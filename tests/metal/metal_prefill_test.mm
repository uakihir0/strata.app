// Metal twins for Prefill's hyper-connection normalization (docs/mac/04 §6.3).
#include "metallib_sha256.inc"
#include "oracles/gdn.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/prefill_dispatch.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"
#include "strata/metal/prefill_gr_read_dispatch.hpp"
#include "strata/metal/prefill_gr_read_mix_dispatch.hpp"
#include "strata/metal/prefill_gr_mix_dispatch.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
#include "strata/metal/prefill_gr_silu_dispatch.hpp"
#include "strata/metal/prefill_gdn_gates_dispatch.hpp"
#include "strata/metal/prefill_gdn_conv_dispatch.hpp"
#include "strata/metal/prefill_gdn_block_dispatch.hpp"
#include "strata/metal/prefill_attention_gdn_transition_dispatch.hpp"
#include "strata/metal/prefill_gdn_dispatch.hpp"
#include "strata/metal/prefill_gdn_l2_dispatch.hpp"
#include "strata/metal/prefill_gdn_out_norm_dispatch.hpp"
#include "strata/metal/prefill_gdn_rec_dispatch.hpp"
#include "strata/metal/prefill_gr_write_norm_dispatch.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"
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
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kStreams = 4u;
constexpr std::uint32_t kGdnHeads = 48u;
constexpr std::uint32_t kGdnConvChannels = 10240u;
constexpr std::uint32_t kGdnConvKernel = 4u;
constexpr std::uint32_t kGdnHeadDim = GDN_HEAD_DIM;
constexpr std::uint32_t kGdnValueDim = GDN_VALUE_DIM;
constexpr std::uint32_t kGdnStateElements = GDN_STATE_ELEMENTS;
constexpr std::uint32_t kThreads = 256u;
constexpr std::size_t kGdnHalfBytes = sizeof(std::uint16_t);
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr double kGemmUnitRoundoff = 0x1.0p-24;
constexpr double kGemmUnderflowAllowance = 0x1.0p-126;
constexpr double kGemmRelativeTolerance = 0x1.0p-16;

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

class ScopedEnvironment {
public:
    ~ScopedEnvironment() { (void)restore(); }

    bool set(std::string name, const char* value) {
        if (active_) return false;
        name_ = std::move(name);
        const char* const current = std::getenv(name_.c_str());
        had_value_ = current != nullptr;
        if (had_value_) old_value_ = current;
        if (::setenv(name_.c_str(), value, 1) != 0) return false;
        active_ = true;
        return true;
    }

    bool restore() {
        if (!active_) return true;
        const int result = had_value_
                               ? ::setenv(name_.c_str(), old_value_.c_str(), 1)
                               : ::unsetenv(name_.c_str());
        if (result != 0) return false;
        active_ = false;
        return true;
    }

private:
    std::string name_;
    std::string old_value_;
    bool had_value_ = false;
    bool active_ = false;
};

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
    std::span<const std::uint8_t> payload() const noexcept {
        return {static_cast<const std::uint8_t*>(view_.contents), payload_bytes_};
    }
    std::span<const std::uint8_t> allocation() const noexcept {
        return {static_cast<const std::uint8_t*>(storage_.contents),
                storage_.length};
    }
    std::span<std::uint8_t> allocation() noexcept {
        return {static_cast<std::uint8_t*>(storage_.contents), storage_.length};
    }

    bool guards_intact() const {
        const auto bytes = allocation();
        const std::size_t suffix = kGuardBytes + payload_bytes_;
        return std::all_of(bytes.begin(), bytes.begin() + kGuardBytes,
                           [](std::uint8_t value) { return value == kPoison; }) &&
               std::all_of(bytes.begin() + suffix, bytes.end(),
                           [](std::uint8_t value) { return value == kPoison; });
    }

private:
    BufferRef storage_{};
    BufferRef view_{};
    std::size_t payload_bytes_ = 0u;
};

std::uint32_t float_ordered(float value) {
    std::uint32_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

std::uint64_t float_ulp_distance(float left, float right) {
    const std::uint32_t a = float_ordered(left);
    const std::uint32_t b = float_ordered(right);
    return a < b ? std::uint64_t{b} - a : std::uint64_t{a} - b;
}

std::uint16_t half_ordered(std::uint16_t bits) {
    return (bits & 0x8000u) != 0u
               ? static_cast<std::uint16_t>(~bits)
               : static_cast<std::uint16_t>(bits | 0x8000u);
}

std::uint32_t half_ulp_distance(std::uint16_t left, std::uint16_t right) {
    const std::uint32_t a = half_ordered(left);
    const std::uint32_t b = half_ordered(right);
    return a < b ? b - a : a - b;
}

std::uint16_t bf16_ordered(std::uint16_t value) {
    return (value & 0x8000u) != 0u
               ? static_cast<std::uint16_t>(~value)
               : static_cast<std::uint16_t>(value | 0x8000u);
}

std::uint16_t bf16_ulp_distance(std::uint16_t left, std::uint16_t right) {
    const std::uint16_t a = bf16_ordered(left);
    const std::uint16_t b = bf16_ordered(right);
    return a < b ? static_cast<std::uint16_t>(b - a)
                 : static_cast<std::uint16_t>(a - b);
}

float xor_butterfly_sum(std::array<float, 32> values) {
    for (std::uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        const auto previous = values;
        for (std::uint32_t lane = 0u; lane < values.size(); ++lane) {
            values[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return values[0];
}

struct Oracle {
    std::vector<float> rs;
    std::vector<std::uint16_t> high;
    std::vector<std::uint16_t> low;
};

Oracle prefill_gr_norm_oracle(std::span<const float> residual,
                              std::span<const float> norm,
                              std::uint32_t rows, bool write_low) {
    const std::size_t stream_rows =
        static_cast<std::size_t>(rows) * kStreams;
    Oracle result;
    result.rs.resize(stream_rows);
    result.high.resize(stream_rows * kWidth);
    if (write_low) result.low.resize(stream_rows * kWidth);

    for (std::size_t row = 0u; row < stream_rows; ++row) {
        std::array<float, kThreads> thread_sums{};
        const std::size_t row_offset = row * kWidth;
        for (std::uint32_t tid = 0u; tid < kThreads; ++tid) {
            float sum = 0.0f;
            for (std::uint32_t column = tid; column < kWidth;
                 column += kThreads) {
                const float value = residual[row_offset + column];
                sum = std::fma(value, value, sum);
            }
            thread_sums[tid] = sum;
        }

        std::array<float, 8> partials{};
        for (std::uint32_t simdgroup = 0u; simdgroup < 8u; ++simdgroup) {
            std::array<float, 32> lanes{};
            for (std::uint32_t lane = 0u; lane < lanes.size(); ++lane) {
                lanes[lane] = thread_sums[simdgroup * 32u + lane];
            }
            partials[simdgroup] = xor_butterfly_sum(lanes);
        }
        std::array<float, 32> warp_zero_lanes{};
        std::copy(partials.begin(), partials.end(), warp_zero_lanes.begin());
        const float total = xor_butterfly_sum(warp_zero_lanes);
        const float mean = total / static_cast<float>(kWidth);
        const float scale =
            1.0f / std::sqrt(mean + kEpsilon);
        result.rs[row] = scale;

        const std::size_t gamma_offset = (row % kStreams) * kWidth;
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const float value =
                (residual[row_offset + column] * scale) *
                norm[gamma_offset + column];
            const std::uint16_t high =
                strata::kernels::bf16_from_f32(value);
            result.high[row_offset + column] = high;
            if (write_low) {
                result.low[row_offset + column] =
                    strata::kernels::bf16_from_f32(
                        value - strata::kernels::f32_from_bf16(high));
            }
        }
    }
    return result;
}

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable after commit";
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = label + " command buffer failed";
        if (native.error.localizedDescription.UTF8String != nullptr) {
            error += ": ";
            error += native.error.localizedDescription.UTF8String;
        }
        return false;
    }
    error.clear();
    return true;
}

bool check_outputs(const std::string& label, const Oracle& expected,
                   const TestBuffer& rs_buffer,
                   const TestBuffer& high_buffer,
                   const TestBuffer* low_buffer,
                   std::span<const float> residual,
                   std::span<const float> norm) {
    bool passed = true;
    const auto actual_rs = std::span<const float>(
        reinterpret_cast<const float*>(rs_buffer.payload().data()),
        expected.rs.size());
    std::uint64_t max_rs_ulp = 0u;
    for (std::size_t index = 0u; index < expected.rs.size(); ++index) {
        max_rs_ulp = std::max(max_rs_ulp,
                              float_ulp_distance(actual_rs[index],
                                                 expected.rs[index]));
    }
    check(max_rs_ulp <= 4u, label + " scale matches ordered CPU oracle",
          "max ULP=" + std::to_string(max_rs_ulp) + ", limit=4");
    passed &= max_rs_ulp <= 4u;

    const auto actual_high = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(high_buffer.payload().data()),
        expected.high.size());
    std::uint16_t max_high_ulp = 0u;
    for (std::size_t index = 0u; index < expected.high.size(); ++index) {
        max_high_ulp = std::max(
            max_high_ulp,
            bf16_ulp_distance(actual_high[index], expected.high[index]));
    }
    check(max_high_ulp <= 1u, label + " BF16 image matches RNE oracle",
          "max ULP=" + std::to_string(max_high_ulp) + ", limit=1");
    passed &= max_high_ulp <= 1u;

    if (low_buffer != nullptr) {
        const auto actual_low = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(low_buffer->payload().data()),
            expected.low.size());
        std::uint16_t max_low_ulp = 0u;
        for (std::size_t row = 0u; row < expected.rs.size(); ++row) {
            const std::size_t row_offset = row * kWidth;
            const std::size_t gamma_offset = (row % kStreams) * kWidth;
            for (std::uint32_t column = 0u; column < kWidth; ++column) {
                const std::size_t index = row_offset + column;
                const float value =
                    (residual[index] * actual_rs[row]) *
                    norm[gamma_offset + column];
                const float remainder = value -
                    strata::kernels::f32_from_bf16(actual_high[index]);
                const std::uint16_t reference =
                    strata::kernels::bf16_from_f32(remainder);
                max_low_ulp = std::max(
                    max_low_ulp,
                    bf16_ulp_distance(actual_low[index], reference));
            }
        }
        check(max_low_ulp == 0u,
              label + " BF16 remainder matches scale-conditioned RNE oracle",
              "max ULP=" + std::to_string(max_low_ulp) + ", limit=0");
        passed &= max_low_ulp == 0u;
    }
    check(rs_buffer.guards_intact(), label + " scale output guards");
    check(high_buffer.guards_intact(), label + " BF16 output guards");
    if (low_buffer != nullptr) {
        check(low_buffer->guards_intact(), label + " remainder output guards");
    }
    return passed;
}

void fill_inputs(std::vector<float>& residual, std::vector<float>& norm,
                 std::uint32_t rows, twin::InputSet input_set,
                 std::uint64_t seed) {
    std::mt19937 rng(static_cast<std::uint32_t>(seed));
    std::uniform_real_distribution<float> random_value(-1.25f, 1.25f);
    for (std::size_t index = 0u; index < residual.size(); ++index) {
        if (input_set == twin::InputSet::edge) {
            constexpr std::array<float, 8> edge{
                -1.0f, 1.0f, -0.5f, 0.5f, -0.125f, 0.125f, -0.001f, 0.001f};
            residual[index] = edge[(index * 7u + index / kWidth) % edge.size()];
        } else if (input_set == twin::InputSet::fixture) {
            const int centered = static_cast<int>((index * 29u + rows * 13u) % 257u) - 128;
            residual[index] = static_cast<float>(centered) * 0.00390625f;
        } else {
            residual[index] = random_value(rng);
        }
    }
    if (input_set == twin::InputSet::edge) {
        residual[0] = 4096.0f;
        for (std::size_t column = 1u; column < 32u; ++column) {
            residual[column] = 1.0f;
        }
    }
    for (std::size_t index = 0u; index < norm.size(); ++index) {
        const int centered = static_cast<int>((index * 17u + 11u) % 37u) - 18;
        norm[index] = 0.75f + static_cast<float>(centered) * 0.015625f;
    }
}

bool run_case(Context& context, std::uint32_t rows,
              PrefillBF16X2Mode mode, twin::InputSet input_set,
              std::uint64_t seed, bool shared_encoder) {
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::string label = "Prefill GR norm T=" + std::to_string(rows) +
        " mode=" + std::to_string(static_cast<std::uint32_t>(mode)) +
        " set=" + twin::input_set_name(input_set) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t stream_rows = static_cast<std::size_t>(rows) * kStreams;
    const std::size_t residual_bytes = stream_rows * kWidth * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * sizeof(float);
    const std::size_t scale_bytes = stream_rows * sizeof(float);
    const std::size_t output_bytes = stream_rows * kWidth * sizeof(std::uint16_t);

    std::vector<float> residual(stream_rows * kWidth);
    std::vector<float> norm(kStreams * kWidth);
    fill_inputs(residual, norm, rows, input_set, seed);
    const Oracle expected =
        prefill_gr_norm_oracle(residual, norm, rows, write_low);
    if (input_set == twin::InputSet::edge) {
        float linear_sum = 0.0f;
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            const float value = residual[column];
            linear_sum = std::fma(value, value, linear_sum);
        }
        const float linear_scale =
            1.0f / std::sqrt(linear_sum / static_cast<float>(kWidth) +
                             kEpsilon);
        const std::uint64_t order_distance =
            float_ulp_distance(expected.rs[0], linear_scale);
        check(order_distance > 4u,
              label + " distinguishes a linear reduction order",
              "scale distance=" + std::to_string(order_distance) +
                  " ULP, required >4");
        if (order_distance <= 4u) return false;
    }

    TestBuffer residual_buffer, norm_buffer, rs_buffer, high_buffer, low_buffer;
    std::string error;
    check(residual_buffer.allocate(context, label + " residual",
                                   residual_bytes, error),
          label + " allocate residual", error);
    check(norm_buffer.allocate(context, label + " norm", norm_bytes, error),
          label + " allocate norm", error);
    check(rs_buffer.allocate(context, label + " rs", scale_bytes, error),
          label + " allocate rs", error);
    check(high_buffer.allocate(context, label + " xn16", output_bytes, error),
          label + " allocate xn16", error);
    if (write_low) {
        check(low_buffer.allocate(context, label + " xn16_lo", output_bytes,
                                  error),
              label + " allocate xn16_lo", error);
    }
    if (!residual_buffer.view() || !norm_buffer.view() || !rs_buffer.view() ||
        !high_buffer.view() || (write_low && !low_buffer.view())) {
        return false;
    }

    std::memcpy(residual_buffer.payload().data(), residual.data(), residual_bytes);
    std::memcpy(norm_buffer.payload().data(), norm.data(), norm_bytes);
    const std::vector<std::uint8_t> residual_before(
        residual_buffer.allocation().begin(), residual_buffer.allocation().end());
    const std::vector<std::uint8_t> norm_before(
        norm_buffer.allocation().begin(), norm_buffer.allocation().end());

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const PrefillGRNormBuffers buffers{
        residual_buffer.view(), norm_buffer.view(), rs_buffer.view(),
        high_buffer.view(), write_low ? low_buffer.view() : BufferRef{}};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared Prefill GR norm",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gr_norm(context, encoder, rows, mode,
                                                buffers, encode_error);
              },
              error)
        : encode_prefill_gr_norm(context, *command, rows, mode, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const bool input_unchanged =
        std::equal(residual_before.begin(), residual_before.end(),
                   residual_buffer.allocation().begin()) &&
        std::equal(norm_before.begin(), norm_before.end(),
                   norm_buffer.allocation().begin());
    check(input_unchanged, label + " inputs remain read-only");
    check(residual_buffer.guards_intact(), label + " residual guards");
    check(norm_buffer.guards_intact(), label + " norm guards");
    bool passed = check_outputs(label, expected, rs_buffer, high_buffer,
                                write_low ? &low_buffer : nullptr,
                                residual, norm);
    return input_unchanged && passed;
}

bool reject_call(Context& context, std::uint32_t rows,
                 PrefillBF16X2Mode mode, const PrefillGRNormBuffers& buffers,
                 std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GR norm",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_norm(context, encoder, rows, mode,
                                          buffers, encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_encoder_validation(Context& context, Context& foreign_context) {
    const std::string label = "Prefill GR norm validation";
    std::string error;
    TestBuffer residual, norm, rs, high, low, foreign_residual;
    constexpr std::size_t kResidualBytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t kNormBytes = kStreams * kWidth * sizeof(float);
    constexpr std::size_t kScaleBytes = kStreams * sizeof(float);
    constexpr std::size_t kOutputBytes =
        kStreams * kWidth * sizeof(std::uint16_t);
    const bool allocated =
        residual.allocate(context, label + " residual", kResidualBytes, error) &&
        norm.allocate(context, label + " norm", kNormBytes, error) &&
        rs.allocate(context, label + " rs", kScaleBytes, error) &&
        high.allocate(context, label + " high", kOutputBytes, error) &&
        low.allocate(context, label + " low", kOutputBytes, error) &&
        foreign_residual.allocate(foreign_context, label + " foreign residual",
                                  kResidualBytes, error);
    check(allocated, label + " allocate validation buffers", error);
    if (!allocated) return false;
    const PrefillGRNormBuffers valid{
        residual.view(), norm.view(), rs.view(), high.view(), low.view()};
    bool passed = true;
    passed &= reject_call(context, 0u, PrefillBF16X2Mode::off, valid,
                          "row count", label + " rejects zero rows");
    passed &= reject_call(context, 1u, static_cast<PrefillBF16X2Mode>(3u),
                          {residual.view(), norm.view(), rs.view(), high.view(), {}},
                          "mode is invalid", label + " rejects unknown mode");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::all_projections,
                          {residual.view(), norm.view(), rs.view(), high.view(), {}},
                          "requires xn16_lo", label + " requires mode-1 remainder");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off, valid,
                          "only valid in mode 1", label + " rejects unused remainder");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off,
                          {residual.view().slice(0u, kResidualBytes - 4u),
                           norm.view(), rs.view(), high.view(), {}},
                          "undersized", label + " rejects short residual");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off,
                          {residual.view(), norm.view().slice(2u, kNormBytes - 2u),
                           rs.view(), high.view(), {}},
                          "misaligned", label + " rejects misaligned norm");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off,
                          {residual.view(), norm.view(), residual.view(), high.view(), {}},
                          "overlaps a read-only input",
                          label + " rejects output/input overlap");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off,
                          {residual.view(), norm.view(), high.view(), high.view(), {}},
                          "output buffer views overlap",
                          label + " rejects output/output overlap");
    passed &= reject_call(context, 1u, PrefillBF16X2Mode::off,
                          {foreign_residual.view(), norm.view(), rs.view(), high.view(), {}},
                          "not owned by this Context",
                          label + " rejects foreign Context view");
    return passed;
}

bool test_prefill_embedding_chain(Context& context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t vocabulary = 2u;
    constexpr bool write_low = true;
    const std::string label = "Prefill embedding to GR norm segment";
    const std::array<std::uint32_t, rows> tokens{0u, 1u};
    PrefillChunkMetadata chunk;
    std::string error;
    const bool metadata_ok =
        build_prefill_chunk_metadata(0u, tokens, chunk, error);
    check(metadata_ok, label + " build chunk metadata", error);
    if (!metadata_ok) return false;

    const std::size_t table_bytes =
        static_cast<std::size_t>(vocabulary) * kWidth * sizeof(std::uint16_t);
    const std::size_t token_bytes = rows * sizeof(std::uint32_t);
    const std::size_t embedding_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(float);
    const std::size_t stream_bytes =
        static_cast<std::size_t>(rows) * kStreams * kWidth * sizeof(float);
    const std::size_t norm_bytes = kStreams * kWidth * sizeof(float);
    const std::size_t scale_bytes = rows * kStreams * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * kStreams * kWidth * sizeof(std::uint16_t);
    TestBuffer table, token_buffer, embedding, streams, norm, rs, high, low;
    const bool allocated =
        table.allocate(context, label + " table", table_bytes, error) &&
        token_buffer.allocate(context, label + " tokens", token_bytes, error) &&
        embedding.allocate(context, label + " embedding", embedding_bytes, error) &&
        streams.allocate(context, label + " streams", stream_bytes, error) &&
        norm.allocate(context, label + " norm", norm_bytes, error) &&
        rs.allocate(context, label + " rs", scale_bytes, error) &&
        high.allocate(context, label + " high", output_bytes, error) &&
        low.allocate(context, label + " low", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    std::vector<std::uint16_t> table_bits(vocabulary * kWidth);
    for (std::size_t index = 0u; index < table_bits.size(); ++index) {
        const int centered = static_cast<int>((index * 23u + index / kWidth) % 101u) - 50;
        table_bits[index] = strata::kernels::bf16_from_f32(
            static_cast<float>(centered) * 0.0078125f);
    }
    std::vector<float> expected_residual(rows * kStreams * kWidth);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t stream = 0u; stream < kStreams; ++stream) {
            for (std::uint32_t column = 0u; column < kWidth; ++column) {
                expected_residual[(row * kStreams + stream) * kWidth + column] =
                    strata::kernels::f32_from_bf16(
                        table_bits[tokens[row] * kWidth + column]);
            }
        }
    }
    std::vector<float> norm_values(kStreams * kWidth);
    for (std::size_t index = 0u; index < norm_values.size(); ++index) {
        norm_values[index] = 0.875f + static_cast<float>(index % 29u) * 0.0078125f;
    }
    std::memcpy(table.payload().data(), table_bits.data(), table_bytes);
    std::memcpy(token_buffer.payload().data(), tokens.data(), token_bytes);
    std::memcpy(norm.payload().data(), norm_values.data(), norm_bytes);
    const Oracle expected = prefill_gr_norm_oracle(
        expected_residual, norm_values, rows, write_low);

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const PrefillEmbeddingBuffers embedding_buffers{
        {table.view(), token_buffer.view(), embedding.view()}, {}, streams.view()};
    const PrefillGRNormBuffers norm_buffers{
        streams.view(), norm.view(), rs.view(), high.view(), low.view()};
    const bool encoded = Encoder::encode(
        context, *command, "Prefill embedding and GR norm",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_embedding(context, encoder, chunk, vocabulary,
                                          EmbedRowsFormat::bf16,
                                          embedding_buffers, encode_error)) {
                return false;
            }
            return encode_prefill_gr_norm(
                context, encoder, rows, PrefillBF16X2Mode::all_projections,
                norm_buffers, encode_error);
        },
        error);
    check(encoded, label + " production composition", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const bool streams_match = std::memcmp(
        streams.payload().data(), expected_residual.data(), stream_bytes) == 0;
    check(streams_match, label + " broadcast feeds all four norm streams");
    const bool passed = check_outputs(label, expected, rs, high, &low,
                                      expected_residual, norm_values);
    check(table.guards_intact() && token_buffer.guards_intact() &&
              embedding.guards_intact() && streams.guards_intact() &&
              norm.guards_intact(),
          label + " input and embedding-stage guards");
    return streams_match && passed;
}

bool test_failure_discards_prior_prefill_dispatch(Context& context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t vocabulary = 1u;
    const std::string label = "Prefill GR norm failure discards earlier dispatch";
    const std::array<std::uint32_t, rows> tokens{0u};
    PrefillChunkMetadata chunk;
    std::string error;
    if (!build_prefill_chunk_metadata(0u, tokens, chunk, error)) {
        check(false, label + " metadata", error);
        return false;
    }
    TestBuffer table, token_buffer, embedding, streams, norm, short_norm, rs, high;
    const std::size_t table_bytes = kWidth * sizeof(std::uint16_t);
    const std::size_t norm_bytes = kStreams * kWidth * sizeof(float);
    const std::size_t output_bytes = kStreams * kWidth * sizeof(std::uint16_t);
    const bool allocated =
        table.allocate(context, label + " table", table_bytes, error) &&
        token_buffer.allocate(context, label + " tokens", sizeof(std::uint32_t), error) &&
        embedding.allocate(context, label + " embedding", kWidth * sizeof(float), error) &&
        streams.allocate(context, label + " streams", norm_bytes, error) &&
        norm.allocate(context, label + " norm", norm_bytes, error) &&
        short_norm.allocate(context, label + " short norm", norm_bytes, error) &&
        rs.allocate(context, label + " rs", kStreams * sizeof(float), error) &&
        high.allocate(context, label + " high", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    std::vector<std::uint16_t> table_values(kWidth, 0x3f80u);
    std::vector<std::uint32_t> token_values(tokens.begin(), tokens.end());
    std::memcpy(table.payload().data(), table_values.data(), table_bytes);
    std::memcpy(token_buffer.payload().data(), token_values.data(), sizeof(std::uint32_t));
    std::memset(short_norm.payload().data(), 0,
                short_norm.payload().size());
    const std::vector<std::uint8_t> streams_before(
        streams.allocation().begin(), streams.allocation().end());
    const std::vector<std::uint8_t> embedding_before(
        embedding.allocation().begin(), embedding.allocation().end());
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const PrefillEmbeddingBuffers embedding_buffers{
        {table.view(), token_buffer.view(), embedding.view()}, {}, streams.view()};
    const PrefillGRNormBuffers invalid_norm_buffers{
        streams.view(), short_norm.view().slice(0u, norm_bytes - sizeof(float)),
        rs.view(), high.view(), {}};
    const bool encoded = Encoder::encode(
        context, *command, "Prefill rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_embedding(context, encoder, chunk, vocabulary,
                                          EmbedRowsFormat::bf16,
                                          embedding_buffers, encode_error)) {
                return false;
            }
            return encode_prefill_gr_norm(
                context, encoder, rows, PrefillBF16X2Mode::off,
                invalid_norm_buffers, encode_error);
        },
        error);
    const bool segment_discarded =
        command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && segment_discarded &&
        error.find("undersized") != std::string::npos &&
        std::equal(streams_before.begin(), streams_before.end(),
                   streams.allocation().begin()) &&
        std::equal(embedding_before.begin(), embedding_before.end(),
                   embedding.allocation().begin());
    check(unchanged, label,
          error + (segment_discarded ? " (command buffer discarded)"
                                    : " (command buffer remained submit-capable)"));
    return unchanged;
}

struct SiluOracle {
    std::vector<std::uint16_t> high;
    std::vector<std::uint16_t> low;
};

SiluOracle prefill_gr_silu_oracle(std::span<const float> low_rank,
                                  bool write_low) {
    SiluOracle result;
    result.high.resize(low_rank.size());
    if (write_low) result.low.resize(low_rank.size());
    for (std::size_t index = 0u; index < low_rank.size(); ++index) {
        const float x = low_rank[index] / 4.0f;
        const float value = x / (1.0f + std::exp(-x));
        const std::uint16_t high =
            strata::kernels::bf16_from_f32(value);
        result.high[index] = high;
        if (write_low) {
            result.low[index] = strata::kernels::bf16_from_f32(
                value - strata::kernels::f32_from_bf16(high));
        }
    }
    return result;
}

bool check_prefill_projection_oracle(
    std::string_view label, std::span<const std::uint16_t> input_high,
    std::span<const std::uint16_t> input_low,
    std::span<const std::uint16_t> weights, std::uint32_t rows,
    std::uint32_t n, std::uint32_t k, std::span<const float> actual) {
    if (input_high.size() != static_cast<std::size_t>(rows) * k ||
        (!input_low.empty() && input_low.size() != input_high.size()) ||
        weights.size() != static_cast<std::size_t>(n) * k ||
        actual.size() != static_cast<std::size_t>(rows) * n) {
        check(false, std::string(label) + " FP64 oracle dimensions");
        return false;
    }

    const double error_scale =
        2.0 * static_cast<double>(k) * kGemmUnitRoundoff;
    bool within_bound = true;
    bool within_relative_tolerance = true;
    double max_relative_l2 = 0.0;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        double row_error_squared = 0.0;
        double row_reference_squared = 0.0;
        for (std::uint32_t column = 0u; column < n; ++column) {
            double reference = 0.0;
            double magnitude = 0.0;
            for (std::uint32_t inner = 0u; inner < k; ++inner) {
                const std::size_t input_index =
                    static_cast<std::size_t>(row) * k + inner;
                const double weight = strata::kernels::f32_from_bf16(
                    weights[static_cast<std::size_t>(column) * k + inner]);
                const double high = strata::kernels::f32_from_bf16(
                    input_high[input_index]);
                const double high_product = high * weight;
                reference += high_product;
                magnitude += std::abs(high_product);
                if (!input_low.empty()) {
                    const double low = strata::kernels::f32_from_bf16(
                        input_low[input_index]);
                    const double low_product = low * weight;
                    reference += low_product;
                    magnitude += std::abs(low_product);
                }
            }
            const float output =
                actual[static_cast<std::size_t>(row) * n + column];
            const double error = static_cast<double>(output) - reference;
            const double bound =
                error_scale * magnitude + kGemmUnderflowAllowance;
            within_bound &= std::isfinite(output) &&
                            std::abs(error) <= bound;
            row_error_squared += error * error;
            row_reference_squared += reference * reference;
        }
        if (row_reference_squared != 0.0) {
            const double relative_l2 =
                std::sqrt(row_error_squared / row_reference_squared);
            max_relative_l2 = std::max(max_relative_l2, relative_l2);
            within_relative_tolerance &=
                relative_l2 <= kGemmRelativeTolerance;
        } else {
            within_bound &= row_error_squared == 0.0;
            within_relative_tolerance &= row_error_squared == 0.0;
        }
    }
    check(within_bound, std::string(label) + " FP64 GEMM U(2K) error bound",
          "max row-relative L2 error " + std::to_string(max_relative_l2));
    check(within_relative_tolerance,
          std::string(label) + " FP64 GEMM R(2^-16) tolerance",
          "max row-relative L2 error " + std::to_string(max_relative_l2));
    return within_bound && within_relative_tolerance;
}

struct PrefillGRReadTestOutputs {
    TestBuffer rs;
    TestBuffer xn16;
    TestBuffer xn16_lo;
    TestBuffer low_rank;
    TestBuffer low_rank_bf16;
    TestBuffer low_rank_bf16_lo;
    TestBuffer gated;
    TestBuffer inject;

    bool allocate(Context& context, std::uint32_t rows, bool write_low,
                  std::string_view label, std::string& error) {
        const std::size_t stream_rows =
            static_cast<std::size_t>(rows) * kStreams;
        const std::size_t normalized_values =
            stream_rows * kWidth;
        const std::size_t low_rank_values =
            static_cast<std::size_t>(rows) * 320u;
        const std::size_t gated_values =
            stream_rows * kWidth;
        const std::string prefix(label);
        return rs.allocate(context, prefix + " rs",
                           stream_rows * sizeof(float), error) &&
               xn16.allocate(context, prefix + " xn16",
                             normalized_values * sizeof(std::uint16_t),
                             error) &&
               (!write_low ||
                xn16_lo.allocate(context, prefix + " xn16 low",
                                 normalized_values * sizeof(std::uint16_t),
                                 error)) &&
               low_rank.allocate(context, prefix + " low rank",
                                 low_rank_values * sizeof(float), error) &&
               low_rank_bf16.allocate(
                   context, prefix + " low rank BF16",
                   low_rank_values * sizeof(std::uint16_t), error) &&
               (!write_low ||
                low_rank_bf16_lo.allocate(
                    context, prefix + " low rank BF16 low",
                    low_rank_values * sizeof(std::uint16_t), error)) &&
               gated.allocate(context, prefix + " gated",
                              gated_values * sizeof(float), error) &&
               inject.allocate(context, prefix + " inject",
                               stream_rows * sizeof(float), error);
    }

    PrefillGRReadBuffers bind(
        BufferRef residual, BufferRef norm, ReadOnlyBufferRef down_weights,
        ReadOnlyBufferRef up_weights,
        ReadOnlyBufferRef inject_weights, bool write_low) const {
        return {residual,
                norm,
                rs.view(),
                xn16.view(),
                write_low ? xn16_lo.view() : BufferRef{},
                down_weights,
                low_rank.view(),
                low_rank_bf16.view(),
                write_low ? low_rank_bf16_lo.view() : BufferRef{},
                up_weights,
                gated.view(),
                inject_weights,
                inject.view()};
    }
};

bool encode_prefill_gr_read_reference(
    Context& context, Encoder& encoder, std::uint32_t rows,
    PrefillBF16X2Mode mode, const PrefillGRReadBuffers& buffers,
    std::string& error) {
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const PrefillGRNormBuffers norm_buffers{
        buffers.residual, buffers.norm, buffers.rs, buffers.xn16,
        write_low ? buffers.xn16_lo : BufferRef{}};
    if (!encode_prefill_gr_norm(context, encoder, rows, mode, norm_buffers,
                                error)) {
        return false;
    }

    const auto xn16 = context.buffers().read_only(buffers.xn16);
    const auto xn16_lo =
        write_low ? context.buffers().read_only(buffers.xn16_lo)
                  : ReadOnlyBufferRef{};
    const PrefillGemmMPPParams down_params{
        rows, 320u, 10240u, 320u, 0u, false};
    const PrefillGemmMPPParams up_params{
        rows, 10240u, 320u, 10240u, 0u, false};
    const PrefillGemmMPPParams inject_params{
        rows, 4u, 10240u, 4u, 0u, false};
    const auto project = [&](const PrefillGemmMPPParams& params,
                             ReadOnlyBufferRef high,
                             ReadOnlyBufferRef low,
                             ReadOnlyBufferRef weights, BufferRef output) {
        if (write_low) {
            return encode_prefill_gemm_bf16x2(
                context, encoder, params,
                {high, low, weights, output}, error);
        }
        return encode_prefill_gemm(
            context, encoder, params, {high, weights, output},
            PrefillGemmFormat::BF16, error);
    };
    if (!project(down_params, xn16, xn16_lo, buffers.down_weights,
                 buffers.low_rank)) {
        return false;
    }

    const PrefillGRSiluBuffers silu_buffers{
        context.buffers().read_only(buffers.low_rank),
        buffers.low_rank_bf16,
        write_low ? buffers.low_rank_bf16_lo : BufferRef{}};
    if (!encode_prefill_gr_silu(context, encoder, rows, mode, silu_buffers,
                                error)) {
        return false;
    }

    const auto low_rank_bf16 =
        context.buffers().read_only(buffers.low_rank_bf16);
    const auto low_rank_bf16_lo =
        write_low
            ? context.buffers().read_only(buffers.low_rank_bf16_lo)
            : ReadOnlyBufferRef{};
    if (!project(up_params, low_rank_bf16, low_rank_bf16_lo,
                 buffers.up_weights, buffers.gated)) {
        return false;
    }
    return project(inject_params, xn16, xn16_lo, buffers.inject_weights,
                   buffers.inject);
}

bool test_prefill_gr_read_chain(Context& context) {
    constexpr std::array<std::uint32_t, 3> rows_by_mode{1u, 2u, 3u};
    constexpr std::array<PrefillBF16X2Mode, 3> modes{
        PrefillBF16X2Mode::off,
        PrefillBF16X2Mode::all_projections,
        PrefillBF16X2Mode::except_gr};
    constexpr std::size_t normalized_values = kStreams * kWidth;
    constexpr std::size_t down_weight_values = 320u * 10240u;
    constexpr std::size_t up_weight_values = 10240u * 320u;
    constexpr std::size_t inject_weight_values = 4u * 10240u;
    const std::string label = "Prefill GR normalization and read projections";
    std::string error;
    TestBuffer residual, norm, down_weights, up_weights, inject_weights;
    const bool inputs_allocated =
        residual.allocate(context, label + " residual",
                          rows_by_mode[2] * normalized_values *
                              sizeof(float),
                          error) &&
        norm.allocate(context, label + " norm weights",
                      normalized_values * sizeof(float), error) &&
        down_weights.allocate(context, label + " down weights",
                              down_weight_values * sizeof(std::uint16_t),
                              error) &&
        up_weights.allocate(context, label + " up weights",
                            up_weight_values * sizeof(std::uint16_t), error) &&
        inject_weights.allocate(context, label + " inject weights",
                                inject_weight_values *
                                    sizeof(std::uint16_t),
                                error);
    check(inputs_allocated, label + " allocate model inputs", error);
    if (!inputs_allocated) return false;

    std::vector<float> residual_values(rows_by_mode[2] * normalized_values);
    std::vector<float> norm_values(normalized_values);
    std::vector<std::uint16_t> down_values(down_weight_values);
    std::vector<std::uint16_t> up_values(up_weight_values);
    std::vector<std::uint16_t> inject_values(inject_weight_values);
    for (std::size_t index = 0u; index < residual_values.size(); ++index) {
        const int centered = static_cast<int>((index * 29u + 17u) % 257u) -
                             128;
        residual_values[index] = static_cast<float>(centered) / 128.0f;
    }
    for (std::size_t index = 0u; index < norm_values.size(); ++index) {
        norm_values[index] =
            0.75f + static_cast<float>((index * 17u + 11u) % 37u) *
                        0.015625f;
    }
    const auto fill_bf16 = [](std::vector<std::uint16_t>& values,
                              std::uint32_t multiplier,
                              std::uint32_t modulus, float scale) {
        for (std::size_t index = 0u; index < values.size(); ++index) {
            const int centered =
                static_cast<int>((index * multiplier + index / 7u) %
                                 modulus) -
                static_cast<int>(modulus / 2u);
            values[index] = strata::kernels::bf16_from_f32(
                static_cast<float>(centered) * scale);
        }
    };
    fill_bf16(down_values, 13u, 31u, 0.00390625f);
    fill_bf16(up_values, 19u, 29u, 0.00390625f);
    fill_bf16(inject_values, 7u, 23u, 0.00390625f);
    std::memcpy(residual.payload().data(), residual_values.data(),
                residual_values.size() * sizeof(float));
    std::memcpy(norm.payload().data(), norm_values.data(),
                norm_values.size() * sizeof(float));
    std::memcpy(down_weights.payload().data(), down_values.data(),
                down_values.size() * sizeof(std::uint16_t));
    std::memcpy(up_weights.payload().data(), up_values.data(),
                up_values.size() * sizeof(std::uint16_t));
    std::memcpy(inject_weights.payload().data(), inject_values.data(),
                inject_values.size() * sizeof(std::uint16_t));
    const std::vector<std::uint8_t> residual_before(
        residual.payload().begin(), residual.payload().end());
    const std::vector<std::uint8_t> norm_before(
        norm.payload().begin(), norm.payload().end());
    const std::vector<std::uint8_t> down_weights_before(
        down_weights.payload().begin(), down_weights.payload().end());
    const std::vector<std::uint8_t> up_weights_before(
        up_weights.payload().begin(), up_weights.payload().end());
    const std::vector<std::uint8_t> inject_weights_before(
        inject_weights.payload().begin(), inject_weights.payload().end());
    const auto down_read = context.buffers().read_only(down_weights.view());
    const auto up_read = context.buffers().read_only(up_weights.view());
    const auto inject_read = context.buffers().read_only(inject_weights.view());

    bool passed = true;
    for (std::size_t mode_index = 0u; mode_index < modes.size();
         ++mode_index) {
        const std::uint32_t rows = rows_by_mode[mode_index];
        const bool write_low =
            modes[mode_index] == PrefillBF16X2Mode::all_projections;
        const bool mix_write_low =
            modes[mode_index] != PrefillBF16X2Mode::off;
        PrefillGRReadTestOutputs composed, reference;
        const bool outputs_allocated =
            composed.allocate(context, rows, write_low,
                              label + " composed " +
                                  std::to_string(mode_index),
                              error) &&
            reference.allocate(context, rows, write_low,
                               label + " reference " +
                                   std::to_string(mode_index),
                               error);
        check(outputs_allocated, label + " allocate chain outputs", error);
        if (!outputs_allocated) return false;

        const PrefillGRReadBuffers composed_buffers = composed.bind(
            residual.view(), norm.view(), down_read, up_read, inject_read,
            write_low);
        const PrefillGRReadBuffers reference_buffers = reference.bind(
            residual.view(), norm.view(), down_read, up_read, inject_read,
            write_low);

        auto composed_command =
            context.make_command_buffer(label + " composed", error);
        if (!composed_command) {
            check(false, label + " composed command buffer", error);
            return false;
        }
        const bool composed_encoded = encode_prefill_gr_read(
            context, *composed_command, rows, modes[mode_index],
            composed_buffers, error);
        check(composed_encoded, label + " coordinator encoding", error);
        if (!composed_encoded) return false;
        const bool composed_completed =
            submit_and_wait(composed_command, label + " composed", error);
        check(composed_completed, label + " coordinator dispatch", error);
        if (!composed_completed) return false;

        auto reference_command =
            context.make_command_buffer(label + " reference", error);
        if (!reference_command) {
            check(false, label + " reference command buffer", error);
            return false;
        }
        const bool reference_encoded = Encoder::encode(
            context, *reference_command, "direct Prefill GR read chain",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_gr_read_reference(
                    context, encoder, rows, modes[mode_index],
                    reference_buffers, encode_error);
            },
            error);
        check(reference_encoded, label + " direct component chain", error);
        if (!reference_encoded) return false;
        const bool reference_completed =
            submit_and_wait(reference_command, label + " reference", error);
        check(reference_completed, label + " direct chain dispatch", error);
        if (!reference_completed) return false;

        const auto compare = [&](const TestBuffer& actual,
                                 const TestBuffer& expected,
                                 std::string_view stage) {
            const bool equal =
                std::equal(actual.payload().begin(), actual.payload().end(),
                           expected.payload().begin());
            check(equal, label + " " + std::string(stage) +
                             " matches direct composition");
            return equal;
        };
        passed &= compare(composed.rs, reference.rs, "scale");
        passed &= compare(composed.xn16, reference.xn16, "normalized input");
        passed &= compare(composed.low_rank, reference.low_rank,
                          "down projection");
        passed &= compare(composed.low_rank_bf16, reference.low_rank_bf16,
                          "SiLU conversion");
        passed &= compare(composed.gated, reference.gated, "up projection");
        passed &= compare(composed.inject, reference.inject,
                          "injection projection");
        if (write_low) {
            passed &= compare(composed.xn16_lo, reference.xn16_lo,
                              "normalized remainder");
            passed &= compare(composed.low_rank_bf16_lo,
                              reference.low_rank_bf16_lo,
                              "SiLU remainder");
        }

        const std::span<const float> active_residual(
            residual_values.data(),
            static_cast<std::size_t>(rows) * normalized_values);
        const Oracle expected_norm = prefill_gr_norm_oracle(
            active_residual, norm_values, rows, write_low);
        passed &= check_outputs(
            label + " mode " + std::to_string(mode_index), expected_norm,
            composed.rs, composed.xn16,
            write_low ? &composed.xn16_lo : nullptr, active_residual,
            norm_values);

        const std::size_t normalized_count =
            static_cast<std::size_t>(rows) * normalized_values;
        const auto* const normalized_high_values =
            reinterpret_cast<const std::uint16_t*>(
                composed.xn16.payload().data());
        const auto* const normalized_low_values =
            write_low
                ? reinterpret_cast<const std::uint16_t*>(
                      composed.xn16_lo.payload().data())
                : nullptr;
        const std::span<const std::uint16_t> normalized_high(
            normalized_high_values, normalized_count);
        const std::span<const std::uint16_t> normalized_low =
            write_low
                ? std::span<const std::uint16_t>(normalized_low_values,
                                                 normalized_count)
                : std::span<const std::uint16_t>{};
        const auto* const low_rank_values =
            reinterpret_cast<const float*>(composed.low_rank.payload().data());
        passed &= check_prefill_projection_oracle(
            label + " down projection", normalized_high, normalized_low,
            down_values, rows, 320u, 10240u,
            std::span<const float>(low_rank_values, rows * 320u));

        const std::size_t low_rank_count =
            static_cast<std::size_t>(rows) * 320u;
        const SiluOracle expected_silu = prefill_gr_silu_oracle(
            std::span<const float>(low_rank_values, low_rank_count),
            write_low);
        const bool silu_high_matches = std::equal(
            expected_silu.high.begin(), expected_silu.high.end(),
            reinterpret_cast<const std::uint16_t*>(
                composed.low_rank_bf16.payload().data()));
        check(silu_high_matches, label + " SiLU BF16 matches scalar oracle");
        passed &= silu_high_matches;
        if (write_low) {
            const auto* const actual_low =
                reinterpret_cast<const std::uint16_t*>(
                    composed.low_rank_bf16_lo.payload().data());
            std::uint16_t max_low_ulp = 0u;
            for (std::size_t index = 0u;
                 index < expected_silu.low.size(); ++index) {
                max_low_ulp = std::max(
                    max_low_ulp,
                    bf16_ulp_distance(actual_low[index],
                                      expected_silu.low[index]));
            }
            const bool silu_low_matches = max_low_ulp <= 2u;
            check(silu_low_matches,
                  label + " SiLU remainder matches scalar oracle",
                  "max ULP=" + std::to_string(max_low_ulp) + ", limit=2");
            passed &= silu_low_matches;
        }

        const auto* const low_rank_high =
            reinterpret_cast<const std::uint16_t*>(
                composed.low_rank_bf16.payload().data());
        const auto* const low_rank_low =
            write_low
                ? reinterpret_cast<const std::uint16_t*>(
                      composed.low_rank_bf16_lo.payload().data())
                : nullptr;
        const auto* const gated_values =
            reinterpret_cast<const float*>(composed.gated.payload().data());
        passed &= check_prefill_projection_oracle(
            label + " up projection",
            std::span<const std::uint16_t>(low_rank_high, low_rank_count),
            write_low
                ? std::span<const std::uint16_t>(low_rank_low,
                                                 low_rank_count)
                : std::span<const std::uint16_t>{},
            up_values, rows, 10240u, 320u,
            std::span<const float>(gated_values,
                                   static_cast<std::size_t>(rows) *
                                       normalized_values));

        const auto* const inject_result =
            reinterpret_cast<const float*>(composed.inject.payload().data());
        passed &= check_prefill_projection_oracle(
            label + " injection projection", normalized_high, normalized_low,
            inject_values, rows, 4u, 10240u,
            std::span<const float>(
                inject_result, static_cast<std::size_t>(rows) * kStreams));

        PrefillGRReadTestOutputs integrated_read, direct_read;
        TestBuffer mixed_bf, mixed_bf_lo, mixed_h;
        TestBuffer direct_mixed_bf, direct_mixed_bf_lo, direct_mixed_h;
        const std::size_t mixed_bytes =
            static_cast<std::size_t>(rows) * kWidth *
            sizeof(std::uint16_t);
        const bool integrated_allocated =
            integrated_read.allocate(context, rows, write_low,
                                     label + " integrated read", error) &&
            direct_read.allocate(context, rows, write_low,
                                 label + " direct read", error) &&
            mixed_bf.allocate(context, label + " integrated mixed BF16",
                              mixed_bytes, error) &&
            (!mix_write_low ||
             mixed_bf_lo.allocate(context,
                                  label + " integrated mixed BF16 low",
                                  mixed_bytes, error)) &&
            mixed_h.allocate(context, label + " integrated mixed F16",
                             mixed_bytes, error) &&
            direct_mixed_bf.allocate(context, label + " direct mixed BF16",
                                     mixed_bytes, error) &&
            (!mix_write_low ||
             direct_mixed_bf_lo.allocate(
                 context, label + " direct mixed BF16 low", mixed_bytes,
                 error)) &&
            direct_mixed_h.allocate(context, label + " direct mixed F16",
                                    mixed_bytes, error);
        check(integrated_allocated, label + " allocate read/mix outputs",
              error);
        if (!integrated_allocated) return false;
        const PrefillGRReadBuffers integrated_read_buffers =
            integrated_read.bind(residual.view(), norm.view(), down_read,
                                 up_read, inject_read, write_low);
        const PrefillGRReadBuffers direct_read_buffers =
            direct_read.bind(residual.view(), norm.view(), down_read,
                             up_read, inject_read, write_low);
        const PrefillGRReadMixBuffers integrated_buffers{
            integrated_read_buffers,
            mixed_bf.view(),
            mix_write_low ? mixed_bf_lo.view() : BufferRef{},
            mixed_h.view()};
        auto integrated_command =
            context.make_command_buffer(label + " read and mix", error);
        if (!integrated_command) {
            check(false, label + " integrated command buffer", error);
            return false;
        }
        const bool integrated_encoded = encode_prefill_gr_read_mix(
            context, *integrated_command, rows, modes[mode_index],
            integrated_buffers, error);
        check(integrated_encoded, label + " complete GR read/mix encode",
              error);
        if (!integrated_encoded) return false;
        const bool integrated_completed =
            submit_and_wait(integrated_command, label + " read and mix",
                            error);
        check(integrated_completed, label + " complete GR read/mix dispatch",
              error);
        if (!integrated_completed) return false;

        const PrefillGRMixBuffers direct_mix_buffers{
            context.buffers().read_only(residual.view()),
            context.buffers().read_only(direct_read.rs.view()),
            context.buffers().read_only(norm.view()),
            context.buffers().read_only(direct_read.gated.view()),
            direct_mixed_bf.view(),
            mix_write_low ? direct_mixed_bf_lo.view() : BufferRef{},
            direct_mixed_h.view()};
        auto direct_mix_command =
            context.make_command_buffer(label + " direct read and mix", error);
        if (!direct_mix_command) {
            check(false, label + " direct read/mix command buffer", error);
            return false;
        }
        const bool direct_mix_encoded = Encoder::encode(
            context, *direct_mix_command, "direct Prefill GR read and mix",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encode_prefill_gr_read_reference(
                        context, encoder, rows, modes[mode_index],
                        direct_read_buffers, encode_error)) {
                    return false;
                }
                return encode_prefill_gr_mix(
                    context, encoder, rows, modes[mode_index],
                    direct_mix_buffers, encode_error);
            },
            error);
        check(direct_mix_encoded, label + " direct component read/mix encode",
              error);
        if (!direct_mix_encoded) return false;
        const bool direct_mix_completed =
            submit_and_wait(direct_mix_command, label + " direct read/mix",
                            error);
        check(direct_mix_completed,
              label + " direct component read/mix dispatch", error);
        if (!direct_mix_completed) return false;

        passed &= compare(integrated_read.rs, direct_read.rs,
                          "complete read/mix scale");
        passed &= compare(integrated_read.xn16, direct_read.xn16,
                          "complete read/mix normalized input");
        passed &= compare(integrated_read.low_rank, direct_read.low_rank,
                          "complete read/mix down projection");
        passed &= compare(integrated_read.low_rank_bf16,
                          direct_read.low_rank_bf16,
                          "complete read/mix SiLU conversion");
        passed &= compare(integrated_read.gated, direct_read.gated,
                          "complete read/mix up projection");
        passed &= compare(integrated_read.inject, direct_read.inject,
                          "complete read/mix injection projection");
        passed &= compare(mixed_bf, direct_mixed_bf, "mixer BF16 output");
        passed &= compare(mixed_h, direct_mixed_h, "mixer F16 output");
        if (write_low) {
            passed &= compare(integrated_read.xn16_lo, direct_read.xn16_lo,
                              "complete read/mix normalized remainder");
            passed &= compare(integrated_read.low_rank_bf16_lo,
                              direct_read.low_rank_bf16_lo,
                              "complete read/mix SiLU remainder");
        }
        if (mix_write_low) {
            passed &= compare(mixed_bf_lo, direct_mixed_bf_lo,
                              "mixer BF16 remainder");
        }

        const bool integrated_guards =
            integrated_read.rs.guards_intact() &&
            integrated_read.xn16.guards_intact() &&
            integrated_read.low_rank.guards_intact() &&
            integrated_read.low_rank_bf16.guards_intact() &&
            integrated_read.gated.guards_intact() &&
            integrated_read.inject.guards_intact() &&
            mixed_bf.guards_intact() && mixed_h.guards_intact() &&
            (!mix_write_low || mixed_bf_lo.guards_intact()) &&
            (!write_low ||
             (integrated_read.xn16_lo.guards_intact() &&
              integrated_read.low_rank_bf16_lo.guards_intact()));
        check(integrated_guards, label + " integrated read/mix guards");
        passed &= integrated_guards;

        struct ReadInputAlias {
            std::string_view name;
            BufferRef view;
        };
        const bool guards_intact =
            composed.rs.guards_intact() && composed.xn16.guards_intact() &&
            composed.low_rank.guards_intact() &&
            composed.low_rank_bf16.guards_intact() &&
            composed.gated.guards_intact() &&
            composed.inject.guards_intact() &&
            (!write_low || (composed.xn16_lo.guards_intact() &&
                            composed.low_rank_bf16_lo.guards_intact())) &&
            residual.guards_intact() && norm.guards_intact() &&
            down_weights.guards_intact() && up_weights.guards_intact() &&
            inject_weights.guards_intact();
        check(guards_intact, label + " output and input guards");
        passed &= guards_intact;

        if (mode_index == 0u) {
            PrefillGRReadBuffers aliased = composed_buffers;
            aliased.rs = composed.low_rank.view();
            const std::vector<std::uint8_t> low_rank_before(
                composed.low_rank.allocation().begin(),
                composed.low_rank.allocation().end());
            auto alias_command =
                context.make_command_buffer(label + " alias rejection", error);
            if (!alias_command) {
                check(false, label + " alias command buffer", error);
                return false;
            }
            const bool alias_encoded = encode_prefill_gr_read(
                context, *alias_command, rows, modes[mode_index], aliased,
                error);
            const bool alias_rejected =
                !alias_encoded &&
                alias_command->unsafe_native_handle() == nullptr &&
                error.find("writable output views overlap") !=
                    std::string::npos &&
                std::equal(low_rank_before.begin(), low_rank_before.end(),
                           composed.low_rank.allocation().begin());
            check(alias_rejected,
                  label + " rejects cross-stage output alias before dispatch",
                  error);
            passed &= alias_rejected;

            const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
            if (dispatch != nullptr &&
                std::string_view(dispatch) == "concurrent") {
                auto group_command = context.make_command_buffer(
                    label + " open concurrent group", error);
                if (!group_command) {
                    check(false, label + " open-group command buffer", error);
                    return false;
                }
                const bool group_encoded = Encoder::encode(
                    context, *group_command, "open Prefill GR read group",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encoder.group_begin(encode_error)) return false;
                        return encode_prefill_gr_read(
                            context, encoder, rows, modes[mode_index],
                            composed_buffers, encode_error);
                    },
                    error);
                const bool group_rejected =
                    !group_encoded &&
                    error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
                check(group_rejected,
                      label + " rejects an open concurrent group", error);
                passed &= group_rejected;

                auto integrated_group_command = context.make_command_buffer(
                    label + " open read/mix concurrent group", error);
                if (!integrated_group_command) {
                    check(false,
                          label + " open read/mix group command buffer",
                          error);
                    return false;
                }
                const bool integrated_group_encoded = Encoder::encode(
                    context, *integrated_group_command,
                    "open concurrent Prefill GR read/mix",
                    [&](Encoder& encoder, std::string& encode_error) {
                        if (!encoder.group_begin(encode_error)) return false;
                        return encode_prefill_gr_read_mix(
                            context, encoder, rows, modes[mode_index],
                            integrated_buffers, encode_error);
                    },
                    error);
                const bool integrated_group_rejected =
                    !integrated_group_encoded &&
                    error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
                check(integrated_group_rejected,
                      label + " read/mix rejects an open concurrent group",
                      error);
                passed &= integrated_group_rejected;
            }
        }

        const std::array<ReadInputAlias, 5> read_inputs{{
            {"residual", residual.view()},
            {"norm", norm.view()},
            {"down weights", down_weights.view()},
            {"up weights", up_weights.view()},
            {"injection weights", inject_weights.view()},
        }};
        const std::array<std::string_view, 3> mixer_output_names{
            "mixed BF16", "mixed F16", "mixed BF16 remainder"};
        const std::size_t active_mixer_outputs =
            mix_write_low ? mixer_output_names.size()
                          : mixer_output_names.size() - 1u;
        for (std::size_t output_index = 0u;
             output_index < active_mixer_outputs; ++output_index) {
            for (const ReadInputAlias& input : read_inputs) {
                PrefillGRReadMixBuffers input_aliased = integrated_buffers;
                const BufferRef alias_view =
                    input.view.slice(0u, mixed_bytes);
                if (!alias_view) {
                    check(false,
                          label + " create " +
                              std::string(mixer_output_names[output_index]) +
                              " input-alias view",
                          std::string(input.name));
                    return false;
                }
                switch (output_index) {
                    case 0u:
                        input_aliased.mixed_bf = alias_view;
                        break;
                    case 1u:
                        input_aliased.mixed_h = alias_view;
                        break;
                    default:
                        input_aliased.mixed_bf_lo = alias_view;
                        break;
                }
                const std::vector<std::uint8_t> input_before(
                    static_cast<const std::uint8_t*>(alias_view.contents),
                    static_cast<const std::uint8_t*>(alias_view.contents) +
                        alias_view.length);
                auto input_alias_command = context.make_command_buffer(
                    label + " read/mix " +
                        std::string(mixer_output_names[output_index]) +
                        " / " + std::string(input.name) + " alias",
                    error);
                if (!input_alias_command) {
                    check(false,
                          label + " read/mix input-alias command buffer",
                          error);
                    return false;
                }
                const bool input_alias_encoded =
                    encode_prefill_gr_read_mix(
                        context, *input_alias_command, rows,
                        modes[mode_index], input_aliased, error);
                const bool input_alias_rejected =
                    !input_alias_encoded &&
                    input_alias_command->unsafe_native_handle() == nullptr &&
                    error.find("mixer output overlaps a read input") !=
                        std::string::npos &&
                    std::equal(
                        input_before.begin(), input_before.end(),
                        static_cast<const std::uint8_t*>(
                            alias_view.contents));
                check(input_alias_rejected,
                      label + " rejects " +
                          std::string(mixer_output_names[output_index]) +
                          " alias with " + std::string(input.name),
                      error);
                passed &= input_alias_rejected;
            }
        }

        if (mode_index != 0u) {
            PrefillGRReadMixBuffers remainder_aliased = integrated_buffers;
            remainder_aliased.mixed_bf_lo =
                mode_index == 1u
                    ? integrated_read.xn16_lo.view().slice(0u, mixed_bytes)
                    : integrated_read.gated.view().slice(0u, mixed_bytes);
            const BufferRef aliased_read_output =
                remainder_aliased.mixed_bf_lo;
            const std::vector<std::uint8_t> remainder_alias_before(
                static_cast<const std::uint8_t*>(
                    aliased_read_output.contents),
                static_cast<const std::uint8_t*>(
                    aliased_read_output.contents) +
                    aliased_read_output.length);
            auto remainder_alias_command = context.make_command_buffer(
                label + " active remainder alias rejection", error);
            if (!remainder_alias_command) {
                check(false,
                      label + " active remainder-alias command buffer",
                      error);
                return false;
            }
            const bool remainder_alias_encoded =
                encode_prefill_gr_read_mix(
                    context, *remainder_alias_command, rows,
                    modes[mode_index], remainder_aliased, error);
            const bool remainder_alias_rejected =
                !remainder_alias_encoded &&
                remainder_alias_command->unsafe_native_handle() == nullptr &&
                error.find("mixer output overlaps a GR read output") !=
                    std::string::npos &&
                std::equal(
                    remainder_alias_before.begin(),
                    remainder_alias_before.end(),
                    static_cast<const std::uint8_t*>(
                        aliased_read_output.contents));
            check(remainder_alias_rejected,
                  label + " rejects mixer remainder alias with active " +
                      (mode_index == 1u ? "GR remainder" : "GR output"),
                  error);
            passed &= remainder_alias_rejected;

            if (mode_index == 1u) {
                TestBuffer low_rank_remainder_alias;
                const bool alias_allocated =
                    low_rank_remainder_alias.allocate(
                        context, label + " low-rank remainder alias",
                        mixed_bytes, error);
                check(alias_allocated,
                      label + " allocate low-rank remainder alias", error);
                if (!alias_allocated) return false;
                PrefillGRReadMixBuffers low_rank_remainder_aliased =
                    integrated_buffers;
                low_rank_remainder_aliased.read.low_rank_bf16_lo =
                    low_rank_remainder_alias.view();
                low_rank_remainder_aliased.mixed_bf_lo =
                    low_rank_remainder_alias.view();
                const std::vector<std::uint8_t> alias_before(
                    low_rank_remainder_alias.allocation().begin(),
                    low_rank_remainder_alias.allocation().end());
                auto low_rank_alias_command = context.make_command_buffer(
                    label + " low-rank remainder alias rejection", error);
                if (!low_rank_alias_command) {
                    check(false,
                          label +
                              " low-rank remainder-alias command buffer",
                          error);
                    return false;
                }
                const bool low_rank_alias_encoded =
                    encode_prefill_gr_read_mix(
                        context, *low_rank_alias_command, rows,
                        modes[mode_index], low_rank_remainder_aliased, error);
                const bool low_rank_alias_rejected =
                    !low_rank_alias_encoded &&
                    low_rank_alias_command->unsafe_native_handle() == nullptr &&
                    error.find("mixer output overlaps a GR read output") !=
                        std::string::npos &&
                    std::equal(
                        alias_before.begin(), alias_before.end(),
                        low_rank_remainder_alias.allocation().begin());
                check(low_rank_alias_rejected,
                      label +
                          " rejects mixer remainder alias with low-rank GR remainder",
                      error);
                passed &= low_rank_alias_rejected;
            }
        }
    }

    PrefillGRReadTestOutputs late_failure;
    TestBuffer late_mixed_bf, late_mixed_h;
    const bool late_failure_allocated =
        late_failure.allocate(context, 1u, false, label + " late failure",
                              error) &&
        late_mixed_bf.allocate(context, label + " late mixed BF16",
                               kWidth * sizeof(std::uint16_t), error) &&
        late_mixed_h.allocate(context, label + " late mixed F16",
                              kWidth * sizeof(std::uint16_t), error);
    check(late_failure_allocated, label + " allocate late-failure outputs",
          error);
    if (!late_failure_allocated) return false;
    const PrefillGRReadBuffers late_failure_buffers = late_failure.bind(
        residual.view(), norm.view(), down_read, up_read, inject_read, false);
    const PrefillGRReadMixBuffers late_failure_mix_buffers{
        late_failure_buffers, late_mixed_bf.view(), {}, late_mixed_h.view()};
    const std::array<TestBuffer*, 8> late_failure_outputs{
        &late_failure.rs, &late_failure.xn16, &late_failure.low_rank,
        &late_failure.low_rank_bf16, &late_failure.gated, &late_failure.inject,
        &late_mixed_bf, &late_mixed_h};
    const auto snapshot_late_failure_outputs = [&] {
        std::array<std::vector<std::uint8_t>, 8> snapshots;
        for (std::size_t index = 0u; index < snapshots.size(); ++index) {
            const auto bytes = late_failure_outputs[index]->allocation();
            snapshots[index].assign(bytes.begin(), bytes.end());
        }
        return snapshots;
    };
    const auto late_failure_outputs_unchanged = [&](const auto& snapshots) {
        for (std::size_t index = 0u; index < snapshots.size(); ++index) {
            const auto bytes = late_failure_outputs[index]->allocation();
            if (!std::equal(snapshots[index].begin(), snapshots[index].end(),
                            bytes.begin())) {
                return false;
            }
        }
        return true;
    };
    const auto run_late_failure = [&](std::string_view form,
                                      bool caller_owned, bool include_mix) {
        const auto before = snapshot_late_failure_outputs();
        auto command = context.make_command_buffer(
            label + " late failure " + std::string(form), error);
        if (!command) {
            check(false, label + " late-failure command buffer", error);
            return false;
        }
        bool encoded = false;
        if (caller_owned) {
            encoded = Encoder::encode(
                context, *command, "caller-owned Prefill GR read/mix failure",
                [&](Encoder& encoder, std::string& encode_error) {
                    return include_mix
                               ? encode_prefill_gr_read_mix(
                                     context, encoder, 1u,
                                     PrefillBF16X2Mode::off,
                                     late_failure_mix_buffers, encode_error)
                               : encode_prefill_gr_read(
                                     context, encoder, 1u,
                                     PrefillBF16X2Mode::off,
                                     late_failure_buffers, encode_error);
                },
                error);
        } else {
            encoded = include_mix
                          ? encode_prefill_gr_read_mix(
                                context, *command, 1u,
                                PrefillBF16X2Mode::off,
                                late_failure_mix_buffers, error)
                          : encode_prefill_gr_read(
                                context, *command, 1u,
                                PrefillBF16X2Mode::off,
                                late_failure_buffers, error);
        }
        const bool discarded =
            command->unsafe_native_handle() == nullptr;
        const bool unchanged =
            late_failure_outputs_unchanged(before);
        const bool rejected =
            !encoded && discarded && unchanged &&
            error.find("STRATA_METAL_GEMM must be") != std::string::npos;
        check(rejected,
              label + " " + std::string(form) +
                  (include_mix ? " read/mix" : " read") +
                  " discards normalization prefix after GEMM selector failure",
              error + (discarded ? " (command buffer discarded)"
                                 : " (command buffer remained submit-capable)"));
        return rejected;
    };

    ScopedEnvironment invalid_gemm_setting;
    const bool invalid_setting_applied =
        invalid_gemm_setting.set("STRATA_METAL_GEMM", "invalid");
    check(invalid_setting_applied,
          label + " installs deterministic late GEMM failure");
    if (invalid_setting_applied) {
        passed &= run_late_failure("transactional", false, false);
        passed &= run_late_failure("caller-owned", true, false);
        passed &= run_late_failure("transactional", false, true);
        passed &= run_late_failure("caller-owned", true, true);
        const bool environment_restored = invalid_gemm_setting.restore();
        check(environment_restored,
              label + " restores STRATA_METAL_GEMM after late-failure checks");
        passed &= environment_restored;
    }

    TestBuffer late_mix_shared;
    const std::size_t late_mix_output_bytes =
        static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t);
    const bool late_mix_shared_allocated =
        late_mix_shared.allocate(context, label + " late mixer overlap",
                                 late_mix_output_bytes * 2u, error);
    check(late_mix_shared_allocated,
          label + " allocate late mixer-overlap fixture", error);
    if (!late_mix_shared_allocated) return false;
    const PrefillGRReadMixBuffers late_mix_overlap_buffers{
        late_failure_buffers,
        late_mix_shared.view(),
        {},
        late_mix_shared.view().slice(late_mix_output_bytes,
                                     late_mix_output_bytes)};
    // The active mixer outputs are adjacent, but the first view's unused tail
    // overlaps the second view. The read prefix encodes before the mixer
    // performs its stricter full-view overlap validation.
    const auto run_late_mix_failure = [&](std::string_view form,
                                          bool caller_owned) {
        const auto before = snapshot_late_failure_outputs();
        const std::vector<std::uint8_t> shared_before(
            late_mix_shared.allocation().begin(),
            late_mix_shared.allocation().end());
        auto command = context.make_command_buffer(
            label + " late mixer validation " + std::string(form), error);
        if (!command) {
            check(false, label + " late mixer-validation command buffer",
                  error);
            return false;
        }
        const auto encode_body = [&](Encoder& encoder,
                                     std::string& encode_error) {
            return encode_prefill_gr_read_mix(
                context, encoder, 1u, PrefillBF16X2Mode::off,
                late_mix_overlap_buffers, encode_error);
        };
        const bool encoded =
            caller_owned
                ? Encoder::encode(context, *command,
                                  "caller-owned Prefill late mixer failure",
                                  encode_body, error)
                : encode_prefill_gr_read_mix(
                      context, *command, 1u, PrefillBF16X2Mode::off,
                      late_mix_overlap_buffers, error);
        const bool rejected =
            !encoded && command->unsafe_native_handle() == nullptr &&
            late_failure_outputs_unchanged(before) &&
            std::equal(shared_before.begin(), shared_before.end(),
                       late_mix_shared.allocation().begin()) &&
            error.find("Prefill GR mix output buffer views overlap") !=
                std::string::npos;
        check(rejected,
              label + " " + std::string(form) +
                  " read/mix failure after GR read discards its encoded prefix",
              error);
        return rejected;
    };
    passed &= run_late_mix_failure("transactional", false);
    passed &= run_late_mix_failure("caller-owned", true);

    const bool immutable_inputs =
        std::equal(residual_before.begin(), residual_before.end(),
                   residual.payload().begin()) &&
        std::equal(norm_before.begin(), norm_before.end(),
                   norm.payload().begin()) &&
        std::equal(down_weights_before.begin(), down_weights_before.end(),
                   down_weights.payload().begin()) &&
        std::equal(up_weights_before.begin(), up_weights_before.end(),
                   up_weights.payload().begin()) &&
        std::equal(inject_weights_before.begin(), inject_weights_before.end(),
                   inject_weights.payload().begin());
    check(immutable_inputs, label + " keeps all model inputs immutable");
    passed &= immutable_inputs;
    return passed;
}

bool run_silu_case(Context& context, std::uint32_t rows,
                   PrefillBF16X2Mode mode, bool shared_encoder) {
    constexpr std::array<float, 19> edge_values{
        -2048.0f, -160.0f, -80.0f, -40.0f, -12.0f, -4.0f, -1.0f,
        -0.5f, -0.001f, -0.0f, 0.0f, 0.001f, 0.5f, 1.0f, 4.0f,
        12.0f, 40.0f, 80.0f, 2048.0f};
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::string label = "Prefill GR SiLU T=" + std::to_string(rows) +
        " mode=" + std::to_string(static_cast<std::uint32_t>(mode)) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t values = static_cast<std::size_t>(rows) * 320u;
    const std::size_t input_bytes = values * sizeof(float);
    const std::size_t output_bytes = values * sizeof(std::uint16_t);
    std::vector<float> low_rank(values);
    for (std::size_t index = 0u; index < values; ++index) {
        low_rank[index] = edge_values[(index * 7u + rows * 3u) %
                                      edge_values.size()];
    }
    const SiluOracle expected = prefill_gr_silu_oracle(low_rank, write_low);

    TestBuffer input, high, low;
    std::string error;
    const bool allocated =
        input.allocate(context, label + " low-rank input", input_bytes, error) &&
        high.allocate(context, label + " high output", output_bytes, error) &&
        (!write_low ||
         low.allocate(context, label + " low output", output_bytes, error));
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(input.payload().data(), low_rank.data(), input_bytes);
    const std::vector<std::uint8_t> input_before(
        input.allocation().begin(), input.allocation().end());

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const PrefillGRSiluBuffers buffers{
        context.buffers().read_only(input.view()), high.view(),
        write_low ? low.view() : BufferRef{}};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared Prefill GR SiLU",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gr_silu(context, encoder, rows, mode,
                                                buffers, encode_error);
              },
              error)
        : encode_prefill_gr_silu(context, *command, rows, mode, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto actual_high = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(high.payload().data()), values);
    std::uint16_t max_high_ulp = 0u;
    for (std::size_t index = 0u; index < values; ++index) {
        max_high_ulp = std::max(
            max_high_ulp,
            bf16_ulp_distance(actual_high[index], expected.high[index]));
    }
    check(max_high_ulp <= 1u,
          label + " high BF16 matches SiLU/RNE oracle",
          "max ULP=" + std::to_string(max_high_ulp) + ", limit=1");

    bool low_matches = true;
    if (write_low) {
        const auto actual_low = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(low.payload().data()),
            values);
        std::uint16_t max_low_ulp = 0u;
        for (std::size_t index = 0u; index < values; ++index) {
            max_low_ulp = std::max(
                max_low_ulp,
                bf16_ulp_distance(actual_low[index], expected.low[index]));
        }
        low_matches = max_low_ulp <= 2u;
        check(low_matches,
              label + " low BF16 remainder matches SiLU/RNE oracle",
              "max ULP=" + std::to_string(max_low_ulp) + ", limit=2");
        check(low.guards_intact(), label + " low output guards");
    }
    const bool input_unchanged =
        std::equal(input_before.begin(), input_before.end(),
                   input.allocation().begin());
    check(input_unchanged, label + " low-rank input remains read-only");
    check(input.guards_intact(), label + " input guards");
    check(high.guards_intact(), label + " high output guards");
    return max_high_ulp <= 1u && low_matches && input_unchanged;
}

bool test_silu_rne_boundaries(Context& context,
                              PrefillBF16X2Mode mode) {
    constexpr std::array<float, 8> midpoint_inputs{
        257.0f, 259.0f, 261.0f, 263.0f,
        256.501953125f, 256.505859375f,
        259.498046875f, 259.494140625f};
    constexpr std::array<std::uint16_t, 8> expected_high{
        0x4280u, 0x4282u, 0x4282u, 0x4284u,
        0x4280u, 0x4280u, 0x4282u, 0x4282u};
    constexpr std::array<std::uint16_t, 8> expected_low{
        0x3e80u, 0xbe80u, 0x3e80u, 0xbe80u,
        0x3e00u, 0x3e02u, 0xbe00u, 0xbe02u};
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::string label =
        "Prefill GR SiLU exact BF16 RNE midpoint mode=" +
        std::to_string(static_cast<std::uint32_t>(mode));
    constexpr std::size_t kValues = 320u;
    const std::size_t input_bytes = kValues * sizeof(float);
    const std::size_t output_bytes = kValues * sizeof(std::uint16_t);
    std::vector<float> low_rank(kValues);
    for (std::size_t index = 0u; index < kValues; ++index) {
        low_rank[index] = midpoint_inputs[index % midpoint_inputs.size()];
    }
    const bool oracle_bits_match =
        strata::kernels::bf16_from_f32(64.25f) == expected_high[0] &&
        strata::kernels::bf16_from_f32(64.75f) == expected_high[1] &&
        strata::kernels::bf16_from_f32(65.25f) == expected_high[2] &&
        strata::kernels::bf16_from_f32(65.75f) == expected_high[3] &&
        strata::kernels::bf16_from_f32(0.25f) == expected_low[0] &&
        strata::kernels::bf16_from_f32(-0.25f) == expected_low[1] &&
        strata::kernels::bf16_from_f32(0.12548828125f) == expected_low[4] &&
        strata::kernels::bf16_from_f32(0.12646484375f) == expected_low[5] &&
        strata::kernels::bf16_from_f32(-0.12548828125f) ==
            expected_low[6] &&
        strata::kernels::bf16_from_f32(-0.12646484375f) ==
            expected_low[7];
    check(oracle_bits_match, label + " fixed RNE expectations");
    if (!oracle_bits_match) return false;

    TestBuffer input, high, low;
    std::string error;
    const bool allocated =
        input.allocate(context, label + " input", input_bytes, error) &&
        high.allocate(context, label + " high", output_bytes, error) &&
        (!write_low ||
         low.allocate(context, label + " low", output_bytes, error));
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(input.payload().data(), low_rank.data(), input_bytes);
    const PrefillGRSiluBuffers buffers{
        context.buffers().read_only(input.view()), high.view(),
        write_low ? low.view() : BufferRef{}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_prefill_gr_silu(
        context, *command, 1u, mode, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto actual_high = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(high.payload().data()),
        kValues);
    bool high_matches = true;
    bool low_matches = true;
    std::string high_mismatch;
    for (std::size_t index = 0u; index < kValues; ++index) {
        const std::size_t tie = index % expected_high.size();
        if (actual_high[index] != expected_high[tie]) {
            high_matches = false;
            if (high_mismatch.empty()) {
                char detail[96]{};
                std::snprintf(detail, sizeof(detail),
                              "index=%zu actual=0x%04x expected=0x%04x",
                              index, actual_high[index], expected_high[tie]);
                high_mismatch = detail;
            }
        }
    }
    check(high_matches, label + " high BF16 bits match RNE ties exactly",
          high_mismatch);
    if (write_low) {
        const auto actual_low = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(low.payload().data()),
            kValues);
        std::string low_mismatch;
        for (std::size_t index = 0u; index < kValues; ++index) {
            const std::size_t tie = index % expected_low.size();
            if (actual_low[index] != expected_low[tie]) {
                low_matches = false;
                if (low_mismatch.empty()) {
                    char detail[96]{};
                    std::snprintf(detail, sizeof(detail),
                                  "index=%zu actual=0x%04x expected=0x%04x",
                                  index, actual_low[index],
                                  expected_low[tie]);
                    low_mismatch = detail;
                }
            }
        }
        check(low_matches, label + " low BF16 remainder bits match exactly",
              low_mismatch);
        check(low.guards_intact(), label + " low output guards");
    }
    check(input.guards_intact() && high.guards_intact(),
          label + " input and high output guards");
    return high_matches && low_matches;
}

bool reject_silu_call(Context& context, std::uint32_t rows,
                      PrefillBF16X2Mode mode,
                      const PrefillGRSiluBuffers& buffers,
                      std::string_view expected_error,
                      const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_prefill_gr_silu(context, *command, rows, mode,
                                                buffers, error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_silu_encoder_validation(Context& context,
                                  Context& foreign_context) {
    const std::string label = "Prefill GR SiLU validation";
    std::string error;
    TestBuffer input, high, low, foreign_input;
    constexpr std::size_t kInputBytes = 320u * sizeof(float);
    constexpr std::size_t kOutputBytes = 320u * sizeof(std::uint16_t);
    constexpr std::size_t kOutputStorageBytes =
        kOutputBytes + sizeof(std::uint16_t);
    const bool allocated =
        input.allocate(context, label + " input", kInputBytes, error) &&
        high.allocate(context, label + " high", kOutputStorageBytes, error) &&
        low.allocate(context, label + " low", kOutputStorageBytes, error) &&
        foreign_input.allocate(foreign_context, label + " foreign input",
                               kInputBytes, error);
    check(allocated, label + " allocate validation buffers", error);
    if (!allocated) return false;

    const ReadOnlyBufferRef input_read_only =
        context.buffers().read_only(input.view());
    const PrefillGRSiluBuffers valid{
        input_read_only, high.view(), low.view()};
    bool passed = true;
    passed &= reject_silu_call(
        context, 0u, PrefillBF16X2Mode::off,
        {input_read_only, high.view(), {}}, "row count",
        label + " rejects zero rows");
    passed &= reject_silu_call(
        context, kPrefillMaxShaderRows + 1u, PrefillBF16X2Mode::off,
        {input_read_only, high.view(), {}}, "row count",
        label + " rejects rows beyond shader range");
    passed &= reject_silu_call(
        context, 1u, static_cast<PrefillBF16X2Mode>(3u),
        {input_read_only, high.view(), {}}, "mode is invalid",
        label + " rejects unknown mode");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {input_read_only, high.view(), {}}, "requires low_bf16",
        label + " requires mode-1 remainder");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off, valid,
        "only valid in mode 1", label + " rejects unused mode-0 remainder");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::except_gr, valid,
        "only valid in mode 1", label + " rejects unused mode-2 remainder");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {context.buffers().read_only(
             input.view().slice(0u, kInputBytes - sizeof(float))),
         high.view(), {}},
        "undersized", label + " rejects short low-rank input");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {context.buffers().read_only(
             input.view().slice(sizeof(std::uint16_t),
                                kInputBytes - sizeof(std::uint16_t))),
         high.view(), {}},
        "misaligned", label + " rejects misaligned low-rank input");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {input_read_only,
         high.view().slice(0u, kOutputBytes - sizeof(std::uint16_t)), {}},
        "undersized", label + " rejects short high output");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {input_read_only, high.view().slice(1u, kOutputBytes), {}},
        "misaligned", label + " rejects misaligned high output");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {input_read_only, high.view(),
         low.view().slice(0u, kOutputBytes - sizeof(std::uint16_t))},
        "undersized", label + " rejects short low output");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {input_read_only, high.view(), low.view().slice(1u, kOutputBytes)},
        "misaligned", label + " rejects misaligned low output");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {foreign_context.buffers().read_only(foreign_input.view()),
         high.view(), {}},
        "not owned by this Context",
        label + " rejects a foreign read-only view");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {input_read_only, high.view(), high.view()},
        "output buffer views overlap",
        label + " rejects overlapping outputs");
    passed &= reject_silu_call(
        context, 1u, PrefillBF16X2Mode::off,
        {context.buffers().read_only(input.view()),
         input.view().slice(0u, kOutputBytes), {}},
        "overlaps a writable buffer view",
        label + " rejects input/output overlap");
    return passed;
}

bool test_silu_failure_discards_prior_dispatch(Context& context) {
    const std::string label = "Prefill GR SiLU failure discards earlier dispatch";
    std::string error;
    TestBuffer input, high;
    const bool allocated =
        input.allocate(context, label + " input", 320u * sizeof(float), error) &&
        high.allocate(context, label + " high", 320u * sizeof(std::uint16_t),
                      error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::fill(input.payload().begin(), input.payload().end(), 0u);
    const std::vector<std::uint8_t> high_before(
        high.allocation().begin(), high.allocation().end());
    const PrefillGRSiluBuffers valid{
        context.buffers().read_only(input.view()), high.view(), {}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GR SiLU rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gr_silu(context, encoder, 1u,
                                        PrefillBF16X2Mode::off, valid,
                                        encode_error)) {
                return false;
            }
            return encode_prefill_gr_silu(
                context, encoder, 0u, PrefillBF16X2Mode::off, valid,
                encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(high_before.begin(), high_before.end(),
                   high.allocation().begin());
    check(unchanged, label,
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));
    return unchanged;
}

bool test_silu_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label = "Prefill GR SiLU rejects open concurrent group";
    std::string error;
    TestBuffer input, high;
    const bool allocated =
        input.allocate(context, label + " input", 320u * sizeof(float), error) &&
        high.allocate(context, label + " high",
                      320u * sizeof(std::uint16_t), error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGRSiluBuffers buffers{
        context.buffers().read_only(input.view()), high.view(), {}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GR SiLU",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gr_silu(context, encoder, 1u,
                                          PrefillBF16X2Mode::off, buffers,
                                          encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

struct GRMixOracle {
    std::vector<std::uint16_t> high;
    std::vector<std::uint16_t> half;
    std::vector<float> values;
    std::vector<float> magnitudes;
};

GRMixOracle prefill_gr_mix_oracle(std::span<const float> residual,
                                  std::span<const float> rs,
                                  std::span<const float> norm,
                                  std::span<const float> gated,
                                  std::uint32_t rows) {
    const std::size_t values = static_cast<std::size_t>(rows) * kWidth;
    GRMixOracle result;
    result.high.resize(values);
    result.half.resize(values);
    result.values.resize(values);
    result.magnitudes.resize(values);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < kWidth; ++column) {
            float sum = 0.0f;
            float magnitude = 0.0f;
            for (std::uint32_t stream = 0u; stream < kStreams; ++stream) {
                const std::size_t input_index =
                    (static_cast<std::size_t>(row) * kStreams + stream) *
                    kWidth +
                    column;
                const float value =
                    (residual[input_index] * rs[row * kStreams + stream]) *
                    norm[stream * kWidth + column];
                const float gate =
                    1.0f / (1.0f + std::exp(-gated[input_index]));
                sum = std::fma(value, gate, sum);
                magnitude += std::abs(value * gate);
            }
            sum /= 4.0f;
            magnitude /= 4.0f;
            const std::size_t output_index =
                static_cast<std::size_t>(row) * kWidth + column;
            result.values[output_index] = sum;
            result.magnitudes[output_index] = magnitude;
            const std::uint16_t high =
                strata::kernels::bf16_from_f32(sum);
            result.high[output_index] = high;
            result.half[output_index] =
                strata::kernels::f16_from_f32(sum);
        }
    }
    return result;
}

void fill_gr_mix_inputs(std::vector<float>& residual, std::vector<float>& rs,
                        std::vector<float>& norm, std::vector<float>& gated,
                        std::uint32_t rows) {
    for (std::size_t index = 0u; index < residual.size(); ++index) {
        const int centered =
            static_cast<int>((index * 13u + index / kWidth * 7u) % 97u) - 48;
        residual[index] = static_cast<float>(centered) * 0.03125f;
        const int gate_centered =
            static_cast<int>((index * 5u + index / kWidth * 11u) % 41u) - 20;
        gated[index] = static_cast<float>(gate_centered) * 0.5f;
    }
    for (std::size_t index = 0u; index < rs.size(); ++index) {
        rs[index] = 0.75f + static_cast<float>(index % kStreams) * 0.125f +
                    static_cast<float>(index / kStreams) * 0.015625f;
    }
    for (std::size_t index = 0u; index < norm.size(); ++index) {
        const int centered = static_cast<int>((index * 17u + rows) % 33u) - 16;
        norm[index] = 0.9375f + static_cast<float>(centered) * 0.0078125f;
    }
}

bool run_gr_mix_case(Context& context, std::uint32_t rows,
                     PrefillBF16X2Mode mode, bool shared_encoder) {
    const bool write_low = mode != PrefillBF16X2Mode::off;
    const std::string label = "Prefill GR mix T=" + std::to_string(rows) +
        " mode=" + std::to_string(static_cast<std::uint32_t>(mode)) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t stream_values =
        static_cast<std::size_t>(rows) * kStreams * kWidth;
    const std::size_t output_values = static_cast<std::size_t>(rows) * kWidth;
    const std::size_t stream_bytes = stream_values * sizeof(float);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(rows) * kStreams * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * sizeof(float);
    const std::size_t output_bytes =
        output_values * sizeof(std::uint16_t);
    std::vector<float> residual(stream_values);
    std::vector<float> rs(static_cast<std::size_t>(rows) * kStreams);
    std::vector<float> norm(kStreams * kWidth);
    std::vector<float> gated(stream_values);
    fill_gr_mix_inputs(residual, rs, norm, gated, rows);
    const GRMixOracle expected =
        prefill_gr_mix_oracle(residual, rs, norm, gated, rows);

    TestBuffer residual_buffer, rs_buffer, norm_buffer, gated_buffer;
    TestBuffer high_buffer, low_buffer, half_buffer;
    std::string error;
    const bool allocated =
        residual_buffer.allocate(context, label + " residual", stream_bytes,
                                 error) &&
        rs_buffer.allocate(context, label + " scale", scale_bytes, error) &&
        norm_buffer.allocate(context, label + " norm", norm_bytes, error) &&
        gated_buffer.allocate(context, label + " gated", stream_bytes, error) &&
        high_buffer.allocate(context, label + " mixed_bf", output_bytes,
                             error) &&
        (!write_low ||
         low_buffer.allocate(context, label + " mixed_bf_lo", output_bytes,
                             error)) &&
        half_buffer.allocate(context, label + " mixed_h", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(residual_buffer.payload().data(), residual.data(), stream_bytes);
    std::memcpy(rs_buffer.payload().data(), rs.data(), scale_bytes);
    std::memcpy(norm_buffer.payload().data(), norm.data(), norm_bytes);
    std::memcpy(gated_buffer.payload().data(), gated.data(), stream_bytes);
    const std::array<std::vector<std::uint8_t>, 4> input_before{{
        {residual_buffer.allocation().begin(),
         residual_buffer.allocation().end()},
        {rs_buffer.allocation().begin(), rs_buffer.allocation().end()},
        {norm_buffer.allocation().begin(), norm_buffer.allocation().end()},
        {gated_buffer.allocation().begin(), gated_buffer.allocation().end()},
    }};

    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const PrefillGRMixBuffers buffers{
        context.buffers().read_only(residual_buffer.view()),
        context.buffers().read_only(rs_buffer.view()),
        context.buffers().read_only(norm_buffer.view()),
        context.buffers().read_only(gated_buffer.view()),
        high_buffer.view(),
        write_low ? low_buffer.view() : BufferRef{}, half_buffer.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared Prefill GR mix",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gr_mix(context, encoder, rows, mode,
                                               buffers, encode_error);
              },
              error)
        : encode_prefill_gr_mix(context, *command, rows, mode, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto actual_high = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(high_buffer.payload().data()),
        output_values);
    const auto actual_half = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(half_buffer.payload().data()),
        output_values);
    std::uint16_t max_high_ulp = 0u;
    std::uint16_t max_half_ulp = 0u;
    for (std::size_t index = 0u; index < output_values; ++index) {
        max_high_ulp = std::max(
            max_high_ulp, bf16_ulp_distance(actual_high[index],
                                            expected.high[index]));
        max_half_ulp = std::max(
            max_half_ulp, bf16_ulp_distance(actual_half[index],
                                            expected.half[index]));
    }
    check(max_high_ulp <= 1u, label + " BF16 high output oracle",
          "max ULP=" + std::to_string(max_high_ulp) + ", limit=1");
    check(max_half_ulp <= 2u, label + " F16 output oracle",
          "max ULP=" + std::to_string(max_half_ulp) + ", limit=2");
    bool low_matches = true;
    if (write_low) {
        const auto actual_low = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(low_buffer.payload().data()),
            output_values);
        constexpr float kBf16x2RelativeTolerance = 0x1p-15f;
        float max_reconstruction_error = 0.0f;
        float max_allowed_error = 0.0f;
        std::size_t improved_count = 0u;
        bool remainder_matches = true;
        for (std::size_t index = 0u; index < output_values; ++index) {
            const float high_value =
                strata::kernels::f32_from_bf16(actual_high[index]);
            const float low_value =
                strata::kernels::f32_from_bf16(actual_low[index]);
            const float reconstructed = high_value + low_value;
            const float reconstruction_error =
                std::abs(reconstructed - expected.values[index]);
            const float allowed_error =
                kBf16x2RelativeTolerance * expected.magnitudes[index] +
                0x1p-126f;
            max_reconstruction_error =
                std::max(max_reconstruction_error, reconstruction_error);
            max_allowed_error = std::max(max_allowed_error, allowed_error);
            remainder_matches &= reconstruction_error <= allowed_error;
            if (std::abs(high_value - expected.values[index]) >
                allowed_error) {
                ++improved_count;
            }
        }
        low_matches = remainder_matches && improved_count > output_values / 4u;
        check(low_matches, label + " BF16X2 reconstruction oracle",
              "max absolute error=" +
                  std::to_string(max_reconstruction_error) +
                  ", max allowed=" + std::to_string(max_allowed_error) +
                  ", high-only values improved=" +
                  std::to_string(improved_count) + "/" +
                  std::to_string(output_values));
        check(low_buffer.guards_intact(), label + " remainder guards");
    }
    const std::array<const TestBuffer*, 4> input_buffers{
        &residual_buffer, &rs_buffer, &norm_buffer, &gated_buffer};
    bool inputs_unchanged = true;
    for (std::size_t index = 0u; index < input_buffers.size(); ++index) {
        const auto allocation = input_buffers[index]->allocation();
        inputs_unchanged &=
            std::equal(input_before[index].begin(), input_before[index].end(),
                       allocation.begin());
        check(input_buffers[index]->guards_intact(),
              label + " input guards " + std::to_string(index));
    }
    check(inputs_unchanged, label + " inputs remain read-only");
    check(high_buffer.guards_intact(), label + " BF16 output guards");
    check(half_buffer.guards_intact(), label + " F16 output guards");
    return max_high_ulp <= 1u && max_half_ulp <= 2u && low_matches &&
           inputs_unchanged;
}

bool test_gr_mix_rne_boundaries(Context& context,
                                PrefillBF16X2Mode mode) {
    constexpr std::array<float, 8> sums{
        64.25f, 64.75f, 65.25f, 65.75f,
        64.12548828125f, 64.12646484375f,
        64.87451171875f, 64.87353515625f};
    constexpr std::array<std::uint16_t, 8> expected_high{
        0x4280u, 0x4282u, 0x4282u, 0x4284u,
        0x4280u, 0x4280u, 0x4282u, 0x4282u};
    constexpr std::array<std::uint16_t, 8> expected_low{
        0x3e80u, 0xbe80u, 0x3e80u, 0xbe80u,
        0x3e00u, 0x3e02u, 0xbe00u, 0xbe02u};
    const bool write_low = mode != PrefillBF16X2Mode::off;
    const std::string label =
        "Prefill GR mix exact BF16 RNE ties mode=" +
        std::to_string(static_cast<std::uint32_t>(mode));
    bool oracle_bits_match = true;
    for (std::size_t index = 0u; index < sums.size(); ++index) {
        const std::uint16_t high =
            strata::kernels::bf16_from_f32(sums[index]);
        oracle_bits_match &= high == expected_high[index];
        if (write_low) {
            const float remainder =
                sums[index] - strata::kernels::f32_from_bf16(high);
            oracle_bits_match &=
                strata::kernels::bf16_from_f32(remainder) ==
                expected_low[index];
        }
    }
    check(oracle_bits_match, label + " fixed oracle bits");
    if (!oracle_bits_match) return false;

    const std::size_t stream_values =
        static_cast<std::size_t>(kStreams) * kWidth;
    const std::size_t input_bytes = stream_values * sizeof(float);
    const std::size_t scale_bytes =
        static_cast<std::size_t>(kStreams) * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(kWidth) * sizeof(std::uint16_t);
    std::vector<float> residual(stream_values, 0.0f);
    std::vector<float> rs(kStreams, 1.0f);
    std::vector<float> norm(stream_values, 1.0f);
    std::vector<float> gated(stream_values, 0.0f);
    for (std::uint32_t column = 0u; column < kWidth; ++column) {
        const float twice_sum = 2.0f * sums[column % sums.size()];
        for (std::uint32_t stream = 0u; stream < kStreams; ++stream) {
            residual[static_cast<std::size_t>(stream) * kWidth + column] =
                twice_sum;
        }
    }

    TestBuffer residual_buffer, rs_buffer, norm_buffer, gated_buffer;
    TestBuffer high_buffer, low_buffer, half_buffer;
    std::string error;
    const bool allocated =
        residual_buffer.allocate(context, label + " residual", input_bytes,
                                 error) &&
        rs_buffer.allocate(context, label + " scale", scale_bytes, error) &&
        norm_buffer.allocate(context, label + " norm", input_bytes, error) &&
        gated_buffer.allocate(context, label + " gated", input_bytes, error) &&
        high_buffer.allocate(context, label + " high", output_bytes, error) &&
        (!write_low ||
         low_buffer.allocate(context, label + " low", output_bytes, error)) &&
        half_buffer.allocate(context, label + " half", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(residual_buffer.payload().data(), residual.data(), input_bytes);
    std::memcpy(rs_buffer.payload().data(), rs.data(), scale_bytes);
    std::memcpy(norm_buffer.payload().data(), norm.data(), input_bytes);
    std::memcpy(gated_buffer.payload().data(), gated.data(), input_bytes);

    const PrefillGRMixBuffers buffers{
        context.buffers().read_only(residual_buffer.view()),
        context.buffers().read_only(rs_buffer.view()),
        context.buffers().read_only(norm_buffer.view()),
        context.buffers().read_only(gated_buffer.view()),
        high_buffer.view(), write_low ? low_buffer.view() : BufferRef{},
        half_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded =
        encode_prefill_gr_mix(context, *command, 1u, mode, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto actual_high = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(high_buffer.payload().data()),
        kWidth);
    const auto actual_half = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(half_buffer.payload().data()),
        kWidth);
    std::span<const std::uint16_t> actual_low;
    if (write_low) {
        actual_low = std::span<const std::uint16_t>(
            reinterpret_cast<const std::uint16_t*>(
                low_buffer.payload().data()),
            kWidth);
    }
    bool high_matches = true;
    bool half_matches = true;
    bool low_matches = true;
    std::string high_mismatch;
    std::string low_mismatch;
    for (std::size_t column = 0u; column < kWidth; ++column) {
        const std::size_t tie = column % sums.size();
        if (actual_high[column] != expected_high[tie]) {
            high_matches = false;
            if (high_mismatch.empty()) {
                char detail[96]{};
                std::snprintf(detail, sizeof(detail),
                              "column=%zu actual=0x%04x expected=0x%04x",
                              column, actual_high[column],
                              expected_high[tie]);
                high_mismatch = detail;
            }
        }
        half_matches &=
            actual_half[column] ==
            strata::kernels::f16_from_f32(sums[tie]);
        if (write_low && actual_low[column] != expected_low[tie]) {
            low_matches = false;
            if (low_mismatch.empty()) {
                char detail[96]{};
                std::snprintf(detail, sizeof(detail),
                              "column=%zu actual=0x%04x expected=0x%04x",
                              column, actual_low[column], expected_low[tie]);
                low_mismatch = detail;
            }
        }
    }
    check(high_matches, label + " high BF16 ties match RNE exactly",
          high_mismatch);
    check(half_matches, label + " F16 matches scalar conversion exactly");
    if (write_low) {
        check(low_matches, label + " low BF16 ties match RNE exactly",
              low_mismatch);
    }
    check(residual_buffer.guards_intact() && rs_buffer.guards_intact() &&
              norm_buffer.guards_intact() && gated_buffer.guards_intact() &&
              high_buffer.guards_intact() && half_buffer.guards_intact() &&
              (!write_low || low_buffer.guards_intact()),
          label + " input and output guards");
    return high_matches && half_matches && low_matches;
}

bool reject_gr_mix_call(Context& context, std::uint32_t rows,
                        PrefillBF16X2Mode mode,
                        const PrefillGRMixBuffers& buffers,
                        std::string_view expected_error,
                        const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gr_mix(context, *command, rows, mode, buffers, error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_gr_mix_encoder_validation(Context& context,
                                    Context& foreign_context) {
    const std::string label = "Prefill GR mix validation";
    std::string error;
    constexpr std::size_t kStreamBytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t kScaleBytes = kStreams * sizeof(float);
    constexpr std::size_t kNormBytes = kStreams * kWidth * sizeof(float);
    constexpr std::size_t kOutputBytes = kWidth * sizeof(std::uint16_t);
    TestBuffer residual, rs, norm, gated, high, low, half, foreign_residual,
        foreign_high;
    const bool allocated =
        residual.allocate(context, label + " residual", kStreamBytes, error) &&
        rs.allocate(context, label + " scale", kScaleBytes + sizeof(float),
                    error) &&
        norm.allocate(context, label + " norm", kNormBytes, error) &&
        gated.allocate(context, label + " gated", kStreamBytes, error) &&
        high.allocate(context, label + " high",
                      kOutputBytes + sizeof(std::uint16_t), error) &&
        low.allocate(context, label + " low",
                     kOutputBytes + sizeof(std::uint16_t), error) &&
        half.allocate(context, label + " half",
                      kOutputBytes + sizeof(std::uint16_t), error) &&
        foreign_residual.allocate(foreign_context, label + " foreign residual",
                                  kStreamBytes, error) &&
        foreign_high.allocate(foreign_context, label + " foreign high",
                              kOutputBytes, error);
    check(allocated, label + " allocate validation buffers", error);
    if (!allocated) return false;

    const ReadOnlyBufferRef residual_ro =
        context.buffers().read_only(residual.view());
    const ReadOnlyBufferRef rs_ro = context.buffers().read_only(rs.view());
    const ReadOnlyBufferRef norm_ro = context.buffers().read_only(norm.view());
    const ReadOnlyBufferRef gated_ro =
        context.buffers().read_only(gated.view());
    const PrefillGRMixBuffers valid{
        residual_ro, rs_ro, norm_ro, gated_ro,
        high.view(), low.view(), half.view()};
    bool passed = true;
    passed &= reject_gr_mix_call(
        context, 0u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, half.view()},
        "row count", label + " rejects zero rows");
    passed &= reject_gr_mix_call(
        context, kPrefillMaxShaderRows + 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, half.view()},
        "row count", label + " rejects rows beyond shader range");
    passed &= reject_gr_mix_call(
        context, 1u, static_cast<PrefillBF16X2Mode>(3u),
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, half.view()},
        "mode is invalid", label + " rejects unknown mode");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, half.view()},
        "require mixed_bf_lo", label + " mode 1 requires remainder");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::except_gr,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, half.view()},
        "require mixed_bf_lo", label + " mode 2 requires remainder");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off, valid,
        "only valid in modes 1 and 2",
        label + " mode 0 rejects remainder");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro.slice(0u, kStreamBytes - sizeof(float)), rs_ro,
         norm_ro, gated_ro, high.view(), {}, half.view()},
        "undersized", label + " rejects short residual");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, context.buffers().read_only(
                          rs.view().slice(1u, kScaleBytes)),
         norm_ro, gated_ro, high.view(), {}, half.view()},
        "misaligned", label + " rejects misaligned scale");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro,
         context.buffers().read_only(
             norm.view().slice(0u, kNormBytes - sizeof(float))),
         gated_ro, high.view(), {}, half.view()},
        "undersized", label + " rejects short norm");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro,
         context.buffers().read_only(
             gated.view().slice(0u, kStreamBytes - sizeof(float))),
         high.view(), {}, half.view()},
        "undersized", label + " rejects short gated input");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro,
         high.view().slice(0u, kOutputBytes - sizeof(std::uint16_t)), {},
         half.view()},
        "undersized", label + " rejects short BF16 output");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro,
         high.view().slice(1u, kOutputBytes), {}, half.view()},
        "misaligned", label + " rejects misaligned BF16 output");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(),
         low.view().slice(1u, kOutputBytes), half.view()},
        "misaligned", label + " rejects misaligned BF16 remainder");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(),
         low.view().slice(0u, kOutputBytes - sizeof(std::uint16_t)),
         half.view()},
        "undersized", label + " rejects short BF16 remainder");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {},
         half.view().slice(0u, kOutputBytes - sizeof(std::uint16_t))},
        "undersized", label + " rejects short F16 output");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {},
         half.view().slice(1u, kOutputBytes)},
        "misaligned", label + " rejects misaligned F16 output");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), {}, high.view()},
        "output buffer views overlap",
        label + " rejects overlapping output views");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), high.view(),
         half.view()},
        "output buffer views overlap",
        label + " rejects high/remainder output alias");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(), half.view(),
         half.view()},
        "output buffer views overlap",
        label + " rejects half/remainder output alias");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual_ro, rs_ro, norm_ro, gated_ro, high.view(),
         residual.view().slice(0u, kOutputBytes), half.view()},
        "read-only input overlaps a writable buffer view",
        label + " rejects remainder/input overlap");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {context.buffers().read_only(residual.view()), rs_ro, norm_ro, gated_ro,
         residual.view().slice(0u, kOutputBytes), {}, half.view()},
        "overlaps a writable buffer view",
        label + " rejects input/output overlap");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {foreign_context.buffers().read_only(foreign_residual.view()), rs_ro,
         norm_ro, gated_ro, high.view(), {}, half.view()},
        "not owned by this Context",
        label + " rejects foreign input");
    passed &= reject_gr_mix_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual_ro, rs_ro, norm_ro, gated_ro,
         foreign_high.view(), {}, half.view()},
        "not owned by this Context",
        label + " rejects foreign output");
    return passed;
}

bool test_gr_mix_failure_discards_prior_dispatch(Context& context) {
    const std::string label = "Prefill GR mix failure discards earlier dispatch";
    std::string error;
    TestBuffer residual, rs, norm, gated, high, half;
    const bool allocated =
        residual.allocate(context, label + " residual",
                          kStreams * kWidth * sizeof(float), error) &&
        rs.allocate(context, label + " scale",
                    kStreams * sizeof(float), error) &&
        norm.allocate(context, label + " norm",
                      kStreams * kWidth * sizeof(float), error) &&
        gated.allocate(context, label + " gated",
                       kStreams * kWidth * sizeof(float), error) &&
        high.allocate(context, label + " high", kWidth * sizeof(std::uint16_t),
                      error) &&
        half.allocate(context, label + " half", kWidth * sizeof(std::uint16_t),
                      error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const std::vector<std::uint8_t> high_before(
        high.allocation().begin(), high.allocation().end());
    const std::vector<std::uint8_t> half_before(
        half.allocation().begin(), half.allocation().end());
    const PrefillGRMixBuffers buffers{
        context.buffers().read_only(residual.view()),
        context.buffers().read_only(rs.view()),
        context.buffers().read_only(norm.view()),
        context.buffers().read_only(gated.view()),
        high.view(), {}, half.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GR mix rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gr_mix(context, encoder, 1u,
                                       PrefillBF16X2Mode::off, buffers,
                                       encode_error)) {
                return false;
            }
            return encode_prefill_gr_mix(
                context, encoder, 0u, PrefillBF16X2Mode::off, buffers,
                encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(high_before.begin(), high_before.end(),
                   high.allocation().begin()) &&
        std::equal(half_before.begin(), half_before.end(),
                   half.allocation().begin());
    check(unchanged, label,
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));
    return unchanged;
}

bool test_gr_mix_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label = "Prefill GR mix rejects open concurrent group";
    std::string error;
    TestBuffer residual, rs, norm, gated, high, half;
    const bool allocated =
        residual.allocate(context, label + " residual",
                          kStreams * kWidth * sizeof(float), error) &&
        rs.allocate(context, label + " scale",
                    kStreams * sizeof(float), error) &&
        norm.allocate(context, label + " norm",
                      kStreams * kWidth * sizeof(float), error) &&
        gated.allocate(context, label + " gated",
                       kStreams * kWidth * sizeof(float), error) &&
        high.allocate(context, label + " high", kWidth * sizeof(std::uint16_t),
                      error) &&
        half.allocate(context, label + " half", kWidth * sizeof(std::uint16_t),
                      error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGRMixBuffers buffers{
        context.buffers().read_only(residual.view()),
        context.buffers().read_only(rs.view()),
        context.buffers().read_only(norm.view()),
        context.buffers().read_only(gated.view()),
        high.view(), {}, half.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GR mix",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gr_mix(context, encoder, 1u,
                                         PrefillBF16X2Mode::off, buffers,
                                         encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool run_gr_write_norm_case(Context& context, std::uint32_t rows,
                            PrefillBF16X2Mode mode, bool shared_encoder) {
    const bool write_low = mode == PrefillBF16X2Mode::all_projections;
    const std::string label =
        "Prefill GR write-norm T=" + std::to_string(rows) +
        " mode=" + std::to_string(static_cast<std::uint32_t>(mode)) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t stream_rows = static_cast<std::size_t>(rows) * kStreams;
    const std::size_t residual_bytes =
        stream_rows * kWidth * sizeof(float);
    const std::size_t block_bytes =
        static_cast<std::size_t>(rows) * kWidth * sizeof(float);
    const std::size_t inject_bytes = stream_rows * sizeof(float);
    const std::size_t norm_bytes =
        static_cast<std::size_t>(kStreams) * kWidth * sizeof(float);
    const std::size_t scale_bytes = stream_rows * sizeof(float);
    const std::size_t output_bytes =
        stream_rows * kWidth * sizeof(std::uint16_t);

    std::mt19937 rng(0x4752574eu ^ rows ^
                     static_cast<std::uint32_t>(mode));
    std::uniform_real_distribution<float> residual_dist(-1.0f, 1.0f);
    std::uniform_real_distribution<float> block_dist(-0.5f, 0.5f);
    std::uniform_real_distribution<float> inject_dist(-5.0f, 5.0f);
    std::vector<float> initial_residual(stream_rows * kWidth);
    std::vector<float> block_out(static_cast<std::size_t>(rows) * kWidth);
    std::vector<float> inject(stream_rows);
    std::vector<float> next_norm(kStreams * kWidth);
    for (float& value : initial_residual) value = residual_dist(rng);
    for (float& value : block_out) value = block_dist(rng);
    for (float& value : inject) value = inject_dist(rng);
    for (std::size_t index = 0u; index < next_norm.size(); ++index) {
        const int centered = static_cast<int>((index * 19u + 7u) % 41u) - 20;
        next_norm[index] =
            0.875f + static_cast<float>(centered) * 0.015625f;
    }

    TestBuffer fused_residual, reference_residual, block_buffer, inject_buffer;
    TestBuffer norm_buffer, fused_rs, reference_rs, fused_high, reference_high;
    TestBuffer fused_low, reference_low;
    std::string error;
    bool allocated =
        fused_residual.allocate(context, label + " fused residual",
                                residual_bytes, error) &&
        reference_residual.allocate(context, label + " reference residual",
                                    residual_bytes, error) &&
        block_buffer.allocate(context, label + " block output", block_bytes,
                              error) &&
        inject_buffer.allocate(context, label + " injection", inject_bytes,
                               error) &&
        norm_buffer.allocate(context, label + " next norm", norm_bytes,
                             error) &&
        fused_rs.allocate(context, label + " fused scale", scale_bytes,
                          error) &&
        reference_rs.allocate(context, label + " reference scale", scale_bytes,
                              error) &&
        fused_high.allocate(context, label + " fused high", output_bytes,
                            error) &&
        reference_high.allocate(context, label + " reference high",
                                output_bytes, error);
    if (write_low) {
        allocated =
            allocated &&
            fused_low.allocate(context, label + " fused low", output_bytes,
                               error) &&
            reference_low.allocate(context, label + " reference low",
                                   output_bytes, error);
    }
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    std::memcpy(fused_residual.payload().data(), initial_residual.data(),
                residual_bytes);
    std::memcpy(reference_residual.payload().data(), initial_residual.data(),
                residual_bytes);
    std::memcpy(block_buffer.payload().data(), block_out.data(), block_bytes);
    std::memcpy(inject_buffer.payload().data(), inject.data(), inject_bytes);
    std::memcpy(norm_buffer.payload().data(), next_norm.data(), norm_bytes);
    const auto block_before = std::vector<std::uint8_t>(
        block_buffer.allocation().begin(), block_buffer.allocation().end());
    const auto inject_before = std::vector<std::uint8_t>(
        inject_buffer.allocation().begin(), inject_buffer.allocation().end());
    const auto norm_before = std::vector<std::uint8_t>(
        norm_buffer.allocation().begin(), norm_buffer.allocation().end());

    const PrefillGRWriteNormBuffers fused_buffers{
        fused_residual.view(),
        context.buffers().read_only(block_buffer.view()),
        context.buffers().read_only(inject_buffer.view()),
        context.buffers().read_only(norm_buffer.view()),
        fused_rs.view(), fused_high.view(),
        write_low ? fused_low.view() : BufferRef{}};
    auto fused_command = context.make_command_buffer(label + " fused", error);
    check(fused_command != nullptr, label + " fused command buffer", error);
    if (!fused_command) return false;
    const bool fused_encoded = shared_encoder
        ? Encoder::encode(
              context, *fused_command, "shared Prefill GR write-norm",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gr_write_norm(
                      context, encoder, rows, mode, fused_buffers,
                      encode_error);
              },
              error)
        : encode_prefill_gr_write_norm(context, *fused_command, rows, mode,
                                       fused_buffers, error);
    check(fused_encoded, label + " production encode", error);
    if (!fused_encoded) {
        fused_command->discard();
        return false;
    }
    if (!submit_and_wait(fused_command, label + " fused", error)) {
        check(false, label + " fused Metal dispatch", error);
        return false;
    }

    const PrefillGRNormBuffers reference_norm_buffers{
        reference_residual.view(), norm_buffer.view(), reference_rs.view(),
        reference_high.view(),
        write_low ? reference_low.view() : BufferRef{}};
    auto reference_command =
        context.make_command_buffer(label + " reference chain", error);
    check(reference_command != nullptr,
          label + " reference command buffer", error);
    if (!reference_command) return false;
    const bool reference_encoded = Encoder::encode(
        context, *reference_command, "GR post followed by Prefill GR norm",
        [&](Encoder& encoder, std::string& encode_error) {
            const GRPostBuffers post_buffers{
                reference_residual.view(), block_buffer.view(),
                inject_buffer.view(), reference_residual.view()};
            return encode_gr_post(context, encoder, rows, post_buffers,
                                  encode_error) &&
                   encode_prefill_gr_norm(context, encoder, rows, mode,
                                          reference_norm_buffers,
                                          encode_error);
        },
        error);
    check(reference_encoded, label + " reference encode", error);
    if (!reference_encoded) {
        reference_command->discard();
        return false;
    }
    if (!submit_and_wait(reference_command, label + " reference", error)) {
        check(false, label + " reference Metal dispatch", error);
        return false;
    }

    const auto same_payload = [](const TestBuffer& left,
                                 const TestBuffer& right) {
        return std::equal(left.payload().begin(), left.payload().end(),
                          right.payload().begin());
    };
    const bool residual_equal = same_payload(fused_residual, reference_residual);
    const bool scale_equal = same_payload(fused_rs, reference_rs);
    const bool high_equal = same_payload(fused_high, reference_high);
    const bool low_equal =
        !write_low || same_payload(fused_low, reference_low);
    check(residual_equal, label + " residual matches GR post");
    check(scale_equal, label + " scale matches separate normalization");
    check(high_equal, label + " BF16 high matches separate normalization");
    check(low_equal, label + " BF16 remainder matches separate normalization");
    const bool inputs_unchanged =
        std::equal(block_before.begin(), block_before.end(),
                   block_buffer.allocation().begin()) &&
        std::equal(inject_before.begin(), inject_before.end(),
                   inject_buffer.allocation().begin()) &&
        std::equal(norm_before.begin(), norm_before.end(),
                   norm_buffer.allocation().begin());
    check(inputs_unchanged, label + " inputs remain read-only");
    const bool guards =
        fused_residual.guards_intact() && reference_residual.guards_intact() &&
        block_buffer.guards_intact() && inject_buffer.guards_intact() &&
        norm_buffer.guards_intact() && fused_rs.guards_intact() &&
        reference_rs.guards_intact() && fused_high.guards_intact() &&
        reference_high.guards_intact() &&
        (!write_low ||
         (fused_low.guards_intact() && reference_low.guards_intact()));
    check(guards, label + " guard regions");
    return residual_equal && scale_equal && high_equal && low_equal &&
           inputs_unchanged && guards;
}

float prefill_gdn_softplus_oracle(float value) {
    if (value > 20.0f) return value;
    const float exponential = std::exp(value);
    const float shifted = 1.0f + exponential;
    return shifted == 1.0f
               ? exponential
               : std::log(shifted) * (exponential / (shifted - 1.0f));
}

bool run_prefill_gdn_gates_case(Context& context, std::uint32_t rows,
                                bool shared_encoder) {
    const std::string label =
        "Prefill GDN gates T=" + std::to_string(rows) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t input_bytes =
        static_cast<std::size_t>(rows) * 2u * kGdnHeads * sizeof(float);
    const std::size_t head_bytes =
        static_cast<std::size_t>(kGdnHeads) * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * kGdnHeads * sizeof(float);
    std::vector<float> alpha_beta(static_cast<std::size_t>(rows) *
                                  2u * kGdnHeads);
    std::vector<float> dt_bias(kGdnHeads);
    std::vector<float> a_values(kGdnHeads);
    std::vector<float> expected_gate(static_cast<std::size_t>(rows) *
                                     kGdnHeads);
    std::vector<float> expected_beta(expected_gate.size());

    for (std::uint32_t head = 0u; head < kGdnHeads; ++head) {
        const int centered = static_cast<int>((head * 17u + 9u) % 41u) - 20;
        dt_bias[head] = static_cast<float>(centered) * 0.125f;
        const int scale = static_cast<int>(head % 17u) - 8;
        a_values[head] = static_cast<float>(scale) * 0.25f;
    }
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t head = 0u; head < kGdnHeads; ++head) {
            const std::size_t out = static_cast<std::size_t>(row) *
                                    kGdnHeads + head;
            const std::size_t ab = static_cast<std::size_t>(row) *
                                   2u * kGdnHeads;
            const int alpha_pattern =
                static_cast<int>((row * 47u + head * 29u + 3u) % 121u) - 60;
            const int beta_pattern =
                static_cast<int>((row * 31u + head * 13u + 11u) % 101u) - 50;
            alpha_beta[ab + head] =
                static_cast<float>(alpha_pattern) * 0.5f;
            alpha_beta[ab + kGdnHeads + head] =
                static_cast<float>(beta_pattern) * 0.5f;
            if (head == 0u) {
                alpha_beta[ab + head] = 20.0f;
                dt_bias[head] = 0.5f;
            } else if (head == 1u) {
                alpha_beta[ab + head] = -40.0f;
                dt_bias[head] = 0.0f;
            } else if (head == 2u) {
                alpha_beta[ab + kGdnHeads + head] = 20.0f;
            } else if (head == 3u) {
                alpha_beta[ab + kGdnHeads + head] = -20.0f;
            }
            const float shifted = alpha_beta[ab + head] + dt_bias[head];
            expected_gate[out] =
                prefill_gdn_softplus_oracle(shifted) * a_values[head];
            expected_beta[out] =
                1.0f /
                (1.0f + std::exp(-alpha_beta[ab + kGdnHeads + head]));
        }
    }

    TestBuffer ab_buffer, dt_buffer, a_buffer, gate_buffer, beta_buffer;
    std::string error;
    const bool allocated =
        ab_buffer.allocate(context, label + " alpha beta", input_bytes,
                           error) &&
        dt_buffer.allocate(context, label + " dt bias", head_bytes, error) &&
        a_buffer.allocate(context, label + " A", head_bytes, error) &&
        gate_buffer.allocate(context, label + " gate", output_bytes, error) &&
        beta_buffer.allocate(context, label + " beta", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(ab_buffer.payload().data(), alpha_beta.data(), input_bytes);
    std::memcpy(dt_buffer.payload().data(), dt_bias.data(), head_bytes);
    std::memcpy(a_buffer.payload().data(), a_values.data(), head_bytes);
    const auto ab_before = std::vector<std::uint8_t>(
        ab_buffer.allocation().begin(), ab_buffer.allocation().end());
    const auto dt_before = std::vector<std::uint8_t>(
        dt_buffer.allocation().begin(), dt_buffer.allocation().end());
    const auto a_before = std::vector<std::uint8_t>(
        a_buffer.allocation().begin(), a_buffer.allocation().end());
    const PrefillGDNGatesBuffers buffers{
        context.buffers().read_only(ab_buffer.view()),
        context.buffers().read_only(dt_buffer.view()),
        context.buffers().read_only(a_buffer.view()), gate_buffer.view(),
        beta_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared Prefill GDN gates",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gdn_gates(
                      context, encoder, rows, buffers, encode_error);
              },
              error)
        : encode_prefill_gdn_gates(context, *command, rows, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual_gate = std::span<const float>(
        reinterpret_cast<const float*>(gate_buffer.payload().data()),
        expected_gate.size());
    const auto actual_beta = std::span<const float>(
        reinterpret_cast<const float*>(beta_buffer.payload().data()),
        expected_beta.size());
    std::uint64_t max_gate_ulp = 0u;
    std::uint64_t max_beta_ulp = 0u;
    for (std::size_t index = 0u; index < expected_gate.size(); ++index) {
        max_gate_ulp = std::max(
            max_gate_ulp,
            float_ulp_distance(actual_gate[index], expected_gate[index]));
        max_beta_ulp = std::max(
            max_beta_ulp,
            float_ulp_distance(actual_beta[index], expected_beta[index]));
    }
    const bool gate_ok = max_gate_ulp <= 256u;
    const bool beta_ok = max_beta_ulp <= 128u;
    check(gate_ok, label + " softplus gate oracle",
          "max ULP=" + std::to_string(max_gate_ulp) + ", limit=256");
    check(beta_ok, label + " sigmoid beta oracle",
          "max ULP=" + std::to_string(max_beta_ulp) + ", limit=128");
    const bool inputs_unchanged =
        std::equal(ab_before.begin(), ab_before.end(),
                   ab_buffer.allocation().begin()) &&
        std::equal(dt_before.begin(), dt_before.end(),
                   dt_buffer.allocation().begin()) &&
        std::equal(a_before.begin(), a_before.end(),
                   a_buffer.allocation().begin());
    check(inputs_unchanged, label + " inputs remain read-only");
    const bool guards =
        ab_buffer.guards_intact() && dt_buffer.guards_intact() &&
        a_buffer.guards_intact() && gate_buffer.guards_intact() &&
        beta_buffer.guards_intact();
    check(guards, label + " guard regions");
    return gate_ok && beta_ok && inputs_unchanged && guards;
}

bool reject_prefill_gdn_gates_call(
    Context& context, std::uint32_t rows,
    const PrefillGDNGatesBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN gates",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_gates(context, encoder, rows, buffers,
                                             encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_gates_open_concurrent_group_rejected(
    Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN gates reject open concurrent group";
    constexpr std::size_t ab_bytes =
        2u * kGdnHeads * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes = kGdnHeads * sizeof(float);
    TestBuffer ab, dt, a, gate, beta;
    std::string error;
    const bool allocated =
        ab.allocate(context, label + " alpha beta", ab_bytes, error) &&
        dt.allocate(context, label + " dt bias", head_bytes, error) &&
        a.allocate(context, label + " A", head_bytes, error) &&
        gate.allocate(context, label + " gate", output_bytes, error) &&
        beta.allocate(context, label + " beta", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGDNGatesBuffers buffers{
        context.buffers().read_only(ab.view()),
        context.buffers().read_only(dt.view()),
        context.buffers().read_only(a.view()), gate.view(), beta.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GDN gates",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_gates(context, encoder, 1u, buffers,
                                             encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_gates_validation(Context& context,
                                       Context& foreign_context) {
    const std::string label = "Prefill GDN gates validation";
    constexpr std::size_t ab_bytes =
        2u * kGdnHeads * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes = kGdnHeads * sizeof(float);
    TestBuffer ab, dt, a, gate, beta, foreign;
    std::string error;
    const bool allocated =
        ab.allocate(context, label + " alpha beta", ab_bytes, error) &&
        dt.allocate(context, label + " dt", head_bytes, error) &&
        a.allocate(context, label + " A", head_bytes, error) &&
        gate.allocate(context, label + " gate", output_bytes, error) &&
        beta.allocate(context, label + " beta", output_bytes, error) &&
        foreign.allocate(foreign_context, label + " foreign", ab_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const ReadOnlyBufferRef ab_ro =
        context.buffers().read_only(ab.view());
    const ReadOnlyBufferRef dt_ro =
        context.buffers().read_only(dt.view());
    const ReadOnlyBufferRef a_ro =
        context.buffers().read_only(a.view());
    const PrefillGDNGatesBuffers valid{
        ab_ro, dt_ro, a_ro, gate.view(), beta.view()};
    bool passed = true;
    auto overlap_command =
        context.make_command_buffer(label + " read-only overlap", error);
    if (!overlap_command) {
        check(false, label + " read-only overlap command buffer", error);
        return false;
    }
    const ReadOnlyBufferRef shared_dt = ab_ro.slice(0u, head_bytes);
    const ReadOnlyBufferRef shared_a =
        ab_ro.slice(head_bytes, head_bytes);
    const bool readonly_overlap_accepted = Encoder::encode(
        context, *overlap_command, "Prefill GDN overlapping read-only inputs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_gates(
                context, encoder, 1u,
                {ab_ro, shared_dt, shared_a, gate.view(), beta.view()},
                encode_error);
        },
        error);
    overlap_command->discard();
    check(readonly_overlap_accepted,
          label + " accepts overlapping read-only inputs", error);
    passed &= readonly_overlap_accepted;
    passed &= reject_prefill_gdn_gates_call(
        context, 0u, valid, "row count",
        label + " rejects zero rows");
    passed &= reject_prefill_gdn_gates_call(
        context, kPrefillMaxShaderRows + 1u, {},
        "row count", label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_gates_call(
        context, 1u,
        {ab_ro.slice(0u, ab_bytes - sizeof(float)), dt_ro, a_ro, gate.view(),
         beta.view()},
        "undersized", label + " rejects short alpha/beta input");
    passed &= reject_prefill_gdn_gates_call(
        context, 1u,
        {ab_ro, dt_ro, a_ro, ab.view(), beta.view()},
        "output overlaps a read-only input",
        label + " rejects output/input overlap");
    passed &= reject_prefill_gdn_gates_call(
        context, 1u, {ab_ro, dt_ro, a_ro, gate.view(), gate.view()},
        "output buffer views overlap",
        label + " rejects output/output overlap");
    passed &= reject_prefill_gdn_gates_call(
        context, 1u,
        {foreign_context.buffers().read_only(foreign.view()), dt_ro, a_ro,
         gate.view(), beta.view()},
        "not owned by this Context",
        label + " rejects foreign Context input");
    return passed;
}

bool test_prefill_gdn_gates_failure_discards_prior_dispatch(
    Context& context) {
    const std::string label =
        "Prefill GDN gates failure discards earlier dispatch";
    constexpr std::size_t ab_bytes =
        2u * kGdnHeads * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes = kGdnHeads * sizeof(float);
    TestBuffer ab, dt, a, gate, beta;
    std::string error;
    const bool allocated =
        ab.allocate(context, label + " alpha beta", ab_bytes, error) &&
        dt.allocate(context, label + " dt", head_bytes, error) &&
        a.allocate(context, label + " A", head_bytes, error) &&
        gate.allocate(context, label + " gate", output_bytes, error) &&
        beta.allocate(context, label + " beta", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const auto gate_before = std::vector<std::uint8_t>(
        gate.allocation().begin(), gate.allocation().end());
    const auto beta_before = std::vector<std::uint8_t>(
        beta.allocation().begin(), beta.allocation().end());
    const PrefillGDNGatesBuffers buffers{
        context.buffers().read_only(ab.view()),
        context.buffers().read_only(dt.view()),
        context.buffers().read_only(a.view()), gate.view(), beta.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GDN gates rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_gates(context, encoder, 1u, buffers,
                                          encode_error)) {
                return false;
            }
            return encode_prefill_gdn_gates(context, encoder, 0u, {},
                                            encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(gate_before.begin(), gate_before.end(),
                   gate.allocation().begin()) &&
        std::equal(beta_before.begin(), beta_before.end(),
                   beta.allocation().begin());
    check(unchanged, label,
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));
    return unchanged;
}

float prefill_gdn_conv_input_oracle(std::span<const float> history,
                                    std::span<const float> qkv, int row,
                                    std::uint32_t channel) {
    if (row < 0) {
        const std::size_t index =
            static_cast<std::size_t>(channel) * (kGdnConvKernel - 1u) +
            static_cast<std::size_t>(row + 3);
        return history[index];
    }
    return qkv[static_cast<std::size_t>(row) * kGdnConvChannels + channel];
}

bool run_prefill_gdn_conv_tiled_case(Context& context, std::uint32_t rows,
                                     bool shared_encoder) {
    const std::string label =
        "Prefill GDN tiled convolution T=" + std::to_string(rows) +
        (shared_encoder ? " shared" : " standalone");
    const std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    const std::size_t row_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * kGdnConvChannels;
    std::vector<float> history(history_bytes / sizeof(float));
    std::vector<float> qkv(value_count);
    std::vector<float> weights(weight_bytes / sizeof(float));
    std::vector<float> expected(value_count);
    const auto input_pattern = [](std::size_t index) {
        const int centered =
            static_cast<int>((index * 37u + 13u) % 127u) - 63;
        return static_cast<float>(centered) * (1.0f / 128.0f);
    };
    const auto weight_pattern = [](std::size_t index) {
        const int centered =
            static_cast<int>((index * 19u + 5u) % 31u) - 15;
        return static_cast<float>(centered) * (1.0f / 64.0f);
    };
    for (std::size_t index = 0u; index < history.size(); ++index) {
        history[index] = input_pattern(index + 101u);
    }
    for (std::size_t index = 0u; index < qkv.size(); ++index) {
        qkv[index] = input_pattern(index + 211u);
    }
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        weights[index] = weight_pattern(index + 307u);
    }
    if (rows > 1u) {
        // For output row 1, the documented nested FMA order produces
        // 31.999998f. Reversing the FMA order rounds the same terms to 32.0f.
        // The non-dyadic 1.0001f × 255.9744f product creates the cancellation
        // residual that the later positive term keeps observable.
        history[1u] = -256.0f;
        history[2u] = 1.0001f;
        qkv[0u] = 32.0f;
        qkv[kGdnConvChannels] = 0.0f;
        weights[0u] = 1.0f;
        weights[1u] = 255.9744f;
        weights[2u] = 1.0f;
        weights[3u] = 1.0f;
    }
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t channel = 0u; channel < kGdnConvChannels;
             ++channel) {
            const std::size_t out =
                static_cast<std::size_t>(row) * kGdnConvChannels + channel;
            const std::size_t weight_base =
                static_cast<std::size_t>(channel) * kGdnConvKernel;
            const float x0 = prefill_gdn_conv_input_oracle(
                history, qkv, static_cast<int>(row) - 3, channel);
            const float x1 = prefill_gdn_conv_input_oracle(
                history, qkv, static_cast<int>(row) - 2, channel);
            const float x2 = prefill_gdn_conv_input_oracle(
                history, qkv, static_cast<int>(row) - 1, channel);
            const float x3 = qkv[out];
            const float conv = std::fma(
                x3, weights[weight_base + 3u],
                std::fma(x2, weights[weight_base + 2u],
                         std::fma(x1, weights[weight_base + 1u],
                                  x0 * weights[weight_base])));
            expected[out] = conv / (1.0f + std::exp(-conv));
        }
    }
    bool order_control_passed = true;
    if (rows > 1u) {
        const float x0 = history[1u];
        const float x1 = history[2u];
        const float x2 = qkv[0u];
        const float x3 = qkv[kGdnConvChannels];
        const float documented = std::fma(
            x3, weights[3u],
            std::fma(x2, weights[2u],
                     std::fma(x1, weights[1u], x0 * weights[0u])));
        const float reordered = std::fma(
            x0, weights[0u],
            std::fma(x1, weights[1u],
                     std::fma(x2, weights[2u], x3 * weights[3u])));
        order_control_passed =
            documented == expected[kGdnConvChannels] &&
            expected[kGdnConvChannels] != reordered;
        check(order_control_passed, label + " FMA order negative control",
              "documented=" + std::to_string(documented) +
                  ", reordered=" + std::to_string(reordered));
    }

    TestBuffer history_buffer, qkv_buffer, weight_buffer, output_buffer;
    std::string error;
    const bool allocated =
        history_buffer.allocate(context, label + " history", history_bytes,
                                error) &&
        qkv_buffer.allocate(context, label + " qkv", row_bytes, error) &&
        weight_buffer.allocate(context, label + " weights", weight_bytes,
                               error) &&
        output_buffer.allocate(context, label + " output", row_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(history_buffer.payload().data(), history.data(),
                history_bytes);
    std::memcpy(qkv_buffer.payload().data(), qkv.data(), row_bytes);
    std::memcpy(weight_buffer.payload().data(), weights.data(), weight_bytes);
    const auto history_before = std::vector<std::uint8_t>(
        history_buffer.allocation().begin(), history_buffer.allocation().end());
    const auto qkv_before = std::vector<std::uint8_t>(
        qkv_buffer.allocation().begin(), qkv_buffer.allocation().end());
    const auto weights_before = std::vector<std::uint8_t>(
        weight_buffer.allocation().begin(), weight_buffer.allocation().end());
    const PrefillGDNConvTiledBuffers buffers{
        context.buffers().read_only(history_buffer.view()),
        context.buffers().read_only(qkv_buffer.view()),
        context.buffers().read_only(weight_buffer.view()),
        output_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, "shared Prefill GDN tiled convolution",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_gdn_conv_tiled(
                      context, encoder, rows, buffers, encode_error);
              },
              error)
        : encode_prefill_gdn_conv_tiled(context, *command, rows, buffers,
                                        error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(output_buffer.payload().data()),
        value_count);
    std::uint64_t max_ulp = 0u;
    for (std::size_t index = 0u; index < expected.size(); ++index) {
        max_ulp =
            std::max(max_ulp, float_ulp_distance(actual[index], expected[index]));
    }
    const bool values_match = max_ulp <= 64u;
    check(values_match, label + " tiled convolution oracle",
          "max ULP=" + std::to_string(max_ulp) + ", limit=64");
    const bool sentinel_match =
        rows <= 1u || actual[kGdnConvChannels] == expected[kGdnConvChannels];
    check(sentinel_match, label + " cancellation-sensitive FMA sentinel");
    const bool inputs_unchanged =
        std::equal(history_before.begin(), history_before.end(),
                   history_buffer.allocation().begin()) &&
        std::equal(qkv_before.begin(), qkv_before.end(),
                   qkv_buffer.allocation().begin()) &&
        std::equal(weights_before.begin(), weights_before.end(),
                   weight_buffer.allocation().begin());
    check(inputs_unchanged, label + " inputs remain read-only");
    const bool guards =
        history_buffer.guards_intact() && qkv_buffer.guards_intact() &&
        weight_buffer.guards_intact() && output_buffer.guards_intact();
    check(guards, label + " guard regions");
    return order_control_passed && values_match && sentinel_match &&
           inputs_unchanged && guards;
}

std::vector<float> prefill_gdn_conv_history_oracle(
    std::span<const float> history, std::span<const float> qkv,
    std::uint32_t rows) {
    std::vector<float> expected(history.begin(), history.end());
    for (std::uint32_t channel = 0u; channel < kGdnConvChannels; ++channel) {
        const std::size_t history_base =
            static_cast<std::size_t>(channel) * (kGdnConvKernel - 1u);
        for (std::uint32_t k = 0u; k < kGdnConvKernel - 1u; ++k) {
            const std::uint32_t source = rows + k;
            expected[history_base + k] =
                source < kGdnConvKernel - 1u
                    ? history[history_base + source]
                    : qkv[static_cast<std::size_t>(
                              source - (kGdnConvKernel - 1u)) *
                              kGdnConvChannels +
                          channel];
        }
    }
    return expected;
}

bool run_prefill_gdn_conv_history_case(Context& context, std::uint32_t rows,
                                      bool shared_encoder,
                                      bool compose_with_convolution) {
    const std::string label =
        "Prefill GDN convolution history T=" + std::to_string(rows) +
        (compose_with_convolution
             ? " composed"
             : (shared_encoder ? " shared" : " standalone"));
    const std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    const std::size_t row_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * kGdnConvChannels;
    std::vector<float> history(history_bytes / sizeof(float));
    std::vector<float> qkv(value_count);
    std::vector<float> weights(weight_bytes / sizeof(float));
    for (std::size_t index = 0u; index < history.size(); ++index) {
        const int centered =
            static_cast<int>((index * 29u + 7u) % 127u) - 63;
        history[index] = static_cast<float>(centered) * (1.0f / 64.0f);
    }
    for (std::size_t index = 0u; index < qkv.size(); ++index) {
        const int centered =
            static_cast<int>((index * 43u + 17u) % 251u) - 125;
        qkv[index] = static_cast<float>(centered) * (1.0f / 128.0f);
    }
    for (std::size_t index = 0u; index < weights.size(); ++index) {
        const int centered =
            static_cast<int>((index * 23u + 11u) % 63u) - 31;
        weights[index] = static_cast<float>(centered) * (1.0f / 64.0f);
    }
    if (compose_with_convolution && rows > 1u) {
        history[1u] = -256.0f;
        history[2u] = 1.0001f;
        qkv[0u] = 32.0f;
        qkv[kGdnConvChannels] = 0.0f;
        weights[0u] = 1.0f;
        weights[1u] = 255.9744f;
        weights[2u] = 1.0f;
        weights[3u] = 1.0f;
    }
    const std::vector<float> expected_history =
        prefill_gdn_conv_history_oracle(history, qkv, rows);
    std::vector<float> expected_convolution;
    if (compose_with_convolution) {
        expected_convolution.resize(value_count);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t channel = 0u; channel < kGdnConvChannels;
                 ++channel) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * kGdnConvChannels +
                    channel;
                const std::size_t weight_base =
                    static_cast<std::size_t>(channel) * kGdnConvKernel;
                const float x0 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 3, channel);
                const float x1 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 2, channel);
                const float x2 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 1, channel);
                const float current = qkv[index];
                const float convolved = std::fma(
                    current, weights[weight_base + 3u],
                    std::fma(x2, weights[weight_base + 2u],
                             std::fma(x1, weights[weight_base + 1u],
                                      x0 * weights[weight_base])));
                expected_convolution[index] =
                    convolved / (1.0f + std::exp(-convolved));
            }
        }
    }

    TestBuffer history_buffer, qkv_buffer, weight_buffer, output_buffer;
    std::string error;
    const bool allocated =
        history_buffer.allocate(context, label + " history", history_bytes,
                                error) &&
        qkv_buffer.allocate(context, label + " qkv", row_bytes, error) &&
        (!compose_with_convolution ||
         (weight_buffer.allocate(context, label + " weights", weight_bytes,
                                 error) &&
          output_buffer.allocate(context, label + " convolution output",
                                 row_bytes, error)));
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(history_buffer.payload().data(), history.data(),
                history_bytes);
    std::memcpy(qkv_buffer.payload().data(), qkv.data(), row_bytes);
    if (compose_with_convolution) {
        std::memcpy(weight_buffer.payload().data(), weights.data(),
                    weight_bytes);
    }
    const auto qkv_before = std::vector<std::uint8_t>(
        qkv_buffer.allocation().begin(), qkv_buffer.allocation().end());
    const auto weights_before = std::vector<std::uint8_t>(
        weight_buffer.allocation().begin(), weight_buffer.allocation().end());
    const auto history_view =
        context.buffers().read_only(history_buffer.view());
    const auto qkv_view = context.buffers().read_only(qkv_buffer.view());
    const PrefillGDNConvHistoryBuffers history_buffers{
        history_buffer.view(), qkv_view};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    bool encoded = false;
    if (!compose_with_convolution && !shared_encoder) {
        encoded = encode_prefill_gdn_conv_history(
            context, *command, rows, history_buffers, error);
    } else {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                if (compose_with_convolution) {
                    const PrefillGDNConvTiledBuffers convolution_buffers{
                        history_view, qkv_view,
                        context.buffers().read_only(weight_buffer.view()),
                        output_buffer.view()};
                    if (!encode_prefill_gdn_conv_tiled(
                            context, encoder, rows, convolution_buffers,
                            encode_error)) {
                        return false;
                    }
                }
                return encode_prefill_gdn_conv_history(
                    context, encoder, rows, history_buffers, encode_error);
            },
            error);
    }
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual_history = std::span<const float>(
        reinterpret_cast<const float*>(history_buffer.payload().data()),
        expected_history.size());
    const bool history_matches =
        std::equal(expected_history.begin(), expected_history.end(),
                   actual_history.begin());
    check(history_matches, label + " history matches scalar oracle");
    const bool qkv_unchanged =
        std::equal(qkv_before.begin(), qkv_before.end(),
                   qkv_buffer.allocation().begin());
    check(qkv_unchanged, label + " QKV remains read-only");
    const bool weights_unchanged =
        !compose_with_convolution ||
        std::equal(weights_before.begin(), weights_before.end(),
                   weight_buffer.allocation().begin());
    check(weights_unchanged, label + " convolution weights remain read-only");
    bool convolution_matches = true;
    if (compose_with_convolution) {
        const auto actual_convolution = std::span<const float>(
            reinterpret_cast<const float*>(output_buffer.payload().data()),
            value_count);
        std::uint64_t max_ulp = 0u;
        for (std::size_t index = 0u; index < expected_convolution.size();
             ++index) {
            max_ulp = std::max(
                max_ulp,
                float_ulp_distance(actual_convolution[index],
                                   expected_convolution[index]));
        }
        convolution_matches = max_ulp <= 64u;
        check(convolution_matches, label + " convolution reads old history",
              "max ULP=" + std::to_string(max_ulp) + ", limit=64");
    }
    const bool guards =
        history_buffer.guards_intact() && qkv_buffer.guards_intact() &&
        (!compose_with_convolution ||
         (weight_buffer.guards_intact() && output_buffer.guards_intact()));
    check(guards, label + " guard regions");
    return history_matches && qkv_unchanged && weights_unchanged &&
           convolution_matches && guards;
}

std::vector<float> prefill_gdn_l2_oracle(std::span<const float> input,
                                         std::uint32_t rows) {
    std::vector<float> expected(input.begin(), input.end());
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_base =
            static_cast<std::size_t>(row) * kGdnConvChannels;
        for (std::uint32_t head = 0u; head < 32u; ++head) {
            const std::size_t head_base =
                row_base + static_cast<std::size_t>(head) * 128u;
            std::array<float, 4u> partials{};
            for (std::uint32_t warp = 0u; warp < 4u; ++warp) {
                std::array<float, 32u> squares{};
                for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                    const float value =
                        input[head_base + warp * 32u + lane];
                    squares[lane] = value * value;
                }
                partials[warp] = xor_butterfly_sum(squares);
            }
            const float sum =
                ((partials[0] + partials[1]) + partials[2]) + partials[3];
            const float scale = 1.0f / std::sqrt(sum + kEpsilon);
            for (std::uint32_t column = 0u; column < 128u; ++column) {
                const std::size_t index = head_base + column;
                expected[index] = input[index] * scale;
            }
        }
    }
    return expected;
}

bool run_prefill_gdn_l2_case(Context& context, std::uint32_t rows,
                             bool shared_encoder,
                             bool compose_with_convolution) {
    const std::string label =
        "Prefill GDN L2 T=" + std::to_string(rows) +
        (compose_with_convolution
             ? " composed"
             : (shared_encoder ? " shared" : " standalone"));
    const std::size_t h_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    const std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    const std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * kGdnConvChannels;
    std::vector<float> h(value_count);
    std::vector<float> history(history_bytes / sizeof(float));
    std::vector<float> qkv(value_count);
    std::vector<float> weights(weight_bytes / sizeof(float));
    const auto input_pattern = [](std::size_t index) {
        const int centered =
            static_cast<int>((index * 31u + 9u) % 127u) - 63;
        return static_cast<float>(centered) * (1.0f / 128.0f);
    };
    const auto weight_pattern = [](std::size_t index) {
        const int centered =
            static_cast<int>((index * 17u + 3u) % 31u) - 15;
        return static_cast<float>(centered) * (1.0f / 64.0f);
    };
    if (compose_with_convolution) {
        for (std::size_t index = 0u; index < history.size(); ++index) {
            history[index] = input_pattern(index + 103u);
        }
        for (std::size_t index = 0u; index < qkv.size(); ++index) {
            qkv[index] = input_pattern(index + 211u);
        }
        for (std::size_t index = 0u; index < weights.size(); ++index) {
            weights[index] = weight_pattern(index + 307u);
        }
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t channel = 0u; channel < kGdnConvChannels;
                 ++channel) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * kGdnConvChannels +
                    channel;
                const std::size_t weight_base =
                    static_cast<std::size_t>(channel) * kGdnConvKernel;
                const float x0 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 3, channel);
                const float x1 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 2, channel);
                const float x2 = prefill_gdn_conv_input_oracle(
                    history, qkv, static_cast<int>(row) - 1, channel);
                const float current = qkv[index];
                const float convolved = std::fma(
                    current, weights[weight_base + 3u],
                    std::fma(x2, weights[weight_base + 2u],
                             std::fma(x1, weights[weight_base + 1u],
                                      x0 * weights[weight_base])));
                h[index] = convolved / (1.0f + std::exp(-convolved));
            }
        }
    } else {
        for (std::size_t index = 0u; index < h.size(); ++index) {
            h[index] = input_pattern(index + 401u);
        }
        if (rows > 0u) {
            std::fill(h.begin(), h.begin() + 128u, 0.0f);
            h[0u] = 0.03125f;
            h[32u] = 0.03125f;
            h[64u] = 0.03125f;
            h[96u] = 128.0f;
        }
    }
    const std::vector<float> expected =
        prefill_gdn_l2_oracle(h, rows);
    bool order_control_passed = true;
    if (!compose_with_convolution) {
        const float p0 = h[0u] * h[0u];
        const float p1 = h[32u] * h[32u];
        const float p2 = h[64u] * h[64u];
        const float p3 = h[96u] * h[96u];
        const float ordered_sum = ((p0 + p1) + p2) + p3;
        const float regrouped_sum = (p0 + p2) + (p1 + p3);
        const float ordered_output =
            h[96u] * (1.0f / std::sqrt(ordered_sum + kEpsilon));
        const float regrouped_output =
            h[96u] * (1.0f / std::sqrt(regrouped_sum + kEpsilon));
        order_control_passed =
            ordered_output == expected[96u] &&
            float_ulp_distance(ordered_output, regrouped_output) >= 2u;
        check(order_control_passed,
              label + " distinguishes the specified reduction order",
              "ordered=" + std::to_string(ordered_output) +
                  ", regrouped=" + std::to_string(regrouped_output));
    }

    TestBuffer h_buffer, convolution_reference, history_buffer, qkv_buffer,
        weights_buffer;
    std::string error;
    const bool allocated =
        h_buffer.allocate(context, label + " h", h_bytes, error) &&
        (!compose_with_convolution ||
         (convolution_reference.allocate(context, label + " convolution "
                                                  "reference",
                                         h_bytes, error) &&
          history_buffer.allocate(context, label + " history", history_bytes,
                                  error) &&
          qkv_buffer.allocate(context, label + " qkv", h_bytes, error) &&
          weights_buffer.allocate(context, label + " weights", weight_bytes,
                                  error)));
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    if (!compose_with_convolution) {
        std::memcpy(h_buffer.payload().data(), h.data(), h_bytes);
    } else {
        std::memcpy(history_buffer.payload().data(), history.data(),
                    history_bytes);
        std::memcpy(qkv_buffer.payload().data(), qkv.data(), h_bytes);
        std::memcpy(weights_buffer.payload().data(), weights.data(),
                    weight_bytes);
    }
    const auto history_before = std::vector<std::uint8_t>(
        history_buffer.allocation().begin(), history_buffer.allocation().end());
    const auto qkv_before = std::vector<std::uint8_t>(
        qkv_buffer.allocation().begin(), qkv_buffer.allocation().end());
    const auto weights_before = std::vector<std::uint8_t>(
        weights_buffer.allocation().begin(), weights_buffer.allocation().end());
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    bool encoded = false;
    if (!compose_with_convolution && !shared_encoder) {
        encoded = encode_prefill_gdn_l2(
            context, *command, rows, h_buffer.view(), error);
    } else {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                if (compose_with_convolution) {
                    PrefillGDNConvTiledBuffers convolution_buffers{
                        context.buffers().read_only(history_buffer.view()),
                        context.buffers().read_only(qkv_buffer.view()),
                        context.buffers().read_only(weights_buffer.view()),
                        convolution_reference.view()};
                    if (!encode_prefill_gdn_conv_tiled(
                            context, encoder, rows, convolution_buffers,
                            encode_error)) {
                        return false;
                    }
                    convolution_buffers.output = h_buffer.view();
                    if (!encode_prefill_gdn_conv_tiled(
                            context, encoder, rows, convolution_buffers,
                            encode_error)) {
                        return false;
                    }
                }
                return encode_prefill_gdn_l2(
                    context, encoder, rows, h_buffer.view(), encode_error);
            },
            error);
    }
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(h_buffer.payload().data()), value_count);
    const auto reference_convolution =
        compose_with_convolution
            ? std::span<const float>(
                  reinterpret_cast<const float*>(
                      convolution_reference.payload().data()),
                  value_count)
            : std::span<const float>(h.data(), h.size());
    std::uint64_t max_qk_ulp = 0u;
    const std::uint64_t sentinel_ulp =
        compose_with_convolution
            ? 0u
            : float_ulp_distance(actual[96u], expected[96u]);
    bool value_heads_unchanged = true;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_base =
            static_cast<std::size_t>(row) * kGdnConvChannels;
        for (std::size_t index = 0u; index < 32u * 128u; ++index) {
            max_qk_ulp = std::max(
                max_qk_ulp,
                float_ulp_distance(actual[row_base + index],
                                   expected[row_base + index]));
        }
        for (std::size_t index = 32u * 128u;
             index < kGdnConvChannels; ++index) {
            value_heads_unchanged &=
                actual[row_base + index] ==
                reference_convolution[row_base + index];
        }
    }
    const bool qk_match = max_qk_ulp <= 32u;
    check(qk_match, label + " q/k reduction oracle",
          "max ULP=" + std::to_string(max_qk_ulp) + ", limit=32");
    const bool sentinel_match =
        compose_with_convolution || sentinel_ulp <= 1u;
    if (!compose_with_convolution) {
        check(sentinel_match,
              label + " cancellation-sensitive reduction sentinel",
              "ULP=" + std::to_string(sentinel_ulp) + ", limit=1");
    }
    check(value_heads_unchanged, label + " value heads remain unchanged");
    const bool inputs_unchanged =
        !compose_with_convolution ||
        (std::equal(history_before.begin(), history_before.end(),
                    history_buffer.allocation().begin()) &&
         std::equal(qkv_before.begin(), qkv_before.end(),
                    qkv_buffer.allocation().begin()) &&
         std::equal(weights_before.begin(), weights_before.end(),
                    weights_buffer.allocation().begin()));
    check(inputs_unchanged, label + " convolution inputs remain read-only");
    const bool guards =
        h_buffer.guards_intact() &&
        (!compose_with_convolution ||
         (convolution_reference.guards_intact() &&
          history_buffer.guards_intact() && qkv_buffer.guards_intact() &&
          weights_buffer.guards_intact()));
    check(guards, label + " guard regions");
    return order_control_passed && qk_match && sentinel_match &&
           value_heads_unchanged && inputs_unchanged && guards;
}

bool test_prefill_gdn_l2_epsilon_sentinel(Context& context) {
    const std::string label = "Prefill GDN L2 epsilon sentinel";
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    std::vector<float> input(kGdnConvChannels, 0.0f);
    input[0u] = 0.001f;
    const std::vector<float> expected =
        prefill_gdn_l2_oracle(input, 1u);
    const float without_epsilon =
        input[0u] *
        (1.0f / std::sqrt(input[0u] * input[0u]));
    const bool epsilon_control =
        float_ulp_distance(expected[0u], without_epsilon) > 100000u;
    check(epsilon_control, label + " distinguishes omitted epsilon",
          "with epsilon=" + std::to_string(expected[0u]) +
              ", without=" + std::to_string(without_epsilon));

    TestBuffer h;
    std::string error;
    if (!h.allocate(context, label + " h", h_bytes, error)) {
        check(false, label + " allocate buffer", error);
        return false;
    }
    bool passed = epsilon_control;
    for (const bool shared_encoder : {false, true}) {
        const std::string form =
            shared_encoder ? " caller-owned" : " standalone";
        std::memcpy(h.payload().data(), input.data(), h_bytes);
        auto command = context.make_command_buffer(label + form, error);
        if (!command) {
            check(false, label + form + " command buffer", error);
            return false;
        }
        const bool encoded =
            shared_encoder
                ? Encoder::encode(
                      context, *command, label + form,
                      [&](Encoder& encoder, std::string& encode_error) {
                          return encode_prefill_gdn_l2(
                              context, encoder, 1u, h.view(), encode_error);
                      },
                      error)
                : encode_prefill_gdn_l2(context, *command, 1u, h.view(),
                                        error);
        check(encoded, label + form + " production encode", error);
        if (!encoded) {
            command->discard();
            return false;
        }
        if (!submit_and_wait(command, label + form, error)) {
            check(false, label + form + " Metal dispatch", error);
            return false;
        }
        const float actual =
            *reinterpret_cast<const float*>(h.payload().data());
        const std::uint64_t ulp = float_ulp_distance(actual, expected[0u]);
        const bool matches = ulp <= 2u;
        check(matches, label + form + " retains epsilon",
              "ULP=" + std::to_string(ulp) + ", limit=2");
        const bool guards = h.guards_intact();
        check(guards, label + form + " guard regions");
        passed &= matches && guards;
    }
    return passed;
}

bool run_prefill_gdn_rec_case(Context& context, std::uint32_t rows,
                              bool shared_encoder,
                              bool compose_with_l2) {
    const std::string label =
        "Prefill GDN recurrence T=" + std::to_string(rows) +
        (compose_with_l2
             ? " L2→recurrence"
             : (shared_encoder ? " caller-owned" : " standalone"));
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    const std::size_t h_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    const std::size_t head_bytes =
        static_cast<std::size_t>(rows) * kGdnHeads * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * kGdnValueDim * sizeof(float);
    std::vector<float> initial_state(kGdnStateElements);
    std::vector<float> h(static_cast<std::size_t>(rows) * kGdnConvChannels);
    std::vector<float> gate(static_cast<std::size_t>(rows) * kGdnHeads);
    std::vector<float> beta(static_cast<std::size_t>(rows) * kGdnHeads);
    for (std::size_t i = 0u; i < initial_state.size(); ++i) {
        const int centered =
            static_cast<int>((i * 29u + 17u) % 101u) - 50;
        initial_state[i] = static_cast<float>(centered) * 0.0002f;
    }
    for (std::size_t i = 0u; i < h.size(); ++i) {
        const int centered =
            static_cast<int>((i * 37u + 43u) % 509u) - 254;
        h[i] = static_cast<float>(centered) * 0.0003f;
    }
    for (std::size_t i = 0u; i < gate.size(); ++i) {
        const int centered =
            static_cast<int>((i * 13u + 5u) % 101u) - 50;
        gate[i] = -2.0f + static_cast<float>(centered) * 0.001f;
        beta[i] = 0.4f + static_cast<float>((i * 7u + 3u) % 101u) *
                             0.0005f;
    }

    const std::vector<float> expected_h =
        compose_with_l2 ? prefill_gdn_l2_oracle(h, rows) : h;
    std::vector<float> expected_state = initial_state;
    std::vector<twin::tf> expected_output;
    std::vector<twin::tf> expected_state_oracle;
    oracle::gdn_rec(expected_state, expected_h, gate, beta, rows,
                    expected_output, expected_state_oracle);

    TestBuffer state_buffer, h_buffer, gate_buffer, beta_buffer, output_buffer;
    std::string error;
    const bool allocated =
        state_buffer.allocate(context, label + " state", state_bytes, error) &&
        h_buffer.allocate(context, label + " h", h_bytes, error) &&
        gate_buffer.allocate(context, label + " gate", head_bytes, error) &&
        beta_buffer.allocate(context, label + " beta", head_bytes, error) &&
        output_buffer.allocate(context, label + " output", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(state_buffer.payload().data(), initial_state.data(),
                state_bytes);
    std::memcpy(h_buffer.payload().data(), h.data(), h_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), head_bytes);
    std::memcpy(beta_buffer.payload().data(), beta.data(), head_bytes);

    const auto gate_before = std::vector<std::uint8_t>(
        gate_buffer.payload().begin(), gate_buffer.payload().end());
    const auto beta_before = std::vector<std::uint8_t>(
        beta_buffer.payload().begin(), beta_buffer.payload().end());
    const ReadOnlyBufferRef h_ro =
        context.buffers().read_only(h_buffer.view());
    const ReadOnlyBufferRef gate_ro =
        context.buffers().read_only(gate_buffer.view());
    const ReadOnlyBufferRef beta_ro =
        context.buffers().read_only(beta_buffer.view());
    const PrefillGDNRecBuffers buffers{
        state_buffer.view(), h_ro, gate_ro, beta_ro, output_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    bool encoded = false;
    if (shared_encoder) {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                if (compose_with_l2 &&
                    !encode_prefill_gdn_l2(context, encoder, rows,
                                            h_buffer.view(), encode_error)) {
                    return false;
                }
                return encode_prefill_gdn_rec(context, encoder, rows, buffers,
                                              encode_error);
            },
            error);
    } else {
        encoded = encode_prefill_gdn_rec(context, *command, rows, buffers,
                                         error);
    }
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual_output = std::span<const float>(
        reinterpret_cast<const float*>(output_buffer.payload().data()),
        expected_output.size());
    const auto output_mismatch = twin::compare_floats(
        actual_output, expected_output,
        {twin::Comparison::units, 512.0, 0.0, 0u});
    check(!output_mismatch, label + " raw recurrent-output oracle",
          output_mismatch ? output_mismatch->reason : "");
    const auto actual_state = std::span<const float>(
        reinterpret_cast<const float*>(state_buffer.payload().data()),
        kGdnStateElements);
    const auto state_mismatch = twin::compare_floats(
        actual_state, expected_state_oracle,
        {twin::Comparison::units, 512.0, 0.0, 0u});
    check(!state_mismatch, label + " recurrent-state oracle",
          state_mismatch ? state_mismatch->reason : "");

    const auto actual_h = std::span<const float>(
        reinterpret_cast<const float*>(h_buffer.payload().data()), h.size());
    bool h_matches = true;
    const std::size_t qk_values =
        static_cast<std::size_t>(2u * GDN_QK_HEADS * GDN_HEAD_DIM);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::size_t row_base =
            static_cast<std::size_t>(row) * kGdnConvChannels;
        for (std::size_t i = 0u; i < qk_values; ++i) {
            const std::uint64_t ulp =
                float_ulp_distance(actual_h[row_base + i],
                                   expected_h[row_base + i]);
            h_matches &= compose_with_l2 ? ulp <= 32u : ulp == 0u;
        }
        for (std::size_t i = qk_values; i < kGdnConvChannels; ++i) {
            h_matches &=
                std::memcmp(h_buffer.payload().data() +
                                (row_base + i) * sizeof(float),
                            h.data() + row_base + i, sizeof(float)) == 0;
        }
    }
    check(h_matches, label + " q/k L2 input and unchanged v heads");
    const bool read_only_inputs_unchanged =
        std::equal(gate_before.begin(), gate_before.end(),
                   gate_buffer.payload().begin()) &&
        std::equal(beta_before.begin(), beta_before.end(),
                   beta_buffer.payload().begin());
    check(read_only_inputs_unchanged, label + " gate/beta remain unchanged");
    const bool guards = state_buffer.guards_intact() &&
                        h_buffer.guards_intact() &&
                        gate_buffer.guards_intact() &&
                        beta_buffer.guards_intact() &&
                        output_buffer.guards_intact();
    check(guards, label + " guard regions");
    return !output_mismatch && !state_mismatch && h_matches &&
           read_only_inputs_unchanged && guards;
}

bool test_prefill_gdn_rec_reduction_sentinel(Context& context) {
    const std::string label = "Prefill GDN recurrence reduction sentinel";
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes = kGdnValueDim * sizeof(float);
    std::vector<float> initial_state(kGdnStateElements, 0.0f);
    std::vector<float> h(kGdnConvChannels, 0.0f);
    std::vector<float> gate(kGdnHeads, 0.0f);
    std::vector<float> beta(kGdnHeads, 0.0f);
    const std::array<float, 4u> partials{
        0x1p-16f, 0x1p-16f, 0x1p-16f, 256.0f};
    const float ordered_sum =
        ((partials[0] + partials[1]) + partials[2]) + partials[3];
    const float regrouped_sum =
        (partials[0] + partials[2]) + (partials[1] + partials[3]);
    h[2u * GDN_QK_HEADS * GDN_HEAD_DIM] = ordered_sum;
    beta[0u] = 1.0f;
    for (std::uint32_t row_group = 0u; row_group < 4u; ++row_group) {
        const std::uint32_t state_row = row_group * 32u;
        const std::size_t state_index =
            (static_cast<std::size_t>(state_row) * GDN_V_HEADS) *
            GDN_HEAD_DIM;
        initial_state[state_index] = partials[row_group];
        const std::size_t q_index = state_row;
        const std::size_t k_index =
            GDN_QK_HEADS * GDN_HEAD_DIM + state_row;
        h[q_index] = 1.0f;
        h[k_index] = 1.0f;
    }
    const float scale = 1.0f / std::sqrt(static_cast<float>(GDN_HEAD_DIM));
    const float ordered_output = ordered_sum * scale;
    const float regrouped_output = regrouped_sum * scale;
    const bool order_control =
        float_ulp_distance(ordered_output, regrouped_output) >= 2u;
    check(order_control, label + " distinguishes regrouped reduction",
          "ordered=" + std::to_string(ordered_output) +
              ", regrouped=" + std::to_string(regrouped_output));

    std::vector<float> expected_state = initial_state;
    std::vector<twin::tf> expected_output;
    std::vector<twin::tf> expected_state_oracle;
    oracle::gdn_rec(expected_state, h, gate, beta, 1u, expected_output,
                    expected_state_oracle);
    TestBuffer state_buffer, h_buffer, gate_buffer, beta_buffer, output_buffer;
    std::string error;
    const bool allocated =
        state_buffer.allocate(context, label + " state", state_bytes, error) &&
        h_buffer.allocate(context, label + " h", h_bytes, error) &&
        gate_buffer.allocate(context, label + " gate", head_bytes, error) &&
        beta_buffer.allocate(context, label + " beta", head_bytes, error) &&
        output_buffer.allocate(context, label + " output", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(state_buffer.payload().data(), initial_state.data(),
                state_bytes);
    std::memcpy(h_buffer.payload().data(), h.data(), h_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), head_bytes);
    std::memcpy(beta_buffer.payload().data(), beta.data(), head_bytes);
    const PrefillGDNRecBuffers buffers{
        state_buffer.view(), context.buffers().read_only(h_buffer.view()),
        context.buffers().read_only(gate_buffer.view()),
        context.buffers().read_only(beta_buffer.view()),
        output_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gdn_rec(context, *command, 1u, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }
    const auto actual_output = std::span<const float>(
        reinterpret_cast<const float*>(output_buffer.payload().data()),
        expected_output.size());
    const auto output_mismatch = twin::compare_floats(
        actual_output, expected_output,
        {twin::Comparison::units, 512.0, 0.0, 0u});
    check(!output_mismatch, label + " output oracle",
          output_mismatch ? output_mismatch->reason : "");
    const std::uint64_t output_ulp =
        float_ulp_distance(actual_output[0u], expected_output[0u].value);
    const bool ordered_output_matches = output_ulp <= 1u;
    check(ordered_output_matches, label + " preserves ordered reduction",
          "ULP=" + std::to_string(output_ulp) + ", limit=1");
    const bool state_unchanged =
        std::memcmp(state_buffer.payload().data(), initial_state.data(),
                    state_bytes) == 0;
    check(state_unchanged, label + " exact zero-delta state");
    const bool guards = state_buffer.guards_intact() &&
                        h_buffer.guards_intact() &&
                        gate_buffer.guards_intact() &&
                        beta_buffer.guards_intact() &&
                        output_buffer.guards_intact();
    check(guards, label + " guard regions");
    return order_control && !output_mismatch && ordered_output_matches &&
           state_unchanged && guards;
}

bool reject_prefill_gdn_rec_call(
    Context& context, std::uint32_t rows,
    const PrefillGDNRecBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN recurrence",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_rec(context, encoder, rows, buffers,
                                          encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_rec_validation(Context& context,
                                     Context& foreign_context) {
    const std::string label = "Prefill GDN recurrence validation";
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    TestBuffer state, h, gate, beta, output, foreign_state, foreign_h,
        input_overlap, output_alias;
    std::string error;
    const bool allocated =
        state.allocate(context, label + " state", state_bytes, error) &&
        h.allocate(context, label + " h", h_bytes, error) &&
        gate.allocate(context, label + " gate", head_bytes, error) &&
        beta.allocate(context, label + " beta", head_bytes, error) &&
        output.allocate(context, label + " output", output_bytes, error) &&
        foreign_state.allocate(foreign_context, label + " foreign state",
                               state_bytes, error) &&
        foreign_h.allocate(foreign_context, label + " foreign h", h_bytes,
                           error) &&
        input_overlap.allocate(context, label + " overlapping inputs",
                               h_bytes, error) &&
        output_alias.allocate(context, label + " output/input alias",
                              h_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    const ReadOnlyBufferRef h_ro = context.buffers().read_only(h.view());
    const ReadOnlyBufferRef gate_ro =
        context.buffers().read_only(gate.view());
    const ReadOnlyBufferRef beta_ro =
        context.buffers().read_only(beta.view());
    const PrefillGDNRecBuffers valid{
        state.view(), h_ro, gate_ro, beta_ro, output.view()};
    bool passed = true;
    passed &= reject_prefill_gdn_rec_call(
        context, 0u, valid, "row count", label + " rejects zero rows");
    passed &= reject_prefill_gdn_rec_call(
        context, kPrefillMaxShaderRows + 1u, {},
        "row count", label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view().slice(0u, state_bytes - sizeof(float)), h_ro, gate_ro,
         beta_ro, output.view()},
        "undersized or misaligned", label + " rejects short state");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), h_ro.slice(0u, h_bytes - sizeof(float)), gate_ro,
         beta_ro, output.view()},
        "undersized or misaligned", label + " rejects short h");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), h_ro, gate_ro.slice(0u, head_bytes - sizeof(float)),
         beta_ro, output.view()},
        "undersized or misaligned", label + " rejects short gate");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), h_ro, gate_ro,
         beta_ro.slice(0u, head_bytes - sizeof(float)), output.view()},
        "undersized or misaligned", label + " rejects short beta");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), h_ro, gate_ro, beta_ro,
         output.view().slice(0u, output_bytes - sizeof(float))},
        "undersized or misaligned", label + " rejects short output");

    BufferRef raw_misaligned_state = context.buffers().allocate(
        label + " misaligned state backing", state_bytes + 1u, error);
    BufferRef raw_misaligned_h = context.buffers().allocate(
        label + " misaligned h backing", h_bytes + 1u, error);
    if (!raw_misaligned_state || !raw_misaligned_h) {
        check(false, label + " allocate misaligned backings", error);
        return false;
    }
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {raw_misaligned_state.slice(1u, state_bytes), h_ro, gate_ro, beta_ro,
         output.view()},
        "undersized or misaligned", label + " rejects misaligned state");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(),
         context.buffers().read_only(raw_misaligned_h.slice(1u, h_bytes)),
         gate_ro, beta_ro, output.view()},
        "undersized or misaligned", label + " rejects misaligned h");

    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {foreign_state.view(), h_ro, gate_ro, beta_ro, output.view()},
        "not owned by this Context",
        label + " rejects foreign Context state");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(),
         foreign_context.buffers().read_only(foreign_h.view()), gate_ro,
         beta_ro, output.view()},
        "not owned by this Context",
        label + " rejects foreign Context h");

    const ReadOnlyBufferRef state_as_h =
        context.buffers().read_only(state.view().slice(0u, h_bytes));
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), state_as_h, gate_ro, beta_ro, output.view()},
        "overlaps a read-only input",
        label + " rejects state/h overlap");
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), h_ro, gate_ro, beta_ro,
         state.view().slice(0u, output_bytes)},
        "writable views overlap",
        label + " rejects state/output overlap");
    const ReadOnlyBufferRef alias_as_h =
        context.buffers().read_only(output_alias.view());
    passed &= reject_prefill_gdn_rec_call(
        context, 1u,
        {state.view(), alias_as_h, gate_ro, beta_ro, output_alias.view()},
        "overlaps a read-only input",
        label + " rejects output/h overlap");

    const ReadOnlyBufferRef shared_input =
        context.buffers().read_only(input_overlap.view());
    auto overlap_command =
        context.make_command_buffer(label + " read-only overlap", error);
    if (!overlap_command) {
        check(false, label + " overlap command buffer", error);
        return false;
    }
    const bool readonly_overlap_accepted = Encoder::encode(
        context, *overlap_command, "Prefill GDN overlapping read-only inputs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_rec(
                context, encoder, 1u,
                {state.view(), shared_input.slice(0u, h_bytes),
                 shared_input.slice(0u, head_bytes),
                 shared_input.slice(0u, head_bytes), output.view()},
                encode_error);
        },
        error);
    overlap_command->discard();
    check(readonly_overlap_accepted,
          label + " accepts overlapping read-only inputs", error);
    passed &= readonly_overlap_accepted;
    return passed;
}

bool test_prefill_gdn_rec_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN recurrence rejects open concurrent group";
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    TestBuffer state, h, gate, beta, output;
    std::string error;
    const bool allocated =
        state.allocate(context, label + " state", state_bytes, error) &&
        h.allocate(context, label + " h", h_bytes, error) &&
        gate.allocate(context, label + " gate", head_bytes, error) &&
        beta.allocate(context, label + " beta", head_bytes, error) &&
        output.allocate(context, label + " output", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGDNRecBuffers buffers{
        state.view(), context.buffers().read_only(h.view()),
        context.buffers().read_only(gate.view()),
        context.buffers().read_only(beta.view()), output.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GDN recurrence",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_rec(context, encoder, 1u, buffers,
                                          encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_rec_failure_discards_prior_dispatch(Context& context) {
    const std::string label =
        "Prefill GDN recurrence late failure discards prior dispatch";
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t head_bytes = kGdnHeads * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    TestBuffer state, h, gate, beta, output;
    std::string error;
    const bool allocated =
        state.allocate(context, label + " state", state_bytes, error) &&
        h.allocate(context, label + " h", h_bytes, error) &&
        gate.allocate(context, label + " gate", head_bytes, error) &&
        beta.allocate(context, label + " beta", head_bytes, error) &&
        output.allocate(context, label + " output", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    std::vector<float> state_values(kGdnStateElements, 0.0001f);
    std::vector<float> h_values(kGdnConvChannels, 0.002f);
    std::vector<float> gate_values(kGdnHeads, -2.0f);
    std::vector<float> beta_values(kGdnHeads, 0.5f);
    std::memcpy(state.payload().data(), state_values.data(), state_bytes);
    std::memcpy(h.payload().data(), h_values.data(), h_bytes);
    std::memcpy(gate.payload().data(), gate_values.data(), head_bytes);
    std::memcpy(beta.payload().data(), beta_values.data(), head_bytes);
    const std::array<std::span<const std::uint8_t>, 5u> before{
        std::span<const std::uint8_t>(state.allocation()),
        std::span<const std::uint8_t>(h.allocation()),
        std::span<const std::uint8_t>(gate.allocation()),
        std::span<const std::uint8_t>(beta.allocation()),
        std::span<const std::uint8_t>(output.allocation())};
    const auto unchanged = [&] {
        const std::array<std::span<const std::uint8_t>, 5u> after{
            std::span<const std::uint8_t>(state.allocation()),
            std::span<const std::uint8_t>(h.allocation()),
            std::span<const std::uint8_t>(gate.allocation()),
            std::span<const std::uint8_t>(beta.allocation()),
            std::span<const std::uint8_t>(output.allocation())};
        for (std::size_t i = 0u; i < before.size(); ++i) {
            if (!std::equal(before[i].begin(), before[i].end(),
                            after[i].begin())) {
                return false;
            }
        }
        return true;
    };
    const PrefillGDNRecBuffers buffers{
        state.view(), context.buffers().read_only(h.view()),
        context.buffers().read_only(gate.view()),
        context.buffers().read_only(beta.view()), output.view()};

    auto caller_command = context.make_command_buffer(label + " shared", error);
    if (!caller_command) {
        check(false, label + " shared command buffer", error);
        return false;
    }
    const bool caller_encoded = Encoder::encode(
        context, *caller_command, "Prefill GDN recurrence shared rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_rec(context, encoder, 1u, buffers,
                                        encode_error)) {
                return false;
            }
            return encode_prefill_gdn_l2(context, encoder, 0u, {},
                                         encode_error);
        },
        error);
    const bool caller_rolled_back =
        !caller_encoded && caller_command->unsafe_native_handle() == nullptr &&
        error.find("row count") != std::string::npos && unchanged();
    check(caller_rolled_back, label + " caller-owned rollback", error);

    auto standalone_command =
        context.make_command_buffer(label + " standalone", error);
    if (!standalone_command) {
        check(false, label + " standalone command buffer", error);
        return false;
    }
    const bool first =
        encode_prefill_gdn_rec(context, *standalone_command, 1u, buffers,
                               error);
    const bool second =
        first && encode_prefill_gdn_l2(context, *standalone_command, 0u, {},
                                       error);
    const bool standalone_rolled_back =
        first && !second &&
        standalone_command->unsafe_native_handle() == nullptr &&
        error.find("row count") != std::string::npos && unchanged();
    check(standalone_rolled_back, label + " standalone rollback", error);
    return caller_rolled_back && standalone_rolled_back;
}

bool run_prefill_gdn_out_norm_case(Context& context, std::uint32_t rows,
                                   bool shared_encoder,
                                   bool compose_with_rec) {
    const std::string label =
        "Prefill GDN output normalization T=" + std::to_string(rows) +
        (compose_with_rec
             ? " L2→recurrence→out_norm"
             : (shared_encoder ? " caller-owned" : " standalone"));
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * kGdnValueDim;
    const std::size_t value_bytes = value_count * sizeof(float);
    const std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    const std::size_t output_bytes = value_count * sizeof(std::uint16_t);
    constexpr std::size_t state_bytes =
        static_cast<std::size_t>(kGdnStateElements) * sizeof(float);
    const std::size_t h_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    const std::size_t head_bytes =
        static_cast<std::size_t>(rows) * kGdnHeads * sizeof(float);

    std::vector<float> oc(value_count);
    std::vector<float> z(value_count);
    std::vector<float> gamma(kGdnHeadDim);
    for (std::size_t i = 0u; i < value_count; ++i) {
        const int centered = static_cast<int>((i * 41u + 29u) % 503u) - 251;
        oc[i] = static_cast<float>(centered) * 0.003f;
        const int gate_centered =
            static_cast<int>((i * 13u + 7u) % 101u) - 50;
        z[i] = static_cast<float>(gate_centered) * 0.05f;
    }
    for (std::size_t i = 0u; i < gamma.size(); ++i) {
        const int centered =
            static_cast<int>((i * 17u + 3u) % 101u) - 50;
        gamma[i] = static_cast<float>(centered) * 0.01f;
    }

    std::vector<float> initial_state;
    std::vector<float> h;
    std::vector<float> gate;
    std::vector<float> beta;
    std::vector<float> expected_h;
    std::vector<twin::tf> expected_oc(value_count);
    std::vector<twin::tf> expected_state;
    if (compose_with_rec) {
        initial_state.resize(kGdnStateElements);
        h.resize(static_cast<std::size_t>(rows) * kGdnConvChannels);
        gate.resize(static_cast<std::size_t>(rows) * kGdnHeads);
        beta.resize(static_cast<std::size_t>(rows) * kGdnHeads);
        for (std::size_t i = 0u; i < initial_state.size(); ++i) {
            const int centered =
                static_cast<int>((i * 29u + 17u) % 101u) - 50;
            initial_state[i] = static_cast<float>(centered) * 0.0002f;
        }
        for (std::size_t i = 0u; i < h.size(); ++i) {
            const int centered =
                static_cast<int>((i * 37u + 43u) % 509u) - 254;
            h[i] = static_cast<float>(centered) * 0.0003f;
        }
        for (std::size_t i = 0u; i < gate.size(); ++i) {
            const int centered =
                static_cast<int>((i * 13u + 5u) % 101u) - 50;
            gate[i] = -2.0f + static_cast<float>(centered) * 0.001f;
            beta[i] = 0.4f +
                      static_cast<float>((i * 7u + 3u) % 101u) * 0.0005f;
        }
        expected_h = prefill_gdn_l2_oracle(h, rows);
        std::vector<float> expected_state_values = initial_state;
        oracle::gdn_rec(expected_state_values, expected_h, gate, beta, rows,
                       expected_oc, expected_state);
    } else {
        for (std::size_t i = 0u; i < value_count; ++i) {
            expected_oc[i] = twin::tf{oc[i]};
        }
    }
    std::vector<std::uint16_t> expected_y_h;
    oracle::gdn_out_norm(expected_oc, z, gamma, rows, expected_y_h);

    TestBuffer oc_buffer, z_buffer, gamma_buffer, y_h_buffer;
    TestBuffer state_buffer, h_buffer, gate_buffer, beta_buffer;
    std::string error;
    bool allocated =
        oc_buffer.allocate(context, label + " oc", value_bytes, error) &&
        z_buffer.allocate(context, label + " z", value_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h_buffer.allocate(context, label + " y_h", output_bytes, error);
    if (compose_with_rec) {
        allocated =
            allocated &&
            state_buffer.allocate(context, label + " state", state_bytes,
                                  error) &&
            h_buffer.allocate(context, label + " h", h_bytes, error) &&
            gate_buffer.allocate(context, label + " gate", head_bytes, error) &&
            beta_buffer.allocate(context, label + " beta", head_bytes, error);
    }
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    if (compose_with_rec) {
        std::memcpy(state_buffer.payload().data(), initial_state.data(),
                    state_bytes);
        std::memcpy(h_buffer.payload().data(), h.data(), h_bytes);
        std::memcpy(gate_buffer.payload().data(), gate.data(), head_bytes);
        std::memcpy(beta_buffer.payload().data(), beta.data(), head_bytes);
    } else {
        std::memcpy(oc_buffer.payload().data(), oc.data(), value_bytes);
    }
    std::memcpy(z_buffer.payload().data(), z.data(), value_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    const auto z_before = std::vector<std::uint8_t>(
        z_buffer.payload().begin(), z_buffer.payload().end());
    const auto gamma_before = std::vector<std::uint8_t>(
        gamma_buffer.payload().begin(), gamma_buffer.payload().end());
    const auto oc_before = std::vector<std::uint8_t>(
        oc_buffer.payload().begin(), oc_buffer.payload().end());

    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc_buffer.view()),
        context.buffers().read_only(z_buffer.view()),
        context.buffers().read_only(gamma_buffer.view()), y_h_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    bool encoded = false;
    if (shared_encoder) {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                if (compose_with_rec) {
                    if (!encode_prefill_gdn_l2(context, encoder, rows,
                                               h_buffer.view(), encode_error)) {
                        return false;
                    }
                    const PrefillGDNRecBuffers rec_buffers{
                        state_buffer.view(),
                        context.buffers().read_only(h_buffer.view()),
                        context.buffers().read_only(gate_buffer.view()),
                        context.buffers().read_only(beta_buffer.view()),
                        oc_buffer.view()};
                    if (!encode_prefill_gdn_rec(context, encoder, rows,
                                                rec_buffers, encode_error)) {
                        return false;
                    }
                }
                return encode_prefill_gdn_out_norm(
                    context, encoder, rows, buffers, encode_error);
            },
            error);
    } else {
        encoded = encode_prefill_gdn_out_norm(context, *command, rows, buffers,
                                              error);
    }
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }

    const auto actual_y_h = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(y_h_buffer.payload().data()),
        expected_y_h.size());
    std::uint32_t max_half_ulp = 0u;
    for (std::size_t i = 0u; i < actual_y_h.size(); ++i) {
        max_half_ulp =
            std::max(max_half_ulp,
                     half_ulp_distance(actual_y_h[i], expected_y_h[i]));
    }
    const bool output_matches = max_half_ulp <= 1u;
    check(output_matches, label + " F16 output oracle",
          "maximum half ULP=" + std::to_string(max_half_ulp) + ", limit=1");

    bool rec_matches = true;
    bool h_matches = true;
    if (compose_with_rec) {
        const auto actual_oc = std::span<const float>(
            reinterpret_cast<const float*>(oc_buffer.payload().data()),
            expected_oc.size());
        const auto oc_mismatch = twin::compare_floats(
            actual_oc, expected_oc,
            {twin::Comparison::units, 512.0, 0.0, 0u});
        const auto actual_state = std::span<const float>(
            reinterpret_cast<const float*>(state_buffer.payload().data()),
            expected_state.size());
        const auto state_mismatch = twin::compare_floats(
            actual_state, expected_state,
            {twin::Comparison::units, 512.0, 0.0, 0u});
        rec_matches = !oc_mismatch && !state_mismatch;
        check(rec_matches, label + " recurrence output/state oracles",
              oc_mismatch ? oc_mismatch->reason
                          : (state_mismatch ? state_mismatch->reason : ""));

        const auto actual_h = std::span<const float>(
            reinterpret_cast<const float*>(h_buffer.payload().data()),
            expected_h.size());
        for (std::size_t i = 0u; i < actual_h.size(); ++i) {
            if (i % kGdnConvChannels <
                2u * GDN_QK_HEADS * GDN_HEAD_DIM) {
                h_matches &=
                    float_ulp_distance(actual_h[i], expected_h[i]) <= 32u;
            } else {
                h_matches &= std::memcmp(
                                 h_buffer.payload().data() + i * sizeof(float),
                                 h.data() + i, sizeof(float)) == 0;
            }
        }
        check(h_matches, label + " q/k normalization and unchanged v");
    } else {
        const bool immutable =
            std::equal(oc_before.begin(), oc_before.end(),
                       oc_buffer.payload().begin());
        check(immutable, label + " oc remains unchanged");
        rec_matches &= immutable;
    }
    const bool readonly_inputs_unchanged =
        std::equal(z_before.begin(), z_before.end(),
                   z_buffer.payload().begin()) &&
        std::equal(gamma_before.begin(), gamma_before.end(),
                   gamma_buffer.payload().begin());
    check(readonly_inputs_unchanged, label + " z/gamma remain unchanged");
    const bool guards =
        oc_buffer.guards_intact() && z_buffer.guards_intact() &&
        gamma_buffer.guards_intact() && y_h_buffer.guards_intact() &&
        (!compose_with_rec ||
         (state_buffer.guards_intact() && h_buffer.guards_intact() &&
          gate_buffer.guards_intact() && beta_buffer.guards_intact()));
    check(guards, label + " guard regions");
    return output_matches && rec_matches && h_matches &&
           readonly_inputs_unchanged && guards;
}

bool test_prefill_gdn_out_norm_epsilon_sentinel(Context& context) {
    const std::string label =
        "Prefill GDN output normalization epsilon sentinel";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    std::vector<float> oc(kGdnValueDim, 0.0f);
    std::vector<float> z(kGdnValueDim, 0.0f);
    std::vector<float> gamma(kGdnHeadDim, 1.0f);
    oc[0u] = 0.001f;
    std::vector<twin::tf> expected_oc(kGdnValueDim);
    for (std::size_t i = 0u; i < oc.size(); ++i) {
        expected_oc[i] = twin::tf{oc[i]};
    }
    std::vector<std::uint16_t> expected_y_h;
    oracle::gdn_out_norm(expected_oc, z, gamma, 1u, expected_y_h);
    const float without_epsilon =
        (oc[0u] * (1.0f / std::sqrt((oc[0u] * oc[0u]) /
                                    static_cast<float>(kGdnHeadDim)))) *
        0.5f;
    const std::uint16_t no_epsilon_bits =
        strata::kernels::f16_from_f32(without_epsilon);
    const bool epsilon_control =
        half_ulp_distance(expected_y_h[0u], no_epsilon_bits) > 100u;
    check(epsilon_control, label + " distinguishes omitted epsilon",
          "with epsilon bits=" + std::to_string(expected_y_h[0u]) +
              ", without=" + std::to_string(no_epsilon_bits));

    TestBuffer oc_buffer, z_buffer, gamma_buffer, y_h_buffer;
    std::string error;
    const bool allocated =
        oc_buffer.allocate(context, label + " oc", value_bytes, error) &&
        z_buffer.allocate(context, label + " z", value_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h_buffer.allocate(context, label + " y_h", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(oc_buffer.payload().data(), oc.data(), value_bytes);
    std::memcpy(z_buffer.payload().data(), z.data(), value_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc_buffer.view()),
        context.buffers().read_only(z_buffer.view()),
        context.buffers().read_only(gamma_buffer.view()), y_h_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gdn_out_norm(context, *command, 1u, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }
    const auto actual = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(y_h_buffer.payload().data()),
        expected_y_h.size());
    const bool matches =
        half_ulp_distance(actual[0u], expected_y_h[0u]) <= 1u;
    check(matches, label + " retains RMS_EPSILON",
          "half ULP=" +
              std::to_string(half_ulp_distance(actual[0u], expected_y_h[0u])));
    const bool guards = oc_buffer.guards_intact() && z_buffer.guards_intact() &&
                        gamma_buffer.guards_intact() &&
                        y_h_buffer.guards_intact();
    check(guards, label + " guard regions");
    return epsilon_control && matches && guards;
}

bool test_prefill_gdn_out_norm_reduction_order_sentinel(Context& context) {
    const std::string label =
        "Prefill GDN output normalization reduction-order sentinel";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    std::vector<float> oc(kGdnValueDim, 0.0f);
    std::vector<float> z(kGdnValueDim, 20.0f);
    std::vector<float> gamma(kGdnHeadDim, 1.0f);
    gamma[1u] = 180.9751434326171875f;
    for (std::size_t head = 0u; head < kGdnHeads; ++head) {
        const std::size_t base = head * kGdnHeadDim;
        oc[base] = 4096.0f;
        oc[base + 1u] = 1.0f;
        oc[base + 32u] = 1.0f;
        oc[base + 64u] = 1.0f;
        oc[base + 96u] = 1.0f;
    }

    std::vector<twin::tf> expected_oc(kGdnValueDim);
    for (std::size_t i = 0u; i < oc.size(); ++i) {
        expected_oc[i] = twin::tf{oc[i]};
    }
    std::vector<std::uint16_t> expected_y_h;
    oracle::gdn_out_norm(expected_oc, z, gamma, 1u, expected_y_h);

    // A right-associated sum gives 2^24 + 3, which rounds to 2^24 + 4.
    // The specified left-associated reduction rounds each +1 away at 2^24.
    const float regrouped_sum =
        16777216.0f + ((1.0f + 1.0f) + 1.0f);
    const float regrouped_mean =
        regrouped_sum / static_cast<float>(kGdnHeadDim) + 1.0e-6f;
    const float regrouped_scale = static_cast<float>(
        1.0 / std::sqrt(static_cast<double>(regrouped_mean)));
    const float regrouped_value =
        (1.0f * regrouped_scale) * gamma[1u];
    const std::uint16_t regrouped_bits =
        strata::kernels::f16_from_f32(regrouped_value);
    const bool order_control = expected_y_h[1u] != regrouped_bits;
    check(order_control, label + " distinguishes regrouped partials",
          "specified bits=" + std::to_string(expected_y_h[1u]) +
              ", regrouped bits=" + std::to_string(regrouped_bits));

    TestBuffer oc_buffer, z_buffer, gamma_buffer, y_h_buffer;
    std::string error;
    const bool allocated =
        oc_buffer.allocate(context, label + " oc", value_bytes, error) &&
        z_buffer.allocate(context, label + " z", value_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h_buffer.allocate(context, label + " y_h", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(oc_buffer.payload().data(), oc.data(), value_bytes);
    std::memcpy(z_buffer.payload().data(), z.data(), value_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc_buffer.view()),
        context.buffers().read_only(z_buffer.view()),
        context.buffers().read_only(gamma_buffer.view()), y_h_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gdn_out_norm(context, *command, 1u, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }
    const auto actual = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(y_h_buffer.payload().data()),
        expected_y_h.size());
    bool all_heads_follow_order = true;
    for (std::size_t head = 0u; head < kGdnHeads; ++head) {
        const std::size_t index = head * kGdnHeadDim + 1u;
        all_heads_follow_order &=
            actual[index] == expected_y_h[index] &&
            actual[index] != regrouped_bits;
    }
    check(all_heads_follow_order,
          label + " output follows the specified four-partial order",
          "specified bits=" + std::to_string(expected_y_h[1u]) +
              ", regrouped bits=" + std::to_string(regrouped_bits) +
              ", actual head 0 bits=" + std::to_string(actual[1u]));
    const bool guards = oc_buffer.guards_intact() && z_buffer.guards_intact() &&
                        gamma_buffer.guards_intact() &&
                        y_h_buffer.guards_intact();
    check(guards, label + " guard regions");
    return order_control && all_heads_follow_order && guards;
}

bool test_prefill_gdn_out_norm_unsaturated_half_overflow(Context& context) {
    const std::string label =
        "Prefill GDN output normalization unsaturated F16 overflow";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    std::vector<float> oc(kGdnValueDim, 0.0f);
    std::vector<float> z(kGdnValueDim, 0.0f);
    std::vector<float> gamma(kGdnHeadDim, 1.0f);
    oc[0u] = 1.0f;
    oc[1u] = -1.0f;
    z[0u] = 20.0f;
    z[1u] = 20.0f;
    gamma[0u] = 10000.0f;
    gamma[1u] = 10000.0f;
    std::vector<twin::tf> expected_oc(kGdnValueDim);
    for (std::size_t i = 0u; i < oc.size(); ++i) {
        expected_oc[i] = twin::tf{oc[i]};
    }
    std::vector<std::uint16_t> expected_y_h;
    oracle::gdn_out_norm(expected_oc, z, gamma, 1u, expected_y_h);
    const bool oracle_unsaturated =
        expected_y_h[0u] == 0x7c00u && expected_y_h[1u] == 0xfc00u;
    check(oracle_unsaturated, label + " oracle preserves signed infinity",
          "positive=" + std::to_string(expected_y_h[0u]) +
              ", negative=" + std::to_string(expected_y_h[1u]));

    TestBuffer oc_buffer, z_buffer, gamma_buffer, y_h_buffer;
    std::string error;
    const bool allocated =
        oc_buffer.allocate(context, label + " oc", value_bytes, error) &&
        z_buffer.allocate(context, label + " z", value_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h_buffer.allocate(context, label + " y_h", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(oc_buffer.payload().data(), oc.data(), value_bytes);
    std::memcpy(z_buffer.payload().data(), z.data(), value_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc_buffer.view()),
        context.buffers().read_only(z_buffer.view()),
        context.buffers().read_only(gamma_buffer.view()), y_h_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gdn_out_norm(context, *command, 1u, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }
    const auto actual = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(y_h_buffer.payload().data()),
        expected_y_h.size());
    const bool unsaturated =
        actual[0u] == 0x7c00u && actual[1u] == 0xfc00u;
    check(unsaturated, label + " does not clamp to finite F16",
          "positive=" + std::to_string(actual[0u]) +
              ", negative=" + std::to_string(actual[1u]));
    const bool guards = oc_buffer.guards_intact() && z_buffer.guards_intact() &&
                        gamma_buffer.guards_intact() &&
                        y_h_buffer.guards_intact();
    check(guards, label + " guard regions");
    return oracle_unsaturated && unsaturated && guards;
}

bool test_prefill_gdn_out_norm_halfway_rne(Context& context) {
    const std::string label =
        "Prefill GDN output normalization F16 halfway RNE";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    std::vector<float> oc(kGdnValueDim, 1.0f);
    std::vector<float> z(kGdnValueDim, 20.0f);
    std::vector<float> gamma(kGdnHeadDim, 1.0f);
    // With 128 unit inputs, rsqrt(1 + epsilon) is 0x3f7ffff8 on the
    // reference compiler. This gamma makes the pre-conversion F32 value the
    // exact midpoint between F16 1.0 and its successor; ties round to even.
    gamma[0u] = 1.000488758087158203125f;
    gamma[1u] = std::nextafter(gamma[0u],
                               std::numeric_limits<float>::infinity());
    std::vector<twin::tf> expected_oc(kGdnValueDim);
    for (std::size_t i = 0u; i < oc.size(); ++i) {
        expected_oc[i] = twin::tf{oc[i]};
    }
    std::vector<std::uint16_t> expected_y_h;
    oracle::gdn_out_norm(expected_oc, z, gamma, 1u, expected_y_h);
    const bool oracle_tie =
        expected_y_h[0u] == 0x3c00u && expected_y_h[1u] == 0x3c01u;
    check(oracle_tie, label + " scalar oracle locks both RNE sides",
          "tie=" + std::to_string(expected_y_h[0u]) +
              ", above=" + std::to_string(expected_y_h[1u]));

    TestBuffer oc_buffer, z_buffer, gamma_buffer, y_h_buffer;
    std::string error;
    const bool allocated =
        oc_buffer.allocate(context, label + " oc", value_bytes, error) &&
        z_buffer.allocate(context, label + " z", value_bytes, error) &&
        gamma_buffer.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h_buffer.allocate(context, label + " y_h", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    std::memcpy(oc_buffer.payload().data(), oc.data(), value_bytes);
    std::memcpy(z_buffer.payload().data(), z.data(), value_bytes);
    std::memcpy(gamma_buffer.payload().data(), gamma.data(), gamma_bytes);
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc_buffer.view()),
        context.buffers().read_only(z_buffer.view()),
        context.buffers().read_only(gamma_buffer.view()), y_h_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_prefill_gdn_out_norm(context, *command, 1u, buffers, error);
    check(encoded, label + " production encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " Metal dispatch", error);
        return false;
    }
    const auto actual = std::span<const std::uint16_t>(
        reinterpret_cast<const std::uint16_t*>(y_h_buffer.payload().data()),
        expected_y_h.size());
    const bool rne_bits =
        actual[0u] == 0x3c00u && actual[1u] == 0x3c01u;
    check(rne_bits, label + " GPU output rounds midpoint to even",
          "tie=" + std::to_string(actual[0u]) +
              ", above=" + std::to_string(actual[1u]));
    const bool guards = oc_buffer.guards_intact() && z_buffer.guards_intact() &&
                        gamma_buffer.guards_intact() &&
                        y_h_buffer.guards_intact();
    check(guards, label + " guard regions");
    return oracle_tie && rne_bits && guards;
}

bool reject_prefill_gdn_out_norm_call(
    Context& context, std::uint32_t rows,
    const PrefillGDNOutNormBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN output normalization",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_out_norm(context, encoder, rows, buffers,
                                               encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_out_norm_validation(Context& context,
                                          Context& foreign_context) {
    const std::string label = "Prefill GDN output normalization validation";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    TestBuffer oc, z, gamma, y_h, foreign_oc, foreign_y_h, output_alias,
        overlapping_inputs;
    std::string error;
    const bool allocated =
        oc.allocate(context, label + " oc", value_bytes, error) &&
        z.allocate(context, label + " z", value_bytes, error) &&
        gamma.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h.allocate(context, label + " y_h", output_bytes, error) &&
        foreign_oc.allocate(foreign_context, label + " foreign oc",
                            value_bytes, error) &&
        foreign_y_h.allocate(foreign_context, label + " foreign y_h",
                             output_bytes, error) &&
        output_alias.allocate(context, label + " output alias backing",
                              value_bytes, error) &&
        overlapping_inputs.allocate(context, label + " shared read inputs",
                                   value_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const ReadOnlyBufferRef oc_ro = context.buffers().read_only(oc.view());
    const ReadOnlyBufferRef z_ro = context.buffers().read_only(z.view());
    const ReadOnlyBufferRef gamma_ro =
        context.buffers().read_only(gamma.view());
    const PrefillGDNOutNormBuffers valid{oc_ro, z_ro, gamma_ro, y_h.view()};
    bool passed = true;
    passed &= reject_prefill_gdn_out_norm_call(
        context, 0u, valid, "row count", label + " rejects zero rows");
    passed &= reject_prefill_gdn_out_norm_call(
        context, kPrefillMaxShaderRows + 1u, {},
        "row count", label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro.slice(0u, value_bytes - sizeof(float)), z_ro, gamma_ro,
         y_h.view()},
        "undersized or misaligned", label + " rejects short oc");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro, z_ro.slice(0u, value_bytes - sizeof(float)), gamma_ro,
         y_h.view()},
        "undersized or misaligned", label + " rejects short z");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro, z_ro, gamma_ro.slice(0u, gamma_bytes - sizeof(float)),
         y_h.view()},
        "undersized or misaligned", label + " rejects short gamma");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro, z_ro, gamma_ro,
         y_h.view().slice(0u, output_bytes - sizeof(std::uint16_t))},
        "undersized or misaligned", label + " rejects short y_h");

    BufferRef raw_float = context.buffers().allocate(
        label + " misaligned F32 backing", value_bytes + 1u, error);
    BufferRef raw_half = context.buffers().allocate(
        label + " misaligned F16 backing", output_bytes + 1u, error);
    if (!raw_float || !raw_half) {
        check(false, label + " allocate misaligned backings", error);
        return false;
    }
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {context.buffers().read_only(raw_float.slice(1u, value_bytes)), z_ro,
         gamma_ro, y_h.view()},
        "undersized or misaligned", label + " rejects misaligned oc");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro, z_ro, gamma_ro,
         raw_half.slice(1u, output_bytes)},
        "undersized or misaligned", label + " rejects misaligned y_h");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {foreign_context.buffers().read_only(foreign_oc.view()), z_ro,
         gamma_ro, y_h.view()},
        "not owned by this Context",
        label + " rejects foreign Context input");
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {oc_ro, z_ro, gamma_ro, foreign_y_h.view()},
        "not owned by this Context",
        label + " rejects foreign Context output");

    const ReadOnlyBufferRef alias_as_oc = context.buffers().read_only(
        output_alias.view().slice(0u, value_bytes));
    passed &= reject_prefill_gdn_out_norm_call(
        context, 1u,
        {alias_as_oc, z_ro, gamma_ro,
         output_alias.view().slice(0u, output_bytes)},
        "overlaps a read-only input", label + " rejects y_h/oc overlap");

    const ReadOnlyBufferRef shared =
        context.buffers().read_only(overlapping_inputs.view());
    auto overlap_command =
        context.make_command_buffer(label + " read-only overlap", error);
    if (!overlap_command) {
        check(false, label + " overlap command buffer", error);
        return false;
    }
    const bool readonly_overlap_accepted = Encoder::encode(
        context, *overlap_command, "Prefill GDN overlapping read-only inputs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_out_norm(
                context, encoder, 1u,
                {shared.slice(0u, value_bytes),
                 shared.slice(0u, value_bytes),
                 shared.slice(0u, gamma_bytes), y_h.view()},
                encode_error);
        },
        error);
    overlap_command->discard();
    check(readonly_overlap_accepted,
          label + " accepts overlapping read-only inputs", error);
    passed &= readonly_overlap_accepted;
    return passed;
}

bool test_prefill_gdn_out_norm_open_concurrent_group_rejected(
    Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN output normalization rejects open concurrent group";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    TestBuffer oc, z, gamma, y_h;
    std::string error;
    const bool allocated =
        oc.allocate(context, label + " oc", value_bytes, error) &&
        z.allocate(context, label + " z", value_bytes, error) &&
        gamma.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h.allocate(context, label + " y_h", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc.view()),
        context.buffers().read_only(z.view()),
        context.buffers().read_only(gamma.view()), y_h.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent GDN output normalization",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_out_norm(
                context, encoder, 1u, buffers, encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_out_norm_failure_discards_prior_dispatch(
    Context& context) {
    const std::string label =
        "Prefill GDN output normalization late failure rollback";
    constexpr std::size_t value_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(float);
    constexpr std::size_t gamma_bytes =
        static_cast<std::size_t>(kGdnHeadDim) * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(kGdnValueDim) * sizeof(std::uint16_t);
    TestBuffer oc, z, gamma, y_h;
    std::string error;
    const bool allocated =
        oc.allocate(context, label + " oc", value_bytes, error) &&
        z.allocate(context, label + " z", value_bytes, error) &&
        gamma.allocate(context, label + " gamma", gamma_bytes, error) &&
        y_h.allocate(context, label + " y_h", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const std::array<std::span<const std::uint8_t>, 4u> before{
        std::span<const std::uint8_t>(oc.allocation()),
        std::span<const std::uint8_t>(z.allocation()),
        std::span<const std::uint8_t>(gamma.allocation()),
        std::span<const std::uint8_t>(y_h.allocation())};
    const auto unchanged = [&] {
        const std::array<std::span<const std::uint8_t>, 4u> after{
            std::span<const std::uint8_t>(oc.allocation()),
            std::span<const std::uint8_t>(z.allocation()),
            std::span<const std::uint8_t>(gamma.allocation()),
            std::span<const std::uint8_t>(y_h.allocation())};
        for (std::size_t i = 0u; i < before.size(); ++i) {
            if (!std::equal(before[i].begin(), before[i].end(),
                            after[i].begin())) {
                return false;
            }
        }
        return true;
    };
    const PrefillGDNOutNormBuffers buffers{
        context.buffers().read_only(oc.view()),
        context.buffers().read_only(z.view()),
        context.buffers().read_only(gamma.view()), y_h.view()};

    auto caller_command = context.make_command_buffer(label + " shared", error);
    if (!caller_command) {
        check(false, label + " shared command buffer", error);
        return false;
    }
    const bool caller_encoded = Encoder::encode(
        context, *caller_command, "Prefill GDN out-norm shared rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_out_norm(context, encoder, 1u, buffers,
                                             encode_error)) {
                return false;
            }
            return encode_prefill_gdn_out_norm(context, encoder, 0u, buffers,
                                               encode_error);
        },
        error);
    const bool caller_rolled_back =
        !caller_encoded && caller_command->unsafe_native_handle() == nullptr &&
        error.find("row count") != std::string::npos && unchanged();
    check(caller_rolled_back, label + " caller-owned rollback", error);

    auto standalone_command =
        context.make_command_buffer(label + " standalone", error);
    if (!standalone_command) {
        check(false, label + " standalone command buffer", error);
        return false;
    }
    const bool first = encode_prefill_gdn_out_norm(
        context, *standalone_command, 1u, buffers, error);
    const bool second =
        first && encode_prefill_gdn_out_norm(
                     context, *standalone_command, 0u, buffers, error);
    const bool standalone_rolled_back =
        first && !second &&
        standalone_command->unsafe_native_handle() == nullptr &&
        error.find("row count") != std::string::npos && unchanged();
    check(standalone_rolled_back, label + " standalone rollback", error);
    return caller_rolled_back && standalone_rolled_back;
}

bool reject_prefill_gdn_conv_tiled_call(
    Context& context, std::uint32_t rows,
    const PrefillGDNConvTiledBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN tiled convolution",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_conv_tiled(
                context, encoder, rows, buffers, encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_conv_tiled_validation(Context& context,
                                            Context& foreign_context) {
    const std::string label = "Prefill GDN tiled convolution validation";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    TestBuffer history, qkv, weights, output, foreign, overlapping_inputs;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error) &&
        weights.allocate(context, label + " weights", weight_bytes, error) &&
        output.allocate(context, label + " output", row_bytes, error) &&
        foreign.allocate(foreign_context, label + " foreign", history_bytes,
                         error) &&
        overlapping_inputs.allocate(context, label + " overlapping read-only",
                                    weight_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const ReadOnlyBufferRef history_ro =
        context.buffers().read_only(history.view());
    const ReadOnlyBufferRef qkv_ro = context.buffers().read_only(qkv.view());
    const ReadOnlyBufferRef weights_ro =
        context.buffers().read_only(weights.view());
    const PrefillGDNConvTiledBuffers valid{
        history_ro, qkv_ro, weights_ro, output.view()};
    bool passed = true;
    auto overlap_command =
        context.make_command_buffer(label + " read-only overlap", error);
    if (!overlap_command) {
        check(false, label + " read-only overlap command buffer", error);
        return false;
    }
    const ReadOnlyBufferRef shared_ro =
        context.buffers().read_only(overlapping_inputs.view());
    const bool readonly_overlap_accepted = Encoder::encode(
        context, *overlap_command, "Prefill GDN overlapping read-only inputs",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_conv_tiled(
                context, encoder, 1u,
                {shared_ro.slice(0u, history_bytes),
                 shared_ro.slice(0u, row_bytes),
                 shared_ro.slice(0u, weight_bytes), output.view()},
                encode_error);
        },
        error);
    overlap_command->discard();
    check(readonly_overlap_accepted,
          label + " accepts overlapping read-only inputs", error);
    passed &= readonly_overlap_accepted;
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 0u, valid, "row count", label + " rejects zero rows");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, kPrefillMaxShaderRows + 1u, {}, "row count",
        label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 1u,
        {history_ro.slice(0u, history_bytes - sizeof(float)), qkv_ro,
         weights_ro, output.view()},
        "undersized", label + " rejects short history");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 1u,
        {history_ro, qkv_ro.slice(0u, row_bytes - sizeof(float)), weights_ro,
         output.view()},
        "undersized", label + " rejects short QKV");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 1u,
        {history_ro, qkv_ro, weights_ro.slice(0u, weight_bytes - sizeof(float)),
         output.view()},
        "undersized", label + " rejects short weights");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 1u,
        {history_ro, qkv_ro, weights_ro, qkv.view()},
        "output overlaps a read-only input",
        label + " rejects output/input overlap");
    passed &= reject_prefill_gdn_conv_tiled_call(
        context, 1u,
        {foreign_context.buffers().read_only(foreign.view()), qkv_ro,
         weights_ro, output.view()},
        "not owned by this Context",
        label + " rejects foreign Context input");
    return passed;
}

bool reject_prefill_gdn_conv_history_call(
    Context& context, std::uint32_t rows,
    const PrefillGDNConvHistoryBuffers& buffers,
    std::string_view expected_error, const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN convolution history",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_conv_history(
                context, encoder, rows, buffers, encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_conv_history_validation(Context& context,
                                              Context& foreign_context) {
    const std::string label = "Prefill GDN convolution history validation";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    TestBuffer history, qkv, foreign_history, foreign_qkv;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error) &&
        foreign_history.allocate(foreign_context, label + " foreign history",
                                 history_bytes, error) &&
        foreign_qkv.allocate(foreign_context, label + " foreign qkv",
                            row_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const ReadOnlyBufferRef qkv_view =
        context.buffers().read_only(qkv.view());
    const PrefillGDNConvHistoryBuffers valid{history.view(), qkv_view};
    bool passed = true;
    passed &= reject_prefill_gdn_conv_history_call(
        context, 0u, valid, "row count", label + " rejects zero rows");
    passed &= reject_prefill_gdn_conv_history_call(
        context, kPrefillMaxShaderRows + 1u, {},
        "row count", label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u,
        {history.view().slice(0u, history_bytes - sizeof(float)), qkv_view},
        "undersized or misaligned",
        label + " rejects short history");
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u,
        {history.view(), qkv_view.slice(0u, row_bytes - sizeof(float))},
        "undersized or misaligned", label + " rejects short QKV");
    BufferRef raw_misaligned = context.buffers().allocate(
        label + " misaligned backing", history_bytes + 1u, error);
    const BufferRef misaligned_history =
        raw_misaligned ? raw_misaligned.slice(1u, history_bytes) : BufferRef{};
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u, {misaligned_history, qkv_view},
        "undersized or misaligned",
        label + " rejects misaligned history");
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u,
        {history.view(),
         context.buffers().read_only(history.view().slice(0u, row_bytes))},
        "overlaps QKV", label + " rejects history/QKV overlap");
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u,
        {foreign_history.view(), qkv_view}, "not owned by this Context",
        label + " rejects foreign Context history");
    passed &= reject_prefill_gdn_conv_history_call(
        context, 1u,
        {history.view(),
         foreign_context.buffers().read_only(foreign_qkv.view())},
        "not owned by this Context",
        label + " rejects foreign Context QKV");
    return passed;
}

bool reject_prefill_gdn_l2_call(Context& context, std::uint32_t rows,
                                BufferRef h, std::string_view expected_error,
                                const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GDN L2",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gdn_l2(context, encoder, rows, h,
                                         encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_prefill_gdn_l2_validation(Context& context,
                                    Context& foreign_context) {
    const std::string label = "Prefill GDN L2 validation";
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    TestBuffer h, foreign_h;
    std::string error;
    const bool allocated =
        h.allocate(context, label + " h", row_bytes, error) &&
        foreign_h.allocate(foreign_context, label + " foreign h", row_bytes,
                           error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    bool passed = true;
    passed &= reject_prefill_gdn_l2_call(
        context, 0u, h.view(), "row count", label + " rejects zero rows");
    passed &= reject_prefill_gdn_l2_call(
        context, kPrefillMaxShaderRows + 1u, {},
        "row count", label + " rejects shader-unsafe rows");
    passed &= reject_prefill_gdn_l2_call(
        context, 1u, h.view().slice(0u, row_bytes - sizeof(float)),
        "undersized or misaligned", label + " rejects short h view");
    BufferRef raw_misaligned =
        context.buffers().allocate(label + " misaligned backing",
                                   row_bytes + 1u, error);
    if (!raw_misaligned) {
        check(false, label + " allocate misaligned backing", error);
        return false;
    }
    const BufferRef misaligned_h =
        raw_misaligned.slice(1u, row_bytes);
    passed &= reject_prefill_gdn_l2_call(
        context, 1u, misaligned_h, "undersized or misaligned",
        label + " rejects misaligned h view");
    passed &= reject_prefill_gdn_l2_call(
        context, 1u, foreign_h.view(), "not owned by this Context",
        label + " rejects foreign Context h");
    return passed;
}

bool test_prefill_gdn_l2_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN L2 rejects open concurrent group";
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    TestBuffer h;
    std::string error;
    if (!h.allocate(context, label + " h", h_bytes, error)) {
        check(false, label + " allocate buffer", error);
        return false;
    }
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GDN L2",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_l2(context, encoder, 1u, h.view(),
                                         encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_l2_failure_discards_prior_dispatch(Context& context) {
    const std::string label =
        "Prefill GDN L2 failure discards prior dispatch";
    constexpr std::size_t h_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    TestBuffer h;
    std::string error;
    if (!h.allocate(context, label + " h", h_bytes, error)) {
        check(false, label + " allocate buffer", error);
        return false;
    }
    std::vector<float> initial(kGdnConvChannels);
    for (std::size_t index = 0u; index < initial.size(); ++index) {
        initial[index] =
            static_cast<float>(static_cast<int>(index % 101u) - 50) / 64.0f;
    }
    std::memcpy(h.payload().data(), initial.data(), h_bytes);
    const auto before = std::vector<std::uint8_t>(
        h.allocation().begin(), h.allocation().end());

    auto caller_command = context.make_command_buffer(label + " shared", error);
    if (!caller_command) {
        check(false, label + " shared command buffer", error);
        return false;
    }
    const bool caller_encoded = Encoder::encode(
        context, *caller_command, "Prefill GDN L2 shared rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_l2(context, encoder, 1u, h.view(),
                                       encode_error)) {
                return false;
            }
            return encode_prefill_gdn_l2(context, encoder, 0u, {},
                                         encode_error);
        },
        error);
    const bool caller_discarded =
        caller_command->unsafe_native_handle() == nullptr;
    const bool caller_unchanged =
        !caller_encoded && caller_discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(before.begin(), before.end(), h.allocation().begin());
    check(caller_unchanged, label + " caller-owned Encoder rollback", error);

    auto standalone_command =
        context.make_command_buffer(label + " standalone", error);
    if (!standalone_command) {
        check(false, label + " standalone command buffer", error);
        return false;
    }
    const bool first_standalone =
        encode_prefill_gdn_l2(context, *standalone_command, 1u, h.view(),
                              error);
    const bool second_standalone =
        first_standalone && encode_prefill_gdn_l2(
                                context, *standalone_command, 0u, {}, error);
    const bool standalone_discarded =
        standalone_command->unsafe_native_handle() == nullptr;
    const bool standalone_unchanged =
        first_standalone && !second_standalone && standalone_discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(before.begin(), before.end(), h.allocation().begin());
    check(standalone_unchanged, label + " standalone wrapper rollback",
          error);
    return caller_unchanged && standalone_unchanged;
}

bool test_prefill_gdn_conv_history_open_concurrent_group_rejected(
    Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN convolution history rejects open concurrent group";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    TestBuffer history, qkv;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGDNConvHistoryBuffers buffers{
        history.view(), context.buffers().read_only(qkv.view())};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GDN history",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_conv_history(
                context, encoder, 1u, buffers, encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_conv_history_failure_discards_prior_dispatch(
    Context& context) {
    const std::string label =
        "Prefill GDN convolution history failure discards prior dispatch";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    TestBuffer history, qkv, weights, output;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error) &&
        weights.allocate(context, label + " weights", weight_bytes, error) &&
        output.allocate(context, label + " output", row_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    for (std::size_t index = 0u; index < history.payload().size(); ++index) {
        history.payload()[index] = static_cast<std::uint8_t>(index * 13u);
    }
    const auto history_before = std::vector<std::uint8_t>(
        history.allocation().begin(), history.allocation().end());
    const auto output_before = std::vector<std::uint8_t>(
        output.allocation().begin(), output.allocation().end());
    const PrefillGDNConvTiledBuffers convolution_buffers{
        context.buffers().read_only(history.view()),
        context.buffers().read_only(qkv.view()),
        context.buffers().read_only(weights.view()), output.view()};
    const PrefillGDNConvHistoryBuffers history_buffers{
        history.view(), context.buffers().read_only(qkv.view())};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GDN history rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_conv_tiled(
                    context, encoder, 1u, convolution_buffers,
                    encode_error)) {
                return false;
            }
            return encode_prefill_gdn_conv_history(
                context, encoder, 0u, {}, encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool caller_owned_unchanged =
        !encoded && discarded && error.find("row count") != std::string::npos &&
        std::equal(history_before.begin(), history_before.end(),
                   history.allocation().begin()) &&
        std::equal(output_before.begin(), output_before.end(),
                   output.allocation().begin());
    check(caller_owned_unchanged, label + " caller-owned Encoder rollback",
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));

    auto standalone_command =
        context.make_command_buffer(label + " standalone", error);
    if (!standalone_command) {
        check(false, label + " standalone command buffer", error);
        return false;
    }
    const bool standalone_first = encode_prefill_gdn_conv_tiled(
        context, *standalone_command, 1u, convolution_buffers, error);
    const bool standalone_second =
        standalone_first && encode_prefill_gdn_conv_history(
                                context, *standalone_command, 0u, {},
                                error);
    const bool standalone_discarded =
        standalone_command->unsafe_native_handle() == nullptr;
    const bool standalone_unchanged =
        standalone_first && !standalone_second && standalone_discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(history_before.begin(), history_before.end(),
                   history.allocation().begin()) &&
        std::equal(output_before.begin(), output_before.end(),
                   output.allocation().begin());
    check(standalone_unchanged, label + " standalone wrapper rollback",
          error +
              (standalone_discarded
                   ? " (command buffer discarded)"
                   : " (command buffer remained submit-capable)"));
    return caller_owned_unchanged && standalone_unchanged;
}

bool test_prefill_gdn_conv_tiled_open_concurrent_group_rejected(
    Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN tiled convolution rejects open concurrent group";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    TestBuffer history, qkv, weights, output;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error) &&
        weights.allocate(context, label + " weights", weight_bytes, error) &&
        output.allocate(context, label + " output", row_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGDNConvTiledBuffers buffers{
        context.buffers().read_only(history.view()),
        context.buffers().read_only(qkv.view()),
        context.buffers().read_only(weights.view()), output.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GDN convolution",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_conv_tiled(
                context, encoder, 1u, buffers, encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_conv_tiled_failure_discards_prior_dispatch(
    Context& context) {
    const std::string label =
        "Prefill GDN tiled convolution failure discards prior dispatch";
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * sizeof(float);
    constexpr std::size_t weight_bytes =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel *
        sizeof(float);
    TestBuffer history, qkv, weights, output;
    std::string error;
    const bool allocated =
        history.allocate(context, label + " history", history_bytes, error) &&
        qkv.allocate(context, label + " qkv", row_bytes, error) &&
        weights.allocate(context, label + " weights", weight_bytes, error) &&
        output.allocate(context, label + " output", row_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const auto output_before = std::vector<std::uint8_t>(
        output.allocation().begin(), output.allocation().end());
    const PrefillGDNConvTiledBuffers buffers{
        context.buffers().read_only(history.view()),
        context.buffers().read_only(qkv.view()),
        context.buffers().read_only(weights.view()), output.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GDN tiled convolution rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gdn_conv_tiled(context, encoder, 1u, buffers,
                                               encode_error)) {
                return false;
            }
            return encode_prefill_gdn_conv_tiled(context, encoder, 0u, {},
                                                 encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && discarded && error.find("row count") != std::string::npos &&
        std::equal(output_before.begin(), output_before.end(),
                   output.allocation().begin());
    check(unchanged, label,
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));
    return unchanged;
}

bool reject_gr_write_norm_call(Context& context, std::uint32_t rows,
                               PrefillBF16X2Mode mode,
                               const PrefillGRWriteNormBuffers& buffers,
                               std::string_view expected_error,
                               const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "rejected Prefill GR write-norm",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_gr_write_norm(context, encoder, rows, mode,
                                                buffers, encode_error);
        },
        error);
    const bool right_error =
        !encoded && error.find(expected_error) != std::string::npos;
    check(right_error, label, error);
    return right_error;
}

bool test_gr_write_norm_validation(Context& context,
                                  Context& foreign_context) {
    const std::string label = "Prefill GR write-norm validation";
    constexpr std::size_t residual_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t block_bytes = kWidth * sizeof(float);
    constexpr std::size_t inject_bytes = kStreams * sizeof(float);
    constexpr std::size_t norm_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t scale_bytes = kStreams * sizeof(float);
    constexpr std::size_t output_bytes =
        kStreams * kWidth * sizeof(std::uint16_t);
    TestBuffer residual, block, inject, norm, rs, high, low, foreign;
    std::string error;
    const bool allocated =
        residual.allocate(context, label + " residual", residual_bytes,
                          error) &&
        block.allocate(context, label + " block", block_bytes, error) &&
        inject.allocate(context, label + " injection", inject_bytes, error) &&
        norm.allocate(context, label + " norm", norm_bytes, error) &&
        rs.allocate(context, label + " scale", scale_bytes, error) &&
        high.allocate(context, label + " high", output_bytes, error) &&
        low.allocate(context, label + " low", output_bytes, error) &&
        foreign.allocate(foreign_context, label + " foreign", residual_bytes,
                         error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const ReadOnlyBufferRef block_ro =
        context.buffers().read_only(block.view());
    const ReadOnlyBufferRef inject_ro =
        context.buffers().read_only(inject.view());
    const ReadOnlyBufferRef norm_ro =
        context.buffers().read_only(norm.view());
    const PrefillGRWriteNormBuffers valid{
        residual.view(), block_ro, inject_ro, norm_ro, rs.view(), high.view(),
        low.view()};
    bool passed = true;
    passed &= reject_gr_write_norm_call(
        context, 0u, PrefillBF16X2Mode::off, valid, "row count",
        label + " rejects zero rows");
    passed &= reject_gr_write_norm_call(
        context, 1u, static_cast<PrefillBF16X2Mode>(3u),
        {residual.view(), block_ro, inject_ro, norm_ro, rs.view(), high.view(),
         {}},
        "mode is invalid", label + " rejects unknown mode");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::all_projections,
        {residual.view(), block_ro, inject_ro, norm_ro, rs.view(), high.view(),
         {}},
        "requires xn16_lo", label + " requires mode-1 remainder");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off, valid,
        "only valid in mode 1", label + " rejects unused remainder");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual.view(), block_ro.slice(0u, block_bytes - sizeof(float)),
         inject_ro, norm_ro, rs.view(), high.view(), {}},
        "undersized", label + " rejects short block output");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual.view(), block_ro, inject_ro.slice(0u, inject_bytes - 4u),
         norm_ro, rs.view(), high.view(), {}},
        "undersized", label + " rejects short injection");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual.view(), block_ro, inject_ro,
         norm_ro.slice(0u, norm_bytes - sizeof(float)), rs.view(), high.view(),
         {}},
        "undersized", label + " rejects short next norm");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual.view(), context.buffers().read_only(residual.view()),
         inject_ro, norm_ro, rs.view(), high.view(), {}},
        "output overlaps a read-only input",
        label + " rejects output/input overlap");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {residual.view(), block_ro, inject_ro, norm_ro, residual.view(),
         high.view(), {}},
        "output buffer views overlap",
        label + " rejects output/output overlap");
    passed &= reject_gr_write_norm_call(
        context, 1u, PrefillBF16X2Mode::off,
        {foreign.view(), block_ro, inject_ro, norm_ro, rs.view(), high.view(),
         {}},
        "not owned by this Context",
        label + " rejects foreign Context view");
    return passed;
}

bool test_gr_write_norm_failure_discards_prior_dispatch(Context& context) {
    const std::string label =
        "Prefill GR write-norm failure discards earlier dispatch";
    constexpr std::size_t residual_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t norm_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t scale_bytes = kStreams * sizeof(float);
    constexpr std::size_t output_bytes =
        kStreams * kWidth * sizeof(std::uint16_t);
    TestBuffer residual, norm, rs, high;
    std::string error;
    const bool allocated =
        residual.allocate(context, label + " residual", residual_bytes,
                          error) &&
        norm.allocate(context, label + " norm", norm_bytes, error) &&
        rs.allocate(context, label + " scale", scale_bytes, error) &&
        high.allocate(context, label + " high", output_bytes, error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    const auto rs_before = std::vector<std::uint8_t>(
        rs.allocation().begin(), rs.allocation().end());
    const auto high_before = std::vector<std::uint8_t>(
        high.allocation().begin(), high.allocation().end());
    const PrefillGRNormBuffers norm_buffers{
        residual.view(), norm.view(), rs.view(), high.view(), {}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "Prefill GR write-norm rollback fixture",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gr_norm(context, encoder, 1u,
                                        PrefillBF16X2Mode::off, norm_buffers,
                                        encode_error)) {
                return false;
            }
            return encode_prefill_gr_write_norm(
                context, encoder, 0u, PrefillBF16X2Mode::off, {},
                encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !encoded && discarded &&
        error.find("row count") != std::string::npos &&
        std::equal(rs_before.begin(), rs_before.end(),
                   rs.allocation().begin()) &&
        std::equal(high_before.begin(), high_before.end(),
                   high.allocation().begin());
    check(unchanged, label,
          error + (discarded ? " (command buffer discarded)"
                             : " (command buffer remained submit-capable)"));
    return unchanged;
}

bool test_gr_write_norm_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GR write-norm rejects open concurrent group";
    constexpr std::size_t residual_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t block_bytes = kWidth * sizeof(float);
    constexpr std::size_t inject_bytes = kStreams * sizeof(float);
    constexpr std::size_t norm_bytes =
        kStreams * kWidth * sizeof(float);
    constexpr std::size_t scale_bytes = kStreams * sizeof(float);
    constexpr std::size_t output_bytes =
        kStreams * kWidth * sizeof(std::uint16_t);
    TestBuffer residual, block, inject, norm, rs, high;
    std::string error;
    const bool allocated =
        residual.allocate(context, label + " residual", residual_bytes,
                          error) &&
        block.allocate(context, label + " block", block_bytes, error) &&
        inject.allocate(context, label + " injection", inject_bytes, error) &&
        norm.allocate(context, label + " norm", norm_bytes, error) &&
        rs.allocate(context, label + " scale", scale_bytes, error) &&
        high.allocate(context, label + " high", output_bytes, error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGRWriteNormBuffers buffers{
        residual.view(), context.buffers().read_only(block.view()),
        context.buffers().read_only(inject.view()),
        context.buffers().read_only(norm.view()), rs.view(), high.view(), {}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill GR write-norm",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gr_write_norm(
                context, encoder, 1u, PrefillBF16X2Mode::off, buffers,
                encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label = "Prefill GR norm rejects open concurrent group";
    std::string error;
    TestBuffer residual, norm, rs, high;
    const bool allocated =
        residual.allocate(context, label + " residual",
                          kStreams * kWidth * sizeof(float), error) &&
        norm.allocate(context, label + " norm",
                      kStreams * kWidth * sizeof(float), error) &&
        rs.allocate(context, label + " rs", kStreams * sizeof(float), error) &&
        high.allocate(context, label + " high",
                      kStreams * kWidth * sizeof(std::uint16_t), error);
    if (!allocated) {
        check(false, label + " allocate buffers", error);
        return false;
    }
    const PrefillGRNormBuffers buffers{
        residual.view(), norm.view(), rs.view(), high.view(), {}};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "open concurrent Prefill group",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gr_norm(context, encoder, 1u,
                                          PrefillBF16X2Mode::off, buffers,
                                          encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

struct GDNTransformTestOutputs;

struct GDNTransformTestInputs {
    TestBuffer alpha_beta;
    TestBuffer dt_bias;
    TestBuffer a;
    TestBuffer qkv;
    TestBuffer conv_weights;
    TestBuffer z;
    TestBuffer gamma;

    bool allocate(Context& context, std::uint32_t rows,
                  const std::string& label, std::string& error) {
        return alpha_beta.allocate(
                   context, label + " alpha beta",
                   static_cast<std::size_t>(rows) * 2u * kGdnHeads *
                       sizeof(float),
                   error) &&
               dt_bias.allocate(context, label + " dt bias",
                                kGdnHeads * sizeof(float), error) &&
               a.allocate(context, label + " A",
                          kGdnHeads * sizeof(float), error) &&
               qkv.allocate(context, label + " qkv",
                             static_cast<std::size_t>(rows) *
                                 kGdnConvChannels * sizeof(float),
                             error) &&
               conv_weights.allocate(
                   context, label + " conv weights",
                   static_cast<std::size_t>(kGdnConvChannels) *
                       kGdnConvKernel * sizeof(float),
                   error) &&
               z.allocate(context, label + " z",
                          static_cast<std::size_t>(rows) * kGdnValueDim *
                              sizeof(float),
                          error) &&
               gamma.allocate(context, label + " gamma",
                              kGdnHeadDim * sizeof(float), error);
    }

    PrefillGDNTransformBuffers bind(Context& context,
                                    GDNTransformTestOutputs& outputs);
};

struct GDNTransformTestOutputs {
    TestBuffer history;
    TestBuffer gate;
    TestBuffer beta;
    TestBuffer h;
    TestBuffer state;
    TestBuffer oc;
    TestBuffer y_h;

    bool allocate(Context& context, std::uint32_t rows,
                  const std::string& label, std::string& error) {
        const std::size_t head_rows_bytes =
            static_cast<std::size_t>(rows) * kGdnHeads * sizeof(float);
        const std::size_t row_bytes =
            static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
        const std::size_t value_bytes =
            static_cast<std::size_t>(rows) * kGdnValueDim * sizeof(float);
        return history.allocate(
                   context, label + " history",
                   static_cast<std::size_t>(kGdnConvChannels) *
                       (kGdnConvKernel - 1u) * sizeof(float),
                   error) &&
               gate.allocate(context, label + " gate", head_rows_bytes,
                             error) &&
               beta.allocate(context, label + " beta", head_rows_bytes,
                             error) &&
               h.allocate(context, label + " h", row_bytes, error) &&
               state.allocate(context, label + " state",
                              static_cast<std::size_t>(kGdnStateElements) *
                                  sizeof(float),
                              error) &&
               oc.allocate(context, label + " oc", value_bytes, error) &&
               y_h.allocate(
                   context, label + " y_h",
                   static_cast<std::size_t>(rows) * kGdnValueDim *
                       sizeof(std::uint16_t),
                   error);
    }
};

PrefillGDNTransformBuffers GDNTransformTestInputs::bind(
    Context& context, GDNTransformTestOutputs& outputs) {
    return {context.buffers().read_only(alpha_beta.view()),
            context.buffers().read_only(dt_bias.view()),
            context.buffers().read_only(a.view()),
            outputs.history.view(),
            context.buffers().read_only(qkv.view()),
            context.buffers().read_only(conv_weights.view()),
            outputs.gate.view(),
            outputs.beta.view(),
            outputs.h.view(),
            outputs.state.view(),
            outputs.oc.view(),
            context.buffers().read_only(z.view()),
            context.buffers().read_only(gamma.view()),
            outputs.y_h.view()};
}

void initialize_gdn_transform_inputs(GDNTransformTestInputs& inputs,
                                     std::uint32_t rows) {
    auto* alpha_beta =
        reinterpret_cast<float*>(inputs.alpha_beta.payload().data());
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t head = 0u; head < kGdnHeads; ++head) {
            const std::size_t base =
                static_cast<std::size_t>(row) * 2u * kGdnHeads;
            const int alpha =
                static_cast<int>((row * 19u + head * 31u + 7u) % 101u) - 50;
            const int beta =
                static_cast<int>((row * 37u + head * 11u + 3u) % 97u) - 48;
            alpha_beta[base + head] = static_cast<float>(alpha) * 0.035f;
            alpha_beta[base + kGdnHeads + head] =
                static_cast<float>(beta) * 0.04f;
        }
    }
    auto* dt = reinterpret_cast<float*>(inputs.dt_bias.payload().data());
    auto* a_values = reinterpret_cast<float*>(inputs.a.payload().data());
    for (std::uint32_t head = 0u; head < kGdnHeads; ++head) {
        dt[head] = static_cast<float>(static_cast<int>(head % 17u) - 8) *
                   0.025f;
        a_values[head] = 0.35f + static_cast<float>(head % 9u) * 0.04f;
    }

    auto* qkv = reinterpret_cast<float*>(inputs.qkv.payload().data());
    const std::size_t qkv_values =
        static_cast<std::size_t>(rows) * kGdnConvChannels;
    for (std::size_t i = 0u; i < qkv_values; ++i) {
        const int centered = static_cast<int>((i * 23u + 5u) % 127u) - 63;
        qkv[i] = static_cast<float>(centered) * 0.0008f;
    }
    auto* conv_weights =
        reinterpret_cast<float*>(inputs.conv_weights.payload().data());
    const std::size_t weight_values =
        static_cast<std::size_t>(kGdnConvChannels) * kGdnConvKernel;
    for (std::size_t i = 0u; i < weight_values; ++i) {
        const int centered = static_cast<int>((i * 13u + 3u) % 53u) - 26;
        conv_weights[i] = static_cast<float>(centered) * 0.0015f;
    }
    auto* z = reinterpret_cast<float*>(inputs.z.payload().data());
    const std::size_t value_values =
        static_cast<std::size_t>(rows) * kGdnValueDim;
    for (std::size_t i = 0u; i < value_values; ++i) {
        const int centered = static_cast<int>((i * 17u + 29u) % 103u) - 51;
        z[i] = static_cast<float>(centered) * 0.03f;
    }
    auto* gamma = reinterpret_cast<float*>(inputs.gamma.payload().data());
    for (std::uint32_t i = 0u; i < kGdnHeadDim; ++i) {
        gamma[i] = 0.75f + static_cast<float>((i * 7u + 5u) % 31u) * 0.0125f;
    }
}

void initialize_gdn_transform_outputs(GDNTransformTestOutputs& outputs) {
    auto* history = reinterpret_cast<float*>(outputs.history.payload().data());
    const std::size_t history_values =
        static_cast<std::size_t>(kGdnConvChannels) * (kGdnConvKernel - 1u);
    for (std::size_t i = 0u; i < history_values; ++i) {
        const int centered = static_cast<int>((i * 11u + 7u) % 61u) - 30;
        history[i] = static_cast<float>(centered) * 0.0007f;
    }
    auto* state = reinterpret_cast<float*>(outputs.state.payload().data());
    for (std::size_t i = 0u; i < kGdnStateElements; ++i) {
        const int centered = static_cast<int>((i * 29u + 17u) % 101u) - 50;
        state[i] = static_cast<float>(centered) * 0.00012f;
    }
}

bool run_prefill_gdn_transforms_case(Context& context, std::uint32_t rows,
                                     bool standalone_api) {
    const std::string label =
        "Prefill GDN transforms T=" + std::to_string(rows) +
        (standalone_api ? " standalone" : " caller-owned");
    GDNTransformTestInputs inputs;
    GDNTransformTestOutputs composed, direct;
    std::string error;
    const bool allocated =
        inputs.allocate(context, rows, label + " inputs", error) &&
        composed.allocate(context, rows, label + " composed", error) &&
        direct.allocate(context, rows, label + " direct", error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    initialize_gdn_transform_inputs(inputs, rows);
    initialize_gdn_transform_outputs(composed);
    initialize_gdn_transform_outputs(direct);
    std::memcpy(direct.history.payload().data(),
                composed.history.payload().data(),
                composed.history.payload().size());
    std::memcpy(direct.state.payload().data(),
                composed.state.payload().data(),
                composed.state.payload().size());

    const std::array<TestBuffer*, 7u> read_only_inputs{
        &inputs.alpha_beta, &inputs.dt_bias, &inputs.a, &inputs.qkv,
        &inputs.conv_weights, &inputs.z, &inputs.gamma};
    std::array<std::vector<std::uint8_t>, 7u> input_snapshots;
    for (std::size_t i = 0u; i < read_only_inputs.size(); ++i) {
        const auto payload = read_only_inputs[i]->payload();
        input_snapshots[i].assign(payload.begin(), payload.end());
    }
    const PrefillGDNTransformBuffers composed_buffers =
        inputs.bind(context, composed);

    auto composed_command =
        context.make_command_buffer(label + " composed command", error);
    check(composed_command != nullptr, label + " composed command buffer",
          error);
    if (!composed_command) return false;
    const bool composed_encoded =
        standalone_api
            ? encode_prefill_gdn_transforms(context, *composed_command, rows,
                                            composed_buffers, error)
            : Encoder::encode(
                  context, *composed_command, label + " caller-owned",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_gdn_transforms(
                          context, encoder, rows, composed_buffers,
                          encode_error);
                  },
                  error);
    check(composed_encoded, label + " composed encode", error);
    if (!composed_encoded) {
        composed_command->discard();
        return false;
    }
    if (!submit_and_wait(composed_command, label + " composed", error)) {
        check(false, label + " composed dispatch", error);
        return false;
    }

    const PrefillGDNTransformBuffers direct_buffers =
        inputs.bind(context, direct);
    auto direct_command =
        context.make_command_buffer(label + " direct command", error);
    check(direct_command != nullptr, label + " direct command buffer", error);
    if (!direct_command) return false;
    const bool direct_encoded = Encoder::encode(
        context, *direct_command, label + " direct component sequence",
        [&](Encoder& encoder, std::string& encode_error) {
            const PrefillGDNGatesBuffers gates{
                direct_buffers.alpha_beta, direct_buffers.dt_bias,
                direct_buffers.a, direct_buffers.gate, direct_buffers.beta};
            const PrefillGDNConvTiledBuffers conv{
                context.buffers().read_only(direct_buffers.history),
                direct_buffers.qkv, direct_buffers.conv_weights,
                direct_buffers.h};
            const PrefillGDNConvHistoryBuffers history{
                direct_buffers.history, direct_buffers.qkv};
            const PrefillGDNRecBuffers rec{
                direct_buffers.state,
                context.buffers().read_only(direct_buffers.h),
                context.buffers().read_only(direct_buffers.gate),
                context.buffers().read_only(direct_buffers.beta),
                direct_buffers.oc};
            const PrefillGDNOutNormBuffers out_norm{
                context.buffers().read_only(direct_buffers.oc), direct_buffers.z,
                direct_buffers.gamma, direct_buffers.y_h};
            return encode_prefill_gdn_gates(context, encoder, rows, gates,
                                             encode_error) &&
                   encode_prefill_gdn_conv_tiled(context, encoder, rows, conv,
                                                 encode_error) &&
                   encode_prefill_gdn_conv_history(context, encoder, rows,
                                                   history, encode_error) &&
                   encode_prefill_gdn_l2(context, encoder, rows,
                                         direct_buffers.h, encode_error) &&
                   encode_prefill_gdn_rec(context, encoder, rows, rec,
                                          encode_error) &&
                   encode_prefill_gdn_out_norm(context, encoder, rows,
                                               out_norm, encode_error);
        },
        error);
    check(direct_encoded, label + " direct component encode", error);
    if (!direct_encoded) {
        direct_command->discard();
        return false;
    }
    if (!submit_and_wait(direct_command, label + " direct", error)) {
        check(false, label + " direct dispatch", error);
        return false;
    }

    const std::array<TestBuffer*, 7u> composed_outputs{
        &composed.history, &composed.gate, &composed.beta, &composed.h,
        &composed.state, &composed.oc, &composed.y_h};
    const std::array<TestBuffer*, 7u> direct_outputs{
        &direct.history, &direct.gate, &direct.beta, &direct.h, &direct.state,
        &direct.oc, &direct.y_h};
    const std::array<std::string_view, 7u> output_names{
        "history update", "gate", "beta", "convolution and q/k h",
        "recurrent state", "oc", "F16 y_h"};
    bool outputs_match = true;
    for (std::size_t i = 0u; i < composed_outputs.size(); ++i) {
        const auto left = composed_outputs[i]->payload();
        const auto right = direct_outputs[i]->payload();
        const bool match =
            left.size() == right.size() &&
            std::equal(left.begin(), left.end(), right.begin());
        check(match, label + " matches direct " + std::string(output_names[i]));
        outputs_match &= match;
    }
    bool inputs_unchanged = true;
    for (std::size_t i = 0u; i < read_only_inputs.size(); ++i) {
        const auto payload = read_only_inputs[i]->payload();
        inputs_unchanged &=
            std::equal(input_snapshots[i].begin(), input_snapshots[i].end(),
                       payload.begin());
    }
    check(inputs_unchanged, label + " read-only projections remain unchanged");
    bool guards = true;
    for (TestBuffer* buffer : read_only_inputs) {
        guards &= buffer->guards_intact();
    }
    for (TestBuffer* buffer : composed_outputs) {
        guards &= buffer->guards_intact();
    }
    for (TestBuffer* buffer : direct_outputs) {
        guards &= buffer->guards_intact();
    }
    check(guards, label + " guard regions");

    const std::size_t value_bytes =
        static_cast<std::size_t>(rows) * kGdnValueDim * sizeof(float);
    PrefillGDNTransformBuffers late_alias = inputs.bind(context, composed);
    late_alias.oc = composed.h.view().slice(0u, value_bytes);
    auto alias_command =
        context.make_command_buffer(label + " late alias", error);
    if (!alias_command) {
        check(false, label + " late alias command buffer", error);
        return false;
    }
    const bool alias_encoded =
        standalone_api
            ? encode_prefill_gdn_transforms(context, *alias_command, rows,
                                            late_alias, error)
            : Encoder::encode(
                  context, *alias_command, label + " caller-owned alias",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_gdn_transforms(
                          context, encoder, rows, late_alias, encode_error);
                  },
                  error);
    const bool alias_rejected =
        !alias_encoded && alias_command->unsafe_native_handle() == nullptr &&
        error.find("writable stages overlap") != std::string::npos &&
        error.find("h and oc") != std::string::npos;
    check(alias_rejected, label + " rejects late cross-stage h/oc alias",
          error);

    const std::size_t y_h_bytes =
        static_cast<std::size_t>(rows) * kGdnValueDim *
        sizeof(std::uint16_t);
    PrefillGDNTransformBuffers short_final = inputs.bind(context, composed);
    short_final.y_h = composed.y_h.view().slice(0u, y_h_bytes - 2u);
    auto short_command =
        context.make_command_buffer(label + " short y_h", error);
    if (!short_command) {
        check(false, label + " short y_h command buffer", error);
        return false;
    }
    const bool short_encoded =
        standalone_api
            ? encode_prefill_gdn_transforms(context, *short_command, rows,
                                            short_final, error)
            : Encoder::encode(
                  context, *short_command, label + " caller-owned short y_h",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_gdn_transforms(
                          context, encoder, rows, short_final, encode_error);
                  },
                  error);
    const bool short_rejected =
        !short_encoded && short_command->unsafe_native_handle() == nullptr &&
        error.find("y_h buffer is undersized") != std::string::npos;
    check(short_rejected, label + " rejects short final output before dispatch",
          error);

    const std::size_t history_bytes =
        static_cast<std::size_t>(kGdnConvChannels) *
        (kGdnConvKernel - 1u) * sizeof(float);
    const std::size_t qkv_bytes =
        static_cast<std::size_t>(rows) * kGdnConvChannels * sizeof(float);
    TestBuffer shared_history_qkv;
    std::string shared_error;
    const bool arena_allocated = shared_history_qkv.allocate(
        context, label + " shared history/qkv arena",
        history_bytes + qkv_bytes, shared_error);
    check(arena_allocated, label + " allocate shared history/qkv arena",
          shared_error);
    if (!arena_allocated) return false;
    PrefillGDNTransformBuffers shared_ranges =
        inputs.bind(context, composed);
    shared_ranges.history =
        shared_history_qkv.view().slice(0u, history_bytes + qkv_bytes);
    shared_ranges.qkv = context.buffers().read_only(
        shared_history_qkv.view().slice(history_bytes, qkv_bytes));
    auto shared_command =
        context.make_command_buffer(label + " disjoint arena ranges",
                                    shared_error);
    check(shared_command != nullptr,
          label + " disjoint arena command buffer", shared_error);
    if (!shared_command) return false;
    const bool shared_encoded =
        standalone_api
            ? encode_prefill_gdn_transforms(context, *shared_command, rows,
                                            shared_ranges, shared_error)
            : Encoder::encode(
                  context, *shared_command,
                  label + " disjoint arena caller-owned",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_gdn_transforms(
                          context, encoder, rows, shared_ranges, encode_error);
                  },
                  shared_error);
    const bool shared_ranges_accepted =
        shared_encoded && shared_command->unsafe_native_handle() != nullptr;
    shared_command->discard();
    check(shared_ranges_accepted,
          label + " accepts disjoint active ranges in shared arena",
          shared_error);
    check(shared_history_qkv.guards_intact(),
          label + " shared arena guard regions");
    return outputs_match && inputs_unchanged && guards && alias_rejected &&
           short_rejected && shared_ranges_accepted &&
           shared_history_qkv.guards_intact();
}

struct GDNBlockTestOutputs;

struct GDNBlockTestInputs {
    TestBuffer mixed_h;
    TestBuffer mixed_bf;
    TestBuffer mixed_bf_lo;
    TestBuffer qkv_weights;
    TestBuffer gate_weights;
    TestBuffer alpha_weights;
    TestBuffer beta_weights;
    TestBuffer out_weights;
    TestBuffer dt_bias;
    TestBuffer a;
    TestBuffer conv_weights;
    TestBuffer gamma;

    bool allocate(Context& context, std::uint32_t rows,
                  const std::string& label, std::string& error) {
        const std::size_t q8_qkv =
            static_cast<std::size_t>(GDN_CONV_CHANNELS) * N_EMBD / 32u *
            sizeof(block_q8_0);
        const std::size_t q8_gate =
            static_cast<std::size_t>(GDN_VALUE_DIM) * N_EMBD / 32u *
            sizeof(block_q8_0);
        const std::size_t q8_out =
            static_cast<std::size_t>(N_EMBD) * GDN_VALUE_DIM / 32u *
            sizeof(block_q8_0);
        const std::size_t bf16_projection =
            static_cast<std::size_t>(GDN_V_HEADS) * N_EMBD * kGdnHalfBytes;
        const std::size_t activation_bytes =
            static_cast<std::size_t>(rows) * N_EMBD * kGdnHalfBytes;
        return mixed_h.allocate(context, label + " mixed_h",
                                activation_bytes, error) &&
               mixed_bf.allocate(context, label + " mixed_bf",
                                 activation_bytes, error) &&
               mixed_bf_lo.allocate(context, label + " mixed_bf_lo",
                                     activation_bytes, error) &&
               qkv_weights.allocate(context, label + " attn_qkv Q8_0",
                                    q8_qkv, error) &&
               gate_weights.allocate(context, label + " attn_gate Q8_0",
                                     q8_gate, error) &&
               alpha_weights.allocate(context, label + " ssm_alpha BF16",
                                      bf16_projection, error) &&
               beta_weights.allocate(context, label + " ssm_beta BF16",
                                     bf16_projection, error) &&
               out_weights.allocate(context, label + " ssm_out Q8_0",
                                    q8_out, error) &&
               dt_bias.allocate(context, label + " ssm_dt.bias",
                                 GDN_V_HEADS * sizeof(float), error) &&
               a.allocate(context, label + " ssm_a",
                          GDN_V_HEADS * sizeof(float), error) &&
               conv_weights.allocate(
                   context, label + " ssm_conv1d",
                   static_cast<std::size_t>(GDN_CONV_CHANNELS) *
                       GDN_CONV_KERNEL * sizeof(float),
                   error) &&
               gamma.allocate(context, label + " ssm_norm",
                              GDN_HEAD_DIM * sizeof(float), error);
    }

    PrefillGDNBlockBuffers bind(Context& context,
                                struct GDNBlockTestOutputs& outputs,
                                PrefillBF16X2Mode mode);
};

struct GDNBlockTestOutputs {
    TestBuffer dequantized_f16;
    TestBuffer qkv;
    TestBuffer z;
    TestBuffer alpha_beta;
    TestBuffer history;
    TestBuffer gate;
    TestBuffer beta;
    TestBuffer h;
    TestBuffer state;
    TestBuffer oc;
    TestBuffer y_h;
    TestBuffer bo;

    bool allocate(Context& context, std::uint32_t rows,
                  const std::string& label, std::string& error) {
        const std::size_t row_values =
            static_cast<std::size_t>(rows) * GDN_CONV_CHANNELS;
        const std::size_t value_values =
            static_cast<std::size_t>(rows) * GDN_VALUE_DIM;
        return dequantized_f16.allocate(
                   context, label + " dequantized_f16",
                   static_cast<std::size_t>(GDN_CONV_CHANNELS) * N_EMBD *
                       kGdnHalfBytes,
                   error) &&
               qkv.allocate(context, label + " qkv",
                            row_values * sizeof(float), error) &&
               z.allocate(context, label + " z",
                          value_values * sizeof(float), error) &&
               alpha_beta.allocate(
                   context, label + " alpha_beta",
                   static_cast<std::size_t>(rows) * 2u * GDN_V_HEADS *
                       sizeof(float),
                   error) &&
               history.allocate(
                   context, label + " history",
                   static_cast<std::size_t>(GDN_HISTORY_ELEMENTS) *
                       sizeof(float),
                   error) &&
               gate.allocate(context, label + " gate",
                             static_cast<std::size_t>(rows) * GDN_V_HEADS *
                                 sizeof(float),
                             error) &&
               beta.allocate(context, label + " beta",
                             static_cast<std::size_t>(rows) * GDN_V_HEADS *
                                 sizeof(float),
                             error) &&
               h.allocate(context, label + " h",
                          row_values * sizeof(float), error) &&
               state.allocate(context, label + " state",
                              static_cast<std::size_t>(GDN_STATE_ELEMENTS) *
                                  sizeof(float),
                              error) &&
               oc.allocate(context, label + " oc",
                           value_values * sizeof(float), error) &&
               y_h.allocate(context, label + " y_h",
                            value_values * kGdnHalfBytes, error) &&
               bo.allocate(context, label + " bo",
                           static_cast<std::size_t>(rows) * N_EMBD *
                               sizeof(float),
                           error);
    }

    void initialize_persistent_state() {
        auto* history_values =
            reinterpret_cast<float*>(history.payload().data());
        for (std::size_t i = 0u; i < history.payload().size() / sizeof(float);
             ++i) {
            const int centered =
                static_cast<int>((i * 13u + 9u) % 71u) - 35;
            history_values[i] = static_cast<float>(centered) * 0.0006f;
        }
        auto* state_values =
            reinterpret_cast<float*>(state.payload().data());
        for (std::size_t i = 0u; i < GDN_STATE_ELEMENTS; ++i) {
            const int centered =
                static_cast<int>((i * 17u + 11u) % 83u) - 41;
            state_values[i] = static_cast<float>(centered) * 0.00008f;
        }
    }
};

PrefillGDNBlockBuffers GDNBlockTestInputs::bind(
    Context& context, GDNBlockTestOutputs& outputs,
    PrefillBF16X2Mode mode) {
    PrefillGDNBlockBuffers buffers;
    buffers.mixed_h = context.buffers().read_only(mixed_h.view());
    buffers.mixed_bf = context.buffers().read_only(mixed_bf.view());
    if (mode != PrefillBF16X2Mode::off) {
        buffers.mixed_bf_lo =
            context.buffers().read_only(mixed_bf_lo.view());
    }
    buffers.attn_qkv_weights =
        context.buffers().read_only(qkv_weights.view());
    buffers.attn_qkv_format = PrefillNativeWeightFormat::Q8_0;
    buffers.attn_gate_weights =
        context.buffers().read_only(gate_weights.view());
    buffers.attn_gate_format = PrefillNativeWeightFormat::Q8_0;
    buffers.ssm_alpha_weights =
        context.buffers().read_only(alpha_weights.view());
    buffers.ssm_beta_weights =
        context.buffers().read_only(beta_weights.view());
    buffers.ssm_out_weights = context.buffers().read_only(out_weights.view());
    buffers.ssm_out_format = PrefillNativeWeightFormat::Q8_0;
    buffers.dt_bias = context.buffers().read_only(dt_bias.view());
    buffers.a = context.buffers().read_only(a.view());
    buffers.conv_weights = context.buffers().read_only(conv_weights.view());
    buffers.gamma = context.buffers().read_only(gamma.view());
    buffers.dequantized_f16 = outputs.dequantized_f16.view();
    buffers.qkv = outputs.qkv.view();
    buffers.z = outputs.z.view();
    buffers.alpha_beta = outputs.alpha_beta.view();
    buffers.history = outputs.history.view();
    buffers.gate = outputs.gate.view();
    buffers.beta = outputs.beta.view();
    buffers.h = outputs.h.view();
    buffers.state = outputs.state.view();
    buffers.oc = outputs.oc.view();
    buffers.y_h = outputs.y_h.view();
    buffers.bo = outputs.bo.view();
    return buffers;
}

void initialize_gdn_block_inputs(GDNBlockTestInputs& inputs,
                                 std::uint32_t rows) {
    auto* mixed_h = reinterpret_cast<std::uint16_t*>(
        inputs.mixed_h.payload().data());
    const std::size_t activation_values =
        static_cast<std::size_t>(rows) * N_EMBD;
    for (std::size_t i = 0u; i < activation_values; ++i) {
        const int centered = static_cast<int>((i * 19u + 7u) % 101u) - 50;
        mixed_h[i] = strata::kernels::f16_from_f32(
            static_cast<float>(centered) * 0.0009f);
    }

    auto* mixed_bf = reinterpret_cast<std::uint16_t*>(
        inputs.mixed_bf.payload().data());
    auto* mixed_bf_lo = reinterpret_cast<std::uint16_t*>(
        inputs.mixed_bf_lo.payload().data());
    for (std::size_t i = 0u; i < activation_values; ++i) {
        const int high = static_cast<int>((i * 23u + 11u) % 79u) - 39;
        const int low = static_cast<int>((i * 31u + 3u) % 43u) - 21;
        mixed_bf[i] = strata::kernels::bf16_from_f32(
            static_cast<float>(high) * 0.0004f);
        mixed_bf_lo[i] = strata::kernels::bf16_from_f32(
            static_cast<float>(low) * 0.000015f);
    }

    auto fill_q8 = [](TestBuffer& buffer, std::uint32_t input_width,
                      std::uint32_t output_width, std::uint32_t salt) {
        auto* blocks = reinterpret_cast<block_q8_0*>(
            buffer.payload().data());
        const std::uint32_t blocks_per_row = input_width / 32u;
        const std::size_t block_count =
            static_cast<std::size_t>(output_width) * blocks_per_row;
        for (std::size_t index = 0u; index < block_count; ++index) {
            block_q8_0& block = blocks[index];
            block.d = strata::kernels::f16_from_f32(1.0f / 32.0f);
            const std::uint32_t output_row =
                static_cast<std::uint32_t>(index / blocks_per_row);
            for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                const int code = static_cast<int>(
                    (output_row * 11u + static_cast<std::uint32_t>(index) * 3u +
                     lane * 5u + salt) %
                    9u) -
                                 4;
                block.qs[lane] = static_cast<std::int8_t>(code);
            }
        }
    };
    fill_q8(inputs.qkv_weights, N_EMBD, GDN_CONV_CHANNELS, 1u);
    fill_q8(inputs.gate_weights, N_EMBD, GDN_VALUE_DIM, 4u);
    fill_q8(inputs.out_weights, GDN_VALUE_DIM, N_EMBD, 7u);

    auto fill_bf16 = [](TestBuffer& buffer, std::uint32_t rows,
                        std::uint32_t columns, std::uint32_t salt) {
        auto* weights =
            reinterpret_cast<std::uint16_t*>(buffer.payload().data());
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < columns; ++column) {
                const int centered = static_cast<int>(
                    (row * 13u + column * 7u + salt) % 17u) -
                                     8;
                weights[static_cast<std::size_t>(row) * columns + column] =
                    strata::kernels::bf16_from_f32(
                        static_cast<float>(centered) * 0.0003f);
            }
        }
    };
    fill_bf16(inputs.alpha_weights, GDN_V_HEADS, N_EMBD, 2u);
    fill_bf16(inputs.beta_weights, GDN_V_HEADS, N_EMBD, 5u);

    auto* dt_bias =
        reinterpret_cast<float*>(inputs.dt_bias.payload().data());
    auto* a = reinterpret_cast<float*>(inputs.a.payload().data());
    for (std::uint32_t head = 0u; head < GDN_V_HEADS; ++head) {
        dt_bias[head] =
            static_cast<float>(static_cast<int>(head % 13u) - 6) * 0.03f;
        a[head] = 0.4f + static_cast<float>(head % 7u) * 0.05f;
    }
    auto* conv_weights =
        reinterpret_cast<float*>(inputs.conv_weights.payload().data());
    for (std::size_t i = 0u;
         i < inputs.conv_weights.payload().size() / sizeof(float); ++i) {
        const int centered = static_cast<int>((i * 29u + 5u) % 47u) - 23;
        conv_weights[i] = static_cast<float>(centered) * 0.0012f;
    }
    auto* gamma = reinterpret_cast<float*>(inputs.gamma.payload().data());
    for (std::uint32_t i = 0u; i < GDN_HEAD_DIM; ++i) {
        gamma[i] = 0.8f + static_cast<float>((i * 3u + 1u) % 23u) * 0.01f;
    }
}

bool encode_direct_gdn_block(Context& context, Encoder& encoder,
                             std::uint32_t rows, PrefillBF16X2Mode mode,
                             const PrefillGDNBlockBuffers& buffers,
                             std::string& error) {
    const PrefillGemmMPPParams qkv_params{
        rows, GDN_CONV_CHANNELS, N_EMBD, GDN_CONV_CHANNELS, 0u, false};
    const PrefillGemmMPPParams gate_params{
        rows, GDN_VALUE_DIM, N_EMBD, GDN_VALUE_DIM, 0u, false};
    const PrefillGemmMPPParams alpha_params{
        rows, GDN_V_HEADS, N_EMBD, 2u * GDN_V_HEADS, 0u, false};
    const PrefillGemmMPPParams beta_params{
        rows, GDN_V_HEADS, N_EMBD, 2u * GDN_V_HEADS, GDN_V_HEADS, false};
    const PrefillGemmMPPParams out_params{
        rows, N_EMBD, GDN_VALUE_DIM, N_EMBD, 0u, false};

    if (!encode_prefill_native_projection(
            context, encoder, qkv_params, buffers.attn_qkv_format,
            {buffers.mixed_h, buffers.attn_qkv_weights,
             buffers.dequantized_f16, buffers.qkv},
            error) ||
        !encode_prefill_native_projection(
            context, encoder, gate_params, buffers.attn_gate_format,
            {buffers.mixed_h, buffers.attn_gate_weights,
             buffers.dequantized_f16, buffers.z},
            error)) {
        return false;
    }
    if (mode == PrefillBF16X2Mode::off) {
        if (!encode_prefill_gemm(
                context, encoder, alpha_params,
                {buffers.mixed_bf, buffers.ssm_alpha_weights,
                 buffers.alpha_beta},
                PrefillGemmFormat::BF16, error) ||
            !encode_prefill_gemm(
                context, encoder, beta_params,
                {buffers.mixed_bf, buffers.ssm_beta_weights,
                 buffers.alpha_beta},
                PrefillGemmFormat::BF16, error)) {
            return false;
        }
    } else {
        if (!encode_prefill_gemm_bf16x2(
                context, encoder, alpha_params,
                {buffers.mixed_bf, buffers.mixed_bf_lo,
                 buffers.ssm_alpha_weights, buffers.alpha_beta},
                error) ||
            !encode_prefill_gemm_bf16x2(
                context, encoder, beta_params,
                {buffers.mixed_bf, buffers.mixed_bf_lo,
                 buffers.ssm_beta_weights, buffers.alpha_beta},
                error)) {
            return false;
        }
    }

    const PrefillGDNTransformBuffers transform_buffers{
        context.buffers().read_only(buffers.alpha_beta),
        buffers.dt_bias,
        buffers.a,
        buffers.history,
        context.buffers().read_only(buffers.qkv),
        buffers.conv_weights,
        buffers.gate,
        buffers.beta,
        buffers.h,
        buffers.state,
        buffers.oc,
        context.buffers().read_only(buffers.z),
        buffers.gamma,
        buffers.y_h};
    return encode_prefill_gdn_transforms(
               context, encoder, rows, transform_buffers, error) &&
           encode_prefill_native_projection(
               context, encoder, out_params, buffers.ssm_out_format,
               {context.buffers().read_only(buffers.y_h),
                buffers.ssm_out_weights, buffers.dequantized_f16, buffers.bo},
               error);
}

struct GDNTransitionTestOutputs {
    TestBuffer residual;
    PrefillGRReadTestOutputs gr;
    TestBuffer mixed_bf;
    TestBuffer mixed_bf_lo;
    TestBuffer mixed_h;
    GDNBlockTestOutputs gdn;

    bool allocate(Context& context, std::uint32_t rows,
                  PrefillBF16X2Mode mode, const std::string& label,
                  std::string& error) {
        const bool read_low =
            mode == PrefillBF16X2Mode::all_projections;
        const bool mix_low = mode != PrefillBF16X2Mode::off;
        const std::size_t residual_bytes =
            static_cast<std::size_t>(rows) * kStreams * kWidth *
            sizeof(float);
        const std::size_t mixer_bytes =
            static_cast<std::size_t>(rows) * kWidth * kGdnHalfBytes;
        return residual.allocate(context, label + " residual",
                                 residual_bytes, error) &&
               gr.allocate(context, rows, read_low, label + " GR", error) &&
               mixed_bf.allocate(context, label + " mixed BF16",
                                 mixer_bytes, error) &&
               (!mix_low ||
                mixed_bf_lo.allocate(context, label + " mixed BF16 low",
                                     mixer_bytes, error)) &&
               mixed_h.allocate(context, label + " mixed F16", mixer_bytes,
                                error) &&
               gdn.allocate(context, rows, label + " GDN", error);
    }

    void collect_buffers(std::vector<TestBuffer*>& output) {
        output.insert(output.end(),
                      {&residual, &gr.rs, &gr.xn16, &gr.xn16_lo,
                       &gr.low_rank, &gr.low_rank_bf16,
                       &gr.low_rank_bf16_lo, &gr.gated, &gr.inject,
                       &mixed_bf, &mixed_bf_lo, &mixed_h,
                       &gdn.dequantized_f16, &gdn.qkv, &gdn.z,
                       &gdn.alpha_beta, &gdn.history, &gdn.gate,
                       &gdn.beta, &gdn.h, &gdn.state, &gdn.oc,
                       &gdn.y_h, &gdn.bo});
    }
};

PrefillGRReadMixBuffers bind_gdn_layer_read_mix(
    GDNTransitionTestOutputs& outputs,
    BufferRef norm, ReadOnlyBufferRef down_weights,
    ReadOnlyBufferRef up_weights, ReadOnlyBufferRef inject_weights,
    PrefillBF16X2Mode mode) {
    const bool read_low =
        mode == PrefillBF16X2Mode::all_projections;
    const bool mix_low = mode != PrefillBF16X2Mode::off;
    return {
        outputs.gr.bind(outputs.residual.view(), norm, down_weights,
                        up_weights, inject_weights, read_low),
        outputs.mixed_bf.view(),
        mix_low ? outputs.mixed_bf_lo.view() : BufferRef{},
        outputs.mixed_h.view()};
}

PrefillAttentionGDNTransitionBuffers bind_gdn_transition(
    Context& context, GDNBlockTestInputs& inputs,
    GDNTransitionTestOutputs& outputs, BufferRef norm,
    ReadOnlyBufferRef down_weights, ReadOnlyBufferRef up_weights,
    ReadOnlyBufferRef inject_weights, ReadOnlyBufferRef next_norm,
    PrefillBF16X2Mode mode) {
    PrefillGDNBlockBuffers gdn =
        inputs.bind(context, outputs.gdn, mode);
    gdn.mixed_h = {};
    gdn.mixed_bf = {};
    gdn.mixed_bf_lo = {};
    return {bind_gdn_layer_read_mix(outputs, norm, down_weights, up_weights,
                                    inject_weights, mode),
            gdn, next_norm};
}

bool run_prefill_gdn_block_case(Context& context, std::uint32_t rows,
                                PrefillBF16X2Mode mode,
                                bool standalone_api) {
    const std::string label =
        "Prefill GDN block T=" + std::to_string(rows) + " mode=" +
        std::to_string(static_cast<std::uint32_t>(mode)) +
        (standalone_api ? " standalone" : " caller-owned");
    GDNBlockTestInputs inputs;
    GDNBlockTestOutputs composed, direct;
    std::string error;
    const bool allocated =
        inputs.allocate(context, rows, label + " inputs", error) &&
        composed.allocate(context, rows, label + " composed", error) &&
        direct.allocate(context, rows, label + " direct", error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    initialize_gdn_block_inputs(inputs, rows);

    const std::array<TestBuffer*, 12u> read_only_inputs{
        &inputs.mixed_h,     &inputs.mixed_bf,   &inputs.mixed_bf_lo,
        &inputs.qkv_weights, &inputs.gate_weights, &inputs.alpha_weights,
        &inputs.beta_weights, &inputs.out_weights, &inputs.dt_bias,
        &inputs.a,           &inputs.conv_weights, &inputs.gamma};
    std::array<std::vector<std::uint8_t>, 12u> input_snapshots;
    for (std::size_t i = 0u; i < read_only_inputs.size(); ++i) {
        const auto payload = read_only_inputs[i]->payload();
        input_snapshots[i].assign(payload.begin(), payload.end());
    }

    composed.initialize_persistent_state();
    direct.initialize_persistent_state();
    std::memcpy(direct.history.payload().data(),
                composed.history.payload().data(),
                composed.history.payload().size());
    std::memcpy(direct.state.payload().data(),
                composed.state.payload().data(),
                composed.state.payload().size());

    const PrefillGDNBlockBuffers composed_buffers =
        inputs.bind(context, composed, mode);
    auto composed_command =
        context.make_command_buffer(label + " composed command", error);
    check(composed_command != nullptr, label + " composed command buffer",
          error);
    if (!composed_command) return false;
    const bool composed_encoded =
        standalone_api
            ? encode_prefill_gdn_block(context, *composed_command, rows, mode,
                                       composed_buffers, error)
            : Encoder::encode(
                  context, *composed_command, label + " caller-owned",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_gdn_block(
                          context, encoder, rows, mode, composed_buffers,
                          encode_error);
                  },
                  error);
    check(composed_encoded, label + " composed encode", error);
    if (!composed_encoded) {
        composed_command->discard();
        return false;
    }
    if (!submit_and_wait(composed_command, label + " composed", error)) {
        check(false, label + " composed dispatch", error);
        return false;
    }

    const PrefillGDNBlockBuffers direct_buffers =
        inputs.bind(context, direct, mode);
    auto direct_command =
        context.make_command_buffer(label + " direct command", error);
    check(direct_command != nullptr, label + " direct command buffer", error);
    if (!direct_command) return false;
    const bool direct_encoded = Encoder::encode(
        context, *direct_command, label + " direct component sequence",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_direct_gdn_block(context, encoder, rows, mode,
                                           direct_buffers, encode_error);
        },
        error);
    check(direct_encoded, label + " direct component encode", error);
    if (!direct_encoded) {
        direct_command->discard();
        return false;
    }
    if (!submit_and_wait(direct_command, label + " direct", error)) {
        check(false, label + " direct dispatch", error);
        return false;
    }

    const std::array<TestBuffer*, 12u> composed_outputs{
        &composed.dequantized_f16, &composed.qkv,   &composed.z,
        &composed.alpha_beta,      &composed.history, &composed.gate,
        &composed.beta,            &composed.h,    &composed.state,
        &composed.oc,              &composed.y_h,  &composed.bo};
    const std::array<TestBuffer*, 12u> direct_outputs{
        &direct.dequantized_f16, &direct.qkv,   &direct.z,
        &direct.alpha_beta,      &direct.history, &direct.gate,
        &direct.beta,            &direct.h,    &direct.state,
        &direct.oc,              &direct.y_h,  &direct.bo};
    const std::array<std::string_view, 12u> output_names{
        "reused F16 dequant scratch", "qkv", "z", "alpha_beta",
        "history", "gate", "beta", "convolution h", "recurrent state",
        "oc", "F16 y_h", "bo"};
    bool outputs_match = true;
    for (std::size_t i = 0u; i < composed_outputs.size(); ++i) {
        const auto left = composed_outputs[i]->payload();
        const auto right = direct_outputs[i]->payload();
        const bool match =
            left.size() == right.size() &&
            std::equal(left.begin(), left.end(), right.begin());
        check(match, label + " matches direct " + std::string(output_names[i]));
        outputs_match &= match;
    }

    bool inputs_unchanged = true;
    for (std::size_t i = 0u; i < read_only_inputs.size(); ++i) {
        const auto payload = read_only_inputs[i]->payload();
        inputs_unchanged &=
            std::equal(input_snapshots[i].begin(), input_snapshots[i].end(),
                       payload.begin());
    }
    check(inputs_unchanged,
          label + " leaves projection inputs and weights unchanged");
    bool guards = true;
    for (TestBuffer* buffer : read_only_inputs) {
        guards &= buffer->guards_intact();
    }
    for (TestBuffer* buffer : composed_outputs) {
        guards &= buffer->guards_intact();
    }
    for (TestBuffer* buffer : direct_outputs) {
        guards &= buffer->guards_intact();
    }
    check(guards, label + " guard regions");

    bool rollback = true;
    if (rows == 1u && mode == PrefillBF16X2Mode::except_gr) {
        ScopedEnvironment invalid_selector;
        const bool environment_set =
            invalid_selector.set("STRATA_METAL_GEMM", "invalid");
        check(environment_set, label + " set invalid GEMM selector");
        rollback &= environment_set;
        if (environment_set) {
            std::array<std::vector<std::uint8_t>, 12u> output_snapshots;
            for (std::size_t i = 0u; i < composed_outputs.size(); ++i) {
                const auto payload = composed_outputs[i]->payload();
                output_snapshots[i].assign(payload.begin(), payload.end());
            }

            const std::array<bool, 2u> standalone_forms{true, false};
            for (const bool standalone_form : standalone_forms) {
                std::string rollback_error;
                const std::string rollback_label =
                    label + (standalone_form ? " standalone rollback"
                                             : " caller-owned rollback");
                auto rollback_command = context.make_command_buffer(
                    rollback_label, rollback_error);
                if (!rollback_command) {
                    check(false, rollback_label + " command buffer",
                          rollback_error);
                    rollback = false;
                    continue;
                }
                bool prefix_dispatch_observed = false;
                const bool encoded =
                    standalone_form
                        ? encode_prefill_gdn_block(
                              context, *rollback_command, rows, mode,
                              composed_buffers, rollback_error)
                        : Encoder::encode(
                              context, *rollback_command, rollback_label,
                              [&](Encoder& encoder,
                                  std::string& encode_error) {
                                  const bool block_encoded =
                                      encode_prefill_gdn_block(
                                      context, encoder, rows, mode,
                                      composed_buffers, encode_error);
                                  prefix_dispatch_observed =
                                      encoder.has_encoded_dispatches();
                                  return block_encoded;
                              },
                              rollback_error);
                if (!standalone_form) {
                    check(prefix_dispatch_observed,
                          rollback_label +
                              " reaches selector failure after an encoded "
                              "dequant dispatch",
                          rollback_error);
                }
                const bool rejected_and_discarded =
                    !encoded &&
                    rollback_command->unsafe_native_handle() == nullptr &&
                    rollback_error.find(
                        "STRATA_METAL_GEMM must be 'mpp' or 'simd'") !=
                        std::string::npos &&
                    (standalone_form || prefix_dispatch_observed);
                check(rejected_and_discarded,
                      rollback_label +
                          " propagates late projection selector failure",
                      rollback_error);
                bool unchanged = true;
                for (std::size_t i = 0u; i < composed_outputs.size(); ++i) {
                    const auto payload = composed_outputs[i]->payload();
                    unchanged &=
                        std::equal(output_snapshots[i].begin(),
                                   output_snapshots[i].end(), payload.begin());
                }
                check(unchanged,
                      rollback_label +
                          " preserves all block outputs and persistent state");
                rollback &= rejected_and_discarded && unchanged;
            }
            const bool environment_restored = invalid_selector.restore();
            check(environment_restored,
                  label + " restores GEMM selector environment");
            rollback &= environment_restored;
        }
    }
    return outputs_match && inputs_unchanged && guards && rollback;
}

bool run_prefill_attention_gdn_transition_case(Context& context, std::uint32_t rows,
                                      PrefillBF16X2Mode mode,
                                      bool standalone_api) {
    const std::string label =
        "Prefill attention-half GDN transition T=" + std::to_string(rows) +
        " mode=" + std::to_string(static_cast<std::uint32_t>(mode)) +
        (standalone_api ? " standalone" : " caller-owned");
    constexpr std::size_t stream_values =
        static_cast<std::size_t>(kStreams) * kWidth;
    constexpr std::size_t down_weight_values =
        static_cast<std::size_t>(HC_LOW_RANK) * stream_values;
    constexpr std::size_t up_weight_values =
        stream_values * HC_LOW_RANK;
    constexpr std::size_t inject_weight_values =
        static_cast<std::size_t>(kStreams) * stream_values;
    const bool read_low = mode == PrefillBF16X2Mode::all_projections;

    TestBuffer norm, next_norm, down_weights, up_weights, inject_weights;
    std::string error;
    const bool gr_inputs_allocated =
        norm.allocate(context, label + " GR norm", stream_values * sizeof(float),
                      error) &&
        next_norm.allocate(context, label + " next GR norm",
                           stream_values * sizeof(float), error) &&
        down_weights.allocate(context, label + " GR down weights",
                              down_weight_values * kGdnHalfBytes, error) &&
        up_weights.allocate(context, label + " GR up weights",
                            up_weight_values * kGdnHalfBytes, error) &&
        inject_weights.allocate(context, label + " GR inject weights",
                                inject_weight_values * kGdnHalfBytes, error);
    check(gr_inputs_allocated, label + " allocate GR model inputs", error);
    if (!gr_inputs_allocated) return false;

    auto* const norm_values =
        reinterpret_cast<float*>(norm.payload().data());
    auto* const next_norm_values =
        reinterpret_cast<float*>(next_norm.payload().data());
    for (std::size_t index = 0u; index < stream_values; ++index) {
        norm_values[index] =
            0.75f + static_cast<float>((index * 17u + 11u) % 37u) *
                        0.015625f;
        next_norm_values[index] =
            0.625f + static_cast<float>((index * 23u + 3u) % 41u) *
                         0.015625f;
    }
    const auto fill_bf16 = [](TestBuffer& buffer, std::uint32_t multiplier,
                              std::uint32_t modulus, float scale) {
        auto* const values =
            reinterpret_cast<std::uint16_t*>(buffer.payload().data());
        const std::size_t count = buffer.payload().size() / sizeof(*values);
        for (std::size_t index = 0u; index < count; ++index) {
            const int centered =
                static_cast<int>((index * multiplier + index / 7u) %
                                 modulus) -
                static_cast<int>(modulus / 2u);
            values[index] = strata::kernels::bf16_from_f32(
                static_cast<float>(centered) * scale);
        }
    };
    fill_bf16(down_weights, 13u, 31u, 0.00390625f);
    fill_bf16(up_weights, 19u, 29u, 0.00390625f);
    fill_bf16(inject_weights, 7u, 23u, 0.00390625f);

    GDNBlockTestInputs gdn_inputs;
    const bool gdn_inputs_allocated =
        gdn_inputs.allocate(context, rows, label + " GDN inputs", error);
    check(gdn_inputs_allocated, label + " allocate GDN inputs", error);
    if (!gdn_inputs_allocated) return false;
    initialize_gdn_block_inputs(gdn_inputs, rows);
    const std::array<TestBuffer*, 17u> gr_model_inputs{
        &norm, &next_norm, &down_weights, &up_weights, &inject_weights,
        &gdn_inputs.qkv_weights, &gdn_inputs.gate_weights,
        &gdn_inputs.alpha_weights, &gdn_inputs.beta_weights,
        &gdn_inputs.out_weights, &gdn_inputs.dt_bias, &gdn_inputs.a,
        &gdn_inputs.conv_weights, &gdn_inputs.gamma,
        &gdn_inputs.mixed_h, &gdn_inputs.mixed_bf, &gdn_inputs.mixed_bf_lo};
    std::array<std::vector<std::uint8_t>, gr_model_inputs.size()>
        model_snapshots;
    for (std::size_t index = 0u; index < gr_model_inputs.size(); ++index) {
        const auto payload = gr_model_inputs[index]->payload();
        model_snapshots[index].assign(payload.begin(), payload.end());
    }

    GDNTransitionTestOutputs composed, direct;
    const bool outputs_allocated =
        composed.allocate(context, rows, mode, label + " composed", error) &&
        direct.allocate(context, rows, mode, label + " direct", error);
    check(outputs_allocated, label + " allocate layer outputs", error);
    if (!outputs_allocated) return false;
    std::vector<float> initial_residual(
        static_cast<std::size_t>(rows) * stream_values);
    for (std::size_t index = 0u; index < initial_residual.size(); ++index) {
        const int centered = static_cast<int>((index * 29u + 17u) % 257u) -
                             128;
        initial_residual[index] = static_cast<float>(centered) / 128.0f;
    }
    const std::size_t residual_bytes =
        initial_residual.size() * sizeof(float);
    std::memcpy(composed.residual.payload().data(), initial_residual.data(),
                residual_bytes);
    std::memcpy(direct.residual.payload().data(), initial_residual.data(),
                residual_bytes);
    composed.gdn.initialize_persistent_state();
    direct.gdn.initialize_persistent_state();

    const auto down_read = context.buffers().read_only(down_weights.view());
    const auto up_read = context.buffers().read_only(up_weights.view());
    const auto inject_read =
        context.buffers().read_only(inject_weights.view());
    const auto next_norm_read =
        context.buffers().read_only(next_norm.view());
    const PrefillAttentionGDNTransitionBuffers composed_buffers =
        bind_gdn_transition(context, gdn_inputs, composed, norm.view(),
                             down_read, up_read, inject_read, next_norm_read,
                             mode);

    auto composed_command =
        context.make_command_buffer(label + " composed command", error);
    check(composed_command != nullptr,
          label + " create composed command buffer", error);
    if (!composed_command) return false;
    const bool composed_encoded =
        standalone_api
            ? encode_prefill_attention_gdn_transition(context, *composed_command, rows,
                                             mode, composed_buffers, error)
            : Encoder::encode(
                  context, *composed_command, label + " caller-owned",
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_prefill_attention_gdn_transition(
                          context, encoder, rows, mode, composed_buffers,
                          encode_error);
                  },
                  error);
    check(composed_encoded, label + " composed encode", error);
    if (!composed_encoded) {
        composed_command->discard();
        return false;
    }
    if (!submit_and_wait(composed_command, label + " composed", error)) {
        check(false, label + " composed dispatch", error);
        return false;
    }

    const PrefillAttentionGDNTransitionBuffers direct_buffers =
        bind_gdn_transition(context, gdn_inputs, direct, norm.view(),
                             down_read, up_read, inject_read, next_norm_read,
                             mode);
    auto direct_command =
        context.make_command_buffer(label + " direct command", error);
    check(direct_command != nullptr, label + " create direct command buffer",
          error);
    if (!direct_command) return false;
    const bool direct_encoded = Encoder::encode(
        context, *direct_command, label + " direct component chain",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_prefill_gr_read_reference(
                    context, encoder, rows, mode,
                    direct_buffers.gr_read_mix.read, encode_error)) {
                return false;
            }
            const PrefillGRReadBuffers& read =
                direct_buffers.gr_read_mix.read;
            const bool mix_low = mode != PrefillBF16X2Mode::off;
            const PrefillGRMixBuffers mix{
                context.buffers().read_only(read.residual),
                context.buffers().read_only(read.rs),
                context.buffers().read_only(read.norm),
                context.buffers().read_only(read.gated),
                direct_buffers.gr_read_mix.mixed_bf,
                direct_buffers.gr_read_mix.mixed_bf_lo,
                direct_buffers.gr_read_mix.mixed_h};
            if (!encode_prefill_gr_mix(context, encoder, rows, mode, mix,
                                       encode_error)) {
                return false;
            }

            PrefillGDNBlockBuffers gdn =
                gdn_inputs.bind(context, direct.gdn, mode);
            gdn.mixed_h = context.buffers().read_only(
                direct_buffers.gr_read_mix.mixed_h);
            gdn.mixed_bf = context.buffers().read_only(
                direct_buffers.gr_read_mix.mixed_bf);
            if (mix_low) {
                gdn.mixed_bf_lo = context.buffers().read_only(
                    direct_buffers.gr_read_mix.mixed_bf_lo);
            }
            if (!encode_direct_gdn_block(context, encoder, rows, mode, gdn,
                                         encode_error)) {
                return false;
            }
            const PrefillGRWriteNormBuffers write_norm{
                read.residual,
                context.buffers().read_only(gdn.bo),
                context.buffers().read_only(read.inject),
                next_norm_read,
                read.rs,
                read.xn16,
                read_low ? read.xn16_lo : BufferRef{}};
            return encode_prefill_gr_write_norm(context, encoder, rows, mode,
                                                write_norm, encode_error);
        },
        error);
    check(direct_encoded, label + " direct component encode", error);
    if (!direct_encoded) {
        direct_command->discard();
        return false;
    }
    if (!submit_and_wait(direct_command, label + " direct", error)) {
        check(false, label + " direct dispatch", error);
        return false;
    }

    bool outputs_match = true;
    const auto compare = [&](const TestBuffer& actual,
                             const TestBuffer& expected,
                             std::string_view name) {
        const auto left = actual.payload();
        const auto right = expected.payload();
        const bool match =
            left.size() == right.size() &&
            std::equal(left.begin(), left.end(), right.begin());
        check(match, label + " matches direct " + std::string(name));
        outputs_match &= match;
    };
    compare(composed.residual, direct.residual, "written residual");
    compare(composed.gr.rs, direct.gr.rs, "next-read scale");
    compare(composed.gr.xn16, direct.gr.xn16, "next-read BF16 values");
    compare(composed.gr.low_rank, direct.gr.low_rank, "GR low rank");
    compare(composed.gr.low_rank_bf16, direct.gr.low_rank_bf16,
            "GR SiLU values");
    compare(composed.gr.gated, direct.gr.gated, "GR gated values");
    compare(composed.gr.inject, direct.gr.inject, "GR injection");
    compare(composed.mixed_bf, direct.mixed_bf, "mixed BF16 values");
    compare(composed.mixed_h, direct.mixed_h, "mixed F16 values");
    if (read_low) {
        compare(composed.gr.xn16_lo, direct.gr.xn16_lo,
                "next-read BF16 remainder");
        compare(composed.gr.low_rank_bf16_lo, direct.gr.low_rank_bf16_lo,
                "GR SiLU remainder");
    }
    if (mode != PrefillBF16X2Mode::off) {
        compare(composed.mixed_bf_lo, direct.mixed_bf_lo,
                "mixed BF16 remainder");
    }
    const std::array<TestBuffer*, 12u> composed_gdn{
        &composed.gdn.dequantized_f16, &composed.gdn.qkv, &composed.gdn.z,
        &composed.gdn.alpha_beta, &composed.gdn.history, &composed.gdn.gate,
        &composed.gdn.beta, &composed.gdn.h, &composed.gdn.state,
        &composed.gdn.oc, &composed.gdn.y_h, &composed.gdn.bo};
    const std::array<TestBuffer*, 12u> direct_gdn{
        &direct.gdn.dequantized_f16, &direct.gdn.qkv, &direct.gdn.z,
        &direct.gdn.alpha_beta, &direct.gdn.history, &direct.gdn.gate,
        &direct.gdn.beta, &direct.gdn.h, &direct.gdn.state,
        &direct.gdn.oc, &direct.gdn.y_h, &direct.gdn.bo};
    constexpr std::array<std::string_view, 12u> gdn_names{
        "reused native scratch", "qkv", "z", "alpha/beta", "history",
        "gate", "beta", "convolution", "recurrent state", "recurrence",
        "normalized output", "block output"};
    for (std::size_t index = 0u; index < composed_gdn.size(); ++index) {
        compare(*composed_gdn[index], *direct_gdn[index], gdn_names[index]);
    }

    bool model_inputs_unchanged = true;
    for (std::size_t index = 0u; index < gr_model_inputs.size(); ++index) {
        const auto payload = gr_model_inputs[index]->payload();
        model_inputs_unchanged &=
            std::equal(model_snapshots[index].begin(),
                       model_snapshots[index].end(), payload.begin());
    }
    check(model_inputs_unchanged, label + " keeps model inputs immutable");

    std::vector<TestBuffer*> composed_buffers_to_check;
    std::vector<TestBuffer*> direct_buffers_to_check;
    composed.collect_buffers(composed_buffers_to_check);
    direct.collect_buffers(direct_buffers_to_check);
    const auto buffer_snapshots = [](const std::vector<TestBuffer*>& list) {
        std::vector<std::vector<std::uint8_t>> result;
        result.reserve(list.size());
        for (const TestBuffer* buffer : list) {
            const auto payload = buffer->payload();
            result.emplace_back(payload.begin(), payload.end());
        }
        return result;
    };
    const auto composed_before_failure =
        buffer_snapshots(composed_buffers_to_check);
    const auto unchanged_after_failure =
        [&](const std::vector<TestBuffer*>& list,
            const std::vector<std::vector<std::uint8_t>>& before) {
            bool unchanged = list.size() == before.size();
            for (std::size_t index = 0u; index < list.size(); ++index) {
                const auto payload = list[index]->payload();
                unchanged &= payload.size() == before[index].size() &&
                             std::equal(before[index].begin(),
                                        before[index].end(), payload.begin());
            }
            return unchanged;
        };

    bool aliases_rejected = true;
    const auto rejects_preflight = [&](const std::string& case_label,
                                       const PrefillAttentionGDNTransitionBuffers& bad,
                                       std::string_view expected_error) {
        std::string rejection_error;
        auto command =
            context.make_command_buffer(case_label + " command", rejection_error);
        if (!command) {
            check(false, case_label + " command buffer", rejection_error);
            return false;
        }
        const bool encoded = encode_prefill_attention_gdn_transition(
            context, *command, rows, mode, bad, rejection_error);
        const bool rejected =
            !encoded && command->unsafe_native_handle() == nullptr &&
            rejection_error.find(expected_error) != std::string::npos;
        check(rejected, case_label, rejection_error);
        return rejected;
    };
    if (rows == 1u) {
        PrefillAttentionGDNTransitionBuffers next_norm_alias = composed_buffers;
        next_norm_alias.next_norm =
            context.buffers().read_only(composed.gr.gated.view());
        aliases_rejected &= rejects_preflight(
            label + " rejects next norm overwritten by GR", next_norm_alias,
            "next GR norm overlaps an earlier GR output");

        PrefillAttentionGDNTransitionBuffers gdn_weight_alias = composed_buffers;
        gdn_weight_alias.gdn.dt_bias =
            context.buffers().read_only(composed.gr.low_rank.view());
        aliases_rejected &= rejects_preflight(
            label + " rejects GDN weight overwritten by GR",
            gdn_weight_alias, "GDN dt bias overlaps an earlier GR output");

        TestBuffer carried_alias;
        const std::size_t bo_bytes =
            static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
        const bool carried_allocated = carried_alias.allocate(
            context, label + " carried injection alias", bo_bytes, error);
        check(carried_allocated, label + " allocate carried alias", error);
        if (!carried_allocated) return false;
        PrefillAttentionGDNTransitionBuffers output_alias = composed_buffers;
        output_alias.gr_read_mix.read.inject = carried_alias.view();
        output_alias.gdn.bo = carried_alias.view();
        aliases_rejected &= rejects_preflight(
            label + " rejects GDN output overwriting carried injection",
            output_alias, "GDN block output overlaps carried GR injection");

        PrefillAttentionGDNTransitionBuffers weight_overwrite =
            composed_buffers;
        weight_overwrite.gdn.bo =
            down_weights.view().slice(0u, bo_bytes);
        aliases_rejected &= rejects_preflight(
            label + " rejects GDN output overwriting GR model weights",
            weight_overwrite,
            "GDN block output overlaps GR down weights");

        const std::size_t history_bytes =
            static_cast<std::size_t>(kGdnConvKernel - 1u) *
            kGdnConvChannels * sizeof(float);
        PrefillAttentionGDNTransitionBuffers history_overwrite =
            composed_buffers;
        history_overwrite.gdn.history =
            down_weights.view().slice(0u, history_bytes);
        aliases_rejected &= rejects_preflight(
            label + " rejects persistent history overwriting GR weights",
            history_overwrite, "GDN history overlaps GR down weights");

        PrefillAttentionGDNTransitionBuffers residual_overwrite =
            composed_buffers;
        residual_overwrite.gr_read_mix.read.residual =
            down_weights.view().slice(0u, residual_bytes);
        aliases_rejected &= rejects_preflight(
            label + " rejects final residual overwriting GR model weights",
            residual_overwrite,
            "residual overlaps GR down weights");

        TestBuffer reuse_alias;
        const bool reuse_allocated = reuse_alias.allocate(
            context, label + " reused GR output alias", bo_bytes, error);
        check(reuse_allocated, label + " allocate reused output alias", error);
        if (!reuse_allocated) return false;
        PrefillAttentionGDNTransitionBuffers reused_output_alias =
            composed_buffers;
        reused_output_alias.gr_read_mix.read.rs =
            reuse_alias.view().slice(0u, HC * sizeof(float));
        reused_output_alias.gdn.bo = reuse_alias.view();
        aliases_rejected &= rejects_preflight(
            label + " rejects GDN output overwriting next GR scale",
            reused_output_alias,
            "GDN block output overlaps carried next GR scale output");

        if (mode == PrefillBF16X2Mode::off && standalone_api) {
            GDNTransitionTestOutputs oversized;
            TestBuffer arena;
            const std::size_t residual_view_bytes = residual_bytes * 2u;
            const std::size_t scale_offset = residual_bytes + 1024u;
            const std::size_t arena_bytes = scale_offset + residual_bytes;
            const bool oversized_allocated =
                oversized.allocate(context, rows, mode,
                                   label + " oversized outputs", error) &&
                arena.allocate(context, label + " reusable arena",
                               arena_bytes, error);
            check(oversized_allocated,
                  label + " allocate oversized disjoint active views", error);
            if (!oversized_allocated) return false;
            std::memcpy(arena.payload().data(), initial_residual.data(),
                        residual_bytes);
            oversized.gdn.initialize_persistent_state();
            PrefillAttentionGDNTransitionBuffers oversized_buffers =
                bind_gdn_transition(context, gdn_inputs, oversized, norm.view(),
                                    down_read, up_read, inject_read,
                                    next_norm_read, mode);
            oversized_buffers.gr_read_mix.read.residual =
                arena.view().slice(0u, residual_view_bytes);
            oversized_buffers.gr_read_mix.read.rs =
                arena.view().slice(scale_offset, residual_bytes);
            const auto byte_ranges_overlap = [](std::size_t left_offset,
                                                std::size_t left_bytes,
                                                std::size_t right_offset,
                                                std::size_t right_bytes) {
                return left_offset <= right_offset
                           ? right_offset - left_offset < left_bytes
                           : left_offset - right_offset < right_bytes;
            };
            const std::size_t residual_offset =
                oversized_buffers.gr_read_mix.read.residual.offset;
            const std::size_t scale_view_offset =
                oversized_buffers.gr_read_mix.read.rs.offset;
            const bool full_views_overlap = byte_ranges_overlap(
                residual_offset, residual_view_bytes, scale_view_offset,
                residual_bytes);
            const bool active_prefixes_disjoint = !byte_ranges_overlap(
                residual_offset, residual_bytes, scale_view_offset,
                HC * sizeof(float));
            check(full_views_overlap && active_prefixes_disjoint,
                  label + " arena views overlap only outside active prefixes");
            auto oversized_command =
                context.make_command_buffer(label + " oversized command",
                                           error);
            check(oversized_command != nullptr,
                  label + " create oversized command", error);
            if (!oversized_command) return false;
            const bool oversized_encoded =
                encode_prefill_attention_gdn_transition(
                    context, *oversized_command, rows, mode,
                    oversized_buffers, error);
            check(oversized_encoded,
                  label + " accepts disjoint active arena prefixes", error);
            bool oversized_ran = oversized_encoded;
            if (oversized_encoded) {
                oversized_ran = submit_and_wait(
                    oversized_command, label + " oversized active views",
                    error);
                check(oversized_ran,
                      label + " dispatches oversized active views", error);
            } else {
                oversized_command->discard();
            }
            if (oversized_ran) {
                const auto arena_payload = arena.payload();
                const auto expected_residual = composed.residual.payload();
                const auto expected_scale = composed.gr.rs.payload();
                const bool residual_matches =
                    std::equal(expected_residual.begin(),
                               expected_residual.begin() + residual_bytes,
                               arena_payload.begin());
                const bool scale_matches =
                    std::equal(expected_scale.begin(),
                               expected_scale.begin() + HC * sizeof(float),
                               arena_payload.begin() + scale_offset);
                check(residual_matches && scale_matches,
                      label + " oversized arena views preserve active results");
            }
            check(arena.guards_intact() && oversized.residual.guards_intact(),
                  label + " oversized arena and output guards");
            aliases_rejected &= full_views_overlap &&
                                active_prefixes_disjoint && oversized_ran;
        }
    }

    PrefillAttentionGDNTransitionBuffers short_late_weight = composed_buffers;
    short_late_weight.gdn.ssm_out_weights =
        context.buffers().read_only(
            gdn_inputs.out_weights.view().slice(0u, 1u));
    bool late_failure_rolled_back = true;
    for (const bool standalone_form : {true, false}) {
        const std::string rollback_label =
            label + (standalone_form ? " standalone late rollback"
                                     : " caller-owned late rollback");
        std::string rollback_error;
        auto rollback_command =
            context.make_command_buffer(rollback_label, rollback_error);
        if (!rollback_command) {
            check(false, rollback_label + " command buffer", rollback_error);
            late_failure_rolled_back = false;
            continue;
        }
        bool prefix_dispatch_observed = false;
        const bool encoded =
            standalone_form
                ? encode_prefill_attention_gdn_transition(
                      context, *rollback_command, rows, mode,
                      short_late_weight, rollback_error)
                : Encoder::encode(
                      context, *rollback_command, rollback_label,
                      [&](Encoder& encoder, std::string& encode_error) {
                          const bool layer_encoded =
                              encode_prefill_attention_gdn_transition(
                                  context, encoder, rows, mode,
                                  short_late_weight, encode_error);
                          prefix_dispatch_observed =
                              encoder.has_encoded_dispatches();
                          return layer_encoded;
                      },
                      rollback_error);
        const bool rejected =
            !encoded &&
            rollback_command->unsafe_native_handle() == nullptr &&
            rollback_error.find("ssm_out weights buffer is undersized") !=
                std::string::npos &&
            (standalone_form || prefix_dispatch_observed);
        check(rejected, rollback_label + " rejects after GR prefix",
              rollback_error);
        if (!standalone_form) {
            check(prefix_dispatch_observed,
                  rollback_label + " observed successful GR dispatch prefix",
                  rollback_error);
        }
        const bool unchanged = unchanged_after_failure(
            composed_buffers_to_check, composed_before_failure);
        check(unchanged, rollback_label + " preserves all outputs and state");
        late_failure_rolled_back &= rejected && unchanged;
    }

    bool open_group_rejected = true;
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch != nullptr && std::string_view(dispatch) == "concurrent") {
        std::string group_error;
        auto group_command =
            context.make_command_buffer(label + " open group", group_error);
        if (!group_command) {
            check(false, label + " open-group command buffer", group_error);
            open_group_rejected = false;
        } else {
            const bool encoded = Encoder::encode(
                context, *group_command, label + " open concurrent group",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encoder.group_begin(encode_error)) return false;
                    return encode_prefill_attention_gdn_transition(
                        context, encoder, rows, mode, composed_buffers,
                        encode_error);
                },
                group_error);
            open_group_rejected =
                !encoded &&
                group_error.find("ordered concurrent-group boundary") !=
                    std::string::npos &&
                group_command->unsafe_native_handle() == nullptr;
            check(open_group_rejected,
                  label + " rejects an open concurrent group", group_error);
        }
    }

    bool guards = true;
    for (const TestBuffer* buffer : gr_model_inputs) {
        guards &= buffer->guards_intact();
    }
    const auto check_output_guards = [&](const std::vector<TestBuffer*>& list) {
        for (const TestBuffer* buffer : list) {
            if (!buffer->allocation().empty()) guards &= buffer->guards_intact();
        }
    };
    check_output_guards(composed_buffers_to_check);
    check_output_guards(direct_buffers_to_check);
    check(guards, label + " all active input/output guard regions");

    return outputs_match && model_inputs_unchanged && aliases_rejected &&
           late_failure_rolled_back && open_group_rejected && guards;
}

bool test_prefill_attention_gdn_transition(Context& context) {
    bool passed = true;
    const std::array<PrefillBF16X2Mode, 3u> modes{
        PrefillBF16X2Mode::off,
        PrefillBF16X2Mode::all_projections,
        PrefillBF16X2Mode::except_gr};
    for (std::size_t index = 0u; index < modes.size(); ++index) {
        passed &= run_prefill_attention_gdn_transition_case(
            context, 1u, modes[index], index % 2u == 0u);
    }
    passed &= run_prefill_attention_gdn_transition_case(
        context, 3u, PrefillBF16X2Mode::except_gr, false);
    passed &= run_prefill_attention_gdn_transition_case(
        context, 65u, PrefillBF16X2Mode::except_gr, false);
    return passed;
}

bool test_prefill_native_projection_buffer_sizes() {
    const PrefillGemmMPPParams qkv{
        1u, GDN_CONV_CHANNELS, N_EMBD, GDN_CONV_CHANNELS, 0u, false};
    PrefillNativeProjectionBufferSizes sizes;
    std::string error;
    const bool sized = prefill_native_projection_buffer_sizes(
        qkv, PrefillNativeWeightFormat::Q8_0, sizes, error);
    const bool expected =
        sized && sizes.x_f16_bytes == N_EMBD * kGdnHalfBytes &&
        sizes.weights_bytes ==
            static_cast<std::size_t>(GDN_CONV_CHANNELS) * N_EMBD / 32u *
                sizeof(block_q8_0) &&
        sizes.dequantized_f16_bytes ==
            static_cast<std::size_t>(GDN_CONV_CHANNELS) * N_EMBD *
                kGdnHalfBytes &&
        sizes.y_f32_bytes == GDN_CONV_CHANNELS * sizeof(float) &&
        sizes.weights_alignment == alignof(block_q8_0);
    check(expected, "Prefill native projection reports Q8_0 active extents",
          error);
    const bool invalid_rejected = !prefill_native_projection_buffer_sizes(
        qkv, PrefillNativeWeightFormat::Invalid, sizes, error);
    check(invalid_rejected,
          "Prefill native projection rejects the invalid format sentinel",
          error);
    return expected && invalid_rejected;
}

bool test_prefill_gdn_block_validation(Context& context) {
    constexpr std::uint32_t rows = 3u;
    const std::string label = "Prefill GDN block preflight T=3";
    GDNBlockTestInputs inputs;
    GDNBlockTestOutputs outputs;
    std::string error;
    const bool allocated =
        inputs.allocate(context, rows, label + " inputs", error) &&
        outputs.allocate(context, rows, label + " outputs", error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;
    initialize_gdn_block_inputs(inputs, rows);

    const auto rejects_before_dispatch =
        [&](std::string_view case_name, PrefillBF16X2Mode mode,
            const PrefillGDNBlockBuffers& buffers,
            std::string_view expected_error) {
            std::string case_error;
            auto command = context.make_command_buffer(
                label + " " + std::string(case_name), case_error);
            if (!command) {
                check(false, label + " " + std::string(case_name) +
                                 " command buffer",
                      case_error);
                return false;
            }
            const bool rejected =
                !encode_prefill_gdn_block(context, *command, rows, mode,
                                          buffers, case_error) &&
                command->unsafe_native_handle() == nullptr &&
                case_error.find(expected_error) != std::string::npos;
            check(rejected,
                  label + " rejects " + std::string(case_name) +
                      " before dispatch",
                  case_error);
            return rejected;
        };

    PrefillGDNBlockBuffers short_bo =
        inputs.bind(context, outputs, PrefillBF16X2Mode::off);
    const std::size_t bo_bytes =
        static_cast<std::size_t>(rows) * N_EMBD * sizeof(float);
    short_bo.bo = outputs.bo.view().slice(0u, bo_bytes - sizeof(float));
    const bool short_bo_rejected = rejects_before_dispatch(
        "short final bo", PrefillBF16X2Mode::off, short_bo,
        "bo buffer is undersized");

    PrefillGDNBlockBuffers missing_low =
        inputs.bind(context, outputs,
                    PrefillBF16X2Mode::all_projections);
    missing_low.mixed_bf_lo = {};
    const bool missing_low_rejected = rejects_before_dispatch(
        "missing mode 1 remainder", PrefillBF16X2Mode::all_projections,
        missing_low, "requires mixed_bf_lo");

    PrefillGDNBlockBuffers unexpected_low =
        inputs.bind(context, outputs, PrefillBF16X2Mode::off);
    unexpected_low.mixed_bf_lo =
        context.buffers().read_only(inputs.mixed_bf_lo.view());
    const bool unexpected_low_rejected = rejects_before_dispatch(
        "unexpected mode 0 remainder", PrefillBF16X2Mode::off,
        unexpected_low, "requires an empty");

    PrefillGDNBlockBuffers late_alias =
        inputs.bind(context, outputs, PrefillBF16X2Mode::off);
    late_alias.bo = outputs.qkv.view().slice(0u, bo_bytes);
    const bool alias_rejected = rejects_before_dispatch(
        "qkv and final bo alias", PrefillBF16X2Mode::off, late_alias,
        "writable stages overlap: qkv and bo");

    PrefillGDNBlockBuffers invalid_late_format =
        inputs.bind(context, outputs, PrefillBF16X2Mode::off);
    invalid_late_format.ssm_out_format = PrefillNativeWeightFormat::Invalid;
    const bool format_rejected = rejects_before_dispatch(
        "invalid final projection format", PrefillBF16X2Mode::off,
        invalid_late_format, "format is invalid");

    return short_bo_rejected && missing_low_rejected &&
           unexpected_low_rejected && alias_rejected && format_rejected;
}

bool test_prefill_gdn_transforms_open_concurrent_group_rejected(
    Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN transforms reject open concurrent group";
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_transforms(context, encoder, 1u, {},
                                                 encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_prefill_gdn_block_open_concurrent_group_rejected(Context& context) {
    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch == nullptr || std::string_view(dispatch) != "concurrent") {
        return true;
    }
    const std::string label =
        "Prefill GDN block rejects open concurrent group";
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            return encode_prefill_gdn_block(
                context, encoder, 1u, PrefillBF16X2Mode::off, {},
                encode_error);
        },
        error);
    const bool rejected =
        !encoded && error.find("ordered concurrent-group boundary") !=
                        std::string::npos;
    check(rejected, label, error);
    return rejected;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_test strata.metallib "
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
        std::fprintf(stderr, "metal_prefill_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    ContextOptions foreign_options = context_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = test_encoder_validation(*context, *foreign_context);
    passed &= test_silu_encoder_validation(*context, *foreign_context);
    passed &= test_gr_write_norm_validation(*context, *foreign_context);
    passed &= test_prefill_gdn_gates_validation(*context, *foreign_context);
    passed &= test_prefill_gdn_conv_tiled_validation(*context,
                                                     *foreign_context);
    passed &= test_prefill_gdn_conv_history_validation(*context,
                                                        *foreign_context);
    passed &= test_prefill_gdn_l2_validation(*context, *foreign_context);
    passed &= test_prefill_gdn_l2_epsilon_sentinel(*context);
    passed &= test_prefill_gdn_rec_validation(*context, *foreign_context);
    passed &= test_prefill_gdn_rec_reduction_sentinel(*context);
    passed &= test_prefill_gdn_out_norm_validation(*context, *foreign_context);
    passed &= test_prefill_gdn_out_norm_epsilon_sentinel(*context);
    passed &=
        test_prefill_gdn_out_norm_reduction_order_sentinel(*context);
    passed &=
        test_prefill_gdn_out_norm_unsaturated_half_overflow(*context);
    passed &= test_prefill_gdn_out_norm_halfway_rne(*context);
    passed &= test_prefill_native_projection_buffer_sizes();
    passed &= test_prefill_gdn_block_validation(*context);
    passed &= test_prefill_attention_gdn_transition(*context);
    const std::array<std::uint32_t, 3> rows{1u, 3u, 8u};
    const std::array<PrefillBF16X2Mode, 3> modes{
        PrefillBF16X2Mode::off,
        PrefillBF16X2Mode::all_projections,
        PrefillBF16X2Mode::except_gr,
    };
    const std::array<twin::InputSet, 3> sets{
        twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
    for (std::size_t index = 0u; index < modes.size(); ++index) {
        passed &= run_case(*context, rows[index], modes[index], sets[index],
                           options.seed + index, false);
    }
    passed &= run_case(*context, 3u, PrefillBF16X2Mode::all_projections,
                       options.input_set, options.seed ^ 0x50464752u, true);
    for (std::size_t index = 0u; index < modes.size(); ++index) {
        passed &= run_silu_case(*context, rows[index], modes[index], false);
        passed &= test_silu_rne_boundaries(*context, modes[index]);
    }
    passed &= test_gr_mix_encoder_validation(*context, *foreign_context);
    for (std::size_t row_index = 0u; row_index < rows.size(); ++row_index) {
        for (std::size_t mode_index = 0u; mode_index < modes.size();
             ++mode_index) {
            const bool shared = row_index == 1u && mode_index == 1u;
            passed &= run_gr_mix_case(*context, rows[row_index],
                                      modes[mode_index], shared);
        }
    }
    for (const PrefillBF16X2Mode mode : modes) {
        passed &= test_gr_mix_rne_boundaries(*context, mode);
    }
    passed &= run_silu_case(*context, 3u,
                            PrefillBF16X2Mode::all_projections, true);
    passed &= test_prefill_embedding_chain(*context);
    passed &= test_prefill_gr_read_chain(*context);
    passed &= test_failure_discards_prior_prefill_dispatch(*context);
    passed &= test_silu_failure_discards_prior_dispatch(*context);
    passed &= test_gr_mix_failure_discards_prior_dispatch(*context);
    passed &= test_gr_write_norm_failure_discards_prior_dispatch(*context);
    passed &= test_prefill_gdn_gates_failure_discards_prior_dispatch(*context);
    passed &=
        test_prefill_gdn_conv_tiled_failure_discards_prior_dispatch(*context);
    passed &= test_prefill_gdn_conv_history_failure_discards_prior_dispatch(
        *context);
    passed &= test_prefill_gdn_l2_failure_discards_prior_dispatch(*context);
    passed &=
        test_prefill_gdn_rec_failure_discards_prior_dispatch(*context);
    passed &=
        test_prefill_gdn_out_norm_failure_discards_prior_dispatch(*context);
    passed &= test_prefill_gdn_transforms_open_concurrent_group_rejected(
        *context);
    passed &= test_prefill_gdn_block_open_concurrent_group_rejected(*context);
    passed &= test_open_concurrent_group_rejected(*context);
    passed &= test_silu_open_concurrent_group_rejected(*context);
    passed &= test_gr_mix_open_concurrent_group_rejected(*context);
    for (std::size_t mode_index = 0u; mode_index < modes.size();
         ++mode_index) {
        for (std::size_t row_index = 0u; row_index < rows.size();
             ++row_index) {
            const bool shared = row_index == 1u && mode_index == 1u;
            passed &= run_gr_write_norm_case(
                *context, rows[row_index], modes[mode_index], shared);
        }
    }
    passed &= test_gr_write_norm_open_concurrent_group_rejected(*context);
    passed &= test_prefill_gdn_gates_open_concurrent_group_rejected(*context);
    passed &=
        test_prefill_gdn_conv_tiled_open_concurrent_group_rejected(*context);
    passed &=
        test_prefill_gdn_conv_history_open_concurrent_group_rejected(*context);
    passed &= test_prefill_gdn_l2_open_concurrent_group_rejected(*context);
    passed &=
        test_prefill_gdn_rec_open_concurrent_group_rejected(*context);
    passed &=
        test_prefill_gdn_out_norm_open_concurrent_group_rejected(*context);
    for (const std::uint32_t gate_rows : {1u, 3u, 8u}) {
        passed &= run_prefill_gdn_gates_case(*context, gate_rows, false);
        passed &= run_prefill_gdn_gates_case(*context, gate_rows, true);
    }
    for (const std::uint32_t conv_rows : {1u, 3u, 64u, 65u, 129u}) {
        passed &= run_prefill_gdn_conv_tiled_case(*context, conv_rows, false);
        passed &= run_prefill_gdn_conv_tiled_case(*context, conv_rows, true);
    }
    for (const std::uint32_t history_rows : {1u, 2u, 3u, 8u, 65u}) {
        passed &=
            run_prefill_gdn_conv_history_case(*context, history_rows, false,
                                              false);
        passed &=
            run_prefill_gdn_conv_history_case(*context, history_rows, true,
                                              false);
    }
    for (const std::uint32_t composed_rows : {1u, 2u, 65u}) {
        passed &=
            run_prefill_gdn_conv_history_case(*context, composed_rows, true,
                                              true);
    }
    for (const std::uint32_t l2_rows : {1u, 2u, 3u, 8u, 65u}) {
        passed &= run_prefill_gdn_l2_case(*context, l2_rows, false, false);
        passed &= run_prefill_gdn_l2_case(*context, l2_rows, true, false);
    }
    for (const std::uint32_t composed_rows : {1u, 2u, 8u, 65u}) {
        passed &=
            run_prefill_gdn_l2_case(*context, composed_rows, true, true);
    }
    for (const std::uint32_t rec_rows : {1u, 2u, 3u, 8u, 65u}) {
        passed &= run_prefill_gdn_rec_case(*context, rec_rows, false, false);
    }
    for (const std::uint32_t rec_rows : {1u, 8u}) {
        passed &= run_prefill_gdn_rec_case(*context, rec_rows, true, false);
    }
    for (const std::uint32_t composed_rows : {1u, 3u, 8u}) {
        passed &=
            run_prefill_gdn_rec_case(*context, composed_rows, true, true);
    }
    for (const std::uint32_t out_norm_rows : {1u, 2u, 3u, 8u, 65u}) {
        passed &=
            run_prefill_gdn_out_norm_case(*context, out_norm_rows, false,
                                          false);
    }
    for (const std::uint32_t out_norm_rows : {1u, 8u}) {
        passed &=
            run_prefill_gdn_out_norm_case(*context, out_norm_rows, true,
                                          false);
    }
    for (const std::uint32_t composed_rows : {1u, 3u, 8u}) {
        passed &=
            run_prefill_gdn_out_norm_case(*context, composed_rows, true,
                                          true);
    }
    passed &= run_prefill_gdn_transforms_case(*context, 1u, true);
    passed &= run_prefill_gdn_transforms_case(*context, 3u, false);
    passed &= run_prefill_gdn_transforms_case(*context, 65u, true);
    for (std::size_t mode_index = 0u; mode_index < modes.size();
         ++mode_index) {
        passed &= run_prefill_gdn_block_case(
            *context, 1u, modes[mode_index], mode_index == 1u);
    }
    passed &= run_prefill_gdn_block_case(
        *context, 3u, PrefillBF16X2Mode::except_gr, false);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill Metal test(s) failed\n", g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill GR and GDN tests passed");
    return 0;
}
