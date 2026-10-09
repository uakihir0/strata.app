// Metal twins for Prefill's hyper-connection normalization (docs/mac/04 §6.3).
#include "metallib_sha256.inc"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/prefill_dispatch.hpp"
#include "strata/metal/prefill_gr_mix_dispatch.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
#include "strata/metal/prefill_gr_silu_dispatch.hpp"
#include "strata/metal/prefill_gdn_gates_dispatch.hpp"
#include "strata/metal/prefill_gdn_conv_dispatch.hpp"
#include "strata/metal/prefill_gr_write_norm_dispatch.hpp"
#include "twin.hpp"

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
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kWidth = 2560u;
constexpr std::uint32_t kStreams = 4u;
constexpr std::uint32_t kGdnHeads = 48u;
constexpr std::uint32_t kGdnConvChannels = 10240u;
constexpr std::uint32_t kGdnConvKernel = 4u;
constexpr std::uint32_t kThreads = 256u;
constexpr float kEpsilon = 1.0e-6f;
constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint8_t kPoison = 0xa5u;

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
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
    passed &= test_failure_discards_prior_prefill_dispatch(*context);
    passed &= test_silu_failure_discards_prior_dispatch(*context);
    passed &= test_gr_mix_failure_discards_prior_dispatch(*context);
    passed &= test_gr_write_norm_failure_discards_prior_dispatch(*context);
    passed &= test_prefill_gdn_gates_failure_discards_prior_dispatch(*context);
    passed &=
        test_prefill_gdn_conv_tiled_failure_discards_prior_dispatch(*context);
    passed &= test_prefill_gdn_conv_history_failure_discards_prior_dispatch(
        *context);
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
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill Metal test(s) failed\n", g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill GR and GDN tests passed");
    return 0;
}
