// Metal twins for Prefill's hyper-connection normalization (docs/mac/04 §6.3).
#include "metallib_sha256.inc"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_dispatch.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
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
    passed &= test_prefill_embedding_chain(*context);
    passed &= test_failure_discards_prior_prefill_dispatch(*context);
    passed &= test_open_concurrent_group_rejected(*context);
    if (g_failures != 0) {
        std::fprintf(stderr, "%d Prefill Metal test(s) failed\n", g_failures);
        return 1;
    }
    if (!passed) return 1;
    std::puts("all Prefill GR normalization tests passed");
    return 0;
}
