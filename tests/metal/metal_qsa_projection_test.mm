// GPU twin for Prefill QSA input projections (docs/mac/04 §6.5, steps 1–2).
#include "metallib_sha256.inc"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_gr_norm_dispatch.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_qsa_projection_dispatch.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

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
#include <tuple>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kRows = 2u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint32_t kK = N_EMBD;
constexpr std::uint32_t kKvWidth = N_HEAD_KV * HEAD_DIM;
constexpr std::uint32_t kQueryWidth = N_HEAD_Q * 2u * HEAD_DIM;
constexpr std::uint32_t kIndexerQueryWidth = INDEXER_HEADS * INDEXER_HEAD_DIM;
constexpr double kProjectionTolerance = 0x1.0p-16;

static_assert(sizeof(block_q4_0) == 18u);
static_assert(sizeof(block_q8_0) == 34u);

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
            error = label + " payload view is invalid";
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
        return {static_cast<const std::uint8_t*>(view_.contents),
                payload_bytes_};
    }

    bool guards_intact() const {
        const auto bytes = std::span<const std::uint8_t>(
            static_cast<const std::uint8_t*>(storage_.contents),
            storage_.length);
        const std::size_t suffix = kGuardBytes + payload_bytes_;
        return std::all_of(bytes.begin(), bytes.begin() + kGuardBytes,
                           [](std::uint8_t byte) { return byte == kPoison; }) &&
               std::all_of(bytes.begin() + suffix, bytes.end(),
                           [](std::uint8_t byte) { return byte == kPoison; });
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

template <class T>
std::span<T> as_span(TestBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class T>
std::span<const T> as_span(const TestBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()),
            bytes.size() / sizeof(T)};
}

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     std::string& error) {
    return command && command->commit(error) &&
           command->wait_until_completed(error);
}

int signed_pattern(std::size_t index, int modulus) {
    return static_cast<int>((index * 37u + 19u) %
                            static_cast<std::size_t>(modulus)) -
           modulus / 2;
}

bool concurrent_dispatch_mode();

std::uint16_t make_f16_input(std::size_t index) {
    return strata::kernels::f16_from_f32(
        static_cast<float>(signed_pattern(index, 31)) * 0.015625f);
}

std::uint16_t make_bf16_input(std::size_t index) {
    return strata::kernels::bf16_from_f32(
        static_cast<float>(signed_pattern(index, 31)) * 0.0078125f);
}

std::uint16_t make_bf16_low(std::size_t index) {
    return strata::kernels::bf16_from_f32(
        static_cast<float>(signed_pattern(index + 11u, 31)) * 0.000030517578125f);
}

std::uint16_t make_bf16_weight(std::size_t index) {
    return strata::kernels::bf16_from_f32(
        static_cast<float>(signed_pattern(index + 7u, 31)) * 0.0009765625f);
}

void fill_q4_0(std::span<block_q4_0> weights, std::uint32_t outputs) {
    constexpr std::uint32_t blocks_per_row = kK / 32u;
    for (std::uint32_t output = 0u; output < outputs; ++output) {
        for (std::uint32_t block_index = 0u;
             block_index < blocks_per_row; ++block_index) {
            block_q4_0& block =
                weights[static_cast<std::size_t>(output) * blocks_per_row +
                        block_index];
            block.d = strata::kernels::f16_from_f32(0.015625f);
            for (std::uint32_t lane = 0u; lane < 16u; ++lane) {
                const std::size_t seed =
                    static_cast<std::size_t>(output) * kK +
                    block_index * 32u + lane;
                const std::uint8_t low = static_cast<std::uint8_t>(
                    (signed_pattern(seed, 16) + 8) & 0x0f);
                const std::uint8_t high = static_cast<std::uint8_t>(
                    (signed_pattern(seed + 13u, 16) + 8) & 0x0f);
                block.qs[lane] = static_cast<std::uint8_t>(low | (high << 4u));
            }
        }
    }
}

void fill_q8_0(std::span<block_q8_0> weights, std::uint32_t outputs,
               std::size_t salt) {
    constexpr std::uint32_t blocks_per_row = kK / 32u;
    for (std::uint32_t output = 0u; output < outputs; ++output) {
        for (std::uint32_t block_index = 0u;
             block_index < blocks_per_row; ++block_index) {
            block_q8_0& block =
                weights[static_cast<std::size_t>(output) * blocks_per_row +
                        block_index];
            block.d = strata::kernels::f16_from_f32(0.0078125f);
            for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                const std::size_t index =
                    salt + static_cast<std::size_t>(output) * kK +
                    block_index * 32u + lane;
                block.qs[lane] =
                    static_cast<std::int8_t>(signed_pattern(index, 31));
            }
        }
    }
}

double q4_0_weight_after_dequant(const block_q4_0& block,
                                 std::uint32_t lane) {
    const std::uint8_t packed = block.qs[lane & 15u];
    const std::uint32_t code =
        lane < 16u ? (packed & 0x0fu) : (packed >> 4u);
    const float scale = strata::kernels::f32_from_f16(block.d);
    const float dequantized =
        scale * (static_cast<float>(code) - 8.0f);
    return strata::kernels::f32_from_f16(
        strata::kernels::f16_from_f32(dequantized));
}

double q8_0_weight_after_dequant(const block_q8_0& block,
                                 std::uint32_t lane) {
    const float scale = strata::kernels::f32_from_f16(block.d);
    const float dequantized = scale * static_cast<float>(block.qs[lane]);
    return strata::kernels::f32_from_f16(
        strata::kernels::f16_from_f32(dequantized));
}

std::vector<double> q4_0_projection_oracle(
    std::span<const std::uint16_t> input,
    std::span<const block_q4_0> weights, std::uint32_t rows,
    std::uint32_t outputs) {
    constexpr std::uint32_t blocks_per_row = kK / 32u;
    std::vector<double> expected(static_cast<std::size_t>(rows) * outputs);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t output = 0u; output < outputs; ++output) {
            double sum = 0.0;
            for (std::uint32_t k = 0u; k < kK; ++k) {
                const block_q4_0& block =
                    weights[static_cast<std::size_t>(output) *
                                blocks_per_row +
                            k / 32u];
                const double x = strata::kernels::f32_from_f16(
                    input[static_cast<std::size_t>(row) * kK + k]);
                sum += x * q4_0_weight_after_dequant(block, k % 32u);
            }
            expected[static_cast<std::size_t>(row) * outputs + output] = sum;
        }
    }
    return expected;
}

std::vector<double> q8_0_projection_oracle(
    std::span<const std::uint16_t> input,
    std::span<const block_q8_0> weights, std::uint32_t rows,
    std::uint32_t outputs) {
    constexpr std::uint32_t blocks_per_row = kK / 32u;
    std::vector<double> expected(static_cast<std::size_t>(rows) * outputs);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t output = 0u; output < outputs; ++output) {
            double sum = 0.0;
            for (std::uint32_t k = 0u; k < kK; ++k) {
                const block_q8_0& block =
                    weights[static_cast<std::size_t>(output) *
                                blocks_per_row +
                            k / 32u];
                const double x = strata::kernels::f32_from_f16(
                    input[static_cast<std::size_t>(row) * kK + k]);
                sum += x * q8_0_weight_after_dequant(block, k % 32u);
            }
            expected[static_cast<std::size_t>(row) * outputs + output] = sum;
        }
    }
    return expected;
}

std::vector<double> bf16_projection_oracle(
    std::span<const std::uint16_t> input_hi,
    std::span<const std::uint16_t> input_lo,
    std::span<const std::uint16_t> weights, std::uint32_t rows,
    std::uint32_t outputs, bool use_low) {
    std::vector<double> expected(static_cast<std::size_t>(rows) * outputs);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t output = 0u; output < outputs; ++output) {
            double sum = 0.0;
            for (std::uint32_t k = 0u; k < kK; ++k) {
                const std::size_t input_index =
                    static_cast<std::size_t>(row) * kK + k;
                const std::size_t weight_index =
                    static_cast<std::size_t>(output) * kK + k;
                double activation =
                    strata::kernels::f32_from_bf16(input_hi[input_index]);
                if (use_low) {
                    activation +=
                        strata::kernels::f32_from_bf16(input_lo[input_index]);
                }
                sum += activation *
                       strata::kernels::f32_from_bf16(weights[weight_index]);
            }
            expected[static_cast<std::size_t>(row) * outputs + output] = sum;
        }
    }
    return expected;
}

bool compare_projection(std::span<const float> actual,
                        std::span<const double> expected,
                        std::uint32_t output_width,
                        const std::string& label) {
    const auto mismatch = twin::compare_relative_l2(
        actual, expected, kProjectionTolerance, output_width);
    if (mismatch) {
        std::fprintf(stderr, "%s mismatch at %zu (%s)\n", label.c_str(),
                     mismatch->index, mismatch->reason.c_str());
    }
    check(!mismatch, label + " R(2^-16) against independent scalar oracle");
    return !mismatch;
}

bool run_projection(Context& context, std::uint32_t rows,
                    PrefillBF16X2Mode mode,
                    const PrefillQsaProjectionBuffers& buffers,
                    bool caller_owned, PrefillGemmBackend& selected,
                    std::string& error) {
    auto command = context.make_command_buffer("Prefill QSA projections",
                                               error);
    check(command != nullptr, "QSA projection command buffer", error);
    if (!command) return false;

    const bool encoded = caller_owned
        ? Encoder::encode(
              context, *command, "caller-owned QSA input projections",
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_qsa_projection(
                      context, encoder, rows, mode, buffers, encode_error,
                      &selected);
              },
              error)
        : encode_prefill_qsa_projection(context, *command, rows, mode,
                                        buffers, error, &selected);
    check(encoded, "QSA projection encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool completed = submit_and_wait(command, error);
    check(completed, "QSA projection command completion", error);
    return completed;
}

bool test_projection_chain(Context& context) {
    std::string error;
    constexpr std::uint32_t rows = kRows;
    TestBuffer mixed_h, mixed_bf, mixed_bf_lo;
    TestBuffer attn_k_weights, attn_v_weights, attn_q_weights;
    TestBuffer indexer_k_weights, indexer_q_weights, dequantized_f16;
    TestBuffer key, value, query_full, indexer_key, indexer_query;

    const std::size_t native_k_bytes =
        static_cast<std::size_t>(kKvWidth) * (kK / 32u) *
        sizeof(block_q4_0);
    const std::size_t native_v_bytes =
        static_cast<std::size_t>(kKvWidth) * (kK / 32u) *
        sizeof(block_q8_0);
    const std::size_t native_q_bytes =
        static_cast<std::size_t>(kQueryWidth) * (kK / 32u) *
        sizeof(block_q8_0);
    const std::size_t indexer_k_weights_bytes =
        static_cast<std::size_t>(N_EMBD) * INDEXER_HEAD_DIM *
        sizeof(std::uint16_t);
    const std::size_t indexer_q_weights_bytes =
        static_cast<std::size_t>(N_EMBD) * kIndexerQueryWidth *
        sizeof(std::uint16_t);
    const std::size_t mixed_half_bytes =
        static_cast<std::size_t>(rows) * kK * sizeof(std::uint16_t);
    const std::size_t key_value_bytes =
        static_cast<std::size_t>(rows) * kKvWidth * sizeof(float);
    const std::size_t query_bytes =
        static_cast<std::size_t>(rows) * kQueryWidth * sizeof(float);
    const std::size_t indexer_k_bytes =
        static_cast<std::size_t>(rows) * INDEXER_HEAD_DIM * sizeof(float);
    const std::size_t indexer_q_bytes =
        static_cast<std::size_t>(rows) * kIndexerQueryWidth * sizeof(float);

    PrefillQsaProjectionBuffers buffers;
    buffers.attn_k_format = PrefillNativeWeightFormat::Q4_0;
    buffers.attn_v_format = PrefillNativeWeightFormat::Q8_0;
    buffers.attn_q_format = PrefillNativeWeightFormat::Q8_0;
    PrefillQsaProjectionBufferSizes sizes;
    bool allocated =
        mixed_h.allocate(context, "QSA projection mixed_h", mixed_half_bytes,
                         error) &&
        mixed_bf.allocate(context, "QSA projection mixed_bf",
                          mixed_half_bytes, error) &&
        mixed_bf_lo.allocate(context, "QSA projection mixed_bf_lo",
                             mixed_half_bytes, error) &&
        attn_k_weights.allocate(context, "QSA attn_k Q4_0 weights",
                                native_k_bytes, error) &&
        attn_v_weights.allocate(context, "QSA attn_v Q8_0 weights",
                                native_v_bytes, error) &&
        attn_q_weights.allocate(context, "QSA attn_q Q8_0 weights",
                                native_q_bytes, error) &&
        indexer_k_weights.allocate(context, "QSA indexer.k_proj BF16 weights",
                                   indexer_k_weights_bytes, error) &&
        indexer_q_weights.allocate(context, "QSA indexer.q_proj BF16 weights",
                                   indexer_q_weights_bytes, error) &&
        prefill_qsa_projection_buffer_sizes(rows, sizes, buffers, error) &&
        dequantized_f16.allocate(context, "QSA projection shared F16 scratch",
                                 sizes.dequantized_f16_bytes, error) &&
        key.allocate(context, "QSA projection Kc", key_value_bytes, error) &&
        value.allocate(context, "QSA projection Vc", key_value_bytes, error) &&
        query_full.allocate(context, "QSA projection Qf", query_bytes, error) &&
        indexer_key.allocate(context, "QSA projection idx_raw",
                             indexer_k_bytes, error) &&
        indexer_query.allocate(context, "QSA projection q_idx",
                               indexer_q_bytes, error);
    check(allocated, "QSA projection buffers allocate", error);
    if (!allocated) return false;

    buffers.mixed_h = context.buffers().read_only(mixed_h.view());
    buffers.mixed_bf = context.buffers().read_only(mixed_bf.view());
    buffers.mixed_bf_lo = context.buffers().read_only(mixed_bf_lo.view());
    buffers.attn_k_weights =
        context.buffers().read_only(attn_k_weights.view());
    buffers.attn_v_weights =
        context.buffers().read_only(attn_v_weights.view());
    buffers.attn_q_weights =
        context.buffers().read_only(attn_q_weights.view());
    buffers.indexer_k_weights =
        context.buffers().read_only(indexer_k_weights.view());
    buffers.indexer_q_weights =
        context.buffers().read_only(indexer_q_weights.view());
    buffers.dequantized_f16 = dequantized_f16.view();
    buffers.key = key.view();
    buffers.value = value.view();
    buffers.query_full = query_full.view();
    buffers.indexer_key = indexer_key.view();
    buffers.indexer_query = indexer_query.view();

    bool passed = true;
    passed &= sizes.attn_k_params.n == kKvWidth &&
              sizes.attn_v_params.n == kKvWidth &&
              sizes.attn_q_params.n == kQueryWidth &&
              sizes.indexer_k_params.n == INDEXER_HEAD_DIM &&
              sizes.indexer_q_params.n == kIndexerQueryWidth &&
              sizes.mixed_bf_lo_bytes == mixed_half_bytes &&
              sizes.dequantized_f16_bytes ==
                  sizes.attn_q.dequantized_f16_bytes &&
              sizes.indexer_k_weights_bytes == indexer_k_weights_bytes &&
              sizes.indexer_q_weights_bytes == indexer_q_weights_bytes;
    check(passed, "QSA projection preflight sizes match all five shapes");

    PrefillQsaProjectionBufferSizes maximum_sizes{};
    const bool maximum_rows_accepted = prefill_qsa_projection_buffer_sizes(
        kPrefillMaxShaderRows, maximum_sizes, buffers, error);
    check(maximum_rows_accepted,
          "QSA projection size query accepts the inclusive shader-row limit",
          error);
    passed &= maximum_rows_accepted;
    PrefillQsaProjectionBufferSizes preserved_row_sizes{};
    preserved_row_sizes.indexer_q_output_bytes = 0x5678u;
    const bool excessive_rows_rejected =
        !prefill_qsa_projection_buffer_sizes(
            std::numeric_limits<std::uint32_t>::max(), preserved_row_sizes,
            buffers, error) &&
        preserved_row_sizes.indexer_q_output_bytes == 0x5678u;
    check(excessive_rows_rejected,
          "QSA projection size query rejects out-of-range rows", error);
    passed &= excessive_rows_rejected;

    auto x_f16 = as_span<std::uint16_t>(mixed_h);
    auto x_bf16 = as_span<std::uint16_t>(mixed_bf);
    auto x_bf16_lo = as_span<std::uint16_t>(mixed_bf_lo);
    for (std::size_t index = 0u; index < x_f16.size(); ++index) {
        x_f16[index] = make_f16_input(index);
        x_bf16[index] = make_bf16_input(index);
        x_bf16_lo[index] = make_bf16_low(index);
    }
    fill_q4_0(as_span<block_q4_0>(attn_k_weights), kKvWidth);
    fill_q8_0(as_span<block_q8_0>(attn_v_weights), kKvWidth, 101u);
    fill_q8_0(as_span<block_q8_0>(attn_q_weights), kQueryWidth, 307u);
    auto indexer_k = as_span<std::uint16_t>(indexer_k_weights);
    auto indexer_q = as_span<std::uint16_t>(indexer_q_weights);
    for (std::size_t index = 0u; index < indexer_k.size(); ++index) {
        indexer_k[index] = make_bf16_weight(index);
    }
    for (std::size_t index = 0u; index < indexer_q.size(); ++index) {
        indexer_q[index] = make_bf16_weight(index + 1709u);
    }

    const auto expected_key = q4_0_projection_oracle(
        x_f16, as_span<block_q4_0>(attn_k_weights), rows, kKvWidth);
    const auto expected_value = q8_0_projection_oracle(
        x_f16, as_span<block_q8_0>(attn_v_weights), rows, kKvWidth);
    const auto expected_query = q8_0_projection_oracle(
        x_f16, as_span<block_q8_0>(attn_q_weights), rows, kQueryWidth);

    const std::vector<std::uint8_t> mixed_h_before(
        mixed_h.payload().begin(), mixed_h.payload().end());
    const std::vector<std::uint8_t> mixed_bf_before(
        mixed_bf.payload().begin(), mixed_bf.payload().end());
    const std::vector<std::uint8_t> mixed_bf_lo_before(
        mixed_bf_lo.payload().begin(), mixed_bf_lo.payload().end());
    const std::vector<std::uint8_t> attn_k_before(
        attn_k_weights.payload().begin(), attn_k_weights.payload().end());
    const std::vector<std::uint8_t> attn_v_before(
        attn_v_weights.payload().begin(), attn_v_weights.payload().end());
    const std::vector<std::uint8_t> attn_q_before(
        attn_q_weights.payload().begin(), attn_q_weights.payload().end());
    const std::vector<std::uint8_t> indexer_k_before(
        indexer_k_weights.payload().begin(), indexer_k_weights.payload().end());
    const std::vector<std::uint8_t> indexer_q_before(
        indexer_q_weights.payload().begin(), indexer_q_weights.payload().end());

    bool ran_all_modes = true;
    for (const auto [mode, caller_owned] :
         std::array<std::pair<PrefillBF16X2Mode, bool>, 3>{
             std::pair{PrefillBF16X2Mode::off, false},
             std::pair{PrefillBF16X2Mode::all_projections, true},
             std::pair{PrefillBF16X2Mode::except_gr, false}}) {
        PrefillQsaProjectionBuffers mode_buffers = buffers;
        if (mode == PrefillBF16X2Mode::off) {
            mode_buffers.mixed_bf_lo = {};
        }
        std::memset(dequantized_f16.payload().data(), kPoison,
                    dequantized_f16.payload().size());
        for (TestBuffer* output :
             {&key, &value, &query_full, &indexer_key, &indexer_query}) {
            std::memset(output->payload().data(), kPoison,
                        output->payload().size());
        }
        PrefillGemmBackend selected = PrefillGemmBackend::SIMD;
        const std::string mode_label =
            mode == PrefillBF16X2Mode::off
                ? "mode 0"
                : mode == PrefillBF16X2Mode::all_projections
                    ? "mode 1"
                    : "mode 2";
        const bool completed = run_projection(
            context, rows, mode, mode_buffers, caller_owned, selected, error);
        check(completed, "QSA projection " + mode_label + " completes", error);
        if (!completed) {
            ran_all_modes = false;
            continue;
        }

        passed &= compare_projection(as_span<float>(key), expected_key,
                                     kKvWidth,
                                     "QSA Kc " + mode_label);
        passed &= compare_projection(as_span<float>(value), expected_value,
                                     kKvWidth,
                                     "QSA Vc " + mode_label);
        passed &= compare_projection(as_span<float>(query_full),
                                     expected_query, kQueryWidth,
                                     "QSA Qf " + mode_label);
        for (const auto [weight_buffer, width, label] :
             std::array<std::tuple<const TestBuffer*, std::uint32_t,
                                   const char*>, 2>{
                 std::tuple{&indexer_k_weights, INDEXER_HEAD_DIM,
                            "indexer.k_proj"},
                 std::tuple{&indexer_q_weights, kIndexerQueryWidth,
                            "indexer.q_proj"}}) {
            const bool use_low = mode != PrefillBF16X2Mode::off;
            const auto expected = bf16_projection_oracle(
                x_bf16, x_bf16_lo,
                as_span<std::uint16_t>(*weight_buffer), rows, width,
                use_low);
            const auto actual = label == std::string_view("indexer.k_proj")
                ? as_span<float>(indexer_key)
                : as_span<float>(indexer_query);
            passed &= compare_projection(
                actual, expected, width,
                "QSA " + std::string(label) + " " + mode_label);
        }
        const auto scratch = as_span<std::uint16_t>(dequantized_f16);
        const auto q_weights = as_span<block_q8_0>(attn_q_weights);
        const std::uint32_t q_output = kQueryWidth - 3u;
        const std::uint32_t q_k = kK - 7u;
        const block_q8_0& q_block =
            q_weights[static_cast<std::size_t>(q_output) * (kK / 32u) +
                      q_k / 32u];
        const std::uint16_t expected_scratch =
            strata::kernels::f16_from_f32(static_cast<float>(
                q8_0_weight_after_dequant(q_block, q_k % 32u)));
        const std::size_t scratch_index =
            static_cast<std::size_t>(q_k) * kQueryWidth + q_output;
        passed &= scratch[scratch_index] == expected_scratch;
        check(scratch[scratch_index] == expected_scratch,
              "QSA projection reuses scratch through the final Q dequant");
        passed &= selected == PrefillGemmBackend::MPP ||
                  selected == PrefillGemmBackend::SIMD;
        check(selected == PrefillGemmBackend::MPP ||
                  selected == PrefillGemmBackend::SIMD,
              "QSA projection returns a selected GEMM backend");
    }

    passed &= ran_all_modes;
    check(ran_all_modes, "QSA projection API forms complete in all modes");

    const auto unchanged = [](const TestBuffer& buffer,
                              const std::vector<std::uint8_t>& before) {
        return std::equal(before.begin(), before.end(),
                          buffer.payload().begin());
    };
    const bool inputs_unchanged =
        unchanged(mixed_h, mixed_h_before) &&
        unchanged(mixed_bf, mixed_bf_before) &&
        unchanged(mixed_bf_lo, mixed_bf_lo_before) &&
        unchanged(attn_k_weights, attn_k_before) &&
        unchanged(attn_v_weights, attn_v_before) &&
        unchanged(attn_q_weights, attn_q_before) &&
        unchanged(indexer_k_weights, indexer_k_before) &&
        unchanged(indexer_q_weights, indexer_q_before);
    check(inputs_unchanged, "QSA projection chain preserves all input bytes");
    passed &= inputs_unchanged;
    const bool guards_intact =
        mixed_h.guards_intact() && mixed_bf.guards_intact() &&
        mixed_bf_lo.guards_intact() && attn_k_weights.guards_intact() &&
        attn_v_weights.guards_intact() && attn_q_weights.guards_intact() &&
        indexer_k_weights.guards_intact() &&
        indexer_q_weights.guards_intact() && dequantized_f16.guards_intact() &&
        key.guards_intact() && value.guards_intact() &&
        query_full.guards_intact() && indexer_key.guards_intact() &&
        indexer_query.guards_intact();
    check(guards_intact, "QSA projection chain respects buffer guards");
    passed &= guards_intact;

    // Every invalid handoff is rejected before the first child stage dispatch.
    const std::vector<std::uint8_t> scratch_before_failure(
        dequantized_f16.payload().begin(), dequantized_f16.payload().end());
    const std::vector<std::uint8_t> key_before_failure(
        key.payload().begin(), key.payload().end());
    PrefillQsaProjectionBuffers short_output = buffers;
    short_output.mixed_bf_lo = {};
    short_output.key = key.view().slice(0u, key_value_bytes - sizeof(float));
    auto short_command =
        context.make_command_buffer("QSA projection short-output rejection",
                                    error);
    check(short_command != nullptr, "short-output command buffer", error);
    if (!short_command) return false;
    PrefillGemmBackend unchanged_backend = PrefillGemmBackend::MPP;
    const bool short_encoded = encode_prefill_qsa_projection(
        context, *short_command, rows, PrefillBF16X2Mode::off, short_output,
        error, &unchanged_backend);
    const bool short_rejected =
        !short_encoded && error.find("Kc buffer is undersized") !=
                              std::string::npos &&
        short_command->unsafe_native_handle() == nullptr &&
        unchanged_backend == PrefillGemmBackend::MPP &&
        std::equal(scratch_before_failure.begin(),
                   scratch_before_failure.end(),
                   dequantized_f16.payload().begin()) &&
        std::equal(key_before_failure.begin(), key_before_failure.end(),
                   key.payload().begin());
    check(short_rejected,
          "QSA projection undersized late-stage output rejects before dispatch",
          error);
    passed &= short_rejected;
    short_command->discard();

    PrefillQsaProjectionBuffers output_alias = buffers;
    output_alias.mixed_bf_lo = {};
    output_alias.indexer_key = indexer_query.view();
    auto alias_command =
        context.make_command_buffer("QSA projection output-alias rejection",
                                    error);
    check(alias_command != nullptr, "output-alias command buffer", error);
    if (!alias_command) return false;
    const bool alias_encoded = encode_prefill_qsa_projection(
        context, *alias_command, rows, PrefillBF16X2Mode::off, output_alias,
        error);
    const bool output_alias_rejected =
        !alias_encoded &&
        error.find("writable stages overlap") != std::string::npos &&
        alias_command->unsafe_native_handle() == nullptr;
    check(output_alias_rejected,
          "QSA projection rejects partial output-to-output aliases", error);
    passed &= output_alias_rejected;
    alias_command->discard();

    PrefillQsaProjectionBuffers input_alias = buffers;
    input_alias.mixed_bf_lo = {};
    input_alias.indexer_key = mixed_bf.view();
    auto input_alias_command =
        context.make_command_buffer("QSA projection input-alias rejection",
                                    error);
    check(input_alias_command != nullptr, "input-alias command buffer", error);
    if (!input_alias_command) return false;
    const bool input_alias_encoded = encode_prefill_qsa_projection(
        context, *input_alias_command, rows, PrefillBF16X2Mode::off,
        input_alias, error);
    const bool input_alias_rejected =
        !input_alias_encoded &&
        error.find("overlaps read-only mixed_bf") != std::string::npos &&
        input_alias_command->unsafe_native_handle() == nullptr;
    check(input_alias_rejected,
          "QSA projection rejects writable/read-only backing aliases", error);
    passed &= input_alias_rejected;
    input_alias_command->discard();

    PrefillQsaProjectionBuffers missing_low = buffers;
    missing_low.mixed_bf_lo = {};
    auto missing_low_command =
        context.make_command_buffer("QSA projection missing-low rejection",
                                    error);
    check(missing_low_command != nullptr, "missing-low command buffer", error);
    if (!missing_low_command) return false;
    const bool missing_low_encoded = encode_prefill_qsa_projection(
        context, *missing_low_command, rows,
        PrefillBF16X2Mode::all_projections, missing_low, error);
    const bool missing_low_rejected =
        !missing_low_encoded &&
        error.find("requires mixed_bf_lo") != std::string::npos &&
        missing_low_command->unsafe_native_handle() == nullptr;
    check(missing_low_rejected,
          "QSA BF16X2 projection requires the low activation view", error);
    passed &= missing_low_rejected;
    missing_low_command->discard();

    PrefillQsaProjectionBuffers off_buffers = buffers;
    off_buffers.mixed_bf_lo = {};
    for (TestBuffer* output :
         {&key, &value, &query_full, &indexer_key, &indexer_query}) {
        std::memset(output->payload().data(), kPoison,
                    output->payload().size());
    }
    std::memset(dequantized_f16.payload().data(), kPoison,
                dequantized_f16.payload().size());
    const std::vector<std::uint8_t> rollback_scratch_before(
        dequantized_f16.payload().begin(), dequantized_f16.payload().end());
    const std::array<std::vector<std::uint8_t>, 5> rollback_outputs_before{
        std::vector<std::uint8_t>(key.payload().begin(), key.payload().end()),
        std::vector<std::uint8_t>(value.payload().begin(),
                                  value.payload().end()),
        std::vector<std::uint8_t>(query_full.payload().begin(),
                                  query_full.payload().end()),
        std::vector<std::uint8_t>(indexer_key.payload().begin(),
                                  indexer_key.payload().end()),
        std::vector<std::uint8_t>(indexer_query.payload().begin(),
                                  indexer_query.payload().end())};
    bool selector_rollback = false;
    {
        ScopedEnvironmentOverride invalid_selector("STRATA_METAL_GEMM",
                                                   "invalid");
        check(invalid_selector.ready(),
              "set invalid GEMM selector for rollback fixture");
        if (invalid_selector.ready()) {
            const auto run_rejected = [&](bool caller_owned) {
                auto command = context.make_command_buffer(
                    caller_owned ? "QSA projection caller rollback"
                                 : "QSA projection standalone rollback",
                    error);
                check(command != nullptr, "selector rollback command buffer",
                      error);
                if (!command) return false;
                PrefillGemmBackend selected = PrefillGemmBackend::MPP;
                const bool encoded = caller_owned
                    ? Encoder::encode(
                          context, *command,
                          "caller-owned QSA selector rollback",
                          [&](Encoder& encoder, std::string& body_error) {
                              return encode_prefill_qsa_projection(
                                  context, encoder, rows,
                                  PrefillBF16X2Mode::off, off_buffers,
                                  body_error, &selected);
                          },
                          error)
                    : encode_prefill_qsa_projection(
                          context, *command, rows, PrefillBF16X2Mode::off,
                          off_buffers, error, &selected);
                const bool outputs_unchanged =
                    std::equal(rollback_outputs_before[0].begin(),
                               rollback_outputs_before[0].end(),
                               key.payload().begin()) &&
                    std::equal(rollback_outputs_before[1].begin(),
                               rollback_outputs_before[1].end(),
                               value.payload().begin()) &&
                    std::equal(rollback_outputs_before[2].begin(),
                               rollback_outputs_before[2].end(),
                               query_full.payload().begin()) &&
                    std::equal(rollback_outputs_before[3].begin(),
                               rollback_outputs_before[3].end(),
                               indexer_key.payload().begin()) &&
                    std::equal(rollback_outputs_before[4].begin(),
                               rollback_outputs_before[4].end(),
                               indexer_query.payload().begin());
                const bool rejected =
                    !encoded &&
                    error.find("STRATA_METAL_GEMM must be 'mpp' or 'simd'") !=
                        std::string::npos &&
                    command->unsafe_native_handle() == nullptr &&
                    selected == PrefillGemmBackend::MPP &&
                    std::equal(rollback_scratch_before.begin(),
                               rollback_scratch_before.end(),
                               dequantized_f16.payload().begin()) &&
                    outputs_unchanged;
                check(rejected,
                      caller_owned
                          ? "caller-owned QSA rollback discards encoded dequant"
                          : "standalone QSA rollback discards encoded dequant",
                      error);
                command->discard();
                return rejected;
            };
            const bool standalone_rollback = run_rejected(false);
            const bool caller_rollback = run_rejected(true);
            selector_rollback = standalone_rollback && caller_rollback;
        }
    }
    passed &= selector_rollback;

    PrefillQsaProjectionBufferSizes preserved_sizes{};
    preserved_sizes.indexer_q_output_bytes = 0x1234u;
    PrefillQsaProjectionBuffers invalid_format = buffers;
    invalid_format.attn_q_format = PrefillNativeWeightFormat::Invalid;
    const bool invalid_size_rejected =
        !prefill_qsa_projection_buffer_sizes(rows, preserved_sizes,
                                             invalid_format, error) &&
        preserved_sizes.indexer_q_output_bytes == 0x1234u &&
        !error.empty();
    check(invalid_size_rejected,
          "QSA projection size query preserves output on invalid format",
          error);
    passed &= invalid_size_rejected;
    return passed;
}

bool test_concurrent_group_rejection(Context& context) {
    if (!concurrent_dispatch_mode()) return true;

    std::string error;
    TestBuffer input, low, weights_q4, weights_q8, weights_q;
    TestBuffer weights_bf_k, weights_bf_q, scratch;
    TestBuffer key, value, query, indexer_key, indexer_query;
    const std::size_t half_bytes =
        static_cast<std::size_t>(kRows) * kK * sizeof(std::uint16_t);
    PrefillQsaProjectionBuffers buffers;
    buffers.attn_k_format = PrefillNativeWeightFormat::Q4_0;
    buffers.attn_v_format = PrefillNativeWeightFormat::Q8_0;
    buffers.attn_q_format = PrefillNativeWeightFormat::Q8_0;
    PrefillQsaProjectionBufferSizes sizes;
    bool allocated =
        input.allocate(context, "group-test mixed_h", half_bytes, error) &&
        low.allocate(context, "group-test mixed_bf", half_bytes, error) &&
        weights_q4.allocate(
            context, "group-test Q4_0 weights",
            static_cast<std::size_t>(kKvWidth) * (kK / 32u) *
                sizeof(block_q4_0),
            error) &&
        weights_q8.allocate(
            context, "group-test Q8_0 weights",
            static_cast<std::size_t>(kKvWidth) * (kK / 32u) *
                sizeof(block_q8_0),
            error) &&
        weights_q.allocate(
            context, "group-test Q8_0 Q weights",
            static_cast<std::size_t>(kQueryWidth) * (kK / 32u) *
                sizeof(block_q8_0),
            error) &&
        weights_bf_k.allocate(
            context, "group-test BF16 indexer K",
            static_cast<std::size_t>(N_EMBD) * INDEXER_HEAD_DIM *
                sizeof(std::uint16_t),
            error) &&
        weights_bf_q.allocate(
            context, "group-test BF16 indexer Q",
            static_cast<std::size_t>(N_EMBD) * kIndexerQueryWidth *
                sizeof(std::uint16_t),
            error);
    allocated &= prefill_qsa_projection_buffer_sizes(kRows, sizes, buffers,
                                                     error);
    allocated &= scratch.allocate(context, "group-test scratch",
                                  sizes.dequantized_f16_bytes, error) &&
                 key.allocate(context, "group-test Kc",
                              static_cast<std::size_t>(kRows) * kKvWidth *
                                  sizeof(float),
                              error) &&
                 value.allocate(context, "group-test Vc",
                                static_cast<std::size_t>(kRows) * kKvWidth *
                                    sizeof(float),
                                error) &&
                 query.allocate(context, "group-test Qf",
                                static_cast<std::size_t>(kRows) *
                                    kQueryWidth * sizeof(float),
                                error) &&
                 indexer_key.allocate(
                     context, "group-test idx_raw",
                     static_cast<std::size_t>(kRows) * INDEXER_HEAD_DIM *
                         sizeof(float),
                     error) &&
                 indexer_query.allocate(
                     context, "group-test q_idx",
                     static_cast<std::size_t>(kRows) * kIndexerQueryWidth *
                         sizeof(float),
                     error);
    check(allocated, "open-group rejection buffers allocate", error);
    if (!allocated) return false;

    buffers.mixed_h = context.buffers().read_only(input.view());
    buffers.mixed_bf = context.buffers().read_only(low.view());
    buffers.mixed_bf_lo = context.buffers().read_only(low.view());
    buffers.attn_k_weights = context.buffers().read_only(weights_q4.view());
    buffers.attn_v_weights = context.buffers().read_only(weights_q8.view());
    buffers.attn_q_weights = context.buffers().read_only(weights_q.view());
    buffers.indexer_k_weights =
        context.buffers().read_only(weights_bf_k.view());
    buffers.indexer_q_weights =
        context.buffers().read_only(weights_bf_q.view());
    buffers.dequantized_f16 = scratch.view();
    buffers.key = key.view();
    buffers.value = value.view();
    buffers.query_full = query.view();
    buffers.indexer_key = indexer_key.view();
    buffers.indexer_query = indexer_query.view();

    auto command =
        context.make_command_buffer("QSA projection open-group rejection",
                                    error);
    check(command != nullptr, "open-group command buffer", error);
    if (!command) return false;
    bool saw_open_group = false;
    bool call_succeeded = false;
    PrefillGemmBackend selected = PrefillGemmBackend::MPP;
    const bool encoded = Encoder::encode(
        context, *command, "QSA projection group boundary",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encoder.group_begin(body_error)) return false;
            saw_open_group = encoder.concurrent_group_open();
            if (saw_open_group) {
                std::string call_error;
                call_succeeded = encode_prefill_qsa_projection(
                    context, encoder, kRows,
                    PrefillBF16X2Mode::all_projections, buffers, call_error,
                    &selected);
                body_error = call_error;
            } else {
                body_error = "concurrent group did not open";
            }
            std::string close_error;
            const bool closed = encoder.group_end(close_error);
            if (!closed) body_error = close_error;
            return false;
        },
        error);
    const bool rejected =
        saw_open_group && !encoded && !call_succeeded &&
        error.find("requires an ordered concurrent-group boundary") !=
            std::string::npos &&
        command->unsafe_native_handle() == nullptr &&
        selected == PrefillGemmBackend::MPP;
    check(rejected, "QSA projection rejects open concurrent groups", error);
    command->discard();
    return rejected;
}

bool concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_qsa_projection_test strata.metallib\n");
        return 2;
    }
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;

    bool passed = test_projection_chain(*context);
    passed &= test_concurrent_group_rejection(*context);
    check(passed && g_failures == 0,
          "Prefill QSA input projections match their scalar oracles");
    std::printf("metal_qsa_projection_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
