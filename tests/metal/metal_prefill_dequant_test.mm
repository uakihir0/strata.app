// GPU oracle and buffer-contract tests for Prefill Q8_0 dequantization.
#include "metallib_sha256.inc"

#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/prefill_dequant_dispatch.hpp"

#include "../../src/metal/kernels/common/types.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr std::uint32_t kColumns = 257u;
constexpr std::uint32_t kK = 64u;
constexpr std::uint32_t kValuesPerBlock = 32u;

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
    std::span<const std::uint8_t> allocation() const noexcept {
        return {static_cast<const std::uint8_t*>(storage_.contents),
                storage_.length};
    }
    bool guards_intact() const {
        const auto bytes = allocation();
        const std::size_t suffix = kGuardBytes + payload_bytes_;
        return std::all_of(bytes.begin(), bytes.begin() + kGuardBytes,
                           [](std::uint8_t value) {
                               return value == kPoison;
                           }) &&
               std::all_of(bytes.begin() + suffix, bytes.end(),
                           [](std::uint8_t value) {
                               return value == kPoison;
                           });
    }

private:
    BufferRef storage_{};
    BufferRef view_{};
    std::size_t payload_bytes_ = 0u;
};

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     std::string& error) {
    return command && command->commit(error) &&
           command->wait_until_completed(error);
}

std::size_t required_weight_bytes(std::uint32_t n_out, std::uint32_t k) {
    return static_cast<std::size_t>(n_out) * (k / kValuesPerBlock) *
           sizeof(block_q8_0);
}

std::size_t required_output_bytes(std::uint32_t n_out, std::uint32_t k) {
    return static_cast<std::size_t>(n_out) * k * sizeof(std::uint16_t);
}

bool expect_rejected(Context& context, const PrefillDequantParams& params,
                     const PrefillDequantBuffers& buffers,
                     std::string_view expected_error,
                     const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = encode_prefill_dequant_q8_0(
        context, *command, params, buffers, error);
    const bool rejected =
        !encoded && command->unsafe_native_handle() == nullptr &&
        error.find(expected_error) != std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_q8_0_dequant(Context& context) {
    constexpr std::size_t blocks_per_row = kK / kValuesPerBlock;
    const std::size_t weight_bytes =
        required_weight_bytes(kColumns, kK);
    const std::size_t output_bytes =
        required_output_bytes(kColumns, kK);
    std::string error;
    TestBuffer weights, output;
    const bool allocated =
        weights.allocate(context, "Prefill Q8_0 weights", weight_bytes + 2u,
                         error) &&
        output.allocate(context, "Prefill Q8_0 K-major F16", output_bytes + 2u,
                        error);
    check(allocated, "Prefill Q8_0 dequant allocate buffers", error);
    if (!allocated) return false;

    constexpr std::array<std::uint16_t, 5> scales{
        0x0001u, 0x0400u, 0x3555u, 0x3c00u, 0x7bffu};
    std::vector<block_q8_0> source(kColumns * blocks_per_row);
    std::vector<std::uint16_t> expected(kColumns * kK);
    for (std::uint32_t column = 0u; column < kColumns; ++column) {
        for (std::uint32_t block_index = 0u;
             block_index < blocks_per_row; ++block_index) {
            block_q8_0& block =
                source[static_cast<std::size_t>(column) * blocks_per_row +
                       block_index];
            const std::uint16_t scale =
                scales[(column + block_index) % scales.size()];
            block.d = static_cast<strata_half>(scale);
            const float expanded_scale =
                strata::kernels::f32_from_f16(scale);
            for (std::uint32_t value = 0u; value < kValuesPerBlock; ++value) {
                const auto centered = static_cast<int>(
                    (column * 13u + block_index * 31u + value * 17u) %
                    255u);
                const auto quantized = static_cast<std::int8_t>(centered - 127);
                block.qs[value] = static_cast<strata_i8>(quantized);
                const std::uint32_t k =
                    block_index * kValuesPerBlock + value;
                expected[static_cast<std::size_t>(k) * kColumns + column] =
                    strata::kernels::f16_from_f32(
                        expanded_scale * static_cast<float>(quantized));
            }
        }
    }
    std::memcpy(weights.payload().data(), source.data(), weight_bytes);
    std::fill(output.payload().begin(), output.payload().end(), kPoison);
    const std::vector<std::uint8_t> weights_before(
        weights.allocation().begin(), weights.allocation().end());

    const PrefillDequantParams params{kColumns, kK};
    const PrefillDequantBuffers buffers{
        context.buffers().read_only(weights.view().slice(0u, weight_bytes)),
        output.view().slice(0u, output_bytes)};
    auto command = context.make_command_buffer("Prefill Q8_0 dequant", error);
    check(command != nullptr, "Prefill Q8_0 dequant command buffer", error);
    if (!command) return false;
    const bool encoded =
        encode_prefill_dequant_q8_0(context, *command, params, buffers, error);
    check(encoded, "Prefill Q8_0 dequant encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, error);
    check(completed, "Prefill Q8_0 dequant GPU dispatch", error);
    if (!completed) return false;

    const bool exact =
        std::memcmp(output.payload().data(), expected.data(), output_bytes) == 0;
    const bool input_unchanged =
        std::equal(weights_before.begin(), weights_before.end(),
                   weights.allocation().begin());
    const auto output_payload = output.payload();
    const bool trailing_bytes_untouched =
        output_payload[output_bytes] == kPoison &&
        output_payload[output_bytes + 1u] == kPoison;
    check(exact, "Q8_0 expansion matches exact F16 rounding and K-major layout");
    check(input_unchanged, "Q8_0 source weights remain unchanged");
    check(trailing_bytes_untouched,
          "Q8_0 dequant leaves unused output bytes untouched");
    check(weights.guards_intact() && output.guards_intact(),
          "Q8_0 dequant input and output guards");
    return exact && input_unchanged && trailing_bytes_untouched &&
           weights.guards_intact() && output.guards_intact();
}

bool test_validation(Context& context, Context& foreign_context) {
    const std::string prefix = "Prefill Q8_0 dequant validation";
    const PrefillDequantParams valid{kColumns, kK};
    const std::size_t weight_bytes =
        required_weight_bytes(kColumns, kK);
    const std::size_t output_bytes =
        required_output_bytes(kColumns, kK);
    std::string error;
    TestBuffer weights, output, alias, foreign_weights, foreign_output;
    const bool allocated =
        weights.allocate(context, prefix + " weights", weight_bytes + 2u,
                         error) &&
        output.allocate(context, prefix + " output", output_bytes + 2u,
                        error) &&
        alias.allocate(context, prefix + " aliased storage",
                       output_bytes + 2u, error) &&
        foreign_weights.allocate(foreign_context, prefix + " foreign weights",
                                 weight_bytes, error) &&
        foreign_output.allocate(foreign_context, prefix + " foreign output",
                                output_bytes, error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;

    const PrefillDequantBuffers valid_buffers{
        context.buffers().read_only(weights.view().slice(0u, weight_bytes)),
        output.view().slice(0u, output_bytes)};
    bool passed = true;
    auto invalid = valid;
    invalid.n_out = 0u;
    passed &= expect_rejected(context, invalid, valid_buffers,
                              "positive signed extents", prefix + " zero N");
    invalid = valid;
    invalid.k = 31u;
    passed &= expect_rejected(context, invalid, valid_buffers,
                              "positive signed extents", prefix + " short K");
    invalid = valid;
    invalid.k = 33u;
    passed &= expect_rejected(context, invalid, valid_buffers,
                              "positive signed extents", prefix + " K remainder");
    invalid = valid;
    invalid.n_out =
        static_cast<std::uint32_t>(
            std::numeric_limits<std::int32_t>::max()) +
        1u;
    passed &= expect_rejected(context, invalid, valid_buffers,
                              "positive signed extents", prefix + " signed N");
    invalid = valid;
    invalid.k =
        (static_cast<std::uint32_t>(
             std::numeric_limits<std::int32_t>::max()) /
         kValuesPerBlock +
         1u) *
        kValuesPerBlock;
    passed &= expect_rejected(context, invalid, valid_buffers,
                              "positive signed extents", prefix + " signed K");

    const PrefillDequantBuffers short_weights{
        context.buffers().read_only(
            weights.view().slice(0u, weight_bytes - 1u)),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, valid, short_weights,
                              "weight buffer is undersized",
                              prefix + " short weights");
    const PrefillDequantBuffers short_output{
        valid_buffers.weights, output.view().slice(0u, output_bytes - 1u)};
    passed &= expect_rejected(context, valid, short_output,
                              "output is undersized", prefix + " short output");

    const PrefillDequantBuffers misaligned_weights{
        context.buffers().read_only(weights.view().slice(1u, weight_bytes)),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, valid, misaligned_weights, "misaligned",
                              prefix + " misaligned weights");
    const PrefillDequantBuffers misaligned_output{
        valid_buffers.weights, output.view().slice(1u, output_bytes)};
    passed &= expect_rejected(context, valid, misaligned_output, "misaligned",
                              prefix + " misaligned output");

    const PrefillDequantBuffers overlapping{
        context.buffers().read_only(alias.view().slice(0u, weight_bytes)),
        alias.view().slice(0u, output_bytes)};
    passed &= expect_rejected(context, valid, overlapping,
                              "overlaps its weights", prefix + " input/output alias");
    const PrefillDequantBuffers partially_overlapping{
        context.buffers().read_only(alias.view().slice(0u, weight_bytes)),
        alias.view().slice(2u, output_bytes)};
    passed &= expect_rejected(
        context, valid, partially_overlapping, "overlaps its weights",
        prefix + " partial input/output overlap");

    const PrefillDequantBuffers foreign{
        foreign_context.buffers().read_only(foreign_weights.view()),
        valid_buffers.k_major_f16};
    passed &= expect_rejected(context, valid, foreign,
                              "not owned by this Context",
                              prefix + " foreign input");
    const PrefillDequantBuffers foreign_output_buffers{
        valid_buffers.weights, foreign_output.view()};
    passed &= expect_rejected(context, valid, foreign_output_buffers,
                              "not owned by this Context",
                              prefix + " foreign output");
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_prefill_dequant_test strata.metallib\n");
        return 2;
    }

    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = test_q8_0_dequant(*context);
    passed &= test_validation(*context, *foreign_context);
    return passed && g_failures == 0 ? 0 : 1;
}
