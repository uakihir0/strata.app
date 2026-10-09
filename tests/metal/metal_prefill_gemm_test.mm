// GPU twins for isolated Prefill F16/BF16 MPP and SIMD GEMM components
// (docs/mac/04 §6.2).
#include "metallib_sha256.inc"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"

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
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::size_t kGuardBytes = 4096u;
constexpr std::uint8_t kPoison = 0xa5u;
constexpr float kInitialOutput = 0.375f;
constexpr double kUnitRoundoff = 0x1.0p-24;
constexpr double kUnderflowAllowance = 0x1.0p-126;
constexpr double kRelativeTolerance = 0x1.0p-16;

int g_failures = 0;
bool g_auto_select = false;

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
    std::span<std::uint8_t> allocation() noexcept {
        return {static_cast<std::uint8_t*>(storage_.contents),
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

bool supports_apple10(const Context& context) {
    const auto& families = context.device_facts().families;
    return std::find(families.begin(), families.end(), "apple10") !=
           families.end();
}

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     std::string& error) {
    if (!command || !command->commit(error)) return false;
    return command->wait_until_completed(error);
}

std::string_view precision_name(bool f16) {
    return f16 ? "F16" : "BF16";
}

std::string_view backend_name(bool simd) {
    return g_auto_select ? "AUTO" : simd ? "SIMD" : "MPP";
}

PrefillGemmBackend expected_auto_backend(
    const Context& context, const PrefillGemmMPPParams& params) {
    const char* const configured = std::getenv("STRATA_METAL_GEMM");
    const std::string_view setting =
        configured == nullptr ? std::string_view{} : std::string_view(configured);
    return params.n >= 64u && supports_apple10(context) &&
                   (setting.empty() || setting == "mpp")
               ? PrefillGemmBackend::MPP
               : PrefillGemmBackend::SIMD;
}

std::uint16_t encode_input(float value, bool f16) {
    return f16 ? strata::kernels::f16_from_f32(value)
               : strata::kernels::bf16_from_f32(value);
}

double decode_input(std::uint16_t bits, bool f16) {
    return f16 ? strata::kernels::f32_from_f16(bits)
               : strata::kernels::f32_from_bf16(bits);
}

bool encode_case(Context& context, bool f16, bool simd, Encoder& encoder,
                 const PrefillGemmMPPParams& params,
                 const PrefillGemmMPPBuffers& buffers, std::string& error) {
    if (g_auto_select) {
        const auto expected = expected_auto_backend(context, params);
        PrefillGemmBackend selected =
            expected == PrefillGemmBackend::MPP ? PrefillGemmBackend::SIMD
                                                : PrefillGemmBackend::MPP;
        const bool encoded = encode_prefill_gemm(
            context, encoder, params, buffers,
            f16 ? PrefillGemmFormat::F16 : PrefillGemmFormat::BF16, error,
            &selected);
        if (encoded) {
            check(selected == expected,
                  "AUTO GEMM selected backend matches the active policy");
        }
        return encoded;
    }
    if (simd) {
        return f16 ? encode_prefill_gemm_f16_simd(context, encoder, params,
                                                  buffers, error)
                   : encode_prefill_gemm_bf16_simd(context, encoder, params,
                                                   buffers, error);
    }
    return f16 ? encode_prefill_gemm_f16_mpp(context, encoder, params, buffers,
                                             error)
               : encode_prefill_gemm_bf16_mpp(context, encoder, params, buffers,
                                              error);
}

bool encode_case(Context& context, bool f16, bool simd, CommandBuffer& command,
                 const PrefillGemmMPPParams& params,
                 const PrefillGemmMPPBuffers& buffers, std::string& error) {
    if (g_auto_select) {
        const auto expected = expected_auto_backend(context, params);
        PrefillGemmBackend selected =
            expected == PrefillGemmBackend::MPP ? PrefillGemmBackend::SIMD
                                                : PrefillGemmBackend::MPP;
        const bool encoded = encode_prefill_gemm(
            context, command, params, buffers,
            f16 ? PrefillGemmFormat::F16 : PrefillGemmFormat::BF16, error,
            &selected);
        if (encoded) {
            check(selected == expected,
                  "AUTO GEMM selected backend matches the active policy");
        }
        return encoded;
    }
    if (simd) {
        return f16 ? encode_prefill_gemm_f16_simd(context, command, params,
                                                  buffers, error)
                   : encode_prefill_gemm_bf16_simd(context, command, params,
                                                   buffers, error);
    }
    return f16 ? encode_prefill_gemm_f16_mpp(context, command, params, buffers,
                                             error)
               : encode_prefill_gemm_bf16_mpp(context, command, params, buffers,
                                              error);
}

bool dispatch_case(Context& context, std::uint32_t rows, std::uint32_t n,
                   std::uint32_t k, bool accumulate, bool use_wrapper,
                   bool f16, bool simd) {
    const std::string label =
        "Prefill " + std::string(precision_name(f16)) +
        " " + std::string(backend_name(simd)) + " GEMM rows=" +
        std::to_string(rows) +
        " n=" + std::to_string(n) + " k=" + std::to_string(k) +
        (accumulate ? " accumulate" : " overwrite");
    const std::uint32_t col0 = 3u;
    const std::uint32_t ldy = col0 + n + 5u;
    const std::size_t x_count = static_cast<std::size_t>(rows) * k;
    const std::size_t w_count = static_cast<std::size_t>(n) * k;
    const std::size_t y_count = static_cast<std::size_t>(rows) * ldy;
    std::string error;
    TestBuffer x, w, y;
    const bool allocated =
        x.allocate(context, label + " X", x_count * sizeof(std::uint16_t),
                   error) &&
        w.allocate(context, label + " W", w_count * sizeof(std::uint16_t),
                   error) &&
        y.allocate(context, label + " Y", y_count * sizeof(float), error);
    check(allocated, label + " allocate buffers", error);
    if (!allocated) return false;

    std::vector<std::uint16_t> x_values(x_count);
    std::vector<std::uint16_t> w_values(w_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const float row_scale =
            1.0f + static_cast<float>(row) * 0x1.0p-7f;
        for (std::uint32_t inner = 0u; inner < k; ++inner) {
            const float value =
                (inner & 1u) == 0u ? row_scale : -row_scale;
            x_values[static_cast<std::size_t>(row) * k + inner] =
                encode_input(value, f16);
        }
    }
    for (std::size_t i = 0u; i < w_count; ++i) {
        const std::uint32_t inner =
            f16 ? static_cast<std::uint32_t>(i / n)
                : static_cast<std::uint32_t>(i % k);
        const std::uint32_t column =
            f16 ? static_cast<std::uint32_t>(i % n)
                : static_cast<std::uint32_t>(i / k);
        const float value =
            column == 0u
                ? 1.0f + (inner + 1u == k ? 0x1.0p-7f : 0.0f)
                : 0.25f +
                      static_cast<float>((i * 13u) % 17u) * 0.03125f;
        w_values[i] = encode_input(value, f16);
    }
    std::memcpy(x.payload().data(), x_values.data(),
                x_values.size() * sizeof(std::uint16_t));
    std::memcpy(w.payload().data(), w_values.data(),
                w_values.size() * sizeof(std::uint16_t));
    auto* y_values = reinterpret_cast<float*>(y.payload().data());
    std::fill(y_values, y_values + y_count, kInitialOutput);

    const std::vector<std::uint8_t> x_before(x.allocation().begin(),
                                             x.allocation().end());
    const std::vector<std::uint8_t> w_before(w.allocation().begin(),
                                             w.allocation().end());
    const PrefillGemmMPPParams params{rows, n, k, ldy, col0, accumulate};
    const PrefillGemmMPPBuffers buffers{
        context.buffers().read_only(x.view()),
        context.buffers().read_only(w.view()), y.view()};
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;

    const bool encoded =
        use_wrapper
            ? encode_case(context, f16, simd, *command, params, buffers, error)
            : Encoder::encode(
                  context, *command, label,
                  [&](Encoder& encoder, std::string& encode_error) {
                      return encode_case(context, f16, simd, encoder, params,
                                         buffers, encode_error);
                  },
                  error);
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, error);
    check(completed, label + " complete", error);
    if (!completed) return false;

    bool within_u = true;
    std::vector<double> row_error_squared(rows, 0.0);
    std::vector<double> row_reference_squared(rows, 0.0);
    const double unit_bound_scale =
        2.0 * static_cast<double>(k) * kUnitRoundoff;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t column = 0u; column < n; ++column) {
            double reference = 0.0;
            double magnitude = 0.0;
            for (std::uint32_t inner = 0u; inner < k; ++inner) {
                const std::size_t weight_index =
                    f16 ? static_cast<std::size_t>(inner) * n + column
                        : static_cast<std::size_t>(column) * k + inner;
                const double left = decode_input(
                    x_values[static_cast<std::size_t>(row) * k + inner], f16);
                const double right =
                    decode_input(w_values[weight_index], f16);
                const double product = left * right;
                reference += product;
                magnitude += std::abs(product);
            }
            if (accumulate) {
                reference += kInitialOutput;
                magnitude += std::abs(static_cast<double>(kInitialOutput));
            }
            const float actual =
                y_values[static_cast<std::size_t>(row) * ldy + col0 + column];
            const double error_value =
                static_cast<double>(actual) - reference;
            const double bound = unit_bound_scale * magnitude +
                                 kUnderflowAllowance;
            within_u &= std::isfinite(actual) &&
                        std::abs(error_value) <= bound;
            row_error_squared[row] += error_value * error_value;
            row_reference_squared[row] += reference * reference;
        }
        for (std::uint32_t column = 0u; column < ldy; ++column) {
            if (column >= col0 && column < col0 + n) continue;
            within_u &= y_values[static_cast<std::size_t>(row) * ldy +
                                 column] == kInitialOutput;
        }
    }
    bool within_r = true;
    double max_relative_l2 = 0.0;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const double reference_norm = std::sqrt(row_reference_squared[row]);
        const double relative_l2 =
            std::sqrt(row_error_squared[row]) / reference_norm;
        max_relative_l2 = std::max(max_relative_l2, relative_l2);
        within_r &= relative_l2 <= kRelativeTolerance;
    }
    check(within_u, label + " U(2K), output padding");
    check(within_r, label + " R(2^-16)",
          "max row-relative L2 error " + std::to_string(max_relative_l2));
    const bool immutable =
        std::equal(x_before.begin(), x_before.end(), x.allocation().begin()) &&
        std::equal(w_before.begin(), w_before.end(), w.allocation().begin());
    check(immutable, label + " read-only inputs");
    check(x.guards_intact() && w.guards_intact() && y.guards_intact(),
          label + " guard zones");
    return within_u && within_r && immutable && x.guards_intact() &&
           w.guards_intact() && y.guards_intact();
}

bool expect_rejected(Context& context,
                     bool f16, bool simd,
                     const PrefillGemmMPPParams& params,
                     const PrefillGemmMPPBuffers& buffers,
                     std::string_view expected_error,
                     const std::string& label) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded =
        encode_case(context, f16, simd, *command, params, buffers, error);
    const bool rejected =
        !encoded && command->unsafe_native_handle() == nullptr &&
        error.find(expected_error) != std::string::npos;
    check(rejected, label, error);
    return rejected;
}

bool test_validation(Context& context, Context& foreign_context, bool f16,
                     bool simd) {
    const std::string prefix =
        "Prefill " + std::string(precision_name(f16)) + " " +
        std::string(backend_name(simd)) + " GEMM validation";
    std::string error;
    TestBuffer x, w, y, oversized;
    const bool allocated =
        x.allocate(context, prefix + " X", 65u, error) &&
        w.allocate(context, prefix + " W", 128u, error) &&
        y.allocate(context, prefix + " Y", 64u, error) &&
        oversized.allocate(context, prefix + " oversized", 160u, error);
    check(allocated, prefix + " allocate buffers", error);
    if (!allocated) return false;
    const PrefillGemmMPPBuffers valid_buffers{
        context.buffers().read_only(x.view()),
        context.buffers().read_only(w.view()), y.view()};
    const PrefillGemmMPPParams valid{1u, 2u, 32u, 5u, 1u, false};
    bool passed = true;
    auto invalid = valid;
    invalid.rows = 0u;
    passed &= expect_rejected(context, f16, simd, invalid, valid_buffers,
                              "row count", prefix + " zero rows");
    invalid = valid;
    invalid.n = 0u;
    passed &= expect_rejected(context, f16, simd, invalid, valid_buffers,
                              "n > 0", prefix + " zero columns");
    invalid = valid;
    invalid.k = 33u;
    passed &= expect_rejected(context, f16, simd, invalid, valid_buffers,
                              "K divisible", prefix + " K alignment");
    invalid = valid;
    invalid.ldy = 2u;
    passed &= expect_rejected(context, f16, simd, invalid, valid_buffers,
                              "extent or stride", prefix + " short output stride");
    invalid = valid;
    invalid.ldy =
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max()) +
        1u;
    passed &= expect_rejected(context, f16, simd, invalid, valid_buffers,
                              "extent or stride", prefix + " signed extent");

    const PrefillGemmMPPBuffers short_x{
        context.buffers().read_only(x.view().slice(0u, 2u)),
        valid_buffers.w, valid_buffers.y};
    passed &= expect_rejected(context, f16, simd, valid, short_x, "X buffer",
                              prefix + " short X");
    const PrefillGemmMPPBuffers short_w{
        valid_buffers.x,
        context.buffers().read_only(w.view().slice(0u, 2u)), valid_buffers.y};
    passed &= expect_rejected(context, f16, simd, valid, short_w, "W buffer",
                              prefix + " short W");
    const PrefillGemmMPPBuffers short_y{
        valid_buffers.x, valid_buffers.w, y.view().slice(0u, 4u)};
    passed &= expect_rejected(context, f16, simd, valid, short_y, "y buffer",
                              prefix + " short Y");
    const PrefillGemmMPPBuffers misaligned_x{
        context.buffers().read_only(x.view().slice(1u, 64u)),
        valid_buffers.w, valid_buffers.y};
    passed &= expect_rejected(context, f16, simd, valid, misaligned_x, "X buffer",
                              prefix + " misaligned X");
    const PrefillGemmMPPBuffers misaligned_y{
        valid_buffers.x, valid_buffers.w, y.view().slice(2u, 62u)};
    passed &= expect_rejected(context, f16, simd, valid, misaligned_y, "y buffer",
                              prefix + " misaligned Y");

    std::string foreign_error;
    TestBuffer foreign_x;
    const bool foreign_allocated =
        foreign_x.allocate(foreign_context, prefix + " foreign X", 64u,
                           foreign_error);
    check(foreign_allocated, prefix + " allocate foreign X", foreign_error);
    if (!foreign_allocated) return false;
    const PrefillGemmMPPBuffers foreign_inputs{
        foreign_context.buffers().read_only(foreign_x.view()),
        valid_buffers.w, valid_buffers.y};
    passed &= expect_rejected(context, f16, simd, valid, foreign_inputs, "not owned",
                              prefix + " foreign input");

    const auto alias_x = context.buffers().read_only(
        oversized.view().slice(0u, 64u));
    const PrefillGemmMPPBuffers exact_alias{
        alias_x, valid_buffers.w, oversized.view().slice(0u, 128u)};
    passed &= expect_rejected(context, f16, simd, valid, exact_alias, "overlaps",
                              prefix + " exact output/input alias");
    const PrefillGemmMPPBuffers partial_alias{
        alias_x, valid_buffers.w, oversized.view().slice(60u, 68u)};
    passed &= expect_rejected(context, f16, simd, valid, partial_alias, "overlaps",
                              prefix + " partial output/input alias");
    const auto alias_w = context.buffers().read_only(
        oversized.view().slice(0u, 128u));
    const PrefillGemmMPPBuffers partial_w_alias{
        valid_buffers.x, alias_w, oversized.view().slice(120u, 12u)};
    passed &= expect_rejected(context, f16, simd, valid, partial_w_alias, "overlaps",
                              prefix + " partial W/output alias");

    auto adjacent_command = context.make_command_buffer(
        prefix + " adjacent views", error);
    if (!adjacent_command) {
        check(false, prefix + " adjacent-view command buffer", error);
        return false;
    }
    const PrefillGemmMPPBuffers adjacent_views{
        alias_x, valid_buffers.w, oversized.view().slice(64u, 64u)};
    const bool adjacent_encoded = Encoder::encode(
        context, *adjacent_command, prefix + " adjacent views",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_case(context, f16, simd, encoder, valid, adjacent_views,
                               encode_error);
        },
        error);
    check(adjacent_encoded, prefix + " accepts adjacent nonoverlapping views",
          error);
    if (adjacent_encoded) adjacent_command->discard();
    passed &= adjacent_encoded;

    auto rollback_command = context.make_command_buffer(
        prefix + " rollback", error);
    if (!rollback_command) {
        check(false, prefix + " rollback command buffer", error);
        return false;
    }
    const auto y_before = std::vector<std::uint8_t>(
        y.allocation().begin(), y.allocation().end());
    const bool rollback_encoded = Encoder::encode(
        context, *rollback_command, prefix + " rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_case(context, f16, simd, encoder, valid, valid_buffers,
                             encode_error)) {
                return false;
            }
            auto late_failure = valid;
            late_failure.rows = 0u;
            return encode_case(context, f16, simd, encoder, late_failure,
                               valid_buffers, encode_error);
        },
        error);
    const bool rolled_back =
        !rollback_encoded &&
        rollback_command->unsafe_native_handle() == nullptr &&
        error.find("row count") != std::string::npos &&
        std::equal(y_before.begin(), y_before.end(), y.allocation().begin());
    check(rolled_back, prefix + " late failure rolls back command buffer",
          error);
    passed &= rolled_back;

    const char* const dispatch = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch != nullptr && std::string_view(dispatch) == "concurrent") {
        auto group_command = context.make_command_buffer(
            prefix + " open group", error);
        if (!group_command) {
            check(false, prefix + " open-group command buffer", error);
            return false;
        }
        const bool group_encoded = Encoder::encode(
            context, *group_command, "open concurrent " + prefix,
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                return encode_case(context, f16, simd, encoder, valid,
                                   valid_buffers,
                                   encode_error);
            },
            error);
        const bool group_rejected =
            !group_encoded &&
            error.find("ordered concurrent-group boundary") !=
                std::string::npos;
        check(group_rejected, prefix + " rejects open concurrent group",
              error);
        passed &= group_rejected;
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 ||
        (std::string_view(argv[2]) != "mpp" &&
         std::string_view(argv[2]) != "simd" &&
         std::string_view(argv[2]) != "auto" &&
         std::string_view(argv[2]) != "invalid-setting")) {
        std::fprintf(stderr,
                     "usage: metal_prefill_gemm_test strata.metallib "
                     "<mpp|simd|auto|invalid-setting>\n");
        return 2;
    }
    const std::string_view mode(argv[2]);
    const bool simd = mode == "simd";
    g_auto_select = mode == "auto" || mode == "invalid-setting";
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal Context", error);
    if (!context) return 1;
    if (!simd && !g_auto_select && !supports_apple10(*context)) {
        std::puts("skip: F16/BF16 MPP GEMM requires an Apple10 GPU");
        return 77;
    }
    if (mode == "invalid-setting") {
        auto command =
            context->make_command_buffer("invalid Prefill GEMM selector",
                                         error);
        if (!command) {
            check(false, "invalid selector command buffer", error);
            return 1;
        }
        const bool encoded = encode_prefill_gemm(
            *context, *command, PrefillGemmMPPParams{1u, 64u, 32u, 64u, 0u,
                                                      false},
            PrefillGemmMPPBuffers{}, PrefillGemmFormat::F16, error);
        const bool rejected =
            !encoded && error == "STRATA_METAL_GEMM must be 'mpp' or 'simd'" &&
            command->unsafe_native_handle() == nullptr;
        check(rejected, "invalid selector rejects and discards command buffer",
              error);
        return rejected ? 0 : 1;
    }

    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal Context", error);
    if (!foreign_context) return 1;

    bool passed = true;
    const std::array<std::uint32_t, 3u> edge_rows{1u, 63u, 65u};
    const std::array<std::uint32_t, 3u> edge_columns{1u, 63u, 65u};
    for (const bool f16 : {true, false}) {
        passed &= test_validation(*context, *foreign_context, f16, simd);
        for (const std::uint32_t rows : edge_rows) {
            for (const std::uint32_t columns : edge_columns) {
                const bool wrapper = rows == 1u && columns == 1u;
                passed &= dispatch_case(*context, rows, columns, 32u, false,
                                        wrapper, f16, simd);
                passed &= dispatch_case(*context, rows, columns, 32u, true,
                                        false, f16, simd);
            }
        }
        passed &= dispatch_case(*context, 65u, 65u, 2560u, false, false, f16,
                                simd);
        passed &= dispatch_case(*context, 65u, 65u, 2560u, true, false, f16,
                                simd);
    }
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr,
                     "%d Prefill F16/BF16 %s GEMM test(s) failed\n",
                     g_failures, backend_name(simd).data());
        return 1;
    }
    std::printf("all Prefill F16/BF16 %s GEMM tests passed\n",
                backend_name(simd).data());
    return 0;
}
