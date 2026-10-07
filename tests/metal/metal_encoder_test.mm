// tests/metal/metal_encoder_test.mm - common encoder validation and dispatch modes.
#include "strata/metal/encoder.hpp"

#include "metallib_sha256.inc"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::size_t kElementCount = 1024;
constexpr std::size_t kFloatBytes = kElementCount * sizeof(float);

struct SelftestMathParams {
    std::uint32_t count;
    std::uint32_t operation;
    float lower;
    float upper;
};
static_assert(sizeof(SelftestMathParams) == 16);

int g_failures = 0;

void check(bool ok, const char* label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", label,
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !ok;
}

class EnvironmentRestore {
public:
    explicit EnvironmentRestore(const char* name) : name_(name) {
        const char* value = std::getenv(name_.c_str());
        had_value_ = value != nullptr;
        if (had_value_) value_ = value;
    }

    ~EnvironmentRestore() {
        if (had_value_) {
            setenv(name_.c_str(), value_.c_str(), 1);
        } else {
            unsetenv(name_.c_str());
        }
    }

    bool set(const char* value) const {
        return value != nullptr ? setenv(name_.c_str(), value, 1) == 0
                                : unsetenv(name_.c_str()) == 0;
    }

private:
    std::string name_;
    std::string value_;
    bool had_value_ = false;
};

BufferRef allocate(Context& context, const char* label, std::size_t bytes,
                   std::string& error) {
    return context.buffers().allocate(label, bytes, error);
}

EncoderBufferBinding float_binding(BufferRef buffer, std::uint32_t index,
                                   std::size_t bytes) {
    return {buffer, index, bytes, alignof(float)};
}

bool dispatch_add(Encoder& encoder, BufferRef buffer, std::string& error) {
    const EncoderBufferBinding binding = float_binding(buffer, 0, kFloatBytes);
    return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                            std::span<const EncoderBufferBinding>(&binding, 1), {}, {}, error);
}

bool run_add_sequence(Context& context, const char* policy,
                      std::array<std::vector<float>, 2>& outputs) {
    EnvironmentRestore environment("STRATA_METAL_DISPATCH");
    if (!environment.set(policy)) {
        check(false, "set dispatch policy");
        return false;
    }

    std::string error;
    BufferRef first = allocate(context, "encoder sequence first", kFloatBytes, error);
    BufferRef second = allocate(context, "encoder sequence second", kFloatBytes, error);
    check(first && second, "allocate dispatch-mode buffers", error);
    if (!first || !second) return false;

    auto* const first_values = static_cast<float*>(first.contents);
    auto* const second_values = static_cast<float*>(second.contents);
    for (std::size_t index = 0; index < kElementCount; ++index) {
        first_values[index] = static_cast<float>(index % 97u + 1u);
        second_values[index] = static_cast<float>(index % 31u + 1u);
    }

    auto command = context.make_command_buffer("encoder dispatch modes", error);
    check(command != nullptr, "create dispatch-mode command buffer", error);
    if (!command) return false;

    const bool encoded = Encoder::encode(
        context, *command, "encoder dispatch mode segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return dispatch_add(encoder, first, encode_error) &&
                   encoder.group_begin(encode_error) &&
                   dispatch_add(encoder, first, encode_error) &&
                   dispatch_add(encoder, second, encode_error) &&
                   encoder.group_end(encode_error) &&
                   dispatch_add(encoder, second, encode_error);
        },
        error);
    check(encoded, "encode dependent and independent dispatches", error);
    if (!encoded || !command->commit(error)) {
        check(false, "commit dispatch-mode command buffer", error);
        return false;
    }
    check(command->wait_until_completed(error), "wait for dispatch-mode command buffer", error);
    if (!error.empty()) return false;

    outputs[0].assign(first_values, first_values + kElementCount);
    outputs[1].assign(second_values, second_values + kElementCount);
    bool expected_values = true;
    for (std::size_t index = 0; index < kElementCount; ++index) {
        const float first_initial = static_cast<float>(index % 97u + 1u);
        const float second_initial = static_cast<float>(index % 31u + 1u);
        expected_values &= outputs[0][index] == first_initial * 4.0f + 3.0f;
        expected_values &= outputs[1][index] == second_initial * 4.0f + 3.0f;
    }
    check(expected_values, "dispatch barriers preserve dependent results");
    return expected_values;
}

bool run_inline_and_threadgroup_bindings(Context& context) {
    std::string error;
    constexpr std::size_t input_count = 1024;
    constexpr std::size_t output_count = input_count / 8;
    BufferRef input = allocate(context, "encoder partials input",
                               input_count * sizeof(float), error);
    BufferRef output = allocate(context, "encoder partials output",
                                output_count * sizeof(float), error);
    check(input && output, "allocate inline and threadgroup test buffers", error);
    if (!input || !output) return false;
    std::fill_n(static_cast<float*>(input.contents), input_count, 1.0f);
    std::fill_n(static_cast<float*>(output.contents), output_count, -1.0f);

    SelftestMathParams params{static_cast<std::uint32_t>(input_count), 8u, 0.0f, 0.0f};
    const std::array bindings{
        float_binding(input, 0, input_count * sizeof(float)),
        float_binding(output, 1, output_count * sizeof(float)),
    };
    const EncoderBytesBinding bytes{&params, sizeof(params), 2};
    const EncoderThreadgroupMemory memory{0, 8u * sizeof(float)};
    auto command = context.make_command_buffer("encoder inline bytes and threadgroup memory",
                                              error);
    check(command != nullptr, "create inline-binding command buffer", error);
    if (!command) return false;
    const bool encoded = Encoder::encode(
        context, *command, "encoder dynamic threadgroup segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encoder.dispatch(
                       "strata_selftest_partials", {output_count, 1, 1}, {8, 1, 1},
                       bindings, std::span<const EncoderBytesBinding>(&bytes, 1),
                       std::span<const EncoderThreadgroupMemory>(&memory, 1), encode_error) &&
                   dispatch_add(encoder, input, encode_error);
        },
        error);
    check(encoded, "encode changing buffer and threadgroup bindings", error);
    if (!encoded || !command->commit(error)) {
        check(false, "commit inline-binding command buffer", error);
        return false;
    }
    check(command->wait_until_completed(error), "wait for inline-binding command buffer", error);
    if (!error.empty()) return false;

    const auto* const values = static_cast<const float*>(output.contents);
    bool all_sums_match = true;
    for (std::size_t index = 0; index < output_count; ++index) {
        all_sums_match &= values[index] == 8.0f;
    }
    check(all_sums_match, "inline parameters and dynamic threadgroup memory reach the kernel");
    return all_sums_match;
}

using FailureBody = Encoder::Body;

bool expect_encode_failure(Context& context, const char* label,
                           const FailureBody& body, const char* policy = "serial",
                           const char* expected_error = nullptr) {
    EnvironmentRestore environment("STRATA_METAL_DISPATCH");
    if (!environment.set(policy)) {
        check(false, "set negative-test dispatch policy");
        return false;
    }
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label, error);
        return false;
    }
    const bool encoded = Encoder::encode(context, *command, "encoder negative segment",
                                         body, error);
    const bool error_matches =
        expected_error == nullptr || error.find(expected_error) != std::string::npos;
    const bool rejected_and_discarded =
        !encoded && command->unsafe_native_handle() == nullptr && !error.empty() &&
        error_matches;
    check(rejected_and_discarded, label, error);
    return rejected_and_discarded;
}

bool test_inline_byte_segment_limit(Context& context, BufferRef buffer) {
    EnvironmentRestore environment("STRATA_METAL_DISPATCH");
    if (!environment.set("serial")) {
        check(false, "set inline-byte boundary dispatch policy");
        return false;
    }

    std::array<std::uint8_t, 2048> first_payload{};
    std::array<std::uint8_t, 2048> second_payload{};
    const SelftestMathParams params{
        static_cast<std::uint32_t>(kElementCount), 0u, 0.0f, 0.0f};
    std::memcpy(first_payload.data(), &params, sizeof(params));
    std::memcpy(second_payload.data(), &params, sizeof(params));
    const std::array bindings{
        float_binding(buffer, 0, kFloatBytes),
        float_binding(buffer, 1, kFloatBytes),
    };
    std::string error;
    auto command =
        context.make_command_buffer("encoder exact inline-byte segment limit", error);
    check(command != nullptr, "create exact inline-byte boundary command buffer",
          error);
    if (!command) return false;

    const bool encoded = Encoder::encode(
        context, *command, "encoder exact inline-byte segment",
        [&](Encoder& encoder, std::string& encode_error) {
            const EncoderBytesBinding first{
                first_payload.data(), first_payload.size(), 2};
            const EncoderBytesBinding second{
                second_payload.data(), second_payload.size(), 2};
            return encoder.dispatch(
                       "strata_selftest_math", {4, 1, 1}, {256, 1, 1},
                       bindings,
                       std::span<const EncoderBytesBinding>(&first, 1), {},
                       encode_error) &&
                   encoder.dispatch(
                       "strata_selftest_math", {4, 1, 1}, {256, 1, 1},
                       bindings,
                       std::span<const EncoderBytesBinding>(&second, 1), {},
                       encode_error);
        },
        error);
    check(encoded, "accept exactly 4 KiB of inline bytes across a segment", error);
    if (!encoded) return false;
    command->discard();
    const bool discarded = command->unsafe_native_handle() == nullptr;
    check(discarded, "discard exact inline-byte boundary command buffer");
    if (!discarded) return false;

    std::array<std::uint8_t, 2049> oversized_segment_payload{};
    std::memcpy(oversized_segment_payload.data(), &params, sizeof(params));
    return expect_encode_failure(
        context, "reject cumulative inline bytes over 4 KiB across dispatches",
        [&](Encoder& encoder, std::string& encode_error) {
            const EncoderBytesBinding first{
                first_payload.data(), first_payload.size(), 2};
            const EncoderBytesBinding second{
                oversized_segment_payload.data(),
                oversized_segment_payload.size(), 2};
            return encoder.dispatch(
                       "strata_selftest_math", {4, 1, 1}, {256, 1, 1},
                       bindings,
                       std::span<const EncoderBytesBinding>(&first, 1), {},
                       encode_error) &&
                   encoder.dispatch(
                       "strata_selftest_math", {4, 1, 1}, {256, 1, 1},
                       bindings,
                       std::span<const EncoderBytesBinding>(&second, 1), {},
                       encode_error);
        },
        "serial", "Metal segment inline-byte bindings exceed the 4 KiB encoder limit");
}

bool test_failure_lifecycle(Context& context, BufferRef valid_buffer) {
    EnvironmentRestore environment("STRATA_METAL_DISPATCH");
    if (!environment.set("serial")) {
        check(false, "set failure-lifecycle dispatch policy");
        return false;
    }
    std::string error;
    auto failed_command =
        context.make_command_buffer("encoder exception after dispatch", error);
    check(failed_command != nullptr, "create exception-test command buffer", error);
    if (!failed_command) return false;

    const bool encoded = Encoder::encode(
        context, *failed_command, "encoder exception lifecycle",
        [valid_buffer](Encoder& encoder, std::string& encode_error) -> bool {
            if (!dispatch_add(encoder, valid_buffer, encode_error)) return false;
            throw std::runtime_error("deliberate exception after dispatch");
        },
        error);
    const bool discarded =
        !encoded && failed_command->unsafe_native_handle() == nullptr &&
        error.find("deliberate exception after dispatch") != std::string::npos;
    check(discarded, "exception after dispatch discards partial encoding", error);
    if (!discarded) return false;

    auto recovery_command =
        context.make_command_buffer("encoder post-exception recovery", error);
    check(recovery_command != nullptr, "create post-exception command buffer", error);
    if (!recovery_command) return false;
    error = "stale diagnostic from an earlier operation";
    const bool recovery_encoded = Encoder::encode(
        context, *recovery_command, "encoder post-exception recovery",
        [valid_buffer](Encoder& encoder, std::string& encode_error) {
            return dispatch_add(encoder, valid_buffer, encode_error);
        },
        error);
    check(recovery_encoded && error.empty(),
          "successful encode clears a reused error string", error);
    if (!recovery_encoded || !error.empty() || !recovery_command->commit(error) ||
        !recovery_command->wait_until_completed(error)) {
        check(false, "post-exception command buffer completes", error);
        return false;
    }
    return true;
}

bool test_validation(Context& context, Context& foreign_context,
                     BufferRef valid_buffer, BufferRef short_buffer,
                     BufferRef misaligned_buffer, BufferRef foreign_buffer) {
    bool passed = true;
    const auto one_dispatch = [valid_buffer](Encoder& encoder, std::string& error) {
        return dispatch_add(encoder, valid_buffer, error);
    };
    const EncoderBufferBinding valid = float_binding(valid_buffer, 0, kFloatBytes);
    const EncoderBufferBinding short_binding = float_binding(short_buffer, 0, kFloatBytes);
    const EncoderBufferBinding misaligned =
        float_binding(misaligned_buffer, 0, kFloatBytes);
    const EncoderBufferBinding foreign = float_binding(foreign_buffer, 0, kFloatBytes);

    passed &= expect_encode_failure(context, "reject unknown manifest kernel",
        [valid](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_not_a_kernel", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&valid, 1), {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject manifest threadgroup mismatch",
        [valid](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {128, 2, 1},
                                    std::span<const EncoderBufferBinding>(&valid, 1), {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject zero dispatch grid",
        [valid](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_selftest_add", {0, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&valid, 1), {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject short buffer view",
        [short_binding](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&short_binding, 1),
                                    {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject misaligned buffer view",
        [misaligned](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&misaligned, 1),
                                    {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject foreign-context buffer",
        [foreign](Encoder& encoder, std::string& error) {
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&foreign, 1),
                                    {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject duplicate argument indices",
        [valid](Encoder& encoder, std::string& error) {
            const std::array duplicate{valid, valid};
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    duplicate, {}, {}, error);
        });
    passed &= expect_encode_failure(
        context, "reject omitted buffer previously bound by prior dispatch",
        [valid_buffer](Encoder& encoder, std::string& error) {
            if (!dispatch_add(encoder, valid_buffer, error)) return false;
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    {}, {}, {}, error);
        });
    passed &= expect_encode_failure(context, "reject null inline bytes",
        [valid](Encoder& encoder, std::string& error) {
            const EncoderBytesBinding invalid_bytes{nullptr, sizeof(std::uint32_t), 1};
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&valid, 1),
                                    std::span<const EncoderBytesBinding>(&invalid_bytes, 1),
                                    {}, error);
        });
    passed &= expect_encode_failure(
        context, "reject inline-byte payload beyond the Metal encoder limit",
        [valid_buffer](Encoder& encoder, std::string& error) {
            std::array<std::uint8_t, 4097> oversized{};
            const std::array bindings{
                float_binding(valid_buffer, 0, kFloatBytes),
                float_binding(valid_buffer, 1, kFloatBytes),
            };
            const EncoderBytesBinding bytes{oversized.data(), oversized.size(), 2};
            return encoder.dispatch(
                "strata_selftest_math", {4, 1, 1}, {256, 1, 1}, bindings,
                std::span<const EncoderBytesBinding>(&bytes, 1), {}, error);
        });
    passed &= test_inline_byte_segment_limit(context, valid_buffer);
    passed &= expect_encode_failure(context, "reject duplicate threadgroup-memory indices",
        [valid](Encoder& encoder, std::string& error) {
            const std::array duplicate_memory{
                EncoderThreadgroupMemory{0, 16}, EncoderThreadgroupMemory{0, 16}};
            return encoder.dispatch("strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                                    std::span<const EncoderBufferBinding>(&valid, 1), {},
                                    duplicate_memory, error);
        });

    const std::uint64_t max_threadgroup_memory =
        context.device_facts().max_threadgroup_memory_length;
    if (max_threadgroup_memory <=
        std::numeric_limits<std::size_t>::max() - 16u) {
        const std::size_t excessive_bytes =
            static_cast<std::size_t>(max_threadgroup_memory) + 16u;
    passed &= expect_encode_failure(context, "reject excessive threadgroup memory",
            [valid_buffer, excessive_bytes](Encoder& encoder, std::string& error) {
                SelftestMathParams params{
                    static_cast<std::uint32_t>(kElementCount), 8u, 0.0f, 0.0f};
                const std::array bindings{
                    float_binding(valid_buffer, 0, kFloatBytes),
                    float_binding(valid_buffer, 1, kFloatBytes),
                };
                const EncoderBytesBinding bytes{&params, sizeof(params), 2};
                const EncoderThreadgroupMemory excessive{0, excessive_bytes};
                return encoder.dispatch(
                    "strata_selftest_partials", {128, 1, 1}, {8, 1, 1},
                    bindings, std::span<const EncoderBytesBinding>(&bytes, 1),
                    std::span<const EncoderThreadgroupMemory>(&excessive, 1), error);
            });
    } else {
        check(false, "device threadgroup-memory limit permits overflow rejection");
        passed = false;
    }

    passed &= expect_encode_failure(context, "reject unmatched concurrent group end",
        [](Encoder& encoder, std::string& error) {
            return encoder.group_end(error);
        }, "concurrent");
    passed &= expect_encode_failure(context, "reject nested concurrent groups",
        [](Encoder& encoder, std::string& error) {
            return encoder.group_begin(error) && encoder.group_begin(error);
        }, "concurrent");
    passed &= expect_encode_failure(context, "reject unclosed concurrent group",
        [](Encoder& encoder, std::string& error) {
            return encoder.group_begin(error);
        }, "concurrent");
    passed &= expect_encode_failure(context, "invalid policy discards command buffer",
        [](Encoder&, std::string&) { return true; }, "parallel-ish");
    passed &= expect_encode_failure(context, "failed body discards partial encoding",
        [valid_buffer](Encoder& encoder, std::string& error) {
            if (!dispatch_add(encoder, valid_buffer, error)) return false;
            error = "deliberate body failure after one dispatch";
            return false;
        });
    passed &= test_failure_lifecycle(context, valid_buffer);
    {
        EnvironmentRestore stale_error_environment("STRATA_METAL_DISPATCH");
        stale_error_environment.set("serial");
        std::string stale_error = "stale diagnostic from an earlier call";
        auto stale_error_command =
            context.make_command_buffer("encoder stale-error test", stale_error);
        check(stale_error_command != nullptr, "create stale-error test command buffer",
              stale_error);
        if (stale_error_command) {
            const bool stale_error_encoded = Encoder::encode(
                context, *stale_error_command, "encoder stale-error test",
                [](Encoder&, std::string&) { return false; }, stale_error);
            const bool diagnostic_replaced =
                !stale_error_encoded &&
                stale_error_command->unsafe_native_handle() == nullptr &&
                stale_error == "Metal encoder body failed";
            check(diagnostic_replaced, "body failure replaces a reused error string",
                  stale_error);
            passed &= diagnostic_replaced;
        } else {
            passed = false;
        }
    }

    EnvironmentRestore environment("STRATA_METAL_DISPATCH");
    environment.set("serial");
    std::string error;
    auto foreign_command = foreign_context.make_command_buffer("foreign encoder command", error);
    check(foreign_command != nullptr, "create foreign-context command buffer", error);
    if (foreign_command) {
        const bool encoded = Encoder::encode(
            context, *foreign_command, "foreign context", one_dispatch, error);
        check(!encoded && foreign_command->unsafe_native_handle() != nullptr,
              "foreign command buffer is rejected without discarding its owner",
              error);
        passed &= !encoded && foreign_command->unsafe_native_handle() != nullptr;
    } else {
        passed = false;
    }
    return passed;
}

bool test_pipeline_absence(const std::string& metallib) {
    ContextOptions options;
    options.metallib = metallib;
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    options.pipelines = false;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open device-only Context for missing-pipeline check", error);
    if (!context) return false;
    BufferRef buffer = allocate(*context, "encoder missing-pipeline buffer", kFloatBytes, error);
    if (!buffer) {
        check(false, "allocate missing-pipeline buffer", error);
        return false;
    }
    const EncoderBufferBinding binding = float_binding(buffer, 0, kFloatBytes);
    return expect_encode_failure(*context, "reject unavailable pipeline",
        [binding](Encoder& encoder, std::string& encode_error) {
            return encoder.dispatch(
                "strata_selftest_add", {4, 1, 1}, {256, 1, 1},
                std::span<const EncoderBufferBinding>(&binding, 1), {}, {}, encode_error);
        });
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc != 2) {
        std::fprintf(stderr, "usage: metal_encoder_test strata.metallib\n");
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
    check(foreign_context != nullptr, "open second Context for ownership checks", error);
    if (!foreign_context) return 1;

    std::array<std::vector<float>, 2> serial_outputs;
    std::array<std::vector<float>, 2> concurrent_outputs;
    std::array<std::vector<float>, 2> default_outputs;
    bool passed = run_add_sequence(*context, "serial", serial_outputs);
    passed &= run_add_sequence(*context, "concurrent", concurrent_outputs);
    {
        EnvironmentRestore environment("STRATA_METAL_DISPATCH");
        if (!environment.set(nullptr)) {
            check(false, "clear dispatch policy to test default");
            passed = false;
        } else {
            passed &= run_add_sequence(*context, nullptr, default_outputs);
        }
    }
    check(serial_outputs == concurrent_outputs,
          "serial and concurrent dispatch outputs are bitwise identical");
    check(serial_outputs == default_outputs,
          "unset dispatch policy defaults to serial behavior");
    passed &= serial_outputs == concurrent_outputs;
    passed &= serial_outputs == default_outputs;
    passed &= run_inline_and_threadgroup_bindings(*context);

    BufferRef valid = allocate(*context, "encoder validation buffer", kFloatBytes, error);
    BufferRef short_buffer = allocate(*context, "encoder short buffer", 64, error);
    BufferRef misaligned_backing =
        allocate(*context, "encoder misaligned backing", kFloatBytes + 1, error);
    BufferRef foreign = allocate(*foreign_context, "encoder foreign buffer", kFloatBytes, error);
    check(valid && short_buffer && misaligned_backing && foreign,
          "allocate encoder-validation buffers", error);
    if (!valid || !short_buffer || !misaligned_backing || !foreign) return 1;
    BufferRef misaligned = misaligned_backing.slice(1, kFloatBytes);
    check(static_cast<bool>(misaligned), "create valid but misaligned BufferRef slice");
    passed &= static_cast<bool>(misaligned);
    passed &= test_validation(*context, *foreign_context, valid, short_buffer,
                              misaligned, foreign);
    passed &= test_pipeline_absence(argv[1]);

    std::printf("metal_encoder_test: %s (%d failures)\n",
                passed && g_failures == 0 ? "OK" : "FAILED", g_failures);
    return passed && g_failures == 0 ? 0 : 1;
}
