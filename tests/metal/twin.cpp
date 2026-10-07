// Objective-C++ implementation of the shared Metal twin harness.
#include "twin.hpp"
#include "../../src/metal/engine/metal_fence.hpp"

#import <Metal/Metal.h>

#include <algorithm>
#include <bit>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

namespace twin {
namespace {

std::uint32_t float_bits(float value) {
    return std::bit_cast<std::uint32_t>(value);
}

bool in_written_range(std::size_t index, std::span<const ByteRange> written) {
    for (const ByteRange range : written) {
        if (range.offset <= index && index - range.offset < range.length) return true;
    }
    return false;
}

bool size_product(GridSize size, std::uint64_t& product) {
    constexpr std::uint64_t limit = std::numeric_limits<std::uint64_t>::max();
    if (size.x == 0 || size.y == 0 || size.z == 0) return false;
    const std::uint64_t xy = static_cast<std::uint64_t>(size.x) * size.y;
    if (xy > limit / size.z) return false;
    product = xy * size.z;
    return true;
}

bool same_nonfinite(float actual, float expected) {
    if (std::isnan(expected)) return std::isnan(actual);
    if (std::isinf(expected)) return std::isinf(actual) && std::signbit(actual) == std::signbit(expected);
    return std::isfinite(actual);
}

Failure float_failure(std::size_t index, float actual, float expected, const char* reason) {
    return {index, float_bits(actual), float_bits(expected), reason};
}

}  // namespace

const char* input_set_name(InputSet set) noexcept {
    switch (set) {
        case InputSet::random: return "random";
        case InputSet::edge: return "edge";
        case InputSet::fixture: return "fixture";
        case InputSet::real: return "real";
    }
    return "unknown";
}

std::optional<InputSet> parse_input_set(const std::string& name) noexcept {
    if (name == "random") return InputSet::random;
    if (name == "edge") return InputSet::edge;
    if (name == "fixture") return InputSet::fixture;
    if (name == "real") return InputSet::real;
    return std::nullopt;
}

bool parse_options(int argc, char** argv, int first_option, Options& options, std::string& err) {
    for (int i = first_option; i < argc; ++i) {
        const std::string option = argv[i];
        if (option == "--seed") {
            if (i + 1 >= argc) {
                err = "--seed needs an unsigned integer";
                return false;
            }
            const char* begin = argv[++i];
            const char* end = begin + std::strlen(begin);
            std::uint64_t value = 0;
            const auto parsed = std::from_chars(begin, end, value, 10);
            if (parsed.ec != std::errc{} || parsed.ptr != end) {
                err = "--seed must be an unsigned 64-bit integer";
                return false;
            }
            options.seed = value;
        } else if (option == "--input-set") {
            if (i + 1 >= argc) {
                err = "--input-set needs random, edge, fixture or real";
                return false;
            }
            const auto parsed = parse_input_set(argv[++i]);
            if (!parsed) {
                err = "--input-set needs random, edge, fixture or real";
                return false;
            }
            options.input_set = *parsed;
            options.input_set_explicit = true;
        } else {
            err = "unknown twin option " + option;
            return false;
        }
    }
    err.clear();
    return true;
}

bool GuardedBuffer::allocate(strata::metal::Context& context, const std::string& label,
                             std::size_t payload_bytes, std::string& err) {
    if (payload_bytes == 0 || payload_bytes > std::numeric_limits<std::size_t>::max() - 2 * kGuardBytes) {
        err = "guarded buffer payload size is invalid";
        return false;
    }
    storage_ = context.buffers().allocate(label, payload_bytes + 2 * kGuardBytes, err, true);
    if (!storage_) return false;
    auto* contents = static_cast<std::uint8_t*>(storage_.contents);
    if (contents == nullptr || storage_.gpu_address > std::numeric_limits<std::uint64_t>::max() - kGuardBytes) {
        storage_ = {};
        err = "guarded buffer has no addressable shared contents";
        return false;
    }
    payload_bytes_ = payload_bytes;
    view_ = storage_.slice(kGuardBytes, payload_bytes);
    if (!view_) {
        storage_ = {};
        payload_bytes_ = 0;
        err = "guarded buffer payload is outside its Metal allocation";
        return false;
    }
    before_.clear();
    err.clear();
    return true;
}

std::span<std::uint8_t> GuardedBuffer::payload() noexcept {
    return {static_cast<std::uint8_t*>(view_.contents), payload_bytes_};
}

std::span<const std::uint8_t> GuardedBuffer::payload() const noexcept {
    return {static_cast<const std::uint8_t*>(view_.contents), payload_bytes_};
}

std::span<const std::uint8_t> GuardedBuffer::allocation() const noexcept {
    if (!storage_ || storage_.contents == nullptr) return {};
    return {static_cast<const std::uint8_t*>(storage_.contents), storage_.length};
}

bool GuardedBuffer::poison_bytes_against(std::span<const std::uint8_t> expected, std::string& err) {
    if (expected.size() != payload_bytes_) {
        err = "byte poison and expected output lengths differ";
        return false;
    }
    auto bytes = payload();
    for (std::size_t i = 0; i < bytes.size(); ++i) bytes[i] = expected[i] ^ 0x01u;
    err.clear();
    return true;
}

bool GuardedBuffer::poison_floats_against(std::span<const tf> expected, Comparison comparison,
                                          std::string& err) {
    if (expected.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        expected.size() * sizeof(float) != payload_bytes_) {
        err = "float poison and expected output lengths differ";
        return false;
    }
    if (comparison != Comparison::bits && comparison != Comparison::units &&
        comparison != Comparison::relative_l2) {
        err = "float poison has an unknown comparison class";
        return false;
    }
    auto bytes = payload();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const float want = expected[i].value;
        float poison = 0.0f;
        if (comparison == Comparison::bits) {
            poison = std::bit_cast<float>(float_bits(want) ^ 1u);
        } else if (std::isfinite(want)) {
            poison = std::numeric_limits<float>::quiet_NaN();
        }
        std::memcpy(bytes.data() + i * sizeof(float), &poison, sizeof(poison));
    }
    err.clear();
    return true;
}

bool GuardedBuffer::poison_floats_against(std::span<const double> expected, Comparison comparison,
                                          std::string& err) {
    if (expected.size() > std::numeric_limits<std::size_t>::max() / sizeof(float) ||
        expected.size() * sizeof(float) != payload_bytes_) {
        err = "float poison and expected output lengths differ";
        return false;
    }
    if (comparison != Comparison::bits && comparison != Comparison::units &&
        comparison != Comparison::relative_l2) {
        err = "float poison has an unknown comparison class";
        return false;
    }
    auto bytes = payload();
    for (std::size_t i = 0; i < expected.size(); ++i) {
        const double want = expected[i];
        float poison = 0.0f;
        if (comparison == Comparison::bits) {
            poison = std::bit_cast<float>(float_bits(static_cast<float>(want)) ^ 1u);
        } else if (std::isfinite(want)) {
            poison = std::numeric_limits<float>::quiet_NaN();
        }
        std::memcpy(bytes.data() + i * sizeof(float), &poison, sizeof(poison));
    }
    err.clear();
    return true;
}

void GuardedBuffer::capture_before_dispatch() {
    const auto bytes = allocation();
    before_.assign(bytes.begin(), bytes.end());
}

std::optional<Failure> GuardedBuffer::first_unwritten_change(std::span<const ByteRange> written) const {
    const auto current = allocation();
    if (before_.size() != current.size()) {
        return Failure{0, current.size(), before_.size(), "buffer allocation changed after dispatch"};
    }
    for (std::size_t i = 0; i < current.size(); ++i) {
        const bool inside_payload = i >= kGuardBytes && i - kGuardBytes < payload_bytes_;
        if (inside_payload && in_written_range(i - kGuardBytes, written)) continue;
        if (current[i] != before_[i]) {
            const bool leading_guard = i < kGuardBytes;
            const std::size_t index = leading_guard ? 0 : i - kGuardBytes;
            const std::string reason = leading_guard
                ? "leading poison guard changed at allocation byte " + std::to_string(i)
                : "poison or read-only byte changed outside declared output range";
            return Failure{index, current[i], before_[i], reason};
        }
    }
    return std::nullopt;
}

bool dispatch(strata::metal::Context& context, const Dispatch& work, std::string& err) {
    std::uint64_t grid_threads = 0;
    std::uint64_t group_threads = 0;
    if (work.pipeline == nullptr || !size_product(work.grid, grid_threads) || grid_threads == 0 ||
        !size_product(work.threads_per_group, group_threads)) {
        err = "twin dispatch has an invalid pipeline or grid";
        return false;
    }
    id<MTLComputePipelineState> pipeline = (__bridge id<MTLComputePipelineState>)work.pipeline;
    if (group_threads > pipeline.maxTotalThreadsPerThreadgroup) {
        err = "twin dispatch threadgroup exceeds the pipeline limit";
        return false;
    }
    auto command = context.make_command_buffer(work.label, err);
    if (!command) return false;
    id<MTLCommandBuffer> native_command =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    id<MTLComputeCommandEncoder> encoder =
        strata::metal::begin_compute_encoder(context, native_command);
    if (encoder == nil) {
        command->discard();
        err = "cannot create compute encoder for " + work.label;
        return false;
    }
    encoder.label = @(work.label.c_str());
    [encoder setComputePipelineState:pipeline];
    for (const Binding& binding : work.buffers) {
        id<MTLBuffer> native_buffer = (__bridge id<MTLBuffer>)binding.buffer.handle;
        if (!binding.buffer || native_buffer == nil || binding.buffer.offset > native_buffer.length ||
            binding.buffer.length > native_buffer.length - binding.buffer.offset) {
            strata::metal::end_compute_encoder(context, encoder);
            command->discard();
            err = "twin dispatch has an invalid buffer binding";
            return false;
        }
        [encoder setBuffer:(__bridge id<MTLBuffer>)binding.buffer.handle
                    offset:binding.buffer.offset
                   atIndex:binding.index];
    }
    if (work.parameter_bytes != 0) {
        if (work.parameters == nullptr) {
            strata::metal::end_compute_encoder(context, encoder);
            command->discard();
            err = "twin dispatch has a null parameter block";
            return false;
        }
        [encoder setBytes:work.parameters length:work.parameter_bytes atIndex:work.parameter_index];
    }
    if (work.threadgroup_memory_bytes != 0) {
        [encoder setThreadgroupMemoryLength:work.threadgroup_memory_bytes
                                    atIndex:work.threadgroup_memory_index];
    }
    const MTLSize grid = MTLSizeMake(work.grid.x, work.grid.y, work.grid.z);
    const MTLSize threadgroup = MTLSizeMake(work.threads_per_group.x,
                                           work.threads_per_group.y,
                                           work.threads_per_group.z);
    if (work.grid_mode == GridMode::threads) {
        [encoder dispatchThreads:grid threadsPerThreadgroup:threadgroup];
    } else if (work.grid_mode == GridMode::threadgroups) {
        [encoder dispatchThreadgroups:grid threadsPerThreadgroup:threadgroup];
    } else {
        strata::metal::end_compute_encoder(context, encoder);
        command->discard();
        err = "twin dispatch has an unknown grid mode";
        return false;
    }
    strata::metal::end_compute_encoder(context, encoder);
    if (!command->commit(err)) return false;
    [native_command waitUntilCompleted];
    if (native_command.status != MTLCommandBufferStatusCompleted) {
        const char* message = native_command.error.localizedDescription.UTF8String;
        err = work.label + " command buffer failed";
        if (message != nullptr && message[0] != '\0') err += ": " + std::string(message);
        return false;
    }
    err.clear();
    return true;
}

bool dispatch_manifest(strata::metal::Context& context, const char* kernel_name,
                       Dispatch work, std::string& err) {
    if (kernel_name == nullptr || kernel_name[0] == '\0') {
        err = "twin manifest dispatch needs a kernel name";
        return false;
    }
    std::size_t count = 0;
    const strata::metal::KernelInfo* manifest = strata::metal::kernel_manifest(count);
    const strata::metal::KernelInfo* found = nullptr;
    for (std::size_t i = 0; i < count; ++i) {
        if (std::strcmp(manifest[i].name, kernel_name) == 0) {
            found = &manifest[i];
            break;
        }
    }
    if (found == nullptr) {
        err = std::string("twin kernel is absent from the manifest: ") + kernel_name;
        return false;
    }
    const bool use_manifest_group =
        work.threads_per_group.x == 0 && work.threads_per_group.y == 0 &&
        work.threads_per_group.z == 0;
    std::uint64_t group_threads = 0;
    if (!use_manifest_group &&
        (!size_product(work.threads_per_group, group_threads) ||
         work.threads_per_group.x != found->threadgroup_x ||
         work.threads_per_group.y != found->threadgroup_y ||
         work.threads_per_group.z != found->threadgroup_z)) {
        err = std::string("twin threadgroup shape does not match the manifest for ") + kernel_name;
        return false;
    }
    work.pipeline = context.pipeline(kernel_name);
    if (work.pipeline == nullptr) {
        err = std::string("twin pipeline is unavailable: ") + kernel_name;
        return false;
    }
    work.label = kernel_name;
    if (use_manifest_group) {
        work.threads_per_group = {
            found->threadgroup_x, found->threadgroup_y, found->threadgroup_z};
    }
    return dispatch(context, work, err);
}

std::optional<Failure> compare_bytes(std::span<const std::uint8_t> actual,
                                     std::span<const std::uint8_t> expected) {
    const std::size_t count = std::min(actual.size(), expected.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (actual[i] != expected[i]) return Failure{i, actual[i], expected[i], "bytewise mismatch"};
    }
    if (actual.size() != expected.size()) {
        return Failure{count, actual.size(), expected.size(), "byte range length mismatch"};
    }
    return std::nullopt;
}

std::optional<Failure> compare_floats(std::span<const float> actual,
                                      std::span<const tf> expected,
                                      Tolerance tolerance) {
    if (actual.size() != expected.size()) {
        return Failure{std::min(actual.size(), expected.size()), actual.size(), expected.size(),
                       "float range length mismatch"};
    }
    if (tolerance.comparison != Comparison::bits && tolerance.comparison != Comparison::units &&
        tolerance.comparison != Comparison::relative_l2) {
        return Failure{0, 0, 0, "unknown floating-point comparison class"};
    }
    if (tolerance.comparison == Comparison::relative_l2) {
        return Failure{0, 0, 0, "R(epsilon) needs the FP64 compare_relative_l2 oracle"};
    }
    if (tolerance.comparison == Comparison::units &&
        (!std::isfinite(tolerance.n) || tolerance.n < 0.0)) {
        return Failure{0, 0, 0, "U(n) tolerance must be finite and nonnegative"};
    }
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float want = expected[i].value;
        if (tolerance.comparison == Comparison::bits) {
            if (float_bits(actual[i]) != float_bits(want)) return float_failure(i, actual[i], want, "B bit mismatch");
            continue;
        }
        if (!same_nonfinite(actual[i], want)) return float_failure(i, actual[i], want, "U(n) non-finite mismatch");
        if (!std::isfinite(want)) continue;
        if (!std::isfinite(expected[i].magnitude) || expected[i].magnitude < 0.0) {
            return Failure{i, float_bits(actual[i]), float_bits(want), "U(n) oracle magnitude is invalid"};
        }
        const double bound = tolerance.n * std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        if (!std::isfinite(bound)) {
            return Failure{i, float_bits(actual[i]), float_bits(want), "U(n) error bound overflowed"};
        }
        if (std::fabs(static_cast<double>(actual[i]) - want) > bound) {
            return float_failure(i, actual[i], want, "U(n) error bound exceeded");
        }
    }
    return std::nullopt;
}

std::optional<Failure> compare_relative_l2(std::span<const float> actual,
                                           std::span<const double> expected,
                                           double epsilon, std::size_t row_width) {
    if (actual.size() != expected.size()) {
        return Failure{std::min(actual.size(), expected.size()), actual.size(), expected.size(),
                       "R(epsilon) range length mismatch"};
    }
    if (!std::isfinite(epsilon) || epsilon < 0.0 || row_width == 0 || actual.size() % row_width != 0) {
        return Failure{0, actual.size(), row_width, "R(epsilon) tolerance or row width is invalid"};
    }
    for (std::size_t row = 0; row < actual.size() / row_width; ++row) {
        double error_scale = 0.0;
        double reference_scale = 0.0;
        for (std::size_t col = 0; col < row_width; ++col) {
            const std::size_t i = row * row_width + col;
            const double reference = expected[i];
            if (std::isnan(reference)) {
                if (!std::isnan(actual[i])) {
                    return Failure{i, float_bits(actual[i]), std::bit_cast<std::uint64_t>(reference),
                                   "R(epsilon) NaN mismatch"};
                }
                continue;
            }
            if (std::isinf(reference)) {
                if (!std::isinf(actual[i]) || std::signbit(actual[i]) != std::signbit(reference)) {
                    return Failure{i, float_bits(actual[i]), std::bit_cast<std::uint64_t>(reference),
                                   "R(epsilon) infinity mismatch"};
                }
                continue;
            }
            if (!std::isfinite(actual[i])) {
                return Failure{i, float_bits(actual[i]), std::bit_cast<std::uint64_t>(reference),
                               "R(epsilon) output is non-finite for a finite reference"};
            }
            const double delta = static_cast<double>(actual[i]) - reference;
            error_scale = std::max(error_scale, std::fabs(delta));
            reference_scale = std::max(reference_scale, std::fabs(reference));
        }
        double relative_error = 0.0;
        if (error_scale != 0.0) {
            if (reference_scale == 0.0) {
                relative_error = std::numeric_limits<double>::infinity();
            } else {
                double error_sq = 0.0;
                double reference_sq = 0.0;
                for (std::size_t col = 0; col < row_width; ++col) {
                    const std::size_t i = row * row_width + col;
                    if (!std::isfinite(expected[i])) continue;
                    const double delta = static_cast<double>(actual[i]) - expected[i];
                    const double scaled_error = delta / error_scale;
                    const double scaled_reference = expected[i] / reference_scale;
                    error_sq += scaled_error * scaled_error;
                    reference_sq += scaled_reference * scaled_reference;
                }
                relative_error = (error_scale / reference_scale) * std::sqrt(error_sq / reference_sq);
            }
        }
        if (relative_error > epsilon) {
            const std::size_t i = row * row_width;
            return Failure{i, float_bits(actual[i]), std::bit_cast<std::uint64_t>(expected[i]),
                           "R(epsilon) row norm exceeded"};
        }
    }
    return std::nullopt;
}

std::optional<Failure> row_t_independent(std::span<const std::uint8_t> single_row_outputs,
                                         std::span<const std::uint8_t> many_rows,
                                         std::size_t row_bytes, std::uint32_t rows,
                                         std::size_t element_bytes) {
    const bool multiplication_overflows =
        rows != 0 && row_bytes > std::numeric_limits<std::size_t>::max() / rows;
    const std::size_t expected_size =
        multiplication_overflows ? 0 : row_bytes * static_cast<std::size_t>(rows);
    if (row_bytes == 0 || rows == 0 || element_bytes == 0 || row_bytes % element_bytes != 0 ||
        multiplication_overflows || single_row_outputs.size() != expected_size ||
        many_rows.size() != expected_size) {
        return Failure{0, many_rows.size(), expected_size, "row_t_independent range shape is invalid"};
    }
    for (std::uint32_t row = 0; row < rows; ++row) {
        const auto expected_row =
            single_row_outputs.subspan(static_cast<std::size_t>(row) * row_bytes, row_bytes);
        const auto candidate = many_rows.subspan(static_cast<std::size_t>(row) * row_bytes, row_bytes);
        if (const auto diff = compare_bytes(candidate, expected_row)) {
            const std::size_t element_offset = (diff->index / element_bytes) * element_bytes;
            std::uint64_t actual_word = 0;
            std::uint64_t expected_word = 0;
            const std::size_t word_bytes = std::min(element_bytes, sizeof(std::uint64_t));
            std::memcpy(&actual_word, candidate.data() + element_offset, word_bytes);
            std::memcpy(&expected_word, expected_row.data() + element_offset, word_bytes);
            const std::size_t elements_per_row = row_bytes / element_bytes;
            return Failure{static_cast<std::size_t>(row) * elements_per_row + diff->index / element_bytes,
                           actual_word, expected_word, "row_t_independent bit mismatch"};
        }
    }
    return std::nullopt;
}

void report_failure(const char* label, const Options& options, std::uint32_t T,
                    const Failure& failure) {
    std::printf("FAIL %s seed=%llu set=%s T=%u first failing index=%zu actual=0x%llx expected=0x%llx: %s\n",
                label, static_cast<unsigned long long>(options.seed), input_set_name(options.input_set), T,
                failure.index, static_cast<unsigned long long>(failure.actual),
                static_cast<unsigned long long>(failure.expected), failure.reason.c_str());
    std::fflush(stdout);
}

}  // namespace twin
