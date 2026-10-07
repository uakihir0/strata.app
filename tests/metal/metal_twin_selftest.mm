// Negative controls that prove the shared twin checks reject planted kernel faults.
#include "synth.hpp"
#include "twin.hpp"

#include "metallib_sha256.inc"

#import <Metal/Metal.h>

#include <array>
#include <bit>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <random>
#include <string>
#include <vector>

using namespace strata::metal;

namespace {

struct RowParams {
    std::uint32_t count;
    std::uint32_t T;
};
static_assert(sizeof(RowParams) == 8);

int g_failures = 0;

void check(bool ok, const char* label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", ok ? "ok  " : "FAIL", label, detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !ok;
}

void test_oracle_primitives() {
    const twin::tf a{2.0f};
    const twin::tf b{3.0f};
    const twin::tf c{5.0f};
    const twin::tf sum = a * b + c;
    check(sum.value == 11.0f && sum.magnitude == 11.0, "oracle arithmetic and magnitude propagation");
    const twin::tf cancellation = twin::tf{1.0f} - twin::tf{1.0f};
    check(cancellation.value == 0.0f && cancellation.magnitude == 2.0,
          "oracle subtraction adds operand magnitudes");
    check(twin::accurate_exp(twin::tf{0.0f}).value == 1.0f &&
              twin::accurate_log(twin::tf{1.0f}).value == 0.0f &&
              twin::accurate_sqrt(twin::tf{4.0f}).value == 2.0f,
          "oracle library functions round double results to float");

    const float x = 1.0f + 0x1.0p-23f;
    const float y = 1.0f - 0x1.0p-23f;
    const std::uint32_t separate_bits = twin::contraction_probe_bits(x, y, -1.0f);
    const std::uint32_t expected_separate = std::bit_cast<std::uint32_t>(0.0f);
    const std::uint32_t fused_bits = std::bit_cast<std::uint32_t>(std::fma(x, y, -1.0f));
    check(separate_bits == expected_separate && fused_bits != expected_separate,
          "oracle library is compiled with contraction disabled");

    const std::array<float, 1> one = {1.0f};
    const std::array<twin::tf, 1> finite_expected = {twin::tf{1.0f}};
    const auto invalid_u = twin::compare_floats(
        one, finite_expected, {twin::Comparison::units, std::numeric_limits<double>::infinity()});
    check(invalid_u.has_value(), "U(n) comparator rejects an infinite tolerance");
    const std::array<twin::tf, 1> invalid_magnitude = {
        twin::tf{1.0f, std::numeric_limits<double>::quiet_NaN()}};
    check(twin::compare_floats(one, invalid_magnitude, {twin::Comparison::units, 2.0}).has_value(),
          "U(n) comparator rejects an invalid oracle magnitude");
    const std::array<twin::tf, 1> huge_magnitude = {
        twin::tf{1.0f, std::numeric_limits<double>::max()}};
    check(twin::compare_floats(one, huge_magnitude,
                               {twin::Comparison::units, std::numeric_limits<double>::max()}).has_value(),
          "U(n) comparator rejects an overflowed error bound");

    const std::array<float, 1> poison_nan = {std::bit_cast<float>(0xffffffffu)};
    const std::array<twin::tf, 1> nan_expected = {twin::tf{std::numeric_limits<float>::quiet_NaN()}};
    check(!twin::compare_floats(
              poison_nan, nan_expected, {twin::Comparison::units, 2.0}),
          "U(n) comparator permits matching NaN output payloads");

    const std::array<std::uint8_t, 4> poison_bytes = {0xff, 0xff, 0xff, 0xff};
    check(!twin::compare_bytes(poison_bytes, poison_bytes),
          "B comparator permits legitimate all-0xFF output bytes");
    const std::array<double, 1> nan_reference = {std::numeric_limits<double>::quiet_NaN()};
    check(!twin::compare_relative_l2(poison_nan, nan_reference, 0.0, 1),
          "R(epsilon) permits a matching NaN reference position");

    const std::array<double, 1> fp64_reference = {1.00000001};
    check(twin::compare_relative_l2(one, fp64_reference, 1.0e-9, 1).has_value(),
          "R(epsilon) uses the FP64 oracle value");
    check(!twin::compare_relative_l2(one, fp64_reference, 2.0e-8, 1).has_value(),
          "R(epsilon) accepts a sufficient FP64-relative tolerance");
    check(twin::compare_relative_l2(one, fp64_reference,
                                    std::numeric_limits<double>::quiet_NaN(), 1).has_value(),
          "R(epsilon) comparator rejects an invalid tolerance");

    const std::array<std::uint32_t, 2> single_rows = {
        std::bit_cast<std::uint32_t>(1.0f), std::bit_cast<std::uint32_t>(2.0f)};
    auto many_rows = single_rows;
    const auto single_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(single_rows.data()), sizeof(single_rows));
    auto many_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(many_rows.data()), sizeof(many_rows));
    check(!twin::row_t_independent(single_bytes, many_bytes, sizeof(float), 2, sizeof(float)).has_value(),
          "row_t_independent compares different rows with their matching T=1 baselines");
    many_rows[1] = std::bit_cast<std::uint32_t>(3.0f);
    const auto row_failure =
        twin::row_t_independent(single_bytes, many_bytes, sizeof(float), 2, sizeof(float));
    check(row_failure.has_value() && row_failure->index == 1,
          "row_t_independent reports the first differing element");
}

id<MTLComputePipelineState> make_pipeline(id<MTLDevice> device, id<MTLLibrary> library,
                                          NSString* name, std::string& err) {
    id<MTLFunction> function = [library newFunctionWithName:name];
    if (function == nil) {
        err = std::string("test library is missing ") + name.UTF8String;
        return nil;
    }
    NSError* error = nil;
    id<MTLComputePipelineState> pipeline = [device newComputePipelineStateWithFunction:function error:&error];
    if (pipeline == nil) err = error != nil ? error.localizedDescription.UTF8String : "unknown pipeline error";
    return pipeline;
}

constexpr const char* kFaultSource =
    "#include <metal_stdlib>\n"
    "using namespace metal;\n"
    "struct RowParams { uint count; uint T; };\n"
    "kernel void overrun_kernel(device uchar* out [[buffer(0)]]) {\n"
    "  out[0] = 0u; out[1] = 0u;\n"
    "}\n"
    "kernel void one_ulp_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]]) {\n"
    "  y[0] = as_type<float>(as_type<uint>(x[0]) + 1u);\n"
    "}\n"
    "kernel void row_count_kernel(device const float* x [[buffer(0)]], device float* y [[buffer(1)]],\n"
    "                             constant RowParams& p [[buffer(2)]], uint i [[thread_position_in_grid]]) {\n"
    "  if (i < p.count) y[i] = x[i] + float(p.T);\n"
    "}\n"
    "kernel void grid_3d_kernel(device uint* out [[buffer(0)]], uint3 gid [[thread_position_in_grid]]) {\n"
    "  out[(gid.z * 4u + gid.y) * 4u + gid.x] = (gid.z * 16u + gid.y * 4u + gid.x);\n"
    "}\n";

void expect_planted_failure(const char* label, const twin::Options& options, std::uint32_t T,
                            std::size_t expected_index,
                            const std::optional<twin::Failure>& failure) {
    if (!failure) {
        check(false, label, "planted fault escaped the twin checks");
        return;
    }
    twin::report_failure(label, options, T, *failure);
    check(failure->index == expected_index, label,
          "planted fault rejected at the expected first failing index " + std::to_string(expected_index));
}

bool dispatch_once(Context& context, id<MTLComputePipelineState> pipeline, const char* label,
                   std::vector<twin::Binding> bindings, std::uint32_t threads,
                   std::uint32_t threads_per_group, const void* parameters,
                   std::size_t parameter_bytes, std::string& err) {
    twin::Dispatch work;
    work.pipeline = (__bridge void*)pipeline;
    work.label = label;
    work.buffers = std::move(bindings);
    work.grid = {threads, 1, 1};
    work.threads_per_group = {threads_per_group, 1, 1};
    work.parameters = parameters;
    work.parameter_bytes = parameter_bytes;
    work.parameter_index = 2;
    return twin::dispatch(context, work, err);
}

BufferRef subview(BufferRef buffer, std::size_t offset, std::size_t length) {
    return buffer.slice(offset, length);
}

void test_overrun(Context& context, id<MTLComputePipelineState> pipeline,
                  const twin::Options& options) {
    std::string err;
    twin::GuardedBuffer output;
    check(output.allocate(context, "twin overrun output", 1, err), "allocate overrun output", err);
    if (output.view().handle == nullptr) return;
    const std::array<std::uint8_t, 1> expected = {0};
    check(output.poison_bytes_against(expected, err), "initialize output against byte oracle", err);
    output.capture_before_dispatch();
    check(dispatch_once(context, pipeline, "twin one-byte overrun",
                        {{0, output.view()}}, 1, 1, nullptr, 0, err),
          "dispatch overrun kernel", err);
    const auto declared = twin::compare_bytes(output.payload(), expected);
    check(!declared, "declared output matches oracle");
    constexpr std::array<twin::ByteRange, 1> writes = {{{0, 1}}};
    expect_planted_failure("poison guard detects one-byte overrun", options, 1, 1,
                           output.first_unwritten_change(writes));
}

void test_one_ulp(Context& context, id<MTLComputePipelineState> pipeline,
                  std::span<const float> inputs, const twin::Options& options) {
    std::string err;
    twin::GuardedBuffer input;
    twin::GuardedBuffer output;
    check(input.allocate(context, "twin one-ulp input", sizeof(float), err), "allocate one-ulp input", err);
    check(output.allocate(context, "twin one-ulp output", sizeof(float), err), "allocate one-ulp output", err);
    if (input.view().handle == nullptr || output.view().handle == nullptr) return;
    const float value = inputs.front();
    const std::array<twin::tf, 1> expected = {twin::tf{value}};
    std::memcpy(input.payload().data(), &value, sizeof(value));
    check(output.poison_floats_against(expected, twin::Comparison::bits, err),
          "initialize one-ULP output against bit oracle", err);
    input.capture_before_dispatch();
    output.capture_before_dispatch();
    check(dispatch_once(context, pipeline, "twin one-ulp kernel",
                        {{0, input.view()}, {1, output.view()}}, 1, 1, nullptr, 0, err),
          "dispatch one-ulp kernel", err);
    check(!input.first_unwritten_change({}), "input and its guards remain unchanged");
    constexpr std::array<twin::ByteRange, 1> writes = {{{0, sizeof(float)}}};
    check(!output.first_unwritten_change(writes), "one-ulp output guards remain unchanged");
    const std::span<const float> actual(static_cast<const float*>(output.view().contents), 1);
    expect_planted_failure("B comparator detects a one-ULP error", options, 1, 0,
                           twin::compare_floats(actual, expected, {twin::Comparison::bits}));
}

void test_row_count(Context& context, id<MTLComputePipelineState> pipeline,
                    std::span<const float> inputs, const twin::Options& options) {
    std::string err;
    twin::GuardedBuffer input;
    twin::GuardedBuffer one;
    twin::GuardedBuffer many;
    check(input.allocate(context, "row T input", 2 * sizeof(float), err), "allocate row T input", err);
    check(one.allocate(context, "row T one-row output", 2 * sizeof(float), err), "allocate one-row output", err);
    check(many.allocate(context, "row T multi-row output", 2 * sizeof(float), err), "allocate multi-row output", err);
    if (input.view().handle == nullptr || one.view().handle == nullptr || many.view().handle == nullptr) return;
    std::memcpy(input.payload().data(), inputs.data(), 2 * sizeof(float));
    const std::array<twin::tf, 2> expected_one = {
        twin::tf{inputs[0]} + twin::tf{1.0f}, twin::tf{inputs[1]} + twin::tf{1.0f}};
    const std::array<twin::tf, 2> expected_many = {
        twin::tf{inputs[0]} + twin::tf{2.0f}, twin::tf{inputs[1]} + twin::tf{2.0f}};
    check(one.poison_floats_against(expected_one, twin::Comparison::bits, err),
          "initialize T=1 output against bit oracle", err);
    check(many.poison_floats_against(expected_many, twin::Comparison::bits, err),
          "initialize T=2 output against bit oracle", err);
    input.capture_before_dispatch();
    one.capture_before_dispatch();
    many.capture_before_dispatch();

    RowParams params{1, 1};
    bool baseline_ok = true;
    for (std::size_t row = 0; row < 2; ++row) {
        baseline_ok &= dispatch_once(context, pipeline, "twin row t T=1 baseline",
                                     {{0, subview(input.view(), row * sizeof(float), sizeof(float))},
                                      {1, subview(one.view(), row * sizeof(float), sizeof(float))}},
                                     1, 2, &params, sizeof(params), err);
    }
    check(baseline_ok, "dispatch each row's T=1 baseline", err);
    params = {2, 2};
    check(dispatch_once(context, pipeline, "twin row T=2 negative control",
                        {{0, input.view()}, {1, many.view()}}, 2, 2,
                        &params, sizeof(params), err),
          "dispatch T=2 negative control", err);
    check(!input.first_unwritten_change({}), "row T input remains unchanged");
    constexpr std::array<twin::ByteRange, 1> one_write = {{{0, 2 * sizeof(float)}}};
    constexpr std::array<twin::ByteRange, 1> two_writes = {{{0, 2 * sizeof(float)}}};
    check(!one.first_unwritten_change(one_write), "T=1 output guards remain unchanged");
    check(!many.first_unwritten_change(two_writes), "T=2 output guards remain unchanged");
    const std::span<const float> actual_one(static_cast<const float*>(one.view().contents), 2);
    check(!twin::compare_floats(actual_one, expected_one, {twin::Comparison::bits}),
          "T=1 output matches its oracle");
    expect_planted_failure("row_t_independent detects output changing with T", options, 2, 0,
                           twin::row_t_independent(one.payload(), many.payload(),
                                                   sizeof(float), 2, sizeof(float)));
}

void test_manifest_dispatch(Context& context, std::span<const float> inputs,
                            const twin::Options& options) {
    std::string err;
    twin::GuardedBuffer input;
    twin::GuardedBuffer output;
    constexpr std::size_t count = 8;
    constexpr std::size_t bytes = count * sizeof(float);
    check(input.allocate(context, "manifest twin input", bytes, err), "allocate manifest twin input", err);
    check(output.allocate(context, "manifest twin output", bytes, err), "allocate manifest twin output", err);
    if (input.view().handle == nullptr || output.view().handle == nullptr) return;
    std::array<twin::tf, count> expected{};
    for (std::size_t i = 0; i < count; ++i) {
        float value = inputs[i];
        if (value == -1.0f) value = 0.5f;  // The in-place kernel must not leave a fixed point unwritten.
        std::memcpy(input.payload().data() + i * sizeof(float), &value, sizeof(value));
        expected[i] = twin::tf{value} * twin::tf{2.0f} + twin::tf{1.0f};
    }
    // strata_selftest_add is intentionally in-place; the input value itself is a
    // valid nonmatching poison for its expected output.
    std::memcpy(output.payload().data(), input.payload().data(), bytes);
    input.capture_before_dispatch();
    output.capture_before_dispatch();

    twin::Dispatch work;
    work.buffers = {{0, output.view()}};
    work.grid = {count, 1, 1};
    work.threads_per_group = {128, 2, 1};
    check(!twin::dispatch_manifest(context, "strata_selftest_add", work, err) &&
              err.find("shape does not match the manifest") != std::string::npos,
          "manifest dispatch rejects a same-product but differently shaped threadgroup");

    work.threads_per_group = {0, 0, 0};
    work.grid_mode = twin::GridMode::threads;
    check(twin::dispatch_manifest(context, "strata_selftest_add", work, err),
          "manifest dispatch uses the declared threadgroup size", err);
    constexpr std::array<twin::ByteRange, 1> writes = {{{0, bytes}}};
    check(!input.first_unwritten_change({}), "manifest source input remains unchanged");
    check(!output.first_unwritten_change(writes), "manifest dispatch preserves both poison guards");
    const std::span<const float> actual(static_cast<const float*>(output.view().contents), count);
    const auto failure = twin::compare_floats(actual, expected, {twin::Comparison::bits});
    if (failure) twin::report_failure("manifest selftest twin", options, 1, *failure);
    check(!failure, "manifest selftest output matches its oracle");
}

void test_three_dimensional_dispatch(Context& context, id<MTLComputePipelineState> pipeline) {
    std::string err;
    twin::GuardedBuffer output;
    constexpr std::size_t count = 64;
    constexpr std::size_t bytes = count * sizeof(std::uint32_t);
    check(output.allocate(context, "3D grid output", bytes, err), "allocate 3D grid output", err);
    if (output.view().handle == nullptr) return;
    std::array<std::uint32_t, count> expected{};
    for (std::size_t i = 0; i < count; ++i) expected[i] = static_cast<std::uint32_t>(i);
    const auto expected_bytes = std::span<const std::uint8_t>(
        reinterpret_cast<const std::uint8_t*>(expected.data()), sizeof(expected));
    check(output.poison_bytes_against(expected_bytes, err),
          "initialize 3D output against byte oracle", err);
    output.capture_before_dispatch();

    twin::Dispatch work;
    work.pipeline = (__bridge void*)pipeline;
    work.label = "twin three-dimensional grid";
    work.buffers = {{0, output.view()}};
    work.grid_mode = twin::GridMode::threadgroups;
    work.grid = {2, 2, 2};
    work.threads_per_group = {2, 2, 2};
    check(twin::dispatch(context, work, err),
          "dispatch multiple 3D threadgroups over a 4x4x4 thread grid", err);
    check(!twin::compare_bytes(output.payload(), expected_bytes), "3D grid output matches its oracle");
    constexpr std::array<twin::ByteRange, 1> writes = {{{0, bytes}}};
    check(!output.first_unwritten_change(writes), "3D grid preserves both poison guards");

    work.label = "twin three-dimensional exact thread grid";
    work.grid_mode = twin::GridMode::threads;
    work.grid = {4, 4, 4};
    check(twin::dispatch(context, work, err), "dispatch an exact 3D thread grid", err);
    check(!twin::compare_bytes(output.payload(), expected_bytes),
          "exact 3D thread grid output matches its oracle");
}

void test_unwritten_outputs(Context& context) {
    std::string err;
    twin::GuardedBuffer bytes;
    twin::GuardedBuffer units;
    twin::GuardedBuffer relative;
    check(bytes.allocate(context, "unwritten byte output", 4, err), "allocate unwritten byte output", err);
    check(units.allocate(context, "unwritten U output", sizeof(float), err), "allocate unwritten U output", err);
    check(relative.allocate(context, "unwritten R output", sizeof(float), err),
          "allocate unwritten R output", err);
    if (!bytes.view() || !units.view() || !relative.view()) return;

    const std::array<std::uint8_t, 4> all_ff = {0xff, 0xff, 0xff, 0xff};
    const std::array<twin::tf, 1> expected_nan = {
        twin::tf{std::numeric_limits<float>::quiet_NaN()}};
    const std::array<double, 1> reference_nan = {
        std::numeric_limits<double>::quiet_NaN()};
    check(bytes.poison_bytes_against(all_ff, err), "prepoison all-0xFF byte oracle output", err);
    check(units.poison_floats_against(expected_nan, twin::Comparison::units, err),
          "prepoison NaN U oracle output", err);
    check(relative.poison_floats_against(reference_nan, twin::Comparison::relative_l2, err),
          "prepoison NaN R oracle output", err);
    bytes.capture_before_dispatch();
    units.capture_before_dispatch();
    relative.capture_before_dispatch();

    check(twin::compare_bytes(bytes.payload(), all_ff).has_value(),
          "an unwritten all-0xFF expected byte range is detected");
    const std::span<const float> units_actual(static_cast<const float*>(units.view().contents), 1);
    check(twin::compare_floats(units_actual, expected_nan,
                               {twin::Comparison::units, 2.0}).has_value(),
          "an unwritten NaN U output is detected without reserving a NaN payload");
    const std::span<const float> relative_actual(static_cast<const float*>(relative.view().contents), 1);
    check(twin::compare_relative_l2(relative_actual, reference_nan, 0.0, 1).has_value(),
          "an unwritten NaN R output is detected without reserving a NaN payload");
}

bool run_set(Context& context, id<MTLComputePipelineState> overrun,
             id<MTLComputePipelineState> one_ulp, id<MTLComputePipelineState> row_count,
             id<MTLComputePipelineState> grid_3d,
             twin::Options options) {
    std::mt19937_64 rng(options.seed);
    std::string err;
    const std::vector<float> inputs = twin::synthesize_floats(options.input_set, 8, rng, err);
    if (!err.empty()) {
        std::printf("SKIP input set %s: %s\n", twin::input_set_name(options.input_set), err.c_str());
        return false;
    }
    check(!inputs.empty(), "synthesize deterministic twin inputs");
    if (inputs.empty()) return false;
    test_manifest_dispatch(context, inputs, options);
    test_three_dimensional_dispatch(context, grid_3d);
    test_overrun(context, overrun, options);
    test_one_ulp(context, one_ulp, inputs, options);
    test_row_count(context, row_count, inputs, options);
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr, "usage: metal_twin_selftest strata.metallib [--seed N] [--input-set random|edge|fixture|real]\n");
        return 2;
    }

    test_oracle_primitives();
    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;
    test_unwritten_outputs(*context);

    id<MTLDevice> device = (__bridge id<MTLDevice>)context->device();
    NSError* compile_error = nil;
    id<MTLLibrary> library =
        [device newLibraryWithSource:@(kFaultSource) options:nil error:&compile_error];
    check(library != nil, "compile the three planted fault kernels",
          compile_error != nil ? compile_error.localizedDescription.UTF8String : "");
    if (library == nil) return 1;

    id<MTLComputePipelineState> overrun = make_pipeline(device, library, @"overrun_kernel", err);
    id<MTLComputePipelineState> one_ulp = make_pipeline(device, library, @"one_ulp_kernel", err);
    id<MTLComputePipelineState> row_count = make_pipeline(device, library, @"row_count_kernel", err);
    id<MTLComputePipelineState> grid_3d = make_pipeline(device, library, @"grid_3d_kernel", err);
    check(overrun != nil && one_ulp != nil && row_count != nil && grid_3d != nil,
          "build all twin selftest pipelines", err);
    if (overrun == nil || one_ulp == nil || row_count == nil || grid_3d == nil) return 1;

    std::vector<twin::InputSet> sets;
    if (options.input_set_explicit) {
        sets.push_back(options.input_set);
    } else {
        sets = {twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
    }
    for (const twin::InputSet set : sets) {
        options.input_set = set;
        if (!run_set(*context, overrun, one_ulp, row_count, grid_3d, options)) ++g_failures;
    }
    std::printf("metal_twin_selftest: %s\n", g_failures == 0 ? "OK" : "FAILED");
    return g_failures == 0 ? 0 : 1;
}
