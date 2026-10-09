// Metal twins for weighted RMS norm and PLE kernels (docs/mac/04 §5.3).
#include "metallib_sha256.inc"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/ple_dispatch.hpp"
#include "strata/metal/window_pending_dispatch.hpp"
#include "strata/metal/window_ple_dispatch.hpp"
#include "twin.hpp"
#include "oracles/bf16_mv.hpp"
#include "oracles/q2_0_mv.hpp"
#include "oracles/q8_0_mv.hpp"
#include "oracles/q8_1.hpp"

#include "../../src/metal/kernels/common/params.h"

#import <Metal/Metal.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kGateThreads = 512u;
constexpr float kGateToleranceUnits = 64.0f;
constexpr float kConvToleranceUnits = 16.0f;
constexpr std::array<std::uint32_t, 7> kHistoryLengths{
    0u, 1u, 2u, 8u, 9u, 10u, 17u};
int g_failures = 0;

void check(bool condition, const std::string& label, const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ", detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

ReadOnlyBufferRef read_only(Context& context, BufferRef buffer,
                            const std::string& label) {
    ReadOnlyBufferRef result = context.buffers().read_only(buffer);
    check(static_cast<bool>(result), label,
          result ? std::string{} : "could not create a read-only view");
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

float value_input(twin::InputSet set, std::uint32_t row, std::uint32_t col,
                  std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 8> edge{
            0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f, 16.0f, -16.0f};
        return edge[(col + 3u * row) % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t numerator =
            static_cast<std::int32_t>((col * 31u + row * 47u) % 1021u) - 510;
        return static_cast<float>(numerator) * (1.0f / 256.0f);
    }
    const std::int32_t numerator =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(numerator) * (1.0f / 8192.0f);
}

float gate_input(twin::InputSet set, std::uint32_t row, std::uint32_t stream,
                 std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 8> edge{
            0.0f, -0.0f, 0.25f, -0.5f, 1.0f, -1.0f, 8.0f, -8.0f};
        return edge[(stream + 3u * row) % edge.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t numerator =
            static_cast<std::int32_t>((stream * 37u + row * 61u) % 257u) - 128;
        return static_cast<float>(numerator) * (1.0f / 64.0f);
    }
    const std::int32_t numerator =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(numerator) * (1.0f / 16384.0f);
}

twin::tf butterfly_sum(std::array<twin::tf, 32> lanes) {
    for (std::uint32_t offset = 16u; offset > 0u; offset >>= 1u) {
        const auto previous = lanes;
        for (std::uint32_t lane = 0u; lane < lanes.size(); ++lane) {
            lanes[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return lanes[0];
}

enum class GateReductionControl {
    ordered,
    linear_thread_lanes,
    serial_simdgroup_partials,
};

twin::tf gate_oracle(std::span<const float> key,
                     std::span<const float> query,
                     float scale,
                     GateReductionControl control =
                         GateReductionControl::ordered) {
    std::array<twin::tf, kGateThreads> thread_sums{};
    for (std::uint32_t tid = 0u; tid < kGateThreads; ++tid) {
        twin::tf sum{0.0f};
        for (std::uint32_t j = 0u; j < 8u; ++j) {
            const std::uint32_t col = tid + kGateThreads * j;
            const twin::tf product =
                col < N_EMBD ? twin::tf{key[col]} * twin::tf{query[col]}
                             : twin::tf{0.0f};
            sum = sum + product;
        }
        thread_sums[tid] = sum;
    }

    twin::tf total{0.0f};
    if (control == GateReductionControl::linear_thread_lanes) {
        for (const twin::tf lane_sum : thread_sums) total = total + lane_sum;
    } else {
        std::array<twin::tf, 16> partials{};
        for (std::uint32_t simdgroup = 0u; simdgroup < partials.size();
             ++simdgroup) {
            std::array<twin::tf, 32> lanes{};
            for (std::uint32_t lane = 0u; lane < lanes.size(); ++lane) {
                lanes[lane] = thread_sums[simdgroup * lanes.size() + lane];
            }
            partials[simdgroup] = butterfly_sum(lanes);
        }
        if (control == GateReductionControl::serial_simdgroup_partials) {
            for (const twin::tf partial : partials) total = total + partial;
        } else {
            std::array<twin::tf, 32> partial_lanes{};
            std::copy(partials.begin(), partials.end(), partial_lanes.begin());
            total = butterfly_sum(partial_lanes);
        }
    }

    const twin::tf scaled =
        twin::fma(twin::tf{scale}, total, twin::tf{0.0f});
    const float sqrt_input = std::fmax(std::fabs(scaled.value), 1e-6f);
    const double sqrt_magnitude =
        std::fmax(std::fabs(scaled.magnitude), 1e-6);
    const twin::tf magnitude = twin::accurate_sqrt(
        twin::tf{sqrt_input, sqrt_magnitude});
    const float sign =
        static_cast<float>((scaled.value > 0.0f) - (scaled.value < 0.0f));
    const twin::tf sigmoid_input = twin::tf{sign} * magnitude;
    const twin::tf exponential = twin::accurate_exp(-sigmoid_input);
    return twin::tf{1.0f} / (twin::tf{1.0f} + exponential);
}

bool run_gate_case(Context& context, std::uint32_t rows,
                   twin::InputSet set, const twin::Options& options,
                   bool reduction_probe,
                   std::vector<std::uint8_t>& output_snapshot) {
    const std::string label = "ple_gate rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (reduction_probe ? " reduction-probe" : "");
    const std::size_t input_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t output_count =
        static_cast<std::size_t>(rows) * HC;
    const std::size_t input_bytes = input_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);
    const float scale = 1.0f / std::sqrt(static_cast<float>(N_EMBD));

    std::vector<float> key(input_count);
    std::vector<float> query(input_count);
    if (reduction_probe) {
        const std::size_t first_stream = 0u;
        key[first_stream] = 16777216.0f;
        key[first_stream + 1u] = 1.0f;
        key[first_stream + 16u] = -16777216.0f;
        query[first_stream] = 1.0f;
        query[first_stream + 1u] = 1.0f;
        query[first_stream + 16u] = 1.0f;

        const std::size_t second_stream = N_EMBD;
        key[second_stream] = 16777216.0f;
        key[second_stream + 32u] = 1.0f;
        key[second_stream + 256u] = -16777216.0f;
        query[second_stream] = 1.0f;
        query[second_stream + 32u] = 1.0f;
        query[second_stream + 256u] = 1.0f;
    } else {
        for (std::uint32_t row = 0u; row < rows; ++row) {
            std::mt19937_64 key_rng(options.seed ^ 0x504c454741544b59ull ^ row);
            std::mt19937_64 query_rng(options.seed ^ 0x504c454741545152ull ^ row);
            for (std::uint32_t stream = 0u; stream < HC; ++stream) {
                for (std::uint32_t col = 0u; col < N_EMBD; ++col) {
                    const std::size_t index =
                        static_cast<std::size_t>(row) * HC_DIM +
                        static_cast<std::size_t>(stream) * N_EMBD + col;
                    key[index] = value_input(set, row, col, key_rng);
                    query[index] = value_input(set, row + 1u, col, query_rng);
                }
            }
        }
    }

    std::vector<twin::tf> expected(output_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t stream = 0u; stream < HC; ++stream) {
            const std::size_t input_offset =
                static_cast<std::size_t>(row) * HC_DIM +
                static_cast<std::size_t>(stream) * N_EMBD;
            const std::size_t output_index =
                static_cast<std::size_t>(row) * HC + stream;
            const auto key_row =
                std::span<const float>(key.data() + input_offset, N_EMBD);
            const auto query_row =
                std::span<const float>(query.data() + input_offset, N_EMBD);
            expected[output_index] =
                gate_oracle(key_row, query_row, scale);
            if (reduction_probe && stream < 2u) {
                const GateReductionControl control =
                    stream == 0u
                        ? GateReductionControl::linear_thread_lanes
                        : GateReductionControl::serial_simdgroup_partials;
                const twin::tf linear =
                    gate_oracle(key_row, query_row, scale, control);
                const double allowed =
                    kGateToleranceUnits *
                        std::ldexp(expected[output_index].magnitude, -24) +
                    std::ldexp(1.0, -126);
                check(std::fabs(static_cast<double>(linear.value) -
                                expected[output_index].value) > allowed,
                      label + (stream == 0u
                                   ? " rejects linear-thread-lane reduction control"
                                   : " rejects serial-simdgroup reduction control"),
                      "tree=" + std::to_string(expected[output_index].value) +
                          " linear=" + std::to_string(linear.value));
            }
        }
    }

    std::string err;
    twin::GuardedBuffer key_buffer;
    twin::GuardedBuffer query_buffer;
    twin::GuardedBuffer gate_buffer;
    const bool shared_encoder_case =
        rows == 1u && set == twin::InputSet::fixture && !reduction_probe;
    twin::GuardedBuffer second_gate_buffer;
    check(key_buffer.allocate(context, label + " key", input_bytes, err),
          label + " key buffer", err);
    check(query_buffer.allocate(context, label + " query", input_bytes, err),
          label + " query buffer", err);
    check(gate_buffer.allocate(context, label + " gate", output_bytes, err),
          label + " output buffer", err);
    if (shared_encoder_case) {
        check(second_gate_buffer.allocate(
                  context, label + " second gate", output_bytes, err),
              label + " second output buffer", err);
    }
    if (!key_buffer.view() || !query_buffer.view() || !gate_buffer.view() ||
        (shared_encoder_case && !second_gate_buffer.view())) {
        return false;
    }
    std::memcpy(key_buffer.payload().data(), key.data(), input_bytes);
    std::memcpy(query_buffer.payload().data(), query.data(), input_bytes);
    check(gate_buffer.poison_floats_against(
              expected, twin::Comparison::units, err),
          label + " output poison", err);
    if (shared_encoder_case) {
        check(second_gate_buffer.poison_floats_against(
                  expected, twin::Comparison::units, err),
              label + " second output poison", err);
    }
    key_buffer.capture_before_dispatch();
    query_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    if (shared_encoder_case) second_gate_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder_case
        ? Encoder::encode(
              context, *command, "PLE gate shared segment",
              [&](Encoder& segment, std::string& encode_error) {
                  if (!encode_ple_gate(
                          context, segment, rows,
                          {key_buffer.view(), query_buffer.view(),
                           gate_buffer.view()},
                          encode_error)) {
                      return false;
                  }
                  return encode_ple_gate(
                      context, segment, rows,
                      {key_buffer.view(), query_buffer.view(),
                       second_gate_buffer.view()},
                      encode_error);
              },
              err)
        : encode_ple_gate(
              context, *command, rows,
              {key_buffer.view(), query_buffer.view(), gate_buffer.view()},
              err);
    check(encoded,
          label + (shared_encoder_case
                       ? " production PLE gate shared-segment encode"
                       : " production PLE gate encode"),
          err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(gate_buffer.payload().data()),
        output_count);
    const auto mismatch = twin::compare_floats(
        actual, expected,
        {twin::Comparison::units, kGateToleranceUnits, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    double max_units = 0.0;
    for (std::size_t i = 0u; i < output_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(expected[i].value)) {
            ++exact_count;
        }
        const double bound =
            std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units,
            std::fabs(static_cast<double>(actual[i]) - expected[i].value) /
                bound);
    }
    check(!mismatch, label + " U(64) ordered gate result",
          "max observed U=" + std::to_string(max_units) +
              ", exact bits=" + std::to_string(exact_count) + "/" +
              std::to_string(output_count));
    check(!key_buffer.first_unwritten_change({}),
          label + " key remains read-only");
    check(!query_buffer.first_unwritten_change({}),
          label + " query remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, output_bytes}}};
    const auto changed = gate_buffer.first_unwritten_change(written);
    check(!changed, label + " output guards remain unchanged",
          changed ? changed->reason : "");
    if (shared_encoder_case) {
        const auto second_actual = std::span<const float>(
            reinterpret_cast<const float*>(
                second_gate_buffer.payload().data()),
            output_count);
        const auto second_mismatch = twin::compare_floats(
            second_actual, expected,
            {twin::Comparison::units, kGateToleranceUnits, 0.0, 0u});
        check(!second_mismatch,
              label + " shared-segment second output matches oracle",
              second_mismatch ? second_mismatch->reason : "");
        check(std::equal(actual.begin(), actual.end(),
                         second_actual.begin()),
              label + " repeated PLE gate dispatch matches in one segment");
        const auto second_changed =
            second_gate_buffer.first_unwritten_change(written);
        check(!second_changed,
              label + " second output guards remain unchanged",
              second_changed ? second_changed->reason : "");
    }
    output_snapshot.assign(gate_buffer.payload().begin(),
                           gate_buffer.payload().begin() + output_bytes);
    return !mismatch;
}

bool test_gate_encoder_validation(Context& context,
                                  const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t input_row_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    constexpr std::size_t gate_row_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    const std::size_t input_bytes = rows * input_row_bytes;
    const std::size_t gate_bytes = rows * gate_row_bytes;

    std::string error;
    auto rejected_command = context.make_command_buffer(
        "PLE gate rejected-input validation", error);
    if (!rejected_command) {
        check(false, "PLE gate rejection-validation command buffer", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t row_count,
                            const PLEGateBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !encode_ple_gate(context, *rejected_command, row_count,
                             buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const PLEGateBuffers empty{};
    passed &= reject("PLE gate encoder rejects zero rows", 0u, empty,
                     "row count");
    passed &= reject("PLE gate encoder rejects rows above eight", 9u, empty,
                     "row count");

    BufferRef key = context.buffers().allocate(
        "PLE gate validation key", input_bytes, error);
    BufferRef query = context.buffers().allocate(
        "PLE gate validation query", input_bytes, error);
    BufferRef gate = context.buffers().allocate(
        "PLE gate validation output", gate_bytes, error);
    BufferRef alias_backing = context.buffers().allocate(
        "PLE gate validation alias backing", input_bytes + sizeof(float), error);
    if (!key || !query || !gate || !alias_backing) {
        check(false, "PLE gate encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const PLEGateBuffers valid{key, query, gate};

    PLEGateBuffers short_key = valid;
    short_key.key.length = input_row_bytes;
    passed &= reject("PLE gate encoder rejects one-row key for T=2",
                     rows, short_key, "undersized");
    PLEGateBuffers short_query = valid;
    short_query.query.length = input_row_bytes;
    passed &= reject("PLE gate encoder rejects one-row query for T=2",
                     rows, short_query, "undersized");
    PLEGateBuffers short_gate = valid;
    short_gate.gate.length = gate_row_bytes;
    passed &= reject("PLE gate encoder rejects one-row gate for T=2",
                     rows, short_gate, "undersized");

    PLEGateBuffers misaligned_key = valid;
    misaligned_key.key = key.slice(1u, key.length - 1u);
    passed &= reject("PLE gate encoder rejects misaligned key",
                     rows, misaligned_key, "misaligned");
    PLEGateBuffers misaligned_query = valid;
    misaligned_query.query = query.slice(1u, query.length - 1u);
    passed &= reject("PLE gate encoder rejects misaligned query",
                     rows, misaligned_query, "misaligned");
    PLEGateBuffers misaligned_gate = valid;
    misaligned_gate.gate = gate.slice(1u, gate.length - 1u);
    passed &= reject("PLE gate encoder rejects misaligned output",
                     rows, misaligned_gate, "misaligned");

    PLEGateBuffers out_of_range = valid;
    out_of_range.gate = gate.slice(gate.length, sizeof(float));
    passed &= reject("PLE gate encoder rejects output past allocation",
                     rows, out_of_range, "not owned by this Context");

    BufferRef exact_alias_input = alias_backing.slice(0u, input_bytes);
    BufferRef exact_alias_output = alias_backing.slice(0u, gate_bytes);
    PLEGateBuffers exact_key_alias{
        exact_alias_input, query, exact_alias_output};
    passed &= reject("PLE gate encoder rejects exact output/key overlap",
                     rows, exact_key_alias, "must not overlap");
    BufferRef missing_metadata_key = exact_alias_input;
    missing_metadata_key.contents = nullptr;
    missing_metadata_key.gpu_address = 0u;
    PLEGateBuffers missing_metadata_key_alias{
        missing_metadata_key, query, exact_alias_output};
    passed &= reject(
        "PLE gate rejects forged CPU/GPU metadata",
        rows, missing_metadata_key_alias, "not owned by this Context");
    PLEGateBuffers exact_query_alias{
        key, exact_alias_input, exact_alias_output};
    passed &= reject("PLE gate encoder rejects exact output/query overlap",
                     rows, exact_query_alias, "must not overlap");

    BufferRef partial_alias_input =
        alias_backing.slice(sizeof(float), input_bytes);
    PLEGateBuffers partial_key_alias{
        partial_alias_input, query, exact_alias_output};
    passed &= reject("PLE gate encoder rejects partial output/key overlap",
                     rows, partial_key_alias, "must not overlap");
    PLEGateBuffers partial_query_alias{
        key, partial_alias_input, exact_alias_output};
    passed &= reject("PLE gate encoder rejects partial output/query overlap",
                     rows, partial_query_alias, "must not overlap");
    rejected_command->discard();

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    if (!foreign_context) {
        check(false, "PLE gate foreign Context", error);
        return false;
    }
    BufferRef foreign_key = foreign_context->buffers().allocate(
        "PLE gate foreign-context key", input_bytes, error);
    if (!foreign_key) {
        check(false, "PLE gate foreign-context buffer", error);
        return false;
    }
    PLEGateBuffers foreign_buffers = valid;
    foreign_buffers.key = foreign_key;
    auto foreign_buffer_command = context.make_command_buffer(
        "PLE gate foreign-buffer validation", error);
    if (!foreign_buffer_command) {
        check(false, "PLE gate foreign-buffer command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_buffer =
        !encode_ple_gate(context, *foreign_buffer_command, rows,
                         foreign_buffers, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejected_foreign_buffer,
          "PLE gate encoder rejects foreign-context buffer", error);
    foreign_buffer_command->discard();
    passed &= rejected_foreign_buffer;

    auto foreign_command = foreign_context->make_command_buffer(
        "PLE gate foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "PLE gate foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_command =
        !encode_ple_gate(context, *foreign_command, rows, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "PLE gate encoder rejects foreign command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "PLE gate submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "PLE gate command-buffer submission", error);
        return false;
    }
    error.clear();
    const bool rejected_submitted =
        !encode_ple_gate(context, *submitted, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "PLE gate encoder rejects submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed, "PLE gate validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "PLE gate enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "PLE gate command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    error.clear();
    const bool rejected_enqueued =
        !encode_ple_gate(context, *enqueued, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "PLE gate encoder rejects enqueued command buffer", error);
    const bool enqueued_completed =
        submit_and_wait(enqueued, "PLE gate enqueued validation", error);
    check(enqueued_completed,
          "PLE gate enqueued command buffer drains", error);
    passed &= rejected_enqueued && enqueued_completed;

    BufferRef shared_input = context.buffers().allocate(
        "PLE gate overlapping read-only inputs", input_bytes, error);
    BufferRef disjoint_gate = context.buffers().allocate(
        "PLE gate overlapping-input output", gate_bytes, error);
    if (!shared_input || !disjoint_gate) {
        check(false, "PLE gate overlapping-input test buffers", error);
        return false;
    }
    std::memset(shared_input.contents, 0, shared_input.length);
    auto alias_command = context.make_command_buffer(
        "PLE gate overlapping read-only inputs", error);
    if (!alias_command) {
        check(false, "PLE gate overlapping-input command buffer", error);
        return false;
    }
    const bool accepted_read_only_overlap = encode_ple_gate(
        context, *alias_command, rows,
        {shared_input, shared_input, disjoint_gate}, error);
    check(accepted_read_only_overlap,
          "PLE gate permits overlapping read-only inputs", error);
    passed &= accepted_read_only_overlap;
    if (!accepted_read_only_overlap) {
        alias_command->discard();
        return false;
    }
    const bool alias_dispatched =
        submit_and_wait(alias_command, "PLE gate overlapping-input test",
                        error);
    check(alias_dispatched,
          "PLE gate overlapping-input command completes", error);
    passed &= alias_dispatched;
    return passed;
}

twin::tf conv_residual_oracle(std::array<float, 4> values,
                              std::array<float, 4> weights,
                              float gated,
                              float residual,
                              bool fused_accumulation = false,
                              std::array<std::uint32_t, 4> order = {0u, 1u, 2u, 3u},
                              bool regroup_residual = false) {
    twin::tf sum{};
    for (std::uint32_t position = 0u; position < order.size(); ++position) {
        const std::uint32_t tap = order[position];
        const twin::tf value{values[tap]};
        const twin::tf weight{weights[tap]};
        if (position == 0u) {
            sum = value * weight;
        } else if (fused_accumulation) {
            sum = twin::fma(value, weight, sum);
        } else {
            sum = sum + value * weight;
        }
    }
    const twin::tf activation =
        sum / (twin::tf{1.0f} + twin::accurate_exp(-sum));
    if (regroup_residual) {
        return (twin::tf{residual} + twin::tf{gated}) + activation;
    }
    return twin::tf{residual} + (twin::tf{gated} + activation);
}

void weighted_rms_norm_oracle(std::span<const float> input,
                              std::span<const float> gamma,
                              std::uint32_t rows, std::uint32_t n_cols,
                              std::uint32_t gamma_period, float epsilon,
                              std::vector<twin::tf>& output) {
    constexpr std::uint32_t threads = 1024u;
    output.resize(static_cast<std::size_t>(rows) * n_cols);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        std::array<std::array<twin::tf, 32>, 32> warp_lanes{};
        const std::size_t row_offset =
            static_cast<std::size_t>(row) * n_cols;
        for (std::uint32_t tid = 0u; tid < threads; ++tid) {
            twin::tf partial{};
            for (std::uint32_t col = tid; col < n_cols; col += threads) {
                const twin::tf value{input[row_offset + col]};
                partial = twin::fma(value, value, partial);
            }
            warp_lanes[tid / 32u][tid % 32u] = partial;
        }
        std::array<twin::tf, 32> warp_sums{};
        for (std::uint32_t warp = 0u; warp < warp_sums.size(); ++warp) {
            warp_sums[warp] = butterfly_sum(warp_lanes[warp]);
        }
        const twin::tf sum = butterfly_sum(warp_sums);
        const twin::tf mean =
            sum / twin::tf{static_cast<float>(n_cols)};
        const twin::tf variance = mean + twin::tf{epsilon};
        const twin::tf scale{static_cast<float>(
            1.0 / std::sqrt(static_cast<double>(variance.value)))};
        const std::size_t gamma_offset =
            static_cast<std::size_t>(row % gamma_period) * n_cols;
        for (std::uint32_t col = 0u; col < n_cols; ++col) {
            output[row_offset + col] =
                (scale * twin::tf{input[row_offset + col]}) *
                twin::tf{gamma[gamma_offset + col]};
        }
    }
}

bool run_shared_ple_segment(Context& context,
                            const twin::Options& options,
                            std::uint32_t token_rows) {
    const std::uint32_t stream_rows = token_rows * HC;
    constexpr float epsilon = 1.0e-6f;
    const std::string label =
        "PLE documented compute segment T=" + std::to_string(token_rows) +
        " fixture";
    const std::size_t activation_count =
        static_cast<std::size_t>(token_rows) * HC_DIM;
    const std::size_t activation_bytes = activation_count * sizeof(float);
    const std::size_t gamma_bytes =
        static_cast<std::size_t>(HC) * N_EMBD * sizeof(float);
    const std::size_t value_count =
        static_cast<std::size_t>(token_rows) * N_EMBD;
    const std::size_t value_bytes = value_count * sizeof(float);
    const std::size_t gate_count =
        static_cast<std::size_t>(token_rows) * HC;
    const std::size_t gate_bytes =
        gate_count * sizeof(float);
    const std::size_t history_count =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS;
    const std::size_t history_bytes = history_count * sizeof(float);
    const std::size_t conv_count = static_cast<std::size_t>(HC_DIM) * 4u;
    const std::size_t conv_bytes = conv_count * sizeof(std::uint16_t);

    std::vector<float> key(activation_count);
    std::vector<float> residual(activation_count);
    std::vector<float> value(value_count);
    std::vector<float> key_gamma(HC_DIM);
    std::vector<float> query_gamma(HC_DIM);
    std::vector<float> conv_gamma(HC_DIM);
    std::vector<float> initial_history(history_count);
    std::vector<std::uint16_t> conv1d(conv_count);
    constexpr twin::InputSet set = twin::InputSet::fixture;
    std::mt19937_64 key_rng(options.seed ^ 0x504c455345474b59ull);
    std::mt19937_64 residual_rng(options.seed ^ 0x504c455345475253ull);
    std::mt19937_64 value_rng(options.seed ^ 0x504c455345475641ull);
    std::mt19937_64 history_rng(options.seed ^ 0x504c455345474849ull);
    std::mt19937_64 weight_rng(options.seed ^ 0x504c455345475745ull);
    for (std::uint32_t stream = 0u; stream < stream_rows; ++stream) {
        for (std::uint32_t col = 0u; col < N_EMBD; ++col) {
            const std::size_t index =
                static_cast<std::size_t>(stream) * N_EMBD + col;
            key[index] = value_input(set, stream, col, key_rng);
            residual[index] = value_input(set, stream + 2u, col, residual_rng);
            const std::uint32_t gamma_stream = stream % HC;
            const std::size_t gamma_index =
                static_cast<std::size_t>(gamma_stream) * N_EMBD + col;
            const std::uint32_t gamma_base =
                (gamma_stream * 73u + col * 17u) % 257u;
            key_gamma[gamma_index] =
                0.5f + static_cast<float>(gamma_base) * (1.0f / 512.0f);
            query_gamma[gamma_index] =
                0.5f + static_cast<float>((gamma_base + 61u) % 257u) *
                           (1.0f / 512.0f);
            conv_gamma[gamma_index] =
                0.5f + static_cast<float>((gamma_base + 139u) % 257u) *
                           (1.0f / 512.0f);
        }
    }
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        for (std::uint32_t col = 0u; col < N_EMBD; ++col) {
            value[static_cast<std::size_t>(row) * N_EMBD + col] =
                value_input(set, row, col, value_rng);
        }
    }
    for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
        const std::size_t history_offset =
            static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
        for (std::uint32_t row = 0u; row < PLE_HISTORY_ROWS; ++row) {
            initial_history[history_offset + row] =
                value_input(set, channel, row, history_rng);
        }
        for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
            const float weight = value_input(set, channel, tap, weight_rng);
            conv1d[static_cast<std::size_t>(channel) * 4u + tap] =
                strata::kernels::f16_from_f32(weight);
        }
    }

    std::vector<twin::tf> key_expected;
    std::vector<twin::tf> query_expected;
    std::vector<twin::tf> gate_expected(gate_count);
    std::vector<twin::tf> gated_expected(activation_count);
    std::vector<twin::tf> ple_norm_expected;
    std::vector<twin::tf> residual_expected(activation_count);
    std::vector<twin::tf> history_expected(history_count);
    weighted_rms_norm_oracle(key, key_gamma, stream_rows, N_EMBD, HC,
                             epsilon, key_expected);
    weighted_rms_norm_oracle(residual, query_gamma, stream_rows, N_EMBD, HC,
                             epsilon, query_expected);

    std::string error;
    twin::GuardedBuffer key_buffer;
    twin::GuardedBuffer query_buffer;
    twin::GuardedBuffer ple_norm_buffer;
    twin::GuardedBuffer reused_query_snapshot_buffer;
    twin::GuardedBuffer reused_key_buffer;
    twin::GuardedBuffer reused_query_ple_norm_buffer;
    twin::GuardedBuffer value_buffer;
    twin::GuardedBuffer key_gamma_buffer;
    twin::GuardedBuffer query_gamma_buffer;
    twin::GuardedBuffer conv_gamma_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer gated_buffer;
    twin::GuardedBuffer reused_gate_buffer;
    twin::GuardedBuffer reused_gated_buffer;
    twin::GuardedBuffer history_buffer;
    twin::GuardedBuffer reused_history_buffer;
    twin::GuardedBuffer conv_buffer;
    twin::GuardedBuffer residual_buffer;
    twin::GuardedBuffer reused_residual_buffer;
    const auto allocate = [&](twin::GuardedBuffer& buffer,
                              const std::string& name,
                              std::size_t bytes) {
        const bool allocated = buffer.allocate(context, label + " " + name,
                                               bytes, error);
        check(allocated, label + " " + name + " buffer", error);
        return allocated;
    };
    bool ready = true;
    ready &= allocate(key_buffer, "key", activation_bytes);
    ready &= allocate(query_buffer, "query", activation_bytes);
    ready &= allocate(ple_norm_buffer, "ple_norm", activation_bytes);
    ready &= allocate(reused_query_snapshot_buffer,
                      "reused-layout query snapshot", activation_bytes);
    ready &= allocate(reused_key_buffer, "reused-layout key",
                      activation_bytes);
    ready &= allocate(reused_query_ple_norm_buffer,
                      "reused-layout query/ple_norm", activation_bytes);
    ready &= allocate(value_buffer, "value", value_bytes);
    ready &= allocate(key_gamma_buffer, "key gamma", gamma_bytes);
    ready &= allocate(query_gamma_buffer, "query gamma", gamma_bytes);
    ready &= allocate(conv_gamma_buffer, "conv gamma", gamma_bytes);
    ready &= allocate(gate_buffer, "gate", gate_bytes);
    ready &= allocate(gated_buffer, "gated", activation_bytes);
    ready &= allocate(reused_gate_buffer, "reused-layout gate", gate_bytes);
    ready &= allocate(reused_gated_buffer, "reused-layout gated",
                      activation_bytes);
    ready &= allocate(history_buffer, "history", history_bytes);
    ready &= allocate(reused_history_buffer, "reused-layout history",
                      history_bytes);
    ready &= allocate(conv_buffer, "conv1d", conv_bytes);
    ready &= allocate(residual_buffer, "residual", activation_bytes);
    ready &= allocate(reused_residual_buffer, "reused-layout residual",
                      activation_bytes);
    if (!ready || !key_buffer.view() || !query_buffer.view() ||
        !ple_norm_buffer.view() || !reused_query_snapshot_buffer.view() ||
        !reused_key_buffer.view() ||
        !reused_query_ple_norm_buffer.view() ||
        !value_buffer.view() || !key_gamma_buffer.view() ||
        !query_gamma_buffer.view() || !conv_gamma_buffer.view() ||
        !gate_buffer.view() || !gated_buffer.view() ||
        !reused_gate_buffer.view() || !reused_gated_buffer.view() ||
        !history_buffer.view() || !reused_history_buffer.view() ||
        !conv_buffer.view() || !residual_buffer.view() ||
        !reused_residual_buffer.view()) {
        return false;
    }

    std::memcpy(key_buffer.payload().data(), key.data(), activation_bytes);
    std::memcpy(reused_key_buffer.payload().data(), key.data(),
                activation_bytes);
    std::memcpy(value_buffer.payload().data(), value.data(), value_bytes);
    std::memcpy(key_gamma_buffer.payload().data(), key_gamma.data(),
                gamma_bytes);
    std::memcpy(query_gamma_buffer.payload().data(), query_gamma.data(),
                gamma_bytes);
    std::memcpy(conv_gamma_buffer.payload().data(), conv_gamma.data(),
                gamma_bytes);
    std::memcpy(history_buffer.payload().data(), initial_history.data(),
                history_bytes);
    std::memcpy(reused_history_buffer.payload().data(),
                initial_history.data(), history_bytes);
    std::memcpy(conv_buffer.payload().data(), conv1d.data(), conv_bytes);
    std::memcpy(residual_buffer.payload().data(), residual.data(),
                activation_bytes);
    std::memcpy(reused_residual_buffer.payload().data(), residual.data(),
                activation_bytes);
    const std::vector<twin::tf> zero_gate(gate_count, twin::tf{0.0f});
    const std::vector<twin::tf> zero_activation(activation_count,
                                                twin::tf{0.0f});
    check(query_buffer.poison_floats_against(
              query_expected, twin::Comparison::units, error),
          label + " query poison", error);
    check(ple_norm_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " ple_norm poison", error);
    check(reused_query_snapshot_buffer.poison_floats_against(
              query_expected, twin::Comparison::units, error),
          label + " reused-layout query snapshot poison", error);
    check(gate_buffer.poison_floats_against(
              zero_gate, twin::Comparison::bits, error),
          label + " gate poison", error);
    check(gated_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " gated poison", error);
    check(reused_query_ple_norm_buffer.poison_floats_against(
              query_expected, twin::Comparison::units, error),
          label + " reused-layout query/ple_norm poison", error);
    check(reused_gate_buffer.poison_floats_against(
              zero_gate, twin::Comparison::bits, error),
          label + " reused-layout gate poison", error);
    check(reused_gated_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " reused-layout gated poison", error);
    key_buffer.capture_before_dispatch();
    query_buffer.capture_before_dispatch();
    ple_norm_buffer.capture_before_dispatch();
    reused_query_snapshot_buffer.capture_before_dispatch();
    reused_key_buffer.capture_before_dispatch();
    reused_query_ple_norm_buffer.capture_before_dispatch();
    value_buffer.capture_before_dispatch();
    key_gamma_buffer.capture_before_dispatch();
    query_gamma_buffer.capture_before_dispatch();
    conv_gamma_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    gated_buffer.capture_before_dispatch();
    reused_gate_buffer.capture_before_dispatch();
    reused_gated_buffer.capture_before_dispatch();
    history_buffer.capture_before_dispatch();
    reused_history_buffer.capture_before_dispatch();
    conv_buffer.capture_before_dispatch();
    residual_buffer.capture_before_dispatch();
    reused_residual_buffer.capture_before_dispatch();

    const PLEPostProjectionBuffers reference_buffers{
        key_buffer.view(),
        residual_buffer.view(),
        query_buffer.view(),
        ple_norm_buffer.view(),
        value_buffer.view(),
        gate_buffer.view(),
        gated_buffer.view(),
        history_buffer.view(),
        read_only(context, conv_buffer.view(), label + " conv weights view"),
        read_only(context, key_gamma_buffer.view(), label + " key gamma view"),
        read_only(context, query_gamma_buffer.view(),
                  label + " query gamma view"),
        read_only(context, conv_gamma_buffer.view(),
                  label + " conv gamma view"),
    };
    const auto snapshot_payload = [](twin::GuardedBuffer& buffer) {
        return std::vector<std::uint8_t>(buffer.payload().begin(),
                                         buffer.payload().end());
    };
    const auto key_before_rejected_segment = snapshot_payload(key_buffer);
    const auto query_before_rejected_segment = snapshot_payload(query_buffer);
    const auto norm_before_rejected_segment = snapshot_payload(ple_norm_buffer);
    const auto gate_before_rejected_segment = snapshot_payload(gate_buffer);
    const auto gated_before_rejected_segment = snapshot_payload(gated_buffer);
    const auto history_before_rejected_segment =
        snapshot_payload(history_buffer);
    const auto residual_before_rejected_segment =
        snapshot_payload(residual_buffer);
    PLEPostProjectionBuffers short_history_buffers = reference_buffers;
    short_history_buffers.history =
        history_buffer.view().slice(0u, history_bytes - sizeof(float));
    auto rejected_segment =
        context.make_command_buffer(label + " late failure discard", error);
    if (!rejected_segment) {
        check(false, label + " late failure command buffer", error);
        return false;
    }
    const bool encoded_rejected_segment = Encoder::encode(
        context, *rejected_segment, "PLE post-projection late failure",
        [&](Encoder& segment, std::string& encode_error) {
            return encode_ple_post_projection(
                context, segment, token_rows,
                PLEHistoryUpdate::advance_rows, short_history_buffers,
                encode_error);
        },
        error);
    const bool late_failure_discarded =
        !encoded_rejected_segment &&
        rejected_segment->unsafe_native_handle() == nullptr &&
        error.find("history") != std::string::npos;
    check(late_failure_discarded,
          label + " late validation failure discards the partial segment",
          error);
    const auto unchanged = [](const std::vector<std::uint8_t>& before,
                              twin::GuardedBuffer& buffer) {
        const auto after = buffer.payload();
        return before.size() == after.size() &&
               std::equal(before.begin(), before.end(), after.begin());
    };
    const bool rejected_segment_left_outputs_unchanged =
        unchanged(key_before_rejected_segment, key_buffer) &&
        unchanged(query_before_rejected_segment, query_buffer) &&
        unchanged(norm_before_rejected_segment, ple_norm_buffer) &&
        unchanged(gate_before_rejected_segment, gate_buffer) &&
        unchanged(gated_before_rejected_segment, gated_buffer) &&
        unchanged(history_before_rejected_segment, history_buffer) &&
        unchanged(residual_before_rejected_segment, residual_buffer);
    check(rejected_segment_left_outputs_unchanged,
          label + " discarded partial segment leaves every output unchanged");

    PLEPostProjectionBuffers partial_alias_buffers = reference_buffers;
    partial_alias_buffers.ple_norm =
        query_buffer.view().slice(sizeof(float), activation_bytes -
                                                       sizeof(float));
    auto rejected_alias_command =
        context.make_command_buffer(label + " partial scratch alias", error);
    if (!rejected_alias_command) {
        check(false, label + " partial scratch alias command buffer", error);
        return false;
    }
    const bool partial_alias_accepted = encode_ple_post_projection(
        context, *rejected_alias_command, token_rows,
        PLEHistoryUpdate::advance_rows, partial_alias_buffers, error);
    const bool partial_alias_rejected =
        !partial_alias_accepted &&
        rejected_alias_command->unsafe_native_handle() == nullptr &&
        error.find("query and ple_norm") != std::string::npos;
    check(partial_alias_rejected,
          label + " rejects partial query/ple_norm scratch overlap", error);

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "PLE weighted-normalization and history segment",
        [&](Encoder& segment, std::string& encode_error) {
            const auto encode_chain =
                [&](BufferRef chain_key, BufferRef chain_query,
                    BufferRef chain_query_snapshot,
                    BufferRef chain_norm, BufferRef chain_gate,
                    BufferRef chain_gated, BufferRef chain_history,
                    BufferRef chain_residual,
                    std::string& chain_error) {
                    if (!encode_weighted_rms_norm(
                            context, segment, stream_rows, N_EMBD, HC,
                            epsilon,
                            {chain_key, key_gamma_buffer.view(), chain_key},
                            chain_error)) {
                        return false;
                    }
                    if (!encode_weighted_rms_norm(
                            context, segment, stream_rows, N_EMBD, HC,
                            epsilon,
                            {chain_residual, query_gamma_buffer.view(),
                             chain_query},
                            chain_error)) {
                        return false;
                    }
                    if (chain_query_snapshot) {
                        const std::array snapshot_bindings{
                            EncoderBufferBinding{
                                chain_query, 0u, activation_bytes,
                                alignof(std::uint32_t)},
                            EncoderBufferBinding{
                                chain_query_snapshot, 1u, activation_bytes,
                                alignof(std::uint32_t)},
                        };
                        if (!segment.dispatch(
                                "strata_selftest_copy_u32",
                                {activation_count / 256u, 1u, 1u},
                                {256u, 1u, 1u},
                                snapshot_bindings, {}, {}, chain_error)) {
                            return false;
                        }
                    }
                    if (!encode_ple_gate(
                            context, segment, token_rows,
                            {chain_key, chain_query, chain_gate},
                            chain_error)) {
                        return false;
                    }
                    if (!encode_ple_broadcast(
                            context, segment, token_rows,
                            {value_buffer.view(), chain_gate, chain_gated},
                            chain_error)) {
                        return false;
                    }
                    if (!encode_weighted_rms_norm(
                            context, segment, stream_rows, N_EMBD, HC,
                            epsilon,
                            {chain_gated, conv_gamma_buffer.view(),
                             chain_norm},
                            chain_error)) {
                        return false;
                    }
                    if (!encode_ple_conv_residual(
                            context, segment, token_rows,
                            {read_only(context, chain_norm,
                                       label + " chain norm"),
                             read_only(context, chain_history,
                                       label + " chain history"),
                             read_only(context, conv_buffer.view(),
                                       label + " chain conv weights"),
                             read_only(context, chain_gated,
                                       label + " chain gated"),
                             chain_residual},
                            chain_error)) {
                        return false;
                    }
                    return encode_ple_history(
                        context, segment, token_rows,
                        {chain_history, chain_norm}, chain_error);
                };
            if (!encode_ple_post_projection(
                    context, segment, token_rows,
                    PLEHistoryUpdate::advance_rows, reference_buffers,
                    encode_error)) {
                return false;
            }
            return encode_chain(
                reused_key_buffer.view(),
                reused_query_ple_norm_buffer.view(),
                reused_query_snapshot_buffer.view(),
                reused_query_ple_norm_buffer.view(),
                reused_gate_buffer.view(), reused_gated_buffer.view(),
                reused_history_buffer.view(),
                reused_residual_buffer.view(), encode_error);
        },
        error);
    check(encoded, label + " shared-segment PLE chain encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto floats = [](twin::GuardedBuffer& buffer, std::size_t count) {
        return std::span<const float>(
            reinterpret_cast<const float*>(buffer.payload().data()), count);
    };
    const auto copy_floats = [&](twin::GuardedBuffer& buffer,
                                 std::size_t count) {
        const auto contents = floats(buffer, count);
        return std::vector<float>(contents.begin(), contents.end());
    };
    const auto manual_reused_key =
        copy_floats(reused_key_buffer, activation_count);
    const auto manual_reused_gate = copy_floats(reused_gate_buffer, gate_count);
    const auto manual_reused_gated =
        copy_floats(reused_gated_buffer, activation_count);
    const auto manual_reused_ple_norm =
        copy_floats(reused_query_ple_norm_buffer, activation_count);
    const auto manual_reused_residual =
        copy_floats(reused_residual_buffer, activation_count);
    const auto manual_reused_history =
        copy_floats(reused_history_buffer, history_count);

    std::memcpy(reused_key_buffer.payload().data(), key.data(),
                activation_bytes);
    std::memcpy(reused_residual_buffer.payload().data(), residual.data(),
                activation_bytes);
    std::memcpy(reused_history_buffer.payload().data(), initial_history.data(),
                history_bytes);
    const BufferRef reused_query_ple_norm =
        reused_query_ple_norm_buffer.view();
    const PLEPostProjectionBuffers alias_buffers{
        reused_key_buffer.view(),
        reused_residual_buffer.view(),
        reused_query_ple_norm,
        reused_query_ple_norm,
        value_buffer.view(),
        reused_gate_buffer.view(),
        reused_gated_buffer.view(),
        reused_history_buffer.view(),
        read_only(context, conv_buffer.view(), label + " alias conv weights"),
        read_only(context, key_gamma_buffer.view(), label + " alias key gamma"),
        read_only(context, query_gamma_buffer.view(),
                  label + " alias query gamma"),
        read_only(context, conv_gamma_buffer.view(),
                  label + " alias conv gamma"),
    };
    error.clear();
    check(reused_query_ple_norm_buffer.poison_floats_against(
              query_expected, twin::Comparison::units, error),
          label + " coordinator exact-alias scratch poison", error);
    check(reused_gate_buffer.poison_floats_against(
              zero_gate, twin::Comparison::bits, error),
          label + " coordinator exact-alias gate poison", error);
    check(reused_gated_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " coordinator exact-alias broadcast poison", error);
    auto composed_alias_command =
        context.make_command_buffer(label + " exact-alias coordinator", error);
    if (!composed_alias_command) {
        check(false, label + " exact-alias coordinator command buffer", error);
        return false;
    }
    const bool composed_alias_encoded = encode_ple_post_projection(
        context, *composed_alias_command, token_rows,
        PLEHistoryUpdate::advance_rows, alias_buffers, error);
    check(composed_alias_encoded,
          label + " coordinator accepts exact query/ple_norm alias", error);
    if (!composed_alias_encoded) return false;
    const bool composed_alias_dispatched =
        submit_and_wait(composed_alias_command, label + " exact-alias", error);
    check(composed_alias_dispatched,
          label + " exact-alias coordinator dispatch", error);
    if (!composed_alias_dispatched) return false;

    const auto actual_key = floats(key_buffer, activation_count);
    const auto actual_query = floats(query_buffer, activation_count);
    const auto actual_reused_query_snapshot =
        floats(reused_query_snapshot_buffer, activation_count);
    const auto actual_gate = floats(gate_buffer, gate_count);
    const auto actual_gated = floats(gated_buffer, activation_count);
    const auto actual_ple_norm = floats(ple_norm_buffer, activation_count);
    const auto actual_reused_key = floats(reused_key_buffer, activation_count);
    const auto actual_reused_gate = floats(reused_gate_buffer, gate_count);
    const auto actual_reused_gated =
        floats(reused_gated_buffer, activation_count);
    const auto actual_reused_ple_norm =
        floats(reused_query_ple_norm_buffer, activation_count);
    const auto actual_reused_residual =
        floats(reused_residual_buffer, activation_count);
    const auto actual_reused_history =
        floats(reused_history_buffer, history_count);
    const float gate_scale =
        1.0f / std::sqrt(static_cast<float>(N_EMBD));
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        for (std::uint32_t stream = 0u; stream < HC; ++stream) {
            const std::size_t offset =
                (static_cast<std::size_t>(row) * HC + stream) * N_EMBD;
            gate_expected[static_cast<std::size_t>(row) * HC + stream] =
                gate_oracle(actual_key.subspan(offset, N_EMBD),
                            actual_query.subspan(offset, N_EMBD), gate_scale);
        }
    }
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
            const std::uint32_t stream = channel / N_EMBD;
            const std::uint32_t col = channel % N_EMBD;
            const std::size_t index =
                static_cast<std::size_t>(row) * HC_DIM + channel;
            gated_expected[index] = twin::tf{
                value[static_cast<std::size_t>(row) * N_EMBD + col] *
                actual_gate[static_cast<std::size_t>(row) * HC + stream]};
        }
    }
    weighted_rms_norm_oracle(actual_gated, conv_gamma, stream_rows, N_EMBD,
                             HC, epsilon, ple_norm_expected);
    for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
        const std::size_t history_offset =
            static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
        std::array<float, 4> tap_weights{};
        for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
            tap_weights[tap] = strata::kernels::f32_from_f16(
                conv1d[static_cast<std::size_t>(channel) * 4u + tap]);
        }
        for (std::uint32_t row = 0u; row < token_rows; ++row) {
            std::array<float, 4> tap_values{};
            for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
                const std::int32_t norm_row =
                    static_cast<std::int32_t>(row) -
                    static_cast<std::int32_t>(PLE_HISTORY_ROWS) +
                    static_cast<std::int32_t>(3u * tap);
                tap_values[tap] =
                    norm_row >= 0
                        ? actual_ple_norm[
                              static_cast<std::size_t>(norm_row) * HC_DIM +
                              channel]
                        : initial_history[
                              history_offset +
                              static_cast<std::size_t>(
                                  static_cast<std::int32_t>(PLE_HISTORY_ROWS) +
                                  norm_row)];
            }
            const std::size_t index =
                static_cast<std::size_t>(row) * HC_DIM + channel;
            residual_expected[index] = conv_residual_oracle(
                tap_values, tap_weights, actual_gated[index], residual[index]);
        }
        for (std::uint32_t history_row = 0u;
             history_row < PLE_HISTORY_ROWS; ++history_row) {
            const std::int32_t norm_row =
                static_cast<std::int32_t>(history_row) +
                static_cast<std::int32_t>(token_rows) -
                static_cast<std::int32_t>(PLE_HISTORY_ROWS);
            history_expected[history_offset + history_row] =
                norm_row >= 0
                    ? twin::tf{actual_ple_norm[
                          static_cast<std::size_t>(norm_row) * HC_DIM +
                          channel]}
                    : twin::tf{initial_history[
                          history_offset + token_rows + history_row]};
        }
    }
    const auto compare = [&](const std::string& stage, std::uint32_t rows,
                             std::span<const float> actual,
                             const std::vector<twin::tf>& expected,
                             twin::Comparison comparison, double units) {
        const auto mismatch = twin::compare_floats(
            actual, expected, {comparison, units, 0.0, 0u});
        if (mismatch) {
            twin::report_failure((label + " " + stage).c_str(), options,
                                 rows, *mismatch);
        }
        check(!mismatch, label + " " + stage + " oracle");
        return !mismatch;
    };
    bool passed = true;
    passed &= compare("key weighted norm", stream_rows,
                      floats(key_buffer, activation_count),
                      key_expected, twin::Comparison::units, 64.0);
    passed &= compare("query weighted norm", stream_rows, actual_query,
                      query_expected,
                      twin::Comparison::units, 64.0);
    passed &= compare("reused-layout query weighted norm snapshot",
                      stream_rows, actual_reused_query_snapshot,
                      query_expected, twin::Comparison::units, 64.0);
    passed &= compare("gate reduction", token_rows,
                      floats(gate_buffer, gate_count),
                      gate_expected, twin::Comparison::units,
                      kGateToleranceUnits);
    passed &= compare("value broadcast", token_rows,
                      floats(gated_buffer, activation_count),
                      gated_expected, twin::Comparison::bits, 0.0);
    passed &= compare("convolution weighted norm", stream_rows,
                      actual_ple_norm,
                      ple_norm_expected, twin::Comparison::units, 64.0);
    passed &= compare("in-place residual update", token_rows,
                      floats(residual_buffer, activation_count), residual_expected,
                      twin::Comparison::units, kConvToleranceUnits);
    passed &= compare("history advancement", PLE_HISTORY_ROWS,
                      floats(history_buffer, history_count), history_expected,
                      twin::Comparison::bits, 0.0);
    const auto compare_layout_bits = [&](const std::string& stage,
                                         std::span<const float> reference,
                                         std::span<const float> reused) {
        const bool equal =
            reference.size() == reused.size() &&
            std::memcmp(reference.data(), reused.data(),
                        reference.size_bytes()) == 0;
        check(equal, label + " query/ple_norm-reuse path matches " + stage +
                         " bitwise");
        return equal;
    };
    passed &= compare_layout_bits("instrumented reuse key",
                                  floats(key_buffer, activation_count),
                                  manual_reused_key);
    passed &= compare_layout_bits("coordinator exact-alias key",
                                  floats(key_buffer, activation_count),
                                  actual_reused_key);
    passed &= compare_layout_bits("instrumented reuse gate",
                                  floats(gate_buffer, gate_count),
                                  manual_reused_gate);
    passed &= compare_layout_bits("coordinator exact-alias gate",
                                  floats(gate_buffer, gate_count),
                                  actual_reused_gate);
    passed &= compare_layout_bits("instrumented reuse broadcast",
                                  floats(gated_buffer, activation_count),
                                  manual_reused_gated);
    passed &= compare_layout_bits("coordinator exact-alias broadcast",
                                  floats(gated_buffer, activation_count),
                                  actual_reused_gated);
    passed &= compare_layout_bits("instrumented reuse ple_norm",
                                  actual_ple_norm,
                                  manual_reused_ple_norm);
    passed &= compare_layout_bits("coordinator exact-alias ple_norm",
                                  actual_ple_norm,
                                  actual_reused_ple_norm);
    passed &= compare_layout_bits("instrumented reuse residual",
                                  floats(residual_buffer, activation_count),
                                  manual_reused_residual);
    passed &= compare_layout_bits("coordinator exact-alias residual",
                                  floats(residual_buffer, activation_count),
                                  actual_reused_residual);
    passed &= compare_layout_bits("instrumented reuse history",
                                  floats(history_buffer, history_count),
                                  manual_reused_history);
    passed &= compare_layout_bits("coordinator exact-alias history",
                                  floats(history_buffer, history_count),
                                  actual_reused_history);
    passed &= compare_layout_bits("instrumented query norm snapshot",
                                  actual_query,
                                  actual_reused_query_snapshot);

    const auto invalid_policy_command = context.make_command_buffer(
        label + " invalid history policy", error);
    if (!invalid_policy_command) {
        check(false, label + " invalid history policy command buffer", error);
        return false;
    }
    const bool invalid_policy_accepted = encode_ple_post_projection(
        context, *invalid_policy_command, token_rows,
        static_cast<PLEHistoryUpdate>(0xffu), alias_buffers, error);
    const bool invalid_policy_rejected =
        !invalid_policy_accepted &&
        invalid_policy_command->unsafe_native_handle() == nullptr &&
        error.find("history update policy") != std::string::npos;
    check(invalid_policy_rejected,
          label + " rejects an invalid history update policy", error);

    const std::uint32_t accepted_rows =
        token_rows == 1u ? 1u : token_rows - 1u;
    std::memcpy(reused_key_buffer.payload().data(), key.data(),
                activation_bytes);
    std::memcpy(reused_residual_buffer.payload().data(), residual.data(),
                activation_bytes);
    std::memcpy(reused_history_buffer.payload().data(),
                initial_history.data(), history_bytes);
    check(reused_query_ple_norm_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " deferred path norm poison", error);
    check(reused_gate_buffer.poison_floats_against(
              zero_gate, twin::Comparison::bits, error),
          label + " deferred path gate poison", error);
    check(reused_gated_buffer.poison_floats_against(
              zero_activation, twin::Comparison::bits, error),
          label + " deferred path broadcast poison", error);
    auto deferred_command =
        context.make_command_buffer(label + " deferred history", error);
    if (!deferred_command) {
        check(false, label + " deferred history command buffer", error);
        return false;
    }
    const bool deferred_encoded = encode_ple_post_projection(
        context, *deferred_command, token_rows, PLEHistoryUpdate::defer,
        alias_buffers, error);
    check(deferred_encoded, label + " deferred history encode", error);
    if (!deferred_encoded) return false;
    const bool deferred_dispatched =
        submit_and_wait(deferred_command, label + " deferred history", error);
    check(deferred_dispatched, label + " deferred history dispatch", error);
    if (!deferred_dispatched) return false;
    const bool deferred_history_unchanged =
        std::memcmp(reused_history_buffer.payload().data(),
                    initial_history.data(), history_bytes) == 0;
    check(deferred_history_unchanged,
          label + " speculative PLE stage leaves committed history unchanged");
    passed &= deferred_history_unchanged;
    passed &= compare_layout_bits(
        "deferred path key", floats(key_buffer, activation_count),
        floats(reused_key_buffer, activation_count));
    passed &= compare_layout_bits(
        "deferred path gate", floats(gate_buffer, gate_count),
        floats(reused_gate_buffer, gate_count));
    passed &= compare_layout_bits(
        "deferred path broadcast", floats(gated_buffer, activation_count),
        floats(reused_gated_buffer, activation_count));
    passed &= compare_layout_bits(
        "deferred path ple_norm",
        floats(ple_norm_buffer, activation_count),
        floats(reused_query_ple_norm_buffer, activation_count));
    passed &= compare_layout_bits(
        "deferred path residual",
        floats(residual_buffer, activation_count),
        floats(reused_residual_buffer, activation_count));

    auto commit_history_command =
        context.make_command_buffer(label + " accepted-prefix history",
                                    error);
    if (!commit_history_command) {
        check(false, label + " accepted-prefix history command buffer",
              error);
        return false;
    }
    const bool commit_history_encoded = encode_ple_history(
        context, *commit_history_command, accepted_rows,
        {reused_history_buffer.view(),
         reused_query_ple_norm_buffer.view()},
        error);
    check(commit_history_encoded,
          label + " accepted-prefix history encode", error);
    if (!commit_history_encoded) return false;
    const bool commit_history_dispatched = submit_and_wait(
        commit_history_command, label + " accepted-prefix history", error);
    check(commit_history_dispatched,
          label + " accepted-prefix history dispatch", error);
    if (!commit_history_dispatched) return false;

    std::vector<twin::tf> accepted_history_expected(history_count);
    for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
        const std::size_t history_offset =
            static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
        for (std::uint32_t history_row = 0u;
             history_row < PLE_HISTORY_ROWS; ++history_row) {
            const std::int32_t norm_row =
                static_cast<std::int32_t>(history_row) +
                static_cast<std::int32_t>(accepted_rows) -
                static_cast<std::int32_t>(PLE_HISTORY_ROWS);
            accepted_history_expected[history_offset + history_row] =
                norm_row >= 0
                    ? twin::tf{manual_reused_ple_norm[
                          static_cast<std::size_t>(norm_row) * HC_DIM +
                          channel]}
                    : twin::tf{initial_history[
                          history_offset + accepted_rows + history_row]};
        }
    }
    passed &= compare(
        "accepted-prefix history advancement n_keep=" +
            std::to_string(accepted_rows),
        PLE_HISTORY_ROWS, floats(reused_history_buffer, history_count),
        accepted_history_expected, twin::Comparison::bits, 0.0);

    check(!value_buffer.first_unwritten_change({}),
          label + " value remains read-only");
    check(!key_gamma_buffer.first_unwritten_change({}),
          label + " key gamma remains read-only");
    check(!query_gamma_buffer.first_unwritten_change({}),
          label + " query gamma remains read-only");
    check(!conv_gamma_buffer.first_unwritten_change({}),
          label + " conv gamma remains read-only");
    check(!conv_buffer.first_unwritten_change({}),
          label + " conv1d remains read-only");
    const auto check_guards = [&](twin::GuardedBuffer& buffer,
                                  std::size_t bytes,
                                  const std::string& name) {
        const std::array<twin::ByteRange, 1> written{{{0u, bytes}}};
        const auto changed = buffer.first_unwritten_change(written);
        check(!changed, label + " " + name + " guards remain unchanged",
              changed ? changed->reason : "");
        return !changed;
    };
    passed &= check_guards(key_buffer, activation_bytes, "key");
    passed &= check_guards(query_buffer, activation_bytes, "query");
    passed &= check_guards(ple_norm_buffer, activation_bytes, "ple_norm");
    passed &= check_guards(reused_query_snapshot_buffer, activation_bytes,
                           "reused-layout query snapshot");
    passed &= check_guards(reused_key_buffer, activation_bytes,
                           "reused-layout key");
    passed &= check_guards(reused_query_ple_norm_buffer, activation_bytes,
                           "reused-layout query/ple_norm");
    passed &= check_guards(gate_buffer, gate_bytes, "gate");
    passed &= check_guards(gated_buffer, activation_bytes, "gated");
    passed &= check_guards(history_buffer, history_bytes, "history");
    passed &= check_guards(residual_buffer, activation_bytes, "residual");
    passed &= check_guards(reused_gate_buffer, gate_bytes,
                           "reused-layout gate");
    passed &= check_guards(reused_gated_buffer, activation_bytes,
                           "reused-layout gated");
    passed &= check_guards(reused_history_buffer, history_bytes,
                           "reused-layout history");
    passed &= check_guards(reused_residual_buffer, activation_bytes,
                           "reused-layout residual");
    return passed;
}

void gr_post_reference(std::span<const float> residual,
                       std::span<const float> block_out,
                       std::span<const float> inject, std::uint32_t rows,
                       std::vector<twin::tf>& output) {
    output.resize(static_cast<std::size_t>(rows) * HC_DIM);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t index = 0u; index < HC_DIM; ++index) {
            const std::uint32_t stream = index / N_EMBD;
            const std::uint32_t column = index % N_EMBD;
            const twin::tf scaled_inject = twin::fma(
                twin::tf{0.25f},
                twin::tf{inject[static_cast<std::size_t>(row) * HC + stream]},
                twin::tf{});
            const twin::tf sigmoid =
                twin::tf{1.0f} /
                (twin::tf{1.0f} + twin::accurate_exp(-scaled_inject));
            const twin::tf weight =
                twin::fma(twin::tf{2.0f}, sigmoid, twin::tf{});
            const std::size_t output_index =
                static_cast<std::size_t>(row) * HC_DIM + index;
            const std::size_t block_index =
                static_cast<std::size_t>(row) * N_EMBD + column;
            output[output_index] = twin::fma(
                twin::tf{block_out[block_index]}, weight,
                twin::tf{residual[output_index]});
        }
    }
}

bool run_ple_block_projection_case(Context& context,
                                   const twin::Options& options,
                                   MatVecWeightFormat key_format,
                                   std::uint32_t rows,
                                   std::uint32_t compiled_columns,
                                   bool check_late_failure,
                                   bool prepend_gr_post = false) {
    constexpr std::uint32_t input_width = N_EMBD;
    constexpr std::uint32_t key_width = HC_DIM;
    constexpr float epsilon = 1.0e-6f;
    const bool quantized_key =
        key_format == MatVecWeightFormat::q2_0 ||
        key_format == MatVecWeightFormat::q8_0;
    const std::string format_name =
        key_format == MatVecWeightFormat::q2_0
            ? "Q2_0"
            : key_format == MatVecWeightFormat::q8_0 ? "Q8_0" : "BF16";
    const std::string label =
        "PLE full block " + format_name + " T=" + std::to_string(rows) +
        " NC=" + std::to_string(compiled_columns);
    const std::size_t active_input_count =
        static_cast<std::size_t>(rows) * input_width;
    const std::size_t compiled_input_count =
        static_cast<std::size_t>(compiled_columns) * input_width;
    const std::size_t key_count = static_cast<std::size_t>(rows) * key_width;
    const std::size_t value_count =
        static_cast<std::size_t>(rows) * input_width;
    const std::size_t post_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t gamma_count = HC_DIM;
    const std::size_t history_count =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS;
    const std::size_t conv_count = static_cast<std::size_t>(HC_DIM) * 4u;
    const std::size_t key_weight_count =
        static_cast<std::size_t>(key_width) * input_width;
    const std::size_t value_weight_count =
        static_cast<std::size_t>(input_width) * input_width;
    const std::size_t q8_count =
        static_cast<std::size_t>(rows) * (input_width / 32u);
    const std::size_t key_padding_count =
        static_cast<std::size_t>(compiled_columns - rows) * key_width;
    const std::size_t value_padding_count =
        static_cast<std::size_t>(compiled_columns - rows) * input_width;
    const std::size_t sink_count =
        std::max<std::size_t>(1u, key_padding_count);

    std::vector<float> input(active_input_count);
    std::vector<float> compiled_input(compiled_input_count);
    std::vector<float> residual(post_count);
    std::vector<float> key_gamma(gamma_count);
    std::vector<float> query_gamma(gamma_count);
    std::vector<float> conv_gamma(gamma_count);
    std::vector<float> initial_history(history_count);
    std::vector<std::uint16_t> conv1d(conv_count);
    std::mt19937_64 input_rng(options.seed ^ 0x504c45424c4f434bull);
    std::mt19937_64 residual_rng(options.seed ^ 0x504c45424c4f4352ull);
    std::mt19937_64 history_rng(options.seed ^ 0x504c45424c4f4348ull);
    std::mt19937_64 weight_rng(options.seed ^ 0x504c45424c4f4357ull);
    for (std::size_t index = 0u; index < input.size(); ++index) {
        input[index] = value_input(twin::InputSet::fixture,
                                   static_cast<std::uint32_t>(index / input_width),
                                   static_cast<std::uint32_t>(index % input_width),
                                   input_rng);
    }
    for (std::uint32_t row = 0u; row < rows; ++row) {
        std::copy_n(input.begin() + static_cast<std::size_t>(row) * input_width,
                    input_width,
                    compiled_input.begin() +
                        static_cast<std::size_t>(row) * input_width);
        for (std::uint32_t column = 0u; column < key_width; ++column) {
            const std::size_t index =
                static_cast<std::size_t>(row) * key_width + column;
            residual[index] =
                value_input(twin::InputSet::fixture, row + 1u, column,
                            residual_rng);
        }
    }

    std::string error;
    std::vector<float> residual_seed = residual;
    std::vector<float> previous_block_out(
        static_cast<std::size_t>(rows) * input_width);
    std::vector<float> previous_inject(static_cast<std::size_t>(rows) * HC);
    std::vector<twin::tf> residual_after_previous_write;
    if (prepend_gr_post) {
        std::mt19937_64 previous_write_rng(
            options.seed ^ 0x57494e504c455752ull);
        for (std::uint32_t row = 0u; row < rows; ++row) {
            for (std::uint32_t column = 0u; column < input_width; ++column) {
                previous_block_out[static_cast<std::size_t>(row) * input_width +
                                   column] =
                    value_input(twin::InputSet::fixture, row + 3u, column,
                                previous_write_rng);
            }
            for (std::uint32_t stream = 0u; stream < HC; ++stream) {
                previous_inject[static_cast<std::size_t>(row) * HC + stream] =
                    gate_input(twin::InputSet::fixture, row + 5u, stream,
                               previous_write_rng);
            }
        }
        gr_post_reference(residual_seed, previous_block_out, previous_inject,
                          rows, residual_after_previous_write);

        twin::GuardedBuffer probe_residual;
        twin::GuardedBuffer probe_block_out;
        twin::GuardedBuffer probe_inject;
        bool probe_allocated = true;
        probe_allocated &= probe_residual.allocate(
            context, label + " prior-write oracle residual",
            post_count * sizeof(float), error);
        probe_allocated &= probe_block_out.allocate(
            context, label + " prior-write oracle block output",
            previous_block_out.size() * sizeof(float), error);
        probe_allocated &= probe_inject.allocate(
            context, label + " prior-write oracle injection",
            previous_inject.size() * sizeof(float), error);
        check(probe_allocated,
              label + " prior-write oracle buffer allocation", error);
        if (!probe_allocated) return false;
        std::memcpy(probe_residual.payload().data(), residual_seed.data(),
                    residual_seed.size() * sizeof(float));
        std::memcpy(probe_block_out.payload().data(),
                    previous_block_out.data(),
                    previous_block_out.size() * sizeof(float));
        std::memcpy(probe_inject.payload().data(), previous_inject.data(),
                    previous_inject.size() * sizeof(float));
        probe_residual.capture_before_dispatch();
        probe_block_out.capture_before_dispatch();
        probe_inject.capture_before_dispatch();

        auto probe_command = context.make_command_buffer(
            label + " prior GR write oracle probe", error);
        if (!probe_command) {
            check(false, label + " prior-write oracle command buffer", error);
            return false;
        }
        const GRPostBuffers probe_buffers{
            probe_residual.view(), probe_block_out.view(), probe_inject.view(),
            probe_residual.view(),
        };
        const bool probe_encoded = encode_gr_post(
            context, *probe_command, rows, probe_buffers, error);
        check(probe_encoded, label + " prior GR write oracle encode", error);
        if (!probe_encoded) {
            probe_command->discard();
            return false;
        }
        const bool probe_dispatched = submit_and_wait(
            probe_command, label + " prior GR write oracle probe", error);
        check(probe_dispatched,
              label + " prior GR write oracle dispatch", error);
        if (!probe_dispatched) return false;

        const auto actual_previous_write = std::span<const float>(
            reinterpret_cast<const float*>(probe_residual.payload().data()),
            post_count);
        const auto prior_write_mismatch = twin::compare_floats(
            actual_previous_write, residual_after_previous_write,
            {twin::Comparison::units, 16.0, 0.0, 0u});
        if (prior_write_mismatch) {
            twin::report_failure(label.c_str(), options, rows,
                                 *prior_write_mismatch);
        }
        check(!prior_write_mismatch,
              label + " prior GR write matches its scalar oracle",
              prior_write_mismatch ? prior_write_mismatch->reason : "");
        const std::array<twin::ByteRange, 1> prior_write_range{{
            {0u, post_count * sizeof(float)},
        }};
        const auto probe_residual_guard =
            probe_residual.first_unwritten_change(prior_write_range);
        check(!probe_residual_guard,
              label + " prior-write residual guards remain unchanged",
              probe_residual_guard ? probe_residual_guard->reason : "");
        check(!probe_block_out.first_unwritten_change({}),
              label + " prior-write block output remains read-only");
        check(!probe_inject.first_unwritten_change({}),
              label + " prior-write injection remains read-only");
        if (prior_write_mismatch || probe_residual_guard ||
            probe_block_out.first_unwritten_change({}) ||
            probe_inject.first_unwritten_change({})) {
            return false;
        }
        std::copy(actual_previous_write.begin(), actual_previous_write.end(),
                  residual.begin());
    }
    for (std::uint32_t column = 0u; column < key_width; ++column) {
        const std::size_t history_offset =
            static_cast<std::size_t>(column) * PLE_HISTORY_ROWS;
        for (std::uint32_t history_row = 0u;
             history_row < PLE_HISTORY_ROWS; ++history_row) {
            initial_history[history_offset + history_row] =
                value_input(twin::InputSet::fixture, column, history_row,
                            history_rng);
        }
        for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
            conv1d[static_cast<std::size_t>(column) * 4u + tap] =
                strata::kernels::f16_from_f32(
                    value_input(twin::InputSet::fixture, column, tap,
                                weight_rng) *
                    0.125f);
        }
        key_gamma[column] =
            0.5f + static_cast<float>((column * 17u) % 257u) / 512.0f;
        query_gamma[column] =
            0.5f + static_cast<float>((column * 29u + 61u) % 257u) /
                       512.0f;
        conv_gamma[column] =
            0.5f + static_cast<float>((column * 43u + 139u) % 257u) /
                       512.0f;
    }
    for (std::uint32_t row = rows; row < compiled_columns; ++row) {
        std::copy_n(input.begin(), input_width,
                    compiled_input.begin() +
                        static_cast<std::size_t>(row) * input_width);
    }

    std::vector<block_q2_0> q2_weights;
    std::vector<block_q8_0> q8_weights;
    std::vector<std::uint16_t> bf16_key_weights;
    std::vector<std::uint16_t> bf16_value_weights(value_weight_count);
    if (key_format == MatVecWeightFormat::q2_0) {
        const std::uint32_t blocks_per_row = input_width / 64u;
        q2_weights.resize(key_weight_count / 64u);
        for (std::size_t block_index = 0u;
             block_index < q2_weights.size(); ++block_index) {
            auto& block = q2_weights[block_index];
            block.d = strata::kernels::f16_from_f32(
                0.015625f * static_cast<float>(1u + block_index % 5u));
            std::fill_n(block.qs, 16u, std::uint8_t{0});
            for (std::uint32_t value = 0u; value < 64u; ++value) {
                const std::uint32_t code = static_cast<std::uint32_t>(
                    (value * 3u + block_index * 5u + block_index /
                                                       blocks_per_row) %
                    4u);
                const std::uint32_t byte_index = value >> 2u;
                const std::uint32_t shift = 2u * (value & 3u);
                block.qs[byte_index] |=
                    static_cast<std::uint8_t>(code << shift);
            }
        }
    } else if (key_format == MatVecWeightFormat::q8_0) {
        q8_weights.resize(key_weight_count / 32u);
        for (std::size_t block_index = 0u;
             block_index < q8_weights.size(); ++block_index) {
            auto& block = q8_weights[block_index];
            block.d = strata::kernels::f16_from_f32(
                0.001953125f * static_cast<float>(1u + block_index % 5u));
            for (std::uint32_t value = 0u; value < 32u; ++value) {
                const int code = static_cast<int>(
                    (value * 7u + block_index * 3u) % 31u) -
                                15;
                block.qs[value] = static_cast<std::int8_t>(code);
            }
        }
    } else {
        bf16_key_weights.resize(key_weight_count);
        for (std::size_t index = 0u; index < key_weight_count; ++index) {
            const int value = static_cast<int>((index * 11u) % 61u) - 30;
            bf16_key_weights[index] = strata::kernels::bf16_from_f32(
                static_cast<float>(value) / 8192.0f);
        }
    }
    for (std::size_t index = 0u; index < bf16_value_weights.size(); ++index) {
        const int value = static_cast<int>((index * 13u + 7u) % 67u) - 33;
        bf16_value_weights[index] = strata::kernels::bf16_from_f32(
            static_cast<float>(value) / 8192.0f);
    }

    std::vector<Q81> q8_reference;
    std::vector<float> key_projection;
    bool oracle_ready = true;
    if (quantized_key) {
        oracle_ready &= twin::quantize_q8_1_reference(
            compiled_input, input_width, compiled_columns, q8_reference,
            error);
        check(oracle_ready, label + " Q8_1 CPU oracle", error);
        if (oracle_ready && key_format == MatVecWeightFormat::q2_0) {
            oracle_ready &= twin::q2_0_matvec_reference(
                q2_weights, q8_reference, input_width, key_width,
                compiled_columns, rows, key_projection, error);
        } else if (oracle_ready) {
            oracle_ready &= twin::q8_0_matvec_reference(
                q8_weights, q8_reference, input_width, key_width,
                compiled_columns, rows, key_projection, error);
        }
    } else {
        oracle_ready &= twin::bf16_matvec_reference(
            bf16_key_weights, compiled_input, input_width, key_width,
            compiled_columns, rows, 256u, key_projection, error);
    }
    check(oracle_ready, label + " key projection CPU oracle", error);
    std::vector<float> value_projection;
    oracle_ready &= twin::bf16_matvec_reference(
        bf16_value_weights, compiled_input, input_width, input_width,
        compiled_columns, rows, 256u, value_projection, error);
    check(oracle_ready, label + " F32-activation value CPU oracle", error);
    if (!oracle_ready) return false;

    std::vector<float> active_key_projection(key_count);
    std::vector<float> active_value_projection(value_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        std::copy_n(key_projection.begin() +
                        static_cast<std::size_t>(row) * key_width,
                    key_width,
                    active_key_projection.begin() +
                        static_cast<std::size_t>(row) * key_width);
        std::copy_n(value_projection.begin() +
                        static_cast<std::size_t>(row) * input_width,
                    input_width,
                    active_value_projection.begin() +
                        static_cast<std::size_t>(row) * input_width);
    }
    std::vector<twin::tf> key_expected;
    std::vector<twin::tf> query_expected;
    std::vector<twin::tf> value_expected;
    std::vector<twin::tf> gate_expected(static_cast<std::size_t>(rows) * HC);
    std::vector<twin::tf> gated_expected(key_count);
    std::vector<float> key_projection_f32(key_count);
    std::vector<float> query_projection_f32(post_count);
    weighted_rms_norm_oracle(active_key_projection, key_gamma, rows * HC,
                             N_EMBD, HC, epsilon, key_expected);
    weighted_rms_norm_oracle(residual, query_gamma, rows * HC, N_EMBD, HC,
                             epsilon, query_expected);
    std::transform(key_expected.begin(), key_expected.end(),
                   key_projection_f32.begin(),
                   [](const twin::tf& value) { return value.value; });
    std::transform(query_expected.begin(), query_expected.end(),
                   query_projection_f32.begin(),
                   [](const twin::tf& value) { return value.value; });
    value_expected.reserve(value_count);
    for (std::uint32_t index = 0u; index < rows * input_width; ++index) {
        value_expected.emplace_back(active_value_projection[index]);
    }
    const float gate_scale = 1.0f / std::sqrt(static_cast<float>(N_EMBD));
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t stream = 0u; stream < HC; ++stream) {
            const std::size_t offset =
                (static_cast<std::size_t>(row) * HC + stream) * N_EMBD;
            gate_expected[static_cast<std::size_t>(row) * HC + stream] =
                gate_oracle(
                    std::span<const float>(
                        key_projection_f32.data() + offset, N_EMBD),
                    std::span<const float>(
                        query_projection_f32.data() + offset, N_EMBD),
                    gate_scale);
        }
        for (std::uint32_t column = 0u; column < N_EMBD; ++column) {
            for (std::uint32_t stream = 0u; stream < HC; ++stream) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * HC_DIM +
                    static_cast<std::size_t>(stream) * N_EMBD + column;
                gated_expected[index] = twin::tf{
                    active_value_projection[
                        static_cast<std::size_t>(row) * N_EMBD + column] *
                    gate_expected[static_cast<std::size_t>(row) * HC +
                                  stream]
                        .value};
            }
        }
    }

    enum BufferIndex : std::size_t {
        input_buffer,
        q8_buffer,
        key_weights_buffer,
        value_weights_buffer,
        sink_buffer,
        active_rows_buffer,
        previous_block_buffer,
        previous_inject_buffer,
        key_buffer,
        residual_buffer,
        query_buffer,
        norm_buffer,
        value_buffer,
        gate_buffer,
        gated_buffer,
        history_buffer,
        conv_buffer,
        key_gamma_buffer,
        query_gamma_buffer,
        conv_gamma_buffer,
        buffer_count,
    };
    std::array<twin::GuardedBuffer, buffer_count> guarded;
    const auto allocate = [&](BufferIndex index, const std::string& name,
                              std::size_t bytes) {
        const bool ok =
            guarded[index].allocate(context, label + " " + name, bytes, error);
        check(ok, label + " " + name + " allocation", error);
        return ok;
    };
    bool allocated = true;
    allocated &= allocate(input_buffer, "F32 input",
                          active_input_count * sizeof(float));
    if (quantized_key) {
        allocated &= allocate(q8_buffer, "Q8_1 activations",
                              q8_count * sizeof(Q81));
    }
    const std::size_t key_weight_bytes =
        key_format == MatVecWeightFormat::q2_0
            ? q2_weights.size() * sizeof(block_q2_0)
            : key_format == MatVecWeightFormat::q8_0
                  ? q8_weights.size() * sizeof(block_q8_0)
                  : bf16_key_weights.size() * sizeof(std::uint16_t);
    allocated &= allocate(key_weights_buffer, "key weights",
                          key_weight_bytes);
    allocated &= allocate(value_weights_buffer, "value weights",
                          bf16_value_weights.size() *
                              sizeof(std::uint16_t));
    allocated &= allocate(sink_buffer, "shared mat-vec sink",
                          sink_count * sizeof(float));
    allocated &= allocate(active_rows_buffer, "active rows",
                          sizeof(std::uint32_t));
    allocated &= allocate(previous_block_buffer, "previous block output",
                          previous_block_out.size() * sizeof(float));
    allocated &= allocate(previous_inject_buffer, "previous injection",
                          previous_inject.size() * sizeof(float));
    allocated &= allocate(key_buffer, "key output",
                          key_count * sizeof(float));
    allocated &= allocate(value_buffer, "value output",
                          value_count * sizeof(float));
    allocated &= allocate(residual_buffer, "residual",
                          post_count * sizeof(float));
    allocated &= allocate(query_buffer, "query",
                          post_count * sizeof(float));
    allocated &= allocate(norm_buffer, "PLE norm",
                          post_count * sizeof(float));
    allocated &= allocate(gate_buffer, "gate",
                          static_cast<std::size_t>(rows) * HC *
                              sizeof(float));
    allocated &= allocate(gated_buffer, "gated value",
                          post_count * sizeof(float));
    allocated &= allocate(history_buffer, "history",
                          history_count * sizeof(float));
    allocated &= allocate(conv_buffer, "convolution weights",
                          conv_count * sizeof(std::uint16_t));
    allocated &= allocate(key_gamma_buffer, "key gamma",
                          gamma_count * sizeof(float));
    allocated &= allocate(query_gamma_buffer, "query gamma",
                          gamma_count * sizeof(float));
    allocated &= allocate(conv_gamma_buffer, "convolution gamma",
                          gamma_count * sizeof(float));
    if (!allocated) return false;
    for (std::size_t index = 0u; index < guarded.size(); ++index) {
        if (index == q8_buffer && !quantized_key) continue;
        if (!guarded[index].view()) return false;
        guarded[index].capture_before_dispatch();
    }
    std::memcpy(guarded[input_buffer].payload().data(), input.data(),
                input.size() * sizeof(float));
    if (key_format == MatVecWeightFormat::q2_0) {
        std::memcpy(guarded[key_weights_buffer].payload().data(),
                    q2_weights.data(), key_weight_bytes);
    } else if (key_format == MatVecWeightFormat::q8_0) {
        std::memcpy(guarded[key_weights_buffer].payload().data(),
                    q8_weights.data(), key_weight_bytes);
    } else {
        std::memcpy(guarded[key_weights_buffer].payload().data(),
                    bf16_key_weights.data(), key_weight_bytes);
    }
    std::memcpy(guarded[value_weights_buffer].payload().data(),
                bf16_value_weights.data(),
                bf16_value_weights.size() * sizeof(std::uint16_t));
    std::memcpy(guarded[active_rows_buffer].payload().data(), &rows,
                sizeof(rows));
    std::memcpy(guarded[previous_block_buffer].payload().data(),
                previous_block_out.data(), previous_block_out.size() * sizeof(float));
    std::memcpy(guarded[previous_inject_buffer].payload().data(),
                previous_inject.data(), previous_inject.size() * sizeof(float));
    std::memcpy(guarded[residual_buffer].payload().data(), residual_seed.data(),
                residual_seed.size() * sizeof(float));
    std::memcpy(guarded[history_buffer].payload().data(),
                initial_history.data(), initial_history.size() * sizeof(float));
    std::memcpy(guarded[conv_buffer].payload().data(), conv1d.data(),
                conv1d.size() * sizeof(std::uint16_t));
    std::memcpy(guarded[key_gamma_buffer].payload().data(), key_gamma.data(),
                key_gamma.size() * sizeof(float));
    std::memcpy(guarded[query_gamma_buffer].payload().data(),
                query_gamma.data(), query_gamma.size() * sizeof(float));
    std::memcpy(guarded[conv_gamma_buffer].payload().data(),
                conv_gamma.data(), conv_gamma.size() * sizeof(float));

    bool host_initialization_guards_clean = true;
    for (std::size_t index = 0u; index < guarded.size(); ++index) {
        if (index == q8_buffer && !quantized_key) continue;
        const std::array<twin::ByteRange, 1> writable_payload{{
            {0u, guarded[index].payload().size()},
        }};
        const auto changed =
            guarded[index].first_unwritten_change(writable_payload);
        check(!changed,
              label + " host initialization preserves " +
                  std::to_string(index) + " buffer guards",
              changed ? changed->reason : "");
        host_initialization_guards_clean &= !changed;
    }
    if (!host_initialization_guards_clean) return false;

    std::vector<twin::tf> zero_post(post_count, twin::tf{0.0f});
    std::vector<twin::tf> norm_expected;
    std::vector<std::uint8_t> sink_expected(sink_count * sizeof(float), 0u);
    if (compiled_columns > rows) {
        const std::size_t key_pad_bytes =
            key_padding_count * sizeof(float);
        const std::size_t value_pad_bytes =
            value_padding_count * sizeof(float);
        std::memcpy(sink_expected.data(),
                    key_projection.data() + key_count, key_pad_bytes);
        std::memcpy(sink_expected.data(),
                    value_projection.data() + value_count, value_pad_bytes);
    }
    bool poison_ready = true;
    poison_ready &= guarded[key_buffer].poison_floats_against(
        key_expected, twin::Comparison::units, error);
    poison_ready &= guarded[value_buffer].poison_floats_against(
        value_expected, twin::Comparison::bits, error);
    poison_ready &= guarded[query_buffer].poison_floats_against(
        query_expected, twin::Comparison::units, error);
    poison_ready &= guarded[norm_buffer].poison_floats_against(
        zero_post, twin::Comparison::bits, error);
    poison_ready &= guarded[gate_buffer].poison_floats_against(
        gate_expected, twin::Comparison::units, error);
    poison_ready &= guarded[gated_buffer].poison_floats_against(
        gated_expected, twin::Comparison::bits, error);
    poison_ready &= guarded[sink_buffer].poison_bytes_against(
        sink_expected, error);
    if (quantized_key) {
        poison_ready &= guarded[q8_buffer].poison_bytes_against(
            std::span<const std::uint8_t>(
                reinterpret_cast<const std::uint8_t*>(q8_reference.data()),
                q8_count * sizeof(Q81)),
            error);
    }
    check(poison_ready, label + " output poison", error);
    if (!poison_ready) return false;
    for (auto& buffer : guarded) buffer.capture_before_dispatch();

    const PLEPostProjectionBuffers post{
        guarded[key_buffer].view(),
        guarded[residual_buffer].view(),
        guarded[query_buffer].view(),
        guarded[norm_buffer].view(),
        guarded[value_buffer].view(),
        guarded[gate_buffer].view(),
        guarded[gated_buffer].view(),
        guarded[history_buffer].view(),
        read_only(context, guarded[conv_buffer].view(),
                  label + " conv weights view"),
        read_only(context, guarded[key_gamma_buffer].view(),
                  label + " key gamma view"),
        read_only(context, guarded[query_gamma_buffer].view(),
                  label + " query gamma view"),
        read_only(context, guarded[conv_gamma_buffer].view(),
                  label + " conv gamma view"),
    };
    const PLEBlockBuffers block{
        guarded[input_buffer].view(),
        quantized_key ? guarded[q8_buffer].view() : BufferRef{},
        read_only(context, guarded[key_weights_buffer].view(),
                  label + " key weights view"),
        read_only(context, guarded[value_weights_buffer].view(),
                  label + " value weights view"),
        guarded[sink_buffer].view(),
        read_only(context, guarded[active_rows_buffer].view(),
                  label + " active rows view"),
        post,
    };
    MoeRoutedSharedFFNBuffers previous_ffn{};
    previous_ffn.shared.output = guarded[previous_block_buffer].view();
    GRReadBuffers previous_gr_read{};
    previous_gr_read.residual = guarded[residual_buffer].view();
    previous_gr_read.inject_out = guarded[previous_inject_buffer].view();
    const WindowMoePLEEntryBuffers window_entry{
        previous_ffn, previous_gr_read, block};
    const WindowMoePendingConsumer pending_consumer{
        WindowMoePendingPLEConsumer{
            window_entry, key_format, compiled_columns,
            PLEHistoryUpdate::defer}};

    if (check_late_failure) {
        std::array<std::vector<std::uint8_t>, buffer_count> before;
        for (std::size_t index = 0u; index < guarded.size(); ++index) {
            before[index].assign(guarded[index].payload().begin(),
                                 guarded[index].payload().end());
        }
        const auto unchanged_after_rejection = [&] {
            for (std::size_t index = 0u; index < guarded.size(); ++index) {
                const auto after = guarded[index].payload();
                if (before[index].size() != after.size() ||
                    !std::equal(before[index].begin(), before[index].end(),
                                after.begin())) {
                    return false;
                }
            }
            return true;
        };
        auto unsupported_format = context.make_command_buffer(
            label + " unsupported key format", error);
        if (!unsupported_format) {
            check(false, label + " unsupported-format command buffer", error);
            return false;
        }
        const bool unsupported_rejected = !encode_ple_block(
            context, *unsupported_format, rows, MatVecWeightFormat::q4_0,
            compiled_columns, PLEHistoryUpdate::defer, block, error);
        check(unsupported_rejected &&
                  unsupported_format->unsafe_native_handle() == nullptr &&
                  error.find("key format") != std::string::npos,
              label + " rejects unsupported PLE key format", error);
        const bool unsupported_unchanged = unchanged_after_rejection();
        check(unsupported_unchanged,
              label + " unsupported-format rejection preserves all buffers");

        PLEBlockBuffers aliased_block = block;
        aliased_block.key_q8_1 = guarded[input_buffer].view();
        auto alias_command = context.make_command_buffer(
            label + " cross-stage alias", error);
        if (!alias_command) {
            check(false, label + " cross-stage alias command buffer", error);
            return false;
        }
        const bool alias_rejected = !encode_ple_block(
            context, *alias_command, rows, key_format, compiled_columns,
            PLEHistoryUpdate::defer, aliased_block, error);
        check(alias_rejected &&
                  alias_command->unsafe_native_handle() == nullptr &&
                  error.find("input and key Q8_1") != std::string::npos,
              label + " rejects projection input/Q8_1 output alias", error);
        const bool alias_unchanged = unchanged_after_rejection();
        check(alias_unchanged,
              label + " alias rejection preserves all buffers");

        if (prepend_gr_post) {
            WindowMoePLEEntryBuffers wrong_residual_entry = window_entry;
            wrong_residual_entry.previous_gr_read.residual =
                guarded[norm_buffer].view();
            auto wrong_residual_command = context.make_command_buffer(
                label + " mismatched pending-MoE residual", error);
            if (!wrong_residual_command) {
                check(false, label + " mismatched-residual command buffer",
                      error);
                return false;
            }
            const bool wrong_residual_rejected =
                !Encoder::encode(
                    context, *wrong_residual_command,
                    "Window pending-MoE PLE residual rejection",
                    [&](Encoder& encoder, std::string& encode_error) {
                        return encode_window_moe_pending_consumer(
                            context, encoder, rows, true,
                            WindowMoePendingConsumer{
                                WindowMoePendingPLEConsumer{
                                    wrong_residual_entry, key_format,
                                    compiled_columns,
                                    PLEHistoryUpdate::defer}},
                            encode_error);
                    },
                    error);
            const bool wrong_residual_discarded =
                wrong_residual_rejected &&
                wrong_residual_command->unsafe_native_handle() == nullptr &&
                error.find("previous GR residual") != std::string::npos;
            check(wrong_residual_discarded,
                  label + " rejects a PLE residual outside the pending GR handoff",
                  error);
            const bool wrong_residual_unchanged =
                unchanged_after_rejection();
            check(wrong_residual_unchanged,
                  label + " mismatched residual rejection preserves all buffers");
            if (!wrong_residual_discarded || !wrong_residual_unchanged) {
                return false;
            }
        }

        PLEBlockBuffers invalid_block = block;
        invalid_block.post.history = guarded[history_buffer].view().slice(
            0u, history_count * sizeof(float) - sizeof(float));
        auto rejected = context.make_command_buffer(
            label + " late history validation failure", error);
        if (!rejected) {
            check(false, label + " late-failure command buffer", error);
            return false;
        }
        WindowMoePLEEntryBuffers invalid_entry = window_entry;
        invalid_entry.ple = invalid_block;
        const bool accepted = prepend_gr_post
                                  ? Encoder::encode(
                                        context, *rejected,
                                        "Window pending-MoE PLE late failure",
                                        [&](Encoder& encoder,
                                            std::string& encode_error) {
                                            return encode_window_moe_pending_consumer(
                                                context, encoder, rows, true,
                                                WindowMoePendingConsumer{
                                                    WindowMoePendingPLEConsumer{
                                                        invalid_entry,
                                                        key_format,
                                                        compiled_columns,
                                                        PLEHistoryUpdate::defer}},
                                                encode_error);
                                        },
                                        error)
                                  : encode_ple_block(
                                        context, *rejected, rows, key_format,
                                        compiled_columns,
                                        PLEHistoryUpdate::defer, invalid_block,
                                        error);
        const bool discarded =
            !accepted && rejected->unsafe_native_handle() == nullptr &&
            error.find("history") != std::string::npos;
        check(discarded,
              label + " late history failure discards all partial dispatches",
              error);
        const bool unchanged = unchanged_after_rejection();
        check(unchanged,
              label + " discarded partial segment leaves every buffer unchanged");
        if (!unsupported_rejected || !unsupported_unchanged || !alias_rejected ||
            !alias_unchanged || !discarded || !unchanged) {
            return false;
        }

        if (prepend_gr_post) {
            auto caller_owned_rejection = context.make_command_buffer(
                label + " caller-owned late history failure", error);
            if (!caller_owned_rejection) {
                check(false,
                      label + " caller-owned late-failure command buffer",
                      error);
                return false;
            }
            const bool caller_owned_accepted = Encoder::encode(
                context, *caller_owned_rejection,
                "Window PLE caller-owned late failure",
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_window_moe_pending_consumer(
                        context, encoder, rows, true,
                        WindowMoePendingConsumer{
                            WindowMoePendingPLEConsumer{
                                invalid_entry, key_format, compiled_columns,
                                PLEHistoryUpdate::defer}},
                        encode_error);
                },
                error);
            const bool caller_owned_discarded =
                !caller_owned_accepted &&
                caller_owned_rejection->unsafe_native_handle() == nullptr &&
                error.find("history") != std::string::npos;
            check(caller_owned_discarded,
                  label + " caller propagates late PLE failure to discard GR",
                  error);
            const bool caller_owned_unchanged = unchanged_after_rejection();
            check(caller_owned_unchanged,
                  label + " caller-owned failure preserves all buffers");
            if (!caller_owned_discarded || !caller_owned_unchanged) {
                return false;
            }
        }

        const char* const dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
        if (prepend_gr_post && dispatch_mode != nullptr &&
            std::strcmp(dispatch_mode, "concurrent") == 0) {
            auto grouped = context.make_command_buffer(
                label + " open concurrent group rejection", error);
            if (!grouped) {
                check(false, label + " open-group command buffer", error);
                return false;
            }
            std::string stage_error;
            const bool outer_encoded = Encoder::encode(
                context, *grouped, "Window PLE open-group rejection",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encoder.group_begin(encode_error)) return false;
                    const bool stage_accepted =
                        encode_window_moe_pending_consumer(
                            context, encoder, rows, true, pending_consumer,
                            stage_error);
                    if (!encoder.group_end(encode_error)) return false;
                    return !stage_accepted;
                },
                error);
            const bool rejected_open_group =
                outer_encoded &&
                stage_error.find("ordered concurrent-group boundary") !=
                    std::string::npos;
            check(rejected_open_group,
                  label + " rejects PLE entry inside concurrent dispatch group",
                  stage_error);
            const bool open_group_unchanged = unchanged_after_rejection();
            check(open_group_unchanged,
                  label + " open-group rejection preserves all buffers");
            grouped->discard();
            if (!rejected_open_group || !open_group_unchanged) return false;
        }
    }

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = prepend_gr_post && rows == 1u
                             ? encode_window_moe_ple_entry(
                                   context, *command, rows, key_format,
                                   compiled_columns,
                                   PLEHistoryUpdate::defer, window_entry, error)
                         : Encoder::encode(
                               context, *command,
                               prepend_gr_post
                                   ? "Window prior write and PLE block segment"
                                   : "full PLE projection and post-op segment",
                               [&](Encoder& encoder,
                                   std::string& encode_error) {
                                   return prepend_gr_post
                                              ? encode_window_moe_pending_consumer(
                                                    context, encoder, rows,
                                                    true, pending_consumer,
                                                    encode_error)
                                              : encode_ple_block(
                                                    context, encoder, rows,
                                                    key_format,
                                                    compiled_columns,
                                                    PLEHistoryUpdate::defer,
                                                    block, encode_error);
                               },
                               error);
    check(encoded, label + " shared-Encoder encode", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, error);
    check(dispatched, label + " Metal dispatch", error);
    if (!dispatched) return false;

    const auto floats = [&](BufferIndex index, std::size_t count) {
        return std::span<const float>(
            reinterpret_cast<const float*>(
                guarded[index].payload().data()),
            count);
    };
    const auto compare = [&](BufferIndex index, std::size_t count,
                             const std::vector<twin::tf>& expected,
                             twin::Comparison comparison, double units,
                             const std::string& stage) {
        const auto mismatch = twin::compare_floats(
            floats(index, count), expected,
            {comparison, units, 0.0, 0u});
        if (mismatch) {
            twin::report_failure(label.c_str(), options, rows, *mismatch);
        }
        check(!mismatch, label + " " + stage + " oracle",
              mismatch ? mismatch->reason : "");
        return !mismatch;
    };
    bool passed = true;
    passed &= compare(key_buffer, key_count, key_expected,
                      twin::Comparison::units, 64.0, "normalized key");
    passed &= compare(query_buffer, post_count, query_expected,
                      twin::Comparison::units, 64.0, "normalized query");
    passed &= compare(value_buffer, value_count, value_expected,
                      twin::Comparison::bits, 0.0, "BF16 value projection");
    passed &= compare(gate_buffer, gate_expected.size(), gate_expected,
                      twin::Comparison::units, kGateToleranceUnits, "gate");

    const auto actual_gate = floats(gate_buffer, gate_expected.size());
    const auto actual_value = floats(value_buffer, value_count);
    std::vector<twin::tf> actual_gate_broadcast(key_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t stream = 0u; stream < HC; ++stream) {
            for (std::uint32_t column = 0u; column < N_EMBD; ++column) {
                const std::size_t index =
                    static_cast<std::size_t>(row) * HC_DIM +
                    static_cast<std::size_t>(stream) * N_EMBD + column;
                actual_gate_broadcast[index] = twin::tf{
                    actual_value[static_cast<std::size_t>(row) * N_EMBD +
                                 column] *
                    actual_gate[static_cast<std::size_t>(row) * HC + stream]};
            }
        }
    }
    passed &= compare(gated_buffer, key_count, actual_gate_broadcast,
                      twin::Comparison::bits, 0.0, "value broadcast");
    weighted_rms_norm_oracle(
        floats(gated_buffer, key_count), conv_gamma, rows * HC, N_EMBD, HC,
        epsilon, norm_expected);
    passed &= compare(norm_buffer, post_count, norm_expected,
                      twin::Comparison::units, 64.0,
                      "convolution weighted norm");
    std::vector<twin::tf> residual_expected(post_count);
    const auto actual_norm = floats(norm_buffer, post_count);
    const auto actual_gated = floats(gated_buffer, key_count);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
            std::array<float, 4> tap_values{};
            std::array<float, 4> tap_weights{};
            const std::size_t history_offset =
                static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
            for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
                const std::int32_t norm_row =
                    static_cast<std::int32_t>(row) -
                    static_cast<std::int32_t>(PLE_HISTORY_ROWS) +
                    static_cast<std::int32_t>(3u * tap);
                tap_values[tap] =
                    norm_row >= 0
                        ? actual_norm[static_cast<std::size_t>(norm_row) *
                                          HC_DIM +
                                      channel]
                        : initial_history[
                              history_offset +
                              static_cast<std::size_t>(
                                  static_cast<std::int32_t>(
                                      PLE_HISTORY_ROWS) +
                                  norm_row)];
                tap_weights[tap] = strata::kernels::f32_from_f16(
                    conv1d[static_cast<std::size_t>(channel) * 4u + tap]);
            }
            const std::size_t index =
                static_cast<std::size_t>(row) * HC_DIM + channel;
            residual_expected[index] = conv_residual_oracle(
                tap_values, tap_weights, actual_gated[index],
                residual[index]);
        }
    }
    passed &= compare(residual_buffer, post_count, residual_expected,
                      twin::Comparison::units, kConvToleranceUnits,
                      "convolution residual");
    const bool history_unchanged =
        std::memcmp(guarded[history_buffer].payload().data(),
                    initial_history.data(),
                    initial_history.size() * sizeof(float)) == 0;
    check(history_unchanged,
          label + " deferred history remains bitwise unchanged");
    passed &= history_unchanged;
    if (compiled_columns > rows) {
        const auto sink_mismatch = twin::compare_bytes(
            guarded[sink_buffer].payload(), sink_expected);
        check(!sink_mismatch, label + " shared sink contains padded value rows",
              sink_mismatch ? sink_mismatch->reason : "");
        passed &= !sink_mismatch;
    }
    check(!guarded[input_buffer].first_unwritten_change({}),
          label + " input and guards remain unchanged");
    check(!guarded[previous_block_buffer].first_unwritten_change({}),
          label + " previous block output remains read-only");
    check(!guarded[previous_inject_buffer].first_unwritten_change({}),
          label + " previous injection remains read-only");
    check(!guarded[key_weights_buffer].first_unwritten_change({}),
          label + " key weights and guards remain unchanged");
    check(!guarded[value_weights_buffer].first_unwritten_change({}),
          label + " value weights and guards remain unchanged");
    check(!guarded[active_rows_buffer].first_unwritten_change({}),
          label + " active-row metadata and guards remain unchanged");
    check(!guarded[history_buffer].first_unwritten_change({}),
          label + " deferred history and guards remain unchanged");
    const auto check_guard_range = [&](BufferIndex index, std::size_t bytes,
                                       const std::string& name) {
        const std::array<twin::ByteRange, 1> written{{{0u, bytes}}};
        const auto changed =
            guarded[index].first_unwritten_change(bytes == 0u
                                                      ? std::span<const twin::ByteRange>{}
                                                      : std::span<const twin::ByteRange>{written});
        check(!changed, label + " " + name + " guards remain unchanged",
              changed ? changed->reason : "");
        return !changed;
    };
    passed &= check_guard_range(key_buffer, key_count * sizeof(float), "key");
    passed &= check_guard_range(value_buffer, value_count * sizeof(float),
                                "value");
    passed &= check_guard_range(residual_buffer, post_count * sizeof(float),
                                "residual");
    passed &= check_guard_range(query_buffer, post_count * sizeof(float),
                                "query");
    passed &= check_guard_range(norm_buffer, post_count * sizeof(float),
                                "PLE norm");
    passed &= check_guard_range(gate_buffer,
                                gate_expected.size() * sizeof(float), "gate");
    passed &= check_guard_range(gated_buffer, post_count * sizeof(float),
                                "gated value");
    passed &= check_guard_range(
        sink_buffer,
        compiled_columns > rows ? sink_count * sizeof(float) : 0u,
        "mat-vec sink");
    if (quantized_key) {
        passed &= check_guard_range(q8_buffer, q8_count * sizeof(Q81),
                                    "Q8_1 activations");
    }
    passed &= check_guard_range(conv_buffer, 0u, "convolution weights");
    passed &= check_guard_range(key_gamma_buffer, 0u, "key gamma");
    passed &= check_guard_range(query_gamma_buffer, 0u, "query gamma");
    passed &= check_guard_range(conv_gamma_buffer, 0u, "convolution gamma");
    return passed;
}

bool run_conv_residual_case(Context& context, std::uint32_t rows,
                            twin::InputSet set, const twin::Options& options,
                            bool arithmetic_probe,
                            std::vector<std::uint8_t>& output_snapshot) {
    const std::string label = "ple_conv_residual rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (arithmetic_probe ? " arithmetic-probe" : "");
    const std::size_t history_count =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS;
    const std::size_t norm_count =
        static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t value_count = norm_count;
    const std::size_t weight_count = static_cast<std::size_t>(HC_DIM) * 4u;
    const std::size_t history_bytes = history_count * sizeof(float);
    const std::size_t norm_bytes = norm_count * sizeof(float);
    const std::size_t value_bytes = value_count * sizeof(float);
    const std::size_t weight_bytes = weight_count * sizeof(std::uint16_t);

    std::vector<float> initial_history(history_count);
    std::vector<float> ple_norm(norm_count);
    std::vector<float> gated(value_count);
    std::vector<float> initial_residual(value_count);
    std::vector<std::uint16_t> conv1d(weight_count);

    std::mt19937_64 history_rng(options.seed ^ 0x504c454348495354ull);
    for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
        for (std::uint32_t history_row = 0u;
             history_row < PLE_HISTORY_ROWS; ++history_row) {
            initial_history[static_cast<std::size_t>(channel) *
                                PLE_HISTORY_ROWS + history_row] =
                value_input(set, channel, history_row, history_rng);
        }
    }
    std::mt19937_64 norm_rng(options.seed ^ 0x504c454e4f524dull);
    std::mt19937_64 gated_rng(options.seed ^ 0x504c454741544544ull);
    std::mt19937_64 residual_rng(options.seed ^ 0x504c455245534944ull);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
            const std::size_t index =
                static_cast<std::size_t>(row) * HC_DIM + channel;
            ple_norm[index] = value_input(set, row, channel, norm_rng);
            gated[index] = value_input(set, row + 2u, channel, gated_rng);
            initial_residual[index] =
                value_input(set, row + 4u, channel, residual_rng);
        }
    }
    std::mt19937_64 weight_rng(options.seed ^ 0x504c45434f4e5631ull);
    for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
        for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
            const float weight = value_input(set, channel, tap, weight_rng);
            conv1d[static_cast<std::size_t>(channel) * 4u + tap] =
                strata::kernels::f16_from_f32(weight);
        }
    }

    if (arithmetic_probe) {
        for (std::uint32_t channel = 0u; channel < 3u; ++channel) {
            const std::size_t history_offset =
                static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
            for (std::uint32_t history_row = 0u;
                 history_row < PLE_HISTORY_ROWS; ++history_row) {
                initial_history[history_offset + history_row] = 0.0f;
            }
            const std::size_t output_index = channel;
            ple_norm[output_index] = 0.0f;
            gated[output_index] = 0.0f;
            initial_residual[output_index] = 0.0f;
        }

        initial_history[0u] = 0.1f;
        initial_history[3u] = -0.1f;
        conv1d[0u] = strata::kernels::f16_from_f32(0.1f);
        conv1d[1u] = strata::kernels::f16_from_f32(0.1f);
        conv1d[2u] = strata::kernels::f16_from_f32(0.0f);
        conv1d[3u] = strata::kernels::f16_from_f32(0.0f);

        const std::size_t channel_one_history = PLE_HISTORY_ROWS;
        initial_history[channel_one_history] = 16777216.0f;
        initial_history[channel_one_history + 3u] = 1.0f;
        initial_history[channel_one_history + 6u] = -16777216.0f;
        for (std::uint32_t tap = 0u; tap < 3u; ++tap) {
            conv1d[4u + tap] = strata::kernels::f16_from_f32(1.0f);
        }
        conv1d[7u] = strata::kernels::f16_from_f32(0.0f);

        const std::size_t channel_two = 2u;
        ple_norm[channel_two] = 1.2f;
        gated[channel_two] = -33554432.0f;
        initial_residual[channel_two] = 33554432.0f;
        conv1d[channel_two * 4u] = strata::kernels::f16_from_f32(0.0f);
        conv1d[channel_two * 4u + 1u] = strata::kernels::f16_from_f32(0.0f);
        conv1d[channel_two * 4u + 2u] = strata::kernels::f16_from_f32(0.0f);
        conv1d[channel_two * 4u + 3u] = strata::kernels::f16_from_f32(1.0f);
    }

    std::vector<twin::tf> expected(value_count);
    std::vector<float> sequential_history = initial_history;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
            std::array<float, 4> tap_values{};
            std::array<float, 4> tap_weights{};
            for (std::uint32_t tap = 0u; tap < 4u; ++tap) {
                if (tap < 3u) {
                    tap_values[tap] =
                        sequential_history[static_cast<std::size_t>(channel) *
                                               PLE_HISTORY_ROWS +
                                           3u * tap];
                } else {
                    tap_values[tap] =
                        ple_norm[static_cast<std::size_t>(row) * HC_DIM +
                                 channel];
                }
                tap_weights[tap] = strata::kernels::f32_from_f16(
                    conv1d[static_cast<std::size_t>(channel) * 4u + tap]);
            }
            const std::size_t index =
                static_cast<std::size_t>(row) * HC_DIM + channel;
            expected[index] = conv_residual_oracle(
                tap_values, tap_weights, gated[index], initial_residual[index]);

            if (arithmetic_probe && row == 0u && channel < 3u) {
                const twin::tf ordered = expected[index];
                constexpr std::array<std::uint32_t, 4> reordered{
                    0u, 2u, 1u, 3u};
                const twin::tf reorder_control = conv_residual_oracle(
                    tap_values, tap_weights, gated[index],
                    initial_residual[index], false, reordered);
                if (channel == 0u) {
                    const twin::tf fused_control = conv_residual_oracle(
                        tap_values, tap_weights, gated[index],
                        initial_residual[index], true);
                    check(std::bit_cast<std::uint32_t>(ordered.value) == 0u,
                          label + " ordered product/add golden is +0");
                    check(std::bit_cast<std::uint32_t>(fused_control.value) != 0u,
                          label + " rejects fused product/add control");
                } else if (channel == 1u) {
                    check(std::bit_cast<std::uint32_t>(ordered.value) == 0u,
                          label + " ordered tap golden is +0");
                    check(std::bit_cast<std::uint32_t>(reorder_control.value) != 0u,
                          label + " rejects reordered tap control");
                } else {
                    const twin::tf regroup_control = conv_residual_oracle(
                        tap_values, tap_weights, gated[index],
                        initial_residual[index], false, {0u, 1u, 2u, 3u}, true);
                    check(std::bit_cast<std::uint32_t>(ordered.value) == 0u,
                          label + " ordered residual golden is +0");
                    check(std::bit_cast<std::uint32_t>(regroup_control.value) != 0u,
                          label + " rejects regrouped residual control");
                }
            }
        }
        for (std::uint32_t channel = 0u; channel < HC_DIM; ++channel) {
            const std::size_t history_offset =
                static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
            for (std::uint32_t history_row = 0u;
                 history_row + 1u < PLE_HISTORY_ROWS; ++history_row) {
                sequential_history[history_offset + history_row] =
                    sequential_history[history_offset + history_row + 1u];
            }
            sequential_history[history_offset + PLE_HISTORY_ROWS - 1u] =
                ple_norm[static_cast<std::size_t>(row) * HC_DIM + channel];
        }
    }

    std::string err;
    twin::GuardedBuffer norm_buffer;
    twin::GuardedBuffer history_buffer;
    twin::GuardedBuffer conv_buffer;
    twin::GuardedBuffer gated_buffer;
    twin::GuardedBuffer residual_buffer;
    const bool shared_encoder_case =
        rows == 1u && set == twin::InputSet::fixture && !arithmetic_probe;
    twin::GuardedBuffer second_residual_buffer;
    check(norm_buffer.allocate(context, label + " ple_norm", norm_bytes, err),
          label + " ple_norm buffer", err);
    check(history_buffer.allocate(context, label + " history", history_bytes, err),
          label + " history buffer", err);
    check(conv_buffer.allocate(context, label + " conv1d", weight_bytes, err),
          label + " conv1d buffer", err);
    check(gated_buffer.allocate(context, label + " gated", value_bytes, err),
          label + " gated buffer", err);
    check(residual_buffer.allocate(context, label + " residual", value_bytes, err),
          label + " residual buffer", err);
    if (shared_encoder_case) {
        check(second_residual_buffer.allocate(
                  context, label + " second residual", value_bytes, err),
              label + " second residual buffer", err);
    }
    if (!norm_buffer.view() || !history_buffer.view() || !conv_buffer.view() ||
        !gated_buffer.view() || !residual_buffer.view() ||
        (shared_encoder_case && !second_residual_buffer.view())) {
        return false;
    }
    std::memcpy(norm_buffer.payload().data(), ple_norm.data(), norm_bytes);
    std::memcpy(history_buffer.payload().data(), initial_history.data(),
                history_bytes);
    std::memcpy(conv_buffer.payload().data(), conv1d.data(), weight_bytes);
    std::memcpy(gated_buffer.payload().data(), gated.data(), value_bytes);
    std::memcpy(residual_buffer.payload().data(), initial_residual.data(),
                value_bytes);
    if (shared_encoder_case) {
        std::memcpy(second_residual_buffer.payload().data(),
                    initial_residual.data(), value_bytes);
    }
    norm_buffer.capture_before_dispatch();
    history_buffer.capture_before_dispatch();
    conv_buffer.capture_before_dispatch();
    gated_buffer.capture_before_dispatch();
    residual_buffer.capture_before_dispatch();
    if (shared_encoder_case) {
        second_residual_buffer.capture_before_dispatch();
    }

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder_case
        ? Encoder::encode(
              context, *command, "PLE conv-residual shared segment",
              [&](Encoder& segment, std::string& encode_error) {
                  if (!encode_ple_conv_residual(
                          context, segment, rows,
                          {read_only(context, norm_buffer.view(),
                                     label + " norm"),
                           read_only(context, history_buffer.view(),
                                     label + " history"),
                           read_only(context, conv_buffer.view(),
                                     label + " conv weights"),
                           read_only(context, gated_buffer.view(),
                                     label + " gated"),
                           residual_buffer.view()},
                          encode_error)) {
                      return false;
                  }
                  return encode_ple_conv_residual(
                      context, segment, rows,
                      {read_only(context, norm_buffer.view(),
                                 label + " second norm"),
                       read_only(context, history_buffer.view(),
                                 label + " second history"),
                       read_only(context, conv_buffer.view(),
                                 label + " second conv weights"),
                       read_only(context, gated_buffer.view(),
                                 label + " second gated"),
                       second_residual_buffer.view()},
                      encode_error);
              },
              err)
        : encode_ple_conv_residual(
              context, *command, rows,
              {read_only(context, norm_buffer.view(), label + " norm"),
               read_only(context, history_buffer.view(), label + " history"),
               read_only(context, conv_buffer.view(),
                         label + " conv weights"),
               read_only(context, gated_buffer.view(), label + " gated"),
               residual_buffer.view()},
              err);
    check(encoded,
          label + (shared_encoder_case
                       ? " production PLE conv-residual shared-segment encode"
                       : " production PLE conv-residual encode"),
          err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(residual_buffer.payload().data()),
        value_count);
    const auto mismatch = twin::compare_floats(
        actual, expected,
        {twin::Comparison::units, kConvToleranceUnits, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    double max_units = 0.0;
    for (std::size_t i = 0u; i < value_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(expected[i].value)) {
            ++exact_count;
        }
        const double bound =
            std::ldexp(expected[i].magnitude, -24) + std::ldexp(1.0, -126);
        max_units = std::max(
            max_units,
            std::fabs(static_cast<double>(actual[i]) - expected[i].value) /
                bound);
    }
    check(!mismatch, label + " U(16) sequential convolution result",
          "max observed U=" + std::to_string(max_units) +
              ", exact bits=" + std::to_string(exact_count) + "/" +
              std::to_string(value_count));
    if (arithmetic_probe) {
        check(std::bit_cast<std::uint32_t>(actual[0]) == 0u,
              label + " FMA-sensitive output remains +0");
        check(std::bit_cast<std::uint32_t>(actual[1]) == 0u,
              label + " tap-order-sensitive output remains +0");
        check(std::bit_cast<std::uint32_t>(actual[2]) == 0u,
              label + " residual-order-sensitive output remains +0");
    }
    check(!norm_buffer.first_unwritten_change({}),
          label + " ple_norm remains read-only");
    check(!history_buffer.first_unwritten_change({}),
          label + " history remains read-only");
    check(!conv_buffer.first_unwritten_change({}),
          label + " conv1d remains read-only");
    check(!gated_buffer.first_unwritten_change({}),
          label + " gated remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, value_bytes}}};
    const auto changed = residual_buffer.first_unwritten_change(written);
    check(!changed, label + " residual guards remain unchanged",
          changed ? changed->reason : "");
    bool second_matches = true;
    if (shared_encoder_case) {
        const auto second_actual = std::span<const float>(
            reinterpret_cast<const float*>(
                second_residual_buffer.payload().data()),
            value_count);
        const auto second_mismatch = twin::compare_floats(
            second_actual, expected,
            {twin::Comparison::units, kConvToleranceUnits, 0.0, 0u});
        check(!second_mismatch,
              label + " shared-segment second output matches oracle",
              second_mismatch ? second_mismatch->reason : "");
        check(std::memcmp(residual_buffer.payload().data(),
                          second_residual_buffer.payload().data(),
                          value_bytes) == 0,
              label + " repeated PLE conv-residual dispatch matches bitwise");
        const auto second_changed =
            second_residual_buffer.first_unwritten_change(written);
        check(!second_changed,
              label + " second residual guards remain unchanged",
              second_changed ? second_changed->reason : "");
        second_matches = !second_mismatch;
    }
    output_snapshot.assign(residual_buffer.payload().begin(),
                           residual_buffer.payload().begin() + value_bytes);
    return !mismatch && second_matches;
}

bool test_conv_residual_encoder_validation(
    Context& context, const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t row_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    constexpr std::size_t conv1d_bytes =
        static_cast<std::size_t>(HC_DIM) * 4u * sizeof(std::uint16_t);
    const std::size_t value_bytes = rows * row_bytes;

    std::string error;
    auto rejected_command = context.make_command_buffer(
        "PLE conv-residual rejected-input validation", error);
    if (!rejected_command) {
        check(false,
              "PLE conv-residual rejection-validation command buffer", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t row_count,
                            const PLEConvResidualBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !encode_ple_conv_residual(context, *rejected_command, row_count,
                                      buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const PLEConvResidualBuffers empty{};
    passed &= reject("PLE conv-residual encoder rejects zero rows",
                     0u, empty, "row count");
    passed &= reject("PLE conv-residual encoder rejects rows above eight",
                     9u, empty, "row count");

    BufferRef norm = context.buffers().allocate(
        "PLE conv-residual validation norm", value_bytes, error);
    BufferRef history = context.buffers().allocate(
        "PLE conv-residual validation history", history_bytes, error);
    BufferRef conv1d = context.buffers().allocate(
        "PLE conv-residual validation weights", conv1d_bytes, error);
    BufferRef gated = context.buffers().allocate(
        "PLE conv-residual validation gated", value_bytes, error);
    BufferRef residual = context.buffers().allocate(
        "PLE conv-residual validation residual", value_bytes, error);
    BufferRef alias_backing = context.buffers().allocate(
        "PLE conv-residual validation alias backing",
        history_bytes + sizeof(float), error);
    if (!norm || !history || !conv1d || !gated || !residual ||
        !alias_backing) {
        check(false, "PLE conv-residual encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const PLEConvResidualBuffers valid{
        read_only(context, norm, "validation norm"),
        read_only(context, history, "validation history"),
        read_only(context, conv1d, "validation conv weights"),
        read_only(context, gated, "validation gated"), residual};

    PLEConvResidualBuffers short_norm = valid;
    short_norm.ple_norm = valid.ple_norm.slice(0u, row_bytes);
    passed &= reject("PLE conv-residual encoder rejects one norm row for T=2",
                     rows, short_norm, "undersized");
    PLEConvResidualBuffers short_history = valid;
    short_history.history =
        valid.history.slice(0u, history_bytes - sizeof(float));
    passed &= reject("PLE conv-residual encoder rejects short history",
                     rows, short_history, "undersized");
    PLEConvResidualBuffers short_conv1d = valid;
    short_conv1d.conv1d = valid.conv1d.slice(
        0u, conv1d_bytes - sizeof(std::uint16_t));
    passed &= reject("PLE conv-residual encoder rejects short F16 weights",
                     rows, short_conv1d, "undersized");
    PLEConvResidualBuffers short_gated = valid;
    short_gated.gated = valid.gated.slice(0u, row_bytes);
    passed &= reject("PLE conv-residual encoder rejects one gated row for T=2",
                     rows, short_gated, "undersized");
    PLEConvResidualBuffers short_residual = valid;
    short_residual.residual.length = row_bytes;
    passed &= reject(
        "PLE conv-residual encoder rejects one residual row for T=2",
        rows, short_residual, "undersized");

    PLEConvResidualBuffers misaligned_norm = valid;
    misaligned_norm.ple_norm =
        valid.ple_norm.slice(1u, valid.ple_norm.length() - 1u);
    passed &= reject("PLE conv-residual encoder rejects misaligned norm",
                     rows, misaligned_norm, "misaligned");
    PLEConvResidualBuffers misaligned_history = valid;
    misaligned_history.history =
        valid.history.slice(1u, valid.history.length() - 1u);
    passed &= reject("PLE conv-residual encoder rejects misaligned history",
                     rows, misaligned_history, "misaligned");
    PLEConvResidualBuffers misaligned_conv1d = valid;
    misaligned_conv1d.conv1d =
        valid.conv1d.slice(1u, valid.conv1d.length() - 1u);
    passed &= reject(
        "PLE conv-residual encoder rejects misaligned F16 weights",
        rows, misaligned_conv1d, "misaligned");
    PLEConvResidualBuffers misaligned_gated = valid;
    misaligned_gated.gated =
        valid.gated.slice(1u, valid.gated.length() - 1u);
    passed &= reject("PLE conv-residual encoder rejects misaligned gated",
                     rows, misaligned_gated, "misaligned");
    PLEConvResidualBuffers misaligned_residual = valid;
    misaligned_residual.residual =
        residual.slice(1u, residual.length - 1u);
    passed &= reject("PLE conv-residual encoder rejects misaligned residual",
                     rows, misaligned_residual, "misaligned");

    PLEConvResidualBuffers out_of_range = valid;
    out_of_range.ple_norm =
        valid.ple_norm.slice(valid.ple_norm.length(), sizeof(float));
    passed &= reject("PLE conv-residual encoder rejects norm past allocation",
                     rows, out_of_range, "not owned by this Context");

    BufferRef alias_residual = alias_backing.slice(0u, value_bytes);
    const auto shifted_view = [](BufferRef view, std::size_t input_bytes,
                                 std::size_t delta) {
        return view.slice(delta, input_bytes);
    };
    const auto reject_alias = [&](const std::string& role, BufferRef input,
                                  std::size_t input_bytes,
                                  std::size_t partial_delta) {
        BufferRef exact_input = input.slice(0u, input_bytes);
        PLEConvResidualBuffers exact = valid;
        if (role == "ple_norm") {
            exact.ple_norm = read_only(context, exact_input, "exact norm");
        }
        else if (role == "history") {
            exact.history = read_only(context, exact_input, "exact history");
        }
        else if (role == "conv1d") {
            exact.conv1d = read_only(
                context, exact_input, "exact-alias conv weights");
        }
        else exact.gated = read_only(context, exact_input, "exact gated");
        exact.residual = alias_residual;
        passed &= reject("PLE conv-residual encoder rejects exact residual/" +
                             role + " overlap",
                         rows, exact, "must not overlap");

        BufferRef partial_input =
            shifted_view(alias_backing, input_bytes, partial_delta);
        PLEConvResidualBuffers partial = valid;
        if (role == "ple_norm") {
            partial.ple_norm =
                read_only(context, partial_input, "partial norm");
        }
        else if (role == "history") {
            partial.history =
                read_only(context, partial_input, "partial history");
        }
        else if (role == "conv1d") {
            partial.conv1d = read_only(
                context, partial_input, "partial-alias conv weights");
        }
        else partial.gated = read_only(context, partial_input, "partial gated");
        partial.residual = alias_residual;
        passed &= reject("PLE conv-residual encoder rejects partial residual/" +
                             role + " overlap",
                         rows, partial, "must not overlap");
    };
    reject_alias("ple_norm", alias_backing, value_bytes, sizeof(float));
    reject_alias("history", alias_backing, history_bytes, sizeof(float));
    reject_alias("conv1d", alias_backing, conv1d_bytes,
                 sizeof(std::uint16_t));
    reject_alias("gated", alias_backing, value_bytes, sizeof(float));

    BufferRef missing_metadata_norm = alias_backing;
    missing_metadata_norm.length = value_bytes;
    missing_metadata_norm.contents = nullptr;
    missing_metadata_norm.gpu_address = 0u;
    const ReadOnlyBufferRef forged_read_only =
        context.buffers().read_only(missing_metadata_norm);
    check(!forged_read_only,
          "PLE conv-residual rejects forged read-only buffer metadata");
    PLEConvResidualBuffers missing_metadata_alias{
        forged_read_only, valid.history,
        read_only(context, conv1d, "alias-test conv weights"), valid.gated,
        alias_residual};
    passed &= reject(
        "PLE conv-residual rejects forged CPU/GPU metadata",
        rows, missing_metadata_alias, "not owned by this Context");
    rejected_command->discard();

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    if (!foreign_context) {
        check(false, "PLE conv-residual foreign Context", error);
        return false;
    }
    BufferRef foreign_norm = foreign_context->buffers().allocate(
        "PLE conv-residual foreign-context norm", value_bytes, error);
    if (!foreign_norm) {
        check(false, "PLE conv-residual foreign-context buffer", error);
        return false;
    }
    PLEConvResidualBuffers foreign_buffers = valid;
    foreign_buffers.ple_norm =
        read_only(*foreign_context, foreign_norm, "foreign norm");
    auto foreign_buffer_command = context.make_command_buffer(
        "PLE conv-residual foreign-buffer validation", error);
    if (!foreign_buffer_command) {
        check(false, "PLE conv-residual foreign-buffer command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_buffer =
        !encode_ple_conv_residual(context, *foreign_buffer_command, rows,
                                  foreign_buffers, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejected_foreign_buffer,
          "PLE conv-residual encoder rejects foreign-context buffer", error);
    foreign_buffer_command->discard();
    passed &= rejected_foreign_buffer;

    auto foreign_command = foreign_context->make_command_buffer(
        "PLE conv-residual foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "PLE conv-residual foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_command =
        !encode_ple_conv_residual(context, *foreign_command, rows, valid,
                                  error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "PLE conv-residual encoder rejects foreign command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "PLE conv-residual submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "PLE conv-residual command-buffer submission", error);
        return false;
    }
    error.clear();
    const bool rejected_submitted =
        !encode_ple_conv_residual(context, *submitted, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "PLE conv-residual encoder rejects submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "PLE conv-residual validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "PLE conv-residual enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "PLE conv-residual command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    error.clear();
    const bool rejected_enqueued =
        !encode_ple_conv_residual(context, *enqueued, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "PLE conv-residual encoder rejects enqueued command buffer", error);
    const bool enqueued_completed =
        submit_and_wait(enqueued, "PLE conv-residual enqueued validation",
                        error);
    check(enqueued_completed,
          "PLE conv-residual enqueued command buffer drains", error);
    passed &= rejected_enqueued && enqueued_completed;

    BufferRef shared_read_only = context.buffers().allocate(
        "PLE conv-residual overlapping read-only inputs", history_bytes, error);
    BufferRef separate_residual = context.buffers().allocate(
        "PLE conv-residual overlapping-input output", value_bytes, error);
    if (!shared_read_only || !separate_residual) {
        check(false, "PLE conv-residual overlapping-input test buffers", error);
        return false;
    }
    std::memset(shared_read_only.contents, 0, shared_read_only.length);
    std::memset(separate_residual.contents, 0, separate_residual.length);
    const std::vector<std::uint8_t> read_only_snapshot(
        static_cast<const std::uint8_t*>(shared_read_only.contents),
        static_cast<const std::uint8_t*>(shared_read_only.contents) +
            shared_read_only.length);
    auto read_only_command = context.make_command_buffer(
        "PLE conv-residual overlapping read-only inputs", error);
    if (!read_only_command) {
        check(false,
              "PLE conv-residual overlapping-input command buffer", error);
        return false;
    }
    const ReadOnlyBufferRef shared_view =
        read_only(context, shared_read_only, "shared read-only input");
    PLEConvResidualBuffers read_only_alias{
        shared_view, shared_view, shared_view, shared_view, separate_residual};
    const bool accepted_read_only_overlap = encode_ple_conv_residual(
        context, *read_only_command, rows, read_only_alias, error);
    check(accepted_read_only_overlap,
          "PLE conv-residual permits overlapping read-only inputs", error);
    passed &= accepted_read_only_overlap;
    if (!accepted_read_only_overlap) {
        read_only_command->discard();
        return false;
    }
    const bool read_only_dispatched =
        submit_and_wait(read_only_command,
                        "PLE conv-residual overlapping-input test", error);
    check(read_only_dispatched,
          "PLE conv-residual overlapping-input command completes", error);
    passed &= read_only_dispatched;
    const bool read_only_unchanged =
        std::memcmp(shared_read_only.contents, read_only_snapshot.data(),
                    read_only_snapshot.size()) == 0;
    check(read_only_unchanged,
          "PLE conv-residual overlapping inputs remain unchanged");
    passed &= read_only_unchanged;
    return passed;
}

bool run_case(Context& context, std::uint32_t rows, twin::InputSet set,
              const twin::Options& options,
              bool repeat_first_row,
              std::vector<std::uint8_t>& output_snapshot) {
    const std::string label = "ple_broadcast rows=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set);
    const std::size_t value_count = static_cast<std::size_t>(rows) * N_EMBD;
    const std::size_t gate_count = static_cast<std::size_t>(rows) * HC;
    const std::size_t output_count = static_cast<std::size_t>(rows) * HC_DIM;
    const std::size_t value_bytes = value_count * sizeof(float);
    const std::size_t gate_bytes = gate_count * sizeof(float);
    const std::size_t output_bytes = output_count * sizeof(float);

    std::vector<float> value(value_count);
    std::vector<float> gate(gate_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        const std::uint32_t input_row = repeat_first_row ? 0u : row;
        std::mt19937_64 value_rng(options.seed ^ 0x504c4556414c5545ull ^
                                  input_row);
        std::mt19937_64 gate_rng(options.seed ^ 0x504c454741544521ull ^
                                 input_row);
        for (std::uint32_t col = 0; col < N_EMBD; ++col) {
            value[static_cast<std::size_t>(row) * N_EMBD + col] =
                value_input(set, input_row, col, value_rng);
        }
        for (std::uint32_t stream = 0; stream < HC; ++stream) {
            gate[static_cast<std::size_t>(row) * HC + stream] =
                gate_input(set, input_row, stream, gate_rng);
        }
    }

    std::vector<twin::tf> expected(output_count);
    for (std::uint32_t row = 0; row < rows; ++row) {
        for (std::uint32_t index = 0; index < HC_DIM; ++index) {
            const std::uint32_t stream = index / N_EMBD;
            const std::uint32_t col = index % N_EMBD;
            const std::size_t input_index =
                static_cast<std::size_t>(row) * N_EMBD + col;
            const std::size_t gate_index =
                static_cast<std::size_t>(row) * HC + stream;
            const std::size_t output_index =
                static_cast<std::size_t>(row) * HC_DIM + index;
            expected[output_index] =
                twin::tf{value[input_index] * gate[gate_index]};
        }
    }

    std::string err;
    twin::GuardedBuffer value_buffer;
    twin::GuardedBuffer gate_buffer;
    twin::GuardedBuffer output_buffer;
    const bool shared_encoder_case =
        rows == 1u && set == twin::InputSet::fixture && !repeat_first_row;
    twin::GuardedBuffer second_output_buffer;
    check(value_buffer.allocate(context, label + " value", value_bytes, err),
          label + " value buffer", err);
    check(gate_buffer.allocate(context, label + " gate", gate_bytes, err),
          label + " gate buffer", err);
    check(output_buffer.allocate(context, label + " gated", output_bytes, err),
          label + " output buffer", err);
    if (shared_encoder_case) {
        check(second_output_buffer.allocate(
                  context, label + " second gated", output_bytes, err),
              label + " second output buffer", err);
    }
    if (!value_buffer.view() || !gate_buffer.view() || !output_buffer.view() ||
        (shared_encoder_case && !second_output_buffer.view())) {
        return false;
    }
    std::memcpy(value_buffer.payload().data(), value.data(), value_bytes);
    std::memcpy(gate_buffer.payload().data(), gate.data(), gate_bytes);
    check(output_buffer.poison_floats_against(
              expected, twin::Comparison::bits, err),
          label + " output poison", err);
    if (shared_encoder_case) {
        check(second_output_buffer.poison_floats_against(
                  expected, twin::Comparison::bits, err),
              label + " second output poison", err);
    }
    value_buffer.capture_before_dispatch();
    gate_buffer.capture_before_dispatch();
    output_buffer.capture_before_dispatch();
    if (shared_encoder_case) second_output_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder_case
        ? Encoder::encode(
              context, *command, "PLE broadcast shared segment",
              [&](Encoder& segment, std::string& encode_error) {
                  if (!encode_ple_broadcast(
                          context, segment, rows,
                          {value_buffer.view(), gate_buffer.view(),
                           output_buffer.view()},
                          encode_error)) {
                      return false;
                  }
                  return encode_ple_broadcast(
                      context, segment, rows,
                      {value_buffer.view(), gate_buffer.view(),
                       second_output_buffer.view()},
                      encode_error);
              },
              err)
        : encode_ple_broadcast(
              context, *command, rows,
              {value_buffer.view(), gate_buffer.view(),
               output_buffer.view()},
              err);
    check(encoded,
          label + (shared_encoder_case
                       ? " production PLE broadcast shared-segment encode"
                       : " production PLE broadcast encode"),
          err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(output_buffer.payload().data()), output_count);
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::bits, 0.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, rows, *mismatch);
    std::size_t exact_count = 0u;
    for (std::size_t i = 0; i < output_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(expected[i].value)) {
            ++exact_count;
        }
    }
    check(!mismatch, label + " bit-exact broadcast",
          "exact bits=" + std::to_string(exact_count) + "/" +
              std::to_string(output_count));
    check(!value_buffer.first_unwritten_change({}),
          label + " value remains read-only");
    check(!gate_buffer.first_unwritten_change({}),
          label + " gate remains read-only");
    const std::array<twin::ByteRange, 1> written{{{0u, output_bytes}}};
    const auto changed = output_buffer.first_unwritten_change(written);
    check(!changed, label + " output guards remain unchanged",
          changed ? changed->reason : "");
    if (shared_encoder_case) {
        const auto second_actual = std::span<const float>(
            reinterpret_cast<const float*>(
                second_output_buffer.payload().data()),
            output_count);
        const auto second_mismatch = twin::compare_floats(
            second_actual, expected,
            {twin::Comparison::bits, 0.0, 0.0, 0u});
        check(!second_mismatch,
              label + " shared-segment second output matches oracle",
              second_mismatch ? second_mismatch->reason : "");
        check(std::memcmp(output_buffer.payload().data(),
                          second_output_buffer.payload().data(),
                          output_bytes) == 0,
              label + " repeated PLE broadcast dispatch matches bitwise");
        const auto second_changed =
            second_output_buffer.first_unwritten_change(written);
        check(!second_changed,
              label + " second output guards remain unchanged",
              second_changed ? second_changed->reason : "");
    }
    output_snapshot.assign(output_buffer.payload().begin(),
                           output_buffer.payload().begin() + output_bytes);
    return !mismatch;
}

bool test_broadcast_encoder_validation(Context& context,
                                      const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t value_row_bytes =
        static_cast<std::size_t>(N_EMBD) * sizeof(float);
    constexpr std::size_t gate_row_bytes =
        static_cast<std::size_t>(HC) * sizeof(float);
    constexpr std::size_t output_row_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t value_bytes = rows * value_row_bytes;
    const std::size_t gate_bytes = rows * gate_row_bytes;
    const std::size_t output_bytes = rows * output_row_bytes;

    std::string error;
    auto rejected_command = context.make_command_buffer(
        "PLE broadcast rejected-input validation", error);
    if (!rejected_command) {
        check(false, "PLE broadcast rejection-validation command buffer", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t row_count,
                            const PLEBroadcastBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !encode_ple_broadcast(context, *rejected_command, row_count,
                                  buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const PLEBroadcastBuffers empty{};
    passed &= reject("PLE broadcast encoder rejects zero rows", 0u, empty,
                     "row count");
    passed &= reject("PLE broadcast encoder rejects rows above eight", 9u,
                     empty, "row count");

    BufferRef value = context.buffers().allocate(
        "PLE broadcast validation value", value_bytes, error);
    BufferRef gate = context.buffers().allocate(
        "PLE broadcast validation gate", gate_bytes, error);
    BufferRef output = context.buffers().allocate(
        "PLE broadcast validation output", output_bytes, error);
    BufferRef output_alias_backing = context.buffers().allocate(
        "PLE broadcast validation output alias", output_bytes, error);
    if (!value || !gate || !output || !output_alias_backing) {
        check(false, "PLE broadcast encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const PLEBroadcastBuffers valid{value, gate, output};

    PLEBroadcastBuffers short_value = valid;
    short_value.value.length = value_row_bytes;
    passed &= reject("PLE broadcast encoder rejects one-row value for T=2",
                     rows, short_value, "undersized");
    PLEBroadcastBuffers short_gate = valid;
    short_gate.gate.length = gate_row_bytes;
    passed &= reject("PLE broadcast encoder rejects one-row gate for T=2",
                     rows, short_gate, "undersized");
    PLEBroadcastBuffers short_output = valid;
    short_output.gated.length = output_row_bytes;
    passed &= reject("PLE broadcast encoder rejects one-row output for T=2",
                     rows, short_output, "undersized");

    PLEBroadcastBuffers misaligned_value = valid;
    misaligned_value.value =
        value.slice(1u, value.length - 1u);
    passed &= reject("PLE broadcast encoder rejects misaligned value",
                     rows, misaligned_value, "misaligned");
    PLEBroadcastBuffers misaligned_gate = valid;
    misaligned_gate.gate =
        gate.slice(1u, gate.length - 1u);
    passed &= reject("PLE broadcast encoder rejects misaligned gate",
                     rows, misaligned_gate, "misaligned");
    PLEBroadcastBuffers misaligned_output = valid;
    misaligned_output.gated =
        output.slice(1u, output.length - 1u);
    passed &= reject("PLE broadcast encoder rejects misaligned output",
                     rows, misaligned_output, "misaligned");

    PLEBroadcastBuffers out_of_range = valid;
    out_of_range.gated = output.slice(output.length, sizeof(float));
    passed &= reject("PLE broadcast encoder rejects view past allocation",
                     rows, out_of_range, "not owned by this Context");

    BufferRef alias_value = output_alias_backing.slice(0u, value_bytes);
    PLEBroadcastBuffers output_value_alias{
        alias_value, gate, output_alias_backing};
    passed &= reject("PLE broadcast encoder rejects output/value overlap",
                     rows, output_value_alias, "must not overlap");
    BufferRef missing_metadata_value = alias_value;
    missing_metadata_value.contents = nullptr;
    missing_metadata_value.gpu_address = 0u;
    PLEBroadcastBuffers missing_metadata_value_alias{
        missing_metadata_value, gate, output_alias_backing};
    passed &= reject(
        "PLE broadcast rejects forged CPU/GPU metadata",
        rows, missing_metadata_value_alias, "not owned by this Context");
    BufferRef partial_value_alias =
        output_alias_backing.slice(sizeof(float), value_bytes);
    PLEBroadcastBuffers partial_output_value_alias{
        partial_value_alias, gate, output_alias_backing};
    passed &= reject(
        "PLE broadcast encoder rejects partial output/value overlap",
        rows, partial_output_value_alias, "must not overlap");
    BufferRef alias_gate = output_alias_backing.slice(0u, gate_bytes);
    PLEBroadcastBuffers output_gate_alias{
        value, alias_gate, output_alias_backing};
    passed &= reject("PLE broadcast encoder rejects output/gate overlap",
                     rows, output_gate_alias, "must not overlap");
    BufferRef partial_gate_alias =
        output_alias_backing.slice(sizeof(float), gate_bytes);
    PLEBroadcastBuffers partial_output_gate_alias{
        value, partial_gate_alias, output_alias_backing};
    passed &= reject(
        "PLE broadcast encoder rejects partial output/gate overlap",
        rows, partial_output_gate_alias, "must not overlap");
    rejected_command->discard();

    BufferRef foreign_value;
    std::unique_ptr<CommandBuffer> foreign_command;
    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    if (!foreign_context) {
        check(false, "PLE broadcast foreign Context", error);
        return false;
    }
    foreign_value = foreign_context->buffers().allocate(
        "PLE broadcast foreign-context value", value_bytes, error);
    if (!foreign_value) {
        check(false, "PLE broadcast foreign-context buffer", error);
        return false;
    }
    PLEBroadcastBuffers foreign_buffers = valid;
    foreign_buffers.value = foreign_value;
    auto foreign_rejected_command = context.make_command_buffer(
        "PLE broadcast foreign-buffer validation", error);
    if (!foreign_rejected_command) {
        check(false, "PLE broadcast foreign-buffer command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_buffer =
        !encode_ple_broadcast(context, *foreign_rejected_command, rows,
                              foreign_buffers, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejected_foreign_buffer,
          "PLE broadcast encoder rejects foreign-context buffers", error);
    foreign_rejected_command->discard();
    passed &= rejected_foreign_buffer;

    foreign_command = foreign_context->make_command_buffer(
        "PLE broadcast foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "PLE broadcast foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_command =
        !encode_ple_broadcast(context, *foreign_command, rows, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "PLE broadcast encoder rejects foreign command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "PLE broadcast submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "PLE broadcast command-buffer submission", error);
        return false;
    }
    error.clear();
    const bool rejected_submitted =
        !encode_ple_broadcast(context, *submitted, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "PLE broadcast encoder rejects submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "PLE broadcast validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "PLE broadcast enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "PLE broadcast command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    error.clear();
    const bool rejected_enqueued =
        !encode_ple_broadcast(context, *enqueued, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "PLE broadcast encoder rejects enqueued command buffer", error);
    const bool enqueued_completed =
        submit_and_wait(enqueued, "PLE broadcast enqueued validation", error);
    check(enqueued_completed,
          "PLE broadcast enqueued command buffer drains", error);
    passed &= rejected_enqueued && enqueued_completed;

    BufferRef shared_input = context.buffers().allocate(
        "PLE broadcast overlapping read-only inputs", value_row_bytes * rows,
        error);
    BufferRef alias_output = context.buffers().allocate(
        "PLE broadcast overlapping-input output", output_bytes, error);
    if (!shared_input || !alias_output) {
        check(false, "PLE broadcast overlapping-input test buffers", error);
        return false;
    }
    std::memset(shared_input.contents, 0, shared_input.length);
    auto alias_command = context.make_command_buffer(
        "PLE broadcast overlapping read-only inputs", error);
    if (!alias_command) {
        check(false, "PLE broadcast overlapping-input command buffer", error);
        return false;
    }
    const bool accepted_read_only_overlap = encode_ple_broadcast(
        context, *alias_command, rows,
        {shared_input, shared_input, alias_output}, error);
    check(accepted_read_only_overlap,
          "PLE broadcast permits overlapping read-only inputs", error);
    passed &= accepted_read_only_overlap;
    if (!accepted_read_only_overlap) {
        alias_command->discard();
        return false;
    }
    const bool alias_dispatched =
        submit_and_wait(alias_command, "PLE broadcast overlapping-input test",
                        error);
    check(alias_dispatched,
          "PLE broadcast overlapping-input command completes", error);
    passed &= alias_dispatched;
    return passed;
}

bool test_history_encoder_validation(Context& context,
                                     const std::string& metallib_path) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::size_t history_bytes =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS * sizeof(float);
    constexpr std::size_t norm_row_bytes =
        static_cast<std::size_t>(HC_DIM) * sizeof(float);
    const std::size_t norm_bytes = rows * norm_row_bytes;

    std::string error;
    auto rejected_command = context.make_command_buffer(
        "PLE history rejected-input validation", error);
    if (!rejected_command) {
        check(false, "PLE history rejection-validation command buffer", error);
        return false;
    }
    const auto reject = [&](const std::string& label, std::uint32_t n,
                            const PLEHistoryBuffers& buffers,
                            const std::string& expected_error) {
        error.clear();
        const bool rejected =
            !encode_ple_history(context, *rejected_command, n, buffers, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, label, error);
        return rejected;
    };

    bool passed = true;
    const PLEHistoryBuffers empty{};
    passed &= reject("PLE history encoder rejects signed-index overflow",
                     0x80000000u, empty, "signed Metal index range");

    BufferRef history = context.buffers().allocate(
        "PLE history validation history", history_bytes, error);
    BufferRef norm = context.buffers().allocate(
        "PLE history validation norm", norm_bytes, error);
    constexpr std::uint32_t long_rows = 17u;
    BufferRef long_norm = context.buffers().allocate(
        "PLE history validation long norm",
        static_cast<std::size_t>(long_rows) * norm_row_bytes, error);
    BufferRef overlap_backing = context.buffers().allocate(
        "PLE history validation overlap backing", history_bytes, error);
    if (!history || !norm || !long_norm || !overlap_backing) {
        check(false, "PLE history encoder validation buffers", error);
        rejected_command->discard();
        return false;
    }
    const PLEHistoryBuffers valid{history, norm};

    PLEHistoryBuffers short_history = valid;
    short_history.history.length -= sizeof(float);
    passed &= reject("PLE history encoder rejects short history",
                     rows, short_history, "undersized");
    PLEHistoryBuffers short_norm = valid;
    short_norm.ple_norm.length = norm_row_bytes;
    passed &= reject("PLE history encoder rejects one norm row for n=2",
                     rows, short_norm, "undersized");
    PLEHistoryBuffers short_long_norm{history, long_norm};
    short_long_norm.ple_norm.length =
        static_cast<std::size_t>(long_rows - 1u) * norm_row_bytes;
    passed &= reject(
        "PLE history encoder rejects 16 norm rows for n=17",
        long_rows, short_long_norm, "undersized");

    PLEHistoryBuffers misaligned_history = valid;
    misaligned_history.history =
        history.slice(1u, history.length - 1u);
    passed &= reject("PLE history encoder rejects misaligned history",
                     rows, misaligned_history, "misaligned");
    PLEHistoryBuffers misaligned_norm = valid;
    misaligned_norm.ple_norm =
        norm.slice(1u, norm.length - 1u);
    passed &= reject("PLE history encoder rejects misaligned ple_norm",
                     rows, misaligned_norm, "misaligned");

    PLEHistoryBuffers out_of_range = valid;
    out_of_range.ple_norm = norm.slice(norm.length, sizeof(float));
    passed &= reject("PLE history encoder rejects norm past allocation",
                     rows, out_of_range, "not owned by this Context");

    PLEHistoryBuffers exact_overlap{
        overlap_backing, overlap_backing};
    passed &= reject("PLE history encoder rejects history/norm overlap",
                     rows, exact_overlap, "must not overlap");
    BufferRef missing_metadata_norm = overlap_backing;
    missing_metadata_norm.length = norm_bytes;
    missing_metadata_norm.contents = nullptr;
    missing_metadata_norm.gpu_address = 0u;
    PLEHistoryBuffers missing_metadata_overlap{
        overlap_backing, missing_metadata_norm};
    passed &= reject(
        "PLE history rejects forged CPU/GPU metadata",
        rows, missing_metadata_overlap, "not owned by this Context");
    BufferRef partial_norm =
        overlap_backing.slice(sizeof(float), norm_bytes);
    PLEHistoryBuffers partial_overlap{overlap_backing, partial_norm};
    passed &= reject("PLE history encoder rejects partial history/norm overlap",
                     rows, partial_overlap, "must not overlap");
    rejected_command->discard();

    ContextOptions foreign_options;
    foreign_options.metallib = metallib_path;
    foreign_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto foreign_context = Context::open(foreign_options, error);
    if (!foreign_context) {
        check(false, "PLE history foreign Context", error);
        return false;
    }
    BufferRef foreign_norm = foreign_context->buffers().allocate(
        "PLE history foreign-context norm", norm_bytes, error);
    if (!foreign_norm) {
        check(false, "PLE history foreign-context buffer", error);
        return false;
    }
    PLEHistoryBuffers foreign_buffers = valid;
    foreign_buffers.ple_norm = foreign_norm;
    auto foreign_buffer_command = context.make_command_buffer(
        "PLE history foreign-buffer validation", error);
    if (!foreign_buffer_command) {
        check(false, "PLE history foreign-buffer command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_buffer =
        !encode_ple_history(context, *foreign_buffer_command, rows,
                            foreign_buffers, error) &&
        error.find("not owned by this Context") != std::string::npos;
    check(rejected_foreign_buffer,
          "PLE history encoder rejects foreign-context buffers", error);
    foreign_buffer_command->discard();
    passed &= rejected_foreign_buffer;

    auto foreign_command = foreign_context->make_command_buffer(
        "PLE history foreign command-buffer validation", error);
    if (!foreign_command) {
        check(false, "PLE history foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected_foreign_command =
        !encode_ple_history(context, *foreign_command, rows, valid, error) &&
        error.find("belongs to a different Context") != std::string::npos;
    check(rejected_foreign_command,
          "PLE history encoder rejects foreign command buffer", error);
    foreign_command->discard();
    passed &= rejected_foreign_command;

    auto submitted = context.make_command_buffer(
        "PLE history submitted-buffer validation", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "PLE history command-buffer submission", error);
        return false;
    }
    error.clear();
    const bool rejected_submitted =
        !encode_ple_history(context, *submitted, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_submitted,
          "PLE history encoder rejects submitted command buffer", error);
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native != nil) [submitted_native waitUntilCompleted];
    const bool submitted_completed =
        submitted_native != nil &&
        submitted_native.status == MTLCommandBufferStatusCompleted;
    check(submitted_completed,
          "PLE history validation command buffer completes");
    passed &= rejected_submitted && submitted_completed;

    auto enqueued = context.make_command_buffer(
        "PLE history enqueued-buffer validation", error);
    if (!enqueued || !enqueued->enqueue(error)) {
        check(false, "PLE history command-buffer enqueue", error);
        if (enqueued) enqueued->discard();
        return false;
    }
    error.clear();
    const bool rejected_enqueued =
        !encode_ple_history(context, *enqueued, rows, valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(rejected_enqueued,
          "PLE history encoder rejects enqueued command buffer", error);
    const bool enqueued_completed =
        submit_and_wait(enqueued, "PLE history enqueued validation", error);
    check(enqueued_completed,
          "PLE history enqueued command buffer drains", error);
    passed &= rejected_enqueued && enqueued_completed;

    constexpr std::size_t empty_norm_offset = 17u * sizeof(float);
    BufferRef empty_norm = history.slice(empty_norm_offset, 0u);
    auto zero_command = context.make_command_buffer(
        "PLE history zero-row empty-norm validation", error);
    if (!zero_command) {
        check(false, "PLE history zero-row command buffer", error);
        return false;
    }
    const bool accepts_zero_rows = encode_ple_history(
        context, *zero_command, 0u, {history, empty_norm}, error);
    check(accepts_zero_rows,
          "PLE history encoder accepts n=0 with an empty aliased norm view",
          error);
    passed &= accepts_zero_rows;
    if (!accepts_zero_rows) {
        zero_command->discard();
        return false;
    }
    const bool zero_completed =
        submit_and_wait(zero_command, "PLE history zero-row validation", error);
    check(zero_completed, "PLE history zero-row command completes", error);
    passed &= zero_completed;
    auto zero_segment_command = context.make_command_buffer(
        "PLE history zero-row shared-segment validation", error);
    if (!zero_segment_command) {
        check(false, "PLE history zero-row shared-segment command buffer",
              error);
        return false;
    }
    const bool accepts_zero_segment = Encoder::encode(
        context, *zero_segment_command, "PLE history zero-row segment",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_ple_history(context, encoder, 0u,
                                      {history, empty_norm}, encode_error);
        },
        error);
    check(accepts_zero_segment,
          "PLE history shared segment accepts n=0 with an empty norm view",
          error);
    passed &= accepts_zero_segment;
    if (!accepts_zero_segment) {
        zero_segment_command->discard();
        return false;
    }
    const bool zero_segment_completed =
        submit_and_wait(zero_segment_command,
                        "PLE history zero-row shared segment", error);
    check(zero_segment_completed,
          "PLE history zero-row shared segment completes", error);
    passed &= zero_segment_completed;
    return passed;
}

bool run_history_case(Context& context, std::uint32_t n,
                      twin::InputSet set, const twin::Options& options) {
    const bool shared_encoder_case =
        n == 1u && set == twin::InputSet::fixture;
    const std::string label = "ple_history n=" + std::to_string(n) +
        " set=" + twin::input_set_name(set) +
        (shared_encoder_case ? " shared-segment-rebind" : "");
    const std::size_t history_count =
        static_cast<std::size_t>(HC_DIM) * PLE_HISTORY_ROWS;
    const std::size_t allocated_norm_rows =
        shared_encoder_case ? 3u : std::max<std::uint32_t>(n, 1u);
    const std::size_t norm_count =
        allocated_norm_rows * static_cast<std::size_t>(HC_DIM);
    const std::size_t history_bytes = history_count * sizeof(float);
    const std::size_t norm_bytes = norm_count * sizeof(float);

    std::vector<float> initial_history(history_count);
    std::vector<float> ple_norm(norm_count);
    std::mt19937_64 rng(options.seed ^ 0x504c45484953544full ^
                        (static_cast<std::uint64_t>(n) << 32u));
    for (std::uint32_t channel = 0; channel < HC_DIM; ++channel) {
        for (std::uint32_t row = 0; row < PLE_HISTORY_ROWS; ++row) {
            initial_history[static_cast<std::size_t>(channel) *
                                PLE_HISTORY_ROWS + row] =
                value_input(set, channel, row, rng);
        }
    }
    for (std::uint32_t row = 0; row < allocated_norm_rows; ++row) {
        for (std::uint32_t channel = 0; channel < HC_DIM; ++channel) {
            ple_norm[static_cast<std::size_t>(row) * HC_DIM + channel] =
                value_input(set, row, channel, rng);
        }
    }

    const auto advance = [&](std::vector<float>& state, std::uint32_t first,
                             std::uint32_t count) {
        for (std::uint32_t step = first; step < first + count; ++step) {
            for (std::uint32_t channel = 0; channel < HC_DIM; ++channel) {
                const std::size_t history_offset =
                    static_cast<std::size_t>(channel) * PLE_HISTORY_ROWS;
                for (std::uint32_t row = 0;
                     row + 1u < PLE_HISTORY_ROWS; ++row) {
                    state[history_offset + row] =
                        state[history_offset + row + 1u];
                }
                state[history_offset + PLE_HISTORY_ROWS - 1u] =
                    ple_norm[static_cast<std::size_t>(step) * HC_DIM +
                             channel];
            }
        }
    };
    std::vector<float> expected = initial_history;
    advance(expected, 0u, n);
    std::vector<float> shared_expected = expected;
    if (shared_encoder_case) advance(shared_expected, 1u, 2u);
    std::vector<twin::tf> expected_oracle(history_count);
    for (std::size_t i = 0; i < history_count; ++i) {
        expected_oracle[i] = twin::tf{expected[i]};
    }
    std::vector<twin::tf> shared_expected_oracle;
    if (shared_encoder_case) {
        shared_expected_oracle.resize(history_count);
        for (std::size_t i = 0; i < history_count; ++i) {
            shared_expected_oracle[i] = twin::tf{shared_expected[i]};
        }
    }

    std::string err;
    twin::GuardedBuffer history_buffer;
    twin::GuardedBuffer second_history_buffer;
    twin::GuardedBuffer norm_buffer;
    check(history_buffer.allocate(context, label + " history",
                                  history_bytes, err),
          label + " history buffer", err);
    if (shared_encoder_case) {
        check(second_history_buffer.allocate(
                  context, label + " second history", history_bytes, err),
              label + " second history buffer", err);
    }
    check(norm_buffer.allocate(context, label + " ple_norm", norm_bytes, err),
          label + " ple_norm buffer", err);
    if (!history_buffer.view() || !norm_buffer.view() ||
        (shared_encoder_case && !second_history_buffer.view())) {
        return false;
    }
    std::memcpy(history_buffer.payload().data(),
                initial_history.data(), history_bytes);
    if (shared_encoder_case) {
        std::memcpy(second_history_buffer.payload().data(),
                    initial_history.data(), history_bytes);
    }
    std::memcpy(norm_buffer.payload().data(), ple_norm.data(), norm_bytes);
    history_buffer.capture_before_dispatch();
    if (shared_encoder_case) second_history_buffer.capture_before_dispatch();
    norm_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, err);
    if (!command) {
        check(false, label + " command buffer", err);
        return false;
    }
    const bool encoded = shared_encoder_case
        ? Encoder::encode(
              context, *command, "PLE history shared-segment rebinding",
              [&](Encoder& segment, std::string& encode_error) {
                  const std::size_t norm_row_bytes =
                      static_cast<std::size_t>(HC_DIM) * sizeof(float);
                  const BufferRef first_norm_row =
                      norm_buffer.view().slice(0u, norm_row_bytes);
                  const BufferRef next_two_norm_rows =
                      norm_buffer.view().slice(norm_row_bytes,
                                               2u * norm_row_bytes);
                  if (!encode_ple_history(
                          context, segment, 1u,
                          {history_buffer.view(), first_norm_row},
                          encode_error)) {
                      return false;
                  }
                  if (!encode_ple_history(
                          context, segment, 2u,
                          {history_buffer.view(), next_two_norm_rows},
                          encode_error)) {
                      return false;
                  }
                  return encode_ple_history(
                      context, segment, 1u,
                      {second_history_buffer.view(), first_norm_row},
                      encode_error);
              },
              err)
        : encode_ple_history(
              context, *command, n,
              {history_buffer.view(), norm_buffer.view()}, err);
    check(encoded,
          label + (shared_encoder_case
                       ? " production PLE history shared-segment encode"
                       : " production PLE history encode"),
          err);
    if (!encoded) {
        command->discard();
        return false;
    }
    const bool dispatched = submit_and_wait(command, label, err);
    check(dispatched, label + " Metal dispatch", err);
    if (!dispatched) return false;

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(history_buffer.payload().data()),
        history_count);
    const std::vector<twin::tf>& actual_expected =
        shared_encoder_case ? shared_expected_oracle : expected_oracle;
    const auto mismatch = twin::compare_floats(
        actual, actual_expected,
        {twin::Comparison::bits, 0.0, 0.0, 0u});
    if (mismatch) twin::report_failure(label.c_str(), options, n, *mismatch);
    std::size_t exact_count = 0u;
    for (std::size_t i = 0; i < history_count; ++i) {
        if (std::bit_cast<std::uint32_t>(actual[i]) ==
            std::bit_cast<std::uint32_t>(actual_expected[i].value)) {
            ++exact_count;
        }
    }
    check(!mismatch, label + " bit-exact sequential history",
          "exact bits=" + std::to_string(exact_count) + "/" +
              std::to_string(history_count));
    const std::array<twin::ByteRange, 1> written{{{0u, history_bytes}}};
    const auto changed = history_buffer.first_unwritten_change(written);
    check(!changed, label + " history guards remain unchanged",
          changed ? changed->reason : "");
    bool second_matches = true;
    if (shared_encoder_case) {
        const auto second_actual = std::span<const float>(
            reinterpret_cast<const float*>(
                second_history_buffer.payload().data()),
            history_count);
        const auto second_mismatch = twin::compare_floats(
            second_actual, expected_oracle,
            {twin::Comparison::bits, 0.0, 0.0, 0u});
        check(!second_mismatch,
              label + " n=1 rebinding result matches oracle",
              second_mismatch ? second_mismatch->reason : "");
        const auto second_changed =
            second_history_buffer.first_unwritten_change(written);
        check(!second_changed,
              label + " second history guards remain unchanged",
              second_changed ? second_changed->reason : "");
        second_matches = !second_mismatch;
    }
    check(!norm_buffer.first_unwritten_change({}),
          label + " ple_norm remains read-only");
    return !mismatch && second_matches;
}

bool run_set(Context& context, twin::InputSet set,
             const twin::Options& options) {
    bool passed = true;
    std::vector<std::uint8_t> gate_one_row;
    std::vector<std::uint8_t> gate_eight_rows;
    std::vector<std::uint8_t> conv_one_row;
    std::vector<std::uint8_t> conv_eight_rows;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        std::vector<std::uint8_t> output;
        passed &= run_gate_case(context, rows, set, options, false, output);
        if (rows == 1u) gate_one_row = std::move(output);
        if (rows == VERIFY_MAX_T) gate_eight_rows = std::move(output);
    }
    std::vector<std::uint8_t> reduction_probe;
    passed &= run_gate_case(context, 1u, set, options, true, reduction_probe);
    if (gate_one_row.size() == HC * sizeof(float) &&
        gate_eight_rows.size() ==
            static_cast<std::size_t>(VERIFY_MAX_T) * HC * sizeof(float)) {
        check(std::equal(gate_one_row.begin(), gate_one_row.end(),
                         gate_eight_rows.begin()),
              "ple_gate first row is independent of batch size");
    } else {
        check(false, "ple_gate row-independence snapshots");
        passed = false;
    }

    std::vector<std::uint8_t> one_row;
    std::vector<std::uint8_t> eight_rows;
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        std::vector<std::uint8_t> output;
        passed &= run_case(context, rows, set, options, false, output);
    }
    passed &= run_case(context, 1u, set, options, true, one_row);
    passed &= run_case(context, VERIFY_MAX_T, set, options, true, eight_rows);
    if (one_row.size() == HC_DIM * sizeof(float) &&
        eight_rows.size() == static_cast<std::size_t>(VERIFY_MAX_T) *
                                 HC_DIM * sizeof(float)) {
        std::vector<std::uint8_t> repeated_single;
        repeated_single.reserve(eight_rows.size());
        for (std::uint32_t row = 0; row < VERIFY_MAX_T; ++row) {
            repeated_single.insert(repeated_single.end(),
                                   one_row.begin(), one_row.end());
        }
        const auto mismatch = twin::row_t_independent(
            repeated_single, eight_rows, HC_DIM * sizeof(float), VERIFY_MAX_T);
        check(!mismatch, "ple_broadcast row independence",
              mismatch ? mismatch->reason : "");
        passed &= !mismatch;
    } else {
        check(false, "ple_broadcast row independence snapshots");
        passed = false;
    }
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        std::vector<std::uint8_t> output;
        passed &= run_conv_residual_case(
            context, rows, set, options, false, output);
        if (rows == 1u) conv_one_row = std::move(output);
        if (rows == VERIFY_MAX_T) conv_eight_rows = std::move(output);
    }
    std::vector<std::uint8_t> arithmetic_probe;
    passed &= run_conv_residual_case(
        context, 1u, set, options, true, arithmetic_probe);
    if (conv_one_row.size() == HC_DIM * sizeof(float) &&
        conv_eight_rows.size() ==
            static_cast<std::size_t>(VERIFY_MAX_T) * HC_DIM * sizeof(float)) {
        check(std::equal(conv_one_row.begin(), conv_one_row.end(),
                         conv_eight_rows.begin()),
              "ple_conv_residual first row is independent of batch size");
    } else {
        check(false, "ple_conv_residual row-independence snapshots");
        passed = false;
    }
    for (const std::uint32_t n : kHistoryLengths) {
        passed &= run_history_case(context, n, set, options);
    }
    if (set == twin::InputSet::fixture) {
        for (const std::uint32_t rows :
             std::array<std::uint32_t, 3>{1u, 2u, 8u}) {
            passed &= run_shared_ple_segment(context, options, rows);
        }
        passed &= run_ple_block_projection_case(
            context, options, MatVecWeightFormat::q2_0, 2u, 4u, true, true);
        passed &= run_ple_block_projection_case(
            context, options, MatVecWeightFormat::q8_0, 8u, 8u, true, true);
        passed &= run_ple_block_projection_case(
            context, options, MatVecWeightFormat::bf16, 1u, 1u, false, true);
    }
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_ple_test strata.metallib "
                     "[--seed N] [--input-set random|edge|fixture]\n");
        return 2;
    }

    twin::Options options;
    std::string err;
    if (!twin::parse_options(argc, argv, 2, options, err)) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 2;
    }
    if (options.input_set == twin::InputSet::real) {
        std::fprintf(stderr, "metal_ple_test has no model-backed real input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, err);
    check(context != nullptr, "open Metal context", err);
    if (!context) return 1;

    bool passed = test_gate_encoder_validation(*context, argv[1]);
    passed &= test_broadcast_encoder_validation(*context, argv[1]);
    passed &= test_history_encoder_validation(*context, argv[1]);
    passed &= test_conv_residual_encoder_validation(*context, argv[1]);
    if (options.input_set_explicit) {
        passed = run_set(*context, options.input_set, options);
    } else {
        constexpr std::array input_sets{
            twin::InputSet::random, twin::InputSet::edge, twin::InputSet::fixture};
        for (const twin::InputSet set : input_sets) {
            twin::Options case_options = options;
            case_options.input_set = set;
            case_options.input_set_explicit = true;
            std::printf("PLE input set: %s\n", twin::input_set_name(set));
            passed &= run_set(*context, set, case_options);
        }
    }
    check(passed && g_failures == 0,
          "Metal PLE gate/broadcast/conv-residual/history parity");
    std::printf("metal_ple_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
