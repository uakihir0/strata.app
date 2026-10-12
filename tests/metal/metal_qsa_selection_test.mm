// Metal twin for QSA block scoring (docs/mac/04 §5.6).
#include "metallib_sha256.inc"
#include "oracles/qsa.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/qsa_dispatch.hpp"
#include "twin.hpp"

#include "../../src/metal/kernels/common/params.h"
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
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kCellCapacity = 64u;
constexpr std::uint32_t kPrefillSelectionRowsPerGroup = 256u;
constexpr float kUntouchedScore = -9876.5f;
constexpr std::int32_t kUntouchedId = -987654321;
constexpr std::array<std::uint32_t, VERIFY_MAX_T> kPositions{
    0u, 1u, 2u, 3u, 4u, 7u, 8u, 15u};

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

template <class T>
std::span<const T> typed(const twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<const T*>(bytes.data()),
            bytes.size() / sizeof(T)};
}

BufferRef subview(BufferRef buffer, std::size_t offset,
                  std::size_t length) {
    return buffer.slice(offset, length);
}

float input_value(twin::InputSet set, std::size_t index,
                  std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 16> values{
            0.0f, -0.0f, 0.125f, -0.125f, 0.25f, -0.25f, 0.5f, -0.5f,
            1.0f, -1.0f, 2.0f, -2.0f, 0x1p-12f, -0x1p-12f, 8.0f, -8.0f};
        return values[index % values.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t value =
            static_cast<std::int32_t>((index * 73u + 29u) % 2047u) - 1023;
        return static_cast<float>(value) * (1.0f / 256.0f);
    }
    const std::int32_t value =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(value) * (1.0f / 32768.0f);
}

struct ScoreBuffers {
    twin::GuardedBuffer pooled;
    twin::GuardedBuffer dead;
    twin::GuardedBuffer queries;
    twin::GuardedBuffer steps;
    twin::GuardedBuffer scores;

    QsaSelectionScoresBuffers views() const {
        return {pooled.view(), dead.view(), queries.view(), steps.view(),
                scores.view()};
    }
};

struct TopkBuffers {
    twin::GuardedBuffer scores;
    twin::GuardedBuffer steps;
    twin::GuardedBuffer ids;

    QsaSelectionTopkBuffers views() const {
        return {scores.view(), steps.view(), ids.view()};
    }
};

struct PrefillSelectionBuffers {
    twin::GuardedBuffer pooled;
    twin::GuardedBuffer dead;
    twin::GuardedBuffer queries;
    twin::GuardedBuffer steps;
    twin::GuardedBuffer scores;
    twin::GuardedBuffer ids;

    QsaPrefillSelectionBuffers views() const {
        return {pooled.view(), dead.view(), queries.view(), steps.view(),
                scores.view(), ids.view()};
    }
};

bool allocate_prefill_selection_buffers(
    Context& context, std::uint32_t rows, std::uint32_t cell_capacity,
    PrefillSelectionBuffers& buffers, std::string& error) {
    constexpr std::size_t key_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t query_row_bytes =
        static_cast<std::size_t>(INDEXER_HEADS) * key_row_bytes;
    constexpr std::size_t id_row_bytes =
        static_cast<std::size_t>(INDEXER_SELECTION_WIDTH_MAX) *
        sizeof(std::int32_t);
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    return buffers.pooled.allocate(
               context, "Prefill selection pooled",
               max_blocks * key_row_bytes, error) &&
           buffers.dead.allocate(
               context, "Prefill selection dead", key_row_bytes, error) &&
           buffers.queries.allocate(
               context, "Prefill selection queries",
               static_cast<std::size_t>(rows) * query_row_bytes, error) &&
           buffers.steps.allocate(
               context, "Prefill selection steps",
               static_cast<std::size_t>(rows) * sizeof(StepRow), error) &&
           buffers.scores.allocate(
               context, "Prefill selection reusable scores",
               static_cast<std::size_t>(kPrefillSelectionRowsPerGroup) *
                   max_blocks * sizeof(float),
               error) &&
           buffers.ids.allocate(
               context, "Prefill selection ids",
               static_cast<std::size_t>(rows) * id_row_bytes, error);
}

bool allocate_topk_buffers(Context& context, std::uint32_t rows,
                           std::uint32_t cell_capacity,
                           TopkBuffers& buffers, std::string& error) {
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t score_bytes =
        static_cast<std::size_t>(rows) * max_blocks * sizeof(float);
    const std::size_t steps_bytes =
        static_cast<std::size_t>(rows) * sizeof(StepRow);
    const std::size_t ids_bytes =
        static_cast<std::size_t>(rows) * INDEXER_SELECTION_WIDTH_MAX *
        sizeof(std::int32_t);
    return buffers.scores.allocate(context, "QSA selection top-k scores",
                                   score_bytes, error) &&
           buffers.steps.allocate(context, "QSA selection top-k steps",
                                  steps_bytes, error) &&
           buffers.ids.allocate(context, "QSA selection top-k ids",
                                ids_bytes, error);
}

std::vector<StepRow> make_topk_steps(
    std::span<const std::uint32_t> n_kv_values) {
    std::vector<StepRow> steps;
    steps.reserve(n_kv_values.size());
    for (const std::uint32_t n_kv : n_kv_values) {
        const std::int32_t signed_n_kv = static_cast<std::int32_t>(n_kv);
        steps.push_back(StepRow{
            signed_n_kv - 1,
            signed_n_kv,
            signed_n_kv / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(
                signed_n_kv,
                static_cast<std::int32_t>(INDEXER_SELECTION_WIDTH_MAX)),
        });
    }
    return steps;
}

bool allocate_score_buffers(Context& context, std::uint32_t rows,
                            std::uint32_t cell_capacity,
                            ScoreBuffers& buffers, std::string& error) {
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t key_row =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    const std::size_t query_row =
        static_cast<std::size_t>(INDEXER_HEADS) * key_row;
    return buffers.pooled.allocate(context, "QSA selection pooled keys",
                                   max_blocks * key_row + 16u, error) &&
           buffers.dead.allocate(context, "QSA selection dead key",
                                 key_row, error) &&
           buffers.queries.allocate(context, "QSA selection queries",
                                    rows * query_row, error) &&
           buffers.steps.allocate(
               context, "QSA selection steps",
               static_cast<std::size_t>(rows) * sizeof(StepRow), error) &&
           buffers.scores.allocate(
               context, "QSA selection scores",
               static_cast<std::size_t>(rows) * max_blocks * sizeof(float),
               error);
}

std::vector<StepRow> make_steps(std::uint32_t rows,
                               std::uint32_t cell_capacity) {
    std::vector<StepRow> steps(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::uint32_t position =
            rows == 1u && cell_capacity > kCellCapacity
                ? cell_capacity - 1u
                : kPositions[row];
        const std::int32_t n_kv = static_cast<std::int32_t>(position + 1u);
        steps[row] = StepRow{
            static_cast<std::int32_t>(position),
            n_kv,
            n_kv / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(
                n_kv, static_cast<std::int32_t>(
                          INDEXER_SELECTION_WIDTH_MAX)),
        };
    }
    return steps;
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

bool run_score_case(Context& context, std::uint32_t rows,
                    std::uint32_t cell_capacity, twin::InputSet set,
                    const twin::Options& options,
                    std::string_view suffix = {},
                    bool shared_encoder = false) {
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::string label =
        "QSA selection scores T=" + std::to_string(rows) +
        " capacity=" + std::to_string(cell_capacity) +
        " set=" + twin::input_set_name(set) + std::string(suffix) +
        (shared_encoder ? " shared Encoder" : "");
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(rows) << 40u) ^
                        (static_cast<std::uint64_t>(cell_capacity) << 8u) ^
                        static_cast<std::uint64_t>(set));
    std::vector<float> pooled(max_blocks * INDEXER_HEAD_DIM);
    std::vector<float> dead(INDEXER_HEAD_DIM);
    std::vector<float> queries(
        static_cast<std::size_t>(rows) * INDEXER_HEADS * INDEXER_HEAD_DIM);
    for (std::size_t i = 0u; i < pooled.size(); ++i) {
        pooled[i] = input_value(set, i + 31u, rng);
    }
    for (std::size_t i = 0u; i < dead.size(); ++i) {
        dead[i] = input_value(set, i + 0x1003u, rng);
    }
    for (std::size_t i = 0u; i < queries.size(); ++i) {
        queries[i] = input_value(set, i + 0x2039u, rng);
    }
    const std::vector<StepRow> steps = make_steps(rows, cell_capacity);

    ScoreBuffers buffers;
    std::string error;
    const bool allocated =
        allocate_score_buffers(context, rows, cell_capacity, buffers, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !buffers.pooled.view() || !buffers.dead.view() ||
        !buffers.queries.view() || !buffers.steps.view() ||
        !buffers.scores.view()) {
        return false;
    }
    std::memcpy(buffers.pooled.payload().data(), pooled.data(),
                pooled.size() * sizeof(float));
    std::memcpy(buffers.dead.payload().data(), dead.data(),
                dead.size() * sizeof(float));
    std::memcpy(buffers.queries.payload().data(), queries.data(),
                queries.size() * sizeof(float));
    std::memcpy(buffers.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    auto initial_scores = typed<float>(buffers.scores);
    std::fill(initial_scores.begin(), initial_scores.end(), kUntouchedScore);

    std::vector<std::int32_t> n_kv(rows), n_bid(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        n_kv[row] = steps[row].n_kv;
        n_bid[row] = steps[row].n_bid;
    }
    std::vector<twin::tf> expected;
    oracle::qsa_selection_scores(
        pooled, dead, queries, n_kv, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedScore, expected);

    buffers.pooled.capture_before_dispatch();
    buffers.dead.capture_before_dispatch();
    buffers.queries.capture_before_dispatch();
    buffers.steps.capture_before_dispatch();
    buffers.scores.capture_before_dispatch();
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    if (shared_encoder) {
        const bool encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_qsa_selection_scores(
                    context, encoder, rows, cell_capacity, buffers.views(),
                    encode_error);
            },
            error);
        if (!encoded) {
            check(false, label + " shared encode", error);
            return false;
        }
    } else {
        if (!encode_qsa_selection_scores(
                context, *command, rows, cell_capacity, buffers.views(),
                error)) {
            check(false, label + " encode", error);
            command->discard();
            return false;
        }
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto actual = typed<float>(buffers.scores);
    const twin::Tolerance exact{twin::Comparison::bits, 0.0, 0.0, 0u};
    const auto difference = twin::compare_floats(actual, expected, exact);
    if (difference) {
        twin::report_failure(label.c_str(), options, rows, *difference);
    }
    check(!difference, label + " scores match ordered scalar oracle");

    const std::span<const twin::ByteRange> no_writes;
    const bool inputs_unchanged =
        !buffers.pooled.first_unwritten_change(no_writes) &&
        !buffers.dead.first_unwritten_change(no_writes) &&
        !buffers.queries.first_unwritten_change(no_writes) &&
        !buffers.steps.first_unwritten_change(no_writes);
    check(inputs_unchanged, label + " inputs are unchanged");
    const std::array written_ranges{
        twin::ByteRange{0u, buffers.scores.payload().size()}};
    const bool output_guards =
        !buffers.scores.first_unwritten_change(written_ranges);
    check(output_guards, label + " output guards are intact");
    return !difference && inputs_unchanged && output_guards;
}

bool run_prefill_selection_case(
    Context& context, std::uint32_t rows, std::uint32_t pos0,
    std::uint32_t cell_capacity, const twin::Options& options,
    bool shared_encoder) {
    const std::string label =
        "Prefill QSA selection T=" + std::to_string(rows) +
        " p0=" + std::to_string(pos0) +
        " capacity=" + std::to_string(cell_capacity) +
        (shared_encoder ? " shared Encoder" : "");
    std::vector<std::uint32_t> tokens(rows, 1u);
    PrefillChunkMetadata metadata;
    std::string error;
    if (!build_prefill_chunk_metadata(pos0, tokens, metadata, error)) {
        check(false, label + " metadata", error);
        return false;
    }
    std::vector<StepRow> steps;
    steps.reserve(rows);
    for (const PrefillStepRow& step : metadata.steps) {
        steps.push_back(
            StepRow{step.pos, step.n_kv, step.n_bid, step.width});
    }
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    constexpr std::size_t key_row_elements = INDEXER_HEAD_DIM;
    constexpr std::size_t query_row_elements =
        INDEXER_HEADS * INDEXER_HEAD_DIM;
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(rows) << 32u) ^
                        (static_cast<std::uint64_t>(pos0) << 8u) ^
                        cell_capacity);
    std::vector<float> pooled(max_blocks * key_row_elements);
    std::vector<float> dead(key_row_elements);
    std::vector<float> queries(
        static_cast<std::size_t>(rows) * query_row_elements);
    for (std::size_t i = 0u; i < pooled.size(); ++i) {
        pooled[i] = input_value(options.input_set, i + 0x3101u, rng);
    }
    for (std::size_t i = 0u; i < dead.size(); ++i) {
        dead[i] = input_value(options.input_set, i + 0x4103u, rng);
    }
    for (std::size_t i = 0u; i < queries.size(); ++i) {
        queries[i] = input_value(
            options.input_set, i + 0x5107u, rng);
    }

    PrefillSelectionBuffers buffers;
    const bool allocated = allocate_prefill_selection_buffers(
        context, rows, cell_capacity, buffers, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !buffers.pooled.view() || !buffers.dead.view() ||
        !buffers.queries.view() || !buffers.steps.view() ||
        !buffers.scores.view() || !buffers.ids.view()) {
        return false;
    }
    std::memcpy(buffers.pooled.payload().data(), pooled.data(),
                pooled.size() * sizeof(float));
    std::memcpy(buffers.dead.payload().data(), dead.data(),
                dead.size() * sizeof(float));
    std::memcpy(buffers.queries.payload().data(), queries.data(),
                queries.size() * sizeof(float));
    std::memcpy(buffers.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    std::fill(typed<float>(buffers.scores).begin(),
              typed<float>(buffers.scores).end(), kUntouchedScore);
    std::fill(typed<std::int32_t>(buffers.ids).begin(),
              typed<std::int32_t>(buffers.ids).end(), kUntouchedId);

    std::vector<std::int32_t> n_kv(rows), n_bid(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        n_kv[row] = steps[row].n_kv;
        n_bid[row] = steps[row].n_bid;
    }
    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        pooled, dead, queries, n_kv, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedScore,
        expected_scores);
    std::vector<float> expected_score_values;
    expected_score_values.reserve(expected_scores.size());
    for (const twin::tf value : expected_scores) {
        expected_score_values.push_back(value.value);
    }
    std::vector<std::int32_t> expected_ids;
    oracle::qsa_selection_topk(
        expected_score_values, n_kv, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedId, expected_ids);

    const std::uint32_t last_group_base =
        ((rows - 1u) / kPrefillSelectionRowsPerGroup) *
        kPrefillSelectionRowsPerGroup;
    const std::uint32_t last_group_rows = rows - last_group_base;
    buffers.pooled.capture_before_dispatch();
    buffers.dead.capture_before_dispatch();
    buffers.queries.capture_before_dispatch();
    buffers.steps.capture_before_dispatch();
    buffers.scores.capture_before_dispatch();
    buffers.ids.capture_before_dispatch();
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    bool encoded = false;
    if (shared_encoder) {
        encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_prefill_qsa_selection(
                    context, encoder, metadata.params, cell_capacity,
                    buffers.views(), encode_error);
            },
            error);
    } else {
        encoded = encode_prefill_qsa_selection(
            context, *command, metadata.params, cell_capacity,
            buffers.views(), error);
    }
    check(encoded, label + " encode", error);
    if (!encoded || !submit_and_wait(command, label, error)) {
        check(false, label + " GPU completion", error);
        if (!shared_encoder) command->discard();
        return false;
    }

    const std::size_t last_score_count =
        static_cast<std::size_t>(last_group_rows) * max_blocks;
    const twin::Tolerance exact{twin::Comparison::bits, 0.0, 0.0, 0u};
    std::vector<twin::tf> expected_reusable_scores(
        static_cast<std::size_t>(kPrefillSelectionRowsPerGroup) * max_blocks,
        twin::tf{kUntouchedScore});
    if (last_group_base != 0u) {
        const std::uint32_t previous_group_base =
            last_group_base - kPrefillSelectionRowsPerGroup;
        std::copy_n(
            expected_scores.begin() +
                static_cast<std::ptrdiff_t>(
                    static_cast<std::size_t>(previous_group_base) *
                    max_blocks),
            expected_reusable_scores.size(),
            expected_reusable_scores.begin());
    }
    std::copy_n(
        expected_scores.begin() +
            static_cast<std::ptrdiff_t>(
                static_cast<std::size_t>(last_group_base) * max_blocks),
        last_score_count, expected_reusable_scores.begin());
    const auto score_difference = twin::compare_floats(
        typed<float>(buffers.scores), expected_reusable_scores, exact);
    if (score_difference) {
        twin::report_failure(label.c_str(), options, rows,
                             *score_difference);
    }
    check(!score_difference,
          label +
              " reusable score scratch matches sequential batch oracle");
    const auto actual_ids = typed<std::int32_t>(buffers.ids);
    const auto id_mismatch =
        std::mismatch(actual_ids.begin(), actual_ids.end(),
                      expected_ids.begin());
    const bool ids_match = id_mismatch.first == actual_ids.end();
    check(ids_match, label + " all IDs and untouched suffixes match oracle",
          ids_match
              ? std::string{}
              : "first differing id index " +
                    std::to_string(id_mismatch.first - actual_ids.begin()));

    const std::span<const twin::ByteRange> no_writes;
    const bool inputs_unchanged =
        !buffers.pooled.first_unwritten_change(no_writes) &&
        !buffers.dead.first_unwritten_change(no_writes) &&
        !buffers.queries.first_unwritten_change(no_writes) &&
        !buffers.steps.first_unwritten_change(no_writes);
    check(inputs_unchanged, label + " read-only inputs are unchanged");
    const std::array score_written{
        twin::ByteRange{0u, buffers.scores.payload().size()}};
    const std::array ids_written{
        twin::ByteRange{0u, buffers.ids.payload().size()}};
    const bool output_guards =
        !buffers.scores.first_unwritten_change(score_written) &&
        !buffers.ids.first_unwritten_change(ids_written);
    check(output_guards, label + " output guards are intact");
    return !score_difference && ids_match && inputs_unchanged &&
           output_guards;
}

bool test_prefill_indexer_selection_composition(
    Context& context, const twin::Options& options, std::uint32_t pos0,
    std::uint32_t cell_capacity) {
    constexpr std::uint32_t rows = 9u;
    constexpr std::size_t key_row_elements = INDEXER_HEAD_DIM;
    constexpr std::size_t query_row_elements =
        INDEXER_HEADS * INDEXER_HEAD_DIM;
    constexpr std::size_t tail_elements = 3u * key_row_elements;
    constexpr float rope_zero_cos = 1.125f;
    constexpr float rope_zero_sin = -0.0f;
    const std::size_t max_blocks =
        static_cast<std::size_t>(
            cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::string label =
        "Prefill QSA indexer append then selection in one Encoder p0=" +
        std::to_string(pos0) + " capacity=" +
        std::to_string(cell_capacity);

    std::vector<std::uint32_t> tokens(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        tokens[row] = row * 11u + 5u;
    }
    PrefillChunkMetadata metadata;
    std::string error;
    if (!build_prefill_chunk_metadata(pos0, tokens, metadata, error)) {
        check(false, label + " metadata", error);
        return false;
    }

    std::mt19937_64 rng(options.seed ^ 0x5a17c0deu);
    std::vector<float> raw_keys(
        static_cast<std::size_t>(rows) * key_row_elements);
    std::vector<float> gamma(key_row_elements);
    std::vector<float> rope(
        static_cast<std::size_t>(cell_capacity) * 32u * 2u);
    std::vector<float> queries(
        static_cast<std::size_t>(rows) * query_row_elements);
    for (std::size_t i = 0u; i < raw_keys.size(); ++i) {
        raw_keys[i] = input_value(options.input_set, i + 0x1103u, rng) * 2.75f;
    }
    for (std::size_t i = 0u; i < gamma.size(); ++i) {
        gamma[i] = 1.0f +
            input_value(options.input_set, i + 0x2101u, rng) * 0.375f;
    }
    for (std::uint32_t cell = 0u; cell < cell_capacity; ++cell) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float angle =
                static_cast<float>((cell * 13u + pair * 7u) % 257u) *
                0.013f;
            const std::size_t offset =
                (static_cast<std::size_t>(cell) * 32u + pair) * 2u;
            rope[offset] = std::cos(angle) * 1.0625f;
            rope[offset + 1u] = std::sin(angle) * 1.0625f;
        }
    }
    for (std::size_t i = 0u; i < queries.size(); ++i) {
        queries[i] =
            input_value(options.input_set, i + 0x3109u, rng) * 0.75f;
    }

    std::vector<std::int32_t> positions(rows);
    std::vector<StepRow> steps(rows);
    std::vector<std::int32_t> n_kv(rows);
    std::vector<std::int32_t> n_bid(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::int32_t position =
            static_cast<std::int32_t>(pos0 + row);
        positions[row] = position;
        const std::int32_t count = position + 1;
        steps[row] = StepRow{
            position, count,
            count / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(
                count,
                static_cast<std::int32_t>(INDEXER_SELECTION_WIDTH_MAX))};
        n_kv[row] = count;
        n_bid[row] = steps[row].n_bid;
    }

    oracle::QsaIndexerState initial;
    initial.tail.assign(tail_elements, -13.25f);
    initial.dead.assign(key_row_elements, -17.5f);
    initial.pooled.assign(max_blocks * key_row_elements, -19.75f);
    initial.block_position = -23;
    std::vector<std::int32_t> prefix_positions(pos0);
    std::vector<float> prefix_keys(
        static_cast<std::size_t>(pos0) * key_row_elements);
    for (std::uint32_t row = 0u; row < pos0; ++row) {
        prefix_positions[row] = static_cast<std::int32_t>(row);
        for (std::size_t dimension = 0u;
             dimension < key_row_elements; ++dimension) {
            const std::size_t index =
                static_cast<std::size_t>(row) * key_row_elements + dimension;
            prefix_keys[index] = input_value(
                options.input_set, index + 0x4103u, rng) * 2.75f;
        }
    }
    oracle::qsa_indexer_append(
        prefix_keys, prefix_positions, gamma, rope,
        rope_zero_cos, rope_zero_sin, initial);
    oracle::QsaIndexerState expected_indexer = initial;
    oracle::qsa_indexer_append(
        raw_keys, positions, gamma, rope,
        rope_zero_cos, rope_zero_sin, expected_indexer);

    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        expected_indexer.pooled, expected_indexer.dead, queries, n_kv, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedScore,
        expected_scores);
    std::vector<float> expected_score_values;
    expected_score_values.reserve(expected_scores.size());
    for (const twin::tf value : expected_scores) {
        expected_score_values.push_back(value.value);
    }
    std::vector<std::int32_t> expected_ids;
    oracle::qsa_selection_topk(
        expected_score_values, n_kv, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedId, expected_ids);

    PrefillSelectionBuffers selection;
    twin::GuardedBuffer raw_buffer, gamma_buffer, rope_buffer, tail_buffer,
        block_position_buffer;
    const std::size_t tail_bytes = tail_elements * sizeof(float);
    const bool allocated =
        allocate_prefill_selection_buffers(
            context, rows, cell_capacity, selection, error) &&
        raw_buffer.allocate(context, label + " raw keys",
                            raw_keys.size() * sizeof(float), error) &&
        gamma_buffer.allocate(context, label + " gamma",
                              gamma.size() * sizeof(float), error) &&
        rope_buffer.allocate(context, label + " RoPE",
                             rope.size() * sizeof(float), error) &&
        tail_buffer.allocate(context, label + " tail", tail_bytes, error) &&
        block_position_buffer.allocate(
            context, label + " block position", sizeof(std::int32_t), error);
    check(allocated, label + " allocations", error);
    if (!allocated || !selection.pooled.view() || !selection.dead.view() ||
        !selection.queries.view() || !selection.steps.view() ||
        !selection.scores.view() || !selection.ids.view() ||
        !raw_buffer.view() || !gamma_buffer.view() || !rope_buffer.view() ||
        !tail_buffer.view() || !block_position_buffer.view()) {
        return false;
    }

    std::memcpy(selection.pooled.payload().data(),
                initial.pooled.data(), initial.pooled.size() * sizeof(float));
    std::memcpy(selection.dead.payload().data(),
                initial.dead.data(), initial.dead.size() * sizeof(float));
    std::memcpy(selection.queries.payload().data(),
                queries.data(), queries.size() * sizeof(float));
    std::memcpy(selection.steps.payload().data(),
                steps.data(), steps.size() * sizeof(StepRow));
    std::fill(typed<float>(selection.scores).begin(),
              typed<float>(selection.scores).end(), kUntouchedScore);
    std::fill(typed<std::int32_t>(selection.ids).begin(),
              typed<std::int32_t>(selection.ids).end(), kUntouchedId);
    std::memcpy(raw_buffer.payload().data(),
                raw_keys.data(), raw_keys.size() * sizeof(float));
    std::memcpy(gamma_buffer.payload().data(),
                gamma.data(), gamma.size() * sizeof(float));
    std::memcpy(rope_buffer.payload().data(),
                rope.data(), rope.size() * sizeof(float));
    std::memcpy(tail_buffer.payload().data(),
                initial.tail.data(), tail_bytes);
    std::memcpy(block_position_buffer.payload().data(),
                &initial.block_position, sizeof(initial.block_position));

    raw_buffer.capture_before_dispatch();
    gamma_buffer.capture_before_dispatch();
    rope_buffer.capture_before_dispatch();
    selection.queries.capture_before_dispatch();
    selection.steps.capture_before_dispatch();
    tail_buffer.capture_before_dispatch();
    selection.dead.capture_before_dispatch();
    selection.pooled.capture_before_dispatch();
    block_position_buffer.capture_before_dispatch();
    selection.scores.capture_before_dispatch();
    selection.ids.capture_before_dispatch();

    const QsaPrefillIndexerAppendBuffers append_buffers{
        raw_buffer.view(), gamma_buffer.view(), rope_buffer.view(),
        tail_buffer.view(), selection.dead.view(), selection.pooled.view(),
        block_position_buffer.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_prefill_qsa_indexer_append(
                       context, encoder, metadata.params, cell_capacity,
                       rope_zero_cos, rope_zero_sin, append_buffers,
                       encode_error) &&
                encode_prefill_qsa_selection(
                    context, encoder, metadata.params, cell_capacity,
                    selection.views(), encode_error);
        },
        error);
    check(encoded, label + " encode", error);
    if (!encoded || !submit_and_wait(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const twin::Tolerance exact{twin::Comparison::bits, 0.0, 0.0, 0u};
    const auto actual_score_count =
        static_cast<std::size_t>(rows) * max_blocks;
    const auto score_difference = twin::compare_floats(
        typed<float>(selection.scores).first(actual_score_count),
        expected_scores, exact);
    const auto exact_float_match = [](
        std::span<const float> actual, std::span<const float> expected) {
        return actual.size() == expected.size() &&
            std::memcmp(actual.data(), expected.data(),
                        actual.size_bytes()) == 0;
    };
    const bool dead_matches = exact_float_match(
        typed<float>(selection.dead), expected_indexer.dead);
    const bool pool_matches = exact_float_match(
        typed<float>(selection.pooled), expected_indexer.pooled);
    const bool tail_matches = exact_float_match(
        typed<float>(tail_buffer), expected_indexer.tail);
    if (score_difference) {
        twin::report_failure(label.c_str(), options, rows, *score_difference);
    }
    const auto actual_ids = typed<std::int32_t>(selection.ids);
    const bool ids_match =
        std::equal(actual_ids.begin(), actual_ids.end(), expected_ids.begin());
    check(!score_difference,
          label + " selection scores use appended indexer keys");
    check(dead_matches && pool_matches && tail_matches,
          label + " append outputs match scalar oracle");
    check(ids_match, label + " selected IDs and untouched suffix match oracle");

    const std::span<const twin::ByteRange> no_writes;
    const bool inputs_unchanged =
        !raw_buffer.first_unwritten_change(no_writes) &&
        !gamma_buffer.first_unwritten_change(no_writes) &&
        !rope_buffer.first_unwritten_change(no_writes) &&
        !selection.queries.first_unwritten_change(no_writes) &&
        !selection.steps.first_unwritten_change(no_writes);
    check(inputs_unchanged, label + " read-only inputs are unchanged");
    const std::array tail_written{twin::ByteRange{0u, tail_bytes}};
    const std::array key_written{
        twin::ByteRange{0u, selection.dead.payload().size()}};
    const std::array pool_written{
        twin::ByteRange{0u, selection.pooled.payload().size()}};
    const std::array block_written{
        twin::ByteRange{0u, sizeof(std::int32_t)}};
    const std::array score_written{
        twin::ByteRange{0u, selection.scores.payload().size()}};
    const std::array ids_written{
        twin::ByteRange{0u, selection.ids.payload().size()}};
    const bool guards_intact =
        !tail_buffer.first_unwritten_change(tail_written) &&
        !selection.dead.first_unwritten_change(key_written) &&
        !selection.pooled.first_unwritten_change(pool_written) &&
        !block_position_buffer.first_unwritten_change(block_written) &&
        !selection.scores.first_unwritten_change(score_written) &&
        !selection.ids.first_unwritten_change(ids_written);
    check(guards_intact, label + " guards are intact");
    return !score_difference && dead_matches && pool_matches &&
           tail_matches && ids_match && inputs_unchanged && guards_intact;
}

float topk_score_value(twin::InputSet set, std::uint32_t row,
                       std::size_t block, std::mt19937_64& rng) {
    if (set == twin::InputSet::edge) {
        constexpr float infinity = std::numeric_limits<float>::infinity();
        constexpr std::array<float, 12> values{
            -std::numeric_limits<float>::infinity(),
            -8.0f,
            -1.0f,
            -0.0f,
            0.0f,
            std::numeric_limits<float>::quiet_NaN(),
            0.125f,
            1.0f,
            std::numeric_limits<float>::infinity(),
            2.0f,
            -infinity,
            std::numeric_limits<float>::quiet_NaN(),
        };
        return values[(block + row * 5u) % values.size()];
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t value = static_cast<std::int32_t>(
            (block * 37u + static_cast<std::size_t>(row) * 19u) % 29u);
        return static_cast<float>(value - 14) * 0.25f;
    }
    return input_value(twin::InputSet::random,
                       block + static_cast<std::size_t>(row) * 0x10001u,
                       rng);
}

enum class TopkScorePattern {
    generated,
    all_equal,
    nan_below_negative_infinity,
    signed_zero_tie,
    tail_block_high,
    high_ids_only,
};

bool run_topk_case(Context& context, std::uint32_t cell_capacity,
                   std::span<const std::uint32_t> n_kv_values,
                   twin::InputSet set, const twin::Options& options,
                   TopkScorePattern pattern, std::string_view suffix = {},
                   std::vector<std::int32_t>* ids_out = nullptr,
                   bool shared_encoder = false) {
    const std::uint32_t rows =
        static_cast<std::uint32_t>(n_kv_values.size());
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::string label =
        "QSA selection top-k T=" + std::to_string(rows) +
        " capacity=" + std::to_string(cell_capacity) +
        " set=" + twin::input_set_name(set) + std::string(suffix) +
        (shared_encoder ? " shared Encoder" : "");
    std::mt19937_64 rng(options.seed ^
                        (static_cast<std::uint64_t>(rows) << 40u) ^
                        (static_cast<std::uint64_t>(cell_capacity) << 8u) ^
                        static_cast<std::uint64_t>(set));
    const std::vector<StepRow> steps = make_topk_steps(n_kv_values);

    TopkBuffers buffers;
    std::string error;
    const bool allocated = allocate_topk_buffers(
        context, rows, cell_capacity, buffers, error);
    check(allocated, label + " allocations", error);
    if (!allocated || !buffers.scores.view() || !buffers.steps.view() ||
        !buffers.ids.view()) {
        return false;
    }

    auto scores = typed<float>(buffers.scores);
    std::fill(scores.begin(), scores.end(), kUntouchedScore);
    std::vector<std::int32_t> n_kv(rows), n_bid(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        n_kv[row] = steps[row].n_kv;
        n_bid[row] = steps[row].n_bid;
        for (std::int32_t block = 0; block <= steps[row].n_bid; ++block) {
            float score = 0.0f;
            switch (pattern) {
                case TopkScorePattern::generated:
                    score = topk_score_value(
                        set, row, static_cast<std::size_t>(block), rng);
                    break;
                case TopkScorePattern::all_equal:
                    score = 3.25f;
                    break;
                case TopkScorePattern::nan_below_negative_infinity:
                    score = block == 0
                        ? std::numeric_limits<float>::quiet_NaN()
                        : -std::numeric_limits<float>::infinity();
                    break;
                case TopkScorePattern::signed_zero_tie:
                    score = block % 2 == 0 ? -0.0f : 0.0f;
                    break;
                case TopkScorePattern::tail_block_high:
                    score = block == steps[row].n_bid
                        ? std::numeric_limits<float>::infinity()
                        : -std::numeric_limits<float>::infinity();
                    break;
                case TopkScorePattern::high_ids_only:
                    score = block >= steps[row].n_bid - 513
                        ? std::numeric_limits<float>::infinity()
                        : -std::numeric_limits<float>::infinity();
                    break;
            }
            scores[static_cast<std::size_t>(row) * max_blocks +
                   static_cast<std::size_t>(block)] = score;
        }
    }
    std::memcpy(buffers.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    auto ids = typed<std::int32_t>(buffers.ids);
    std::fill(ids.begin(), ids.end(), kUntouchedId);

    std::vector<std::int32_t> expected;
    oracle::qsa_selection_topk(scores, n_kv, n_bid,
                               static_cast<std::uint32_t>(max_blocks),
                               kUntouchedId, expected);
    buffers.scores.capture_before_dispatch();
    buffers.steps.capture_before_dispatch();
    buffers.ids.capture_before_dispatch();

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    if (shared_encoder) {
        const bool encoded = Encoder::encode(
            context, *command, label,
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_qsa_selection_topk(
                    context, encoder, rows, cell_capacity, buffers.views(),
                    encode_error);
            },
            error);
        if (!encoded) {
            check(false, label + " shared encode", error);
            return false;
        }
    } else {
        if (!encode_qsa_selection_topk(context, *command, rows, cell_capacity,
                                       buffers.views(), error)) {
            check(false, label + " encode", error);
            command->discard();
            return false;
        }
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto actual = typed<std::int32_t>(buffers.ids);
    const auto mismatch =
        std::mismatch(actual.begin(), actual.end(), expected.begin());
    const bool ids_match = mismatch.first == actual.end();
    const std::string detail =
        ids_match ? std::string{}
                  : "first differing id index " +
                        std::to_string(mismatch.first - actual.begin());
    check(ids_match, label + " IDs match sorted scalar oracle", detail);
    if (ids_out != nullptr) ids_out->assign(actual.begin(), actual.end());
    bool pattern_checks = true;
    if (pattern == TopkScorePattern::nan_below_negative_infinity) {
        const bool nan_is_below_negative_infinity =
            actual.size() >= 4u && actual[0] == 0 && actual[1] == 1 &&
            actual[2] == 2 && actual[3] == 4 &&
            actual[INDEXER_SELECTION_WIDTH_MAX - 1u] == 2051;
        check(nan_is_below_negative_infinity,
              label + " NaN key ranks below negative infinity");
        pattern_checks &= nan_is_below_negative_infinity;
    } else if (pattern == TopkScorePattern::signed_zero_tie) {
        bool signed_zeros_are_tied = true;
        for (std::size_t id = 0u;
             id < INDEXER_SELECTION_WIDTH_MAX; ++id) {
            signed_zeros_are_tied &=
                actual[id] == static_cast<std::int32_t>(id);
        }
        check(signed_zeros_are_tied,
              label + " signed zeros tie and prefer lower ids");
        pattern_checks &= signed_zeros_are_tied;
    } else if (pattern == TopkScorePattern::tail_block_high) {
        bool every_tail_cell_selected = true;
        for (std::uint32_t row = 0u; row < rows; ++row) {
            const std::uint32_t tail_weight =
                n_kv_values[row] % INDEXER_BLOCK_CELLS;
            const std::size_t row_begin =
                static_cast<std::size_t>(row) *
                INDEXER_SELECTION_WIDTH_MAX;
            const auto selected = actual.subspan(
                row_begin, static_cast<std::size_t>(steps[row].width));
            for (std::uint32_t tail = 0u; tail < tail_weight; ++tail) {
                const std::int32_t tail_id =
                    steps[row].n_kv -
                    static_cast<std::int32_t>(tail_weight) +
                    static_cast<std::int32_t>(tail);
                every_tail_cell_selected &=
                    std::find(selected.begin(), selected.end(), tail_id) !=
                    selected.end();
            }
        }
        check(every_tail_cell_selected,
              label + " every highest-scoring tail cell is selected");
        pattern_checks &= every_tail_cell_selected;
    } else if (pattern == TopkScorePattern::high_ids_only) {
        const auto selected = actual.first(
            static_cast<std::size_t>(steps[0].width));
        const bool selected_ids_are_high =
            std::all_of(selected.begin(), selected.end(),
                        [](std::int32_t id) { return id > 65535; });
        check(selected_ids_are_high,
              label + " selected ids are all above 65,535");
        pattern_checks &= selected_ids_are_high;
    }

    const std::span<const twin::ByteRange> no_writes;
    const bool inputs_unchanged =
        !buffers.scores.first_unwritten_change(no_writes) &&
        !buffers.steps.first_unwritten_change(no_writes);
    check(inputs_unchanged, label + " score and StepRow inputs are unchanged");
    const std::array written_ranges{
        twin::ByteRange{0u, buffers.ids.payload().size()}};
    const bool output_guards =
        !buffers.ids.first_unwritten_change(written_ranges);
    check(output_guards, label + " output guards are intact");
    return ids_match && pattern_checks && inputs_unchanged && output_guards;
}

bool test_topk_correctness(Context& context,
                           const twin::Options& options) {
    bool passed = true;
    constexpr std::array<std::uint32_t, VERIFY_MAX_T> direct_n_kv{
        1u, 2u, 3u, 4u, 7u, 64u, 1024u, 2051u};
    for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
        passed &= run_topk_case(
            context, 2052u,
            std::span<const std::uint32_t>(direct_n_kv.data(), rows),
            twin::InputSet::fixture, options, TopkScorePattern::generated,
            " direct path");
    }
    constexpr std::array<std::uint32_t, 1> first_radix_n_kv{2052u};
    passed &= run_topk_case(context, 2052u, first_radix_n_kv,
                            twin::InputSet::fixture, options,
                            TopkScorePattern::all_equal,
                            " first radix row; tied partial threshold");
    passed &= run_topk_case(
        context, 2052u, first_radix_n_kv, twin::InputSet::edge, options,
        TopkScorePattern::nan_below_negative_infinity,
        " NaN order-key boundary");
    passed &= run_topk_case(
        context, 2052u, first_radix_n_kv, twin::InputSet::edge, options,
        TopkScorePattern::signed_zero_tie,
        " signed-zero tie boundary");

    constexpr std::array<std::uint32_t, 3> tail_n_kv{
        2053u, 2054u, 2055u};
    passed &= run_topk_case(context, 2056u, tail_n_kv,
                            twin::InputSet::edge, options,
                            TopkScorePattern::tail_block_high,
                            " highest-scoring partial tail cells");

    constexpr std::array<std::uint32_t, 4> register_n_kv{
        8192u, 8193u, 8194u, 8195u};
    for (const auto set : {twin::InputSet::random, twin::InputSet::edge,
                           twin::InputSet::fixture}) {
        passed &= run_topk_case(context, 8196u, register_n_kv, set, options,
                                TopkScorePattern::generated,
                                " register path");
    }
    constexpr std::array<std::uint32_t, 4> shared_register_n_kv{
        8192u, 8193u, 8194u, 8195u};
    passed &= run_topk_case(
        context, 8196u, shared_register_n_kv, twin::InputSet::fixture, options,
        TopkScorePattern::generated, " shared register path", nullptr, true);
    constexpr std::array<std::uint32_t, 1> tied_n_kv{8192u};
    passed &= run_topk_case(context, 8192u, tied_n_kv,
                            twin::InputSet::fixture, options,
                            TopkScorePattern::all_equal,
                            " all scores tied; partial threshold block");

    constexpr std::array<std::uint32_t, 1> register_limit_n_kv{135160u};
    passed &= run_topk_case(context, 135160u, register_limit_n_kv,
                            twin::InputSet::fixture, options,
                            TopkScorePattern::all_equal,
                            " register-cache 33-key boundary");

    constexpr std::array<std::uint32_t, 1> memory_limit_n_kv{135164u};
    passed &= run_topk_case(context, 135164u, memory_limit_n_kv,
                            twin::InputSet::edge, options,
                            TopkScorePattern::generated,
                            " memory-form dispatch boundary");
    passed &= run_topk_case(
        context, 135164u, memory_limit_n_kv, twin::InputSet::fixture, options,
        TopkScorePattern::all_equal,
        " shared memory-form dispatch boundary", nullptr, true);

    constexpr std::array<std::uint32_t, 4> memory_n_kv{
        139997u, 139998u, 139999u, 140000u};
    passed &= run_topk_case(context, 140000u, memory_n_kv,
                            twin::InputSet::edge, options,
                            TopkScorePattern::generated,
                            " memory path; high ids and all tail widths");
    constexpr std::array<std::uint32_t, 1> high_ids_n_kv{140000u};
    passed &= run_topk_case(context, 140000u, high_ids_n_kv,
                            twin::InputSet::edge, options,
                            TopkScorePattern::high_ids_only,
                            " memory path; ids above 65,535");

    constexpr std::array<std::uint32_t, VERIFY_MAX_T> equivalent_n_kv{
        8192u, 8192u, 8192u, 8192u, 8192u, 8192u, 8192u, 8192u};
    std::vector<std::int32_t> register_ids;
    std::vector<std::int32_t> memory_ids;
    passed &= run_topk_case(context, 8192u, equivalent_n_kv,
                            twin::InputSet::fixture, options,
                            TopkScorePattern::generated,
                            " register/memory equivalence reference",
                            &register_ids);
    passed &= run_topk_case(context, 140000u, equivalent_n_kv,
                            twin::InputSet::fixture, options,
                            TopkScorePattern::generated,
                            " register/memory equivalence memory",
                            &memory_ids);
    passed &= run_topk_case(
        context, 8192u, equivalent_n_kv, twin::InputSet::fixture, options,
        TopkScorePattern::generated, " shared T=8 register form", nullptr,
        true);
    constexpr std::array<std::uint32_t, VERIFY_MAX_T> shared_memory_batch{
        135164u, 135164u, 135164u, 135164u,
        135164u, 135164u, 135164u, 135164u};
    passed &= run_topk_case(
        context, 135164u, shared_memory_batch, twin::InputSet::fixture,
        options, TopkScorePattern::generated, " shared T=8 memory form",
        nullptr, true);
    const bool equivalent = register_ids == memory_ids;
    check(equivalent,
          "QSA top-k register and memory forms return identical ids");
    return passed && equivalent;
}

bool test_topk_encoder_validation(Context& context,
                                 Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t capacity = 64u;
    constexpr std::array<std::uint32_t, rows> n_kv{1u, 2u};
    std::string error;
    TopkBuffers storage;
    if (!allocate_topk_buffers(context, rows, capacity, storage, error)) {
        check(false, "QSA top-k validation allocations", error);
        return false;
    }
    const std::vector<StepRow> steps = make_topk_steps(n_kv);
    std::memcpy(storage.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    QsaSelectionTopkBuffers valid = storage.views();
    bool passed = true;
    auto reject = [&](const std::string& label,
                      const QsaSelectionTopkBuffers& candidate,
                      std::uint32_t token_rows, std::uint32_t cell_capacity,
                      std::string_view message) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !encode_qsa_selection_topk(
                context, *command, token_rows, cell_capacity, candidate,
                error) &&
            error.find(message) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    passed &= reject("QSA top-k rejects zero rows", valid, 0u, capacity,
                     "row count");
    passed &= reject("QSA top-k rejects too many rows", valid, 9u, capacity,
                     "row count");
    passed &= reject("QSA top-k rejects zero cell capacity", valid, rows, 0u,
                     "cell capacity");
    passed &= reject("QSA top-k rejects unaligned cell capacity", valid,
                     rows, capacity + 1u, "cell capacity");
    passed &= reject("QSA top-k rejects capacity above StepRow range", valid,
                     rows, 0x80000000u, "cell capacity");

    const std::size_t max_blocks =
        static_cast<std::size_t>(capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t scores_bytes =
        static_cast<std::size_t>(rows) * max_blocks * sizeof(float);
    const std::size_t steps_bytes =
        static_cast<std::size_t>(rows) * sizeof(StepRow);
    QsaSelectionTopkBuffers short_scores = valid;
    short_scores.scores.length -= sizeof(float);
    passed &= reject("QSA top-k rejects short score view", short_scores,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionTopkBuffers short_steps = valid;
    short_steps.steps.length -= sizeof(std::int32_t);
    passed &= reject("QSA top-k rejects short StepRow view", short_steps,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionTopkBuffers short_ids = valid;
    short_ids.ids.length -= sizeof(std::int32_t);
    passed &= reject("QSA top-k rejects short id view", short_ids, rows,
                     capacity, "undersized or misaligned");

    twin::GuardedBuffer extra_scores;
    if (!extra_scores.allocate(context, "QSA top-k misaligned scores",
                               scores_bytes + 16u, error)) {
        check(false, "QSA top-k misaligned-score allocation", error);
        return false;
    }
    QsaSelectionTopkBuffers misaligned_scores = valid;
    misaligned_scores.scores =
        subview(extra_scores.view(), 1u, scores_bytes);
    passed &= reject("QSA top-k rejects misaligned score view",
                     misaligned_scores, rows, capacity,
                     "undersized or misaligned");

    std::memcpy(storage.ids.payload().data(), steps.data(), steps_bytes);
    QsaSelectionTopkBuffers score_alias = valid;
    score_alias.scores = subview(storage.ids.view(), 0u, scores_bytes);
    passed &= reject("QSA top-k rejects partial score/id overlap", score_alias,
                     rows, capacity, "must not overlap");
    QsaSelectionTopkBuffers steps_alias = valid;
    steps_alias.steps = subview(storage.ids.view(), 0u, steps_bytes);
    passed &= reject("QSA top-k rejects StepRow/id overlap", steps_alias,
                     rows, capacity, "must not overlap");

    auto reject_step = [&](const std::string& label, StepRow invalid) {
        std::memcpy(storage.steps.payload().data(), &invalid,
                    sizeof(invalid));
        const bool rejected =
            reject(label, valid, rows, capacity, "StepRow");
        std::memcpy(storage.steps.payload().data(), steps.data(),
                    steps.size() * sizeof(StepRow));
        return rejected;
    };
    StepRow invalid_step = steps[0];
    invalid_step.pos = static_cast<std::int32_t>(capacity);
    invalid_step.n_kv = invalid_step.pos + 1;
    invalid_step.n_bid =
        invalid_step.n_kv /
        static_cast<std::int32_t>(INDEXER_BLOCK_CELLS);
    invalid_step.width = invalid_step.n_kv;
    passed &= reject_step("QSA top-k rejects out-of-range StepRow pos",
                          invalid_step);
    invalid_step = steps[0];
    invalid_step.n_kv = 2;
    passed &= reject_step("QSA top-k rejects inconsistent StepRow n_kv",
                          invalid_step);
    invalid_step = steps[0];
    ++invalid_step.n_bid;
    passed &= reject_step("QSA top-k rejects inconsistent StepRow n_bid",
                          invalid_step);
    invalid_step = steps[0];
    ++invalid_step.width;
    passed &= reject_step("QSA top-k rejects inconsistent StepRow width",
                          invalid_step);

    twin::GuardedBuffer foreign_buffer;
    if (!foreign_buffer.allocate(foreign_context, "foreign QSA top-k scores",
                                 scores_bytes, error)) {
        check(false, "QSA top-k foreign allocation", error);
        return false;
    }
    QsaSelectionTopkBuffers foreign_view = valid;
    foreign_view.scores = foreign_buffer.view();
    passed &= reject("QSA top-k rejects foreign Context buffer", foreign_view,
                     rows, capacity, "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA top-k foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA top-k foreign command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_qsa_selection_topk(context, *foreign_command, rows, capacity,
                                    valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_command_rejected,
          "QSA top-k rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA top-k submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA top-k submitted command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_selection_topk(context, *submitted, rows, capacity, valid,
                                   error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA top-k rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA top-k submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "QSA top-k submitted command-buffer completion");
            passed = false;
        }
    }

    auto shared_command = context.make_command_buffer(
        "QSA top-k shared validation and late failure", error);
    if (!shared_command) {
        check(false, "QSA top-k shared command-buffer setup", error);
        return false;
    }
    const auto reject_shared = [&](Encoder& encoder,
                                   std::string& encode_error,
                                   const std::string& label,
                                   const QsaSelectionTopkBuffers& candidate,
                                   std::uint32_t token_rows,
                                   std::uint32_t cell_capacity,
                                   std::string_view message) {
        const bool rejected =
            !encode_qsa_selection_topk(
                context, encoder, token_rows, cell_capacity, candidate,
                encode_error) &&
            encode_error.find(message) != std::string::npos;
        check(rejected, label, encode_error);
        encode_error.clear();
        return rejected;
    };
    bool shared_validation_passed = true;
    const bool shared_validation_encoded = Encoder::encode(
        context, *shared_command, "QSA top-k shared validation",
        [&](Encoder& encoder, std::string& encode_error) {
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects zero rows", valid, 0u, capacity,
                "row count");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects invalid capacity", valid, rows,
                capacity + 1u, "cell capacity");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects a short score view", short_scores,
                rows, capacity, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects a short StepRow view", short_steps,
                rows, capacity, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects a short id view", short_ids, rows,
                capacity, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects id/score overlap", score_alias,
                rows, capacity, "must not overlap");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects id/StepRow overlap", steps_alias,
                rows, capacity, "must not overlap");
            StepRow invalid = steps[0];
            ++invalid.width;
            std::memcpy(storage.steps.payload().data(), &invalid,
                        sizeof(invalid));
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects inconsistent StepRow", valid,
                rows, capacity, "StepRow");
            std::memcpy(storage.steps.payload().data(), steps.data(),
                        steps.size() * sizeof(StepRow));
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects foreign Context buffer",
                foreign_view, rows, capacity, "not owned");
            QsaSelectionTopkBuffers forged_cpu = valid;
            forged_cpu.steps.contents =
                static_cast<char*>(forged_cpu.steps.contents) +
                sizeof(StepRow);
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects forged CPU metadata", forged_cpu,
                rows, capacity, "not owned");
            QsaSelectionTopkBuffers forged_gpu = valid;
            forged_gpu.ids.gpu_address ^= 0x10u;
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared top-k rejects forged GPU metadata", forged_gpu,
                rows, capacity, "not owned");
            return true;
        },
        error);
    check(shared_validation_encoded,
          "QSA shared top-k validation body completes", error);
    passed &= shared_validation_encoded && shared_validation_passed;

    std::fill(typed<std::int32_t>(storage.ids).begin(),
              typed<std::int32_t>(storage.ids).end(), kUntouchedId);
    QsaSelectionTopkBuffers late_short_scores = valid;
    late_short_scores.scores.length -= sizeof(float);
    error.clear();
    bool first_topk_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *shared_command, "QSA top-k shared late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_selection_topk(
                    context, encoder, rows, capacity, valid, encode_error)) {
                return false;
            }
            first_topk_encoded = true;
            return encode_qsa_selection_topk(
                context, encoder, rows, capacity, late_short_scores,
                encode_error);
        },
        error) &&
        first_topk_encoded &&
        error.find("undersized or misaligned") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared top-k propagates a late segment failure", error);
    passed &= late_failure_propagated;
    const bool late_ids_untouched =
        std::all_of(typed<std::int32_t>(storage.ids).begin(),
                    typed<std::int32_t>(storage.ids).end(),
                    [](std::int32_t id) { return id == kUntouchedId; });
    check(late_ids_untouched,
          "QSA shared top-k late failure leaves ids untouched");
    passed &= late_ids_untouched;
    error.clear();
    const bool late_failure_discarded =
        !shared_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared top-k late failure discards earlier work", error);
    passed &= late_failure_discarded;

    return passed;
}

bool prefill_concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
}

bool test_prefill_qsa_selection_validation(
    Context& context, Context& foreign_context) {
    constexpr std::uint32_t rows = 9u;
    constexpr std::uint32_t capacity = 64u;
    const std::vector<std::uint32_t> tokens(rows, 1u);
    PrefillChunkMetadata metadata;
    std::string error;
    if (!build_prefill_chunk_metadata(3u, tokens, metadata, error)) {
        check(false, "Prefill selection validation metadata", error);
        return false;
    }
    PrefillSelectionBuffers storage;
    if (!allocate_prefill_selection_buffers(
            context, rows, capacity, storage, error)) {
        check(false, "Prefill selection validation allocations", error);
        return false;
    }
    std::vector<StepRow> steps;
    for (const PrefillStepRow& step : metadata.steps) {
        steps.push_back(
            StepRow{step.pos, step.n_kv, step.n_bid, step.width});
    }
    std::memcpy(storage.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    std::fill(typed<float>(storage.scores).begin(),
              typed<float>(storage.scores).end(), kUntouchedScore);
    std::fill(typed<std::int32_t>(storage.ids).begin(),
              typed<std::int32_t>(storage.ids).end(), kUntouchedId);
    const QsaPrefillSelectionBuffers valid = storage.views();

    const auto rejected = [&](const std::string& label,
                              const strata::metal::PrefillParams& params,
                              std::uint32_t cell_capacity,
                              const QsaPrefillSelectionBuffers& buffers,
                              std::string_view expected_error) {
        error.clear();
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        const bool encoded = encode_prefill_qsa_selection(
            context, *command, params, cell_capacity, buffers, error);
        const bool is_rejected =
            !encoded && command->unsafe_native_handle() == nullptr &&
            error.find(expected_error) != std::string::npos;
        check(is_rejected, label, error);
        return is_rejected;
    };

    bool passed = true;
    strata::metal::PrefillParams invalid = metadata.params;
    invalid.idx_f += INDEXER_BLOCK_CELLS;
    passed &= rejected("Prefill selection rejects mismatched idx_f", invalid,
                       capacity, valid, "parameters do not match");
    invalid = metadata.params;
    invalid.idx_e += INDEXER_BLOCK_CELLS;
    passed &= rejected("Prefill selection rejects mismatched idx_e", invalid,
                       capacity, valid, "parameters do not match");
    invalid = metadata.params;
    invalid.T = 0u;
    passed &= rejected("Prefill selection rejects zero rows", invalid,
                       capacity, valid, "row count");
    invalid = metadata.params;
    invalid.T = kPrefillMaxShaderRows + 1u;
    passed &= rejected("Prefill selection rejects rows above shader limit",
                       invalid, capacity, valid, "row count");
    invalid = metadata.params;
    invalid.p0 = 60u;
    passed &= rejected("Prefill selection rejects chunk beyond capacity",
                       invalid, capacity, valid, "exceeds the cell capacity");
    passed &= rejected("Prefill selection rejects unaligned cell capacity",
                       metadata.params, capacity - 1u, valid,
                       "cell capacity is invalid");
    strata::metal::PrefillParams at_limit{
        kPrefillMaxShaderRows,
        0u,
        INDEXER_BLOCK_CELLS,
        kPrefillMaxShaderRows &
            ~(INDEXER_BLOCK_CELLS - 1u),
    };
    const std::uint32_t limit_capacity =
        (at_limit.T + INDEXER_BLOCK_CELLS - 1u) &
        ~(INDEXER_BLOCK_CELLS - 1u);
    passed &= rejected("Prefill selection accepts inclusive shader row limit",
                       at_limit, limit_capacity,
                       valid, "pooled keys");

    QsaPrefillSelectionBuffers short_scores = valid;
    short_scores.scores.length -= sizeof(float);
    passed &= rejected("Prefill selection rejects short score scratch",
                       metadata.params, capacity, short_scores,
                       "undersized or misaligned");
    QsaPrefillSelectionBuffers short_ids = valid;
    short_ids.ids.length -= sizeof(std::int32_t);
    passed &= rejected("Prefill selection rejects short ID output",
                       metadata.params, capacity, short_ids,
                       "undersized or misaligned");
    QsaPrefillSelectionBuffers output_input_alias = valid;
    output_input_alias.pooled = valid.scores;
    passed &= rejected("Prefill selection rejects score/input alias",
                       metadata.params, capacity, output_input_alias,
                       "outputs must not overlap inputs");
    QsaPrefillSelectionBuffers output_alias = valid;
    output_alias.scores = valid.ids;
    passed &= rejected("Prefill selection rejects score/ID output alias",
                       metadata.params, capacity, output_alias,
                       "score and ID outputs must be disjoint");

    twin::GuardedBuffer foreign_scores;
    if (!foreign_scores.allocate(
            foreign_context, "foreign Prefill selection scratch",
            storage.scores.payload().size(), error)) {
        check(false, "Prefill selection foreign allocation", error);
        return false;
    }
    QsaPrefillSelectionBuffers foreign_view = valid;
    foreign_view.scores = foreign_scores.view();
    passed &= rejected("Prefill selection rejects foreign Context buffer",
                       metadata.params, capacity, foreign_view, "not owned");

    StepRow invalid_step = steps[4u];
    ++invalid_step.pos;
    std::memcpy(storage.steps.payload().data() +
                    4u * sizeof(StepRow),
                &invalid_step, sizeof(invalid_step));
    passed &= rejected("Prefill selection rejects noncontiguous StepRows",
                       metadata.params, capacity, valid,
                       "contiguous positions");
    std::memcpy(storage.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));

    if (prefill_concurrent_dispatch_mode()) {
        auto group_command = context.make_command_buffer(
            "Prefill selection open-group validation", error);
        if (!group_command) {
            check(false, "Prefill selection open-group command buffer",
                  error);
            return false;
        }
        bool rejected_in_group = false;
        const bool group_encoded = Encoder::encode(
            context, *group_command, "Prefill selection open group",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encoder.group_begin(encode_error)) return false;
                std::string selection_error;
                rejected_in_group =
                    !encode_prefill_qsa_selection(
                        context, encoder, metadata.params, capacity, valid,
                        selection_error) &&
                    selection_error.find("ordered concurrent-group") !=
                        std::string::npos;
                const bool group_closed = encoder.group_end(encode_error);
                return rejected_in_group && group_closed;
            },
            error);
        check(group_encoded && rejected_in_group,
              "Prefill selection rejects an open concurrent group", error);
        passed &= group_encoded && rejected_in_group;
        group_command->discard();
    }

    const std::size_t max_blocks =
        static_cast<std::size_t>(capacity / INDEXER_BLOCK_CELLS) + 2u;
    const QsaSelectionScoresBuffers earlier_scores{
        valid.pooled,
        valid.dead,
        valid.indexer_queries,
        valid.steps,
        valid.scores.slice(
            0u, static_cast<std::size_t>(rows) * max_blocks *
                    sizeof(float)),
    };
    strata::metal::PrefillParams late_invalid = metadata.params;
    late_invalid.idx_f += INDEXER_BLOCK_CELLS;
    auto rollback_command = context.make_command_buffer(
        "Prefill selection shared rollback", error);
    if (!rollback_command) {
        check(false, "Prefill selection rollback command buffer", error);
        return false;
    }
    bool earlier_dispatch_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
            context, *rollback_command, "Prefill selection shared rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            earlier_dispatch_encoded = encode_qsa_selection_scores(
                context, encoder, VERIFY_MAX_T, capacity, earlier_scores,
                encode_error);
            return earlier_dispatch_encoded &&
                encode_prefill_qsa_selection(
                    context, encoder, late_invalid, capacity, valid,
                    encode_error);
        },
        error) &&
        earlier_dispatch_encoded &&
        error.find("parameters do not match") != std::string::npos;
    check(late_failure_propagated && earlier_dispatch_encoded,
          "Prefill selection propagates late metadata failure", error);
    passed &= late_failure_propagated && earlier_dispatch_encoded;
    const bool scores_unchanged = std::all_of(
        typed<float>(storage.scores).begin(),
        typed<float>(storage.scores).end(),
        [](float value) { return value == kUntouchedScore; });
    check(scores_unchanged,
          "Prefill selection rollback preserves prior score output");
    passed &= scores_unchanged;
    error.clear();
    const bool rollback_discarded =
        !rollback_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(rollback_discarded,
          "Prefill selection rollback discards prior score dispatch", error);
    passed &= rollback_discarded;
    return passed;
}

bool test_scores_then_topk_same_command(
    Context& context, const twin::Options& options,
    bool shared_encoder = false) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t capacity = 8196u;
    constexpr std::array<std::uint32_t, rows> n_kv{8193u};
    const std::size_t max_blocks =
        static_cast<std::size_t>(capacity / INDEXER_BLOCK_CELLS) + 2u;
    std::mt19937_64 rng(options.seed ^ 0x515341746f706bULL);
    ScoreBuffers score_buffers;
    TopkBuffers topk_storage;
    std::string error;
    if (!allocate_score_buffers(context, rows, capacity, score_buffers,
                                error) ||
        !allocate_topk_buffers(context, rows, capacity, topk_storage, error)) {
        check(false, "QSA scores-to-top-k allocations", error);
        return false;
    }

    std::vector<float> pooled(max_blocks * INDEXER_HEAD_DIM);
    std::vector<float> dead(INDEXER_HEAD_DIM);
    std::vector<float> queries(
        static_cast<std::size_t>(rows) * INDEXER_HEADS * INDEXER_HEAD_DIM);
    for (std::size_t i = 0u; i < pooled.size(); ++i) {
        pooled[i] = input_value(twin::InputSet::fixture, i + 17u, rng);
    }
    for (std::size_t i = 0u; i < dead.size(); ++i) {
        dead[i] = input_value(twin::InputSet::fixture, i + 0x1001u, rng);
    }
    for (std::size_t i = 0u; i < queries.size(); ++i) {
        queries[i] = input_value(twin::InputSet::fixture, i + 0x2003u, rng);
    }
    const std::vector<StepRow> steps = make_topk_steps(n_kv);
    std::memcpy(score_buffers.pooled.payload().data(), pooled.data(),
                pooled.size() * sizeof(float));
    std::memcpy(score_buffers.dead.payload().data(), dead.data(),
                dead.size() * sizeof(float));
    std::memcpy(score_buffers.queries.payload().data(), queries.data(),
                queries.size() * sizeof(float));
    std::memcpy(score_buffers.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    std::fill(typed<float>(score_buffers.scores).begin(),
              typed<float>(score_buffers.scores).end(), kUntouchedScore);
    std::fill(typed<std::int32_t>(topk_storage.ids).begin(),
              typed<std::int32_t>(topk_storage.ids).end(), kUntouchedId);

    const std::vector<std::int32_t> n_kv_i32{
        static_cast<std::int32_t>(n_kv[0])};
    const std::vector<std::int32_t> n_bid{steps[0].n_bid};
    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        pooled, dead, queries, n_kv_i32, n_bid,
        static_cast<std::uint32_t>(max_blocks), kUntouchedScore,
        expected_scores);
    std::vector<float> expected_score_values;
    expected_score_values.reserve(expected_scores.size());
    for (const twin::tf value : expected_scores) {
        expected_score_values.push_back(value.value);
    }
    std::vector<std::int32_t> expected_ids;
    oracle::qsa_selection_topk(expected_score_values, n_kv_i32, n_bid,
                               static_cast<std::uint32_t>(max_blocks),
                               kUntouchedId, expected_ids);

    score_buffers.pooled.capture_before_dispatch();
    score_buffers.dead.capture_before_dispatch();
    score_buffers.queries.capture_before_dispatch();
    score_buffers.steps.capture_before_dispatch();
    score_buffers.scores.capture_before_dispatch();
    topk_storage.ids.capture_before_dispatch();
    const std::string label = shared_encoder
        ? "QSA scores then top-k shared Encoder segment"
        : "QSA scores then top-k same command";
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, "QSA scores-to-top-k command buffer", error);
        return false;
    }
    const QsaSelectionTopkBuffers topk_buffers{
        score_buffers.scores.view(), score_buffers.steps.view(),
        topk_storage.ids.view()};
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_selection_scores(
                             context, encoder, rows, capacity,
                             score_buffers.views(), encode_error) &&
                         encode_qsa_selection_topk(
                             context, encoder, rows, capacity, topk_buffers,
                             encode_error);
              },
              error)
        : encode_qsa_selection_scores(
              context, *command, rows, capacity, score_buffers.views(),
              error) &&
              encode_qsa_selection_topk(
                  context, *command, rows, capacity, topk_buffers, error);
    if (!encoded) {
        check(false, "QSA scores-to-top-k encode chain", error);
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, "QSA scores-to-top-k GPU completion", error);
        return false;
    }

    const twin::Tolerance exact{twin::Comparison::bits, 0.0, 0.0, 0u};
    const auto score_difference = twin::compare_floats(
        typed<float>(score_buffers.scores), expected_scores, exact);
    if (score_difference) {
        twin::report_failure("QSA scores-to-top-k", options, rows,
                             *score_difference);
    }
    check(!score_difference,
          "QSA same-command scores match the ordered scalar oracle");
    const auto actual_ids = typed<std::int32_t>(topk_storage.ids);
    const auto mismatch =
        std::mismatch(actual_ids.begin(), actual_ids.end(),
                      expected_ids.begin());
    const bool ids_match = mismatch.first == actual_ids.end();
    check(ids_match,
          "QSA same-command top-k ids match the sorted scalar oracle",
          ids_match ? std::string{}
                    : "first differing id index " +
                          std::to_string(mismatch.first - actual_ids.begin()));

    const std::span<const twin::ByteRange> no_writes;
    const bool inputs_unchanged =
        !score_buffers.pooled.first_unwritten_change(no_writes) &&
        !score_buffers.dead.first_unwritten_change(no_writes) &&
        !score_buffers.queries.first_unwritten_change(no_writes) &&
        !score_buffers.steps.first_unwritten_change(no_writes);
    check(inputs_unchanged, "QSA scores-to-top-k inputs are unchanged");
    const std::array scores_written{
        twin::ByteRange{0u, score_buffers.scores.payload().size()}};
    const bool score_guards =
        !score_buffers.scores.first_unwritten_change(scores_written);
    const std::array ids_written{
        twin::ByteRange{0u, topk_storage.ids.payload().size()}};
    const bool id_guards =
        !topk_storage.ids.first_unwritten_change(ids_written);
    check(score_guards && id_guards,
          "QSA scores-to-top-k output guards are intact");
    return !score_difference && ids_match && inputs_unchanged &&
           score_guards && id_guards;
}

bool test_relu_is_applied_per_head(Context& context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::uint32_t capacity = kCellCapacity;
    const std::size_t max_blocks =
        static_cast<std::size_t>(capacity / INDEXER_BLOCK_CELLS) + 2u;
    std::string error;
    ScoreBuffers buffers;
    if (!allocate_score_buffers(context, rows, capacity, buffers, error)) {
        check(false, "QSA selection ReLU fixture allocations", error);
        return false;
    }
    auto pooled = typed<float>(buffers.pooled);
    auto dead = typed<float>(buffers.dead);
    auto queries = typed<float>(buffers.queries);
    auto scores = typed<float>(buffers.scores);
    std::fill(pooled.begin(), pooled.end(), 1.0f);
    std::fill(dead.begin(), dead.end(), 1.0f);
    std::fill(scores.begin(), scores.end(), kUntouchedScore);
    constexpr std::array<float, INDEXER_HEADS> head_values{
        1.0f, -2.0f, 0.5f, -0.25f};
    for (std::uint32_t head = 0u; head < INDEXER_HEADS; ++head) {
        std::fill_n(queries.begin() +
                        static_cast<std::size_t>(head) * INDEXER_HEAD_DIM,
                    INDEXER_HEAD_DIM, head_values[head]);
    }
    const StepRow step{3, 4, 1, 4};
    std::memcpy(buffers.steps.payload().data(), &step, sizeof(step));

    auto command = context.make_command_buffer(
        "QSA selection per-head ReLU fixture", error);
    if (!command ||
        !encode_qsa_selection_scores(context, *command, rows, capacity,
                                     buffers.views(), error)) {
        check(false, "QSA selection per-head ReLU encode", error);
        if (command) command->discard();
        return false;
    }
    if (!submit_and_wait(command, "QSA selection per-head ReLU", error)) {
        check(false, "QSA selection per-head ReLU GPU completion", error);
        return false;
    }
    const bool exact_expected =
        scores[0] == 192.0f && scores[1] == 192.0f;
    check(exact_expected,
          "QSA selection applies ReLU to each head before summing");
    return exact_expected && max_blocks == scores.size();
}

bool test_encoder_validation(Context& context, Context& foreign_context) {
    constexpr std::uint32_t rows = 2u;
    constexpr std::uint32_t capacity = kCellCapacity;
    std::string error;
    ScoreBuffers storage;
    if (!allocate_score_buffers(context, rows, capacity, storage, error)) {
        check(false, "QSA selection validation allocations", error);
        return false;
    }
    const std::vector<StepRow> steps = make_steps(rows, capacity);
    std::memcpy(storage.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    QsaSelectionScoresBuffers valid = storage.views();
    bool passed = true;
    auto reject = [&](const std::string& label,
                      const QsaSelectionScoresBuffers& candidate,
                      std::uint32_t token_rows, std::uint32_t cell_capacity,
                      std::string_view message) {
        auto command = context.make_command_buffer(label, error);
        if (!command) {
            check(false, label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !encode_qsa_selection_scores(
                context, *command, token_rows, cell_capacity, candidate,
                error) &&
            error.find(message) != std::string::npos;
        check(rejected, label, error);
        command->discard();
        return rejected;
    };

    passed &= reject("QSA selection rejects zero rows", valid, 0u, capacity,
                     "row count");
    passed &= reject("QSA selection rejects too many rows", valid, 9u,
                     capacity, "row count");
    passed &= reject("QSA selection rejects unaligned cell capacity", valid,
                     rows, capacity + 1u, "cell capacity");
    passed &= reject("QSA selection rejects zero cell capacity", valid, rows,
                     0u, "cell capacity");
    passed &= reject("QSA selection rejects capacity above StepRow range",
                     valid, rows, 0x80000000u, "cell capacity");

    const std::size_t max_blocks =
        static_cast<std::size_t>(capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t score_bytes =
        static_cast<std::size_t>(rows) * max_blocks * sizeof(float);

    QsaSelectionScoresBuffers short_pooled = valid;
    short_pooled.pooled.length -= 16u + sizeof(float);
    passed &= reject("QSA selection rejects short pooled view", short_pooled,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionScoresBuffers short_dead = valid;
    short_dead.dead.length -= sizeof(float);
    passed &= reject("QSA selection rejects short dead-key view", short_dead,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionScoresBuffers short_queries = valid;
    short_queries.indexer_queries.length -= sizeof(float);
    passed &= reject("QSA selection rejects short query view", short_queries,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionScoresBuffers short_steps = valid;
    short_steps.steps.length -= sizeof(std::int32_t);
    passed &= reject("QSA selection rejects short StepRow view", short_steps,
                     rows, capacity, "undersized or misaligned");
    QsaSelectionScoresBuffers short_scores = valid;
    short_scores.scores.length -= sizeof(float);
    passed &= reject("QSA selection rejects short score view", short_scores,
                     rows, capacity, "undersized or misaligned");

    QsaSelectionScoresBuffers output_alias = valid;
    output_alias.scores =
        subview(storage.pooled.view(), sizeof(float), score_bytes);
    passed &= reject("QSA selection rejects partial pooled/output overlap",
                     output_alias, rows, capacity,
                     "must not overlap an input");
    QsaSelectionScoresBuffers dead_alias = valid;
    dead_alias.scores = subview(storage.dead.view(), 0u, score_bytes);
    passed &= reject("QSA selection rejects dead/output overlap", dead_alias,
                     rows, capacity, "must not overlap an input");
    QsaSelectionScoresBuffers query_alias = valid;
    query_alias.scores = subview(storage.queries.view(), 0u, score_bytes);
    passed &= reject("QSA selection rejects query/output overlap",
                     query_alias, rows, capacity,
                     "must not overlap an input");
    twin::GuardedBuffer steps_alias_storage;
    if (!steps_alias_storage.allocate(context, "QSA selection steps alias",
                                      score_bytes, error)) {
        check(false, "QSA selection steps-alias allocation", error);
        return false;
    }
    std::memcpy(steps_alias_storage.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    QsaSelectionScoresBuffers steps_alias = valid;
    steps_alias.steps = subview(
        steps_alias_storage.view(), 0u, rows * sizeof(StepRow));
    steps_alias.scores = steps_alias_storage.view();
    passed &= reject("QSA selection rejects steps/output overlap",
                     steps_alias, rows, capacity,
                     "must not overlap an input");
    QsaSelectionScoresBuffers misaligned_pool = valid;
    misaligned_pool.pooled = subview(storage.pooled.view(), sizeof(float),
                                     storage.pooled.view().length -
                                         sizeof(float));
    passed &= reject("QSA selection rejects misaligned vector input",
                     misaligned_pool, rows, capacity,
                     "undersized or misaligned");

    auto reject_step = [&](const std::string& label, StepRow invalid) {
        std::memcpy(storage.steps.payload().data(), &invalid,
                    sizeof(invalid));
        const bool rejected =
            reject(label, valid, rows, capacity, "StepRow");
        std::memcpy(storage.steps.payload().data(), steps.data(),
                    steps.size() * sizeof(StepRow));
        return rejected;
    };
    StepRow invalid_step = steps[0];
    invalid_step.pos = static_cast<std::int32_t>(capacity);
    invalid_step.n_kv = invalid_step.pos + 1;
    invalid_step.n_bid =
        invalid_step.n_kv /
        static_cast<std::int32_t>(INDEXER_BLOCK_CELLS);
    invalid_step.width = invalid_step.n_kv;
    passed &= reject_step("QSA selection rejects out-of-range StepRow pos",
                         invalid_step);
    invalid_step = steps[0];
    invalid_step.n_kv = 2;
    invalid_step.n_bid = 0;
    invalid_step.width = 2;
    passed &= reject_step("QSA selection rejects inconsistent StepRow n_kv",
                         invalid_step);
    invalid_step = steps[0];
    ++invalid_step.n_bid;
    passed &= reject_step("QSA selection rejects inconsistent StepRow n_bid",
                         invalid_step);
    invalid_step = steps[0];
    ++invalid_step.width;
    passed &= reject_step("QSA selection rejects inconsistent StepRow width",
                         invalid_step);

    twin::GuardedBuffer foreign_buffer;
    if (!foreign_buffer.allocate(foreign_context, "foreign QSA scores",
                                 score_bytes, error)) {
        check(false, "QSA selection foreign allocation", error);
        return false;
    }
    QsaSelectionScoresBuffers foreign_view = valid;
    foreign_view.scores = foreign_buffer.view();
    passed &= reject("QSA selection rejects foreign Context buffer",
                     foreign_view, rows, capacity, "not owned");

    auto foreign_command = foreign_context.make_command_buffer(
        "QSA selection foreign command buffer", error);
    if (!foreign_command) {
        check(false, "QSA selection foreign command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_qsa_selection_scores(context, *foreign_command, rows,
                                     capacity, valid, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_command_rejected,
          "QSA selection rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "QSA selection submitted command buffer", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "QSA selection submitted command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_selection_scores(context, *submitted, rows, capacity,
                                     valid, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "QSA selection rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "QSA selection submitted command buffer native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false,
                  "QSA selection submitted command-buffer completion");
            passed = false;
        }
    }

    auto shared_command = context.make_command_buffer(
        "QSA selection shared validation", error);
    if (!shared_command) {
        check(false, "QSA selection shared validation command buffer", error);
        return false;
    }
    const auto reject_shared = [&](Encoder& encoder,
                                   std::string& encode_error,
                                   const std::string& label,
                                   const QsaSelectionScoresBuffers& candidate,
                                   std::uint32_t token_rows,
                                   std::uint32_t cell_capacity,
                                   std::string_view message) {
        const bool rejected =
            !encode_qsa_selection_scores(
                context, encoder, token_rows, cell_capacity, candidate,
                encode_error) &&
            encode_error.find(message) != std::string::npos;
        check(rejected, label, encode_error);
        encode_error.clear();
        return rejected;
    };
    bool shared_validation_passed = true;
    const bool shared_validation_encoded = Encoder::encode(
        context, *shared_command, "QSA selection shared validation",
        [&](Encoder& encoder, std::string& encode_error) {
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects zero rows", valid, 0u,
                capacity, "row count");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects invalid capacity", valid, rows,
                capacity + 1u, "cell capacity");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects a short score view",
                short_scores, rows, capacity, "undersized or misaligned");
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects output/input overlap",
                output_alias, rows, capacity, "must not overlap an input");
            StepRow invalid = steps[0];
            ++invalid.width;
            std::memcpy(storage.steps.payload().data(), &invalid,
                        sizeof(invalid));
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects inconsistent StepRow", valid,
                rows, capacity, "StepRow");
            std::memcpy(storage.steps.payload().data(), steps.data(),
                        steps.size() * sizeof(StepRow));
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects foreign Context buffer",
                foreign_view, rows, capacity, "not owned");
            QsaSelectionScoresBuffers forged_cpu = valid;
            forged_cpu.steps.contents =
                static_cast<char*>(forged_cpu.steps.contents) +
                sizeof(StepRow);
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects forged CPU metadata",
                forged_cpu, rows, capacity, "not owned");
            QsaSelectionScoresBuffers forged_gpu = valid;
            forged_gpu.scores.gpu_address ^= 0x10u;
            shared_validation_passed &= reject_shared(
                encoder, encode_error,
                "QSA shared selection rejects forged GPU metadata",
                forged_gpu, rows, capacity, "not owned");
            return true;
        },
        error);
    check(shared_validation_encoded,
          "QSA shared selection validation body completes", error);
    passed &= shared_validation_encoded && shared_validation_passed;

    std::fill(typed<float>(storage.scores).begin(),
              typed<float>(storage.scores).end(), kUntouchedScore);
    QsaSelectionScoresBuffers late_short_scores = valid;
    late_short_scores.scores.length -= sizeof(float);
    error.clear();
    bool first_score_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *shared_command, "QSA selection shared late failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_selection_scores(
                    context, encoder, rows, capacity, valid, encode_error)) {
                return false;
            }
            first_score_encoded = true;
            return encode_qsa_selection_scores(
                context, encoder, rows, capacity, late_short_scores,
                encode_error);
        },
        error) &&
        first_score_encoded &&
        error.find("undersized or misaligned") != std::string::npos;
    check(late_failure_propagated,
          "QSA shared selection propagates a late segment failure", error);
    passed &= late_failure_propagated;
    const bool late_scores_untouched = std::all_of(
        typed<float>(storage.scores).begin(),
        typed<float>(storage.scores).end(),
        [](float value) { return value == kUntouchedScore; });
    check(late_scores_untouched,
          "QSA shared selection late failure leaves scores untouched");
    passed &= late_scores_untouched;
    error.clear();
    const bool late_failure_discarded =
        !shared_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "QSA shared selection late failure discards earlier work", error);
    passed &= late_failure_discarded;
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_qsa_selection_test strata.metallib "
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
        std::fprintf(stderr,
                     "metal_qsa_selection_test has no model-backed real "
                     "input set\n");
        return 77;
    }

    ContextOptions context_options;
    context_options.metallib = argv[1];
    context_options.metallib_sha256 = STRATA_METALLIB_SHA256;
    auto context = Context::open(context_options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;
    ContextOptions foreign_options = context_options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign validation context",
          error);
    if (!foreign_context) return 1;

    bool passed = test_encoder_validation(*context, *foreign_context);
    passed &= test_topk_encoder_validation(*context, *foreign_context);
    passed &= test_prefill_qsa_selection_validation(
        *context, *foreign_context);
    passed &= test_topk_correctness(*context, options);
    passed &= test_scores_then_topk_same_command(*context, options);
    passed &= test_scores_then_topk_same_command(*context, options, true);
    passed &= test_relu_is_applied_per_head(*context);
    constexpr std::array sets{twin::InputSet::random, twin::InputSet::edge,
                              twin::InputSet::fixture};
    for (const auto set : sets) {
        for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
            passed &= run_score_case(*context, rows, kCellCapacity, set,
                                     options);
        }
    }
    passed &= run_score_case(*context, 1u, kCellCapacity,
                             twin::InputSet::fixture, options,
                             " shared-window", true);
    passed &= run_score_case(*context, VERIFY_MAX_T, kCellCapacity,
                             twin::InputSet::fixture, options,
                             " shared-batch", true);
    passed &= run_score_case(*context, 1u, 131072u,
                             twin::InputSet::fixture, options,
                             " grid-stride");
    struct PrefillSelectionCase {
        std::uint32_t pos0;
        std::uint32_t rows;
        std::uint32_t capacity;
    };
    constexpr std::array prefill_selection_cases{
        PrefillSelectionCase{0u, 1u, 1024u},
        PrefillSelectionCase{1u, 7u, 1024u},
        PrefillSelectionCase{3u, 8u, 1024u},
        PrefillSelectionCase{4u, 9u, 1024u},
        PrefillSelectionCase{6u, 255u, 1024u},
        PrefillSelectionCase{0u, 256u, 1024u},
        PrefillSelectionCase{3u, 257u, 1024u},
        PrefillSelectionCase{6u, 300u, 1024u},
        PrefillSelectionCase{2200u, 9u, 4096u},
        PrefillSelectionCase{100000u, 1u, 135168u},
    };
    for (const PrefillSelectionCase& prefill_case :
         prefill_selection_cases) {
        passed &= run_prefill_selection_case(
            *context, prefill_case.rows, prefill_case.pos0,
            prefill_case.capacity, options, false);
        passed &= run_prefill_selection_case(
            *context, prefill_case.rows, prefill_case.pos0,
            prefill_case.capacity, options, true);
    }
    passed &= test_prefill_indexer_selection_composition(
        *context, options, 3u, 64u);
    passed &= test_prefill_indexer_selection_composition(
        *context, options, 2047u, 4096u);
    check(passed && g_failures == 0,
          "QSA selection score and top-k Metal encoder/kernel twins");
    std::printf("metal_qsa_selection_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
