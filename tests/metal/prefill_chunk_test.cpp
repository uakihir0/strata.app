// Device-free tests for the host metadata of a Metal Prefill chunk.
#include "strata/metal/prefill.hpp"

#include <array>
#include <cstdint>
#include <iostream>
#include <limits>
#include <span>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) return;
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

bool build(std::uint32_t pos0, std::span<const std::uint32_t> tokens,
           strata::metal::PrefillChunkMetadata& metadata,
           std::string& error) {
    return strata::metal::build_prefill_chunk_metadata(pos0, tokens, metadata,
                                                        error);
}

void builds_chunk_metadata() {
    using namespace strata::metal;
    constexpr std::uint32_t image_token = kPrefillImageTokenFlag | 17u;
    const std::vector<std::uint32_t> tokens{3u, image_token, 5u, 7u, 11u};
    PrefillChunkMetadata metadata;
    std::string error;
    const bool built = build(0u, tokens, metadata, error);
    check(built, "builds a non-empty prompt chunk");
    if (!built) {
        std::cerr << "metadata error: " << error << '\n';
        return;
    }

    check(metadata.params.T == 5u && metadata.params.p0 == 0u &&
              metadata.params.idx_f == 4u && metadata.params.idx_e == 4u,
          "PrefillParams match the first chunk's indexer boundaries");
    check(metadata.batch_rows ==
              std::array<std::uint32_t, 6>{8u, 5u, 256u, 5u, 32u, 5u},
          "the six query slots carry the specified full and remainder sizes");
    check(metadata.tokens == tokens,
          "input token ids and image flags are copied verbatim");
    check(metadata.has_image_rows,
          "image-marked tokens request the image embedding pass");
    check(metadata.steps.size() == tokens.size(),
          "one StepRow is produced for each live token");
    if (metadata.steps.size() == 5u) {
        check(metadata.steps[0].pos == 0 && metadata.steps[0].n_kv == 1 &&
                  metadata.steps[0].n_bid == 0 &&
                  metadata.steps[0].width == 1,
              "first StepRow starts with one visible KV cell");
        check(metadata.steps[3].pos == 3 && metadata.steps[3].n_kv == 4 &&
                  metadata.steps[3].n_bid == 1 &&
                  metadata.steps[3].width == 4,
              "StepRow counts the first complete four-cell indexer block");
        check(metadata.steps[4].pos == 4 && metadata.steps[4].n_kv == 5 &&
                  metadata.steps[4].n_bid == 1 &&
                  metadata.steps[4].width == 5,
              "StepRow advances positions and selected width by one");
    }
}

void computes_position_boundaries() {
    using namespace strata::metal;
    std::string error;
    PrefillChunkMetadata metadata;

    const std::vector<std::uint32_t> two{1u, 2u};
    check(build(3u, two, metadata, error),
          "builds a chunk crossing the first four-cell boundary");
    check(metadata.params.idx_f == 4u && metadata.params.idx_e == 4u,
          "idx_f rounds p0 up and idx_e rounds the exclusive end down");

    const std::vector<std::uint32_t> three{1u, 2u, 3u};
    check(build(2049u, three, metadata, error),
          "builds a chunk across the indexer and width boundary");
    check(metadata.params.idx_f == 2052u && metadata.params.idx_e == 2052u,
          "indexer boundaries use four-cell blocks");
    check(metadata.steps.size() == 3u &&
              metadata.steps[0].n_bid == 512 &&
              metadata.steps[1].n_bid == 512 &&
              metadata.steps[2].n_bid == 513 &&
              metadata.steps[0].width == 2050 &&
              metadata.steps[1].width == 2051 &&
              metadata.steps[2].width == 2051,
          "StepRow clamps width at 2051 and advances complete-block count");

    const std::vector<std::uint32_t> one{9u};
    const auto last_valid_pos =
        static_cast<std::uint32_t>(std::numeric_limits<std::int32_t>::max() - 1);
    check(build(last_valid_pos, one, metadata, error),
          "accepts the largest position whose n_kv fits signed StepRow");
    check(metadata.steps.size() == 1u &&
              metadata.steps[0].pos ==
                  std::numeric_limits<std::int32_t>::max() - 1 &&
              metadata.steps[0].n_kv ==
                  std::numeric_limits<std::int32_t>::max(),
          "the last valid StepRow does not overflow signed fields");
}

void computes_batch_remainders() {
    using namespace strata::metal;
    std::string error;
    for (const std::uint32_t rows : {1u, 8u, 9u, 31u, 32u, 33u, 255u, 256u,
                                     257u}) {
        const std::vector<std::uint32_t> tokens(rows, 23u);
        PrefillChunkMetadata metadata;
        check(build(64u, tokens, metadata, error),
              "builds batch-slot boundary row counts");
        if (!metadata.steps.empty()) {
            check(metadata.params.T == rows &&
                      metadata.batch_rows[0] == 8u &&
                      metadata.batch_rows[1] == rows % 8u &&
                      metadata.batch_rows[2] == 256u &&
                      metadata.batch_rows[3] == rows % 256u &&
                      metadata.batch_rows[4] == 32u &&
                      metadata.batch_rows[5] == rows % 32u,
                  "full and remainder batch slots match the documented order");
            check(!metadata.has_image_rows,
                  "text-only chunks do not request the image pass");
        }
    }
}

void builds_chunk_schedule_without_padding() {
    using namespace strata::metal;
    const std::vector<std::uint32_t> tokens{
        3u, 5u, kPrefillImageTokenFlag | 7u, 11u, 13u,
        17u, 19u, 23u, 29u, 31u,
    };
    std::vector<PrefillChunkRange> schedule;
    std::string error;
    check(build_prefill_chunk_schedule(7u, tokens.size(), 4u, schedule,
                                       error),
          "plans a prompt as fixed-size chunks with a short tail");
    check(schedule.size() == 3u,
          "the schedule contains one range for each non-empty chunk");
    if (schedule.size() != 3u) return;

    const std::array<PrefillChunkRange, 3> expected{{
        {0u, 7u, 4u},
        {4u, 11u, 4u},
        {8u, 15u, 2u},
    }};
    std::size_t covered_rows = 0u;
    for (std::size_t index = 0u; index < schedule.size(); ++index) {
        const auto& range = schedule[index];
        check(range.token_offset == expected[index].token_offset &&
                  range.pos0 == expected[index].pos0 &&
                  range.rows == expected[index].rows,
              "scheduled ranges are contiguous and retain absolute positions");
        check(range.rows <= 4u && range.rows != 0u,
              "scheduled chunks are non-empty and never exceed the requested size");
        covered_rows += range.rows;

        PrefillChunkMetadata metadata;
        const auto chunk_tokens =
            std::span<const std::uint32_t>(tokens).subspan(range.token_offset,
                                                           range.rows);
        check(build_prefill_chunk_metadata(range.pos0, chunk_tokens, metadata,
                                           error),
              "each scheduled range builds valid chunk metadata");
        if (metadata.params.T != 0u) {
            check(metadata.params.T == range.rows &&
                      metadata.params.p0 == range.pos0 &&
                      metadata.tokens.size() == range.rows,
                  "chunk metadata matches its scheduled token slice");
        }
    }
    check(covered_rows == tokens.size(),
          "scheduled chunks cover every prompt token exactly once");

    std::vector<PrefillChunkRange> empty_schedule{{0u, 9u, 1u}};
    check(build_prefill_chunk_schedule(
              std::numeric_limits<std::uint32_t>::max(), 0u, 4u,
              empty_schedule, error) &&
              empty_schedule.empty(),
          "an empty prompt produces an empty schedule and clears old ranges");
}

void rejects_invalid_chunk_schedules_transactionally() {
    using namespace strata::metal;
    std::vector<PrefillChunkRange> schedule{{5u, 17u, 3u}};
    std::string error;
    const auto remains_unchanged = [&] {
        return schedule.size() == 1u && schedule[0].token_offset == 5u &&
               schedule[0].pos0 == 17u && schedule[0].rows == 3u;
    };

    check(!build_prefill_chunk_schedule(0u, 3u, 0u, schedule, error) &&
              error.find("shader row limit") != std::string::npos,
          "rejects a zero-row chunk size");
    check(remains_unchanged(),
          "zero-row chunk-size failure preserves the previous schedule");

    check(!build_prefill_chunk_schedule(
              0u, 3u, kPrefillMaxShaderRows + 1u, schedule, error) &&
              error.find("shader row limit") != std::string::npos,
          "rejects a chunk size above the shader row limit");
    check(remains_unchanged(),
          "oversized chunk-size failure preserves the previous schedule");

    const auto last_valid_pos = static_cast<std::uint32_t>(
        std::numeric_limits<std::int32_t>::max() - 1);
    check(build_prefill_chunk_schedule(last_valid_pos, 1u, 1u, schedule,
                                       error) &&
              schedule.size() == 1u &&
              schedule[0].pos0 == last_valid_pos &&
              schedule[0].rows == 1u,
          "accepts the final position whose n_kv fits signed StepRow");

    check(!build_prefill_chunk_schedule(last_valid_pos, 2u, 1u, schedule,
                                        error) &&
              error.find("signed StepRow") != std::string::npos,
          "rejects a prompt whose final chunk would overflow signed StepRow");
    check(schedule.size() == 1u && schedule[0].pos0 == last_valid_pos &&
              schedule[0].rows == 1u,
          "position-overflow failure preserves the previous schedule");

    check(!build_prefill_chunk_schedule(
              1u, std::numeric_limits<std::size_t>::max(), 1u, schedule,
              error) &&
              error.find("signed StepRow") != std::string::npos,
          "rejects an unrepresentable prompt count before chunk arithmetic");
    check(schedule.size() == 1u && schedule[0].pos0 == last_valid_pos &&
              schedule[0].rows == 1u,
          "unrepresentable-count failure preserves the previous schedule");
}

void rejects_invalid_chunks_transactionally() {
    using namespace strata::metal;
    std::string error;
    PrefillChunkMetadata metadata;
    metadata.params = {7u, 91u, 92u, 93u};
    metadata.batch_rows = {1u, 2u, 3u, 4u, 5u, 6u};
    metadata.tokens = {77u, 88u};
    metadata.steps = {{11, 12, 13, 14}};
    metadata.has_image_rows = true;

    const auto remains_unchanged = [&] {
        return metadata.params.T == 7u && metadata.params.p0 == 91u &&
               metadata.params.idx_f == 92u && metadata.params.idx_e == 93u &&
               metadata.batch_rows ==
                   std::array<std::uint32_t, 6>{1u, 2u, 3u, 4u, 5u, 6u} &&
               metadata.tokens == std::vector<std::uint32_t>{77u, 88u} &&
               metadata.steps.size() == 1u &&
               metadata.steps[0].pos == 11 &&
               metadata.has_image_rows;
    };

    check(!build(0u, {}, metadata, error) &&
              error.find("at least one") != std::string::npos,
          "rejects an empty chunk");
    check(remains_unchanged(),
          "empty-chunk failure leaves the output object untouched");

    const std::vector<std::uint32_t> one{1u};
    check(!build(std::numeric_limits<std::int32_t>::max(), one, metadata,
                 error) &&
              error.find("signed StepRow") != std::string::npos,
          "rejects positions whose n_kv would overflow StepRow");
    check(remains_unchanged(),
          "position-overflow failure leaves the output object untouched");

    const std::vector<std::uint32_t> max_rows(kPrefillMaxShaderRows, 0u);
    PrefillChunkMetadata max_metadata;
    check(build(0u, max_rows, max_metadata, error),
          "accepts the exact maximum row count supported by the shader index");
    check(max_metadata.params.T == kPrefillMaxShaderRows &&
              max_metadata.steps.size() == kPrefillMaxShaderRows &&
              (!max_metadata.steps.empty() &&
               max_metadata.steps.back().pos ==
                   static_cast<std::int32_t>(kPrefillMaxShaderRows - 1u)),
          "the exact shader-row limit produces the final live StepRow");

    const std::vector<std::uint32_t> too_many(
        static_cast<std::size_t>(kPrefillMaxShaderRows) + 1u, 0u);
    check(!build(0u, too_many, metadata, error) &&
              error.find("shader index") != std::string::npos,
          "rejects rows that overflow the GR broadcast shader index");
    check(remains_unchanged(),
          "shader-index failure leaves the output object untouched");
}

}  // namespace

int main() {
    builds_chunk_metadata();
    computes_position_boundaries();
    computes_batch_remainders();
    builds_chunk_schedule_without_padding();
    rejects_invalid_chunk_schedules_transactionally();
    rejects_invalid_chunks_transactionally();
    if (failures != 0) {
        std::cerr << "prefill_chunk_test: " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "prefill_chunk_test: all checks passed\n";
    return 0;
}
