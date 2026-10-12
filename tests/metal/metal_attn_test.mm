// Standalone Metal twins for sparse QSA attention (docs/mac/04 §5.7).
#include "metallib_sha256.inc"
#include "oracles/attention.hpp"
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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kSmallCapacity = 128u;
constexpr std::uint32_t kSmallStride = 65u;
constexpr std::uint32_t kLargestStride = INDEXER_SELECTION_WIDTH_MAX;
constexpr std::uint32_t kLargestCapacity = 2112u;
constexpr std::uint32_t kPrefillAttentionBatchRows = 32u;
constexpr std::int32_t kUntouchedId = -987654321;
constexpr float kUntouchedScore = -9876.5f;
constexpr std::array<std::uint32_t, VERIFY_MAX_T> kPositions{
    0u, 1u, 62u, 63u, 64u, 65u, 126u, 127u};

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

BufferRef buffer_subview(BufferRef buffer, std::size_t offset,
                         std::size_t length) {
    return buffer.slice(offset, length);
}

std::uint16_t half_bits(float value) {
    const __fp16 rounded = static_cast<__fp16>(value);
    return std::bit_cast<std::uint16_t>(rounded);
}

void write_u16(std::vector<std::uint8_t>& bytes, std::size_t offset,
               std::uint16_t value) {
    bytes[offset] = static_cast<std::uint8_t>(value & 0xffu);
    bytes[offset + 1u] = static_cast<std::uint8_t>(value >> 8u);
}

float sample(twin::InputSet set, std::mt19937_64& rng,
             std::size_t index, float scale) {
    if (set == twin::InputSet::edge) {
        constexpr std::array<float, 16> values{
            0.0f, -0.0f, 0.125f, -0.125f, 0.25f, -0.25f, 0.5f, -0.5f,
            1.0f, -1.0f, 2.0f, -2.0f, 0x1p-12f, -0x1p-12f, 4.0f, -4.0f};
        return values[index % values.size()] * scale;
    }
    if (set == twin::InputSet::fixture) {
        const std::int32_t value =
            static_cast<std::int32_t>((index * 73u + 29u) % 2047u) - 1023;
        return static_cast<float>(value) * (scale / 1024.0f);
    }
    const std::int32_t value =
        static_cast<std::int32_t>(rng() & 0xffffu) - 32768;
    return static_cast<float>(value) * (scale / 32768.0f);
}

const char* mode_name(QsaAttentionMode mode) {
    switch (mode) {
        case QsaAttentionMode::f16: return "f16";
        case QsaAttentionMode::i8: return "i8";
        case QsaAttentionMode::q4: return "q4";
        case QsaAttentionMode::k8v4: return "k8v4";
    }
    return "invalid";
}

struct CacheData {
    std::vector<std::uint8_t> key_data;
    std::vector<std::uint16_t> key_scales;
    std::vector<std::uint8_t> value_data;
    std::vector<std::uint16_t> value_scales;
};

void write_q4_blocks(std::vector<std::uint8_t>& destination,
                     std::uint32_t pool_row, twin::InputSet set,
                     std::mt19937_64& rng, std::size_t seed) {
    constexpr std::size_t kBlockBytes = sizeof(block_q4_0);
    for (std::uint32_t block = 0u; block < 8u; ++block) {
        const std::size_t block_offset =
            (static_cast<std::size_t>(pool_row) * 8u + block) * kBlockBytes;
        const bool negative = ((pool_row + block) & 1u) != 0u;
        const float magnitude =
            0.03125f * static_cast<float>(1u + (pool_row + block) % 5u);
        const float scale = negative ? -magnitude : magnitude;
        write_u16(destination, block_offset, half_bits(scale));
        for (std::uint32_t index = 0u; index < 16u; ++index) {
            std::uint32_t low = 0u;
            std::uint32_t high = 0u;
            if (set == twin::InputSet::edge) {
                low = (index + block) % 16u;
                high = (15u - index + pool_row) % 16u;
            } else if (set == twin::InputSet::fixture) {
                low = (index * 7u + block * 3u + pool_row) % 16u;
                high = (index * 11u + block + pool_row * 5u) % 16u;
            } else {
                low = static_cast<std::uint32_t>(rng() & 0x0fu);
                high = static_cast<std::uint32_t>(rng() & 0x0fu);
            }
            destination[block_offset + 2u + index] =
                static_cast<std::uint8_t>(low | (high << 4u));
        }
    }
    (void)seed;
}

CacheData make_cache(QsaAttentionMode mode, std::uint32_t cell_capacity,
                     twin::InputSet set, std::uint64_t seed) {
    constexpr std::size_t kHeadValues = HEAD_DIM;
    const std::size_t pool_rows =
        static_cast<std::size_t>(cell_capacity) * N_HEAD_KV;
    CacheData cache;
    std::mt19937_64 rng(seed ^ (static_cast<std::uint64_t>(mode) << 48u) ^
                        (static_cast<std::uint64_t>(set) << 40u));

    const bool key_is_i8 = mode == QsaAttentionMode::i8 ||
                           mode == QsaAttentionMode::k8v4;
    const bool value_is_i8 = mode == QsaAttentionMode::i8;
    const bool key_is_q4 = mode == QsaAttentionMode::q4;
    const bool value_is_q4 = mode == QsaAttentionMode::q4 ||
                             mode == QsaAttentionMode::k8v4;
    const bool key_is_f16 = mode == QsaAttentionMode::f16;

    if (key_is_f16) {
        cache.key_data.resize(pool_rows * kHeadValues * sizeof(std::uint16_t));
        cache.value_data.resize(pool_rows * kHeadValues * sizeof(std::uint16_t));
        for (std::size_t i = 0u; i < pool_rows * kHeadValues; ++i) {
            write_u16(cache.key_data, i * 2u,
                      half_bits(sample(set, rng, i + 0x913u, 1.25f)));
            write_u16(cache.value_data, i * 2u,
                      half_bits(sample(set, rng, i + 0x1271u, 0.75f)));
        }
    }

    if (key_is_i8) {
        cache.key_data.resize(pool_rows * kHeadValues);
        cache.key_scales.resize(pool_rows * 4u);
        for (std::size_t row = 0u; row < pool_rows; ++row) {
            for (std::uint32_t group = 0u; group < 4u; ++group) {
                const float scale =
                    0.0078125f *
                    static_cast<float>(1u + (row * 3u + group * 5u) % 11u);
                cache.key_scales[row * 4u + group] = half_bits(scale);
            }
            for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                const std::size_t index = row * kHeadValues + dimension;
                const float source =
                    sample(set, rng, index + 0x2713u, 1.5f);
                const int code = std::clamp(
                    static_cast<int>(std::nearbyint(source * 48.0f)),
                    -127, 127);
                cache.key_data[index] =
                    std::bit_cast<std::uint8_t>(
                        static_cast<std::int8_t>(code));
            }
        }
    }

    if (value_is_i8) {
        cache.value_data.resize(pool_rows * kHeadValues);
        cache.value_scales.resize(pool_rows * 4u);
        for (std::size_t row = 0u; row < pool_rows; ++row) {
            for (std::uint32_t group = 0u; group < 4u; ++group) {
                const float scale =
                    0.01171875f *
                    static_cast<float>(1u + (row * 7u + group * 3u) % 9u);
                cache.value_scales[row * 4u + group] = half_bits(scale);
            }
            for (std::uint32_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                const std::size_t index = row * kHeadValues + dimension;
                const float source =
                    sample(set, rng, index + 0x411du, 1.25f);
                const int code = std::clamp(
                    static_cast<int>(std::nearbyint(source * 40.0f)),
                    -127, 127);
                cache.value_data[index] =
                    std::bit_cast<std::uint8_t>(
                        static_cast<std::int8_t>(code));
            }
        }
    }

    if (key_is_q4) {
        cache.key_data.resize(pool_rows * 8u * sizeof(block_q4_0));
        for (std::uint32_t row = 0u; row < pool_rows; ++row) {
            write_q4_blocks(cache.key_data, row, set, rng, row + 0x617u);
        }
    }
    if (value_is_q4) {
        cache.value_data.resize(pool_rows * 8u * sizeof(block_q4_0));
        for (std::uint32_t row = 0u; row < pool_rows; ++row) {
            write_q4_blocks(cache.value_data, row, set, rng, row + 0x921u);
        }
    }
    return cache;
}

struct AttentionInput {
    std::vector<float> query;
    std::vector<StepRow> steps;
    std::vector<std::int32_t> ids;
};

AttentionInput make_input(std::uint32_t rows, std::uint32_t cell_capacity,
                          std::uint32_t id_row_stride,
                          twin::InputSet set, std::uint64_t seed) {
    AttentionInput input;
    input.query.resize(
        static_cast<std::size_t>(rows) * N_HEAD_Q * HEAD_DIM);
    std::mt19937_64 rng(seed ^ (static_cast<std::uint64_t>(set) << 32u));
    for (std::size_t i = 0u; i < input.query.size(); ++i) {
        input.query[i] = sample(set, rng, i + 0x7103u, 0.625f);
    }

    input.steps.resize(rows);
    input.ids.assign(
        static_cast<std::size_t>(rows) * id_row_stride, kUntouchedId);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const std::uint32_t position =
            rows == VERIFY_MAX_T && cell_capacity == kSmallCapacity &&
                    id_row_stride == kSmallStride
                ? kPositions[row]
                : cell_capacity - 1u;
        const std::int32_t n_kv =
            static_cast<std::int32_t>(position + 1u);
        const std::int32_t width = std::min<std::int32_t>(
            n_kv, static_cast<std::int32_t>(id_row_stride));
        input.steps[row] = StepRow{
            static_cast<std::int32_t>(position),
            n_kv,
            n_kv / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            width,
        };

        const std::uint32_t count = static_cast<std::uint32_t>(width);
        for (std::uint32_t index = 0u; index < count; ++index) {
            const std::uint32_t cell =
                index + 1u == count && count < static_cast<std::uint32_t>(n_kv)
                    ? static_cast<std::uint32_t>(n_kv - 1)
                    : index;
            input.ids[static_cast<std::size_t>(row) * id_row_stride + index] =
                static_cast<std::int32_t>(cell);
        }
    }
    return input;
}

struct AttentionBuffers {
    twin::GuardedBuffer query;
    twin::GuardedBuffer steps;
    twin::GuardedBuffer ids;
    twin::GuardedBuffer key_data;
    twin::GuardedBuffer key_scales;
    twin::GuardedBuffer value_data;
    twin::GuardedBuffer value_scales;
    twin::GuardedBuffer partial_acc;
    twin::GuardedBuffer partial_m;
    twin::GuardedBuffer partial_l;
    twin::GuardedBuffer output;
};

bool allocate_case_buffers(Context& context, std::uint32_t rows,
                           std::uint32_t id_row_stride,
                           const AttentionInput& input,
                           const CacheData& cache,
                           AttentionBuffers& buffers,
                           std::size_t& partial_acc_bytes,
                           std::size_t& partial_stats_bytes,
                           std::string& error,
                           std::uint32_t scratch_rows = 0u) {
    const std::uint32_t allocated_scratch_rows =
        scratch_rows == 0u ? rows : scratch_rows;
    const std::uint32_t chunks =
        (id_row_stride - 1u) / 64u + 1u;
    const std::size_t slots =
        static_cast<std::size_t>(allocated_scratch_rows) *
        N_HEAD_KV * chunks;
    partial_acc_bytes =
        slots * (N_HEAD_Q / N_HEAD_KV) * HEAD_DIM * sizeof(float);
    partial_stats_bytes =
        slots * (N_HEAD_Q / N_HEAD_KV) * sizeof(float);
    const std::size_t output_bytes =
        static_cast<std::size_t>(rows) * N_HEAD_Q * HEAD_DIM * sizeof(float);
    const bool base_allocated =
        buffers.query.allocate(context, "attention query",
                               input.query.size() * sizeof(float), error) &&
        buffers.steps.allocate(context, "attention steps",
                               input.steps.size() * sizeof(StepRow), error) &&
        buffers.ids.allocate(context, "attention ids",
                             input.ids.size() * sizeof(std::int32_t), error) &&
        buffers.key_data.allocate(context, "attention key data",
                                  cache.key_data.size(), error) &&
        buffers.value_data.allocate(context, "attention value data",
                                    cache.value_data.size(), error) &&
        buffers.partial_acc.allocate(context, "attention partial accumulators",
                                      partial_acc_bytes, error) &&
        buffers.partial_m.allocate(context, "attention partial maxima",
                                   partial_stats_bytes, error) &&
        buffers.partial_l.allocate(context, "attention partial denominators",
                                   partial_stats_bytes, error) &&
        buffers.output.allocate(context, "attention output", output_bytes,
                                error);
    if (!base_allocated) return false;
    if ((!cache.key_scales.empty() &&
         !buffers.key_scales.allocate(context, "attention key scales",
                                      cache.key_scales.size() *
                                          sizeof(std::uint16_t),
                                      error)) ||
        (!cache.value_scales.empty() &&
         !buffers.value_scales.allocate(context, "attention value scales",
                                        cache.value_scales.size() *
                                            sizeof(std::uint16_t),
                                        error))) {
        return false;
    }
    return true;
}

struct PrefillSelectionBuffers {
    twin::GuardedBuffer pooled;
    twin::GuardedBuffer dead;
    twin::GuardedBuffer indexer_queries;
    twin::GuardedBuffer scores;
};

bool allocate_prefill_selection_buffers(
    Context& context, std::uint32_t rows, std::uint32_t cell_capacity,
    PrefillSelectionBuffers& buffers, std::string& error) {
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    constexpr std::size_t key_row_bytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t query_row_bytes =
        static_cast<std::size_t>(INDEXER_HEADS) * key_row_bytes;
    return buffers.pooled.allocate(
               context, "Prefill attention selection pooled",
               max_blocks * key_row_bytes, error) &&
           buffers.dead.allocate(
               context, "Prefill attention selection dead", key_row_bytes,
               error) &&
           buffers.indexer_queries.allocate(
               context, "Prefill attention selection indexer queries",
               static_cast<std::size_t>(rows) * query_row_bytes, error) &&
           buffers.scores.allocate(
               context, "Prefill attention selection scratch",
               static_cast<std::size_t>(256u) * max_blocks * sizeof(float),
               error);
}

bool wait_for_command(std::unique_ptr<CommandBuffer>& command,
                      const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " native command buffer is unavailable";
        return false;
    }
    [native waitUntilCompleted];
    if (native.status != MTLCommandBufferStatusCompleted) {
        error = native.error.localizedDescription.UTF8String != nullptr
            ? native.error.localizedDescription.UTF8String
            : "Metal command buffer failed";
        return false;
    }
    error.clear();
    return true;
}

bool run_attention_case(Context& context, QsaAttentionMode mode,
                        std::uint32_t token_rows,
                        std::uint32_t cell_capacity,
                        std::uint32_t id_row_stride,
                        twin::InputSet set, const twin::Options& options,
                        const AttentionInput& input,
                        const CacheData& cache, std::string_view suffix,
                        std::vector<float>* actual_output = nullptr,
                        bool shared_encoder = false) {
    const std::string label =
        "QSA attention " + std::string(mode_name(mode)) +
        " T=" + std::to_string(token_rows) +
        " stride=" + std::to_string(id_row_stride) +
        " set=" + twin::input_set_name(set) + std::string(suffix) +
        (shared_encoder ? " shared Encoder" : "");
    if (input.query.size() <
            static_cast<std::size_t>(token_rows) * N_HEAD_Q * HEAD_DIM ||
        input.steps.size() < token_rows ||
        input.ids.size() <
            static_cast<std::size_t>(token_rows) * id_row_stride) {
        check(false, label + " host input extents");
        return false;
    }

    std::vector<twin::tf> expected;
    oracle::qsa_attention(
        mode, input.query, input.steps, input.ids, cache.key_data,
        cache.key_scales, cache.value_data, cache.value_scales, token_rows,
        cell_capacity, id_row_stride, expected);

    std::string error;
    AttentionBuffers buffers;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    if (!allocate_case_buffers(context, token_rows, id_row_stride, input,
                               cache, buffers, partial_acc_bytes,
                               partial_stats_bytes, error)) {
        check(false, label + " buffer allocations", error);
        return false;
    }
    std::memcpy(buffers.query.payload().data(), input.query.data(),
                input.query.size() * sizeof(float));
    std::memcpy(buffers.steps.payload().data(), input.steps.data(),
                input.steps.size() * sizeof(StepRow));
    std::memcpy(buffers.ids.payload().data(), input.ids.data(),
                input.ids.size() * sizeof(std::int32_t));
    std::memcpy(buffers.key_data.payload().data(), cache.key_data.data(),
                cache.key_data.size());
    std::memcpy(buffers.value_data.payload().data(), cache.value_data.data(),
                cache.value_data.size());
    if (!cache.key_scales.empty()) {
        std::memcpy(buffers.key_scales.payload().data(),
                    cache.key_scales.data(),
                    cache.key_scales.size() * sizeof(std::uint16_t));
    }
    if (!cache.value_scales.empty()) {
        std::memcpy(buffers.value_scales.payload().data(),
                    cache.value_scales.data(),
                    cache.value_scales.size() * sizeof(std::uint16_t));
    }
    std::fill(buffers.partial_acc.payload().begin(),
              buffers.partial_acc.payload().end(), 0xa5u);
    std::fill(buffers.partial_m.payload().begin(),
              buffers.partial_m.payload().end(), 0xa5u);
    std::fill(buffers.partial_l.payload().begin(),
              buffers.partial_l.payload().end(), 0xa5u);
    if (!buffers.output.poison_floats_against(
            expected, twin::Comparison::units, error)) {
        check(false, label + " output poison", error);
        return false;
    }

    buffers.query.capture_before_dispatch();
    buffers.steps.capture_before_dispatch();
    buffers.ids.capture_before_dispatch();
    buffers.key_data.capture_before_dispatch();
    buffers.value_data.capture_before_dispatch();
    if (!cache.key_scales.empty()) buffers.key_scales.capture_before_dispatch();
    if (!cache.value_scales.empty()) {
        buffers.value_scales.capture_before_dispatch();
    }
    buffers.partial_acc.capture_before_dispatch();
    buffers.partial_m.capture_before_dispatch();
    buffers.partial_l.capture_before_dispatch();
    buffers.output.capture_before_dispatch();

    const QsaAttentionScratch scratch{
        buffers.partial_acc.view(), buffers.partial_m.view(),
        buffers.partial_l.view()};
    const QsaAttentionChunkBuffers chunk_buffers{
        buffers.query.view(),
        buffers.steps.view(),
        buffers.ids.view(),
        buffers.key_data.view(),
        buffers.key_scales.view(),
        buffers.value_data.view(),
        buffers.value_scales.view(),
        scratch,
    };
    const QsaAttentionMergeBuffers merge_buffers{
        scratch, buffers.output.view()};
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_qsa_attention_chunk(
                             context, encoder, mode, token_rows,
                             cell_capacity, id_row_stride, chunk_buffers,
                             encode_error) &&
                         encode_qsa_attention_merge(
                             context, encoder, token_rows, id_row_stride,
                             merge_buffers, encode_error);
              },
              error)
        : encode_qsa_attention_chunk(
              context, *command, mode, token_rows, cell_capacity,
              id_row_stride, chunk_buffers, error) &&
              encode_qsa_attention_merge(
                  context, *command, token_rows, id_row_stride, merge_buffers,
                  error);
    if (!encoded) {
        check(false, label + " encoding", error);
        command->discard();
        return false;
    }
    if (!wait_for_command(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(buffers.output.payload().data()),
        expected.size());
    const auto mismatch = twin::compare_floats(
        actual, expected, {twin::Comparison::units, 32.0, 0.0, 0u});
    if (mismatch) {
        twin::report_failure(label.c_str(), options, token_rows, *mismatch);
    }
    check(!mismatch, label + " U(32) against ordered scalar oracle");
    if (actual_output != nullptr) {
        actual_output->assign(actual.begin(), actual.end());
    }

    const std::array<twin::ByteRange, 1> output_written{{
        {0u, expected.size() * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> acc_written{{
        {0u, partial_acc_bytes}}};
    const std::array<twin::ByteRange, 1> stats_written{{
        {0u, partial_stats_bytes}}};
    const auto query_change = buffers.query.first_unwritten_change({});
    const auto steps_change = buffers.steps.first_unwritten_change({});
    const auto ids_change = buffers.ids.first_unwritten_change({});
    const auto key_change = buffers.key_data.first_unwritten_change({});
    const auto value_change = buffers.value_data.first_unwritten_change({});
    const auto key_scale_change = cache.key_scales.empty()
        ? std::optional<twin::Failure>{}
        : buffers.key_scales.first_unwritten_change({});
    const auto value_scale_change = cache.value_scales.empty()
        ? std::optional<twin::Failure>{}
        : buffers.value_scales.first_unwritten_change({});
    const auto acc_guard =
        buffers.partial_acc.first_unwritten_change(acc_written);
    const auto m_guard = buffers.partial_m.first_unwritten_change(stats_written);
    const auto l_guard = buffers.partial_l.first_unwritten_change(stats_written);
    const auto output_guard =
        buffers.output.first_unwritten_change(output_written);
    const bool guards_ok =
        !query_change && !steps_change && !ids_change && !key_change &&
        !value_change && !key_scale_change && !value_scale_change &&
        !acc_guard && !m_guard && !l_guard && !output_guard;
    check(guards_ok, label + " read-only inputs and buffer guards",
          query_change ? query_change->reason
          : steps_change ? steps_change->reason
          : ids_change ? ids_change->reason
          : key_change ? key_change->reason
          : value_change ? value_change->reason
          : key_scale_change ? key_scale_change->reason
          : value_scale_change ? value_scale_change->reason
          : acc_guard ? acc_guard->reason
          : m_guard ? m_guard->reason
          : l_guard ? l_guard->reason
          : output_guard ? output_guard->reason
                         : "");
    return !mismatch && guards_ok;
}

bool test_row_independence(Context& context, QsaAttentionMode mode,
                           const twin::Options& options) {
    constexpr std::uint32_t rows = VERIFY_MAX_T;
    const auto input = make_input(rows, kSmallCapacity, kSmallStride,
                                  twin::InputSet::fixture, options.seed);
    const auto cache = make_cache(mode, kSmallCapacity,
                                  twin::InputSet::fixture, options.seed);
    std::vector<float> batched;
    bool passed = run_attention_case(
        context, mode, rows, kSmallCapacity, kSmallStride,
        twin::InputSet::fixture, options, input, cache,
        " batch-independence batch", &batched);
    constexpr std::size_t kQueryRow =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    for (std::uint32_t row = 0u; row < rows; ++row) {
        AttentionInput single;
        const auto query = std::span<const float>(input.query).subspan(
            static_cast<std::size_t>(row) * kQueryRow, kQueryRow);
        single.query.assign(query.begin(), query.end());
        single.steps.push_back(input.steps[row]);
        const auto ids = std::span<const std::int32_t>(input.ids).subspan(
            static_cast<std::size_t>(row) * kSmallStride, kSmallStride);
        single.ids.assign(ids.begin(), ids.end());
        std::vector<float> isolated;
        passed &= run_attention_case(
            context, mode, 1u, kSmallCapacity, kSmallStride,
            twin::InputSet::fixture, options, single, cache,
            " row-independence isolated", &isolated);

        const auto batched_row = std::span<const float>(batched).subspan(
            static_cast<std::size_t>(row) * kQueryRow, kQueryRow);
        std::vector<twin::tf> expected;
        expected.reserve(batched_row.size());
        for (const float value : batched_row) expected.emplace_back(value);
        const auto mismatch = twin::compare_floats(
            isolated, expected, {twin::Comparison::bits, 0.0, 0.0, 0u});
        check(!mismatch,
              "QSA attention row result is bitwise batch independent " +
                  std::string(mode_name(mode)) + " row=" +
                  std::to_string(row));
        passed &= !mismatch;
    }
    return passed;
}

bool test_invalid_ids_are_masked(Context& context, QsaAttentionMode mode,
                                 const twin::Options& options) {
    auto input = make_input(1u, kSmallCapacity, 64u,
                            twin::InputSet::fixture, options.seed);
    input.steps[0] = StepRow{63, 64, 16, 64};
    input.ids[0] = -1;
    input.ids[1] = static_cast<std::int32_t>(kSmallCapacity);
    input.ids[2] = input.steps[0].n_kv;

    const auto cache = make_cache(mode, kSmallCapacity,
                                  twin::InputSet::fixture, options.seed);
    bool passed = run_attention_case(
        context, mode, 1u, kSmallCapacity, 64u,
        twin::InputSet::fixture, options, input, cache,
        " invalid-negative-capacity-future-ids");

    std::fill(input.ids.begin(), input.ids.end(), -1);
    std::vector<float> output;
    passed &= run_attention_case(
        context, mode, 1u, kSmallCapacity, 64u,
        twin::InputSet::fixture, options, input, cache,
        " invalid-only-chunk", &output);
    const bool all_positive_zero = std::all_of(
        output.begin(), output.end(),
        [](float value) { return std::bit_cast<std::uint32_t>(value) == 0u; });
    check(all_positive_zero,
          "QSA attention invalid-only chunks return exact positive zero " +
              std::string(mode_name(mode)));
    passed &= all_positive_zero;
    return passed;
}

bool run_selection_attention_case(Context& context, QsaAttentionMode mode,
                                  std::uint32_t token_rows,
                                  const twin::Options& options,
                                  bool varied_prefixes = false,
                                  std::uint32_t cell_capacity =
                                      kLargestCapacity) {
    constexpr std::uint32_t id_row_stride = kLargestStride;
    const std::size_t max_blocks =
        cell_capacity / INDEXER_BLOCK_CELLS + 2u;
    const std::size_t score_count =
        static_cast<std::size_t>(token_rows) * max_blocks;
    const std::size_t id_count =
        static_cast<std::size_t>(token_rows) * id_row_stride;
    const std::size_t attention_row =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    const std::size_t attention_count =
        static_cast<std::size_t>(token_rows) * attention_row;
    const std::size_t q_full_row =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM;
    const std::string label =
        "QSA selection-to-attention " + std::string(mode_name(mode)) +
        " T=" + std::to_string(token_rows) + " shared Encoder" +
        (varied_prefixes
             ? (cell_capacity == 8u ? " short-cache" : " varied-prefixes")
             : "");
    const std::uint64_t seed =
        options.seed ^ (static_cast<std::uint64_t>(mode) << 36u) ^
        (static_cast<std::uint64_t>(token_rows) << 24u) ^
        (varied_prefixes ? 0x766172696564ULL : 0x51534173656cULL);
    std::mt19937_64 rng(seed);

    AttentionInput input = make_input(
        token_rows, cell_capacity, id_row_stride, twin::InputSet::fixture,
        seed ^ 0x243f6a8885a308d3ULL);
    input.ids.assign(id_count, kUntouchedId);
    constexpr std::uint32_t kShortContextCapacity = 8u;
    constexpr std::array<std::uint32_t, VERIFY_MAX_T> varied_positions{
        0u, 1u, 3u, 4u, 63u, 64u, 2047u, 2111u};
    constexpr std::array<std::uint32_t, VERIFY_MAX_T> short_context_positions{
        0u, 1u, 2u, 3u, 4u, 5u, 6u, 7u};
    std::vector<std::int32_t> n_kv(token_rows);
    std::vector<std::int32_t> n_bid(token_rows);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::uint32_t position = varied_prefixes
            ? (cell_capacity == kShortContextCapacity
                   ? short_context_positions[row]
                   : varied_positions[row])
            : token_rows == 1u
                ? cell_capacity - 1u
                : 2051u + (row * 60u + 3u) / 7u;
        const std::int32_t count = static_cast<std::int32_t>(position + 1u);
        input.steps[row] = StepRow{
            static_cast<std::int32_t>(position), count,
            count / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS),
            std::min<std::int32_t>(
                count, static_cast<std::int32_t>(id_row_stride)),
        };
        n_kv[row] = count;
        n_bid[row] = input.steps[row].n_bid;
    }

    std::vector<float> pooled(max_blocks * INDEXER_HEAD_DIM);
    std::vector<float> dead(INDEXER_HEAD_DIM);
    std::vector<float> indexer_queries(
        static_cast<std::size_t>(token_rows) * INDEXER_HEADS *
        INDEXER_HEAD_DIM);
    std::vector<float> q_full(
        static_cast<std::size_t>(token_rows) * q_full_row);
    const auto fill_fixture = [&](std::span<float> values,
                                  std::size_t offset, float scale) {
        for (std::size_t i = 0u; i < values.size(); ++i) {
            values[i] = sample(twin::InputSet::fixture, rng,
                               i + offset, scale);
        }
    };
    fill_fixture(pooled, 0x1003u, 0.75f);
    fill_fixture(dead, 0x2003u, 0.75f);
    fill_fixture(indexer_queries, 0x3003u, 0.75f);
    fill_fixture(q_full, 0x4003u, 0.625f);

    const CacheData cache = make_cache(
        mode, cell_capacity, twin::InputSet::fixture, seed ^ 0x9e3779b97f4a7c15ULL);
    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        pooled, dead, indexer_queries, n_kv, n_bid,
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

    std::vector<twin::tf> expected_attention;
    oracle::qsa_attention(
        mode, input.query, input.steps, expected_ids, cache.key_data,
        cache.key_scales, cache.value_data, cache.value_scales,
        token_rows, cell_capacity, id_row_stride, expected_attention);
    if (mode == QsaAttentionMode::q4 || mode == QsaAttentionMode::k8v4) {
        std::vector<float> unrotated;
        unrotated.reserve(expected_attention.size());
        for (const twin::tf value : expected_attention) {
            unrotated.push_back(value.value);
        }
        std::vector<twin::tf> fwht_values;
        oracle::qsa_fwht256(unrotated, token_rows * N_HEAD_Q, fwht_values);
        std::vector<twin::tf> propagated(expected_attention.size());
        for (std::size_t row = 0u; row < token_rows * N_HEAD_Q; ++row) {
            std::array<twin::tf, HEAD_DIM> values{};
            const std::size_t base = row * HEAD_DIM;
            for (std::size_t index = 0u; index < HEAD_DIM; ++index) {
                values[index] = expected_attention[base + index] *
                                twin::tf{0.0625f};
            }
            for (std::size_t span = 1u; span < HEAD_DIM; span <<= 1u) {
                for (std::size_t block = 0u; block < HEAD_DIM;
                     block += 2u * span) {
                    for (std::size_t index = 0u; index < span; ++index) {
                        const twin::tf lower = values[block + index];
                        const twin::tf upper = values[block + index + span];
                        values[block + index] = lower + upper;
                        values[block + index + span] = lower - upper;
                    }
                }
            }
            for (std::size_t index = 0u; index < HEAD_DIM; ++index) {
                propagated[base + index] = twin::tf{
                    fwht_values[base + index].value,
                    values[index].magnitude};
            }
        }
        expected_attention = std::move(propagated);
    }
    std::vector<float> expected_gate_input;
    expected_gate_input.reserve(expected_attention.size());
    for (const twin::tf value : expected_attention) {
        expected_gate_input.push_back(value.value);
    }
    std::vector<twin::tf> expected_gated;
    oracle::qsa_gate(expected_gate_input, q_full, token_rows, expected_gated);
    for (std::size_t row = 0u; row < token_rows; ++row) {
        const std::size_t attn_row = row * attention_row;
        const std::size_t q_row = row * q_full_row;
        for (std::size_t head = 0u; head < N_HEAD_Q; ++head) {
            const std::size_t attn_base = attn_row + head * HEAD_DIM;
            const std::size_t q_base = q_row + head * 2u * HEAD_DIM + HEAD_DIM;
            for (std::size_t dimension = 0u; dimension < HEAD_DIM;
                 ++dimension) {
                const twin::tf gate_value{q_full[q_base + dimension]};
                const twin::tf sigmoid = twin::tf{1.0f} /
                    (twin::tf{1.0f} +
                     twin::accurate_exp(-gate_value));
                const twin::tf propagated =
                    expected_attention[attn_base + dimension] * sigmoid;
                expected_gated[attn_base + dimension] = twin::tf{
                    expected_gated[attn_base + dimension].value,
                    propagated.magnitude};
            }
        }
    }

    std::string error;
    twin::GuardedBuffer pooled_buffer, dead_buffer, query_buffer;
    twin::GuardedBuffer scores_buffer, q_full_buffer, gated_buffer;
    AttentionBuffers attention;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    const bool allocated =
        pooled_buffer.allocate(context, label + " pooled",
                               pooled.size() * sizeof(float), error) &&
        dead_buffer.allocate(context, label + " dead",
                             dead.size() * sizeof(float), error) &&
        query_buffer.allocate(context, label + " indexer queries",
                              indexer_queries.size() * sizeof(float), error) &&
        scores_buffer.allocate(context, label + " scores",
                              score_count * sizeof(float), error) &&
        allocate_case_buffers(context, token_rows, id_row_stride, input,
                              cache, attention, partial_acc_bytes,
                              partial_stats_bytes, error) &&
        q_full_buffer.allocate(context, label + " q_full",
                               q_full.size() * sizeof(float), error) &&
        gated_buffer.allocate(context, label + " gated output",
                              attention_count * sizeof(float), error);
    check(allocated, label + " allocations", error);
    if (!allocated) return false;

    std::memcpy(pooled_buffer.payload().data(), pooled.data(),
                pooled.size() * sizeof(float));
    std::memcpy(dead_buffer.payload().data(), dead.data(),
                dead.size() * sizeof(float));
    std::memcpy(query_buffer.payload().data(), indexer_queries.data(),
                indexer_queries.size() * sizeof(float));
    std::fill(typed<float>(scores_buffer).begin(),
              typed<float>(scores_buffer).end(), kUntouchedScore);
    std::memcpy(attention.query.payload().data(), input.query.data(),
                input.query.size() * sizeof(float));
    std::memcpy(attention.steps.payload().data(), input.steps.data(),
                input.steps.size() * sizeof(StepRow));
    std::fill(typed<std::int32_t>(attention.ids).begin(),
              typed<std::int32_t>(attention.ids).end(), kUntouchedId);
    std::memcpy(attention.key_data.payload().data(), cache.key_data.data(),
                cache.key_data.size());
    std::memcpy(attention.value_data.payload().data(), cache.value_data.data(),
                cache.value_data.size());
    if (!cache.key_scales.empty()) {
        std::memcpy(attention.key_scales.payload().data(),
                    cache.key_scales.data(),
                    cache.key_scales.size() * sizeof(std::uint16_t));
    }
    if (!cache.value_scales.empty()) {
        std::memcpy(attention.value_scales.payload().data(),
                    cache.value_scales.data(),
                    cache.value_scales.size() * sizeof(std::uint16_t));
    }
    std::fill(attention.partial_acc.payload().begin(),
              attention.partial_acc.payload().end(), 0xa5u);
    std::fill(attention.partial_m.payload().begin(),
              attention.partial_m.payload().end(), 0xa5u);
    std::fill(attention.partial_l.payload().begin(),
              attention.partial_l.payload().end(), 0xa5u);
    std::memcpy(q_full_buffer.payload().data(), q_full.data(),
                q_full.size() * sizeof(float));
    const bool poisoned =
        attention.output.poison_floats_against(
            expected_attention, twin::Comparison::units, error) &&
        gated_buffer.poison_floats_against(
            expected_gated, twin::Comparison::units, error);
    check(poisoned, label + " output poison", error);
    if (!poisoned) return false;

    const QsaAttentionScratch scratch{
        attention.partial_acc.view(), attention.partial_m.view(),
        attention.partial_l.view()};
    const QsaSelectionScoresBuffers selection_scores{
        pooled_buffer.view(), dead_buffer.view(), query_buffer.view(),
        attention.steps.view(), scores_buffer.view()};
    const QsaSelectionTopkBuffers selection_topk{
        scores_buffer.view(), attention.steps.view(), attention.ids.view()};
    const QsaAttentionChunkBuffers attention_chunk{
        attention.query.view(), attention.steps.view(), attention.ids.view(),
        attention.key_data.view(), attention.key_scales.view(),
        attention.value_data.view(), attention.value_scales.view(), scratch};
    const QsaAttentionMergeBuffers attention_merge{
        scratch, attention.output.view()};
    const QsaSelectionAttentionBuffers flow{
        selection_scores, selection_topk, attention_chunk, attention_merge};
    const QsaGateBuffers gate{
        attention.output.view(), q_full_buffer.view(), gated_buffer.view()};

    bool passed = true;
    if (token_rows == 1u && mode == QsaAttentionMode::f16) {
        const auto reject_preflight = [&](
            std::string_view case_name,
            const QsaSelectionAttentionBuffers& candidate,
            std::uint32_t capacity, std::uint32_t stride,
            std::string_view message) {
            std::string validation_error;
            auto validation_command = context.make_command_buffer(
                label + " " + std::string(case_name), validation_error);
            if (!validation_command) {
                check(false, label + " " + std::string(case_name) +
                                 " command buffer", validation_error);
                return false;
            }
            const bool encoded = Encoder::encode(
                context, *validation_command,
                label + " " + std::string(case_name),
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_qsa_selection_attention(
                        context, encoder, mode, token_rows, capacity,
                        stride, candidate, encode_error);
                },
                validation_error);
            const bool rejected =
                !encoded && validation_error.find(message) != std::string::npos;
            check(rejected, label + " rejects " + std::string(case_name),
                  validation_error);
            const bool discarded =
                !validation_command->commit(validation_error) &&
                validation_error.find("discarded") != std::string::npos;
            check(discarded, label + " discards " + std::string(case_name),
                  validation_error);
            return rejected && discarded;
        };
        passed &= reject_preflight(
            "non-fixed ID stride", flow, cell_capacity,
            id_row_stride - 1u, "must be 2051");
        passed &= reject_preflight(
            "non-interleaved cell capacity", flow,
            INDEXER_SELECTION_WIDTH_MAX - 1u, id_row_stride,
            "multiple of four");
        auto mismatched_ids = flow;
        mismatched_ids.attention_chunk.ids = attention.ids.view().slice(
            sizeof(std::int32_t),
            attention.ids.payload().size() - sizeof(std::int32_t));
        passed &= reject_preflight(
            "different top-k and attention ID views", mismatched_ids,
            cell_capacity, id_row_stride, "share one ID view");
        auto cross_stage_alias = flow;
        cross_stage_alias.selection_scores.scores = attention.query.view();
        cross_stage_alias.selection_topk.scores = attention.query.view();
        passed &= reject_preflight(
            "score output overlapping attention query",
            cross_stage_alias, cell_capacity, id_row_stride,
            "overlap unrelated stage inputs");

        std::string group_error;
        auto group_command = context.make_command_buffer(
            label + " concurrent group precondition", group_error);
        if (!group_command) {
            check(false, label + " concurrent group command buffer",
                  group_error);
            passed = false;
        } else {
            bool was_concurrent_group = false;
            bool rejected_open_group = false;
            const bool group_body_encoded = Encoder::encode(
                context, *group_command,
                label + " concurrent group precondition",
                [&](Encoder& encoder, std::string& encode_error) {
                    if (!encoder.group_begin(encode_error)) return false;
                    was_concurrent_group = encoder.concurrent_group_open();
                    if (was_concurrent_group) {
                        std::string composer_error;
                        rejected_open_group =
                            !encode_qsa_selection_attention(
                                context, encoder, mode, token_rows,
                                cell_capacity, id_row_stride, flow,
                                composer_error) &&
                            composer_error.find("outside a concurrent") !=
                                std::string::npos;
                    }
                    return encoder.group_end(encode_error);
                },
                group_error);
            const bool group_contract_ok =
                group_body_encoded &&
                (!was_concurrent_group || rejected_open_group);
            check(group_contract_ok,
                  label + " rejects composition inside an open concurrent group",
                  group_error);
            passed &= group_contract_ok;
            group_command->discard();
        }
    }

    attention.query.capture_before_dispatch();
    attention.steps.capture_before_dispatch();
    attention.ids.capture_before_dispatch();
    attention.key_data.capture_before_dispatch();
    attention.value_data.capture_before_dispatch();
    if (!cache.key_scales.empty()) attention.key_scales.capture_before_dispatch();
    if (!cache.value_scales.empty()) {
        attention.value_scales.capture_before_dispatch();
    }
    attention.partial_acc.capture_before_dispatch();
    attention.partial_m.capture_before_dispatch();
    attention.partial_l.capture_before_dispatch();
    attention.output.capture_before_dispatch();
    pooled_buffer.capture_before_dispatch();
    dead_buffer.capture_before_dispatch();
    query_buffer.capture_before_dispatch();
    scores_buffer.capture_before_dispatch();
    q_full_buffer.capture_before_dispatch();
    gated_buffer.capture_before_dispatch();

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, label,
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_selection_attention(
                    context, encoder, mode, token_rows, cell_capacity,
                    id_row_stride, flow, encode_error)) {
                return false;
            }
            if ((mode == QsaAttentionMode::q4 ||
                 mode == QsaAttentionMode::k8v4) &&
                !encode_qsa_fwht256(
                    context, encoder, token_rows * N_HEAD_Q,
                    attention.output.view(), encode_error)) {
                return false;
            }
            return encode_qsa_gate(
                context, encoder, token_rows, gate, encode_error);
        },
        error);
    if (!encoded) {
        check(false, label + " shared Encoder encoding", error);
        command->discard();
        return false;
    }
    if (!wait_for_command(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto score_mismatch = twin::compare_floats(
        typed<float>(scores_buffer), expected_scores,
        {twin::Comparison::bits, 0.0, 0.0, 0u});
    if (score_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows,
                             *score_mismatch);
    }
    check(!score_mismatch, label + " GPU scores match the scalar oracle");
    passed &= !score_mismatch;

    const auto actual_ids = typed<std::int32_t>(attention.ids);
    const bool ids_match =
        actual_ids.size() == expected_ids.size() &&
        std::equal(actual_ids.begin(), actual_ids.end(), expected_ids.begin());
    bool ids_valid = true;
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const std::size_t row_start =
            static_cast<std::size_t>(row) * id_row_stride;
        for (std::uint32_t index = 0u; index <
             static_cast<std::uint32_t>(input.steps[row].width); ++index) {
            const std::int32_t id = actual_ids[row_start + index];
            ids_valid &= id >= 0 && id < input.steps[row].n_kv;
        }
    }
    check(ids_match, label + " GPU top-k IDs match the scalar oracle");
    check(ids_valid, label + " GPU top-k IDs are valid attention inputs");
    passed &= ids_match && ids_valid;

    const double attention_units =
        mode == QsaAttentionMode::q4 || mode == QsaAttentionMode::k8v4
            ? 256.0 : 32.0;
    const auto actual_attention = typed<float>(attention.output)
                                     .first(attention_count);
    const auto attention_mismatch = twin::compare_floats(
        actual_attention, expected_attention,
        {twin::Comparison::units, attention_units, 0.0, 0u});
    if (attention_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows,
                             *attention_mismatch);
    }
    check(!attention_mismatch,
          label + " merged attention (and O-class FWHT) matches oracle");
    passed &= !attention_mismatch;

    const auto actual_gated = typed<float>(gated_buffer).first(attention_count);
    const double gate_units =
        mode == QsaAttentionMode::q4 || mode == QsaAttentionMode::k8v4
            ? 256.0 : 16.0;
    const auto gate_mismatch = twin::compare_floats(
        actual_gated, expected_gated,
        {twin::Comparison::units, gate_units, 0.0, 0u});
    if (gate_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows, *gate_mismatch);
    }
    check(!gate_mismatch,
          label + " post-attention gate matches the scalar oracle");
    passed &= !gate_mismatch;

    std::vector<twin::ByteRange> score_written;
    score_written.reserve(token_rows);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        score_written.push_back(twin::ByteRange{
            static_cast<std::size_t>(row) * max_blocks * sizeof(float),
            static_cast<std::size_t>(n_bid[row] + 1) * sizeof(float)});
    }
    const std::array<twin::ByteRange, 1> all_ids_written{{
        {0u, id_count * sizeof(std::int32_t)}}};
    const std::array<twin::ByteRange, 1> all_attention_written{{
        {0u, attention_count * sizeof(float)}}};
    const std::array<twin::ByteRange, 1> acc_written{{
        {0u, partial_acc_bytes}}};
    const std::array<twin::ByteRange, 1> stats_written{{
        {0u, partial_stats_bytes}}};
    const auto scores_guard = scores_buffer.first_unwritten_change(score_written);
    const auto ids_guard = attention.ids.first_unwritten_change(all_ids_written);
    const auto attention_guard =
        attention.output.first_unwritten_change(all_attention_written);
    const auto gate_guard = gated_buffer.first_unwritten_change(all_attention_written);
    const auto acc_guard = attention.partial_acc.first_unwritten_change(acc_written);
    const auto m_guard = attention.partial_m.first_unwritten_change(stats_written);
    const auto l_guard = attention.partial_l.first_unwritten_change(stats_written);
    const bool read_only_ok =
        !pooled_buffer.first_unwritten_change({}) &&
        !dead_buffer.first_unwritten_change({}) &&
        !query_buffer.first_unwritten_change({}) &&
        !attention.query.first_unwritten_change({}) &&
        !attention.steps.first_unwritten_change({}) &&
        !attention.key_data.first_unwritten_change({}) &&
        !attention.value_data.first_unwritten_change({}) &&
        (cache.key_scales.empty() ||
         !attention.key_scales.first_unwritten_change({})) &&
        (cache.value_scales.empty() ||
         !attention.value_scales.first_unwritten_change({})) &&
        !q_full_buffer.first_unwritten_change({});
    const bool output_guards_ok =
        !scores_guard && !ids_guard && !attention_guard && !gate_guard &&
        !acc_guard && !m_guard && !l_guard;
    check(read_only_ok, label + " read-only inputs remain unchanged");
    check(output_guards_ok, label + " score, ID, scratch and output guards",
          scores_guard ? scores_guard->reason
          : ids_guard ? ids_guard->reason
          : attention_guard ? attention_guard->reason
          : gate_guard ? gate_guard->reason
          : acc_guard ? acc_guard->reason
          : m_guard ? m_guard->reason
          : l_guard ? l_guard->reason : "");
    passed &= read_only_ok && output_guards_ok;
    return passed;
}

bool run_prefill_attention_case(
    Context& context, QsaAttentionMode mode, std::uint32_t token_rows,
    std::uint32_t pos0, std::uint32_t cell_capacity,
    const twin::Options& options, bool shared_encoder) {
    constexpr std::uint32_t id_row_stride = INDEXER_SELECTION_WIDTH_MAX;
    const std::string label =
        "Prefill QSA selection→attention " +
        std::string(mode_name(mode)) + " T=" +
        std::to_string(token_rows) + " p0=" + std::to_string(pos0) +
        " capacity=" + std::to_string(cell_capacity) +
        (shared_encoder ? " shared Encoder" : "");
    std::vector<std::uint32_t> tokens(token_rows, 1u);
    PrefillChunkMetadata metadata;
    std::string error;
    if (!build_prefill_chunk_metadata(pos0, tokens, metadata, error)) {
        check(false, label + " metadata", error);
        return false;
    }

    const std::size_t max_blocks =
        static_cast<std::size_t>(
            cell_capacity / INDEXER_BLOCK_CELLS) + 2u;
    const std::size_t selection_query_row_elements =
        static_cast<std::size_t>(INDEXER_HEADS) * INDEXER_HEAD_DIM;
    std::mt19937_64 rng(
        options.seed ^ (static_cast<std::uint64_t>(mode) << 48u) ^
        (static_cast<std::uint64_t>(token_rows) << 24u) ^ pos0 ^
        (shared_encoder ? 0x534841524544ULL : 0x5354414e44414cULL));
    AttentionInput input = make_input(
        token_rows, cell_capacity, id_row_stride,
        twin::InputSet::fixture, options.seed ^ 0x81c2d3e4u);
    input.ids.assign(
        static_cast<std::size_t>(token_rows) * id_row_stride, kUntouchedId);
    std::vector<std::int32_t> n_kv(token_rows), n_bid(token_rows);
    std::vector<StepRow> steps;
    steps.reserve(token_rows);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        const PrefillStepRow& prefill_step = metadata.steps[row];
        const StepRow step{
            prefill_step.pos, prefill_step.n_kv,
            prefill_step.n_bid, prefill_step.width};
        steps.push_back(step);
        input.steps[row] = step;
        n_kv[row] = step.n_kv;
        n_bid[row] = step.n_bid;
    }

    std::vector<float> pooled(max_blocks * INDEXER_HEAD_DIM);
    std::vector<float> dead(INDEXER_HEAD_DIM);
    std::vector<float> indexer_queries(
        static_cast<std::size_t>(token_rows) *
        selection_query_row_elements);
    for (std::size_t i = 0u; i < pooled.size(); ++i) {
        pooled[i] = sample(twin::InputSet::fixture, rng, i + 0x1021u, 0.75f);
    }
    for (std::size_t i = 0u; i < dead.size(); ++i) {
        dead[i] = sample(twin::InputSet::fixture, rng, i + 0x2043u, 0.75f);
    }
    for (std::size_t i = 0u; i < indexer_queries.size(); ++i) {
        indexer_queries[i] =
            sample(twin::InputSet::fixture, rng, i + 0x3087u, 0.75f);
    }

    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        pooled, dead, indexer_queries, n_kv, n_bid,
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

    const CacheData cache = make_cache(
        mode, cell_capacity, twin::InputSet::fixture,
        options.seed ^ (static_cast<std::uint64_t>(mode) << 32u) ^
            (static_cast<std::uint64_t>(token_rows) << 8u) ^ pos0);
    std::vector<twin::tf> expected_attention;
    oracle::qsa_attention(
        mode, input.query, steps, expected_ids, cache.key_data,
        cache.key_scales, cache.value_data, cache.value_scales,
        token_rows, cell_capacity, id_row_stride, expected_attention);

    std::string allocation_error;
    PrefillSelectionBuffers selection;
    AttentionBuffers attention;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    const bool allocated =
        allocate_prefill_selection_buffers(
            context, token_rows, cell_capacity, selection, allocation_error) &&
        allocate_case_buffers(
            context, token_rows, id_row_stride, input, cache, attention,
            partial_acc_bytes, partial_stats_bytes, allocation_error,
            kPrefillAttentionBatchRows);
    check(allocated, label + " allocations", allocation_error);
    if (!allocated) return false;

    std::memcpy(selection.pooled.payload().data(), pooled.data(),
                pooled.size() * sizeof(float));
    std::memcpy(selection.dead.payload().data(), dead.data(),
                dead.size() * sizeof(float));
    std::memcpy(selection.indexer_queries.payload().data(),
                indexer_queries.data(), indexer_queries.size() * sizeof(float));
    std::fill(typed<float>(selection.scores).begin(),
              typed<float>(selection.scores).end(), kUntouchedScore);
    std::memcpy(attention.query.payload().data(), input.query.data(),
                input.query.size() * sizeof(float));
    std::memcpy(attention.steps.payload().data(), steps.data(),
                steps.size() * sizeof(StepRow));
    std::fill(typed<std::int32_t>(attention.ids).begin(),
              typed<std::int32_t>(attention.ids).end(), kUntouchedId);
    std::memcpy(attention.key_data.payload().data(), cache.key_data.data(),
                cache.key_data.size());
    std::memcpy(attention.value_data.payload().data(),
                cache.value_data.data(), cache.value_data.size());
    if (!cache.key_scales.empty()) {
        std::memcpy(attention.key_scales.payload().data(),
                    cache.key_scales.data(),
                    cache.key_scales.size() * sizeof(std::uint16_t));
    }
    if (!cache.value_scales.empty()) {
        std::memcpy(attention.value_scales.payload().data(),
                    cache.value_scales.data(),
                    cache.value_scales.size() * sizeof(std::uint16_t));
    }
    std::fill(attention.partial_acc.payload().begin(),
              attention.partial_acc.payload().end(), 0xa5u);
    std::fill(attention.partial_m.payload().begin(),
              attention.partial_m.payload().end(), 0xa5u);
    std::fill(attention.partial_l.payload().begin(),
              attention.partial_l.payload().end(), 0xa5u);
    if (!attention.output.poison_floats_against(
            expected_attention,
            twin::Comparison::units, allocation_error)) {
        check(false, label + " output poison", allocation_error);
        return false;
    }

    selection.pooled.capture_before_dispatch();
    selection.dead.capture_before_dispatch();
    selection.indexer_queries.capture_before_dispatch();
    selection.scores.capture_before_dispatch();
    attention.query.capture_before_dispatch();
    attention.steps.capture_before_dispatch();
    attention.ids.capture_before_dispatch();
    attention.key_data.capture_before_dispatch();
    if (!cache.key_scales.empty()) {
        attention.key_scales.capture_before_dispatch();
    }
    attention.value_data.capture_before_dispatch();
    if (!cache.value_scales.empty()) {
        attention.value_scales.capture_before_dispatch();
    }
    attention.partial_acc.capture_before_dispatch();
    attention.partial_m.capture_before_dispatch();
    attention.partial_l.capture_before_dispatch();
    attention.output.capture_before_dispatch();

    const QsaPrefillSelectionBuffers selection_views{
        selection.pooled.view(), selection.dead.view(),
        selection.indexer_queries.view(), attention.steps.view(),
        selection.scores.view(), attention.ids.view()};
    const QsaAttentionScratch scratch{
        attention.partial_acc.view(), attention.partial_m.view(),
        attention.partial_l.view()};
    const QsaPrefillAttentionBuffers attention_views{
        QsaAttentionChunkBuffers{
            attention.query.view(), attention.steps.view(),
            attention.ids.view(), attention.key_data.view(),
            attention.key_scales.view(), attention.value_data.view(),
            attention.value_scales.view(), scratch},
        QsaAttentionMergeBuffers{scratch, attention.output.view()}};

    auto command = context.make_command_buffer(label, allocation_error);
    if (!command) {
        check(false, label + " command buffer", allocation_error);
        return false;
    }
    const bool encoded = shared_encoder
        ? Encoder::encode(
              context, *command, label,
              [&](Encoder& encoder, std::string& encode_error) {
                  return encode_prefill_qsa_selection(
                             context, encoder, metadata.params,
                             cell_capacity, selection_views, encode_error) &&
                         encode_prefill_qsa_attention(
                             context, encoder, metadata.params, mode,
                             cell_capacity, attention_views, encode_error);
              },
              allocation_error)
        : encode_prefill_qsa_selection(
              context, *command, metadata.params, cell_capacity,
              selection_views, allocation_error) &&
              encode_prefill_qsa_attention(
                  context, *command, metadata.params, mode, cell_capacity,
                  attention_views, allocation_error);
    check(encoded, label + " encode selection→attention", allocation_error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!wait_for_command(command, label, allocation_error)) {
        check(false, label + " GPU completion", allocation_error);
        return false;
    }

    bool passed = true;
    const auto actual_scores = typed<float>(selection.scores).first(
        expected_scores.size());
    const auto score_mismatch = twin::compare_floats(
        actual_scores, expected_scores,
        {twin::Comparison::bits, 0.0, 0.0, 0u});
    if (score_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows,
                             *score_mismatch);
    }
    check(!score_mismatch, label + " selection scores match scalar oracle");
    passed &= !score_mismatch;

    const auto actual_ids = typed<std::int32_t>(attention.ids);
    const bool ids_match =
        actual_ids.size() == expected_ids.size() &&
        std::equal(actual_ids.begin(), actual_ids.end(), expected_ids.begin());
    check(ids_match, label + " selected IDs match scalar oracle");
    passed &= ids_match;

    const auto actual_attention =
        typed<float>(attention.output).first(expected_attention.size());
    const auto attention_mismatch = twin::compare_floats(
        actual_attention, expected_attention,
        {twin::Comparison::units, 32.0, 0.0, 0u});
    if (attention_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows,
                             *attention_mismatch);
    }
    check(!attention_mismatch,
          label + " batched chunk→merge matches scalar oracle");
    passed &= !attention_mismatch;

    std::vector<twin::ByteRange> score_written;
    score_written.reserve(token_rows);
    for (std::uint32_t row = 0u; row < token_rows; ++row) {
        score_written.push_back(twin::ByteRange{
            static_cast<std::size_t>(row) * max_blocks * sizeof(float),
            static_cast<std::size_t>(n_bid[row] + 1) * sizeof(float)});
    }
    const std::array<twin::ByteRange, 1> all_ids_written{{
        {0u, expected_ids.size() * sizeof(std::int32_t)}}};
    const std::array<twin::ByteRange, 1> output_written{{
        {0u, expected_attention.size() * sizeof(float)}}};
    const std::size_t acc_bytes_per_row =
        partial_acc_bytes / kPrefillAttentionBatchRows;
    const std::size_t stats_bytes_per_row =
        partial_stats_bytes / kPrefillAttentionBatchRows;
    const std::uint32_t scratch_rows_written =
        std::min(token_rows, kPrefillAttentionBatchRows);
    const std::array<twin::ByteRange, 1> acc_written{{
        {0u, static_cast<std::size_t>(scratch_rows_written) *
                 acc_bytes_per_row}}};
    const std::array<twin::ByteRange, 1> stats_written{{
        {0u, static_cast<std::size_t>(scratch_rows_written) *
                 stats_bytes_per_row}}};
    const bool inputs_unchanged =
        !selection.pooled.first_unwritten_change({}) &&
        !selection.dead.first_unwritten_change({}) &&
        !selection.indexer_queries.first_unwritten_change({}) &&
        !attention.query.first_unwritten_change({}) &&
        !attention.steps.first_unwritten_change({}) &&
        !attention.key_data.first_unwritten_change({}) &&
        !attention.value_data.first_unwritten_change({}) &&
        (cache.key_scales.empty() ||
         !attention.key_scales.first_unwritten_change({})) &&
        (cache.value_scales.empty() ||
         !attention.value_scales.first_unwritten_change({}));
    const auto scores_guard =
        selection.scores.first_unwritten_change(score_written);
    const auto ids_guard = attention.ids.first_unwritten_change(all_ids_written);
    const auto acc_guard =
        attention.partial_acc.first_unwritten_change(acc_written);
    const auto m_guard =
        attention.partial_m.first_unwritten_change(stats_written);
    const auto l_guard =
        attention.partial_l.first_unwritten_change(stats_written);
    const auto output_guard =
        attention.output.first_unwritten_change(output_written);
    const bool guards_ok =
        !scores_guard && !ids_guard && !acc_guard && !m_guard &&
        !l_guard && !output_guard;
    check(inputs_unchanged, label + " read-only inputs are unchanged");
    check(guards_ok, label + " score, ID, scratch and output guards",
          scores_guard ? scores_guard->reason
          : ids_guard ? ids_guard->reason
          : acc_guard ? acc_guard->reason
          : m_guard ? m_guard->reason
          : l_guard ? l_guard->reason
          : output_guard ? output_guard->reason : "");
    passed &= inputs_unchanged && guards_ok;
    return passed;
}

struct ValidationBuffers {
    AttentionBuffers storage;
    QsaAttentionScratch scratch;
    QsaAttentionChunkBuffers chunk;
    QsaAttentionMergeBuffers merge;
};

bool make_validation_buffers(Context& context, ValidationBuffers& buffers,
                             std::string& error) {
    const auto input = make_input(2u, 4u, 2u, twin::InputSet::fixture, 23u);
    const auto cache = make_cache(QsaAttentionMode::f16, 4u,
                                  twin::InputSet::fixture, 41u);
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    if (!allocate_case_buffers(context, 2u, 2u, input, cache,
                               buffers.storage, partial_acc_bytes,
                               partial_stats_bytes, error)) {
        return false;
    }
    auto steps = typed<StepRow>(buffers.storage.steps);
    steps[0] = StepRow{0, 1, 0, 1};
    steps[1] = StepRow{1, 2, 0, 2};
    auto ids = typed<std::int32_t>(buffers.storage.ids);
    ids[0] = 0;
    ids[1] = kUntouchedId;
    ids[2] = 0;
    ids[3] = 1;
    buffers.scratch = {
        buffers.storage.partial_acc.view(),
        buffers.storage.partial_m.view(),
        buffers.storage.partial_l.view(),
    };
    buffers.chunk = {
        buffers.storage.query.view(),
        buffers.storage.steps.view(),
        buffers.storage.ids.view(),
        buffers.storage.key_data.view(),
        {},
        buffers.storage.value_data.view(),
        {},
        buffers.scratch,
    };
    buffers.merge = {buffers.scratch, buffers.storage.output.view()};
    return true;
}

bool test_prefill_attention_encoder_validation(Context& context) {
    constexpr std::uint32_t rows = 33u;
    constexpr std::uint32_t pos0 = 3u;
    constexpr std::uint32_t cell_capacity = 128u;
    constexpr std::uint32_t id_row_stride = INDEXER_SELECTION_WIDTH_MAX;
    std::vector<std::uint32_t> tokens(rows, 1u);
    PrefillChunkMetadata metadata;
    std::string error;
    if (!build_prefill_chunk_metadata(pos0, tokens, metadata, error)) {
        check(false, "Prefill attention validation metadata", error);
        return false;
    }

    AttentionInput input = make_input(
        rows, cell_capacity, id_row_stride, twin::InputSet::fixture, 31u);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        const PrefillStepRow& source = metadata.steps[row];
        input.steps[row] =
            StepRow{source.pos, source.n_kv, source.n_bid, source.width};
    }
    const CacheData cache = make_cache(
        QsaAttentionMode::f16, cell_capacity,
        twin::InputSet::fixture, 47u);
    AttentionBuffers storage;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    if (!allocate_case_buffers(
            context, rows, id_row_stride, input, cache, storage,
            partial_acc_bytes, partial_stats_bytes, error,
            kPrefillAttentionBatchRows)) {
        check(false, "Prefill attention validation allocations", error);
        return false;
    }
    std::memcpy(storage.query.payload().data(), input.query.data(),
                input.query.size() * sizeof(float));
    std::memcpy(storage.steps.payload().data(), input.steps.data(),
                input.steps.size() * sizeof(StepRow));
    std::memcpy(storage.ids.payload().data(), input.ids.data(),
                input.ids.size() * sizeof(std::int32_t));
    std::memcpy(storage.key_data.payload().data(), cache.key_data.data(),
                cache.key_data.size());
    std::memcpy(storage.value_data.payload().data(), cache.value_data.data(),
                cache.value_data.size());
    const QsaAttentionScratch scratch{
        storage.partial_acc.view(), storage.partial_m.view(),
        storage.partial_l.view()};
    const QsaPrefillAttentionBuffers valid{
        QsaAttentionChunkBuffers{
            storage.query.view(), storage.steps.view(), storage.ids.view(),
            storage.key_data.view(), {}, storage.value_data.view(), {},
            scratch},
        QsaAttentionMergeBuffers{scratch, storage.output.view()}};
    bool passed = true;
    const auto reject = [&](const std::string& label,
                            const strata::metal::PrefillParams& params,
                            std::uint32_t capacity,
                            const QsaPrefillAttentionBuffers& candidate,
                            std::string_view expected_message) {
        std::string call_error;
        auto command = context.make_command_buffer(label, call_error);
        if (!command) {
            check(false, label + " command buffer", call_error);
            return false;
        }
        const bool rejected =
            !encode_prefill_qsa_attention(
                context, *command, params, QsaAttentionMode::f16, capacity,
                candidate, call_error) &&
            call_error.find(expected_message) != std::string::npos;
        check(rejected, label, call_error);
        command->discard();
        return rejected;
    };

    strata::metal::PrefillParams invalid_params = metadata.params;
    ++invalid_params.idx_f;
    passed &= reject(
        "Prefill attention rejects mismatched idx_f", invalid_params,
        cell_capacity, valid, "parameters do not match");
    invalid_params = metadata.params;
    invalid_params.T = 0u;
    passed &= reject(
        "Prefill attention rejects zero rows", invalid_params,
        cell_capacity, valid, "row count");
    invalid_params = metadata.params;
    invalid_params.T = kPrefillMaxShaderRows + 1u;
    passed &= reject(
        "Prefill attention rejects rows above shader limit", invalid_params,
        cell_capacity, valid, "row count");
    passed &= reject(
        "Prefill attention rejects unaligned capacity", metadata.params,
        cell_capacity + 2u, valid, "cell capacity is invalid");

    auto short_scratch = valid;
    short_scratch.attention_chunk.scratch.partial_acc.length -= sizeof(float);
    short_scratch.attention_merge.scratch.partial_acc.length -= sizeof(float);
    passed &= reject(
        "Prefill attention rejects short 32-row scratch", metadata.params,
        cell_capacity, short_scratch, "undersized or misaligned");

    auto split_scratch = valid;
    split_scratch.attention_merge.scratch.partial_m =
        storage.partial_l.view();
    passed &= reject(
        "Prefill attention requires shared chunk/merge scratch",
        metadata.params, cell_capacity, split_scratch,
        "must share the same scratch views");

    auto output_alias = valid;
    output_alias.attention_merge.output =
        storage.query.view().slice(
            0u, storage.output.payload().size());
    passed &= reject(
        "Prefill attention rejects output/query alias", metadata.params,
        cell_capacity, output_alias, "must not overlap an input");

    auto mutable_steps = typed<StepRow>(storage.steps);
    mutable_steps[rows - 1u].pos += 1;
    auto late_failure_command =
        context.make_command_buffer("Prefill attention late metadata failure",
                                    error);
    if (!late_failure_command) {
        check(false, "Prefill attention late-failure command buffer", error);
        return false;
    }
    bool late_metadata_rejected = false;
    bool no_partial_dispatches = false;
    const bool late_body = Encoder::encode(
        context, *late_failure_command,
        "Prefill attention late metadata failure",
        [&](Encoder& encoder, std::string& encode_error) {
            late_metadata_rejected =
                !encode_prefill_qsa_attention(
                    context, encoder, metadata.params,
                    QsaAttentionMode::f16, cell_capacity, valid,
                    encode_error) &&
                encode_error.find("contiguous positions") !=
                    std::string::npos;
            no_partial_dispatches = !encoder.has_encoded_dispatches();
            return false;
        },
        error);
    const bool late_transaction_discarded =
        !late_body && late_metadata_rejected && no_partial_dispatches;
    check(late_transaction_discarded,
          "Prefill attention preflights a malformed final batch before "
          "dispatching earlier rows",
          error);
    passed &= late_transaction_discarded;
    const bool late_commit_rejected =
        !late_failure_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_commit_rejected,
          "Prefill attention late validation discards the Encoder segment",
          error);
    passed &= late_commit_rejected;
    mutable_steps[rows - 1u].pos -= 1;

    storage.query.capture_before_dispatch();
    storage.output.capture_before_dispatch();
    auto standalone_rollback_command = context.make_command_buffer(
        "Prefill attention standalone rollback", error);
    if (!standalone_rollback_command) {
        check(false,
              "Prefill attention standalone rollback command buffer", error);
        return false;
    }
    std::string standalone_error;
    const bool earlier_dispatch_encoded = encode_qsa_fwht256(
        context, *standalone_rollback_command, 1u, storage.query.view(),
        standalone_error);
    check(earlier_dispatch_encoded,
          "Prefill attention standalone rollback seeds earlier work",
          standalone_error);
    passed &= earlier_dispatch_encoded;
    mutable_steps[rows - 1u].pos += 1;
    const bool standalone_rejected =
        earlier_dispatch_encoded &&
        !encode_prefill_qsa_attention(
            context, *standalone_rollback_command, metadata.params,
            QsaAttentionMode::f16, cell_capacity, valid, standalone_error) &&
        standalone_error.find("contiguous positions") != std::string::npos;
    const bool standalone_commit_rejected =
        !standalone_rollback_command->commit(standalone_error) &&
        standalone_error.find("discarded") != std::string::npos;
    mutable_steps[rows - 1u].pos -= 1;
    const bool standalone_outputs_untouched =
        !storage.query.first_unwritten_change({}) &&
        !storage.output.first_unwritten_change({});
    check(standalone_rejected && standalone_commit_rejected &&
              standalone_outputs_untouched,
          "Prefill attention standalone failure discards earlier work and "
          "preserves outputs",
          standalone_error);
    passed &= standalone_rejected && standalone_commit_rejected &&
              standalone_outputs_untouched;

    auto group_command = context.make_command_buffer(
        "Prefill attention concurrent-group precondition", error);
    if (!group_command) {
        check(false, "Prefill attention concurrent-group command buffer",
              error);
        return false;
    }
    bool group_was_open = false;
    bool rejected_open_group = false;
    const bool group_body = Encoder::encode(
        context, *group_command,
        "Prefill attention concurrent-group precondition",
        [&](Encoder& encoder, std::string& group_error) {
            if (!encoder.group_begin(group_error)) return false;
            group_was_open = encoder.concurrent_group_open();
            if (group_was_open) {
                std::string call_error;
                rejected_open_group =
                    !encode_prefill_qsa_attention(
                        context, encoder, metadata.params,
                        QsaAttentionMode::f16, cell_capacity, valid,
                        call_error) &&
                    call_error.find("ordered concurrent-group") !=
                        std::string::npos;
            }
            return encoder.group_end(group_error);
        },
        error);
    const bool group_contract_ok =
        group_body && (!group_was_open || rejected_open_group);
    check(group_contract_ok,
          "Prefill attention rejects calls inside an open concurrent group",
          error);
    passed &= group_contract_ok;
    group_command->discard();

    storage.output.capture_before_dispatch();
    auto downstream_failure_command = context.make_command_buffer(
        "Prefill attention downstream failure", error);
    if (!downstream_failure_command) {
        check(false,
              "Prefill attention downstream-failure command buffer", error);
        return false;
    }
    bool attention_dispatches_encoded = false;
    bool downstream_failure_propagated = false;
    const bool downstream_body = Encoder::encode(
        context, *downstream_failure_command,
        "Prefill attention downstream failure",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_prefill_qsa_attention(
                    context, encoder, metadata.params,
                    QsaAttentionMode::f16, cell_capacity, valid,
                    body_error)) {
                return false;
            }
            attention_dispatches_encoded = encoder.has_encoded_dispatches();
            downstream_failure_propagated =
                !encode_qsa_attention_merge(
                    context, encoder, 0u, id_row_stride, valid.attention_merge,
                    body_error) &&
                body_error.find("row count") != std::string::npos;
            return false;
        },
        error);
    const bool downstream_rollback =
        !downstream_body && attention_dispatches_encoded &&
        downstream_failure_propagated;
    check(downstream_rollback,
          "Prefill attention propagates downstream failure after batch "
          "dispatches",
          error);
    passed &= downstream_rollback;
    const bool downstream_commit_rejected =
        !downstream_failure_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    const bool output_untouched =
        !storage.output.first_unwritten_change({});
    check(downstream_commit_rejected && output_untouched,
          "Prefill attention downstream failure discards all encoded batches",
          error);
    passed &= downstream_commit_rejected && output_untouched;
    return passed;
}

bool reject_chunk(Context& context, const std::string& label,
                  QsaAttentionMode mode, std::uint32_t rows,
                  std::uint32_t cell_capacity, std::uint32_t id_row_stride,
                  const QsaAttentionChunkBuffers& candidate,
                  std::string_view message) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected =
        !encode_qsa_attention_chunk(
            context, *command, mode, rows, cell_capacity, id_row_stride,
            candidate, error) &&
        error.find(message) != std::string::npos;
    check(rejected, label, error);
    command->discard();
    return rejected;
}

bool reject_merge(Context& context, const std::string& label,
                  std::uint32_t rows, std::uint32_t id_row_stride,
                  const QsaAttentionMergeBuffers& candidate,
                  std::string_view message) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    error.clear();
    const bool rejected =
        !encode_qsa_attention_merge(context, *command, rows, id_row_stride,
                                    candidate, error) &&
        error.find(message) != std::string::npos;
    check(rejected, label, error);
    command->discard();
    return rejected;
}

bool test_attention_encoder_validation(Context& context,
                                      Context& foreign_context) {
    ValidationBuffers buffers;
    std::string error;
    if (!make_validation_buffers(context, buffers, error)) {
        check(false, "QSA attention validation allocations", error);
        return false;
    }
    bool passed = true;
    passed &= reject_chunk(
        context, "attention rejects zero rows", QsaAttentionMode::f16, 0u,
        4u, 2u, buffers.chunk, "row count");
    passed &= reject_chunk(
        context, "attention rejects too many rows", QsaAttentionMode::f16,
        9u, 4u, 2u, buffers.chunk, "row count");
    passed &= reject_chunk(
        context, "attention rejects invalid mode",
        static_cast<QsaAttentionMode>(99u), 2u, 4u, 2u,
        buffers.chunk, "KV mode");
    passed &= reject_chunk(
        context, "attention rejects zero capacity", QsaAttentionMode::f16,
        2u, 0u, 1u, buffers.chunk, "cell capacity");
    passed &= reject_chunk(
        context, "attention rejects unaligned cell capacity",
        QsaAttentionMode::f16, 2u, 6u, 2u, buffers.chunk,
        "cell capacity");
    passed &= reject_chunk(
        context, "attention rejects oversized cell capacity",
        QsaAttentionMode::f16, 2u, 262148u, 2u, buffers.chunk,
        "cell capacity");
    passed &= reject_chunk(
        context, "attention rejects zero id stride", QsaAttentionMode::f16,
        2u, 4u, 0u, buffers.chunk, "id row stride");
    QsaAttentionChunkBuffers short_view = buffers.chunk;
    short_view.query.length -= sizeof(float);
    passed &= reject_chunk(
        context, "attention rejects short query view",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.steps.length -= sizeof(StepRow);
    passed &= reject_chunk(
        context, "attention rejects short StepRow view",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.ids.length -= sizeof(std::int32_t);
    passed &= reject_chunk(
        context, "attention rejects short ID view",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.key_data.length -= sizeof(std::uint16_t);
    passed &= reject_chunk(
        context, "attention rejects short fp16 key pool",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.value_data.length -= sizeof(std::uint16_t);
    passed &= reject_chunk(
        context, "attention rejects short fp16 value pool",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.scratch.partial_acc.length -= sizeof(float);
    passed &= reject_chunk(
        context, "attention rejects short accumulator scratch",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.scratch.partial_m.length -= sizeof(float);
    passed &= reject_chunk(
        context, "attention rejects short maximum scratch",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");
    short_view = buffers.chunk;
    short_view.scratch.partial_l.length -= sizeof(float);
    passed &= reject_chunk(
        context, "attention rejects short denominator scratch",
        QsaAttentionMode::f16, 2u, 4u, 2u, short_view,
        "undersized or misaligned");

    twin::GuardedBuffer unaligned_storage;
    const std::size_t query_bytes =
        2u * N_HEAD_Q * HEAD_DIM * sizeof(float);
    if (!unaligned_storage.allocate(context, "unaligned attention query",
                                    query_bytes + 4u, error)) {
        check(false, "attention misaligned-query allocation", error);
        return false;
    }
    QsaAttentionChunkBuffers unaligned = buffers.chunk;
    unaligned.query =
        buffer_subview(unaligned_storage.view(), 1u, query_bytes);
    passed &= reject_chunk(
        context, "attention rejects misaligned query",
        QsaAttentionMode::f16, 2u, 4u, 2u, unaligned,
        "undersized or misaligned");

    QsaAttentionChunkBuffers scratch_alias = buffers.chunk;
    scratch_alias.scratch.partial_acc = buffers.chunk.query;
    passed &= reject_chunk(
        context, "attention rejects scratch/input overlap",
        QsaAttentionMode::f16, 2u, 4u, 2u, scratch_alias,
        "disjoint from all inputs");
    scratch_alias = buffers.chunk;
    scratch_alias.scratch.partial_l = buffers.chunk.scratch.partial_m;
    passed &= reject_chunk(
        context, "attention rejects scratch/output overlap",
        QsaAttentionMode::f16, 2u, 4u, 2u, scratch_alias,
        "disjoint from one another");

    QsaAttentionChunkBuffers readonly_alias = buffers.chunk;
    readonly_alias.value_data = buffers.chunk.key_data;
    auto readonly_command = context.make_command_buffer(
        "attention overlapping read-only inputs", error);
    if (!readonly_command) {
        check(false, "attention read-only alias command buffer", error);
        return false;
    }
    error.clear();
    const bool readonly_accepted = encode_qsa_attention_chunk(
        context, *readonly_command, QsaAttentionMode::f16, 2u, 4u, 2u,
        readonly_alias, error);
    check(readonly_accepted,
          "attention permits overlapping read-only cache inputs", error);
    passed &= readonly_accepted;
    readonly_command->discard();

    twin::GuardedBuffer foreign_buffer;
    if (!foreign_buffer.allocate(foreign_context, "foreign attention query",
                                 query_bytes, error)) {
        check(false, "attention foreign-buffer allocation", error);
        return false;
    }
    QsaAttentionChunkBuffers foreign_view = buffers.chunk;
    foreign_view.query = foreign_buffer.view();
    passed &= reject_chunk(
        context, "attention rejects a foreign Context buffer",
        QsaAttentionMode::f16, 2u, 4u, 2u, foreign_view, "not owned");

    auto foreign_command =
        foreign_context.make_command_buffer("foreign attention command",
                                            error);
    if (!foreign_command) {
        check(false, "attention foreign command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_qsa_attention_chunk(
            context, *foreign_command, QsaAttentionMode::f16, 2u, 4u, 2u,
            buffers.chunk, error) &&
        error.find("different Context") != std::string::npos;
    check(foreign_command_rejected,
          "attention rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted = context.make_command_buffer(
        "submitted attention command", error);
    if (!submitted || !submitted->commit(error)) {
        check(false, "attention submitted command-buffer setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_qsa_attention_chunk(
            context, *submitted, QsaAttentionMode::f16, 2u, 4u, 2u,
            buffers.chunk, error) &&
        error.find("before command-buffer submission") != std::string::npos;
    check(submitted_rejected,
          "attention rejects a submitted command buffer", error);
    passed &= submitted_rejected;
    id<MTLCommandBuffer> submitted_native =
        (__bridge id<MTLCommandBuffer>)submitted->unsafe_native_handle();
    if (submitted_native == nil) {
        check(false, "attention submitted command native handle");
        passed = false;
    } else {
        [submitted_native waitUntilCompleted];
        if (submitted_native.status != MTLCommandBufferStatusCompleted) {
            check(false, "attention submitted command completion");
            passed = false;
        }
    }

    QsaAttentionMergeBuffers merge_short = buffers.merge;
    merge_short.output.length -= sizeof(float);
    passed &= reject_merge(
        context, "attention merge rejects short output", 2u, 2u,
        merge_short, "undersized or misaligned");
    merge_short = buffers.merge;
    merge_short.scratch.partial_acc = buffers.merge.output;
    passed &= reject_merge(
        context, "attention merge rejects output/scratch overlap",
        2u, 2u, merge_short, "disjoint from scratch");
    merge_short = buffers.merge;
    merge_short.output.length -= sizeof(float);

    auto shared_command = context.make_command_buffer(
        "shared attention validation and late failure", error);
    if (!shared_command) {
        check(false, "shared attention command-buffer setup", error);
        return false;
    }
    const auto reject_shared_chunk =
        [&](Encoder& encoder, std::string& encode_error,
            const std::string& label, QsaAttentionMode mode,
            std::uint32_t rows, std::uint32_t cell_capacity,
            std::uint32_t id_row_stride,
            const QsaAttentionChunkBuffers& candidate,
            std::string_view message) {
            const bool rejected =
                !encode_qsa_attention_chunk(
                    context, encoder, mode, rows, cell_capacity,
                    id_row_stride, candidate, encode_error) &&
                encode_error.find(message) != std::string::npos;
            check(rejected, label, encode_error);
            encode_error.clear();
            return rejected;
        };
    const auto reject_shared_merge =
        [&](Encoder& encoder, std::string& encode_error,
            const std::string& label, std::uint32_t rows,
            std::uint32_t id_row_stride,
            const QsaAttentionMergeBuffers& candidate,
            std::string_view message) {
            const bool rejected =
                !encode_qsa_attention_merge(
                    context, encoder, rows, id_row_stride, candidate,
                    encode_error) &&
                encode_error.find(message) != std::string::npos;
            check(rejected, label, encode_error);
            encode_error.clear();
            return rejected;
        };
    bool shared_validation_passed = true;
    const bool shared_validation_encoded = Encoder::encode(
        context, *shared_command, "shared attention validation",
        [&](Encoder& encoder, std::string& encode_error) {
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects zero rows", QsaAttentionMode::f16,
                0u, 4u, 2u, buffers.chunk, "row count");
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects invalid mode",
                static_cast<QsaAttentionMode>(99u), 2u, 4u, 2u,
                buffers.chunk, "KV mode");
            QsaAttentionChunkBuffers shared_short = buffers.chunk;
            shared_short.query.length -= sizeof(float);
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects a short query view",
                QsaAttentionMode::f16, 2u, 4u, 2u, shared_short,
                "undersized or misaligned");
            shared_short = buffers.chunk;
            shared_short.steps.length -= sizeof(StepRow);
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects a short StepRow view",
                QsaAttentionMode::f16, 2u, 4u, 2u, shared_short,
                "undersized or misaligned");
            shared_short = buffers.chunk;
            shared_short.ids.length -= sizeof(std::int32_t);
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects a short ID view",
                QsaAttentionMode::f16, 2u, 4u, 2u, shared_short,
                "undersized or misaligned");
            auto shared_steps = typed<StepRow>(buffers.storage.steps);
            const StepRow original_step = shared_steps[0];
            shared_steps[0].width = original_step.width + 1;
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects an inconsistent StepRow",
                QsaAttentionMode::f16, 2u, 4u, 2u, buffers.chunk,
                "StepRow is inconsistent");
            shared_steps[0] = original_step;
            QsaAttentionChunkBuffers shared_alias = buffers.chunk;
            shared_alias.scratch.partial_acc = buffers.chunk.query;
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects scratch/input overlap",
                QsaAttentionMode::f16, 2u, 4u, 2u, shared_alias,
                "disjoint from all inputs");
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects foreign Context buffer",
                QsaAttentionMode::f16, 2u, 4u, 2u, foreign_view,
                "not owned");
            QsaAttentionChunkBuffers forged_cpu = buffers.chunk;
            forged_cpu.query.contents =
                static_cast<char*>(forged_cpu.query.contents) +
                sizeof(float);
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects forged CPU metadata",
                QsaAttentionMode::f16, 2u, 4u, 2u, forged_cpu,
                "not owned");
            QsaAttentionChunkBuffers forged_gpu = buffers.chunk;
            forged_gpu.ids.gpu_address ^= 0x10u;
            shared_validation_passed &= reject_shared_chunk(
                encoder, encode_error,
                "shared attention rejects forged GPU metadata",
                QsaAttentionMode::f16, 2u, 4u, 2u, forged_gpu,
                "not owned");
            shared_validation_passed &= reject_shared_merge(
                encoder, encode_error,
                "shared attention merge rejects zero rows", 0u, 2u,
                buffers.merge, "row count");
            shared_validation_passed &= reject_shared_merge(
                encoder, encode_error,
                "shared attention merge rejects a short output", 2u, 2u,
                merge_short, "undersized or misaligned");
            QsaAttentionMergeBuffers shared_merge_alias = buffers.merge;
            shared_merge_alias.scratch.partial_acc =
                buffers.merge.output;
            shared_validation_passed &= reject_shared_merge(
                encoder, encode_error,
                "shared attention merge rejects output/scratch overlap",
                2u, 2u, shared_merge_alias, "disjoint from scratch");
            return true;
        },
        error);
    check(shared_validation_encoded,
          "shared attention validation body completes", error);
    passed &= shared_validation_encoded && shared_validation_passed;

    std::fill(buffers.storage.output.payload().begin(),
              buffers.storage.output.payload().end(), 0xa5u);
    const std::vector<std::uint8_t> output_before(
        buffers.storage.output.payload().begin(),
        buffers.storage.output.payload().end());
    error.clear();
    bool shared_chunk_encoded = false;
    const bool late_failure_propagated = !Encoder::encode(
        context, *shared_command, "shared attention late merge failure",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_qsa_attention_chunk(
                    context, encoder, QsaAttentionMode::f16, 2u, 4u, 2u,
                    buffers.chunk, encode_error)) {
                return false;
            }
            shared_chunk_encoded = true;
            return encode_qsa_attention_merge(
                context, encoder, 2u, 2u, merge_short, encode_error);
        },
        error) &&
        shared_chunk_encoded &&
        error.find("undersized or misaligned") != std::string::npos;
    check(late_failure_propagated,
          "shared attention propagates a late merge failure", error);
    passed &= late_failure_propagated;
    const bool output_untouched = std::equal(
        output_before.begin(), output_before.end(),
        buffers.storage.output.payload().begin());
    check(output_untouched,
          "shared attention late failure leaves output untouched");
    passed &= output_untouched;
    error.clear();
    const bool late_failure_discarded =
        !shared_command->commit(error) &&
        error.find("discarded") != std::string::npos;
    check(late_failure_discarded,
          "shared attention late failure discards chunk work", error);
    passed &= late_failure_discarded;
    return passed;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc < 2) {
        std::fprintf(stderr,
                     "usage: metal_attn_test strata.metallib "
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
                     "metal_attn_test has no model-backed real input set\n");
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
    check(foreign_context != nullptr,
          "open foreign attention validation context", error);
    if (!foreign_context) return 1;

    constexpr std::array modes{
        QsaAttentionMode::f16, QsaAttentionMode::i8,
        QsaAttentionMode::q4, QsaAttentionMode::k8v4};
    constexpr std::array all_sets{
        twin::InputSet::random, twin::InputSet::edge,
        twin::InputSet::fixture};
    const std::span<const twin::InputSet> selected_sets =
        options.input_set_explicit
            ? std::span<const twin::InputSet>(&options.input_set, 1u)
            : std::span<const twin::InputSet>(all_sets);

    bool passed = true;
    passed &= test_attention_encoder_validation(*context, *foreign_context);
    passed &= test_prefill_attention_encoder_validation(*context);
    for (const QsaAttentionMode mode : modes) {
        for (const twin::InputSet set : selected_sets) {
            const auto input = make_input(
                VERIFY_MAX_T, kSmallCapacity, kSmallStride, set,
                options.seed ^ (static_cast<std::uint64_t>(mode) << 36u));
            const auto cache = make_cache(
                mode, kSmallCapacity, set,
                options.seed ^ (static_cast<std::uint64_t>(mode) << 36u));
            if (options.input_set_explicit ||
                set == twin::InputSet::random) {
                for (std::uint32_t rows = 1u; rows <= VERIFY_MAX_T; ++rows) {
                    passed &= run_attention_case(
                        *context, mode, rows, kSmallCapacity, kSmallStride,
                        set, options, input, cache, "");
                }
            } else {
                passed &= run_attention_case(
                    *context, mode, VERIFY_MAX_T, kSmallCapacity,
                    kSmallStride, set, options, input, cache, "");
            }
        }
    }

    for (const QsaAttentionMode mode : modes) {
        const auto run_shared_fixture =
            [&](std::uint32_t rows, std::uint32_t capacity,
                std::uint32_t stride, std::string_view suffix) {
                const std::uint64_t seed =
                    options.seed ^
                    (static_cast<std::uint64_t>(mode) << 36u) ^
                    (static_cast<std::uint64_t>(rows) << 24u) ^ stride;
                const auto input = make_input(
                    rows, capacity, stride, twin::InputSet::fixture, seed);
                const auto cache = make_cache(
                    mode, capacity, twin::InputSet::fixture, seed);
                return run_attention_case(
                    *context, mode, rows, capacity, stride,
                    twin::InputSet::fixture, options, input, cache, suffix,
                    nullptr, true);
            };
        passed &= run_shared_fixture(1u, 4u, 1u,
                                     " shared single-chunk");
        passed &= run_shared_fixture(VERIFY_MAX_T, kSmallCapacity,
                                     kSmallStride, " shared two-chunk T=8");
    }

    for (const QsaAttentionMode mode : modes) {
        passed &= run_selection_attention_case(*context, mode, 1u, options);
        passed &= run_selection_attention_case(
            *context, mode, VERIFY_MAX_T, options);
        passed &= run_selection_attention_case(
            *context, mode, VERIFY_MAX_T, options, true);
        passed &= run_selection_attention_case(
            *context, mode, VERIFY_MAX_T, options, true, 8u);
    }

    for (const QsaAttentionMode mode : modes) {
        passed &= run_prefill_attention_case(
            *context, mode, 33u, 3u, 128u, options, false);
        passed &= run_prefill_attention_case(
            *context, mode, 33u, 3u, 128u, options, true);
    }
    passed &= run_prefill_attention_case(
        *context, QsaAttentionMode::f16, 1u, 3u, 128u, options, false);
    passed &= run_prefill_attention_case(
        *context, QsaAttentionMode::f16, 31u, 3u, 128u, options, true);
    passed &= run_prefill_attention_case(
        *context, QsaAttentionMode::f16, 32u, 3u, 128u, options, false);
    passed &= run_prefill_attention_case(
        *context, QsaAttentionMode::f16, 64u, 3u, 128u, options, true);
    passed &= run_prefill_attention_case(
        *context, QsaAttentionMode::f16, 33u, 2047u, kLargestCapacity,
        options, true);

    if (!options.input_set_explicit ||
        options.input_set == twin::InputSet::fixture) {
        for (const QsaAttentionMode mode : modes) {
            passed &= test_row_independence(*context, mode, options);
        }
    }

    const auto boundaries = std::array{
        std::pair{1u, 4u}, std::pair{63u, 64u},
        std::pair{64u, 64u}, std::pair{kLargestStride, kLargestCapacity}};
    for (const QsaAttentionMode mode : modes) {
        passed &= test_invalid_ids_are_masked(*context, mode, options);
        for (const auto [stride, capacity] : boundaries) {
            const auto input = make_input(
                1u, capacity, stride, twin::InputSet::fixture,
                options.seed ^ (static_cast<std::uint64_t>(mode) << 32u) ^
                    stride);
            const auto cache = make_cache(
                mode, capacity, twin::InputSet::fixture,
                options.seed ^ (static_cast<std::uint64_t>(mode) << 32u) ^
                    stride);
            passed &= run_attention_case(
                *context, mode, 1u, capacity, stride,
                twin::InputSet::fixture, options, input, cache,
                " chunk-boundary");
        }
    }

    check(passed && g_failures == 0,
          "QSA sparse attention chunk/merge encoder and kernel twins");
    std::printf("metal_attn_test: %d failures\n", g_failures);
    return g_failures == 0 ? 0 : 1;
}
