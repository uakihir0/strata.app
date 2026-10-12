// Standalone Metal twins for sparse QSA attention (docs/mac/04 §5.7).
#include "metallib_sha256.inc"
#include "oracles/attention.hpp"
#include "oracles/qsa.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill_gemm_dispatch.hpp"
#include "strata/metal/prefill_native_projection.hpp"
#include "strata/metal/prefill_qsa_gate_dispatch.hpp"
#include "strata/metal/prefill_qsa_layer_dispatch.hpp"
#include "strata/metal/prefill_qsa_tail_dispatch.hpp"
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
#include <cstdlib>
#include <cstring>
#include <optional>
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

std::uint16_t half_order_key(std::uint16_t bits);

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    if (!condition) ++g_failures;
}

class ScopedEnvironmentOverride {
public:
    ScopedEnvironmentOverride(const char* name, const char* value)
        : name_(name) {
        if (const char* const previous = std::getenv(name)) {
            previous_ = previous;
        }
        ready_ = ::setenv(name_.c_str(), value, 1) == 0;
    }

    ~ScopedEnvironmentOverride() {
        if (!ready_) return;
        if (previous_) {
            ::setenv(name_.c_str(), previous_->c_str(), 1);
        } else {
            ::unsetenv(name_.c_str());
        }
    }

    bool ready() const noexcept { return ready_; }

private:
    std::string name_;
    std::optional<std::string> previous_;
    bool ready_ = false;
};

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

bool concurrent_dispatch_mode() {
    const char* const value = std::getenv("STRATA_METAL_DISPATCH");
    return value != nullptr && std::string_view(value) == "concurrent";
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
    const twin::Options& options, bool shared_encoder,
    bool exercise_output_tail = false, bool apply_tail_fwht = true) {
    constexpr std::uint32_t id_row_stride = INDEXER_SELECTION_WIDTH_MAX;
    const std::string label =
        "Prefill QSA selection→attention " +
        std::string(mode_name(mode)) + " T=" +
        std::to_string(token_rows) + " p0=" + std::to_string(pos0) +
        " capacity=" + std::to_string(cell_capacity) +
        (exercise_output_tail
             ? (apply_tail_fwht ? " output tail FWHT"
                                : " output tail no FWHT")
             : "") +
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

    constexpr std::uint32_t projection_n = 2560u;
    constexpr std::uint32_t projection_k = 24u * 256u;
    std::vector<float> q_full_values;
    std::vector<twin::tf> expected_fwht;
    std::vector<float> expected_attention_values;
    std::vector<float> expected_fwht_values;
    std::vector<twin::tf> expected_gate;
    std::vector<std::uint16_t> expected_gate_f16;
    if (exercise_output_tail) {
        q_full_values.resize(
            static_cast<std::size_t>(token_rows) * N_HEAD_Q * 2u * HEAD_DIM);
        for (std::size_t i = 0u; i < q_full_values.size(); ++i) {
            q_full_values[i] =
                sample(twin::InputSet::fixture, rng, i + 0x42d91u, 4.0f);
        }
        expected_attention_values.reserve(expected_attention.size());
        for (const twin::tf value : expected_attention) {
            expected_attention_values.push_back(value.value);
        }
        std::vector<float> gate_attention_values =
            expected_attention_values;
        if (apply_tail_fwht) {
            oracle::qsa_fwht256(
                expected_attention_values,
                token_rows * static_cast<std::uint32_t>(N_HEAD_Q),
                expected_fwht);
            expected_fwht_values.reserve(expected_fwht.size());
            for (const twin::tf value : expected_fwht) {
                expected_fwht_values.push_back(value.value);
            }
            gate_attention_values = expected_fwht_values;
        }
        oracle::qsa_gate(gate_attention_values, q_full_values, token_rows,
                         expected_gate);
        expected_gate_f16.reserve(expected_gate.size());
        for (const twin::tf value : expected_gate) {
            expected_gate_f16.push_back(half_bits(value.value));
        }
    }

    std::string allocation_error;
    PrefillSelectionBuffers selection;
    AttentionBuffers attention;
    twin::GuardedBuffer q_full, gate_f16, native_weights, dequantized_f16,
        projected_y;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    const bool allocated =
        allocate_prefill_selection_buffers(
            context, token_rows, cell_capacity, selection, allocation_error) &&
        allocate_case_buffers(
            context, token_rows, id_row_stride, input, cache, attention,
            partial_acc_bytes, partial_stats_bytes, allocation_error,
            kPrefillAttentionBatchRows) &&
        (!exercise_output_tail ||
         (q_full.allocate(
              context, label + " output Qf",
              q_full_values.size() * sizeof(float), allocation_error) &&
          gate_f16.allocate(
              context, label + " gated F16",
              static_cast<std::size_t>(token_rows) * N_HEAD_Q * HEAD_DIM *
                  sizeof(std::uint16_t),
              allocation_error) &&
          native_weights.allocate(
              context, label + " Q8_0 output weights",
              static_cast<std::size_t>(projection_n) *
                  (projection_k / 32u) * sizeof(block_q8_0),
              allocation_error) &&
          dequantized_f16.allocate(
              context, label + " dequantized output weights",
              static_cast<std::size_t>(projection_n) * projection_k *
                  sizeof(std::uint16_t),
              allocation_error) &&
          projected_y.allocate(
              context, label + " projected Y",
              static_cast<std::size_t>(token_rows) * projection_n *
                  sizeof(float),
              allocation_error)));
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
    if (exercise_output_tail) {
        std::memcpy(q_full.payload().data(), q_full_values.data(),
                    q_full_values.size() * sizeof(float));
        auto weights = typed<block_q8_0>(native_weights);
        constexpr std::array<std::uint16_t, 4u> q8_scale_bits{
            0x2c00u, 0x3000u, 0x3200u, 0x3400u};
        for (std::uint32_t column = 0u; column < projection_n; ++column) {
            for (std::uint32_t block_index = 0u;
                 block_index < projection_k / 32u; ++block_index) {
                block_q8_0& block =
                    weights[static_cast<std::size_t>(column) *
                                (projection_k / 32u) +
                            block_index];
                block = {};
                const std::size_t scale_index =
                    (static_cast<std::size_t>(column) +
                     static_cast<std::size_t>(block_index) * 3u) %
                    q8_scale_bits.size();
                block.d =
                    static_cast<strata_half>(q8_scale_bits[scale_index]);
                for (std::uint32_t value = 0u; value < 32u; ++value) {
                    std::uint32_t code_hash =
                        column * 0x9e3779b9u ^
                        block_index * 0x85ebca6bu ^
                        value * 0xc2b2ae35u;
                    code_hash ^= code_hash >> 16u;
                    code_hash *= 0x7feb352du;
                    code_hash ^= code_hash >> 15u;
                    code_hash *= 0x846ca68bu;
                    code_hash ^= code_hash >> 16u;
                    const int code =
                        static_cast<int>(code_hash % 11u) - 5;
                    block.qs[value] = static_cast<strata_i8>(code);
                }
            }
        }
        std::fill(gate_f16.payload().begin(), gate_f16.payload().end(),
                  0xa5u);
        std::fill(dequantized_f16.payload().begin(),
                  dequantized_f16.payload().end(), 0xa5u);
        std::fill(projected_y.payload().begin(), projected_y.payload().end(),
                  0xa5u);
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

    PrefillQsaTailBuffers tail_views{};
    PrefillGemmBackend tail_backend = PrefillGemmBackend::SIMD;
    if (exercise_output_tail) {
        const PrefillQsaOutputBuffers output_views{
            {context.buffers().read_only(attention.output.view()),
             context.buffers().read_only(q_full.view()), gate_f16.view()},
            context.buffers().read_only(native_weights.view()),
            dequantized_f16.view(), projected_y.view(),
            apply_tail_fwht ? attention.output.view() : BufferRef{}};
        tail_views = PrefillQsaTailBuffers{
            selection_views, attention_views, output_views};
        q_full.capture_before_dispatch();
        gate_f16.capture_before_dispatch();
        native_weights.capture_before_dispatch();
        dequantized_f16.capture_before_dispatch();
        projected_y.capture_before_dispatch();
    }

    const auto encode_tail = [&](CommandBuffer& target,
                                 const PrefillQsaTailBuffers& views,
                                 PrefillGemmBackend* backend,
                                 std::string& error) {
        if (shared_encoder) {
            return Encoder::encode(
                context, target, label,
                [&](Encoder& encoder, std::string& encode_error) {
                    return encode_prefill_qsa_tail(
                        context, encoder, metadata.params, mode,
                        cell_capacity, PrefillNativeWeightFormat::Q8_0,
                        views, encode_error, backend,
                        apply_tail_fwht
                            ? PrefillQsaAttentionTransform::fwht256
                            : PrefillQsaAttentionTransform::none);
                },
                error);
        }
        return encode_prefill_qsa_tail(
            context, target, metadata.params, mode, cell_capacity,
            PrefillNativeWeightFormat::Q8_0, views, error, backend,
            apply_tail_fwht ? PrefillQsaAttentionTransform::fwht256
                            : PrefillQsaAttentionTransform::none);
    };

    if (exercise_output_tail) {
        const auto expect_tail_preflight_rejection =
            [&](const PrefillQsaTailBuffers& invalid,
                std::string_view expected_error,
                std::string_view scenario) {
                const std::string case_label =
                    label + " " + std::string(scenario);
                auto rejected_command =
                    context.make_command_buffer(case_label, allocation_error);
                if (!rejected_command) {
                    check(false, case_label + " command buffer",
                          allocation_error);
                    return false;
                }
                PrefillGemmBackend rejected_backend =
                    PrefillGemmBackend::SIMD;
                allocation_error.clear();
                const bool encoded = encode_tail(
                    *rejected_command, invalid, &rejected_backend,
                    allocation_error);
                const bool rejected =
                    !encoded &&
                    rejected_command->unsafe_native_handle() == nullptr &&
                    allocation_error.find(expected_error) !=
                        std::string::npos &&
                    rejected_backend == PrefillGemmBackend::SIMD &&
                    std::equal(input.query.begin(), input.query.end(),
                               typed<float>(attention.query).begin());
                check(rejected, case_label, allocation_error);
                return rejected;
            };

        PrefillQsaTailBuffers invalid_handoff = tail_views;
        invalid_handoff.selection.steps = selection.pooled.view();
        if (!expect_tail_preflight_rejection(
                invalid_handoff, "selection and attention must share StepRows",
                "rejects mismatched StepRows")) {
            return false;
        }
        invalid_handoff = tail_views;
        invalid_handoff.selection.ids = selection.scores.view();
        if (!expect_tail_preflight_rejection(
                invalid_handoff, "selection IDs must feed attention",
                "rejects mismatched selection IDs")) {
            return false;
        }
        invalid_handoff = tail_views;
        invalid_handoff.output.gate.attn =
            context.buffers().read_only(attention.query.view());
        if (!expect_tail_preflight_rejection(
                invalid_handoff,
                "attention output must feed the output gate",
                "rejects mismatched attention output")) {
            return false;
        }
        if (apply_tail_fwht) {
            invalid_handoff = tail_views;
            invalid_handoff.output.attention_transform =
                q_full.view().slice(0u, attention.output.view().length);
            if (!expect_tail_preflight_rejection(
                    invalid_handoff,
                    "FWHT view must match the attention output",
                    "rejects mismatched FWHT handoff")) {
                return false;
            }
        }
        invalid_handoff = tail_views;
        invalid_handoff.selection.scores = attention.query.view().slice(
            sizeof(float), selection.scores.view().length);
        if (!expect_tail_preflight_rejection(
                invalid_handoff,
                "selection scores overlaps active attention queries",
                "rejects a partial score/query overlap")) {
            return false;
        }
    }

    auto command = context.make_command_buffer(label, allocation_error);
    if (!command) {
        check(false, label + " command buffer", allocation_error);
        return false;
    }
    const bool encoded = exercise_output_tail
        ? encode_tail(*command, tail_views, &tail_backend, allocation_error)
        : shared_encoder
              ? Encoder::encode(
                    context, *command, label,
                    [&](Encoder& encoder, std::string& encode_error) {
                        return encode_prefill_qsa_selection(
                                   context, encoder, metadata.params,
                                   cell_capacity, selection_views,
                                   encode_error) &&
                               encode_prefill_qsa_attention(
                                   context, encoder, metadata.params, mode,
                                   cell_capacity, attention_views,
                                   encode_error);
                    },
                    allocation_error)
              : encode_prefill_qsa_selection(
                    context, *command, metadata.params, cell_capacity,
                    selection_views, allocation_error) &&
                    encode_prefill_qsa_attention(
                        context, *command, metadata.params, mode,
                        cell_capacity, attention_views, allocation_error);
    check(encoded, label + " encode selection→attention→output", allocation_error);
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

    const std::vector<twin::tf>& expected_composed_attention =
        exercise_output_tail && apply_tail_fwht
            ? expected_fwht
            : expected_attention;
    const auto actual_attention =
        typed<float>(attention.output).first(expected_composed_attention.size());
    const auto attention_mismatch = twin::compare_floats(
        actual_attention, expected_composed_attention,
        {twin::Comparison::units, 32.0, 0.0, 0u});
    if (attention_mismatch) {
        twin::report_failure(label.c_str(), options, token_rows,
                             *attention_mismatch);
    }
    check(!attention_mismatch,
          label + (exercise_output_tail && apply_tail_fwht
                       ? " attention FWHT matches scalar oracle"
                       : exercise_output_tail
                             ? " attention output matches scalar oracle"
                             : " batched chunk→merge matches scalar oracle"));
    passed &= !attention_mismatch;

    if (exercise_output_tail) {
        const auto actual_gate = typed<std::uint16_t>(gate_f16)
                                     .first(expected_gate_f16.size());
        std::uint16_t max_gate_ulp = 0u;
        bool gate_matches = actual_gate.size() == expected_gate_f16.size();
        for (std::size_t i = 0u; gate_matches && i < actual_gate.size(); ++i) {
            const std::uint16_t actual_order = half_order_key(actual_gate[i]);
            const std::uint16_t expected_order =
                half_order_key(expected_gate_f16[i]);
            const std::uint16_t distance =
                actual_order >= expected_order
                    ? static_cast<std::uint16_t>(actual_order - expected_order)
                    : static_cast<std::uint16_t>(expected_order - actual_order);
            max_gate_ulp = std::max(max_gate_ulp, distance);
            gate_matches &= distance <= 2u;
        }
        check(gate_matches,
              label + " F16 gate matches scalar oracle within 2 ULP",
              "maximum distance=" + std::to_string(max_gate_ulp));
        passed &= gate_matches;

        const auto projected_values = typed<float>(projected_y).first(
            static_cast<std::size_t>(token_rows) * projection_n);
        bool projected_finite = true;
        bool projected_nonzero = false;
        for (const float value : projected_values) {
            projected_finite &= std::isfinite(value);
            projected_nonzero |= value != 0.0f;
        }
        check(projected_finite, label + " native projection is finite");
        check(projected_nonzero, label + " native projection is nonzero");
        passed &= projected_finite && projected_nonzero;

        std::vector<twin::tf> expected_projected;
        std::vector<double> expected_projected_double;
        expected_projected.reserve(projected_values.size());
        expected_projected_double.reserve(projected_values.size());
        const auto q8_weights = typed<block_q8_0>(native_weights);
        constexpr std::uint32_t projection_blocks_per_column =
            projection_k / 32u;
        for (std::uint32_t row = 0u; row < token_rows; ++row) {
            for (std::uint32_t column = 0u; column < projection_n; ++column) {
                double sum = 0.0;
                double magnitude = 0.0;
                for (std::uint32_t block_index = 0u;
                     block_index < projection_blocks_per_column;
                     ++block_index) {
                    const block_q8_0& block =
                        q8_weights[static_cast<std::size_t>(column) *
                                       projection_blocks_per_column +
                                   block_index];
                    const float scale = strata::kernels::f32_from_f16(
                        static_cast<std::uint16_t>(block.d));
                    for (std::uint32_t value = 0u; value < 32u; ++value) {
                        const std::uint32_t k = block_index * 32u + value;
                        const std::uint16_t weight_f16 =
                            strata::kernels::f16_from_f32(
                                scale * static_cast<float>(block.qs[value]));
                        const double product =
                            static_cast<double>(strata::kernels::f32_from_f16(
                                expected_gate_f16[
                                    static_cast<std::size_t>(row) *
                                        projection_k +
                                    k])) *
                            static_cast<double>(
                                strata::kernels::f32_from_f16(weight_f16));
                        sum += product;
                        magnitude += std::fabs(product);
                    }
                }
                expected_projected.emplace_back(
                    static_cast<float>(sum), magnitude);
                expected_projected_double.push_back(sum);
            }
        }
        const auto projection_units_mismatch = twin::compare_floats(
            projected_values, expected_projected,
            {twin::Comparison::units, 2.0 * projection_k, 0.0, 0u});
        if (projection_units_mismatch) {
            twin::report_failure(label.c_str(), options, token_rows,
                                 *projection_units_mismatch);
        }
        check(!projection_units_mismatch,
              label + " native projection matches independent scalar oracle under U(2K)");
        const auto projection_relative_mismatch = twin::compare_relative_l2(
            projected_values, expected_projected_double, 0x1p-16,
            projection_n);
        if (projection_relative_mismatch) {
            twin::report_failure(label.c_str(), options, token_rows,
                                 *projection_relative_mismatch);
        }
        check(!projection_relative_mismatch,
              label + " native projection matches FP64 oracle under R(2^-16)");
        passed &= !projection_units_mismatch &&
                  !projection_relative_mismatch;

        const std::vector<std::uint8_t> tail_gate(
            gate_f16.payload().begin(), gate_f16.payload().end());
        const std::vector<std::uint8_t> tail_y(
            projected_y.payload().begin(), projected_y.payload().end());
        std::fill(gate_f16.payload().begin(), gate_f16.payload().end(), 0xa5u);
        std::fill(dequantized_f16.payload().begin(),
                  dequantized_f16.payload().end(), 0xa5u);
        std::fill(projected_y.payload().begin(), projected_y.payload().end(),
                  0xa5u);
        auto components = context.make_command_buffer(
            label + " explicit output components", allocation_error);
        if (!components) {
            check(false, label + " component command buffer", allocation_error);
            return false;
        }
        PrefillGemmBackend component_backend = PrefillGemmBackend::SIMD;
        const PrefillGemmMPPParams projection_params{
            token_rows, projection_n, projection_k, projection_n, 0u, false};
        const PrefillNativeProjectionBuffers projection_buffers{
            context.buffers().read_only(gate_f16.view()),
            tail_views.output.native_weights, dequantized_f16.view(),
            projected_y.view()};
        const bool components_encoded = Encoder::encode(
            context, *components, label + " explicit output components",
            [&](Encoder& encoder, std::string& encode_error) {
                if (!encode_prefill_qsa_gate_attn(
                        context, encoder, token_rows,
                        tail_views.output.gate, encode_error)) {
                    return false;
                }
                return encode_prefill_native_projection(
                    context, encoder, projection_params,
                    PrefillNativeWeightFormat::Q8_0, projection_buffers,
                    encode_error, &component_backend);
            },
            allocation_error);
        check(components_encoded,
              label + " explicit gate and projection encode",
              allocation_error);
        if (!components_encoded) return false;
        const bool components_completed =
            wait_for_command(components, label + " explicit output components",
                             allocation_error);
        check(components_completed,
              label + " explicit gate and projection complete",
              allocation_error);
        if (!components_completed) return false;
        const bool component_match =
            std::equal(tail_gate.begin(), tail_gate.end(),
                       gate_f16.payload().begin()) &&
            std::equal(tail_y.begin(), tail_y.end(),
                       projected_y.payload().begin()) &&
            component_backend == tail_backend;
        check(component_match,
              label + " tail output matches explicit gate/projection chain");
        passed &= component_match;

        std::fill(typed<float>(selection.scores).begin(),
                  typed<float>(selection.scores).end(), kUntouchedScore);
        std::fill(typed<std::int32_t>(attention.ids).begin(),
                  typed<std::int32_t>(attention.ids).end(), kUntouchedId);
        std::fill(attention.partial_acc.payload().begin(),
                  attention.partial_acc.payload().end(), 0xa5u);
        std::fill(attention.partial_m.payload().begin(),
                  attention.partial_m.payload().end(), 0xa5u);
        std::fill(attention.partial_l.payload().begin(),
                  attention.partial_l.payload().end(), 0xa5u);
        std::fill(attention.output.payload().begin(),
                  attention.output.payload().end(), 0xa5u);
        std::fill(gate_f16.payload().begin(), gate_f16.payload().end(), 0xa5u);
        std::fill(dequantized_f16.payload().begin(),
                  dequantized_f16.payload().end(), 0xa5u);
        std::fill(projected_y.payload().begin(), projected_y.payload().end(),
                  0xa5u);
        ScopedEnvironmentOverride invalid_selector("STRATA_METAL_GEMM",
                                                    "invalid");
        const bool selector_ready = invalid_selector.ready();
        check(selector_ready, label + " install late invalid selector");
        if (!selector_ready) return false;

        auto rollback_command = context.make_command_buffer(
            label + " late projection rollback", allocation_error);
        if (!rollback_command) {
            check(false, label + " rollback command buffer",
                  allocation_error);
            return false;
        }
        PrefillGemmBackend rollback_backend = PrefillGemmBackend::MPP;
        allocation_error.clear();
        const bool rollback_encoded = encode_tail(
            *rollback_command, tail_views, &rollback_backend,
            allocation_error);
        const bool rollback_ok =
            !rollback_encoded &&
            rollback_command->unsafe_native_handle() == nullptr &&
            allocation_error.find("STRATA_METAL_GEMM") != std::string::npos &&
            rollback_backend == PrefillGemmBackend::MPP &&
            std::all_of(
                typed<float>(selection.scores).begin(),
                typed<float>(selection.scores).end(),
                [](float value) { return value == kUntouchedScore; }) &&
            std::all_of(
                typed<std::int32_t>(attention.ids).begin(),
                typed<std::int32_t>(attention.ids).end(),
                [](std::int32_t value) { return value == kUntouchedId; }) &&
            std::all_of(attention.output.payload().begin(),
                        attention.output.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; }) &&
            std::all_of(gate_f16.payload().begin(), gate_f16.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; }) &&
            std::all_of(dequantized_f16.payload().begin(),
                        dequantized_f16.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; }) &&
            std::all_of(projected_y.payload().begin(),
                        projected_y.payload().end(),
                        [](std::uint8_t value) { return value == 0xa5u; });
        check(rollback_ok,
              label + " late projection failure discards the complete tail",
              allocation_error);
        passed &= rollback_ok;
    }

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
         !attention.value_scales.first_unwritten_change({})) &&
        (!exercise_output_tail ||
         (!q_full.first_unwritten_change({}) &&
          !native_weights.first_unwritten_change({})));
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
    const std::array<twin::ByteRange, 1> gate_written{{
        {0u, exercise_output_tail
                 ? expected_gate_f16.size() * sizeof(std::uint16_t)
                 : 0u}}};
    const std::array<twin::ByteRange, 1> dequantized_written{{
        {0u, exercise_output_tail
                 ? static_cast<std::size_t>(projection_n) * projection_k *
                       sizeof(std::uint16_t)
                 : 0u}}};
    const std::array<twin::ByteRange, 1> projected_written{{
        {0u, exercise_output_tail
                 ? static_cast<std::size_t>(token_rows) * projection_n *
                       sizeof(float)
                 : 0u}}};
    const auto gate_guard = gate_f16.first_unwritten_change(gate_written);
    const auto dequantized_guard =
        dequantized_f16.first_unwritten_change(dequantized_written);
    const auto projected_guard =
        projected_y.first_unwritten_change(projected_written);
    const bool guards_ok =
        !scores_guard && !ids_guard && !acc_guard && !m_guard &&
        !l_guard && !output_guard && !gate_guard && !dequantized_guard &&
        !projected_guard;
    check(inputs_unchanged, label + " read-only inputs are unchanged");
    check(guards_ok,
          label + " score, ID, attention/output scratch and output guards",
          scores_guard ? scores_guard->reason
          : ids_guard ? ids_guard->reason
          : acc_guard ? acc_guard->reason
          : m_guard ? m_guard->reason
          : l_guard ? l_guard->reason
          : output_guard ? output_guard->reason
          : gate_guard ? gate_guard->reason
          : dequantized_guard ? dequantized_guard->reason
          : projected_guard ? projected_guard->reason : "");
    passed &= inputs_unchanged && guards_ok;
    return passed;
}

std::uint16_t half_order_key(std::uint16_t bits) {
    return (bits & 0x8000u) != 0u
        ? static_cast<std::uint16_t>(~bits)
        : static_cast<std::uint16_t>(bits | 0x8000u);
}

bool run_prefill_qsa_layer_case(
    Context& context, QsaAttentionMode mode, const twin::Options& options,
    bool shared_encoder) {
    constexpr std::uint32_t kRows = 1u;
    constexpr std::uint32_t kPosition = 31u;
    constexpr std::uint32_t kCapacity = 128u;
    constexpr std::uint32_t kIdStride = INDEXER_SELECTION_WIDTH_MAX;
    constexpr std::uint32_t kMaxBlocks =
        kCapacity / INDEXER_BLOCK_CELLS + 2u;
    constexpr std::uint32_t kProjectionN = 2560u;
    constexpr std::uint32_t kProjectionK = N_HEAD_Q * HEAD_DIM;
    constexpr float kRopeZeroCos = 1.0625f;
    constexpr float kRopeZeroSin = -0.0f;
    const std::string label =
        "Prefill QSA projected-row layer " +
        std::string(mode_name(mode)) +
        std::string(shared_encoder ? " shared Encoder" : " standalone");
    std::string error;

    PrefillChunkMetadata metadata;
    const std::array token_ids{1u};
    if (!build_prefill_chunk_metadata(kPosition, token_ids, metadata, error)) {
        check(false, label + " metadata", error);
        return false;
    }
    const StepRow step{
        metadata.steps[0].pos, metadata.steps[0].n_kv,
        metadata.steps[0].n_bid, metadata.steps[0].width};
    constexpr std::size_t kKvInputBytes =
        static_cast<std::size_t>(2u) * HEAD_DIM * sizeof(float);
    constexpr std::size_t kQFullBytes =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * sizeof(float);
    constexpr std::size_t kIndexerQueryBytes =
        static_cast<std::size_t>(INDEXER_HEADS) * INDEXER_HEAD_DIM *
        sizeof(float);
    constexpr std::size_t kIndexerKeyBytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t kIndexerGammaBytes =
        static_cast<std::size_t>(INDEXER_HEAD_DIM) * sizeof(float);
    constexpr std::size_t kIndexerTailBytes =
        3u * INDEXER_HEAD_DIM * sizeof(float);
    constexpr std::size_t kIndexerPoolBytes =
        static_cast<std::size_t>(kMaxBlocks) * INDEXER_HEAD_DIM *
        sizeof(float);
    const std::size_t rope_bytes =
        static_cast<std::size_t>(kCapacity) * 32u * 2u * sizeof(float);

    AttentionInput attention_input;
    attention_input.query.assign(
        static_cast<std::size_t>(kRows) * N_HEAD_Q * HEAD_DIM, 0.0f);
    attention_input.steps.push_back(step);
    attention_input.ids.assign(kIdStride, kUntouchedId);
    CacheData cache = make_cache(
        mode, kCapacity, twin::InputSet::fixture,
        options.seed ^ (static_cast<std::uint64_t>(mode) << 32u) ^ 0x91d4u);
    AttentionBuffers attention;
    PrefillSelectionBuffers selection;
    std::size_t partial_acc_bytes = 0u;
    std::size_t partial_stats_bytes = 0u;
    if (!allocate_case_buffers(
            context, kRows, kIdStride, attention_input, cache, attention,
            partial_acc_bytes, partial_stats_bytes, error) ||
        !allocate_prefill_selection_buffers(
            context, kRows, kCapacity, selection, error)) {
        check(false, label + " selection and attention buffers", error);
        return false;
    }
    const std::vector<std::uint8_t> initial_attention_query(
        attention.query.payload().begin(), attention.query.payload().end());

    twin::GuardedBuffer keys, values, q_full, key_gamma, query_gamma, rope,
        indexer_query_gamma, raw_indexer_keys, indexer_gamma, indexer_tail,
        indexer_block_position, gate_f16, native_weights, dequantized_f16,
        projected_y;
    const bool allocated =
        keys.allocate(context, label + " projected K", kKvInputBytes, error) &&
        values.allocate(context, label + " projected V", kKvInputBytes,
                        error) &&
        q_full.allocate(context, label + " projected Qf", kQFullBytes,
                        error) &&
        key_gamma.allocate(context, label + " K gamma",
                           HEAD_DIM * sizeof(float), error) &&
        query_gamma.allocate(context, label + " Q gamma",
                             HEAD_DIM * sizeof(float), error) &&
        rope.allocate(context, label + " RoPE", rope_bytes, error) &&
        indexer_query_gamma.allocate(
            context, label + " indexer query gamma", kIndexerGammaBytes,
            error) &&
        raw_indexer_keys.allocate(context, label + " raw indexer K",
                                  kIndexerKeyBytes, error) &&
        indexer_gamma.allocate(context, label + " indexer gamma",
                               kIndexerGammaBytes, error) &&
        indexer_tail.allocate(context, label + " indexer tail",
                              kIndexerTailBytes, error) &&
        indexer_block_position.allocate(
            context, label + " indexer block position",
            sizeof(std::int32_t), error) &&
        gate_f16.allocate(
            context, label + " gate F16",
            static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM *
                sizeof(std::uint16_t),
            error) &&
        native_weights.allocate(
            context, label + " Q8_0 output weights",
            static_cast<std::size_t>(kProjectionN) *
                (kProjectionK / 32u) * sizeof(block_q8_0),
            error) &&
        dequantized_f16.allocate(
            context, label + " dequantized output weights",
            static_cast<std::size_t>(kProjectionN) * kProjectionK *
                sizeof(std::uint16_t),
            error) &&
        projected_y.allocate(
            context, label + " projected Y",
            static_cast<std::size_t>(kProjectionN) * sizeof(float), error);
    check(allocated, label + " preparation and output buffers", error);
    if (!allocated) return false;

    std::mt19937_64 rng(options.seed ^ 0x91d4a3u);
    const auto generate = [&](std::span<float> output, float scale,
                              std::size_t seed) {
        for (std::size_t i = 0u; i < output.size(); ++i) {
            output[i] = sample(twin::InputSet::fixture, rng, i + seed, scale);
        }
    };
    std::vector<float> source_keys(kKvInputBytes / sizeof(float));
    std::vector<float> source_values(source_keys.size());
    std::vector<float> source_q_full(kQFullBytes / sizeof(float));
    std::vector<float> source_indexer_query(
        kIndexerQueryBytes / sizeof(float));
    std::vector<float> source_raw_indexer(
        kIndexerKeyBytes / sizeof(float));
    std::vector<float> host_gamma(HEAD_DIM, 1.0f);
    std::vector<float> host_indexer_gamma(INDEXER_HEAD_DIM, 1.0f);
    std::vector<float> host_rope(rope_bytes / sizeof(float));
    generate(source_keys, 4.0f, 0x1121u);
    generate(source_values, 2.0f, 0x2243u);
    generate(source_q_full, 6.0f, 0x3365u);
    generate(source_indexer_query, 3.0f, 0x4487u);
    generate(source_raw_indexer, 2.0f, 0x55a9u);
    for (std::uint32_t cell = 0u; cell < kCapacity; ++cell) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float angle =
                static_cast<float>((cell * 11u + pair * 5u) % 251u) *
                0.011f;
            const std::size_t index =
                (static_cast<std::size_t>(cell) * 32u + pair) * 2u;
            host_rope[index] = std::cos(angle);
            host_rope[index + 1u] = std::sin(angle);
        }
    }
    std::memcpy(keys.payload().data(), source_keys.data(), kKvInputBytes);
    std::memcpy(values.payload().data(), source_values.data(), kKvInputBytes);
    std::memcpy(q_full.payload().data(), source_q_full.data(), kQFullBytes);
    std::memcpy(key_gamma.payload().data(), host_gamma.data(),
                HEAD_DIM * sizeof(float));
    std::memcpy(query_gamma.payload().data(), host_gamma.data(),
                HEAD_DIM * sizeof(float));
    std::memcpy(indexer_query_gamma.payload().data(),
                host_indexer_gamma.data(), kIndexerGammaBytes);
    std::memcpy(indexer_gamma.payload().data(), host_indexer_gamma.data(),
                kIndexerGammaBytes);
    std::memcpy(rope.payload().data(), host_rope.data(), rope_bytes);
    std::memcpy(selection.indexer_queries.payload().data(),
                source_indexer_query.data(), kIndexerQueryBytes);
    std::memcpy(raw_indexer_keys.payload().data(), source_raw_indexer.data(),
                kIndexerKeyBytes);
    std::memcpy(attention.steps.payload().data(), &step, sizeof(step));
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
    std::fill(typed<float>(selection.scores).begin(),
              typed<float>(selection.scores).end(), kUntouchedScore);
    std::fill(typed<std::int32_t>(attention.ids).begin(),
              typed<std::int32_t>(attention.ids).end(), kUntouchedId);

    oracle::QsaIndexerState indexer_state;
    indexer_state.tail.assign(3u * INDEXER_HEAD_DIM, -13.25f);
    indexer_state.dead.assign(INDEXER_HEAD_DIM, -17.5f);
    indexer_state.pooled.assign(kIndexerPoolBytes / sizeof(float), -19.75f);
    indexer_state.block_position = -23;
    std::vector<std::int32_t> prefix_positions(kPosition);
    std::vector<float> prefix_keys(
        static_cast<std::size_t>(kPosition) * INDEXER_HEAD_DIM);
    for (std::uint32_t row = 0u; row < kPosition; ++row) {
        prefix_positions[row] = static_cast<std::int32_t>(row);
        for (std::uint32_t dimension = 0u; dimension < INDEXER_HEAD_DIM;
             ++dimension) {
            const std::size_t index =
                static_cast<std::size_t>(row) * INDEXER_HEAD_DIM + dimension;
            prefix_keys[index] =
                sample(twin::InputSet::fixture, rng, index + 0x661bu, 2.0f);
        }
    }
    oracle::qsa_indexer_append(
        prefix_keys, prefix_positions, host_indexer_gamma, host_rope,
        kRopeZeroCos, kRopeZeroSin, indexer_state);
    std::memcpy(selection.pooled.payload().data(),
                indexer_state.pooled.data(), kIndexerPoolBytes);
    std::memcpy(selection.dead.payload().data(), indexer_state.dead.data(),
                INDEXER_HEAD_DIM * sizeof(float));
    std::memcpy(indexer_tail.payload().data(), indexer_state.tail.data(),
                kIndexerTailBytes);
    std::memcpy(indexer_block_position.payload().data(),
                &indexer_state.block_position, sizeof(std::int32_t));
    const auto snapshot_bytes = [](auto& buffer) {
        const auto payload = buffer.payload();
        return std::vector<std::uint8_t>(payload.begin(), payload.end());
    };
    const std::vector<std::uint8_t> initial_projected_keys =
        snapshot_bytes(keys);
    const std::vector<std::uint8_t> initial_projected_values =
        snapshot_bytes(values);
    const std::vector<std::uint8_t> initial_projected_q =
        snapshot_bytes(q_full);
    const std::vector<std::uint8_t> initial_steps =
        snapshot_bytes(attention.steps);
    const std::vector<std::uint8_t> initial_key_gamma =
        snapshot_bytes(key_gamma);
    const std::vector<std::uint8_t> initial_query_gamma =
        snapshot_bytes(query_gamma);
    const std::vector<std::uint8_t> initial_rope = snapshot_bytes(rope);
    const std::vector<std::uint8_t> initial_indexer_query =
        snapshot_bytes(selection.indexer_queries);
    const std::vector<std::uint8_t> initial_raw_indexer_key =
        snapshot_bytes(raw_indexer_keys);
    const std::vector<std::uint8_t> initial_indexer_gamma =
        snapshot_bytes(indexer_gamma);
    const std::vector<std::uint8_t> initial_indexer_query_gamma =
        snapshot_bytes(indexer_query_gamma);
    const std::vector<std::uint8_t> initial_indexer_tail =
        snapshot_bytes(indexer_tail);
    const std::vector<std::uint8_t> initial_indexer_dead =
        snapshot_bytes(selection.dead);
    const std::vector<std::uint8_t> initial_indexer_pool =
        snapshot_bytes(selection.pooled);
    const std::vector<std::uint8_t> initial_indexer_block_position =
        snapshot_bytes(indexer_block_position);
    const std::vector<std::uint8_t> initial_key_cache =
        snapshot_bytes(attention.key_data);
    const std::vector<std::uint8_t> initial_value_cache =
        snapshot_bytes(attention.value_data);
    const std::vector<std::uint8_t> initial_key_scales =
        snapshot_bytes(attention.key_scales);
    const std::vector<std::uint8_t> initial_value_scales =
        snapshot_bytes(attention.value_scales);

    std::vector<twin::tf> expected_keys_tf;
    const std::array positions{static_cast<std::int32_t>(kPosition)};
    oracle::qsa_norm_rope(source_keys, host_gamma, positions, host_rope,
                          kRows, QsaNormRopeMode::key, expected_keys_tf);
    const auto tf_to_float = [](std::span<const twin::tf> source) {
        std::vector<float> output;
        output.reserve(source.size());
        for (const auto value : source) output.push_back(value.value);
        return output;
    };
    const bool rotate_keys = mode == QsaAttentionMode::q4;
    const bool rotate_values =
        mode == QsaAttentionMode::q4 || mode == QsaAttentionMode::k8v4;
    const bool rotate_query = mode == QsaAttentionMode::q4;
    const bool transform_attention =
        mode == QsaAttentionMode::q4 || mode == QsaAttentionMode::k8v4;
    std::vector<twin::tf> expected_keys_transformed = expected_keys_tf;
    if (rotate_keys) {
        oracle::qsa_fwht256(tf_to_float(expected_keys_tf), 2u,
                           expected_keys_transformed);
    }
    std::vector<twin::tf> expected_values_transformed;
    if (rotate_values) {
        oracle::qsa_fwht256(source_values, 2u, expected_values_transformed);
    } else {
        expected_values_transformed.reserve(source_values.size());
        for (const float value : source_values) {
            expected_values_transformed.emplace_back(value);
        }
    }
    const std::vector<float> expected_key_values =
        tf_to_float(expected_keys_transformed);
    const std::vector<float> expected_value_values =
        tf_to_float(expected_values_transformed);
    cache.key_data.assign(attention.key_data.payload().begin(),
                          attention.key_data.payload().end());
    cache.value_data.assign(attention.value_data.payload().begin(),
                            attention.value_data.payload().end());
    if (mode == QsaAttentionMode::f16) {
        std::vector<std::uint16_t> key_pool(cache.key_data.size() / 2u);
        std::vector<std::uint16_t> value_pool(cache.value_data.size() / 2u);
        std::memcpy(key_pool.data(), cache.key_data.data(),
                    cache.key_data.size());
        std::memcpy(value_pool.data(), cache.value_data.data(),
                    cache.value_data.size());
        oracle::qsa_kv_append_f16(expected_key_values, expected_value_values,
                                  positions, key_pool, value_pool);
        std::memcpy(cache.key_data.data(), key_pool.data(),
                    cache.key_data.size());
        std::memcpy(cache.value_data.data(), value_pool.data(),
                    cache.value_data.size());
    } else if (mode == QsaAttentionMode::i8) {
        std::vector<std::int8_t> key_codes(cache.key_data.size());
        std::vector<std::int8_t> value_codes(cache.value_data.size());
        std::memcpy(key_codes.data(), cache.key_data.data(),
                    cache.key_data.size());
        std::memcpy(value_codes.data(), cache.value_data.data(),
                    cache.value_data.size());
        oracle::qsa_kv_append_i8(
            expected_key_values, expected_value_values, positions, key_codes,
            cache.key_scales, value_codes, cache.value_scales);
        std::memcpy(cache.key_data.data(), key_codes.data(),
                    cache.key_data.size());
        std::memcpy(cache.value_data.data(), value_codes.data(),
                    cache.value_data.size());
    } else if (mode == QsaAttentionMode::q4) {
        oracle::qsa_kv_append_q4(
            expected_key_values, expected_value_values, positions,
            cache.key_data, cache.value_data);
    } else {
        std::vector<std::int8_t> key_codes(cache.key_data.size());
        std::vector<std::int8_t> ignored_value_codes(
            static_cast<std::size_t>(kCapacity) * N_HEAD_KV * HEAD_DIM);
        std::vector<std::uint16_t> ignored_value_scales(
            static_cast<std::size_t>(kCapacity) * N_HEAD_KV * 4u);
        std::memcpy(key_codes.data(), cache.key_data.data(),
                    cache.key_data.size());
        oracle::qsa_kv_append_i8(
            expected_key_values, source_values, positions, key_codes,
            cache.key_scales, ignored_value_codes, ignored_value_scales);
        std::memcpy(cache.key_data.data(), key_codes.data(),
                    cache.key_data.size());
        std::vector<std::uint8_t> ignored_key_pool(cache.value_data.size());
        oracle::qsa_kv_append_q4(
            expected_key_values, expected_value_values, positions,
            ignored_key_pool, cache.value_data);
    }

    std::vector<twin::tf> expected_qcur;
    oracle::qsa_norm_rope(source_q_full, host_gamma, positions, host_rope,
                          kRows, QsaNormRopeMode::query, expected_qcur);
    std::vector<twin::tf> expected_qcur_transformed;
    if (rotate_query) {
        oracle::qsa_fwht256(tf_to_float(expected_qcur), N_HEAD_Q,
                           expected_qcur_transformed);
    } else {
        expected_qcur_transformed = expected_qcur;
    }
    const std::vector<float> expected_qcur_transformed_values =
        tf_to_float(expected_qcur_transformed);
    std::vector<twin::tf> expected_indexer_query;
    oracle::qsa_norm_rope(
        source_indexer_query, host_indexer_gamma, positions, host_rope,
        kRows, QsaNormRopeMode::indexer_query, expected_indexer_query);
    oracle::qsa_indexer_append(
        source_raw_indexer, positions, host_indexer_gamma, host_rope,
        kRopeZeroCos, kRopeZeroSin, indexer_state);

    const std::vector<std::int32_t> n_kv{step.n_kv};
    const std::vector<std::int32_t> n_bid{step.n_bid};
    const std::vector<float> expected_indexer_query_values =
        tf_to_float(expected_indexer_query);
    std::vector<twin::tf> expected_scores;
    oracle::qsa_selection_scores(
        indexer_state.pooled, indexer_state.dead,
        expected_indexer_query_values, n_kv, n_bid, kMaxBlocks,
        kUntouchedScore, expected_scores);
    const std::vector<float> expected_score_values =
        tf_to_float(expected_scores);
    std::vector<std::int32_t> expected_ids;
    oracle::qsa_selection_topk(expected_score_values, n_kv, n_bid,
                               kMaxBlocks, kUntouchedId, expected_ids);
    std::vector<twin::tf> expected_attention;
    oracle::qsa_attention(
        mode, expected_qcur_transformed_values,
        std::span<const StepRow>(&step, 1u), expected_ids, cache.key_data,
        cache.key_scales, cache.value_data, cache.value_scales, kRows,
        kCapacity, kIdStride, expected_attention);
    std::vector<twin::tf> expected_attention_transformed;
    if (transform_attention) {
        oracle::qsa_fwht256(tf_to_float(expected_attention), N_HEAD_Q,
                           expected_attention_transformed);
    } else {
        expected_attention_transformed = expected_attention;
    }
    std::vector<twin::tf> expected_gate;
    oracle::qsa_gate(tf_to_float(expected_attention_transformed), source_q_full,
                     kRows, expected_gate);
    std::vector<std::uint16_t> expected_gate_f16;
    expected_gate_f16.reserve(expected_gate.size());
    for (const auto value : expected_gate) {
        expected_gate_f16.push_back(half_bits(value.value));
    }

    const QsaNormRopeBuffers key_norm{
        keys.view(), key_gamma.view(), attention.steps.view(), rope.view(),
        keys.view()};
    const QsaNormRopeBuffers query_norm{
        q_full.view(), query_gamma.view(), attention.steps.view(), rope.view(),
        attention.query.view()};
    const QsaNormRopeBuffers indexer_query_norm{
        selection.indexer_queries.view(), indexer_query_gamma.view(),
        attention.steps.view(), rope.view(), selection.indexer_queries.view()};
    PrefillQsaKvAppendBuffers kv_append;
    switch (mode) {
        case QsaAttentionMode::f16:
            kv_append = QsaKvAppendF16Buffers{
                keys.view(), values.view(), attention.steps.view(),
                attention.key_data.view(), attention.value_data.view()};
            break;
        case QsaAttentionMode::i8:
            kv_append = QsaKvAppendI8Buffers{
                keys.view(), values.view(), attention.steps.view(),
                attention.key_data.view(), attention.key_scales.view(),
                attention.value_data.view(), attention.value_scales.view()};
            break;
        case QsaAttentionMode::q4:
            kv_append = QsaKvAppendQ4Buffers{
                keys.view(), values.view(), attention.steps.view(),
                attention.key_data.view(), attention.value_data.view()};
            break;
        case QsaAttentionMode::k8v4:
            kv_append = QsaKvAppendK8V4Buffers{
                keys.view(), values.view(), attention.steps.view(),
                attention.key_data.view(), attention.key_scales.view(),
                attention.value_data.view()};
            break;
    }
    const QsaPrefillIndexerAppendBuffers indexer_append{
        raw_indexer_keys.view(), indexer_gamma.view(), rope.view(),
        indexer_tail.view(), selection.dead.view(), selection.pooled.view(),
        indexer_block_position.view()};
    const QsaAttentionScratch scratch{
        attention.partial_acc.view(), attention.partial_m.view(),
        attention.partial_l.view()};
    const QsaPrefillSelectionBuffers selection_views{
        selection.pooled.view(), selection.dead.view(),
        selection.indexer_queries.view(), attention.steps.view(),
        selection.scores.view(), attention.ids.view()};
    const QsaPrefillAttentionBuffers attention_views{
        QsaAttentionChunkBuffers{
            attention.query.view(), attention.steps.view(),
            attention.ids.view(), attention.key_data.view(),
            attention.key_scales.view(), attention.value_data.view(),
            attention.value_scales.view(), scratch},
        QsaAttentionMergeBuffers{scratch, attention.output.view()}};
    constexpr std::uint16_t kProjectionScale = 0x3000u;  // F16 0.125
    auto q8_weights = typed<block_q8_0>(native_weights);
    for (block_q8_0& block : q8_weights) {
        block = {};
        block.d = static_cast<strata_half>(kProjectionScale);
        std::fill(std::begin(block.qs), std::end(block.qs),
                  static_cast<strata_i8>(1));
    }
    double expected_projection_sum = 0.0;
    double expected_projection_magnitude = 0.0;
    const float projection_scale =
        strata::kernels::f32_from_f16(kProjectionScale);
    for (const std::uint16_t activation_bits : expected_gate_f16) {
        const double product =
            static_cast<double>(
                strata::kernels::f32_from_f16(activation_bits)) *
            static_cast<double>(projection_scale);
        expected_projection_sum += product;
        expected_projection_magnitude += std::fabs(product);
    }
    std::vector<twin::tf> expected_projection;
    expected_projection.reserve(kProjectionN);
    for (std::uint32_t column = 0u; column < kProjectionN; ++column) {
        expected_projection.emplace_back(
            static_cast<float>(expected_projection_sum),
            expected_projection_magnitude);
    }
    if (!projected_y.poison_floats_against(
            expected_projection, twin::Comparison::units, error)) {
        check(false, label + " projection output poison", error);
        return false;
    }
    std::fill(attention.partial_acc.payload().begin(),
              attention.partial_acc.payload().end(), 0xa5u);
    std::fill(attention.partial_m.payload().begin(),
              attention.partial_m.payload().end(), 0xa5u);
    std::fill(attention.partial_l.payload().begin(),
              attention.partial_l.payload().end(), 0xa5u);
    if (!attention.output.poison_floats_against(
            expected_attention_transformed, twin::Comparison::units, error)) {
        check(false, label + " output poison", error);
        return false;
    }
    const PrefillQsaLayerBuffers layer_buffers{
        PrefillQsaPrepareBuffers{
            key_norm, query_norm, indexer_query_norm, kv_append,
            indexer_append},
        PrefillQsaTailBuffers{
            selection_views, attention_views,
            PrefillQsaOutputBuffers{
                {context.buffers().read_only(attention.output.view()),
                 context.buffers().read_only(q_full.view()), gate_f16.view()},
                context.buffers().read_only(native_weights.view()),
                dequantized_f16.view(), projected_y.view(),
                transform_attention ? attention.output.view() : BufferRef{}}}};
    const auto encode_layer = [&](CommandBuffer& command,
                                  const PrefillQsaLayerBuffers& buffers,
                                  PrefillGemmBackend* backend,
                                  std::string& encode_error) {
        if (shared_encoder) {
            return Encoder::encode(
                context, command, label,
                [&](Encoder& encoder, std::string& inner_error) {
                    return encode_prefill_qsa_layer(
                        context, encoder, metadata.params, mode,
                        kCapacity, kRopeZeroCos, kRopeZeroSin, false,
                        PrefillNativeWeightFormat::Q8_0, buffers, inner_error,
                        backend);
                },
                encode_error);
        }
        return encode_prefill_qsa_layer(
            context, command, metadata.params, mode, kCapacity,
            kRopeZeroCos, kRopeZeroSin, false,
            PrefillNativeWeightFormat::Q8_0, buffers, encode_error, backend);
    };
    auto invalid_buffers = layer_buffers;
    invalid_buffers.tail.attention.attention_chunk.query =
        selection.pooled.view();
    auto rejected_command =
        context.make_command_buffer(label + " invalid handoff", error);
    if (!rejected_command) {
        check(false, label + " invalid handoff command buffer", error);
        return false;
    }
    PrefillGemmBackend rejected_backend = PrefillGemmBackend::SIMD;
    const bool rejected =
        !encode_layer(*rejected_command, invalid_buffers, &rejected_backend,
                      error) &&
        rejected_command->unsafe_native_handle() == nullptr &&
        error.find("qcur must feed attention") != std::string::npos &&
        std::equal(source_keys.begin(), source_keys.end(),
                   typed<float>(keys).begin()) &&
        std::all_of(typed<float>(selection.scores).begin(),
                    typed<float>(selection.scores).end(),
                    [](float value) { return value == kUntouchedScore; });
    check(rejected, label + " rejects a broken handoff before dispatch", error);
    if (!rejected && rejected_command->unsafe_native_handle() != nullptr) {
        rejected_command->discard();
    }

    auto invalid_kv_buffers = layer_buffers;
    switch (mode) {
        case QsaAttentionMode::f16:
            invalid_kv_buffers.tail.attention.attention_chunk.key_data =
                attention.value_data.view();
            break;
        case QsaAttentionMode::i8:
            invalid_kv_buffers.tail.attention.attention_chunk.value_scales =
                attention.key_scales.view();
            break;
        case QsaAttentionMode::q4:
            invalid_kv_buffers.tail.attention.attention_chunk.value_data =
                attention.key_data.view();
            break;
        case QsaAttentionMode::k8v4:
            invalid_kv_buffers.tail.attention.attention_chunk.key_data =
                attention.value_data.view();
            break;
    }
    auto invalid_kv_command =
        context.make_command_buffer(label + " invalid KV handoff", error);
    if (!invalid_kv_command) {
        check(false, label + " invalid KV handoff command buffer", error);
        return false;
    }
    PrefillGemmBackend invalid_kv_backend = PrefillGemmBackend::SIMD;
    std::string invalid_kv_error;
    const bool invalid_kv_encoded =
        encode_layer(*invalid_kv_command, invalid_kv_buffers,
                     &invalid_kv_backend, invalid_kv_error);
    std::string invalid_kv_commit_error;
    const bool invalid_kv_committed =
        invalid_kv_command->commit(invalid_kv_commit_error);
    constexpr std::string_view kInvalidKvHandoffError =
        "Prefill QSA layer KV append outputs must feed attention";
    const bool invalid_kv_rejected =
        !invalid_kv_encoded && !invalid_kv_committed &&
        invalid_kv_command->unsafe_native_handle() == nullptr &&
        invalid_kv_error == kInvalidKvHandoffError &&
        invalid_kv_commit_error.find("discarded") != std::string::npos;
    check(invalid_kv_rejected,
          label + " rejects a mode-specific KV handoff before dispatch",
          invalid_kv_error +
              (invalid_kv_commit_error.empty()
                   ? ""
                   : "; " + invalid_kv_commit_error));

    auto late_invalid_buffers = layer_buffers;
    late_invalid_buffers.tail.selection.scores.length -= sizeof(float);
    auto late_rejected_command =
        context.make_command_buffer(label + " late rollback", error);
    if (!late_rejected_command) {
        check(false, label + " late rollback command buffer", error);
        return false;
    }
    PrefillGemmBackend late_rejected_backend = PrefillGemmBackend::SIMD;
    std::string late_encode_error;
    const bool late_encode_succeeded =
        encode_layer(*late_rejected_command, late_invalid_buffers,
                     &late_rejected_backend, late_encode_error);
    std::string late_commit_error;
    const bool late_commit_succeeded =
        late_rejected_command->commit(late_commit_error);
    const bool late_reject_reason =
        !late_encode_succeeded && !late_commit_succeeded &&
        late_encode_error.find("scores view") != std::string::npos &&
        late_commit_error.find("discarded") != std::string::npos;
    const auto bytes_unchanged = [](const std::vector<std::uint8_t>& expected,
                                    auto& buffer) {
        const auto actual = buffer.payload();
        return expected.size() == actual.size() &&
               std::equal(expected.begin(), expected.end(), actual.begin());
    };
    const bool late_inputs_unchanged =
        bytes_unchanged(initial_projected_keys, keys) &&
        bytes_unchanged(initial_projected_values, values) &&
        bytes_unchanged(initial_projected_q, q_full) &&
        bytes_unchanged(initial_steps, attention.steps) &&
        bytes_unchanged(initial_key_gamma, key_gamma) &&
        bytes_unchanged(initial_query_gamma, query_gamma) &&
        bytes_unchanged(initial_rope, rope) &&
        bytes_unchanged(initial_indexer_query, selection.indexer_queries) &&
        bytes_unchanged(initial_raw_indexer_key, raw_indexer_keys) &&
        bytes_unchanged(initial_indexer_gamma, indexer_gamma) &&
        bytes_unchanged(initial_indexer_query_gamma, indexer_query_gamma) &&
        bytes_unchanged(initial_attention_query, attention.query);
    const bool late_cache_unchanged =
        bytes_unchanged(initial_key_cache, attention.key_data) &&
        bytes_unchanged(initial_value_cache, attention.value_data) &&
        bytes_unchanged(initial_key_scales, attention.key_scales) &&
        bytes_unchanged(initial_value_scales, attention.value_scales);
    const bool late_indexer_unchanged =
        bytes_unchanged(initial_indexer_tail, indexer_tail) &&
        bytes_unchanged(initial_indexer_dead, selection.dead) &&
        bytes_unchanged(initial_indexer_pool, selection.pooled) &&
        bytes_unchanged(initial_indexer_block_position,
                        indexer_block_position);
    const bool late_rejected = late_reject_reason && late_inputs_unchanged &&
                               late_cache_unchanged &&
                               late_indexer_unchanged;
    check(late_reject_reason,
          label + " discards the command after a late tail failure",
          late_encode_error + (late_commit_error.empty()
                                   ? ""
                                   : "; " + late_commit_error));
    check(late_inputs_unchanged,
          label + " late rollback preserves raw projection, query and norm inputs");
    check(late_cache_unchanged,
          label + " late rollback preserves KV cache and scale bytes");
    check(late_indexer_unchanged,
          label + " late rollback preserves indexer query and state bytes");

    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    PrefillGemmBackend selected_backend = PrefillGemmBackend::SIMD;
    const bool encoded =
        encode_layer(*command, layer_buffers, &selected_backend, error);
    check(encoded, label + " encodes preparation through projection", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!wait_for_command(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    bool passed = rejected && late_rejected;
    const auto float_matches = [&](std::span<const float> actual,
                                   std::span<const twin::tf> expected,
                                   std::string_view stage) {
        const auto mismatch = twin::compare_floats(
            actual, expected, {twin::Comparison::units, 64.0, 0.0, 0u});
        check(!mismatch,
              label + " " + std::string(stage) +
                  " matches the scalar oracle under U(64)");
        passed &= !mismatch;
    };
    float_matches(typed<float>(keys), expected_keys_transformed,
                  "transformed K");
    float_matches(typed<float>(values), expected_values_transformed,
                  "transformed V");
    float_matches(typed<float>(attention.query), expected_qcur_transformed,
                  "qcur");
    float_matches(typed<float>(selection.indexer_queries),
                  expected_indexer_query, "indexer query");
    const bool key_bytes_match =
        std::equal(cache.key_data.begin(), cache.key_data.end(),
                   attention.key_data.payload().begin());
    const bool value_bytes_match =
        std::equal(cache.value_data.begin(), cache.value_data.end(),
                   attention.value_data.payload().begin());
    const auto scale_mismatch_detail =
        [](std::span<const std::uint16_t> expected,
           std::span<const std::uint16_t> actual) {
            if (expected.size() != actual.size()) {
                return "length expected=" + std::to_string(expected.size()) +
                       " actual=" + std::to_string(actual.size());
            }
            for (std::size_t i = 0u; i < expected.size(); ++i) {
                if (expected[i] != actual[i]) {
                    return "index=" + std::to_string(i) +
                           " expected=" + std::to_string(expected[i]) +
                           " actual=" + std::to_string(actual[i]);
                }
            }
            return std::string{};
        };
    const auto actual_key_scales =
        typed<std::uint16_t>(attention.key_scales);
    const auto actual_value_scales =
        typed<std::uint16_t>(attention.value_scales);
    const bool key_scales_match =
        cache.key_scales.empty() ||
        std::equal(cache.key_scales.begin(), cache.key_scales.end(),
                   actual_key_scales.begin());
    const bool value_scales_match =
        cache.value_scales.empty() ||
        std::equal(cache.value_scales.begin(), cache.value_scales.end(),
                   actual_value_scales.begin());
    const bool cache_matches = key_bytes_match && value_bytes_match &&
                               key_scales_match && value_scales_match;
    check(key_bytes_match, label + " key cache codes match oracle");
    check(value_bytes_match, label + " value cache codes match oracle");
    check(key_scales_match, label + " key cache scales match oracle",
          scale_mismatch_detail(cache.key_scales, actual_key_scales));
    check(value_scales_match, label + " value cache scales match oracle",
          scale_mismatch_detail(cache.value_scales, actual_value_scales));
    check(cache_matches, label + " KV cache matches oracle bytes");
    passed &= cache_matches;
    const bool indexer_matches =
        std::equal(indexer_state.tail.begin(), indexer_state.tail.end(),
                   typed<float>(indexer_tail).begin()) &&
        std::equal(indexer_state.dead.begin(), indexer_state.dead.end(),
                   typed<float>(selection.dead).begin()) &&
        std::equal(indexer_state.pooled.begin(), indexer_state.pooled.end(),
                   typed<float>(selection.pooled).begin());
    std::int32_t actual_indexer_block_position = 0;
    std::memcpy(&actual_indexer_block_position,
                indexer_block_position.payload().data(),
                sizeof(actual_indexer_block_position));
    const bool indexer_block_position_matches =
        actual_indexer_block_position == indexer_state.block_position;
    check(indexer_block_position_matches,
          label + " indexer block position matches scalar oracle");
    const bool indexer_state_matches =
        indexer_matches && indexer_block_position_matches;
    check(indexer_state_matches,
          label + " indexer append matches scalar oracle");
    passed &= indexer_state_matches;
    const auto scores_mismatch = twin::compare_floats(
        typed<float>(selection.scores).first(expected_scores.size()),
        expected_scores,
        {twin::Comparison::units, 64.0, 0.0, 0u});
    if (scores_mismatch) {
        twin::report_failure(label.c_str(), options, kRows, *scores_mismatch);
    }
    check(!scores_mismatch,
          label + " selection scores match scalar oracle under U(64)");
    passed &= !scores_mismatch;
    const bool ids_match =
        std::equal(expected_ids.begin(), expected_ids.end(),
                   typed<std::int32_t>(attention.ids).begin());
    check(ids_match, label + " selected IDs match scalar oracle");
    passed &= ids_match;
    const auto attention_mismatch = twin::compare_floats(
        typed<float>(attention.output), expected_attention_transformed,
        {twin::Comparison::units, 64.0, 0.0, 0u});
    if (attention_mismatch) {
        twin::report_failure(label.c_str(), options, kRows,
                             *attention_mismatch);
    }
    check(!attention_mismatch,
          label + " transformed attention matches scalar oracle");
    passed &= !attention_mismatch;
    bool gate_matches =
        typed<std::uint16_t>(gate_f16).size() >= expected_gate_f16.size();
    for (std::size_t i = 0u; gate_matches && i < expected_gate_f16.size(); ++i) {
        const std::uint16_t actual = half_order_key(
            typed<std::uint16_t>(gate_f16)[i]);
        const std::uint16_t expected = half_order_key(expected_gate_f16[i]);
        gate_matches &= (actual >= expected ? actual - expected
                                            : expected - actual) <= 2u;
    }
    check(gate_matches, label + " F16 gate matches scalar oracle within 2 ULP");
    passed &= gate_matches;
    const auto projection_mismatch = twin::compare_floats(
        typed<float>(projected_y), expected_projection,
        {twin::Comparison::units, 2.0 * kProjectionK, 0.0, 0u});
    check(!projection_mismatch,
          label + " Q8_0 projection matches the scalar FP64 oracle");
    passed &= !projection_mismatch;
    return passed;
}

bool run_prefill_qsa_gate_attn_case(
    Context& context, std::uint32_t rows, twin::InputSet set,
    const twin::Options& options, bool shared_encoder, bool apply_fwht) {
    constexpr std::size_t attn_row_values =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM;
    constexpr std::size_t q_full_row_values =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM;
    const std::size_t attn_values_count =
        static_cast<std::size_t>(rows) * attn_row_values;
    const std::size_t q_full_values_count =
        static_cast<std::size_t>(rows) * q_full_row_values;
    const std::string label =
        "Prefill QSA attention gate T=" + std::to_string(rows) +
        " set=" + twin::input_set_name(set) +
        (apply_fwht ? " FWHT" : "") +
        (shared_encoder ? " shared Encoder" : "");
    const std::uint64_t seed =
        options.seed ^ (static_cast<std::uint64_t>(rows) << 32u) ^
        (static_cast<std::uint64_t>(set) << 24u) ^
        (shared_encoder ? 0x534841524544ULL : 0x5354414e44414cULL);
    std::mt19937_64 rng(seed);
    std::vector<float> attn_values(attn_values_count);
    std::vector<float> q_full_values(q_full_values_count);
    for (std::size_t i = 0u; i < attn_values.size(); ++i) {
        attn_values[i] = sample(set, rng, i + 0x19731u, 8.0f);
    }
    for (std::uint32_t row = 0u; row < rows; ++row) {
        for (std::uint32_t head = 0u; head < N_HEAD_Q; ++head) {
            const std::size_t base =
                (static_cast<std::size_t>(row) * N_HEAD_Q + head) *
                (2u * HEAD_DIM);
            for (std::uint32_t dim = 0u; dim < HEAD_DIM; ++dim) {
                q_full_values[base + dim] =
                    32.0f + static_cast<float>(row * 3u + head) * 0.25f +
                    static_cast<float>(dim) * 0.001f;
                q_full_values[base + HEAD_DIM + dim] =
                    sample(set, rng, base + HEAD_DIM + dim, 12.0f);
            }
        }
    }

    const bool exercise_f16_boundaries =
        rows == 1u && set == twin::InputSet::edge && !apply_fwht;
    if (exercise_f16_boundaries) {
        attn_values[0] = 90000.0f;
        attn_values[1] = -90000.0f;
        attn_values[2] = 3.0f * std::ldexp(1.0f, -24);
        attn_values[3] = -3.0f * std::ldexp(1.0f, -24);
        attn_values[4] = -0.0f;
        q_full_values[HEAD_DIM] = 30.0f;
        q_full_values[HEAD_DIM + 1u] = 30.0f;
        q_full_values[HEAD_DIM + 2u] = 0.0f;
        q_full_values[HEAD_DIM + 3u] = 0.0f;
        q_full_values[HEAD_DIM + 4u] = 0.0f;
    }

    std::vector<twin::tf> expected_fwht;
    std::vector<float> expected_attn;
    if (apply_fwht) {
        oracle::qsa_fwht256(
            attn_values, rows * static_cast<std::uint32_t>(N_HEAD_Q),
            expected_fwht);
        expected_attn.reserve(expected_fwht.size());
        for (const twin::tf value : expected_fwht) {
            expected_attn.push_back(value.value);
        }
    } else {
        expected_attn = attn_values;
    }
    std::vector<twin::tf> expected_gate;
    oracle::qsa_gate(expected_attn, q_full_values, rows, expected_gate);
    std::vector<std::uint16_t> expected_half(expected_gate.size());
    for (std::size_t i = 0u; i < expected_gate.size(); ++i) {
        expected_half[i] = half_bits(expected_gate[i].value);
    }

    const std::size_t attn_bytes = attn_values_count * sizeof(float);
    const std::size_t q_full_bytes = q_full_values_count * sizeof(float);
    const std::size_t output_bytes =
        attn_values_count * sizeof(std::uint16_t);
    std::string error;
    twin::GuardedBuffer attn, q_full, output;
    if (!attn.allocate(context, label + " attn", attn_bytes, error) ||
        !q_full.allocate(context, label + " Qf", q_full_bytes, error) ||
        !output.allocate(context, label + " attn_h", output_bytes, error)) {
        check(false, label + " allocations", error);
        return false;
    }
    std::memcpy(attn.payload().data(), attn_values.data(), attn_bytes);
    std::memcpy(q_full.payload().data(), q_full_values.data(), q_full_bytes);
    std::fill(output.payload().begin(), output.payload().end(), 0xa5u);
    attn.capture_before_dispatch();
    q_full.capture_before_dispatch();
    output.capture_before_dispatch();
    const PrefillQsaGateAttnBuffers buffers{
        context.buffers().read_only(attn.view()),
        context.buffers().read_only(q_full.view()), output.view()};

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
                if (apply_fwht &&
                    !encode_qsa_fwht256(
                        context, encoder,
                        rows * static_cast<std::uint32_t>(N_HEAD_Q),
                        attn.view(), encode_error)) {
                    return false;
                }
                return encode_prefill_qsa_gate_attn(
                    context, encoder, rows, buffers, encode_error);
            },
            error);
    } else {
        encoded = !apply_fwht ||
            encode_qsa_fwht256(
                context, *command,
                rows * static_cast<std::uint32_t>(N_HEAD_Q),
                attn.view(), error);
        if (encoded) {
            encoded = encode_prefill_qsa_gate_attn(
                context, *command, rows, buffers, error);
        }
    }
    check(encoded, label + " encodes FWHT→gate", error);
    if (!encoded) {
        command->discard();
        return false;
    }
    if (!wait_for_command(command, label, error)) {
        check(false, label + " command completes", error);
        return false;
    }

    bool passed = true;
    if (apply_fwht) {
        const auto mismatch = twin::compare_floats(
            typed<float>(attn).first(attn_values_count), expected_fwht,
            {twin::Comparison::bits, 0.0, 0.0, 0u});
        check(!mismatch, label + " attention FWHT oracle",
              mismatch ? mismatch->reason : "");
        passed &= !mismatch;
    } else {
        const bool unchanged = !attn.first_unwritten_change({});
        check(unchanged, label + " attention input is unchanged");
        passed &= unchanged;
    }

    const auto actual_half = typed<std::uint16_t>(output)
                                 .first(expected_half.size());
    std::uint16_t max_ulp = 0u;
    bool output_matches = actual_half.size() == expected_half.size();
    for (std::size_t i = 0u; output_matches && i < actual_half.size(); ++i) {
        const std::uint16_t actual_order = half_order_key(actual_half[i]);
        const std::uint16_t expected_order = half_order_key(expected_half[i]);
        const std::uint16_t distance =
            actual_order >= expected_order
                ? static_cast<std::uint16_t>(actual_order - expected_order)
                : static_cast<std::uint16_t>(expected_order - actual_order);
        max_ulp = std::max(max_ulp, distance);
        output_matches &= distance <= 1u;
    }
    check(output_matches, label + " F16 gate matches scalar oracle within 1 ULP",
          "maximum distance=" + std::to_string(max_ulp));
    passed &= output_matches;
    if (exercise_f16_boundaries) {
        const bool boundaries =
            actual_half[0] == 0x7c00u && actual_half[1] == 0xfc00u &&
            actual_half[2] == 0x0002u && actual_half[3] == 0x8002u &&
            actual_half[4] == 0x8000u;
        check(boundaries,
              label + " unsaturated overflow, subnormal ties and signed zero");
        passed &= boundaries;
    }

    const bool q_full_unchanged = !q_full.first_unwritten_change({});
    const std::array<twin::ByteRange, 1> output_written{{
        {0u, output_bytes},
    }};
    const bool output_guards =
        !output.first_unwritten_change(output_written);
    check(q_full_unchanged, label + " Qf input is unchanged");
    check(output_guards, label + " F16 output guards");
    passed &= q_full_unchanged && output_guards;
    return passed;
}

bool test_prefill_qsa_gate_attn_validation(Context& context,
                                           Context& foreign_context) {
    constexpr std::uint32_t rows = 1u;
    constexpr std::size_t attn_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM * sizeof(float);
    constexpr std::size_t q_full_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * 2u * HEAD_DIM * sizeof(float);
    constexpr std::size_t output_bytes =
        static_cast<std::size_t>(N_HEAD_Q) * HEAD_DIM *
        sizeof(std::uint16_t);
    const std::string label = "Prefill QSA gate validation";
    std::string error;
    twin::GuardedBuffer attn, q_full, output;
    if (!attn.allocate(context, label + " attn", attn_bytes, error) ||
        !q_full.allocate(context, label + " Qf", q_full_bytes, error) ||
        !output.allocate(context, label + " output", output_bytes, error)) {
        check(false, label + " allocations", error);
        return false;
    }
    const PrefillQsaGateAttnBuffers valid{
        context.buffers().read_only(attn.view()),
        context.buffers().read_only(q_full.view()), output.view()};
    bool passed = true;
    auto reject = [&](const std::string& case_label,
                      const PrefillQsaGateAttnBuffers& candidate,
                      std::uint32_t candidate_rows,
                      std::string_view expected_error) {
        auto command = context.make_command_buffer(case_label, error);
        if (!command) {
            check(false, case_label + " command buffer", error);
            return false;
        }
        error.clear();
        const bool rejected =
            !encode_prefill_qsa_gate_attn(
                context, *command, candidate_rows, candidate, error) &&
            error.find(expected_error) != std::string::npos;
        check(rejected, case_label, error);
        command->discard();
        return rejected;
    };

    passed &= reject(label + " rejects zero rows", valid, 0u, "row count");
    passed &= reject(label + " rejects rows above maximum", valid,
                     kPrefillMaxShaderRows + 1u, "row count");
    passed &= reject(label + " checks maximum-row extents", valid,
                     kPrefillMaxShaderRows, "undersized or misaligned");

    PrefillQsaGateAttnBuffers short_attn = valid;
    short_attn.attn = context.buffers().read_only(buffer_subview(
        attn.view(), 0u, attn_bytes - sizeof(float)));
    passed &= reject(label + " rejects short attention", short_attn, rows,
                     "undersized or misaligned");
    PrefillQsaGateAttnBuffers short_q_full = valid;
    short_q_full.q_full = context.buffers().read_only(buffer_subview(
        q_full.view(), 0u, q_full_bytes - sizeof(float)));
    passed &= reject(label + " rejects short Qf", short_q_full, rows,
                     "undersized or misaligned");
    PrefillQsaGateAttnBuffers short_output = valid;
    short_output.attn_h =
        output.view().slice(0u, output_bytes - sizeof(std::uint16_t));
    passed &= reject(label + " rejects short attn_h", short_output, rows,
                     "undersized or misaligned");

    twin::GuardedBuffer misaligned_input, misaligned_output;
    if (!misaligned_input.allocate(context, label + " misaligned input",
                                   q_full_bytes + 8u, error) ||
        !misaligned_output.allocate(context, label + " misaligned output",
                                    output_bytes + 8u, error)) {
        check(false, label + " misaligned allocations", error);
        return false;
    }
    PrefillQsaGateAttnBuffers misaligned_attn = valid;
    misaligned_attn.attn = context.buffers().read_only(buffer_subview(
        misaligned_input.view(), 1u, attn_bytes));
    passed &= reject(label + " rejects misaligned attention",
                     misaligned_attn, rows, "undersized or misaligned");
    PrefillQsaGateAttnBuffers misaligned_q = valid;
    misaligned_q.q_full = context.buffers().read_only(buffer_subview(
        misaligned_input.view(), 2u, q_full_bytes));
    passed &= reject(label + " rejects misaligned Qf", misaligned_q, rows,
                     "undersized or misaligned");
    PrefillQsaGateAttnBuffers misaligned_half = valid;
    misaligned_half.attn_h =
        misaligned_output.view().slice(1u, output_bytes);
    passed &= reject(label + " rejects misaligned F16 output",
                     misaligned_half, rows, "undersized or misaligned");

    PrefillQsaGateAttnBuffers exact_attn_alias = valid;
    exact_attn_alias.attn_h = attn.view();
    passed &= reject(label + " rejects output/attention alias",
                     exact_attn_alias, rows, "overlaps a read-only input");
    PrefillQsaGateAttnBuffers exact_q_alias = valid;
    exact_q_alias.attn_h =
        q_full.view().slice(0u, output_bytes);
    passed &= reject(label + " rejects output/Qf alias", exact_q_alias,
                     rows, "overlaps a read-only input");

    twin::GuardedBuffer partial_alias;
    if (!partial_alias.allocate(context, label + " partial alias",
                                attn_bytes + output_bytes, error)) {
        check(false, label + " partial-alias allocation", error);
        return false;
    }
    PrefillQsaGateAttnBuffers partial = valid;
    partial.attn = context.buffers().read_only(
        buffer_subview(partial_alias.view(), 0u, attn_bytes));
    partial.attn_h = partial_alias.view().slice(
        attn_bytes - sizeof(std::uint16_t), output_bytes);
    passed &= reject(label + " rejects partial output/attention alias",
                     partial, rows, "overlaps a read-only input");

    twin::GuardedBuffer partial_q_alias;
    if (!partial_q_alias.allocate(context, label + " partial Qf alias",
                                  q_full_bytes + output_bytes, error)) {
        check(false, label + " partial Qf-alias allocation", error);
        return false;
    }
    PrefillQsaGateAttnBuffers partial_q = valid;
    partial_q.q_full = context.buffers().read_only(buffer_subview(
        partial_q_alias.view(), 0u, q_full_bytes));
    partial_q.attn_h = partial_q_alias.view().slice(
        q_full_bytes - sizeof(std::uint16_t), output_bytes);
    passed &= reject(label + " rejects partial output/Qf alias", partial_q,
                     rows, "overlaps a read-only input");

    twin::GuardedBuffer foreign_buffer;
    if (!foreign_buffer.allocate(foreign_context, label + " foreign input",
                                 attn_bytes, error)) {
        check(false, label + " foreign-buffer allocation", error);
        return false;
    }
    PrefillQsaGateAttnBuffers foreign = valid;
    foreign.attn = foreign_context.buffers().read_only(
        foreign_buffer.view());
    passed &= reject(label + " rejects a foreign Context input", foreign,
                     rows, "not owned");

    twin::GuardedBuffer foreign_q, foreign_output;
    if (!foreign_q.allocate(foreign_context, label + " foreign Qf",
                            q_full_bytes, error) ||
        !foreign_output.allocate(foreign_context, label + " foreign output",
                                 output_bytes, error)) {
        check(false, label + " foreign Qf/output allocations", error);
        return false;
    }
    PrefillQsaGateAttnBuffers foreign_query = valid;
    foreign_query.q_full =
        foreign_context.buffers().read_only(foreign_q.view());
    passed &= reject(label + " rejects foreign Qf", foreign_query, rows,
                     "not owned");
    PrefillQsaGateAttnBuffers foreign_write = valid;
    foreign_write.attn_h = foreign_output.view();
    passed &= reject(label + " rejects a foreign Context output",
                     foreign_write, rows, "not owned");

    PrefillQsaGateAttnBuffers forged_output = valid;
    forged_output.attn_h.gpu_address += sizeof(std::uint16_t);
    passed &= reject(label + " rejects forged output metadata", forged_output,
                     rows, "not owned");
    PrefillQsaGateAttnBuffers out_of_bounds = valid;
    out_of_bounds.attn_h.length += sizeof(std::uint16_t);
    passed &= reject(label + " rejects out-of-range output metadata",
                     out_of_bounds, rows, "not owned");

    PrefillQsaGateAttnBuffers readonly_overlap = valid;
    readonly_overlap.attn = context.buffers().read_only(
        buffer_subview(q_full.view(), 0u, attn_bytes));
    auto readonly_command =
        context.make_command_buffer(label + " read-only overlap", error);
    if (!readonly_command) {
        check(false, label + " read-only-overlap command buffer", error);
        return false;
    }
    error.clear();
    const bool readonly_accepted = encode_prefill_qsa_gate_attn(
        context, *readonly_command, rows, readonly_overlap, error);
    check(readonly_accepted,
          label + " permits overlapping read-only inputs", error);
    passed &= readonly_accepted;
    readonly_command->discard();

    auto foreign_command = foreign_context.make_command_buffer(
        label + " foreign command", error);
    if (!foreign_command) {
        check(false, label + " foreign command buffer", error);
        return false;
    }
    error.clear();
    const bool foreign_command_rejected =
        !encode_prefill_qsa_gate_attn(context, *foreign_command, rows, valid,
                                      error);
    check(foreign_command_rejected,
          label + " rejects a foreign command buffer", error);
    passed &= foreign_command_rejected;
    foreign_command->discard();

    auto submitted_command =
        context.make_command_buffer(label + " submitted command", error);
    if (!submitted_command || !submitted_command->commit(error) ||
        !submitted_command->wait_until_completed(error)) {
        check(false, label + " submitted command setup", error);
        return false;
    }
    error.clear();
    const bool submitted_rejected =
        !encode_prefill_qsa_gate_attn(context, *submitted_command, rows,
                                      valid, error);
    check(submitted_rejected,
          label + " rejects a submitted command buffer", error);
    passed &= submitted_rejected;

    auto standalone_rollback =
        context.make_command_buffer(label + " standalone rollback", error);
    if (!standalone_rollback) {
        check(false, label + " standalone rollback command buffer", error);
        return false;
    }
    std::fill(attn.payload().begin(), attn.payload().end(), 0u);
    std::fill(output.payload().begin(), output.payload().end(), 0xa5u);
    attn.capture_before_dispatch();
    output.capture_before_dispatch();
    const bool fwht_encoded = encode_qsa_fwht256(
        context, *standalone_rollback, N_HEAD_Q, attn.view(), error);
    PrefillQsaGateAttnBuffers late_invalid = valid;
    late_invalid.q_full = short_q_full.q_full;
    error.clear();
    const bool gate_rejected =
        fwht_encoded &&
        !encode_prefill_qsa_gate_attn(context, *standalone_rollback, rows,
                                      late_invalid, error) &&
        error.find("undersized or misaligned") != std::string::npos;
    const bool standalone_commit_rejected =
        !standalone_rollback->commit(error) &&
        error.find("discarded") != std::string::npos;
    const bool standalone_rollback_clean =
        !attn.first_unwritten_change({}) &&
        !output.first_unwritten_change({});
    check(gate_rejected && standalone_commit_rejected &&
              standalone_rollback_clean,
          label + " standalone late failure discards FWHT and preserves data",
          error);
    passed &= gate_rejected && standalone_commit_rejected &&
              standalone_rollback_clean;

    auto caller_rollback =
        context.make_command_buffer(label + " caller rollback", error);
    if (!caller_rollback) {
        check(false, label + " caller rollback command buffer", error);
        return false;
    }
    bool caller_gate_rejected = false;
    const bool caller_encoded = Encoder::encode(
        context, *caller_rollback, label + " caller rollback",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_qsa_fwht256(context, encoder, N_HEAD_Q,
                                    attn.view(), body_error)) {
                return false;
            }
            caller_gate_rejected =
                !encode_prefill_qsa_gate_attn(
                    context, encoder, rows, late_invalid, body_error) &&
                body_error.find("undersized or misaligned") !=
                    std::string::npos;
            return caller_gate_rejected ? false : true;
        },
        error);
    const bool caller_commit_rejected =
        !caller_rollback->commit(error) &&
        error.find("discarded") != std::string::npos;
    const bool caller_rollback_clean =
        !attn.first_unwritten_change({}) &&
        !output.first_unwritten_change({});
    check(!caller_encoded && caller_gate_rejected &&
              caller_commit_rejected && caller_rollback_clean,
          label + " caller-owned late failure discards FWHT and preserves data",
          error);
    passed &= !caller_encoded && caller_gate_rejected &&
              caller_commit_rejected && caller_rollback_clean;

    const PrefillGemmMPPParams invalid_projection_params{};
    const PrefillGemmMPPBuffers invalid_projection_buffers{
        context.buffers().read_only(output.view()),
        context.buffers().read_only(q_full.view()), output.view()};
    auto downstream_standalone =
        context.make_command_buffer(label + " downstream standalone", error);
    if (!downstream_standalone) {
        check(false, label + " downstream standalone command buffer", error);
        return false;
    }
    error.clear();
    const bool standalone_gate_encoded = encode_prefill_qsa_gate_attn(
        context, *downstream_standalone, rows, valid, error);
    const bool standalone_projection_rejected =
        standalone_gate_encoded &&
        !encode_prefill_gemm(context, *downstream_standalone,
                             invalid_projection_params,
                             invalid_projection_buffers,
                             PrefillGemmFormat::F16, error);
    const bool standalone_projection_error_expected =
        standalone_projection_rejected &&
        error == "STRATA_METAL_GEMM selection requires n > 0";
    const bool downstream_standalone_commit_rejected =
        !downstream_standalone->commit(error) &&
        error.find("discarded") != std::string::npos;
    const bool downstream_standalone_untouched =
        !output.first_unwritten_change({});
    check(standalone_gate_encoded && standalone_projection_error_expected &&
              downstream_standalone_commit_rejected &&
              downstream_standalone_untouched,
          label + " downstream standalone failure discards encoded gate",
          error);
    passed &= standalone_gate_encoded && standalone_projection_error_expected &&
              downstream_standalone_commit_rejected &&
              downstream_standalone_untouched;

    auto downstream_caller =
        context.make_command_buffer(label + " downstream caller", error);
    if (!downstream_caller) {
        check(false, label + " downstream caller command buffer", error);
        return false;
    }
    bool caller_projection_rejected = false;
    bool caller_projection_error_expected = false;
    const bool downstream_caller_encoded = Encoder::encode(
        context, *downstream_caller, label + " downstream caller",
        [&](Encoder& encoder, std::string& body_error) {
            if (!encode_prefill_qsa_gate_attn(
                    context, encoder, rows, valid, body_error)) {
                return false;
            }
            caller_projection_rejected = !encode_prefill_gemm(
                context, encoder, invalid_projection_params,
                invalid_projection_buffers, PrefillGemmFormat::F16,
                body_error);
            caller_projection_error_expected =
                caller_projection_rejected &&
                body_error ==
                    "STRATA_METAL_GEMM selection requires n > 0";
            return caller_projection_rejected ? false : true;
        },
        error);
    const bool downstream_caller_commit_rejected =
        !downstream_caller->commit(error) &&
        error.find("discarded") != std::string::npos;
    const bool downstream_caller_untouched =
        !output.first_unwritten_change({});
    check(!downstream_caller_encoded &&
              caller_projection_error_expected &&
              downstream_caller_commit_rejected &&
              downstream_caller_untouched,
          label + " downstream caller failure discards encoded gate", error);
    passed &= !downstream_caller_encoded &&
              caller_projection_error_expected &&
              downstream_caller_commit_rejected &&
              downstream_caller_untouched;

    if (concurrent_dispatch_mode()) {
        auto group_command =
            context.make_command_buffer(label + " open group", error);
        if (!group_command) {
            check(false, label + " open-group command buffer", error);
            return false;
        }
        bool saw_open_group = false;
        bool gate_succeeded = false;
        const bool group_encoded = Encoder::encode(
            context, *group_command, label + " open group",
            [&](Encoder& encoder, std::string& body_error) {
                if (!encoder.group_begin(body_error)) return false;
                saw_open_group = encoder.concurrent_group_open();
                if (saw_open_group) {
                    std::string gate_error;
                    gate_succeeded = encode_prefill_qsa_gate_attn(
                        context, encoder, rows, valid, gate_error);
                    body_error = gate_error;
                }
                std::string group_error;
                const bool group_closed = encoder.group_end(group_error);
                if (!group_closed) body_error = group_error;
                return false;
            },
            error);
        const bool group_rejected =
            !group_encoded && saw_open_group && !gate_succeeded &&
            error.find("ordered concurrent-group boundary") !=
                std::string::npos;
        check(group_rejected,
              label + " rejects an open concurrent group", error);
        passed &= group_rejected;
    }
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
    passed &= test_prefill_qsa_gate_attn_validation(*context,
                                                    *foreign_context);
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
    for (const QsaAttentionMode mode : modes) {
        passed &= run_prefill_qsa_layer_case(
            *context, mode, options, concurrent_dispatch_mode());
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
    for (const QsaAttentionMode mode : modes) {
        passed &= run_prefill_attention_case(
            *context, mode, 2u, 3u, 128u, options,
            concurrent_dispatch_mode(), true);
        passed &= run_prefill_attention_case(
            *context, mode, 2u, 3u, 128u, options,
            concurrent_dispatch_mode(), true, false);
    }
    for (const twin::InputSet set : selected_sets) {
        passed &= run_prefill_qsa_gate_attn_case(
            *context, 1u, set, options, false, false);
        passed &= run_prefill_qsa_gate_attn_case(
            *context, 9u, set, options, true, true);
        passed &= run_prefill_qsa_gate_attn_case(
            *context, 65u, set, options, false, true);
    }

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
