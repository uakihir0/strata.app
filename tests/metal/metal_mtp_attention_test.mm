// Metal twin for MTP drafter stage 13 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "oracles/attention.hpp"
#include "oracles/qsa.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kCellCapacity = 64u;
constexpr std::uint32_t kIdStride = 64u;
constexpr std::uint32_t kWindow = 17u;
constexpr std::int32_t kPosition = 39;
constexpr std::int32_t kKvCount = kPosition + 1;
constexpr std::uint32_t kQueryValues = N_HEAD_Q * HEAD_DIM;
constexpr std::size_t kQueryBytes =
    static_cast<std::size_t>(kQueryValues) * sizeof(float);
constexpr std::size_t kOutputBytes = kQueryBytes;
constexpr std::int32_t kIdSentinel = 0x1234567;
constexpr float kFloatSentinel = -913.25f;

int g_failures = 0;

void check(bool condition, const std::string& label,
           const std::string& detail = {}) {
    std::printf("%s %s%s%s\n", condition ? "ok  " : "FAIL", label.c_str(),
                detail.empty() ? "" : ": ",
                detail.empty() ? "" : detail.c_str());
    std::fflush(stdout);
    g_failures += !condition;
}

template <class T>
std::span<T> typed(twin::GuardedBuffer& buffer) {
    auto bytes = buffer.payload();
    return {reinterpret_cast<T*>(bytes.data()), bytes.size() / sizeof(T)};
}

struct CacheData {
    std::vector<std::uint8_t> key_data;
    std::vector<std::uint16_t> key_scales;
    std::vector<std::uint8_t> value_data;
    std::vector<std::uint16_t> value_scales;
};

std::vector<float> make_values(std::size_t count, std::size_t seed,
                               float scale) {
    std::vector<float> values(count);
    for (std::size_t i = 0u; i < count; ++i) {
        const auto pattern = static_cast<std::int32_t>(
            (i * 73u + seed * 29u + (i / 256u) * 17u) % 1021u);
        values[i] = static_cast<float>(pattern - 510) * (scale / 512.0f);
    }
    return values;
}

void rotate_rows(std::vector<float>& values, std::uint32_t rows) {
    std::vector<twin::tf> rotated;
    oracle::qsa_fwht256(values, rows, rotated);
    std::transform(rotated.begin(), rotated.end(), values.begin(),
                   [](twin::tf value) { return value.value; });
}

CacheData make_cache(QsaAttentionMode mode, bool kv_rot,
                     std::uint32_t cell_capacity) {
    const std::uint32_t cache_rows = cell_capacity * N_HEAD_KV;
    std::vector<float> keys = make_values(
        static_cast<std::size_t>(cache_rows) * HEAD_DIM, 13u, 0.375f);
    std::vector<float> values = make_values(
        static_cast<std::size_t>(cache_rows) * HEAD_DIM, 71u, 0.25f);
    if (kv_rot) {
        rotate_rows(keys, cache_rows);
        rotate_rows(values, cache_rows);
    }

    std::vector<std::int32_t> positions(cell_capacity);
    for (std::uint32_t cell = 0u; cell < cell_capacity; ++cell) {
        positions[cell] = static_cast<std::int32_t>(cell);
    }

    CacheData cache;
    if (mode == QsaAttentionMode::f16) {
        std::vector<std::uint16_t> key_pool(
            static_cast<std::size_t>(cache_rows) * HEAD_DIM);
        std::vector<std::uint16_t> value_pool(key_pool.size());
        oracle::qsa_kv_append_f16(keys, values, positions, key_pool,
                                  value_pool);
        cache.key_data.resize(key_pool.size() * sizeof(std::uint16_t));
        cache.value_data.resize(value_pool.size() * sizeof(std::uint16_t));
        std::memcpy(cache.key_data.data(), key_pool.data(),
                    cache.key_data.size());
        std::memcpy(cache.value_data.data(), value_pool.data(),
                    cache.value_data.size());
    } else if (mode == QsaAttentionMode::i8) {
        std::vector<std::int8_t> key_codes(
            static_cast<std::size_t>(cache_rows) * HEAD_DIM);
        std::vector<std::int8_t> value_codes(key_codes.size());
        cache.key_scales.resize(static_cast<std::size_t>(cache_rows) * 4u);
        cache.value_scales.resize(cache.key_scales.size());
        oracle::qsa_kv_append_i8(keys, values, positions, key_codes,
                                 cache.key_scales, value_codes,
                                 cache.value_scales);
        cache.key_data.resize(key_codes.size());
        cache.value_data.resize(value_codes.size());
        std::memcpy(cache.key_data.data(), key_codes.data(),
                    cache.key_data.size());
        std::memcpy(cache.value_data.data(), value_codes.data(),
                    cache.value_data.size());
    } else {
        cache.key_data.resize(static_cast<std::size_t>(cache_rows) * 8u *
                              sizeof(block_q4_0));
        cache.value_data.resize(cache.key_data.size());
        oracle::qsa_kv_append_q4(keys, values, positions, cache.key_data,
                                 cache.value_data);
    }
    return cache;
}

QsaAttentionMode attention_mode(KvFormat format) {
    switch (format) {
        case KvFormat::Fp16: return QsaAttentionMode::f16;
        case KvFormat::Int8:
        case KvFormat::K8v4: return QsaAttentionMode::i8;
        case KvFormat::Q4_0: return QsaAttentionMode::q4;
    }
    return QsaAttentionMode::f16;
}

struct Fixture {
    twin::GuardedBuffer params;
    twin::GuardedBuffer qcur;
    twin::GuardedBuffer mtp_ids;
    twin::GuardedBuffer att_scratch;
    twin::GuardedBuffer attn;
    std::unique_ptr<MtpKvStore> kv_store;
    MtpKvLayout kv_layout;
    MtpKvViews kv_views;
    CacheData cache;
    bool kv_rot = false;
    std::int32_t position = kPosition;
    std::int32_t kv_count = kKvCount;
    std::uint32_t window = kWindow;
    std::uint32_t id_row_stride = kIdStride;
    std::size_t scratch_active_bytes = 0u;
    std::size_t scratch_bytes = 0u;

    bool allocate(Context& context, KvFormat format, bool kv_rot,
                  const std::string& label,
                  std::uint32_t requested_cells = kCellCapacity,
                  std::int32_t requested_position = kPosition,
                  std::uint32_t requested_window = kWindow,
                  std::uint32_t requested_id_stride = kIdStride) {
        this->kv_rot = kv_rot;
        id_row_stride = requested_id_stride;
        const std::size_t chunks =
            (static_cast<std::size_t>(id_row_stride) + 63u) / 64u;
        scratch_active_bytes = chunks * 24u * 258u * sizeof(float);
        scratch_bytes =
            scratch_active_bytes + 64u * sizeof(float);
        std::string error;
        kv_store =
            MtpKvStore::open(context, requested_cells, format, error);
        if (!kv_store) {
            check(false, label + " MTP KV allocation", error);
            return false;
        }
        kv_layout = kv_store->layout();
        position = std::min(
            requested_position,
            static_cast<std::int32_t>(kv_layout.cell_capacity - 1u));
        kv_count = position + 1;
        window = std::min(requested_window,
                          static_cast<std::uint32_t>(kv_count));
        if (!kv_store->views(kv_views, error)) {
            check(false, label + " MTP KV views", error);
            return false;
        }

        const bool allocated =
            params.allocate(context, label + " params", 256u, error) &&
            qcur.allocate(context, label + " qcur", scratch_bytes, error) &&
            mtp_ids.allocate(
                context, label + " IDs",
                static_cast<std::size_t>(id_row_stride) *
                    sizeof(std::int32_t),
                error) &&
            att_scratch.allocate(context, label + " attention scratch",
                                 scratch_bytes, error) &&
            attn.allocate(context, label + " attention output",
                          kOutputBytes, error);
        if (!allocated) {
            check(false, label + " buffers allocated", error);
            return false;
        }

        cache = make_cache(attention_mode(format), kv_rot,
                           kv_layout.cell_capacity);
        install_cache();
        return true;
    }

    void install_cache() {
        if (!cache.key_data.empty()) {
            std::memcpy(kv_views.k.contents, cache.key_data.data(),
                        cache.key_data.size());
        }
        if (!cache.key_scales.empty()) {
            std::memcpy(kv_views.k_scale.contents, cache.key_scales.data(),
                        cache.key_scales.size() * sizeof(std::uint16_t));
        }
        if (!cache.value_data.empty()) {
            std::memcpy(kv_views.v.contents, cache.value_data.data(),
                        cache.value_data.size());
        }
        if (!cache.value_scales.empty()) {
            std::memcpy(kv_views.v_scale.contents, cache.value_scales.data(),
                        cache.value_scales.size() * sizeof(std::uint16_t));
        }
    }

    void initialize(bool kv_rot, bool host_ids) {
        MtpParams block{};
        block.T = 1u;
        block.step[0].pos = position;
        block.step[0].n_kv = kv_count;
        block.step[0].n_bid =
            kv_count / static_cast<std::int32_t>(INDEXER_BLOCK_CELLS);
        block.step[0].width = static_cast<std::int32_t>(window);
        std::memcpy(params.payload().data(), &block, sizeof(block));

        std::vector<float> query =
            make_values(kQueryValues, 97u, 0.125f);
        if (kv_rot) rotate_rows(query, N_HEAD_Q);
        std::memcpy(qcur.payload().data(), query.data(), kQueryBytes);
        std::fill(typed<std::int32_t>(mtp_ids).begin(),
                  typed<std::int32_t>(mtp_ids).end(), kIdSentinel);
        if (host_ids) {
        const std::int32_t first =
            kv_count - static_cast<std::int32_t>(window);
        for (std::uint32_t index = 0u; index < window; ++index) {
                typed<std::int32_t>(mtp_ids)[index] =
                    first + static_cast<std::int32_t>(index);
            }
        }
        std::fill(typed<float>(att_scratch).begin(),
                  typed<float>(att_scratch).end(), kFloatSentinel);
        std::fill(typed<float>(attn).begin(), typed<float>(attn).end(),
                  kFloatSentinel);
    }

    MtpAttentionBuffers attention_buffers() {
        return {params.view(), qcur.view(), mtp_ids.view(), kv_layout,
                kv_views, att_scratch.view(), attn.view(), id_row_stride,
                kv_rot};
    }

    MtpWindowIdsBuffers window_buffers(Context& context) {
        return {context.buffers().read_only(params.view()), mtp_ids.view(),
                window};
    }

    std::vector<std::uint8_t> cache_snapshot() const {
        const BufferRef allocation = kv_store->allocation();
        const auto* bytes =
            static_cast<const std::uint8_t*>(allocation.contents);
        return {bytes, bytes + kv_layout.allocation_bytes};
    }
};

bool submit_and_wait(std::unique_ptr<CommandBuffer>& command,
                     const std::string& label, std::string& error) {
    if (!command || !command->commit(error)) return false;
    id<MTLCommandBuffer> native =
        (__bridge id<MTLCommandBuffer>)command->unsafe_native_handle();
    if (native == nil) {
        error = label + " command buffer became unavailable";
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

std::vector<twin::tf> expected_attention(const Fixture& fixture,
                                         QsaAttentionMode mode,
                                         bool kv_rot) {
    const auto* params =
        static_cast<const MtpParams*>(fixture.params.view().contents);
    const std::span<const StepRow> steps(params->step, 1u);
    std::vector<std::int32_t> ids(fixture.id_row_stride);
    const std::int32_t first =
        fixture.kv_count - static_cast<std::int32_t>(fixture.window);
    for (std::uint32_t index = 0u; index < fixture.window; ++index) {
        ids[index] = first + static_cast<std::int32_t>(index);
    }
    const auto query = std::span<const float>(
        reinterpret_cast<const float*>(fixture.qcur.payload().data()),
        kQueryValues);
    std::vector<twin::tf> output;
    oracle::qsa_attention(
        mode, query, steps, ids, fixture.cache.key_data,
        fixture.cache.key_scales, fixture.cache.value_data,
        fixture.cache.value_scales, 1u, fixture.kv_layout.cell_capacity,
        fixture.id_row_stride, output);
    if (kv_rot) {
        std::vector<float> floats;
        floats.reserve(output.size());
        for (const twin::tf value : output) floats.push_back(value.value);
        std::vector<twin::tf> rotated;
        oracle::qsa_fwht256(floats, N_HEAD_Q, rotated);
        output = std::move(rotated);
    }
    return output;
}

bool run_case(Context& context, KvFormat format, bool kv_rot,
              bool compose_window, const std::string& label,
              std::uint32_t requested_cells = kCellCapacity,
              std::int32_t position = kPosition,
              std::uint32_t window = kWindow,
              std::uint32_t id_row_stride = kIdStride) {
    Fixture fixture;
    if (!fixture.allocate(context, format, kv_rot, label, requested_cells,
                          position, window, id_row_stride)) {
        return false;
    }
    fixture.initialize(kv_rot, !compose_window);
    const auto expected =
        expected_attention(fixture, attention_mode(format), kv_rot);
    const std::vector<std::uint8_t> kv_before = fixture.cache_snapshot();
    fixture.params.capture_before_dispatch();
    fixture.qcur.capture_before_dispatch();
    fixture.mtp_ids.capture_before_dispatch();
    fixture.att_scratch.capture_before_dispatch();
    fixture.attn.capture_before_dispatch();

    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }

    bool encoded = false;
    if (compose_window) {
        encoded = Encoder::encode(
            context, *command, label + " combined stages",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_window_ids(
                           context, encoder,
                           fixture.window_buffers(context), encode_error) &&
                       encode_mtp_attention(
                           context, encoder, fixture.attention_buffers(),
                           encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_attention(
            context, *command, fixture.attention_buffers(), error);
    }
    if (!encoded) {
        check(false, label + " encoding", error);
        command->discard();
        return false;
    }
    if (!submit_and_wait(command, label, error)) {
        check(false, label + " GPU completion", error);
        return false;
    }

    const auto actual = std::span<const float>(
        reinterpret_cast<const float*>(fixture.attn.payload().data()),
        expected.size());
    const double tolerance =
        format == KvFormat::Q4_0 ? 256.0 : 32.0;
    const auto mismatch = twin::compare_floats(
        actual, expected,
        {twin::Comparison::units, tolerance, 0.0, 0u});
    if (mismatch) {
        check(false, label + " scalar attention oracle",
              "index " + std::to_string(mismatch->index) + " " +
                  mismatch->reason);
    } else {
        check(true, label + " scalar attention oracle");
    }

    const std::array written_ids{
        twin::ByteRange{0u, static_cast<std::size_t>(fixture.window) *
                                sizeof(std::int32_t)}};
    const bool ids_valid = [&] {
        const auto actual_ids = typed<std::int32_t>(fixture.mtp_ids);
        const std::int32_t first =
            fixture.kv_count - static_cast<std::int32_t>(fixture.window);
        for (std::uint32_t index = 0u; index < fixture.window; ++index) {
            if (actual_ids[index] !=
                first + static_cast<std::int32_t>(index)) {
                return false;
            }
        }
        return !fixture.mtp_ids.first_unwritten_change(
            compose_window ? std::span<const twin::ByteRange>(written_ids)
                           : std::span<const twin::ByteRange>{});
    }();
    check(ids_valid, label + " absolute IDs and untouched suffix");
    const bool inputs_unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.qcur.first_unwritten_change({});
    check(inputs_unchanged, label + " immutable params and query");
    const std::array written_scratch{
        twin::ByteRange{0u, fixture.scratch_active_bytes}};
    const bool scratch_tail_unchanged =
        !fixture.att_scratch.first_unwritten_change(written_scratch);
    check(scratch_tail_unchanged,
          label + " reserved scratch tail and allocation guards");
    const std::array written_output{
        twin::ByteRange{0u, kOutputBytes}};
    check(!fixture.attn.first_unwritten_change(written_output),
          label + " output allocation guards");
    check(fixture.cache_snapshot() == kv_before,
          label + " immutable K/V cache");
    return !mismatch && ids_valid && inputs_unchanged &&
           scratch_tail_unchanged &&
           !fixture.attn.first_unwritten_change(written_output) &&
           fixture.cache_snapshot() == kv_before;
}

bool reject_case(Context& context, const MtpAttentionBuffers& buffers,
                 const std::string& label, std::string_view expected_error) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    const bool rejected =
        !encode_mtp_attention(context, *command, buffers, error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool message_matches =
        error.find(expected_error) != std::string::npos;
    check(rejected && discarded && message_matches, label, error);
    command->discard();
    return rejected && discarded && message_matches;
}

bool test_validation(Context& context) {
    Fixture fixture;
    if (!fixture.allocate(context, KvFormat::Fp16, false,
                          "MTP attention validation")) {
        return false;
    }
    fixture.initialize(false, true);
    const auto good = fixture.attention_buffers();
    bool passed = true;

    auto bad = good;
    bad.id_row_stride = kIdStride + 1u;
    passed &= reject_case(context, bad, "MTP attention rejects non-chunk stride",
                          "multiple of 64");

    bad = good;
    bad.mtp_ids = bad.mtp_ids.slice(0u, (kIdStride - 1u) *
                                             sizeof(std::int32_t));
    passed &= reject_case(context, bad, "MTP attention rejects short IDs",
                          "undersized or misaligned");

    bad = good;
    bad.qcur = bad.qcur.slice(0u, kQueryBytes - sizeof(float));
    passed &= reject_case(context, bad, "MTP attention rejects short qcur",
                          "undersized or misaligned");

    bad = good;
    bad.att_scratch =
        bad.att_scratch.slice(0u, fixture.scratch_bytes - sizeof(float));
    passed &= reject_case(context, bad,
                          "MTP attention rejects short bind scratch",
                          "undersized or misaligned");

    bad = good;
    bad.attn = bad.attn.slice(0u, kOutputBytes - sizeof(float));
    passed &= reject_case(context, bad, "MTP attention rejects short output",
                          "undersized or misaligned");

    bad = good;
    bad.att_scratch = bad.qcur;
    passed &= reject_case(context, bad,
                          "MTP attention rejects scratch/query alias",
                          "overlaps input qcur");

    bad = good;
    bad.attn = bad.qcur;
    passed &= reject_case(context, bad,
                          "MTP attention rejects output/query alias",
                          "overlaps input qcur");

    bad = good;
    bad.attn = bad.kv_views.k;
    passed &= reject_case(context, bad,
                          "MTP attention rejects output/cache alias",
                          "overlaps input K cache");

    bad = good;
    bad.kv_rot = true;
    passed &= reject_case(context, bad,
                          "MTP attention rejects fp16 rotation mismatch",
                          "kv_rot does not match");

    bad = good;
    bad.kv_layout.storage_format = KvFormat::Q4_0;
    passed &= reject_case(context, bad,
                          "MTP attention rejects forged cache layout",
                          "not canonical");

    return passed;
}

bool test_late_rollback(Context& context) {
    Fixture fixture;
    if (!fixture.allocate(context, KvFormat::Int8, false,
                          "MTP attention rollback")) {
        return false;
    }
    fixture.initialize(false, false);
    const auto good_attention = fixture.attention_buffers();
    const auto good_window = fixture.window_buffers(context);
    auto bad_attention = good_attention;
    bad_attention.attn =
        bad_attention.attn.slice(0u, kOutputBytes - sizeof(float));
    std::string error;
    auto command = context.make_command_buffer("MTP attention rollback",
                                               error);
    if (!command) {
        check(false, "MTP attention rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stages 12-13 rollback",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_mtp_window_ids(context, encoder, good_window,
                                         encode_error) &&
                   encode_mtp_attention(context, encoder, bad_attention,
                                        encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const auto ids_after = typed<std::int32_t>(fixture.mtp_ids);
    const bool unsubmitted =
        std::all_of(ids_after.begin(), ids_after.end(),
                    [](std::int32_t id) { return id == kIdSentinel; });
    check(rejected && discarded && unsubmitted,
          "MTP stage-13 late failure discards stage-12 work", error);
    command->discard();
    return rejected && discarded && unsubmitted;
}

bool test_concurrent_group_boundary(Context& context) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    Fixture fixture;
    if (!fixture.allocate(context, KvFormat::Int8, false,
                          "MTP attention group boundary")) {
        return false;
    }
    fixture.initialize(false, true);
    std::string error;
    auto command =
        context.make_command_buffer("MTP attention group boundary", error);
    if (!command) {
        check(false, "MTP attention group-boundary command buffer", error);
        return false;
    }
    const bool encoded = Encoder::encode(
        context, *command, "MTP attention group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_attention(
                context, encoder, fixture.attention_buffers(), stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP attention accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    check(encoded, "MTP attention rejects an open concurrent group", error);
    command->discard();
    return encoded;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_attention_test <strata.metallib>\n");
        return 2;
    }
    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;

    bool passed = true;
    passed &= run_case(*context, KvFormat::Fp16, false, false,
                       "MTP stage 13 fp16");
    passed &= run_case(*context, KvFormat::Int8, false, false,
                       "MTP stage 13 int8");
    passed &= run_case(*context, KvFormat::Int8, true, false,
                       "MTP stage 13 rotated int8");
    passed &= run_case(*context, KvFormat::Q4_0, true, false,
                       "MTP stage 13 rotated q4_0");
    passed &= run_case(*context, KvFormat::K8v4, true, true,
                       "MTP stages 12-13 K8v4-requested int8");
    passed &= run_case(*context, KvFormat::K8v4, false, false,
                       "MTP stage 13 K8v4-requested unrotated int8");
    passed &= run_case(*context, KvFormat::Int8, false, true,
                       "MTP stage 13 padded ID stride", 20u, 19, 17u);
    passed &= run_case(*context, KvFormat::Int8, false, true,
                       "MTP stage 13 multi-chunk merge", 128u, 99, 80u,
                       128u);
    passed &= test_validation(*context);
    passed &= test_late_rollback(*context);
    passed &= test_concurrent_group_boundary(*context);
    if (g_failures != 0) {
        std::fprintf(stderr, "metal_mtp_attention_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_attention_test: MTP stage 13 OK");
    return passed ? 0 : 1;
}
