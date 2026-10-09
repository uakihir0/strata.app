// Metal twin for MTP drafter stage 9 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
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
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <span>
#include <string>
#include <string_view>
#include <vector>

using namespace strata::metal;

namespace {

constexpr std::uint32_t kMaxRows = VERIFY_MAX_T;
constexpr std::uint32_t kCapacity = 16u;
constexpr std::uint32_t kRequestedCapacity = 13u;
constexpr std::uint32_t kHeadRowsPerToken = 2u;
constexpr std::uint32_t kHeadWidth = 256u;
constexpr std::uint8_t kCachePoison = 0x5au;
constexpr std::array<std::int32_t, kMaxRows> kPositions{
    3, 0, 14, 5, 12, 2, 9, 7};

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

std::vector<float> make_rope(std::uint32_t cells) {
    std::vector<float> result(static_cast<std::size_t>(cells) * 32u * 2u);
    for (std::uint32_t cell = 0u; cell < cells; ++cell) {
        for (std::uint32_t pair = 0u; pair < 32u; ++pair) {
            const float angle =
                0.009f * static_cast<float>((cell + 2u) * (pair + 5u));
            const std::size_t offset =
                (static_cast<std::size_t>(cell) * 32u + pair) * 2u;
            result[offset] = std::cos(angle);
            result[offset + 1u] = std::sin(angle);
        }
    }
    return result;
}

struct Fixture {
    twin::GuardedBuffer params;
    twin::GuardedBuffer kcur;
    twin::GuardedBuffer vcur;
    twin::GuardedBuffer k_norm;
    twin::GuardedBuffer rope;
    twin::GuardedBuffer kv_allocation;
    MtpKvLayout layout;
    MtpKvViews views;
    bool kv_rot = false;

    bool allocate(Context& context, KvFormat format, bool rotate,
                  const std::string& label) {
        kv_rot = rotate;
        std::string error;
        if (!make_mtp_kv_layout(kRequestedCapacity, format, layout, error)) {
            check(false, label + " canonical KV layout", error);
            return false;
        }
        const auto rope_values = make_rope(layout.cell_capacity);
        const std::array<std::pair<twin::GuardedBuffer*, std::pair<const char*,
                                                                  std::size_t>>,
                         5>
            buffers{{
                {&params, {"params", 256u}},
                {&kcur, {"kcur", static_cast<std::size_t>(kMaxRows) *
                                      kHeadRowsPerToken * kHeadWidth *
                                      sizeof(float)}},
                {&vcur, {"vcur", static_cast<std::size_t>(kMaxRows) *
                                      kHeadRowsPerToken * kHeadWidth *
                                      sizeof(float)}},
                {&k_norm, {"K norm", kHeadWidth * sizeof(float)}},
                {&rope, {"RoPE", rope_values.size() * sizeof(float)}},
            }};
        for (const auto& [buffer, metadata] : buffers) {
            if (!buffer->allocate(context, label + " " + metadata.first,
                                  metadata.second, error)) {
                check(false, label + " " + metadata.first + " allocation",
                      error);
                return false;
            }
        }
        if (!kv_allocation.allocate(context, label + " KV allocation",
                                    layout.allocation_bytes, error)) {
            check(false, label + " KV allocation", error);
            return false;
        }
        if (!bind_views(context, label)) return false;

        std::memcpy(rope.payload().data(), rope_values.data(),
                    rope_values.size() * sizeof(float));
        auto norm = typed<float>(k_norm);
        for (std::size_t i = 0u; i < norm.size(); ++i) {
            norm[i] = 0.65f + static_cast<float>(i % 23u) * 0.013f;
        }
        return reset(1u);
    }

    bool bind_views(Context& context, const std::string& label) {
        const BufferRef allocation = kv_allocation.view();
        const auto slice = [&](KvRegion region, const char* name,
                               BufferRef& result) {
            if (region.bytes == 0u) {
                result = {};
                return true;
            }
            result = allocation.slice(region.offset, region.bytes);
            if (!result || !context.buffers().owns(result)) {
                check(false, label + " " + name + " view");
                return false;
            }
            return true;
        };
        return slice(layout.k, "K cache", views.k) &&
               slice(layout.k_scale, "K scales", views.k_scale) &&
               slice(layout.v, "V cache", views.v) &&
               slice(layout.v_scale, "V scales", views.v_scale);
    }

    bool reset(std::uint32_t rows) {
        std::fill(params.payload().begin(), params.payload().end(), 0u);
        MtpParams block{};
        block.T = rows;
        for (std::uint32_t row = 0u; row < rows; ++row) {
            const std::int32_t position = kPositions[row];
            block.step[row] = StepRow{
                position, position + 1, position / 4, position + 1};
        }
        std::memcpy(params.payload().data(), &block, sizeof(block));

        auto keys = typed<float>(kcur);
        auto values = typed<float>(vcur);
        for (std::size_t i = 0u; i < keys.size(); ++i) {
            const auto centered =
                static_cast<std::int32_t>((i * 29u + i / 31u) % 251u) - 125;
            keys[i] = static_cast<float>(centered) * 0.0078125f;
            values[i] =
                static_cast<float>(static_cast<std::int32_t>(
                    (i * 17u + i / 19u + 61u) % 233u) -
                                  116) *
                0.009765625f;
        }
        std::fill(kv_allocation.payload().begin(),
                  kv_allocation.payload().end(), kCachePoison);
        return true;
    }

    MtpAttentionKvAppendBuffers mtp_buffers() {
        return {params.view(), kcur.view(), vcur.view(), k_norm.view(),
                rope.view(), layout, views, kv_rot};
    }

    void capture() {
        params.capture_before_dispatch();
        kcur.capture_before_dispatch();
        vcur.capture_before_dispatch();
        k_norm.capture_before_dispatch();
        rope.capture_before_dispatch();
        kv_allocation.capture_before_dispatch();
    }

    bool inputs_unchanged() const {
        return !params.first_unwritten_change({}) &&
               !k_norm.first_unwritten_change({}) &&
               !rope.first_unwritten_change({});
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

std::vector<std::uint8_t> expected_cache_bytes(
    const Fixture& fixture, std::span<const float> keys,
    std::span<const float> values, std::span<const std::int32_t> positions) {
    std::vector<std::uint8_t> expected(fixture.layout.allocation_bytes,
                                       kCachePoison);
    const std::size_t pool_rows =
        static_cast<std::size_t>(fixture.layout.cell_capacity) *
        kHeadRowsPerToken;
    if (fixture.layout.storage_format == KvFormat::Fp16) {
        std::vector<std::uint16_t> key_pool(pool_rows * kHeadWidth,
                                             0x5a5au);
        std::vector<std::uint16_t> value_pool(pool_rows * kHeadWidth,
                                               0x5a5au);
        oracle::qsa_kv_append_f16(keys, values, positions, key_pool,
                                  value_pool);
        std::memcpy(expected.data() + fixture.layout.k.offset,
                    key_pool.data(), fixture.layout.k.bytes);
        std::memcpy(expected.data() + fixture.layout.v.offset,
                    value_pool.data(), fixture.layout.v.bytes);
    } else if (fixture.layout.storage_format == KvFormat::Int8) {
        std::vector<std::int8_t> key_codes(fixture.layout.k.bytes,
                                           static_cast<std::int8_t>(
                                               kCachePoison));
        std::vector<std::int8_t> value_codes(fixture.layout.v.bytes,
                                             static_cast<std::int8_t>(
                                                 kCachePoison));
        std::vector<std::uint16_t> key_scales(
            fixture.layout.k_scale.bytes / sizeof(std::uint16_t), 0x5a5au);
        std::vector<std::uint16_t> value_scales(
            fixture.layout.v_scale.bytes / sizeof(std::uint16_t), 0x5a5au);
        oracle::qsa_kv_append_i8(keys, values, positions, key_codes,
                                 key_scales, value_codes, value_scales);
        std::memcpy(expected.data() + fixture.layout.k.offset,
                    key_codes.data(), fixture.layout.k.bytes);
        std::memcpy(expected.data() + fixture.layout.k_scale.offset,
                    key_scales.data(), fixture.layout.k_scale.bytes);
        std::memcpy(expected.data() + fixture.layout.v.offset,
                    value_codes.data(), fixture.layout.v.bytes);
        std::memcpy(expected.data() + fixture.layout.v_scale.offset,
                    value_scales.data(), fixture.layout.v_scale.bytes);
    } else {
        std::vector<std::uint8_t> key_pool(fixture.layout.k.bytes,
                                           kCachePoison);
        std::vector<std::uint8_t> value_pool(fixture.layout.v.bytes,
                                             kCachePoison);
        oracle::qsa_kv_append_q4(keys, values, positions, key_pool,
                                 value_pool);
        std::memcpy(expected.data() + fixture.layout.k.offset,
                    key_pool.data(), fixture.layout.k.bytes);
        std::memcpy(expected.data() + fixture.layout.v.offset,
                    value_pool.data(), fixture.layout.v.bytes);
    }
    return expected;
}

bool execute_rows(Context& context, Fixture& fixture, std::uint32_t rows,
                  const std::string& mode_name) {
    fixture.reset(rows);
    std::vector<float> original_keys(
        typed<float>(fixture.kcur).begin(),
        typed<float>(fixture.kcur).begin() +
            static_cast<std::ptrdiff_t>(rows) * kHeadRowsPerToken *
                kHeadWidth);
    std::vector<float> original_values(
        typed<float>(fixture.vcur).begin(),
        typed<float>(fixture.vcur).begin() +
            static_cast<std::ptrdiff_t>(rows) * kHeadRowsPerToken *
                kHeadWidth);
    MtpParams params{};
    std::memcpy(&params, fixture.params.payload().data(), sizeof(params));
    std::vector<std::int32_t> positions;
    positions.reserve(rows);
    for (std::uint32_t row = 0u; row < rows; ++row) {
        positions.push_back(params.step[row].pos);
    }
    const auto norm = std::span<const float>(typed<float>(fixture.k_norm))
                          .first(kHeadWidth);
    const auto rope = std::span<const float>(typed<float>(fixture.rope));
    std::vector<twin::tf> normalized_keys;
    oracle::qsa_norm_rope(original_keys, norm, positions, rope, rows,
                          QsaNormRopeMode::key, normalized_keys);
    std::vector<float> expected_keys(normalized_keys.size());
    std::transform(normalized_keys.begin(), normalized_keys.end(),
                   expected_keys.begin(),
                   [](twin::tf value) { return value.value; });
    std::vector<float> expected_values = original_values;
    if (fixture.kv_rot) {
        std::vector<twin::tf> rotated_keys;
        std::vector<twin::tf> rotated_values;
        oracle::qsa_fwht256(expected_keys, rows * kHeadRowsPerToken,
                            rotated_keys);
        oracle::qsa_fwht256(expected_values, rows * kHeadRowsPerToken,
                            rotated_values);
        std::transform(rotated_keys.begin(), rotated_keys.end(),
                       expected_keys.begin(),
                       [](twin::tf value) { return value.value; });
        std::transform(rotated_values.begin(), rotated_values.end(),
                       expected_values.begin(),
                       [](twin::tf value) { return value.value; });
    }

    fixture.capture();
    const std::string label =
        "MTP stage-9 " + mode_name + " T=" + std::to_string(rows);
    std::string error;
    auto command = context.make_command_buffer(label, error);
    if (!command) {
        check(false, label + " command buffer", error);
        return false;
    }
    bool encoded = false;
    if (rows == kMaxRows) {
        encoded = Encoder::encode(
            context, *command, label + " caller-owned",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_attention_kv_append(
                    context, encoder, fixture.mtp_buffers(), encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_attention_kv_append(
            context, *command, fixture.mtp_buffers(), error);
    }
    check(encoded, label + " encode", error);
    if (!encoded) return false;
    const bool completed = submit_and_wait(command, label, error);
    check(completed, label + " GPU completion", error);
    if (!completed) return false;

    const std::size_t active_values =
        static_cast<std::size_t>(rows) * kHeadRowsPerToken * kHeadWidth;
    const auto actual_keys =
        std::span<const float>(typed<float>(fixture.kcur)).first(active_values);
    const auto actual_values =
        std::span<const float>(typed<float>(fixture.vcur)).first(active_values);
    std::vector<twin::tf> expected_key_tf;
    std::vector<twin::tf> expected_value_tf;
    expected_key_tf.reserve(expected_keys.size());
    expected_value_tf.reserve(expected_values.size());
    for (float value : expected_keys) expected_key_tf.emplace_back(value);
    for (float value : expected_values) expected_value_tf.emplace_back(value);
    const auto key_mismatch = twin::compare_floats(
        actual_keys, expected_key_tf,
        {twin::Comparison::units, 16.0, 0.0, 0u});
    const auto value_mismatch = twin::compare_floats(
        actual_values, expected_value_tf,
        {twin::Comparison::units, 16.0, 0.0, 0u});
    check(!key_mismatch, label + " K norm/RoPE and optional FWHT oracle");
    check(!value_mismatch,
          label + (fixture.kv_rot ? " V FWHT oracle" : " V preserved"));

    const auto expected_cache = expected_cache_bytes(
        fixture, actual_keys, actual_values, positions);
    const bool cache_exact =
        std::memcmp(fixture.kv_allocation.payload().data(),
                    expected_cache.data(), expected_cache.size()) == 0;
    check(cache_exact, label + " full K/V cache matches scalar append oracle");
    const std::array<twin::ByteRange, 1> k_written{{
        {0u, active_values * sizeof(float)},
    }};
    const std::array<twin::ByteRange, 1> v_written{{
        {0u, fixture.kv_rot ? active_values * sizeof(float) : 0u},
    }};
    const std::array<twin::ByteRange, 4> cache_written{{
        {fixture.layout.k.offset, fixture.layout.k.bytes},
        {fixture.layout.k_scale.offset, fixture.layout.k_scale.bytes},
        {fixture.layout.v.offset, fixture.layout.v.bytes},
        {fixture.layout.v_scale.offset, fixture.layout.v_scale.bytes},
    }};
    const bool guards_ok =
        fixture.inputs_unchanged() &&
        !fixture.kcur.first_unwritten_change(k_written) &&
        !fixture.vcur.first_unwritten_change(
            fixture.kv_rot ? std::span<const twin::ByteRange>(v_written)
                           : std::span<const twin::ByteRange>{}) &&
        !fixture.kv_allocation.first_unwritten_change(cache_written);
    check(guards_ok, label + " input, inactive-row and allocation guards");
    return !key_mismatch && !value_mismatch && cache_exact && guards_ok;
}

bool reject_case(Context& context,
                 const MtpAttentionKvAppendBuffers& buffers,
                 const std::string& label,
                 const std::string& expected_error = {}) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_attention_kv_append(context, *command, buffers, error);
    const bool expected =
        expected_error.empty() ||
        error.find(expected_error) != std::string::npos;
    check(rejected && expected, label, error);
    command->discard();
    return rejected && expected;
}

bool test_validation(Context& context, Context& foreign_context,
                     Fixture& fixture) {
    bool passed = true;
    fixture.reset(1u);
    auto buffers = fixture.mtp_buffers();

    MtpParams invalid{};
    std::memcpy(fixture.params.payload().data(), &invalid, sizeof(invalid));
    passed &= reject_case(context, buffers, "MTP stage 9 rejects T=0",
                          "T must be in 1..8");

    fixture.reset(2u);
    auto* params =
        reinterpret_cast<MtpParams*>(fixture.params.payload().data());
    params->step[1].pos = params->step[0].pos;
    buffers = fixture.mtp_buffers();
    passed &= reject_case(context, buffers,
                          "MTP stage 9 rejects duplicate KV positions",
                          "positions must be unique");

    fixture.reset(1u);
    params = reinterpret_cast<MtpParams*>(fixture.params.payload().data());
    params->step[0].pos = static_cast<std::int32_t>(kCapacity);
    buffers = fixture.mtp_buffers();
    passed &= reject_case(context, buffers,
                          "MTP stage 9 rejects a position outside capacity",
                          "outside the KV capacity");

    fixture.reset(1u);
    buffers = fixture.mtp_buffers();
    buffers.kv_layout.cell_capacity -= 4u;
    passed &= reject_case(context, buffers,
                          "MTP stage 9 rejects a forged KV layout",
                          "not canonical");

    buffers = fixture.mtp_buffers();
    buffers.kv_views.v =
        buffers.kv_views.v.slice(0u, buffers.kv_views.v.length - 1u);
    passed &= reject_case(context, buffers,
                          "MTP stage 9 rejects a short cache view",
                          "buffer is undersized");

    buffers = fixture.mtp_buffers();
    buffers.kv_rot = true;
    passed &= reject_case(context, buffers,
                          "MTP stage 9 rejects rotation for fp16 KV",
                          "kv_rot does not match");

    Fixture q4_fixture;
    const bool q4_allocated =
        q4_fixture.allocate(context, KvFormat::Q4_0, false,
                            "MTP stage-9 invalid q4");
    check(q4_allocated, "allocate MTP stage-9 q4 validation fixture");
    if (q4_allocated) {
        passed &= reject_case(
            context, q4_fixture.mtp_buffers(),
            "MTP stage 9 requires q4 rotation",
            "kv_rot does not match");
    } else {
        passed = false;
    }

    Fixture foreign_fixture;
    const bool foreign_allocated =
        foreign_fixture.allocate(foreign_context, KvFormat::Fp16, false,
                                 "foreign MTP stage-9 cache");
    check(foreign_allocated, "allocate foreign MTP stage-9 cache");
    if (foreign_allocated) {
        auto foreign_views = foreign_fixture.mtp_buffers();
        auto local_buffers = fixture.mtp_buffers();
        local_buffers.kv_views = foreign_views.kv_views;
        passed &= reject_case(
            context, local_buffers,
            "MTP stage 9 rejects a foreign cache allocation",
            "not owned by this Context");
    } else {
        passed = false;
    }
    return passed;
}

bool test_concurrent_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    fixture.reset(1u);
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP stage-9 group boundary", error);
    if (!command) {
        check(false, "MTP stage-9 group-boundary command buffer", error);
        return false;
    }
    const auto buffers = fixture.mtp_buffers();
    const bool wrapper_checked = Encoder::encode(
        context, *command, "MTP stage-9 group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool rejected = !encode_mtp_attention_kv_append(
                context, encoder, buffers, stage_error);
            if (!encoder.group_end(encode_error)) return false;
            if (!rejected ||
                stage_error.find("ordered concurrent-group boundary") ==
                    std::string::npos) {
                encode_error =
                    "MTP stage 9 accepted an open concurrent group";
                return false;
            }
            return true;
        },
        error);
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.kcur.first_unwritten_change({}) &&
        !fixture.vcur.first_unwritten_change({}) &&
        !fixture.kv_allocation.first_unwritten_change({});
    check(wrapper_checked, "MTP stage 9 rejects an open concurrent group",
          error);
    check(unchanged,
          "MTP stage-9 open-group rejection preserves all fixture bytes");
    command->discard();
    return wrapper_checked && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.reset(2u);
    fixture.capture();
    const auto good = fixture.mtp_buffers();
    auto bad = good;
    bad.kv_rot = true;  // Invalid for fp16, after the valid segment was encoded.

    std::string error;
    auto command = context.make_command_buffer("MTP stage-9 rollback", error);
    if (!command) {
        check(false, "MTP stage-9 rollback command buffer", error);
        return false;
    }
    const bool rejected = !Encoder::encode(
        context, *command, "MTP stage-9 caller-owned segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_attention_kv_append(context, encoder, good,
                                                encode_error)) {
                return false;
            }
            return encode_mtp_attention_kv_append(context, encoder, bad,
                                                  encode_error);
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged =
        !fixture.params.first_unwritten_change({}) &&
        !fixture.kcur.first_unwritten_change({}) &&
        !fixture.vcur.first_unwritten_change({}) &&
        !fixture.kv_allocation.first_unwritten_change({});
    check(rejected && discarded,
          "MTP stage-9 late failure discards the caller-owned segment", error);
    check(unchanged, "MTP stage-9 rollback preserves all fixture bytes");
    return rejected && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_kv_append_test <strata.metallib>\n");
        return 2;
    }

    ContextOptions options;
    options.metallib = argv[1];
    options.metallib_sha256 = STRATA_METALLIB_SHA256;
    std::string error;
    auto context = Context::open(options, error);
    check(context != nullptr, "open Metal context", error);
    if (!context) return 1;
    ContextOptions foreign_options = options;
    foreign_options.pipelines = false;
    auto foreign_context = Context::open(foreign_options, error);
    check(foreign_context != nullptr, "open foreign Metal context", error);
    if (!foreign_context) return 1;

    const std::array<std::pair<KvFormat, bool>, 5> modes{{
        {KvFormat::Fp16, false},
        {KvFormat::Int8, false},
        {KvFormat::Int8, true},
        {KvFormat::Q4_0, true},
        {KvFormat::K8v4, true},
    }};
    bool passed = true;
    for (std::size_t index = 0u; index < modes.size(); ++index) {
        const auto [format, rotate] = modes[index];
        const std::string mode =
            "format=" + std::to_string(static_cast<unsigned>(format)) +
            (rotate ? " rotated" : "");
        Fixture fixture;
        const bool allocated = fixture.allocate(
            *context, format, rotate, "MTP stage-9 " + mode);
        check(allocated, "allocate MTP stage-9 " + mode + " fixture");
        if (!allocated) {
            passed = false;
            continue;
        }
        for (std::uint32_t rows = 1u; rows <= kMaxRows; ++rows) {
            passed &= execute_rows(*context, fixture, rows, mode);
        }
        if (index == 0u) {
            passed &= test_validation(*context, *foreign_context, fixture);
            passed &= test_concurrent_group_boundary(*context, fixture);
            passed &= test_late_rollback(*context, fixture);
        }
    }
    if (g_failures != 0 || !passed) {
        std::fprintf(stderr, "metal_mtp_kv_append_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_kv_append_test: MTP stage 9 OK");
    return 0;
}
