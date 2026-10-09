// Metal twin for MTP drafter stage 17 (docs/mac/04 §5.11).
#include "metallib_sha256.inc"
#include "strata/kernels/f16_bits.hpp"
#include "strata/metal/context.hpp"
#include "strata/metal/encoder.hpp"
#include "strata/metal/mtp.hpp"
#include "strata/metal/mtp_dispatch.hpp"
#include "twin.hpp"

#import <Metal/Metal.h>

#include "../../src/metal/kernels/common/params.h"
#include "../../src/metal/kernels/common/types.h"

#include <algorithm>
#include <array>
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

constexpr std::uint32_t kTopK = 10u;
constexpr std::array<std::int32_t, kTopK> kRouteIds{
    0, 1, 2, 3, 4, 5, 6, 7, 8, 511};
constexpr std::uint32_t kExpertInputWidth = 2'560u;
constexpr std::uint32_t kExpertHiddenWidth = 640u;
constexpr std::uint32_t kQuantBlock = 32u;
constexpr std::size_t kExpertBlobBytes = 1'382'400u;
constexpr std::size_t kExpertGateUpCodesOffset = 0u;
constexpr std::size_t kExpertDownCodesOffset = 819'200u;
constexpr std::size_t kExpertGateUpScalesOffset = 1'228'800u;
constexpr std::size_t kExpertDownScalesOffset = 1'331'200u;
constexpr std::size_t kMixedBytes =
    static_cast<std::size_t>(kExpertInputWidth) * sizeof(float);
constexpr std::size_t kIdsBytes = kTopK * sizeof(std::int32_t);
constexpr std::size_t kMixedQ8Bytes = kExpertInputWidth;
constexpr std::size_t kMixedScaleBytes =
    kExpertInputWidth / kQuantBlock * sizeof(float);
constexpr std::size_t kHiddenValues =
    static_cast<std::size_t>(kTopK) * kExpertHiddenWidth;
constexpr std::size_t kHiddenBytes = kHiddenValues * sizeof(float);
constexpr std::size_t kHiddenQ8Bytes = kHiddenValues;
constexpr std::size_t kHiddenScaleBytes =
    kHiddenValues / kQuantBlock * sizeof(float);
constexpr std::size_t kPartsBytes =
    static_cast<std::size_t>(kTopK) * kExpertInputWidth * sizeof(float);

enum OutputIndex : std::size_t {
    kMixedQ8,
    kMixedScales,
    kGate,
    kUp,
    kSwiGlu,
    kHiddenQ8,
    kHiddenScales,
    kParts,
    kOutputCount,
};

constexpr std::array<const char*, kOutputCount> kOutputNames{
    "mixed Q8", "mixed scales", "gate", "up",
    "SwiGLU", "hidden Q8", "hidden scales", "down parts"};
constexpr std::array<std::size_t, kOutputCount> kOutputSizes{
    kMixedQ8Bytes, kMixedScaleBytes, kHiddenBytes, kHiddenBytes,
    kHiddenBytes, kHiddenQ8Bytes, kHiddenScaleBytes, kPartsBytes};

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

float half_value(const std::uint8_t* address) {
    std::uint16_t bits = 0u;
    std::memcpy(&bits, address, sizeof(bits));
    return strata::kernels::f32_from_f16(bits);
}

void set_half(std::uint8_t* address, float value) {
    const std::uint16_t bits = strata::kernels::f16_from_f32(value);
    std::memcpy(address, &bits, sizeof(bits));
}

void quantize_q8_scaled(std::span<const float> input,
                        std::vector<std::int8_t>& codes,
                        std::vector<float>& scales) {
    codes.resize(input.size());
    scales.resize(input.size() / kQuantBlock);
    for (std::size_t block = 0u; block < scales.size(); ++block) {
        const std::size_t first = block * kQuantBlock;
        float amax = 0.0f;
        for (std::size_t index = 0u; index < kQuantBlock; ++index) {
            amax = std::max(amax, std::fabs(input[first + index]));
        }
        const float scale = amax > 0.0f ? amax / 127.0f : 0.0f;
        const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
        scales[block] = scale;
        for (std::size_t index = 0u; index < kQuantBlock; ++index) {
            const float value = input[first + index] * inverse;
            const float rounded = value + (value >= 0.0f ? 0.5f : -0.5f);
            const int quantized =
                std::clamp(static_cast<int>(rounded), -127, 127);
            codes[first + index] = static_cast<std::int8_t>(quantized);
        }
    }
}

float butterfly_sum(std::array<float, 32u> lanes) {
    for (std::size_t offset = 16u; offset != 0u; offset >>= 1u) {
        const auto previous = lanes;
        for (std::size_t lane = 0u; lane < lanes.size(); ++lane) {
            lanes[lane] = previous[lane] + previous[lane ^ offset];
        }
    }
    return lanes[0];
}

int s2_chunk_dot(const std::uint8_t* packed_codes,
                 const std::int8_t* activation_codes) {
    int dot = 0;
    int activation_sum = 0;
    for (std::size_t word = 0u; word < 8u; ++word) {
        const std::uint8_t packed = packed_codes[word];
        for (std::size_t code_index = 0u; code_index < 4u; ++code_index) {
            const int code =
                static_cast<int>((packed >> (2u * code_index)) & 0x03u);
            const int activation =
                static_cast<int>(activation_codes[4u * word + code_index]);
            dot += code * activation;
            activation_sum += activation;
        }
    }
    return dot - activation_sum;
}

struct Oracle {
    std::vector<std::int8_t> mixed_q8;
    std::vector<float> mixed_scales;
    std::vector<float> gate;
    std::vector<float> up;
    std::vector<float> swiglu;
    std::vector<std::int8_t> hidden_q8;
    std::vector<float> hidden_scales;
    std::vector<float> parts;

    void build(std::span<const float> mixed,
               const std::array<std::vector<std::uint8_t>, kTopK>& expert_blobs) {
        quantize_q8_scaled(mixed, mixed_q8, mixed_scales);
        gate.assign(kHiddenValues, 0.0f);
        up.assign(kHiddenValues, 0.0f);
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            const std::uint8_t* expert_blob = expert_blobs[rank].data();
            for (std::uint32_t slot = 0u;
                 slot < 2u * kExpertHiddenWidth; ++slot) {
                std::array<float, 32u> lanes{};
                for (std::uint32_t lane = 0u; lane < 32u; ++lane) {
                    for (std::uint32_t chunk = lane; chunk < 80u;
                         chunk += 32u) {
                        const std::size_t codes_offset =
                            kExpertGateUpCodesOffset +
                            static_cast<std::size_t>(slot) * 640u +
                            static_cast<std::size_t>(chunk) * 8u;
                        const int dot = s2_chunk_dot(
                            expert_blob + codes_offset,
                            mixed_q8.data() + chunk * kQuantBlock);
                        const std::size_t scale_offset =
                            kExpertGateUpScalesOffset +
                            static_cast<std::size_t>(slot) * 80u +
                            2u * (chunk >> 1u);
                        const float weight_scale =
                            half_value(expert_blob + scale_offset);
                        lanes[lane] = std::fma(
                            weight_scale * mixed_scales[chunk],
                            static_cast<float>(dot), lanes[lane]);
                    }
                }
                const float value = butterfly_sum(lanes);
                const std::size_t index =
                    rank * kExpertHiddenWidth + slot / 2u;
                (slot & 1u ? up : gate)[index] = value;
            }
        }

        swiglu.resize(kHiddenValues);
        for (std::size_t index = 0u; index < kHiddenValues; ++index) {
            const float sigmoid =
                1.0f / (1.0f + std::exp(-gate[index]));
            swiglu[index] = (gate[index] * sigmoid) * up[index];
        }
        quantize_q8_scaled(swiglu, hidden_q8, hidden_scales);

        parts.assign(static_cast<std::size_t>(kTopK) * kExpertInputWidth,
                     0.0f);
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            const std::uint8_t* expert_blob = expert_blobs[rank].data();
            for (std::uint32_t row = 0u; row < kExpertInputWidth; ++row) {
                std::array<float, 32u> lanes{};
                for (std::uint32_t lane = 0u; lane < 20u; ++lane) {
                    const std::size_t codes_offset =
                        kExpertDownCodesOffset +
                        static_cast<std::size_t>(row) * 160u +
                        static_cast<std::size_t>(lane) * 8u;
                    const int dot = s2_chunk_dot(
                        expert_blob + codes_offset,
                        hidden_q8.data() +
                            rank * kExpertHiddenWidth +
                            lane * kQuantBlock);
                    const std::size_t scale_offset =
                        kExpertDownScalesOffset +
                        static_cast<std::size_t>(row) * 20u +
                        2u * (lane >> 1u);
                    const float weight_scale =
                        half_value(expert_blob + scale_offset);
                    lanes[lane] = std::fma(
                        weight_scale *
                            hidden_scales[rank * 20u + lane],
                        static_cast<float>(dot), 0.0f);
                }
                parts[rank * kExpertInputWidth + row] =
                    butterfly_sum(lanes);
            }
        }
    }
};

struct Fixture {
    twin::GuardedBuffer mixed;
    twin::GuardedBuffer ids;
    std::array<twin::GuardedBuffer, kOutputCount> outputs;
    BufferRef expert_blob;
    std::vector<BufferRef> expert_window_buffers;
    std::vector<MtpExpertBlobWindow> expert_windows;
    std::array<std::vector<std::uint8_t>, kTopK> selected_expert_data;
    std::vector<std::uint8_t> selected_experts_before;
    Oracle oracle;

    BufferRef output(OutputIndex index) { return outputs[index].view(); }

    BufferRef expert_view(std::uint32_t expert_id) const {
        if (expert_id >= kMtpExpertCount) return {};
        if (expert_blob) {
            return expert_blob.slice(
                static_cast<std::size_t>(expert_id) * kExpertBlobBytes,
                kExpertBlobBytes);
        }
        for (std::size_t index = 0u; index < expert_windows.size(); ++index) {
            const MtpExpertBlobWindow& window = expert_windows[index];
            if (expert_id >= window.first_expert &&
                expert_id - window.first_expert < window.expert_count) {
                const std::size_t local =
                    static_cast<std::size_t>(expert_id - window.first_expert);
                return expert_window_buffers[index].slice(
                    local * kExpertBlobBytes, kExpertBlobBytes);
            }
        }
        return {};
    }

    MtpRoutedExpertsBuffers buffers(Context& context) {
        MtpRoutedExpertsBuffers result{
            context.buffers().read_only(mixed.view()),
            context.buffers().read_only(ids.view()),
            context.buffers().read_only(expert_blob),
            output(kMixedQ8),
            output(kMixedScales),
            output(kGate),
            output(kUp),
            output(kSwiGlu),
            output(kHiddenQ8),
            output(kHiddenScales),
            output(kParts),
            {}};
        if (!expert_blob) result.expert_windows = expert_windows;
        return result;
    }

    MtpRoutedExpertsBuffers windowed_buffers(Context& context) {
        MtpRoutedExpertsBuffers result = buffers(context);
        if (!expert_blob) return result;
        result.expert_blob = {};
        constexpr std::array<std::uint32_t, 4> boundaries{0u, 4u, 9u, 512u};
        for (std::size_t index = 0u; index + 1u < boundaries.size(); ++index) {
            const std::uint32_t first = boundaries[index];
            const std::uint32_t count = boundaries[index + 1u] - first;
            const std::size_t offset =
                static_cast<std::size_t>(first) * kExpertBlobBytes;
            const std::size_t bytes =
                static_cast<std::size_t>(count) * kExpertBlobBytes;
            const BufferRef window = expert_blob.slice(offset, bytes);
            result.expert_windows.push_back(
                {context.buffers().read_only(window), first, count});
        }
        return result;
    }

    bool allocate(Context& context) {
        std::string error;
        const std::uint64_t max_buffer_length =
            context.device_facts().max_buffer_length;
        if (max_buffer_length < kExpertBlobBytes) {
            check(false, "one S2 expert blob fits maxBufferLength",
                  "the device cannot hold even one complete MTP expert");
            return false;
        }
        if (!mixed.allocate(context, "MTP stage 17 mixed", kMixedBytes,
                            error) ||
            !ids.allocate(context, "MTP stage 17 route IDs", kIdsBytes,
                          error)) {
            check(false, "MTP stage 17 input allocation", error);
            return false;
        }
        constexpr std::array<std::size_t, kOutputCount> output_sizes{
            kMixedQ8Bytes, kMixedScaleBytes, kHiddenBytes, kHiddenBytes,
            kHiddenBytes, kHiddenQ8Bytes, kHiddenScaleBytes, kPartsBytes};
        for (std::size_t index = 0u; index < outputs.size(); ++index) {
            if (!outputs[index].allocate(context,
                                         "MTP stage 17 " + std::string(kOutputNames[index]),
                                         output_sizes[index], error)) {
                check(false, "MTP stage 17 output allocation", error);
                return false;
            }
        }
        if (max_buffer_length >= kMtpExpertsFileBytes) {
            expert_blob = context.buffers().allocate(
                "MTP stage 17 complete S2 expert image",
                static_cast<std::size_t>(kMtpExpertsFileBytes), error);
            if (!expert_blob || expert_blob.contents == nullptr) {
                check(false, "MTP stage 17 complete expert-image allocation",
                      error.empty() ? "CPU-visible shared mapping unavailable"
                                    : error);
                return false;
            }
        } else {
            const std::uint64_t experts_per_window =
                max_buffer_length / kExpertBlobBytes;
            for (std::uint32_t first = 0u; first < kMtpExpertCount;) {
                const std::uint32_t count = static_cast<std::uint32_t>(
                    std::min<std::uint64_t>(
                        experts_per_window, kMtpExpertCount - first));
                const std::size_t bytes =
                    static_cast<std::size_t>(count) * kExpertBlobBytes;
                BufferRef window = context.buffers().allocate(
                    "MTP stage 17 expert window " +
                        std::to_string(expert_window_buffers.size()),
                    bytes, error);
                if (!window || window.contents == nullptr) {
                    check(false, "MTP stage 17 expert-window allocation",
                          error.empty()
                              ? "CPU-visible shared mapping unavailable"
                              : error);
                    return false;
                }
                expert_window_buffers.push_back(window);
                expert_windows.push_back(
                    {context.buffers().read_only(window), first, count});
                first += count;
            }
        }

        std::span<float> mixed_values = typed<float>(mixed);
        for (std::size_t index = 0u; index < mixed_values.size(); ++index) {
            const std::int32_t centered =
                static_cast<std::int32_t>((index * 31u + 11u) % 251u) - 125;
            mixed_values[index] = static_cast<float>(centered) * 0.004f;
        }
        auto route_ids = typed<std::int32_t>(ids);
        for (std::size_t index = 0u; index < route_ids.size(); ++index) {
            route_ids[index] = kRouteIds[index];
        }

        selected_experts_before.resize(kTopK * kExpertBlobBytes);
        for (std::uint32_t expert = 0u; expert < kTopK; ++expert) {
            const std::uint32_t expert_id =
                static_cast<std::uint32_t>(kRouteIds[expert]);
            std::vector<std::uint8_t>& expert_data =
                selected_expert_data[expert];
            expert_data.resize(kExpertBlobBytes);
            for (std::size_t index = 0u; index < kExpertBlobBytes; ++index) {
                expert_data[index] = static_cast<std::uint8_t>(
                    (index * 13u + expert_id * 29u + index / 640u * 3u) &
                    0xffu);
            }
            for (std::uint32_t slot = 0u;
                 slot < 2u * kExpertHiddenWidth; ++slot) {
                for (std::uint32_t pair = 0u; pair < 40u; ++pair) {
                    const float scale =
                        0.012f + static_cast<float>(expert_id % 7u) * 0.0007f +
                        static_cast<float>(slot % 13u) * 0.00021f +
                        static_cast<float>(pair % 3u) * 0.00008f;
                    set_half(expert_data.data() + kExpertGateUpScalesOffset +
                                 static_cast<std::size_t>(slot) * 80u +
                                 static_cast<std::size_t>(pair) * 2u,
                             scale);
                }
            }
            for (std::uint32_t row = 0u; row < kExpertInputWidth; ++row) {
                for (std::uint32_t pair = 0u; pair < 10u; ++pair) {
                    const float scale =
                        0.004f + static_cast<float>(expert_id % 7u) * 0.0003f +
                        static_cast<float>(row % 9u) * 0.00007f +
                        static_cast<float>(pair % 2u) * 0.00004f;
                    set_half(expert_data.data() + kExpertDownScalesOffset +
                                 static_cast<std::size_t>(row) * 20u +
                                 static_cast<std::size_t>(pair) * 2u,
                             scale);
                }
            }
            std::memcpy(selected_experts_before.data() +
                            static_cast<std::size_t>(expert) *
                                kExpertBlobBytes,
                        expert_data.data(), kExpertBlobBytes);
            BufferRef expert_gpu_view = expert_view(expert_id);
            if (!expert_gpu_view || expert_gpu_view.contents == nullptr) {
                check(false, "MTP stage-17 selected expert window",
                      "selected expert is outside runtime test windows");
                return false;
            }
            std::memcpy(expert_gpu_view.contents, expert_data.data(),
                        kExpertBlobBytes);
        }

        oracle.build(mixed_values, selected_expert_data);
        reset_outputs();
        return true;
    }

    void reset_outputs() {
        for (auto& buffer : outputs) {
            std::memset(buffer.payload().data(), 0xa5, buffer.payload().size());
        }
    }

    void capture() {
        mixed.capture_before_dispatch();
        ids.capture_before_dispatch();
        for (auto& buffer : outputs) buffer.capture_before_dispatch();
    }

    std::array<std::vector<std::uint8_t>, kOutputCount> snapshot() const {
        std::array<std::vector<std::uint8_t>, kOutputCount> result;
        for (std::size_t index = 0u; index < outputs.size(); ++index) {
            const auto bytes = outputs[index].payload();
            result[index].assign(bytes.begin(), bytes.end());
        }
        return result;
    }

    bool inputs_unchanged() const {
        if (mixed.first_unwritten_change({}) || ids.first_unwritten_change({})) {
            return false;
        }
        for (std::size_t rank = 0u; rank < kTopK; ++rank) {
            const BufferRef expert_gpu_view =
                expert_view(static_cast<std::uint32_t>(kRouteIds[rank]));
            const std::size_t snapshot_base = rank * kExpertBlobBytes;
            if (!expert_gpu_view || expert_gpu_view.contents == nullptr ||
                std::memcmp(expert_gpu_view.contents,
                            selected_experts_before.data() + snapshot_base,
                            kExpertBlobBytes) != 0) {
                return false;
            }
        }
        return true;
    }

    bool output_guards_unchanged() const {
        for (std::size_t index = 0u; index < outputs.size(); ++index) {
            const std::array<twin::ByteRange, 1u> written{{
                {0u, kOutputSizes[index]},
            }};
            if (outputs[index].first_unwritten_change(written)) return false;
        }
        return true;
    }

    bool all_buffers_unchanged() const {
        if (!inputs_unchanged()) return false;
        return std::all_of(outputs.begin(), outputs.end(),
                           [](const twin::GuardedBuffer& buffer) {
                               return !buffer.first_unwritten_change({});
                           });
    }
};

std::span<const std::uint8_t> output_bytes(const Fixture& fixture,
                                          OutputIndex index) {
    return fixture.outputs[index].payload();
}

std::span<const float> output_floats(const Fixture& fixture,
                                    OutputIndex index) {
    const auto bytes = output_bytes(fixture, index);
    return {reinterpret_cast<const float*>(bytes.data()),
            bytes.size() / sizeof(float)};
}

bool compare_exact_floats(std::span<const float> actual,
                          std::span<const float> expected,
                          const std::string& label) {
    if (actual.size() != expected.size()) {
        check(false, label, "float range length mismatch");
        return false;
    }
    std::size_t mismatches = 0u;
    std::size_t first = 0u;
    double max_abs_error = 0.0;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        if (std::memcmp(&actual[index], &expected[index], sizeof(float)) != 0) {
            if (mismatches == 0u) first = index;
            ++mismatches;
            max_abs_error = std::max(
                max_abs_error,
                std::fabs(static_cast<double>(actual[index]) -
                          static_cast<double>(expected[index])));
        }
    }
    if (mismatches != 0u) {
        char detail[192]{};
        std::snprintf(detail, sizeof(detail),
                      "%zu bit mismatches; first=%zu actual=%.9g expected=%.9g max_abs=%.9g",
                      mismatches, first, actual[first], expected[first],
                      max_abs_error);
        check(false, label, detail);
        return false;
    }
    check(true, label);
    return true;
}

bool compare_close_floats(std::span<const float> actual,
                          std::span<const float> expected,
                          const std::string& label,
                          double absolute_tolerance = 2.0e-6,
                          double relative_tolerance = 3.0e-5) {
    if (actual.size() != expected.size()) {
        check(false, label, "float range length mismatch");
        return false;
    }
    std::size_t violations = 0u;
    std::size_t first = 0u;
    double worst_ratio = 0.0;
    for (std::size_t index = 0u; index < actual.size(); ++index) {
        const double want = expected[index];
        const double error = std::fabs(static_cast<double>(actual[index]) -
                                       want);
        const double bound =
            absolute_tolerance + relative_tolerance * std::fabs(want);
        if (error > bound) {
            if (violations == 0u) first = index;
            ++violations;
            worst_ratio = std::max(worst_ratio, error / bound);
        }
    }
    if (violations != 0u) {
        char detail[192]{};
        std::snprintf(detail, sizeof(detail),
                      "%zu values exceed %.3g + %.3g*|ref|; first=%zu actual=%.9g expected=%.9g worst_ratio=%.6g",
                      violations, absolute_tolerance, relative_tolerance,
                      first, actual[first], expected[first], worst_ratio);
        check(false, label, detail);
        return false;
    }
    check(true, label);
    return true;
}

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

bool run(Context& context, Fixture& fixture, bool caller_owned,
         std::array<std::vector<std::uint8_t>, kOutputCount>& outputs,
         bool windowed = false) {
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command = context.make_command_buffer(
        caller_owned ? "MTP routed experts caller-owned"
                     : "MTP routed experts standalone",
        error);
    check(command != nullptr, "MTP stage-17 command buffer", error);
    if (!command) return false;

    bool encoded = false;
    const auto views = windowed ? fixture.windowed_buffers(context)
                                : fixture.buffers(context);
    if (caller_owned) {
        encoded = Encoder::encode(
            context, *command, "MTP routed experts caller-owned segment",
            [&](Encoder& encoder, std::string& encode_error) {
                return encode_mtp_routed_experts(context, encoder, views,
                                                 encode_error);
            },
            error);
    } else {
        encoded = encode_mtp_routed_experts(context, *command, views, error);
    }
    check(encoded, "MTP stage-17 full ordered encode", error);
    if (!encoded) return false;
    const bool completed =
        submit_and_wait(command, "MTP stage 17", error);
    check(completed, "MTP stage-17 completion", error);
    if (!completed) return false;
    outputs = fixture.snapshot();
    return true;
}

bool verify_oracle(Fixture& fixture) {
    bool passed = true;
    const std::array<std::int8_t, kExpertInputWidth> expected_mq = [&] {
        std::array<std::int8_t, kExpertInputWidth> values{};
        std::copy(fixture.oracle.mixed_q8.begin(),
                  fixture.oracle.mixed_q8.end(), values.begin());
        return values;
    }();
    const auto actual_mq =
        std::span<const std::int8_t>(
            reinterpret_cast<const std::int8_t*>(
                output_bytes(fixture, kMixedQ8).data()),
            kExpertInputWidth);
    passed &= std::memcmp(actual_mq.data(), expected_mq.data(),
                          kMixedQ8Bytes) == 0;
    check(passed, "MTP stage-17 mixed Q8 codes match scalar oracle");

    const auto actual_ms = output_floats(fixture, kMixedScales);
    passed &= compare_exact_floats(actual_ms, fixture.oracle.mixed_scales,
                                   "MTP stage-17 mixed Q8 scales exact");
    passed &= compare_exact_floats(output_floats(fixture, kGate),
                                   fixture.oracle.gate,
                                   "MTP stage-17 S2 gate rows");
    passed &= compare_exact_floats(output_floats(fixture, kUp),
                                   fixture.oracle.up,
                                   "MTP stage-17 S2 up rows");
    passed &= compare_close_floats(output_floats(fixture, kSwiGlu),
                                   fixture.oracle.swiglu,
                                   "MTP stage-17 SwiGLU rows");

    const auto actual_hq =
        std::span<const std::int8_t>(
            reinterpret_cast<const std::int8_t*>(
                output_bytes(fixture, kHiddenQ8).data()),
            kHiddenQ8Bytes);
    const bool hidden_codes_exact =
        std::memcmp(actual_hq.data(), fixture.oracle.hidden_q8.data(),
                    kHiddenQ8Bytes) == 0;
    passed &= hidden_codes_exact;
    check(hidden_codes_exact,
          "MTP stage-17 SwiGLU Q8 codes match scalar oracle");
    passed &= compare_close_floats(
        output_floats(fixture, kHiddenScales), fixture.oracle.hidden_scales,
        "MTP stage-17 SwiGLU Q8 scales", 2.0e-11, 2.0e-7);
    passed &= compare_close_floats(
        output_floats(fixture, kParts), fixture.oracle.parts,
        "MTP stage-17 S2 down parts", 2.0e-9, 1.0e-6);
    return passed;
}

bool expect_rejected(Context& context,
                     const MtpRoutedExpertsBuffers& buffers,
                     const std::string& label,
                     std::string_view expected_error) {
    std::string error;
    auto command = context.make_command_buffer(label, error);
    check(command != nullptr, label + " command buffer", error);
    if (!command) return false;
    const bool rejected =
        !encode_mtp_routed_experts(context, *command, buffers, error);
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
    fixture.capture();
    const auto valid = fixture.buffers(context);

    auto bad = valid;
    bad.mixed = context.buffers().read_only(
        fixture.mixed.view().slice(0u, kMixedBytes - sizeof(float)));
    passed &= expect_rejected(context, bad, "Stage 17 rejects short mixed",
                              "undersized or misaligned");

    bad = fixture.windowed_buffers(context);
    if (fixture.expert_blob) {
        bad.expert_blob = context.buffers().read_only(
            fixture.expert_blob.slice(1u,
                                      fixture.expert_blob.length - 1u));
        bad.expert_windows.clear();
    } else {
        const BufferRef first_window =
            fixture.expert_window_buffers.front();
        bad.expert_windows.front().bytes = context.buffers().read_only(
            first_window.slice(1u, first_window.length - 1u));
    }
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects misaligned expert image",
                              "undersized or misaligned");

    bad = fixture.windowed_buffers(context);
    bad.expert_windows.pop_back();
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects incomplete expert windows",
                              "do not cover all 512 experts");

    bad = fixture.windowed_buffers(context);
    ++bad.expert_windows[1].first_expert;
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects gapped expert windows",
                              "cover the expert table in order");

    bad = fixture.windowed_buffers(context);
    bad.expert_blob = valid.expert_blob;
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects ambiguous expert inputs",
                              "either a complete expert image or expert windows");

    bad = valid;
    bad.ids = context.buffers().read_only(
        fixture.ids.view().slice(0u, kIdsBytes - sizeof(std::int32_t)));
    passed &= expect_rejected(context, bad, "Stage 17 rejects short route IDs",
                              "undersized or misaligned");

    std::string error;
    BufferRef foreign_buffer = foreign_context.buffers().allocate(
        "foreign MTP stage-17 mixed", kMixedBytes, error);
    check(static_cast<bool>(foreign_buffer),
          "MTP stage-17 foreign input allocation", error);
    if (foreign_buffer) {
        bad = valid;
        bad.mixed = foreign_context.buffers().read_only(foreign_buffer);
        passed &= expect_rejected(context, bad,
                                  "Stage 17 rejects foreign Context input",
                                  "not owned by this Context");
    } else {
        passed = false;
    }

    bad = valid;
    bad.mixed = context.buffers().read_only(
        fixture.output(kParts).slice(0u, kMixedBytes));
    bad.mq = fixture.output(kParts).slice(0u, kMixedQ8Bytes);
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects input/output overlap",
                              "overlaps input");

    bad = valid;
    bad.up = bad.gate;
    passed &= expect_rejected(context, bad,
                              "Stage 17 rejects overlapping outputs",
                              "outputs up rows and gate rows overlap");

    const bool unchanged = fixture.all_buffers_unchanged();
    check(unchanged,
          "Stage-17 validation failures preserve inputs and all outputs");
    return passed && unchanged;
}

bool test_group_boundary(Context& context, Fixture& fixture) {
    const char* dispatch_mode = std::getenv("STRATA_METAL_DISPATCH");
    if (dispatch_mode == nullptr ||
        std::string_view(dispatch_mode) != "concurrent") {
        return true;
    }
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command = context.make_command_buffer(
        "MTP routed experts group boundary", error);
    if (!command) {
        check(false, "MTP stage-17 group-boundary command buffer", error);
        return false;
    }
    const auto views = fixture.buffers(context);
    const bool encoded = Encoder::encode(
        context, *command, "MTP routed experts group-boundary validation",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encoder.group_begin(encode_error)) return false;
            std::string stage_error;
            const bool accepted =
                encode_mtp_routed_experts(context, encoder, views,
                                          stage_error);
            std::string end_error;
            if (!encoder.group_end(end_error)) {
                encode_error = end_error;
                return false;
            }
            if (!accepted &&
                stage_error.find("ordered concurrent-group boundary") !=
                    std::string::npos) {
                return true;
            }
            encode_error = "MTP stage 17 accepted an open concurrent group";
            return false;
        },
        error);
    const bool unchanged = fixture.all_buffers_unchanged();
    check(encoded, "MTP stage 17 rejects an open concurrent group", error);
    check(unchanged,
          "MTP stage-17 open-group rejection preserves every buffer");
    command->discard();
    return encoded && unchanged;
}

bool test_late_rollback(Context& context, Fixture& fixture) {
    fixture.reset_outputs();
    fixture.capture();
    std::string error;
    auto command =
        context.make_command_buffer("MTP routed experts late rollback", error);
    if (!command) {
        check(false, "MTP stage-17 rollback command buffer", error);
        return false;
    }
    const auto valid = fixture.buffers(context);
    const bool encoded = Encoder::encode(
        context, *command, "MTP routed experts late rollback segment",
        [&](Encoder& encoder, std::string& encode_error) {
            if (!encode_mtp_routed_experts(context, encoder, valid,
                                           encode_error)) {
                return false;
            }
            auto invalid = valid;
            invalid.ids = context.buffers().read_only(
                fixture.ids.view().slice(0u, kIdsBytes - 1u));
            std::string late_error;
            if (encode_mtp_routed_experts(context, encoder, invalid,
                                          late_error)) {
                encode_error = "MTP stage 17 accepted invalid late route IDs";
                return false;
            }
            encode_error = late_error;
            return false;
        },
        error);
    const bool discarded = command->unsafe_native_handle() == nullptr;
    const bool unchanged = fixture.all_buffers_unchanged();
    check(!encoded && discarded &&
              error.find("undersized or misaligned") != std::string::npos,
          "MTP stage-17 late failure discards its encoded prefix", error);
    check(unchanged,
          "MTP stage-17 rollback preserves every buffer");
    return !encoded && discarded && unchanged;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr,
                     "usage: metal_mtp_routed_experts_test <strata.metallib>\n");
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

    Fixture fixture;
    if (!fixture.allocate(*context)) return 1;

    bool passed = true;
    std::array<std::vector<std::uint8_t>, kOutputCount> standalone_outputs;
    passed &= run(*context, fixture, false, standalone_outputs);
    passed &= verify_oracle(fixture);
    std::array<std::vector<std::uint8_t>, kOutputCount> windowed_outputs;
    passed &= run(*context, fixture, false, windowed_outputs, true);
    bool window_parity = true;
    for (std::size_t index = 0u; index < kOutputCount; ++index) {
        window_parity &= standalone_outputs[index] == windowed_outputs[index];
    }
    check(window_parity,
          "MTP stage-17 windowed and complete-image outputs are byte-identical");
    passed &= window_parity;
    const bool immutable = fixture.inputs_unchanged();
    const bool guards = fixture.output_guards_unchanged();
    check(immutable,
          "MTP stage 17 preserves mixed, route IDs and selected expert bytes");
    check(guards, "MTP stage 17 preserves every output guard");
    passed &= immutable && guards;

    std::array<std::vector<std::uint8_t>, kOutputCount> caller_outputs;
    passed &= run(*context, fixture, true, caller_outputs);
    bool parity = true;
    for (std::size_t index = 0u; index < kOutputCount; ++index) {
        parity &= standalone_outputs[index] == caller_outputs[index];
    }
    check(parity,
          "MTP stage-17 standalone and caller-owned outputs are byte-identical");
    passed &= parity;
    passed &= test_validation(*context, *foreign_context, fixture);
    passed &= test_group_boundary(*context, fixture);
    passed &= test_late_rollback(*context, fixture);

    if (g_failures != 0 || !passed) {
        std::fprintf(stderr,
                     "metal_mtp_routed_experts_test: %d failure(s)\n",
                     g_failures);
        return 1;
    }
    std::puts("metal_mtp_routed_experts_test: MTP stage 17 OK");
    return 0;
}
