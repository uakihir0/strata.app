// Validated Metal MoE routing dispatches.
#include "strata/metal/moe_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/tiered.hpp"

#include "../kernels/common/types.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

namespace strata::metal {
namespace {

constexpr std::uint32_t kMoeTopK = 10u;
constexpr std::uint32_t kMoeRouteThreads = 32u;
constexpr std::uint32_t kMoePlanThreads = 128u;
constexpr std::uint32_t kMoeInputBlocksPerRow = 80u;
constexpr std::uint32_t kMoeExpertOutputWidth = 640u;
constexpr std::uint32_t kMoeDownInputBlocksPerEntry = 20u;
constexpr std::uint32_t kMoeDownOutputWidth = 2560u;
constexpr std::uint32_t kMoeDownOutputRowGroups =
    kMoeDownOutputWidth / 8u;
constexpr std::uint32_t kMoeExpertOutputRowGroups =
    2u * kMoeExpertOutputWidth / 8u;
constexpr std::uint32_t kMoeSwiGLURowGroups =
    (kMoeExpertOutputWidth + 255u) / 256u;

struct MoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(MoeExpertAddress) == 24u);

struct MoeGroupedGateUpParams {
    std::uint32_t n_expert;
    std::uint32_t layer;
    std::uint32_t n_rows;
};
static_assert(sizeof(MoeGroupedGateUpParams) == 12u);
static_assert(sizeof(Q81) == 36u);

struct MoeGroupedDownParams {
    std::uint32_t n_expert;
    std::uint32_t layer;
    std::uint32_t n_rows;
};
static_assert(sizeof(MoeGroupedDownParams) == 12u);

struct MoePlanParams {
    std::uint32_t rows;
    std::uint32_t n_expert;
    std::uint32_t tiered;
    std::uint32_t layer;
};
static_assert(sizeof(MoePlanParams) == 16u);

bool checked_bytes(std::uint64_t count, std::size_t element_bytes,
                   std::size_t& bytes) {
    if (element_bytes != 0u &&
        count > std::numeric_limits<std::size_t>::max() / element_bytes) {
        return false;
    }
    bytes = static_cast<std::size_t>(count) * element_bytes;
    return true;
}

bool ranges_overlap(std::uintptr_t left, std::size_t left_length,
                    std::uintptr_t right, std::size_t right_length) {
    if (left_length == 0u || right_length == 0u) return false;
    if (left <= right) return right - left < left_length;
    return left - right < right_length;
}

bool views_overlap(BufferRef left, BufferRef right) {
    if (left.handle != nullptr && left.handle == right.handle &&
        ranges_overlap(left.offset, left.length, right.offset, right.length)) {
        return true;
    }
    if (left.contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.contents),
                       left.length,
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length)) {
        return true;
    }
    return left.gpu_address != 0u && right.gpu_address != 0u &&
           ranges_overlap(static_cast<std::uintptr_t>(left.gpu_address),
                          left.length,
                          static_cast<std::uintptr_t>(right.gpu_address),
                          right.length);
}

bool views_overlap(ReadOnlyBufferRef left, BufferRef right) {
    const void* left_contents = left.contents();
    if (left_contents != nullptr && right.contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right.contents),
                       right.length)) {
        return true;
    }
    const std::uint64_t left_gpu = left.gpu_address();
    return left_gpu != 0u && right.gpu_address != 0u &&
           ranges_overlap(static_cast<std::uintptr_t>(left_gpu), left.length(),
                          static_cast<std::uintptr_t>(right.gpu_address),
                          right.length);
}

bool views_overlap(ReadOnlyBufferRef left, ReadOnlyBufferRef right) {
    const void* left_contents = left.contents();
    const void* right_contents = right.contents();
    if (left_contents != nullptr && right_contents != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left_contents),
                       left.length(),
                       reinterpret_cast<std::uintptr_t>(right_contents),
                       right.length())) {
        return true;
    }
    const std::uint64_t left_gpu = left.gpu_address();
    const std::uint64_t right_gpu = right.gpu_address();
    return left_gpu != 0u && right_gpu != 0u &&
           ranges_overlap(static_cast<std::uintptr_t>(left_gpu), left.length(),
                          static_cast<std::uintptr_t>(right_gpu), right.length());
}

const char* moe_expert_kernel(MoeExpertFormat format) {
    switch (format) {
        case MoeExpertFormat::q2_0: return "strata_moe_gu_q2_0";
        case MoeExpertFormat::iq1_m: return "strata_moe_gu_iq1_m";
        case MoeExpertFormat::iq2_xxs: return "strata_moe_gu_iq2_xxs";
        case MoeExpertFormat::iq2_xs: return "strata_moe_gu_iq2_xs";
        case MoeExpertFormat::iq2_s: return "strata_moe_gu_iq2_s";
        case MoeExpertFormat::iq3_xxs: return "strata_moe_gu_iq3_xxs";
        case MoeExpertFormat::iq3_s: return "strata_moe_gu_iq3_s";
        case MoeExpertFormat::iq4_xs: return "strata_moe_gu_iq4_xs";
    }
    return nullptr;
}

const char* moe_down_kernel(MoeDownFormat format) {
    switch (format) {
        case MoeDownFormat::q2_0: return "strata_moe_down_q2_0";
        case MoeDownFormat::iq4_nl: return "strata_moe_down_iq4_nl";
    }
    return nullptr;
}

bool validate_route(Context& context, Encoder& encoder, std::uint32_t rows,
                    std::uint32_t experts, const MoeRouteBuffers& buffers,
                    std::size_t& logits_bytes, std::size_t& output_bytes,
                    std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MoE route requires an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u ||
        (experts != 256u && experts != 512u)) {
        error = "MoE route requires positive rows and 256 or 512 experts";
        return false;
    }

    const std::uint64_t logit_count =
        static_cast<std::uint64_t>(rows) * experts;
    const std::uint64_t output_count =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    if (logit_count > std::numeric_limits<std::uint32_t>::max() ||
        output_count > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(logit_count, sizeof(float), logits_bytes) ||
        !checked_bytes(output_count, sizeof(float), output_bytes)) {
        error = "MoE route shape exceeds the shader index range";
        return false;
    }
    std::size_t ids_bytes = 0u;
    if (!checked_bytes(output_count, sizeof(std::int32_t), ids_bytes)) {
        error = "MoE route output size overflows the host byte range";
        return false;
    }
    if (!buffers.logits || !context.buffers().owns(buffers.logits) ||
        buffers.logits.length() < logits_bytes ||
        buffers.logits.offset() % alignof(float) != 0u) {
        error = "MoE route logits are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.ids || !context.buffers().owns(buffers.ids) ||
        buffers.ids.length < ids_bytes ||
        buffers.ids.offset % alignof(std::int32_t) != 0u) {
        error = "MoE route IDs are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.weights || !context.buffers().owns(buffers.weights) ||
        buffers.weights.length < output_bytes ||
        buffers.weights.offset % alignof(float) != 0u) {
        error = "MoE route weights are stale, foreign, undersized or misaligned";
        return false;
    }
    if (views_overlap(buffers.logits, buffers.ids) ||
        views_overlap(buffers.logits, buffers.weights) ||
        views_overlap(buffers.ids, buffers.weights)) {
        error = "MoE route outputs must not overlap each other or the logits";
        return false;
    }

    error.clear();
    return true;
}

bool validate_plan(Context& context, Encoder& encoder,
                   std::uint32_t rows, std::uint32_t experts,
                   std::uint32_t layer, const MoePlanBuffers& buffers,
                   std::size_t& ids_bytes, std::size_t& table_bytes,
                   std::size_t& plans_bytes, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MoE plan requires an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u || rows > 8u || experts == 0u ||
        experts > kTieredMaxExperts || layer >= kTieredMaxLayers) {
        error = "MoE plan requires 1..8 rows, 1..512 experts, and a valid layer";
        return false;
    }

    const std::uint64_t entries =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    const std::uint64_t table_entries =
        (static_cast<std::uint64_t>(layer) + 1u) * experts;
    const std::uint64_t plan_count =
        static_cast<std::uint64_t>(layer) + 1u;
    std::size_t resident_bytes = 0u;
    if (entries > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(entries, sizeof(std::int32_t), ids_bytes) ||
        !checked_bytes(table_entries, sizeof(std::int32_t), resident_bytes) ||
        !checked_bytes(table_entries, sizeof(float), table_bytes) ||
        !checked_bytes(plan_count, sizeof(TieredMoePlan), plans_bytes)) {
        error = "MoE plan shape exceeds the host byte range";
        return false;
    }

    if (!buffers.ids || !context.buffers().owns(buffers.ids) ||
        buffers.ids.length() < ids_bytes ||
        buffers.ids.offset() % alignof(std::int32_t) != 0u) {
        error = "MoE plan IDs are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.resident || !context.buffers().owns(buffers.resident) ||
        buffers.resident.length() < resident_bytes ||
        buffers.resident.offset() % alignof(std::int32_t) != 0u) {
        error = "MoE plan resident table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.usage || !context.buffers().owns(buffers.usage) ||
        buffers.usage.length < table_bytes ||
        buffers.usage.offset % alignof(float) != 0u) {
        error = "MoE plan usage table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.plans || !context.buffers().owns(buffers.plans) ||
        buffers.plans.length < plans_bytes ||
        buffers.plans.offset % alignof(TieredMoePlan) != 0u) {
        error = "MoE plan output is stale, foreign, undersized or misaligned";
        return false;
    }

    if (views_overlap(buffers.ids, buffers.usage) ||
        views_overlap(buffers.ids, buffers.plans) ||
        views_overlap(buffers.resident, buffers.usage) ||
        views_overlap(buffers.resident, buffers.plans) ||
        views_overlap(buffers.usage, buffers.plans)) {
        error = "MoE plan writable views must not overlap any input or each other";
        return false;
    }
    return true;
}

bool encode_moe_plan_impl(Context& context, Encoder& encoder,
                          std::uint32_t rows, std::uint32_t experts,
                          bool tiered, std::uint32_t layer,
                          const MoePlanBuffers& buffers,
                          std::string& error) {
    std::size_t ids_bytes = 0u;
    std::size_t table_bytes = 0u;
    std::size_t plans_bytes = 0u;
    if (!validate_plan(context, encoder, rows, experts, layer, buffers,
                       ids_bytes, table_bytes, plans_bytes, error)) {
        return false;
    }

    const MoePlanParams params{rows, experts, tiered ? 1u : 0u, layer};
    const std::array writable{
        EncoderBufferBinding{buffers.usage, 2u, table_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.plans, 3u, plans_bytes,
                             alignof(TieredMoePlan)},
    };
    const std::size_t resident_bytes =
        static_cast<std::size_t>(layer + 1u) * experts * sizeof(std::int32_t);
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.ids, 0u, ids_bytes,
                                     alignof(std::int32_t)},
        EncoderReadOnlyBufferBinding{buffers.resident, 1u, resident_bytes,
                                     alignof(std::int32_t)},
    };
    const std::array bytes{
        EncoderBytesBinding{&params, sizeof(params), 4u},
    };
    return encoder.dispatch("strata_moe_plan", {1u, 1u, 1u},
                            {kMoePlanThreads, 1u, 1u}, writable,
                            read_only, bytes, {}, error);
}

bool encode_moe_route_impl(Context& context, Encoder& encoder,
                           std::uint32_t rows, std::uint32_t experts,
                           bool divide_weights,
                           const MoeRouteBuffers& buffers,
                           std::string& error) {
    std::size_t logits_bytes = 0u;
    std::size_t weights_bytes = 0u;
    if (!validate_route(context, encoder, rows, experts, buffers,
                        logits_bytes, weights_bytes, error)) {
        return false;
    }

    const std::string kernel = experts == 512u
        ? "strata_moe_route_r16"
        : "strata_moe_route_r8";
    const std::uint32_t divide = divide_weights ? 1u : 0u;
    const std::array writable{
        EncoderBufferBinding{buffers.ids, 1u,
                             static_cast<std::size_t>(rows) * kMoeTopK *
                                 sizeof(std::int32_t),
                             alignof(std::int32_t)},
        EncoderBufferBinding{buffers.weights, 2u, weights_bytes,
                             alignof(float)},
    };
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.logits, 0u, logits_bytes,
                                     alignof(float)},
    };
    const std::array bytes{
        EncoderBytesBinding{&divide, sizeof(divide), 3u},
    };
    return encoder.dispatch(kernel, {rows, 1u, 1u},
                            {kMoeRouteThreads, 1u, 1u}, writable,
                            read_only, bytes, {}, error);
}

bool encode_moe_experts_impl(Context& context, Encoder& encoder,
                             std::uint32_t rows, std::uint32_t experts,
                             std::uint32_t layer, MoeExpertFormat format,
                             const MoeGroupedExpertBuffers& buffers,
                             std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "grouped MoE experts require an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u || rows > 8u || experts == 0u ||
        experts > kTieredMaxExperts || layer >= kTieredMaxLayers) {
        error = "grouped MoE experts require 1..8 rows, 1..512 experts, and a valid layer";
        return false;
    }
    const char* kernel = moe_expert_kernel(format);
    if (kernel == nullptr) {
        error = "grouped MoE expert format is unsupported";
        return false;
    }

    const std::uint64_t entry_count =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    const std::uint64_t address_count =
        (static_cast<std::uint64_t>(layer) + 1u) * experts;
    const std::uint64_t plan_count =
        static_cast<std::uint64_t>(layer) + 1u;
    std::size_t addresses_bytes = 0u;
    std::size_t activations_bytes = 0u;
    std::size_t plans_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (entry_count > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(address_count, sizeof(MoeExpertAddress),
                       addresses_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeInputBlocksPerRow,
                       sizeof(Q81), activations_bytes) ||
        !checked_bytes(plan_count, sizeof(TieredMoePlan), plans_bytes) ||
        !checked_bytes(entry_count * kMoeExpertOutputWidth, sizeof(float),
                       output_bytes)) {
        error = "grouped MoE expert shape exceeds the host byte range";
        return false;
    }

    if (!buffers.addresses || !context.buffers().owns(buffers.addresses) ||
        buffers.addresses.length() < addresses_bytes ||
        buffers.addresses.offset() % alignof(MoeExpertAddress) != 0u) {
        error = "grouped MoE expert address table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.activations ||
        !context.buffers().owns(buffers.activations) ||
        buffers.activations.length() < activations_bytes ||
        buffers.activations.offset() % alignof(Q81) != 0u) {
        error = "grouped MoE Q8_1 activations are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.plans || !context.buffers().owns(buffers.plans) ||
        buffers.plans.length() < plans_bytes ||
        buffers.plans.offset() % alignof(TieredMoePlan) != 0u) {
        error = "grouped MoE plan table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.gate || !context.buffers().owns(buffers.gate) ||
        buffers.gate.length < output_bytes ||
        buffers.gate.offset % alignof(float) != 0u) {
        error = "grouped MoE gate output is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.up || !context.buffers().owns(buffers.up) ||
        buffers.up.length < output_bytes ||
        buffers.up.offset % alignof(float) != 0u) {
        error = "grouped MoE up output is stale, foreign, undersized or misaligned";
        return false;
    }

    if (views_overlap(buffers.addresses, buffers.activations) ||
        views_overlap(buffers.addresses, buffers.plans) ||
        views_overlap(buffers.activations, buffers.plans) ||
        views_overlap(buffers.addresses, buffers.gate) ||
        views_overlap(buffers.addresses, buffers.up) ||
        views_overlap(buffers.activations, buffers.gate) ||
        views_overlap(buffers.activations, buffers.up) ||
        views_overlap(buffers.plans, buffers.gate) ||
        views_overlap(buffers.plans, buffers.up) ||
        views_overlap(buffers.gate, buffers.up)) {
        error = "grouped MoE expert outputs must not overlap their inputs or each other";
        return false;
    }

    const MoeGroupedGateUpParams params{experts, layer, rows};
    const std::array writable{
        EncoderBufferBinding{buffers.gate, 3u, output_bytes,
                             alignof(float)},
        EncoderBufferBinding{buffers.up, 4u, output_bytes,
                             alignof(float)},
    };
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.addresses, 0u,
                                     addresses_bytes,
                                     alignof(MoeExpertAddress)},
        EncoderReadOnlyBufferBinding{buffers.activations, 1u,
                                     activations_bytes, alignof(Q81)},
        EncoderReadOnlyBufferBinding{buffers.plans, 2u, plans_bytes,
                                     alignof(TieredMoePlan)},
    };
    const std::array bytes{
        EncoderBytesBinding{&params, sizeof(params), 5u},
    };
    error.clear();
    return encoder.dispatch(kernel, {kMoeExpertOutputRowGroups,
                                     static_cast<std::uint32_t>(entry_count),
                                     1u},
                            {256u, 1u, 1u}, writable, read_only, bytes, {},
                            error);
}

bool encode_moe_down_impl(Context& context, Encoder& encoder,
                          std::uint32_t rows, std::uint32_t experts,
                          std::uint32_t layer, MoeDownFormat format,
                          const MoeGroupedDownBuffers& buffers,
                          std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "grouped MoE down projection requires an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u || rows > 8u || experts == 0u ||
        experts > kTieredMaxExperts || layer >= kTieredMaxLayers) {
        error = "grouped MoE down projection requires 1..8 rows, 1..512 experts, and a valid layer";
        return false;
    }
    const char* kernel = moe_down_kernel(format);
    if (kernel == nullptr) {
        error = "grouped MoE down format is unsupported";
        return false;
    }

    const std::uint64_t entry_count =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    const std::uint64_t address_count =
        (static_cast<std::uint64_t>(layer) + 1u) * experts;
    const std::uint64_t plan_count =
        static_cast<std::uint64_t>(layer) + 1u;
    std::size_t addresses_bytes = 0u;
    std::size_t activations_bytes = 0u;
    std::size_t plans_bytes = 0u;
    std::size_t parts_bytes = 0u;
    if (entry_count > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(address_count, sizeof(MoeExpertAddress),
                       addresses_bytes) ||
        !checked_bytes(entry_count * kMoeDownInputBlocksPerEntry,
                       sizeof(Q81), activations_bytes) ||
        !checked_bytes(plan_count, sizeof(TieredMoePlan), plans_bytes) ||
        !checked_bytes(entry_count * kMoeDownOutputWidth, sizeof(float),
                       parts_bytes)) {
        error = "grouped MoE down shape exceeds the host byte range";
        return false;
    }

    if (!buffers.addresses || !context.buffers().owns(buffers.addresses) ||
        buffers.addresses.length() < addresses_bytes ||
        buffers.addresses.offset() % alignof(MoeExpertAddress) != 0u) {
        error = "grouped MoE down address table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.activations ||
        !context.buffers().owns(buffers.activations) ||
        buffers.activations.length() < activations_bytes ||
        buffers.activations.offset() % alignof(Q81) != 0u) {
        error = "grouped MoE down Q8_1 activations are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.plans || !context.buffers().owns(buffers.plans) ||
        buffers.plans.length() < plans_bytes ||
        buffers.plans.offset() % alignof(TieredMoePlan) != 0u) {
        error = "grouped MoE down plan table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.parts || !context.buffers().owns(buffers.parts) ||
        buffers.parts.length < parts_bytes ||
        buffers.parts.offset % alignof(float) != 0u) {
        error = "grouped MoE down parts output is stale, foreign, undersized or misaligned";
        return false;
    }

    if (views_overlap(buffers.addresses, buffers.activations) ||
        views_overlap(buffers.addresses, buffers.plans) ||
        views_overlap(buffers.activations, buffers.plans) ||
        views_overlap(buffers.addresses, buffers.parts) ||
        views_overlap(buffers.activations, buffers.parts) ||
        views_overlap(buffers.plans, buffers.parts)) {
        error = "grouped MoE down output must not overlap its inputs";
        return false;
    }

    const MoeGroupedDownParams params{experts, layer, rows};
    const std::array writable{
        EncoderBufferBinding{buffers.parts, 3u, parts_bytes,
                             alignof(float)},
    };
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.addresses, 0u,
                                     addresses_bytes,
                                     alignof(MoeExpertAddress)},
        EncoderReadOnlyBufferBinding{buffers.activations, 1u,
                                     activations_bytes, alignof(Q81)},
        EncoderReadOnlyBufferBinding{buffers.plans, 2u, plans_bytes,
                                     alignof(TieredMoePlan)},
    };
    const std::array bytes{
        EncoderBytesBinding{&params, sizeof(params), 4u},
    };
    return encoder.dispatch(kernel, {kMoeDownOutputRowGroups,
                                     static_cast<std::uint32_t>(entry_count),
                                     1u},
                            {256u, 1u, 1u}, writable, read_only, bytes, {},
                            error);
}

bool encode_moe_swiglu_impl(Context& context, Encoder& encoder,
                            std::uint32_t rows, bool shared_expert,
                            const MoeSwiGLUBuffers& buffers,
                            std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MoE SwiGLU requires an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "MoE SwiGLU requires 1..8 rows";
        return false;
    }

    const std::uint64_t entries =
        static_cast<std::uint64_t>(rows) *
        (shared_expert ? 1u : kMoeTopK);
    std::size_t output_bytes = 0u;
    if (entries > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(entries * kMoeExpertOutputWidth, sizeof(float),
                       output_bytes)) {
        error = "MoE SwiGLU shape exceeds the host byte range";
        return false;
    }
    if (!buffers.gate || !context.buffers().owns(buffers.gate) ||
        buffers.gate.length() < output_bytes ||
        buffers.gate.offset() % alignof(float) != 0u) {
        error = "MoE SwiGLU gate input is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.up || !context.buffers().owns(buffers.up) ||
        buffers.up.length() < output_bytes ||
        buffers.up.offset() % alignof(float) != 0u) {
        error = "MoE SwiGLU up input is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.output || !context.buffers().owns(buffers.output) ||
        buffers.output.length < output_bytes ||
        buffers.output.offset % alignof(float) != 0u) {
        error = "MoE SwiGLU output is stale, foreign, undersized or misaligned";
        return false;
    }
    if (views_overlap(buffers.gate, buffers.output) ||
        views_overlap(buffers.up, buffers.output)) {
        error = "MoE SwiGLU output must not overlap either input";
        return false;
    }

    const char* kernel = shared_expert
        ? "strata_shx_swiglu"
        : "strata_moe_swiglu";
    const std::array writable{
        EncoderBufferBinding{buffers.output, 2u, output_bytes,
                             alignof(float)},
    };
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.gate, 0u, output_bytes,
                                     alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.up, 1u, output_bytes,
                                     alignof(float)},
    };
    return encoder.dispatch(kernel,
                            {kMoeSwiGLURowGroups,
                             static_cast<std::uint32_t>(entries), 1u},
                            {256u, 1u, 1u}, writable, read_only, {}, {},
                            error);
}

bool encode_moe_combine_impl(Context& context, Encoder& encoder,
                             std::uint32_t rows,
                             const MoeCombineBuffers& buffers,
                             std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "MoE combine requires an ordered dispatch boundary";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "MoE combine requires 1..8 rows";
        return false;
    }

    const std::uint64_t entry_count =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    std::size_t parts_bytes = 0u;
    std::size_t weights_bytes = 0u;
    std::size_t shared_bytes = 0u;
    std::size_t gate_bytes = 0u;
    std::size_t output_bytes = 0u;
    if (!checked_bytes(entry_count * kMoeDownOutputWidth, sizeof(float),
                       parts_bytes) ||
        !checked_bytes(entry_count, sizeof(float), weights_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownOutputWidth,
                       sizeof(float), shared_bytes) ||
        !checked_bytes(rows, sizeof(float), gate_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownOutputWidth,
                       sizeof(float), output_bytes)) {
        error = "MoE combine shape exceeds the host byte range";
        return false;
    }

    if (!buffers.parts || !context.buffers().owns(buffers.parts) ||
        buffers.parts.length() < parts_bytes ||
        buffers.parts.offset() % alignof(float) != 0u) {
        error = "MoE combine routed parts are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.route_weights ||
        !context.buffers().owns(buffers.route_weights) ||
        buffers.route_weights.length() < weights_bytes ||
        buffers.route_weights.offset() % alignof(float) != 0u) {
        error = "MoE combine route weights are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.shared_output ||
        !context.buffers().owns(buffers.shared_output) ||
        buffers.shared_output.length() < shared_bytes ||
        buffers.shared_output.offset() % alignof(float) != 0u) {
        error = "MoE combine shared output is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.shared_gate_logits ||
        !context.buffers().owns(buffers.shared_gate_logits) ||
        buffers.shared_gate_logits.length() < gate_bytes ||
        buffers.shared_gate_logits.offset() % alignof(float) != 0u) {
        error = "MoE combine shared gate logits are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.output || !context.buffers().owns(buffers.output) ||
        buffers.output.length < output_bytes ||
        buffers.output.offset % alignof(float) != 0u) {
        error = "MoE combine output is stale, foreign, undersized or misaligned";
        return false;
    }
    if (views_overlap(buffers.parts, buffers.output) ||
        views_overlap(buffers.route_weights, buffers.output) ||
        views_overlap(buffers.shared_output, buffers.output) ||
        views_overlap(buffers.shared_gate_logits, buffers.output)) {
        error = "MoE combine output must not overlap any input";
        return false;
    }

    const std::array writable{
        EncoderBufferBinding{buffers.output, 4u, output_bytes,
                             alignof(float)},
    };
    const std::array read_only{
        EncoderReadOnlyBufferBinding{buffers.parts, 0u, parts_bytes,
                                     alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.route_weights, 1u,
                                     weights_bytes, alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.shared_output, 2u,
                                     shared_bytes, alignof(float)},
        EncoderReadOnlyBufferBinding{buffers.shared_gate_logits, 3u,
                                     gate_bytes, alignof(float)},
    };
    return encoder.dispatch("strata_moe_combine",
                           {kMoeTopK, rows, 1u},
                           {256u, 1u, 1u}, writable, read_only, {}, {},
                           error);
}

}  // namespace

bool encode_moe_route(Context& context, Encoder& encoder,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error) {
    return encode_moe_route_impl(context, encoder, rows, experts,
                                 divide_weights, buffers, error);
}

bool encode_moe_route(Context& context, CommandBuffer& command_buffer,
                      std::uint32_t rows, std::uint32_t experts,
                      bool divide_weights, const MoeRouteBuffers& buffers,
                      std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MoE route",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_route_impl(context, encoder, rows, experts,
                                         divide_weights, buffers,
                                         encode_error);
        },
        error);
}

bool encode_moe_plan(Context& context, Encoder& encoder,
                     std::uint32_t rows, std::uint32_t experts,
                     bool tiered, std::uint32_t layer,
                     const MoePlanBuffers& buffers, std::string& error) {
    return encode_moe_plan_impl(context, encoder, rows, experts, tiered,
                                layer, buffers, error);
}

bool encode_moe_plan(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, std::uint32_t experts,
                     bool tiered, std::uint32_t layer,
                     const MoePlanBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MoE plan",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_plan_impl(context, encoder, rows, experts,
                                        tiered, layer, buffers, encode_error);
        },
        error);
}

bool encode_moe_experts(Context& context, Encoder& encoder,
                        std::uint32_t rows, std::uint32_t experts,
                        std::uint32_t layer, MoeExpertFormat format,
                        const MoeGroupedExpertBuffers& buffers,
                        std::string& error) {
    return encode_moe_experts_impl(context, encoder, rows, experts, layer,
                                   format, buffers, error);
}

bool encode_moe_experts(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows, std::uint32_t experts,
                        std::uint32_t layer, MoeExpertFormat format,
                        const MoeGroupedExpertBuffers& buffers,
                        std::string& error) {
    return Encoder::encode(
        context, command_buffer, "grouped MoE experts",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_experts_impl(context, encoder, rows, experts,
                                           layer, format, buffers,
                                           encode_error);
        },
            error);
}

bool encode_moe_down(Context& context, Encoder& encoder,
                     std::uint32_t rows, std::uint32_t experts,
                     std::uint32_t layer, MoeDownFormat format,
                     const MoeGroupedDownBuffers& buffers,
                     std::string& error) {
    return encode_moe_down_impl(context, encoder, rows, experts, layer,
                                format, buffers, error);
}

bool encode_moe_down(Context& context, CommandBuffer& command_buffer,
                     std::uint32_t rows, std::uint32_t experts,
                     std::uint32_t layer, MoeDownFormat format,
                     const MoeGroupedDownBuffers& buffers,
                     std::string& error) {
    return Encoder::encode(
        context, command_buffer, "grouped MoE down projection",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_down_impl(context, encoder, rows, experts, layer,
                                        format, buffers, encode_error);
        },
        error);
}

bool encode_moe_swiglu(Context& context, Encoder& encoder,
                       std::uint32_t rows, bool shared_expert,
                       const MoeSwiGLUBuffers& buffers,
                       std::string& error) {
    return encode_moe_swiglu_impl(context, encoder, rows, shared_expert,
                                  buffers, error);
}

bool encode_moe_swiglu(Context& context, CommandBuffer& command_buffer,
                       std::uint32_t rows, bool shared_expert,
                       const MoeSwiGLUBuffers& buffers,
                       std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MoE SwiGLU",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_swiglu_impl(context, encoder, rows,
                                          shared_expert, buffers,
                                          encode_error);
        },
        error);
}

bool encode_moe_combine(Context& context, Encoder& encoder,
                        std::uint32_t rows,
                        const MoeCombineBuffers& buffers,
                        std::string& error) {
    return encode_moe_combine_impl(context, encoder, rows, buffers, error);
}

bool encode_moe_combine(Context& context, CommandBuffer& command_buffer,
                        std::uint32_t rows,
                        const MoeCombineBuffers& buffers,
                        std::string& error) {
    return Encoder::encode(
        context, command_buffer, "MoE combine",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_combine_impl(context, encoder, rows, buffers,
                                           encode_error);
        },
        error);
}

}  // namespace strata::metal
