// Validated Metal MoE routing dispatches.
#include "strata/metal/moe_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/gr_dispatch.hpp"
#include "strata/metal/mv_dispatch.hpp"
#include "strata/metal/tiered.hpp"

#include "../kernels/common/types.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string>
#include <utility>

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

bool same_buffer_start(const void* left_contents, std::uint64_t left_gpu,
                       const void* right_contents,
                       std::uint64_t right_gpu) {
    bool comparable = false;
    if (left_contents != nullptr && right_contents != nullptr) {
        if (left_contents != right_contents) return false;
        comparable = true;
    }
    if (left_gpu != 0u && right_gpu != 0u) {
        if (left_gpu != right_gpu) return false;
        comparable = true;
    }
    return comparable;
}

bool same_buffer_start(ReadOnlyBufferRef left, ReadOnlyBufferRef right) {
    return same_buffer_start(left.contents(), left.gpu_address(),
                             right.contents(), right.gpu_address());
}

bool same_buffer_start(ReadOnlyBufferRef left, BufferRef right) {
    return same_buffer_start(left.contents(), left.gpu_address(),
                             right.contents, right.gpu_address);
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

bool encode_moe_routed_ffn_impl(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                std::uint32_t experts,
                                std::uint32_t layer,
                                MoeExpertFormat gate_up_format,
                                MoeDownFormat down_format,
                                const MoeRoutedFFNBuffers& buffers,
                                std::string& error,
                                bool validate_only = false) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "routed MoE FFN requires ordered dispatch boundaries";
        return false;
    }
    if (rows == 0u || rows > 8u || experts == 0u ||
        experts > kTieredMaxExperts || layer >= kTieredMaxLayers) {
        error = "routed MoE FFN requires 1..8 rows, 1..512 experts, and a valid layer";
        return false;
    }
    if (moe_expert_kernel(gate_up_format) == nullptr ||
        moe_down_kernel(down_format) == nullptr) {
        error = "routed MoE FFN weight format is unsupported";
        return false;
    }

    const std::uint64_t entries =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    const std::uint64_t address_count =
        (static_cast<std::uint64_t>(layer) + 1u) * experts;
    const std::uint64_t plan_count =
        static_cast<std::uint64_t>(layer) + 1u;
    std::size_t addresses_bytes = 0u;
    std::size_t input_bytes = 0u;
    std::size_t plans_bytes = 0u;
    std::size_t f32_bytes = 0u;
    std::size_t down_activations_bytes = 0u;
    std::size_t parts_bytes = 0u;
    if (entries > std::numeric_limits<std::uint32_t>::max() ||
        !checked_bytes(address_count, sizeof(MoeExpertAddress),
                       addresses_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeInputBlocksPerRow,
                       sizeof(Q81), input_bytes) ||
        !checked_bytes(plan_count, sizeof(TieredMoePlan), plans_bytes) ||
        !checked_bytes(entries * kMoeExpertOutputWidth, sizeof(float),
                       f32_bytes) ||
        !checked_bytes(entries * kMoeDownInputBlocksPerEntry, sizeof(Q81),
                       down_activations_bytes) ||
        !checked_bytes(entries * kMoeDownOutputWidth, sizeof(float),
                       parts_bytes)) {
        error = "routed MoE FFN shape exceeds the host byte range";
        return false;
    }

    if (!buffers.addresses || !context.buffers().owns(buffers.addresses) ||
        buffers.addresses.length() < addresses_bytes ||
        buffers.addresses.offset() % alignof(MoeExpertAddress) != 0u) {
        error = "routed MoE FFN address table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.input_activations ||
        !context.buffers().owns(buffers.input_activations) ||
        buffers.input_activations.length() < input_bytes ||
        buffers.input_activations.offset() % alignof(Q81) != 0u) {
        error = "routed MoE FFN input activations are stale, foreign, undersized or misaligned";
        return false;
    }
    if (!buffers.plans || !context.buffers().owns(buffers.plans) ||
        buffers.plans.length() < plans_bytes ||
        buffers.plans.offset() % alignof(TieredMoePlan) != 0u) {
        error = "routed MoE FFN plan table is stale, foreign, undersized or misaligned";
        return false;
    }
    if (views_overlap(buffers.addresses, buffers.input_activations) ||
        views_overlap(buffers.addresses, buffers.plans) ||
        views_overlap(buffers.input_activations, buffers.plans)) {
        error = "routed MoE FFN read-only inputs must be disjoint";
        return false;
    }
    const std::array writable{
        std::pair{buffers.gate, f32_bytes},
        std::pair{buffers.up, f32_bytes},
        std::pair{buffers.hidden, f32_bytes},
        std::pair{buffers.down_activations, down_activations_bytes},
        std::pair{buffers.parts, parts_bytes},
    };
    for (std::size_t i = 0; i < writable.size(); ++i) {
        const BufferRef output = writable[i].first;
        if (!output || !context.buffers().owns(output) ||
            output.length < writable[i].second) {
            error = "routed MoE FFN scratch output is stale, foreign or undersized";
            return false;
        }
        const std::size_t alignment =
            i == 3u ? alignof(Q81) : alignof(float);
        if (output.offset % alignment != 0u) {
            error = "routed MoE FFN scratch output is misaligned";
            return false;
        }
        for (std::size_t j = 0; j < i; ++j) {
            if (views_overlap(output, writable[j].first)) {
                error = "routed MoE FFN scratch outputs must be disjoint";
                return false;
            }
        }
        if (views_overlap(buffers.addresses, output) ||
            views_overlap(buffers.input_activations, output) ||
            views_overlap(buffers.plans, output)) {
            error = "routed MoE FFN scratch outputs must be disjoint from inputs";
            return false;
        }
    }

    if (validate_only) return true;

    const MoeGroupedExpertBuffers gate_up{
        buffers.addresses, buffers.input_activations, buffers.plans,
        buffers.gate, buffers.up};
    if (!encode_moe_experts_impl(context, encoder, rows, experts, layer,
                                 gate_up_format, gate_up, error)) {
        return false;
    }

    const MoeSwiGLUBuffers swiglu{
        context.buffers().read_only(buffers.gate),
        context.buffers().read_only(buffers.up),
        buffers.hidden};
    if (!encode_moe_swiglu_impl(context, encoder, rows, false, swiglu,
                                error)) {
        return false;
    }

    const Q8_1Buffers quantized_hidden{
        buffers.hidden, buffers.down_activations};
    if (!encode_q8_1(context, encoder,
                     static_cast<std::uint32_t>(entries),
                     kMoeExpertOutputWidth, quantized_hidden, error)) {
        return false;
    }

    const MoeGroupedDownBuffers down{
        buffers.addresses,
        context.buffers().read_only(buffers.down_activations),
        buffers.plans,
        buffers.parts};
    return encode_moe_down_impl(context, encoder, rows, experts, layer,
                                down_format, down, error);
}

bool encode_moe_shared_ffn_impl(Context& context, Encoder& encoder,
                                std::uint32_t rows,
                                const MoeSharedFFNBuffers& buffers,
                                std::string& error,
                                const MoeSharedFFNWriteBuffers* writeback =
                                    nullptr,
                                bool validate_only = false) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "shared MoE FFN requires ordered dispatch boundaries";
        return false;
    }
    if (rows == 0u || rows > 8u) {
        error = "shared MoE FFN requires 1..8 rows";
        return false;
    }

    const std::uint32_t compiled_columns =
        rows <= 1u ? 1u : (rows <= 2u ? 2u : (rows <= 4u ? 4u : 8u));
    const std::uint64_t padded_rows = compiled_columns - rows;
    std::size_t input_activations_bytes = 0u;
    std::size_t mixed_bytes = 0u;
    std::size_t gate_weights_bytes = 0u;
    std::size_t up_weights_bytes = 0u;
    std::size_t down_weights_bytes = 0u;
    std::size_t shared_gate_weights_bytes = 0u;
    std::size_t routed_parts_bytes = 0u;
    std::size_t route_weights_bytes = 0u;
    std::size_t gate_bytes = 0u;
    std::size_t up_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    std::size_t down_activations_bytes = 0u;
    std::size_t shared_output_bytes = 0u;
    std::size_t shared_gate_logits_bytes = 0u;
    std::size_t output_bytes = 0u;
    std::size_t gate_sink_bytes = 0u;
    std::size_t up_sink_bytes = 0u;
    std::size_t down_sink_bytes = 0u;
    std::size_t shared_gate_sink_bytes = 0u;
    const auto float_bytes = [](std::uint64_t count,
                                std::size_t& bytes) {
        return checked_bytes(count, sizeof(float), bytes);
    };
    const auto padded_sink_bytes = [&](std::uint32_t width,
                                       std::size_t& bytes) {
        const std::uint64_t count =
            std::max<std::uint64_t>(1u, padded_rows * width);
        return float_bytes(count, bytes);
    };
    if (!checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeInputBlocksPerRow,
                       sizeof(Q81), input_activations_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeDownOutputWidth,
                     mixed_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(kMoeExpertOutputWidth) *
                           kMoeInputBlocksPerRow,
                       sizeof(block_q8_0), gate_weights_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(kMoeExpertOutputWidth) *
                           kMoeInputBlocksPerRow,
                       sizeof(block_q8_0), up_weights_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(kMoeDownOutputWidth) *
                           kMoeDownInputBlocksPerEntry,
                       sizeof(block_q8_0), down_weights_bytes) ||
        !checked_bytes(kMoeDownOutputWidth, sizeof(std::uint16_t),
                       shared_gate_weights_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) * kMoeTopK *
                         kMoeDownOutputWidth,
                     routed_parts_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) * kMoeTopK,
                     route_weights_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeExpertOutputWidth,
                     gate_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeExpertOutputWidth,
                     up_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeExpertOutputWidth,
                     hidden_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownInputBlocksPerEntry,
                       sizeof(Q81), down_activations_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeDownOutputWidth,
                     shared_output_bytes) ||
        !float_bytes(rows, shared_gate_logits_bytes) ||
        !float_bytes(static_cast<std::uint64_t>(rows) *
                         kMoeDownOutputWidth,
                     output_bytes) ||
        !padded_sink_bytes(kMoeExpertOutputWidth, gate_sink_bytes) ||
        !padded_sink_bytes(kMoeExpertOutputWidth, up_sink_bytes) ||
        !padded_sink_bytes(kMoeDownOutputWidth, down_sink_bytes) ||
        !padded_sink_bytes(1u, shared_gate_sink_bytes)) {
        error = "shared MoE FFN shape exceeds the host byte range";
        return false;
    }

    const auto valid_read_only = [&](ReadOnlyBufferRef view,
                                     std::size_t required_bytes,
                                     std::size_t alignment) {
        if (!view || !context.buffers().owns(view) ||
            view.length() < required_bytes || alignment == 0u ||
            view.offset() % alignment != 0u || view.contents() == nullptr) {
            return false;
        }
        const auto cpu_address =
            reinterpret_cast<std::uintptr_t>(view.contents());
        const std::uint64_t gpu_address = view.gpu_address();
        return cpu_address % alignment == 0u &&
               (gpu_address == 0u || gpu_address % alignment == 0u);
    };
    const auto valid_output = [&](BufferRef view,
                                  std::size_t required_bytes,
                                  std::size_t alignment) {
        if (!view || !context.buffers().owns(view) ||
            view.length < required_bytes || alignment == 0u ||
            view.offset % alignment != 0u || view.contents == nullptr) {
            return false;
        }
        const auto cpu_address =
            reinterpret_cast<std::uintptr_t>(view.contents);
        return cpu_address % alignment == 0u &&
               (view.gpu_address == 0u ||
                view.gpu_address % alignment == 0u);
    };
    const auto require_input = [&](ReadOnlyBufferRef view,
                                   std::size_t required_bytes,
                                   std::size_t alignment,
                                   const char* name) {
        if (valid_read_only(view, required_bytes, alignment)) return true;
        error = std::string("shared MoE FFN ") + name +
                " is stale, foreign, undersized or misaligned";
        return false;
    };
    if (!require_input(buffers.input_activations, input_activations_bytes,
                       256u, "Q8_1 input") ||
        !require_input(buffers.mixed, mixed_bytes, 2u * alignof(float),
                       "mixed input") ||
        !require_input(buffers.gate_weights, gate_weights_bytes,
                       alignof(std::uint16_t), "gate weights") ||
        !require_input(buffers.up_weights, up_weights_bytes,
                       alignof(std::uint16_t), "up weights") ||
        !require_input(buffers.down_weights, down_weights_bytes,
                       alignof(std::uint16_t), "down weights") ||
        !require_input(buffers.shared_gate_weights,
                       shared_gate_weights_bytes, alignof(std::uint32_t),
                       "shared gate weights") ||
        !require_input(buffers.active_rows, sizeof(std::uint32_t),
                       alignof(std::uint32_t), "active-row count") ||
        !require_input(buffers.routed_parts, routed_parts_bytes,
                       alignof(float), "routed parts") ||
        !require_input(buffers.route_weights, route_weights_bytes,
                       alignof(float), "route weights")) {
        return false;
    }

    std::uint32_t live_rows = 0u;
    std::memcpy(&live_rows, buffers.active_rows.contents(),
                sizeof(live_rows));
    if (live_rows != rows) {
        error = "shared MoE FFN active-row count does not match rows";
        return false;
    }

    const std::array<std::pair<BufferRef, std::size_t>, 11u> writable{
        std::pair{buffers.gate, gate_bytes},
        std::pair{buffers.up, up_bytes},
        std::pair{buffers.hidden, hidden_bytes},
        std::pair{buffers.down_activations, down_activations_bytes},
        std::pair{buffers.shared_output, shared_output_bytes},
        std::pair{buffers.shared_gate_logits, shared_gate_logits_bytes},
        std::pair{buffers.output, output_bytes},
        std::pair{buffers.gate_sink, gate_sink_bytes},
        std::pair{buffers.up_sink, up_sink_bytes},
        std::pair{buffers.down_sink, down_sink_bytes},
        std::pair{buffers.shared_gate_sink, shared_gate_sink_bytes},
    };
    for (std::size_t i = 0u; i < writable.size(); ++i) {
        const std::size_t alignment =
            i == 3u ? alignof(Q81) : alignof(float);
        const BufferRef view = writable[i].first;
        if (!valid_output(view, writable[i].second, alignment)) {
            error = "shared MoE FFN scratch/output view is stale, foreign, "
                    "undersized or misaligned";
            return false;
        }
    }

    std::size_t residual_bytes = 0u;
    std::size_t inject_bytes = 0u;
    BufferRef residual_range;
    BufferRef inject_range;
    if (writeback != nullptr) {
        if (!checked_bytes(static_cast<std::uint64_t>(rows) * 4u *
                               kMoeDownOutputWidth,
                           sizeof(float), residual_bytes) ||
            !checked_bytes(static_cast<std::uint64_t>(rows) * 4u,
                           sizeof(float), inject_bytes)) {
            error = "shared MoE FFN GR write shape exceeds the host byte range";
            return false;
        }
        if (!valid_output(writeback->residual, residual_bytes,
                          alignof(float)) ||
            !valid_output(writeback->inject, inject_bytes, alignof(float))) {
            error = "shared MoE FFN GR residual or injection is stale, foreign, "
                    "undersized or misaligned";
            return false;
        }
        residual_range = writeback->residual.slice(0u, residual_bytes);
        inject_range = writeback->inject.slice(0u, inject_bytes);
        if (!residual_range || !inject_range) {
            error = "shared MoE FFN GR write view became stale";
            return false;
        }
        if (views_overlap(residual_range, inject_range)) {
            error = "shared MoE FFN GR residual and injection must be disjoint";
            return false;
        }
    }

    const std::array<std::pair<ReadOnlyBufferRef, std::size_t>, 9u> inputs{
        std::pair{buffers.input_activations, input_activations_bytes},
        std::pair{buffers.mixed, mixed_bytes},
        std::pair{buffers.gate_weights, gate_weights_bytes},
        std::pair{buffers.up_weights, up_weights_bytes},
        std::pair{buffers.down_weights, down_weights_bytes},
        std::pair{buffers.shared_gate_weights, shared_gate_weights_bytes},
        std::pair{buffers.active_rows, sizeof(std::uint32_t)},
        std::pair{buffers.routed_parts, routed_parts_bytes},
        std::pair{buffers.route_weights, route_weights_bytes},
    };
    std::array<BufferRef, 11u> writable_ranges{};
    for (std::size_t i = 0u; i < writable.size(); ++i) {
        writable_ranges[i] = writable[i].first.slice(0u, writable[i].second);
        if (!writable_ranges[i]) {
            error = "shared MoE FFN writable view became stale";
            return false;
        }
        for (std::size_t j = 0u; j < i; ++j) {
            const BufferRef prior =
                writable[j].first.slice(0u, writable[j].second);
            if (views_overlap(writable_ranges[i], prior)) {
                error = "shared MoE FFN writable views must be disjoint";
                return false;
            }
        }
        for (const auto& [input, input_bytes] : inputs) {
            const ReadOnlyBufferRef input_range =
                input.slice(0u, input_bytes);
            if (views_overlap(input_range, writable_ranges[i])) {
                error = "shared MoE FFN writable views must not overlap inputs";
                return false;
            }
        }
    }
    if (writeback != nullptr) {
        const std::array write_ranges{residual_range, inject_range};
        for (const BufferRef& write_view : write_ranges) {
            for (const BufferRef& shared_write : writable_ranges) {
                if (views_overlap(write_view, shared_write)) {
                    error = "shared MoE FFN GR write views must be disjoint "
                            "from shared FFN outputs";
                    return false;
                }
            }
            for (const auto& [input, input_bytes] : inputs) {
                const ReadOnlyBufferRef input_range =
                    input.slice(0u, input_bytes);
                if (views_overlap(input_range, write_view)) {
                    error = "shared MoE FFN GR write views must be disjoint "
                            "from shared FFN inputs";
                    return false;
                }
            }
        }
    }

    if (validate_only) return true;

    const ReadOnlyQ8_0MatVecBuffers gate_projection{
        buffers.input_activations, buffers.gate_weights,
        buffers.active_rows, buffers.gate, buffers.gate_sink};
    if (!encode_q8_0_matvec(context, encoder, kMoeDownOutputWidth,
                            kMoeExpertOutputWidth, rows, compiled_columns,
                            gate_projection, error)) {
        return false;
    }
    const ReadOnlyQ8_0MatVecBuffers up_projection{
        buffers.input_activations, buffers.up_weights,
        buffers.active_rows, buffers.up, buffers.up_sink};
    if (!encode_q8_0_matvec(context, encoder, kMoeDownOutputWidth,
                            kMoeExpertOutputWidth, rows, compiled_columns,
                            up_projection, error)) {
        return false;
    }

    const MoeSwiGLUBuffers swiglu{
        context.buffers().read_only(buffers.gate),
        context.buffers().read_only(buffers.up),
        buffers.hidden};
    if (!encode_moe_swiglu_impl(context, encoder, rows, true, swiglu,
                                error)) {
        return false;
    }

    const Q8_1Buffers hidden_activations{
        buffers.hidden, buffers.down_activations};
    if (!encode_q8_1(context, encoder, rows, kMoeExpertOutputWidth,
                     hidden_activations, error)) {
        return false;
    }

    const ReadOnlyQ8_0MatVecBuffers down_projection{
        context.buffers().read_only(buffers.down_activations),
        buffers.down_weights, buffers.active_rows, buffers.shared_output,
        buffers.down_sink};
    if (!encode_q8_0_matvec(context, encoder, kMoeExpertOutputWidth,
                            kMoeDownOutputWidth, rows, compiled_columns,
                            down_projection, error)) {
        return false;
    }

    const ReadOnlyBF16MatVecBuffers shared_gate{
        buffers.mixed, buffers.shared_gate_weights, buffers.active_rows,
        buffers.shared_gate_logits, buffers.shared_gate_sink};
    if (!encode_bf16_matvec(context, encoder, kMoeDownOutputWidth, 1u, rows,
                            compiled_columns, shared_gate, error)) {
        return false;
    }

    const MoeCombineBuffers combine{
        buffers.routed_parts, buffers.route_weights,
        context.buffers().read_only(buffers.shared_output),
        context.buffers().read_only(buffers.shared_gate_logits),
        buffers.output};
    if (!encode_moe_combine_impl(context, encoder, rows, combine, error)) {
        return false;
    }
    if (writeback == nullptr) return true;

    const GRPostBuffers post{residual_range, writable_ranges[6],
                             inject_range, residual_range};
    return encode_gr_post(context, encoder, rows, post, error);
}

bool encode_moe_routed_shared_ffn_impl(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedFFNBuffers& routed,
    const MoeSharedFFNBuffers& shared_ffn,
    const MoeSharedFFNWriteBuffers* writeback, std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "routed/shared MoE FFN requires ordered dispatch boundaries";
        return false;
    }

    if (!encode_moe_routed_ffn_impl(
            context, encoder, rows, experts, layer, gate_up_format,
            down_format, routed, error, true)) {
        return false;
    }

    if (!same_buffer_start(shared_ffn.input_activations,
                           routed.input_activations)) {
        error = "routed/shared MoE FFN activation views must share the "
                "routed Q8_1 producer";
        return false;
    }
    if (!same_buffer_start(shared_ffn.routed_parts, routed.parts)) {
        error = "routed/shared MoE FFN parts view must use the routed "
                "producer output";
        return false;
    }

    if (!encode_moe_shared_ffn_impl(
            context, encoder, rows, shared_ffn, error, writeback, true)) {
        return false;
    }

    const std::uint64_t routed_entries =
        static_cast<std::uint64_t>(rows) * kMoeTopK;
    const std::uint32_t shared_columns =
        rows <= 1u ? 1u : (rows <= 2u ? 2u : (rows <= 4u ? 4u : 8u));
    const std::uint64_t padded_rows = shared_columns - rows;
    std::size_t routed_addresses_bytes = 0u;
    std::size_t routed_input_bytes = 0u;
    std::size_t routed_plan_bytes = 0u;
    std::size_t routed_gate_up_bytes = 0u;
    std::size_t routed_down_activations_bytes = 0u;
    std::size_t routed_parts_bytes = 0u;
    std::size_t shared_input_bytes = 0u;
    std::size_t shared_mixed_bytes = 0u;
    std::size_t shared_gate_up_weights_bytes = 0u;
    std::size_t shared_down_weights_bytes = 0u;
    std::size_t shared_gate_weights_bytes = 0u;
    std::size_t shared_active_rows_bytes = sizeof(std::uint32_t);
    std::size_t shared_route_weights_bytes = 0u;
    std::size_t shared_projection_bytes = 0u;
    std::size_t shared_down_activations_bytes = 0u;
    std::size_t shared_output_bytes = 0u;
    std::size_t shared_gate_logits_bytes = 0u;
    std::size_t shared_result_bytes = 0u;
    std::size_t shared_gate_sink_bytes = 0u;
    std::size_t shared_up_sink_bytes = 0u;
    std::size_t shared_down_sink_bytes = 0u;
    std::size_t shared_gate_logits_sink_bytes = 0u;
    std::size_t residual_bytes = 0u;
    std::size_t inject_bytes = 0u;
    const auto sink_bytes = [&](std::uint32_t width, std::size_t& bytes) {
        return checked_bytes(std::max<std::uint64_t>(1u, padded_rows * width),
                             sizeof(float), bytes);
    };
    if (!checked_bytes((static_cast<std::uint64_t>(layer) + 1u) * experts,
                       sizeof(MoeExpertAddress), routed_addresses_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeInputBlocksPerRow,
                       sizeof(Q81), routed_input_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(layer) + 1u,
                       sizeof(TieredMoePlan), routed_plan_bytes) ||
        !checked_bytes(routed_entries * kMoeExpertOutputWidth,
                       sizeof(float), routed_gate_up_bytes) ||
        !checked_bytes(routed_entries * kMoeDownInputBlocksPerEntry,
                       sizeof(Q81), routed_down_activations_bytes) ||
        !checked_bytes(routed_entries * kMoeDownOutputWidth,
                       sizeof(float), routed_parts_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeInputBlocksPerRow,
                       sizeof(Q81), shared_input_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownOutputWidth,
                       sizeof(float), shared_mixed_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(kMoeExpertOutputWidth) *
                           kMoeInputBlocksPerRow,
                       sizeof(block_q8_0), shared_gate_up_weights_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(kMoeDownOutputWidth) *
                           kMoeDownInputBlocksPerEntry,
                       sizeof(block_q8_0), shared_down_weights_bytes) ||
        !checked_bytes(kMoeDownOutputWidth, sizeof(std::uint16_t),
                       shared_gate_weights_bytes) ||
        !checked_bytes(routed_entries, sizeof(float),
                       shared_route_weights_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeExpertOutputWidth,
                       sizeof(float), shared_projection_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownInputBlocksPerEntry,
                       sizeof(Q81), shared_down_activations_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownOutputWidth,
                       sizeof(float), shared_output_bytes) ||
        !checked_bytes(rows, sizeof(float), shared_gate_logits_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) *
                           kMoeDownOutputWidth,
                       sizeof(float), shared_result_bytes) ||
        !sink_bytes(kMoeExpertOutputWidth, shared_gate_sink_bytes) ||
        !sink_bytes(kMoeExpertOutputWidth, shared_up_sink_bytes) ||
        !sink_bytes(kMoeDownOutputWidth, shared_down_sink_bytes) ||
        !sink_bytes(1u, shared_gate_logits_sink_bytes) ||
        !checked_bytes(static_cast<std::uint64_t>(rows) * 4u *
                           kMoeDownOutputWidth,
                       sizeof(float), residual_bytes) ||
        (writeback != nullptr &&
         !checked_bytes(static_cast<std::uint64_t>(rows) * 4u,
                        sizeof(float), inject_bytes))) {
        error = "routed/shared MoE FFN active range exceeds the host byte range";
        return false;
    }

    const auto read_range = [](ReadOnlyBufferRef view, std::size_t bytes) {
        return view.slice(0u, bytes);
    };
    const auto write_range = [](BufferRef view, std::size_t bytes) {
        return view.slice(0u, bytes);
    };
    const std::array routed_inputs{
        read_range(routed.addresses, routed_addresses_bytes),
        read_range(routed.input_activations, routed_input_bytes),
        read_range(routed.plans, routed_plan_bytes),
    };
    const std::array routed_outputs{
        write_range(routed.gate, routed_gate_up_bytes),
        write_range(routed.up, routed_gate_up_bytes),
        write_range(routed.hidden, routed_gate_up_bytes),
        write_range(routed.down_activations, routed_down_activations_bytes),
        write_range(routed.parts, routed_parts_bytes),
    };
    const std::array shared_inputs{
        read_range(shared_ffn.input_activations, shared_input_bytes),
        read_range(shared_ffn.mixed, shared_mixed_bytes),
        read_range(shared_ffn.gate_weights, shared_gate_up_weights_bytes),
        read_range(shared_ffn.up_weights, shared_gate_up_weights_bytes),
        read_range(shared_ffn.down_weights, shared_down_weights_bytes),
        read_range(shared_ffn.shared_gate_weights, shared_gate_weights_bytes),
        read_range(shared_ffn.active_rows, shared_active_rows_bytes),
        read_range(shared_ffn.routed_parts, routed_parts_bytes),
        read_range(shared_ffn.route_weights, shared_route_weights_bytes),
    };
    const std::array shared_outputs{
        write_range(shared_ffn.gate, shared_projection_bytes),
        write_range(shared_ffn.up, shared_projection_bytes),
        write_range(shared_ffn.hidden, shared_projection_bytes),
        write_range(shared_ffn.down_activations, shared_down_activations_bytes),
        write_range(shared_ffn.shared_output, shared_output_bytes),
        write_range(shared_ffn.shared_gate_logits, shared_gate_logits_bytes),
        write_range(shared_ffn.output, shared_result_bytes),
        write_range(shared_ffn.gate_sink, shared_gate_sink_bytes),
        write_range(shared_ffn.up_sink, shared_gate_sink_bytes),
        write_range(shared_ffn.down_sink, shared_down_sink_bytes),
        write_range(shared_ffn.shared_gate_sink, shared_gate_logits_sink_bytes),
    };
    std::array<BufferRef, 2u> residual_writes{};
    if (writeback != nullptr) {
        residual_writes = {write_range(writeback->residual, residual_bytes),
                           write_range(writeback->inject, inject_bytes)};
    }
    const auto valid_ranges = [](const auto& views) {
        return std::all_of(views.begin(), views.end(),
                           [](const auto& view) { return static_cast<bool>(view); });
    };
    if (!valid_ranges(routed_inputs) || !valid_ranges(routed_outputs) ||
        !valid_ranges(shared_inputs) || !valid_ranges(shared_outputs) ||
        (writeback != nullptr && !valid_ranges(residual_writes))) {
        error = "routed/shared MoE FFN active view is stale or undersized";
        return false;
    }

    for (std::size_t output_index = 0u;
         output_index < routed_outputs.size(); ++output_index) {
        for (std::size_t input_index = 0u;
             input_index < shared_inputs.size(); ++input_index) {
            const bool routed_parts_handoff =
                output_index == 4u && input_index == 7u;
            if (!routed_parts_handoff &&
                views_overlap(shared_inputs[input_index],
                              routed_outputs[output_index])) {
                error = "routed/shared MoE FFN cross-stage output/input views "
                        "must be disjoint";
                return false;
            }
        }
        for (const BufferRef& shared_output : shared_outputs) {
            if (views_overlap(routed_outputs[output_index], shared_output)) {
                error = "routed/shared MoE FFN stage outputs must be disjoint";
                return false;
            }
        }
        for (const BufferRef& residual_write : residual_writes) {
            if (views_overlap(routed_outputs[output_index], residual_write)) {
                error = "routed/shared MoE FFN residual writes must be "
                        "disjoint from routed outputs";
                return false;
            }
        }
    }

    for (const ReadOnlyBufferRef& routed_input : routed_inputs) {
        for (const BufferRef& shared_output : shared_outputs) {
            if (views_overlap(routed_input, shared_output)) {
                error = "routed/shared MoE FFN shared outputs must be "
                        "disjoint from routed inputs";
                return false;
            }
        }
        for (const BufferRef& residual_write : residual_writes) {
            if (views_overlap(routed_input, residual_write)) {
                error = "routed/shared MoE FFN residual writes must be "
                        "disjoint from routed inputs";
                return false;
            }
        }
    }

    if (!encode_moe_routed_ffn_impl(context, encoder, rows, experts, layer,
                                    gate_up_format, down_format, routed,
                                    error)) {
        return false;
    }
    return encode_moe_shared_ffn_impl(context, encoder, rows, shared_ffn,
                                      error, writeback);
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

bool encode_moe_routed_ffn(Context& context, Encoder& encoder,
                           std::uint32_t rows, std::uint32_t experts,
                           std::uint32_t layer,
                           MoeExpertFormat gate_up_format,
                           MoeDownFormat down_format,
                           const MoeRoutedFFNBuffers& buffers,
                           std::string& error) {
    return encode_moe_routed_ffn_impl(
        context, encoder, rows, experts, layer, gate_up_format, down_format,
        buffers, error);
}

bool encode_moe_routed_ffn(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows, std::uint32_t experts,
                           std::uint32_t layer,
                           MoeExpertFormat gate_up_format,
                           MoeDownFormat down_format,
                           const MoeRoutedFFNBuffers& buffers,
                           std::string& error) {
    return Encoder::encode(
        context, command_buffer, "routed MoE FFN",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_routed_ffn_impl(
                context, encoder, rows, experts, layer, gate_up_format,
                down_format, buffers, encode_error);
        },
        error);
}

bool encode_moe_shared_ffn(Context& context, Encoder& encoder,
                           std::uint32_t rows,
                           const MoeSharedFFNBuffers& buffers,
                           std::string& error) {
    return encode_moe_shared_ffn_impl(context, encoder, rows, buffers,
                                       error);
}

bool encode_moe_shared_ffn(Context& context, CommandBuffer& command_buffer,
                           std::uint32_t rows,
                           const MoeSharedFFNBuffers& buffers,
                           std::string& error) {
    return Encoder::encode(
        context, command_buffer, "shared MoE FFN and combine",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_shared_ffn_impl(context, encoder, rows,
                                              buffers, encode_error);
        },
        error);
}

bool encode_moe_shared_ffn_write(
    Context& context, Encoder& encoder, std::uint32_t rows,
    const MoeSharedFFNWriteBuffers& buffers, std::string& error) {
    return encode_moe_shared_ffn_impl(context, encoder, rows, buffers.ffn,
                                       error, &buffers);
}

bool encode_moe_shared_ffn_write(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    const MoeSharedFFNWriteBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "shared MoE FFN and GR residual write",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_shared_ffn_impl(
                context, encoder, rows, buffers.ffn, encode_error, &buffers);
        },
        error);
}

bool encode_moe_routed_shared_ffn_write(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNWriteBuffers& buffers, std::string& error) {
    return encode_moe_routed_shared_ffn_impl(
        context, encoder, rows, experts, layer, gate_up_format, down_format,
        buffers.routed, buffers.shared.ffn, &buffers.shared, error);
}

bool encode_moe_routed_shared_ffn_write(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNWriteBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "routed/shared MoE FFN and GR residual write",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_routed_shared_ffn_impl(
                context, encoder, rows, experts, layer, gate_up_format,
                down_format, buffers.routed, buffers.shared.ffn,
                &buffers.shared, encode_error);
        },
        error);
}

bool encode_moe_routed_shared_ffn(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNBuffers& buffers, std::string& error) {
    return encode_moe_routed_shared_ffn_impl(
        context, encoder, rows, experts, layer, gate_up_format, down_format,
        buffers.routed, buffers.shared, nullptr, error);
}

bool encode_moe_routed_shared_ffn(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t experts, std::uint32_t layer,
    MoeExpertFormat gate_up_format, MoeDownFormat down_format,
    const MoeRoutedSharedFFNBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "routed/shared MoE FFN combine",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_moe_routed_shared_ffn_impl(
                context, encoder, rows, experts, layer, gate_up_format,
                down_format, buffers.routed, buffers.shared, nullptr,
                encode_error);
        },
        error);
}

}  // namespace strata::metal
