// Validated Metal MoE routing dispatches.
#include "strata/metal/moe_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/tiered.hpp"

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

}  // namespace strata::metal
