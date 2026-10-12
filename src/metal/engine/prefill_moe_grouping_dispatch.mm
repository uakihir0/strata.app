// Validated Prefill MoE route grouping and row gathering.
#include "strata/metal/prefill_moe_grouping.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/prefill.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>

namespace strata::metal {
namespace {

constexpr std::uint32_t kCountThreads = 256u;
constexpr std::uint32_t kScanThreads = 512u;
constexpr std::uint32_t kGatherWidth = 2560u;
constexpr std::size_t kIntBytes = sizeof(std::int32_t);
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::size_t kGatherVectorBytes = 16u;
constexpr std::size_t kTileBytes = 2u * sizeof(std::uint32_t);

struct PrefillMoeGroupingParams {
    std::uint32_t rows;
    std::uint32_t n_experts;
    std::uint32_t layer;
    std::uint32_t tiered;
};
static_assert(sizeof(PrefillMoeGroupingParams) == 16u);

struct ActiveRange {
    const void* cpu = nullptr;
    std::uint64_t gpu = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
    bool writable = false;
};

template <class Address>
bool ranges_overlap(Address left, std::size_t left_bytes, Address right,
                    std::size_t right_bytes) {
    if (left == 0 || right == 0 || left_bytes == 0u || right_bytes == 0u) {
        return false;
    }
    return left <= right ? right - left < left_bytes
                         : left - right < right_bytes;
}

bool overlaps(const ActiveRange& left, const ActiveRange& right) {
    const bool cpu_overlap =
        left.cpu != nullptr && right.cpu != nullptr &&
        ranges_overlap(reinterpret_cast<std::uintptr_t>(left.cpu), left.bytes,
                       reinterpret_cast<std::uintptr_t>(right.cpu),
                       right.bytes);
    const bool gpu_overlap =
        left.gpu != 0u && right.gpu != 0u &&
        ranges_overlap(left.gpu, left.bytes, right.gpu, right.bytes);
    return cpu_overlap || gpu_overlap;
}

bool checked_product(std::initializer_list<std::size_t> factors,
                     std::size_t& result) {
    result = 1u;
    for (const std::size_t factor : factors) {
        if (factor != 0u &&
            result > std::numeric_limits<std::size_t>::max() / factor) {
            return false;
        }
        result *= factor;
    }
    return true;
}

std::size_t round_up_div(std::size_t value, std::size_t divisor) {
    return value / divisor + static_cast<std::size_t>(value % divisor != 0u);
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE grouping " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill MoE grouping " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE grouping " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill MoE grouping " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

ReadOnlyBufferRef active_view(ReadOnlyBufferRef view,
                              std::size_t active_bytes) {
    return view.slice(0u, active_bytes);
}

BufferRef active_view(BufferRef view, std::size_t active_bytes) {
    return view.slice(0u, active_bytes);
}

ActiveRange read_range(ReadOnlyBufferRef view, std::size_t bytes,
                       std::string_view name) {
    return ActiveRange{view.contents(), view.gpu_address(), bytes, name, false};
}

ActiveRange write_range(BufferRef view, std::size_t bytes,
                        std::string_view name) {
    return ActiveRange{view.contents, view.gpu_address, bytes, name, true};
}

bool validate_dispatch(Context& context, std::uint32_t rows,
                       std::uint32_t n_experts, bool tiered,
                       std::uint32_t layer,
                       const PrefillMoeGroupingBuffers& buffers,
                       std::size_t& entries, std::size_t& count_tiles,
                       std::size_t& histogram_bytes,
                       std::size_t& tile_capacity,
                       std::size_t& tile_bytes, std::size_t& route_bytes,
                       std::size_t& mixed_bytes, std::size_t& gathered_bytes,
                       std::string& error) {
    error.clear();
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill MoE grouping row count exceeds the shader-safe range";
        return false;
    }
    if (n_experts != 256u && n_experts != 512u) {
        error = "Prefill MoE grouping supports 256 or 512 experts";
        return false;
    }
    if (layer >= kPrefillMoePlanLayers) {
        error = "Prefill MoE grouping layer is outside the PfPlan table";
        return false;
    }

    entries = static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    if (entries > std::numeric_limits<std::uint32_t>::max()) {
        error = "Prefill MoE grouping entry count exceeds shader indexing";
        return false;
    }
    count_tiles = round_up_div(entries, kCountThreads);
    tile_capacity =
        round_up_div(entries, 64u) + kPrefillMoeHistogramExperts;
    if (!checked_product({count_tiles, kPrefillMoeHistogramExperts,
                          sizeof(std::uint32_t)},
                         histogram_bytes) ||
        !checked_product({tile_capacity, kTileBytes}, tile_bytes) ||
        !checked_product({entries, kIntBytes}, route_bytes) ||
        !checked_product({rows, kGatherWidth, kHalfBytes}, mixed_bytes) ||
        !checked_product({entries, kGatherWidth, kHalfBytes},
                         gathered_bytes)) {
        error = "Prefill MoE grouping buffer size overflows size_t";
        return false;
    }

    std::size_t resident_count = 0u;
    std::size_t resident_bytes = 0u;
    std::size_t resident_layer_bytes = 0u;
    if (!checked_product({static_cast<std::size_t>(layer) + 1u, n_experts},
                         resident_count) ||
        !checked_product({resident_count, kIntBytes}, resident_bytes)) {
        error = "Prefill MoE grouping residency extent overflows size_t";
        return false;
    }
    if (!checked_product({n_experts, kIntBytes}, resident_layer_bytes)) {
        error = "Prefill MoE grouping residency layer extent overflows size_t";
        return false;
    }
    const std::size_t resident_required_bytes =
        tiered ? resident_bytes : kIntBytes;
    if (!validate_read_view(context, buffers.ids, route_bytes, kIntBytes,
                            "route IDs", error) ||
        !validate_read_view(context, buffers.resident, resident_required_bytes,
                            kIntBytes, "residency", error) ||
        !validate_read_view(context, buffers.mixed_h, mixed_bytes,
                            kGatherVectorBytes, "mixed activation", error) ||
        !validate_write_view(context, buffers.histogram, histogram_bytes,
                             alignof(std::uint32_t), "histogram", error) ||
        !validate_write_view(context, buffers.plans,
                             kPrefillMoePlanStrideBytes, alignof(std::uint32_t),
                             "PfPlan slot", error) ||
        !validate_write_view(context, buffers.tiles, tile_bytes,
                             alignof(std::uint64_t), "tile list", error) ||
        !validate_write_view(context, buffers.source_rows, route_bytes,
                             alignof(std::int32_t), "source rows", error) ||
        !validate_write_view(context, buffers.destination_rows, route_bytes,
                             alignof(std::int32_t), "destination rows", error) ||
        !validate_write_view(context, buffers.gathered_rows, gathered_bytes,
                             kGatherVectorBytes, "gathered rows", error)) {
        return false;
    }

    const auto resident_active =
        tiered ? buffers.resident.slice(
                     static_cast<std::size_t>(layer) * resident_layer_bytes,
                     resident_layer_bytes)
               : buffers.resident.slice(0u, kIntBytes);
    const std::array<ActiveRange, 3u> inputs{{
        read_range(buffers.ids, route_bytes, "route IDs"),
        read_range(resident_active, tiered ? resident_layer_bytes : 0u,
                   "residency"),
        read_range(buffers.mixed_h, mixed_bytes, "mixed activation"),
    }};
    const std::array<ActiveRange, 6u> outputs{{
        write_range(buffers.histogram, histogram_bytes, "histogram"),
        write_range(buffers.plans, kPrefillMoePlanStrideBytes, "PfPlan slot"),
        write_range(buffers.tiles, tile_bytes, "tile list"),
        write_range(buffers.source_rows, route_bytes, "source rows"),
        write_range(buffers.destination_rows, route_bytes, "destination rows"),
        write_range(buffers.gathered_rows, gathered_bytes, "gathered rows"),
    }};
    for (const auto& output : outputs) {
        for (const auto& input : inputs) {
            if (overlaps(output, input)) {
                error = "Prefill MoE grouping " + std::string(output.name) +
                        " overlaps active " + std::string(input.name);
                return false;
            }
        }
    }
    for (std::size_t i = 0u; i < outputs.size(); ++i) {
        for (std::size_t j = i + 1u; j < outputs.size(); ++j) {
            if (overlaps(outputs[i], outputs[j])) {
                error = "Prefill MoE grouping " +
                        std::string(outputs[i].name) +
                        " overlaps active " + std::string(outputs[j].name);
                return false;
            }
        }
    }
    return true;
}

bool encode_grouping(Context& context, Encoder& encoder, std::uint32_t rows,
                     std::uint32_t n_experts, bool tiered,
                     std::uint32_t layer,
                     const PrefillMoeGroupingBuffers& buffers,
                     std::string& error) {
    if (encoder.concurrent_group_open()) {
        error = "Prefill MoE grouping requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t entries = 0u;
    std::size_t count_tiles = 0u;
    std::size_t histogram_bytes = 0u;
    std::size_t tile_capacity = 0u;
    std::size_t tile_bytes = 0u;
    std::size_t route_bytes = 0u;
    std::size_t mixed_bytes = 0u;
    std::size_t gathered_bytes = 0u;
    if (!validate_dispatch(
            context, rows, n_experts, tiered, layer, buffers, entries,
            count_tiles, histogram_bytes, tile_capacity, tile_bytes,
            route_bytes, mixed_bytes, gathered_bytes, error)) {
        return false;
    }

    const PrefillMoeGroupingParams params{
        rows, n_experts, layer, tiered ? 1u : 0u};
    const auto ids = active_view(buffers.ids, route_bytes);
    const std::size_t resident_layer_bytes =
        static_cast<std::size_t>(n_experts) * kIntBytes;
    const auto resident =
        tiered ? buffers.resident.slice(
                     static_cast<std::size_t>(layer) * resident_layer_bytes,
                     resident_layer_bytes)
               : active_view(buffers.resident, kIntBytes);
    const auto mixed_h = active_view(buffers.mixed_h, mixed_bytes);
    const auto histogram = active_view(buffers.histogram, histogram_bytes);
    const auto plans =
        active_view(buffers.plans, kPrefillMoePlanStrideBytes);
    const auto tiles = active_view(buffers.tiles, tile_bytes);
    const auto source_rows = active_view(buffers.source_rows, route_bytes);
    const auto destination_rows =
        active_view(buffers.destination_rows, route_bytes);
    const auto gathered_rows =
        active_view(buffers.gathered_rows, gathered_bytes);
    const std::array<EncoderBufferBinding, 1u> count_outputs{{
        {histogram, 1u, histogram_bytes, alignof(std::uint32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 1u> count_inputs{{
        {ids, 0u, route_bytes, kIntBytes},
    }};
    const std::array<EncoderBytesBinding, 1u> count_bytes{{
        {&params, sizeof(params), 2u},
    }};
    constexpr std::array<EncoderThreadgroupMemory, 0u> no_threadgroup{};
    if (!encoder.dispatch(
            "strata_pf_sort_count", {count_tiles, 1u, 1u},
            {kCountThreads, 1u, 1u}, count_outputs, count_inputs,
            count_bytes, no_threadgroup, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 3u> scan_buffers{{
        {histogram, 0u, histogram_bytes, alignof(std::uint32_t)},
        {plans, 2u, kPrefillMoePlanStrideBytes,
         alignof(std::uint32_t)},
        {tiles, 3u, tile_bytes, alignof(std::uint64_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 1u> scan_inputs{{
        {resident, 1u, tiered ? resident_layer_bytes : kIntBytes, kIntBytes},
    }};
    const std::array<EncoderBytesBinding, 1u> scan_bytes{{
        {&params, sizeof(params), 4u},
    }};
    if (!encoder.dispatch(
            "strata_pf_sort_scan", {1u, 1u, 1u},
            {kScanThreads, 1u, 1u}, scan_buffers, scan_inputs,
            scan_bytes, no_threadgroup, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 2u> scatter_outputs{{
        {source_rows, 2u, route_bytes, alignof(std::int32_t)},
        {destination_rows, 3u, route_bytes, alignof(std::int32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 2u> scatter_inputs{{
        {ids, 0u, route_bytes, kIntBytes},
        {context.buffers().read_only(histogram), 1u, histogram_bytes,
         alignof(std::uint32_t)},
    }};
    const std::array<EncoderBytesBinding, 1u> scatter_bytes{{
        {&params, sizeof(params), 4u},
    }};
    if (!encoder.dispatch(
            "strata_pf_sort_scatter", {count_tiles, 1u, 1u},
            {kCountThreads, 1u, 1u}, scatter_outputs, scatter_inputs,
            scatter_bytes, no_threadgroup, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 1u> gather_outputs{{
        {gathered_rows, 3u, gathered_bytes, kGatherVectorBytes},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 3u> gather_inputs{{
        {mixed_h, 0u, mixed_bytes, kGatherVectorBytes},
        {context.buffers().read_only(source_rows), 1u, route_bytes,
         kIntBytes},
        {context.buffers().read_only(plans), 2u,
         kPrefillMoePlanBytes, alignof(std::uint32_t)},
    }};
    const std::array<EncoderBytesBinding, 1u> gather_bytes{{
        {&params, sizeof(params), 4u},
    }};
    const std::size_t gather_vectors =
        entries * (kGatherWidth / (kGatherVectorBytes / kHalfBytes));
    const std::size_t gather_groups =
        round_up_div(gather_vectors, kCountThreads);
    if (!encoder.dispatch(
            "strata_pf_gather_f16", {gather_groups, 1u, 1u},
            {kCountThreads, 1u, 1u}, gather_outputs, gather_inputs,
            gather_bytes, no_threadgroup, error)) {
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool encode_prefill_moe_grouping(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, bool tiered, std::uint32_t layer,
    const PrefillMoeGroupingBuffers& buffers, std::string& error) {
    return encode_grouping(context, encoder, rows, n_experts, tiered, layer,
                           buffers, error);
}

bool encode_prefill_moe_grouping(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, bool tiered, std::uint32_t layer,
    const PrefillMoeGroupingBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE grouping",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_grouping(context, encoder, rows, n_experts, tiered,
                                   layer, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
