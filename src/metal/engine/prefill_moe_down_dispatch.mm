// Validated Prefill MoE F16-activation down-tile encoder.
#include "strata/metal/prefill_moe_down_dispatch.hpp"

#include "strata/metal/encoder.hpp"
#include "strata/metal/experts.hpp"
#include "strata/metal/prefill.hpp"
#include "strata/metal/prefill_moe_grouping.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::size_t kThreads = 128u;
constexpr std::size_t kTileRows = 64u;
constexpr std::size_t kTileColumnGroups = 40u;
constexpr std::size_t kModelWidth = 2560u;
constexpr std::size_t kHiddenWidth = 640u;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kQ2BlockBytes = 18u;
constexpr std::size_t kQ2BlocksPerRow = kHiddenWidth / 64u;
constexpr std::size_t kIQ4NLBlocksPerRow = kHiddenWidth / 32u;

enum class WeightFormat : std::uint32_t { q2_0 = 0u, iq4_nl = 1u };

struct PrefillMoeExpertAddress {
    std::uint64_t role[3]{};
};
static_assert(sizeof(PrefillMoeExpertAddress) == 24u);

struct PrefillMoeTileParams {
    std::uint32_t rows;
    std::uint32_t n_expert;
    std::uint32_t tile_capacity;
};
static_assert(sizeof(PrefillMoeTileParams) == 12u);

struct ActiveRange {
    const void* cpu = nullptr;
    std::uint64_t gpu = 0u;
    std::size_t bytes = 0u;
    std::string_view name;
};

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
    return value / divisor +
           static_cast<std::size_t>(value % divisor != 0u);
}

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

bool contains_gpu_range(ReadOnlyBufferRef view, std::uint64_t address,
                        std::size_t bytes, std::size_t& offset) {
    const std::uint64_t base = view.gpu_address();
    if (base == 0u || address < base) return false;
    const std::uint64_t distance = address - base;
    if (distance > view.length() ||
        bytes > view.length() - static_cast<std::size_t>(distance)) {
        return false;
    }
    offset = static_cast<std::size_t>(distance);
    return true;
}

ActiveRange read_range(ReadOnlyBufferRef view, std::size_t bytes,
                       std::string_view name) {
    return ActiveRange{view.contents(), view.gpu_address(), bytes, name};
}

ActiveRange write_range(BufferRef view, std::size_t bytes,
                        std::string_view name) {
    return ActiveRange{view.contents, view.gpu_address, bytes, name};
}

bool validate_read_view(Context& context, ReadOnlyBufferRef view,
                        std::size_t required_bytes, std::size_t alignment,
                        std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE down " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill MoE down " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE down " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill MoE down " + std::string(name) +
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

bool validate_dispatch(
    Context& context, std::uint32_t rows, std::uint32_t n_experts,
    std::uint32_t layer, WeightFormat format,
    const PrefillMoeDownTileBuffers& buffers,
    std::size_t& selected_address_offset, std::size_t& address_bytes,
    std::size_t& selected_address_bytes, std::size_t& weight_expert_bytes,
    std::size_t& hidden_bytes,
    std::size_t& plan_bytes, std::size_t& tile_capacity,
    std::size_t& tile_bytes, std::size_t& destination_bytes,
    std::size_t& destination_seen_bytes, std::size_t& parts_bytes,
    bool& requires_tiered_lease,
    std::array<PrefillMoeExpertAddress, 512u>& host_address_snapshot,
    std::array<std::uint8_t, 512u>& tiered_lease_mask,
    std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill MoE down row count exceeds the shader-safe range";
        return false;
    }
    if (n_experts != 256u && n_experts != 512u) {
        error = "Prefill MoE down supports 256 or 512 experts";
        return false;
    }
    if (layer >= kPrefillMoePlanLayers) {
        error = "Prefill MoE down layer is outside the PfPlan table";
        return false;
    }

    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    if (entries > std::numeric_limits<std::uint32_t>::max()) {
        error = "Prefill MoE down entries exceed shader indexing";
        return false;
    }
    tile_capacity =
        round_up_div(entries, kTileRows) +
        std::min<std::size_t>(n_experts, entries);
    const std::size_t weight_blocks_per_row =
        format == WeightFormat::q2_0 ? kQ2BlocksPerRow : kIQ4NLBlocksPerRow;
    if (!checked_product(
            {static_cast<std::size_t>(layer) + 1u, n_experts,
             sizeof(PrefillMoeExpertAddress)},
            address_bytes) ||
        !checked_product({static_cast<std::size_t>(layer), n_experts,
                         sizeof(PrefillMoeExpertAddress)},
                        selected_address_offset) ||
        !checked_product({n_experts, sizeof(PrefillMoeExpertAddress)},
                        selected_address_bytes) ||
        !checked_product({kModelWidth, weight_blocks_per_row, kQ2BlockBytes},
                         weight_expert_bytes) ||
        !checked_product({entries, kHiddenWidth, kHalfBytes}, hidden_bytes) ||
        !checked_product({tile_capacity, sizeof(std::uint32_t) * 2u},
                         tile_bytes) ||
        !checked_product({entries, sizeof(std::uint32_t)},
                         destination_bytes) ||
        !checked_product({round_up_div(entries, 32u), sizeof(std::uint32_t)},
                         destination_seen_bytes) ||
        !checked_product({entries, kModelWidth, kFloatBytes}, parts_bytes)) {
        error = "Prefill MoE down buffer size overflows size_t";
        return false;
    }
    if (destination_seen_bytes > parts_bytes) {
        error = "Prefill MoE down destination validation scratch exceeds "
                "the output buffer";
        return false;
    }
    plan_bytes = kPrefillMoePlanBytes;

    const std::size_t snapshot_bytes = selected_address_bytes;
    if (!validate_read_view(context, buffers.addresses, address_bytes,
                            alignof(PrefillMoeExpertAddress),
                            "address table", error) ||
        !validate_write_view(context, buffers.address_snapshot,
                             snapshot_bytes,
                             alignof(PrefillMoeExpertAddress),
                             "address snapshot", error) ||
        !validate_read_view(context, buffers.hidden, hidden_bytes, 16u,
                            "hidden input", error) ||
        !validate_read_view(context, buffers.plan, plan_bytes,
                            alignof(std::uint32_t), "PfPlan slot", error) ||
        !validate_read_view(context, buffers.tiles, tile_bytes,
                            alignof(std::uint64_t), "tile list", error) ||
        !validate_read_view(context, buffers.destinations, destination_bytes,
                            alignof(std::uint32_t), "destination map", error) ||
        !validate_write_view(context, buffers.parts, parts_bytes,
                             alignof(float), "down output", error) ||
        !validate_write_view(context, buffers.status, sizeof(std::uint32_t),
                             alignof(std::uint32_t), "status", error)) {
        return false;
    }
    if (buffers.status.contents == nullptr) {
        error = "Prefill MoE down status buffer is not host-writable";
        return false;
    }

    const ReadOnlyBufferRef selected_addresses = buffers.addresses.slice(
        selected_address_offset, selected_address_bytes);
    if (!selected_addresses || selected_addresses.contents() == nullptr) {
        error = "Prefill MoE down selected address row is not host-readable";
        return false;
    }

    const ActiveRange parts_output =
        write_range(buffers.parts, parts_bytes, "down output");
    const ActiveRange status_output =
        write_range(buffers.status, sizeof(std::uint32_t), "status");
    const ActiveRange address_snapshot_output = write_range(
        buffers.address_snapshot, snapshot_bytes, "address snapshot");
    if (overlaps(parts_output, status_output) ||
        overlaps(address_snapshot_output, parts_output) ||
        overlaps(address_snapshot_output, status_output) ||
        overlaps(address_snapshot_output,
                 read_range(buffers.addresses, buffers.addresses.length(),
                            "address table"))) {
        error = "Prefill MoE down address snapshot or output aliases an "
                "active input or output";
        return false;
    }
    const std::array inputs{
        read_range(selected_addresses, selected_address_bytes,
                   "selected address row"),
        read_range(buffers.hidden, hidden_bytes, "hidden input"),
        read_range(buffers.plan, plan_bytes, "PfPlan slot"),
        read_range(buffers.tiles, tile_bytes, "tile list"),
        read_range(buffers.destinations, destination_bytes,
                   "destination map"),
    };
    for (const ActiveRange& input : inputs) {
        if (overlaps(parts_output, input) ||
            overlaps(status_output, input) ||
            overlaps(address_snapshot_output, input)) {
            error = "Prefill MoE down address snapshot or output overlaps "
                    "active " +
                    std::string(input.name);
            return false;
        }
    }

    if (buffers.weight_storage.empty()) {
        error = "Prefill MoE down has no weight storage views";
        return false;
    }
    for (const PrefillMoeWeightStorageView& weight_storage :
         buffers.weight_storage) {
        if (!validate_read_view(context, weight_storage.buffer, 1u,
                                alignof(std::uint16_t), "weight storage",
                                error) ||
            weight_storage.buffer.contents() == nullptr ||
            weight_storage.buffer.gpu_address() == 0u) {
            if (error.empty()) {
                error = "Prefill MoE down weight storage has no GPU "
                        "address or host mapping";
            }
            return false;
        }
        if (overlaps(address_snapshot_output,
                     read_range(weight_storage.buffer,
                                weight_storage.buffer.length(),
                                "weight storage"))) {
            error = "Prefill MoE down address snapshot overlaps expert "
                    "weight storage";
            return false;
        }
    }

    std::memcpy(host_address_snapshot.data(), selected_addresses.contents(),
                selected_address_bytes);
    tiered_lease_mask.fill(1u);
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        const std::uint64_t address = host_address_snapshot[expert].role[2];
        if (address == 0u) continue;
        if (address % alignof(std::uint16_t) != 0u) {
            error = "Prefill MoE down weight address is misaligned";
            return false;
        }

        bool backed = false;
        bool address_is_tiered = false;
        for (const PrefillMoeWeightStorageView& weight_storage :
             buffers.weight_storage) {
            std::size_t offset = 0u;
            if (!contains_gpu_range(weight_storage.buffer, address,
                                    weight_expert_bytes, offset)) {
                continue;
            }
            const auto cpu_base = reinterpret_cast<std::uintptr_t>(
                weight_storage.buffer.contents());
            if (offset >
                std::numeric_limits<std::uintptr_t>::max() - cpu_base) {
                error = "Prefill MoE down weight host range overflows";
                return false;
            }
            const ActiveRange weight{
                reinterpret_cast<const void*>(cpu_base + offset), address,
                weight_expert_bytes, "Prefill down weights"};
            if (overlaps(parts_output, weight) ||
                overlaps(status_output, weight)) {
                error = "Prefill MoE down output overlaps active "
                        "expert weights";
                return false;
            }
            address_is_tiered = weight_storage.buffer.is_tiered_slot_arena();
            backed = true;
            break;
        }
        if (!backed) {
            error = "Prefill MoE down weight address is outside "
                    "provided storage";
            return false;
        }
        if (address_is_tiered) {
            requires_tiered_lease = true;
            tiered_lease_mask[expert] =
                buffers.tiered_leases != nullptr &&
                        buffers.tiered_leases->pins(layer, address,
                                                    weight_expert_bytes)
                    ? 1u
                    : 0u;
        }
    }
    return true;
}

bool encode_impl(Context& context, Encoder& encoder, std::uint32_t rows,
                 std::uint32_t n_experts, std::uint32_t layer,
                 WeightFormat format,
                 const PrefillMoeDownTileBuffers& buffers,
                 std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill MoE down requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t selected_address_offset = 0u;
    std::size_t address_bytes = 0u;
    std::size_t selected_address_bytes = 0u;
    std::size_t weight_expert_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    std::size_t plan_bytes = 0u;
    std::size_t tile_capacity = 0u;
    std::size_t tile_bytes = 0u;
    std::size_t destination_bytes = 0u;
    std::size_t destination_seen_bytes = 0u;
    std::size_t parts_bytes = 0u;
    bool requires_tiered_lease = false;
    std::array<PrefillMoeExpertAddress, 512u> address_snapshot{};
    std::array<std::uint8_t, 512u> tiered_lease_mask{};
    if (!validate_dispatch(
            context, rows, n_experts, layer, format, buffers,
            selected_address_offset, address_bytes, selected_address_bytes,
            weight_expert_bytes, hidden_bytes, plan_bytes, tile_capacity,
            tile_bytes,
            destination_bytes, destination_seen_bytes, parts_bytes,
            requires_tiered_lease,
            address_snapshot, tiered_lease_mask, error)) {
        return false;
    }
    if (requires_tiered_lease &&
        (!buffers.tiered_leases ||
         !buffers.tiered_leases->prepared_for(encoder))) {
        error = "Prefill MoE down requires the matching prepared "
                "ExpertLeaseSet";
        return false;
    }

    const std::size_t snapshot_bytes = selected_address_bytes;
    if (!encoder.claim_unique_staging_range(
            buffers.address_snapshot, snapshot_bytes,
            alignof(PrefillMoeExpertAddress), error)) {
        return false;
    }
    std::memcpy(buffers.address_snapshot.contents, address_snapshot.data(),
                snapshot_bytes);
    const ReadOnlyBufferRef addresses = context.buffers().read_only(
        active_view(buffers.address_snapshot, snapshot_bytes));
    if (!addresses) {
        error = "cannot create Prefill MoE down address snapshot view";
        return false;
    }

    const auto hidden = active_view(buffers.hidden, hidden_bytes);
    const auto plan = active_view(buffers.plan, plan_bytes);
    const auto tiles = active_view(buffers.tiles, tile_bytes);
    const auto destinations =
        active_view(buffers.destinations, destination_bytes);
    const auto parts = active_view(buffers.parts, parts_bytes);
    const auto status = active_view(buffers.status, sizeof(std::uint32_t));
    const PrefillMoeTileParams params{
        rows, n_experts, static_cast<std::uint32_t>(tile_capacity)};

    std::vector<EncoderResourceBinding> weight_resources;
    weight_resources.reserve(buffers.weight_storage.size());
    for (const PrefillMoeWeightStorageView& weight_storage :
         buffers.weight_storage) {
        weight_resources.push_back(
            {weight_storage.buffer, 1u, alignof(std::uint16_t)});
    }
    if (!encoder.use_resources(weight_resources, error)) return false;

    const std::array<EncoderBufferBinding, 1u> validation_outputs{{
        {status, 2u, sizeof(std::uint32_t), alignof(std::uint32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 2u> validation_inputs{{
        {plan, 0u, plan_bytes, alignof(std::uint32_t)},
        {tiles, 1u, tile_bytes, alignof(std::uint64_t)},
    }};
    const EncoderBytesBinding validation_params{&params, sizeof(params), 3u};
    constexpr std::array<EncoderThreadgroupMemory, 0u> no_threadgroup{};
    if (!encoder.dispatch(
            "strata_pf_moe_plan_validate", {1u, 1u, 1u}, {128u, 1u, 1u},
            validation_outputs, validation_inputs,
            std::span<const EncoderBytesBinding>(&validation_params, 1u),
            no_threadgroup, error)) {
        return false;
    }

    const BufferRef destination_seen =
        active_view(buffers.parts, destination_seen_bytes);
    const std::array<EncoderBufferBinding, 1u> clear_outputs{{
        {destination_seen, 0u, destination_seen_bytes,
         alignof(std::uint32_t)},
    }};
    const EncoderBytesBinding clear_params{
        &params, sizeof(params), 1u};
    const std::size_t seen_words =
        destination_seen_bytes / sizeof(std::uint32_t);
    if (!encoder.dispatch(
            "strata_pf_moe_destination_clear",
            {round_up_div(seen_words, kThreads), 1u, 1u}, {kThreads, 1u, 1u},
            clear_outputs, {}, std::span<const EncoderBytesBinding>(
                                 &clear_params, 1u),
            no_threadgroup, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 2u> destination_validation_outputs{{
        {destination_seen, 1u, destination_seen_bytes,
         alignof(std::uint32_t)},
        {status, 2u, sizeof(std::uint32_t), alignof(std::uint32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 1u>
        destination_validation_inputs{{
            {destinations, 0u, destination_bytes,
             alignof(std::uint32_t)},
        }};
    const EncoderBytesBinding destination_validation_params{
        &params, sizeof(params), 3u};
    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    if (!encoder.dispatch(
            "strata_pf_moe_destination_validate",
            {round_up_div(entries, kThreads), 1u, 1u},
            {kThreads, 1u, 1u}, destination_validation_outputs,
            destination_validation_inputs,
            std::span<const EncoderBytesBinding>(
                &destination_validation_params, 1u),
            no_threadgroup, error)) {
        return false;
    }

    const std::array<EncoderBufferBinding, 2u> outputs{{
        {parts, 5u, parts_bytes, alignof(float)},
        {status, 7u, sizeof(std::uint32_t), alignof(std::uint32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 5u> inputs{{
        {addresses, 0u, selected_address_bytes,
         alignof(PrefillMoeExpertAddress)},
        {hidden, 1u, hidden_bytes, 16u},
        {plan, 2u, plan_bytes, alignof(std::uint32_t)},
        {tiles, 3u, tile_bytes, alignof(std::uint64_t)},
        {destinations, 4u, destination_bytes, alignof(std::uint32_t)},
    }};
    const std::uint32_t weight_format =
        static_cast<std::uint32_t>(format);
    const std::array<EncoderBytesBinding, 3u> expert_bytes{{
        {&params, sizeof(params), 6u},
        {tiered_lease_mask.data(), n_experts, 8u},
        {&weight_format, sizeof(weight_format), 9u},
    }};
    if (!encoder.dispatch(
            "strata_pf_moe_down_f16",
            {kTileColumnGroups, tile_capacity, 1u}, {kThreads, 1u, 1u},
            outputs, inputs, expert_bytes, no_threadgroup, error)) {
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_moe_down_q2_0(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error) {
    return encode_impl(context, encoder, rows, n_experts, layer,
                       WeightFormat::q2_0, buffers, error);
}

bool encode_prefill_moe_down_iq4_nl(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error) {
    return encode_impl(context, encoder, rows, n_experts, layer,
                       WeightFormat::iq4_nl, buffers, error);
}

bool encode_prefill_moe_down_q2_0(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE down tiles",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_impl(context, encoder, rows, n_experts, layer,
                               WeightFormat::q2_0, buffers, encode_error);
        },
        error);
}

bool encode_prefill_moe_down_iq4_nl(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeDownTileBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE IQ4_NL down tiles",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_impl(context, encoder, rows, n_experts, layer,
                               WeightFormat::iq4_nl, buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
