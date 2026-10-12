// Validated Prefill MoE Q2_0 gate/up expert-tile encoder.
#include "strata/metal/prefill_moe_expert_dispatch.hpp"

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
constexpr std::size_t kTileColumnGroups = 20u;
constexpr std::size_t kModelWidth = 2560u;
constexpr std::size_t kHiddenWidth = 640u;
constexpr std::size_t kHalfBytes = sizeof(std::uint16_t);
constexpr std::size_t kQ2BlockBytes = 18u;
constexpr std::size_t kQ2BlocksPerRow = kModelWidth / 64u;
constexpr std::size_t kQ2ExpertBytes =
    kHiddenWidth * kQ2BlocksPerRow * kQ2BlockBytes;

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
        error = "Prefill MoE Q2_0 tile " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length() < required_bytes || alignment == 0u ||
        view.offset() % alignment != 0u) {
        error = "Prefill MoE Q2_0 tile " + std::string(name) +
                " view is undersized or misaligned";
        return false;
    }
    return true;
}

bool validate_write_view(Context& context, BufferRef view,
                         std::size_t required_bytes, std::size_t alignment,
                         std::string_view name, std::string& error) {
    if (!view || !context.buffers().owns(view)) {
        error = "Prefill MoE Q2_0 tile " + std::string(name) +
                " view is not owned by this Context";
        return false;
    }
    if (view.length < required_bytes || alignment == 0u ||
        view.offset % alignment != 0u) {
        error = "Prefill MoE Q2_0 tile " + std::string(name) +
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

bool validate_dispatch(Context& context, std::uint32_t rows,
                       std::uint32_t n_experts, std::uint32_t layer,
                       const PrefillMoeExpertTileBuffers& buffers,
                       std::size_t& address_bytes,
                       std::size_t& selected_address_offset,
                       std::size_t& gathered_bytes, std::size_t& plan_bytes,
                       std::size_t& tile_capacity,
                       std::size_t& tile_bytes, std::size_t& hidden_bytes,
                       bool& requires_tiered_lease,
                       std::array<PrefillMoeExpertAddress, 512u>&
                           host_address_snapshot,
                       std::array<std::uint8_t, 512u>& tiered_lease_mask,
                       std::string& error) {
    if (rows == 0u || rows > kPrefillMaxShaderRows) {
        error = "Prefill MoE Q2_0 tile row count exceeds the shader-safe range";
        return false;
    }
    if (n_experts != 256u && n_experts != 512u) {
        error = "Prefill MoE Q2_0 tile supports 256 or 512 experts";
        return false;
    }
    if (layer >= kPrefillMoePlanLayers) {
        error = "Prefill MoE Q2_0 tile layer is outside the PfPlan table";
        return false;
    }

    const std::size_t entries =
        static_cast<std::size_t>(rows) * kPrefillMoeTopK;
    if (entries > std::numeric_limits<std::uint32_t>::max()) {
        error = "Prefill MoE Q2_0 tile entries exceed shader indexing";
        return false;
    }
    tile_capacity =
        round_up_div(entries, kTileRows) +
        std::min<std::size_t>(n_experts, entries);
    if (!checked_product(
            {static_cast<std::size_t>(layer) + 1u, n_experts,
             sizeof(PrefillMoeExpertAddress)},
            address_bytes) ||
        !checked_product({static_cast<std::size_t>(layer), n_experts,
                         sizeof(PrefillMoeExpertAddress)},
                        selected_address_offset) ||
        !checked_product({entries, kModelWidth, kHalfBytes},
                         gathered_bytes) ||
        !checked_product({tile_capacity, sizeof(std::uint32_t) * 2u},
                         tile_bytes) ||
        !checked_product({entries, kHiddenWidth, kHalfBytes},
                         hidden_bytes)) {
        error = "Prefill MoE Q2_0 tile buffer size overflows size_t";
        return false;
    }
    plan_bytes = kPrefillMoePlanBytes;

    if (!validate_read_view(context, buffers.addresses, address_bytes,
                            alignof(PrefillMoeExpertAddress),
                            "address table", error) ||
        !validate_write_view(
            context, buffers.address_snapshot,
            static_cast<std::size_t>(n_experts) *
                sizeof(PrefillMoeExpertAddress),
            alignof(PrefillMoeExpertAddress), "address snapshot", error) ||
        !validate_read_view(context, buffers.gathered_rows, gathered_bytes,
                            16u, "gathered input", error) ||
        !validate_read_view(context, buffers.plan, plan_bytes,
                            alignof(std::uint32_t), "PfPlan slot", error) ||
        !validate_read_view(context, buffers.tiles, tile_bytes,
                            alignof(std::uint64_t), "tile list", error) ||
        !validate_write_view(context, buffers.hidden, hidden_bytes,
                             kHalfBytes, "SwiGLU output", error) ||
        !validate_write_view(context, buffers.status, sizeof(std::uint32_t),
                             alignof(std::uint32_t), "status", error)) {
        return false;
    }
    if (buffers.status.contents == nullptr) {
        error = "Prefill MoE Q2_0 tile status buffer is not host-writable";
        return false;
    }

    const ReadOnlyBufferRef selected_addresses = buffers.addresses.slice(
        selected_address_offset,
        static_cast<std::size_t>(n_experts) *
            sizeof(PrefillMoeExpertAddress));
    if (!selected_addresses || selected_addresses.contents() == nullptr) {
        error = "Prefill MoE Q2_0 tile selected address row is not host-readable";
        return false;
    }
    const ActiveRange hidden_output =
        write_range(buffers.hidden, hidden_bytes, "SwiGLU output");
    const ActiveRange status_output =
        write_range(buffers.status, sizeof(std::uint32_t), "status");
    const ActiveRange address_snapshot_output = write_range(
        buffers.address_snapshot,
        static_cast<std::size_t>(n_experts) *
            sizeof(PrefillMoeExpertAddress),
        "address snapshot");
    if (overlaps(hidden_output, status_output) ||
        overlaps(address_snapshot_output, hidden_output) ||
        overlaps(address_snapshot_output, status_output) ||
        overlaps(address_snapshot_output,
                 read_range(buffers.addresses, buffers.addresses.length(),
                            "address table"))) {
        error = "Prefill MoE Q2_0 tile address snapshot overlaps an active "
                "address, input or output range";
        return false;
    }
    const std::array inputs{
        read_range(selected_addresses,
                   static_cast<std::size_t>(n_experts) *
                       sizeof(PrefillMoeExpertAddress),
                   "selected address row"),
        read_range(buffers.gathered_rows, gathered_bytes, "gathered input"),
        read_range(buffers.plan, plan_bytes, "PfPlan slot"),
        read_range(buffers.tiles, tile_bytes, "tile list"),
    };
    for (const ActiveRange& input : inputs) {
        if (overlaps(hidden_output, input) ||
            overlaps(status_output, input) ||
            overlaps(address_snapshot_output, input)) {
            error = "Prefill MoE Q2_0 tile address snapshot or output overlaps "
                    "active " +
                    std::string(input.name);
            return false;
        }
    }

    if (buffers.weight_storage.empty()) {
        error = "Prefill MoE Q2_0 tile has no weight storage views";
        return false;
    }
    for (const PrefillMoeWeightStorageView& weight_storage :
         buffers.weight_storage) {
        if (!validate_read_view(context, weight_storage.buffer, 1u,
                                alignof(std::uint16_t), "Q2_0 weight storage",
                                error) ||
            weight_storage.buffer.contents() == nullptr ||
            weight_storage.buffer.gpu_address() == 0u) {
            if (error.empty()) {
                error = "Prefill MoE Q2_0 tile weight storage has no GPU "
                        "address or host mapping";
            }
            return false;
        }
        if (overlaps(address_snapshot_output,
                     read_range(weight_storage.buffer,
                                weight_storage.buffer.length(),
                                "Q2_0 weight storage"))) {
            error = "Prefill MoE Q2_0 tile address snapshot overlaps expert "
                    "weight storage";
            return false;
        }
    }

    const std::size_t selected_address_bytes =
        static_cast<std::size_t>(n_experts) *
        sizeof(PrefillMoeExpertAddress);
    std::memcpy(host_address_snapshot.data(), selected_addresses.contents(),
                selected_address_bytes);
    const auto* const expert_addresses = host_address_snapshot.data();
    tiered_lease_mask.fill(1u);
    for (std::uint32_t expert = 0u; expert < n_experts; ++expert) {
        const PrefillMoeExpertAddress& expert_address =
            expert_addresses[expert];
        if (expert_address.role[0] == 0u &&
            expert_address.role[1] == 0u) {
            continue;
        }
        if (expert_address.role[0] == 0u ||
            expert_address.role[1] == 0u) {
            error = "Prefill MoE Q2_0 tile has only one gate/up address";
            return false;
        }
        bool has_tiered_address = false;
        bool all_tiered_addresses_are_pinned = true;
        for (const std::uint64_t address :
             {expert_address.role[0], expert_address.role[1]}) {
            if (address % alignof(std::uint16_t) != 0u) {
                error = "Prefill MoE Q2_0 tile weight address is misaligned";
                return false;
            }
            bool backed = false;
            for (const PrefillMoeWeightStorageView& weight_storage :
                 buffers.weight_storage) {
                std::size_t offset = 0u;
                if (!contains_gpu_range(weight_storage.buffer, address,
                                        kQ2ExpertBytes, offset)) {
                    continue;
                }
                const auto cpu_base = reinterpret_cast<std::uintptr_t>(
                    weight_storage.buffer.contents());
                if (offset >
                    std::numeric_limits<std::uintptr_t>::max() - cpu_base) {
                    error = "Prefill MoE Q2_0 tile weight host range overflows";
                    return false;
                }
                const ActiveRange weight{
                    reinterpret_cast<const void*>(cpu_base + offset), address,
                    kQ2ExpertBytes, "Q2_0 expert weights"};
                if (overlaps(hidden_output, weight) ||
                    overlaps(status_output, weight)) {
                    error = "Prefill MoE Q2_0 tile output overlaps active "
                            "expert weights";
                    return false;
                }
                if (weight_storage.buffer.is_tiered_slot_arena()) {
                    has_tiered_address = true;
                    requires_tiered_lease = true;
                    all_tiered_addresses_are_pinned &=
                        buffers.tiered_leases != nullptr &&
                        buffers.tiered_leases->pins(layer, address,
                                                    kQ2ExpertBytes);
                }
                backed = true;
                break;
            }
            if (!backed) {
                error = "Prefill MoE Q2_0 tile weight address is outside "
                        "provided storage";
                return false;
            }
        }
        if (has_tiered_address && !all_tiered_addresses_are_pinned) {
            tiered_lease_mask[expert] = 0u;
        }
    }
    return true;
}

bool encode_impl(Context& context, Encoder& encoder, std::uint32_t rows,
                 std::uint32_t n_experts, std::uint32_t layer,
                 const PrefillMoeExpertTileBuffers& buffers,
                 std::string& error) {
    error.clear();
    if (encoder.concurrent_group_open()) {
        error = "Prefill MoE Q2_0 tile requires an ordered concurrent-group "
                "boundary";
        return false;
    }

    std::size_t address_bytes = 0u;
    std::size_t selected_address_offset = 0u;
    std::size_t gathered_bytes = 0u;
    std::size_t plan_bytes = 0u;
    std::size_t tile_capacity = 0u;
    std::size_t tile_bytes = 0u;
    std::size_t hidden_bytes = 0u;
    bool requires_tiered_lease = false;
    std::array<PrefillMoeExpertAddress, 512u> address_snapshot{};
    std::array<std::uint8_t, 512u> tiered_lease_mask{};
    if (!validate_dispatch(context, rows, n_experts, layer, buffers,
                           address_bytes, selected_address_offset,
                           gathered_bytes, plan_bytes, tile_capacity,
                           tile_bytes, hidden_bytes, requires_tiered_lease,
                           address_snapshot,
                           tiered_lease_mask, error)) {
        return false;
    }
    if (requires_tiered_lease &&
        (!buffers.tiered_leases ||
         !buffers.tiered_leases->prepared_for(encoder))) {
        error = "Prefill MoE Q2_0 tile requires the matching prepared "
                "ExpertLeaseSet";
        return false;
    }

    const std::size_t selected_address_bytes =
        static_cast<std::size_t>(n_experts) *
        sizeof(PrefillMoeExpertAddress);
    if (!encoder.claim_unique_staging_range(
            buffers.address_snapshot, selected_address_bytes,
            alignof(PrefillMoeExpertAddress), error)) {
        return false;
    }
    std::memcpy(buffers.address_snapshot.contents, address_snapshot.data(),
                selected_address_bytes);
    const ReadOnlyBufferRef addresses =
        context.buffers().read_only(
            active_view(buffers.address_snapshot, selected_address_bytes));
    if (!addresses) {
        error = "cannot create Prefill MoE Q2_0 tile address snapshot view";
        return false;
    }
    const auto gathered = active_view(buffers.gathered_rows, gathered_bytes);
    const auto plan = active_view(buffers.plan, plan_bytes);
    const auto tiles = active_view(buffers.tiles, tile_bytes);
    const auto hidden = active_view(buffers.hidden, hidden_bytes);
    const auto status =
        active_view(buffers.status, sizeof(std::uint32_t));
    const PrefillMoeTileParams params{
        rows, n_experts, static_cast<std::uint32_t>(tile_capacity)};
    std::vector<EncoderResourceBinding> weight_resources;
    weight_resources.reserve(buffers.weight_storage.size());
    for (const PrefillMoeWeightStorageView& weight_storage :
         buffers.weight_storage) {
        weight_resources.push_back({weight_storage.buffer, 1u,
                                    alignof(std::uint16_t)});
    }
    if (!encoder.use_resources(weight_resources, error)) return false;
    const std::array<EncoderBufferBinding, 2u> outputs{{
        {hidden, 4u, hidden_bytes, kHalfBytes},
        {status, 6u, sizeof(std::uint32_t), alignof(std::uint32_t)},
    }};
    const std::array<EncoderReadOnlyBufferBinding, 4u> inputs{{
        {addresses, 0u, selected_address_bytes,
         alignof(PrefillMoeExpertAddress)},
        {gathered, 1u, gathered_bytes, 16u},
        {plan, 2u, plan_bytes, alignof(std::uint32_t)},
        {tiles, 3u, tile_bytes, alignof(std::uint64_t)},
    }};
    constexpr std::array<EncoderThreadgroupMemory, 0u> no_threadgroup{};
    const std::array<EncoderReadOnlyBufferBinding, 2u> validation_inputs{{
        {plan, 0u, plan_bytes, alignof(std::uint32_t)},
        {tiles, 1u, tile_bytes, alignof(std::uint64_t)},
    }};
    const EncoderBufferBinding validation_status{
        status, 2u, sizeof(std::uint32_t), alignof(std::uint32_t)};
    const EncoderBytesBinding validation_params{&params, sizeof(params), 3u};
    if (!encoder.dispatch(
            "strata_pf_moe_plan_validate", {1u, 1u, 1u}, {128u, 1u, 1u},
            std::span<const EncoderBufferBinding>(&validation_status, 1u),
            validation_inputs,
            std::span<const EncoderBytesBinding>(&validation_params, 1u),
            no_threadgroup, error)) {
        return false;
    }
    const std::array<EncoderBytesBinding, 2u> expert_bytes{{
        {&params, sizeof(params), 5u},
        {tiered_lease_mask.data(), n_experts, 7u},
    }};
    if (!encoder.dispatch(
        "strata_pf_moe_gu_q2_0",
        {kTileColumnGroups, tile_capacity, 1u}, {kThreads, 1u, 1u},
        outputs, inputs, expert_bytes, no_threadgroup, error)) {
        return false;
    }
    return true;
}

}  // namespace

bool encode_prefill_moe_gu_q2_0(
    Context& context, Encoder& encoder, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeExpertTileBuffers& buffers, std::string& error) {
    return encode_impl(context, encoder, rows, n_experts, layer, buffers,
                       error);
}

bool encode_prefill_moe_gu_q2_0(
    Context& context, CommandBuffer& command_buffer, std::uint32_t rows,
    std::uint32_t n_experts, std::uint32_t layer,
    const PrefillMoeExpertTileBuffers& buffers, std::string& error) {
    return Encoder::encode(
        context, command_buffer, "Prefill MoE Q2_0 gate/up tiles",
        [&](Encoder& encoder, std::string& encode_error) {
            return encode_impl(context, encoder, rows, n_experts, layer,
                               buffers, encode_error);
        },
        error);
}

}  // namespace strata::metal
