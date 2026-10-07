// src/metal/engine/tiered.cpp - validate and lay out tiered dense/resident sources.
#include "strata/metal/tiered.hpp"

#include <algorithm>
#include <charconv>
#include <filesystem>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <string_view>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::uint64_t kDenseTensorAlignment = 256;
constexpr std::uint64_t kResidentExpertAlignment = 256;
constexpr std::uint64_t kDenseBaseAlignment = 16 * 1024;
constexpr std::uint64_t kDenseArenaLengthAlignment = 16 * 1024;
constexpr std::uint32_t kExpertRoles = 3;
constexpr std::size_t kAddressEntries =
    static_cast<std::size_t>(kTieredMaxLayers) * kTieredMaxExperts * kExpertRoles;
constexpr std::size_t kResidentEntries =
    static_cast<std::size_t>(kTieredMaxLayers) * kTieredMaxExperts;

struct ExpertIdentity {
    std::uint32_t layer = 0;
    std::uint32_t role = 0;
};

struct ExpertDraft {
    std::array<const WindowTensorSource*, kExpertRoles> tensors{};
    std::array<std::size_t, kExpertRoles> shards{};
};

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    const std::uint64_t remainder = value & (alignment - 1);
    if (remainder == 0) {
        result = value;
        return true;
    }
    return add(value, alignment - remainder, result);
}

std::optional<ExpertIdentity> parse_expert_name(std::string_view name) {
    if (!name.starts_with("blk.")) return std::nullopt;
    const std::size_t layer_end = name.find(".ffn_", 4);
    if (layer_end == std::string_view::npos || layer_end == 4) return std::nullopt;
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(name.data() + 4, name.data() + layer_end, layer);
    if (parsed.ec != std::errc{} || parsed.ptr != name.data() + layer_end) return std::nullopt;
    const std::string_view role = name.substr(layer_end + 1);
    if (role == "ffn_gate_exps.weight") return ExpertIdentity{layer, 0};
    if (role == "ffn_up_exps.weight") return ExpertIdentity{layer, 1};
    if (role == "ffn_down_exps.weight") return ExpertIdentity{layer, 2};
    return std::nullopt;
}

bool is_expert_role_name(std::string_view name) {
    return name.find(".ffn_") != std::string_view::npos &&
           name.ends_with("_exps.weight");
}

bool is_router_name(std::string_view name) {
    return name.ends_with(".ffn_gate_inp.weight") ||
           name.ends_with(".ffn_gate_inp_shexp.weight");
}

bool source_range_valid(std::uint64_t offset, std::uint64_t bytes,
                        std::uint64_t file_size) {
    return bytes != 0 && offset <= file_size && bytes <= file_size - offset;
}

bool address_range_valid(std::uint64_t base, std::uint64_t offset,
                         std::uint64_t bytes, std::uint64_t& address) {
    std::uint64_t end = 0;
    return base != 0 && bytes != 0 && add(base, offset, address) &&
           add(address, bytes, end);
}

bool append_dense_copy(const std::string& name, bool from_pack, std::size_t shard,
                       std::uint64_t source_offset, std::uint64_t source_bytes,
                       std::uint64_t bytes, std::uint64_t& cursor,
                       TieredLayout& layout, std::string& error) {
    std::uint64_t aligned = 0;
    std::uint64_t end = 0;
    if (name.empty() || source_bytes == 0 || bytes == 0 ||
        !align_up(cursor, kDenseTensorAlignment, aligned) ||
        !add(aligned, bytes, end) || end > std::numeric_limits<std::size_t>::max()) {
        error = "tiered dense-arena layout overflows at " + name;
        return false;
    }
    layout.dense_copies.push_back(
        {name, from_pack, shard, source_offset, source_bytes, bytes, aligned});
    cursor = end;
    return true;
}

}  // namespace

bool build_tiered_layout(const Geometry& geometry, const WindowSources& sources,
                         const PlanResult& memory_plan, std::uint64_t dense_gpu_base,
                         std::uint64_t resident_gpu_base, TieredLayout& layout,
                         std::string& error) {
    layout = {};
    error.clear();
    if (!validate_geometry(geometry, error)) return false;
    if (!memory_plan.fits || memory_plan.mode != "tiered" ||
        geometry.n_layers > kTieredMaxLayers || geometry.n_expert > kTieredMaxExperts ||
        sources.files.size() != geometry.shards.size() ||
        memory_plan.resident_order.size() != memory_plan.experts_resident ||
        memory_plan.experts_resident > kResidentEntries ||
        (dense_gpu_base != 0 && dense_gpu_base % kDenseBaseAlignment != 0) ||
        (resident_gpu_base != 0 && resident_gpu_base % kResidentExpertAlignment != 0)) {
        error = "tiered layout inputs do not match the model, memory plan or arena alignment";
        return false;
    }

    std::map<std::string, const DensePackTensorSource*> pack_tensors;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pack_ranges;
    std::uint64_t packed_destination_bytes = 0;
    for (const DensePackTensorSource& tensor : sources.dense_pack_tensors) {
        std::uint64_t pack_end = 0;
        if (tensor.tensor_name.empty() || tensor.kind != 4 ||
            tensor.source_byte_count != tensor.byte_count ||
            !source_range_valid(tensor.file_offset, tensor.source_byte_count,
                                sources.dense_pack_file_size) ||
            !add(tensor.file_offset, tensor.source_byte_count, pack_end) ||
            !pack_tensors.emplace(tensor.tensor_name, &tensor).second ||
            !add(packed_destination_bytes, tensor.byte_count, packed_destination_bytes)) {
            error = "tiered dense-pack inventory is invalid at " + tensor.tensor_name;
            return false;
        }
        pack_ranges.emplace_back(tensor.file_offset, pack_end);
    }
    std::sort(pack_ranges.begin(), pack_ranges.end());
    for (std::size_t i = 1; i < pack_ranges.size(); ++i) {
        if (pack_ranges[i].first < pack_ranges[i - 1].second) {
            error = "tiered dense-pack tensor ranges overlap";
            return false;
        }
    }
    if (packed_destination_bytes != geometry.dense_pack_bytes ||
        (packed_destination_bytes != 0 &&
         (sources.dense_pack_path.empty() || sources.dense_pack_file_size == 0))) {
        error = "tiered dense-pack inventory does not match dense_pack_bytes";
        return false;
    }

    std::vector<ExpertDraft> expert_drafts(
        static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert);
    std::set<std::string> tensor_names;
    std::set<std::string> excluded_names;
    std::uint64_t nonexpert_bytes = 0;
    std::uint64_t dense_cursor = 0;
    std::uint64_t logical_dense_bytes = 0;

    for (std::size_t shard = 0; shard < sources.files.size(); ++shard) {
        const WindowFileSource& file = sources.files[shard];
        const ShardGeometry& planned_shard = geometry.shards[shard];
        if (file.path.empty() || file.file_size == 0 ||
            std::filesystem::path(file.path).filename().string() != planned_shard.name ||
            file.file_size != planned_shard.bytes) {
            error = file.path + ": tiered shard inventory differs from the memory plan";
            return false;
        }
        std::set<std::string> file_names;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> file_ranges;
        for (const WindowTensorSource& tensor : file.tensors) {
            std::uint64_t end = 0;
            if (!file_names.insert(tensor.tensor_name).second ||
                !tensor_names.insert(tensor.tensor_name).second ||
                !source_range_valid(tensor.file_offset, tensor.byte_count, file.file_size) ||
                !add(tensor.file_offset, tensor.byte_count, end)) {
                error = file.path + ": invalid or duplicate tiered tensor source " + tensor.tensor_name;
                return false;
            }
            file_ranges.emplace_back(tensor.file_offset, end);
            const std::optional<ExpertIdentity> identity = parse_expert_name(tensor.tensor_name);
            if (!identity) {
                if (is_expert_role_name(tensor.tensor_name) || tensor.expert_slab_bytes != 0) {
                    error = file.path + ": malformed expert tensor source " + tensor.tensor_name;
                    return false;
                }
                if (!add(nonexpert_bytes, tensor.byte_count, nonexpert_bytes) ||
                    !add(logical_dense_bytes, tensor.byte_count, logical_dense_bytes) ||
                    !append_dense_copy(tensor.tensor_name, false, shard, tensor.file_offset,
                                       tensor.byte_count, tensor.byte_count, dense_cursor,
                                       layout, error)) {
                    if (error.empty()) error = "tiered non-expert byte count overflows";
                    return false;
                }
                continue;
            }
            if (identity->layer >= geometry.n_layers || tensor.expert_slab_bytes == 0 ||
                tensor.byte_count % tensor.expert_slab_bytes != 0 ||
                tensor.byte_count / tensor.expert_slab_bytes != geometry.n_expert) {
                error = file.path + ": invalid expert slab source " + tensor.tensor_name;
                return false;
            }
            const std::size_t expert_base =
                static_cast<std::size_t>(identity->layer) * geometry.n_expert;
            for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
                ExpertDraft& expert_draft = expert_drafts[expert_base + expert];
                if (expert_draft.tensors[identity->role] != nullptr) {
                    error = file.path + ": duplicate expert role source " + tensor.tensor_name;
                    return false;
                }
                expert_draft.tensors[identity->role] = &tensor;
                expert_draft.shards[identity->role] = shard;
            }
        }
        for (const WindowTensorExclusion& tensor : file.excluded_tensors) {
            std::uint64_t end = 0;
            if (!file_names.insert(tensor.tensor_name).second ||
                !tensor_names.insert(tensor.tensor_name).second ||
                !excluded_names.insert(tensor.tensor_name).second ||
                !source_range_valid(tensor.file_offset, tensor.byte_count, file.file_size) ||
                !add(tensor.file_offset, tensor.byte_count, end)) {
                error = file.path + ": invalid or duplicate tiered excluded tensor " + tensor.tensor_name;
                return false;
            }
            file_ranges.emplace_back(tensor.file_offset, end);
            const bool ple = tensor.tensor_name == "per_layer_token_embd.weight";
            if ((ple && tensor.reason != WindowTensorExclusionReason::ple) ||
                (!ple && (tensor.reason != WindowTensorExclusionReason::converted ||
                          !is_router_name(tensor.tensor_name)))) {
                error = file.path + ": unsupported tiered tensor exclusion " + tensor.tensor_name;
                return false;
            }
        }
        std::sort(file_ranges.begin(), file_ranges.end());
        for (std::size_t i = 1; i < file_ranges.size(); ++i) {
            if (file_ranges[i].first < file_ranges[i - 1].second) {
                error = file.path + ": tiered GGUF tensor ranges overlap";
                return false;
            }
        }
    }
    if (nonexpert_bytes != geometry.nonexpert_bytes) {
        error = "tiered non-expert source bytes do not match geometry";
        return false;
    }

    for (const auto& [name, pack_tensor] : pack_tensors) {
        if (!excluded_names.contains(name) || !is_router_name(name) ||
            !append_dense_copy(name, true, 0, pack_tensor->file_offset,
                               pack_tensor->source_byte_count, pack_tensor->byte_count,
                               dense_cursor, layout, error) ||
            !add(logical_dense_bytes, pack_tensor->byte_count, logical_dense_bytes)) {
            if (error.empty()) error = "converted dense-pack tensor has no matching GGUF exclusion: " + name;
            return false;
        }
    }
    for (const std::string& name : excluded_names) {
        if (name != "per_layer_token_embd.weight" && !pack_tensors.contains(name)) {
            error = "converted GGUF tensor has no dense-pack source: " + name;
            return false;
        }
    }

    layout.experts.resize(expert_drafts.size());
    for (std::uint32_t layer = 0; layer < geometry.n_layers; ++layer) {
        std::array<std::uint64_t, kExpertRoles> role_bytes{};
        std::array<std::uint64_t, kExpertRoles> slab_bytes{};
        for (std::uint32_t expert = 0; expert < geometry.n_expert; ++expert) {
            const std::size_t index = static_cast<std::size_t>(layer) * geometry.n_expert + expert;
            const ExpertDraft& draft = expert_drafts[index];
            for (std::uint32_t role = 0; role < kExpertRoles; ++role) {
                if (draft.tensors[role] == nullptr) {
                    error = "tiered expert inventory is missing a role in layer " +
                            std::to_string(layer);
                    return false;
                }
                const WindowTensorSource& tensor = *draft.tensors[role];
                if (expert == 0) {
                    slab_bytes[role] = tensor.expert_slab_bytes;
                    role_bytes[role] = tensor.byte_count;
                } else if (slab_bytes[role] != tensor.expert_slab_bytes ||
                           role_bytes[role] != tensor.byte_count) {
                    error = "tiered expert role has inconsistent slab sizes in layer " +
                            std::to_string(layer);
                    return false;
                }
            }
            std::uint64_t blob = 0;
            if (slab_bytes[0] != slab_bytes[1] ||
                !add(slab_bytes[0], slab_bytes[1], blob) ||
                !add(blob, slab_bytes[2], blob) ||
                blob != geometry.layers[layer].blob_bytes) {
                error = "tiered expert role sizes do not match geometry in layer " +
                        std::to_string(layer);
                return false;
            }
            TieredExpert& output = layout.experts[index];
            output.layer = layer;
            output.expert = expert;
            output.blob_bytes = blob;
            for (std::uint32_t role = 0; role < kExpertRoles; ++role) {
                const WindowTensorSource& tensor = *draft.tensors[role];
                std::uint64_t expert_delta = 0;
                std::uint64_t source_offset = 0;
                if (expert != 0 &&
                    tensor.expert_slab_bytes >
                        std::numeric_limits<std::uint64_t>::max() / expert) {
                    error = "tiered expert source offset overflows";
                    return false;
                }
                expert_delta = static_cast<std::uint64_t>(expert) * tensor.expert_slab_bytes;
                if (!add(tensor.file_offset, expert_delta, source_offset) ||
                    !source_range_valid(source_offset, tensor.expert_slab_bytes,
                                        sources.files[draft.shards[role]].file_size)) {
                    error = "tiered expert source range falls outside its GGUF shard";
                    return false;
                }
                output.roles[role] = {draft.shards[role], source_offset,
                                      tensor.expert_slab_bytes};
            }
        }
    }

    std::uint64_t expected_resident_bytes = 0;
    std::set<std::size_t> resident_indices;
    for (const auto& [layer, expert] : memory_plan.resident_order) {
        if (layer >= geometry.n_layers || expert >= geometry.n_expert) {
            error = "memory plan resident expert is outside model dimensions";
            return false;
        }
        const std::size_t index = static_cast<std::size_t>(layer) * geometry.n_expert + expert;
        if (!resident_indices.insert(index).second) {
            error = "memory plan contains a duplicate resident expert";
            return false;
        }
        if (!add(expected_resident_bytes, layout.experts[index].blob_bytes,
                 expected_resident_bytes)) {
            error = "tiered resident byte count overflows";
            return false;
        }
    }
    if (expected_resident_bytes != memory_plan.experts_resident_bytes) {
        error = "tiered resident experts do not match the memory plan byte count";
        return false;
    }

    std::uint64_t dense_base = dense_gpu_base;
    for (const TieredDenseCopy& copy : layout.dense_copies) {
        std::uint64_t address = 0;
        if (!address_range_valid(dense_base, copy.arena_offset, copy.byte_count, address)) {
            error = "tiered dense GPU address base is unavailable or its extent overflows";
            return false;
        }
        layout.tensors.push_back(
            {copy.tensor_name, address, copy.byte_count, copy.arena_offset});
    }
    if (!align_up(dense_cursor, kDenseArenaLengthAlignment,
                  layout.dense_arena_bytes)) {
        error = "tiered dense arena page-rounded length overflows";
        return false;
    }
    layout.expert_addresses.assign(kAddressEntries, 0);
    layout.resident.assign(kResidentEntries, 0);
    const std::size_t active_entries =
        static_cast<std::size_t>(geometry.n_layers) * geometry.n_expert;
    std::fill(layout.resident.begin(),
              layout.resident.begin() + static_cast<std::ptrdiff_t>(active_entries), -1);

    std::uint64_t resident_cursor = 0;
    std::int32_t slot = 0;
    for (const auto& [layer, expert] : memory_plan.resident_order) {
        const std::size_t index = static_cast<std::size_t>(layer) * geometry.n_expert + expert;
        TieredExpert& output = layout.experts[index];
        std::uint64_t aligned = 0;
        std::uint64_t end = 0;
        if (!align_up(resident_cursor, kResidentExpertAlignment, aligned) ||
            !add(aligned, output.blob_bytes, end) ||
            end > std::numeric_limits<std::size_t>::max()) {
            error = "tiered resident arena layout overflows";
            return false;
        }
        output.resident_slot = slot;
        output.resident_arena_offset = aligned;
        layout.resident[index] = slot;
        std::uint64_t role_offset = 0;
        for (std::uint32_t role = 0; role < kExpertRoles; ++role) {
            std::uint64_t arena_offset = 0;
            std::uint64_t gpu_address = 0;
            if (!add(aligned, role_offset, arena_offset) ||
                !address_range_valid(resident_gpu_base, arena_offset,
                                     output.roles[role].byte_count, gpu_address)) {
                error = "tiered resident GPU address base is unavailable or its extent overflows";
                return false;
            }
            output.resident_addresses[role] = gpu_address;
            layout.expert_addresses[index * kExpertRoles + role] = gpu_address;
            role_offset += output.roles[role].byte_count;
        }
        resident_cursor = end;
        ++slot;
    }
    layout.resident_arena_bytes = resident_cursor;

    std::uint64_t logical_bytes = 0;
    if (!add(geometry.nonexpert_bytes, geometry.dense_pack_bytes, logical_bytes) ||
        logical_dense_bytes != logical_bytes ||
        layout.dense_arena_bytes < logical_bytes ||
        layout.resident_arena_bytes < memory_plan.experts_resident_bytes) {
        error = "tiered arena layout does not account for the planned tensor bytes";
        return false;
    }
    std::uint64_t alignment_slack = 0;
    if (!add(layout.dense_arena_bytes - logical_bytes,
             layout.resident_arena_bytes - memory_plan.experts_resident_bytes,
             alignment_slack)) {
        error = "tiered arena alignment padding overflows";
        return false;
    }
    if (memory_plan.free_bytes < 0 ||
        alignment_slack > static_cast<std::uint64_t>(memory_plan.free_bytes)) {
        error = "tiered arena alignment padding exceeds the memory plan's free bytes";
        return false;
    }

    std::uint64_t arena_end = 0;
    if ((layout.dense_arena_bytes != 0 &&
         !add(dense_gpu_base, layout.dense_arena_bytes, arena_end)) ||
        (layout.resident_arena_bytes != 0 &&
         !add(resident_gpu_base, layout.resident_arena_bytes, arena_end))) {
        error = "tiered arena GPU address range overflows";
        return false;
    }

    error.clear();
    return true;
}

}  // namespace strata::metal
