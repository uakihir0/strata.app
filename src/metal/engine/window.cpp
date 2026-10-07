// src/metal/engine/window.cpp - full-mode tensor and expert address layout.
#include "strata/metal/window.hpp"

#include <algorithm>
#include <charconv>
#include <limits>
#include <map>
#include <set>
#include <string_view>

namespace strata::metal {
namespace {

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
    const std::uint64_t remainder = value & (alignment - 1);
    return remainder == 0 ? (result = value, true)
                          : add(value, alignment - remainder, result);
}

bool mapped_extent(std::uint64_t file_size, std::uint64_t& extent) {
    const std::uint64_t remainder = file_size % kWindowPageBytes;
    return remainder == 0 ? (extent = file_size, true)
                          : add(file_size, kWindowPageBytes - remainder, extent);
}

enum class ExpertRole : std::uint64_t { gate = 0, up = 1, down = 2 };

bool expert_tensor(std::string_view name, std::uint32_t& layer, ExpertRole& role) {
    if (!name.starts_with("blk.")) return false;
    const std::size_t layer_end = name.find('.', 4);
    if (layer_end == std::string_view::npos || layer_end == 4) return false;
    const auto parsed = std::from_chars(name.data() + 4, name.data() + layer_end, layer);
    if (parsed.ec != std::errc{} || parsed.ptr != name.data() + layer_end) return false;

    const std::string_view suffix = name.substr(layer_end + 1);
    if (suffix == "ffn_gate_exps.weight") {
        role = ExpertRole::gate;
        return true;
    }
    if (suffix == "ffn_up_exps.weight") {
        role = ExpertRole::up;
        return true;
    }
    if (suffix == "ffn_down_exps.weight") {
        role = ExpertRole::down;
        return true;
    }
    return false;
}

bool router_name(std::string_view name, std::uint32_t n_layers) {
    if (!name.starts_with("blk.") ||
        !(name.ends_with(".ffn_gate_inp.weight") ||
          name.ends_with(".ffn_gate_inp_shexp.weight"))) {
        return false;
    }
    const std::size_t layer_end = name.find('.', 4);
    if (layer_end == std::string_view::npos || layer_end == 4) return false;
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(name.data() + 4, name.data() + layer_end, layer);
    return parsed.ec == std::errc{} && parsed.ptr == name.data() + layer_end &&
           layer < n_layers &&
           name.substr(4, layer_end - 4) == std::to_string(layer);
}

}  // namespace

bool order_full_mode_warm_experts(
    std::uint32_t n_layers, std::uint32_t n_expert,
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& profile_order,
    std::vector<std::pair<std::uint32_t, std::uint32_t>>& ordered_experts,
    std::string& error) {
    error.clear();
    ordered_experts.clear();
    if (n_layers == 0 || n_layers > kFullModeMaxLayers ||
        n_expert == 0 || n_expert > kFullModeMaxExperts) {
        error = "expert warm-up order has invalid model dimensions";
        return false;
    }

    std::set<std::pair<std::uint32_t, std::uint32_t>> seen;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> candidate;
    candidate.reserve(static_cast<std::size_t>(n_layers) * n_expert);
    for (const auto& [layer, expert] : profile_order) {
        if (layer >= n_layers || expert >= n_expert || !seen.emplace(layer, expert).second) {
            error = "expert warm-up order contains an out-of-range or duplicate pair";
            return false;
        }
        candidate.emplace_back(layer, expert);
    }
    for (std::uint32_t layer = 0; layer < n_layers; ++layer) {
        for (std::uint32_t expert = 0; expert < n_expert; ++expert) {
            if (seen.emplace(layer, expert).second) candidate.emplace_back(layer, expert);
        }
    }

    ordered_experts = std::move(candidate);
    return true;
}

bool build_full_mode_layout(const WindowSources& sources, const WindowPlan& plan,
                            const std::vector<std::vector<std::uint64_t>>& window_gpu_addresses,
                            std::uint64_t max_buffer_length, std::uint32_t n_layers,
                            std::uint32_t n_expert, FullModeLayout& layout, std::string& error,
                            std::uint64_t dense_pack_gpu_base) {
    error.clear();
    layout = {};
    if (n_layers == 0 || n_layers > kFullModeMaxLayers || n_expert == 0 ||
        n_expert > kFullModeMaxExperts ||
        static_cast<std::uint64_t>(kFullModeMaxLayers) * kFullModeMaxExperts >
            std::numeric_limits<std::size_t>::max() / kFullModeExpertRoles ||
        max_buffer_length == 0 || sources.files.size() != plan.files.size() ||
        sources.files.size() != window_gpu_addresses.size()) {
        error = "full-mode layout has invalid dimensions or file counts";
        return false;
    }

    FullModeLayout candidate;
    constexpr std::size_t kTableEntries =
        static_cast<std::size_t>(kFullModeMaxLayers) * kFullModeMaxExperts;
    candidate.expert_addresses.assign(kTableEntries * kFullModeExpertRoles, 0);
    candidate.resident.assign(kTableEntries, 0);
    std::vector<bool> expert_address_written(kTableEntries * kFullModeExpertRoles, false);
    candidate.tensors.reserve(plan.files.size() + sources.dense_pack_tensors.size());
    std::set<std::string> all_tensor_names;
    std::set<std::string> converted_exclusions;

    for (std::size_t file_index = 0; file_index < sources.files.size(); ++file_index) {
        const WindowFileSource& source = sources.files[file_index];
        const FileWindowPlan& file_plan = plan.files[file_index];
        if (source.path != file_plan.path ||
            file_plan.windows.size() != window_gpu_addresses[file_index].size()) {
            error = source.path + ": window plan or GPU buffer count does not match the source";
            return false;
        }
        std::uint64_t file_mapping_extent = 0;
        if (!mapped_extent(source.file_size, file_mapping_extent)) {
            error = source.path + ": mapped file extent overflows";
            return false;
        }

        for (std::size_t window_index = 0; window_index < file_plan.windows.size(); ++window_index) {
            const FileWindow& window = file_plan.windows[window_index];
            std::uint64_t length = 0;
            if (window_gpu_addresses[file_index][window_index] == 0) {
                error = source.path + ": GPU address is unavailable for window " +
                        std::to_string(window_index);
                return false;
            }
            if (window.start >= window.end || window.end > file_mapping_extent ||
                window.start % kWindowPageBytes != 0 || window.end % kWindowPageBytes != 0 ||
                window.end - window.start > max_buffer_length ||
                !add(window_gpu_addresses[file_index][window_index], window.end - window.start, length)) {
                error = source.path + ": invalid or oversized GPU window " + std::to_string(window_index);
                return false;
            }
        }

        std::map<std::string, const WindowTensorSource*> tensor_sources;
        for (const WindowTensorSource& tensor : source.tensors) {
            std::uint64_t end = 0;
            if (tensor.tensor_name == "per_layer_token_embd.weight") {
                error = source.path + ": PLE tensor must not be mapped into the full-mode layout";
                return false;
            }
            if (tensor.tensor_name.empty() || tensor.byte_count == 0 ||
                !add(tensor.file_offset, tensor.byte_count, end) || end > source.file_size ||
                !tensor_sources.emplace(tensor.tensor_name, &tensor).second) {
                error = source.path + ": invalid or duplicate tensor source";
                return false;
            }
            if (!all_tensor_names.insert(tensor.tensor_name).second) {
                error = source.path + ": duplicate tensor name across full-mode shards: " +
                        tensor.tensor_name;
                return false;
            }
            if (tensor.expert_slab_bytes != 0 &&
                (tensor.byte_count % tensor.expert_slab_bytes != 0 ||
                 tensor.byte_count / tensor.expert_slab_bytes != n_expert)) {
                error = source.path + ": expert tensor has an invalid slab count: " + tensor.tensor_name;
                return false;
            }
        }
        for (const WindowTensorExclusion& tensor : source.excluded_tensors) {
            if (tensor.reason == WindowTensorExclusionReason::converted &&
                (!router_name(tensor.tensor_name, n_layers) ||
                 !converted_exclusions.insert(tensor.tensor_name).second)) {
                error = source.path +
                        ": converted exclusion is not a unique supported router: " +
                        tensor.tensor_name;
                return false;
            }
        }

        std::map<std::string, std::set<std::uint64_t>> seen;
        for (const WindowSlice& slice : file_plan.slices) {
            const auto source_it = tensor_sources.find(slice.tensor_name);
            if (source_it == tensor_sources.end()) {
                error = source.path + ": window plan contains an unknown tensor " + slice.tensor_name;
                return false;
            }
            const WindowTensorSource& tensor = *source_it->second;
            std::uint64_t expected_offset = tensor.file_offset;
            std::uint64_t expected_bytes = tensor.byte_count;
            const bool is_expert = tensor.expert_slab_bytes != 0;
            if (is_expert) {
                if (slice.expert_index >= n_expert ||
                    slice.expert_index > std::numeric_limits<std::uint64_t>::max() /
                                             tensor.expert_slab_bytes ||
                    !add(tensor.file_offset, slice.expert_index * tensor.expert_slab_bytes,
                         expected_offset)) {
                    error = source.path + ": invalid expert index for " + slice.tensor_name;
                    return false;
                }
                expected_bytes = tensor.expert_slab_bytes;
            } else if (slice.expert_index != kNoExpertIndex) {
                error = source.path + ": whole tensor has an expert index: " + slice.tensor_name;
                return false;
            }

            if (slice.file_offset != expected_offset || slice.byte_count != expected_bytes ||
                !seen[slice.tensor_name].insert(slice.expert_index).second ||
                slice.window_index >= file_plan.windows.size()) {
                error = source.path + ": invalid or duplicate tensor slice " + slice.tensor_name;
                return false;
            }
            const FileWindow& window = file_plan.windows[static_cast<std::size_t>(slice.window_index)];
            const std::uint64_t window_length = window.end - window.start;
            if (slice.file_offset < window.start || slice.file_offset - window.start != slice.window_offset ||
                slice.window_offset > window_length || slice.byte_count > window_length - slice.window_offset) {
                error = source.path + ": tensor slice falls outside its file window: " + slice.tensor_name;
                return false;
            }

            std::uint64_t gpu_address = 0;
            if (!add(window_gpu_addresses[file_index][static_cast<std::size_t>(slice.window_index)],
                     slice.window_offset, gpu_address)) {
                error = source.path + ": GPU tensor address overflows for " + slice.tensor_name;
                return false;
            }
            candidate.tensors.push_back({slice.tensor_name, slice.expert_index, gpu_address, slice.byte_count});

            std::uint32_t layer = 0;
            ExpertRole role = ExpertRole::gate;
            const bool has_expert_role = expert_tensor(slice.tensor_name, layer, role);
            if (is_expert != has_expert_role || (has_expert_role && layer >= n_layers)) {
                error = source.path + ": expert slab does not match a supported layer role: " +
                        slice.tensor_name;
                return false;
            }
            if (has_expert_role) {
                const std::size_t expert_offset =
                    (static_cast<std::size_t>(layer) * n_expert +
                     static_cast<std::size_t>(slice.expert_index)) * kFullModeExpertRoles +
                    static_cast<std::uint64_t>(role);
                if (expert_address_written[expert_offset]) {
                    error = source.path + ": duplicate address for " + slice.tensor_name;
                    return false;
                }
                candidate.expert_addresses[expert_offset] = gpu_address;
                expert_address_written[expert_offset] = true;
                candidate.resident[expert_offset / kFullModeExpertRoles] = 1;
            }
        }

        for (const auto& [name, tensor] : tensor_sources) {
            const std::uint64_t expected_slices =
                tensor->expert_slab_bytes == 0 ? 1 : tensor->byte_count / tensor->expert_slab_bytes;
            const auto found = seen.find(name);
            if (found == seen.end() || found->second.size() != expected_slices) {
                error = source.path + ": window plan does not cover every byte of " + name;
                return false;
            }
            if (tensor->expert_slab_bytes == 0 && !found->second.contains(kNoExpertIndex)) {
                error = source.path + ": window plan is missing the whole tensor " + name;
                return false;
            }
        }
    }

    std::vector<std::pair<std::uint64_t, std::uint64_t>> pack_ranges;
    pack_ranges.reserve(sources.dense_pack_tensors.size());
    std::set<std::string> packed_names;
    std::uint64_t dense_cursor = 0;
    for (const DensePackTensorSource& tensor : sources.dense_pack_tensors) {
        std::uint64_t source_end = 0;
        std::uint64_t arena_offset = 0;
        std::uint64_t arena_end = 0;
        std::uint64_t address = 0;
        std::uint64_t address_end = 0;
        if (tensor.tensor_name.empty() || !router_name(tensor.tensor_name, n_layers) ||
            tensor.kind != 4 || tensor.source_byte_count == 0 ||
            tensor.source_byte_count != tensor.byte_count ||
            tensor.byte_count % 256 != 0 ||
            !add(tensor.file_offset, tensor.source_byte_count, source_end) ||
            source_end > sources.dense_pack_file_size ||
            all_tensor_names.contains(tensor.tensor_name) ||
            !packed_names.insert(tensor.tensor_name).second ||
            !converted_exclusions.erase(tensor.tensor_name) ||
            dense_pack_gpu_base == 0 || dense_pack_gpu_base % 256 != 0 ||
            !align_up(dense_cursor, 256, arena_offset) ||
            !add(arena_offset, tensor.byte_count, arena_end) ||
            arena_end > std::numeric_limits<std::size_t>::max() ||
            !add(dense_pack_gpu_base, arena_offset, address) ||
            !add(address, tensor.byte_count, address_end)) {
            error = "full-mode converted dense-pack row is invalid: " + tensor.tensor_name;
            return false;
        }
        pack_ranges.emplace_back(tensor.file_offset, source_end);
        candidate.tensors.push_back(
            {tensor.tensor_name, kNoExpertIndex, address, tensor.byte_count});
        dense_cursor = arena_end;
        if (!add(candidate.dense_pack_bytes, tensor.byte_count,
                 candidate.dense_pack_bytes)) {
            error = "full-mode converted dense-pack byte count overflows";
            return false;
        }
    }
    std::sort(pack_ranges.begin(), pack_ranges.end());
    for (std::size_t i = 1; i < pack_ranges.size(); ++i) {
        if (pack_ranges[i].first < pack_ranges[i - 1].second) {
            error = "full-mode converted dense-pack ranges overlap";
            return false;
        }
    }
    if (!converted_exclusions.empty()) {
        error = "full-mode converted router is missing its dense-pack row: " +
                *converted_exclusions.begin();
        return false;
    }
    if (candidate.dense_pack_bytes != 0) {
        if (sources.dense_pack_path.empty() ||
            !align_up(dense_cursor, kWindowPageBytes, candidate.dense_pack_arena_bytes)) {
            error = "full-mode converted dense-pack arena dimensions are invalid";
            return false;
        }
        if (candidate.dense_pack_arena_bytes > max_buffer_length) {
            error = "full-mode converted dense-pack arena exceeds maxBufferLength";
            return false;
        }
    }

    for (std::size_t entry = 0; entry < static_cast<std::size_t>(n_layers) * n_expert; ++entry) {
        const std::size_t role_base = entry * kFullModeExpertRoles;
        for (std::size_t role = 0; role < kFullModeExpertRoles; ++role) {
            if (!expert_address_written[role_base + role]) {
                error = "full-mode expert address table is missing one or more gate, up or down slabs";
                return false;
            }
        }
    }
    layout = std::move(candidate);
    return true;
}

}  // namespace strata::metal
