// src/metal/engine/window_storage.cpp - map validated GGUF windows into Context-owned Metal buffers.
#include "strata/metal/window.hpp"
#include "window_source_lock.hpp"

#include "strata/artifact/gguf_reader.hpp"
#include "strata/kernels/ngram.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <cerrno>
#include <charconv>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <condition_variable>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <pthread/qos.h>
#include <set>
#include <string>
#include <string_view>
#include <system_error>
#include <sys/mman.h>
#include <sys/stat.h>
#include <thread>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

constexpr std::uint32_t kGgmlTypeBf16 = 30;

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool is_router_tensor_name(std::string_view name) {
    return name.ends_with(".ffn_gate_inp.weight") ||
           name.ends_with(".ffn_gate_inp_shexp.weight");
}

enum class ExpertRole : std::size_t { gate = 0, up = 1, down = 2 };

bool expert_tensor_name(std::string_view name, std::uint32_t& layer, ExpertRole& role) {
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

bool looks_like_expert_tensor(std::string_view name) {
    return name.find(".ffn_") != std::string_view::npos &&
           name.ends_with("_exps.weight");
}

bool valid_router_layer(std::string_view name, std::uint32_t n_layers) {
    if (!name.starts_with("blk.")) return false;
    const std::size_t layer_end = name.find('.', 4);
    if (layer_end == std::string_view::npos || layer_end == 4) return false;
    std::uint32_t layer = 0;
    const auto parsed = std::from_chars(name.data() + 4, name.data() + layer_end, layer);
    return parsed.ec == std::errc{} && parsed.ptr == name.data() + layer_end &&
           layer < n_layers &&
           name.substr(4, layer_end - 4) == std::to_string(layer);
}

bool valid_converted_router(const strata::TensorInfo& tensor, const Geometry& geometry) {
    if (std::string_view(tensor.type_name()) != "F32" ||
        !valid_router_layer(tensor.name, geometry.n_layers)) {
        return false;
    }
    if (tensor.name.ends_with(".ffn_gate_inp.weight")) {
        return tensor.shape.size() == 2 && tensor.shape[0] == 2'560 &&
               tensor.shape[1] == geometry.n_expert;
    }
    if (tensor.name.ends_with(".ffn_gate_inp_shexp.weight")) {
        return tensor.shape.size() == 1 && tensor.shape[0] == 2'560;
    }
    return false;
}

bool validate_model_sources(const strata::GgufModel& model, const WindowSources& sources,
                            const Geometry& geometry, std::string& error) {
    if (model.size() != sources.files.size()) {
        error = "GGUF model shard count does not match the window sources";
        return false;
    }
    std::uint64_t converted_source_bytes = 0;
    std::uint64_t nonexpert_bytes = 0;
    std::vector<std::array<std::uint64_t, 3>> expert_role_bytes(geometry.n_layers);
    std::vector<std::array<bool, 3>> expert_role_seen(geometry.n_layers);
    std::set<std::string> model_tensor_names;
    std::map<std::string, const DensePackTensorSource*> pack_tensors;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> pack_ranges;
    std::uint64_t packed_destination_bytes = 0;
    for (const DensePackTensorSource& tensor : sources.dense_pack_tensors) {
        std::uint64_t pack_end = 0;
        if (tensor.tensor_name.empty() || tensor.kind != 4 ||
            tensor.source_byte_count == 0 ||
            tensor.source_byte_count != tensor.byte_count ||
            tensor.file_offset > sources.dense_pack_file_size ||
            tensor.source_byte_count >
                sources.dense_pack_file_size - tensor.file_offset ||
            !add(tensor.file_offset, tensor.source_byte_count, pack_end) ||
            !pack_tensors.emplace(tensor.tensor_name, &tensor).second ||
            !add(packed_destination_bytes, tensor.byte_count,
                 packed_destination_bytes)) {
            error = "converted dense-pack inventory is invalid at " + tensor.tensor_name;
            return false;
        }
        pack_ranges.emplace_back(tensor.file_offset, pack_end);
    }
    std::sort(pack_ranges.begin(), pack_ranges.end());
    for (std::size_t i = 1; i < pack_ranges.size(); ++i) {
        if (pack_ranges[i].first < pack_ranges[i - 1].second) {
            error = "converted dense-pack file ranges overlap";
            return false;
        }
    }
    if (packed_destination_bytes != geometry.dense_pack_bytes ||
        (packed_destination_bytes != 0 &&
         (sources.dense_pack_path.empty() || sources.dense_pack_file_size == 0))) {
        error = "converted dense-pack inventory does not match inspected geometry";
        return false;
    }
    std::size_t ple_count = 0;
    for (std::size_t shard = 0; shard < model.size(); ++shard) {
        const strata::GgufFile& file = model.shard(shard);
        const WindowFileSource& source = sources.files[shard];
        if (file.path() != source.path || file.file_size() != source.file_size ||
            file.file_device_id() != source.file_device_id ||
            file.file_object_id() != source.file_object_id ||
            file.mapping_data() == nullptr) {
            error = source.path +
                    ": mapped GGUF shard no longer matches its inspected file identity";
            return false;
        }
        std::map<std::string, const WindowTensorSource*> included;
        std::map<std::string, const WindowTensorExclusion*> excluded;
        std::vector<std::pair<std::uint64_t, std::uint64_t>> payload_ranges;
        payload_ranges.reserve(file.tensors().size());
        for (const WindowTensorSource& tensor : source.tensors) {
            if (!included.emplace(tensor.tensor_name, &tensor).second) {
                error = source.path + ": duplicate included tensor source " + tensor.tensor_name;
                return false;
            }
        }
        for (const WindowTensorExclusion& tensor : source.excluded_tensors) {
            if (!excluded.emplace(tensor.tensor_name, &tensor).second) {
                error = source.path + ": duplicate excluded tensor source " + tensor.tensor_name;
                return false;
            }
        }

        for (const strata::TensorInfo& tensor : file.tensors()) {
            if (!model_tensor_names.insert(tensor.name).second) {
                error = "GGUF model contains a duplicate tensor name " + tensor.name;
                return false;
            }
            const auto included_it = included.find(tensor.name);
            const auto excluded_it = excluded.find(tensor.name);
            if ((included_it == included.end()) == (excluded_it == excluded.end())) {
                error = source.path + ": tensor must be accounted for exactly once: " + tensor.name;
                return false;
            }
            std::uint64_t offset = 0;
            std::uint64_t end = 0;
            const std::uint64_t bytes = strata::tensor_payload_bytes(tensor);
            if (bytes == 0 || !add(file.data_start(), tensor.offset, offset) ||
                !add(offset, bytes, end) || end > file.file_size() ||
                (included_it != included.end() &&
                 (included_it->second->file_offset != offset ||
                  included_it->second->byte_count != bytes)) ||
                (excluded_it != excluded.end() &&
                 (excluded_it->second->file_offset != offset ||
                  excluded_it->second->byte_count != bytes))) {
                error = source.path + ": inspected tensor range changed for " + tensor.name;
                return false;
            }
            payload_ranges.emplace_back(offset, end);
            if (included_it != included.end() && tensor.name == "per_layer_token_embd.weight") {
                error = source.path + ": the PLE tensor must not be mapped into a Metal window";
                return false;
            }
            const bool is_ple = tensor.name == "per_layer_token_embd.weight";
            std::uint32_t expert_layer = 0;
            ExpertRole expert_role = ExpertRole::gate;
            const bool is_expert = expert_tensor_name(tensor.name, expert_layer, expert_role);
            if (is_expert) {
                const std::size_t role = static_cast<std::size_t>(expert_role);
                if (expert_layer >= geometry.n_layers || excluded_it != excluded.end() ||
                    included_it == included.end() || tensor.shape.size() != 3 ||
                    tensor.shape[2] != geometry.n_expert || bytes % geometry.n_expert != 0 ||
                    included_it->second->expert_slab_bytes != bytes / geometry.n_expert ||
                    expert_role_seen[expert_layer][role]) {
                    error = source.path + ": expert payload dimensions do not match geometry for " +
                            tensor.name;
                    return false;
                }
                expert_role_seen[expert_layer][role] = true;
                expert_role_bytes[expert_layer][role] = bytes / geometry.n_expert;
            } else if (looks_like_expert_tensor(tensor.name)) {
                error = source.path + ": unsupported expert tensor role " + tensor.name;
                return false;
            } else if (!is_ple && excluded_it == excluded.end()) {
                if (included_it == included.end() ||
                    included_it->second->expert_slab_bytes != 0 ||
                    !add(nonexpert_bytes, bytes, nonexpert_bytes)) {
                    error = source.path + ": non-expert payload inventory is invalid for " +
                            tensor.name;
                    return false;
                }
            }
            if (excluded_it != excluded.end()) {
                const WindowTensorExclusionReason expected_reason =
                    is_ple ? WindowTensorExclusionReason::ple
                           : WindowTensorExclusionReason::converted;
                if (excluded_it->second->reason != expected_reason) {
                    error = source.path + ": tensor has an invalid no-copy exclusion reason: " + tensor.name;
                    return false;
                }
                if (is_ple) ++ple_count;
                if (!is_ple) {
                    const auto packed = pack_tensors.find(tensor.name);
                    if (!valid_converted_router(tensor, geometry) || bytes % 2 != 0 ||
                        packed == pack_tensors.end() ||
                        packed->second->source_byte_count != bytes / 2 ||
                        packed->second->byte_count != bytes / 2 ||
                        !add(converted_source_bytes, bytes, converted_source_bytes)) {
                        error = source.path + ": only a supported F32 router may be excluded as converted: " +
                                tensor.name;
                        return false;
                    }
                    pack_tensors.erase(packed);
                }
            } else if (is_router_tensor_name(tensor.name) &&
                       std::string_view(tensor.type_name()) == "F32") {
                error = source.path + ": an F32 router must have a validated dense-pack conversion: " +
                        tensor.name;
                return false;
            }
        }
        if (included.size() + excluded.size() != file.tensors().size()) {
            error = source.path + ": window sources contain a tensor absent from the mapped GGUF";
            return false;
        }
        std::sort(payload_ranges.begin(), payload_ranges.end());
        for (std::size_t i = 1; i < payload_ranges.size(); ++i) {
            if (payload_ranges[i].first < payload_ranges[i - 1].second) {
                error = source.path + ": GGUF tensor payload ranges overlap";
                return false;
            }
        }
    }
    if (ple_count != 1 || nonexpert_bytes != geometry.nonexpert_bytes) {
        error = "GGUF non-expert or PLE payload bytes do not match inspected geometry";
        return false;
    }
    for (std::size_t layer = 0; layer < geometry.layers.size(); ++layer) {
        const auto& seen = expert_role_seen[layer];
        const auto& bytes = expert_role_bytes[layer];
        std::uint64_t blob_bytes = 0;
        if (!seen[0] || !seen[1] || !seen[2] ||
            !add(bytes[0], bytes[1], blob_bytes) ||
            !add(blob_bytes, bytes[2], blob_bytes) ||
            blob_bytes != geometry.layers[layer].blob_bytes) {
            error = "GGUF expert role payload bytes do not match geometry in layer " +
                    std::to_string(layer);
            return false;
        }
    }
    if (!pack_tensors.empty() || converted_source_bytes % 2 != 0 ||
        converted_source_bytes / 2 != geometry.dense_pack_bytes) {
        error = "converted router payload bytes do not match dense_pack_bytes in geometry";
        return false;
    }
    return true;
}

bool same_plan(const WindowPlan& left, const WindowPlan& right) {
    if (left.wrapped_unread_bytes != right.wrapped_unread_bytes ||
        left.files.size() != right.files.size()) {
        return false;
    }
    for (std::size_t file = 0; file < left.files.size(); ++file) {
        const FileWindowPlan& a = left.files[file];
        const FileWindowPlan& b = right.files[file];
        if (a.path != b.path || a.windows.size() != b.windows.size() ||
            a.slices.size() != b.slices.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.windows.size(); ++i) {
            if (a.windows[i].start != b.windows[i].start ||
                a.windows[i].end != b.windows[i].end) {
                return false;
            }
        }
        for (std::size_t i = 0; i < a.slices.size(); ++i) {
            const WindowSlice& x = a.slices[i];
            const WindowSlice& y = b.slices[i];
            if (x.tensor_name != y.tensor_name || x.window_index != y.window_index ||
                x.file_offset != y.file_offset || x.window_offset != y.window_offset ||
                x.byte_count != y.byte_count || x.expert_index != y.expert_index) {
                return false;
            }
        }
    }
    return true;
}

bool validate_geometry_sources(const Geometry& geometry, const WindowSources& sources,
                               const WindowPlan& plan, std::string& error) {
    if (geometry.shards.size() != sources.files.size() ||
        plan.files.size() != sources.files.size()) {
        error = "full-mode sources do not match the shard inventory in the memory plan";
        return false;
    }
    for (std::size_t shard = 0; shard < sources.files.size(); ++shard) {
        const WindowFileSource& source = sources.files[shard];
        const ShardGeometry& planned = geometry.shards[shard];
        if (planned.name != std::filesystem::path(source.path).filename().string() ||
            planned.bytes != source.file_size) {
            error = source.path + ": shard name or size differs from the memory plan";
            return false;
        }
    }
    if (geometry.wrapped_unread_bytes != plan.wrapped_unread_bytes) {
        error = "full-mode window union does not match wrapped_unread_bytes in the memory plan";
        return false;
    }
    return true;
}

const WindowSourceFileLock* find_inspection_lock(const WindowSources& sources,
                                                  const std::string& path) {
    const auto found = std::find_if(
        sources.inspected_file_locks.begin(), sources.inspected_file_locks.end(),
        [&](const WindowSourceFileLock& lock) {
            return lock.path == path && lock.lock_lease() != nullptr;
        });
    return found == sources.inspected_file_locks.end() ? nullptr : &*found;
}

bool validate_locked_path(const WindowSources& sources, const std::string& path,
                          std::uint64_t expected_size, std::uint64_t expected_device,
                          std::uint64_t expected_object, std::string& error) {
    const WindowSourceFileLock* lock = find_inspection_lock(sources, path);
    if (lock == nullptr || lock->file_size != expected_size ||
        lock->file_device_id != expected_device || lock->file_object_id != expected_object) {
        error = path + ": missing or mismatched WeightStore inspection lock";
        return false;
    }
    if (!lock->lock_lease()->matches(path, expected_size, expected_device, expected_object)) {
        error = path + ": WeightStore inspection lock lease does not own the inspected file";
        return false;
    }
    struct stat status {};
    if (::stat(path.c_str(), &status) != 0 || status.st_size < 0 ||
        static_cast<std::uint64_t>(status.st_size) != expected_size ||
        static_cast<std::uint64_t>(status.st_dev) != expected_device ||
        static_cast<std::uint64_t>(status.st_ino) != expected_object) {
        error = path + ": source identity changed after WeightStore inspection";
        return false;
    }
    return true;
}

bool artifact_set_lock_path(const std::filesystem::path& directory,
                            std::filesystem::path& result, std::string& error) {
    std::error_code ec;
    const std::filesystem::path resolved = std::filesystem::weakly_canonical(directory, ec);
    if (ec || resolved.filename().empty()) {
        error = "cannot resolve artifact-set directory " + directory.string() +
                (ec ? ": " + ec.message() : "");
        return false;
    }
    result = resolved.parent_path() /
             ("." + resolved.filename().string() + ".strata-artifact-set.lock");
    return true;
}

bool validate_inspection_locks(const WindowSources& sources, std::string& error) {
    if (sources.files.empty() || sources.dense_pack_path.empty()) {
        error = "full-mode sources are missing their WeightStore inspection locks";
        return false;
    }
    for (const WindowFileSource& source : sources.files) {
        if (!validate_locked_path(sources, source.path, source.file_size,
                                  source.file_device_id, source.file_object_id, error)) {
            return false;
        }
    }
    if (!validate_locked_path(sources, sources.dense_pack_path,
                              sources.dense_pack_file_size,
                              sources.dense_pack_device_id,
                              sources.dense_pack_object_id, error)) {
        return false;
    }
    const WindowSourceFileLock* dense_lock =
        find_inspection_lock(sources, sources.dense_pack_path);
    if (dense_lock == nullptr ||
        !dense_lock->lock_lease()->matches_dense_pack_catalog(
            sources.dense_pack_tensors)) {
        error = sources.dense_pack_path +
                ": converted row inventory differs from the WeightStore index inspection";
        return false;
    }

    const std::filesystem::path source_path(sources.files.front().path);
    const std::filesystem::path dense_path(sources.dense_pack_path);
    const std::filesystem::path source_directory = source_path.has_parent_path()
        ? source_path.parent_path()
        : std::filesystem::current_path();
    const std::filesystem::path dense_directory = dense_path.has_parent_path()
        ? dense_path.parent_path()
        : std::filesystem::current_path();
    std::filesystem::path source_lock_path;
    std::filesystem::path pack_lock_path;
    if (!artifact_set_lock_path(source_directory, source_lock_path, error) ||
        !artifact_set_lock_path(dense_directory, pack_lock_path, error)) {
        return false;
    }
    for (const std::filesystem::path& lock_path :
         {source_lock_path, pack_lock_path}) {
        const WindowSourceFileLock* lock =
            find_inspection_lock(sources, lock_path.string());
        if (lock == nullptr ||
            !lock->lock_lease()->matches(lock_path.string(), lock->file_size,
                                         lock->file_device_id, lock->file_object_id)) {
            error = lock_path.string() +
                    ": missing WeightStore artifact-set inspection lock";
            return false;
        }
        struct stat status {};
        if (::stat(lock_path.c_str(), &status) != 0 || status.st_size < 0 ||
            static_cast<std::uint64_t>(status.st_size) != lock->file_size ||
            static_cast<std::uint64_t>(status.st_dev) != lock->file_device_id ||
            static_cast<std::uint64_t>(status.st_ino) != lock->file_object_id) {
            error = lock_path.string() +
                    ": artifact-set identity changed after WeightStore inspection";
            return false;
        }
    }
    error.clear();
    return true;
}

bool validate_ple_window_exclusion(const strata::GgufModel& model,
                                   const WindowSources& sources,
                                   const WindowPlan& plan,
                                   std::string& error) {
    std::size_t ple_shard = 0;
    const strata::TensorInfo* ple = model.find("per_layer_token_embd.weight", &ple_shard);
    if (ple == nullptr || ple_shard >= model.size() || ple_shard >= sources.files.size() ||
        ple_shard >= plan.files.size()) {
        error = "full-mode GGUF is missing its excluded PLE tensor";
        return false;
    }
    const strata::GgufFile& file = model.shard(ple_shard);
    std::uint64_t ple_start = 0;
    std::uint64_t ple_bytes = strata::tensor_payload_bytes(*ple);
    std::uint64_t ple_end = 0;
    if (ple_bytes == 0 || !add(file.data_start(), ple->offset, ple_start) ||
        !add(ple_start, ple_bytes, ple_end) || ple_end > file.file_size()) {
        error = file.path() + ": PLE payload has an invalid file range";
        return false;
    }
    for (const WindowTensorSource& tensor : sources.files[ple_shard].tensors) {
        if (tensor.tensor_name == ple->name) {
            error = file.path() + ": the PLE tensor must not be a full-mode window source";
            return false;
        }
    }

    for (const FileWindowPlan& file_plan : plan.files) {
        for (const WindowSlice& slice : file_plan.slices) {
            if (slice.tensor_name == ple->name) {
                error = file_plan.path + ": the PLE tensor must not have a window slice";
                return false;
            }
        }
    }

    // Overlap, padding, and union rules are shared with the device-free tests.
    return verify_window_never_wrap(sources, plan, error);
}

std::string window_label(const WindowFileSource& source, std::size_t index) {
    return "GGUF " + std::filesystem::path(source.path).filename().string() +
           " window " + std::to_string(index);
}

struct WarmTask {
    std::size_t file = 0;
    std::size_t window = 0;
    std::uint64_t first_page = 0;
    std::uint64_t page_count = 0;
};

struct SliceLocation {
    std::size_t file = 0;
    const WindowSlice* slice = nullptr;
};

bool append_warm_tasks(std::vector<WarmTask>& tasks, std::size_t file, std::size_t window,
                       std::uint64_t first_page, std::uint64_t page_count) {
    constexpr std::uint64_t kChunkPages = (64ull * 1024 * 1024) / kWindowPageBytes;
    while (page_count != 0) {
        const std::uint64_t pages = std::min(page_count, kChunkPages);
        tasks.push_back({file, window, first_page, pages});
        first_page += pages;
        page_count -= pages;
    }
    return true;
}

bool mark_slice_pages(std::vector<WarmTask>& tasks, std::vector<std::vector<std::vector<std::uint8_t>>>& warmed,
                      std::size_t file, const WindowSlice& slice, std::string& error) {
    if (slice.window_index >= warmed[file].size()) {
        error = "expert warm-up slice references a missing file window";
        return false;
    }
    std::uint64_t end = 0;
    if (!add(slice.window_offset, slice.byte_count, end) ||
        end > warmed[file][static_cast<std::size_t>(slice.window_index)].size() * kWindowPageBytes) {
        error = "expert warm-up slice falls outside its file window";
        return false;
    }
    const std::uint64_t first_page = slice.window_offset / kWindowPageBytes;
    const std::uint64_t past_page = end / kWindowPageBytes + (end % kWindowPageBytes != 0 ? 1 : 0);
    auto& pages = warmed[file][static_cast<std::size_t>(slice.window_index)];
    std::uint64_t run_start = kNoExpertIndex;
    for (std::uint64_t page = first_page; page < past_page; ++page) {
        if (pages[static_cast<std::size_t>(page)] == 0) {
            pages[static_cast<std::size_t>(page)] = 1;
            if (run_start == kNoExpertIndex) run_start = page;
        } else if (run_start != kNoExpertIndex) {
            append_warm_tasks(tasks, file, static_cast<std::size_t>(slice.window_index),
                              run_start, page - run_start);
            run_start = kNoExpertIndex;
        }
    }
    if (run_start != kNoExpertIndex) {
        append_warm_tasks(tasks, file, static_cast<std::size_t>(slice.window_index),
                          run_start, past_page - run_start);
    }
    return true;
}

bool make_warm_tasks(const WindowPlan& plan, const std::vector<std::vector<BufferRef>>& windows,
                     std::uint32_t n_layers, std::uint32_t n_expert,
                     const std::vector<std::pair<std::uint32_t, std::uint32_t>>& expert_order,
                     std::vector<std::vector<WarmTask>>& prioritized,
                     std::vector<WarmTask>& remaining, std::string& error) {
    prioritized.clear();
    remaining.clear();
    if (plan.files.size() != windows.size()) {
        error = "warm-up window plan does not match mapped buffers";
        return false;
    }
    std::vector<std::vector<std::vector<std::uint8_t>>> warmed(windows.size());
    std::map<std::pair<std::string, std::uint64_t>, SliceLocation> expert_slices;
    for (std::size_t file = 0; file < windows.size(); ++file) {
        if (plan.files[file].windows.size() != windows[file].size()) {
            error = plan.files[file].path + ": warm-up buffer count does not match the plan";
            return false;
        }
        warmed[file].reserve(windows[file].size());
        for (const BufferRef& buffer : windows[file]) {
            if (!buffer || buffer.length == 0 || buffer.length % kWindowPageBytes != 0) {
                error = plan.files[file].path + ": warm-up buffer is not page aligned";
                return false;
            }
            const std::size_t pages = buffer.length / kWindowPageBytes;
            warmed[file].emplace_back(pages, 0);
        }
        for (const WindowSlice& slice : plan.files[file].slices) {
            if (slice.expert_index == kNoExpertIndex) continue;
            if (!expert_slices.emplace(std::make_pair(slice.tensor_name, slice.expert_index),
                                       SliceLocation{file, &slice}).second) {
                error = "warm-up plan contains a duplicate expert slice";
                return false;
            }
        }
    }

    std::vector<std::pair<std::uint32_t, std::uint32_t>> ordered_experts;
    if (!order_full_mode_warm_experts(n_layers, n_expert, expert_order, ordered_experts, error)) {
        return false;
    }

    for (const auto& [layer, expert] : ordered_experts) {
        std::vector<WarmTask> expert_tasks;
        const std::string prefix = "blk." + std::to_string(layer) + ".ffn_";
        const char* roles[] = {"gate_exps.weight", "up_exps.weight", "down_exps.weight"};
        for (const char* role : roles) {
            const auto found = expert_slices.find({prefix + role, expert});
            if (found == expert_slices.end() ||
                !mark_slice_pages(expert_tasks, warmed, found->second.file, *found->second.slice, error)) {
                if (error.empty()) error = "expert warm-up order references a missing tensor slab";
                return false;
            }
        }
        prioritized.push_back(std::move(expert_tasks));
    }

    for (std::size_t file = 0; file < warmed.size(); ++file) {
        for (std::size_t window = 0; window < warmed[file].size(); ++window) {
            const auto& pages = warmed[file][window];
            std::uint64_t run_start = kNoExpertIndex;
            for (std::uint64_t page = 0; page < pages.size(); ++page) {
                if (pages[static_cast<std::size_t>(page)] == 0) {
                    if (run_start == kNoExpertIndex) run_start = page;
                } else if (run_start != kNoExpertIndex) {
                    append_warm_tasks(remaining, file, window, run_start, page - run_start);
                    run_start = kNoExpertIndex;
                }
            }
            if (run_start != kNoExpertIndex) {
                append_warm_tasks(remaining, file, window, run_start, pages.size() - run_start);
            }
        }
    }
    return true;
}

bool warm_cpu_pages(const std::vector<std::vector<BufferRef>>& windows,
                    const std::vector<std::vector<WarmTask>>& prioritized,
                    const std::vector<WarmTask>& remaining, std::string& error) {
    std::atomic<std::uint64_t> checksum{0};
    std::atomic<int> advise_error{0};
    auto warm_task = [&](const WarmTask& task) {
        const BufferRef& buffer = windows[task.file][task.window];
        const std::uint64_t offset = task.first_page * kWindowPageBytes;
        const std::size_t length = static_cast<std::size_t>(task.page_count * kWindowPageBytes);
        auto* chunk = static_cast<std::uint8_t*>(buffer.contents) + offset;
        if (madvise(chunk, length, MADV_WILLNEED) != 0) {
            int expected = 0;
            (void)advise_error.compare_exchange_strong(expected, errno, std::memory_order_relaxed);
        }
        const volatile std::uint8_t* pages = chunk;
        std::uint64_t local = 0;
        for (std::uint64_t page = 0; page < task.page_count; ++page) {
            local ^= pages[page * kWindowPageBytes];
        }
        checksum.fetch_xor(local, std::memory_order_relaxed);
    };
    if (!run_full_mode_warmup_waves(
            prioritized.size(),
            [&](std::size_t, std::size_t expert_index) {
                for (const WarmTask& task : prioritized[expert_index]) warm_task(task);
            },
            {}, error)) {
        return false;
    }
    if (!run_full_mode_warmup_waves(
            remaining.size(),
            [&](std::size_t, std::size_t task_index) { warm_task(remaining[task_index]); },
            {}, error)) {
        return false;
    }
    (void)checksum.load(std::memory_order_relaxed);
    const int advice = advise_error.load(std::memory_order_relaxed);
    if (advice != 0) {
        error = "madvise(MADV_WILLNEED) failed during GGUF warm-up: " +
                std::error_code(advice, std::generic_category()).message();
        return false;
    }
    error.clear();
    return true;
}

}  // namespace

bool run_full_mode_warmup_waves(
    std::size_t work_count,
    const std::function<void(std::size_t worker, std::size_t item)>& work,
    const std::function<bool(std::size_t wave, std::size_t completed_items)>& observer,
    std::string& error) {
    error.clear();
    if (work_count == 0) return true;
    if (!work || work_count > std::numeric_limits<std::size_t>::max() -
                                   (kFullModeWarmWorkers - 1)) {
        error = "invalid full-mode warm-up work";
        return false;
    }
    const std::size_t wave_count =
        (work_count + kFullModeWarmWorkers - 1) / kFullModeWarmWorkers;
    if (observer) {
        try {
            if (!observer(0, 0)) {
                error = "full-mode warm-up wave observer rejected the initial wave";
                return false;
            }
        } catch (...) {
            error = "full-mode warm-up wave observer failed on the initial wave";
            return false;
        }
    }
    std::barrier wave_barrier(static_cast<std::ptrdiff_t>(kFullModeWarmWorkers));
    std::atomic<std::size_t> completed{0};
    std::atomic<bool> callback_failed{false};
    std::mutex start_mutex;
    std::condition_variable start_condition;
    bool workers_started = false;
    bool abort_workers = false;
    std::vector<std::thread> workers;
    workers.reserve(kFullModeWarmWorkers);
    try {
        for (std::size_t worker_index = 0; worker_index < kFullModeWarmWorkers; ++worker_index) {
            workers.emplace_back([&, worker_index]() {
                {
                    std::unique_lock<std::mutex> lock(start_mutex);
                    start_condition.wait(lock, [&]() { return workers_started || abort_workers; });
                    if (abort_workers) return;
                }
                (void)pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
                for (std::size_t wave = 0; wave < wave_count; ++wave) {
                    if (wave != 0 && worker_index == 0 && observer) {
                        const std::size_t done = completed.load(std::memory_order_acquire);
                        try {
                            if (!observer(wave, done)) {
                                callback_failed.store(true, std::memory_order_release);
                            }
                        } catch (...) {
                            callback_failed.store(true, std::memory_order_release);
                        }
                    }
                    wave_barrier.arrive_and_wait();
                    const std::size_t item = wave * kFullModeWarmWorkers + worker_index;
                    if (item < work_count && !callback_failed.load(std::memory_order_acquire)) {
                        try {
                            work(worker_index, item);
                            completed.fetch_add(1, std::memory_order_release);
                        } catch (...) {
                            callback_failed.store(true, std::memory_order_release);
                        }
                    }
                    wave_barrier.arrive_and_wait();
                }
            });
        }
    } catch (const std::exception&) {
        {
            std::lock_guard<std::mutex> lock(start_mutex);
            abort_workers = true;
        }
        start_condition.notify_all();
        for (std::thread& thread : workers) if (thread.joinable()) thread.join();
        error = "cannot create the eight full-mode warm-up workers";
        return false;
    }
    {
        std::lock_guard<std::mutex> lock(start_mutex);
        workers_started = true;
    }
    start_condition.notify_all();
    for (std::thread& thread : workers) thread.join();
    if (callback_failed.load(std::memory_order_acquire) ||
        completed.load(std::memory_order_acquire) != work_count) {
        error = "full-mode warm-up wave failed before completing all work";
        return false;
    }
    return true;
}

struct FullModeStorage::Impl {
    Context* context = nullptr;
    std::shared_ptr<strata::GgufModel> model;
    Geometry geometry;
    WindowSources sources;
    WindowPlan plan;
    std::vector<std::vector<BufferRef>> windows;
    std::vector<std::vector<ReadOnlyBufferRef>> read_only_windows;
    BufferRef dense_pack_arena;
    FullModeLayout layout;
    BufferRef expert_addresses;
    BufferRef resident;
    BufferRef usage;
    std::unique_ptr<strata::kernels::PleTable> ple_table;
    strata::kernels::PleTableReader ple_reader;
};

FullModeStorage::FullModeStorage(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}
FullModeStorage::~FullModeStorage() = default;

std::unique_ptr<FullModeStorage> FullModeStorage::open(Context& context,
                                                        const std::string& native_shard,
                                                        const Geometry& geometry,
                                                        const WindowSources& sources,
                                                        const WindowPlan& plan,
                                                        std::string& error) {
    error.clear();
    if (native_shard.empty() || geometry.n_layers == 0 || geometry.n_expert == 0) {
        error = "full-mode storage needs a GGUF shard and non-zero model dimensions";
        return nullptr;
    }
    if (!validate_geometry(geometry, error) ||
        !validate_inspection_locks(sources, error)) {
        return nullptr;
    }

    auto candidate = std::make_unique<Impl>();
    candidate->context = &context;
    candidate->geometry = geometry;
    candidate->sources = sources;
    candidate->plan = plan;
    try {
        candidate->model = std::make_shared<strata::GgufModel>(
            strata::gguf_split_paths(native_shard));
    } catch (const std::exception& exception) {
        error = exception.what();
        return nullptr;
    }
    if (!validate_model_sources(*candidate->model, sources, geometry, error)) return nullptr;
    const DeviceFacts facts = context.device_facts();
    if (facts.max_buffer_length == 0) {
        error = "Metal device reported an invalid maxBufferLength";
        return nullptr;
    }
    WindowPlan verified_plan;
    if (!plan_windows(sources, facts.max_buffer_length, verified_plan, error)) return nullptr;
    if (!same_plan(plan, verified_plan)) {
        error = "full-mode window plan does not match the exact planned tensor union";
        return nullptr;
    }
    if (!validate_geometry_sources(geometry, sources, verified_plan, error) ||
        !validate_ple_window_exclusion(*candidate->model, sources, verified_plan, error)) {
        return nullptr;
    }
    std::size_t ple_shard = 0;
    if (candidate->model->find("per_layer_token_embd.weight", &ple_shard) == nullptr ||
        ple_shard >= candidate->model->size()) {
        error = "full-mode GGUF is missing its excluded PLE tensor";
        return nullptr;
    }
    candidate->ple_table = std::make_unique<strata::kernels::PleTable>();
    if (!candidate->ple_table->open_mapped(candidate->model, ple_shard, error)) {
        return nullptr;
    }
    candidate->ple_table->reader_view(candidate->ple_reader);
    std::vector<std::vector<std::uint64_t>> gpu_bases(sources.files.size());
    for (std::size_t shard = 0; shard < sources.files.size(); ++shard) {
        gpu_bases[shard].assign(plan.files[shard].windows.size(), 1);
    }
    FullModeLayout preflight;
    if (!build_full_mode_layout(sources, plan, gpu_bases, facts.max_buffer_length,
                                geometry.n_layers, geometry.n_expert, preflight, error,
                                0x300000000ull)) {
        return nullptr;
    }
    if (preflight.dense_pack_arena_bytes > facts.max_buffer_length ||
        preflight.dense_pack_arena_bytes > std::numeric_limits<std::size_t>::max()) {
        error = "full-mode converted dense-pack arena exceeds the live Metal buffer limit";
        return nullptr;
    }

    auto buffer_transaction = context.buffers().transaction();
    candidate->windows.resize(sources.files.size());
    candidate->read_only_windows.resize(sources.files.size());
    for (std::size_t shard = 0; shard < sources.files.size(); ++shard) {
        const WindowFileSource& source = sources.files[shard];
        const FileWindowPlan& file_plan = plan.files[shard];
        if (file_plan.path != source.path) {
            error = source.path + ": file window plan does not match its source";
            return nullptr;
        }
        const strata::GgufFile& file = candidate->model->shard(shard);
        candidate->windows[shard].reserve(file_plan.windows.size());
        candidate->read_only_windows[shard].reserve(file_plan.windows.size());
        gpu_bases[shard].clear();
        gpu_bases[shard].reserve(file_plan.windows.size());
        for (std::size_t index = 0; index < file_plan.windows.size(); ++index) {
            const FileWindow& range = file_plan.windows[index];
            if (range.start > std::numeric_limits<std::size_t>::max() ||
                range.end <= range.start || range.end - range.start > std::numeric_limits<std::size_t>::max()) {
                error = source.path + ": file window cannot be represented by this process";
                return nullptr;
            }
            auto* address = const_cast<std::uint8_t*>(file.mapping_data() + range.start);
            BufferRef buffer = context.buffers().wrap_no_copy(
                window_label(source, index), address, static_cast<std::size_t>(range.end - range.start),
                std::static_pointer_cast<void>(candidate->model), error);
            if (!buffer) return nullptr;
            candidate->windows[shard].push_back(buffer);
            candidate->read_only_windows[shard].push_back(ReadOnlyBufferRef(buffer));
            gpu_bases[shard].push_back(buffer.gpu_address);
        }
    }

    if (preflight.dense_pack_arena_bytes != 0) {
        candidate->dense_pack_arena = context.buffers().allocate(
            "full-mode converted dense-pack arena",
            static_cast<std::size_t>(preflight.dense_pack_arena_bytes), error);
        if (!candidate->dense_pack_arena) return nullptr;
        if (candidate->dense_pack_arena.gpu_address == 0 ||
            candidate->dense_pack_arena.gpu_address % 256 != 0) {
            error = "full-mode converted dense-pack arena has an invalid GPU base address";
            return nullptr;
        }
        std::memset(candidate->dense_pack_arena.contents, 0,
                    candidate->dense_pack_arena.length);
        const WindowSourceFileLock* pack_lock =
            find_inspection_lock(sources, sources.dense_pack_path);
        if (pack_lock == nullptr || pack_lock->lock_lease() == nullptr) {
            error = sources.dense_pack_path +
                    ": missing WeightStore dense-pack descriptor lease";
            return nullptr;
        }
        std::uint64_t arena_cursor = 0;
        for (const DensePackTensorSource& tensor : sources.dense_pack_tensors) {
            std::uint64_t remainder = arena_cursor % 256;
            if (remainder != 0 &&
                !add(arena_cursor, 256 - remainder, arena_cursor)) {
                error = "full-mode converted dense-pack arena offset overflows";
                return nullptr;
            }
            std::uint64_t arena_end = 0;
            if (arena_cursor > candidate->dense_pack_arena.length ||
                !add(arena_cursor, tensor.byte_count, arena_end) ||
                arena_end > candidate->dense_pack_arena.length ||
                arena_cursor > std::numeric_limits<std::size_t>::max()) {
                error = "full-mode converted dense-pack row exceeds its arena";
                return nullptr;
            }
            auto* destination = static_cast<std::uint8_t*>(
                                    candidate->dense_pack_arena.contents) +
                                static_cast<std::size_t>(arena_cursor);
            if (!pack_lock->lock_lease()->read_exact(
                    tensor.file_offset, destination, tensor.source_byte_count, error)) {
                return nullptr;
            }
            arena_cursor = arena_end;
        }
    }

    if (!build_full_mode_layout(sources, plan, gpu_bases, facts.max_buffer_length,
                                geometry.n_layers, geometry.n_expert, candidate->layout, error,
                                candidate->dense_pack_arena
                                    ? candidate->dense_pack_arena.gpu_address : 0)) {
        return nullptr;
    }
    if (candidate->layout.dense_pack_arena_bytes !=
            candidate->dense_pack_arena.length ||
        candidate->layout.dense_pack_bytes != geometry.dense_pack_bytes) {
        error = "full-mode converted dense-pack arena does not match validated geometry";
        return nullptr;
    }
    const std::size_t usage_entries =
        static_cast<std::size_t>(kFullModeMaxLayers) * geometry.n_expert;
    if (candidate->layout.expert_addresses.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::uint64_t) ||
        candidate->layout.resident.size() >
            std::numeric_limits<std::size_t>::max() / sizeof(std::int32_t) ||
        usage_entries > std::numeric_limits<std::size_t>::max() / sizeof(float)) {
        error = "full-mode address, resident or usage table size overflows";
        return nullptr;
    }
    const std::size_t address_bytes = candidate->layout.expert_addresses.size() * sizeof(std::uint64_t);
    const std::size_t resident_bytes = candidate->layout.resident.size() * sizeof(std::int32_t);
    const std::size_t usage_bytes = usage_entries * sizeof(float);
    candidate->expert_addresses =
        context.buffers().allocate("expert address table", address_bytes, error);
    if (!candidate->expert_addresses) return nullptr;
    candidate->resident = context.buffers().allocate("resident table", resident_bytes, error);
    if (!candidate->resident) return nullptr;
    candidate->usage = context.buffers().allocate("MoE usage table", usage_bytes, error);
    if (!candidate->usage) return nullptr;
    std::memcpy(candidate->expert_addresses.contents, candidate->layout.expert_addresses.data(),
                address_bytes);
    std::memcpy(candidate->resident.contents, candidate->layout.resident.data(), resident_bytes);
    std::memset(candidate->usage.contents, 0, usage_bytes);
    if (!context.buffers().commit_residency(error)) return nullptr;

    auto storage = std::unique_ptr<FullModeStorage>(new FullModeStorage(std::move(candidate)));
    buffer_transaction.commit();
    return storage;
}

const FullModeLayout& FullModeStorage::layout() const noexcept { return impl_->layout; }

const std::vector<std::vector<ReadOnlyBufferRef>>& FullModeStorage::windows() const noexcept {
    return impl_->read_only_windows;
}

ReadOnlyBufferRef FullModeStorage::dense_pack_arena() const noexcept {
    return ReadOnlyBufferRef(impl_->dense_pack_arena);
}

ReadOnlyBufferRef FullModeStorage::expert_address_table() const noexcept {
    return ReadOnlyBufferRef(impl_->expert_addresses);
}

ReadOnlyBufferRef FullModeStorage::resident_table() const noexcept {
    return ReadOnlyBufferRef(impl_->resident);
}

BufferRef FullModeStorage::usage_table() const noexcept { return impl_->usage; }

strata::kernels::PleTableReader* FullModeStorage::ple_reader() noexcept {
    return impl_ != nullptr && impl_->ple_table != nullptr
               ? &impl_->ple_reader
               : nullptr;
}

bool FullModeStorage::dense_tensor_view(const std::string& tensor_name,
                                        FullModeTensorView& view,
                                        std::string& error) const {
    error.clear();
    view = {};
    if (impl_ == nullptr || impl_->context == nullptr || impl_->model == nullptr) {
        error = "full-mode storage is not attached to a live model and Context";
        return false;
    }
    if (tensor_name.empty()) {
        error = "full-mode tensor name must not be empty";
        return false;
    }

    std::size_t shard = 0;
    const strata::TensorInfo* tensor = impl_->model->find(tensor_name, &shard);
    if (tensor == nullptr) {
        error = "full-mode model has no tensor named " + tensor_name;
        return false;
    }
    const auto binding = std::find_if(
        impl_->layout.tensors.begin(), impl_->layout.tensors.end(),
        [&](const TensorGpuBinding& candidate) {
            return candidate.tensor_name == tensor_name &&
                   candidate.expert_index == kNoExpertIndex;
        });
    if (binding == impl_->layout.tensors.end()) {
        const bool has_expert_slices = std::any_of(
            impl_->layout.tensors.begin(), impl_->layout.tensors.end(),
            [&](const TensorGpuBinding& candidate) {
                return candidate.tensor_name == tensor_name;
            });
        error = has_expert_slices
                    ? "full-mode tensor is stored as per-expert slabs: " + tensor_name
                    : "full-mode tensor is excluded from the dense view catalog: " + tensor_name;
        return false;
    }

    BufferRef candidate_view;
    std::uint32_t runtime_type = tensor->type;
    const auto packed = std::find_if(
        impl_->sources.dense_pack_tensors.begin(),
        impl_->sources.dense_pack_tensors.end(),
        [&](const DensePackTensorSource& source) {
            return source.tensor_name == tensor_name;
        });
    if (packed != impl_->sources.dense_pack_tensors.end()) {
        std::uint64_t elements = 1;
        for (const std::uint64_t dimension : tensor->shape) {
            if (dimension == 0 ||
                elements > std::numeric_limits<std::uint64_t>::max() / dimension) {
                error = "converted full-mode tensor has invalid dimensions: " + tensor_name;
                return false;
            }
            elements *= dimension;
        }
        const std::uint64_t source_payload = strata::tensor_payload_bytes(*tensor);
        if (packed->kind != 4 || tensor->type != 0 ||
            source_payload == 0 || source_payload % 2 != 0 ||
            source_payload / 2 != packed->source_byte_count ||
            elements > std::numeric_limits<std::uint64_t>::max() / 2 ||
            elements * 2 != packed->byte_count ||
            binding->byte_count != packed->byte_count ||
            !impl_->dense_pack_arena ||
            binding->gpu_address < impl_->dense_pack_arena.gpu_address) {
            error = "converted full-mode tensor metadata does not match its BF16 arena row: " +
                    tensor_name;
            return false;
        }
        const std::uint64_t offset =
            binding->gpu_address - impl_->dense_pack_arena.gpu_address;
        if (offset > std::numeric_limits<std::size_t>::max() ||
            offset > impl_->dense_pack_arena.length ||
            binding->byte_count > impl_->dense_pack_arena.length - offset) {
            error = "converted full-mode tensor view exceeds its dense-pack arena: " +
                    tensor_name;
            return false;
        }
        candidate_view = impl_->dense_pack_arena.slice(
            static_cast<std::size_t>(offset),
            static_cast<std::size_t>(binding->byte_count));
        runtime_type = kGgmlTypeBf16;
    } else {
        if (shard >= impl_->sources.files.size() ||
            shard >= impl_->plan.files.size() ||
            shard >= impl_->windows.size()) {
            error = "full-mode tensor shard is missing from the window catalog: " +
                    tensor_name;
            return false;
        }
        const std::uint64_t payload = strata::tensor_payload_bytes(*tensor);
        if (payload == 0 || payload != binding->byte_count) {
            error = "full-mode dense tensor byte count does not match its GGUF metadata: " +
                    tensor_name;
            return false;
        }
        const WindowSlice* slice = nullptr;
        for (const WindowSlice& planned : impl_->plan.files[shard].slices) {
            if (planned.tensor_name != tensor_name ||
                planned.expert_index != kNoExpertIndex) {
                continue;
            }
            if (slice != nullptr) {
                error = "full-mode dense tensor has duplicate window slices: " + tensor_name;
                return false;
            }
            slice = &planned;
        }
        if (slice == nullptr || slice->byte_count != binding->byte_count ||
            slice->byte_count > std::numeric_limits<std::size_t>::max() ||
            slice->window_index >= impl_->windows[shard].size()) {
            error = "full-mode dense tensor has no matching window slice: " + tensor_name;
            return false;
        }
        const BufferRef& window =
            impl_->windows[shard][static_cast<std::size_t>(slice->window_index)];
        if (slice->window_offset > std::numeric_limits<std::size_t>::max()) {
            error = "full-mode dense tensor window offset exceeds process limits: " +
                    tensor_name;
            return false;
        }
        candidate_view = window.slice(static_cast<std::size_t>(slice->window_offset),
                                      static_cast<std::size_t>(slice->byte_count));
    }

    if (!candidate_view || !impl_->context->buffers().owns(candidate_view) ||
        candidate_view.length != binding->byte_count ||
        candidate_view.gpu_address != binding->gpu_address) {
        error = "full-mode tensor view does not match its live GPU binding: " + tensor_name;
        return false;
    }

    view.ggml_type = runtime_type;
    view.shape = tensor->shape;
    view.buffer = ReadOnlyBufferRef(candidate_view);
    return true;
}

bool FullModeStorage::warm_up(
    const std::vector<std::pair<std::uint32_t, std::uint32_t>>& expert_order,
    std::string& error) {
    error.clear();
    if (impl_ == nullptr || impl_->context == nullptr) {
        error = "full-mode storage is not attached to a live Metal Context";
        return false;
    }
    try {
        std::vector<std::vector<WarmTask>> prioritized;
        std::vector<WarmTask> remaining;
        if (!make_warm_tasks(impl_->plan, impl_->windows, impl_->geometry.n_layers,
                             impl_->geometry.n_expert, expert_order, prioritized, remaining, error)) {
            return false;
        }
        std::uint64_t total_bytes = 0;
        std::vector<BufferRef> warm_buffers;
        for (const auto& file_windows : impl_->windows) {
            for (const BufferRef& window : file_windows) {
                if (!add(total_bytes, window.length, total_bytes)) {
                    error = "full-mode warm-up byte count overflows";
                    return false;
                }
                warm_buffers.push_back(window);
            }
        }
        if (impl_->dense_pack_arena) {
            if (!add(total_bytes, impl_->dense_pack_arena.length, total_bytes)) {
                error = "full-mode warm-up byte count overflows";
                return false;
            }
            warm_buffers.push_back(impl_->dense_pack_arena);
        }

        const auto started = std::chrono::steady_clock::now();
        if (!warm_cpu_pages(impl_->windows, prioritized, remaining, error) ||
            !impl_->context->warm_no_copy_windows(warm_buffers, error)) {
            return false;
        }
        const double seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const double gib = static_cast<double>(total_bytes) / (1024.0 * 1024 * 1024);
        const double gigabytes_per_second =
            seconds > 0 ? static_cast<double>(total_bytes) / seconds / 1'000'000'000.0 : 0;
        std::fprintf(stderr, "strata: warmed %.2f GiB in %.2f s (%.1f GB/s)\n",
                     gib, seconds, gigabytes_per_second);
        error.clear();
        return true;
    } catch (const std::exception& exception) {
        error = std::string("full-mode warm-up failed: ") + exception.what();
        return false;
    }
}

}  // namespace strata::metal
