// src/metal/engine/window_plan.cpp - validate and pack GGUF ranges into no-copy windows.
#include "strata/metal/window_plan.hpp"

#include <algorithm>
#include <limits>
#include <map>
#include <utility>

namespace strata::metal {
namespace {

constexpr std::uint64_t kMaximumExpertSlabs = 512;

struct Atom {
    std::string tensor_name;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint64_t expert_index = kNoExpertIndex;
};

struct Group {
    std::vector<Atom> atoms;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
    std::uint64_t read_bytes = 0;
};

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    const std::uint64_t remainder = value % alignment;
    return remainder == 0 || add(value, alignment - remainder, result)
        ? (remainder == 0 ? (result = value, true) : true)
        : false;
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value - value % alignment;
}

bool append_group(std::vector<Atom>& atoms, Group& group, std::uint64_t& unread,
                  const std::string& path, std::string& error) {
    if (atoms.empty()) return true;
    group = {};
    group.atoms = std::move(atoms);
    group.start = group.atoms.front().start;
    group.end = group.atoms.back().end;
    for (const Atom& atom : group.atoms) {
        const std::uint64_t size = atom.end - atom.start;
        if (!add(group.read_bytes, size, group.read_bytes)) {
            error = path + ": window read-byte total overflows";
            return false;
        }
    }
    const std::uint64_t aligned_start = align_down(group.start, kWindowPageBytes);
    std::uint64_t aligned_end = 0;
    if (!align_up(group.end, kWindowPageBytes, aligned_end) ||
        aligned_end < aligned_start || aligned_end - aligned_start < group.read_bytes ||
        !add(unread, aligned_end - aligned_start - group.read_bytes, unread)) {
        error = path + ": wrapped-unread byte count overflows";
        return false;
    }
    atoms.clear();
    return true;
}

bool make_groups(const WindowFileSource& file, std::vector<Group>& groups,
                 std::uint64_t& unread, std::string& error) {
    groups.clear();
    unread = 0;
    if (file.path.empty() || file.file_size == 0) {
        error = "window source has an empty path or file size";
        return false;
    }

    std::vector<Atom> atoms;
    std::map<std::string, bool> names;
    for (const WindowTensorSource& tensor : file.tensors) {
        std::uint64_t end = 0;
        if (tensor.tensor_name.empty() || !names.emplace(tensor.tensor_name, true).second) {
            error = file.path + ": tensor names must be non-empty and unique";
            return false;
        }
        if (tensor.byte_count == 0 || !add(tensor.file_offset, tensor.byte_count, end) ||
            end > file.file_size) {
            error = file.path + ": invalid or out-of-file tensor range " + tensor.tensor_name;
            return false;
        }
        if (tensor.expert_slab_bytes == 0) {
            atoms.push_back({tensor.tensor_name, tensor.file_offset, end, kNoExpertIndex});
            continue;
        }
        if (tensor.byte_count % tensor.expert_slab_bytes != 0) {
            error = file.path + ": expert tensor size is not divisible into slabs: " +
                    tensor.tensor_name;
            return false;
        }
        const std::uint64_t expert_count = tensor.byte_count / tensor.expert_slab_bytes;
        if (expert_count == 0 || expert_count > kMaximumExpertSlabs) {
            error = file.path + ": invalid expert slab count: " + tensor.tensor_name;
            return false;
        }
        for (std::uint64_t expert = 0; expert < expert_count; ++expert) {
            const std::uint64_t offset = tensor.file_offset + expert * tensor.expert_slab_bytes;
            atoms.push_back({tensor.tensor_name, offset, offset + tensor.expert_slab_bytes, expert});
        }
    }

    std::sort(atoms.begin(), atoms.end(), [](const Atom& left, const Atom& right) {
        if (left.start != right.start) return left.start < right.start;
        if (left.end != right.end) return left.end < right.end;
        if (left.tensor_name != right.tensor_name) return left.tensor_name < right.tensor_name;
        return left.expert_index < right.expert_index;
    });
    for (std::size_t i = 1; i < atoms.size(); ++i) {
        if (atoms[i].start < atoms[i - 1].end) {
            error = file.path + ": tensor payload ranges overlap";
            return false;
        }
    }

    std::vector<Atom> current;
    for (const Atom& atom : atoms) {
        if (!current.empty() && atom.start - current.back().end >= kWindowMergeGapBytes) {
            Group group;
            if (!append_group(current, group, unread, file.path, error)) return false;
            groups.push_back(std::move(group));
        }
        current.push_back(atom);
    }
    if (!current.empty()) {
        Group group;
        if (!append_group(current, group, unread, file.path, error)) return false;
        groups.push_back(std::move(group));
    }
    return true;
}

bool add_file_plan(const WindowFileSource& source, const std::vector<Group>& groups,
                  std::uint64_t max_buffer_length, FileWindowPlan& result,
                  std::string& error) {
    result = {};
    result.path = source.path;
    for (const Group& group : groups) {
        std::uint64_t active_window = kNoExpertIndex;
        for (const Atom& atom : group.atoms) {
            const std::uint64_t aligned_start = align_down(atom.start, kWindowPageBytes);
            std::uint64_t aligned_end = 0;
            if (!align_up(atom.end, kWindowPageBytes, aligned_end) ||
                aligned_end < aligned_start || aligned_end - aligned_start > max_buffer_length) {
                error = source.path + ": tensor or expert slab exceeds maxBufferLength: " +
                        atom.tensor_name;
                return false;
            }

            std::uint64_t window_index = active_window;
            if (active_window == kNoExpertIndex) {
                window_index = result.windows.size();
                result.windows.push_back({aligned_start, aligned_end});
                active_window = window_index;
            } else {
                FileWindow& current = result.windows[static_cast<std::size_t>(active_window)];
                const std::uint64_t combined_start = std::min(current.start, aligned_start);
                const std::uint64_t combined_end = std::max(current.end, aligned_end);
                if (combined_end - combined_start <= max_buffer_length) {
                    current.start = combined_start;
                    current.end = combined_end;
                    window_index = active_window;
                } else {
                    std::uint64_t gap_cursor = current.end;
                    while (gap_cursor < aligned_start) {
                        const std::uint64_t gap_end =
                            gap_cursor + std::min(max_buffer_length, aligned_start - gap_cursor);
                        result.windows.push_back({gap_cursor, gap_end});
                        gap_cursor = gap_end;
                    }
                    window_index = result.windows.size();
                    result.windows.push_back({aligned_start, aligned_end});
                    active_window = window_index;
                }
            }

            const FileWindow& window = result.windows[static_cast<std::size_t>(window_index)];
            if (atom.start < window.start || atom.end > window.end) {
                error = source.path + ": internal window mapping exceeds its window";
                return false;
            }
            result.slices.push_back({atom.tensor_name, window_index, atom.start,
                                     atom.start - window.start, atom.end - atom.start,
                                     atom.expert_index});
        }
    }
    return true;
}

bool prepare_sources(const WindowSources& sources, std::vector<std::vector<Group>>& groups_by_file,
                     std::uint64_t& unread, std::string& error) {
    groups_by_file.clear();
    unread = 0;
    std::map<std::string, bool> paths;
    for (const WindowFileSource& file : sources.files) {
        if (!paths.emplace(file.path, true).second) {
            error = "window source paths must be unique";
            return false;
        }
        std::vector<Group> groups;
        std::uint64_t file_unread = 0;
        if (!make_groups(file, groups, file_unread, error) ||
            !add(unread, file_unread, unread)) {
            if (error.empty()) error = file.path + ": wrapped-unread byte count overflows";
            return false;
        }
        groups_by_file.push_back(std::move(groups));
    }
    return true;
}

}  // namespace

bool plan_windows(const WindowSources& sources, std::uint64_t max_buffer_length,
                  WindowPlan& plan, std::string& error) {
    plan = {};
    error.clear();
    if (max_buffer_length < kWindowPageBytes) {
        error = "maxBufferLength is smaller than one 16 KiB window page";
        return false;
    }
    const std::uint64_t effective_max =
        max_buffer_length - max_buffer_length % kWindowPageBytes;
    std::vector<std::vector<Group>> groups_by_file;
    std::uint64_t unread = 0;
    if (!prepare_sources(sources, groups_by_file, unread, error)) return false;

    WindowPlan candidate;
    candidate.wrapped_unread_bytes = unread;
    candidate.files.reserve(sources.files.size());
    for (std::size_t i = 0; i < sources.files.size(); ++i) {
        FileWindowPlan file_plan;
        if (!add_file_plan(sources.files[i], groups_by_file[i], effective_max, file_plan, error)) {
            return false;
        }
        candidate.files.push_back(std::move(file_plan));
    }
    plan = std::move(candidate);
    return true;
}

bool compute_wrapped_unread(const WindowSources& sources, std::uint64_t& unread_bytes,
                            std::string& error) {
    unread_bytes = 0;
    error.clear();
    std::vector<std::vector<Group>> groups_by_file;
    std::uint64_t candidate = 0;
    if (!prepare_sources(sources, groups_by_file, candidate, error)) return false;
    unread_bytes = candidate;
    return true;
}

}  // namespace strata::metal
