// src/metal/engine/mtp.cpp - bounded MTP runtime file-window planning.
#include "strata/metal/mtp.hpp"
#include "window_source_lock.hpp"

#include <algorithm>
#include <filesystem>
#include <limits>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

namespace strata::metal {
namespace {

bool add(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (right > std::numeric_limits<std::uint64_t>::max() - left) return false;
    result = left + right;
    return true;
}

bool multiply(std::uint64_t left, std::uint64_t right, std::uint64_t& result) {
    if (left != 0 && right > std::numeric_limits<std::uint64_t>::max() / left) return false;
    result = left * right;
    return true;
}

bool align_up(std::uint64_t value, std::uint64_t alignment, std::uint64_t& result) {
    if (alignment == 0) return false;
    const std::uint64_t remainder = value % alignment;
    return remainder == 0 ? (result = value, true)
                          : add(value, alignment - remainder, result);
}

std::uint64_t align_down(std::uint64_t value, std::uint64_t alignment) {
    return value - value % alignment;
}

bool mapped_extent(std::uint64_t file_size, std::uint64_t& extent) {
    return align_up(file_size, kWindowPageBytes, extent);
}

bool dense_weight_bytes(const MtpDenseWeightSource& tensor, std::uint64_t& bytes) {
    bytes = 0;
    if (tensor.rows == 0 || tensor.columns == 0) return false;
    std::uint64_t elements = 0;
    if (!multiply(tensor.rows, tensor.columns, elements)) return false;
    switch (tensor.kind) {
        case MtpDenseWeightKind::q8_0: {
            std::uint64_t row_bytes = 0;
            return tensor.columns % 32 == 0 &&
                   multiply(tensor.columns / 32, 34, row_bytes) &&
                   multiply(tensor.rows, row_bytes, bytes);
        }
        case MtpDenseWeightKind::bf16:
            return multiply(elements, 2, bytes);
        case MtpDenseWeightKind::f32:
            return multiply(elements, 4, bytes);
    }
    return false;
}

struct DenseRowRange {
    const MtpDenseWeightSource* source = nullptr;
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

}  // namespace

bool build_mtp_runtime_layout(const MtpRuntimeSources& sources,
                              std::uint64_t max_buffer_length,
                              MtpRuntimeLayout& layout, std::string& error) {
    error.clear();
    layout = {};
    if (sources.catalog == nullptr || sources.generation_lock.path.empty() ||
        sources.generation_lock.lock_lease() == nullptr) {
        error = "MTP layout requires a WeightStore-inspected catalog and generation lock";
        return false;
    }
    const MtpRuntimeCatalog& catalog = *sources.catalog;
    const WindowSourceFileLock& generation_lock = sources.generation_lock;
    const auto lease = generation_lock.lock_lease();
    if (!lease->matches(generation_lock.path, generation_lock.file_size,
                        generation_lock.file_device_id, generation_lock.file_object_id) ||
        !lease->matches_mtp_catalog(sources)) {
        error = "MTP layout catalog does not match its live generation-lock lease";
        return false;
    }
    if (catalog.directory.empty() ||
        catalog.experts_path != (std::filesystem::path(catalog.directory) / "experts.bin").string() ||
        catalog.dense_path != (std::filesystem::path(catalog.directory) / "dense.bin").string() ||
        catalog.experts_file_size != kMtpExpertsFileBytes ||
        catalog.dense_file_size != kMtpDenseFileBytes ||
        catalog.dense_tensors.size() != 29) {
        error = "MTP layout source inventory does not match the supported runtime";
        return false;
    }
    if (catalog.draft_vocab_file_size % sizeof(std::uint32_t) != 0 ||
        catalog.draft_vocab_ids.size() !=
            catalog.draft_vocab_file_size / sizeof(std::uint32_t)) {
        error = "MTP layout draft-vocabulary inventory has an invalid byte count";
        return false;
    }
    std::vector<std::uint32_t> sorted_vocab = catalog.draft_vocab_ids;
    std::sort(sorted_vocab.begin(), sorted_vocab.end());
    if (std::any_of(sorted_vocab.begin(), sorted_vocab.end(),
                    [](std::uint32_t id) { return id >= 248'320; }) ||
        std::adjacent_find(sorted_vocab.begin(), sorted_vocab.end()) != sorted_vocab.end()) {
        error = "MTP layout draft-vocabulary inventory has duplicate or out-of-range ids";
        return false;
    }
    if (max_buffer_length == 0) {
        error = "MTP layout requires a positive maxBufferLength";
        return false;
    }
    const std::uint64_t usable_window_bytes =
        max_buffer_length - max_buffer_length % kWindowPageBytes;
    if (usable_window_bytes == 0) {
        error = "MTP maxBufferLength is smaller than one no-copy page";
        return false;
    }

    MtpRuntimeLayout candidate;
    candidate.experts.reserve(kMtpExpertCount);
    candidate.dense_weights.reserve(catalog.dense_tensors.size());
    if (!add(catalog.experts_file_size, catalog.dense_file_size,
             candidate.mapped_file_bytes)) {
        error = "MTP mapped file byte count overflows";
        return false;
    }

    const std::uint64_t expert_boundary_bytes =
        (kMtpExpertBlobBytes /
         std::gcd(kMtpExpertBlobBytes, kWindowPageBytes)) * kWindowPageBytes;
    const std::uint64_t experts_per_boundary =
        expert_boundary_bytes / kMtpExpertBlobBytes;
    const std::uint64_t expert_groups_per_window =
        usable_window_bytes / expert_boundary_bytes;
    if (expert_groups_per_window == 0) {
        error = "MTP maxBufferLength cannot hold one page-aligned group of expert blobs";
        return false;
    }
    std::uint64_t experts_per_window = 0;
    if (!multiply(expert_groups_per_window, experts_per_boundary, experts_per_window)) {
        error = "MTP expert window capacity overflows";
        return false;
    }
    experts_per_window = std::min<std::uint64_t>(experts_per_window, kMtpExpertCount);
    if (experts_per_window == 0 || experts_per_window % experts_per_boundary != 0) {
        error = "MTP expert window cannot preserve page-aligned blob boundaries";
        return false;
    }
    std::uint64_t expert_file_extent = 0;
    if (!mapped_extent(catalog.experts_file_size, expert_file_extent)) {
        error = "MTP experts mapping extent overflows";
        return false;
    }
    for (std::uint64_t first_expert = 0; first_expert < kMtpExpertCount;
         first_expert += experts_per_window) {
        const std::uint64_t end_expert =
            std::min<std::uint64_t>(kMtpExpertCount, first_expert + experts_per_window);
        std::uint64_t start = 0;
        std::uint64_t end = 0;
        if (!multiply(first_expert, kMtpExpertBlobBytes, start) ||
            !multiply(end_expert, kMtpExpertBlobBytes, end) ||
            start % kWindowPageBytes != 0 || end % kWindowPageBytes != 0 ||
            end > expert_file_extent || end - start > usable_window_bytes) {
            error = "MTP expert window violates page, file or maxBufferLength bounds";
            return false;
        }
        const std::uint64_t window_index = candidate.expert_windows.size();
        candidate.expert_windows.push_back({start, end});
        for (std::uint64_t expert = first_expert; expert < end_expert; ++expert) {
            std::uint64_t window_offset = 0;
            if (!multiply(expert - first_expert, kMtpExpertBlobBytes, window_offset) ||
                window_offset > end - start ||
                kMtpExpertBlobBytes > end - start - window_offset) {
                error = "MTP expert binding falls outside its bounded window";
                return false;
            }
            candidate.experts.push_back(
                {static_cast<std::uint32_t>(expert), window_index,
                 window_offset, kMtpExpertBlobBytes});
        }
    }
    if (candidate.experts.size() != kMtpExpertCount) {
        error = "MTP expert windows do not cover all runtime experts";
        return false;
    }

    std::uint64_t dense_file_extent = 0;
    if (!mapped_extent(catalog.dense_file_size, dense_file_extent)) {
        error = "MTP dense mapping extent overflows";
        return false;
    }
    std::set<std::string> tensor_names;
    std::vector<DenseRowRange> rows;
    rows.reserve(catalog.dense_tensors.size());
    std::uint64_t expected_next = 0;
    for (const MtpDenseWeightSource& tensor : catalog.dense_tensors) {
        std::uint64_t bytes = 0;
        std::uint64_t end = 0;
        std::uint64_t next = 0;
        if (tensor.tensor_name.empty() ||
            !tensor_names.insert(tensor.tensor_name).second ||
            tensor.file_offset % 256 != 0 || tensor.file_offset < expected_next ||
            !dense_weight_bytes(tensor, bytes) || bytes != tensor.byte_count ||
            !add(tensor.file_offset, tensor.byte_count, end) ||
            end > catalog.dense_file_size || !align_up(end, 256, next)) {
            error = "MTP dense tensor inventory has an invalid or overlapping row";
            return false;
        }
        const std::uint64_t start_page = align_down(tensor.file_offset, kWindowPageBytes);
        std::uint64_t end_page = 0;
        if (!align_up(end, kWindowPageBytes, end_page) ||
            end_page > dense_file_extent || end_page - start_page > usable_window_bytes) {
            error = "MTP maxBufferLength cannot hold dense tensor " + tensor.tensor_name;
            return false;
        }
        rows.push_back({&tensor, start_page, end_page});
        expected_next = next;
    }
    if (expected_next != catalog.dense_file_size) {
        error = "MTP dense tensor rows do not cover the validated dense file extent";
        return false;
    }

    std::vector<std::size_t> current_rows;
    std::uint64_t current_start = 0;
    std::uint64_t current_end = 0;
    auto finish_dense_window = [&]() -> bool {
        if (current_rows.empty()) return true;
        if (current_end <= current_start ||
            current_end - current_start > usable_window_bytes) {
            error = "MTP dense window exceeds maxBufferLength";
            return false;
        }
        const std::uint64_t window_index = candidate.dense_windows.size();
        candidate.dense_windows.push_back({current_start, current_end});
        for (const std::size_t row_index : current_rows) {
            const MtpDenseWeightSource& tensor = *rows[row_index].source;
            if (tensor.file_offset < current_start ||
                tensor.file_offset > current_end ||
                tensor.byte_count > current_end - tensor.file_offset) {
                error = "MTP dense tensor binding falls outside its bounded window";
                return false;
            }
            candidate.dense_weights.push_back(
                {tensor.tensor_name, tensor.kind, tensor.rows, tensor.columns,
                 tensor.file_offset, window_index, tensor.file_offset - current_start,
                 tensor.byte_count});
        }
        current_rows.clear();
        return true;
    };

    for (std::size_t row_index = 0; row_index < rows.size(); ++row_index) {
        const DenseRowRange& row = rows[row_index];
        if (current_rows.empty()) {
            current_start = row.start;
            current_end = row.end;
            current_rows.push_back(row_index);
            continue;
        }
        const std::uint64_t proposed_end = std::max(current_end, row.end);
        if (proposed_end - current_start <= usable_window_bytes) {
            current_end = proposed_end;
            current_rows.push_back(row_index);
            continue;
        }
        if (!finish_dense_window()) return false;
        current_start = row.start;
        current_end = row.end;
        current_rows.push_back(row_index);
    }
    if (!finish_dense_window()) return false;
    if (candidate.dense_weights.size() != catalog.dense_tensors.size()) {
        error = "MTP dense windows do not cover every indexed tensor";
        return false;
    }

    layout = std::move(candidate);
    error.clear();
    return true;
}

}  // namespace strata::metal
