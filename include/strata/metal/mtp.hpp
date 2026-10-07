// include/strata/metal/mtp.hpp - validated MTP runtime source inventory.
#pragma once

#include "strata/metal/window_plan.hpp"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::uint32_t kMtpExpertCount = 512;
inline constexpr std::uint64_t kMtpExpertBlobBytes = 1'382'400;
inline constexpr std::uint64_t kMtpExpertsFileBytes =
    static_cast<std::uint64_t>(kMtpExpertCount) * kMtpExpertBlobBytes;
inline constexpr std::uint64_t kMtpDenseFileBytes = 116'099'072;
/// Default-window geometry baseline for 106,299 draft logits, excluding the
/// device-side draft-vocabulary ID table. Other vocabulary sizes adjust `hl`.
inline constexpr std::uint64_t kMtpBindBaselineBytes = 15'874'804;

enum class MtpDenseWeightKind {
    q8_0,
    bf16,
    f32,
};

struct MtpDenseWeightSource {
    std::string tensor_name;
    MtpDenseWeightKind kind = MtpDenseWeightKind::bf16;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t byte_count = 0;
};

/// Exact file inventory and vocabulary validated by WeightStore.
struct MtpRuntimeCatalog {
    std::string directory;
    std::string experts_path;
    std::uint64_t experts_file_size = 0;
    std::uint64_t experts_file_device_id = 0;
    std::uint64_t experts_file_object_id = 0;
    std::string dense_path;
    std::uint64_t dense_file_size = 0;
    std::uint64_t dense_file_device_id = 0;
    std::uint64_t dense_file_object_id = 0;
    std::string draft_vocab_path;
    std::uint64_t draft_vocab_file_size = 0;
    std::vector<MtpDenseWeightSource> dense_tensors;
    // Preserve the file order; the runtime builds any sorted lookup separately.
    std::vector<std::uint32_t> draft_vocab_ids;
};

/// Shares an immutable catalog with its descriptor-backed generation lock.
struct MtpRuntimeSources {
    std::shared_ptr<const MtpRuntimeCatalog> catalog;
    WindowSourceFileLock generation_lock;
};

struct MtpExpertBinding {
    std::uint32_t expert_index = 0;
    std::uint64_t window_index = 0;
    std::uint64_t window_offset = 0;
    std::uint64_t byte_count = 0;
};

struct MtpDenseWeightBinding {
    std::string tensor_name;
    MtpDenseWeightKind kind = MtpDenseWeightKind::bf16;
    std::uint64_t rows = 0;
    std::uint64_t columns = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t window_index = 0;
    std::uint64_t window_offset = 0;
    std::uint64_t byte_count = 0;
};

struct MtpRuntimeLayout {
    std::vector<FileWindow> expert_windows;
    std::vector<MtpExpertBinding> experts;
    std::vector<FileWindow> dense_windows;
    std::vector<MtpDenseWeightBinding> dense_weights;
    std::uint64_t mapped_file_bytes = 0;
};

/// Plan page-aligned, maxBufferLength-bounded windows for the inspected MTP files.
/// Every expert blob and dense tensor remains wholly addressable through one buffer view.
bool build_mtp_runtime_layout(const MtpRuntimeSources& sources,
                              std::uint64_t max_buffer_length,
                              MtpRuntimeLayout& layout, std::string& error);

}  // namespace strata::metal
