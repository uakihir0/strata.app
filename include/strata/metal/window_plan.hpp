// include/strata/metal/window_plan.hpp - device-free planning for no-copy GGUF windows.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::uint64_t kWindowPageBytes = 16 * 1024;
inline constexpr std::uint64_t kWindowMergeGapBytes = 256ull * 1024 * 1024;
inline constexpr std::uint64_t kNoExpertIndex = UINT64_MAX;

struct WindowTensorSource {
    std::string tensor_name;
    std::uint64_t file_offset = 0;
    std::uint64_t byte_count = 0;
    // Zero makes the tensor indivisible. Otherwise byte_count is split into equal expert slabs.
    std::uint64_t expert_slab_bytes = 0;
};

enum class WindowTensorExclusionReason {
    ple,
    converted,
};

struct WindowTensorExclusion {
    std::string tensor_name;
    std::uint64_t file_offset = 0;
    std::uint64_t byte_count = 0;
    WindowTensorExclusionReason reason = WindowTensorExclusionReason::ple;
};

struct DensePackTensorSource {
    std::string tensor_name;
    std::uint64_t file_offset = 0;
    std::uint64_t source_byte_count = 0;
    std::uint64_t byte_count = 0;
    std::uint32_t kind = 0;
};

struct WindowFileSource {
    std::string path;
    std::uint64_t file_size = 0;
    std::vector<WindowTensorSource> tensors;
    // Accounted GGUF payloads intentionally omitted from no-copy windows.
    std::vector<WindowTensorExclusion> excluded_tensors;
    // Captured by WeightStore so tiered reads can reject a same-size path replacement.
    std::uint64_t file_device_id = 0;
    std::uint64_t file_object_id = 0;
};

class WindowSourceLockLease;
class WeightStore;
#if defined(STRATA_METAL_TEST_LOCK_ACCESS)
class WindowSourceLockTestAccess;
#endif

struct WindowSourceFileLock {
    std::string path;
    std::uint64_t file_size = 0;
    std::uint64_t file_device_id = 0;
    std::uint64_t file_object_id = 0;
    // Read-only access to the opaque, descriptor-owning lease.
    const std::shared_ptr<const WindowSourceLockLease>& lock_lease() const noexcept {
        return lease_;
    }

private:
    std::shared_ptr<const WindowSourceLockLease> lease_;
    friend class WeightStore;
#if defined(STRATA_METAL_TEST_LOCK_ACCESS)
    friend class WindowSourceLockTestAccess;
#endif
};

struct WindowSources {
    std::vector<WindowFileSource> files;
    std::string dense_pack_path;
    std::uint64_t dense_pack_file_size = 0;
    std::vector<DensePackTensorSource> dense_pack_tensors;
    std::uint64_t dense_pack_device_id = 0;
    std::uint64_t dense_pack_object_id = 0;
    std::vector<WindowSourceFileLock> inspected_file_locks;
};

struct FileWindow {
    std::uint64_t start = 0;
    std::uint64_t end = 0;
};

struct WindowSlice {
    std::string tensor_name;
    std::uint64_t window_index = 0;
    std::uint64_t file_offset = 0;
    std::uint64_t window_offset = 0;
    std::uint64_t byte_count = 0;
    // kNoExpertIndex identifies a whole, indivisible tensor.
    std::uint64_t expert_index = kNoExpertIndex;
};

struct FileWindowPlan {
    std::string path;
    std::vector<FileWindow> windows;
    std::vector<WindowSlice> slices;
};

struct WindowPlan {
    std::vector<FileWindowPlan> files;
    std::uint64_t wrapped_unread_bytes = 0;
};

/// Plan page-aligned windows and tensor/expert-slab mappings, bounded by max_buffer_length.
bool plan_windows(const WindowSources& sources, std::uint64_t max_buffer_length,
                  WindowPlan& plan, std::string& error);

/// Count bytes covered by full-mode windows but not read, independent of max_buffer_length.
bool compute_wrapped_unread(const WindowSources& sources, std::uint64_t& unread_bytes,
                            std::string& error);

/// Check the never-wrap rules of docs/mac/05-memory-and-residency.md, section 6, against a plan
/// that plan_windows returned for these sources. A window must be page-aligned and lie within the
/// mapped file. An excluded PLE payload may be covered only as outward padding of at most one page,
/// beside a read tensor in the same page, and never from inside the payload. Every slice must tile
/// its included tensor, and the union of the windows minus the included bytes must equal
/// wrapped_unread_bytes. Read-only: the plan and sources are unchanged on success and on failure.
bool verify_window_never_wrap(const WindowSources& sources, const WindowPlan& plan,
                              std::string& error);

}  // namespace strata::metal
