// src/metal/engine/window_source_lock.hpp - private descriptor-owning source-lock lease.
#pragma once

#include "strata/metal/mtp.hpp"
#include "strata/metal/window_plan.hpp"

#include <memory>
#include <string>

namespace strata::metal {

/// Owns the descriptor and shared flock retained by WeightStore inspection.
class WindowSourceLockLease final {
public:
    ~WindowSourceLockLease();
    WindowSourceLockLease(const WindowSourceLockLease&) = delete;
    WindowSourceLockLease& operator=(const WindowSourceLockLease&) = delete;

    const std::string& path() const noexcept { return path_; }
    std::uint64_t file_size() const noexcept { return file_size_; }
    std::uint64_t file_device_id() const noexcept { return file_device_id_; }
    std::uint64_t file_object_id() const noexcept { return file_object_id_; }
    bool read_exact(std::uint64_t offset, void* destination, std::uint64_t bytes,
                    std::string& error) const;
    bool matches(const std::string& path, std::uint64_t file_size,
                 std::uint64_t file_device_id,
                 std::uint64_t file_object_id) const noexcept;
    bool matches_dense_pack_catalog(
        const std::vector<DensePackTensorSource>& rows) const noexcept;
    bool matches_mtp_catalog(const MtpRuntimeSources& sources) const noexcept;

private:
    static std::shared_ptr<const WindowSourceLockLease> acquire_shared(
        const std::string& path, bool create, std::string& error);
    WindowSourceLockLease(int descriptor, std::string path, std::uint64_t file_size,
                          std::uint64_t file_device_id, std::uint64_t file_object_id);
    void bind_dense_pack_catalog(const std::vector<DensePackTensorSource>& rows);
    void bind_mtp_catalog(std::shared_ptr<const MtpRuntimeCatalog> catalog);

    int descriptor_ = -1;
    std::string path_;
    std::uint64_t file_size_ = 0;
    std::uint64_t file_device_id_ = 0;
    std::uint64_t file_object_id_ = 0;
    std::vector<DensePackTensorSource> dense_pack_catalog_;
    bool dense_pack_catalog_bound_ = false;
    std::shared_ptr<const MtpRuntimeCatalog> mtp_catalog_;

    friend class WeightStore;
#if defined(STRATA_METAL_TEST_LOCK_ACCESS)
    friend class WindowSourceLockTestAccess;
#endif
};

}  // namespace strata::metal
