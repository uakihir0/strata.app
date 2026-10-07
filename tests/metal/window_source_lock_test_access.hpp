// tests/metal/window_source_lock_test_access.hpp - fixture-only source-lock issuer.
#pragma once

#if !defined(STRATA_METAL_TEST_LOCK_ACCESS)
#error "source-lock fixture access is available only in the isolated storage-test build"
#endif

#include "../../src/metal/engine/window_source_lock.hpp"

namespace strata::metal {

/// The production engine never links or declares this fixture-only issuer.
class WindowSourceLockTestAccess final {
public:
    static WindowSourceFileLock acquire(const std::string& path, bool create,
                                        std::string& error) {
        WindowSourceFileLock result;
        auto lease = WindowSourceLockLease::acquire_shared(path, create, error);
        if (!lease) return result;
        result.path = lease->path();
        result.file_size = lease->file_size();
        result.file_device_id = lease->file_device_id();
        result.file_object_id = lease->file_object_id();
        result.lease_ = std::move(lease);
        return result;
    }

    static void clear(WindowSourceFileLock& lock) {
        lock.lease_.reset();
    }

    static bool bind_dense_pack_catalog(
        WindowSourceFileLock& lock,
        const std::vector<DensePackTensorSource>& rows) {
        if (!lock.lease_) return false;
        std::const_pointer_cast<WindowSourceLockLease>(lock.lease_)
            ->bind_dense_pack_catalog(rows);
        return true;
    }
};

}  // namespace strata::metal
