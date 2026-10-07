// include/strata/platform/memory.hpp - macOS resident-memory helpers.
#pragma once

#include <cstdint>
#include <string>

namespace strata::platform {

struct LockResult {
    bool ok = false;
    uint64_t locked_bytes = 0;   ///< bytes successfully locked with mlock
    std::string note;            ///< what was done or why it failed
};

/// Lock [p, p + bytes) into physical memory. Failure is reported without changing process limits.
LockResult lock_resident(void* p, uint64_t bytes);

/// Undo lock_resident for the same region (best effort).
void unlock_resident(void* p, uint64_t bytes);

/// Physical RAM in bytes (0 when the macOS sysctl query fails).
uint64_t total_physical_memory();

}  // namespace strata::platform
