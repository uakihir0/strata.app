// src/platform/memory.cpp - macOS resident-memory helpers.
#include "strata/platform/memory.hpp"

#include <sys/mman.h>
#include <sys/sysctl.h>

namespace strata::platform {

LockResult lock_resident(void* p, uint64_t bytes) {
    LockResult result;
    if (p == nullptr || bytes == 0) {
        result.note = "nothing to lock";
        return result;
    }
    if (mlock(p, bytes) != 0) {
        result.note = "mlock failed (check the process locked-memory limit)";
        return result;
    }
    result.ok = true;
    result.locked_bytes = bytes;
    result.note = "mlock";
    return result;
}

void unlock_resident(void* p, uint64_t bytes) {
    if (p != nullptr && bytes != 0) munlock(p, bytes);
}

uint64_t total_physical_memory() {
    uint64_t bytes = 0;
    std::size_t size = sizeof(bytes);
    return sysctlbyname("hw.memsize", &bytes, &size, nullptr, 0) == 0 && size == sizeof(bytes) ? bytes : 0;
}

}  // namespace strata::platform
