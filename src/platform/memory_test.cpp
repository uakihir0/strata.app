// src/platform/memory_test.cpp - plan v0.3 P0.1: lock_resident on a 256 MiB region (CPU only, no GPU, no model).
#include "strata/platform/memory.hpp"
#include "strata/platform/memory_probe.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>



int main() {
    const std::uint64_t total = strata::platform::total_memory();
    const auto available = strata::platform::available_memory();
    const auto process = strata::platform::process_memory();
    const bool invalid_process_sample = !process ||
        process->phys_footprint == 0 || process->resident == 0;
    if (total == 0 || !available || *available > total || invalid_process_sample) {
        std::fprintf(stderr, "memory probe returned an invalid or incomplete sample\n");
        return 2;
    }
    std::printf("memory_probe: total=%llu available=%llu",
                static_cast<unsigned long long>(total),
                static_cast<unsigned long long>(*available));
    if (process) {
        std::printf(" footprint=%llu resident=%llu swap=%llu major_faults=%llu\n",
                    static_cast<unsigned long long>(process->phys_footprint),
                    static_cast<unsigned long long>(process->resident),
                    static_cast<unsigned long long>(process->swap_used),
                    static_cast<unsigned long long>(process->major_faults));
    } else {
        std::printf(" process_sample=unavailable\n");
    }

    const uint64_t bytes = 256ull << 20;
    void* p = std::malloc(bytes);
    if (p == nullptr) return 2;
    std::memset(p, 1, bytes);
    const strata::platform::LockResult r = strata::platform::lock_resident(p, bytes);
    std::printf("lock_resident: ok=%d locked=%llu MiB (%s)\n", (int) r.ok, (unsigned long long) (r.locked_bytes >> 20),
                r.note.c_str());
    int fail = !(r.ok && r.locked_bytes == bytes);
    strata::platform::unlock_resident(p, bytes);
    std::free(p);
    std::printf("platform_memory_test: %s\n", fail ? "FAILED" : "OK");
    return fail;
}
