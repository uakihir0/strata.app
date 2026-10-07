// include/strata/platform/memory_probe.hpp - physical and available memory probes.
#pragma once

#include <cstdint>
#include <optional>

namespace strata::platform {

struct ProcessMemory {
    std::uint64_t phys_footprint = 0;
    std::uint64_t resident = 0;
    std::uint64_t swap_used = 0;
    std::uint64_t major_faults = 0;
};

/// Bytes a new allocation can use without paging, or nullopt when the OS probe fails.
std::optional<std::uint64_t> available_memory();

/// Physical RAM in bytes, or zero when unknown.
std::uint64_t total_memory();

/// Current process memory sample, or nullopt when a complete sample is unavailable.
std::optional<ProcessMemory> process_memory();

}  // namespace strata::platform
