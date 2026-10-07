// src/platform/memory_probe.cpp - macOS memory readings used by the Metal plan and diagnostics.
#include "strata/platform/memory_probe.hpp"

#include <cstdint>
#include <limits>
#include <sys/resource.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/mach_host.h>
#include <mach/task_info.h>
#include <mach/vm_statistics.h>

namespace strata::platform {
namespace {

template <typename T>
bool sysctl_value(const char* name, T& value) {
    std::size_t size = sizeof(value);
    return sysctlbyname(name, &value, &size, nullptr, 0) == 0 && size == sizeof(value);
}

bool add_pages(std::uint64_t& sum, std::uint64_t pages) {
    if (pages > std::numeric_limits<std::uint64_t>::max() - sum) return false;
    sum += pages;
    return true;
}

}  // namespace

std::optional<std::uint64_t> available_memory() {
    const host_t host = mach_host_self();
    if (host == MACH_PORT_NULL) return std::nullopt;
    vm_statistics64_data_t stats{};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    const kern_return_t status =
        host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count);
    mach_port_deallocate(mach_task_self(), host);
    if (status != KERN_SUCCESS || count < HOST_VM_INFO64_COUNT || vm_page_size == 0) {
        return std::nullopt;
    }
    std::uint64_t pages = 0;
    if (!add_pages(pages, stats.free_count) || !add_pages(pages, stats.inactive_count) ||
        !add_pages(pages, stats.purgeable_count) || !add_pages(pages, stats.speculative_count) ||
        pages > std::numeric_limits<std::uint64_t>::max() / vm_page_size) {
        return std::nullopt;
    }
    return pages * vm_page_size;
}

std::uint64_t total_memory() {
    std::uint64_t bytes = 0;
    return sysctl_value("hw.memsize", bytes) ? bytes : 0;
}

std::optional<ProcessMemory> process_memory() {
    task_vm_info_data_t task{};
    mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&task), &count) != KERN_SUCCESS ||
        count < TASK_VM_INFO_COUNT) {
        return std::nullopt;
    }
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_majflt < 0) return std::nullopt;
    xsw_usage swap{};
    std::size_t swap_size = sizeof(swap);
    if (sysctlbyname("vm.swapusage", &swap, &swap_size, nullptr, 0) != 0 || swap_size < sizeof(swap)) {
        return std::nullopt;
    }
    return ProcessMemory{static_cast<std::uint64_t>(task.phys_footprint),
                         static_cast<std::uint64_t>(task.resident_size),
                         static_cast<std::uint64_t>(swap.xsu_used),
                         static_cast<std::uint64_t>(usage.ru_majflt)};
}

}  // namespace strata::platform
