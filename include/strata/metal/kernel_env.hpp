// include/strata/metal/kernel_env.hpp - readings of kernel-scope environment variables (docs/mac/04, section 9).
//
// The engine calls these with the value of getenv(); kernel_env_dispositions_test calls them with the values it sets,
// so the test runs the production reading, not a copy of it.
#pragma once

#include <cerrno>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>

namespace strata::metal {

enum class DispatchMode : std::uint8_t { serial, concurrent };

/// STRATA_METAL_DISPATCH (docs/mac/04, section 8.2; docs/mac/06, section 4.4). Unset, empty or "serial" selects serial
/// dispatch; "concurrent" selects concurrent dispatch. Any other value fails and selects nothing.
inline bool parse_dispatch_mode(const char* value, DispatchMode& mode, std::string& error) {
    if (value == nullptr || value[0] == '\0' || std::string_view(value) == "serial") {
        mode = DispatchMode::serial;
        return true;
    }
    if (std::string_view(value) == "concurrent") {
        mode = DispatchMode::concurrent;
        return true;
    }
    error = "STRATA_METAL_DISPATCH must be 'serial' or 'concurrent'";
    return false;
}

/// STRATA_PREFILL_BF16X2 as the engine reads it for the memory plan (src/metal/program/main.cpp). Unset, empty, or a
/// value that strtol does not read as one whole number, leaves the feature off. Any other whole number that is not
/// zero turns it on.
inline bool prefill_bf16x2_enabled(const char* value) {
    if (value == nullptr || *value == '\0') return false;
    char* end = nullptr;
    errno = 0;
    const long parsed = std::strtol(value, &end, 10);
    return errno == 0 && end != value && *end == '\0' && parsed != 0;
}

}  // namespace strata::metal
