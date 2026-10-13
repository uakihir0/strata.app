// include/strata/metal/kernel_env.hpp - readings of kernel-scope environment variables (docs/mac/04, section 9).
//
// The engine calls these with the value of getenv(); kernel_env_dispositions_test calls them with the values it sets,
// so the test runs the production reading, not a copy of it.
#pragma once

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

/// STRATA_PREFILL_BF16X2 as the engine reads it for the memory plan (src/metal/program/main.cpp). The value is read
/// with atoi, as upstream reads it (docs/mac/06, section 4.2): a leading whole number counts, so "1x" is 1, and unset,
/// empty, 0 or a value with no leading number leaves the feature off. Any other non-zero number turns it on.
/// The code takes strtol's value and narrows it to int, which is what atoi returns on this platform, and it has no
/// undefined overflow.
inline bool prefill_bf16x2_enabled(const char* value) {
    if (value == nullptr) return false;
    return static_cast<int>(std::strtol(value, nullptr, 10)) != 0;
}

/// STRATA_METAL_ICB (docs/mac/04, section 8.3; docs/mac/06, section 4.4). Only "1" requests indirect command buffers,
/// which the Mac does not implement. The engine prints kMetalIcbNotice once at start and ignores the request.
inline constexpr const char* kMetalIcbNotice = "strata: STRATA_METAL_ICB is not implemented; encoding each window";

inline bool metal_icb_requested(const char* value) {
    return value != nullptr && std::string_view(value) == "1";
}

}  // namespace strata::metal
