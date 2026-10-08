#pragma once

#include "strata/metal/window_dispatch.hpp"

#include <cstdint>
#include <string>

namespace strata::metal::detail {

/// Testable portion of the T=1 consumer preflight that binds views to their
/// actual producer prefix. Full-mode table provenance is checked separately by
/// the public coordinator before this helper runs.
bool validate_window_t1_moe_consumer_handoffs(
    Context& context, std::uint32_t vocabulary_size, std::uint32_t experts,
    const WindowT1MoeLayer0FullBuffers& buffers, std::string& error);

}  // namespace strata::metal::detail
