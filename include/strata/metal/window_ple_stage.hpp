// Host-side staging of Window PLE rows into the shared scratch allocation.
#pragma once

#include "strata/metal/window_ple_rows.hpp"
#include "strata/metal/window_scratch.hpp"

#include <string>

namespace strata::metal {

/// Gather the rows for a validated T=1..8 Window into its bounded Shared
/// `ple_input` view. On success, `staged_input` covers exactly T * 2560 floats.
/// This performs no GPU submission; the caller must ensure the scratch region
/// is not still in use by earlier GPU work.
bool stage_window_ple_input(kernels::PleTableReader& reader,
                            const WindowPleRows& rows,
                            const WindowScratchStorage& scratch,
                            BufferRef& staged_input, std::string& error);

}  // namespace strata::metal
