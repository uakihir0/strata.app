// Host-side PLE row gathering into the named Window scratch region.
#include "strata/metal/window_ple_stage.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace strata::metal {

bool stage_window_ple_input(kernels::PleTableReader& reader,
                            const WindowPleRows& rows,
                            const WindowScratchStorage& scratch,
                            BufferRef& staged_input, std::string& error) {
    error.clear();
    staged_input = {};

    const std::uint32_t token_count = rows.token_count();
    if (token_count == 0u || token_count > kWindowPleMaxTokens) {
        error = "Window PLE staging requires 1..8 prepared token rows";
        return false;
    }
    const std::span<const std::uint32_t> row_ids = rows.all_rows();
    const std::size_t expected_row_ids =
        static_cast<std::size_t>(token_count) * kWindowPleRowsPerToken;
    if (row_ids.size() != expected_row_ids) {
        error = "Window PLE row index count does not match the prepared token count";
        return false;
    }
    if (!reader.is_open()) {
        error = "Window PLE staging needs an open table reader";
        return false;
    }
    const std::uint64_t table_rows = reader.rows();
    if (table_rows != kernels::PLE_TABLE_ROWS) {
        error = "Window PLE table row count does not match the artifact geometry";
        return false;
    }
    for (const std::uint32_t row_id : row_ids) {
        if (row_id >= table_rows) {
            error = "Window PLE row index exceeds the opened table";
            return false;
        }
    }

    constexpr std::size_t kFloatsPerToken =
        static_cast<std::size_t>(kernels::NG_N_EMBD);
    constexpr std::size_t kFloatBytes = sizeof(float);
    const std::size_t output_floats =
        static_cast<std::size_t>(token_count) * kFloatsPerToken;
    if (output_floats >
        std::numeric_limits<std::size_t>::max() / kFloatBytes) {
        error = "Window PLE staging output size overflows";
        return false;
    }
    const std::size_t output_bytes = output_floats * kFloatBytes;

    BufferRef scratch_view;
    if (!scratch.view(scratch.layout().ple_input, scratch_view, error)) {
        return false;
    }
    if (scratch_view.contents == nullptr ||
        scratch_view.length < output_bytes ||
        reinterpret_cast<std::uintptr_t>(scratch_view.contents) %
                alignof(float) !=
            0u) {
        error = "Window PLE scratch view is too small or misaligned";
        return false;
    }
    const BufferRef active_view = scratch_view.slice(0u, output_bytes);
    if (!active_view) {
        error = "Window PLE active input range is not a live bounded view";
        return false;
    }

    auto* output = static_cast<float*>(scratch_view.contents);
    if (!reader.gather_batch(row_ids.data(), token_count, output, error)) {
        if (error.empty()) error = "Window PLE table row gathering failed";
        return false;
    }

    staged_input = active_view;
    error.clear();
    return true;
}

}  // namespace strata::metal
