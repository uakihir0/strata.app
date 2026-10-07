#pragma once

#include "strata/metal/qsa_dispatch.hpp"
#include "twin.hpp"

#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <vector>

namespace oracle {

void qsa_attention(
    strata::metal::QsaAttentionMode mode,
    std::span<const float> query,
    std::span<const StepRow> steps,
    std::span<const std::int32_t> ids,
    std::span<const std::uint8_t> key_data,
    std::span<const std::uint16_t> key_scales,
    std::span<const std::uint8_t> value_data,
    std::span<const std::uint16_t> value_scales,
    std::uint32_t token_rows, std::uint32_t cell_capacity,
    std::uint32_t id_row_stride, std::vector<twin::tf>& output);

}  // namespace oracle
