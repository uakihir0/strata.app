// Scalar oracle for native IQ4_XS-weight/Q8_1-activation mat-vecs (docs/mac/04-metal-kernels.md §5.8).
#pragma once

#include "oracles/common.hpp"
#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool iq4_xs_matvec_reference(std::span<const block_iq4_xs> weights,
                             std::span<const Q81> activations,
                             std::uint32_t n_in, std::uint32_t n_out,
                             std::uint32_t nc, std::uint32_t live_rows,
                             std::vector<float>& output, std::string& err);

bool iq4_xs_matvec_tf_reference(
    std::span<const block_iq4_xs> weights,
    std::span<const Q81> activations, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    std::vector<tf>& output, std::string& err);

}  // namespace twin
