// Scalar oracle for native Q2_0-weight/Q8_1-activation mat-vecs.
#pragma once

#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool q2_0_matvec_reference(std::span<const block_q2_0> weights,
                           std::span<const Q81> activations,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::vector<float>& output, std::string& err);

}  // namespace twin
