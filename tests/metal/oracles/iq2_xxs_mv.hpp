// Scalar oracle for native IQ2_XXS-weight/Q8_1-activation row dots.
#pragma once

#include "oracles/common.hpp"
#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool iq2_xxs_matvec_reference(std::span<const block_iq2_xxs> weights,
                              std::span<const Q81> activations,
                              std::uint32_t n_in, std::uint32_t n_out,
                              std::uint32_t nc, std::uint32_t live_rows,
                              std::vector<float>& output, std::string& err);

bool iq2_xxs_matvec_tf_reference(std::span<const block_iq2_xxs> weights,
                                 std::span<const Q81> activations,
                                 std::uint32_t n_in, std::uint32_t n_out,
                                 std::uint32_t nc, std::uint32_t live_rows,
                                 std::vector<tf>& output, std::string& err);

/// Canonical IQ2_XXS block dequantization, following ggml's dq_iq2_xxs.
void iq2_xxs_dequantize_reference(
    const block_iq2_xxs& weight, std::span<float, 256u> output);

}  // namespace twin
