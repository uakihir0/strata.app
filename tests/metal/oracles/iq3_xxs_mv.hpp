#pragma once

#include "oracles/common.hpp"
#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

/// Canonical IQ3_XXS block dequantization, decoded in packed byte order.
void iq3_xxs_dequantize_reference(const block_iq3_xxs& weight,
                                  std::span<float, 256> output);

bool iq3_xxs_matvec_reference(std::span<const block_iq3_xxs> weights,
                              std::span<const Q81> activations,
                              std::uint32_t n_in, std::uint32_t n_out,
                              std::uint32_t nc, std::uint32_t live_rows,
                              std::vector<float>& output, std::string& err);

bool iq3_xxs_matvec_tf_reference(
    std::span<const block_iq3_xxs> weights,
    std::span<const Q81> activations, std::uint32_t n_in,
    std::uint32_t n_out, std::uint32_t nc, std::uint32_t live_rows,
    std::vector<tf>& output, std::string& err);

}  // namespace twin
