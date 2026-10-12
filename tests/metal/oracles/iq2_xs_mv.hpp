#pragma once

#include "oracles/common.hpp"
#include "../../../src/metal/kernels/common/types.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

/// Canonical IQ2_XS block dequantization following ggml's dq_iq2_xs contract.
void iq2_xs_dequantize_reference(
    const block_iq2_xs& weight, std::span<float, 256u> output);

bool iq2_xs_matvec_reference(std::span<const block_iq2_xs> weights,
                             std::span<const Q81> activations,
                             std::uint32_t n_in, std::uint32_t n_out,
                             std::uint32_t nc, std::uint32_t live_rows,
                             std::vector<float>& output, std::string& err);

bool iq2_xs_matvec_tf_reference(std::span<const block_iq2_xs> weights,
                                std::span<const Q81> activations,
                                std::uint32_t n_in, std::uint32_t n_out,
                                std::uint32_t nc, std::uint32_t live_rows,
                                std::vector<tf>& output, std::string& err);

}  // namespace twin
