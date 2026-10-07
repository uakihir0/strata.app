#pragma once

#include "../../../src/metal/kernels/common/types.h"
#include "oracles/common.hpp"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool iq1_m_matvec_reference(std::span<const block_iq1_m> weights,
                            std::span<const Q81> activations,
                            std::uint32_t n_in, std::uint32_t n_out,
                            std::uint32_t nc, std::uint32_t live_rows,
                            std::vector<float>& output, std::string& err);

bool iq1_m_matvec_tf_reference(std::span<const block_iq1_m> weights,
                               std::span<const Q81> activations,
                               std::uint32_t n_in, std::uint32_t n_out,
                               std::uint32_t nc, std::uint32_t live_rows,
                               std::vector<tf>& output, std::string& err);

}  // namespace twin
