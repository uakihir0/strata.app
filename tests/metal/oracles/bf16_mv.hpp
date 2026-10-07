// Scalar oracle for the F32-activation BF16 mat-vec (docs/mac/04-metal-kernels.md §5.9).
#pragma once

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace twin {

bool bf16_matvec_reference(std::span<const std::uint16_t> weights,
                           std::span<const float> inputs,
                           std::uint32_t n_in, std::uint32_t n_out,
                           std::uint32_t nc, std::uint32_t live_rows,
                           std::uint32_t block_size,
                           std::vector<float>& output, std::string& err);

}  // namespace twin
