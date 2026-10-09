#pragma once

#include "oracles/common.hpp"
#include "strata/metal/gdn_dispatch.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oracle {

void gdn_conv_l2(std::span<const float> history,
                 std::span<const float> qkv,
                 std::span<const float> conv_weights,
                 std::uint32_t rows, std::vector<twin::tf>& output);

void gdn_ab(std::span<const float> input,
            std::span<const std::uint16_t> alpha,
            std::span<const std::uint16_t> beta_weights,
            std::span<const float> dt_bias,
            std::span<const float> a_values,
            std::uint32_t rows,
            std::vector<twin::tf>& gate,
            std::vector<twin::tf>& beta);

void gdn_step_norm(std::vector<float>& state,
                   std::span<const float> h,
                   std::span<const float> gate,
                   std::span<const float> beta,
                   std::span<const float> z,
                   std::span<const float> gamma,
                   std::uint32_t rows, strata::metal::GDNStateMode mode,
                   std::vector<twin::tf>& output,
                   std::vector<twin::tf>& final_state);

void gdn_rec(std::vector<float>& state,
             std::span<const float> h,
             std::span<const float> gate,
             std::span<const float> beta,
             std::uint32_t rows,
             std::vector<twin::tf>& output,
             std::vector<twin::tf>& final_state);

void gdn_conv_commit(std::vector<float>& history,
                     std::span<const float> qkv,
                     std::uint32_t rows);

}  // namespace oracle
