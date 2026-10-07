#pragma once

#include "common.hpp"
#include "strata/metal/qsa_dispatch.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace oracle {

void qsa_norm_rope(std::span<const float> input,
                   std::span<const float> gamma,
                   std::span<const std::int32_t> positions,
                   std::span<const float> rope,
                   std::uint32_t token_rows,
                   strata::metal::QsaNormRopeMode mode,
                   std::vector<twin::tf>& output);

void qsa_fwht256(std::span<const float> input, std::uint32_t rows,
                 std::vector<twin::tf>& output);

void qsa_kv_append_f16(std::span<const float> keys,
                       std::span<const float> values,
                       std::span<const std::int32_t> positions,
                       std::vector<std::uint16_t>& key_pool,
                       std::vector<std::uint16_t>& value_pool);

void qsa_kv_append_i8(std::span<const float> keys,
                      std::span<const float> values,
                      std::span<const std::int32_t> positions,
                      std::vector<std::int8_t>& key_codes,
                      std::vector<std::uint16_t>& key_scales,
                      std::vector<std::int8_t>& value_codes,
                      std::vector<std::uint16_t>& value_scales);

void qsa_kv_append_q4(std::span<const float> keys,
                      std::span<const float> values,
                      std::span<const std::int32_t> positions,
                      std::vector<std::uint8_t>& key_pool,
                      std::vector<std::uint8_t>& value_pool);

struct QsaIndexerState {
    std::vector<float> tail;
    std::vector<float> dead;
    std::vector<float> pooled;
    std::int32_t block_position = 0;
};

void qsa_indexer_append(std::span<const float> raw_keys,
                        std::span<const std::int32_t> positions,
                        std::span<const float> gamma,
                        std::span<const float> rope,
                        float rope_zero_cos, float rope_zero_sin,
                        QsaIndexerState& state);

void qsa_selection_scores(
    std::span<const float> pooled, std::span<const float> dead,
    std::span<const float> indexer_queries,
    std::span<const std::int32_t> n_kv,
    std::span<const std::int32_t> n_bid, std::uint32_t max_blocks,
    float untouched_score, std::vector<twin::tf>& output);

void qsa_selection_topk(
    std::span<const float> scores,
    std::span<const std::int32_t> n_kv,
    std::span<const std::int32_t> n_bid, std::uint32_t max_blocks,
    std::int32_t untouched_id, std::vector<std::int32_t>& output);

void qsa_gate(std::span<const float> attn,
              std::span<const float> q_full,
              std::uint32_t token_rows,
              std::vector<twin::tf>& output);

}  // namespace oracle
