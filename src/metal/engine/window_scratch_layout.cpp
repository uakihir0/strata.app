// Checked T=8 Metal Window scratch layout.
#include "strata/metal/window_scratch.hpp"

#include "strata/metal/tiered.hpp"
#include "strata/metal/window_params.hpp"
#include "strata/kernels/qsa.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

namespace strata::metal {
namespace {

constexpr std::size_t kRegionAlignment = 256u;
constexpr std::size_t kFloatBytes = sizeof(float);
constexpr std::size_t kInt32Bytes = sizeof(std::int32_t);
constexpr std::size_t kQ81BlockBytes = 36u;
constexpr std::size_t kQ81BlockValues = 32u;
constexpr std::size_t kQsaChunkCells = 64u;
constexpr std::uint32_t kQsaHeadCount = 24u;
constexpr std::uint32_t kQsaKvHeadCount = 2u;
constexpr std::uint32_t kQsaHeadWidth = 256u;
constexpr std::uint32_t kQsaIndexHeadCount = 4u;
constexpr std::uint32_t kQsaIndexWidth = 128u;
constexpr std::uint32_t kQsaIndexBlockCells = 4u;
constexpr std::uint32_t kQsaTailRows = kQsaIndexBlockCells - 1u;
constexpr std::uint32_t kGdnConvWidth = 10'240u;
constexpr std::uint32_t kGdnValueHeads = 48u;
constexpr std::uint32_t kGdnValueWidth = 6'144u;
constexpr std::uint32_t kPleWidth = 10'240u;
constexpr std::uint32_t kPleInputWidth = 2'560u;
constexpr std::uint32_t kPleGateCount = 4u;
constexpr std::uint32_t kVocabularySize = 248'320u;
constexpr std::uint32_t kQsaTopkCells = 32'768u;
constexpr std::uint32_t kQsaSelectionWidth = 2'051u;
constexpr std::uint32_t kSamplerHistoryLength = 4'096u;
constexpr std::uint32_t kSamplerBlocksPerRow = 64u;
constexpr std::uint32_t kSamplerEntriesPerBlock = 64u;

static_assert(sizeof(TieredMoePlan) == 1'620u,
              "Metal MoePlan host layout must match the shader ABI");
static_assert(sizeof(std::uint32_t) == kInt32Bytes);

bool multiply(std::size_t lhs, std::size_t rhs, std::size_t& result) {
    if (lhs != 0u && rhs > std::numeric_limits<std::size_t>::max() / lhs) {
        return false;
    }
    result = lhs * rhs;
    return true;
}

bool product_bytes(std::initializer_list<std::size_t> factors,
                   std::size_t element_bytes, std::size_t& result) {
    result = element_bytes;
    for (const std::size_t factor : factors) {
        if (!multiply(result, factor, result)) return false;
    }
    return true;
}

bool append_region(std::size_t bytes, std::size_t& cursor,
                   WindowScratchRegion& region) {
    if (bytes == 0u) return false;
    const std::size_t remainder = cursor % kRegionAlignment;
    if (remainder != 0u) {
        const std::size_t padding = kRegionAlignment - remainder;
        if (padding > std::numeric_limits<std::size_t>::max() - cursor) {
            return false;
        }
        cursor += padding;
    }
    if (bytes > std::numeric_limits<std::size_t>::max() - cursor) return false;
    region = {cursor, bytes};
    cursor += bytes;
    return true;
}

bool append_elements(std::size_t count, std::size_t element_bytes,
                     std::size_t& cursor, WindowScratchRegion& region) {
    std::size_t bytes = 0;
    return multiply(count, element_bytes, bytes) &&
           append_region(bytes, cursor, region);
}

bool append_product(std::initializer_list<std::size_t> factors,
                   std::size_t element_bytes, std::size_t& cursor,
                   WindowScratchRegion& region) {
    std::size_t bytes = 0;
    return product_bytes(factors, element_bytes, bytes) &&
           append_region(bytes, cursor, region);
}

bool append_q81(std::size_t row_count, std::size_t values_per_row,
                std::size_t& cursor, WindowScratchRegion& region) {
    if (values_per_row == 0u || values_per_row % kQ81BlockValues != 0u) {
        return false;
    }
    return append_product({row_count, values_per_row / kQ81BlockValues},
                          kQ81BlockBytes, cursor, region);
}

bool supported_geometry(const Geometry& geometry) {
    return is_qwen4exp_runtime_geometry(geometry) &&
           (geometry.n_expert == 256u || geometry.n_expert == 512u);
}

}  // namespace

bool make_window_scratch_layout(const Geometry& geometry,
                                std::uint32_t cell_capacity,
                                WindowScratchLayout& layout,
                                std::string& error) {
    layout = {};
    error.clear();
    if (!supported_geometry(geometry)) {
        error = "Window scratch requires qwen4exp geometry with 48 layers, 12 QSA layers, "
                "2 KV heads, 256-wide heads, 2560 embedding width, vocabulary 248320, "
                "and 256 or 512 experts";
        return false;
    }
    if (cell_capacity < kQsaIndexBlockCells ||
        cell_capacity > kWindowScratchMaxCellCapacity ||
        cell_capacity % kQsaIndexBlockCells != 0u) {
        error = "Window scratch cell capacity must be a multiple of four in [4, 262144]";
        return false;
    }

    const strata::kernels::QsaShapes qsa = strata::kernels::qsa_real_shapes();
    const std::size_t qsa_width = static_cast<std::size_t>(
        strata::kernels::qsa_selection_width(kQsaTopkCells, qsa));
    if (qsa_width != kQsaSelectionWidth || qsa.n_head != kQsaHeadCount ||
        qsa.n_head_kv != kQsaKvHeadCount || qsa.head_dim != kQsaHeadWidth ||
        qsa.idx_n_head != kQsaIndexHeadCount ||
        qsa.idx_dim != kQsaIndexWidth ||
        qsa.idx_block != kQsaIndexBlockCells) {
        error = "compiled QSA shapes do not match the Window scratch ABI";
        return false;
    }

    const std::size_t token_rows = kWindowScratchTokenRows;
    const std::size_t gdn_layers = kSessionGdnLayers;
    const std::size_t qsa_layers = kSessionQsaLayers;
    const std::size_t n_expert = geometry.n_expert;
    const std::size_t max_blocks =
        static_cast<std::size_t>(cell_capacity / kQsaIndexBlockCells) + 2u;
    const std::size_t attention_chunks =
        (qsa_width + kQsaChunkCells - 1u) / kQsaChunkCells;
    std::size_t attention_floats = 0;
    std::size_t attention_per_chunk = 0;
    if (!multiply(kQsaHeadCount,
                  static_cast<std::size_t>(kQsaHeadWidth) + 2u,
                  attention_per_chunk) ||
        !multiply(attention_chunks, attention_per_chunk, attention_floats) ||
        attention_floats >
            std::numeric_limits<std::size_t>::max() - 64u) {
        error = "Window attention scratch size overflows";
        return false;
    }
    attention_floats += 64u;

    WindowScratchLayout candidate;
    candidate.cell_capacity = cell_capacity;
    candidate.max_blocks = static_cast<std::uint32_t>(max_blocks);
    candidate.n_expert = geometry.n_expert;
    candidate.attention_scratch_floats_per_row = attention_floats;
    std::size_t cursor = 0;

    const bool ok =
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.ple_input) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.embedding) &&
        append_product({token_rows, kPleGateCount, kPleInputWidth}, kFloatBytes,
                       cursor, candidate.residual) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.mixed) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.block_output) &&
        append_product({token_rows, kPleGateCount}, kFloatBytes, cursor,
                       candidate.inject) &&
        append_product({token_rows, kPleGateCount}, kFloatBytes, cursor,
                       candidate.inject2) &&
        append_product({token_rows, kSessionHyperconnections * 80u}, kFloatBytes,
                       cursor, candidate.gr_lo) &&
        append_product({token_rows, kSessionHyperconnections}, kFloatBytes,
                       cursor, candidate.gr_rs) &&
        append_product({token_rows, kSessionHyperconnections,
                        kPleInputWidth}, kFloatBytes, cursor, candidate.gr_xn) &&
        append_q81(token_rows, kGdnValueWidth, cursor, candidate.xq) &&

        append_product({gdn_layers, token_rows, kGdnConvWidth}, kFloatBytes,
                       cursor, candidate.gdn_qkv_replay) &&
        append_product({gdn_layers, token_rows, kGdnConvWidth}, kFloatBytes,
                       cursor, candidate.gdn_h_replay) &&
        append_product({gdn_layers, token_rows, kGdnValueHeads}, kFloatBytes,
                       cursor, candidate.gdn_gate_replay) &&
        append_product({gdn_layers, token_rows, kGdnValueHeads}, kFloatBytes,
                       cursor, candidate.gdn_beta_replay) &&
        append_product({token_rows, kGdnValueWidth}, kFloatBytes, cursor,
                       candidate.gdn_z) &&
        append_product({token_rows, kGdnValueWidth}, kFloatBytes, cursor,
                       candidate.gdn_y) &&

        append_product({token_rows, kPleWidth}, kFloatBytes, cursor,
                       candidate.ple_key) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.ple_value) &&
        append_product({token_rows, kPleWidth}, kFloatBytes, cursor,
                       candidate.ple_query) &&
        append_product({token_rows, kPleGateCount}, kFloatBytes, cursor,
                       candidate.ple_gate) &&
        append_product({token_rows, kPleWidth}, kFloatBytes, cursor,
                       candidate.ple_gated) &&
        append_product({token_rows, kPleWidth}, kFloatBytes, cursor,
                       candidate.ple_norm) &&

        append_product({token_rows, kQsaHeadCount, 2u * kQsaHeadWidth},
                       kFloatBytes, cursor, candidate.qsa_q_full) &&
        append_product({token_rows, kQsaHeadCount, kQsaHeadWidth}, kFloatBytes,
                       cursor, candidate.qsa_q) &&
        append_product({token_rows, kQsaKvHeadCount, kQsaHeadWidth}, kFloatBytes,
                       cursor, candidate.qsa_k) &&
        append_product({token_rows, kQsaKvHeadCount, kQsaHeadWidth}, kFloatBytes,
                       cursor, candidate.qsa_v) &&
        append_product({qsa_layers, token_rows, kQsaIndexWidth}, kFloatBytes,
                       cursor, candidate.qsa_indexer_key_replay) &&
        append_product({token_rows, kQsaIndexHeadCount, kQsaIndexWidth},
                       kFloatBytes, cursor, candidate.qsa_indexer_query) &&
        append_product({token_rows, max_blocks}, kFloatBytes, cursor,
                       candidate.qsa_scores) &&
        append_product({token_rows, qsa_width}, kInt32Bytes, cursor,
                       candidate.qsa_ids) &&
        append_product({token_rows, kQsaHeadCount, kQsaHeadWidth}, kFloatBytes,
                       cursor, candidate.qsa_attention) &&
        append_product({token_rows, kQsaHeadCount, kQsaHeadWidth}, kFloatBytes,
                       cursor, candidate.qsa_attention_gated) &&
        append_product({token_rows, attention_floats}, kFloatBytes, cursor,
                       candidate.qsa_attention_scratch) &&
        append_product({qsa_layers, kQsaTailRows, kQsaIndexWidth}, kFloatBytes,
                       cursor, candidate.qsa_tail_snapshot) &&
        append_elements(qsa_layers, kInt32Bytes, cursor,
                        candidate.qsa_block_position_snapshot) &&

        append_product({token_rows, n_expert}, kFloatBytes, cursor,
                       candidate.router_logits) &&
        append_product({token_rows, kWindowScratchExpertsPerRow}, kInt32Bytes,
                       cursor, candidate.router_ids) &&
        append_product({token_rows, kWindowScratchExpertsPerRow}, kFloatBytes,
                       cursor, candidate.router_weights) &&
        append_elements(static_cast<std::size_t>(kSessionModelLayers) *
                            sizeof(TieredMoePlan),
                        1u, cursor, candidate.moe_plans) &&
        append_product({token_rows, kWindowScratchExpertsPerRow,
                        kWindowScratchFeedForwardWidth}, kFloatBytes, cursor,
                       candidate.moe_gate) &&
        append_product({token_rows, kWindowScratchExpertsPerRow,
                        kWindowScratchFeedForwardWidth}, kFloatBytes, cursor,
                       candidate.moe_up) &&
        append_product({token_rows, kWindowScratchExpertsPerRow,
                        kWindowScratchFeedForwardWidth}, kFloatBytes, cursor,
                       candidate.moe_hidden) &&
        append_q81(token_rows * kWindowScratchExpertsPerRow,
                   kWindowScratchFeedForwardWidth, cursor,
                   candidate.moe_hidden_q) &&
        append_product({token_rows, kWindowScratchExpertsPerRow,
                        kPleInputWidth}, kFloatBytes, cursor,
                       candidate.moe_parts) &&
        append_product({token_rows, kWindowScratchFeedForwardWidth},
                       kFloatBytes, cursor, candidate.shared_gate) &&
        append_product({token_rows, kWindowScratchFeedForwardWidth},
                       kFloatBytes, cursor, candidate.shared_up) &&
        append_product({token_rows, kWindowScratchFeedForwardWidth},
                       kFloatBytes, cursor, candidate.shared_hidden) &&
        append_q81(token_rows, kWindowScratchFeedForwardWidth, cursor,
                   candidate.shared_hidden_q) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.shared_output) &&
        append_elements(token_rows, kFloatBytes, cursor,
                        candidate.shared_gate_logit) &&

        append_product({token_rows, kSessionHyperconnections,
                        kSessionEmbeddingWidth},
                       kFloatBytes, cursor, candidate.head_normalized) &&
        append_product({token_rows, kSessionHyperconnections, 320u},
                       kFloatBytes, cursor, candidate.head_low_rank) &&
        append_product({token_rows, kSessionHyperconnections,
                        kSessionEmbeddingWidth},
                       kFloatBytes, cursor, candidate.head_gate) &&
        append_product({token_rows, kPleInputWidth}, kFloatBytes, cursor,
                       candidate.head_mixed) &&
        append_q81(token_rows, kPleInputWidth, cursor, candidate.head_q8_1) &&
        append_product({token_rows, kVocabularySize}, kFloatBytes, cursor,
                       candidate.head_logits) &&
        append_product({token_rows, kSamplerHistoryLength}, kInt32Bytes,
                       cursor, candidate.sampler_history) &&
        append_product({token_rows, kSamplerBlocksPerRow,
                        kSamplerEntriesPerBlock, 2u}, kInt32Bytes, cursor,
                       candidate.sampler_candidates) &&
        append_product({token_rows, 64u, 2u}, kInt32Bytes, cursor,
                       candidate.sampler_selection) &&
        append_elements(token_rows, kInt32Bytes, cursor,
                        candidate.sampler_output) &&
        append_product({3u, kVocabularySize}, kFloatBytes, cursor,
                       candidate.matvec_sink);
    if (!ok) {
        error = "Window scratch layout size overflows or contains an empty region";
        return false;
    }

    candidate.allocation_bytes = cursor;
    layout = candidate;
    error.clear();
    return true;
}

}  // namespace strata::metal
