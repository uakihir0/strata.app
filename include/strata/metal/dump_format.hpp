// include/strata/metal/dump_format.hpp - byte layouts of the --dump-* files and STRATA_DUMP_FIRST_LOGITS
// (docs/mac/03, sections 9.1 and 9.2). Every number is little-endian.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <span>
#include <string>
#include <vector>

namespace strata::metal {

inline constexpr std::int32_t kDumpEmbeddingWidth = 2560;
inline constexpr std::int32_t kDumpStreams = 4;
inline constexpr std::int32_t kDumpLayerCount = 48;
inline constexpr std::size_t kDumpLadderSlotFloats = static_cast<std::size_t>(kDumpStreams) * kDumpEmbeddingWidth;
inline constexpr std::size_t kDumpLadderPositionFloats =
    static_cast<std::size_t>(kDumpLayerCount + 1) * kDumpLadderSlotFloats;
inline constexpr std::size_t kDumpHalfLayerFloats = 13840;

/// The rows that --logits-stride N keeps from a `rows`-row logits dump: row 0, N, 2N, ... and the final row.
std::vector<std::int64_t> logits_stride_rows(std::int64_t rows, std::int64_t stride, std::string& error);

/// --dump-logits: int32 n_vocab, int32 rows, then rows x n_vocab F32. `logits` holds the rows back to back.
bool write_logits_dump(const std::string& path, std::int32_t n_vocab, std::span<const float> logits,
                       std::string& error);
/// STRATA_DUMP_FIRST_LOGITS: n_vocab F32 with no header.
bool write_first_logits_dump(const std::string& path, std::int32_t n_vocab, std::span<const float> logits,
                             std::string& error);
/// --dump-residual: int32 4, int32 2560, then 4 x 2560 F32.
bool write_residual_dump(const std::string& path, std::span<const float> residual, std::string& error);
/// --dump-layers: per position (48 + 1) x 4 x 2560 F32 with no header. Slot 0 is the embedding, slot l + 1 is R after
/// layer l.
bool write_layers_dump(const std::string& path, std::span<const float> ladder, std::string& error);

/// The slices of one layer of --dump-halves. An empty QSA span leaves its slots zero, as GDN layers do. The slots at
/// float offset 11784 are always zero.
struct HalfLayer {
    std::span<const float> attention;          // 2560 floats at 0
    std::span<const float> ffn;                // 2560 floats at 2560
    std::span<const float> attention_inject;   // 4 floats at 5120
    std::span<const float> ffn_inject;         // 4 floats at 5124
    std::span<const float> qsa_pre_gate;       // 6144 floats at 5128
    std::span<const float> qsa_vcur;           // 512 floats at 11272
    std::span<const float> qsa_gather_v;       // 256 floats at 12296
    std::span<const std::int32_t> qsa_ids;     // 4 int32 at 12552
    std::span<const std::int32_t> qsa_steps;   // 4 int32 at 12556
    std::span<const float> qsa_v_pool;         // 256 floats at 12560, FP16 KV only (may be empty)
    std::span<const float> qsa_vcur_appended;  // 512 floats at 12816
    std::span<const float> qsa_kcur;           // 512 floats at 13328
};

/// Lays one layer out as its 13,840 floats in `out`.
bool pack_half_layer(const HalfLayer& layer, std::span<float> out, std::string& error);
/// --dump-halves: per position 48 layers x 13,840 F32 with no header. `halves` holds the packed layers back to back.
bool write_halves_dump(const std::string& path, std::span<const float> halves, std::string& error);

/// One --dump-routing record: int32 layer, int32 k, k x int32 ids, k x F32 weights. Verify-window records carry zeros.
/// k must be at least 1, the ids distinct and non-negative, and the weights finite and non-negative.
bool append_routing_record(std::FILE* file, std::int32_t layer, std::span<const std::int32_t> ids,
                           std::span<const float> weights, std::string& error);

}  // namespace strata::metal
