// include/strata/metal/mtp_bind_layout.hpp - device-free MTP bind-arena layout.
#pragma once

#include "strata/metal/mtp.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace strata::metal {

inline constexpr std::uint32_t kMtpDefaultBindCapacity = 32'768;
inline constexpr std::uint32_t kMtpDraftVocabularyBaseline = 106'299;
inline constexpr std::uint32_t kMtpFullVocabularySize = 248'320;
inline constexpr std::uint32_t kMtpBindRows = 8;
inline constexpr std::uint32_t kMtpBindAlignment = 256;

struct MtpBindRegion {
    std::uint64_t offset = 0;
    std::uint64_t bytes = 0;
};

enum class MtpBindRegionId : std::size_t {
    mtp_rin,
    mtp_tok,
    emb,
    en,
    e2,
    hn,
    h2,
    xq,
    residual,
    xn,
    lo,
    rs,
    inj,
    inj2,
    mixed,
    gate,
    kcur,
    vcur,
    q_full,
    qcur,
    attn,
    attn32,
    mtp_ids,
    draft_vocab_ids,
    att_scratch,
    bo,
    y,
    logits_r,
    ids,
    wts,
    mq,
    ms,
    eg,
    eu,
    eh,
    hq,
    hs,
    parts,
    sg,
    su,
    sh,
    shq,
    shx,
    gl,
    sample,
    hl,
    out_ids,
    probs,
    mtp_out,
    mtp_prob,
    count,
};

inline constexpr std::size_t kMtpBindRegionCount =
    static_cast<std::size_t>(MtpBindRegionId::count);

/// Byte-exact views into the normal-mode MTP arena. Coupled sampler scratch
/// has a separate conditional layout and is intentionally not represented here.
struct MtpBindLayout {
    std::uint32_t capacity = 0;
    std::uint32_t draft_vocab_count = 0;
    std::uint32_t output_vocab_count = 0;
    std::uint32_t chunk_count = 0;
    std::uint64_t used_bytes = 0;
    std::uint64_t reserved_bytes = 0;
    std::uint64_t allocation_bytes = 0;
    std::array<MtpBindRegion, kMtpBindRegionCount> regions{};

    MtpBindRegion region(MtpBindRegionId id) const noexcept;
};

/// Build the normal-mode MTP bind layout without a Metal device or allocation.
/// A zero draft vocabulary selects the full model vocabulary for `hl`.
/// Failure clears `layout`.
bool make_mtp_bind_layout(std::uint32_t capacity,
                          std::uint32_t draft_vocab_count,
                          MtpBindLayout& layout, std::string& error);

std::string_view mtp_bind_region_name(MtpBindRegionId id) noexcept;

}  // namespace strata::metal
