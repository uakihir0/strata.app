// Shared tensor and parameter layouts. Keep this header valid in both MSL and C++.
#pragma once

#ifdef __METAL_VERSION__
#include <metal_stdlib>
using namespace metal;
using strata_half = half;
using strata_half2 = half2;
using strata_u8 = uchar;
using strata_i8 = char;
using strata_u16 = ushort;
using strata_i16 = short;
using strata_u32 = uint;
using strata_i32 = int;
#else
#include <cstddef>
#include <cstdint>
using strata_half = std::uint16_t;
using strata_half2 = std::uint32_t;
using strata_u8 = std::uint8_t;
using strata_i8 = std::int8_t;
using strata_u16 = std::uint16_t;
using strata_i16 = std::int16_t;
using strata_u32 = std::uint32_t;
using strata_i32 = std::int32_t;
#endif

// Quantized GGUF blocks used by the Phase-1 model catalogue. All payload
// members have their natural 1- or 2-byte alignment; tensor rows are packed.
struct block_q2_0 {
    strata_half d;
    strata_u8 qs[16];
};

struct block_q3_K {
    strata_u8 hmask[32];
    strata_u8 qs[64];
    strata_u8 scales[12];
    strata_half d;
};

struct block_q4_K {
    strata_half2 dm;
    strata_u8 scales[12];
    strata_u8 qs[128];
};

struct block_q5_K {
    strata_half2 dm;
    strata_u8 scales[12];
    strata_u8 qh[32];
    strata_u8 qs[128];
};

struct block_q6_K {
    strata_u8 ql[128];
    strata_u8 qh[64];
    strata_i8 scales[16];
    strata_half d;
};

struct block_iq4_xs {
    strata_half d;
    strata_u16 scales_h;
    strata_u8 scales_l[4];
    strata_u8 qs[128];
};

struct block_q4_0 {
    strata_half d;
    strata_u8 qs[16];
};

struct block_q5_0 {
    strata_half d;
    strata_u8 qh[4];
    strata_u8 qs[16];
};

struct block_q8_0 {
    strata_half d;
    strata_i8 qs[32];
};

struct block_iq3_s {
    strata_half d;
    strata_u8 qs[64];
    strata_u8 qh[8];
    strata_u8 signs[32];
    strata_u8 scales[4];
};

struct block_iq2_xxs {
    strata_half d;
    strata_u16 qs[32];
};

struct block_iq2_xs {
    strata_half d;
    strata_u16 qs[32];
    strata_u8 scales[8];
};

struct block_iq2_s {
    strata_half d;
    strata_u8 qs[64];
    strata_u8 qh[8];
    strata_u8 scales[8];
};

struct block_iq3_xxs {
    strata_half d;
    strata_u8 qs[96];
};

struct block_iq1_m {
    strata_u8 qs[32];
    strata_u8 qh[16];
    strata_u8 scales[8];
};

struct block_iq4_nl {
    strata_half d;
    strata_u8 qs[16];
};

// Activations used by the Q8_1 mat-vec path.
struct Q81 {
    strata_half2 ds;
    strata_i8 qs[32];
};

// Logical S2 activation block. The MTP command path stores its code and scale
// planes separately; this value layout is used only while decoding a block.
struct Q80S {
    float d;
    strata_i8 qs[32];
};

struct StepRow {
    strata_i32 pos;
    strata_i32 n_kv;
    strata_i32 n_bid;
    strata_i32 width;
};

struct SampRow {
    float temperature;
    float top_p;
    float min_p;
    float penalty_repeat;
    float penalty_freq;
    float penalty_present;
    strata_i32 top_k;
    strata_i32 min_keep;
    strata_i32 greedy;
    strata_u32 counter;
    strata_u32 hist_off;
    strata_u32 hist_len;
};

struct SampArgs {
    strata_u32 n_vocab;
    strata_u32 n_blocks;
    strata_u32 kpart;
    strata_u32 penalize;
};

struct SampPenalizeArgs {
    strata_u32 nv;
    strata_u32 n_vocab;
    strata_u32 has_map;
};

struct WindowParams {
    strata_u32 T;
    strata_u32 nc;
    strata_u32 pos0;
    strata_u32 n_keep;
    strata_u32 seed_lo;
    strata_u32 seed_hi;
    strata_u32 tokens[8];
    StepRow step[8];
    SampRow samp[8];
};

struct PrefillParams {
    strata_u32 T;
    strata_u32 p0;
    strata_u32 idx_f;
    strata_u32 idx_e;
};

struct MtpParams {
    strata_u32 T;
    strata_u32 hid[4];
    strata_u32 a;
    strata_u32 j;
    strata_u32 tokens[8];
    StepRow step[8];
    SampRow samp;
};

static_assert(sizeof(block_q2_0) == 18, "block_q2_0 layout");
static_assert(sizeof(block_q3_K) == 110, "block_q3_K layout");
static_assert(sizeof(block_q4_K) == 144, "block_q4_K layout");
static_assert(sizeof(block_q5_K) == 176, "block_q5_K layout");
static_assert(sizeof(block_q6_K) == 210, "block_q6_K layout");
static_assert(sizeof(block_iq4_xs) == 136, "block_iq4_xs layout");
static_assert(sizeof(block_q4_0) == 18, "block_q4_0 layout");
static_assert(sizeof(block_q5_0) == 22, "block_q5_0 layout");
static_assert(sizeof(block_q8_0) == 34, "block_q8_0 layout");
static_assert(sizeof(block_iq3_s) == 110, "block_iq3_s layout");
static_assert(sizeof(block_iq2_xxs) == 66, "block_iq2_xxs layout");
static_assert(sizeof(block_iq2_xs) == 74, "block_iq2_xs layout");
static_assert(sizeof(block_iq2_s) == 82, "block_iq2_s layout");
static_assert(sizeof(block_iq3_xxs) == 98, "block_iq3_xxs layout");
static_assert(sizeof(block_iq1_m) == 56, "block_iq1_m layout");
static_assert(sizeof(block_iq4_nl) == 18, "block_iq4_nl layout");
static_assert(sizeof(Q81) == 36, "Q81 layout");
static_assert(sizeof(Q80S) == 36, "Q80S layout");

static_assert(sizeof(StepRow) == 16, "StepRow layout");
static_assert(sizeof(SampRow) == 48, "SampRow layout");
static_assert(sizeof(SampArgs) == 16, "SampArgs layout");
static_assert(sizeof(SampPenalizeArgs) == 12, "SampPenalizeArgs layout");
static_assert(sizeof(WindowParams) == 568, "WindowParams layout");
static_assert(sizeof(PrefillParams) == 16, "PrefillParams layout");
static_assert(sizeof(MtpParams) == 236, "MtpParams layout");

#ifndef __METAL_VERSION__
static_assert(offsetof(WindowParams, tokens) == 24, "WindowParams::tokens offset");
static_assert(offsetof(WindowParams, step) == 56, "WindowParams::step offset");
static_assert(offsetof(WindowParams, samp) == 184, "WindowParams::samp offset");
static_assert(offsetof(MtpParams, hid) == 4, "MtpParams::hid offset");
static_assert(offsetof(MtpParams, a) == 20, "MtpParams::a offset");
static_assert(offsetof(MtpParams, j) == 24, "MtpParams::j offset");
static_assert(offsetof(MtpParams, tokens) == 28, "MtpParams::tokens offset");
static_assert(offsetof(MtpParams, step) == 60, "MtpParams::step offset");
static_assert(offsetof(MtpParams, samp) == 188, "MtpParams::samp offset");
static_assert(offsetof(SampArgs, penalize) == 12, "SampArgs::penalize offset");
static_assert(offsetof(SampPenalizeArgs, has_map) == 8, "SampPenalizeArgs::has_map offset");
#endif
