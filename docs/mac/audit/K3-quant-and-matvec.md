> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Part 03: Quantization formats, mat-vec, dequant and activation-quant kernels

Scope: every weight and activation format Strata reads, and every GPU and CPU kernel that multiplies,
dequantizes or quantizes them. Each item carries a note on how it maps to Apple Silicon (Metal, unified
memory, NEON).

Conventions:
- All source paths are relative to `/tmp/strata-upstream`. That clone is read-only.
- "ggml" means the vendored, pinned llama.cpp `3cf03257` (`third_party/ggml/`, `VERSION.txt`).
- "ggml-metal" means `ggml/src/ggml-metal/kernels/*.metal` at that same commit. Coverage was checked against the
  GitHub tree at `3cf03257`.
- H = n_embd = 2560 and FF = expert n_ff = 640. There are 512 routed experts per layer, top-10, 48 layers.

---

## 0. The essentials for a Metal port

1. **Two families of weight storage coexist.**
   - **Canonical / S-form packs** (`tools/strata_pack.py`) take the ISTA-DASLab GSQ-RCO Q2_0 GGUF and re-express
     it as planar codes, scales and offsets (S2/S4/S8, P16/P32).
   - **Native packs** (`tools/iq_pack.py`, `--native`) keep the raw GGUF blocks: IQ1_M/IQ2_*/IQ3_*, IQ4_*, Q2_0,
     K-quants, Q5_0/Q5_1/Q8_0. The engine uses llama.cpp-transcribed kernels on them.
   - The Mac port must support both, or pick one. Native is the future direction: every new model file is native.
2. **Activation contract is per weight TYPE, not per tensor role**:
   - **CPU (ggml vec_dot_type).** The legacy types (Q2_0, Q4_0, Q5_0, Q8_0, IQ4_NL) take Q8_0, and Q5_1 takes Q8_1.
     The K-quants, IQ4_XS and all IQ1/IQ2/IQ3 types take Q8_K.
   - **GPU canonical path.** S2 takes Q8_0 and the other S-forms take Q8_K (`src/core/layer.cpp:129-131`).
   - **GPU native path.** Everything takes ggml-cuda's Q8_1: fp16 d, fp16 sum of the float x, `roundf`.
   - Strata has several Q8_0 variants (section 2) and parity depends on choosing the right one.
3. **Bit-exactness is a design goal**, and several kernels reproduce CPU summation order on the GPU:
   - `__fmul_rn` and `__fmaf_rn` pin the rounding.
   - one quantizer uses double precision.
   - one kernel replays an AVX `hadd` order.
   - A Metal port must disable fast-math, avoid implicit FMA contraction, and emulate the one `double` division.
4. **ggml-metal at the pinned commit covers every storage format Strata uses for decode mat-vec**
   (`mul_mv`, `mul_mv_id`). That includes Q2_0, all of the IQ types and all of the K-quants. It also covers
   `mul_mm`/`mul_mm_id`, `get_rows` and `cpy`.
   - **Different numerics.** ggml-metal multiplies by float activations and does not quantize them to Q8. Its
     results will not bit-match Strata's CUDA or CPU output. They would be slightly more accurate, about 0.6-0.95%
     relative off Strata's quantized-activation numbers (section 2.3).
   - **Gap for verify windows.** `mul_mv_ext` (the small-batch r1=2..5 path) has no IQ1/IQ2/IQ3 kernels, which
     matters for multi-token verify windows on those types.
5. **No dp4a in Metal.** Strata's integer dot products need replacing:
   - Every S2 and native dot is int8 x int8 with exact integer sums of at most about 2^19 in magnitude.
   - These can be done exactly in fp32 FMA, since partial sums stay below 2^24, or with char4 to int4 multiply-adds.
   - The integer results, and so the final float expressions, can stay bit-identical to CUDA.
6. **Unified memory removes the whole PCIe tier.** That tier is the `pcie_mode` 0/1/2 fetch and mapped
   paths, the pinned registration and the VRAM expert cache. On a Mac every expert is effectively "resident"
   at `arena + layer_offset + id*blob`. The device-built grouping (`group_resident_kernel`, today used only for
   the MTP layer) becomes the general case.

---

## 1. Per-format reference

### 1.1 Master table

The "vec_dot_type" column is the ggml-cpu activation format. "Strata GPU act." is what the CUDA path feeds the
kernel. The ggml-metal columns are mv = `kernel_mul_mv_<t>_f32`, id = `kernel_mul_mv_id_<t>_f32`,
ext = `kernel_mul_mv_ext_<t>_f32_r1_{2..5}`, mm = `kernel_mul_mm(_id)_<t>_{f16,f32}`, rows = `kernel_get_rows_<t>`.

| Format | id | Block elems / bytes | bpw | vec_dot_type (CPU) | Strata GPU act. | Strata uses it for | Metal mv / id / ext / mm / rows |
|---|---|---|---|---|---|---|---|
| F32 | 0 | 1 / 4 | 32 | F32 | F32 | norms, ssm_a, ssm_conv1d, ssm_dt.bias | y / y / - / y / y |
| F16 | 1 | 1 / 2 | 16 | F16 | F32 | ple_conv1d; P16 form; prefill dequant target | y / y / y / y / y |
| Q4_0 | 2 | 32 / 18 | 4.5 | Q8_0 | Q8_1 (native), Q8_0 (S4) | some `ffn_down_shexp` | y / y / y / y / y |
| Q5_0 | 6 | 32 / 22 | 5.5 | Q8_0 | Q8_1 / Q8_0 | some `ffn_down_shexp`; Orca expert downs (CPU only); Orca PLE table | y / y / y / y / y |
| Q5_1 | 7 | 32 / 24 | 6.0 | Q8_1 | Q8_1 | UD-Q4_K_XL expert downs; Orca blk.0 `ffn_down_shexp` | y / y / y / y / y |
| Q8_0 | 8 | 32 / 34 | 8.5 | Q8_0 | Q8_1 / Q8_0 | some `ffn_down_shexp`; UD-Q4_K_XL dense, embd, head, a few expert downs; PLE key option; activation format | y / y / y / y / y |
| Q8_1 | 9 | 32 / 36 | 9.0 | - | - | activation only: GPU native path and Q5_1 on CPU | (none: Metal does not quantize src1) |
| Q2_K | 10 | 256 / 84 | 2.625 | Q8_K | - | **not used**; no Strata kernel | y / y / y / y / y |
| Q3_K | 11 | 256 / 110 | 3.4375 | Q8_K | Q8_1 / Q8_K (S4) | dense attn_qkv, attn_gate, attn_q (many layers) | y / y / y / y / y |
| Q4_K | 12 | 256 / 144 | 4.5 | Q8_K | Q8_1 / Q8_K (S4+off) | dense (some layers); UD-Q4_K_XL / Orca Q4_K_S expert gate/up | y / y / y / y / y |
| Q5_K | 13 | 256 / 176 | 5.5 | Q8_K | Q8_1 / Q8_K (S8+off) | dense (some); UD-Q4_K_XL layer-2 gate/up | y / y / y / y / y |
| Q6_K | 14 | 256 / 210 | 6.5625 | Q8_K | Q8_1 / Q8_K (S8) | dense (some; f32-scale S8) | y / y / y / y / y |
| Q8_K | 15 | 256 / 292 | 9.125 | - | Q8_K | activation only (S4/S8 dense GEMV; CPU IQ/K-quant dots) | (no Metal kernel: not a storage type) |
| IQ2_XXS | 16 | 256 / 66 | 2.0625 | Q8_K | Q8_1 | expert gate/up (some layers of the IQ files) | y / y / **-** / y / y |
| IQ2_XS | 17 | 256 / 74 | 2.3125 | Q8_K | Q8_1 | expert gate/up (IQ2_XS file) | y / y / **-** / y / y |
| IQ3_XXS | 18 | 256 / 98 | 3.0625 | Q8_K | Q8_1 | expert gate/up (IQ3_XXS file, Orca IQ3_XXS); PLE key option | y / y / **-** / y / y |
| IQ1_S | 19 | 256 / 50 | 1.5625 | Q8_K | - | **no Strata kernel** (GGUF readers know the size only) | y / y / **-** / y / y |
| IQ4_NL | 20 | 32 / 18 | 4.5 | Q8_0 | Q8_1 / Q8_0 (S4 cb) | expert downs (IQ files, Orca); `ffn_down_shexp`; **PLE table (28.8 GB, CPU-decoded)** | y / y / y / y / y |
| IQ3_S | 21 | 256 / 110 | 3.4375 | Q8_K | Q8_1 | expert gate/up (IQ3_S file, Coder) | y / y / **-** / y / y |
| IQ2_S | 22 | 256 / 82 | 2.5625 | Q8_K | Q8_1 | expert gate/up (Coder, IQ mixes) | y / y / **-** / y / y |
| IQ4_XS | 23 | 256 / 136 | 4.25 | Q8_K | Q8_1 / Q8_K (S4 cb) | dense attn_qkv, attn_gate, attn_q (some layers); Coder gate/up; PLE key option | y / y / y / y / y |
| I8 | 24 | 1 / 1 | 8 | - | - | carrier type for the FP8 E4M3 PLE table (`strata.ple.format="f8_e4m3"`) | n/a |
| IQ1_M | 29 | 256 / 56 | 1.75 | Q8_K | Q8_1 | GPU and CPU kernels exist; **no shipped expert rows** (Coder gate/up are IQ2_S/IQ3_XXS/IQ3_S) | y / y / **-** / y / y |
| BF16 | 30 | 1 / 2 | 16 | BF16 (CPU fp32 dot) | F32 or BF16 | routers, hc_* (GR), indexer q/k_proj, ssm_alpha/beta, ple_key/value, `--embd-gguf` | y / y / y / y / y (Metal needs bfloat support) |
| MXFP4 | 39 | 32 / 17 | 4.25 | Q8_0 | - | not used | y / y / y / y / y |
| NVFP4 | 40 | 64 / 36 | 4.5 | Q8_0 | - | not used (a fork's `nvfp4_convert.py` only) | (not checked) |
| Q2_0 | 42 | 64 / 18 | 2.25 | Q8_0 | Q8_1 (native), Q8_0 (S2) | **all routed experts of the Q2_0 file**, many expert downs of the IQ files, some dense tensors, PLE key | y / y / y / y / y |
| FP8 E4M3 "fn" | (I8) | 1 / 1 | 8 (+1 f32 per tensor) | - | - | PLE table option, CPU LUT decode (51.2 GB) | none in ggml |

Notes:
- **256-superblock formats cannot be used for any n_in=640 tensor.** FF = 640 = 2.5 x 256. So every expert `down`
  and every `ffn_down_shexp` is a 32- or 64-block format: Q2_0, IQ4_NL, Q4_0, Q5_0, Q5_1 or Q8_0. That is why Q8_K
  is "structurally impossible" for those tensors (`src/kernels/cuda/s_gemv.cu:73-80`), and why the down kernels
  have a separate, shorter type list (`STRATA_D_FMTS = 20 23 42 7 8` in `iq_kernels.cu`).
  - IQ4_XS (23) appears in that list but can only apply at a different n_ff. At FF=640, only 20, 42, 7 and 8 occur.
- `tools/pack_index.py:21-27` gives the scale widths across the 303 quantized dense tensors of the canonical
  pack (1,079 tensors in total):
  - 90 tensors carry fp16 scales: every Q2_0, Q4_0, Q5_0, Q8_0 and IQ4_NL.
  - 213 carry f32 scales: IQ4_XS, Q3_K, Q4_K, Q5_K and Q6_K.
  - 54 Q4_K/Q5_K tensors have an offset plane.
  - The engine widens fp16 scales to f32 at load.

### 1.2 Which tensors use which format

**Dense tensors in the ISTA-DASLab GSQ-RCO files.** These are the same across the Q2_0, IQ2_XS and IQ3_XXS
files: `iq_pack.py --base` hard-links the dense.bin because the float tensors are byte-identical. The per-layer
distribution, quoted from `src/core/layer.cpp:113-121`:

| Tensor | Types across layers |
|---|---|
| `attn_qkv` (36 GDN layers) | IQ4_XS x13, Q3_K x18, Q2_0 x1, Q4_K x4 |
| `attn_gate` (36 GDN layers) | Q3_K x25, IQ4_XS x4, Q4_K x3, Q2_0 x4 |
| `attn_q` (12 QSA layers) | Q3_K x6, IQ4_XS x4, Q2_0 x2 |
| `ffn_down_shexp` (n_in 640) | IQ4_NL / Q4_0 / Q5_0 / Q8_0 (legacy, Q8_0 activation) |
| `ssm_out`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate_shexp`, `ffn_up_shexp` | mixed K-quant / IQ4_XS / Q2_0 (served by native_mmvq in native packs) |
| `blk.1.ple_key` | Q2_0 / Q8_0 (NATIVE_PLE_KEY); the native kernel also accepts IQ3_XXS and IQ4_XS |
| `ple_value` (layer 1) | BF16 [n_embd, n_embd] |
| routers `ffn_gate_inp`, `ffn_gate_inp_shexp` | BF16 (2.6 MB per layer) |
| `hc_attn_{down,up,inject}`, `hc_ffn_{down,up,inject}`, `output_hc_{down,up}` | BF16 |
| `indexer.{k,q}_proj` | BF16 [n_embd, 128] / [n_embd, 512] |
| `ssm_alpha`, `ssm_beta` | BF16 [n_embd, 48] |
| `ple_conv1d` | F16 |
| norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | F32 |
| `token_embd` | native type, via `iq_embed_rows` (GPU, may be mapped host memory); BF16 with `--embd-gguf` |
| `output.weight` (head) | native type, via `native_mmvq` (`src/core/native_head.cpp`) |
| `per_layer_token_embd.weight` (PLE table, [160, 320,001,536]) | IQ4_NL (90 B/row, 28.8 GB), never packed; CPU decode (`src/kernels/ngram.cpp:99-165`) |

These forms come from the iq_pack.py FORM table and the native_dense.cpp eligibility list.

**Routed experts by model file:**

| Model file | gate/up | down | Notes |
|---|---|---|---|
| ISTA Q2_0 (37.6 GB) | Q2_0 | Q2_0 | Canonical packs repack to the 1,382,400 B S2 blob. A native pack keeps GGUF Q2_0 rows. 34.0 GB of experts, 0.66 GB read per token. |
| ISTA IQ2_XS (39.2 GB) | IQ2_XS (plus other IQ types per layer) | Q2_0 / IQ4_NL | per layer, `native_experts.txt` |
| ISTA IQ3_XXS (47.0 GB) | IQ3_XXS mix | Q2_0 / IQ4_NL | |
| ISTA IQ3_S (54.8 GB) | IQ3_S mix | Q2_0 / IQ4_NL | |
| Coder "IQ1_M" (256 experts, 29.6 GB) | IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS | IQ4_NL / Q2_0 | no IQ1_M expert rows despite the name |
| Swift 1.5 | as ISTA (same three sizes) | | layers 13-47 in shard 2 |
| Orca IQ3_XXS | IQ3_XXS | IQ4_NL (Q5_0 in some) | GPU expert path has no Q5_0 down |
| Orca Q4_K_S | Q4_K | Q5_0 / Q5_1 | Q5_0 PLE table (Mmap reader only); blk.0 `ffn_down_shexp` Q5_1 |
| Unsloth UD-Q4_K_XL (71.7 GiB experts) | Q4_K (Q5_K in layer 2) | Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) | dense/embd/head/PLE key Q8_0, read as Q8_0 or as BF16 via `--compat-bf16`; PLE table IQ4_NL |

**PLE table formats**, from `include/strata/kernels/ngram.hpp:44-104`:

| Format | Row bytes | Reader |
|---|---|---|
| IQ4_NL | 90 (5 blocks of 18) | Direct or Mmap |
| Q5_0 | 110 | Mmap only (#296) |
| FP8 E4M3 | 160 + per-tensor f32 `strata.ple.scale` | |

### 1.3 Format details

All decode formulas are ggml's (`ggml-quants.c` `dequantize_row_*`; `ggml-cuda/vecdotq.cuh`). Strata
transcribes them rather than reinterpreting them. `d` is fp16 unless stated.

#### Q2_0 (id 42): the GSQ-RCO 2-bit format

- **Layout** `block_q2_0 { ggml_half d; uint8_t qs[16]; }`: 18 B per 64 weights, 2.25 bpw.
  - Element j is at byte `j/4`, bits `(j%4)*2`, LSB-first.
  - Defined as `QK2_0 64` in the pinned ggml-common.h. ggml also has the unrelated `block_tq2_0` (ternary, 256 elements).
- **Dequant:** `w_j = (code_j - 1) * d`, code in {0,1,2,3}, so values are {-d, 0, d, 2d}. These values are exact in fp16.
- **Scale granularity:** one fp16 per 64 weights, while activation chunks are 32 elements. **A weight block spans
  TWO activation scales.** Collapsing them is the "natural-looking mistake" (`include/strata/kernels/cpu/expert.hpp`).
- **Tensors:** all routed experts in the Q2_0 file (34.0 GB, "31.64 GiB of the 38 GiB pack"), many expert downs
  of the IQ files, several dense tensors, and `blk.1.ple_key`.
- **GPU kernels:**

  | Path | Kernels | Activation |
  |---|---|---|
  | Canonical S2 (dense) | `s2_gemv_q8_kernel` | Q8_0 |
  | Canonical S2 (experts) | `s2_expert_grouped` family | Q8_0 + fp32 scales |
  | Native (dense) | `native_q2_0_mmvq_kernel` | Q8_1 |
  | Native (experts) | `iq_kernels.cu` grouped gate/up/down, `Fmt<42>` | Q8_1 |
  | Dequant | `dequant_s2_kernel`; `dequant_bf16` `group32<42>`; `dequant_flat` | - |

- **CPU kernels:**
  - AVX-512: `s2_expert_vnni*` and `q2_0_gguf_rows_multi` in `cpu/expert.cpp` (VBMI + VNNI).
  - AVX2: `q2_avx2.cpp`.
  - ggml-cpu has a scalar x86 vec_dot only. The paper says Strata's kernel is 7x faster.
- **ggml-metal:**
  - `kernel_mul_mv_q2_0_f32` (`mul_mv.metal:386`, host name `:453`).
    - Dot: `block_q_n_dot_y(block_q2_0*, sumy, yl, il)` at `:33`.
    - Computes `d * (sum_lo + 2*sum_hi - sumy)` with per-bit `select()` over **float** activations.
    - 4 lanes per block, `nr0` rows per simdgroup, `simd_sum` at the end.
  - `mul_mv_ext` r1_2..5 via `dequantize_q2_0_t4` (`:862-865`), `mul_mv_id` (`:3373`), `mul_mm`/`mul_mm_id`,
    `get_rows`, `cpy`.
  - ggml-cpu ARM also has a NEON `ggml_vec_dot_q2_0_q8_0` (`arch/arm/quants.c:222`): `vqtbl1q_u8` byte replication,
    `vshlq_u8` by {0,-2,-4,-6}, `& 3`, `- 1`, two `vdotq_s32`, and `vmlaq_n_f32(sumv, (float)p, d0*d1)`.

#### Canonical S-forms (S2 / S4 / S8 / P16 / P32): `tools/strata_pack.py`, `tools/canonical_xcheck.py:563-576`

- **Layout.** A tensor is stored as planar `codes | scales | offsets`.
  - `code_bits` is 2, 4 or 8 for S2/S4/S8, packed LSB-first.
  - The group size is a power of two (16, 32 or 64).
  - Pack scales are fp16 when the source block stores fp16 (and the round-trip is exact), otherwise f32. The
    engine always holds them as f32.
  - Offsets exist only for Q4_K and Q5_K. They are f32 (`offsets_fp16` is false for all 54).
- **Decode:**
  ```
  w = decode(code) * scale[g] + offset[g]
  decode(code) = (code + code_bias)       when codebook == none   (bias applied in the INTEGER domain)
               = kvalues_iq4nl[code & 15] when codebook == IQ4NL
  ```
  The offset is applied to the weight before the activation multiply. Writing it the other way rounds
  differently and only Q4_K could tell (`s_gemv.cu`).
- **Mappings** (raw bytes, block, group, form, codebook, bias, offset):

  | Source | Raw bytes | Block | Group | Form | Codebook | Bias | Offset |
  |---|---|---|---|---|---|---|---|
  | Q2_0 | 18 | 64 | 64 | S2 | - | -1 | no |
  | Q3_K | 110 | 256 | 16 | S4 | - | -4 | no |
  | Q4_0 | 18 | 32 | 32 | S4 | - | -8 | no |
  | Q4_K | 144 | 256 | 32 | S4 | - | 0 | yes |
  | Q5_0 | 22 | 32 | 32 | S8 | - | -16 | no |
  | Q5_K | 176 | 256 | 32 | S8 | - | 0 | yes |
  | Q6_K | 210 | 256 | 16 | S8 | - | -32 | no |
  | Q8_0 | 34 | 32 | 32 | S8 | - | -128 | no |
  | IQ4_NL | 18 | 32 | 32 | S4 | IQ4NL | 0 | no |
  | IQ4_XS | 136 | 256 | 32 | S4 | IQ4NL | 0 | no |
  | F32 | | | | P32 | | | |
  | BF16 | | | | P32 (pack) -> BF16 (engine) | | | |
  | F16 | | | | P16 | | | |

  - The repack is verified bit-exact: `decode(pack) == dequant_ggml(source)`. It was measured over
    757,139,840 blocks in 1021 tensors.
  - Q3_K's -4 is an integer code bias. Folding it into a float offset would not be exact.
- **The S-form does not determine the activation type.** Q5_0 and Q5_K, and IQ4_NL and IQ4_XS, are
  indistinguishable from S-form fields alone. So the activation kind (Q8_0 vs Q8_K) comes from the manifest's
  `source_type` through `index.txt` (`tools/pack_index.py:84-91`, `VDT`).
- **Missing doc:** `docs/pack-format.md` and `docs/activation-contract.md` are cited in comments but absent from `docs/`.
- **GPU kernels:** `s_gemv*` (S4/S8, Q8_K or Q8_0 activations), `s2_gemv*`, `dequant_s2`.
- **ggml-metal:** none. This is a Strata-private format. On a Mac, either keep the native GGUF blocks and use
  ggml-metal-style kernels, or port `s_gemv_q8_split` and `s2_gemv_q8`. Both are simple, one warp or simdgroup per row.

#### IQ1_S (id 19)

- **Layout:** `block_iq1_s { half d; uint8 qs[32]; uint16 qh[8]; }`, 50 B per 256, 1.5625 bpw.
- **Dequant:** per 32-element sub-block ib:
  - `dl = d * (2*((qh[ib]>>12)&7) + 1)`
  - `delta = (qh[ib] & 0x8000) ? -0.125 : +0.125` (`IQ1S_DELTA`)
  - for each 8-element group l, the 11-bit grid index is `qs[4ib+l] | (((qh[ib] >> 3l) & 7) << 8)`, into
    `iq1s_grid[2048]` (uint64 of 8 int8 values in {-1,0,1})
  - `w = dl * (grid + delta)`
- **Strata:** no GPU or CPU kernel. `iq_supported()` does not list 19. It appears only in the type-size tables
  (`tools/gguf_reader.py`, `ggml-common.h`).
- **ggml-metal:** `mul_mv`, `mul_mv_id`, `mul_mm`, `get_rows`. Not in `mul_mv_ext`.

#### IQ1_M (id 29)

- **Layout:** `block_iq1_m { uint8 qs[32]; uint8 qh[16]; uint8 scales[8]; }`, 56 B per 256, 1.75 bpw.
  - There is no `d` field. The fp16 super-scale is assembled from the top nibbles of the four uint16 `scales` words:
    `u16 = (sc[0]>>12) | ((sc[1]>>8)&0xF0) | ((sc[2]>>4)&0xF00) | (sc[3]&0xF000)`.
- **Dequant:**
  - 3-bit sub-scales per 16 elements: `dl = d * (2*((sc[ib/2] >> (6*(ib%2) + 3*k)) & 7) + 1)`.
  - 11-bit grid index from `qs` plus 3 bits of `qh`, into `iq1s_grid`.
  - Delta of +/-0.125 per 8 elements, from qh bits 0x08 / 0x80.
  - `w = dl*(grid + delta)`.
- **Strata kernels:**
  - GPU: `Fmt<29>` in `iq_kernels.cu` (STRATA_GU_FMTS and STRATA_MMVQ_FMTS), and `native_mmvq` dispatches 29 to `iq_mmvq`.
  - CPU: ggml-cpu vec_dot only. It is excluded from `iq_avx512.cpp`/`iq_avx2.cpp` because no shipped model has
    IQ1_M expert rows.
- **ggml-metal:** as IQ1_S. No `mul_mv_ext`.

#### IQ2_XXS (id 16)

- **Layout:** `block_iq2_xxs { half d; uint16 qs[32]; }`, 66 B per 256, 2.0625 bpw.
- **Dequant:** per 32 elements, `aux32[0..1]` come from 8 bytes of qs.
  - Four 8-bit grid indices go into `iq2xxs_grid[256]`. Each entry is a uint64 of 8 magnitudes from {0x08, 0x19, 0x2b}.
  - Four 7-bit sign indices go into `ksigns_iq2xs[128]`. These expand to 8-bit masks, with the 8th bit as parity
    (CUDA `unpack_ksigns` uses `__popc`).
  - Scale `db = d * (0.5 + (aux32[1]>>28)) * 0.25`.
  - `w = db * grid * (+/-1)`.
- **Kernels:**
  - GPU: `Fmt<16>` (gate/up and mmvq lists).
  - CPU: `iq_avx512.cpp` / `iq_avx2.cpp` multi-token, or ggml vec_dot.
- **ggml-metal:** mv, id, mm, rows. No ext.

#### IQ2_XS (id 17)

- **Layout:** `{ half d; uint16 qs[32]; uint8 scales[8]; }`, 74 B per 256, 2.3125 bpw.
  - Each qs u16 is a 9-bit grid index into `iq2xs_grid[512]` plus a 7-bit sign index.
  - A 4-bit scale applies per 16 elements: `db = d*(0.5+nibble)*0.25`.
- **Kernels:** GPU `Fmt<17>`. CPU: an AVX-512/AVX2 dedicated `row_dot_iq2xs` (`iq_avx2.cpp:215`), or ggml.
- **ggml-metal:** mv, id, mm, rows. No ext.

#### IQ2_S (id 22)

- **Layout:** `{ half d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]; }`, 82 B per 256, 2.5625 bpw.
  - The first 32 bytes of qs are the low 8 bits of grid indices. `qh` adds 2 bits each, making a 10-bit index into `iq2s_grid[1024]`.
  - The last 32 bytes of qs are explicit sign bytes.
  - Scales are as in IQ2_XS.
- **Kernels:** GPU `Fmt<22>`. CPU AVX-512/AVX2 or ggml.
- **ggml-metal:** mv, id, mm, rows. No ext.

#### IQ3_XXS (id 18)

- **Layout:** `{ half d; uint8 qs[96]; }`, 98 B per 256, 3.0625 bpw.
  - The first 64 bytes are 8-bit indices into `iq3xxs_grid[256]` (uint32 of 4 magnitudes).
  - The last 32 bytes are 8 x uint32, each holding 4 x 7-bit sign indices (via `ksigns_iq2xs`) and a 4-bit scale.
  - `db = d*(0.5 + (aux>>28))*0.5`.
- **Kernels:** GPU `Fmt<18>`. CPU AVX-512/AVX2 or ggml. It is also accepted for the native `ple_key`.
- **ggml-metal:** mv, id, mm, rows. No ext.

#### IQ3_S (id 21)

- **Layout:** `{ half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]; }`, 110 B per 256, 3.4375 bpw.
  - A 9-bit grid index (qs plus 1 qh bit) goes into `iq3s_grid[512]` (uint32 of 4).
  - Sign bytes are explicit.
  - A 4-bit scale applies per 32 elements: `db = d*(1 + 2*nibble)`.
- **Kernels:** GPU `Fmt<21>`. CPU AVX-512/AVX2 or ggml.
- **ggml-metal:** mv, id, mm, rows. No ext.

#### IQ4_NL (id 20)

- **Layout:** `{ half d; uint8 qs[16]; }`, 18 B per 32, 4.5 bpw.
  - Split-half nibble order: `qs[j]` low nibble is element j and high nibble is element j+16.
- **Dequant:** `w = d * kvalues_iq4nl[nib]`, where
  `kvalues_iq4nl = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}`.
- **Tensors:** expert downs (IQ files, Coder, Orca), `ffn_down_shexp` in some layers, and the PLE table (90 B per 160-value row).
- **GPU kernels:**
  - `native_small_mmvq_kernel` (`iq4_table_lookup`).
  - `Fmt<20>` in the down list and mmvq list.
  - `s_gemv_q8_split<4,false>` with the IQ4NL codebook (canonical).
  - `dequant_bf16` `group32<20>` with `__constant__ kv_iq4nl`.
  - Not on the GPU for PLE; that path is CPU.
- **CPU kernels:** `iq4nl256_down_rows` (`iq_avx2.cpp:424`, multi-token AVX2), ggml vec_dot, and the PLE decoder
  `ngram.cpp`.
- **Codebook placement:** `s_gemv.cu:18-35` measured that keeping the codebook in `__constant__` is **2.12x slower**,
  because divergent indices are the worst case for constant memory. It is copied to shared memory.
  - On Metal, put the codebook in `threadgroup` memory or registers, not in a `constant` buffer read with
    divergent indices. ggml-metal uses `constexpr constant` plus a threadgroup copy for `kvalues_iq4nl_f`.
- **ggml-metal:** full coverage, including ext.

#### IQ4_XS (id 23)

- **Layout:** `{ half d; uint16 scales_h; uint8 scales_l[4]; uint8 qs[128]; }`, 136 B per 256, 4.25 bpw.
- **Dequant:** a 6-bit scale applies per 32 elements:
  - `ls = ((scales_l[ib/2] >> 4*(ib%2)) & 0xF) | (((scales_h >> 2*ib) & 3) << 4)`
  - `w = d*(ls-32)*kvalues_iq4nl[nib]`
- **Tensors:** dense attention projections (13+4+4 layers), Coder gate/up, PLE key option.
- **Kernels:** `native_iq4_xs_mmvq_kernel` (`native_mmvq.cu:458`), `Fmt<23>`, `s_gemv_q8_split<4,true>`
  (canonical, Q8_K), `dequant_bf16 group32<23>`.
- **ggml-metal:** full, including ext.

#### Q4_0 (2), Q5_0 (6), Q5_1 (7), Q8_0 (8)

| Format | Layout | Bytes | bpw | Dequant |
|---|---|---|---|---|
| Q4_0 | `{half d; uint8 qs[16]}` | 18 | 4.5 | `(nib - 8)*d`, split-half nibbles |
| Q5_0 | `{half d; uint8 qh[4]; uint8 qs[16]}` | 22 | 5.5 | `((nib | hbit<<4) - 16)*d` |
| Q5_1 | `{half d; half m; uint8 qh[4]; uint8 qs[16]}` | 24 | 6.0 | `d*q5 + m` |
| Q8_0 | `{half d; int8 qs[32]}` | 34 | 8.5 | `q*d` |

- **GPU native dot products** (`native_mmvq.cu`, transcribed vecdotq):
  - Q4_0: `d*(sumi*ds.x - 4*ds.y)`.
  - Q5_0: `d*(sumi*ds.x - 8*ds.y)`, with a qh bit-scatter.
  - Q8_0: `d0*d1*sumi`.
  - Q5_1 goes to `iq_mmvq` (`Fmt<7>`). Its min term uses the activation's sum (ds.y).
- **CPU:** Q5_1's vec_dot_type is Q8_1. On the ggml-cpu side, Q8_1's s field is `d*sum(q)` (quantized sum), while
  ggml-cuda's ds.y is the float sum. This is the "Q5_1 min term" difference that `native_expert_parity` tests
  (e_q < 1e-3).
- **Q5 projections:**
  - Q5_1 `ffn_down_shexp` (Orca blk.0) and the Q5_0 PLE table.
  - Parity: `src/kernels/q5_projection_parity.cpp` and `ple_q5_parity.cpp` check `dequant_f32`/`dequant_bf16` against
    ggml with max_abs <= 1e-6.
  - **The GPU expert path does not support Q5_0 downs.** Orca's Q5_0-down layers run on the CPU.
- **ggml-metal:** full coverage, including ext and mm.

#### Q8_1 (activation only), 36 B per 32: `{ half2 ds = {d, s}; int8 qs[32] }`

- **ggml-cuda (and Strata native).**
  - Warp-wide amax and sum (`__shfl_xor_sync`), then `d = amax/127` (IEEE division) and `q = roundf(x/d)` (half away from zero).
  - `ds = half2(d, sum_of_float_x)`.
  - Implemented by `native_quantize_q8_1_kernel` (`native_mmvq.cu:141`) and `quantize_q8_1_kernel`
    (`iq_kernels.cu:979`): one thread per element, one warp per block.
- **Metal.** No ggml-metal equivalent. Metal's mul_mv uses float src1.
  - `simd_max` and `simd_sum` give the amax and sum. Metal `round()` is half away from zero, matching `roundf`.
  - The division must be correctly rounded (fast-math off).

#### Q8_K (activation only), 292 B per 256: `{ float d; int8 qs[256]; int16 bsums[16] }`

- **Quantizer (`quantize_q8_K_kernel`, `quantize_act.cu:161`).** It reproduces `quantize_row_q8_K_ref`:
  - `max` is the signed value with the largest |x|, first occurrence wins.
  - `iscale = -127/max`, then `q = nearest_int(iscale*x)`, where nearest_int is the 12582912.0f magic-number trick.
  - Then `min(127, q)` (only the upper clamp).
  - `d = 1/iscale`, with `bsums` per 16 elements.
  - An all-zero block gives d=0 and zeroed qs and bsums.
- **Exactness details:**
  - `__fmul_rn` is used to forbid FMA contraction. Once, 1 of 524,288 elements differed without it.
  - Using -128 instead of -127 is a 0.79% error.
- **Readers.** The Strata GPU S-form kernels read only `d` and `qs`. `bsums` is part of the stride and is used by ggml-cpu AVX dots.
- **Metal:** none in ggml-metal. Port the kernel verbatim with fast-math off. The magic-number `nearest_int` works
  identically in fp32.

#### K-quants Q2_K (10), Q3_K (11), Q4_K (12), Q5_K (13), Q6_K (14)

| Format | Layout | Bytes | Dequant |
|---|---|---|---|
| Q2_K | `{u8 scales[16]; u8 qs[64]; half d; half dmin}` | 84 | `d*(sc&15)*q2 - dmin*(sc>>4)` |
| Q3_K | `{u8 hmask[32]; u8 qs[64]; u8 scales[12]; half d}` | 110 | 16 x 16-element sub-blocks, 6-bit packed scales, `d*(sc-32)*(q2 - (hbit?0:4))` |
| Q4_K | `{half d; half dmin; u8 scales[12]; u8 qs[128]}` | 144 | 8 x 32, 6-bit sc and m (`get_scale_min_k4`), `d*sc*q4 - dmin*m` |
| Q5_K | `{half d; half dmin; u8 scales[12]; u8 qh[32]; u8 qs[128]}` | 176 | `d*sc*(q4 + 16*hbit) - dmin*m` |
| Q6_K | `{u8 ql[128]; u8 qh[64]; i8 scales[16]; half d}` | 210 | 16 x 16, `d*sc*(q6 - 32)` |

- **Q2_K** has no Strata kernel (not in `native_mmvq_supported`, not in dequant_bf16).
- **GPU native kernels:**
  - `native_q3_k_mmvq_kernel` (`load_int_b2` for the 110 B stride, `__vsubss4`).
  - `native_q4_k_mmvq_kernel`, `native_q5_k_mmvq_kernel`, `native_q6_k_mmvq_kernel`. These are transcribed
    `vec_dot_q*_K_q8_1_impl_vmmq`, with Q8_1 activations.
  - `Fmt<12>` and `Fmt<13>` for UD-Q4_K_XL experts (gate/up list).
- **Canonical:** `s_gemv_q8_split<4|8,true>`, with Q8_K activations.
- **CPU:** ggml vec_dot (`kq_avx2.cpp` `q4k_dot` is a bit-exact AVX2 multi-token transcription).
- **Prefill:** `dequant_bf16` `group32<11..14>`. With `-DSTRATA_MMQ_KQUANTS=ON`, MMQ prefill products are checked
  by `tests/cuda/prefill_mmq_kquant_test.cpp` (rel <= 0.04, max_abs/rms <= 0.35).
- **ggml-metal:** all five types in mv, id, ext, mm and rows.

#### BF16 (30), F16 (1), F32 (0)

- **BF16 kernels:**
  - `bf16_gemv*` (`bf16_gemv.cu`) take BF16 x.
  - `bf16_f32_mmvf*` (`native_bf16.cu`) take F32 x, and are an adaptation of ggml-cuda mmvf.
  - CPU: `bf16_rows_dot` and `bf16_rows_dot_multi` (`kq_avx2.cpp:204-219`), used for the router (2.6 MB per layer).
- **Conversion.** `include/strata/kernels/bf16_bits.hpp` does RNE with the tie on bit 16, and quiets NaN with `|64`.
  `f16_bits.hpp` is a custom RNE fp16 conversion:
  - It exists because `__float2half` was once observed producing wrong bits (0x2600 vs 0x26DB).
  - Round 198 fixed a bug where finite overflow became NaN instead of inf.
- **Metal.** MSL has `bfloat` (MSL 3.1+) and ggml-metal enables BF16 kernels only when the device supports it.
  The safe route is manual `as_type<float>(uint(bits) << 16)`, which is exact. For F16, use `half` /
  `as_type<half>`, and replicate `f16_bits` RNE for any host-side packing.

#### FP8 E4M3 "fn" (PLE table option)

- **Storage.** The GGUF type is I8, with metadata `strata.ple.format = "f8_e4m3"` and a single f32 `strata.ple.scale`
  (from the checkpoint's BF16 `weight_scale`). Rows are 160 B.
- **Decode** through a 256-entry LUT:
  - Normal values: sign, 4-bit exponent (bias 7) and 3-bit mantissa.
  - Subnormals: `m/8 * 2^-6`.
  - `e==15 && m==7` is NaN and is read as 0. This is the "fn" variant: no inf.
- **Size.** 51.2 GB, against 28.8 GB for IQ4_NL. IQ4_NL is about 8% off per row relative to FP8.
- **Code and tests:** CPU-only (`ngram.cpp`; packer `tools/ple_fp8_pack.py`; test `ple_fp8_parity.cpp`).
  No ggml type exists; on a Mac it stays a CPU LUT decode.

---

## 2. Activation-quant contract

### 2.1 The rule

| Weight | Path | Activation | Quantizer |
|---|---|---|---|
| S2 (Q2_0 canonical), dense | GPU `s2_gemv_q8` | Q8_0 (34 B) | `quantize_q8_0` |
| S4/S8 from a legacy source (Q4_0, Q5_0, Q8_0, IQ4_NL) | GPU `s_gemv_q8_0_split` | Q8_0 | `quantize_q8_0` |
| S4/S8 from a K-quant or IQ4_XS | GPU `s_gemv_q8k_split` | Q8_K (292 B) | `quantize_q8_K` |
| S2 routed experts, GPU cache hit | `moe_hit_grouped_s2*`, `moe_grouped_s2` | Q8_0 bytes plus **fp32 scales** | `quantize_q8_0_scaled` (CPU-identical) |
| S2 routed experts, CPU miss | `s2_expert_vnni*` / `q2_avx2` | `ActQ` (int8 plus fp32 scale, sum, hx) | `act_quant_q8_1` (legacy) or `quantize_oracle_q8_0` |
| Any native type, GPU (dense, head, experts) | `native_mmvq`, `iq_mmvq`, `native_expert_grouped` | Q8_1 (36 B) | `native_quantize_q8_1` / `quantize_q8_1_rows` |
| Native IQ/K types, CPU | `native_expert.cpp` | the type's vec_dot_type (Q8_K / Q8_0 / Q8_1) | ggml-cpu `from_float` |
| Native Q2_0 gate/up/down, CPU | `pool.cpp:543-563` -> `q2_rows_any` | `ActQ` | `act_quant_any` (Strata legacy, **not** ggml) |
| BF16 | GPU | BF16 x (`bf16_gemv`) or F32 x (`bf16_f32_mmvf`) | - |

`layer.cpp:129-131` states the dense rule as `code_bits == 2 -> Q8_0 (s2_gemv_q8)`, and any other quantized
form goes to Q8_K (`s_gemv_q8k_split`), except legacy sources that take Q8_0 through `wants_q8k()`. Choosing
the wrong one costs 0.6-1.4% on the GEMV: a plausible output, not a broken one.

### 2.2 The quantizers, exactly

| Quantizer | File:line | Scale | Rounding | Clamp | Stored |
|---|---|---|---|---|---|
| `quantize_q8_0_kernel` | `quantize_act.cu:47` | `d32 = amax/127.0f` (fp32). **Quantize by d32, not by the fp16 value.** | `rint((double)x / (double)d32)`, half to even, **double precision** | [-128, 127] | fp16 d (custom RNE), int8 qs. Zero block: d=0, q=0 |
| `quantize_q8_0_scaled_kernel` | `quantize_act.cu:100` | `s = amax/127.f`, `inv = 1.f/s` (CPU verbatim) | `t = x*inv; (int)(t + copysign(0.5,t))`, half away from zero | [-127, 127] | fp16 d **and** fp32 `scales[]` |
| `cpu_order_quantize_kernel` | `s2_expert_grouped.cu:445` | as above, with `__fdiv_rn` | `__fmul_rn` and `__fadd_rn` (no contraction) | [-127, 127] | fp16 d, fp32 scale, `hx = s*sum` |
| CPU `act_quant_q8_1` (legacy ActQ) | `cpu/expert.cpp:386` (AVX2 twin `q2_avx2.cpp:90`) | fp32 `amax/127`, reciprocal multiply | half away from zero | +/-127 | `ActQ{q[2560], scale[80] fp32, sum[80], hx[80]}` |
| CPU `quantize_oracle_q8_0` | `cpu/expert.cpp:77` | `d = amax/127`, stored fp16, **inverse = 127/amax** | `_mm256_round_ps` nearest-even (x86 ggml) | via packs | ActQ, scale = fp16(d) widened. Opt-in `expert_set_oracle_q8_0` |
| `native_quantize_q8_1_kernel` / `quantize_q8_1_kernel` | `native_mmvq.cu:141` / `iq_kernels.cu:979` | `d = amax/127` | `roundf(x/d)`, half away | implicit | `half2(d, sum float x)` |
| `quantize_q8_K_kernel` | `quantize_act.cu:161` | `iscale = -127/max_signed` | magic 12582912 nearest (half even), `__fmul_rn` | `min(127)` | f32 d = 1/iscale, bsums |
| ggml-cpu ARM `quantize_row_q8_0` (reference for a Mac oracle) | `ggml-cpu/arch/arm/quants.c:41` | `d = amax/127`, `id = 1/d` | `vcvtnq_s32_f32` (nearest even) | - | fp16 d |

**The ggml-cpu Q8_0 activation is not identical across architectures.** x86 AVX multiplies by `127/amax` and
ARM NEON multiplies by `1/(amax/127)`, so near-tie elements can differ. A Mac-side "ggml oracle" will not
reproduce the x86 numbers that Strata's parity logs were taken against.

### 2.3 Measured errors and gaps (from the source comments and tests)

| Quantity | Value | Where |
|---|---|---|
| FP16 activation vs quantized activation, Q2_0 GEMV | **0.9498% relative** (tolerance for FP16 paths 1e-3) | `s2_gemv_q8.cu:3-8` |
| Wrong activation kind (Q8_0 vs Q8_K) on a dense GEMV | 0.6-1.4% | `layer.cpp` comment |
| FP16 vs quantized activation, attention / K-quant GEMV | 0.6-0.9% | prior read of layer and attention code |
| fp32 vs fp16 activation scale (`quantize_q8_0` vs CPU ActQ) | 80 of 80 chunks differ, max rel **4.761e-04** (2^-11); int8 codes 0 of 2560 differ | `quantize_act.cu:84-92` |
| Quantize by fp16 d instead of fp32 d32 | 18 of 80 blocks differ from reference bytes; worse end-to-end KL | `quantize_act.cu:16-19` |
| Q8_K `-128` vs `-127` | 0.79% | `quantize_act.cu` |
| Q8_K without `__fmul_rn` | 1 of 524,288 elements differ | `quantize_act.cu` |
| CPU VNNI expert vs scalar ggml-formula oracle | rel **1.461e-06** (P0.T2) | `cpu/expert.hpp` |
| Oracle quantizing only the intermediate (bug in the old bench) | ~1.2% | `cpu/expert.hpp` |
| `s2_gemv_q8` vs reference | 1.479e-05 measured (tol 1e-4) | `s2_gemv_q8_parity.cpp` |
| `s_gemv` vs reference | ~1e-5 (FMA contraction), tol 1e-4 | `s_gemv_parity.cpp` |
| Native IQ gate rows vs ggml vec_dot | 1e-7 measured (tol 1e-5) | `native_expert_parity.cpp` |
| UD-Q4_K_XL Q8_0 to BF16 conversion of small projections | max abs 0.0144 (`hc_ffn_up`) | `docs/UNSLOTH_Q4.md:88` |

### 2.4 Reproducing the contract in Metal

- **Disable fast-math.** Use `MTLCompileOptions.mathMode = .safe` on newer SDKs, or `fastMathEnabled = NO`.
  - Use explicit `fma()` only where CUDA uses `__fmaf_rn`.
  - Elsewhere, prevent contraction. Check `#pragma METAL fp contract(off)` availability on the target MSL
    version, or split expressions through `volatile`-free barriers.
  - Without this, `__fmul_rn`-dependent results (Q8_K, the cpu_order path) will drift.
- **Correctly rounded division.** `amax/127.0f` and `x/d` need it. Confirm that MSL `precise::divide` or safe-mode `/`
  is correctly rounded on the target GPU. Otherwise compute the quotient and fix it with an `fma` residual.
- **No `double`.** For `quantize_q8_0_kernel`:
  1. Compute `r = rint(x / d32)` in fp32.
  2. Test the two half-integer boundaries exactly with `sign(fma(r +/- 0.5, d32, -x))`. The sign of an fma is
     exact because the rounding cannot flip it.
  3. Adjust r, applying half-to-even at exact ties.
  - This reproduces the float64 reference bit for bit.
  - Alternatively, adopt `quantize_q8_0_scaled` everywhere. That is the CPU-identical contract, and it is what
    the hit path already uses.
- **Rounding functions.** `round()` in MSL is half away from zero, the same as `roundf`. `rint()` is half to even.
  Both exist, so map them 1:1.
- **Exact integer dots in float.** Every integer dot here is exactly representable in fp32:
  - S2: |s| <= 32*3*128.
  - Q8: products <= 16,384, block sums <= 2^19.
  - So `dot(float4(char4), float4(char4))`, or chained fma, reproduces `__dp4a` exactly, provided each partial stays
    below 2^24.

---

## 3. Per `__global__` kernel reference

Columns:
- **Launch:** grid x block and static/dynamic shared memory.
- **Features:** the CUDA intrinsics and constructs used.
- **Multi-col:** the multi-column (several tokens) semantics.
- **Parity:** the test that covers the kernel, with its tolerance.

Common Metal mapping:

| CUDA | Metal |
|---|---|
| warp (32) | simdgroup (32) |
| `__shfl_xor_sync` | `simd_shuffle_xor` |
| `__shfl_down_sync` | `simd_shuffle_down` |
| `__syncthreads` | `threadgroup_barrier(mem_flags::mem_threadgroup)` |
| `__shared__` | `threadgroup` (32 KB max) |
| `__ldg` | plain `device const` load |
| `__byte_perm` | `as_type<uchar4>` swizzle / shifts |
| `__dp4a` | `char4` widen + `dot` / fma |
| `__expf` | `fast::exp` (different ulp) |
| `expf` | `precise::exp` |
| `__launch_bounds__` | `max_total_threads_per_threadgroup` |

### 3.1 `src/kernels/cuda/s_gemv.cu` (canonical S4/S8 dense GEMV)

| Kernel (line) | Purpose / I/O | Launch | Features | Multi-col | Parity | Metal note |
|---|---|---|---|---|---|---|
| `s_gemv_q8k_kernel<CB>` (91) | S4/S8 row x Q8_K activation (`q8k_at`), naive; y[o] f32 | 1 thread per row, 1-D | `__ldg`, smem codebook 16 B, barrier before early return | 1 col | `s_gemv_q8k_parity` 1e-5 | trivial; reference only |
| `s_gemv_q8_split_kernel<CB,Q8K>` (143) | **production** S4/S8 GEMV with Q8_K (`Q8K=true`) or Q8_0 activations; 16 accumulators per lane (quads), group index by shift | 256 threads = 8 warps = 8 rows per block, grid `ceil(n_out/8)` | `__shfl_down_sync` reduction, smem IQ4NL codebook, 64-bit division avoided (`group_shift`) | 1 col (called per token) | `s_gemv_q8k_parity` 1e-5, `s_gemv_parity` 1e-4 | one simdgroup per row; codebook in threadgroup memory; deterministic reduction order: keep `((a0+a1)+(a2+a3))...` tree |
| `s_gemv_kernel<CB>` (302) | S2/S4/S8 x FP16 x, naive | 128 threads, thread per row | smem codebook | 1 | `s_gemv_parity` 1e-4 | reference only |
| `s_gemv_split_kernel<CB>` (401) | S x FP16 x, block per row, tree reduction (used by `concurrent_main`/`overlap_main` tools) | grid n_out, block tpr (32), dyn smem tpr*4 B | `extern __shared__`, tree reduce | 1 | `s_gemv_parity` | not on the engine's main path |

Host entry points: `s_gemv` :344, `s_gemv_split(_async)` :533/:538, `s_gemv_q8k` :580, `s_gemv_q8k_split` :607,
`s_gemv_q8_0_split` :647. The callers are in `src/core/layer.cpp:172` and `:989`.

### 3.2 `s2_gemv.cu`, `s2_gemv_quads.cu`, `s2_gemv_fast.cu`, `s2_gemv_q8.cu`

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `s2_gemv_kernel` (s2_gemv.cu:19) | S2 x FP16, naive reference | thread per row | none | `s2_gemv_parity` 1e-5 | reference |
| `s2_gemv_quads_kernel` (s2_gemv_quads.cu:33) | S2 x FP16, one code byte plus one uint2 (4 halves) per quad | grid n_out, block tpr, smem tpr*4 | `uint2` vector load, `__half2` | `s2_gemv_parity` | experiment, not called by the engine |
| `s2_gemv_fast_kernel<STAGE_X>` (s2_gemv_fast.cu:70) | as quads plus optional smem-staged x plus `__constant__ c_codes[256][4]` LUT. **Measured not faster.** | grid n_out, block tpr | `cudaMemcpyToSymbol` per device | `s2_gemv_parity` | do not port |
| `s2_gemv_q8_kernel` (s2_gemv_q8.cu:33) | **production** S2 dense x Q8_0. Activation dequantized per element (`xq*dx`), float math, no dp4a | grid n_out, block tpr (32), smem tpr*4 | tree reduction | `s2_gemv_q8_parity` 1e-4 (1.479e-5) | straightforward |

### 3.3 `src/kernels/cuda/s2_expert_grouped.cu` (S2 routed experts on the GPU, 1190 LOC)

The arithmetic matches the CPU's: `sum (c-1)*dw*xq*dx = dw*dx*(sum c*xq - sum xq)`. Both sums are int8 dp4a.
`dw` is fp16 per 64 and `dx` is fp16 or fp32 per 32. Blob geometry is restated as literals at lines 36-45.

| Kernel (line) | Purpose / I/O | Launch | Features | Multi-col / exactness | Parity | Metal note |
|---|---|---|---|---|---|---|
| `gu_kernel` (121) | per-hit gate/up rows; `blob = blob_base + slot_index[h]*blob_bytes`; output planar [gate(FF) per hit, then up(FF)] by row parity | 256 threads, warp per row, grid `ceil(n_hits*1280/8)` | `STRATA_DP4A` x2 per word (codes, ones), `memcpy` 4-B loads (34-B blocks are unaligned), `__shfl_down_sync` | per hit | `s2_expert_grouped_parity`, `moe_hit_parity` | char4 dot exact; unaligned 34-B blocks: read bytes or restage to aligned words |
| `swiglu_kernel` (166) | `g/(1+__expf(-g))*u` in place | 256 | `__expf` | - | (via grouped parity) | `fast::exp` differs in ulp; tolerance only |
| `down_kernel` (178) | per-hit down rows from the quantized intermediate; `out[dst_index[h]*H + r]` | warp per row | dp4a | per hit | same | |
| `gu_pair_kernel` (284), `down_pair_kernel` (327) | "same integers from fewer instructions": two rows per warp; activations regrouped by a 4x4 byte transpose (8 `__byte_perm`); code fields `(w >> 2f) & 0x03030303`; `hx` once per chunk | warp per 2 rows | `__byte_perm`, uint2 code loads (needs 8-B aligned arena) | **bitwise equal** to `gu_kernel`/`down_kernel` | grouped parity (new vs old bitwise) | the regrouping is the best NEON and Metal trick too: no code unpack transpose |
| `activation_correction_kernel` (368) | `hx[c] = __fmul_rn(scale[c], sum q)` for the cpu-order path | 256 | `__fmul_rn` | - | cpu-order test | |
| `cpu_order_projection_kernel<DOWN>` (412) | **replays the CPU AVX-512 summation order bit-exactly**: 8 lanes per row; per 64-block `acc = fma(d*xs, lo, acc)` twice; correction subtracted at the end; shuffle order 4,1,2 (`_mm_add_ps` + 2 `hadd`) | `blockDim/8` rows per block | `__fmaf_rn`, `__fmul_rn`, `__fadd_rn`, `__fsub_rn`, width-8 `__shfl_down_sync` | per hit | opt-in `--expert-cache-cpu-order`; CPU equality | width-8 shuffles need manual lane math in Metal; only needed if the CPU NEON order is the reference |
| `cpu_order_swiglu_kernel` (435) | accurate `expf`, `__fdiv_rn`, `__fmul_rn` | 256 | | | | `precise::exp`; libm last-bit differences are a known caveat |
| `cpu_order_quantize_kernel` (445) | CPU ActQ quantizer verbatim | 256, thread per 32-chunk | `__fdiv_rn`, `__float2half_rn` | | | |
| `hit_select_kernel` (623) | from router ids and `res_row` (VRAM slot per expert, or -1), build slot / dst / count for hits | `<<<1,32>>>` | | k <= 32 | | becomes unnecessary with unified memory (everything is a hit) |
| `hit_select_multi_kernel` (643) | the same for n <= 128 entries (verify window) | `<<<1,128>>>` | | | | as above |
| `add_hits_kernel` (666) | `parts[dst] += hit_out` | 256 | | | | |
| `gu_grouped_kernel` (783) | **grouped**: one block = GU_ROWS=32 gate/up rows of ONE expert group; up to GMAX=8 entries (tokens) staged in smem as aligned words; each code chunk loaded once and dotted with every entry | grid (40, cap_groups), 256 threads (4 rows per warp); smem `int[8][640]` + `float[8][80]` = 23,040 B | `__launch_bounds__(256)`, dp4a, `static_assert(GMAX >= kVerifyMaxT)` | **every entry bitwise equal** to the per-hit kernel | grouped parity, bitwise | fits the 32 KB threadgroup memory limit; groups beyond 8 entries would be dropped silently (assert) |
| `down_grouped_kernel` (845) | grouped down, D_ROWS=64 per block (8 per warp) | grid (40, cap_groups), 256 | | bitwise | | |
| `gu_grouped_t_kernel` (918), `down_grouped_t_kernel` (984) | fast variants: word-major staging `xs_w[j*STRIDE+i]` plus `int2 (dx,hx)`; two rows per warp; bank-conflict-free | as above | | **bitwise** equal to the non-t kernels (`STRATA_OLD_GROUPED=1` selects old) | grouped parity | |
| `group_resident_kernel` (1047) | builds groups on the device when every expert is resident at `base + id*blob` (today only the MTP layer): first-appearance order, entries in routing order | `<<<1,128>>>`, smem 5x128 int | O(n^2) scan in shared memory | | | **becomes the general path on Apple Silicon** (whole arena addressable) |

Host entry points:
- `moe_hit_grouped_s2_dev` / `_multi` (lines 690-745). Their sequence is gu, swiglu, `quantize_q8_0_scaled`
  (fp32 scales) or `quantize_q8_0`, then down.
- `moe_grouped_s2` (1098).
- `moe_hit_add` (1138).
- `moe_hit_grouped_s2_cpu_order` (1146).

### 3.4 `src/kernels/cuda/native_mmvq.cu` (native dense / head mat-vec, 1495 LOC)

These are transcriptions of llama.cpp `mmvq.cu` and `vecdotq.cuh`. All take Q8_1 activations (36-B blocks, `Q81Block`).

| Kernel (line) | Purpose | Launch | Features | Multi-col | Parity | Metal note |
|---|---|---|---|---|---|---|
| `native_quantize_q8_1_kernel` (141) | f32 to Q8_1, ggml-cuda contract | 256 threads, thread per element | `__shfl_xor` amax/sum, `roundf`, `make_half2` | n rows | (`iq_parity`, `mmvq_multi_parity`) | `simd_max`/`simd_sum` |
| `native_q5_k_mmvq_kernel<SmallK>` (221) | Q5_K x Q8_1 (`q5_q8_dot_impl`, exact `vec_dot_q5_K_q8_1_impl_vmmq`) | block (32,4); `SmallK`: 4 rows per block, else 1 row per block | dp4a, smem `partial[3][ROWS][32]`, warp sum | 1 (ncols>1 goes to multi kernel) | `iq_parity`-style; head/dense end-to-end | ggml-metal `kernel_mul_mv_q5_K_f32` exists but float activations |
| `native_q2_0_mmvq_kernel<SmallK>` (289) | Q2_0 x Q8_1: qs read as int16; `__byte_perm(0x020100ff, 0x020100ff, q>>0 / q>>2)` maps codes to signed bytes {-1,0,1,2}; `__byte_perm(qe,qo,0x5140/0x7362)` interleaves; 2 dp4a per j; `d2*d8*sumi` (no hx, uses ds.x only) | same | `__byte_perm`, dp4a | 1 | `iq_parity` (Q2_0 fixture) | LUT by `(code - 1)` on `char4` is simpler in MSL |
| `native_q3_k_mmvq_kernel` (377) | Q3_K x Q8_1; `load_int_b2` (110-B stride is only 2-B aligned), `__vsubss4` | same | `__vsubss4` (per-byte saturating subtract) | 1 | `iq_parity` (Q3_K) | emulate `__vsubss4` with `char4` subtraction (no saturation occurs in range) |
| `native_iq4_xs_mmvq_kernel` (458) | IQ4_XS x Q8_1, `get_int_from_table_16` | same | `__byte_perm` table lookups | 1 | `iq_parity` | threadgroup codebook |
| `native_q4_k_mmvq_kernel` (553) | Q4_K | same | dp4a | 1 | | |
| `native_q6_k_mmvq_kernel` (628) | Q6_K | same | dp4a | 1 | | |
| `native_small_mmvq_kernel<Weight,Qi,SmallK>` (740) | Q4_0, Q5_0, Q8_0, IQ4_NL (32-element blocks) | same | dp4a, qh bit-scatter for Q5_0 | 1 | | |
| `native_mmvq_multi_kernel<F,NCOLS,NW,ROWS>` (1001) | ncols 2..8: `F::load` once per (row, block), `F::apply` per column | exact mode (`g_multi_exact=true`, default): NW=4, ROWS=1 (or 4 for small K), identical to ncols=1; upstream mode: ncols 2-4 use 4 warps, 5-8 use 2 warps, 2 rows per block | smem `partial[NW-1][NCOLS][ROWS][32]` | **exact: every column bitwise equal to a 1-column call**; upstream: equal to rounding | `mmvq_multi_parity` (bitwise with multi_exact on) | keep exact mode for reproducibility |

Dispatch (`native_mmvq.cu:1477-1493`):
- 2, 6, 8 and 20 go to the small kernel.
- 11, 12, 13, 14, 23 and 42 go to the dedicated kernels.
- **7 (Q5_1) and 16/17/18/21/22/29 go to `iq_mmvq`** (iq_kernels.cu).

Requirements: ncols 1..8, a non-null stream, and 4-B aligned pointers.

### 3.5 `src/kernels/cuda/iq_kernels.cu` (IQ formats, native experts, embedding, prefill dequant, 1474 LOC)

Helpers transcribed from llama.cpp: `get_int_b2/b4`, `unpack_ksigns` (`__popc` parity), `get_int_from_table_16`
(`__byte_perm`), and the codebook grids from `ggml-common.h`. `Fmt<TY>` supplies qk, ipb (int8s per block) and step.

| Kernel (line) | Purpose | Launch | Features | Multi-col / exactness | Parity | Metal note |
|---|---|---|---|---|---|---|
| `mmvq_kernel<TY>` (501) | IQ/Q2_0/K/Q5_1/Q8_0 x Q8_1 rows; `row_dot` = one warp per row, lanes stride over blocks, xor warp sum; per-column loop | block (32,4), row = `blockIdx.x*4 + y`, `__launch_bounds__(128)` | `__shfl_xor`, dp4a, `__popc`, `__byte_perm` | per column, re-decodes | `iq_parity` (dq < 1e-6, mm < 2e-2 rel, columns bitwise) | ggml-metal has mv for all; ext missing for IQ1/2/3 |
| `mmvq_multi_kernel<TY,NC>` (833) | decode-once variant via `Split<TY>::load/apply`, NC in {1,2,4,8}, chunks of 8 beyond | same | | **bitwise equal** to `mmvq_kernel`; `kSplit` false for 12, 13, 7, 8 (they keep the old kernel); `STRATA_OLD_IQ_MMVQ` restores | `iq_multi_parity` (bitwise new vs old; rel < 1e-2 vs reference) | |
| `native_gu_kernel<TG>` (858) | grouped native gate/up: rows < n_ff are gate, otherwise up; entries of the group read token `ent_tok[e]`'s Q8_1 row; out `gate/up[e*n_ff + r]` | grid (`ceil(2*n_ff/8)`, cap_groups), 256 threads, GU_ROWS=8 (one warp per row), `__launch_bounds__(256)` | dp4a | per entry | `native_expert_parity` (rel <= 1e-5 vs ggml vec_dot, 1e-7 measured) | |
| `native_gu_multi_kernel<TG>` (887) | decode once per (row, block), apply to GRP_NC=4 entries at a time (8 would need 64-80 registers against 48) | same | | **bitwise equal** to per-entry | `native_expert_parity` (decode-once vs per-entry bitwise) | register pressure differs on Apple GPUs; retune NC |
| `swiglu_entries_kernel` (918) | `h = g/(1+__expf(-g))*u` | 256 | `__expf` | | | |
| `native_down_kernel<TD>` (927) / `native_down_multi_kernel<TD>` (950) | grouped down, writes `out[ent_dst[e]*n_embd + r]` | grid (`ceil(n_embd/8)`, cap_groups), 256 | | multi bitwise equal to single | `native_expert_parity` | |
| `quantize_q8_1_kernel<TD>` (979) | f32 to Q8_1 (inputs and the intermediate h) | 256, thread per element | xor shuffles, `roundf` | | | |
| `dequant_flat_kernel<dst_t>` (1254) | contiguous IQ/K/Q2_0/... to fp16/fp32 (`iq_dequant_f16/_f32`), types per `dq_dispatch` (16,17,18,20,21,22,29,23,11,42,12,13,7,8,30) | grid n/256, 32 threads (one superblock per block) | `__float2half` | | `iq_parity` dq_err < 1e-6 | ggml-metal `get_rows_*` / `dequantize_*` functions cover these |
| `dequant_gu_kernel` (1259) | one expert's gate+up to the interleaved fp16 prefill layout (row 2r = gate r, 2r+1 = up r) | grid (n_ff*per_row, 2), 32 | | | | prefill path (part on prefill) |
| `embed_rows_kernel` (1384) | token_embd rows for token ids to f32; **table may be mapped host memory** | grid (n_embd/256, n_tok), 32 | reads host-mapped memory over PCIe | | | unified memory: direct read from the mmapped GGUF; ggml-metal `get_rows` exists for all types |

Lists (macros):
- `STRATA_GU_FMTS = 16 17 18 21 22 23 29 42 12 13 8`
- `STRATA_D_FMTS = 20 23 42 7 8`
- `STRATA_MMVQ_FMTS = 16 17 18 20 21 22 23 29 42 12 13 7 8`

Geometry: `native_expert_layout` (1423) gives `up_off = n_ff*gu_row` and `down_off = 2*up_off`. Scratch is three
float arrays plus hq, each 256-B aligned.

### 3.6 `bf16_gemv.cu` and `native_bf16.cu`

| Kernel (line) | Purpose | Launch | Features | Multi-col | Parity | Metal note |
|---|---|---|---|---|---|---|
| `bf16_gemv_naive_kernel` (bf16_gemv.cu:18) | BF16 W x BF16 x, thread per row (uncoalesced): 4.66 ms per token, 194 us per call, ~210 GB/s vs 641 achievable | 1-D | | 1 | `bf16_gemv_parity` 1e-5 | do not port |
| `bf16_gemv_warp_kernel` (32) | warp per row, used when n_out >= 64 | 256 threads (8 rows per block) | `__shfl_down_sync` | 1 | same | simdgroup per row |
| `bf16_gemv_split_kernel` (48) | block per row, tpr threads, shared tree | grid n_out, dyn smem tpr*4 | | 1 | same | |
| `bf16_f32_mmvf_kernel<BS>` (native_bf16.cu:44) | F32 x times BF16 W (llama.cpp mmvf adaptation): pair loads, two ordered `__fmaf_rn`, xor warp sum, shared `partials[32]`; BS chosen 32..256 by iteration count | grid n_out, block BS | `__fmaf_rn`, `__shfl_xor_sync` | 1 | (layer/verify parity; matches ggml-cuda mmvf) | ggml-metal `kernel_mul_mv_bf16_f32` exists (float x); requires n_in even and x 8-B aligned |
| `bf16_f32_mmvf_multi_kernel<BS,NT>` (77) | n_tok <= 8 tokens, NT 4 or 8 | grid n_out | | **bit-identical** to single | | |

### 3.7 `dequant_bf16.cu`, `dequant_s2.cu`, `quantize_act.cu`

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `dequant_kernel<TY,T>` (dequant_bf16.cu:169), body `group32<TYPE,T>` (:41) | GGUF rows to bf16 (RNE, NaN quieted), fp16 or f32 for the prefill tensor-core GEMMs; types 2, 6, 7, 8, 20, 11, 12, 13, 14, 23, 42 | 256 threads, thread per 32-element group, `gpr` groups per row; block table :179-189 | `__constant__ kv_iq4nl` | `dequant_bf16_test` (F32 1e-6 rel, BF16 half-ulp 2^-8); `q5_projection_parity`, `ple_q5_parity` (max_abs 1e-6) | prefill can use ggml-metal-style `mul_mm` (dequant inside a simdgroup_matrix tile) instead of a dequant pass |
| `dequant_s2_kernel` (dequant_s2.cu:24) | S2 to f32/f16, `(code-1)*d`, thread per 64-block | 1-D, synchronous | | `dequant_s2_parity` (memcmp, bit-exact) | trivial |
| `quantize_q8_0_kernel` (quantize_act.cu:47) | see 2.2 | 128 threads, thread per 32-block, grid `ceil(nb/128)` | **double division**, `rint`, custom f16 | `quantize_act_parity` (bit-exact incl. ties, 1e-6/1e-7 magnitudes) | emulate double (2.4) |
| `quantize_q8_0_scaled_kernel` (100) | CPU-identical Q8_0 plus fp32 scales | 128 | | same | direct port |
| `dequant_q8_0_kernel` (129) | Q8_0 to f32 (`q*d16`) | 128 | | same | |
| `quantize_q8_K_kernel` (161) | see 2.2 | 128, thread per 256-block | `__fmul_rn`, magic rounding | same | direct port, fast-math off |
| `dequant_q8_K_kernel` (210) | Q8_K to f32 | 128 | | same | |

---

## 4. Per-.cu file table

| File | LOC | Purpose | Kernels | Notable CUDA features | Parity test(s) | Metal note |
|---|---|---|---|---|---|---|
| `src/kernels/cuda/s_gemv.cu` | 694 | canonical S4/S8 (and S2) dense GEMV, FP16 / Q8_K / Q8_0 activations | 4 (templated on code bits) | smem codebook, warp shuffles, shift-not-divide | `s_gemv_parity` (1e-4), `s_gemv_q8k_parity` (1e-5) | port `s_gemv_q8_split` only, if canonical packs are kept |
| `s2_gemv.cu` | 63 | naive S2 x FP16 reference | 1 | - | `s2_gemv_parity` 1e-5 | reference only |
| `s2_gemv_quads.cu` | 93 | S2 quad loads (L5 experiment) | 1 | uint2, half2 | `s2_gemv_parity` | skip |
| `s2_gemv_fast.cu` | 153 | negative result (smem x, constant LUT) | 1 | `__constant__`, `cudaMemcpyToSymbol` | `s2_gemv_parity` | skip |
| `s2_gemv_q8.cu` | 96 | production S2 dense x Q8_0 | 1 | tree reduction | `s2_gemv_q8_parity` 1e-4 | port |
| `s2_expert_grouped.cu` | 1190 | S2 routed experts: per-hit, grouped (GMAX 8), cpu-order bit-exact path, device grouping | 17 | dp4a, `__byte_perm`, `__launch_bounds__`, `__f*_rn`, width-8 shuffles, getenv A/B switches | `s2_expert_grouped_parity` (<= 4e-5*mag vs CPU; new vs old bitwise; mapped host blob), `moe_hit_parity` | core of a canonical-pack Mac port; grouped kernel maps cleanly |
| `native_mmvq.cu` | 1495 | native dense/head mat-vec for 17 GGUF types, Q8_1 quantizer, exact multi-col | 9 (plus template instances) | dp4a, `__byte_perm`, `__vsubss4`, `__launch_bounds__(...,1)` | `iq_parity`, `mmvq_multi_parity` (bitwise), `native_dense_ple_key_test` | could be replaced by ggml-metal `mul_mv` (float x) at the cost of contract parity |
| `iq_kernels.cu` | 1474 | IQ formats: mmvq, grouped native experts (gate/up/down), Q8_1 quantizer, dequant (flat, gu), embedding rows | 11 | dp4a, `__popc`, `__byte_perm`, grids `ksigns`/`iq*_grid` in device memory | `iq_parity`, `iq_multi_parity`, `native_expert_parity` | ggml-metal has every IQ dequant/dot routine to borrow |
| `bf16_gemv.cu` | 141 | BF16 x BF16 GEMV (naive / warp / split) | 3 | shuffles | `bf16_gemv_parity` 1e-5 | simdgroup per row |
| `native_bf16.cu` | 186 | F32 x BF16 mmvf (llama.cpp), single and multi-token | 2 | `__fmaf_rn` ordered | (bit-identical multi) | `kernel_mul_mv_bf16_f32` / ext |
| `dequant_bf16.cu` | 261 | GGUF rows to bf16/f16/f32 for prefill GEMM | 1 template (11 types) | `__constant__` codebook | `dequant_bf16_test`, `q5_projection_parity`, `ple_q5_parity` | replace with dequant-in-GEMM |
| `dequant_s2.cu` | 51 | S2 to float | 1 | - | `dequant_s2_parity` (bit-exact) | trivial |
| `quantize_act.cu` | 313 | Q8_0 (ggml-ref and CPU-identical) and Q8_K activation quantizers and dequantizers | 5 | double, `rint`, `__fmul_rn`, custom f16 | `quantize_act_parity` (bit-exact) | double emulation required for `quantize_q8_0` |
| `src/core/pinned.cu` | 549 | host arena reserve (large pages / normal / shared file), `cudaHostRegister(Portable\|Mapped)` with cap and slices | 0 | host registration | (`expert_layout_test`, integration) | replace by `mmap` plus `newBufferWithBytesNoCopy` (16 KB page alignment) |

Related headers:
- `include/strata/kernels/dp4a.hpp`: `STRATA_DP4A` = `__dp4a`, with a signed-byte fallback for sm < 6.1 and
  `__dp4a` provided through `hip_compat` on HIP.
- `strata_spin_pause` = `__nanosleep(100)`.
- `f16_bits.hpp` and `bf16_bits.hpp`: the conversions.

---

## 5. Expert blob layout, grouping and host-mapped / PCIe paths

### 5.1 Canonical S2 blob (Q2_0 pack): `include/strata/kernels/cpu/expert.hpp`

There is one blob per (layer, expert). `BLOB = 1,382,400 B` and every layer uses the same size. Within one blob:

| Offset | Size | Content |
|---|---|---|
| 0 (`O_GU_CODES`) | 819,200 | 1280 interleaved gate/up rows x 640 B of 2-bit codes. Row-slot 2r is gate row r, 2r+1 is up row r. |
| 819,200 (`O_D_CODES`) | 409,600 | 2560 down rows x 160 B |
| 1,228,800 (`O_GU_SCALES`) | 102,400 | 1280 rows x 40 fp16 scales (one per 64) |
| 1,331,200 (`O_D_SCALES`) | 51,200 | 2560 rows x 10 fp16 |

- **Totals.** A layer is 512 x 1,382,400 = 707,788,800 B; all 48 layers are 33.97 GB. Ten experts x 48 layers is
  **0.664 GB read per token**.
- **Row interleaving.** The gate/up interleaving was once decoded correctly but written to the wrong output
  slot. That gave a worst-case relative error of 2.2e+03, with plausible-looking numbers. `moe_hit_parity` caught it.
  Gate and up rows of the same r are adjacent, which the pair kernels exploit.
- **Arena offset.** In the arena, `blob_offset(layer, e) = layer*512*BLOB + e*BLOB` (`ExpertLayout`, `expert_layout.hpp`).

### 5.2 Native blob (IQ / K-quant packs): `include/strata/kernels/cpu/native_expert.hpp`, `iq_kernels.hpp`

- **Layout.** A native blob is `[gate rows | up rows | down rows]`, the raw GGUF slices back to back:
  - `gu_row = ggml_row_size(gu_type, 2560)` and `d_row = ggml_row_size(d_type, 640)`
  - `up_off = 640*gu_row`, `down_off = 2*up_off` and `bytes = down_off + 2560*d_row`
- **Per-layer table.** The blob size and types change per layer. `native_experts.txt` (v3/v4) lists, one line per layer:
  `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down shards]`.
  - The per-role shard column exists because UD-Q4_K_XL's layer 11 has its down in shard 2 and its gate/up in shard 3.
  - The arena can be filled directly from GGUF offsets when there is no `experts.bin`.
- **Row sizes** (bytes):

  | Format | gate/up row (2560) | down row (640) |
  |---|---|---|
  | IQ1_M | 560 | - |
  | IQ2_XXS | 660 | - |
  | Q2_0 | 720 | 180 |
  | IQ2_XS | 740 | - |
  | IQ2_S | 820 | - |
  | IQ3_XXS | 980 | - |
  | IQ3_S | 1100 | - |
  | IQ4_XS | 1360 | - |
  | Q4_K | 1440 | - |
  | Q5_K | 1760 | - |
  | IQ4_NL | - | 360 |
  | Q4_0 | - | 360 |
  | Q5_0 | - | 440 |
  | Q5_1 | - | 480 |
  | Q8_0 | - | 680 |

- **Example blobs:**
  - IQ2_XS gate/up with Q2_0 down: 2*640*740 + 2560*180 = 1,408,000 B.
  - IQ3_XXS with IQ4_NL down: 2,176,000 B.
  - Q4_K with Q5_1 down: 3,072,000 B.
- **Activations per token:** Q8_1 is 80 x 36 = 2880 B (gate/up) and 20 x 36 = 720 B (down). The CPU path uses
  Q8_K, 10 x 292 B (`kNativeActBytes = 4096`, `kNativeHBytes = 1024`).

### 5.3 Grouping (S2 and native)

**Per-hit path** (`moe_hit_grouped_s2_dev`, one token):
1. `hit_select_kernel` turns the top-10 ids and `res_row` (VRAM slot per expert) into slot / dst / count.
2. `gu_kernel` (or `gu_pair_kernel`) runs over n_hits x 1280 rows.
3. `swiglu_kernel`, then `quantize_q8_0_scaled` produce `h_q8_0` and `h_scales`.
4. `down_kernel` writes `out[dst*H + r]`.
5. `moe_hit_add` adds the result into the layer output.

**Grouped path** (`moe_grouped_s2`, verify window of T <= kVerifyMaxT tokens, plan v0.3 P6):
- The host (or `group_resident_kernel`) builds:
  - `grp_ptr[g]`: the device address of an expert blob, either a VRAM slot or the device alias of a mapped host blob
  - `grp_start[g]`, `n_groups`
  - `ent_tok[e]`: the token whose activation entry e reads
  - `ent_dst[e]`: the output row
- Groups are distinct experts in first-appearance order; entries are in routing order. A window routes about 1.4-1.5
  tokens to each distinct expert.
- **Kernels.** `gu_grouped(_t)` runs on a grid of (40, cap_groups) with 32 rows per block; up to 8 entries are
  staged in shared memory. Then come swiglu and the quantize, and `down_grouped(_t)` on (40, cap_groups) with
  64 rows per block.
- **Exactness.** Every entry is bitwise the per-hit kernel's result.

**Native grouped path** (`native_expert_grouped`, `iq_kernels.cu`): the same structure.
- `native_gu(_multi)` runs on (`ceil(1280/8)`, cap_groups), followed by `swiglu_entries`, `quantize_q8_1` (h) and
  `native_down(_multi)` on (320, cap_groups).
- `native_expert_supported(gu, d, 2560, 640)` is checked for every layer at startup.
- Q5_0 down is **not** in `STRATA_D_FMTS`. This is why Orca's Q5_0-down layers cannot use the GPU path.

**CPU pool** (`src/kernels/cpu/pool.cpp`, 753 LOC):
- Distinct experts are split into row ranges (`s2_expert_gu_rows` / `_down_rows`, and the `_multi` forms for up
  to MAXT=8 tokens).
- An intermediate `act_quant_q8_1` (or `act_quant_any`) runs between the gate/up and down phases.
- Native layers go through `run_split_multi_native`, which uses `native_gu_rows` / `native_down_rows`.
  `mode_ == 5 && gu_type == 42` takes the Q2_0 path through `q2_rows_any`.

### 5.4 Host-mapped and PCIe paths (all CUDA/HIP specific)

- **Arena registration.** `src/core/pinned.cu` reserves the arena with large pages when it can, then registers it with
  `cudaHostRegister(base, bytes, cudaHostRegisterPortable | cudaHostRegisterMapped)`.
  - A cap applies (`STRATA_ARENA_PIN_GIB`). On failure it falls back to per-slice registration of a prefix
    (`registered_bytes`, `slice_bytes`), and then to `NormalPages` with a working-set lock.
- **Planning** (`src/core/expert_source.cpp:1620-1705`):
  - Per window, the distinct experts in routing order are classified per entry as -1 CPU, 0 VRAM (cache hit) or 1 PCIe.
  - The last `m = (nmiss * pcie_num) >> 8` misses go to the GPU over PCIe. This is the `--pcie-frac` setting, in
    units of 1/256.
  - `GpuPlanSink` publishes:
    - counts: [0] VRAM groups, [1] entries, [2] PCIe groups
    - arrays: `start`, `dst`, `tok`, `ptr` (cache slot: `cache_base + slot_off[slot]` for native variable-size
      blobs, else `slot*blob`), `ptr2`, `start2`, `staging`
  - The plan is published first, so the GPU starts while the CPU works.
- **`pcie_mode` determines how a PCIe blob reaches the kernel:**
  - **0**: a DMA copy-engine fetch into VRAM staging (`P.fetch(ctx, dma_src, fetches, bytes)`).
  - **1**: the grouped kernel reads the mapped host arena directly through `device_alias(layer, id)`, a
    zero-copy PCIe read per row.
  - **2**: a copy kernel inside the CUDA graph stages the blobs (`fetch_blobs` + `rebase_ptrs`).
- **Verify windows** (`src/core/verify.cpp:754-757`):
  - Native windows quantize into `nat_xq_` (36-B Q8_1 blocks) and call `native_expert_grouped`.
  - S2 windows use `moe_grouped_s2` with `hit_xq_` (34-B Q8_0) plus `hit_xs_` (fp32 scales).
- **Remote experts** (`src/core/remote_experts.cpp:190-205`):
  - Zero-copy mapped input/output buffers (`cudaHostAllocMapped` + `cudaHostGetDevicePointer`).
  - Activations are staged with `quantize_q8_1_rows` (native) or `quantize_q8_0_scaled` (S2).
- **Other mapped memory:**
  - the doorbell / sequence word (a kernel writes into mapped pinned memory because `cudaEventRecord` inside a
    capture was silently dropped; `layer.cpp:109`)
  - the session `y_miss` (CPU expert results summed on the GPU)
  - the MTP buffers
  - `native_head` and the `token_embd` table in mapped host memory, read by `embed_rows_kernel`
  - test-only: `s2_expert_grouped_parity` checks a `cudaHostAllocMapped` blob (`:374`)

### 5.5 What this becomes on Apple Silicon

- **One address space.** mmap the pack (or `experts.bin` / GGUF shards) and wrap it in `MTLBuffer`s with
  `newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared`.
  - The pointer and length must be page-aligned, and the page size is 16 KB on Apple Silicon.
  - A Q2_0 layer (707.8 MB) or a native layer slice is a natural buffer granularity, and keeps each buffer under
    `maxBufferLength`.
  - Then `grp_ptr[g] = gpu_address(buffer) + offset`, using `MTLBuffer.gpuAddress` with argument buffers or
    pointer arithmetic in MSL 3.
- **What disappears:**
  - `pinned.cu` registration, `pcie_mode` 0/1/2, the VRAM expert cache (`res_row`, `hit_select`, LRU/profile
    slot management) and the `--pcie-frac` split.
  - `pcie_mode = 1` semantics (kernel reads host memory directly) are the native behaviour, at full memory
    bandwidth instead of PCIe.
- **General grouping.** `group_resident_kernel`'s "every expert resident at `base + id*blob`" assumption becomes
  true for all layers. Native packs need `base + layer_offset + id*blob_bytes(layer)`.
- **CPU/GPU split.** On a discrete GPU the CPU pool exists to use host DRAM bandwidth that the GPU cannot reach.
  On Apple Silicon CPU and GPU share one memory system, so running some experts on the CPU adds little
  aggregate bandwidth.
  - The likely design is all experts on the GPU, with the CPU kernels kept as the oracle and fallback.
  - A chosen activation contract must then be used everywhere: `quantize_q8_0_scaled` (CPU-identical) for S2,
    and Q8_1 for native.
- **Memory limits.** `recommendedMaxWorkingSetSize` (and the `iogpu.wired_limit_mb` sysctl) bounds how much the
  GPU can keep wired. 34-77 GB of experts plus about 6 GB dense needs a 64-128 GB Mac and may need that limit
  raised. Use `MTLResidencySet` (macOS 15+) or `useResource:` for residency.

---

## 6. CPU kernels and their NEON / Apple equivalents

| File (LOC) | Function(s) | ISA used | What it does | Parity | NEON / Apple equivalent |
|---|---|---|---|---|---|
| `src/kernels/cpu/expert.cpp` (683) | `act_quant_q8_1` :386, `quantize_oracle_q8_0` :77, `s2_expert_vnni(_q)` :457/:462, `s2_expert_gu_rows`/`_down_rows` :484/:494, `_multi` :522/:536, `s2_expert_vnni_multi` :611, `q2_0_gguf_rows_multi` :597, `q2g_rows` :588, `s2_expert_scalar` :629 (oracle) | AVX-512 F/BW/VL/**VNNI** (`vpdpbusd`) and **VBMI** (`vpmultishiftqb`, argument order (control, data), found empirically: following the Intel guide gave 43 of 64 wrong codes); ZMM by default, `STRATA_CPU_YMM` for YMM | Q2_0 experts: unpack 2-bit codes 8 at a time, u8 x s8 dot, float accumulator vector reduced once per row (per-block reduction made it compute-bound at 21 GB/s instead of 32). 6 cores at 42.55 GB/s; scalar 9.89 GB/s | `expert_parity` (VNNI vs oracle 1.461e-6, trap > 1e-4), `expert_multi_test` (bitwise n=1..8) | **NEON `vdotq_s32` (DotProd, all M-series)**: codes 0..3 are valid s8, so `sdot(codes, xq)` plus `sdot(ones, xq)` (or the precomputed `hx`) works. Unpack: `vshrq_n_u8`/`vandq_u8` per 2-bit field, or **better, the GPU kernels' trick of regrouping activations** (`(w >> 2f) & 0x03` per byte gives fields without a transpose). `vpmultishiftqb` has no NEON equivalent. i8mm `vmmlaq_s32` (FEAT_I8MM, M2+) helps multi-token (2x8 by 8x2 tiles) |
| `q2_avx2.cpp` (130) | `unpack64`, `row_multi`, `q2_0_gguf_rows_multi_avx2` :75, `act_quant_q8_1_avx2` :90 | AVX2 + FMA + F16C (`_mm_cvtph_ps`) | same arithmetic for non-AVX-512 CPUs; quantizer bit-identical to scalar | same tests | `vcvt_f32_f16` for fp16; `vdotq_s32` |
| `iq_avx512.cpp` (273) | `Fmt<17/22/18/16/21>`, `row_dot` :156, `gu_rows(_nt)`, `dot_rows(_nt)`, `iq512_gu_rows` :249, `iq512_rows` :261 | AVX-512: masked subtract for signs, `maddubs`, `madd`; optional gather (`STRATA_IQ_GATHER`) | multi-token IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S x Q8_K: decode 64 values once (grid lookups, a 64-bit sign mask, scales), apply to each token. Only float-addition order differs from ggml | `native_expert_parity` | ggml-cpu ARM already has NEON vec_dots for all IQ types (`arch/arm/quants.c:3631-4256`), single-token. A multi-token NEON port would decode with `vqtbl` (grid) and `vmulq_s8`/`vbslq` (signs), then `vdotq_s32` per token |
| `iq_avx2.cpp` (440) | `Fmt32<...>`, `row_dot` :181, `row_dot_iq2xs` :215, `iq256_gu_rows` :363, `iq256_rows` :375, `iq4nl_rows` :393, `iq4nl256_down_rows` :424 | AVX2: `pshufb` sign broadcast plus bit selector, `cmpeq`, `vpsignb`, `maddubs`, `madd` | same, in 256-bit lanes; IQ4_NL multi-token down rows | `native_expert_parity` | NEON: `vqtbl1q_s8` for the IQ4_NL codebook (ggml does exactly this), `vdotq_s32` |
| `kq_avx2.cpp` (265) | `q4k_dot` :85, `q5_1_dot` :148, `q8_0_dot` :172, `dot_rows` :191, `kq256_supported` :202, `bf16_rows_dot_multi` :204, `bf16_rows_dot` :219, `kq256_gu_rows` :239, `kq256_rows` :253 | AVX2 (built with GGML_AVX2=ON, AVX512=OFF in STRATA_PORTABLE) | Q4_K x Q8_K, Q5_1 x Q8_1, Q8_0 x Q8_0 multi-token (MAXT 8), **bit-exact against the ggml-cpu AVX2 branch** for any token count; opt-in `STRATA_KQ256=1` (measured no faster). BF16 router dot | `native_expert_parity` | bit-exactness targets the x86 branch; on ARM the reference becomes ggml-cpu's NEON/i8mm branch (`__ARM_FEATURE_MATMUL_INT8` paths in `ggml_vec_dot_q4_K_q8_K`). BF16: `vbfdotq_f32`/`vbfmmlaq_f32` (FEAT_BF16, M2+) or shift-to-f32 plus `vfmaq_f32` |
| `native_expert.cpp` (152) | `native_fmt`, `native_quant_act`/`_h` (ggml `from_float`), `native_gu_rows`, `native_down_rows` | ggml-cpu type traits | any GGUF expert on the CPU, exactly as llama.cpp's CPU backend computes it; SwiGLU `g/(1+exp(-g))*u`; dispatch: `STRATA_KQ256` -> kq256; `nt >= STRATA_IQ_MT_MIN` (default 2) -> iq512/iq256; IQ4_NL down -> `iq4nl256_down_rows`; else ggml `vec_dot` per token | `native_expert_parity` | **portable as is**: ggml-cpu builds on Apple Silicon with NEON/DotProd/i8mm. Note #152: multi-token and single-token kernels round differently, so greedy output can depend on draft count unless `STRATA_IQ_MT_MIN=1` |
| `expert_layout.cpp` (240) | `cpu_avx512_ok` :24, `cpu_avx2_ok` :56 (CPUID), `q2_rows_any` :111, `act_quant_any` :117, `expert_layout_load` :132 | CPUID dispatch | layout table and ISA dispatch | `tests/core/expert_layout_test` (memcmp) | replace CPUID with `sysctlbyname("hw.optional.arm.FEAT_DotProd" / "FEAT_I8MM" / "FEAT_BF16")` |
| `pool.cpp` (753) | `ExpertPool` worker, run, run_split(_multi)(_native), thread pinning | threads, affinity, spin | parallel expert rows | `pool_test`, `pool_stress` | macOS has no hard affinity (`thread_policy_set` affinity tags are hints only); P-cores vs E-cores need QoS classes (`QOS_CLASS_USER_INTERACTIVE`) |
| `src/kernels/ngram.cpp` :99-165 | PLE row decoders: IQ4_NL (90 B), Q5_0 (110 B), FP8 (160 B, 256-entry LUT, x scale) | scalar/compiler-vectorized | 160 values per token per layer from the 28.8-51.2 GB table | `ple_parity`, `ple_q5_parity`, `ple_fp8_parity` | IQ4_NL: `vqtbl1q_s8` codebook; FP8: `vqtbl4q_u8` on a byte LUT, or a direct `vcvt` path (no hardware FP8 on M-series NEON). Random-row mmap reads favour unified memory |

Other points:
- **Feature gating.** On x86 the build requires AVX2 as a floor (`cpu_avx2_ok`), and AVX-512 is used only when
  F/BW/VL/VNNI/VBMI are all present (`cpu_avx512_ok`; `STRATA_FORCE_AVX2` forces it off).
  `cpu_require_expert_support` exits early with a reason. A Mac build would gate on DotProd (always present on
  M1+) and optionally i8mm/bf16.
- **x86 as the oracle.** "ggml-cpu has no SIMD Q2_0 on x86" (paper 3.6) motivated the VNNI kernel. On ARM, ggml-cpu
  **does** have a NEON Q2_0 dot (`arch/arm/quants.c:222`), so a Mac port can start from ggml-cpu for every format
  and add multi-token NEON kernels only where measured necessary.
- **Bit-exactness on Mac.** The CPU parity references (1.461e-6, `expert_multi_test` bitwise, kq256 bit-exact) were
  all established on x86 summation orders. They must be re-established against NEON summation order. This includes
  `vmlaq_n_f32`, which may or may not contract to `fmla` depending on `-ffp-contract`.

---

## 7. ggml-metal coverage vs Strata needs (pinned commit 3cf03257)

| Need | ggml-metal | Gap / note |
|---|---|---|
| Decode mat-vec, 1 token, every storage type (Q2_0, IQ1_M..IQ4_XS, Q4_0/5_0/5_1/8_0, Q3_K..Q6_K, BF16, F16) | `kernel_mul_mv_<t>_f32` | Covered. **Float activations**, so the contract differs from Strata's Q8_0/Q8_1/Q8_K (0.6-0.95% relative gaps; section 2.3). |
| MoE decode (expert id indirection) | `kernel_mul_mv_id_<t>_f32` | Covered for all types, but llama.cpp's MoE layout (one 3-D tensor per projection) differs from Strata's per-expert blob. Adopting it needs a blob-to-3D view: native blobs are `[gate|up|down]` per expert, not per-projection contiguous. |
| Multi-token verify window (2-8 tokens) | `kernel_mul_mv_ext_<t>_f32_r1_{2..5}` | **Missing for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S.** Present for Q2_0, legacy, K-quants, IQ4_NL, IQ4_XS, BF16, F16, MXFP4. For the missing ones ggml falls back to mul_mm or repeated mul_mv. Strata's grouped decode-once kernels are the better model. |
| Prefill GEMM | `kernel_mul_mm(_id)_<t>_{f16,f32}` (simdgroup_matrix) | Covered for every type including Q2_0 and IQ. |
| Embedding rows | `kernel_get_rows_<t>` | Covered for every type including Q2_0. |
| Activation quantizers Q8_0 / Q8_1 / Q8_K | none | must be written (simple) |
| S-form canonical planes | none | Strata-private; port or drop canonical packs |
| FP8 E4M3 PLE | none (no ggml type) | CPU LUT |
| Bit-exact CPU-order expert path | none | only if CPU parity is required |

---

## 8. Surprises and risks worth flagging

1. **Q2_0 runs under two different activation contracts** inside one engine:
   - Native Q2_0 on the GPU uses Q8_1: fp16 d, `roundf`, `__byte_perm` LUT and **no hx correction**, because the
     -1 bias is folded into the byte LUT.
   - Native Q2_0 on the CPU uses Strata's legacy ActQ: fp32 scale, round half away, hx correction (`pool.cpp:543-563`).
   - Canonical S2 hits use `quantize_q8_0_scaled`, chosen deliberately to match the CPU ActQ.
   - A Mac port that merges CPU and GPU expert execution must pick one of these.
2. **`quantize_q8_0_kernel` divides in double** to match the numpy float64 reference. Metal has no double, so it
   needs an fma-residual tie-fix or a switch to the scaled contract.
3. **ggml-metal does support Q2_0** at the pinned commit, in mul_mv, mul_mv_id, mul_mv_ext, mul_mm(_id), get_rows
   and cpy. It does so with float activations, so it is not a drop-in for Strata's numerics.
4. **`mul_mv_ext` lacks every IQ1/IQ2/IQ3 type.** These are the gate/up formats of the recommended IQ2_XS/IQ3 files,
   which matters for speculative-verify windows.
5. **Down projections can never be 256-superblock formats** (FF=640). This constrains which kernels are needed for
   downs: Q2_0, IQ4_NL, Q4_0, Q5_0, Q5_1, Q8_0.
6. **Q5_0 expert downs** (Orca files) have no GPU kernel. **IQ1_S** has no kernel at all. **IQ1_M** has kernels
   but no shipped expert rows. **Q2_K** is entirely unused.
7. **The PLE table is the largest single tensor** (28.8 GB IQ4_NL; 51.2 GB FP8; Q5_0 for Orca). It is decoded on
   the CPU, 160 values per token per layer, and never goes to the GPU.
8. **The canonical S-form loses the activation type.** It needs `source_type` from the manifest: Q5_0 vs Q5_K and
   IQ4_NL vs IQ4_XS are indistinguishable from S-form fields.
9. **Scale widths vary.** Canonical dense scales are fp16 or f32 per tensor in the pack (90 vs 213 tensors) and are
   widened to f32 at load. The expert arena keeps fp16.
10. **Strata does not trust `__float2half`.** It has custom RNE fp16 and bf16 converters (`f16_bits.hpp`,
    `bf16_bits.hpp`) after observing wrong bits (0x2600 vs 0x26DB) and a NaN-on-overflow bug. A Metal port should
    reuse those host-side and use `as_type` plus explicit RNE in shaders.
11. **Constant memory for divergent codebook lookups cost 2.12x.** The same caution applies to Metal `constant`
    address space: use threadgroup memory or registers.
12. **`s2_gemv_fast.cu` is named fast and is not faster.** `s2_gemv_quads.cu` is unused. Neither should be ported.
13. **Several environment A/B switches change kernels while keeping bitwise results:** `STRATA_OLD_GROUPED`,
    `STRATA_OLD_IQ_MMVQ`, `STRATA_GROUPED_PAIR_MIN_HITS`. Others change numerics: `STRATA_IQ_MT_MIN`,
    `STRATA_KQ256`, `--expert-cache-cpu-order`, `expert_set_oracle_q8_0`. A port should keep only the defaults
    unless parity tests need the others.
14. **Doc drift.** `docs/pack-format.md` and `docs/activation-contract.md` are referenced in code comments but do
    not exist in `docs/`. The `layer.cpp` comment that "Q5_0 and Q5_K are both 8-bit with bias -16" disagrees with
    the authoritative `canonical_xcheck.py` mapping (Q5_K: bias 0 with an offset plane).
