> **Audit report: research input, kept verbatim, except that the appendices are replaced by links to the identical
> audit files K1-K5.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Report A: Qwen3.8-Flash-Next in Strata, the forward pass and every GPU kernel

This report is the implementation-grade inventory for a macOS / Apple Silicon (Metal) port of the Strata
inference engine. Its goal is that the model can be re-implemented in MSL without reading the CUDA sources.

## Scope, sources, conventions

**Upstream.**
- Read-only clone at `/tmp/strata-upstream`.
- Commit `1678de333d0e0711bc414ad992b640e1a37dd814` (2026-10-02, "README, MODELS: the AMD baseline - RX 9070 XT").
- The clone was not modified. `git status` is clean.

**Model.** Qwen3.8-Flash-Next is a 125B-parameter MoE. In the engine's terms:
- 48 layers: 36 Gated DeltaNet (GDN) layers and 12 QSA (sparse attention) layers.
- 512 routed experts with top-10 routing, plus one shared expert per layer.
- 4-stream "GR" hyper-connections.
- A 28.8 GB per-layer n-gram embedding (PLE) table.
- An MTP draft layer.
- M-RoPE for vision.

**What was read.**
- Every `.cu` file in `src/kernels/cuda` (44 files, 205 `__global__`), `src/prefill` (4 files, 37 `__global__`)
  and `src/core` (`device.cu`, `pinned.cu`). That is 50 `.cu` files and 243 `__global__` definitions in total.
- The host orchestration: `src/core/layer.cpp`, `verify.cpp`, `mtp.cpp`, `src/prefill/prefill.cpp`,
  `src/program/generate.cpp`, `expert_cache.cpp` and `expert_source.cpp`.
- The headers in `include/strata/{core,kernels,prefill}`.
- The parity tests and `CMakeLists.txt`.

**Conventions.**
- `file:line` refers to the upstream tree. Paths are relative to `/tmp/strata-upstream` unless absolute.
- "Canonical" means Strata's own pack format and reference kernels (often FP64).
- "Native" means unmodified GGUF/ggml blocks with kernels transcribed from llama.cpp commit `3cf03257`, selected by
  `--native`. **setup.py always passes `--native --prefill auto --spec 4`, so native is production.**
- "Fused" means the GR/GDN fused kernels that are hard-wired on (`layer.cpp:42`).
- Tensor shapes use the GGUF `[in, out]` view unless stated otherwise.
- "Bitwise" means bit-identical fp32 output. "Tolerance" means a numeric bound from a parity test.

**How this document is organised.**

| Part | Content |
|---|---|
| Section 0 | Executive summary for the port |
| Section A | Architecture: constants, tensors, per-layer dataflow, every sub-block |
| Section B | Quantization formats, the activation contract and the ggml-metal map |
| Section C | Unified kernel table (all 50 `.cu` files), the PCIe/VRAM-only list, the do-not-port list and the minimal Phase-1 kernel set |
| Section D | Decode and prefill structure, and everything that assumes discrete VRAM |
| Section E | Test strategy: parity tests, oracle vectors, golden outputs |
| Section F | Metal implementation rules distilled from the above |
| Appendix 1 | GR, router, shared expert, RoPE/YaRN/M-RoPE, cvec, elementwise: per-kernel detail |
| Appendix 2 | QSA, indexer, KV formats, KV streaming, GDN: per-kernel detail |
| Appendix 3 | Quant formats, activation quantizers, mat-vec and expert kernels, blob layouts |
| Appendix 4 | PLE, embedding, head, MTP, sampler, verify kernels |
| Appendix 5 | Decode orchestration (verifier, doorbell, CPU pool), expert cache, prefill, VRAM assumption tables, HIP/warp notes, tests |

Sections 0-F are the integrated reference. The appendices hold the per-kernel detail they cite, with line numbers.
When a Section A-F statement summarises an appendix, the appendix is authoritative for the exact arithmetic.

---

## 0. Executive summary for the port

1. **The production path is narrow.** setup.py runs `--native --prefill auto --spec 4`, with KV `f16` up to
   8192 context and `q8` above it, and `--kv-resident 32768` from 65536 context.
   - Every decoded token goes through the speculative **Verifier**, with windows of T = 1..8 tokens. There is one
     CUDA graph per T.
   - The window is followed by an asynchronous **commit** graph and the **MTP drafter**.
   - Of the 243 `__global__` kernels, only 115 (47%) are in the Phase-1 port set. 106 (44%) are FP64/canonical
     references, alternate variants, diagnostic/HIP-only or PCIe/VRAM-only, and a Metal port can skip them. 12 (5%)
     are needed only if the canonical S-packs are kept, and 10 (4%) are optional features such as the Q4 KV cache
     and the control vector (Section C.3).
2. **The forward pass is unusual in four places**, and each needs custom kernels:
   - **GR hyper-connections.** 4 residual streams, D = 10240, with a 10240→320→10240 mixer per half-layer. The
     write of one half is folded into the read of the next.
   - **QSA.** Block-sparse attention with a 4-head indexer, a cell-weighted radix top-k of width 2051, and split-K
     gathered attention over 4 KV formats.
   - **GDN.** A 48-head delta-rule recurrence with 3.12 MiB of fp32 state per layer, decoded token by token. Even
     prefill is a sequential loop, with no chunked WY form.
   - **PLE.** A host-side n-gram hash and a gather from a 28.8 GB table, then a small GPU block at layer 1 only.
3. **Numerics are pinned to bits.**
   - Many kernels pin rounding with `__fmul_rn`/`__fmaf_rn` and fixed reduction trees, and avoid fast math.
   - The parity suite asserts bitwise equality in many places: decode ≡ verify, multi ≡ single, streamed ≡ resident,
     KV quantizers, sampler tokens.
   - Metal compiles with fast math by default and has no FP64.
   - The port must therefore choose, per pipeline, between "bitwise to CUDA" (safe math, explicit `fma`, xor
     butterflies, integer-RNE bf16) and "tolerance to CUDA". Section F lists the rules.
4. **FP64 appears only in canonical or legacy kernels and in the sampler tail.**
   - The production native and fused path is FP32, except the ≤64-candidate sampler tail.
   - That tail goes to the CPU on Apple. It costs no new synchronisation, because sampled decoding already re-runs
     outside the graph after a sync.
5. **The whole PCIe tier disappears on unified memory.** That is:
   - the VRAM expert cache and its adaptive swaps
   - `pcie_mode` 0/1/2 fetches, pinned/mapped staging and the doorbell spin kernels
   - the CPU expert pool handshake
   - KV streaming, prefill expert streaming and slot lending

   The Phase-1 design (Appendix 5 §4.8) is **GPU-only experts read in place from one no-copy arena, one command
   buffer per verify window**, with a GPU-built expert grouping (the generalised E-6 `resident_plan`). This also
   removes the residency-dependent nondeterminism of the CUDA build.
6. **Weight formats.**
   - The native packs use IQ2/IQ3/IQ4 types, K-quants, Q2_0 (ggml id 42) and the legacy 32/64-block types.
   - The FF dimension is 640, so every down projection (routed and shared) must use a 32/64-block format.
   - ggml-metal at the pinned commit has `mul_mv`, `mul_mv_id`, `mul_mm(_id)` and `get_rows` for every type used,
     including Q2_0.
   - **But ggml-metal multiplies by float activations, while Strata quantizes activations to Q8_1, Q8_0 or Q8_K.**
     The difference is 0.6-0.95% relative.
   - `mul_mv_ext` (2-5 columns) lacks IQ1/IQ2/IQ3, which are exactly the gate/up formats of the recommended packs.
   - Strata's grouped decode-once expert kernels are the better template for verify windows.
7. **Apple limits that bite.**
   - **32 KB threadgroup memory**, exceeded by:
     - `gr_down_multi` (40/80 KB), `gr_down_staged`, `gr_down_v3`
     - `block_topk_reg` (about 32 KB of histograms)
     - `block_scores_tc`, both prompt-attention kernels
     - `sampler_split_merge` / `coupled_merge` (33,792 B)
   - No FP64.
   - No dp4a (int8 dots are exact in fp32 FMA because partial sums stay below 2^24).
   - 16 KiB page alignment for no-copy buffers.
   - `maxBufferLength` limits the size of single arena buffers.
8. **Known gaps in the public tree.**
   - `bench/micro` (the ggml-oracle parity suite) is absent, as are `tests/CMakeLists.txt`, `gguf-py`, the PLE
     capture fixtures and CI.
   - `STRATA_BUILD_TESTS` is off.
   - The 23 `verify_kernels.cu` kernels have no tests.
   - A Metal port must regenerate the oracles from llama.cpp 3cf03257 (Section E).

---

## A. Architecture

### A.0 Constants

| Constant | Value | Source |
|---|---|---|
| `n_embd` N | 2560 | `include/strata/core/layout.hpp:27-55` |
| layers | 48. QSA iff `l % 4 == 3` (L3, L7, ..., L47): 12 QSA, 36 GDN | layout.hpp |
| hyper-connection streams `hc` | 4. Residual D = hc·N = 10240 | layout.hpp |
| GR low rank `hc_lr` | 320 | layout.hpp |
| routed experts / top-k | 512 / 10 | layout.hpp |
| expert FF `n_ff` | 640, for both routed and shared experts | layout.hpp |
| GDN state S | 128 (k-dim = v-dim) | layout.hpp |
| GDN heads | h_k = 16 (q/k), h_v = 48 (v). The q/k head of v-head h is **`h % 16`** | `fused_gdn.cu:30` |
| GDN conv | d_conv 4 (3 history taps + current) over 10240 channels: q [0,2048), k [2048,4096), v [4096,10240) | layout.hpp |
| GDN value_dim | 6144 = 48·128 | layout.hpp |
| QSA heads | 24 query, 2 KV, head_dim 256. GQA: q head h uses KV head **`h / 12`** | `include/strata/kernels/qsa.hpp:94-117` |
| QSA RoPE | NEOX, n_rot 64 (pairs i, i+32 for i < 32), base 1e7, `theta_scale = powf(1e7, -2/64)` | qsa.hpp:125, `native_rope.cu:134` |
| Indexer | 4 query heads × 128, **one shared 128-dim key**; block r = 4 cells; top_k 2048 cells | qsa.hpp, layout.hpp:7-9 |
| Selection width | `min(n_kv, idx_top_k + idx_block - 1) = min(n_kv, 2051)` | qsa.hpp:133-136 |
| Step buffer | int32[4] = {pos, n_kv = pos+1, n_bid = n_kv/4, width} | qsa.hpp:156-162, `qsa.cu:699-714` |
| Attention scale | 1/sqrt(256) = 1/16, applied before the max | `qsa_decode_attn.cu:227` |
| RMS eps | 1e-6 everywhere (GR, QSA, indexer, GDN, PLE) | qsa.hpp:120 |
| Router weight clamp | the sum of the 10 selected probabilities is clamped to ≥ 2^-14 = 6.103515625e-5 | `native_router.cu`, `router_top10.cu` |
| M-RoPE sections | {11, 11, 10}, which reduces exactly to `pair % 3` | `include/strata/kernels/mrope.hpp` |
| YaRN defaults | type None, orig_ctx 262144, beta_fast 32, beta_slow 1. Shipped config: no rope keys, so no scaling | `rope_scaling.hpp:65-74` |
| Vocab | 248320 (`output.weight` ne1) | `native_head.cpp` |
| kFusedGrMaxT / kVerifyMaxT | 8 / 8 tokens per verify window | `fused_gr.hpp`, `verify_kernels.hpp` |
| MTP window | 32768 cells of dense attention | `mtp.hpp:45`, `generate.cpp:352` |
| PLE | 16 heads × 160 = 2560; table 320,001,536 rows; history 9 rows × 10240; EOS 248044 cuts the n-gram context | `include/strata/kernels/ngram.hpp:31-49` |
| Image pad token | 248056 (vision; M-RoPE positions from mtmd) | `tools/vision/strata_vision.cpp` |

### A.1 Tensors, shapes, dtypes (`src/core/layout.cpp`)

Per layer, every layer:

| Tensor | Shape | Dtype in the engine | Notes |
|---|---|---|---|
| `hc_attn_norm.weight`, `hc_ffn_norm.weight` | [10240] | F32, **stored as 1 + w** | layout.cpp:97-98. Do not add 1 again |
| `hc_attn_down.weight`, `hc_ffn_down.weight` | [320][10240] | `Bf16InF32` (bf16 re-rounded, 2 B/elem in the arena) | layout.cpp:53, 56 |
| `hc_attn_up.weight`, `hc_ffn_up.weight` | [10240][320] | bf16 | layout.cpp:54, 57 |
| `hc_attn_inject.weight`, `hc_ffn_inject.weight` | [4][10240] | bf16 | layout.cpp:55, 58 |
| `ffn_gate_inp.weight` (router) | [512][2560] | bf16 | |
| `ffn_gate_inp_shexp.weight` | [2560] | bf16 | scalar shared-expert gate |
| `ffn_gate_shexp`, `ffn_up_shexp` | [640][2560] | quantized. `gate` is Q2_0 on 21 layers and a K-quant elsewhere; `up` is Q2_0 on 13 layers | |
| `ffn_down_shexp` | [2560][640] | 32/64-block legacy formats only (FF = 640) | |
| routed experts gate/up/down | 512 × ([640][2560], [640][2560], [2560][640]) | one blob per expert (Section B.4) | Q2_0 blob 1,382,400 B |

GDN layers (36):

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `attn_qkv` | [2560, 10240] | native: IQ4_XS ×13, Q3_K ×18, Q2_0 ×1, Q4_K ×4 | |
| `attn_gate` (z) | [2560, 6144] | Q3_K ×25, IQ4_XS ×4, Q4_K ×3, Q2_0 ×4 | |
| `ssm_out` | [6144, 2560] | quantized | |
| `ssm_conv1d` | [4, 10240] | F32 | |
| `ssm_alpha`, `ssm_beta` | [2560, 48] | BF16 | |
| `ssm_a`, `ssm_dt.bias` | [48] | F32 | `ssm_a < 0` |
| `ssm_norm` | [128] | F32 | |

QSA layers (12):

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `attn_q` | [2560, 12288] | Q3_K ×6, IQ4_XS ×4, Q2_0 ×2 | per head 512 = 256 q, then 256 output gate |
| `attn_k`, `attn_v` | [2560, 512] | quantized | 2 KV heads × 256 |
| `attn_output` | [6144, 2560] | quantized | |
| `attn_q_norm`, `attn_k_norm` | [256] | F32 | |
| `indexer.q_proj` | [2560, 512] | BF16 | 4 heads × 128 |
| `indexer.k_proj` | [2560, 128] | BF16 | one key head |
| `indexer.q_norm`, `indexer.k_norm` | [128] | F32 | |

Global:

| Tensor | Shape | Dtype | Notes |
|---|---|---|---|
| `token_embd.weight` | [2560, 248320] | a native IQ/K/legacy type, or BF16 via `--embd-gguf` (1.27 GB) | `native_head.cpp:108-188` |
| `output_hc_norm`, `output_hc_down`, `output_hc_up` | as hc_* | F32 / bf16 | head mixer; **no inject**, no separate `output_norm` |
| `output.weight` | [2560, 248320] | native type (a Q6_K head is 521 MB) | `native_head.cpp:19-69` |
| `blk.1.ple_key` | [2560, 10240] | native type 42/18/23/8 or BF16 | PLE, layer 1 only |
| `blk.1.ple_value` | [2560, 2560] | BF16 | |
| `blk.1.ple_norm_{key,query,conv}` | [10240] | F32 | |
| `blk.1.ple_conv1d` | [4 × 10240] | F16 | |
| PLE table | [160, 320,001,536] | IQ4_NL 90 B/row (28.8 GB, default); FP8 E4M3 160 B/row (51.2 GB); Q5_0 110 B/row (mmap only) | separate GGUF shard; CPU-decoded |

MTP (`rt/` directory): `experts.bin` (512 Q2_0 blobs × 1,382,400 B), `dense.bin` (Q8_0 projections, BF16 GR and
router, F32 norms stored as 1 + w), and an optional `draft_vocab.bin` (int32 ids). See A.11.

### A.2 Token lifecycle (decode, production `--native`)

Function map in `src/core/layer.cpp`:

| Function | Line |
|---|---|
| `gdn_layer` | 223 |
| `moe_route` | 355 |
| `moe_shared` | 390 |
| `moe_combine_parts` | 425 |
| `moe_finish` | 442 |
| `layer_verify_compatible` | 448 |
| `moe_layer` | 467 |
| `qsa_layer` | 848 |
| `embed_row` | 1005 |
| `lm_head_mix` | 1060 |
| `lm_head` | 1081 |
| `block_layer_pre` | 1142 |
| `ple_issue_token` / `ple_finish_token` / `ple_stage_token` | 1262 / 1270 / 1281 |
| `block_layer_post` | 1291 |
| `block_layer` | 1316 |

The verify graph (`verify.cpp:387-859`) runs the same stages over T rows. The pseudo-code below is per token.

```
# ---- HOST, before the graph (ple_stage_token, layer.cpp:1262-1284; verify: gather_batch into mapped h_ple_)
rows[16]   = ngram_rows(tok, prev1, prev2)          # u64 multiply/xor hash, ngram.cpp:67-93
emb_ple    = cpu_decode(table rows) -> float[2560]  # IQ4_NL split halves / FP8 LUT / Q5_0; head-slowest
upload emb_ple, tok, step[4], positions             # mapped memory or cudaMemcpyAsync

# ---- DEVICE
e          = embed(tok)                    # NativeEmbed::gather_dev / gather_one, or canonical embedding_gather
R[c][:]    = e  for c in 0..3              # broadcast_streams (verify_kernels.cu:203)
pending    = false                         # a deferred FFN write (block_out, inject2) from the previous layer

for l in 0..47:
    # block_layer_pre (layer.cpp:1142-1260)
    if l == 1 and PLE:                     # layer.cpp:1164-1199, THE call whose absence once failed gate C1
        if pending: gr_write(R, block_out, inject2); pending = false
        ple_block(emb_ple, R, hist)        # R = R + gated + silu(conv), in place
        ple_history_advance(hist, normalized)
    mixed, inject  = fused_gr_read(hc_attn_*, R, apply=pending, bo_prev=block_out, inj_prev=inject2)
    block_out      = QSA(mixed, pos) if l % 4 == 3 else GDN(mixed)
    mixed, inject2 = fused_gr_read(hc_ffn_*, R, apply=true, bo_prev=block_out, inj_prev=inject)  # folds the attn write
    ids[10], w[10] = route(ffn_gate_inp . mixed)        # moe_route: mmvf + native route
    shared         = moe_shared(mixed)                  # g_shared_early = true: runs while the host serves experts
    # block_layer_post (layer.cpp:1291-1314)
    parts[k]       = expert_{ids[k]}(mixed)             # produced by the VRAM/PCIe/CPU tiers on CUDA
    block_out      = combine(parts, w, shared)          # native_moe combine
    if l == 47:  gr_write(R, block_out, inject2); cvec if steered
    elif steered: cvec_apply(write=true, bo=block_out, inj=inject2)   # performs the fold, then the vector
    else:        pending = true                          # the next layer's attn read folds it

mixed  = gr_read(R, output_hc_norm, output_hc_down, output_hc_up, inject=null)   # lm_head_mix, layer.cpp:1060
logits = native_mmvq(output.weight, q8_1(mixed))       # NativeHead::run; Q5_K uses native_q5_k_f32
token  = greedy argmax (in graph) or sampler (outside the graph, Philox(seed, pos))
```

Rules that are easy to get wrong:

1. **Order inside a layer:**
   1. attn read
   2. block
   3. attn write
   4. FFN read
   5. router
   6. shared expert
   7. experts
   8. combine
   9. FFN write

   In fused decode, step 3 lives inside step 4. The FFN write lives inside the next layer's step 1, except at the
   PLE layer 1 and the last layer.
2. **`fused_gr_read` rejects `apply && inj_prev == inject_out`** (`fused_gr.cu:884-899`). The attn read writes
   `inject` and folds with `inject2`. The FFN read writes `inject2` and folds with `inject`. Keep two inject buffers.
3. **The non-fused path reuses `inject`** for both halves and calls `gr_write` after each half
   (`layer.cpp:1233`, `1298-1301`).
4. **PLE placement.** PLE runs once, before layer 1's attn read, after the pending layer-0 FFN write has been
   materialised. The hash and gather are host-side by design. A captured graph would otherwise replay the
   capture-time token's rows forever (comment at `layer.cpp:1143-1158`).
5. **`layer_verify_compatible`** (`layer.cpp:448-460`). The verifier, and so all of production decode, requires:
   - `native_bf16_projections`
   - `g_fused_gr`
   - `g_fused_gdn` + `native_gdn`
   - `g_fast_attn` (and not `native_flash_attn_short`)
   - `g_fast_select`
   - `native_qsa_indexer`

   A Metal port only has to implement this combination.
6. **Debug dumps.** `--dump-layers` writes per layer `2N + 2hc + NH·HD + 5·NKV·HD + 8` floats
   (`dump_stride_floats`). `dump_half` writes block_out at offset 0, the MoE out at N, attn inject at 2N and
   FFN inject at 2N+hc. Use the same layout for Metal golden comparisons (Section E).

### A.3 GR hyper-connections (`include/strata/kernels/gr.hpp:79-95`; Appendix 1 §3)

```
gr_read(R[4][2560], w_norm(1+w), W_down[320][10240], W_up[10240][320], W_inject[4][10240]):
  rs[c]    = 1 / sqrt(mean_d R[c,d]^2 + eps)                      # per stream
  xn[c,d]  = R[c,d] * rs[c] * w_norm[c*N+d]                       # fused computes (R*g)*rs; others (R*rs)*g
  lo[k]    = silu( (W_down[k,:] . xn) / hc )                      # the /hc is INSIDE the silu
  gate[i]  = sigmoid( W_up[i,:] . lo )
  mixed[d] = (1/hc) * sum_c xn[c,d] * gate[c*N+d]
  inject[c]= W_inject[c,:] . xn                                   # omitted for the head
gr_write(R, block_out, inject):  R[c,d] += block_out[d] * 2*sigmoid(inject[c]/hc)
```

**Kernels by path.**

| Path | Read | Write |
|---|---|---|
| Decode (fused) | `fused_gr.cu`: `gr_down_kernel` (41 blocks × 256 threads; the norm plus the 320-row down GEMV plus the inject; 40 KB smem) + `gr_up_kernel` (80 × 256; the up GEMV, gate and stream mean, with the pending write folded in) | folded |
| Verify / MTP | `gr_down_multi` (40/80 KB smem), `gr_up_multi`, `gr_norm_split`, `gr_down_staged`, `gr_norm_multi`. A selftest at verify start picks staged → split → plain, whichever is bitwise equal to single-token | per-token `gr_write` at the last layer |
| Head | native six-kernel `gr_read`: `weighted_rms_norm` → mmvf down → `down_silu` → mmvf up → `pre_gated<false>` → mean | n/a |
| Explicit write | n/a | native `post` (`native_gr_postops.cu:65`) at PLE layer 1 and the last layer; `cvec_kernel(write=true)` when steered |
| Prefill | `gr_norm_rs` + BF16 GEMMs (`hc_down` [T,10240]→[T,320], `gr_silu`, `hc_up`, `hc_inject`) + `gr_mix_r` | `gr_write_norm_rs` (fused write + next norm) |

**Parity facts.**
- Multi ≡ single bitwise.
- v3 (`STRATA_GR_V3=1`) is not bitwise.
- The three write/sigmoid flavours (canonical, native `post`, fused) are not interchangeable bitwise. `cvec(write=true)`
  must use the fused flavour (`__expf` + IEEE division), so that "vector off" ≡ stock.

**dot8 structure.**
- Each 8-element chunk restarts from 0 with eight FMAs (lo half first), then `acc += chunk`.
- Chunk index `j = lane + 32q`, ascending, then an xor butterfly.
- A Metal tiling must preserve this lane-to-chunk mapping to keep multi ≡ single (Appendix 1 §12).

### A.4 GDN layer (`gdn_layer`, `layer.cpp:223-333`; Appendix 2 §2)

```
qkv  = W_qkv x                               [10240]   native_mmvq (Q8_1 activations)
c    = silu(conv4(hist, qkv))                per channel: s = h0*w0 + h1*w1 + h2*w2 + x*w3
q,k  = c[q/k head] * rsqrt(sum c^2 + eps)    eps on the SUM (16 q heads, 16 k heads, 128 each)
v    = c[4096 + h*128 ...]                   48 v heads
gate = softplus(alpha_h + dt_bias_h) * ssm_a_h     softplus: v > 20 ? v : log1p(exp(v))
beta = sigmoid(beta_raw_h);  g = exp(gate)
kv[j]    = sum_i S[i][j] * k[i]                           (old state; k from head h % 16)
delta[j] = (v[j] - g*kv[j]) * beta
S[i][j]  = fmaf(g, S[i][j], k[i]*delta[j])
o[j]     = (sum_i S[i][j] * q[i]) * rsqrt(128)            (the q scale is applied to the output)
y[j]     = o[j] * rsqrt(mean_j o^2 + eps) * ssm_norm[j] * sigmoid(z[j])     z = W_gate x [6144]
out  = W_ssm_out y                                        [2560]
```

**State.**
- `float[128 i][48 h][128 j]` (j fastest), followed by the conv history `float[10240][3]`.
- 3.12 MiB per layer, 112 MiB for 36 layers.

**Decode kernels** (`fused_gdn.cu`):
1. `gdn_conv_l2_kernel`: 80 × 128, conv + SiLU + L2, and slides the history.
2. `gdn_ab_kernel`: one warp per row over 96 rows (alpha, beta), BF16 decode by shift.
3. `gdn_step_norm_kernel`: 48 × (128, 4). 32 state rows per thread in registers. `kv = red0+red1+red2+red3`.
   The output norm is fused.

**Fused-path preconditions.** The fused prologue needs
`g_fused_gdn && native_gdn && native_bf16 && d_conv == 4 && S == 128`.
- The non-native path prescales q by 1/sqrt(S).
- The z GEMV reuses the Q8_1 activation scratch.
- The canonical path builds both Q8_K and Q8_0 images of x, but only for canonical projections.

**Verify.**
- `gdn_conv_l2_multi`, `gdn_ab_multi` and `gdn_step_norm_multi` run with **no state write**.
- The per-window inputs `qkv_L_`/`h_L_`/`gate_L_`/`beta_L_` are kept.
- **Commit** replays the kept tokens: `gdn_conv_commit` plus `gdn_step_norm_multi(n_keep)`. No state snapshot is
  taken (a snapshot would be a 113 MB copy).

**Prefill.**
1. `gdn_gates`.
2. `gdn_conv` (serial when T ≤ 64) or `gdn_conv_tiled` + `gdn_conv_hist`.
3. `gdn_l2`.
4. `gdn_rec_cols_pipe`: 192 blocks, a per-token sequential loop with the same per-element arithmetic as decode.
5. `gdn_out_norm`, which also writes the fp16 `y16` that feeds the out projection.

### A.5 QSA layer, indexer, KV (`qsa_layer`, `layer.cpp:848-991`; Appendix 2 §1, §3)

**Decode, 17 steps.**

| # | Step | Production kernel |
|---|---|---|
| 0 | step fill {pos, pos+1, (pos+1)/4, min(pos+1, 2051)} and the per-head positions | `qsa_step_fill` + uploads |
| 1 | indexer raw key `idx_raw[128] = W_ik · x` (before any norm) | `bf16_gemv_fp32_mmvf` |
| 2 | K, V = [2,256] each | `native_mmvq` |
| 3 | K: RMS(`attn_k_norm`) + NEOX RoPE(64) at `pos_base + pos` | `native_qsa` `norm<256>` + `native_rope` `apply` |
| 4 | optional FWHT, then KV append at cell pos (fp16 / int8-g64 / q4 / k8v4) | `kv_append_step` / `kv_append_q8` / `kv_append_q4` |
| 5 | indexer append (fp16-round raw key into the tail; on block completion mean → RMS → RoPE at the block's first cell) | `native_qsa_indexer` `append` |
| 6 | Q projection [12288] and split of the first 256 of every 512 | `native_mmvq` + 2-D memcpy |
| 7 | q: RMS(`attn_q_norm`) + RoPE | as step 3 |
| 8 | indexer q [4,128]: BF16 GEMV + RMS(`indexer.q_norm`) + RoPE | mmvf + norm + rope |
| 9 | block scores `score[b] = sum_h relu(key_b · q_h)`, b ∈ [0, n_bid] | `block_scores_multi_kernel` |
| 10 | top-k over blocks (cell-weighted) → `width` cell ids, ascending | `block_topk_reg_kernel<33>` (≤ 33,792 blocks), else `block_topk_kernel` |
| 11 | KV streaming only: page in the selected pages | `kv_stream` `resolve` + `copy` (**drop on Apple**) |
| 12 | FWHT(q) if K is rotated | `fwht256_kernel` |
| 13 | split-K gathered attention over 64-cell chunks, then merge | `attn_chunk_kernel<mode>` + `attn_merge_kernel` |
| 14 | FWHT back if V is rotated | `fwht256_kernel` |
| 15 | gate `attn *= sigmoid(q_full[h*512 + 256 + d])` | `native_qsa` `gate` |
| 16 | output projection [6144] → [2560] | `native_mmvq` |

**Selection semantics** (must be exact for id parity; Appendix 2 §1.3).
- Score = ReLU-sum over the 4 heads of raw dots. No softmax, no scale.
- The tail block `b == n_bid` uses the constant `dead` key and gets `+1e9` when `n_kv % 4 != 0`.
- Identity when `n_kv ≤ 2051`.
- Otherwise a 4-pass 8-bit radix select on `order_key(s + 0.0f)`:
  - -0 equals +0. NaN sorts lowest.
  - Block weights are 4 per block, and 0..3 for the tail.
  - Ties at the threshold are emitted in ascending cell order. A threshold block may be emitted partially.
- The ids are ascending, which fixes the PV accumulation order.

**Attention.**
- 12 q heads share each KV head.
- `__expf(s - m_chunk)` with the scale applied before the max.
- Unnormalised `fmaf` PV in ascending cell order.
- The merge computes `w = __expf(m_c - M)` and returns `acc / L` (a division).
- The same kernel serves decode, verify (`grid.z = n`), MTP (dense window of 32768) and the prefill fallback, so
  decode ≡ verify bitwise.

**Prefill.**
- `qsa_block_scores_tc` (3xTF32) and `qsa_prompt_attn_batch` (fp16 hi/lo MMA, exp2 online softmax) are **not bitwise**
  with decode. They are accepted at about 1e-4 relative.

**KV formats** (bytes per cell per layer, K + V, 2 heads).

| Format | Encoding | B | @256K, 12 layers |
|---|---|---|---|
| fp16 | fp16 RNE | 2048 | 6.0 GiB |
| int8 (`--kv q8`) | int8 per 64-dim group, **fp16-rounded scale**, `__float2int_rn(x/sf)` with a precise divide, clamp ±127. Not ggml q8_0 | 1056 | 3.09 GiB |
| q4 | FWHT256 (1/16 on load), then ggml q4_0: `d = mval/-8`, `trunc(fma(x, id, 8.5))` | 576 | 1.69 GiB |
| k8v4 | int8 K (unrotated), FWHT q4 V | 816 | 2.39 GiB |

**Pool layout.** A row is `(page*2 + kvh)*4 + cell%4`, with an identity page table when not streaming.

**Indexer cache.**
- `tail[3][128]` (fp16-rounded raw keys), `dead[128]`, `pooled[(max_cells/4)+2][128]` f32, `block_pos`.
- 128 B/cell/layer, 384 MiB at 256K.
- The pooled key is the fp16-rounded sum `((t0+t1)+t2)+t3`, then `fma(0.25, sum, 0)`, then RMS, then RoPE at
  `pos_base + 4b`. The `dead` key is written only at pos 0, with angle 0.

**Verify and commit.**
- Snapshot the indexer tail before the window.
- Append K/V and the indexer per token.
- Commit restores the tail and replays `append` for the kept tokens (position -1 skips).
- K/V of rejected tokens is simply overwritten later.

### A.6 MoE: router, shared expert, routed experts, combine (Appendix 1 §4-5; Appendix 3 §5)

**Router** (native `route`, `native_router.cu:45`, fast-math).
- `logits[512] = ffn_gate_inp · mixed` via `bf16_gemv_fp32_mmvf`.
- `<<<n_tok, (32, 8)>>>`, where the `y != 0` threads return before a `__syncthreads`, so effectively one warp per
  token.
- Lane l holds experts `l + 32i`.
- Max by xor butterfly, `__expf`, an approximate reciprocal of the warp sum, and NaN → -FLT_MAX.
- 10 rounds of argmax; ties go to the lower id. Taken experts are marked `-INF`.
- `w = selected * (1 / max(sum_selected, 2^-14))`.

Other router forms:
- The canonical FP64 `router_top10` gives the same ids but different weight bits.
- Prefill `route_kernel` uses precise `expf` and a true division.
- **Metal: dispatch exactly 32 threads per token** to mirror the early-return shape.

**Shared expert.**
- `h = silu(gate_sh · x) * (up_sh · x)`, then `y = down_sh · h`.
- `g = sigmoid(w_sg · x)` with a BF16 weight via mmvf, then `__fdividef(1, 1 + __expf(-g))`.
- `shared = g * y`, rounded before the combine.

**Routed expert** e: `y_e = down_e · (silu(gate_e · x) * (up_e · x))`.
- Activations are quantized per the weight-type contract (Section B.2).
- Native kernels: `native_gu` / `native_down` (`iq_kernels.cu`).
- Q2_0 S2: the `gu_grouped` / `down_grouped` kernels (`s2_expert_grouped.cu`).
- Grouped kernels decode each expert's weights once for all window tokens routed to it.

**Combine** (native `combine`, `native_moe.cu:34`): `bo = parts_0*w_0`, then `fma(parts_k, w_k, bo)` for k = 1..9,
then a plain add of `shared`.
- The order is fixed and does not depend on which device produced an expert.
- Prefill instead computes `fma(shared, sigm, s)`, one rounding fewer.

### A.7 RoPE, YaRN, M-RoPE (Appendix 1 §6)

- **Rotation.** NEOX partial rotation on the first 64 dims, pairs (i, i+32): `x' = (a·c - b·s, a·s + b·c)`. The
  remaining dims are copied.
- **Where it applies.** Q (24 × 256), K (2 × 256), the indexer q (4 × 128) and the pooled indexer keys.
- **Angles.** `theta = mrope_pos(pos, pair) * base^(-2·pair/64)`, through `rope_scaled_angle` (YaRN-capable). Three
  implementations:

| Implementation | Where used | Arithmetic |
|---|---|---|
| canonical | non-native path | f64-built host table `[cell][32]` cos/sin |
| native `apply` | decode | fast-math `__powf`/`__cosf`/`__sinf`; rope_parity tolerance 3e-3 |
| prefill `rope_kernel` | prefill | precise `powf`/`cosf`/`sinf` |

- **Decode and prefill already differ in the last bits on CUDA.** `STRATA_ROPE_TABLE=1` makes both use the f64 table.
  - Recommended for Metal: a CPU-built f64 table, uploaded once (shared memory), used by every path. It also fixes
    the large-angle error at 256K positions (about 0.02 rad with fast trig).
- **YaRN.** Implemented (`rope_scaling.hpp:65-193`: ramp, `mscale`, `corr_dim` in double), but the shipped config has no
  rope keys, so production is unscaled.
- **M-RoPE.** `tab[pos*3 + pair%3]` (t, h, w).
  - Positions from mtmd: t = p, h = p + y, w = p + x. Text after an image continues at p + max(nx, ny).
  - With `--vision` the device table holds `max_context + 64` cells, initialised to the identity.

### A.8 Control vectors (cvec, Appendix 1 §7)

- **Loader.** `direction.<l>` tensors; `d = Σ scale·src`.
  - Mode 1 (add, default): `dir = d`, `s = 1`.
  - Mode 0 (project): `dir = d / ||d||`, `s = ||d||`.
- **Kernel.** `cvec_kernel<<<(hc, T), 256>>>`.
  - Mode 1: `R_c += dir`.
  - Mode 0: `R_c -= s·<R_c, dir>·dir`, with an xor-warp then 8-partial reduction.
- **The `write = true` form performs the fused path's pending FFN write first**, using the fused sigmoid flavour.
  It is bitwise equal to the fused write, which keeps "vector loaded but off" ≡ stock (`cvec_parity.cpp:179`).
- Not on the critical path; port it directly.

### A.9 PLE: per-layer n-gram embedding (Appendix 4 §1)

**Host side.**
- Hash:
  - 16 heads: 8 bigram and 8 trigram. Host u64 multiply/xor with `mult = {23703573157769, 20109073645365, 8052911324071}`.
  - `row = (u32)(mixed % vocab[h] + offset[h])`, with per-head prime vocab sizes from 20000003.
  - An EOS (248044) cuts the context.
- Row read: `--ple-io direct` (default, `O_DIRECT` + a 16-thread pread pool), `mmap` or `ram` (mlock).
  - **On macOS, `O_DIRECT` becomes `fcntl(F_NOCACHE)`.**
  - A row cache (1,048,576 rows) dedups pages.
- Decode:
  - IQ4_NL rows are 5 blocks of {f16 d, 16 B}, with **split halves**: element j uses the low nibble, j+16 the high
    nibble. Codebook `{-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113}`.
  - FP8 uses a LUT times the table scale.
  - The output is head-slowest: head h at `[h*160, (h+1)*160)`.

**GPU block** (layer 1 only, before the attn read).

```
key    = gnorm_c(ple_key . emb)                 # 2560 -> 10240, RMS per stream c, gamma norm_key
query  = gnorm_c(R)                             # gamma norm_query
value  = ple_value . bf16(emb)                  # BF16 weight, BF16-rounded activation
s[c]   = (key[c] . query[c]) / sqrt(2560)
gate   = sigmoid(sgn(s) * sqrt(max(|s|, 1e-6)))
gated  = value * gate[c]                        # broadcast to 4 streams
norm   = gnorm_c(gated)                         # gamma norm_conv
conv   = silu(sum_k f16(kW[k + 4ch]) * tap_k)   # taps: hist rows 0, 3, 6 and norm (dilation 3)
R      = R + gated + conv                       # in place
hist   = shift each channel up one row, append norm     # layout hist[row + 9*ch], strided
```

**Two GPU implementations.**

| Implementation | Arithmetic | Use |
|---|---|---|
| native postops (`native_ple_postops.cu`, 7 kernels) | llama.cpp arithmetic, fast-math | use for Metal |
| legacy `ple.cu` (8 kernels) | FP64 gnorm/gate/value | skip |

**Verify.**
- Per-row history snapshots.
- Commit rolls back with `copy_indexed(ple.hist, hist_snap_, commit_ + 1)`.
- The host advances `ple_prev` by n_keep tokens.

### A.10 Embedding, output head, sampler (Appendix 4 §2, §4)

**Embedding.**
- `NativeEmbed::gather_dev` → `embed_rows_kernel` (one 256-value superblock per block), or `gather_one` →
  `dequant_flat_kernel`.
- Canonical S2/S4/S8: `embedding_gather_kernel` with `__fadd_rn(__fmul_rn(code + bias, scale), offset)`. The
  canonical path does not support IQ4_NL.
- On CUDA the table lives in pinned host-mapped memory. On Apple it is one shared buffer.
- `broadcast_streams` then fills the 4 streams.

**Head.**
- `lm_head_mix` (the GR read with `output_hc_*`, no inject), then Q8_1 + `native_mmvq` over 248,320 rows.
- Q5_K uses `native_q5_k_f32` with f32 activations.
- **The MTP requires the native head** (`mtp.cpp:383`).
- Argmax ties go to the lowest index.

**Sampler** (`sampler.cu`, defaults `sampler.hpp:15-28`).
- Defaults: top_k 20 (cap 64), top_p 0.95, min_p 0, temperature 1.
- Chain: penalties → top_k → top_p → min_p → temperature → inverse-CDF pick.
- Philox is a non-standard 10-round variant: constants 0x9E3779B9 / 0xBB67AE85, key (i, 0) per round,
  `u = (c0 >> 8) · 2^-24`.
  - The counter is `pos0 + t`, so a draw depends on the absolute position. Speculative acceptance is therefore exact
    under sampling.
- Inside the graph only greedy runs. Sampled requests re-run after the replay.
- Split path: `sampler_split_part` (61 blocks × 128) + `sampler_split_merge` (33,792 B smem). The ≤64-candidate tail
  uses FP64, which **becomes a CPU tail on Apple**.

### A.11 MTP draft layer (Appendix 4 §3)

**What it is.** One extra layer with its own INT8 K/V (dense attention over the last 32768 cells, no indexer) and
512 resident Q2_0 experts.
- At cell c it consumes the main model's final residual at c and the embedding of the token at c+1.
- It predicts token c+2 through the main head, or through a compact draft-vocabulary head `dhead_` built by
  `gather_rows`.

**Forward** (`record_forward`, `mtp.cpp:413-580`, 22 steps).
1. Embedding: `enorm`, Q8_0 `fc_embedding`.
2. Hidden: `hnorm` over 10240, per-stream Q8_0 `fc_hidden`, `add_streams_broadcast`.
3. Attention:
   - `fused_gr_read_multi` (apply = false).
   - K/V: Q8_0, norm, RoPE, append.
   - Q + gate.
   - `window_ids` (also overwrites `step[3]` with the width), then `qsa_decode_attn_batch`.
   - Gate + o-proj.
4. MoE:
   - `fused_gr_read_multi` (apply = true).
   - Router.
   - Routed experts: `moe_group_resident` → `quantize_q8_0_scaled` → `moe_grouped_s2`.
   - Shared expert, combine, `gr_write`.
5. Head: `gr_read(hyper_connection_mixer)`, then the draft head.
6. Selection: greedy + `row_top_prob` + `map_ids`, or coupled sampling.

**Graphs and drafting.**
- Graphs: `capture_round(T)`, a K/V-only pass over the window's T cells followed by the first full step, and
  `capture_step(j)`.
- One host sync per draft step, to read the probability.
- setup uses `--spec 4 --spec-min-p 0.5 --mtp rt`: at most 3 drafts, stopping when p < 0.5.
- **`row_top_prob` normalises over the draft subset only, so it overestimates** the full-vocabulary probability.
- Acceptance is an exact match against the target's own pick for each row, then `commit(a+1)`.

### A.12 Verify, commit, speculative loop (Appendix 5 §1; Appendix 4 §4)

**Round loop** (`generate.cpp:6128-6232`).
1. Choose T: `S_mtp`, shrunk by `spec_min_p`. The first native window is T = 1.
2. Build the window `[x, d0..]`.
3. `apply_pending` (adaptive swaps).
4. `ver.run(T)`.
5. Accept the leading matches.
6. Adapt every 4 rounds.
7. `commit(a+1)` (async graph).
8. Emit.
9. `mtp.draft`, which overlaps the commit.

**`Verifier::run`** (`verify.cpp:1016-1204`).
- A lazy graph per T.
- Inputs staged in mapped memory.
- Doorbell words reset.
- `cudaGraphLaunch`.
- A host loop of one ring per layer: spin on `seq`, run the CPU expert pool, raise flags A/B/M.
- `cudaStreamSynchronize`.

**Commit graph.**
- GDN `conv_commit` + `step_norm_multi(n_keep)`.
- QSA indexer tail restore + re-append.
- PLE `copy_indexed`.
- The KV cache needs no fix-up.

**Invariants a port must keep.**
- Verify must not mutate GDN state, conv history, the indexer tail or the PLE history.
- Commit replays exactly the kept tokens.
- Decode ≡ verify bitwise is achieved by sharing kernels or transcribing their order.

### A.13 Vision

- `tools/vision/strata_vision.cpp` (205 LOC) uses llama.cpp's mtmd to encode images into embedding rows.
- The prompt carries pad id 248056. The engine replaces those rows with the image rows (prefill `:1103` uploads
  them) and supplies M-RoPE positions.
- Only the M-RoPE table and the row substitution touch this report's kernels. The vision encoder itself is
  mtmd/ggml and can run with ggml-metal unchanged.

---

## B. Quantization formats, the activation contract and the ggml-metal map

Appendix 3 is the per-format and per-kernel authority. This section gives what the port needs to decide and build.
H = n_embd = 2560 and FF = expert n_ff = 640 throughout.

### B.0 Decisions the port has to make

1. **Two storage families coexist.**
   - **Native packs** keep raw GGUF blocks (IQ1_M/IQ2_*/IQ3_*, IQ4_NL/IQ4_XS, K-quants, Q2_0, Q4_0/Q5_0/Q5_1/Q8_0)
     and use kernels transcribed from ggml-cuda `mmvq.cu` / `vecdotq.cuh`. setup.py always runs `--native`, and
     every new model file is native. **This is production.**
   - **Canonical packs** (`tools/strata_pack.py`) re-express GGUF as planar S2/S4/S8/P16/P32 code, scale and offset
     planes, with Strata's own kernels.
   - **The MTP drafter is the exception.** Its routed experts are S2 Q2_0 blobs in `rt/experts.bin`
     (707.8 MB) even under `--native`. So the S2 grouped expert kernels and `quantize_q8_0_scaled` are part of the
     Phase-1 set (Appendix 4 §3, `mtp.cpp`).
2. **Activations are quantized, and the format depends on the weight type and the path** (B.2). On the native GPU
   path every weight type takes ggml-cuda's Q8_1. ggml-metal instead multiplies by float activations. The
   difference is 0.6-0.95% relative per mat-vec, which is far above any bitwise or 1e-5 tolerance gate.
3. **Recommendation.**
   - Phase 1: native GGUF blocks with Q8_1 activations, written as MSL transcriptions of the Strata/ggml-cuda dot
     products. This can be bitwise to CUDA native, because every integer dot is exact in fp32 (B.2.4).
   - MTP: S2 blobs with the `quantize_q8_0_scaled` contract.
   - Canonical S4/S8 dense GEMVs only if canonical packs must be supported.
   - Use ggml-metal `mul_mv` only as a tolerance oracle, and its `mul_mm(_id)` structure as the prefill template.

### B.1 Format table

Legend for the ggml-metal column: mv = `kernel_mul_mv_<t>_f32`, id = `kernel_mul_mv_id_<t>_f32`,
ext = `kernel_mul_mv_ext_<t>_f32_r1_{2..5}`, mm = `kernel_mul_mm(_id)_<t>_{f16,f32}`, rows = `kernel_get_rows_<t>`.
"y" means present at pinned commit 3cf03257. The consumer column lists the production native path first, then
canonical (can.) and prefill (pre.).

| Format (ggml id) | Block: elems / bytes, bpw | Layout and decode | Tensors (shipped files) | Strata GPU consumers | ggml-metal mv/id/ext/mm/rows | Port note |
|---|---|---|---|---|---|---|
| Q2_0 (42) | 64 / 18, 2.25 | `{half d; u8 qs[16]}`; element j at byte j/4, bits (j%4)·2, LSB-first; `w = (code-1)·d`, so {-d, 0, d, 2d} | all experts of the ISTA Q2_0 file; many expert downs of the IQ files; dense `attn_qkv`×1, `attn_gate`×4, `attn_q`×2; `blk.1.ple_key`; MTP experts (as S2 blobs) | `native_q2_0_mmvq` (dense, PLE key); `native_gu`/`native_down` `Fmt<42>` (experts); `dequant_flat`; can.: `s2_gemv_q8`, `s2_expert_grouped`, `dequant_s2`; pre.: MMQ `q2_0`, `strata_q2` repack, `dequant_bf16` `group32<42>` | y/y/y/y/y | One fp16 scale per 64 weights spans **two** 32-element activation blocks. Native folds the -1 into a byte LUT (`__byte_perm`, no hx); S2 uses `dw·dx·(Σc·xq − Σxq)` with an hx correction |
| IQ2_XXS (16) | 256 / 66, 2.06 | 8 × {u16 grid idx, signs+scale}; `iq2xxs_grid[256]` (u64), `ksigns_iq2xs` | gate/up of some IQ-file layers | `iq_mmvq` / `mmvq_multi` (dense via dispatch), `native_gu(_multi)`, `dequant_flat`; pre.: MMQ | y/y/**-**/y/y | codebook in threadgroup memory or registers |
| IQ2_XS (17) | 256 / 74, 2.31 | u16 qs (9-bit grid idx + 7-bit sign idx), 4-bit scales; `iq2xs_grid[512]` | gate/up of the IQ2_XS file | as IQ2_XXS | y/y/**-**/y/y | as above |
| IQ2_S (22) | 256 / 82, 2.56 | qs + qh (10-bit idx), explicit sign bytes; `iq2s_grid[1024]` | Coder gate/up, IQ mixes | as IQ2_XXS | y/y/**-**/y/y | as above |
| IQ3_XXS (18) | 256 / 98, 3.06 | 8-bit grid idx into `iq3xxs_grid[256]` (u32) + packed signs/scales | gate/up of the IQ3_XXS files; PLE key option | as IQ2_XXS | y/y/**-**/y/y | as above |
| IQ3_S (21) | 256 / 110, 3.44 | qs + qh (9-bit idx into `iq3s_grid[512]`), sign bytes, 4-bit scales | gate/up of the IQ3_S file, Coder | as IQ2_XXS | y/y/**-**/y/y | as above |
| IQ1_M (29) | 256 / 56, 1.75 | `iq1s_grid_gpu`, 3-bit scales spread over the block | **no shipped rows** | kernels exist (`iq_mmvq`, `native_gu`); not in the MMQ type list, so prefill falls back to FP16 (`prefill.cpp:463`) | y/y/**-**/y/y | low priority |
| IQ1_S (19) | 256 / 50, 1.56 | | none | **no Strata kernel** | y/y/**-**/y/y | skip |
| IQ4_NL (20) | 32 / 18, 4.5 | `{half d; u8 qs[16]}`, nibbles index `kvalues_iq4nl[16]` (low nibbles = elements 0-15, high = 16-31) | expert downs (IQ files, Orca); `ffn_down_shexp`; **PLE table, 28.8 GB, CPU only** | `native_small_mmvq`, `native_down(_multi)`, `dequant_flat`; pre.: MMQ, `dequant_bf16`; CPU: `ngram.cpp:99-165` | y/y/y/y/y | 16-entry codebook; NEON `vqtbl1q_s8` on the CPU side |
| IQ4_XS (23) | 256 / 136, 4.25 | half d, u16 scales_h, u8 scales_l[4], qs[128]; 6-bit sub-block scales; same codebook | dense `attn_qkv`×13, `attn_gate`×4, `attn_q`×4; Coder gate/up; PLE key option | `native_iq4_xs_mmvq`, `native_gu`, dequant; pre.: MMQ | y/y/y/y/y | in `STRATA_D_FMTS`, but a 256-block format cannot occur at FF = 640 |
| Q3_K (11) | 256 / 110, 3.44 | hmask[32], qs[64], 12 B packed 6-bit scales, half d; 110-B stride is only 2-B aligned | dense `attn_qkv`×18, `attn_gate`×25, `attn_q`×6 | `native_q3_k_mmvq` (`load_int_b2`, `__vsubss4`); pre.: `dequant_bf16` + GEMM | y/y/y/y/y | not in `STRATA_GU_FMTS`, so dense only; emulate `__vsubss4` with `char4` subtraction (no saturation in range) |
| Q4_K (12) | 256 / 144, 4.5 | half d, dmin, 12 B 6-bit scales/mins, qs[128] | some dense; UD-Q4_K_XL and Orca Q4_K_S expert gate/up | `native_q4_k_mmvq`, `native_gu`; `iq_mmvq` (no decode-once split); pre.: MMQ only with `STRATA_MMQ_KQUANTS` | y/y/y/y/y | |
| Q5_K (13) | 256 / 176, 5.5 | Q4_K + qh[32] | some dense; UD layer-2 gate/up; head option (`native_q5_k_f32`) | `native_q5_k_mmvq`, `native_gu` | y/y/y/y/y | |
| Q6_K (14) | 256 / 210, 6.56 | ql[128], qh[64], i8 scales[16], half d | some dense; **head** (Q6_K, 521 MB per pass) | `native_q6_k_mmvq` | y/y/y/y/y | head GEMV over 248,320 rows is bandwidth-bound |
| Q4_0 (2) | 32 / 18, 4.5 | half d, nibbles, `(q-8)·d` | some `ffn_down_shexp` | `native_small_mmvq` | y/y/y/y/y | |
| Q5_0 (6) | 32 / 22, 5.5 | half d, u32 qh, nibbles, `(q-16)·d` | some `ffn_down_shexp`; Orca expert downs; Orca PLE table | `native_small_mmvq`; **no GPU expert-down kernel** (not in `STRATA_D_FMTS`); pre.: MMQ only with `STRATA_ORCA_Q4KS_MMQ` | y/y/y/y/y | Orca Q5_0-down layers run on the CPU in CUDA Strata. On Metal, add Q5_0 to the down list (trivial) |
| Q5_1 (7) | 32 / 24, 6.0 | half d, half m, qh, nibbles, `q·d + m` | UD-Q4_K_XL expert downs; Orca blk.0 `ffn_down_shexp` | `iq_mmvq` (dense, via dispatch), `native_down`; pre.: MMQ with `STRATA_MMQ_KQUANTS` | y/y/y/y/y | the only weight type whose CPU vec_dot_type is Q8_1 |
| Q8_0 (8) | 32 / 34, 8.5 | half d, int8 qs[32] | some `ffn_down_shexp`; UD dense, embd, head, a few expert downs; PLE key option; MTP dense and shared expert | `native_small_mmvq`, `native_gu`/`native_down`; pre.: MMQ | y/y/y/y/y | also an activation format (B.2) |
| BF16 (30) | 1 / 2 | upper 16 bits of fp32 | routers `ffn_gate_inp` (2560→512) and `ffn_gate_inp_shexp`; `hc_*` and `output_hc_*`; indexer `k_proj`/`q_proj`; `ssm_alpha`/`ssm_beta`; `ple_value`; `--embd-gguf` | `bf16_f32_mmvf(_multi)` (F32 x); fused/verify `gdn_ab(_multi)` read alpha/beta directly; can.: `bf16_gemv`; pre.: cuBLAS BF16 GEMM | y/y/y/y/y | decode as `as_type<float>(uint(w) << 16)`; bf16 × bf16 products are exact in fp32 |
| F16 (1) | 1 / 2 | IEEE half | `ple_conv1d`; prefill GEMM operand | `native_ple_postops` conv taps | y/y/y/y/y | |
| F32 (0) | 1 / 4 | | norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | elementwise and fused kernels | y/y/-/y/y | |
| FP8 E4M3 (via I8, 24) | 1 / 1 + one f32 per tensor | 256-entry LUT × `strata.ple.scale` | 51.2 GB PLE table option | CPU only | none in ggml | CPU LUT |
| Q2_K (10), MXFP4 (39), NVFP4 (40) | | | unused | none | Q2_K/MXFP4 y | skip |
| Q8_1 (activation) | 32 / 36 | `{half2 ds = (d, Σx); int8 qs[32]}` | GPU native activations; Q5_1 on the CPU | quantizers B.2.2 | none | write |
| Q8_K (activation) | 256 / 292 | `{float d; int8 qs[256]; int16 bsums[16]}` | canonical S4/S8 dense; CPU IQ/K-quant dots | `quantize_q8_K` | none | write only for canonical or CPU |

**Hard constraint: FF = 640 = 2.5 × 256.** No 256-superblock format can store an n_in = 640 tensor. Every
routed-expert down and every `ffn_down_shexp` is Q2_0, IQ4_NL, Q4_0, Q5_0, Q5_1 or Q8_0. The type lists encode this
(`iq_kernels.cu`):

| List | Types | Role |
|---|---|---|
| `STRATA_GU_FMTS` | 16 17 18 21 22 23 29 42 12 13 8 | grouped expert gate/up |
| `STRATA_D_FMTS` | 20 23 42 7 8 | grouped expert down |
| `STRATA_MMVQ_FMTS` | 16 17 18 20 21 22 23 29 42 12 13 7 8 | `iq_mmvq` |

**Dense dispatch** (`native_mmvq.cu:1477-1493`):
- Types 2, 6, 8 and 20 go to `native_small_mmvq`.
- Types 11, 12, 13, 14, 23 and 42 go to the dedicated kernels.
- Types 7, 16, 17, 18, 21, 22 and 29 go to `iq_mmvq` in `iq_kernels.cu`.
- Constraints: ncols 1..8, a non-null stream, 4-B aligned pointers.

**Canonical S-forms** (Appendix 3 §1.3). Planar `codes | scales | offsets`, with `w = decode(code)·scale[g] + offset[g]`:

| Source | Form | Group | Code bias | Offset | Codebook |
|---|---|---|---|---|---|
| Q2_0 | S2 | 64 | -1 | | |
| Q3_K | S4 | 16 | -4 | | |
| Q4_0 | S4 | 32 | -8 | | |
| Q4_K | S4 | 32 | 0 | f32 plane | |
| Q5_0 | S8 | 32 | -16 | | |
| Q5_K | S8 | 32 | 0 | f32 plane | |
| Q6_K | S8 | 16 | -32 | | |
| Q8_0 | S8 | 32 | -128 | | |
| IQ4_NL / IQ4_XS | S4 | 32 | 0 | | IQ4NL |
| F32 / BF16 / F16 | P32 / BF16 / P16 | | | | |

- The bias is applied in the integer domain. The offset is added to the weight before the activation multiply.
- Scale widths across the 303 quantized dense tensors: 90 fp16, 213 f32, and 54 Q4_K/Q5_K tensors carry an offset plane.
- The S-form loses the source type (Q5_0 vs Q5_K, IQ4_NL vs IQ4_XS look the same). The activation kind therefore
  comes from the manifest's `source_type` (`tools/pack_index.py:84-91`).

### B.2 The activation contract

#### B.2.1 Which activation each path uses

| Weight / path | Kernel(s) | Activation | Quantizer |
|---|---|---|---|
| **Any native type, GPU (dense, head, experts, PLE key)** | `native_mmvq`, `iq_mmvq`, `native_expert_grouped` | **Q8_1** (36 B per 32) | `native_quantize_q8_1` / `quantize_q8_1_rows` |
| **S2 routed experts on the GPU (MTP; canonical cache hits)** | `moe_grouped_s2`, `moe_hit_grouped_s2*` | Q8_0 bytes + **fp32** scales | `quantize_q8_0_scaled` (CPU-identical) |
| S2 canonical dense | `s2_gemv_q8` | Q8_0 | `quantize_q8_0` |
| S4/S8 from a legacy source (Q4_0, Q5_0, Q8_0, IQ4_NL) | `s_gemv_q8_0_split` | Q8_0 | `quantize_q8_0` |
| S4/S8 from a K-quant or IQ4_XS | `s_gemv_q8k_split` | Q8_K (292 B per 256) | `quantize_q8_K` |
| S2 experts on the CPU | `s2_expert_vnni*`, `q2_avx2` | ActQ (int8 + fp32 scale, sum, hx) | `act_quant_q8_1` (legacy) or `quantize_oracle_q8_0` |
| Native IQ/K types on the CPU | `native_expert.cpp` | the type's ggml vec_dot_type (Q8_K / Q8_0 / Q8_1) | ggml-cpu `from_float` |
| Native Q2_0 on the CPU | `pool.cpp:543-563` → `q2_rows_any` | ActQ | `act_quant_any` (Strata legacy, not ggml) |
| BF16 | `bf16_f32_mmvf` (native), `bf16_gemv` (canonical) | F32 x / BF16 x | none |

`layer.cpp:129-131` states the canonical dense rule: `code_bits == 2` takes Q8_0, every other quantized form takes
Q8_K, and legacy sources take Q8_0 through `wants_q8k()`. Choosing the wrong one costs 0.6-1.4% on a GEMV. That
gives plausible output, not broken output, so only a parity test catches it.

**Q2_0 runs under three contracts in one engine:**
- GPU native: Q8_1, with no hx.
- GPU S2: Q8_0-scaled, with hx.
- CPU: ActQ.

A Metal port that moves all experts to the GPU must pick one per pack and use it everywhere.

#### B.2.2 The quantizers, bit by bit

| Quantizer (file:line) | Scale | Rounding | Clamp | Stored |
|---|---|---|---|---|
| `native_quantize_q8_1_kernel` (`native_mmvq.cu:141`), `quantize_q8_1_kernel` (`iq_kernels.cu:979`) | `d = amax/127` | `roundf(x/d)`, half away from zero | implicit | `half2(d, Σx)`, where Σx is the sum of the **float** inputs |
| `quantize_q8_0_scaled_kernel` (`quantize_act.cu:100`) | `s = amax/127.f`, `inv = 1.f/s` | `t = x·inv; (int)(t + copysign(0.5, t))` | ±127 | fp16 d **and** fp32 `scales[]` |
| `quantize_q8_0_kernel` (`quantize_act.cu:47`) | `d32 = amax/127.0f`; quantize by d32, not by the fp16 value | `rint((double)x / (double)d32)`, half to even, **in double** | [-128, 127] | fp16 d (custom RNE); a zero block gives d = 0, q = 0 |
| `quantize_q8_K_kernel` (`quantize_act.cu:161`) | `iscale = -127/max_signed` | magic constant 12582912 (nearest even), `__fmul_rn` | `min(127)` | f32 d = 1/iscale, bsums |
| `cpu_order_quantize_kernel` (`s2_expert_grouped.cu:445`) | as scaled, with `__fdiv_rn` | `__fmul_rn`, `__fadd_rn` | ±127 | fp16 d, fp32 scale, hx = s·Σq |
| ggml-cpu ARM `quantize_row_q8_0` (`ggml-cpu/arch/arm/quants.c:41`) | `d = amax/127`, `id = 1/d` | `vcvtnq_s32_f32` (nearest even) | | fp16 d |

ggml-cpu Q8_0 differs between architectures: x86 multiplies by `127/amax`, ARM by `1/(amax/127)`. Near-tie
elements can therefore differ. A Mac-built ggml oracle will not reproduce the x86 numbers behind Strata's parity
logs.

#### B.2.3 Measured sensitivities (from source comments and tests)

| Quantity | Value | Source |
|---|---|---|
| FP16 vs quantized activation, Q2_0 GEMV | 0.9498% relative | `s2_gemv_q8.cu:3-8` |
| Wrong activation kind (Q8_0 vs Q8_K) | 0.6-1.4% | `layer.cpp` |
| fp32 vs fp16 activation scale | 80 of 80 chunks differ, max relative 4.761e-4 (2^-11); int8 codes identical | `quantize_act.cu:84-92` |
| Quantize by fp16 d instead of fp32 d32 | 18 of 80 blocks differ; worse end-to-end KL | `quantize_act.cu:16-19` |
| Q8_K clamp -128 vs -127 | 0.79% | `quantize_act.cu` |
| Q8_K without `__fmul_rn` | 1 of 524,288 elements differ | `quantize_act.cu` |
| CPU VNNI expert vs scalar oracle | 1.461e-6 relative | `cpu/expert.hpp` |
| `s2_gemv_q8` vs reference | 1.479e-5 (tolerance 1e-4) | `s2_gemv_q8_parity.cpp` |
| Native IQ gate rows vs ggml vec_dot | 1e-7 measured (tolerance 1e-5) | `native_expert_parity.cpp` |
| UD-Q4_K_XL Q8_0 to BF16 for small projections | max abs 0.0144 (`hc_ffn_up`) | `docs/UNSLOTH_Q4.md:88` |

#### B.2.4 Reproducing the contract in MSL

- **Compile these kernels with safe math.** Use `MTLCompileOptions.mathMode = MTLMathModeSafe`, or
  `fastMathEnabled = NO` on older SDKs.
  - Use `fma()` explicitly exactly where CUDA uses `__fmaf_rn`, or where nvcc contracts `a*b+c` in a non-fast-math
    translation unit (the default `--fmad=true`).
  - Prevent contraction elsewhere with `#pragma METAL fp contract(off)`. Its availability must be checked on the
    target MSL version.
- **Division** (`amax/127`, `x/d`) must be correctly rounded. Use `precise::divide`, or fix the quotient with an fma
  residual.
- **No double.** For `quantize_q8_0_kernel`:
  1. Compute `r = rint(x/d32)` in fp32.
  2. Test both half-integer boundaries exactly with `sign(fma(r ± 0.5, d32, -x))`. The sign of an fma result is
     exact.
  3. Adjust r, applying half-to-even at exact ties.

  This reproduces the double reference bit for bit. Alternatively, retire `quantize_q8_0` and use the scaled
  contract.
- **Rounding functions.** `round()` is half away from zero, the same as `roundf`. `rint()` is half to even. Map
  them 1:1.
- **Integer dots.** All S2, Q8 and IQ dots are int8 × int8 with exact sums: S2 |s| ≤ 32·3·128, Q8 products
  ≤ 16,384, block sums ≤ 2^19.
  - `dot(float4(char4), float4(char4))`, or chained fma, reproduces `__dp4a` exactly while every partial stays below
    2^24.
  - Integer `int4` multiply-add is also exact.
  - The float epilogue (`d_w·d_x·sumi`, then the `-m·Σx` terms) must keep CUDA's operation order.
- **fp16 and bf16 conversion.** Strata uses custom RNE converters (`f16_bits.hpp`, `bf16_bits.hpp`). It does so
  after seeing wrong bits from `__float2half` (0x2600 vs 0x26DB) and a NaN-on-overflow bug. For bf16, use
  `as_type<uint>` with integer RNE and the NaN quiet rule (`|64`) in MSL. For fp16, MSL's `half(x)` conversion is
  RNE, but verify subnormals and overflow against `f16_bits.hpp`.

### B.3 Tensor-to-format map per model file

**Dense tensors.** These are identical across the ISTA Q2_0, IQ2_XS and IQ3_XXS files; `iq_pack.py --base`
hard-links `dense.bin`. Per-layer distribution from `layer.cpp:113-121`:

| Tensor | Formats across layers |
|---|---|
| `attn_qkv` (36 GDN layers, 2560→10240) | IQ4_XS ×13, Q3_K ×18, Q2_0 ×1, Q4_K ×4 |
| `attn_gate` (36 GDN layers, 2560→6144) | Q3_K ×25, IQ4_XS ×4, Q4_K ×3, Q2_0 ×4 |
| `attn_q` (12 QSA layers) | Q3_K ×6, IQ4_XS ×4, Q2_0 ×2 |
| `ffn_down_shexp` (n_in 640) | IQ4_NL / Q4_0 / Q5_0 / Q8_0 |
| `ssm_out`, `attn_k`, `attn_v`, `attn_output`, `ffn_gate_shexp`, `ffn_up_shexp` | mixed K-quant / IQ4_XS / Q2_0 |
| `blk.1.ple_key` (2560→10240) | Q2_0 or Q8_0; the native kernel also accepts IQ3_XXS and IQ4_XS |
| `ple_value` (layer 1, 2560→2560) | BF16 |
| routers `ffn_gate_inp` (2560→512), `ffn_gate_inp_shexp` (2560→1) | BF16 (2.6 MB per layer) |
| `hc_{attn,ffn}_{down,up,inject}`, `output_hc_{down,up}` | BF16 |
| `indexer.k_proj` / `indexer.q_proj` | BF16, [2560, 128] / [2560, 512] |
| `ssm_alpha`, `ssm_beta` | BF16, [2560, 48] |
| `ple_conv1d` | F16 |
| norms, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias` | F32 |
| `token_embd` | native type via `embed_rows` (CUDA keeps it in mapped host memory); BF16 with `--embd-gguf` |
| `output.weight` (head, 248,320 rows) | native type via `native_mmvq` (`native_head.cpp`) |
| `per_layer_token_embd.weight` (PLE, [160, 320,001,536]) | IQ4_NL (90 B per row, 28.8 GB), Q5_0 (Orca, 110 B), or FP8 (160 B); CPU decode only |

**Routed experts by model file:**

| Model file | gate/up | down | Notes |
|---|---|---|---|
| ISTA Q2_0 (37.6 GB) | Q2_0 | Q2_0 | 34.0 GB of experts, 0.664 GB read per token. Canonical repacks to the S2 blob; native keeps GGUF rows |
| ISTA IQ2_XS (39.2 GB) | IQ2_XS, plus other IQ types per layer | Q2_0 / IQ4_NL | per layer in `native_experts.txt` |
| ISTA IQ3_XXS (47.0 GB) | IQ3_XXS mix | Q2_0 / IQ4_NL | |
| ISTA IQ3_S (54.8 GB) | IQ3_S mix | Q2_0 / IQ4_NL | |
| Coder "IQ1_M" (256 experts, 29.6 GB) | IQ2_S / IQ3_XXS / IQ3_S / IQ4_XS | IQ4_NL / Q2_0 | no IQ1_M expert rows despite the name |
| Swift 1.5 | as ISTA | | layers 13-47 in shard 2 |
| Orca IQ3_XXS | IQ3_XXS | IQ4_NL, Q5_0 in some layers | no GPU kernel for the Q5_0 down |
| Orca Q4_K_S | Q4_K | Q5_0 / Q5_1 | Q5_0 PLE table (Mmap reader only) |
| Unsloth UD-Q4_K_XL (71.7 GiB experts) | Q4_K (Q5_K in layer 2) | Q5_1 (Q8_0 in layers 2, 4, 30, 46, 47) | dense, embd, head and PLE key are Q8_0, or BF16 via `--compat-bf16` |
| MTP drafter (`rt/`) | S2 Q2_0 blob | S2 Q2_0 blob | 512 × 1,382,400 B; shared expert and dense are Q8_0 via `native_mmvq`; router and hc are BF16 |

### B.4 Expert blob layouts and grouping pipelines

#### B.4.1 S2 blob (canonical Q2_0 and the MTP layer; `include/strata/kernels/cpu/expert.hpp`)

| Offset | Size (B) | Content |
|---|---|---|
| 0 (`O_GU_CODES`) | 819,200 | 1280 interleaved gate/up rows × 640 B of 2-bit codes (row-slot 2r = gate r, 2r+1 = up r) |
| 819,200 (`O_D_CODES`) | 409,600 | 2560 down rows × 160 B |
| 1,228,800 (`O_GU_SCALES`) | 102,400 | 1280 rows × 40 fp16 (one per 64) |
| 1,331,200 (`O_D_SCALES`) | 51,200 | 2560 rows × 10 fp16 |
| **Total** | **1,382,400** | per (layer, expert). A layer is 707,788,800 B; 48 layers are 33,973,862,400 B. Arena offset `layer·512·BLOB + e·BLOB` |

- The interleaving once had a bug: the rows were decoded correctly but written to the wrong output slot, giving a
  worst-case relative error of 2.2e+03 with plausible-looking numbers. `moe_hit_parity` caught it.
- Keep this parity test in the Metal suite.

#### B.4.2 Native blob (`native_expert.hpp`, `iq_kernels.hpp`)

- **Layout.** `[gate rows | up rows | down rows]`, the raw GGUF slices back to back:
  - `gu_row = ggml_row_size(gu_type, 2560)` and `d_row = ggml_row_size(d_type, 640)`
  - `up_off = 640·gu_row`, `down_off = 2·up_off`, `bytes = down_off + 2560·d_row`
- **Per-layer table.** Sizes and types vary per layer, as listed in `native_experts.txt`:
  `layer gu_type d_type offset blob_bytes gate_off up_off down_off [shards]`. Layer starts are 16 KiB aligned.
- **Row bytes:**

  | Format | gate/up row (2560) | down row (640) |
  |---|---|---|
  | Q2_0 | 720 | 180 |
  | IQ2_XXS | 660 | |
  | IQ2_XS | 740 | |
  | IQ2_S | 820 | |
  | IQ3_XXS | 980 | |
  | IQ3_S | 1100 | |
  | IQ4_XS | 1360 | |
  | Q4_K | 1440 | |
  | Q5_K | 1760 | |
  | IQ4_NL / Q4_0 | | 360 |
  | Q5_0 | | 440 |
  | Q5_1 | | 480 |
  | Q8_0 | | 680 |

- **Example blobs:**
  - IQ2_XS + Q2_0: 1,408,000 B.
  - IQ3_XXS + IQ4_NL: 2,176,000 B.
  - Q4_K + Q5_1: 3,072,000 B.
- **Activation sizes per token.** Q8_1 is 80 × 36 = 2880 B for gate/up input and 20 × 36 = 720 B for the down
  input.

#### B.4.3 The expert pipelines in the CUDA build

| Pipeline | Used by | Steps |
|---|---|---|
| **Native grouped** (`native_expert_grouped`, `iq_kernels.cu`) | production decode and verify windows (T = 1..8) for VRAM-cache hits and PCIe-mapped blobs | (1) The plan comes from the CPU (`GpuPlanSink`) or the GPU (`resident_plan_kernel`, when every routed id is resident): `grp_ptr[g]`, `grp_start[g]`, `ent_tok[e]`, `ent_dst[e]`, n_groups. (2) `quantize_q8_1` of the T input rows. (3) `native_gu(_multi)` on (`ceil(1280/8)`, cap_groups) × 256: rows < 640 are gate, the rest up; GRP_NC = 4 entries decoded once. (4) `swiglu_entries`. (5) `quantize_q8_1` of h. (6) `native_down(_multi)` on (320, cap_groups), writing `out[ent_dst[e]·2560 + r]`. (7) Combine (B.4.4) |
| **S2 grouped** (`moe_grouped_s2`) | MTP (always); canonical verify windows | (1) `group_resident_kernel` `<<<1,128>>>`: distinct experts in first-appearance order, entries in routing order, O(n²) scan in shared memory, blob at `base + id·BLOB`. (2) `quantize_q8_0_scaled`. (3) `gu_grouped_t` on (40, cap_groups) × 256: 32 rows per block, GMAX = 8 entries staged as aligned words in smem `int[8][640] + float[8][80]` = 23,040 B. (4) S2 `swiglu`. (5) `quantize_q8_0_scaled` of h. (6) `down_grouped_t` on (40, cap_groups), 64 rows per block. Every entry is bitwise equal to the per-hit kernel; the `_t` variants are bitwise equal to the non-`_t` ones |
| **S2 per-hit** (`moe_hit_grouped_s2_dev`) | canonical decode with a VRAM expert cache | `hit_select` (VRAM slot per expert or -1) → `gu`/`gu_pair` over n_hits × 1280 rows → `swiglu` → `quantize_q8_0_scaled` → `down`/`down_pair` → `moe_hit_add` |
| **CPU pool** (`pool.cpp`) | cache misses, the `--pcie-frac` remainder | row-range split over cores; `s2_expert_vnni*`, `iq512/iq256`, ggml vec_dot; results land in mapped `y_miss`, which the GPU adds with `add_kernel` |
| **Prefill MMQ** | prompt chunks | host counting sort after a sync → `copy16`/`copy1` or `strata_q2` gathers into 16-expert group buffers → ggml `mul_mat_q` (gate/up) → `mmq::swiglu` → `quantize_mmq_q8_1` → `mul_mat_q` (down); FP16 cuBLAS fallback (Section D.3) |

#### B.4.4 Combine and order

- `native_moe` `combine` computes `out = Σ_k w_k·y_k + σ(sg)·shared`.
  - The first term is a plain product; the following terms go through an fma chain in k order.
  - The shared expert is gated by a scalar sigmoid of `x·w_gate_shexp`.
- Prefill `moe_combine_kernel` computes `bo = Σ_k w·Dm[slot] + shared·σ(sg)`.

#### B.4.5 The Apple form: one pipeline

- **Address every expert in place.** Use `arena_buffer.gpuAddress + layer_offset + id·blob_bytes(layer)` on
  `newBufferWithBytesNoCopy` buffers over the mmapped pack. Use one buffer per layer, 16 KiB aligned, which keeps
  each buffer under `maxBufferLength`.
- **Build the plan on the GPU.** `resident_plan` / `group_resident` become the only planners, because the resident
  case is always true.
  - With T ≤ 8 tokens × 10 picks there are at most 80 entries.
  - A one-threadgroup, 128-thread first-appearance scan is enough. It is deterministic, unlike CUDA's
    residency-dependent CPU/GPU split.
- **Kernels.** Then `quantize_q8_1` → grouped gate/up (decode-once over the group's entries) → SwiGLU →
  `quantize_q8_1` → grouped down → `combine`.
  - The S2 variant (MTP) swaps in `quantize_q8_0_scaled` and the S2 dot.
- **Retune for Apple register files.** GMAX = 8 and GRP_NC = 4 are register-pressure choices for NVIDIA. The S2
  smem stage (23,040 B) fits in 32 KB.
- **Delete the rest.** `hit_select*`, `add_hits`, the `cpu_order_*` family, `fetch_blobs`/`rebase_ptrs` and the
  `--pcie-frac` split all go (C.2).

### B.5 ggml-metal coverage vs Strata needs (pinned 3cf03257)

| Need | ggml-metal | Gap / port action |
|---|---|---|
| Decode mat-vec, one token, every storage type in use (Q2_0, IQ1_M..IQ4_XS, legacy, K-quants, BF16, F16) | `kernel_mul_mv_<t>_f32` | Covered, but with **float activations**: not Strata's contract (0.6-0.95%). Use as an oracle or as a starting template; write Q8_1-activation kernels for parity |
| MoE decode with expert ids | `kernel_mul_mv_id_<t>_f32` | Covered for all types. llama.cpp's MoE layout is one 3-D tensor per projection; Strata's native blob is `[gate|up|down]` per expert. Needs a blob-to-3D view (per-projection stride `blob_bytes`, offsets `0 / up_off / down_off`) or Strata's own grouped kernels |
| Multi-token verify window (2-8 tokens) | `kernel_mul_mv_ext_<t>_f32_r1_{2..5}` | **Missing for IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ3_S**, which are the gate/up types of the recommended IQ files. Present for Q2_0, legacy, K-quants, IQ4_NL, IQ4_XS, BF16 and F16. Strata's decode-once grouped kernels are the better model |
| Prefill GEMM | `kernel_mul_mm(_id)_<t>_{f16,f32}` (`simdgroup_matrix`) | Covered for every type including Q2_0 and IQ. This is the template for replacing both MMQ (int8 MMA with Q8_1 activations) and dequant + cuBLAS. Numerics change versus CUDA prefill; gate with KL |
| Embedding rows | `kernel_get_rows_<t>` | Covered for every type |
| Activation quantizers Q8_0 / Q8_1 / Q8_K | none (ggml-metal does not quantize src1) | write (small) |
| S-form canonical planes | none | Strata-private: port `s2_gemv_q8` / `s_gemv_q8_split` or drop canonical packs |
| S2 blob experts (MTP) | none | port `gu/down_grouped_t` |
| FP8 E4M3 PLE | none | CPU LUT (already CPU-side) |
| Bit-exact CPU-order expert path | none | not needed without a CPU/GPU split |
| GDN recurrence | `kernel_gated_delta_net`: **unsure** whether it exists at 3cf03257 | write the fused kernels (Appendix 2) |
| RMS norm, RoPE NEOX/multi, BF16 mat-vec, get/set rows, ssm_conv | present | usable as templates; Strata's reduction orders differ |

---

## C. Kernel inventory: all 50 `.cu` files and 243 `__global__` kernels

### C.0 Classification

Every `__global__` kernel carries one class code:

| Code | Meaning | Count | Share |
|---|---|---|---|
| **P** | Phase-1 port: on the production path (`--native`, fused, verify, MTP, prefill auto, sampler) | **115** | 47.3% |
| C | needed only if canonical S-packs are supported | 12 | 4.9% |
| R | canonical / FP64 reference, or the non-native fallback; useful as a CPU-side oracle | 42 | 17.3% |
| V | exists only for the PCIe / discrete-VRAM / mapped-memory / CPU-split design | 28 | 11.5% |
| A | alternate variant of a ported kernel (bitwise twin, env A/B, sm80/sm75 tuning) | 18 | 7.4% |
| D | dead, diagnostic, HIP-only or test helper | 18 | 7.4% |
| O | optional feature (Q4/K8V4 KV, control vectors, coupled draft sampling, FP16 prefill fallback) | 10 | 4.1% |
| **Total** | | **243** | |

R + A + D + V = 106 kernels (43.6%) can be skipped outright. With C and O, 128 (52.7%) are outside the Phase-1 set.

| Area | Files | Kernels | P | C | R | V | A | D | O |
|---|---|---|---|---|---|---|---|---|---|
| core (`device.cu`) | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 | 0 |
| GR / router / shared / RoPE / cvec / elementwise | 12 | 49 | 19 | 0 | 17 | 7 | 3 | 2 | 1 |
| Quant / mat-vec / experts | 13 | 57 | 27 | 9 | 5 | 7 | 2 | 6 | 1 |
| GDN / QSA / KV | 15 | 53 | 19 | 0 | 13 | 4 | 8 | 5 | 4 |
| PLE / sampler / verify | 4 | 46 | 25 | 1 | 7 | 7 | 0 | 4 | 2 |
| Prefill | 2 | 37 | 25 | 2 | 0 | 3 | 5 | 0 | 2 |
| **Total** (47 files with kernels; plus `pinned.cu`, `prefill/gemm.cu` and `prefill/ggml_cuda_host.cu` with none) | 47 | 243 | 115 | 12 | 42 | 28 | 18 | 18 | 10 |

**Not counted.** The vendored ggml-cuda MMQ kernels (`mul_mat_q`, the stream-k fixup, `quantize_mmq_q8_1`) are
compiled from `third_party/ggml` into `strata_mmq` and are not counted. On Metal they are replaced by the indexed
GEMM (D.3).

**Fast-math split** (`CMakeLists.txt:288-308`):
- `--use_fast_math`: only the `native_*.cu` files and `strata_mmq`.
- Every other file: nvcc defaults, i.e. FMA contraction on, IEEE div/sqrt, no FTZ.
- HIP uses `-ffp-contract=off` for embedding/quantize.

The Metal library split should follow this: a fast-math library for the `native_*` transcriptions (with tolerance
tests), and a safe-math library for the rest.

### C.1 Unified per-file table

The kernel lists are in source order, with the class in brackets. LOC is the file's line count. "absent" means the
test is registered in `CMakeLists.txt` but its source (`bench/micro/*_parity.cpp`, `CMakeLists.txt:311-334`) is not in
the public clone.

#### C.1.1 GR, router, shared expert, RoPE, cvec, elementwise (Appendix 1)

`native_bf16.cu` appears here because it serves GR, but the C.0 area counts include it under quant/mat-vec.

| File (LOC) | Kernels [class] | Purpose and I/O | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|
| `kernels/cuda/fused_gr.cu` (1081) | gr_down [P], gr_up [P], gr_norm_multi [P], gr_down_multi [P], gr_up_multi [P], gr_down_v3 [A], gr_up_v3 [A], gr_norm_split [P], gr_down_staged [A] | Decode GR read in two kernels. K7: weighted RMS norm of the 4 streams (R [4×2560] f32) fused with the BF16 `hc_down` 10240→320 and `hc_inject` 10240→4. K8: SiLU(lo/hc), `hc_up` 320→10240 and the pre-gated stream mix → `mixed[2560]`. `_multi` variants take a T ≤ 8 verify window. `v3` is the non-bitwise `STRATA_GR_V3` read; `staged` (cp.async double buffer) is the CUDA default for verify | 40 KB static smem (`xn[10240]`); 40-80 KB dynamic tiles at T = 8; `cp.async.cg`; uint4 `__ldg`; `__expf`; xor butterflies; `gpu_stamp` | `gr_parity` :351-404 (multi bitwise = single); `fused_gr_selftest` (runtime choice staged → split → plain, `fused_gr.cu:1036-1044`) | **Re-tile to 32 KB.** Have the norm write xn to device memory (bitwise, as `gr_norm_split`/`gr_norm_multi` already do) and let down read xn tiles. Keep the `lane + 32q` chunk order and dot8 structure. Do not port v3 or staged |
| `kernels/cuda/gr.cu` (432) | gr_norm, gr_down, gr_gate, gr_mean, gr_inject, gr_write [all R] | Canonical BF16-activation GR read/write. The host dispatchers `gr_read` (:344) and `gr_write` (:413) call the **native** kernels under `--native` (`native_gr_rms_norm`, mmvf, `native_gr_down_silu`, `native_gr_pre_gated`, `native_gr_post`). The head mix and the MTP final mixer go through these dispatchers | shfl_down + broadcast, smem, precise `expf` | `gr_parity` (770 LOC) | Port the dispatcher logic (native branch, including the `w_inject == nullptr` final-mixer contract); the kernels are references |
| `kernels/cuda/native_gr_norm.cu` (102) | weighted_rms_norm [P] | ggml `rms_norm` × weight per stream | 1024-thread blocks, xor; fast-math | `gr_parity` mode 2; `native_gr_norm_parity` (absent) | direct |
| `kernels/cuda/native_gr_postops.cu` (115) | down_silu [P], pre_gated [P], post [P] | The ggml scale/silu/sigmoid/mul/add sequence of the unfused native GR: SiLU(lo/hc); the pre-gated mix (`pre_gated<false>` for the final mixer); `post` = HC write | `_rn` intrinsics; fast-math | `gr_parity` mode 2; `native_gr_postops_parity` (absent) | elementwise; contract off |
| `kernels/cuda/native_bf16.cu` (186) | bf16_f32_mmvf [P], bf16_f32_mmvf_multi [P] | F32 x × BF16 W GEMV (llama.cpp mmvf): routers, `hc_*`, indexer q/k, `ple_value`, the MTP router. Multi = up to 8 tokens, bit-identical to single | pair loads, two ordered `__fmaf_rn`, xor warp sum, `partials[32]`, BS 32..256 by iteration count | `gr_parity` mode 2; `native_bf16_parity` (absent) | simdgroup GEMV; needs n_in even and x 8-B aligned. `kernel_mul_mv_bf16_f32` as oracle |
| `kernels/cuda/native_router.cu` (127) | route [P] | ggml topk-moe: softmax over 512, top-10 (lowest id on ties), renormalise with a 2^-14 clamp on the sum | xor, `__expf`, early return before the barrier; fast-math | `native_router_parity` (absent) | **dispatch exactly 32 threads per token** (the early-return pattern is only safe in one simdgroup) |
| `kernels/cuda/router_top10.cu` (401) | router_top10 [R], router_top10_fast [D] | canonical FP64 router; the fast variant is HIP | FP64, shfl_down | `router_top10_parity` (167); `tests/hip/router_fast.cpp` | no FP64; do not port |
| `kernels/cuda/shared_expert.cu` (346) | swiglu [R], native_swiglu [P], to_f16 [D], scalar_gate [R], scale [P], native_scalar_sigmoid [P], native_scalar_sigmoid_multi [P], moe_combine [R], scale_rows [P] | Shared-expert host (`shared_expert`, :209): gate/up via `native_mmvq`, SwiGLU, Q8_1, down; scalar gate σ(x·w_g) via mmvf; canonical double forms | FP64 (canonical); `__fdividef`/`__expf` | `shared_expert_parity` (482) | port the native forms |
| `kernels/cuda/native_moe.cu` (86) | combine [P] | ggml-order MoE combine (B.4.4) | fma; fast-math | `shared_expert_parity`; `native_moe_parity` (absent) | elementwise; keep the k order and the plain first product |
| `kernels/cuda/rope.cu` (136) | rope_neox [R] | NEOX RoPE from a host-built f64 table | none | `rope_parity` (442) | the semantics are reused: **build the table on the CPU in f64 and upload it**; this is the recommended Metal path |
| `kernels/cuda/native_rope.cu` (147) | apply [P] | Analytic NEOX RoPE over the first 64 dims; M-RoPE sections {11, 11, 10}; YaRN. With `STRATA_ROPE_TABLE=1` it reads the table and is bitwise equal to `rope_neox` | fast `powf`/`cosf`/`sinf`; atomic host table pointer | `rope_parity` (3e-3 tolerance; 1.4e-3 rad error at 32K) | prefer the table; fast-math trig loses accuracy at 2.6e5 rad |
| `kernels/cuda/cvec.cu` (166) | cvec [O] | control vectors at the HC write, plus the fused-write fold | `__expf`, xor | `cvec_parity` (223) | optional |
| `kernels/cuda/elementwise.cu` (325) | embedding_gather [R], gdn_gate [R], scale [R], add [V], to_f16 [R], to_bf16 [P], silu [R], rms_norm_weighted [R], doorbell_ring [V], doorbell_wait [V], copy_from_mapped [V], copy_rows_from_mapped [V], doorbell_publish [V], copy_i32_from_mapped [V] | Small ops. `embedding_gather` = canonical code-plane rows. `add` sums the CPU experts' `y_miss` into the layer output (`expert_source.cpp:1913`). The doorbells and mapped copies feed captured graphs from mapped pinned memory | volatile, `__threadfence_system`, spin + `__nanosleep` | `elementwise_parity` (475) | Keep `to_bf16` (integer RNE, NaN `|64`). Doorbells become `MTLSharedEvent` between command buffers; mapped copies become shared buffers |

#### C.1.2 Quantization, mat-vec and routed experts (Appendix 3)

| File (LOC) | Kernels [class] | Purpose and I/O | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|
| `kernels/cuda/native_mmvq.cu` (1495) | native_quantize_q8_1 [P], native_q5_k_mmvq [P], native_q2_0_mmvq [P], native_q3_k_mmvq [P], native_iq4_xs_mmvq [P], native_q4_k_mmvq [P], native_q6_k_mmvq [P], native_small_mmvq [P] (Q4_0/Q5_0/Q8_0/IQ4_NL), native_mmvq_multi [P] (ncols 2..8) | Dense projections (`attn_qkv`, `attn_gate`, `attn_q/k/v`, `attn_output`, `ssm_out`, shared gate/up/down), the head (248,320 rows) and the PLE key, all with Q8_1 activations. Block (32, 4); SmallK uses 4 rows per block | dp4a, `__byte_perm`, `__vsubss4`, `__launch_bounds__(…,1)`, smem partials | `iq_parity`; `mmvq_multi_parity` (bitwise); `native_dense_ple_key_test`; `native_mmvq_parity` (absent) | transcribe; char4 dots; **keep exact multi mode** (`g_multi_exact`: every column bitwise equal to a 1-column call) |
| `kernels/cuda/iq_kernels.cu` (1474) | mmvq [P], mmvq_multi [P], native_gu [P], native_gu_multi [P], swiglu_entries [P], native_down [P], native_down_multi [P], quantize_q8_1 [P], dequant_flat [P], dequant_gu [O], embed_rows [P] | IQ-type and Q5_1 mat-vec; grouped native routed experts (B.4.3); the Q8_1 quantizer; dequant to f16/f32 (prefill FP16 fallback, single-row embedding); `embed_rows` = token embedding gather (the table may be mapped host memory) | dp4a, `__popc`, `__byte_perm`, codebooks in device memory, `__launch_bounds__(256)` | `iq_parity` (fixtures, skip code 3); `iq_multi_parity` (bitwise); `native_expert_parity` (≤1e-5, 1e-7 measured) | Port all but `dequant_gu`. Put codebooks in threadgroup memory or registers, **not `constant`** (divergent constant lookups cost 2.12× on CUDA). Retune GRP_NC for the Apple register file |
| `kernels/cuda/s2_expert_grouped.cu` (1190) | gu [C], swiglu [P], down [C], gu_pair [C], down_pair [C], activation_correction [V], cpu_order_projection [V], cpu_order_swiglu [V], cpu_order_quantize [V], hit_select [V], hit_select_multi [V], add_hits [V], gu_grouped [A], down_grouped [A], gu_grouped_t [P], down_grouped_t [P], group_resident [P] | S2 routed experts. Per-hit kernels (VRAM cache); grouped kernels (verify, MTP); a CPU-order replay of the AVX-512 summation for CPU/GPU split consistency; device grouping | dp4a, `__byte_perm`, `__f*_rn`, width-8 shuffles, 23,040 B smem, env A/B (`STRATA_OLD_GROUPED`, `STRATA_GROUPED_PAIR_MIN_HITS`) | `s2_expert_grouped_parity` (≤4e-5·mag vs CPU; new vs old bitwise; mapped blob); `moe_hit_parity` | Port `group_resident`, `gu_grouped_t`, `down_grouped_t`, `swiglu` for MTP. Per-hit kernels only for canonical packs. 34-B Q8_0 blocks are unaligned: restage them to aligned words, as the `_t` kernels do |
| `kernels/cuda/quantize_act.cu` (313) | quantize_q8_0 [C], quantize_q8_0_scaled [P], dequant_q8_0 [D], quantize_q8_K [C], dequant_q8_K [D] | activation quantizers (B.2.2) and test dequantizers | double division, `rint`, `__fmul_rn`, custom f16 | `quantize_act_parity` (bit-exact including ties) | scaled: direct port. `q8_0`: fma-residual tie emulation. `q8_K`: safe math |
| `kernels/cuda/bf16_gemv.cu` (141) | bf16_gemv_naive [D], bf16_gemv_warp [R], bf16_gemv_split [R] | BF16 x × BF16 W GEMV (canonical `layer.cpp:98-99`); naive measured at 4.66 ms per token | shfl_down | `bf16_gemv_parity` (1e-5) | not needed; native uses mmvf |
| `kernels/cuda/dequant_bf16.cu` (261) | dequant [P] | GGUF rows → bf16/f16/f32 for 11 types (2, 6, 7, 8, 20, 11, 12, 13, 14, 23, 42); `dequant_f16` feeds the prefill dense GEMM (`gemm.cu:423`) | `__constant__` IQ4_NL codebook | `dequant_bf16_test` (F32 1e-6, BF16 half-ulp); `q5_projection_parity`; `ple_q5_parity` | P for prefill, unless folded into a quantized `simdgroup_matrix` GEMM (recommended) |
| `kernels/cuda/dequant_s2.cu` (51) | dequant_s2 [C] | S2 → f32/f16, `(code-1)·d` | synchronous | `dequant_s2_parity` (memcmp) | trivial |
| `kernels/cuda/s2_gemv.cu` (63) | s2_gemv [R] | naive S2 × FP16 | | `s2_gemv_parity` (1e-5) | reference only |
| `kernels/cuda/s2_gemv_fast.cu` (153) | s2_gemv_fast [D] | smem x + constant LUT; measured not faster | `__constant__`, `cudaMemcpyToSymbol` | `s2_gemv_parity` | skip |
| `kernels/cuda/s2_gemv_q8.cu` (96) | s2_gemv_q8 [C] | production canonical S2 dense × Q8_0, float math, no dp4a | tree reduction | `s2_gemv_q8_parity` (1e-4; 1.479e-5 measured) | canonical only |
| `kernels/cuda/s2_gemv_quads.cu` (93) | s2_gemv_quads [D] | experiment | uint2, half2 | `s2_gemv_parity` | skip |
| `kernels/cuda/s_gemv.cu` (694) | s_gemv_q8k [R], s_gemv_q8_split [C], s_gemv [R], s_gemv_split [D] | canonical S4/S8 dense GEMV (Q8_K or Q8_0 activation; 8 rows per 256-thread block); `s_gemv_split` serves the `concurrent_main`/`overlap_main` tools | smem codebook, shuffles, shift instead of 64-bit divide | `s_gemv_parity` (1e-4); `s_gemv_q8k_parity` (1e-5) | canonical only; keep the `((a0+a1)+(a2+a3))` tree |

#### C.1.3 GDN, QSA, indexer, KV (Appendix 2)

| File (LOC) | Kernels [class] | Purpose and I/O | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|
| `kernels/cuda/fused_gdn.cu` (166) | gdn_step_norm [P], gdn_conv_l2 [P], gdn_ab [P] | Decode-default GDN. `conv_l2`: 4-tap causal conv + SiLU + L2 norm of the q/k heads over 10240 channels. `ab`: BF16 alpha/beta GEMV → `beta = σ`, `gate = softplus(a + dt)·ssm_a`. `step_norm`: delta-rule update of 48 × 128 × 128 f32 state (3.12 MiB per layer) + gated RMSNorm | `__launch_bounds__(512)`, state in registers, `__expf`, `rsqrtf`; not fast-math | `native_gdn_pipeline_parity` (absent); `gdn_parity` checks the canonical semantics | Custom, 512 threads. Keep `red0+red1+red2+red3`, the `h % 16` head pairing, the q scale on the output, and eps on the sum of squares (L2) vs on the mean (out-norm) |
| `kernels/cuda/gdn.cu` (248) | gdn_step, gdn_conv, gdn_l2, gdn_out_norm, gdn_beta_gate [all R] | canonical GDN (double in l2/out-norm) | FP64 | `gdn_parity` (1e-5 / 1e-6) | CPU reference |
| `kernels/cuda/native_gdn.cu` (122) | step [A] | llama.cpp `gated_delta_net` port (`--native` without fused) | fast-math | `native_gdn_parity` (absent) | skip |
| `kernels/cuda/native_gdn_preprocess.cu` (199) | conv_silu, l2_norm, beta_sigmoid, gate_softplus, out_norm [all A] | non-fused native GDN preprocessing | fast-math | `native_gdn_preprocess_parity` (absent) | skip |
| `kernels/cuda/native_qsa.cu` (127) | norm [P], gate [P] | QSA q/k RMS norm (`norm<256>`); output gate `attn·σ(gate)` in fp32 | fast-math | `native_qsa_parity` (absent) | rms_norm-like + elementwise |
| `kernels/cuda/native_qsa_indexer.cu` (290) | append [P], append_first [P], append_blocks [P], append_tail [P] | Indexer keys: fp16-rounded, pooled per 4-cell block (`((t0+t1)+t2)+t3`, then `fma(0.25, sum, 0)`), RoPE at the block's first cell; single-token and batched (prefill) | fast-math, `__fadd_rn` | `qsa_parity` | custom, small |
| `kernels/cuda/native_qsa_score.cu` (213) | score, scalar_score [D] | TF32 / scalar block scorer, unused | TF32 MMA | `tests/hip/native_qsa_score.cpp` | skip |
| `kernels/cuda/native_flash_attn.cu` (212) | attend [D] | FA-vec D = 256, diagnostic flag only | | `native_flash_attn_parity` (absent) | skip |
| `kernels/cuda/qsa.cu` (821) | kv_append [P], indexer_key_append, qsa_index, topk, kv_gather, qsa_attend, qsa_gate_apply, qsa_gate_apply_f32 [R] | Canonical QSA. The fp16 KV append (`kv_append_step`, :595) is production for `--kv f16` | FP64 | `qsa_parity` (ids exact, scores 1e-6, attention 1e-5) | port `kv_append`; the rest is a CPU reference |
| `kernels/cuda/qsa_select.cu` (788) | block_scores [A], block_topk [P], block_scores_tc [P], block_scores_wmma [D], block_scores_tail [P], block_topk_reg [P], block_scores_multi [P] | Block score = ReLU-sum of 4 indexer-head dots against pooled keys (+1e9 for the tail block). Cell-weighted radix top-k of width 2051, lowest cell first on ties, ids ascending. `tc` (3xTF32) is the prefill scorer; `reg` handles ≤ 33,792 blocks | TF32 MMA, smem histograms (~32 KB in `reg`), xor | `qsa_parity`; `qsa_select_bench` (HIP) | Re-tile `block_topk_reg` under 32 KB. `tc` becomes an fp32 `simdgroup_matrix` (no hi/lo split needed). `block_scores` is a bitwise twin of `multi` |
| `kernels/cuda/qsa_decode_attn.cu` (298) | attn_chunk [P], attn_merge [P] | Gathered split-K sparse attention over the selected cells, 4 KV modes (f16, int8-g64, q4, k8v4), then merge; `__expf(s - m_chunk)`, scale 1/16 before the max, `acc/L` | templates, `__expf` | `kv_hybrid_parity` (<5e-3); `kv_stream_parity` (bitwise); `qsa_prompt_attn_parity` | primary custom kernel |
| `kernels/cuda/qsa_prompt_attn.cu` (1032) | prompt_attn [P], prompt_attn_i8 [A], prompt_attn_wmma [D] | prefill sparse attention on tensor cores (fp16 hi/lo MMA, exp2 with `log2(e)/16`); the i8 variant is cp.async on sm80; wmma is HIP | `mma.sync`, `cp.async`, smem above 32 KB | `qsa_prompt_attn_parity` (≤1e-4·scale) | rewrite with fp32 `simdgroup_matrix` under 32 KB, or start with the batched decode kernel |
| `kernels/cuda/kv_q8.cu` (127) | kv_append_q8 [P], kv_gather_q8 [R] | int8 KV, groups of 64, scale stored fp16; quantize against the fp16-rounded scale with `x/sf` and RNE (default above 8192 context) | `__float2int_rn`, shfl max | `kv_q8_parity` (bitwise, ≤0.624 steps) | `precise::divide` + `rint` |
| `kernels/cuda/kv_q4.cu` (231) | fwht256, kv_append_q4, kv_append_q4_batch, kv_gather_q4 [all O] | q4_0 / k8v4 KV with the FWHT256 rotation | FMA-contracted `trunc(x·id + 8.5)` | `kv_q4_parity` (bitwise); `kv_hybrid_parity` | optional; not chosen by setup.py |
| `kernels/cuda/kv_stream.cu` (266) | resolve, copy, reset, ring [all V] | KV residency ring for context ≥ 65536 (`--kv-resident 32768`) | | `kv_stream_parity` (bitwise) | drop; the ring becomes a modulo |

#### C.1.4 PLE, sampler, verify and MTP control (Appendix 4)

| File (LOC) | Kernels [class] | Purpose and I/O | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|
| `kernels/cuda/ple.cu` (368) | history_advance [P], gnorm [R], gate [R], bcast [R], conv [R], add3 [R], bf16_gemv [R], to_bf16 [R] | Canonical PLE block. `history_advance` shifts the 9-row per-channel history and appends the new normalised row ([9 × 10240]) | FP64 sums (gnorm, gate, bf16_gemv), double shuffles | `ple_parity` (Part C host reference ≤1e-4; fixtures absent) | port `history_advance`; the rest via the native path |
| `kernels/cuda/native_ple_postops.cu` (221) | gate, broadcast, conv_residual, rms_rep, broadcast_batch, conv_residual_batch, history_batch [all P] | Native PLE post-ops at layer 1. Gate = ggml SUM_ROWS emulation (512 threads) → sigmoid; dilated conv (k = 4, dilation 3) + SiLU + residual; weighted RMS norm (1024 threads); batched forms for prefill | `_rn` intrinsics, fast-math, 512/1024-thread blocks | `ple_parity` native paths ≤2e-6; `native_ple_postops_parity` (absent) | keep the reduction shapes; tolerance tests (fast `exp`) |
| `kernels/cuda/sampler.cu` (933) | sampler_greedy [P], sampler_kernel [D], sampler_one_block [P], sampler_split_part [P], sampler_split_merge [P], coupled_stage [V], coupled_penalize [O], coupled_merge [O] | Penalised greedy argmax (in-graph); sampled path = split top-k (61 blocks × 4096 logits) + merge + ≤64-candidate tail (FP64); `one_block` is the fallback under capture or above 64 rows/blocks; coupled = draft sampling | smem `atomicOr`, `__umulhi`, FP64 tail, 33,792 B smem in merge | `sampler_parity` ×3 CTest variants, 18 fixtures with exact tokens | GPU penalties + top-k; **FP64 tail on the CPU** (sampled decoding already syncs). Size the merge lists to 61·64·8 = 31,232 B. Keep the Philox variant and the tie rules |
| `kernels/cuda/verify_kernels.cu` (559) | gdn_conv_l2_multi [P], gdn_conv_commit [P], gdn_ab_multi [P], gdn_step_norm_multi [P], embedding_gather_dev [C], broadcast_streams [P], copy_indexed [P], fetch_blobs [V], rebase_ptrs [V], add_streams_broadcast [P], ident_hits [D], gather_rows [P], map_ids [P], row_top_prob [P], mtp_select [P], dense_steps [D], window_ids [P], wait_flag_ge [V], resident_plan [P], wait_flag_ge_or [V], copy_i32_unless [V], copy_or_zero [V], gpu_stamp [D] | T-token GDN (verify without state mutation, then commit replay with `n_keep`); stream broadcast; PLE rollback (`copy_indexed`); MTP helpers (row gather, subset id map, top prob, select, window ids); `resident_plan` (device-built expert groups); PCIe blob fetch and doorbell spins | volatile mapped memory, `__nanosleep` spins, `__threadfence_system`, `%globaltimer`, uint4 loads | **none** | Port the P kernels. Replace waits with `MTLSharedEvent` between command buffers. `gpu_stamp` becomes `MTLCounterSampleBuffer`. New tests needed (Section E) |

#### C.1.5 Prefill, core (Appendix 5 §3)

| File (LOC) | Kernels [class] | Purpose and I/O | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|
| `prefill/kernels.cu` (842) | gr_norm [A], gr_norm_rs [P], gr_mix_r [P], gr_write_norm_rs [P], gr_silu [P], gr_mix [A], gr_write [A], gr_broadcast [P], gdn_gates [P], gdn_conv [P], gdn_conv_tiled [P], gdn_conv_hist [P], gdn_l2 [P], gdn_rec [A], gdn_rec_cols [A], gdn_rec_cols_pipe [P], gdn_out_norm [P], route [P], blob_dequant [C], swiglu_il [O], swiglu_pair [P], gather_rows16 [O], moe_combine [P], rms_rows [P], rope [P], split_q [P], gate_attn [P], kv_append [P], to_f16 [P], round_f16 [P], to_bf16 [P], copy_i32 [V] | Prompt-chunk forms of every block: GR norm/mix/write (fused `gr_write_norm_rs` is the default); GDN gates, conv (serial for T ≤ 64, tiled with hist above), L2, the per-head recurrence `gdn_rec_cols_pipe` (192 × (32, 4), register prefetch); route (precise `expf`, REG = 16); RoPE template (TAB); KV append (f16 / int8); conversions | `block_sum` (xor + smem[32]), `__expf` vs `expf` chosen explicitly, `__float2int_rn`; **not fast-math** | `tests/cuda/prefill_mmq_kquant_test`; `tests/hip/prefill_native_batch` | Mostly elementwise and row kernels; fuse freely. The serial GDN loop is latency-bound. The A variants are bitwise twins or A/B arms |
| `prefill/moe_mmq.cu` (231) | copy16 [V], copy1 [V], strata_q2 [C], swiglu [P], iota [P] | MMQ glue: gather native blobs into 16-expert group buffers; Strata blob → GGUF Q2_0; SwiGLU before Q8_1; `ids_identity`; `mmq::fits` (:110-136) mirrors `mul_mat_q_switch_J` | uint4; fast-math TU | `prefill_mmq_kquant_test`; `tests/hip/prefill_mmq_parity` | replace with an indexed grouped GEMM; fuse SwiGLU into its epilogue |
| `prefill/gemm.cu` (432) | none | `Gemm::native` (dense: `dequant_f16` into 64 MiB FP16 scratch → `cublasGemmEx` f16, row-sliced above 32M elements) and `Gemm::bf16`; hipBLASLt tuning | cuBLAS / hipBLASLt | `hip_prefill_hipblaslt_gemm` | quantized `simdgroup_matrix` GEMM; MPS optional for BF16 |
| `prefill/ggml_cuda_host.cu` (134) | none | `ggml_cuda_info` shim (`STRATA_EMULATE_CC`) and the `CachingPool` for the MMQ stream-k fixup | | | drop |
| `core/device.cu` (243) | poison [D] | NaN-fill a new `DeviceArena` (debug); device arena and stream helpers | `__int_as_float` | | `fillBuffer` blit if wanted |
| `core/pinned.cu` (549) | none | host arena reservation (large pages, normal pages or a shared file), `cudaHostRegister(Portable|Mapped)` with a cap and slices | host registration | `expert_layout_test`, `pinned_shared_test` | replace with `mmap` + `newBufferWithBytesNoCopy` (16 KiB alignment); drop `pinned_shared_test` |

### C.2 Kernels that exist only for the PCIe / discrete-VRAM design (class V, 28)

| File | Kernels | What they do on CUDA | On unified memory |
|---|---|---|---|
| `elementwise.cu` | `doorbell_ring`, `doorbell_wait`, `doorbell_publish` | A captured graph writes and spins on a sequence word in mapped pinned memory, because `cudaEventRecord` inside a capture was silently dropped (`layer.cpp:109`) | `MTLSharedEvent` signal/wait between command buffers. **Never spin inside a kernel**: Metal gives no forward-progress guarantee and CPU writes are not observed mid-command-buffer |
| `elementwise.cu` | `copy_from_mapped`, `copy_rows_from_mapped`, `copy_i32_from_mapped` | volatile reads of per-replay inputs (tokens, positions, PLE rows) from mapped memory into graph-fixed buffers | the CPU writes shared buffers before `commit` |
| `elementwise.cu` | `add` | adds the CPU pool's `y_miss` into the layer output (`expert_source.cpp:1913`) | gone with GPU-only experts. Phase 2 (a CPU helper) would reintroduce a plain add |
| `s2_expert_grouped.cu` | `hit_select`, `hit_select_multi`, `add_hits` | VRAM-cache hit lists and output accumulation | every expert is a hit; grouping replaces them |
| `s2_expert_grouped.cu` | `activation_correction`, `cpu_order_projection`, `cpu_order_swiglu`, `cpu_order_quantize` | replay the CPU AVX-512 summation order so cache-hit results match CPU-computed misses bit for bit (`--expert-cache-cpu-order`) | no CPU/GPU split, so nothing to match |
| `kv_stream.cu` | `resolve`, `copy`, `reset`, `ring` | KV cells streamed from a host mirror above 64K context | KV stays resident (int8 KV at 256K is about 3.1 GiB) |
| `sampler.cu` | `coupled_stage` | copies sampler params and the history ring from mapped memory | params written into a shared buffer |
| `verify_kernels.cu` | `fetch_blobs`, `rebase_ptrs` | `pcie_mode 2`: in-graph copy of PCIe-miss blobs into VRAM staging | experts read in place |
| `verify_kernels.cu` | `wait_flag_ge`, `wait_flag_ge_or`, `copy_i32_unless`, `copy_or_zero` | in-graph spin on CPU/DMA progress flags; skip protocol from `resident_plan` | `MTLSharedEvent`, or nothing (GPU-only experts) |
| `prefill/kernels.cu` | `copy_i32` | mapped grouping tables, to avoid queuing behind the copy engine | GPU counting sort |
| `prefill/moe_mmq.cu` | `copy16`, `copy1` | gather expert blobs into MMQ group buffers with a fixed stride | indexed GEMM reading experts in place |

**Host-side machinery with the same status** (no kernels; Section D.4 and Appendix 5 §4):
- `pinned.cu`, and the VRAM expert cache with its profile slots, adaptive swaps and slot lending.
- `pcie_mode` 0/1/2, `--pcie-frac`, and the `GpuPlanSink` CPU/GPU split.
- The CPU expert pool handshake.
- Remote experts, and the prefill copy stream with its stager and issuer threads.
- KV and MTP host mirrors, the multi-GPU layer split, and `cudaMemGetInfo` budgeting.

### C.3 Do-not-port list

| Class | Count | Kernels |
|---|---|---|
| R (reference / FP64 / non-native fallback) | 42 | `bf16_gemv_warp`, `bf16_gemv_split`; elementwise `embedding_gather`, `gdn_gate`, `scale`, `to_f16`, `silu`, `rms_norm_weighted`; all 6 of `gr.cu`; `rope_neox`; `router_top10`; shared `swiglu`, `scalar_gate`, `moe_combine`; `s2_gemv`; `s_gemv_q8k`, `s_gemv`; all 5 of `gdn.cu`; `kv_gather_q8`; qsa `indexer_key_append`, `qsa_index`, `topk`, `kv_gather`, `qsa_attend`, `qsa_gate_apply`, `qsa_gate_apply_f32`; ple `gnorm`, `gate`, `bcast`, `conv`, `add3`, `bf16_gemv`, `to_bf16` |
| A (alternate variants) | 18 | `gr_down_v3`, `gr_up_v3`, `gr_down_staged`; `gu_grouped`, `down_grouped`; `native_gdn` `step`; the 5 `native_gdn_preprocess` kernels; `block_scores`; `prompt_attn_i8`; prefill `gr_norm`, `gr_mix`, `gr_write`, `gdn_rec`, `gdn_rec_cols` |
| D (dead / diagnostic / HIP / test) | 18 | `poison`; `bf16_gemv_naive`; `router_top10_fast` (HIP); shared `to_f16`; `dequant_q8_0`, `dequant_q8_K`; `s2_gemv_fast`; `s2_gemv_quads`; `s_gemv_split`; `native_flash_attn` `attend`; `native_qsa_score` ×2; `prompt_attn_wmma`; `block_scores_wmma`; `sampler_kernel` (legacy); `ident_hits`, `dense_steps`, `gpu_stamp` |
| V (PCIe / VRAM / mapped) | 28 | C.2 |
| C (canonical packs only; conditional) | 12 | `dequant_s2`; `s2_gemv_q8`; `s_gemv_q8_split`; `quantize_q8_0`, `quantize_q8_K`; S2 per-hit `gu`, `down`, `gu_pair`, `down_pair`; `embedding_gather_dev`; prefill `blob_dequant`; `strata_q2` |
| O (optional features) | 10 | `cvec`; `kv_q4` ×4 (`fwht256`, `kv_append_q4`, `kv_append_q4_batch`, `kv_gather_q4`); `coupled_penalize`, `coupled_merge`; `dequant_gu`; prefill `swiglu_il`, `gather_rows16` (the FP16 MoE fallback) |

**Keep the R kernels as CPU-side oracles.** They use FP64, which is fine on the CPU, and they encode the reference
semantics that the parity suite checks against (Section E). `rope_neox` is the exception: its f64-table semantics
are the recommended Metal RoPE.

### C.4 Minimal Phase-1 kernel set (115), by stage

Many of these merge naturally in MSL (for example the prefill elementwise kernels, or norm + projection pairs), so
the MSL pipeline count can be lower. The decode and verify window is the latency-critical path. MTP and prefill
come next.

| Stage | Kernels (source file) | # |
|---|---|---|
| Embedding and stream setup | `embed_rows`, `dequant_flat` (iq_kernels); `to_bf16` (elementwise); `broadcast_streams`, `add_streams_broadcast`, `gather_rows` (verify) | 6 |
| PLE (layer 1) | 7 × `native_ple_postops`; `history_advance` (ple); `copy_indexed` (verify; rollback). The key GEMV uses `native_mmvq`, the value GEMV mmvf | 9 |
| GR read and write | `gr_down`, `gr_up`, `gr_norm_multi`, `gr_norm_split`, `gr_down_multi`, `gr_up_multi` (fused_gr); `weighted_rms_norm`; `down_silu`, `pre_gated`, `post`; `bf16_f32_mmvf`, `bf16_f32_mmvf_multi` | 12 |
| GDN (36 layers) | `gdn_conv_l2`, `gdn_ab`, `gdn_step_norm` (fused_gdn); `gdn_conv_l2_multi`, `gdn_conv_commit`, `gdn_ab_multi`, `gdn_step_norm_multi` (verify) | 7 |
| QSA (12 layers) | `norm`, `gate` (native_qsa); `apply` (native_rope); `kv_append` (qsa.cu, f16); `kv_append_q8`; 4 × indexer `append*`; `block_scores_multi`, `block_scores_tail`, `block_topk`, `block_topk_reg`, `block_scores_tc` (prefill); `attn_chunk`, `attn_merge`; `prompt_attn` (prefill) | 17 |
| Dense mat-vec and quantizers | 9 × `native_mmvq.cu`; `mmvq`, `mmvq_multi`, `quantize_q8_1` (iq_kernels) | 12 |
| MoE (native) | `route` (native_router); `native_gu`, `native_gu_multi`, `swiglu_entries`, `native_down`, `native_down_multi`; shared `native_swiglu`, `scale`, `native_scalar_sigmoid`, `native_scalar_sigmoid_multi`, `scale_rows`; `combine` (native_moe); `resident_plan` (verify, generalised to every layer) | 13 |
| MoE (MTP S2 experts) | `group_resident`, `gu_grouped_t`, `down_grouped_t`, S2 `swiglu`; `quantize_q8_0_scaled` | 5 |
| Head and sampler | head = `native_mmvq`; `sampler_greedy`, `sampler_one_block`, `sampler_split_part`, `sampler_split_merge` (FP64 tail moved to the CPU) | 4 |
| Verify / MTP control | `map_ids`, `row_top_prob`, `mtp_select`, `window_ids` | 4 |
| Prefill | 23 × `prefill/kernels.cu` P; `swiglu`, `iota` (moe_mmq); `dequant` (dequant_bf16) | 26 |
| **Total** | | **115** |

**Not in the 115, but required in Phase 1:**
- A grouped/indexed quantized GEMM for prefill experts and dense weights, replacing ggml-cuda MMQ and cuBLAS
  (D.3).
- A GPU counting sort for prefill expert grouping (replacing the host sort and its 48 syncs per chunk).
- An optional strided-copy kernel for the MTP q/gate split (`cudaMemcpy2DAsync`, `mtp.cpp:493`).
- `fwht256` only if `kv_rot` or the Q4 KV formats are enabled.

---

## D. Decode and prefill structure, and everything that assumes discrete VRAM

Appendix 5 §1-§4 has the line-by-line detail. This section gives the structure a port has to reproduce, or delete
on purpose.

### D.0 Execution contexts on CUDA

| Context | What runs | CUDA vehicle | Host participation |
|---|---|---|---|
| Verify window (T = 1..8) | embed → 48 layers → head → greedy argmax | one graph per T (`exec_[9]`, lazy capture), on stream `cs_` | **one doorbell ring per layer**: the host plans experts, the CPU pool computes the misses, and the flags release the graph |
| Commit (n_keep = a+1) | GDN replay of the kept tokens into persistent state, indexer tail restore + re-append, PLE history rollback | one graph (`commit_exec_`), async, on `cs_` | writes `h_commit_` before launch |
| MTP drafter | catch-up over T cells, then up to 3 chained drafts | `round_exec_[9]`, `step_exec_[9]`, `prefill_exec_[9]` on its own stream | one sync per draft step (reads the probability for `spec_min_p`) |
| Sampled / penalised decoding | `sample_tokens` again over the head logits, outside the graph | plain launches + `cudaStreamSynchronize` | Philox counter = pos0 |
| Adaptive tier | VRAM-cache residency swaps | `adapt_stream` + `adapt_ev`, on a `std::thread` | every 4 rounds |
| Prefill chunk (≤ 8192 tokens) | all layers over T tokens | plain launches on `m.cs`, expert copies on `m.copy`, issuer and stager threads | **48 `cudaStreamSynchronize` per chunk** (expert grouping on the host) |

### D.1 Decode

#### D.1.1 One round (`generate.cpp:6128-6232`)

| Step | Host thread | GPU `cs_` | Other |
|---|---|---|---|
| 1 | choose T (MTP `S_mtp`, shrunk by `spec_min_p`; the first native window is T = 1; optional suffix drafter via `DraftPolicy`) | | |
| 2 | `apply_pending`: admit finished adaptive swaps, re-upload `d_res` | | adapt copies done? |
| 3 | `ver.run(T, window)`: stage the inputs in mapped memory, reset the doorbell words, launch the graph | window graph | |
| 4 | serve 48 rings: spin on `seq`, plan, raise flag A (plan) and B (PCIe share), run the CPU pool, raise flag M (CPU rows) | `pre(l)` … wait A … VRAM groups … wait B … PCIe groups … wait M … combine … `pre(l+1)` | copy stream: PCIe DMA (mode 0) |
| 5 | `cudaStreamSynchronize(cs_)`, then `(copy_)`; read `out[]` | head + argmax done | |
| 6 | accept a = leading matches; every 4th round start `adapt()` | | adapt thread |
| 7 | `ver.commit(a+1)` (async) | commit graph | |
| 8 | emit tokens | | |
| 9 | `mtp.draft(...)` | (ordered after the commit by the stream) | MTP stream overlaps the commit |
| 10 | join adapt; `x = outv[a]`, `p += a+1`; `DraftPolicy::observe` | | |

#### D.1.2 The window graph, per layer (`record_window`, `verify.cpp:387-859`)

`pre(l, grp)` runs everything up to the doorbell. `post(l, grp)` is the MoE tail. With G = 1 (the default; G = 2
`--spec-split` measured about 7% slower) the order is `pre0 | post0 pre1 | post1 pre2 | …`.

| Phase | Step | CUDA kernels | Phase-1 Metal kernel(s) |
|---|---|---|---|
| inputs | tokens, steps, positions, PLE rows | `copy_i32_from_mapped` ×3, `copy_from_mapped` | none: kernels read the shared input buffer |
| embed | rows → 4 streams | `NativeEmbed::gather_dev` (`embed_rows` / `dequant_flat`), `broadcast_streams` | same |
| pre, layer 1 | PLE block | `gr_write`, `ple_block` (native post-ops), `ple_history_advance`, snapshot `hist_snap_` | same, plus `copy_indexed` at commit |
| pre | GR attention read (folds the previous FFN write) | `gr_norm_multi` + `gr_down_staged`/`gr_down_multi` + `gr_up_multi` (T ≥ 2), or `gr_down` + `gr_up` (T = 1) | re-tiled `gr_down_multi` (32 KB) |
| pre, GDN | qkv, conv, a/b, gate, recurrence, out | `native_quantize_q8_1`, `native_mmvq` (qkv), `gdn_conv_l2_multi`, `gdn_ab_multi`, `native_mmvq` (z), `gdn_step_norm_multi` (no state advance), q8_1 + `native_mmvq` (`ssm_out`) | same |
| pre, QSA | indexer k, K/V, append, indexer append, Q, split, select, attention, gate, o-proj | mmvf (indexer k), `native_mmvq` K/V, `norm`, `apply` (RoPE), `tail_snap_` copy, `kv_append_*_step`, `native_qsa_indexer_append`, `native_mmvq` Q, `cudaMemcpy2DAsync` q/gate split, `block_scores_multi` + `block_topk(_reg)`, `qsa_kv_resolve` (KV streaming), `attn_chunk` + `attn_merge`, `gate`, `native_mmvq` (out) | same, minus `qsa_kv_resolve`; the q/gate split becomes a strided read |
| pre | GR FFN read (folds the attention write) | fused GR as above | same |
| pre | router | `bf16_f32_mmvf_multi`, `route` (native top-10) | same |
| pre | E-6 plan (opt-in) | `resident_plan` | **always**: the generalised plan |
| pre | **doorbell** | `doorbell_publish` (x, ids, w → mapped; `seq++`) | **deleted** |
| pre | shared expert, while the host works | `f32_to_bf16_bulk`, `shared_expert_multi` (`native_scalar_sigmoid_multi`, `native_mmvq` gate/up, `native_swiglu`, q8_1, down) | same |
| pre | activation quantization for the expert kernels | `quantize_q8_1_rows` (native) / `quantize_q8_0_scaled` (S2) | same |
| post | plan ready | `wait_flag_ge(A)`, `copy_i32_from_mapped(plan)` | deleted |
| post | VRAM groups | `native_expert_grouped` (`native_gu_multi`, `swiglu_entries`, q8_1, `native_down_multi`) or `moe_grouped_s2` | same, over **all** routed experts, reading the arena in place |
| post | PCIe groups | `wait_flag_ge(B)`, `fetch_blobs`, `rebase_ptrs`, grouped kernels on `staging_` | deleted |
| post | CPU rows | `wait_flag_ge(M)`, `copy_rows_from_mapped(y_miss)` | deleted (Phase 2: read the shared rows) |
| post | combine | `moe_hit_add`, `native_moe_combine_multi` (fixed k order + shared) | `combine` only (the GPU writes `parts_` directly) |
| post, l = 47 | final HC write, cvec | `gr_write`, `cvec_apply` | same |
| head | output mixer, head GEMV, argmax | `lm_head_mix` (gr dispatcher with `inject == nullptr`), `native_quantize_q8_1` + `native_mmvq` (T columns, 248,320 rows), `sampler_greedy` | same |

**Dispatch count (estimate).** About 30 dispatches per GDN layer and 35 per QSA layer, so roughly 1,500 per window.
At an assumed 1-2 µs of CPU per encoded dispatch, that is 1.5-3 ms per window. Encoding can overlap the previous
window's GPU time. Measure before choosing between per-window re-encoding and an indirect command buffer per T
(F.13).

#### D.1.3 The doorbell protocol (summary; Appendix 5 §1.5 has every word)

- **Mapped buffers.** All are `cudaHostAlloc(Mapped)` with device aliases: the inputs (`tok`, `step`, `pos`, `ple`,
  `commit`), the outputs (`out`, and `x`/`ids`/`w` published by the GPU), the plan, `ymiss` (T·K·N f32, 409.6 KB at
  T = 4), and four 64-B words.
- **The four words:**
  - `seq` (GPU → host, incremented in memory).
  - `flagA` (plan ready).
  - `flagB` (PCIe share in staging; a monotonic CAS from a host function on `copy_`).
  - `flag` M (CPU rows ready).
- **Rings.** Ring `k = (l - lb)·G + grp + 1`. All words are reset to 0 at window start, and the GPU waits for
  `≥ ring`, so a late raise is harmless.
- **GPU waits.** One-thread `<<<1,1>>>` spin kernels with `__nanosleep(100)`.
- **Host waits.** `_mm_pause` spins, with a `cudaStreamQuery` every 2 ms (WDDM flush and fault detection) and the
  #267 watchdog after 20 s (`release_gpu_waits` writes `UINT32_MAX`).
- **Lessons recorded in the code:**
  - The doorbell value must be incremented in memory, not baked in at capture (the "cloned literal" bug).
  - GPU stores to mapped memory need volatile plus `__threadfence_system` (RDNA4 kept them in L2).
  - Without the fence the host fell back to `cudaEventQuery`, costing about 10 ms per token.

#### D.1.4 Producer tiers and the combine

- Each routed entry `i ∈ [0, n·K)` has exactly one producer, decided per expert by the host planner
  (`expert_pool_dispatch_multi`, `expert_source.cpp:1588-1807`):

  | Tier | Condition | Where the bytes come from |
  |---|---|---|
  | VRAM | `host_res ≥ 0` | the expert-cache slot |
  | PCIe | one of the last `m = nmiss·pcie_frac` misses, pinned layer, staging capacity 16 (8 per group), < 64 fetches | staging, or the mapped alias |
  | CPU | everything else | AVX-512/VNNI pool, writing `y_miss` rows directly |
  | Remote GPU | optional second GPU | |

- **The plan is published before any CPU work**, so the GPU starts early. A plan holds the VRAM groups first, then
  the PCIe groups, with `counts = {vram_groups, entries, pcie_groups}`.
- **Combine.** GPU rows land in `hit_out_`, and `copy_rows_from_mapped` brings in only the CPU rows (GPU rows are
  +0.0). `moe_hit_add` then fills `parts_` in routing order, and `native_moe_combine_multi` reduces in fixed k order.
- **Nondeterminism.** The summation order does not depend on the producer, but the per-expert arithmetic does: the
  CPU and GPU kernels differ. Output therefore depends on residency, and with the adaptive tier, sampled output is
  not reproducible (`DETAILS.md:539-541`). `moe_hit_grouped_s2_cpu_order` exists only to close this gap for the S2
  format.

#### D.1.5 Commit and the verify invariants

- The window must not mutate the GDN state, the conv history, the indexer tail or the PLE history. It caches
  `qkv_L_`, `h_L_`, `gate_L_` and `beta_L_` per GDN layer.
- **The commit graph:**
  1. `copy_i32_from_mapped(commit_)`.
  2. Per GDN layer: `gdn_conv_commit` and `gdn_step_norm_multi(…, commit_)`, which replays the kept tokens.
  3. Per QSA layer: restore `tail_snap_` and re-append the kept rows (`-1` skips).
  4. `copy_indexed(ple.hist, hist_snap_, commit_+1)`.
- K/V rows of rejected positions are simply overwritten later.
- **Async commit.** `commit_done_` is recorded, and the next window is ordered after it by the stream.
  `wait_commit` happens before any host read of the session.
- **Exactness requirement.** Speculative decoding equals greedy decoding token for token
  (`verify.hpp:1-18`). Decode ≡ verify needs the per-row arithmetic of every kernel to be independent of T.

#### D.1.6 Graph rules (`graph.hpp`, `graph.cpp`)

1. Kernel arguments are frozen at capture, so every per-step value lives in device or mapped buffers.
2. Fixed addresses: one arena, bump-allocated once (`verify.cpp:290-330`).
3. Capture is lazy per T, and a sampler with parameters is never captured: it would replay the same draws.
4. On WDDM, waiting must poll the driver. Driver calls cost 10-22 µs each.

#### D.1.7 MTP drafter structure (Appendix 4 §3)

- **Graphs.** `capture_round(T)` is a K/V-only pass over the window's T cells followed by the first full step.
  `capture_step(j)` is one draft step.
- **Inputs.** The verifier's `final_R` rows and the accepted tokens.
- **Streams.** It runs on its own stream, overlapping the async commit. This is safe because it reads only this
  window's final rows and its own K/V.
- **Routed experts.** `moe_group_resident` → `quantize_q8_0_scaled` → `moe_grouped_s2`, over 512 resident S2 blobs.
- **Head.** The draft head is the main head, or the compact `dhead_` built by `gather_rows`.
- **Selection.** Greedy + `row_top_prob` + `map_ids`, then a host sync per step to apply `spec_min_p` (0.5 in setup).
  `row_top_prob` normalises over the draft subset only, so it overestimates the probability.

#### D.1.8 Measured decode numbers (paper findings, CUDA)

| Quantity | Value |
|---|---|
| Window (Q2_0, 4K) | about 33 ms: the CPU waits about 14 ms for the GPU, the GPU about 13 ms for the CPU |
| CPU expert rate | about 42 GB/s (41 concurrent with the GPU, 52 alone) |
| Speculation (MTP) | 1.6-1.8× |
| Adaptive tier | hit rate 50% → 72%; asynchronous refill 91.7 → 94.4 tok/s (Q2_0, 4K) |
| Byte-sized slots | 3,556 IQ3_XXS experts instead of 2,673, +13% at 1K |
| G = 2 split | 85.8 vs 91.8 tok/s (slower) |
| Without the expert cache | 663.6 MB of expert bytes per token at about 40 GB/s = 16.2 ms of a ~53 ms token |
| Polling instead of a blocking sync at window end | -3% (IQ3_S) |

### D.2 The expert cache and the adaptive tier (CUDA only)

| Piece | Behaviour | Numbers |
|---|---|---|
| Storage | one `cudaMalloc` of `n_slots·blob` (or byte-sized slots, prefix sums of `roundup256(blob_bytes(layer))`); `cudaMemset(0)` so an unfilled slot is deterministically wrong; `residency_[48×512]` | Q2_0 blob 1,382,400 B; native IQ3_XXS 1.3-2.3 MB per layer |
| Sizing (`--expert-cache auto`) | `(free − reserve)/max_blob`, with reserve = 700 MiB + prefill `160 + chunk·680/1024` MiB (unless borrowing) + MTP; allocated **last**; capped at the profile | |
| Profile (`STRP`) | `"STRP"`, `<5I>` header, `n_ranked × <HH>(layer, expert)`, a 48×512 int32 table (ignored by the reader) | 196,632 B for a full profile; `tools/make_profile.py` from `--dump-routing` traces |
| Hit rate | 4,105 slots give h = 0.6447 [0.6110, 0.6750] (ten-fold leave-one-out); only 4.6% of (layer, token) pairs have all ten resident, so the split is per expert | |
| Fill | `admit` + `fill_slot_blocking` in rank order; **never evicts**; `verify_slot` (1.38 MB D2H + memcmp) at startup | |
| `d_res` | 48×512 slot table in VRAM for the device-side hit decision and E-6 | |
| Adaptive tier | every `adapt_every` = 4 rounds on a thread: candidates (non-resident, usage ≥ 2.0) vs victims (resident), pairs while `cand ≥ vict + 1.5`, at most `adapt_swaps` = 96 by gain; H2D on `adapt_stream`, `host_res[out] = -1` at once, admit on event completion; `usage *= 0.7` | |
| Prefill lending | the last k slots are lent to prefill scratch, then refilled (`fill_slot_queued` + one `sync_queued`) | 12 GB card, 32K Q2_0: chunks 4096 / 6144 / 8192 give 791 / 878 / 973 tok/s with 69% of slots lent |

**On Apple: delete the whole tier** (Appendix 5 §2.7).
- Every expert is GPU-addressable in a no-copy arena.
- A "slot" becomes `blob_offset(layer, expert)`.
- The profile format survives only as a page-cache warm-up order, or as the resident-set policy for packs larger
  than RAM (D.5.5).
- The adaptive tier's usage counters and gain rule survive only as a load-balancing or RAM-residency policy.

### D.3 Prefill

#### D.3.1 Chunking

- **`--prefill auto`** (setup's default):
  - Tries 32768 down to 256 against the slot-lending rule (`k + 128 ≤ slots`, `k ≤ 85-90%` of slots).
  - The default ceiling is 8192 (`prefill_auto_max`). `auto:N` allows 16384 or 32768.
  - With no profile, or with `--no-prefill-borrow`, it uses a fixed 2048. Serve falls back to 1024.
- **Coverage.** Batched prefill covers `n_prompt - 1` tokens. The last token goes through the decode path.
- **Memory per chunk.** About 680 KB per token, plus:
  - 64 MiB of FP16 GEMM scratch and a 32 MiB cuBLAS workspace.
  - A DQ ring of about 19.7 MB.
  - 16-expert MMQ group buffers with a 4096-B tail.
  - The expert ring of 8 / 96 / 384 × MAXBLOB.
- **The chunk size changes the kernels.** T ≥ 1024 selects stream-all (smaller chunks gave different Q2_0 output).
  T ≤ 64 selects the serial GDN conv. QSA selection runs per 256 queries; the fallback attention per 32.

#### D.3.2 One chunk (`prefill.cpp:994-1941`)

1. **Preamble:**
   - `iota(ids_identity)`.
   - An async PLE gather for the next chunk into a pinned double buffer.
   - Embedding plus `gr_broadcast`.
   - Image rows if vision is used.
   - The stream-all plan of every non-resident (l, e), with an issuer thread copying expert blobs H2D into a ring
     on `m.copy`, using `used[sl]` / `copied[sl]` events.
2. **Layer 1:** batched PLE (`native_ple_postops` `*_batch`).
3. **Each half:**
   - `gr_norm_rs`.
   - BF16 GEMMs `hc_down` [T,10240]→[T,320], then `gr_silu`, then `hc_up` → [T,10240], then `hc_inject` → [T,4].
   - `gr_mix_r`.
4. **GDN** (36 layers):
   - Native FP16 GEMMs `attn_qkv` and `attn_gate`; BF16 `ssm_alpha`/`ssm_beta`.
   - `gdn_gates`, then `gdn_conv` (serial ≤ 64, else `gdn_conv_tiled` + `gdn_conv_hist`), then `gdn_l2`.
   - `gdn_rec_cols_pipe` (a serial T loop; state [k][head][v] f32), then `gdn_out_norm`.
   - `ssm_out`.
5. **QSA** (12 layers):
   - Native K/V/Q; BF16 indexer.
   - `rms_rows`, `rope`, `kv_append`, `split_q`, `rms_rows`, `rope`.
   - Indexer `append_blocks`/`append_tail`.
   - Per 256 queries: `block_scores_tc` (3xTF32, "not bitwise") + `block_topk`.
   - `prompt_attn` (tensor cores), or the batched decode attention.
   - Inverse FWHT if rotated, `gate_attn`, `attn_output`.
6. **MoE:**
   - BF16 router, then `route` (precise `expf`, REG = 16).
   - Shared expert via `swiglu_pair`.
   - **Host grouping**: `copy_i32` → `cudaStreamSynchronize` (`:1592`) → CPU counting sort → mapped
     `slot`/`src`. This happens once per layer, 48 times per chunk.
   - `quantize_mmq_q8_1`.
   - Per 16-expert group: `copy16`/`copy1`/`strata_q2` gather, a 4096-B zero tail, GU `mul_mat_q` →
     `mmq::swiglu` → quantize → down `mul_mat_q`.
   - `moe_combine`.
7. **HC write:** `gr_write_norm_rs` (fused with the next norm).

#### D.3.3 MMQ vs cuBLAS

| Product | Path |
|---|---|
| Routed experts | llama.cpp MMQ (`mul_mat_q`, int8 MMA, Q8_1 activations), if built, enabled, the type is in the MMQ list (B.1) and `mmq::fits` |
| Routed experts, fallback | dequant to FP16 (`blob_dequant`, `dequant_flat`) + 2 `cublasGemmEx` per expert (the IQ1_M path) |
| Dense quantized weights | **always** `dequant_f16` → 64 MiB scratch → `cublasGemmEx` FP16 (row-sliced above 32M elements) |
| BF16 weights | `cublasGemmEx` BF16 × BF16 → F32 |
| MTP K/V (E-9) | Q8_0 MMQ, or FP16 |
| HIP | hipBLAS (hipBLASLt with a tuning file); MMQ opt-in |

There is no token-count threshold between MMQ and GEMM.

**Measured** (RTX 5070, 32K): Q2_0 572 → 1,290 tok/s and IQ3_S 383 → 1,208 tok/s, comparing the paper's
2048-chunk FP16 path with the current one. On a 5090, IQ2_XS reaches 5,624 tok/s with 8192-token chunks and 6,465
with one 32768-token chunk.

#### D.3.4 Prefill on Apple

| CUDA mechanism | Apple replacement |
|---|---|
| Expert streaming (copy stream, 512 events, issuer and stager threads, slot lending, refill) | delete: grouped kernels read `arena + blob_offset(l, e)` in place |
| Host grouping, 48 syncs per chunk | GPU counting sort: an atomic histogram over `ids[T·10]`, an exclusive scan over 512, a scatter of (token, k) → slot. Expert bounds are read by the GEMM from a buffer (indirect dispatch, or a fixed grid with early exit) |
| ggml-cuda MMQ (int8 MMA, Q8_1 activations) | an indexed grouped `simdgroup_matrix` GEMM modelled on ggml-metal `kernel_mul_mm_id`: dequantize weight tiles into threadgroup memory, multiply with float activations, FP32 accumulate, SwiGLU fused into the gate/up epilogue |
| `dequant_f16` + cuBLAS for dense | the same quantized GEMM without the expert index |
| cuBLAS BF16 | the same GEMM template with bf16 tiles (or MPS / MPSGraph) |
| `block_scores_tc` (3xTF32) | fp32 `simdgroup_matrix` (no tf32 on Apple, so no hi/lo split) |
| `prompt_attn` (fp16 hi/lo MMA, smem above 32 KB) | re-tile under 32 KB with fp32 `simdgroup_matrix`, or start with the batched decode attention |
| PLE double buffer + H2D | the host thread writes directly into one of two shared buffers |
| KV streaming and the host mirror | KV resident |

**Numerics.** Float activations in the expert GEMM change prefill numerics against CUDA's Q8_1 MMQ. That is
acceptable, because CUDA prefill is already not bitwise with decode and already differs between chunk sizes. It must
still be gated by KL (E.6).

**Mac chunk rule.** Use the largest of {8192, 6144, 4096, 2048, 1024} whose `bytes_needed` (about 680 KB per token,
so about 5.4 GB at 8192) fits a fixed fraction of `recommendedMaxWorkingSetSize` minus the model. Keep 1024 as the
floor if parity with CUDA output matters.

### D.4 Everything that assumes discrete VRAM

Appendix 5 §4 lists 84 numbered assumptions, each with code locations and an Apple disposition:
(D) delete, (R) replace with a Metal primitive, (K) keep, (M) measure.

| Group | Items | Typical content | Disposition |
|---|---|---|---|
| A. Allocation and budgeting | A1-A18 | `DeviceArena`, weight upload through pinned stages, `cudaMemGetInfo` sizing everywhere, "allocate the cache last", KV streaming from about 64K, eager module loading | mostly D/R: shared or private `MTLBuffer`s with the same bump carving; one fit check against `recommendedMaxWorkingSetSize`; KV resident |
| B. Mapped pinned memory | B1-B16 | doorbell I/O, mapped staging, the `cudaHostRegister` expert arena (whole or sliced, Windows pin budgets), hugetlb, the 4 KiB shared-arena header, device aliases for PCIe reads, volatile + system fences | R: `StorageModeShared` buffers; `newBufferWithBytesNoCopy` over a 16 KiB-aligned mmap; `MTLResidencySet`. D: hugetlb, aliases, remote and layer-split buffers. B7: the 4096-B header needs a buffer offset or a 16 KiB header |
| C. Explicit copies and the PCIe tier | C-1 to C-13 | `pcie_frac` and its probe, `pcie_mode` 0/1/2, 16-blob staging, cache fills, adaptive copies, prefill streaming, `copy_rows_from_mapped`, the q/gate `cudaMemcpy2DAsync`, snapshot copies | D, except C-11 (strided view) and C-12 (`memcpy` from `.contents` after completion) |
| S. Synchronisation inside a window | S1-S11 | the doorbell ring, WDDM flushes, the #267 watchdog, `_mm_sfence`, monotonic CAS, `cudaLaunchHostFunc`, blocking stream syncs, the async commit event, device-wide syncs (including 48 per prefill chunk), the HitFn seam | R: no host in the loop (Phase 1); `MTLSharedEvent` across command-buffer boundaries if a CPU helper returns; queue order replaces stream order |
| G. Graphs, streams, device queries | G1-G11 | graph per T, `wait_ms` polling, non-blocking streams, smem above 48 KiB at 8 sites, `kMinCc` 75, `STRATA_EMULATE_CC`, multi-GPU, `cudaGetLastError`, `%globaltimer` | R: per-window re-encode or ICB; one compute queue (plus one for MTP overlap); 32 KiB re-tiles; `MTLGPUFamily` + `threadExecutionWidth == 32` checks; command-buffer error status; `MTLCounterSampleBuffer` |
| O. CPU / OS / ISA | O1-O9 | `_mm_pause`, thread affinity, AVX-512 expert kernels, `/proc/meminfo`, `posix_fadvise`, `O_DIRECT`, hugetlb, WDDM detection, `mlock` | R: `__builtin_arm_yield`, QoS classes, NEON or no CPU experts, `host_statistics64` / `os_proc_available_memory`, `madvise` with a 16 KiB page size, `F_NOCACHE` |

**Policy knobs from the PCIe machine** (Appendix 5 §4.7):
- **Delete on Apple:** `--expert-cache auto`, `--vram-reserve-mib 700`, `--pcie-frac`, `--pcie-mode`,
  `--expert-cache-per-layer` and `--no-prefill-borrow`. The same goes for `STRATA_DEC_BATCH`, `STRATA_GROUP_COPY`,
  `STRATA_REFILL_BLOCKING`, `STRATA_TG_FLUSH_US`, the arena pin/lock variables and `CUDA_MODULE_LOADING`.
- **Repurpose or delete:** `--expert-profile` and `--adapt-*`.
- **Make it the design:** `STRATA_VERIFY_DEVICE_PLAN` (E-6).

**The 28 class-V kernels** are listed in C.2.

### D.5 The Apple design

#### D.5.1 Phase 1: GPU-only experts, one command buffer per verify window

1. **Host.**
   - Write `tok[T]`, `step`, `pos[T·(NH+NKV+IQ)]` and the PLE rows (CPU hash + gather + decode) into a shared input
     buffer.
   - Encode or replay the window for this T, then commit.
2. **GPU, in one command buffer:**
   - embed → broadcast.
   - Then, for l = 0..47:
     - `pre(l)`: GR read, GDN or QSA block, GR read, router.
     - `resident_plan`: device grouping; every expert is "resident".
     - `quantize_q8_1`, then the shared expert.
     - Grouped GU → `swiglu_entries` → `quantize_q8_1` → grouped down, all reading the arena in place.
     - `combine`.
   - Then the head mix → `quantize_q8_1` → head GEMV (T columns) → `sampler_greedy` → `out[T]`.
3. **Host.**
   - `waitUntilCompleted`, or an `MTLSharedEvent` wait, and read `out[T]`. Shared memory is coherent at command-buffer
     completion.
   - If the request is sampled: GPU penalties + split top-k in a second small command buffer, then the ≤64-candidate
     FP64 tail and the Philox pick on the CPU.
4. **Commit.** Encode it as its own command buffer on the same queue: queue order replaces stream order, and the next
   window runs after it.
5. **MTP.** Run the drafter on a second `MTLCommandQueue` so it can overlap the commit. It must wait (`MTLSharedEvent`)
   on the window's completion for `final_R`, as it does today through stream order. One host sync per draft step
   remains (`spec_min_p`).
6. **What goes away:**
   - Every mapped copy kernel, the doorbell and spin kernels, `fetch_blobs`/`rebase_ptrs` and the CPU pool handshake.
   - `hit_select`/`add_hits`, the cpu-order kernels and `copy_rows_from_mapped`.
   - Residency-dependent nondeterminism.

#### D.5.2 Buffers

| Buffer | Mode | Content |
|---|---|---|
| Expert arena | NoCopy over an mmap of `experts.bin` / GGUF shards; one buffer per layer (16 KiB-aligned starts, under `maxBufferLength`); in an `MTLResidencySet` | Q2_0: 707.8 MB per layer; native IQ3: about 0.9 GB per layer |
| Dense weights, head, embedding | NoCopy or converted once into shared buffers | head Q6_K 521 MB |
| Session state | private, or shared when snapshots are wanted | 36 × 3.12 MiB GDN state, conv history, paged KV (f16 ≤ 8192, q8 above), indexer keys, PLE history [9 × 10240] |
| Window scratch | private, bump-carved once with fixed `gpuAddress`es (the CUDA-graph rule carries over) | `parts_` [T·K·N], `mixed`, Q8_1 activations, plan arrays, `qkv_L_`/`h_L_`/`gate_L_`/`beta_L_` for the commit |
| Inputs and outputs | shared | tokens, steps, positions, PLE rows, `commit_`, `out` |
| PLE table | mmap, CPU only (`F_NOCACHE` reads or the page cache) | 28.8 GB |

#### D.5.3 Roofline

- **Q2_0 window at T = 4.** At most 4 × 10 × 48 distinct blobs (about 2.65 GB). In practice the union per layer is
  20-30 experts, about 1.3-2.0 GB.
- **Fixed costs on top:** dense weights, the head (521 MB per window, read once for all T columns) and KV reads.
- **Lower bound for experts alone:** about 4-6 ms at 400 GB/s, and about 12-18 ms at 120 GB/s.
- **Compared with CUDA,** a 33 ms window with a CPU/GPU wait split. The theoretical Apple window is bandwidth-bound,
  and decode tokens per second scale with accepted tokens per window.
- **A CPU helper does not raise this bound.** The CPU sits on the same memory controller (D.5.4).

#### D.5.4 Phase 2 (optional): a CPU helper

Build it only if measurement shows idle GPU bandwidth.
- Port the planner's "CPU share" as a fraction, not a residency decision.
- Run NEON (`sdot`/i8mm) kernels whose arithmetic order matches the GPU kernels bit for bit, as
  `moe_hit_grouped_s2_cpu_order` does on CUDA.
- Write rows into a shared buffer.
- Hand off per layer with `MTLSharedEvent` signal and wait values across command-buffer or encoder boundaries.
- **Never spin a GPU thread on host memory.** Metal has no forward-progress guarantee, and GPU writes are visible
  only at completion or signal boundaries.

#### D.5.5 Capacity tiers by Mac RAM size

| Unified memory | Fit | Design |
|---|---|---|
| ≥ 96-128 GB | experts (34-43 GB) + dense + KV + scratch fit; the PLE table (28.8 GB) can also be RAM-resident | Phase 1 as is; `mlock` optional |
| 64 GB | Q2_0 / IQ2_XS experts fit; the PLE table stays on SSD (mmap or `F_NOCACHE` reads, 2.1-2.6 ms per token of faults measured on CUDA hosts) | Phase 1; watch `recommendedMaxWorkingSetSize` |
| 32-48 GB | experts do not fit | two-level store: a GPU-visible RAM set chosen by the existing profile and usage counters, refreshed from SSD with `pread` + `F_NOCACHE` into arena slots. Misses need one of: GPU page faults on a file-backed NoCopy buffer (unverified, probably slow), a CPU helper, or a host prefetch after routing (a doorbell over `MTLSharedEvent`). Reuse the `FileExpertSource` lookahead and the 8-thread prefetch |

---

## E. Test strategy: parity tests, oracle vectors, golden outputs

Appendix 5 §6 and Appendix 4 §8 have every test with its CMake line.

### E.0 Method

1. **A reference chain, not self-comparison.** Each GPU kernel is compared with a scalar host reference. That
   reference is itself validated against llama.cpp/ggml through gate C1 (`CMakeLists.txt:220-222`: "a kernel checked
   against another copy of itself proves nothing").
2. **Two contracts:**
   - **Bitwise**, where the order is fixed by construction: quantizers, KV quantization, decode ≡ verify, multi ≡
     single, batched ≡ sequential QSA, streamed ≡ resident KV, sampler tokens, router ids.
   - **Tolerance** against FP64/FP32, where the order is free: GEMV, GEMM, attention.
3. **Negative controls.** The same comparison with the exact mode switched off must find differences
   (`mmvq_multi_parity.cpp:24,181,237`; `qsa_parity.cpp:1298,1373`).
4. **Skip codes.** 77 means skip. 3 means skip for the IQ fixture pair (`CMakeLists.txt:527-533`), when numpy or
   gguf-py is missing. Some tests fail instead of skipping: `ple_parity` without its fixtures, and the device
   self-tests without a GPU.
5. **Cross-backend contract.** The HIP port makes no cross-backend bitwise claim (`docs/AMD_HIP.md:16`). Within-backend
   determinism, plus the model-level gates, is the contract. Metal should adopt the same contract.

### E.1 State of the public tree

- **No CI.** `.github/` has only `FUNDING.yml`.
- **Tests are off by default.** `STRATA_BUILD_TESTS` defaults OFF, because `tests/CMakeLists.txt` and `bench/micro`
  are missing (`CMakeLists.txt:52-57`).
- **Counts.** About 33 ctests register in a default build, about 61 with a GPU backend. The private tree had 70.
- **Missing assets:**
  - `bench/micro`: the ggml-oracle parity suite (`CMakeLists.txt:311-334`) and the PLE capture fixtures
    `ple_in.bin`/`ple_out.bin`.
  - `scripts/`, `tests/python` (including `gen_ple_oracle.py`), most of `ref/`, and `gguf-py`.
  - `tools/gpu_session.py`, `tools/kv_precision_compare.py`, and the sources for `oracle`, `native_mmvq_multi`,
    `graph_registry` and `pinned_capture`.
- **The 16 absent ggml-oracle tests** (`CMakeLists.txt:315`): `native_mmvq`, `native_bf16`, `native_shared`,
  `native_gr_norm`, `native_moe`, `native_gdn`, `native_gr_postops`, `native_router`, `native_gdn_preprocess`,
  `native_gdn_pipeline`, `native_qsa`, `native_rope`, `native_ple_postops`, `native_flash_attn`,
  `native_qsa_indexer`, `native_qsa_score`.

### E.2 Coverage map: kernel file → existing test

| Test | Kernel files covered | Contract |
|---|---|---|
| `quantize_act_parity` | quantize_act | bitwise including ties |
| `elementwise_parity` | elementwise, dequant_bf16 | bitwise / tight |
| `dequant_bf16_test`, `q5_projection_parity` | dequant_bf16 | F32 1e-6, BF16 half-ulp |
| `dequant_s2_parity` | dequant_s2 | memcmp |
| `s2_gemv_parity` | s2_gemv (+ fast, quads) | 1e-5 |
| `s2_gemv_q8_parity` | s2_gemv_q8 | 1e-4 (1.479e-5 measured) |
| `s_gemv_parity`, `s_gemv_q8k_parity` | s_gemv | 1e-4 / 1e-5 |
| `bf16_gemv_parity` | bf16_gemv (**not** mmvf) | 1e-5 |
| `iq_parity` (+ fixtures, skip 3) | iq_kernels, native_mmvq | against gguf-py fixtures |
| `iq_multi_parity`, `mmvq_multi_parity` | iq_kernels, native_mmvq | bitwise multi ≡ single, with a negative control |
| `native_expert_parity` | iq_kernels and CPU native kernels | ≤ 1e-5 (1e-7 measured) |
| `s2_expert_grouped_parity`, `moe_hit_parity` | s2_expert_grouped | ≤ 4e-5·magnitude vs CPU; new ≡ old bitwise |
| `router_top10_parity` | router_top10 | ids exact, weights 1e-5 |
| `shared_expert_parity` | shared_expert, native_moe | tolerance |
| `rope_parity` | rope, native_rope | 3e-3 (table path bitwise) |
| `gr_parity` | fused_gr, gr; mode 2 = native_gr_norm, native_gr_postops, native_bf16 | multi ≡ single bitwise |
| `cvec_parity` | cvec, fused_gr | tolerance |
| `gdn_parity` | gdn.cu only (canonical) | 1e-5 / 1e-6 |
| `qsa_parity` | qsa, native_qsa_indexer, rope, qsa_select | ids exact, scores 1e-6, attention 1e-5, batch ≡ sequential |
| `qsa_prompt_attn_parity` | qsa, qsa_decode_attn, qsa_prompt_attn | ≤ 1e-4·scale |
| `kv_q8_parity`, `kv_q4_parity`, `kv_stream_parity`, `kv_hybrid_parity` | kv_q8, kv_q4, kv_stream, decode/prompt attention | bitwise / < 5e-3 |
| `ple_parity` (fixtures absent → fails) | ple, native_ple_postops, native_mmvq, ngram | per stage ≤ 1e-4, native ≤ 2e-6 |
| `ple_q5_parity`, `ple_fp8_parity` | ngram (CPU decode) | 1e-6 / bit-exact |
| `sampler_parity` ×3 | sampler (split, one_block, old) | 18 exact fixtures |
| `qsa_select_bench`, `tests/hip/*` | qsa_select (HIP), router_fast, native_qsa_score, prompt_attn_wmma | bitwise / tolerance |
| `tests/cuda/prefill_mmq_kquant_test`, `tests/hip/prefill_mmq_parity` | prefill MMQ | vs CPU FP32 |
| CPU `expert_parity`, `expert_multi_test`, `pool_test` | CPU expert kernels | vs scalar |

**No test at all:**
- `fused_gdn.cu`, the production decode GDN. `gdn_parity` covers only the canonical semantics.
- `verify_kernels.cu`: all 23 kernels, including the T-token GDN, commit, rollback, `resident_plan` and the MTP
  helpers.
- `native_router`, `native_moe`, `native_qsa`, `native_ple_postops` and `native_gdn`, outside the absent bench/micro
  suite (`native_moe` is reached indirectly by `shared_expert_parity`).

### E.3 Oracle vectors and golden outputs that exist

| Asset | Content | Use on Metal |
|---|---|---|
| `src/kernels/ple_oracle_vectors.inc` (105 lines) | 6 hash cases (token, prev → 16 rows); table constants (`kTableFileSize` 28,800,138,432); probe rows {0, 12345, 320001535} with head/tail values and `kProbeSum`; `dense.bin` offsets of the PLE key/value | the CPU hash and decode tests first (exact) |
| `sampler_parity.cpp` fixtures | 18 cases with exact expected tokens, plus `host_philox_uniform` (L134-159) | port first: pins the Philox variant, tie rules and FP64 tail |
| `tools/iq_fixture.py` + gguf-py | IQ dequant/gemv fixtures for `iq_parity` | restore gguf-py, then run unchanged |
| `tests/data/native_experts` | native expert layout | `expert_layout_test` |
| Host references inside each `*_parity.cpp` | scalar / FP64 implementations of every canonical and native kernel | the main asset: split into a backend-neutral reference and a backend launch half |
| `tools/metal/tf_compare.py` (retained dump format from the upstream baseline) | logits dump reader: int32 `[n_vocab, rows]`, then f32 rows | the KL gate |
| `serve/chat_golden.json` | chat-template golden outputs, no runner | write a runner |
| Dump hooks | `--dump-logits`, `--dump-layers` (session loop only), `--dump-halves`, `--dump-routing`, `STRATA_DUMP_FIRST_LOGITS`, `STRATA_STATE_HASH`, `STRATA_CKPT_REREAD`, `STRATA_LOGPOS(_TOPK)` | golden comparisons Metal vs CUDA vs llama.cpp |
| **Missing** | `ple_in.bin`/`ple_out.bin` (ggml capture), `gen_ple_oracle.py`, the bench/micro ggml-oracle suite | regenerate from llama.cpp 3cf03257. On macOS, build it with ggml-metal and use it as a second reference |

### E.4 Host tests to keep from day one

These are backend-independent and should run on macOS unchanged, or with small guards:
- GGUF: `gguf_reader_test`, `gguf_split_test`.
- Conversation cache: `conversation_cache`, `memory`, `snapshot` and `validation`. `transfer` is currently
  `NOT APPLE`.
- Experts: `file_expert_source_test`, `expert_layout_test`, `native_dense_ple_key_test`.
- Spec and drafting: `suffix_drafter`, `controller`, `draft_policy`, `conv_cache`, `coupled_draft`, `bf16_bits`.
- PLE: `ple_reader_test`, `ple_q5_parity`, `ple_fp8_parity`.
- CPU expert scalar references, `platform_memory_test`, and the Python suites (`tools/test_*.py`: 21 files;
  `serve/test_*.py`: 7 files).

Guards and drops:
- Guard the unconditional AVX-512 flags (`CMakeLists.txt:784-789`) and `pool_stress` on x86.
- Rewrite `direct_file_async` for `F_NOCACHE`, or drop it.
- Drop `pinned_shared_test` and `expert_cache_per_layer_test` with the mechanisms they test.

### E.5 Metal kernel parity plan

Each Metal twin reuses the host reference and the exact pass criteria, including negative controls. Run these tests
with the bitwise kernels built in safe math, and record the GPU family and OS version in the output.

| Priority | Tests (existing → Metal twin) | Notes |
|---|---|---|
| P0 | `quantize_act` (scaled Q8_0, Q8_1), `elementwise` (`to_bf16`), `dequant_s2`, plus a new `metal_intrinsics` (shuffles, `simd_sum` order, ballot, char4 dot vs dp4a, half/bf16 conversion) | layout and intrinsic bugs |
| P1 | `iq_parity` / `iq_multi_parity` / `mmvq_multi_parity` (all `native_mmvq` types, multi-exact with a negative control), `native_bf16` (new: mmvf vs FP64), `dequant_bf16_test`, `native_expert_parity` (grouped GU/down) | restore gguf-py so `iq_parity` stops skipping |
| P2 | `router_top10_parity` (ids exact) + a new `native_router` test with the 65,536-row tie/NaN generator from `tests/hip/router_fast.cpp`; `sampler_parity` (18 fixtures, GPU part + CPU tail); `rope_parity` (table path bitwise); `shared_expert_parity`; `kv_q8_parity`; `cvec_parity` (optional) | spec ≡ greedy depends on the router and argmax tie rules |
| P3 | `gr_parity` (re-tiled `gr_down_multi`, multi ≡ single); **new** `fused_gdn` test (conv_l2/ab/step_norm vs `gdn.cu` semantics in FP64); **new** `verify_kernels` tests (`gdn_step_norm_multi` verify-then-commit ≡ sequential decode, `copy_indexed` rollback, `window_ids`, `mtp_select`, `row_top_prob`, `map_ids`, `resident_plan` vs the host planner); `qsa_parity` (batch ≡ sequential, re-tiled `block_topk_reg`); `qsa_prompt_attn_parity`; `ple_parity` (Part C host reference; recreate fixtures) | the untested production kernels need new tests here |
| P4 | `s2_expert_grouped_parity` (MTP S2 kernels), prefill grouped GEMM vs CPU FP32 (twin of `prefill_mmq_parity`, plus ggml-metal `mul_mm_id` as a second reference), GPU counting sort vs the host sort | largest kernels |

**A guard to add on the existing backends.** Add a "pretend 32 KiB" option to the CUDA/HIP build that caps the
eight `MaxDynamicSharedMemorySize` sites and forces the small-tile branches. The re-tiled kernel shapes can then be
validated on working hardware before the Metal port exists (Appendix 5 §6.5 step 6).

### E.6 Model-level gates (none is scripted upstream)

| Gate | Procedure | Metal pass criterion |
|---|---|---|
| C1: logits vs llama.cpp | same token ids via `--tokens`; `--dump-logits`, `--dump-layers` / `--dump-halves` for bisection (dump layout in A.2 rule 6) | first gate, on a T = 1 path before any optimisation; add `--dump-layers` to the verify window |
| KL | `docs/UNSLOTH_Q4.md:170-205` with `STRATA_LOGPOS(_TOPK)`; the paper reports KL 0.022 | KL to llama.cpp no worse than the CUDA build of the same pack |
| Spec ≡ greedy | `--spec 0` vs `--spec 4`, `--spec-oracle`, `--spec-corrupt N` | identical token streams; any T-dependent reduction order breaks it (known CUDA exception #152) |
| Determinism | N = 8 repeats of the `DETAILS.md` prompt | **1 distinct answer with no flags** (CUDA needs `--adapt-swaps 0 --pcie-frac 0` and still gives 1-2 of 4) |
| Conversation cache | `tools/conversation_cache_*.py`, `STRATA_STATE_HASH` cold vs `STRATA_CKPT_REREAD` | equal hashes; state is read from shared memory after completion |
| Prefill vs decode | prefill a prompt, then compare the next-token logits with token-by-token decode of the same prompt | tolerance and KL, not bitwise (CUDA is not bitwise either) |
| Golden first window | `STRATA_DUMP_FIRST_LOGITS` vs the CUDA build | rel-L1 and top-5 overlap |

### E.7 Metal-only tests

These have no CUDA twin:
- `metal_device_selftest`: family, `threadExecutionWidth == 32`, `maxThreadgroupMemoryLength`, `maxBufferLength`,
  `recommendedMaxWorkingSetSize`. It skips with code 77 when no GPU is present.
- `metal_shared_event_handoff`: 100+ rounds, CPU → GPU → CPU, with changing payloads.
- `metal_nocopy_alias`: anonymous and file-backed mmap. Skip 77 if file-backed is refused.
- `metal_residency_set`.
- A `maxBufferLength` split test: a synthetic arena just over the limit with per-slot patterns.
- An in-flight visibility diagnosis.
- A command-buffer timeout test for the longest prefill chunk.

### E.8 Bring-up order

1. Host suite green on macOS, with a `macos-15` arm64 CI job.
2. The Appendix 4 §8.2 ladder for the CPU-exact parts: the PLE hash and decode, Philox, `coupled_draft`,
   `suffix_drafter`.
3. P0 → P4 kernel twins (E.5).
4. T = 1 end-to-end with the dump hooks, then gate C1 and KL.
5. T = 2..8 verify window + commit, then spec ≡ greedy.
6. MTP: replay a recorded window on the CPU and on Metal, then compare draft ids, `row_top_prob` and the K/V written.
   Then measure acceptance under `--spec 4 --spec-min-p 0.5`.
7. Prefill: grouped GEMM tests, then prefill vs decode KL, then determinism and the conversation-cache gates.

---

## F. Metal implementation rules

These rules are distilled from Sections A-E and the appendices. "Unverified" marks Apple behaviour that was not
checked on hardware.

### F.1 Libraries and math modes

1. **Build two libraries**, matching the CUDA fast-math split (`CMakeLists.txt:288-308`):
   - A fast-math library for the transcriptions of the `native_*` kernels and the prefill GEMM. These are tested with
     tolerances.
   - A safe-math library for everything else: fused GR/GDN, quantizers, KV quantizers, router, sampler, `verify_*`
     and the prefill elementwise kernels.
2. **Setting the mode.** Use `MTLCompileOptions.mathMode = MTLMathModeSafe` on macOS 15+, or `fastMathEnabled = NO`
   on earlier releases.
3. **Within-backend bitwise, cross-backend tolerance.** Strata's bitwise contracts compare a kernel with itself under
   different shapes: decode vs verify, multi vs single, batch vs sequential.
   - Metal must keep those properties. Share one kernel, or the same per-row operation order, across T, columns and
     batches.
   - Bit-equality with CUDA is not achievable in general. nvcc contracts `a*b+c` into FMA in non-fast-math files
     (default `--fmad=true`) at places only SASS shows.
   - The exceptions are the CUDA kernels that pin every operation (`__f*_rn`, integer math, RNE converters). Their
     transcriptions can be bitwise to CUDA.

### F.2 Arithmetic

- **FMA.** Use `fma()` exactly where CUDA writes `__fmaf_rn`. Where CUDA writes `__fmul_rn`/`__fadd_rn`, use separate
  operations with contraction off (`#pragma METAL fp contract(off)`; availability unverified).
- **Division.** Use `precise::divide` wherever the result feeds a quantizer or a comparison: `amax/127`, `x/d`, KV
  `x/sf`, and the router renormalisation.
- **Transcendentals.** Use `precise::exp`/`log` where CUDA uses `expf`/`logf`: the prefill route, `gate_attn`, the
  sampler tail. `fast::exp` is fine where CUDA uses `__expf`, because those paths are tolerance-tested.
- **Rounding.** `round()` = `roundf` (half away from zero), and `rint()` = half to even.
- **Integer dots.** `char4` → `float4` `dot` or chained fma is exact while partial sums stay below 2^24, which holds
  for every Strata block. Copy the SWAR `vsub4`/`vsubss4`/`vcmpne4` from `hip_compat/intrinsics.hpp:55-79`. Replace
  `__byte_perm` with shifts and masks.
- **Activation epilogue.** Keep CUDA's operation order: `d_w·d_x·sumi`, then the min/offset terms.

### F.3 No FP64

| CUDA FP64 use | Metal answer |
|---|---|
| Sampler tail (≤ 64 candidates) | CPU. Sampled decoding already syncs, so there is no new sync |
| Canonical references (`gdn.cu`, `qsa.cu`, `ple.cu`, `router_top10`, shared-expert canonical forms) | CPU oracles only |
| `quantize_q8_0` (`rint((double)x/(double)d32)`) | fp32 `rint` plus an exact fma-residual tie check (B.2.4), or use the scaled contract |
| RoPE table | built on the CPU in f64 and uploaded. The table path is bitwise to `rope_neox`; fast-math trig loses about 1.4e-3 rad at 32K |

### F.4 Reductions and simdgroups

- **Width.** Assert `threadExecutionWidth == 32`, and `threads_per_simdgroup == 32` in the kernels.
- **Lane and warp index.** Take them from `thread_index_in_simdgroup` and `simdgroup_index_in_threadgroup`, not from
  the thread position.
- **Butterflies.** Use explicit `simd_shuffle_xor` butterflies (the `o = 16, 8, 4, 2, 1` order) wherever a bitwise
  contract exists. The order of `simd_sum`/`simd_max` is unspecified (unverified), so use them only in
  tolerance-tested kernels.
- **`block_sum`** becomes `threadgroup float[32]` plus a barrier. Check `maxTotalThreadsPerThreadgroup` per pipeline
  for the 512- and 1024-thread kernels (`gdn_step_norm`, `gr_norm`, the PLE gate and `rms_rep`, the greedy sampler).
- **Full participation.** All 32 lanes must reach every shuffle. The native router's early return before the barrier
  is safe only with exactly 32 threads per token, so dispatch it that way.
- **Ballot.** `simd_ballot` returns a 64-bit vote; use its low 32 bits.
- **Sub-width shuffles** (width 8) exist only in the cpu-order kernels, which are not ported.

### F.5 Threadgroup memory (32 KB)

These kernels need re-tiling:

| Kernel | CUDA footprint | Re-tile |
|---|---|---|
| `gr_down` | 40 KB (`xn[10240]`) | the norm writes `xn` to device memory (as `gr_norm_split` already does, bitwise); stream 32 KB tiles; keep the `j = lane + 32q` chunk order and the xor butterfly |
| `gr_down_multi`, `gr_down_staged`, `gr_down_v3` | 40-80 KB at T = 8 | 640-wide tiles, or slice the tokens (outputs are per token); port only `multi` |
| `block_topk_reg` | about 32 KB of histograms | fewer bins per pass, or an extra pass |
| `block_scores_tc` | `(QT·QS + NB·KS)·4` | smaller query or key tiles; fp32 `simdgroup_matrix` |
| `prompt_attn` | struct above 32 KB | fewer queries per threadgroup |
| `sampler_split_merge`, `coupled_merge` | 33,792 B | 61 × 64 × 8 = 31,232 B for the vocabulary of 248,320 |
| `qsa_attend` (canonical) | `exit(1)` above the limit | not ported |

The S2 grouped stage (23,040 B) and the sampler bitmap (31,040 B) fit.

### F.6 Memory and I/O

- **Expert arena.** `mmap` the pack and wrap it with `newBufferWithBytesNoCopy`, with 16 KiB alignment of address
  and length. Use one buffer per layer under `maxBufferLength`, add them to an `MTLResidencySet` (macOS 15+), and
  address experts by offset.
  - Whether file-backed mappings may be wrapped NoCopy and made resident is unverified. Test it with
    `metal_nocopy_alias`.
- **Shared arena files.** The 4096-B header (`pinned.cu:41`) is not 16 KiB-aligned. Wrap the whole mapping and use
  offset 4096, or move to a 16 KiB header.
- **Memory budget.** Use `recommendedMaxWorkingSetSize`, `currentAllocatedSize` and `os_proc_available_memory()`
  instead of `cudaMemGetInfo`.
- **PLE table.** Read with `F_NOCACHE` (+ `F_RDAHEAD`) instead of `O_DIRECT`, or through the page cache. Use
  `madvise(MADV_WILLNEED / MADV_DONTNEED)` with `getpagesize()` = 16 KiB. `mlock` works within its limits.
- **Unaligned blocks.** Q3_K (110 B) and Q8_0 (34 B) blocks are only 2-B aligned. Use `ushort` or byte loads, or
  restage into aligned threadgroup words as `gu_grouped_t` does. Do not issue `uint`/`uint4` loads at misaligned
  addresses.
- **Codebooks** (IQ grids, `kvalues_iq4nl`). Copy them into threadgroup memory or keep them in registers. Divergent
  `__constant__` lookups cost 2.12× on CUDA (`s_gemv.cu:20-31`), and `constant` address-space behaviour on Apple is
  to be measured.

### F.7 Synchronisation and encoding

- **No GPU spin.** Never spin a GPU thread on a host-written flag. CPU ↔ GPU handoffs happen at command-buffer
  boundaries, with `MTLSharedEvent` signal and wait values: monotonic values give the `≥ ring` semantics. GPU writes
  are read by the CPU only after completion or a signal.
- **Ordering.** Queue order replaces stream order (window → commit → next window). Use a second queue only for the
  MTP overlap, synchronised with events.
- **Per-step values live in buffers**, never in encoded constants, if command buffers or ICBs are reused (the CUDA
  "cloned literal" rule).
- **Fixed addresses.** Bump-carve all scratch once.
- **Pipelines.** Build every `MTLComputePipelineState` at startup (this replaces eager module loading).
- **Window encoding.** Re-encode each window (about 1,500 dispatches, D.1.2), or build an `MTLIndirectCommandBuffer`
  per T. Measure first.
- **Errors and timing.** In debug builds use `errorOptions = .encoderExecutionStatus`, and check every command
  buffer's status. Get timestamps from `MTLCounterSampleBuffer` instead of `%globaltimer`.

### F.8 Conversions and formats

- **bf16.** Use integer RNE on `as_type<uint>` with the NaN quiet rule (`| 64`), as `bf16_bits.hpp` does. Decode with
  `as_type<float>(uint(b) << 16)`. Native `bfloat` needs MSL 3.1 and is not required.
- **fp16.** `half(x)` is RNE. Verify subnormals and overflow against `f16_bits.hpp`, keep the explicit ±65504
  saturation (`hf_sat`) in the prefill SwiGLU, and keep the "fp16-rounded scale" semantics of the KV quantizers.
- **Q5_0 expert downs** (Orca packs). Add the format to the grouped down kernel. CUDA has no GPU kernel for it, and
  sends those layers to the CPU.

### F.9 Determinism rules for the verify window

- Make every kernel's per-row arithmetic independent of T and of the number of columns. Keep `g_multi_exact` (every
  column bitwise equal to a 1-column call) for `native_mmvq_multi` and the grouped expert kernels.
- Build the expert grouping on the GPU in first-appearance order (`group_resident` / `resident_plan`), so the
  grouping does not depend on timing.
- Combine in fixed k order, with the first product unfused.
- Keep the router and argmax tie rules: lowest id first, NaN → `-FLT_MAX`.

---

# Appendices

The appendices of the original report were the five detailed research parts, unchanged except that their headings
were demoted by one level. They are identical to the audit files below, so this copy does not repeat them. Inside
this report, "Part 0N" and "Appendix N" mean the file in row N, and section numbers are the same (for example,
"Part 05 §4" and "Appendix 5 §4" are section 4 of K5).

| Appendix | Audit file |
| --- | --- |
| 1 | [K1-gr-router-shared-rope-cvec.md](K1-gr-router-shared-rope-cvec.md): hyper-connections (GR), router, shared expert, elementwise, RoPE / M-RoPE / YaRN, control vectors |
| 2 | [K2-qsa-kv-gdn.md](K2-qsa-kv-gdn.md): QSA attention, KV cache formats and streaming, GDN (Gated DeltaNet) |
| 3 | [K3-quant-and-matvec.md](K3-quant-and-matvec.md): quantization formats, mat-vec, dequant and activation-quant kernels |
| 4 | [K4-ple-mtp-head-sampler.md](K4-ple-mtp-head-sampler.md): PLE n-gram embedding, token embedding and output head, MTP draft layer, sampler and verify-window kernels, prompt-lookup drafting |
| 5 | [K5-orchestration-vram-tests.md](K5-orchestration-vram-tests.md): decode/prefill orchestration, discrete-VRAM assumptions, test strategy |
