> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Part 02: QSA attention, KV cache formats and streaming, GDN (Gated DeltaNet), for the Metal port

Source: `/tmp/strata-upstream` (read-only clone). Paths below are relative to that root.

- Kernels live in `src/kernels/cuda/`. Prompt-path kernels live in `src/prefill/kernels.cu`.
- Orchestration:
  - Decode: `src/core/layer.cpp`.
  - Speculative verify and commit: `src/core/verify.cpp`.
  - Prefill: `src/prefill/prefill.cpp`.
  - MTP drafter: `src/core/mtp.cpp`.
  - CLI and defaults: `src/program/generate.cpp`, `setup.py`.
- Headers: `include/strata/kernels/*.hpp` and `include/strata/core/layout.hpp`.
- `src/core/layer.cpp` stores many statements on very long physical lines. Line numbers there point at the statement's line, but they can be off by a few.

Model: Qwen3.8-Flash-Next. 48 layers. Layer `l` is QSA when `l % 4 == 3`, which gives 12 QSA layers; the other 36 are GDN.

---

## 0. Key constants

| Constant | Value | Where |
|---|---|---|
| n_embd | 2560 | `include/strata/core/layout.hpp:27-53` |
| layers / QSA interval | 48 / 4. QSA iff `l%4==3` (L3, L7, ..., L47). 12 QSA + 36 GDN. | layout.hpp L30, L63-65 |
| GDN head state size S | 128 (k-dim = v-dim = 128) | layout.hpp |
| GDN h_k (q/k heads) | 16 | layout.hpp |
| GDN h_v (v heads) | 48. q/k head for v-head h is **`h % 16`** (modulo, not `h/3`). | fused_gdn.cu:30, gdn_parity |
| GDN d_conv | 4 (3 history taps + current) | layout.hpp |
| GDN conv channels | 10240 = q[0,2048) + k[2048,4096) + v[4096,10240) | layout.hpp |
| GDN value_dim | 6144 = 48*128 | layout.hpp |
| QSA n_head / n_head_kv / head_dim | 24 / 2 / 256. GQA group 12. q head h uses kv head **`h/12`**. | `include/strata/kernels/qsa.hpp:94-117` |
| QSA n_rot | 64. Partial NEOX rotary on the first 64 dims, pairs (i, i+32), i<32. Applies to head_dim 256 and to the indexer's 128-dim keys/queries. | qsa.hpp:94-117 |
| Indexer heads / dim | 4 query heads x 128. **One shared 128-dim key head** ("Round 194", layout.hpp:7-9). | layout.hpp, qsa.hpp |
| idx_block r | 4 cells per pooled key | qsa.hpp |
| idx_top_k | 2048 cells of budget | qsa.hpp |
| Selection width | `min(n_kv, idx_top_k + idx_block - 1) = min(n_kv, 2051)`. Identity for n_kv <= 2051. | qsa.hpp:133-136 |
| page_size | 4 cells (1 page = 1 indexer block) | qsa.hpp:108-113 |
| rms_eps | 1e-6 (QSA norms, indexer norms, GDN norms; prefill passes EPS) | qsa.hpp:120 |
| rope freq_base | 1e7. `theta_scale = powf(1e7, -2/64)`. | qsa.hpp:125, native_rope.cu:134 |
| Step buffer | int32[4] = {pos, n_kv=pos+1, n_bid=n_kv/4, width} | qsa.hpp:156-162, qsa.cu:699-714 |
| kTopkMaxCells | 32768. Only caps the canonical `topk_512` path. `cap = qsa_selection_width(32768) = 2051`. | qsa.hpp:279-282 |
| Attention scale | 1/sqrt(256) = 1/16 | qsa_decode_attn.cu:227 |
| kVerifyMaxT | 8 tokens per verify window. setup writes `--spec 4`. | verify_kernels.hpp, verify.cpp:207 |
| MTP window | `--mtp-window` 32768 cells of dense attention | generate.cpp:352 |
| Prefill chunk | `--prefill auto`: largest of {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256} fitting the budget. **Capped at 8192** unless `auto:16384`, `auto:32768`, or `STRATA_PREFILL_AUTO_MAX`. Fallbacks: 2048 (split, L1361-1375, L1441), 1024 (no cache, L3949/L3964). | generate.cpp:326-332, 3748-3765 |

Tensor shapes (GGUF `[in, out]` view, `src/core/layout.cpp` L65-106):

- GDN:
  - `attn_qkv [2560,10240]`, `attn_gate [2560,6144]` (z), `ssm_out [6144,2560]`.
  - `ssm_conv1d [4,10240]` F32.
  - `ssm_alpha [2560,48]` and `ssm_beta [2560,48]`, both BF16.
  - `ssm_a[48]`, `ssm_dt.bias[48]`, `ssm_norm[128]`, all F32.
- QSA:
  - `attn_q [2560,12288]`: per head 512 = 256 q followed by 256 gate, interleaved per head.
  - `attn_k [2560,512]`, `attn_v [2560,512]`, `attn_output [6144,2560]`.
  - `attn_q_norm[256]`, `attn_k_norm[256]`.
  - `indexer.q_proj [2560,512]` BF16, `indexer.k_proj [2560,128]` BF16.
  - `indexer.q_norm[128]`, `indexer.k_norm[128]`, both F32.

---

## 1. QSA layer dataflow

### 1.1 Common building blocks

- **Step buffer.** Every per-token scalar is read from device memory: `step[4]`, `pos_dev`, or the per-row `steps[q*4]`. This is so a CUDA graph can replay with constant arguments (qsa.hpp:139-162).
  - Metal: write these into a shared-storage `MTLBuffer` per token. There is no H2D copy on unified memory.
- **Causality is implicit.** Each query's `n_kv = pos+1` bounds both the selection and the attention. There is no attention mask tensor.
  - In verify and prefill, all T tokens' K/V and indexer keys are appended **before** any token's selection.
  - Correctness relies on per-row steps (`b > n_bid` returns; ids only `< n_kv`).
- **V is never normalized or rotated** (only K gets `attn_k_norm` + RoPE).
- **Output gate.** `attn_out[h,d] *= sigmoid(q_full[h*512 + 256 + d])`. This is the second half of each head's 512-wide projection.

### 1.2 Decode (one token). `qsa_layer`, `src/core/layer.cpp:848-991`

Production path is `--native`, `g_fast_select`, `g_fast_attn`; the KV format is fp16 or int8.

| # | Step | Kernel(s) / call | File:line |
|---|---|---|---|
| 0 | Fill the host step {pos, pos+1, (pos+1)/4, min(pos+1, 2051)} and per-head positions (pos_base+pos); upload | `qsa_step_fill` + 2x cudaMemcpyAsync | layer.cpp:880-888, qsa.cu:699-714 |
| 1 | Indexer raw key: BF16 GEMV x[2560] -> idx_raw[128] (fp32 out). Taken **before** any norm. | `bf16_gemv_fp32_mmvf` (native_bf16, other slice) | layer.cpp:889-895 |
| 2 | K and V projections -> kcur[2,256], vcur[2,256] | `gemv_quantized` (native_mmvq, other slice) | layer.cpp:896 |
| 3 | K: RMS-norm(attn_k_norm, eps 1e-6) per kv head, then NEOX RoPE(64) at pos_base+pos | native: `native_qsa_rms_norm_weighted` (native_qsa.cu `norm<256>` L45) + `native_rope_apply`. Canonical: `rms_norm_weighted` (elementwise.cu:88) + `rope_neox_apply` (rope.cu:97, f64-built table). | layer.cpp:849-857, 896 |
| 4 | Optional FWHT, then KV append at cell pos. k8v4: FWHT(V), then q8 K + q4 V. q4: FWHT K and V, then q4 append. int8: q8 append. fp16: f16 append. Each writes the host mirror if streaming. | `fwht256_inplace_cuda` (kv_q4.cu:25), `kv_append_q8_step` (kv_q8.cu:30), `kv_append_q4_step` (kv_q4.cu:101), `kv_append_step` (qsa.cu:122) | layer.cpp:897-915 |
| 5 | Indexer append: fp16-round raw key, write tail slot; on a block completion, mean -> RMS(k_norm) -> RoPE at the block's first cell -> pooled row | native: `native_qsa_indexer_append` (native_qsa_indexer.cu:46). Canonical: `indexer_key_append` (qsa.cu:155). | layer.cpp:916-921 |
| 6 | Q projection -> q_full[24*512]; split the first 256 of each 512 into qcur[24,256] (strided 2-D memcpy) | `gemv_quantized`, `cudaMemcpy2DAsync` | layer.cpp:922-932 |
| 7 | q: RMS-norm(attn_q_norm) per head + RoPE(64) at pos | same as step 3 | layer.cpp:932 |
| 8 | Indexer query: BF16 GEMV -> q_idx[4,128]; RMS-norm(indexer.q_norm) per head; RoPE(64) at pos | `bf16_gemv_fp32_mmvf` + norm/rope | layer.cpp:933-940 |
| 9 | Block scores: `score[b] = sum_h relu(dot(key_b, q_idx_h))`, `b in [0, n_bid]` | `qsa_block_scores` -> `block_scores_multi_kernel` (qsa_select.cu:606), nq=1 | layer.cpp:941-942 |
| 10 | Top-k over blocks (cell-weighted) -> ids[width] ascending | `qsa_block_topk` -> `block_topk_reg_kernel<33>` (qsa_select.cu:507) if max_blocks <= 33,792, else `block_topk_kernel` (L57) | layer.cpp:943 |
| 10' | Canonical alternative (g_fast_select=false): per-cell double-precision scores + 32-pass binary-lifting top-k | `qsa_index_kernel` (qsa.cu:249) + `topk_kernel` (qsa.cu:307) | layer.cpp:945 |
| 11 | KV streaming only: make the selected pages resident | `qsa_kv_resolve` -> kv_stream.cu kernels | layer.cpp:948 |
| 12 | If kv_rot: FWHT(q) so that <Hq,Hk> = <q,k> | `fwht256_inplace_cuda` | layer.cpp:949 |
| 13 | Sparse attention over the `width` selected cells: split-K over 64-cell chunks + merge | `qsa_decode_attn_step` -> `attn_chunk_kernel<mode>` (qsa_decode_attn.cu:84) + `attn_merge_kernel` (L184) | layer.cpp:950-952 |
| 13' | Canonical alternative (g_fast_attn=false): gather to fp16 scratch + one block per head | `kv_gather_kernel` / `kv_gather_q8` / `kv_gather_q4` + `qsa_attend_kernel` (qsa.cu:439) | layer.cpp:953-974 |
| 14 | If kv_rot or hybrid: FWHT(attn) back. H is self-inverse. | `fwht256_inplace_cuda` | layer.cpp:976 |
| 15 | Gate: `attn32 = attn * sigmoid(q_full[h*512+256+d])` | native: `native_qsa_gate_apply` (native_qsa.cu:69, fp32). Canonical: `qsa_gate_apply_f32` (qsa.cu:793, double sigmoid). | layer.cpp:980-984 |
| 16 | Output projection [6144] -> [2560] | `gemv_quantized` / `s_gemv_q8k_split` | layer.cpp:986-989 |

### 1.3 Selection semantics (exact; required for parity of the ids)

Source: `qsa_select.cu:28-191`. The canonical `qsa.cu:249-401` implements the same rule over cells.

1. **Scores per block** `b = 0..n_bid`, where `n_bid = n_kv/4` is the number of completed blocks.
   - Key: `pooled[b]` for `b < n_bid`. For `b == n_bid` (the incomplete tail block, possibly empty) the key is the **`dead`** row: cell 0's pooled key, see section 3.7.
   - Score: `sum over h=0..3 (in order) of max(0, dot(key, q_idx_h))`. There is no softmax, no 1/sqrt(d), no bias.
   - Production dot (`block_scores_kernel` L44-51, `block_scores_multi` same order):
     - Each lane takes 4 consecutive dims and computes `k.x*q.x + k.y*q.y + k.z*q.z + k.w*q.w` (nvcc may contract to FMAs).
     - Then a butterfly `__shfl_xor` sum over 32 lanes.
   - If `b == n_bid && n_kv % 4 != 0`, add **`+1e9f`** so the tail block always wins (llama `set_input_qsa`).
   - Blocks `b > n_bid` return without writing. This lets the grid be the constant `max_blocks`.
2. **Identity shortcut.** If `n_kv <= width` (i.e. n_kv <= 2051), `ids = 0..n_kv-1`.
3. **Cell-weighted radix select** (`block_topk_kernel` L57-191; `block_topk_reg_kernel` gives the same ids with keys in registers):
   - `order_key(s)`: `v = s + 0.0f` maps -0 to +0. NaN maps to 0. Then `b ^ (sign ? 0xffffffff : 0x80000000)`, i.e. a total order.
   - Block weight is 4 for `b < n_bid` and `n_kv - 4*n_bid` (0..3) for the tail.
   - Four 8-bit radix passes find the largest threshold `thr` such that `sum of weights with key >= thr` is at least `width`.
   - `eq_budget = width - (weight strictly above thr)`.
   - Emit, walking blocks ascending:
     - All cells of blocks with key > thr.
     - For blocks with key == thr, cells in ascending block and cell order until eq_budget runs out. **A threshold block may be emitted partially** (only its first cells).
   - Output: exactly `width` cell ids, **ascending**.
4. **Threshold and dispatch** (`qsa_select.cu:745-786`):
   - `TK_T=1024`, `TK_PER=33` give the register kernel when `max_blocks <= 33,792` (about 135K cells). Otherwise the memory kernel runs.
   - `STRATA_TOPK_OLD=1` forces the memory kernel.
   - HIP has its own counted rule.

### 1.4 Decode attention kernel (production): `qsa_decode_attn.cu`

**`attn_chunk_kernel<KV_MODE>` (L83-182)**

- Launch: `grid = (ceil(cap/64), n_head_kv=2 [, n_q])`, 256 threads.
- One block handles one 64-cell chunk for one KV head and all 12 of its q heads.
- Static shared memory: `sq[12][256]` fp32 (12 KB) + `sp[12][64]` + `srow[64]` (long long). About 15.9 KB in total.
- Steps:
  1. Load q for the 12 heads.
  2. Resolve each cell's pool row as `(page_table[cell/4]*2 + kvh)*4 + cell%4`. Page -1 means not resident: the row is -1 and the cell is masked.
  3. Scores: each warp takes cells `warp, warp+8, ...`. Each lane holds 8 dims.
     - The dot is computed as a plain `k8[0]*qa.x + ... + k8[7]*qb.w` expression (nvcc contracts it), then a shfl_xor warp sum, then `*scale`.
  4. Per head (warp w takes heads w and w+8):
     - `m` = max over the 64 cells, computed over all 64 sp entries, including the -FLT_MAX fill.
     - `e = __expf(s - m)`, with masked cells set to 0.
     - `l = warp_sum(e)`.
  5. PV: thread t owns dim t. For each cell in ascending order, `acc[h] = fmaf(p[h][c], v, acc[h])`. The weights are **unnormalized** (the exp-sum is normalized in the merge).
  6. Write `part_acc[slot][h][256]`, `part_m`, `part_l`, where `slot = kvh*n_chunks + chunk`.
- Empty chunk: `m = -FLT_MAX`, `l = 0`.

**`attn_merge_kernel` (L184-207)**

- Launch: grid `(24 [, n_q])`, 256 threads.
- Computes `M = max_c m_c`, then sequentially over chunks `w = __expf(m_c - M)`, `L = fmaf(l_c, w, L)`, `acc = fmaf(acc_c, w, acc)`.
- Output: `attn = L > 0 ? acc/L : 0`.

**Scratch and modes**

- Scratch size: `chunks*24*258 + 64` floats per query (L251-254).
- KV mode selection (L219-220, L263-264): `k_q4` gives 2 (q4). `k_q && v_q4` gives 3 (k8v4). `k_q` gives 1 (int8). Otherwise 0 (fp16).
- Dequantization:
  - int8: `code * f16 scale` per 64-dim group.
  - q4: `(nibble - 8) * f16 d` per 32-dim block, with element t in the low nibble and t+16 in the high nibble of byte t.
- Determinism: the reduction order is fixed, so results are deterministic and run-to-run bitwise. The same kernel serves decode, verify (`qsa_decode_attn_batch`, L211-249) and MTP. That is why verify matches decode bitwise.

### 1.5 Verify (speculative window of n <= kVerifyMaxT = 8 tokens). `src/core/verify.cpp:545-664`

The verifier requires the native path (`layer_verify_compatible`, layer.cpp:448-458). Per QSA layer, for the window's tokens `[tb, te)` (a window may be split into groups `grp`, e.g. for a layer split):

1. **Activations and projections:**
   - `native_quantize_q8_1(xm)`: activations quantized to Q8_1 for MMVQ, as in llama.cpp.
   - `bf16_gemv_fp32_mmvf_multi` gives idx_raw for n rows (L569-571).
   - `native_mmvq` gives kcur and vcur for n rows (L573-574).
2. **Batched path flag.** `qb = dec_batch && n>1 && native_qsa && native_rope && !kv_q4` (L566). With it, norm+RoPE run over `n*2` rows with per-row positions `pos_k` (L575). Otherwise they run token by token.
3. **FWHT** of K and V (rot), or of V only (hybrid) (L577-582).
4. **Indexer tail snapshot.** `copy_from_mapped(tail_snap_, st.idx_tail, TS)` (L583). This saves the 3x128 tail **before** the window's appends.
5. **KV append per token** `t` with its own step (L584-600): `kv_append_q8_step` / `kv_append_q4_step` / `kv_append_step`. Rejected tokens' KV cells are simply overwritten by the next window.
6. **Indexer append per token:** `native_qsa_indexer_append(idx_raw+t*128, step_+t*4+kStepPos, 0, ...)` (L601-605). This must be sequential per token because of the tail slot logic.
7. **Queries:**
   - `native_mmvq` gives q_full for n rows.
   - The 2-D memcpy split gives qcur.
   - norm+RoPE run with per-row positions.
   - FWHT if rot.
   - The indexer q goes through BF16 GEMV (multi), then norm+RoPE (L606-635).
8. **Selection:** `qsa_block_scores(..., n, ...)` (multi kernel since n <= 8) and `qsa_block_topk(..., n, ...)` (L638-641). The rows of different queries are independent.
9. **Attention:** `qsa_kv_resolve` (L644), then `qsa_decode_attn_batch` with `grid.z = n` (L646-648).
10. **Output:** FWHT back (L650), gate (`native_qsa_gate_apply` over n*24 rows, L651-660), then Q8_1 + MMVQ output projection (L662-663).

**Commit** (graph `capture_commit`, verify.cpp:950-1010; host `Verifier::commit(n_keep)`, L1320-1347):

- Host side:
  - `h_commit_[0] = n_keep`.
  - `h_commit_[1] = n_keep-1`.
  - `h_commit_[2+t] = last_pos0_ + t` for `t < n_keep`, else `-1`.
- Per QSA layer: restore `idx_tail` from `tail_snap_` (L987), then replay `native_qsa_indexer_append` for each kept token at position `commit_[2+t]` (L990). A `-1` position makes the kernel return immediately.
- Why this is sufficient:
  - Replay rewrites accepted complete-block rows. Rejected writes after that prefix are ignored; `pooled[n_bid]` can remain stale when the accepted prefix ends in a partial block, because selection reads `dead` for its tail block rather than `pooled[n_bid]`. The T=8 fixture accepts positions 1..3 and rejects position 7: replay rewrites pooled row 1, while rejected output in spare row 2 lies beyond the accepted `n_bid`.
  - The KV cache needs no fix-up.

### 1.6 Prefill (chunk of T tokens starting at p0, T <= 8192 by default). `src/prefill/prefill.cpp:1366-1560`

| # | Step | Kernel | File:line |
|---|---|---|---|
| 1 | K, V, Q_full projections (tensor-core GEMM via cuBLAS or MMQ) | `native_proj` | prefill.cpp:1377-1379 |
| 2 | idx_raw[T,128], q_idx[T,512] (BF16 GEMM) | `bf16_proj` | L1380-1381 |
| 3 | K: rms_rows over T*2 rows of 256, then rope (ld 512, pos p0+t) | `rms_rows_kernel` (prefill/kernels.cu:553), `rope_kernel<TAB>` (L566) | L1382-1383 |
| 4 | Streaming only (kv_mode 1): stage the whole layer's existing KV from the host copy into an identity-layout staging pool | `kv_stage_from_host` (kv_stream.cu) | L1389-1392 |
| 5a | k8v4: FWHT(V); int8 K append; q4 V append | `fwht256_kernel`; `kv_append_kernel` (prefill/kernels.cu:602, int8 branch); `kv_append_q4_batch_kernel` (kv_q4.cu:119) | L1395-1400 |
| 5b | q4: FWHT K and V; q4 append | as above | L1403-1408 |
| 5c | fp16 / int8: optional FWHT if `STRATA_KV_ROT`; `kv_append_kernel` writes the device pool (if resident), the host mirror and the staging pool | prefill/kernels.cu:602 | L1410-1412 |
| 6 | split_q (first 256 of each 512); rms_rows (T*24); rope (ld 6144); FWHT if rot | `split_q_kernel` (L586), `rms_rows_kernel`, `rope_kernel` | L1414-1417 |
| 7 | q_idx: rms_rows (T*4, 128); rope (ld 512) | same | L1418-1419 |
| 8 | Indexer append for the whole chunk: `native_qsa_indexer_append_batch` = `append_first` (only if p0==0), `append_blocks` (one CTA per completed block), `append_tail` (3 CTAs). `STRATA_INDEXER_PER_TOKEN=1` gives the old per-token loop. | native_qsa_indexer.cu:145, 171, 206 | L1420-1439 |
| 9 | Per `sel_batch` of queries: try `qsa_block_scores_tc` (3xTF32 mma, sm80+, **not bitwise** vs decode); fall back to `qsa_block_scores`; then `qsa_block_topk` | qsa_select.cu:195 + 438 (tail); fallback L28/L606; topk L57/L507 | L1441-1456 |
| 10 | Prompt attention for all T queries: `qsa_prompt_attn_batch`. int8 on sm80 uses `prompt_attn_i8_kernel` (cp.async). Turing or v1 uses `prompt_attn_kernel<1>`. fp16 uses `<0>`, hybrid `<3>`. q4 returns false. Fallback: `qsa_decode_attn_batch` in groups of `attn_batch` (32). | qsa_prompt_attn.cu:102, 416; dispatcher L975-1030 | L1543-1556 |
| 11 | FWHT back (rot or hybrid); `gate_attn_kernel` writes **fp16** gated output (uses `expf`, not `__expf`); output projection | prefill/kernels.cu:592 | L1558-1560 |

Diagnostics:

- `STRATA_IDX_FP16_CHECK` (L1477-1520) measures whether fp16 pooled indexer keys would select identically. This is relevant if Metal wants fp16 pooled keys.
- `STRATA_QSA_DUMP` (L1526-1540) dumps selections for `tools/qsa_locality.py`.

**Prompt attention kernel detail (`qsa_prompt_attn.cu`)**

- Constants: `HD 256`, `G 12`, `CH 32` cells per chunk, `THREADS 128` (4 warps), `QS 264`.
- Grid: `(n_q up to 65535 per launch, 2 kv heads)`. One block is one query token times one KV head. The M dimension is 12 heads padded to 16 rows.
- Q handling:
  - q is scaled by a power of two so that `max|q|` is near 2^14 (exact scaling).
  - It is then split into fp16 hi + lo parts. The score is `mma(hi,K) + mma(lo,K)`, i.e. fp32-like accuracy.
  - K is fp16, or int8 codes converted exactly to half via the `1024+c` mantissa trick (`i8x2_to_h2`, L67-73).
- Online softmax in the base-2 domain: `scale_log2 = log2(e)/16`, `exp2f`, running m, l and alpha per row.
- PV: P is split the same way (hi/lo with a 2^14 rescale), and the per-64-dim int8 scale groups are summed in a fixed order.
- Dynamic smem `sizeof(Smem<mode>)` is above 48 KB, using an opt-in attribute.
- Determinism: deterministic, FP32-level accurate, but **not bitwise** equal to the decode kernel. Turing's two k8 MMAs round differently from sm80's k16 (L41-45).

### 1.7 MTP drafter attention (not QSA-sparse)

- The MTP draft layer uses **dense attention over a sliding window** of the last `--mtp-window = 32768` cells. There is no indexer.
- `window_ids_kernel` (verify_kernels.cu:361-371) writes `ids[j] = start + j` with `start = max(0, n_kv - window)`, and **overwrites `step[3]` (width) = n_kv - start**.
- `dense_steps_kernel` (L296-305) builds steps with `width = n_kv`.
- Attention then runs through `qsa_decode_attn_batch` (`src/core/mtp.cpp` ~L440-505).
- The drafter has its own KV. When streaming, it is a ring of `window + 4*max_t + 64` cells (mtp.cpp:212) using `ring_kernel` (kv_stream.cu:184).
- For Metal: the cap for the MTP ids is 32768, not 2051. The decode attention kernel handles this with 512 chunks.

---

## 2. GDN layer dataflow

### 2.1 State, layout, math

Per layer:

- **Recurrent state:** `float[128 (i = k-row)][48 (head)][128 (j = v-col)]`, with j fastest. Element (i, h, j) is at `(i*48 + h)*128 + j`.
- **Conv history:** `float[10240][3]`, stored immediately after the state.
- **Size:** 786,432 + 30,720 floats = 3.12 MiB per layer. Over 36 layers that is 112 MiB (about 113 MB).

Math per token, per v-head h (q/k head `qh = h % 16`):

```
qkv = W_qkv x                         (10240)
c   = silu(conv4(hist, qkv))           per channel: s = h0*w0 + h1*w1 + h2*w2 + x*w3 ; y = s/(1+exp(-s))
q,k = l2norm(c[q-head]), l2norm(c[k-head])   y * rsqrt(sum y^2 + eps)  (eps on the SUM, not the mean)
v   = c[4096 + h*128 ...]
gate_h = softplus(alpha_h + dt_bias_h) * ssm_a_h     (softplus threshold 20: v>20 ? v : log1p(exp(v)))
beta_h = sigmoid(beta_raw_h)
g   = exp(gate_h)                                     (ssm_a is negative, so g in (0,1))
kv_col[j] = sum_i S[i][j] * k[i]                      (using the OLD state)
delta[j]  = (v[j] - g * kv_col[j]) * beta_h
S[i][j]   = g*S[i][j] + k[i]*delta[j]                 (fmaf(g, s, k*delta))
o[j]      = (sum_i S_new[i][j] * q[i]) * rsqrt(128)   (q scale 1/sqrt(S) applied to the output)
y[j]      = o[j] * rsqrt(mean_j o^2 + eps) * ssm_norm[j] * sigmoid(z[j])
out = W_ssm_out y
```

Note on the algebra:

- In production, `delta = (v - g*kv_col)*beta` with the decay applied to the dot product. This is algebraically equal to the canonical "decay first, then `kv = (gS)k`".
- The canonical `gdn.cu` literally decays first (`s*dec`), then adds `k*d`. The CUDA bits differ.

### 2.2 Decode (fused, production). `gdn_layer`, `src/core/layer.cpp:223-333`

The fused path requires `g_fused_gdn` (always true, layer.cpp:42) together with `native_gdn` and `native_bf16`.

| # | Step | Kernel | File:line |
|---|---|---|---|
| 1 | qkv GEMV [2560] -> [10240] | native_mmvq | layer.cpp ~L240 |
| 2 | Conv + SiLU + L2 (q, k heads), updating history | `gdn_conv_l2_kernel` | fused_gdn.cu:73-93 |
| 3 | alpha/beta BF16 GEMV fused with softplus*ssm_a and sigmoid | `gdn_ab_kernel` | fused_gdn.cu:95-123 |
| 4 | z = attn_gate GEMV [2560] -> [6144] | native_mmvq | layer.cpp |
| 5 | Delta-rule step + RMS norm * gamma * sigmoid(z) | `gdn_step_norm_kernel` | fused_gdn.cu:16-71 |
| 6 | ssm_out GEMV [6144] -> [2560] | native_mmvq | layer.cpp |

Kernel details:

- **`gdn_conv_l2_kernel`:**
  - Grid 80 (= 10240/128) x 128 threads. One thread per channel.
  - The history slides `h0 <- h1 <- h2 <- x`.
  - Blocks `< 32` (the 16 q heads and 16 k heads) L2-normalize over 128: `sum` = warp shfl + 4 partials added in order; `y *= rsqrtf(ss + eps)`.
  - SiLU is `s/(1+__expf(-s))`.
- **`gdn_ab_kernel`:**
  - Grid `ceil(96/8)`, 256 threads. One warp per output row. Rows 0-47 are alpha, 48-95 are beta.
  - BF16 is decoded via `<<16` and `&0xffff0000`. Accumulation is `fmaf` over 8 elements per uint4, lane-strided, then a butterfly reduce.
- **`gdn_step_norm_kernel`:**
  - Grid 48 (v heads), block (128, 4). Thread (col j, row-group rg) keeps 32 state rows `i = rg*32 .. +31` of column j in registers (RPG = 32).
  - `kv` is accumulated with `fmaf` over the thread's 32 rows, then `kv_col = red0 + red1 + red2 + red3` (fixed order).
  - The state is written back in the same loop.
  - The output norm: warp shfl over cols, 4 partials summed in order, then `rsqrtf(ss/128 + eps)`.
  - `y = oc*scale*gamma*(1/(1+__expf(-z)))`.

### 2.3 Verify (T <= 8) and commit replay. `verify.cpp:509-543`, `src/kernels/cuda/verify_kernels.cu`

Design: there is **no state snapshot**. A 113 MB copy is avoided (verify_kernels.hpp:4-14).

- The verify pass reads the state and **does not write it**.
- The commit replays the kept tokens from the stored per-layer inputs. The buffers `qkv_L_`, `h_L_`, `gate_L_` and `beta_L_` persist per GDN layer for the window.
- Header claim: "a verify window reproduces plain decode bit for bit" (verify_kernels.cu:3-4).

Verify pass, per GDN layer:

1. Q8_1 quantize + `native_mmvq` gives qkv for n rows (stored in `qkv_L_`).
2. **`gdn_conv_l2_multi_kernel`** (verify_kernels.cu:25-51):
   - Grid (80, n) x 128.
   - Token t's window is the last 4 entries of `[hist(3) | qkv_0..qkv_t]`.
   - It does **not** update the history. The output goes to `h_L_`.
3. **`gdn_ab_multi_kernel`** (L70-113): same as `gdn_ab_kernel`, but each weight row is read once for all n tokens (`acc[kVerifyMaxT]`). It writes `gate_L_` and `beta_L_`.
4. z GEMV (n rows).
5. **`gdn_step_norm_multi_kernel`** (L115-181):
   - Grid 48 x (128, 4).
   - Starts from the untouched state and loops `t = 0..T-1` (T = te) using the stored inputs.
   - Outputs y only for `t >= t_out_begin` (tb). Earlier group tokens are replayed for state only.
   - The state is written back **only if `n_keep != nullptr`**. In verify it is null, so nothing is written.
6. Q8_1 + MMVQ ssm_out.

Commit (verify.cpp:978-981, inside the captured commit graph):

- **`gdn_conv_commit_kernel`** (L53-68): `hist <-` the last 3 of `[hist | qkv_0..qkv_{n_keep-1}]`, where `n_keep` is device memory (`commit_[0]`).
- **`gdn_step_norm_multi`** is called with `n_keep = commit_`: replay the n_keep tokens and write the state.

### 2.4 Prefill (chunk form). `prefill.cpp:1342-1364`, `src/prefill/kernels.cu`

1. `native_proj` qkv[T,10240] and z[T,6144]. `bf16_proj` alpha into `ab[:, 0:48]` and beta into `ab[:, 48:96]` (ld 96) (L1353-1356).
2. **`gdn_gates_kernel`** (kernels.cu:206-215): `gate = (v>20 ? v : log1pf(__expf(v))) * ssm_a`, with `v = alpha + dt`. `beta = sigm(beta_raw)`.
3. **`gdn_conv`** (launcher kernels.cu:732-742):
   - `T <= 64` or `STRATA_GDN_CONV_SERIAL`: **`gdn_conv_kernel`** (L217-233), one thread per channel walking all T and updating the history.
   - Otherwise:
     - **`gdn_conv_tiled_kernel`** (L235-251): grid (80, ceil(T/64)) x 128. Each tile reads its 3 predecessors from the chunk or the history. Same expression, same bits.
     - **`gdn_conv_hist_kernel`** (L253-262) writes the new history.
   - Then **`gdn_l2_kernel`** (L263-275): grid (32, T) x 128, `v * rsqrtf(ss + eps)`.
4. **`gdn_recurrence`** (launcher L744-758):
   - Default **`gdn_rec_cols_pipe_kernel`** (L380-434):
     - Grid 48*4 = 192, block (32 cols, 4 row-groups).
     - Each thread keeps 32 state rows of one column in registers and loops sequentially over all T tokens.
     - Next-token inputs are prefetched into registers (software pipelining).
     - The output is `oc` (pre-norm) in `y`.
   - Then **`gdn_out_norm_kernel`** (L436-450): grid (T, 48) x 128. It writes y fp32 and **y16 = fp16(y)**, the input of the out projection.
   - `STRATA_GDN_PIPELINE=0` selects `gdn_rec_cols_kernel` (L337-378). `STRATA_GDN_REC_HEADS` selects `gdn_rec_kernel` (L277-334, one block per head with the norm fused).
   - **There is no chunkwise-parallel (WY / UT-transform) form.** The prompt recurrence is a per-token sequential loop inside one kernel, with the same per-element arithmetic as the fused decode ("same bits" comments, L325-329).
5. `native_proj` ssm_out from y16 (L1363).

### 2.5 GDN variants

| Variant | Files | Used by | Notes |
|---|---|---|---|
| Canonical | gdn.cu (`gdn_step_kernel` L43, `gdn_conv_kernel` L99, `gdn_l2_kernel` L118, `gdn_out_norm_kernel` L133, `gdn_beta_gate_kernel` L217), elementwise.cu (`gdn_gate_kernel` L40, `silu_kernel` L72, `scale_kernel` L50) | Decode only when `--native` is off (engine default without setup); the oracle for `gdn_parity` | MAX_H 1. q is prescaled by 1/sqrt(128) (scale_inplace). Decay first, then `s + k*d`. **L2 and out-norm sums in double**. SiLU via double exp. |
| Native (llama.cpp port) | native_gdn.cu (`gdn` kernel L46-81), native_gdn_preprocess.cu (`conv_silu` L53, `l2_norm` L70, `beta_sigmoid` L86, `gate_softplus` L90, `out_norm` L99) | `--native` with the fused path off | Built with `--use_fast_math`. Grid (48,1,32), block (32,4): one warp per (head, column), 4 rows per lane. `g*s + k*delta`; `warp_sum(s*q)*scale`. l2 via `rsqrtf(sum/128 + eps/128) * x * (1/sqrt(128))`. |
| **Fused (production)** | fused_gdn.cu, verify_kernels.cu, prefill/kernels.cu GDN kernels | Decode, verify, commit and prefill by default | Not built with fast math, but uses explicit `__expf` and `rsqrtf`. Decode, verify and prefill share the per-element arithmetic. |

---

## 3. KV cache formats, indexer cache, KV streaming

### 3.1 Pool layout (all formats)

- A row is one (page, kv_head, slot) holding 256 dims: `row = (page*2 + kvh)*4 + cell%4`, with `page = page_table[cell/4]`.
- Without streaming, page_table is the identity (`page = cell/4`).
- `page < 0` means not resident. Readers mask the cell (decode attention) or skip it.
- K and V live in separate pools.

### 3.2 Per-format encoding

| Format | K | V | Bytes/cell/layer (K+V, 2 heads) | 12 layers @32K | @128K | @256K |
|---|---|---|---|---|---|---|
| fp16 (`--kv f16`) | fp16 RNE of the rotated K | fp16 | 2048 | 768 MiB | 3.0 GiB | 6.0 GiB |
| int8 (`--kv q8`) | int8 codes + fp16 scale per 64 dims | same | 2*2*(256+8) = 1056 | 396 MiB | 1.55 GiB | 3.09 GiB |
| q4_0 + FWHT (`--kv q4`) | FWHT256 then ggml q4_0 (32-value blocks, 18 B) | same | 2*2*144 = 576 | 216 MiB | 864 MiB | 1.69 GiB |
| k8v4 (hybrid) | int8, **unrotated** | FWHT256 then q4_0; V scales are inside its 18-byte blocks | 528 + 288 = 816 | 306 MiB | 1.20 GiB | 2.39 GiB |

Format details:

- **int8** (kv_q8.cu:30-66; prefill kernels.cu:602-643 identical):
  - `amax` over 64 values: two warp maxima combined in fixed order.
  - `sbits = f16(amax/127)`; `sf = f32(sbits)`. Quantization uses **the stored scale**: `q = clamp(__float2int_rn(x/sf), -127, 127)`. If `sf == 0`, q = 0.
  - Dequant: `code*sf`.
  - **Not ggml q8_0.** ggml q8_0 uses 32-value blocks, `d = amax/127` and `round(x*id)` with `id = 1/d` in fp32.
- **q4_0** (kv_q4.cu:75-92): ggml rule.
  - `mval` = the value with max |x|. Ties go to the **larger** value, so all lanes agree.
  - `d = mval / -8`, `id = 1/d`, `q = clamp(__float2int_rz(x*id + 8.5f), 0, 15)`.
  - Byte j holds element j (low nibble) and element j+16 (high nibble).
  - Stored `d` is fp16. Quantization used the fp32 d.
  - **Both K and V are FWHT-rotated before quantization.** q is rotated before attention and the output is rotated back.
- **FWHT256** (kv_q4.cu:25-67):
  - Natural (Sylvester) Hadamard order, orthonormal: values are multiplied by `scale = 1/16` **on load**, which makes it self-inverse.
  - One warp per row; lane l holds elements `i*32 + l`, i = 0..7.
  - Stages h = 1..16 run across lanes (`__shfl_xor`, `(lane&h)==0 ? a+b : b-a`). Stages 32..128 run across each lane's 8 registers.
  - `STRATA_KV_ROT=1` also rotates the int8 format.
- **k8v4**:
  - Only V and the attention output are rotated; q is not.
  - **Not streamable with `--kv-resident`** (generate.cpp:1485).
  - The decode attention kernel's mode 3 reads K as int8 and V as q4.

### 3.3 Format selection (setup / CLI)

- `setup.py:2900` sets `--kv f16` when ctx <= 8192, else `--kv q8` (int8).
- `--kv-resident 32768` is set when ctx >= 65536, which turns on KV streaming with 32768 resident cells.
- Engine defaults (generate.cpp:184-228): fp16, all native flags off. Setup always passes `--native --prefill auto --spec 4`.

### 3.4 Indexer key cache (FP32; production = `native_qsa_indexer.cu`)

Buffers per QSA layer (`QsaIndexerBuffers {tail, dead, pooled, block_pos}`):

- `tail[3][128]`: raw keys of the current block's first three cells, **rounded through fp16** (`__half2float(__float2half_rn(raw))`, "SET_ROWS stores F16", L57-59).
- `dead[128]`: the "spare" key used for the incomplete tail block.
- `pooled[(max_cells/4 + 2)][128]` fp32 holds the pooled, normed and rotated keys.
- `block_pos`: the rope position of the last completed block.
- Footprint: 128 B/cell/layer. 12 layers @256K = 384 MiB.

**`append<TAB>`** (L46-118), single token, 1 block x 256 threads (only 128 active per dimension):

1. `slot = pos % 4`. If `slot < 3`, store `incoming` in `tail[slot]`. The kernel returns unless `pos == 0` or `slot == 3`.
2. If `pos == 0`: build the spare from cell 0 repeated 4 times. `sum = incoming` 4 times (`__fadd_rn` chain), so the mean is cell 0.
3. If `slot == 3`: `sum = tail0 + tail1 + tail2 + incoming` (`__fadd_rn`, this order).
4. `mean = __fmaf_rn(0.25f, sum, 0.0f)`.
5. RMS: `square_sum` = warp sums plus 8 partials (fp32), then `scale = rsqrtf(sum/128 + eps)`, then `value = scale*mean*gamma`.
6. RoPE on the first 64 dims (pairs d, d+32). The angle is `mrope_pos(rope_pos, pair) * powf(theta_scale, pair)`, then `rope_scaled_angle` (YaRN-capable; `cosf`/`sinf`).
   - `rope_pos = pos_base + 4*b`: **the block's FIRST cell position**.
   - The spare uses angle 0.
7. Write `pooled[b] = y`.
   - If `pos == 0`: `dead = y`. Otherwise `pooled[b+1] = dead` (the next row holds the spare) and `*block_pos = rope_pos`.
8. `dead` is only ever written at pos 0, so the tail block's key is constant. Its score only needs the +1e9 to rank first.

Batched prefill version (C-2): same arithmetic, same bits.

- `append_first` (L145): pos 0 spare.
- `append_blocks` (L171): one CTA per completed block of the chunk. A key comes either from the batch rows or from the previous tail.
- `append_tail` (L206): 3 CTAs x 128, writing the trailing incomplete block into `tail`.

Canonical `indexer_key_append_kernel` (qsa.cu:155-247):

- Mean and RMS in **double**.
- **No fp16 rounding** of raw keys.
- The spare is `rms_norm(raw[0])`.
- RoPE via the f64-built cos/sin table (`rope_neox_pair`).
- This is NOT bitwise equal to native, and it is used only without `--native`.

### 3.5 KV streaming (`kv_stream.cu`): needed on CUDA, **unnecessary on Apple unified memory**

Purpose: hold the full KV in pinned host memory and only a resident window of pages in VRAM. Selected pages are resolved and copied on demand inside the graph.

- **`resolve_kernel`** (L82-158), 1 block x 1024 threads:
  - Hits stamp an epoch and set a reference bit.
  - Misses claim a slot via CAS (-1 to -2). The victim is picked by a clock sweep with `block_scan`.
  - It re-points `page_table` and updates the `ctl` counters (ctl[3] counts unresolved pages).
  - Requires `n_slots >= 1024`. The engine minimum resident is 20480 cells (5120 slots).
- **`copy_kernel`** (L161): 96 x 128 threads, uint4 copies of the claimed pages from the host-mapped copy into VRAM slots.
- **`reset_kernel`** (L173): resets state.
- **`ring_kernel`** (L184): `table[i] = i % n_slots` (the MTP ring). `kv_ring_restore` uses memcpy.
- Prefill stages the whole layer from host into an identity-layout staging pool (`kv_stage_from_host`) and appends into the host mirror as well.
- Parity: `kv_stream_parity` requires streamed output to be **bitwise** equal to resident output.

**Apple note:** with unified memory, the full KV (for example int8 @256K = 3.09 GiB) is directly GPU-addressable.

- Use mode 0: identity page table, no host mirror, no `resolve`/`copy`/`stage` kernels.
- Keep `page_table` (or drop it and use `cell` directly). The masking path (`page < 0`) becomes dead code.
- The only reason to keep a page indirection would be the MTP ring (bounded drafter KV). That can be a modulo index instead.

---

## 4. Kernel catalog (every `__global__` in the slice)

Abbreviations:

- Det.: determinism. "B" means bitwise-reproducible run to run (fixed reduction order, no atomics on floats). "B*" means deterministic but not bitwise equal to the alternative path.
- Par.: parity test and tolerance.
- ggml-metal names are from upstream llama.cpp's `ggml-metal.metal` (kernel names vary by version; verify against pinned commit `3cf03257`).

### 4.1 `src/kernels/cuda/qsa.cu`: canonical QSA (oracle and fallback)

| Kernel (line) | Purpose, I/O | Launch / smem | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `kv_append_kernel` (122) | fp32 kcur/vcur [2,256] to fp16 K/V pool row at step.pos, plus host mirror | grid_for(n,THREADS) | `f16_from_f32` (RNE); B | qsa_parity: fp16 conversions exact | Trivial compute kernel (half cast). Or `kernel_set_rows_f16`. |
| `indexer_key_append_kernel` (155) | raw[128] to tail/dead/pooled. Double-precision mean and RMS, f64 rope table, no fp16 rounding. | 1 block, `threads`, dyn smem | double math; B | qsa_parity: pooled keys 1e-6 relative to rope terms; spare exact | **No fp64 on Metal.** Port the native variant instead. |
| `qsa_index_kernel` (249) | Per-block score in double (`__dmul_rn`/`__dadd_rn`), ReLU-sum over 4 heads, optional bias, +1e9 tail; writes **per-cell** scores[n_kv] | grid max_blocks, idx_heads*32 threads, smem double[32] | double; B | scores 1e-6 relative to sum of abs terms | No fp64. Use the fp32 block_scores semantics. |
| `topk_kernel` (307) | Binary lifting (32 passes) on order_key over cells; ties by lowest index; ascending ids[width] | 1 x 256, smem 2 KB | B. n_kv <= 32768 | ids EXACT | Custom; replaced by the block radix select. |
| `kv_gather_kernel` (406) | Gather selected fp16 K/V rows to contiguous scratch [width][2][256] | grid_for(total,256) | uint2 copies; B | gather exact | `kernel_get_rows_f16`, or fuse into attention. |
| `qsa_attend_kernel` (439) | One block per head over gathered K/V. Softmax in fp32 with `expf`, `-FLT_MAX` init, 1/sum; optional weights out. kv = h/12. | grid 24 x 256, dyn smem (cap+32) floats | B | attention 1e-5 vs the same fp16 cache | `kernel_flash_attn_ext_vec` over a gathered K/V would be equivalent (D=256 supported). |
| `qsa_gate_apply_kernel` (513) | attn * sigmoid(gate), double sigmoid, fp16 output | grid_for(n,256) | double; B | gate <= 1e-3 | Elementwise. Use the fp32 variant. |
| `qsa_gate_apply_f32_kernel` (793) | Same, fp32 output (double sigmoid) | grid_for(n,256) | double; B | 1e-3 | Elementwise. |

### 4.2 `src/kernels/cuda/qsa_select.cu`: production selection

| Kernel (line) | Purpose, I/O | Launch / smem | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `block_scores_kernel` (28) | scores[nq][max_blocks]. Warp per block, float4 per lane, 4-head ReLU sum, +1e9 tail; dead key for b==n_bid | grid (ceil(max_blocks/8), nq) x 256 | shfl_xor; B | qsa_select_bench reports diffs; decode vs canonical within 1e-6 | Simple: one simdgroup per block, `simd_sum`. |
| `block_topk_kernel` (57) | Cell-weighted 4x8-bit radix select over blocks, eq-budget, ascending ids[cap] per query | grid nq x 256; smem hist[256] + s_a/s_b (3 KB) | shared atomicAdd on **int** histograms (order-independent). Thread-0 serial prefix scans. B. | ids exact vs canonical when scores are equal | Custom kernel. Apple threadgroup atomics on int are fine. ggml `argsort`/`top_k` do NOT reproduce the cell-weighted, partial-threshold-block rule. |
| `block_scores_tc_kernel` (195) | Prefill: many queries times blocks as a GEMM (rows = query x indexer head, K=128), 3xTF32 (hi*hi + hi*lo + lo*hi), ReLU per head summed. Blocks < n_bid only. | 128 threads, dynamic smem (16 queries x 516 floats + 32 keys x 132 floats), 4 tiles per CTA | `cvt.rna.tf32`, `mma.sync m16n8k8 tf32`, sm80+; B* (not bitwise vs warp kernel) | Accepted: near-ties may select differently | `simdgroup_matrix<float,8,8>` multiply is native fp32 on Apple, so no 3xTF32 split is needed. **Threadgroup memory > 32 KB: retile.** |
| `block_scores_wmma_kernel` (329) | HIP gfx12 variant of _tc (opt-in) | 128 threads | HIP WMMA | n/a | Ignore. |
| `block_scores_tail_kernel` (438) | Scores only the tail block (dead key + 1e9) for the _tc/_wmma paths | grid nq x 32 | B | n/a | Fold into the main scorer. |
| `block_topk_reg_kernel<PER>` (507) | Same selection as `block_topk_kernel`, with each query's block keys in registers (PER = 33 or 66 per thread), per-warp histograms, block scans | grid nq x **1024**; smem hist[32][256] int = 32 KB + scans | B, same ids as the ref | same | 1024 threads is OK on Apple, but **hist[32][256] alone is 32 KB, the Apple threadgroup limit**. Use per-simdgroup 16-bit histograms, or fewer histograms. |
| `block_scores_multi_kernel` (606) | Decode/verify scorer: fixed 256-block grid strides over blocks, all nq<=8 queries' q_idx in smem (16 KB); same arithmetic order as `block_scores_kernel` | 256 x 256, smem 16 KB + 128 B | B, bitwise equal to `block_scores_kernel` | same | Good Metal shape: q in threadgroup memory, loop over blocks. |

### 4.3 `src/kernels/cuda/qsa_decode_attn.cu`: production sparse attention (decode, verify, MTP, prefill fallback)

| Kernel (line) | Purpose, I/O | Launch / smem | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `attn_chunk_kernel<0/1/2/3>` (84) | q[nq][24][256] fp32; pools by mode; ids[cap]; produces partial acc/m/l per (kvh, chunk, head). See section 1.4. | grid (ceil(cap/64), 2, nq) x 256; static smem ~15.9 KB | `__expf`, fmaf, shfl; B | kv_hybrid_parity attention < 5e-3 (L309, L354); vs canonical 1e-5 (fp16) | **Primary port target.** Not a fit for `kernel_flash_attn_ext` as-is (that kernel expects contiguous K/V plus a mask). Write a custom gathered split-K kernel: threadgroup = (chunk, kvh), 8 simdgroups, `simd_sum`/`simd_max`, `fast::exp` or `exp`. 15.9 KB of smem fits within 32 KB. Alternative: get_rows-gather 2051 rows, then `kernel_flash_attn_ext_vec` (f16 or q8_0 only, so the int8 format would need to change). |
| `attn_merge_kernel` (184) | Merge chunks: M, then sequential `w = __expf(m-M)`, L and acc via fmaf, `acc/L` | grid (24, nq) x 256 | B | same | Trivial. ggml FA-vec has its own reduce kernel (`kernel_flash_attn_ext_vec_reduce`, name unsure). |

### 4.4 `src/kernels/cuda/qsa_prompt_attn.cu`: prefill sparse attention

| Kernel (line) | Purpose, I/O | Launch / smem | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `prompt_attn_kernel<0/1/3>` (102) | One query token x one KV head: 12 heads padded to M=16, CH=32 cells per chunk over the selected ids. q is split into hi/lo fp16 after a 2^14 power-of-two prescale. int8 codes are converted exactly to half. Online softmax in exp2. Output fp32 attn[T][24][256]. | grid (T, 2) x 128; dynamic `sizeof(Smem<mode>)` (about 50+ KB, opt-in) | `mma.sync m16n8k16 f16` (Turing uses 2x k8); B* (FP32-level, not bitwise vs decode) | qsa_prompt_attn_parity: new vs old <= 1e-4 * scale (L187); kv_hybrid prompt < 1e-2 (L347) | Use `simdgroup_matrix<float,8,8>` (fp32, exact enough) or half matrices with the hi/lo trick. **Smem > 32 KB on Apple: reduce CH to 16 and/or drop ql.** M3+/M4: simdgroup matrix; M5: Metal 4 tensor ops (unsure). |
| `prompt_attn_i8_kernel` (416) | sm80 int8-KV variant: Smem2, cp.async double buffering, swizzled rows, fixed-order per-64 group sums, scale folded into p | grid (T, 2) x 128, dynamic smem | cp.async, mma; B* | same | No cp.async on Metal (simdgroup_async_copy is not public). Use plain loads plus threadgroup barriers. |
| `prompt_attn_wmma_kernel` (740) | HIP gfx12 version | (T,2) x 128 | WMMA | hip_prompt_attn_wmma | Ignore. |

### 4.5 `src/kernels/cuda/kv_q4.cu`

| Kernel (line) | Purpose, I/O | Launch | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `fwht256_kernel` (25) | In-place or out-of-place orthonormal FWHT over rows of 256 fp32 (q, K, V, attn) | grid ceil(rows/rpb) x (32, rpb) | shfl_xor; B | kv_q4_parity: FWHT bitwise, dot invariance | No ggml-metal equivalent (unsure if a Hadamard op exists at the pinned commit). Custom: one simdgroup per row, `simd_shuffle_xor`. Identical algorithm, so bitwise-portable if the add order is kept. |
| `kv_append_q4_kernel` (101) | One 32-value group of K (z=0) or V (z=1) per block, ggml q4_0, writes device page (if resident) + host mirror | grid (2 heads, 8 groups, 2) x 32 | shfl; B | Q4 blocks bitwise vs ggml reference | `kernel_set_rows_q4_0` exists in ggml-metal, but check its tie rule and FMA contraction (`x*id + 8.5`). |
| `kv_append_q4_batch_kernel` (119) | Prompt path: grid (T, 2, 8) x 32, K then V, plus staging pool | same | B | same | same |
| `kv_gather_q4_kernel` (139) | Dequantize selected cells to fp16 scratch (canonical attention path) | grid x (32, rpb) | B | | Fold into attention. |

### 4.6 `src/kernels/cuda/kv_q8.cu`

| Kernel (line) | Purpose, I/O | Launch | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `kv_append_q8_kernel` (30) | One 64-value group per block: amax, f16 scale, `rn(x/sf)` clamp ±127 | grid (2, 4, 2) x 64; smem 8 B | B | kv_q8_parity: codes and scales bitwise; <= 0.624 steps vs fp16 | Custom (not q8_0). Use `precise::divide` or IEEE division. Metal fast-math division is not correctly rounded. |
| `kv_gather_q8_kernel` (69) | Dequantize selected cells to fp16 scratch (canonical path) | grid_for(.., 256) | B | | Fold into attention. |

### 4.7 `src/kernels/cuda/kv_stream.cu` (section 3.5; drop on Apple)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `resolve_kernel` (82) | Page residency: hits, CAS claims, clock sweep, table update, ctl counters | 1 x 1024 | atomicCAS, block_scan; B given the same inputs | kv_stream_parity bitwise | **Not needed** (unified memory). |
| `copy_kernel` (161) | Copy claimed pages host-mapped -> VRAM slots | 96 x 128 | uint4 | same | Not needed. |
| `reset_kernel` (173) | Reset residency state | | | | Not needed. |
| `ring_kernel` (184) | `table[i] = i % n_slots` for the MTP ring | | | | Replace with modulo indexing if the drafter KV stays bounded. |

### 4.8 `src/kernels/cuda/native_qsa.cu` (fast-math; llama.cpp norm/unary port)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `norm<256/1024>` (45) | Weighted RMS norm per row (q heads 256, kv heads, indexer 128); llama.cpp `rms_norm_f32` arithmetic | rows x 256 (or 1024) | fast-math; B | native_qsa_parity vs pinned ggml (bench/micro, absent from the clone) | `kernel_rms_norm` (+ mul) in ggml-metal is the same algorithm, but the reduction order differs. |
| `gate` (69) | `attn * 1/(1+expf(-q_full[h*512+256+d]))`, fp32 | grid_for(count,256) | fast-math (`expf` becomes `__expf`) | same | Elementwise; `ggml_sigmoid` + `mul`. |

### 4.9 `src/kernels/cuda/native_qsa_indexer.cu` (fast-math; production indexer)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `append<TAB>` (46) | Single-token indexer append (section 3.4) | 1 x 256 | `__fadd_rn`, `__fmaf_rn`, fast-math powf/cosf/sinf; B | native_qsa_indexer_parity (absent from the clone); verify replay relies on it being bitwise | Custom kernel (1 threadgroup x 128). Keep `__fadd_rn` order; in Metal write explicit `fma(0.25, sum, 0)`. |
| `append_first<TAB>` (145) | Prompt path: pos-0 spare | 1 x 256 | same | same | same |
| `append_blocks<TAB>` (171) | Prompt path: one CTA per completed block of the chunk | (#blocks) x 256 | same | same | same |
| `append_tail` (206) | Prompt path: the last incomplete block's raw keys to tail (fp16-rounded) | 3 x 128 | | | same |

### 4.10 `src/kernels/cuda/native_qsa_score.cu` (NOT used by the engine)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `score_kernel` (54) | llama.cpp mmf-style TF32 MMA block scorer (raw F32 bits as TF32) | grid ceil(max_blocks/ROWS) x (32, WARPS), launch_bounds(64,1) | mma tf32 | native_qsa_score_parity (absent) | Skip. |
| `scalar_score_kernel` (148) | HIP scalar scorer | max_blocks x 128 | | tests/hip/native_qsa_score.cpp | Skip. |

### 4.11 `src/kernels/cuda/native_flash_attn.cu` (diagnostic only)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `attend` (50) | llama.cpp `fattn-vec` specialized D=256, ctx <= 256 (`--native-flash-attn-short`); keeps the fma-ordering quirk and the `+3*0.6931` (ln 8) max offset | 24 x (32, 4) | fast-math | native_flash_attn_parity (absent) | Equivalent to `kernel_flash_attn_ext_vec_f16_h256`. Not on the production path. |

### 4.12 `src/kernels/cuda/gdn.cu`: canonical GDN (oracle and fallback)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `gdn_step_kernel` (43) | Delta-rule step; q prescaled; decay first; MAX_H=1 head per block | grid (4, 48) x 32 (JTHREADS) | B | gdn_parity: o and state <= 1e-5 (L200-201) | Reference only. |
| `gdn_conv_kernel` (99) | `ggml_ssm_conv` + state slide; kernel[0] reads the oldest state | ceil(C/256) x 256 | B | conv out/state <= 1e-6 (L297-298) | `kernel_ssm_conv_f32_f32` (ggml-metal) has the same semantics (state layout differs: ggml keeps the conv state in a concatenated input). |
| `gdn_l2_kernel` (118) | x / sqrt(sum x^2 + eps), **double** sum | rows x 32 | double | norms <= 1e-6 (L339) | No fp64; ggml `kernel_l2_norm_f32` uses `1/max(sqrt(sum), eps)`, which is different. |
| `gdn_out_norm_kernel` (133) | rms(o) * ssm_norm * sigmoid(z), double sum | 48 x 32 | double | <= 1e-6 (L372) | Use the fused variant's fp32 math. |
| `gdn_beta_gate_kernel` (217) | beta = sigmoid(beta) ("the C1 bug" fix) | ceil(48/128) x 128 | | | Elementwise. |

Related canonical helpers in `elementwise.cu`:

- `gdn_gate_kernel` (L40): softplus * a.
- `scale_kernel` (L50).
- `silu_kernel` (L72), using a double exp.
- `rms_norm_weighted_kernel` (L88).

### 4.13 `src/kernels/cuda/fused_gdn.cu`: production GDN decode (section 2.2)

| Kernel (line) | Purpose | Launch / smem | Features, determinism | Parity | Metal note |
|---|---|---|---|---|---|
| `gdn_step_norm_kernel` (16) | State update + output RMS norm + z gate; state in registers | grid 48 x (128,4); smem sk/sq 1 KB, red 2 KB, wsum 64 B | `__expf`, `rsqrtf`, fmaf; B; bitwise equal to verify_multi and prefill rec | verify bitwise; gdn_parity-class 1e-5 | ggml-metal: upstream llama.cpp has `GGML_OP_GATED_DELTA_NET` (the pinned commit has ggml-cuda `gated_delta_net.cu`, which `native_gdn.cu` ports). A Metal `kernel_gated_delta_net` may exist at that commit (**unsure**). `kernel_rwkv_wkv7_f32` is the closest structural analogue. Custom port is easy: threadgroup (128,4) = 512 threads, 32 floats per thread in registers. |
| `gdn_conv_l2_kernel` (73) | 4-tap conv + SiLU (`__expf`) + L2 for q/k heads; history slide | grid 80 x 128 | B | | `kernel_ssm_conv` + `kernel_silu` + l2 (via rms_norm with eps/128 and scale). Custom fused is simple. |
| `gdn_ab_kernel` (95) | BF16 GEMV of alpha/beta (96 rows x 2560) + softplus*a and sigmoid | ceil(96/8) x 256 | fmaf chain per lane, then butterfly; B | | `kernel_mul_mv_bf16_f32` + elementwise. BF16 is supported in Metal (`bfloat`, M2+/macOS 14+). |

### 4.14 `src/kernels/cuda/native_gdn.cu` and `native_gdn_preprocess.cu` (fast-math llama.cpp ports; non-fused native path)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `gdn` (native_gdn.cu:46) | gated_delta_net arithmetic: `g*s + k*delta`, `warp_sum(s*q)*scale`, state index `(i*h_v+head)*S+col` | grid (48,1,32) x (32,4) | fast-math; B | native_gdn_parity, native_gdn_pipeline_parity (absent) | Same as the fused note. |
| `conv_silu` (pre:53) | ssm_conv + SiLU | | fast-math | native_gdn_preprocess_parity (absent) | `kernel_ssm_conv` + silu |
| `l2_norm` (pre:70) | `rsqrtf(sum/128 + eps/128) * x * (1/sqrt(128))` (llama graph: rms_norm then scale) | | | | `kernel_rms_norm` + `kernel_scale` |
| `beta_sigmoid` (pre:86) | sigmoid | | | | elementwise |
| `gate_softplus` (pre:90) | softplus(a + dt) * ssm_a | | | | elementwise |
| `out_norm` (pre:99) | rms * gamma * sigmoid(z) | | | | `kernel_rms_norm` + mul |

### 4.15 `src/kernels/cuda/verify_kernels.cu` (GDN and QSA-relevant kernels; the other kernels belong to other slices)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `gdn_conv_l2_multi_kernel` (25) | T-token conv+SiLU+L2 from [hist \| qkv_0..t]; no history write | grid (80, n) x 128 | B; bitwise = decode | header claim | Same code as decode with a window index. |
| `gdn_conv_commit_kernel` (53) | hist <- last 3 of [hist \| qkv_0..n_keep-1], n_keep on device | ceil(C/256) x 256 | B | | trivial |
| `gdn_ab_multi_kernel` (70) | alpha/beta GEMV for n <= 8 tokens, weights read once | ceil(96/8) x 256 | B; bitwise = decode | | `mul_mm`/`mul_mv` with ne11 = n |
| `gdn_step_norm_multi_kernel` (115) | Sequential replay over T tokens from the untouched state; outputs for t >= t_out_begin; writes the state iff n_keep | grid 48 x (128,4) | B | | Same as fused, with a token loop. On Metal, use one kernel source for decode, verify, commit and prefill. |
| `dense_steps_kernel` (296) | MTP steps {c, c+1, (c+1)/4, c+1} | 1 x 64 | | | trivial |
| `window_ids_kernel` (361) | MTP window ids and width override | (8, n) x 256 | | | trivial |

Other kernels in this file belong to other slices: `embedding_gather_dev`, `broadcast_streams`, `copy_indexed`, `fetch_blobs`, `rebase_ptrs`, `add_streams_broadcast`, `ident_hits`, `gather_rows`, `map_ids`, `row_top_prob`, `mtp_select`, `wait_flag*`, `resident_plan`, `copy_i32_unless`, `copy_or_zero`, `gpu_stamp`.

### 4.16 `src/prefill/kernels.cu` (QSA/GDN prompt kernels; not fast-math; strata_prefill, CMakeLists:395)

| Kernel (line) | Purpose | Launch | Features | Parity | Metal note |
|---|---|---|---|---|---|
| `gdn_gates_kernel` (206) | gate/beta for T tokens from ab[T][96] | blocks_for(T*48) x 256 | `__expf`, `log1pf`; B | | elementwise |
| `gdn_conv_kernel` (217) | Serial conv over T per channel (T <= 64) | 80 x 128 | B | | `kernel_ssm_conv_f32_f32` handles T tokens per launch. |
| `gdn_conv_tiled_kernel` (235) | Tiled conv, 64 tokens per tile, same bits | (80, ceil(T/64)) x 128 | B | | same |
| `gdn_conv_hist_kernel` (253) | New history = last 3 inputs | 80 x 128 | B | | trivial |
| `gdn_l2_kernel` (263) | L2 per (t, head<32) | (32, T) x 128 | B | | rms_norm + scale |
| `gdn_rec_kernel` (277) | One block per head: recurrence + norm, sequential over T | 48 x (128,4) | B, same bits | | Alternative layout. |
| `gdn_rec_cols_kernel` (337) | Recurrence split into 4 column blocks per head | 192 x (32,4) | B | | |
| `gdn_rec_cols_pipe_kernel` (380) | **Default.** Same as `gdn_rec_cols_kernel` with register prefetch of t+1 | 192 x (32,4) | B, same bits | | Main port target for prefill. Consider a chunked WY form on Metal for speed (tolerance-level parity only). |
| `gdn_out_norm_kernel` (436) | Per (t, head) RMS * gamma * sigmoid(z); y fp32 + y16 fp16 | (T, 48) x 128 | B | | rms_norm + mul + sigmoid |
| `rms_rows_kernel` (553) | Weighted RMS per row (`block_sum`, `s*r*w`) | rows x 256 | | | `kernel_rms_norm` |
| `rope_kernel<TAB>` (566) | NEOX(64) RoPE with M-RoPE `mrope_pos` and YaRN `rope_scaled_angle`; angle via `powf` (accurate, since there is no fast-math here) | rows x 32 | | | `kernel_rope_neox` / `kernel_rope_multi` (n_dims = 64) |
| `split_q_kernel` (586) | First 256 of each 512 | | | | view/copy |
| `gate_attn_kernel` (592) | `fp16(a * 1/(1+expf(-gate)))` | | `expf` (accurate) | | elementwise |
| `kv_append_kernel` (602) | Batched fp16 or int8 append (same int8 rule as kv_q8.cu) to device, host and stage pools | (T, 2, 8) x 64 | B | kv_hybrid_parity bitwise appends | custom/`set_rows` |
| `to_f16`, `round_f16`, `to_bf16` (645-660) | Casts (other slice) | | | | |

### 4.17 Related helpers outside the slice (used by QSA/GDN)

- `rope.cu:97 rope_neox_kernel`: canonical, cos/sin table built in float64 (`build_rope_table`).
- `native_rope.cu`: `--native` decode rope. Fast-math `powf`/`cosf`/`sinf`; host `theta_scale = powf(1e7, -2/64)`. `STRATA_ROPE_TABLE=1` uses the f64 table.
- `elementwise.cu`: `rms_norm_weighted` (L88), `silu` (L72), `gdn_gate` (L40).

---

## 5. Per-file summary

| .cu file | Kernels | Role | Path default? | Metal plan |
|---|---|---|---|---|
| qsa.cu | 8 | Canonical QSA: f16 append, f64 indexer, f64 scores, cell top-k, gather, attend, f64 gate | Fallback/oracle; `kv_append_step` (fp16 append) IS used in production for fp16 KV | Port only `kv_append`. The rest serves as a CPU-side reference (fp64 is OK on CPU). |
| qsa_select.cu | 7 | Block scores (warp, multi, TF32-TC, tail) and block radix top-k (ref, reg) | Decode/verify: multi + reg/ref. Prefill: TC + reg/ref. | Custom scorer + custom radix select (keep the exact tie rule); watch the 32 KB smem cap. |
| qsa_decode_attn.cu | 2 | Gathered split-K sparse attention + merge, 4 KV modes | Decode, verify, MTP, prefill fallback | Custom Metal kernel (primary). |
| qsa_prompt_attn.cu | 3 | Tensor-core prompt sparse attention (fp16 hi/lo MMA), int8 cp.async variant, HIP WMMA | Prefill (not q4) | Custom simdgroup_matrix kernel; or reuse the decode kernel batched over T first. |
| kv_q4.cu | 4 | FWHT256, q4_0 append (step/batch), q4 gather | q4/k8v4 formats; FWHT for rot/hybrid | Custom FWHT; q4_0 via custom or set_rows. |
| kv_q8.cu | 2 | Group-64 int8 append and gather | int8 format (ctx > 8192 by setup) | Custom. |
| kv_stream.cu | 4 | Residency resolve/copy/reset/ring | ctx >= 65536 (setup `--kv-resident`) | **Drop** (unified memory). Ring becomes modulo. |
| native_qsa.cu | 2 | RMS norm, gate (llama.cpp port) | `--native` decode/verify | `kernel_rms_norm`-like + elementwise. |
| native_qsa_indexer.cu | 4 | Indexer append (single + batched) | Decode/verify/commit/prefill | Custom (small). |
| native_qsa_score.cu | 2 | TF32 / scalar block scorer | **Unused** (tests only) | Skip. |
| native_flash_attn.cu | 1 | FA-vec D=256 short ctx | Diagnostic flag only | Skip (`kernel_flash_attn_ext_vec` if ever needed). |
| gdn.cu | 5 | Canonical GDN step/conv/l2/out-norm/beta | Non-native fallback/oracle | CPU reference. |
| fused_gdn.cu | 3 | Fused conv+L2, ab, step+norm | **Decode default** | Custom (primary). |
| native_gdn.cu | 1 | llama.cpp gated_delta_net port | `--native` without fused | Skip (fused covers it). |
| native_gdn_preprocess.cu | 5 | llama.cpp conv/l2/sigmoid/softplus/out-norm | `--native` without fused | Skip. |
| verify_kernels.cu (GDN/QSA part) | 6 of 23 | T-token GDN conv/ab/step, commit, MTP steps/window | Verify + commit + MTP | Custom; share code with decode. |
| prefill/kernels.cu (GDN/QSA part) | 15 of ~35 | Prompt GDN, norms, rope, split/gate, KV append | Prefill | Custom + ggml-like elementwise. |

---

## 6. Canonical vs native vs fused/fast: what runs by default

Flags:

- Engine defaults with no CLI flags (generate.cpp:184-228): native_* off, kv fp16. `g_fused_gdn`, `g_fast_select` and `g_fast_attn` are hard-wired true (layer.cpp:42).
- `--native` (generate.cpp:1498-1523) turns on native_gdn, native_qsa, native_qsa_indexer, native_rope, native_bf16, mmvq and so on.
- **setup.py always passes `--native --prefill auto --spec 4`.** It sets `--kv f16` when ctx <= 8192 and `--kv q8` otherwise, and adds `--kv-resident 32768` when ctx >= 65536.

| Stage | Decode (production `--native`) | Verify (n <= 8) | Prefill (chunk <= 8192) | Non-native fallback (decode) |
|---|---|---|---|---|
| QSA q/k norm | native_qsa `norm<256>` | same, batched rows | `rms_rows_kernel` | `rms_norm_weighted` (elementwise.cu) |
| QSA RoPE | `native_rope_apply` (fast-math angles) | same, per-row positions | `rope_kernel<false>` (accurate powf/cosf/sinf) | `rope_neox` (f64 table) |
| KV append | `kv_append_q8_step` / `kv_append_step` / q4 | same, per token | prefill `kv_append_kernel` / `kv_append_q4_batch` | same as decode |
| Indexer append | native `append` | native `append` per token + commit replay | `append_first/blocks/tail` | `indexer_key_append_kernel` (f64) |
| Scores | `block_scores_multi` | `block_scores_multi` (n <= 8) | `block_scores_tc` (3xTF32) + tail; fallback warp | `qsa_index_kernel` (f64, only if `!g_fast_select`) |
| Top-k | `block_topk_reg<33>` (<= 33,792 blocks), else `block_topk` | same | same | `topk_kernel` |
| Attention | `attn_chunk<mode>` + merge | same, batched | `prompt_attn_kernel<0/1/3>` or `prompt_attn_i8` (sm80 int8); q4 uses decode batch | gather + `qsa_attend` (only if `!g_fast_attn`) |
| Gate | native `gate` (fp32) | same | `gate_attn_kernel` (fp16 out) | `qsa_gate_apply_f32` (double) |
| GDN conv+L2 | `gdn_conv_l2_kernel` | `gdn_conv_l2_multi` | `gdn_conv(_tiled)` + `gdn_l2_kernel` | `gdn_conv_kernel` + `gdn_l2_kernel` (gdn.cu) |
| GDN gates | `gdn_ab_kernel` | `gdn_ab_multi` | `gdn_gates_kernel` (after bf16_proj) | `gdn_gate`/`gdn_beta_gate` |
| GDN recurrence | `gdn_step_norm_kernel` | `gdn_step_norm_multi` (no write) + commit replay | `gdn_rec_cols_pipe` + `gdn_out_norm` | `gdn_step_kernel` + `gdn_out_norm` |

Bitwise relationships on CUDA:

- **Verify equals decode** bitwise: same kernels or a transcribed order (verify_kernels.cu:3-4).
- **Prefill GDN recurrence** has per-element arithmetic identical to decode. But its inputs differ:
  - GEMM vs GEMV projections.
  - Prefill rope vs native rope.
  - The fp16 `y16` input to the out projection.
- **Prefill QSA selection and attention are not bitwise** with decode (TF32 scorer, MMA prompt attention).
- **Streamed equals resident** bitwise.

---

## 7. Numerical subtleties to reproduce for parity

1. **Indexer keys are rounded to fp16** before pooling (native, L57-59). The sum order is `((t0+t1)+t2)+t3` with `__fadd_rn`; the mean is `fma(0.25, sum, 0)`. The canonical version uses doubles and no fp16, and differs.
2. **Pooled key RoPE position** = position of the block's **first** cell (`pos_base + 4*b`), not the last cell and not the mean. The spare/dead key is rotated at angle 0.
3. **The tail block always wins**: `+1e9f` is added to the tail block's score when `n_kv % 4 != 0`, so its 1-3 cells are always selected and reduce the remaining budget.
4. **The width is 2051, not 2048.** The identity applies for n_kv <= 2051. Ids are emitted ascending, so the attention order is ascending cell order (it matters for the fmaf PV accumulation order).
5. **Threshold ties**: lowest cell index first. A threshold block may contribute only its first cells. `-0` equals `+0`. NaN sorts as the lowest key (never selected unless forced).
6. **Score is ReLU-sum of 4 raw dots** with no scaling. The fp32 production order is per-lane float4 dot, then butterfly. TF32-3x in prefill moves last bits, and near-ties may flip (accepted).
7. **GQA mapping** is `h / 12` (heads 0-11 use kv 0), not `h % 2`. For GDN the head pairing is **`h % 16`** (modulo/tile). GGUF reorders V heads to make this valid.
8. **Attention softmax**: `__expf(s - m_chunk)`, chunk partials merged with `__expf(m_c - M)`. The scale 1/16 is applied to the score before the max. Empty or masked chunks give `m = -FLT_MAX`, `l = 0`. The output is `acc/L` (division, not reciprocal-multiply).
9. **Prompt attention uses exp2** with `scale_log2 = log2(e)/16`. It is not bitwise with decode; tolerance about 1e-4 relative.
10. **int8 KV**: quantize against the **fp16-rounded** scale, with correctly rounded division `x/sf` and round-half-even. Metal fast-math division is approximate, so use `precise::divide` (or compile those kernels without fast math).
11. **q4_0**: `d = mval/-8` (sign carried; ties go to the larger value). `id = 1/d` in fp32 (unrounded). `trunc(x*id + 8.5)` is likely contracted to an FMA by nvcc (`--fmad=true` default). Match with explicit `fma(x, id, 8.5f)` and confirm with `kv_q4_parity`. Stored d is fp16.
12. **FWHT**: the 1/16 is applied on load; add/sub order is lanes (bits 0-4) then registers (bits 5-7). It is exactly self-inverse only up to rounding.
13. **GDN L2 eps is on the sum of squares** (`rsqrt(sum + eps)`), while the output norm eps is on the mean (`rsqrt(sum/128 + eps)`). ggml `l2_norm` uses `1/max(sqrt(sum), eps)`, which is different. The native/llama path's `rsqrt(sum/128 + eps/128) * (1/sqrt128)` is algebraically the same as Strata's but not bitwise.
14. **GDN q scale** 1/sqrt(128) is applied to the **output** (`oc = o * rsqrtf(128)`), not to q (the canonical version prescales q). Decay is folded as `delta = (v - g*kv_old)*beta` and `s = fmaf(g, s, k*delta)`. Row-group partial sums are added as `red0+red1+red2+red3`.
15. **Softplus threshold 20**, `log1pf(__expf(v))`. `g = __expf(gate)`. SiLU is `s/(1+__expf(-s))`. Sigmoid is `1/(1+__expf(-z))`. On CUDA these are hardware approximations (`ex2.approx`). Metal `fast::exp` behaves similarly but is not bitwise. Expect about 1e-6 relative differences.
16. **Fast-math compile split**:
    - `native_*.cu` uses `--use_fast_math` (CMakeLists.txt:289-303), so `powf`, `cosf`, `sinf`, `expf` and division are approximate there.
    - `fused_gdn.cu`, `verify_kernels.cu`, `qsa_*.cu`, `kv_*.cu` and **`prefill/kernels.cu`** are not fast-math.
    - Result: the **prefill RoPE (accurate `powf`/`cosf`/`sinf`) and the decode native RoPE (fast-math) differ in the last bits**. K cached by prefill vs decode carries slightly different rotations. `STRATA_ROPE_TABLE=1` makes both use the f64 table.
    - Metal compiles with fast math by default; set `MTLCompileOptions.mathMode` (or `fastMathEnabled = false`) per library where parity matters.
17. **Large-angle trig**: positions up to 256K times frequencies up to 1 rad/token give angles up to about 2.6e5 rad. Fast-math sin/cos lose accuracy there. CUDA native already has this behaviour (and so does llama.cpp). Consider a precise rope or an f64-built table on Metal (CPU-built table, uploaded once).
18. **Output projections**: verify and decode quantize activations to Q8_1 before MMVQ (llama.cpp style). Prefill uses fp16 `y16` and fp16 gated attention (`gate_attn_kernel`) as GEMM inputs. This belongs to the GEMM slice but dominates prefill-vs-decode differences.
19. **No fp64 on Apple GPUs**: every canonical kernel with double math (qsa_index, indexer_key_append, gdn_l2, gdn_out_norm, qsa_gate_apply, silu) can only be reproduced on the CPU side. The Metal port should target the native/fused fp32 semantics and validate against CUDA dumps with the tolerances below.
20. **Verify/commit invariants to keep**:
    - The verify pass must not mutate GDN state, conv history or the indexer tail.
    - Commit replays the kept tokens from the stored h/gate/beta/qkv and the restored tail.
    - KV cells of rejected tokens are left for overwrite.

### Parity tests and tolerances (to mirror in a Metal test suite)

| Test | File | Checks / tolerance |
|---|---|---|
| qsa_parity | `src/kernels/qsa_parity.cpp` (header L1-50) | fp16 conversions, pool, gather, spare EXACT; pooled keys 1e-6 (relative to rope terms); scores 1e-6 (relative to sum of abs terms); ids EXACT; attention vs same fp16 cache 1e-5; fp16 cache cost <= 1e-3; gate <= 1e-3 |
| gdn_parity | `src/kernels/gdn_parity.cpp` | o, state <= 1e-5 (L200-201); conv out/state <= 1e-6 (L297-298); l2 / out-norm <= 1e-6 (L339, L372). Also checks the modulo pairing, decay-first, conv slide direction, eps on the squared norm and the state layout. |
| kv_q8_parity | `src/kernels/kv_q8_parity.cpp` | codes and scales bitwise; <= 0.624 quant steps vs fp16 |
| kv_q4_parity | `src/kernels/kv_q4_parity.cpp` | FWHT and Q4 blocks bitwise; dot-product invariance under rotation |
| kv_hybrid_parity | `src/kernels/kv_hybrid_parity.cpp` | appends bitwise; attention < 5e-3 (L309, L354); prompt < 1e-2 (L347) |
| kv_stream_parity | `src/kernels/kv_stream_parity.cpp` | streamed vs resident BITWISE |
| qsa_prompt_attn_parity | `src/kernels/qsa_prompt_attn_parity.cpp` | new vs old <= 1e-4 * scale (L187) |
| qsa_select_bench | `src/kernels/qsa_select_bench.cpp` | reports score diffs and differing selections (HIP; `32768 256 3 262144`) |
| native_*_parity | `bench/micro/*_parity.cpp` (**absent from the clone**; CMakeLists.txt:313-330, against a pinned llama.cpp build) | native_gdn, native_gdn_preprocess, native_gdn_pipeline, native_qsa, native_qsa_indexer, native_qsa_score, native_flash_attn, native_rope ... |
| HIP scorer | `tests/hip/native_qsa_score.cpp` | scalar scorer parity |

---

## 8. Metal porting summary for this slice

1. **Drop entirely:**
   - `kv_stream.cu` (resolve, copy, reset; the ring becomes modulo).
   - Host KV mirrors and the staging pool (`kv_stage_from_host`).
   - Mapped H2D step uploads (write step/pos into shared buffers).
   - TF32/MMA/cp.async/WMMA variants. Use them only as design references.
2. **Must write custom Metal kernels** (no ggml-metal equivalent with Strata's exact semantics):
   - Indexer append (single + batched).
   - Block scorer.
   - Cell-weighted radix top-k with the tie rule and the 2051 width.
   - Gathered split-K sparse attention (4 KV modes) + merge.
   - FWHT256.
   - Group-64 int8 append.
   - Fused GDN conv+L2, ab, and step+norm (decode, T-token verify/commit, and the prompt loop).
3. **Can map to ggml-metal-style kernels**, or reuse them if the port links ggml:
   - `kernel_rms_norm`, `kernel_rope_neox`/`kernel_rope_multi` (n_dims 64), `kernel_ssm_conv_f32_f32`.
   - Elementwise sigmoid/silu/softplus, `kernel_mul_mv_bf16_f32`, `kernel_get_rows`/`kernel_set_rows`.
   - `kernel_flash_attn_ext(_vec)`: only after an explicit gather, and only with ggml KV types (f16, q8_0, q4_0), not Strata's int8-g64.
   - `kernel_gated_delta_net`: **unsure** whether it exists in ggml-metal at the pinned commit.
4. **Apple constraints to design for:**
   - 32 KB threadgroup memory. This is exceeded by `block_topk_reg` (32 KB of histograms), `block_scores_tc` and both prompt-attention kernels.
   - SIMD width 32 (shuffles map 1:1).
   - No fp64.
   - Fast math is the compiler default.
   - fp32 `simdgroup_matrix` is native, so the hi/lo fp16 and 3xTF32 splits are unnecessary.
5. **Memory on unified memory** (12 QSA layers + indexer + GDN state):
   - int8 KV @256K is about 3.09 GiB, plus the indexer at 384 MiB, plus 112 MiB GDN state. Fully resident.
   - fp16 @256K is 6 GiB.
   - MTP drafter KV: window 32768 cells x 1 layer, small.
