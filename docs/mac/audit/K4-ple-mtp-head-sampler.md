> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Part 04: PLE n-gram embedding, token embedding and output head, MTP draft layer, sampler and verify-window kernels, prompt-lookup drafting

Source tree: `/tmp/strata-upstream` (read-only). Every reference below is `path:line` in that tree. The model is
Qwen3.8-Flash-Next (n_embd 2560, hc 4, hc_dim 10240, hc_lr 320, 48 layers, n_head 24, n_head_kv 2, head_dim 256,
512 experts with top-10 routing, n_ff 640, vocab 248320). `kVerifyMaxT = 8` and `DraftPolicy::kMaxT = 8`.

---------------------------------------------------------------------------------------------------------------------

## 0. Overview

Per decoded or verified token:

1. **Host hash and gather.**
   - The **host** hashes the current token plus its two predecessors into 16 row indices of a 320,001,536-row table
     (`src/kernels/ngram.cpp:67-93`).
   - It reads 16 rows of 90 B (IQ4_NL), 160 B (FP8) or 110 B (Q5_0) from the 28.8 GB GGUF shard, using unbuffered
     SSD reads, mmap, or mlock'd RAM.
   - It dequantizes the rows **on the CPU** into 2560 floats (`ngram.cpp:98-166`, `:349-375`).
   - It uploads those 10 KB to the GPU. Decode uses `cudaMemcpyAsync` (`src/core/layer.cpp:1270-1279`). Verify
     uses mapped memory plus a copy kernel (`src/core/verify.cpp:416`).
2. **PLE block on the GPU.** The PLE block runs **only before layer 1's `gr_read`** (`layer.cpp:1143-1199`). It
   computes `R += gated + silu(conv)` in place on the 4-stream residual R[10240] and advances a 9-row × 10240-channel
   conv history.
3. **Head.** After 48 layers:
   - `lm_head_mix` (`layer.cpp:1060`) is a `gr_read` with `output_hc_norm/down/up`. It produces `mixed[2560]`.
   - Q8_1 quantization plus MMVQ over `output.weight` [2560 × 248320] gives f32 logits.
   - The sampler then runs. Greedy is captured in the CUDA graph. Sampled decoding is re-run outside the graph with
     Philox(seed, position).
4. **MTP draft layer.** This is one extra transformer layer with its own K/V (dense attention over the last 32768
   cells) and 512 resident 2-bit experts.
   - At cell c it consumes the main model's final residual at position c and the embedding of the token at c+1.
     It predicts the token at c+2 through the main model's head, optionally restricted to a draft-vocabulary
     subset.
   - It chains up to `min(max_t-1, max_drafts)` drafts while the draft's probability stays at or above
     `spec_min_p`.
   - The drafts form the next verify window `[x, d0, d1, ...]`. Acceptance is exact-match against the target's
     own pick for each row.
5. **Suffix drafter.** An optional trigram-hash suffix drafter (`src/spec/suffix_drafter.cpp`) can replace the MTP
   window when `DraftPolicy` predicts more tokens per millisecond.

### Files in this slice

| File | Lines | Role |
|---|---|---|
| `include/strata/kernels/ngram.hpp` | - | PLE constants (L31-49) |
| `src/kernels/ngram.cpp` | ~441 | hash, row decode, `PleTable` (mmap / direct / ram) |
| `include/strata/ngram/ple_reader.hpp`, `src/ngram/ple_reader.cpp` | - | async unbuffered row reader, row cache, SSD keep-alive |
| `src/platform/direct_file.cpp` | - | O_DIRECT + pread thread pool (POSIX), IOCP (Windows) |
| `src/kernels/cuda/ple.cu` | 368 | legacy/reference PLE block (8 kernels) |
| `src/kernels/cuda/native_ple_postops.cu` | 221 | llama.cpp-arithmetic PLE post-ops, single + batch (7 kernels) |
| `src/core/native_head.cpp` | 199 | `NativeHead` (output.weight) and `NativeEmbed` (token_embd) |
| `src/core/native_dense.cpp` | - | loader of native GGUF dense tensors (includes `blk.1.ple_key`); no kernels |
| `src/core/mtp.cpp`, `include/strata/core/mtp.hpp` | 840 / - | MTP drafter |
| `include/strata/core/coupled_draft.hpp` | - | coupled draft sampling counter/penalty arithmetic |
| `src/kernels/cuda/sampler.cu`, `include/strata/kernels/sampler.hpp` | 933 / - | sampler (8 kernels) |
| `src/kernels/cuda/verify_kernels.cu` | 559 | verify/MTP helper kernels (23 kernels, 2 dead) |
| `src/core/verify.cpp` | 1367 | verify window, PLE snapshot/rollback, head, commit |
| `src/spec/{suffix_drafter,draft_policy,controller}.cpp` | 88/101/73 | prompt-lookup drafting and the window policy |
| `src/program/generate.cpp` | - | CLI, wiring, the speculative loop (L6101-6234) |
| `tools/{mtp_rt,draft_vocab,embd_bf16_pack,ple_fp8_pack,iq_pack}.py` | - | offline repacks |

---------------------------------------------------------------------------------------------------------------------

## 1. PLE (per-layer n-gram embedding)

### 1.1 Constants (`include/strata/kernels/ngram.hpp`)

| Line | Constant | Value | Meaning |
|---|---|---|---|
| L31 | `NG_N_EMBD` | 2560 | output width = n_embd |
| L32 | `PLE_HEAD_DIM` | 160 | one table row |
| L33 | `NGRAM_SIZE` | 3 | orders 2 and 3 |
| L34 | `HEADS_PER_NGRAM` | 8 | |
| L35 | `PLE_N_HEADS` | 16 | 16 × 160 = 2560 |
| L36 | `PLE_CONV_KERNEL` | 4 | depthwise conv taps |
| L37 | `PLE_EOS_TOKEN_ID` | 248044 | context cut token |
| L38 | `NG_HC` | 4 | hyper-connection streams |
| L39 | `NG_HC_DIM` | 10240 | 4 × 2560 |
| L40 | `NG_HIST` | 9 | conv history rows (dilation 3 × (4-1)) |
| L41 | `TOKEN_NULL` | -1 | |
| L42 | `NG_RMS_EPS` | 1e-6 | |
| L46 | `PLE_TABLE_ROWS` | 320,001,536 | = offset[15] + vocab[15] |
| L47 | `PLE_ROW_BYTES` | 90 | IQ4_NL: 5 blocks × 18 B |
| L48 | `PLE_ROW_BYTES_FP8` | 160 | E4M3 |
| L49 | `PLE_ROW_BYTES_MAX` | max of the above | raw staging size |

Hash constants come from `ple_artifact_consts()` (`src/kernels/ngram.cpp:37-56`, copied verbatim from
`docs/gguf-dump-shard1.txt`):

- `mult` (L42-44) = {23703573157769, 20109073645365, 8052911324071}
- `vocab[16]` (L45-47) = {20000003, 20000023, 20000033, 20000047, 20000059, 20000063, 20000069, 20000077,
  20000081, 20000093, 20000107, 20000147, 20000153, 20000159, 20000161, 20000171}. These are primes, one per head.
- `offset[16]` (L48-50) = {0, 20000003, 40000026, 60000059, 80000106, 100000165, 120000228, 140000297, 160000374,
  180000455, 200000548, 220000655, 240000802, 260000955, 280001114, 300001275}. These are prefix sums of `vocab`.

### 1.2 Tensors

| Tensor | Shape (ggml ne0 first) | Dtype | Notes |
|---|---|---|---|
| `per_layer_token_embd.weight` | [160, 320001536] | IQ4_NL (default); I8 with metadata `strata.ple.format = "f8_e4m3"` and `strata.ple.scale > 0` (FP8, `tools/ple_fp8_pack.py`); Q5_0 (#296, mmap only) | validated in `PleTable::open` (`ngram.cpp:171-270`): shape[0] must be 160; the size check covers both a lone tensor and a multi-tensor shard |
| `blk.1.ple_key.weight` | [2560, 10240] | canonical pack: S2 codes + f32 scales; native pack: GGUF type 42 (Q2_0), 18 (IQ3_XXS), 23 (IQ4_XS) or 8 (Q8_0); IQ model files: raw BF16 from `extra.bin` (`key_bf16`) | wired at `generate.cpp:2058-2087`; native eligibility in `native_dense.cpp:16-28`, `:56-64` |
| `blk.1.ple_value.weight` | [2560, 2560] | BF16 | `w.value_bf16` |
| `blk.1.ple_norm_key.weight` | 10240 (4 × 2560 gamma rows) | F32 | one gamma row per stream (`ple.cu:97`, `wc = w + c*n_embd`) |
| `blk.1.ple_norm_query.weight` | 10240 | F32 | |
| `blk.1.ple_norm_conv.weight` | 10240 | F32 | |
| `blk.1.ple_conv1d.weight` | [4, 10240], ne0 = 4 fast | **F16 required** | `generate.cpp:2094-2104` refuses any other storage (#255); index `kW[k + 4*c]` (`ple.cu:144-146`) |
| conv history `ss.ple.hist` | 10240 channels × 9 rows, **row-fastest** `hist[row + 9*c]` | F32 | 368,640 B (`ple.cu:155-159`) |

Table sizes:
- IQ4_NL: 320,001,536 × 90 B = 28,800,138,240 B, plus a 192 B header. That gives `kTableFileSize = 28,800,138,432`
  and `kTableDataStart = 192` (`src/kernels/ple_oracle_vectors.inc`).
- FP8: 51.2 GB.
- Q5_0: 35.2 GB.

### 1.3 Hash and row addressing (`ngram.cpp:58-93`)

```
ngram_rows(tokens, prev /*oldest first, 2 per token*/, n_tokens, consts, out[16 per token]):
  ctx[0] = token
  cut = false
  for s = 1..2:
      t   = cut ? -1 : prev[i*2 + (2 - s)]        // s=1 reads the NEWEST predecessor
      cut = cut || t < 0 || t == 248044           // evaluated BEFORE storing
      ctx[s] = cut ? 248044 : t
  for n = 2..3:
      mixed = (u64)ctx[0]*mult[0]; for j in 1..n-1: mixed ^= (u64)ctx[j]*mult[j]      // mod 2^64
      for g in 0..7: h = (n-2)*8 + g; out[h] = (u32)(mixed % vocab[h] + offset[h])
```

- Heads 0-7 are the bigram heads (current token and the previous token). Heads 8-15 are the trigram heads.
- An EOS predecessor turns that slot and every older slot into EOS. The current token being EOS does not cut.
- All arithmetic is 64-bit unsigned multiply, xor and modulo. It is a host operation by design: "ggml has no int64
  and no xor" (`layer.cpp:1151-1157`).
- Callers:
  - Decode: `ple_issue_token` (`layer.cpp:1262-1268`) calls `ngram_rows(p.token, p.prev, 1, ...)`.
  - Verify: `verify.cpp:1038-1047` hashes the T window tokens with a local `prev[2]` that rolls forward per row.
  - Prefill: the chunk is hashed the same way.
- The session keeps `ss.ple_prev[2]`, oldest first. It is shifted at commit for the accepted tokens only
  (`verify.cpp:1340-1345`).

### 1.4 Row decode

- **IQ4_NL** (`ngram.cpp:100-113`):
  - A row is 5 blocks of 18 B. Each block is an f16 `d` followed by 16 bytes `qs`.
  - Elements use **split halves**: `out[b*32 + j] = d*kIq4Nl[qs[j] & 15]` and `out[b*32 + j + 16] = d*kIq4Nl[qs[j] >> 4]`.
  - `kIq4Nl` (L96) = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113}.
- **FP8 E4M3fn** (`ngram.cpp:116-135`):
  - A 256-entry LUT. `e == 15 && m == 7` (NaN) reads as 0.
  - Subnormals use `ldexp(m/8, -6)`. Normals use `ldexp(1 + m/8, e - 7)`.
  - Each value is multiplied by the one table-wide `scale`.
- **Q5_0**: `strata::dequantize_q5_0` over 5 blocks of 22 B (`ngram.cpp:158-163`).
- **Output layout is head-slowest**: head h's 160 values are at `[h*160, (h+1)*160)`, which is ggml_get_rows order
  (`ngram.cpp:432-436`).

### 1.5 Table I/O (`PleTable`, `PleReader`, `DirectFile`)

**Modes** (`generate.cpp:222-226`, CLI L1055-1061, validation L1456):

| Option | Default |
|---|---|
| `--ple-io` | `direct` (also `mmap`, `ram`) |
| `--ple-row-cache` | 1,048,576 rows |
| `--ple-inflight` | 64 |
| `--ple-sync-submit` | off |
| `--ple-delay-us` | 0 (fault injection) |

**Mmap** (`ngram.cpp:303-322`): `read_row` decodes straight from the mapping. `--ple-io ram` adds `madvise(WILLNEED)`
and `mlock`, or touches every page if mlock fails (`ngram.cpp:266-272`). On Windows, `PrefetchVirtualMemory` issues
all 16 page faults at once (`ngram.cpp:342`, `:432`). The comment at L410-418 explains why: 16 serial page
faults cost 2.1-2.6 ms per token.

**Direct**: the mapping is dropped after the header parse. `PleReader` issues unbuffered reads into `raw[16 × row]`.
- Split API: `issue` (L324) and `collect` (L349), with one token in flight per table. `gather_batch` (L363) handles
  prompts and verify windows.
- Row cache with clock eviction and page dedup; PAGE = 4096.
- SSD keep-alive: `STRATA_SSD_KEEPALIVE` (ms, default 100) and `STRATA_SSD_KEEPALIVE_WINDOW` (s, default 60). It
  works around SN7100-class power-state stalls of 50-150 ms (`generate.cpp:2034-2041`).

**DirectFile POSIX** (`src/platform/direct_file.cpp:262-330`): `open(path, O_RDONLY | O_DIRECT)` (L308) plus a pool
of `io_threads` (16) workers, each doing a blocking `pread` (L284). The thread count is the queue depth. On Windows
it uses `FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED` and an IOCP (L166-180).

**macOS port of DirectFile:**
- `O_DIRECT` does not exist on macOS (compile error). Use `open` + `fcntl(fd, F_NOCACHE, 1)`, optionally with
  `F_RDAHEAD 0`, and the same pread thread pool.
- Alignment is still required for the uncached path: 4 KiB offsets and buffers are fine. Keep page-aligned reads
  and row-to-page dedup.
- Alternatively use `mmap` + `madvise(MADV_RANDOM)` and parallel prefetch: a 16-thread touch, or `madvise(WILLNEED)`
  on 16 pages. The Windows `PrefetchVirtualMemory` path has no direct equivalent.
- On a 128 GB+ Mac, `--ple-io ram` (mlock 28.8 GB) is viable. On a 64 GB Mac the table competes with the
  ~31.6 GB expert arena (`ngram.cpp:415-416` describes the same pressure on a 63 GB PC), so direct reads remain
  the default.

### 1.6 Block math

Inputs:
- `emb[2560]`: the gathered PLE vector.
- `hidden = R[10240]`: the 4-stream residual stack before layer 1's `gr_read`, with layer 0's pending FFN write
  already applied (`layer.cpp:1165-1170`).
- `hist[9 × 10240]`.

```
key[10240]   = gnorm_c( ple_key · emb )                 # 2560 -> 10240, then RMSNorm per stream c (4 × 2560), gamma norm_key
query[10240] = gnorm_c( hidden )                        # gamma norm_query
value[2560]  = ple_value · bf16(emb)                    # BF16 weight and BF16-rounded activation
s[c]         = (Σ_d key[c][d]·query[c][d]) / sqrt(2560) # c = 0..3
gate[c]      = sigmoid( sgn(s)·sqrt(max(|s|, 1e-6)) )
gated[c][d]  = value[d] · gate[c]                       # broadcast over the 4 streams
norm         = gnorm_c( gated )                         # gamma norm_conv
conv[ch]     = silu( Σ_{k=0..3} f16(kW[k + 4ch]) · tap_k ),  taps = hist rows 0, 3, 6 and `norm` (dilation 3)
R            = hidden + gated + conv                    # in place
hist         = shift each channel's 9 rows up by one, append norm
```

**Key projection paths** (`ple.cu:306-315`):
- `key_bf16`: `bf16_gemv_fp32_mmvf` (`native_bf16.cu:156`).
- native type 42/18/23/8: `native_quantize_q8_1` + `native_mmvq` (`native_mmvq.cu:1152`, `:1475`).
- canonical: `quantize_q8_0` (`quantize_act.cu:224`) + `s2_gemv_q8(..., tpr = 8)` (`s2_gemv_q8.cu:77`).

**Value projection paths** (`ple.cu:322-328`):
- `--native-bf16`: `bf16_gemv_fp32_mmvf`.
- Otherwise: `to_bf16_kernel` + `bf16_gemv_kernel` with **double** accumulation.

**Post-op paths:**
- **Legacy, `ple.cu`** (`ple.cu:316-346`):
  - `gnorm_kernel` rounds the product in f32 and accumulates in double; `scale = 1/sqrtf(mean + eps)`.
  - `gate_kernel` accumulates in double; `bcast_kernel`; `gnorm_kernel`; `conv_kernel`; `add3_kernel`, which
    computes `(hidden + gated) + conv`.
- **Native, `native_ple_postops`** (`native_ple_postops.cu:175-203`), llama.cpp 3cf03257 arithmetic, compiled with
  `--use_fast_math` (`CMakeLists.txt:288-303`):
  - `native_gr_rms_norm_weighted` (`native_gr_norm.cu:85`) for the key and query norms.
  - `gate_kernel`: `__fmul_rn` products, 8 partial sums, a two-level xor warp reduction, `s = __fmaf_rn(scale, sum, 0)`.
  - `broadcast_kernel`, then `native_gr_rms_norm_weighted` for norm_conv.
  - `conv_residual_kernel`: `__fmul_rn`/`__fadd_rn` with no FMA; `result = hidden + (gated + act)`.
  - Buffer aliasing: `NativePlePostopsBuffers{key = d_query, query = d_norm, gate, gated, normalized = d_norm, conv,
    result}` (`ple.cu:334`). query and normalized are allowed to alias, as are result and hidden (`:186-193`).
- **Prefill batch** (`src/prefill/prefill.cpp:1266-1306`):
  - Key and value projections run as GEMMs over sub-batches carved from idle scratch. `bf16x2()` splits the
    activation into hi and lo BF16 halves and accumulates twice to approximate an f32 activation.
  - Then `native_ple_postops_batch` (`native_ple_postops.cu:205-220`):
    1. three `rms_rep_kernel`s with `<<<T*4, 1024>>>`;
    2. `gate_kernel<<<T*4, 512>>>`;
    3. `broadcast_batch`;
    4. `conv_residual_batch`, whose taps at `p = t - 9 + 3k` read history when `p < 0`;
    5. `history_batch`, which keeps the last 9 rows.
  - The non-batch prefill fallback is `ple_block` + `ple_history_advance` per token (`prefill.cpp:1307-1321`).

**Scratch** (`ple.cu:218-225`):
- Size: `(5*10240 + 2560 + 4)` floats (16-aligned), then the Q8_0 activation (80 × 34 B, 16-aligned), then the BF16
  embedding (5120 B), plus 256.
- The run scratch adds a disjoint 10240-float `normalized` export (`layer.cpp:1286-1289`).
- `ple_block` carves this space and never allocates or syncs, so it is capture-safe (`ple.cu:252-256`, `:363-365`).

### 1.7 Where PLE runs and the history

**Decode** (`layer.cpp:1143-1199`, inside `block_layer_pre` at layer 1, half 0 or 1, captured into the graph):
`gr_write` of the pending FFN, then `ple_block(emb_dev, R, hist, w, po{normalized, result = R})`, then
`ple_history_advance(hist, normalized)`.

The host half runs **before** the graph launch: `ple_stage_token` (L1281) = `ple_issue_token` (L1262) +
`ple_finish_token` (L1270). The latter collects the rows, then does `cudaMemcpyAsync` H2D of 2560 floats from
`emb_host`. A captured host gather would replay the capture-time token forever (`layer.cpp:1151-1159`).

**Verify window** (`verify.cpp`):
1. Host: hash T tokens and call `gather_batch` into mapped `h_ple_` (L1038-1047).
2. Graph: `copy_from_mapped(ple_, m_ple_, T*N)` (L416).
3. At layer 1, for each row t (L472-490):
   - `gr_write(Rt(t))`
   - `ple_block(ple_ + t*N, Rt(t), ss.ple.hist, ...)`
   - `ple_history_advance(ss.ple.hist, normalized)`: this **mutates the live history** row by row.
   - `copy_from_mapped(hist_snap_ + t*HS, ss.ple.hist, HS)`, with `HS = 9 × 10240` floats (368,640 B per row,
     2.95 MB at T = 8).
4. **Rollback** in the commit graph: `copy_indexed(ss.ple.hist, hist_snap_, HS, commit_ + 1, HS)` (L996). The index
   `commit_[1] = n_keep - 1` lives in device memory, staged by the host in `h_commit_` (`verify.cpp:1325-1327`). The
   history becomes the snapshot after the last accepted row. `copy_indexed_kernel` is a no-op if the index is
   negative.
5. Host: `commit()` shifts `ss.ple_prev` by the accepted tokens (L1340-1345).

Contrast: GDN state is not mutated during verify. Commit replays the accepted tokens
(`gdn_step_norm_multi(..., n_keep)`, L979). PLE uses snapshots instead because its state is small and its update
is not a cheap replay.

### 1.8 Flags

| Item | Location |
|---|---|
| option fields `native_bf16_extra`, `native_ple_postops` | `generate.cpp:185`, `:207` |
| CLI `--native-bf16-extra`, `--native-ple-postops` | L1069, L1077 |
| setters `ple_set_native_bf16`, `ple_set_native_postops` | L1927, L1952 (`ple.cu:199-201`) |
| `--native-ple-key` | `generate.cpp:2077-2087` |
| `--native` preset | `generate.cpp:1514-1523` |

The `--native` preset requires `--ple-gguf` (auto-found from the shard that holds `per_layer_token_embd.weight`,
L1508). It turns on `stream_token`, `gr_native_mmvf`, `native_bf16` (+extra), `native_ple_key`, `moe_combine`,
`gdn`, `router`, `qsa`, `qsa_indexer`, `rope` and `ple_postops`, and it sets `native_head_gguf`. Without
`--ple-gguf` the PLE is off and the output "is a DIFFERENT MODEL" (`generate.cpp:2128-2131`).

---------------------------------------------------------------------------------------------------------------------

## 2. Token embedding, final norm, output head, argmax

### 2.1 Token embedding

**Tensor:** `token_embd.weight` [2560, 248320].

**Native path** (`NativeEmbed`, `src/core/native_head.cpp:108-188`):
- Accepted types: `embed_type_supported` = `is_iq` {16, 17, 18, 20, 21, 22, 23, 29, 42, 11, 12, 13, 7, 8}
  (`iq_kernels.cu:1270-1273`) plus **30 (BF16)**. The BF16 file comes from `tools/embd_bf16_pack.py` (GGUF arch
  `strata-embd`, passed with `--embd-gguf`) and is 248320 × 5120 B = 1.27 GB.
- Row bytes come from `iq_row_bytes` (`iq_kernels.cu:1335-1354`).
- **Storage: pinned host-mapped memory.** `cudaHostAlloc(Mapped | Portable)` + memcpy + device alias, so the GPU
  reads rows over PCIe and saves VRAM.
- Fallbacks to VRAM: when pinning fails (WSL2), and when the HIP device alias is missing (#325).

**Kernels:**
- `gather_dev(tokens, n, out)` (L190) → `iq_embed_rows` → `embed_rows_kernel` (`iq_kernels.cu:1384`, launch L1397,
  grid `(n_embd/256, n_tok)` × 32 threads). It runs `dq_dispatch<float>` on one 256-value superblock per block.
- `gather_one` (L194) → `iq_dequant_f32` → `dequant_flat_kernel<float>` (`iq_kernels.cu:1254`, launch L1404,
  `<<<n/256, 32>>>`).

**Canonical path:**
- `embed_row` (`layer.cpp:1005`) → `embedding_gather_kernel` (`src/kernels/cuda/elementwise.cu:19-31`, launched at
  L137). It computes `__fadd_rn(__fmul_rn(code + bias, scale[group]), offset[group])`.
- Verify and MTP use the batched `embedding_gather_dev` (`verify_kernels.cu:183`).
- `elementwise.cu` is built with `-ffp-contract=off` on HIP (`CMakeLists.txt:304-308`).

**After the gather:** `broadcast_streams` copies `emb` into all 4 residual streams (`verify_kernels.cu:203`). This is
the initial R.

### 2.2 Final norm and head mix

There is **no separate `output_norm`**. `lm_head_mix` (`layer.cpp:1060`) is
`gr_read(R, output_hc_norm.weight (F32), output_hc_down.weight / output_hc_up.weight (BF16)) → mixed[2560]`. The
final RMSNorm is the hyper-connection read's `hc_norm`. Verify does this per row at `verify.cpp:824-838`.

### 2.3 Output head

**Canonical:** `lm_head` (`layer.cpp:1081`) = `quantize_q8_K` (`quantize_act.cu:281`) + `gemv_quantized`
(`layer.cpp:140`) over the pack's code-plane `output.weight`.

**Native:**
- `NativeHead::load` (`native_head.cpp:19-69`):
  - `output.weight` from whichever shard holds it, of any `native_mmvq_supported` type (`native_mmvq.cu:1441-1446`):
    {2, 6, 7, 8, 11, 12, 13, 14, 20, 23, 42, 16, 17, 18, 21, 22, 29}.
  - Shape must be [n_in = 2560, n_out = 248320], and `n_in % 256 == 0`.
  - Memory: `cudaMalloc` + `cudaMemcpy` of the weights, plus a Q8_1 scratch.
  - A Q6_K row is 2100 B (10 × 210), so a Q6_K head is 521 MB.
- `NativeHead::run` (L71-93):
  - type 13 (Q5_K): `native_q5_k_f32` (`native_mmvq.cu:1192`, f32 activation).
  - otherwise: `native_quantize_q8_1` (L1152) + `native_mmvq` (L1475).

**Verify head** (`verify.cpp:839-857`):
- `native_quantize_q8_1(head_mixed_, T columns)` + `native_mmvq(type, weights, T columns)` produce `head_logits_`
  [T × 248320] f32 (7.9 MB at T = 8).
- Inside the graph, a greedy `sample_tokens` writes into mapped `m_out_`.

### 2.4 Argmax

`sampler_greedy_kernel` (section 4.2). It applies penalties first if a history is given, and ties go to the
**lowest index**.

### 2.5 Canonical versus native

- **Canonical** is Strata's own pack format: code planes + f32 scales (+ offsets), S2 2-bit, and kernels such as
  `s2_gemv_q8`, `gemv_quantized` and `embedding_gather`.
- **Native** is unmodified GGUF/ggml block formats: `native_mmvq` and `iq_kernels`, transcribed from llama.cpp
  3cf03257 and compiled with `--use_fast_math` to match ggml-cuda.
- **The MTP requires the native head** (`mtp.cpp:383`).
- `native_dense.cpp` launches no kernels. Its parts are `eligible` (L16-28: `blk.*` attn_qkv/attn_gate/ssm_out/
  attn_q/k/v/output/ffn_*_shexp, plus `blk.1.ple_key` for types 42/18/23/8), `served_names` (L38),
  `keep_unquantized_ple_key` (L56-64) and `load` (L71-194).

---------------------------------------------------------------------------------------------------------------------

## 3. MTP draft layer (`src/core/mtp.cpp`, `include/strata/core/mtp.hpp`)

### 3.1 Runtime files (`tools/mtp_rt.py`)

The tool converts `mtp-q2_0.gguf` into an `rt/` directory:

- **`experts.bin`**: 512 blobs of `BLOB = 3 × (2560 × 640 × 18 / 64)` = 1,382,400 B (Q2_0: 18 B per 64 weights).
  - Layout: gate/up rows interleaved (2r = gate r, 2r+1 = up r), then down rows.
  - The codes are in one plane and the fp16 scales in another (`cpu/expert.hpp` blob layout).
  - Total 707.8 MB (675 MiB).
- **`dense.bin`**: every other tensor, 256 B aligned.
  - The `Q8` set (`mtp_rt.py:29-31`) is quantized to Q8_0 with ggml reference rounding: `fc_embedding`,
    `fc_hidden`, `self_attn.{q,k,v,o}_proj`, `self_attn.indexer.index_qk_proj`, and
    `mlp.shared_expert.{gate,up,down}_proj`.
  - Hyper-connection and router weights stay BF16.
  - RMSNorm weights are stored as F32 **with +1 applied** (GemmaRMSNorm, `mtp_rt.py:86-88`).
- **`dense.txt`**: one line per tensor, `name kind rows cols offset bytes`, with kind = `q8_0|bf16|f32`. Names have
  the `mtp.` and `layers.0.` prefixes stripped.
- **`draft_vocab.bin`** (optional, from `tools/draft_vocab.py`): raw little-endian int32 token ids with no header;
  `size % 4 == 0`.
  - CLI: `--gguf --base --add han|kana|hangul|cjk_punct|cjk|cyrillic --out --stats`.
  - Ids keep the base order, and added ids follow in ascending id order (#137).

### 3.2 Load (`mtp.cpp:142-302`) and bind (L375-410)

**Load:**
- `max_t` must be in 1..`kVerifyMaxT`.
- Uploads: `dense.bin` is read whole and `cudaMemcpy`'d. `experts.bin` goes in 64 MiB `fread` chunks, each
  `cudaMemcpy`'d (L180-197).
- Required Q8_0 tensors: L201-204.
- **K/V** (L206-236):
  - The drafter's own QSA state, dense attention with no indexer.
  - With `window` (default 32768, `mtp.hpp:45`) below `max_cells`, it is a ring of `window + 4*max_t + 64` cells
    over a pinned host copy.
  - K8V4 is forced to plain INT8.
  - If pinned RAM is exhausted, `ring = -1` (fully resident in VRAM).
  - `cap_ = round_up_64(window or max_cells)` (L239-240).
- Total about 0.9 GB (`mtp.hpp:43-44`).

**Bind:**
- Requires a loaded `NativeHead`. `n_vocab = output.weight.ne1`, and `head_logits_` is `max_t × n_vocab` floats.
- If `draft_vocab.bin` exists:
  - `dvocab_` holds the ids.
  - `dhead_ = gather_rows(head weights, row_bytes, dvocab_)` (`verify_kernels.cu:339`) builds a compact subset of
    the head rows.
  - `dinv_` is the inverse map used by coupled penalties.
- `setup_coupled` runs if coupled drafting is on.

### 3.3 Forward (`MtpDrafter::record_forward`, `mtp.cpp:413-580`)

`step_row0 >= 0` means the full layer for 1 row. `step_row0 < 0` means a K/V-only pass for T rows.

| # | Stage | Tensors / kernels | mtp.cpp |
|---|---|---|---|
| 1 | token embedding of `tok_` (the token at c+1) | `NativeEmbed::gather_dev` or `embedding_gather_dev` | L428-437 |
| 2 | enorm | `native_qsa_rms_norm_weighted(emb, pre_fc_norm_embedding.weight)` (`native_qsa.cu:108`) | L439 |
| 3 | e-proj | Q8_1 + `native_mmvq(Q8_0, fc_embedding.weight)` 2560→2560 | L440-441 |
| 4 | hnorm over the full 10240 (`Rin_` = main final residual at c) | `native_qsa_rms_norm_weighted(Rin_, pre_fc_norm_hidden.weight, cols = HC*N)` | L442 |
| 5 | h-proj **per stream** (T×4 columns, chunks of 8) | `native_mmvq(Q8_0, fc_hidden.weight)` 2560→2560 | L443-447 |
| 6 | combine | `add_streams_broadcast`: R[t][c] = h2[t][c] + e2[t] | L448 |
| 7 | attention hyper-connection read (apply = false) | `fused_gr_read_multi` with `attn_hyper_connection.{hc_norm, input_mix_weight_down, input_mix_weight_up, block_inject_weight}` (`fused_gr.cu:803`) | L450-462 |
| 8 | K, V | Q8_1 + `native_mmvq(Q8_0, self_attn.k_proj / v_proj)` 2560→512 | L469-471 |
| 9 | k_norm + RoPE at the cell position | `native_qsa_rms_norm_weighted` + `native_rope_apply` (`native_rope.cu:118`) or `rope_neox_apply` (`rope.cu:117`); `fwht256_inplace_cuda` if `kv_rot` | L472-477 |
| 10 | K/V append | `kv_append_q4_step` (`kv_q4.cu:193`), `kv_append_q8_step` (`kv_q8.cu:103`) or `kv_append_step` (`qsa.cu:595`) | L478-487 |
| - | **return here if K/V-only** | | L488 |
| 11 | Q + output gate | `native_mmvq(Q8_0, self_attn.q_proj)` 2560→12288 (per head [q 256, gate 256]); `cudaMemcpy2DAsync` splits q | L489-497 |
| 12 | q_norm + RoPE (+FWHT) | | L498-499 |
| 13 | window | `window_ids(step, T, 32768, ident_, cap_)`: cells `[n_kv-32768, n_kv)` | L502 |
| 14 | dense GQA attention 24q/2kv | `qsa_decode_attn_batch(qcur_, pools, ident_, step, cap_)` (`qsa_decode_attn.cu:211`) | L503 |
| 15 | gate + o-proj | (FWHT⁻¹), `native_qsa_gate_apply` (`native_qsa.cu:120`) with the gate half of q_proj, Q8_1 + `native_mmvq(o_proj)` 6144→2560 | L504-508 |
| 16 | MLP hyper-connection (apply = true, folds the attention write) | `fused_gr_read_multi` with `mlp_hyper_connection.*` | L511-523 |
| 17 | router | `bf16_gemv_fp32_mmvf(mlp.gate.weight)` 2560→512, then `native_router_top10` (`native_router.cu:110`) or `router_top10` (`router_top10.cu:345`) | L525-529 |
| 18 | routed experts (2-bit, all 512 resident) | `moe_group_resident(experts_, BLOB)` (`s2_expert_grouped.cu:1090`) → `quantize_q8_0_scaled` (`quantize_act.cu:245`) → `moe_grouped_s2` (`s2_expert_grouped.cu:1099`) → `parts_` [T×10×2560] | L530-533 |
| 19 | shared expert + combine + write | `f32_to_bf16_bulk` (`elementwise.cu:172`); `shared_expert` (`shared_expert.cu:209`), Q8_0 gate/up/down through `NativeSharedWeights` with BF16 `mlp.shared_expert_gate.weight`; `native_moe_combine` (`native_moe.cu:60`) or `moe_combine` (`shared_expert.cu:325`); `gr_write` (`gr.cu:413`) | L534-551 |
| 20 | final mixer | `gr_read(hyper_connection_mixer.*)` (`gr.cu:344`) → `sample_[2560]` | L553-557 |
| 21 | draft head | Q8_1 + `native_mmvq(head type, dhead_ (n_dvocab rows) or the full head)` | L558-561 |
| 22a | coupled mode | `coupled_draft_sample(...)` (section 4.6) | L562-568 |
| 22b | default | greedy `sample_tokens`; `row_top_prob` = softmax probability of the argmax **over the subset only**; `map_ids` maps subset ids to token ids | L569-574 |

**Probability caveat:** with a draft vocabulary, `row_top_prob` normalizes over the subset. The reported probability
is therefore an overestimate of the full-vocab probability, and the 0.5 threshold is applied to it.

### 3.4 Graphs and the draft loop

**Graphs:**
- `capture_prefill` (L603) and `capture_prefill_dev`.
- `capture_round(T, coupled)` (L621-651):
  1. Coupled stage.
  2. Copies from mapped memory: tok[T], step[2·max_t × 4], pos[2·max_t × NH], row[2], and
     `Rin_ ← window_R_` (T × 10240, the verify window's final residual rows).
  3. `record_forward` K/V-only for T rows (cells p..p+T-1, pairs (R_t, outv[t])).
  4. `mtp_select(row = a)` → Rin_/tok_.
  5. Full layer at step row `2·max_t - 1` (cell p+a).
  6. `mtp_select(R_, out_ids_)` → Rin_/tok_ for the next step, plus volatile writes of `m_out_[0]` and `m_prob_[0]`.
- `capture_step(j)` (L655-670): full layer at step row `max_t + j - 1`, then `mtp_select` into `m_out_[j]`.
- `finish_capture` (L583-600). `kv_restore` (L672-680).

**`draft()` (L771-823):**
- The host stages `put(row, cell)`: `step = [cell, cell+1, (cell+1)/4, cell+1]` and `pos[h] = cell` for every head.
- Rows t < T get cells p+t with `tok = outv[t]`. Row `2·max_t - 1` gets cell `p + a`. `h_row_ = {a, 0}`.
- `cudaGraphLaunch(round) + cudaStreamSynchronize`, then a volatile read of `drafts[0]` and `pj`.
- Chain loop: `for j = 1; j < min(max_t - 1, max_drafts) && pj >= min_p`. Each step runs
  `capture_step(j)`, `put(max_t+j-1, p+a+j)`, launch + **sync**, then reads `drafts[j]` and `pj`. That is **one
  host round trip per draft**.
- Unused slots are zeroed.
- `draft_first` (L825-838) handles the first round without a native pack: it copies the current R row into all T
  rows, uses the token, and sets `cell = p - 1`.
- `prefill` (L682-769): K/V-only passes over the prompt in groups of max_t rows. It skips cells the window can never
  reach (`first_needed = prompt_len - window - 64`). All records are uploaded once, with a single sync at the end.

**Number of drafts and the threshold:**
- `S_mtp = min(mtp_max_t, spec)` (`generate.cpp:6117`). `set_max_drafts(S_mtp - 1)` (L6118, L4664).
- The engine default `spec_min_p` is 0.0 (L362). With 0 the window is always S_mtp and the chain always runs to
  `max_drafts`.
- `setup.py:3126` passes `--spec 4 --spec-min-p 0.5 --mtp rt`, so at most 3 drafts per window (T ≤ 4).
- The engine maximum is 7 drafts (kVerifyMaxT 8).
- CLI flags: `generate.cpp:1103-1169`.

### 3.5 How drafts feed verify (`generate.cpp:6101-6234`)

```
T = S_mtp; if (use_mtp && spec_min_p > 0) { T = 1; while (T < S_mtp && dprob[T-1] >= spec_min_p) ++T; }   // L6131-6134
first_window (native pack) -> T = 1
suffix proposal may replace the window (section 4.8)
window = [x, drafts[0..T-2]]
ver.run(T, window, p, ...) -> outv[0..T-1]          // outv[t] = target pick after window[0..t]
a = 0; while (a < T-1 && window[a+1] == outv[a]) ++a   // L6175-6176
ver.commit(a+1)                                      // KV positions, GDN replay, PLE rollback, ple_prev
produced += outv[0..a]
mtp.draft(T, outv, p, a, drafts, dprob, spec_min_p)   // uses this window's rows
x = outv[a]; p += a + 1
```

MTP K/V for rejected rows (cells > p+a) is overwritten by the next round before any query can read it, because
attention at cell c reads only cells ≤ c.

---------------------------------------------------------------------------------------------------------------------

## 4. Sampler, speculative acceptance, verify kernels

### 4.1 Parameters and chain (`include/strata/kernels/sampler.hpp`)

`SamplerParams` (L15-28):

| Field | Default | Note |
|---|---|---|
| `top_k` | 20 | 1..64 as given; 0 or more than 64 means 64 |
| `top_p` | 0.95 | |
| `min_p` | 0 | |
| `temperature` | 1 | ≤ 0 means greedy |
| `min_keep` | 1 | |
| `penalty_last_n` | 0 | |
| `penalty_repeat` | 1 | |
| `penalty_freq` | 0 | |
| `penalty_present` | 0 | |
| `seed` | 0 | |
| `counter` | 0 | |
| `greedy` | false | |

- **Chain** (header L3-6): penalties → top_k → top_p → min_p → temperature → inverse-CDF pick. This is llama.cpp's
  default order: top_p cuts before min_p, and temperature comes after truncation. top_p and min_p therefore see the
  untempered logits.
- **`penalty_rows`** (~L68-79): row t of a verify window gets the last h tokens of (consumed, window drafts 1..t),
  padded with -1.
- **Penalties** (`sampler.cu:67-79`, from `llama_sampler_penalties_apply`):
  - `count = history_count(window, v)`.
  - If count > 0: multiply by `repeat` when logit ≤ 0, else divide.
  - Then `logit -= count*freq + (count > 0)*present`.
- **Philox** (`sampler.cu:39-61`). This is a **non-standard variant**: the key is (i, 0) per round, no key bump, and
  the word placement differs from Random123.
  ```
  round(c, k0, k1): hi0=umulhi(0x9E3779B9,c0); hi1=umulhi(0xBB67AE85,c2); lo0=0x9E3779B9*c0; lo1=0xBB67AE85*c2
                    c = (hi1^c1^k0, lo1, hi0^c3^k1, lo0)
  uniform(seed, counter): c = (lo32(counter), hi32(counter), lo32(seed), hi32(seed)); for i in 0..9: round(c, i, 0)
                          u = (c0 >> 8) * 2^-24          // [0,1), 24 bits
  ```
  The host mirror is `host_philox_uniform` (`src/kernels/sampler_parity.cpp:134-159`). Row t uses
  `counter = p.counter + t`.
- **Position-tied draws:** verify sets `counter = pos0` (`verify.cpp:1175`). Row t's draw is therefore
  Philox(seed, pos0 + t). It depends on the absolute position, not on how text was cut into windows or what the
  drafts were. That makes speculative acceptance under sampling **exact**: the output is a function of the seed only,
  and a rejected row's draw is simply discarded (`verify.cpp:1164-1167`).

### 4.2 Sampling paths (`sample_tokens`, `sampler.cu:836-886`)

**Greedy** (`greedy || temperature <= 0`): `sampler_greedy_kernel<<<rows, 1024, bitmap>>>`.

**Sampled**, by path:
- default Split: `sampler_split_part_kernel<<<(61, rows), 128>>>` + `sampler_split_merge_kernel<<<rows, 32>>>`.
- `STRATA_SAMPLER_ONE_BLOCK=1`: `sampler_one_block_kernel<<<rows, 1024, bitmap>>>`. This is also the fallback when
  Split cannot run: under stream capture (`stream_capturing`, L771), with no scratch, with more than 64 rows, or with
  more than 64 blocks (vocab > 262,144).
- `STRATA_OLD_SAMPLER=1`: `sampler_kernel<<<rows, 1024, bitmap>>>` (engine 0.1.20).

**Bitmap and scratch sizes:**
- Bitmap smem = `ceil(n_vocab/32) × 4` = 31,040 B for 248,320, allocated only when penalties are on.
- Split scratch: `split_scratch` (L789-832) holds a `cudaMalloc` per (device, stream). It is sized for at least
  16 rows × n_blocks × 64 entries, grows by doubling, and retires old buffers instead of freeing them under use.

**Where sampling runs:**
- Inside the verify graph only greedy runs.
- A sampled or penalized request is sampled again after the replay, outside the graph, with its own params,
  `counter = pos0` and `hist_d_`, followed by `cudaStreamSynchronize` (`verify.cpp:1164-1180`). A captured sampler
  would bake in its parameters and replay the same draws forever.

### 4.3 The selection and the tail

**Selection order** (all paths): value descending, ties by lower id. `take_first` (L337) and `warp_first` (L344, an
xor butterfly that is exact because the order is a strict total order).

**`sampled_k`** (L331-334): k = top_k if 1 ≤ top_k < 64, else 64, capped at n_vocab.

**One-block rounds:** round i takes the candidates strictly after round i-1's pick (`s < prev_v || (s == prev_v &&
v > prev_i)`). An empty round yields the sentinel (-inf, n_vocab), stored as id 0.

**`sampled_tail_warp<kProb>`** (L362-424). One warp runs it. It uses **FP64**.
1. `mx = max(sel_logit[0..k))`.
2. **top_p** (if < 1):
   - `ex[i] = exp((double)l_i - mx)` across lanes.
   - Lane 0 computes the ordered sum, then `ex /= sum`.
   - Lane 0's cumulative scan picks `cut` = the first i with `cum >= top_p`, plus 1, with at least `min_keep`.
3. **min_p**: `thresh = sel_logit[0] + logf(min_p)` (float). Keep the prefix with `l ≥ thresh`.
4. **temperature**:
   - `inv_t = 1/T`, `smx = max(l·inv_t)`, `ex = exp((double)(l·inv_t) - smx)`.
   - Lane 0 computes the ordered sum, then normalizes.
5. **pick**: lane 0 computes `u = philox_uniform(seed, counter + t)`. The pick is the first i with `u < cum`,
   defaulting to the last kept entry.
6. With `kProb` (coupled only), lane 0 also writes `ex[pick]`.

### 4.4 Kernel-level description of the sampler

See section 5.3 for launch details. In short:
- **greedy:** a bitmap of history ids via `atomicOr`, a strided per-thread argmax, then a shfl_down reduction with
  the lowest-index tie rule.
- **one-block:** k block-argmax rounds, then warp 0's tail.
- **split part:** each block handles 4096 logits. Each lane holds 32 logits in registers, with penalties applied
  through a 512 B block-local bitmap. There are k rounds per warp over two chains (even/odd), merged with
  `warp_first`. Warp 0 merges the 4 warp lists into `cand[(t*n_blocks + b)*k]`.
- **split merge:** one warp per row. It loads `n_blocks × k` candidates into a 32 KB smem list, `warp_merge_lists`
  (L522-561) takes the global top k, then runs the tail.

### 4.5 Speculative acceptance under sampling

- Acceptance is **exact-match**: `window[a+1] == outv[a]`.
- Under greedy decoding, outv[t] is the argmax.
- Under sampling, outv[t] is the target's sample with u = Philox(seed, pos0 + t).
- There is **no rejection-sampling ratio**. The draft only changes how many tokens a window yields, never which
  tokens are produced.

### 4.6 Coupled draft sampling (`include/strata/core/coupled_draft.hpp`, off by default)

**Enabling:** `STRATA_SPEC_COUPLED=1` or `--coupled-draft`.

**Idea:** the MTP **samples** its draft with the target's chain and **the same uniform** the target will draw for
the row that verifies it. Two inverse-CDF picks from one u agree far more often than an argmax agrees with a sample.

**Counter arithmetic:**
- Chain step j runs at cell `p + a + j` (`coupled_draft_cell`).
- It is verified with counter `cell + 1` (`coupled_draft_counter`). That is row j of the next window, which starts
  at p' = p + a + 1.

**Penalty ring:**
- `kCoupledHistCap = 4096` slots plus max_t.
- The host stages the base (the last h of consumed' + x') at `[cap - h, cap)`, and draft j writes `ring[cap + j]`.
- Draft j's window is therefore `[cap + j - h, cap + j)`, with `h = min(penalty_last_n, cap)`.

**Kernels** (`coupled_draft_sample`, `sampler.cu:912-931`, captured in the drafter graphs):
1. `coupled_stage_kernel`: params and ring base from mapped memory.
2. `coupled_penalize_kernel`.
3. `sampler_split_part_kernel` with grid `(n_blocks, 1)`.
4. `coupled_merge_kernel`: reads the counter from `step_rec[0]`, runs `sampled_tail_warp<true>`, and writes the
   token id, its probability and `ring[cap + j]`.

**Test:** `coupled_draft_test` (`tests/core/coupled_draft_test.cpp`, CMake L769-771).

### 4.7 Verify-window kernels

`verify_kernels.cu` holds 23 `__global__` kernels; see section 5.4. Their roles:
- **GDN multi-token verify and replay:** `gdn_conv_l2_multi`, `gdn_conv_commit`, `gdn_ab_multi`,
  `gdn_step_norm_multi`.
- **Embedding and stream helpers:** `embedding_gather_dev`, `broadcast_streams`, `add_streams_broadcast`.
- **Rollback:** `copy_indexed`.
- **Expert staging over PCIe:** `fetch_blobs`, `rebase_ptrs`, `resident_plan`.
- **Host/GPU doorbells:** `wait_flag_ge`, `wait_flag_ge_or`, `copy_i32_unless`, `copy_or_zero`.
- **MTP helpers:** `window_ids`, `gather_rows`, `map_ids`, `row_top_prob`, `mtp_select`.
- **Profiling:** `gpu_stamp`.
- **Dead code (no callers):** `ident_hits`, `dense_steps`.

The file header (L3-4) claims every per-token arithmetic is a bit-for-bit transcription of the single-token kernel.
**No unit test exists for this file.**

Mapped-memory helpers used by verify and MTP live in `elementwise.cu`:
- `copy_from_mapped_kernel` (L226, wrapper L264) and `copy_i32_from_mapped_kernel` (L297, wrapper L301): volatile
  reads of zero-copy host memory.
- `doorbell_wait_kernel` (L216).
- `strata_spin_pause` (`dp4a.hpp:36`): `__nanosleep(100)` on sm70+.

### 4.8 Prompt-lookup / suffix drafting (brief)

**`SuffixDrafter`** (`src/spec/suffix_drafter.cpp`, header `include/strata/spec/suffix_drafter.hpp`):
- Construction: `SuffixDrafter(min_match = max(1, --suffix-draft), max_match 64, capacity max_context + 4096)` at
  `generate.cpp:6119`. Internally `min_match_ = max(3, min_match)`.
- Hash table: open addressing at load ≤ 0.5. Each slot is keyed by a mixed hash of the last 3 tokens and holds the 4
  most recent end positions (WAYS 4).
- `propose(max_k)`: find the longest backward match (≤ 64) among the 4 candidates, preferring the newest on ties.
  Require length ≥ min_match, then return up to max_k continuation tokens.

**`DraftPolicy`** (`src/spec/draft_policy.cpp`, margin 0.03):
- Compares the expected tokens per ms of the MTP window (`mtp_tokens(T)/cost_ms(T)`, EMA-learned) against lookup
  windows. The lookup value is E(k) = 1 + q + … + q^k, with q per match-length bucket {<6, <12, <24, ≥24} and
  priors {0.75, 0.88, 0.93, 0.96}.
- Costs come from EMAs of measured round times, with the shape prior `kShape` for unmeasured sizes.
- It sometimes probes unmeasured sizes (`kProbes 3`, q ≥ 0.85).

**Use in the loop:** a proposal is used only if `k > 0 && (!use_mtp || sbuf[0] == drafts[0])`
(`generate.cpp:6139-6145`).

**Files:** `controller.cpp` (73 lines) and `suffix_drafter_test` (CMake L753).

**Metal port:** pure host C++, portable as-is.

---------------------------------------------------------------------------------------------------------------------

## 5. Kernel catalog (every `__global__` in the four files, plus those launched for embed/head/MTP)

**Legend:**
- grid/block are the launch shapes; smem is shared memory.
- "fm" means compiled with `--use_fast_math` (only `native_*.cu`, `CMakeLists.txt:288-303`). The others use nvcc
  defaults: FMA contraction on, IEEE div/sqrt, no FTZ.

**Metal primitive mapping used below:**

| CUDA | Metal |
|---|---|
| warp | simdgroup (32 lanes on all Apple GPUs) |
| `__shfl_xor_sync` | `simd_shuffle_xor` |
| `__shfl_down_sync` | `simd_shuffle_down` |
| `__shfl_sync(v, 0)` | `simd_broadcast(v, 0)` |
| `__syncwarp` | `simdgroup_barrier(mem_threadgroup)` |
| `__syncthreads` | `threadgroup_barrier(mem_threadgroup)` |
| smem `atomicOr` | `atomic_fetch_or_explicit` on `threadgroup atomic_uint` |
| `__umulhi` | `mulhi` |
| `__launch_bounds__(N)` | `[[max_total_threads_per_threadgroup(N)]]` |
| `extern __shared__` | `[[threadgroup(0)]]` + `setThreadgroupMemoryLength` |
| `__half2float(__ushort_as_half(x))` | `float(as_type<half>(x))` |
| `__fmul_rn` / `__fadd_rn` (no contraction) | compile with fast math off (`MTLCompileOptions.mathMode = .safe`, or `fastMathEnabled = false` on older SDKs) and `#pragma METAL fp contract(off)`; use explicit `fma()` where CUDA used `__fmaf_rn` |

- **MSL has no `double`.** Every FP64 accumulation needs float-float emulation, or must move to the CPU.
- **Threadgroup memory limit is 32 KB**; CUDA allows 48 KB static.

### 5.1 `src/kernels/cuda/ple.cu` (8 kernels, `THREADS = 256`, L36)

| Kernel (line) | Purpose | I/O | grid × block, smem | CUDA features | Parity | Metal note |
|---|---|---|---|---|---|---|
| `history_advance_kernel` (L46) | shift each channel's 9 rows, append the new normalized row | inout `history[9×10240]` (row-fastest), in `normalized[10240]` | ⌈10240/256⌉=40 × 256 (L209), 0 | `__restrict__` (the host rejects overlap, L206-208) | `ple_parity` guard tests L207, L227-229 (exact); Part C | trivial; 36 B strided per thread. Keep the layout, because snapshot/rollback copies the whole state |
| `gnorm_kernel` (L92) | per-stream weighted RMSNorm; product rounded in f32, **double** sum; `scale = 1/sqrtf(mean+eps)` | in x[4×2560], w[4×2560]; out y (in place allowed) | 4 × 256, `double scratch[8]` | `block_sum` (L74-87): double `__shfl_down_sync`, leading `__syncthreads` | Part C host ref ≤ 1e-4 per stage (L1131), result rel_l1 ≤ 1e-2 (L784-794) | no FP64: use float-float, or use the native path (`rms_rep`/`native_gr_rms_norm`, float), which is the native-pack default |
| `gate_kernel` (L115) | `s = Σ(k·q)` (f32 product, double sum) × 1/√2560 → `sigmoid(sgn·sqrt(max(|s|,1e-6)))` | in key, query [4×2560]; out gate[4] | 4 × 256, `double[8]` | double shuffles, `expf` | as above | same FP64 issue; prefer the native gate |
| `bcast_kernel` (L130) | `gated[i] = value[i%2560]·gate[i/2560]` | | 40 × 256, 0 | | Part C | trivial |
| `conv_kernel` (L147) | dilated causal depthwise conv (k=4, dil=3) over hist rows 0, 3, 6 + new norm, F16 weights `kW[k+4c]`, SiLU | in hist, norm, kW (u16); out conv | 40 × 256, 0 | f16 decode (`f32_from_f16`); `acc += w·v` (nvcc contracts to FMA) | Part C | use `fma()` explicitly to match the nvcc contraction; `expf` precise |
| `add3_kernel` (L166) | `result = (hidden + gated) + conv` | in place on R allowed | 40 × 256, 0 | | Part C | trivial |
| `bf16_gemv_kernel` (L175) | value projection: `y[o] = Σ bf16(x)·bf16(w)` in **double**, one thread per output row | x u16[2560], w u16[2560×2560]; y[2560] | 10 × 256, 0 | naive serial row loop, uncoalesced | native value ≤ 2e-6 vs BF16 separation > 1e-5 (L827) | rewrite as a simdgroup-cooperative GEMV. bf16×bf16 products are exact in f32, so only the sum rounding differs: use a float-float sum for parity, or the native `bf16_gemv_fp32_mmvf` arithmetic |
| `to_bf16_kernel` (L185) | f32 → bf16 RNE (`bf16_bits`, L54; NaN not special-cased) | | 10 × 256, 0 | `memcpy` bit casts | | `as_type<uint>` + the same rounding formula, not `bfloat` conversion (that would need NaN-equivalence checks) |

### 5.2 `src/kernels/cuda/native_ple_postops.cu` (7 kernels, fm; N=2560, H=4, D=10240, HISTORY=9 at L35)

| Kernel (line) | Purpose | I/O | grid × block, smem | Features | Parity | Metal note |
|---|---|---|---|---|---|---|
| `gate_kernel` (L40) | ggml SUM_ROWS emulation: `sums[8]` per thread (d = tid + j·512), `__fmul_rn` products, xor warp sum, `partials[32]`, 2nd warp sum over lanes < 16; `s = __fmaf_rn(scale,sum,0)`; `gate = 1/(1+expf(-sgn·mag))` | key, query [rows×2560] → gate[rows] | H (or T·H) × 512, 128 B | `__shfl_xor_sync`, `_rn` intrinsics, fast-math `expf`/`sqrtf` | ggml-oracle `native_ple_postops_parity` (bench/micro, **absent**); `ple_parity` native paths ≤ 2e-6 | keep the 512-thread shape and reduction order (16 simdgroups); `fma(scale,sum,0)`; Metal fast `exp` ≠ CUDA `__expf`, so expect ULP-level gate differences and use tolerance tests |
| `broadcast_kernel` (L67) | `gated = __fmul_rn(value[i%N], gate[i/N])` | | 40 × 256 | | | trivial |
| `conv_residual_kernel` (L71) | taps `history[c·9 + 3k]` (k<3) + normalized; `__fmul_rn`/`__fadd_rn` sequence (no FMA); `act = sum/(1+expf(-sum))`; `result = hidden + (gated + act)` | out conv, result (`result == hidden` allowed) | 40 × 256 | `__half2float(__ushort_as_half)` | | contract(off); note the addition order differs from `add3_kernel` |
| `rms_rep_kernel` (L96) | weighted RMSNorm, 1024 threads per 2560-row, float partials, xor warp sums, `sums[32]`; `rsqrtf(mean+eps)`; gamma row = `blockIdx % 4` | in/out may alias | T·4 × 1024, 128 B | `rsqrtf` (fm) | | 1024-thread threadgroup is allowed on Apple GPUs; check `pipeline.maxTotalThreadsPerThreadgroup`. Metal `rsqrt` precision differs, so use a tolerance |
| `broadcast_batch_kernel` (L121) | batch broadcast | value[T×N], gate[T×4] → gated[T×D] | ⌈T·D/256⌉ × 256 | | | trivial |
| `conv_residual_batch_kernel` (L128) | batch conv; tap `p = t-9+3k`, reads `history[c·9 + 9+p]` when p<0; `hidden += gated + act` in place | | ⌈T·D/256⌉ × 256 | | | trivial |
| `history_batch_kernel` (L146) | new history = last 9 normalized rows (older from history when T<9) | | 40 × 256 | registers h[9] | | trivial |

Host wrappers (L175-220) validate non-null, 4-byte aligned, non-overlapping spans. `native_ple_postops` is used by
`ple_block` when `--native-ple-postops`.

### 5.3 `src/kernels/cuda/sampler.cu` (8 kernels)

| Kernel (line) | Purpose | I/O | grid × block, smem | Features | Parity | Metal note |
|---|---|---|---|---|---|---|
| `sampler_greedy_kernel` (L100) | penalized argmax; a history-tail bitmap makes membership O(1) and only hits pay the count; ties go to the lowest id; all -inf → 0 | logits [rows×V], history [rows×len] (tail = last `penalty_last_n`) → out[rows] | rows × 1024; dynamic bitmap ⌈V/32⌉·4 = 31,040 B + `sv/si[32]` 256 B | smem `atomicOr`, `__shfl_down_sync` | `sampler_parity` fixtures 1-4, 13, 15 (exact tokens) | fits in 32 KB (31,296 B). The argmax with a total-order tie rule is exact for any thread count |
| `sampler_kernel` (L187) | old (0.1.20) sampled path: k argmax rounds with a `taken` sweep; tail in double on **all** threads | | rows × 1024; bitmap + `sel_ids/sel_logit[KMAX]` + `sv/si` | double | `sampler_parity_old` (CMake L581-582) | **do not port** (env-only legacy) |
| `sampler_one_block_kernel` (L433) | k block-argmax rounds over the "after the previous pick" set, then warp-0 tail | → out[rows] | rows × 1024, `__launch_bounds__(1024)`; bitmap 31,040 + `sel_ids/logit[64]` 512 + `ex double[64]` 512 + `sv/si` 256 = 32,320 B | double tail | `sampler_parity_one_block` (L579-580) | exactly at the 32 KB limit: shrink `ex` (float-float = 8 B, same size) or run the tail on the CPU |
| `sampler_split_part_kernel` (L568) | stage 1: 4096 logits per block (32 per lane in registers), block-local 512 B penalty bitmap, k rounds per warp (2 chains + `warp_first`), warp 0 merges 4 lists | → `cand[(t·n_blocks+b)·k]` int2 (id, float bits) | (⌈V/4096⌉=61, rows) × 128, `__launch_bounds__(128)`; `bits[128]` 512 B + `wl[4][64]` int2 2 KB | xor shuffles, `__float_as_int` | `sampler_parity` (default split, L573-578) | `as_type<int>`; 32 registers per lane fine; dispatch `(61, rows, 1)` × 128 |
| `sampler_split_merge_kernel` (L651) | stage 2: one warp per row; load n_blocks×k candidates into smem, `warp_merge_lists` (L522-561), then tail | → out[rows] | rows × 32, `__launch_bounds__(32)`; `lists[64·64]` int2 = 32,768 B + 1,024 B = **33,792 B** | double tail | default `sampler_parity` | **exceeds Metal's 32 KB**: size lists to n_blocks·k (61·64·8 = 31,232 B), or merge from device memory; FP64 tail → CPU or float-float |
| `coupled_stage_kernel` (L678) | copy `SamplerParams` and the ring base `[cap-h, cap)` from mapped host memory into device copies | → `dp`, `ring` | 1 × 256, 0 | volatile reads of mapped memory | `coupled_draft_test` (host arithmetic) | unnecessary with shared buffers: write params into the buffer before commit |
| `coupled_penalize_kernel` (L692) | apply penalties in place to draft logits (subset via `id_to_sub`), each distinct token once | logits[nv] | 1 × 1024; bitmap ⌈nv/32⌉·4 | `atomicOr` return value used for dedup | | `atomic_fetch_or` returns the old value, so same logic |
| `coupled_merge_kernel` (L717) | merge + `sampled_tail_warp<true>` with `counter = step_rec[0]+1`; writes id, prob, `ring[cap+j]` | | 1 × 32; lists 32,768 + ~1,032 B | double | | same 32 KB issue; since `draft()` already syncs per step, run this tail on the CPU at no extra sync cost |

**Host side:**
- `SampledPath` / `env_flag` (L755-767), `stream_capturing` (L771), `split_scratch` (L789-832), `sample_tokens`
  (L836-886), coupled wrappers (L888-931).
- Scratch: under Metal, preallocate one 64 rows × 64 blocks × 64 entries × 8 B (2 MB) buffer.

### 5.4 `src/kernels/cuda/verify_kernels.cu` (23 kernels; S=128, RG=4, RPG=32 at L16-18)

| Kernel (line) | Purpose | I/O | grid × block, smem | Features | Callers | Metal note |
|---|---|---|---|---|---|---|
| `gdn_conv_l2_multi_kernel` (L25) | GDN causal conv (k=4) over `[hist(3) | qkv rows]` for T tokens + SiLU (`__expf`) + L2 norm per q/k head (blocks < qk_heads) | hist[C×3], qkv[T×C], w[C×4] → h[T×C] | (C/128, n_tok) × 128, `part[4]` | `__launch_bounds__(128)`, xor shuffles | `verify.cpp:530` | 4 simdgroups per head; `fast::exp` vs `__expf` differ in ULP |
| `gdn_conv_commit_kernel` (L53) | after acceptance, hist = last 3 of `[hist | x_0..x_{n-1}]`, n = `*n_keep` (device) | inout hist | ⌈C/256⌉ × 256 | device-resident count | L978 | trivial |
| `gdn_ab_multi_kernel` (L70) | BF16 GEMV of alpha/beta rows for ≤8 tokens: one warp per row, uint4 = 8 bf16 per load, `fmaf` chain; `beta = sigmoid`, `gate = softplus(a+dt)·ssm_a` | x[T×n], wa/wb u16 → gate/beta [T×h_v] | ⌈2h_v/8⌉ × 256 | `__ldg`, bf16 via `<<16` / `&0xffff0000` | L532 | `as_type<float>(w << 16)`; `log1p(exp)` precise |
| `gdn_step_norm_multi_kernel` (L115) | gated delta-rule update over n tokens per head (state 128×128 in registers, 32 rows per thread × 4 row groups) + gated RMSNorm output (`sigmoid(z)`); `n_keep != null` → replay and write state back; `t_out_begin` skips outputs | inout state, out y | h_v × (128,4), smem sk/sq[128], red[4][128], wsum[16] ≈ 3.1 KB | `__launch_bounds__(512)`, `rsqrtf`, `__expf` | verify L539 (no write-back), commit L979 (replay) | 512 threads, 32 float registers per thread for state; check occupancy. Verify-without-mutation then replay-on-commit maps 1:1 |
| `embedding_gather_dev_kernel` (L183) | canonical code-plane embedding rows for T tokens; `__fadd_rn(__fmul_rn(code+bias, scale), offset)` | codes, scales, offsets, tokens → out[T×n] | (⌈n/256⌉, n_tok) × 256 | `_rn` intrinsics | verify L441, mtp L436, prefill L841/L1094 | contract(off) |
| `broadcast_streams_kernel` (L203) | `R[t][c·n+i] = x[t][i]`, c < hc | | (⌈n·hc/256⌉, n_tok) × 256 | | verify L429/L443 | trivial |
| `copy_indexed_kernel` (L210) | `dst = src[*index · stride ...]`; no-op if index < 0 (the PLE rollback) | | min(⌈n/256⌉, 64) × 256 grid-stride | device-resident index | verify L996 | trivial; or do it on the host after the commit buffer completes |
| `fetch_blobs_kernel` (L218) | gather `*n` blobs (uint4) from a pointer table, possibly mapped host pointers, into contiguous VRAM | | 384 × 256 | zero-copy reads over PCIe | verify L768-769 | **drop**: on unified memory experts are addressed in place (pointer table or argument buffer) |
| `rebase_ptrs_kernel` (L228) | `ptr[k] = base + k·bytes` for k < *n | | 1 × 128 | | verify L768-769 | drop with `fetch_blobs` |
| `add_streams_broadcast_kernel` (L233) | `R = h + broadcast(e)` | | (⌈n·hc/256⌉, n_tok) × 256 | | mtp L448, prefill L853 | trivial |
| `ident_hits_kernel` (L241) | identity hit list | | 1 × 1024 | | **none (dead)** | do not port |
| `gather_rows_kernel<E>` (L250) | copy whole rows `src[ids[r]]` with E = uint4/uint32/uint8 chosen by row-size divisibility (Q6_K head row 2100 B uses uint32) | | 384 × 256 grid-stride | template | mtp bind L401 | blit per row or a generic copy kernel; one-time at load |
| `map_ids_kernel` (L259) | `ids[i] = table[ids[i]]` (subset → token id) | | 1 × 64 | | mtp L574 | trivial |
| `row_top_prob_kernel` (L264) | `p = 1/Σ exp(l_i − l_pick)` per row | logits[T×nv], ids → probs | rows × 1024, `part[32]` | `__expf`, xor shuffles | mtp L573 | sum order differs under Metal, giving ULP differences in p. This only affects the T heuristic, not tokens |
| `mtp_select_kernel` (L282) | copy row `*row_dev` of R_src into R_dst; `tok_dst = ids[row]`; volatile writes `out[j]`, `out_p[j]` to mapped host | | 16 × 256 | volatile stores to mapped memory | mtp L640/649/668 | shared buffer; the CPU reads after `waitUntilCompleted` |
| `dense_steps_kernel` (L296) | step records from cells | | 1 × 64 | | **none (dead)** | do not port |
| `window_ids_kernel` (L361) | attention ids `[max(0, n_kv−window), n_kv)` per query; writes `st[3] = width` in place | | (8, n) × 256 | in-block `__syncthreads` only (other blocks do not read st[3]) | mtp L502 | trivial; or compute the window start inside the attention kernel |
| `wait_flag_ge_kernel` (L426) | spin until a mapped host flag ≥ value, then `__threadfence_system` | | 1 × 1 | `__nanosleep` spin, system fence | verify L734, L764, L779 | **not portable**: Metal has no forward-progress guarantee and CPU writes are not observed mid-command-buffer. Replace with `MTLSharedEvent` wait/signal between command buffers |
| `resident_plan_kernel` (L433) | one thread: if all routed ids are VRAM-resident, build the grouped plan (counts/start/dst/tok/ptr) and set `*skip = ring` so later waits and copies are skipped; else `*skip = 0` | | 1 × 1 | single-thread host-loop transcription | verify L687 | with all experts resident in unified memory, always the "resident" case; keep as a tiny kernel or move to the CPU |
| `wait_flag_ge_or_kernel` (L474) | spin unless `*skip == value` | | 1 × 1 | | verify L731, L763, L775 | as `wait_flag_ge` |
| `copy_i32_unless_kernel` (L479) | copy from mapped unless skip | | 1 × 128 | volatile | verify L732 | drop (shared buffers) |
| `copy_or_zero_kernel` (L484) | float4 copy from mapped, or zero if skip | | ≤64 × 256 | volatile float4 | verify L776 | drop or simple copy |
| `gpu_stamp_kernel` (L546) | ns timestamp (`%globaltimer`; HIP `wall_clock64()*10`) | | 1 × 1 | inline PTX | `fused_gr.cu:765/785/852/855`, verify L405 | no in-kernel timer on Metal: use `MTLCounterSampleBuffer` at stage boundaries or command buffer `GPUStartTime`/`GPUEndTime` |

### 5.5 Kernels launched by native_head, native_dense and mtp (outside the four files)

| Kernel / wrapper | Location | Launched by | Metal note |
|---|---|---|---|
| `embed_rows_kernel` | `iq_kernels.cu:1384` (launch L1397), grid (n_embd/256, n_tok) × 32 | `NativeEmbed::gather_dev` (`native_head.cpp:190`); MTP L431 | one simdgroup per 256-value superblock via `dq_dispatch` (see the quant-format part); the table lives in a shared buffer |
| `dequant_flat_kernel<float>` | `iq_kernels.cu:1254` (launch L1404), n/256 × 32 | `NativeEmbed::gather_one` (L194) | same |
| `embedding_gather_kernel` | `elementwise.cu:19-31` (launch L137) | canonical `embed_row` (`layer.cpp:1005`) | contract(off) |
| `native_quantize_q8_1` | `native_mmvq.cu:1152` | head (`native_head.cpp:84`), verify head (L840), MTP (many), PLE native key | see the matvec part |
| `native_mmvq` | `native_mmvq.cu:1475` | the same | 248,320-row GEMV: bandwidth-bound (Q6_K 521 MB per pass; T columns amortize) |
| `native_q5_k_f32` | `native_mmvq.cu:1192` | `NativeHead::run` for type 13 | |
| `quantize_q8_K` + `gemv_quantized` | `quantize_act.cu:281`, `layer.cpp:140` | canonical `lm_head` (`layer.cpp:1081`) | |
| `gr_read` / `fused_gr_read_multi` / `gr_write` | `gr.cu:344`, `fused_gr.cu:803`, `gr.cu:413` | head mix and MTP | see the GR part |
| `native_qsa_rms_norm_weighted`, `native_qsa_gate_apply` | `native_qsa.cu:108`, `:120` | MTP | |
| `native_rope_apply` / `rope_neox_apply` | `native_rope.cu:118`, `rope.cu:117` | MTP | |
| `kv_append_q4_step`, `kv_append_q8_step`, `kv_append_step` | `kv_q4.cu:193`, `kv_q8.cu:103`, `qsa.cu:595` | MTP | |
| `qsa_decode_attn_batch` | `qsa_decode_attn.cu:211` | MTP | |
| `bf16_gemv_fp32_mmvf` | `native_bf16.cu:156` | MTP router, PLE key/value | |
| `native_router_top10` / `router_top10` | `native_router.cu:110` / `router_top10.cu:345` | MTP | |
| `moe_group_resident`, `moe_grouped_s2` | `s2_expert_grouped.cu:1090`, `:1099` | MTP | Q2_0 blob experts |
| `quantize_q8_0_scaled`, `quantize_q8_0` | `quantize_act.cu:245`, `:224` | MTP, PLE canonical key | |
| `shared_expert`, `moe_combine` / `native_moe_combine` | `shared_expert.cu:209`, `:325` / `native_moe.cu:60` | MTP | |
| `f32_to_bf16_bulk` | `elementwise.cu:172` | MTP | |
| `fwht256_inplace_cuda` | definition not located in this slice (used with `kv_rot`) | MTP | |
| `cudaMemcpy2DAsync` (q/gate split) | `mtp.cpp:493` | MTP | strided-copy kernel, or let attention read q with stride 2·HD |

---------------------------------------------------------------------------------------------------------------------

## 6. Per-file summary

| File | `__global__` | FP64 | fast-math | CUDA-specific features | Tests | Metal effort |
|---|---|---|---|---|---|---|
| `ple.cu` | 8 | yes (gnorm, gate, bf16_gemv) | no | double shuffles, `memcpy` bit casts | `ple_parity` (fixtures absent) | low; prefer the native float path |
| `native_ple_postops.cu` | 7 | no | yes | `_rn` intrinsics, 1024-thread blocks | ggml oracle (absent), `ple_parity` native | low; tolerance parity |
| `sampler.cu` | 8 (1 legacy) | yes (tails) | no | smem atomics, `__umulhi`, 32-register lanes, capture query, per-stream scratch, mapped reads | `sampler_parity` ×3 (synthetic, exact) | medium-high: 32 KB smem and FP64 tails. Recommend GPU penalties + top-k with a CPU FP64 tail |
| `verify_kernels.cu` | 23 (2 dead) | no | no | volatile mapped memory, spin waits, `__threadfence_system`, `%globaltimer`, uint4 loads | none | medium: mostly trivial kernels, but the doorbell/skip protocol needs redesigning around MTLSharedEvent |
| `iq_kernels.cu` (embed part) | 2 used here | no | - | `dq_dispatch` | via quant tests | covered by the quant-format part |
| `elementwise.cu` (embed, mapped copies) | 3-4 used here | no | contract off on HIP | volatile mapped reads | - | trivial, or delete |
| `native_mmvq.cu` (head) | wrappers | no | yes | dp4a/Q8_1 | matvec part | see the matvec part |
| `ngram.cpp`, `ple_reader.cpp`, `direct_file.cpp` | host | - | - | O_DIRECT, IOCP, PrefetchVirtualMemory, mlock | `ple_parity` A/B, `ple_reader_test`, `ple_q5_parity`, `ple_fp8_parity` | low: F_NOCACHE plus a pread pool |
| `mtp.cpp` | host | - | - | CUDA graphs per T / step, mapped staging, per-step sync | none direct | medium: re-encode per step; buffers shared |
| `verify.cpp` (this slice's parts) | host | - | - | graph + mapped staging + snapshot/rollback | end-to-end only | medium |
| `suffix_drafter.cpp`, `draft_policy.cpp` | host | - | - | none | `suffix_drafter_test` | none |

---------------------------------------------------------------------------------------------------------------------

## 7. Discrete-VRAM / PCIe assumptions and their unified-memory simplification

| # | Assumption in the CUDA code | Where | Unified-memory (Apple Silicon) treatment |
|---|---|---|---|
| 1 | PLE rows decoded on the CPU into a host buffer, then H2D | `layer.cpp:1270-1279` (pageable `ple_emb_host`, `generate.cpp:2022`); verify mapped `h_ple_` + `copy_from_mapped` (L416) | decode straight into an `MTLStorageModeShared` buffer slot per token or row; no copy kernel. Hashing and decoding 16 rows (1.4 KB) stays on the CPU |
| 2 | Small per-replay inputs (tokens, steps, positions, commit counts, outputs, probabilities, flags) in **mapped pinned memory**, read through volatile copy kernels so a captured graph sees new values | verify L410-416, mtp `capture_round`, `mtp_select`, `coupled_stage` | plain shared buffers written by the CPU before `commit`. Visibility is guaranteed at command-buffer boundaries. No `copy_*_from_mapped` kernels needed |
| 3 | CUDA graphs: fixed pointers, device-resident parameters (`commit_[1]`, `step_rec[0]`, `row_dev`) | verify/MTP | either re-encode each window (hundreds of dispatches, cheap on Apple CPUs) or use indirect command buffers. Device-resident indices can become plain arguments set by the CPU |
| 4 | Token embedding kept in **pinned host-mapped** memory to save 1.27 GB VRAM; VRAM fallbacks for WSL2 and HIP (#325) | `native_head.cpp:108-188` | one shared buffer (or no-copy over an mmap: 16 KiB page-aligned address and length); the fallbacks disappear |
| 5 | Head, `dense.bin`, `experts.bin` uploaded with `cudaMalloc` + `cudaMemcpy` (64 MiB chunks) | `native_head.cpp:46-51`, `mtp.cpp:167-197` | `newBufferWithBytesNoCopy` over mmap'd files (page-aligned offsets required; the GGUF tensor offsets are 32 B aligned, so either copy once into a shared buffer or map the whole file and pass offsets) |
| 6 | MTP K/V as a VRAM ring (window + 4·max_t + 64) over a pinned host copy; resident fallback | `mtp.cpp:206-236` | keep the K/V whole in a shared buffer (32768-cell window × 2 heads × 256 × INT8 is small); streaming unnecessary |
| 7 | Expert streaming over PCIe: `fetch_blobs` from mapped host pointers, `wait_flag_ge*` spin kernels for CPU/DMA progress, `resident_plan` skip protocol | verify L687-779, `verify_kernels.cu:218-231`, `:426-489` | experts addressed in place, so no staging and no doorbells. If CPU expert compute remains (hybrid), synchronize with `MTLSharedEvent` signal/wait **between** command buffers, never with in-kernel spins |
| 8 | Split-sampler scratch allocated per (device, stream) on demand, retiring old buffers | `sampler.cu:789-832` | one preallocated 2 MB shared buffer |
| 9 | Multi-GPU layer split: the drafter lives on the last stage's GPU (`cudaGetDevice` at load, `OnDevice` guards); the Verifier `next_` chain | `mtp.cpp:144`, `verify.cpp:1168-1170` | single device; drop |
| 10 | Host sync per MTP draft step to read `pj` from mapped memory | `mtp.cpp:795-815` | same pattern (commit + wait). Or encode all steps with an on-GPU early-exit flag, then one wait |
| 11 | O_DIRECT reads bypass a page cache contended by a 31.6 GB expert arena on a 63 GB host; SSD keep-alive | `direct_file.cpp:308`, `ngram.cpp:415-416` | `F_NOCACHE`. On unified memory the expert arena and page cache share the same RAM, so the contention argument is stronger. With ≥128 GB, mlock the table (`--ple-io ram`) |
| 12 | `--use_fast_math` on native files to match ggml-cuda; `-ffp-contract=off` for embedding/quantize on HIP | `CMakeLists.txt:288-308` | Metal's default is fast math; choose per pipeline: safe math + contract(off) for `_rn` kernels, fast math acceptable where CUDA used fm, with tolerance tests |
| 13 | FP64 in ple.cu and the sampler tail (FP64 is 1/64 rate on GeForce, which motivated the warp tail) | `sampler.cu:355-357` | no FP64 on Apple GPUs: CPU tail (verify and MTP already sync), float-float on the GPU, or the native float PLE path |

---------------------------------------------------------------------------------------------------------------------

## 8. Oracle vectors and tests: what exists, what is missing, how to reuse

### 8.1 Inventory

| Asset | Location | Content | Needs | Reuse on macOS |
|---|---|---|---|---|
| `ple_oracle_vectors.inc` | `src/kernels/ple_oracle_vectors.inc` (105 lines) | 6 HashCases (token, prev → 16 rows); table constants `kTableRows`, `kTableDataStart 192`, `kTableFileSize 28,800,138,432`; probe rows {0, 12345, 320001535} with head/tail [3][8] values and `kProbeSum {2.0119, 0.99685, 0}`; `kInterleavedSeparation 1.2538`; `dense.bin` offsets (key codes 1,017,723,776 / 6,553,600 B; scales 1,024,277,376 / 819,200 B; value 1,025,219,456 / 26,214,400 B) | hash cases: nothing; probes: the 28.8 GB shard | port the hash test first (pure CPU, exact). Probes validate the IQ4_NL split-halves decode and F_NOCACHE/mmap reads |
| `ple_parity` | `src/kernels/ple_parity.cpp` (1184 lines; CMake L632-640, `WORKING_DIRECTORY` = source) | A: hash (L309). B: table, head/tail exact, `|sum − kProbeSum| ≤ 1e-9` (L503), separation > 0.05 (L526, L542). C: block (L574), skips without capture or GPU (L577, L589); result rel_l1 ≤ 1e-2 (L784-794); native value ≤ 2e-6 with BF16 separation > 1e-5 (L827); native key wiring ≤ 2e-6 (L922); gated-norm separation > 5% (L999-1000, L1035); **host f32 reference** per stage key/value/gate/gated/normalized/conv_out/result ≤ 1e-4 (L1131); history guards L207, L227-229 | `bench/micro/ple_in.bin`, `ple_out.bin` (ggml capture, default `--in` L257) — **absent from the clone**; generator `tests/python/gen_ple_oracle.py` **absent** | lift Part C's host f32 reference into a Metal-vs-CPU test with the same tolerances. The ggml capture must be regenerated from pinned llama.cpp 3cf03257 or obtained from upstream |
| `ple_q5_parity` | `src/kernels/ple_q5_parity.cpp` (CMake L971) | Q5_0 rows vs ggml's dequantizer, max_abs ≤ 1e-6 (L57-58) | Q5_0 table | CPU decode check |
| `ple_fp8_parity` | `src/kernels/ple_fp8_parity.cpp` (CMake L977) | FP8 rows bit-exact through Direct, Mmap and `gather_batch`; correlation ~0.997 vs IQ4_NL | `reference.bin` from torch | exercises the macOS I/O backends |
| `ple_reader_test` | `src/ngram/ple_reader_test.cpp` (CMake L695-698) | reader / cache / dedup on synthetic files | nothing | port with the F_NOCACHE DirectFile |
| `native_dense_ple_key_test` | `tests/core/native_dense_ple_key_test.cpp` (CMake L361-364, skip code 77) | native key loading | model | loader check |
| `hip_ple_iq4` | `tests/hip/ple_iq4.cpp` (CMake L946) | IQ3_XXS/IQ4_XS key 2560→10240 vs reference, rel_l1_q8 ≤ 5e-5, worst ≤ 1e-4 (L178) | synthetic | template for a Metal key-GEMV parity test |
| `sampler_parity` | `src/kernels/sampler_parity.cpp` (1284 lines; CTest variants split / one_block / old at CMake L573-582) | 18 fixtures with **exact** expected tokens: 1 greedy (L457), 2 order (L468), 3 greedy uses no RNG (L506), 4 penalties (L539), 5 temp 0 (L615), 6 sampled penalties (L646), 7 min_p (L681), 8 window tail (L721), 9 distribution share within 0.014 (L755, L775), 10 top_p before min_p (L782), 11 stale history (L806), 12 one history per row (L832), 13 out-of-vocab ids (L896), 14 top_k contract (L940), 15 `penalty_rows` (L962), 16 list under ties (L1028), 17 sampled ties (L1122), 18 fallbacks (L1200); `host_philox_uniform` L134-159 | nothing (synthetic) | **port first**: it pins the Philox variant, the tie rules, the chain order and the FP64 tail. Run it against both the Metal GPU path and the CPU tail |
| `coupled_draft_test` | `tests/core/coupled_draft_test.cpp` (CMake L769-771) | counter and penalty-window arithmetic | nothing | pure host |
| `suffix_drafter_test` | `src/spec/suffix_drafter_test.cpp` (CMake L753) | suffix drafter | nothing | pure host, unchanged |
| ggml-oracle parity suite (includes `native_ple_postops_parity`) | `bench/micro` (CMake L311-334) — **absent**; `tests/CMakeLists.txt` absent (CMake L52-54) | native kernels vs a pinned llama.cpp build (`STRATA_ORACLE_SOURCE_DIR` / `BUILD_DIR`) | llama.cpp 3cf03257 build | on macOS, build the same llama.cpp with its Metal backend and compare Strata-Metal against ggml-Metal kernels |
| verify kernels | - | **no tests** | - | write new CPU-reference tests: `gdn_step_norm_multi` (verify vs replay with `n_keep`), `copy_indexed` rollback, `window_ids`, `mtp_select` |
| end-to-end dumps | `STRATA_DUMP_FIRST_LOGITS` (`generate.cpp`, L6183, first-window logits), `STRATA_DBG_NAN` (`verify.cpp:1183`) | logits files | model | compare CUDA and Metal first-window logits (rel L1 / top-k overlap); compare greedy token streams under a fixed prompt |

### 8.2 Recommended parity ladder for the Metal port

1. Pure-CPU, exact:
   - the PLE hash (oracle HashCases)
   - the IQ4_NL / FP8 / Q5_0 decode (probes)
   - `host_philox_uniform` against the Metal Philox kernel over many (seed, counter) pairs
   - `coupled_draft_test`, `suffix_drafter_test`
2. Sampler: all 18 `sampler_parity` fixtures, which should be exact. A float-float or CPU-double tail is needed for
   fixtures 6-12 and 17.
3. PLE block:
   - Part C host reference: ≤ 1e-4 per stage, result rel_l1 ≤ 1e-2.
   - Separately, the native postops path against the ggml-Metal oracle (ULP-level tolerance).
   - The history advance and the verify snapshot/rollback (`copy_indexed` with n_keep-1) checked exactly.
4. Head: Q8_1 + MMVQ over the full vocabulary against CPU dequant-dot. Greedy argmax exactness depends on near-ties.
   Compare the top-5 sets and logit rel-L1, not just the argmax.
5. MTP: replay a recorded window (R rows, outv, p, a) on the CPU and on Metal. Compare draft ids and `row_top_prob`
   (tolerance), and the K/V cells written.
6. End-to-end: greedy token streams and first-window logits against the CUDA build for the same prompt and seed.
   Also check the acceptance rate under `--spec 4 --spec-min-p 0.5`.

### 8.3 Porting conclusions specific to this slice

- **PLE:**
  - The 28.8 GB table never touches the GPU; keep hash, read and decode on the CPU and write into a shared buffer.
  - The macOS I/O backend needs `F_NOCACHE` in place of `O_DIRECT`.
  - The GPU block is about 10 small kernels, dominated by the key GEMV (2560×10240) and the value GEMV
    (2560×2560 BF16).
  - Use the native float post-ops; the legacy `ple.cu` path depends on FP64.
- **Head:** a bandwidth-bound 248,320-row GEMV (Q6_K 521 MB). On unified memory it is simply resident. The draft
  vocabulary subset (`dhead_`) remains a large saving for the MTP.
- **Sampler:**
  - Implement penalties + top-k selection on the GPU (bitmap and split kernels) and the ≤64-entry FP64 tail on the
    CPU. Both the verify and the MTP paths already synchronize where the tail is needed, so this costs no extra
    sync.
  - Keep the exact Philox variant and the lowest-index tie rule.
  - Watch the 32 KB threadgroup limit (`split_merge` and `coupled_merge` use 33.8 KB).
- **Verify and MTP control flow:**
  - Delete the mapped-memory copy kernels, `fetch_blobs`/`rebase_ptrs` and the doorbell spin kernels.
  - Device-resident indices (`commit_+1`, `step_rec`, `row_dev`) can stay as small buffers written by the CPU.
  - Replace graph capture with per-window encoding.
  - `ident_hits` and `dense_steps` are dead code and need no port.
