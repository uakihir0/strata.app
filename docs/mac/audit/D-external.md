> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# D. External landscape: porting Strata (Qwen3.8-Flash-Next engine) to macOS / Apple Silicon / Metal

Research date: 2026-10-02. Sources: GitHub (via authenticated `gh` / GraphQL), Hugging Face API and model cards, apple.com / support.apple.com / developer.apple.com (raw HTML fetched with curl), local probes on this machine.

Legend:
- **Verified** = read directly at the given URL (or measured locally, command noted) on 2026-10-02.
- **Unverified / inferred** = my own calculation, extrapolation, prior knowledge not re-checked, or a claim I could not fetch. Always labelled.
- GB = 10^9 bytes, GiB = 2^30 bytes; units are kept as each source states them.

---

## 0. Executive summary

1. **Qwen3.8-Flash-Next already runs on Metal in at least six engines.** llama.cpp (official, merged 2026-08-27), mlx-vlm (official since v0.6.17), DwarfStar/antirez-ds4 (Metal is its primary target), MTPLX, oMLX, LM Studio (llama.cpp GGUF) and Ollama (MLX NVFP4). A Strata Metal port would be entering a crowded field, not an empty one.
2. **Strata itself has no Metal plans.** The owner closed issue #193 with "Not planned for now ... a Metal backend would be a new port". Of 444 forks scanned, one is Mac-related, and it is a UI/Python shim. It does not port the engine.
3. **Best measured Mac numbers so far:**
   - M3 Ultra, DwarfStar Q2: about 1,330 tok/s prefill and about 50 tok/s decode (up to 90 with MTP).
   - M5 Max 128 GB, MTPLX 4-bit: 1,453 tok/s prefill at 4K, 43.8 plain decode, 73.5 with MTP.
   - M1 Ultra, llama.cpp IQ4_NL: 548 pp512 and 37.7 tg128.
   - For comparison, Strata on an RTX 5070 12 GB does 93 tok/s decode and 2,170 tok/s prefill at 32K (Q2_0).
4. **What is still unsolved on Mac is the 32 to 48 GB tier.** That means streaming routed experts from SSD, which is exactly Strata's design idea (hot-expert cache plus cold misses). Only these projects do it:
   - DwarfStar PR #1056 (open): runs Q2 on a 32 GiB M1 Max at about 7.9 tok/s with MTP.
   - oMLX expert offload: it cannot be combined with its Lightning MTP.
5. **Moving the PLE / n-gram table (28.8 GB at IQ4_NL) off-GPU is common practice.** DwarfStar, MTPLX, oMLX, mlx-vlm's `ple_storage.py` and llama.cpp's lazy-read mode all do it. In llama.cpp, macOS page-cache-backed reads were "already smart enough" (tarruda). One open llama.cpp bug still matters: the per-file mmap counts the CPU-resident PLE against the GPU working set (#29465).
6. **Memory budget on this test machine** (M5 Pro, 64 GB, macOS 27.0):
   - Metal reports `recommendedMaxWorkingSetSize` = 55.66 GB (81%).
   - `maxBufferLength` = 41.75 GB.
   - `iogpu.wired_limit_mb` = 0, meaning the OS default.
   - What fits on 64 GB: ISTA Q2_0 / IQ2_XS / IQ3_XXS shard 1 (37.6 / 39.2 / 47.0 GB) and DwarfStar Q2 (41.73 GiB resident).
   - What does not fit: 4-bit builds (about 70 GiB and up).
7. **Hardware (Apple specs pages):**
   - M5 Pro: 307 GB/s, up to 64 GB.
   - M5 Max: 460 GB/s (32-core GPU, 36 GB) or 614 GB/s (40-core GPU, 48/64/128 GB).
   - M5 Ultra: 1.2 TB/s, 96/256/512 GB.
   - M6: up to 170 GB/s, up to 32 GB.
   - M5-generation GPUs have a Neural Accelerator in every GPU core. Apple says M5 matmul is 4x faster than M4, and the gain translates almost 1:1 to prompt processing.
   - MacBook Pro SSD: up to 14.5 GB/s.
8. **Metal 4 on macOS 27 adds quantized MTLTensor formats.** These are 4/8-bit FP and 2-bit int (4/8-bit int existed since macOS 26), plus FP8 E8M0 block-scale planes, which suit MXFP4.
   - Tensor ops use the Neural Accelerators automatically.
   - llama.cpp enables its tensor-API path by default only on M5, M6, A19 and A20 chips.
   - GSQ/IQ codebook formats (IQ2_XS, IQ3_*) have no native tensor format, so they would need custom dequant kernels.

---

## 1. The model and its artifacts

### 1.1 Verified (with URL)

**Model card: https://huggingface.co/Qwen/Qwen3.8-Flash-Next**

- **Size:** "125B with 6B activated, plus 51B n-gram embedding and 4B MTP".
- **Dimensions:** hidden 2560, 48 layers, vocab 248320 (padded).
- **Layer layout:** "12 x (3 x (Gated DeltaNet -> MoE) -> 1 x (Qwen Sparse Attention -> MoE))".
- **Gated DeltaNet:** 48 V heads and 16 QK heads, head dim 128.
- **QSA:** 24 Q heads and 2 KV heads, head dim 256, RoPE dim 64.
  - Indexer: MQA with 4 query heads and 1 shared key head, dim 128.
  - Budget: "512 blocks or 2048 tokens".
- **MoE:** 512 experts, "10 Routed + 1 Shared", expert intermediate dim 640.
- **Gated Residual:** 4 branches, bottleneck rank 320.
- **N-gram embedding:** 20,000,000 entries (bigrams/trigrams at layer 2).
- **MTP:** 1 layer.
- **Context:** 262,144 native, extensible to 1,000,000.
- **License:** `qwen-community-1.0`.
- **Recommended serving engines:** vLLM, SGLang, KTransformers, TokenSpeed. The card does not mention Apple, MLX or llama.cpp.
- **Config:** `qwen4_exp` / `Qwen4ExpForConditionalGeneration`, `ngram_vocab_size_base` 20M, `hc_count` 4, `indexer_budget` 2048. Source: config.json in the same repo.
- **BF16 weights:** about 360 GB across 131 shards.

**ISTA-DASLab GGUF: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF**

- License apache-2.0, 952,084 downloads (HF API, 2026-10-02).
- **Shard 2 is the same in every variant:** the 28.8 GB `per_layer_token_embd` n-gram table (51.2B params) at IQ4_NL, which the card calls "a lookup, not a matmul weight".

| Quant | bpw | Shard 1 (must be resident) | Shard 2 (n-gram, can stay on disk) | Total |
| --- | ---: | ---: | ---: | ---: |
| Q2_0 | 2.40 | 37.62 GB | 28.80 GB | 66.4 GB |
| IQ2_XS | 2.50 | 39.23 GB | 28.80 GB | 68.0 GB |
| IQ3_XXS | 3.00 | 47.04 GB | 28.80 GB | 75.8 GB |
| IQ3_S | 3.50 | 54.82 GB | 28.80 GB | 83.6 GB |

- **Card guidance:** "Run with `-lm mmap --lazy-mode on` ... plan for shard 1 in VRAM, and keep shard 2 on an SSD".
- **Card's llama.cpp measurement** (hardware not stated):
  - Q2_0: 367.49 tok/s prompt, 93.79 tok/s decode.
  - IQ2_XS: 108.19 tok/s prompt, 70.30 tok/s decode.
  - Explanation given: lookup-table formats "cost real time, and on this model that cost dominates inference".

**ISTA Coder GGUF: https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF**

- apache-2.0, 148,142 downloads.
- IQ1_M ships as 29.61 GB + 28.80 GB shards (58.4 GB total), plus a 0.91 GB mmproj.

**Swift 1.5 GGUF: https://huggingface.co/ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF**

- License "other", base `ukisai/Swift-Qwen3.8-Flash-Next`, 96,094 downloads.

| Quant | Shard 1 | Shard 2 |
| --- | ---: | ---: |
| IQ2_XS | 39.79 GB | 28.36 GB |
| IQ3_XXS | 39.79 GB | 36.18 GB |
| Q2_0 | 39.80 GB | 26.75 GB |

- The repo also includes an imatrix (0.58 GB) and an mmproj (0.91 GB).
- The base repo `ukisai/Swift-Qwen3.8-Flash-Next` could not be read through the HF API (non-JSON response, probably gated).

**Strata: https://github.com/Niko1221/Strata**

- Created 2026-09-24, MIT, about 5,019 stars and 449 forks.
- README speeds (RTX 5070 12 GB + 64 GB RAM, decode / PP at 32K, tok/s):

| Quant | Decode | PP@32K |
| --- | ---: | ---: |
| Q2_0 | 93 | 2,170 |
| IQ2_XS | 79 | 2,090 |
| IQ3_XXS | 62 | 1,750 |
| IQ3_S | 53 | 1,620 |
| Coder | 55 | 2,180 |

- RX 9070 XT, Q2_0: 60 decode / 1,160 PP.
- Architecture: hot experts on the GPU, all experts in RAM with the CPU computing misses, n-gram table on SSD, and MTP for 1.6 to 1.8x.
- The ISTA card also points to Strata, saying it runs "on a single NVIDIA GPU from 8 GB of VRAM".
- **Mac status:** issue #193 was closed by the owner: "Not planned for now ... a Metal backend would be a new port". https://github.com/Niko1221/Strata/issues/193#issuecomment-5908103145

### 1.2 Unverified / inferred

- **Per-expert size.** Each expert is gate + up + down = 3 x 2560 x 640 = 4.92M params. Across 512 experts x 48 layers that is about 120.8B params, which matches the 125B total, so experts dominate the weights.
- **Routed bytes per token:** 10 x 48 x 4.92M = 2.36B params.
  - At about 2.5 bpw: roughly 0.74 GB per token.
  - At 4.5 bpw: roughly 1.33 GB per token.
  - Total active weight bytes per token come to roughly 1.5 to 2 GB at 2 to 2.5 bpw, and 3 to 3.5 GB at 4 bpw. This is my arithmetic, not a published figure.
- **Swift 1.5 sharding.** Shard 1 is a constant 39.8 GB, so Swift appears to split at a fixed maximum size rather than isolating the PLE table. On Swift, "shard 2 = PLE" would then not hold; this is inferred from the sizes. In the Coder repo, shard 2 is 28.80 GB, the same as ISTA's PLE shard, so it is probably the same table (inferred).
- **Q2_0 vs IQ formats on Metal.** ISTA's explanation (codebook lookup cost dominates) should apply even more strongly on Metal: IQ2_XS / IQ3_* need grid lookups in threadgroup or constant memory, and there is no native MTLTensor format for them. Inferred.

---

## 2. llama.cpp

### 2.1 Verified (with URL)

**Core support**

- **PR #27742** "model: add Qwen3.8-Flash-Next (qwen4exp)": merged 2026-08-27T19:32Z. https://github.com/ggml-org/llama.cpp/pull/27742
  - The first build containing it is b10660.
  - The latest prerelease build today is b11342 (2026-10-02). https://github.com/ggml-org/llama.cpp/releases
- **Stable release v0.5.0** (2026-09-23) lists "Add Qwen4Exp hyper-connection ops and sparse flash attention (#28901, #28770)" and "Add Metal MoE and SSM_CONV fusion optimizations (#28948)". https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0
- **Lazy loading:** PR #27794 "model_loader: add TENSOR_READ_LAZY" (merged 2026-08-27) and PR #27837 "improve TENSOR_READ_LAZY handling" (merged 2026-08-30). These are what make `-lm mmap --lazy-mode on` work.
  - https://github.com/ggml-org/llama.cpp/pull/27794
  - https://github.com/ggml-org/llama.cpp/pull/27837
- **Metal op coverage:** the Metal backend's `supports_op` covers every op the model needs, including LIGHTNING_INDEXER, the DSV4_HC_* hyper-connection ops, GATED_DELTA_NET and sparse FA. Checked in the local clone at commit fb4b2737 (2026-10-02), ggml/src/ggml-metal/.

**Open or recent Metal issues for this model**

- **#29465 (open):** "Metal: per-file mmap mapping spans CPU-resident tensors, so a 27 GiB lazy-read table between GPU tensors counts against the GPU working set (Qwen3.8-Flash-Next OOM)". Reported on an M4 Max 128 GB with a default working set of about 96 GiB. https://github.com/ggml-org/llama.cpp/issues/29465
- **#28805 (open):** on Metal, "decode emits 1 token then EOS at long context ... threshold moves with model quant / KV quant / n_ctx". Seen on 2x M1 Max. https://github.com/ggml-org/llama.cpp/issues/28805
- **#29335 (closed):** "Metal decode throughput collapses when several long sequences are batched (M3 Ultra; 2 separate -np 1 processes are 4.6x faster)". Reported M3 Ultra Q8_0 TG was 29.6 to 32.8 tok/s. https://github.com/ggml-org/llama.cpp/issues/29335
- **#27473 (open):** "Metal tensor API silently inert on M5+/A19+ — MTLCompileOptions.languageVersion never set". https://github.com/ggml-org/llama.cpp/issues/27473
  - Current master code (local clone fb4b2737, ggml-metal-device.m lines 201-208) now sets `languageVersion = 4.0` whenever `has_tensor` is true. The issue is still open.

**How the Metal backend handles memory** (local clone fb4b2737, ggml/src/ggml-metal/)

- **Tensor-API gating** (ggml-metal-device.m around lines 1145-1162):
  - `has_tensor = supportsFamily(MTLGPUFamilyMetal4)`.
  - It is then disabled unless the device name contains M5, M6, A19 or A20, or `GGML_METAL_TENSOR_ENABLE` is set.
  - Code comment: "M2 Ultra: ~5% slower; M4, M4 Max: no significant difference".
  - A test kernel using `mpp::tensor_ops::matmul2d` is compiled first.
- **Memory reporting:**
  - `max_working_set_size = recommendedMaxWorkingSetSize` (around line 1290).
  - `ggml_metal_device_get_memory` reports total = recommendedMaxWorkingSetSize and free = total minus currentAllocatedSize (around line 1503).
  - llama.cpp does not touch `iogpu.wired_limit_mb`.
- **Wrapping host memory** (around lines 2190-2260):
  - The pointer is page-aligned.
  - If size <= `maxBufferLength`: a single `newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil`.
  - Otherwise: overlapping views of `max_buffer_size`, with step = max_buffer_size minus overlap, where overlap = largest tensor rounded up plus one page.
  - `buffer_from_host_ptr = true` (ggml-metal.cpp line 698), so mmapped GGUF files are wrapped with no copy.
- **Residency sets** (around lines 19-24 and 931-1001):
  - Created only on macOS >= 15.
  - A background heartbeat keeps them wired. Keep-alive defaults to 180 s and can be changed with `GGML_METAL_RESIDENCY_KEEP_ALIVE_S`.

**Mac speeds for this model**

- **Discussion #27965** (tarruda, M1 Ultra, branch `qwen4exp-metal-optimizations`, IQ4_NL 95.20 GiB / 176.94B params). https://github.com/ggml-org/llama.cpp/discussions/27965

| Depth | pp512 (tok/s) | tg128 (tok/s) |
| ---: | ---: | ---: |
| 0 | 548.09 | 37.70 |
| 10K | 453.29 | 34.14 |
| 20K | 427.72 | 33.87 |
| 30K | 407.90 | 33.32 |
| 50K | 376.66 | 32.44 |
| 80K | 335.10 | 31.19 |

- **Discussion #27864**, "Qwen 4 PLE SSD offloading" (tarruda, created 2026-08-28; the hardware is presumably the same M1 Ultra, see 2.2). Branch `metal-qwen4exp-split-ngram`, using `--model-ngram` with `--ngram-load-mode resident|read`; weights are 68.37 GiB. https://github.com/ggml-org/llama.cpp/discussions/27864

| Depth | resident pp512 | resident tg128 | read-from-SSD pp512 | read-from-SSD tg128 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 554.57 | 39.56 | 272.77 | 36.08 |
| 100K | 319.90 | 26.80 | 205.40 | 29.28 |
| 250K | 212.82 | 24.15 | 198.80 | 26.40 |

  - In short: reading the PLE from SSD roughly halves short-context prefill, costs about 9% of decode, and has little effect at long context.
  - tarruda's comment: a smarter cache gave no gain over "naive fseek/fread ... filesystem paging is already smart enough". He also pointed users to the published split GGUF at https://huggingface.co/tarruda/Qwen3.8-Flash-Next-GGUF.
  - Later commenters say `--tensor-read-lazy` (PR #27794) makes the manual split unnecessary.

### 2.2 Unverified / inferred

- Hardware for #27864 is not restated in the body. Since it is the same author, the same week and similar numbers to #27965, I assume an M1 Ultra.
- **Working-set accounting.** The `newBufferWithBytesNoCopy` approach means every byte of a mapped GGUF range counts toward the Metal working set, including the 28.8 GB PLE when it sits in the same file range. That is the root cause of #29465. A Strata port should keep PLE and GPU weights in **separate files or separate mappings** from day one.
- **`maxBufferLength` is about 41.7 GB on a 64 GB M5 Pro** (see section 6). ISTA's IQ3_XXS / IQ3_S shard 1 (47 / 55 GB) cannot be wrapped as a single MTLBuffer. Any engine needs either the multi-view scheme or per-tensor or per-layer buffers.

---

## 3. MLX ecosystem, LM Studio, Ollama

### 3.1 Verified (with URL)

**mlx-vlm**

- Official support since **v0.6.17** (2026-08-26) via PR #2032 by lucasnewman. https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.17
- Model code lives in https://github.com/Blaizzy/mlx-vlm/tree/main/mlx_vlm/models/qwen4_exp: config.py, fp8.py, language.py, `ple_storage.py`, `qsa_kernel.py`, vision.py.
- `ple_storage.py` docstring: "Loading those shards as ordinary MLX parameters wires roughly 30 GiB of cold lookup data. This module instead maps the existing safetensors byte ranges and materializes only the rows selected for the current token."
- Follow-ups: MTP (#2040), continuous batching, FP8 (#2058).
- Latest release v0.7.4 (2026-09-28). License MIT.

**mlx-lm**

- **PR #1788** "Add Qwen3.8-Flash-Next (qwen4_exp) model support" by eauchs: OPEN since 2026-08-26, last updated 2026-09-30, with one approval. https://github.com/ml-explore/mlx-lm/pull/1788
- Numbers in the PR thread:
  - 4-bit: 24.0 tok/s generation at 111 GB peak, Mac not specified (Vontra).
  - 8-bit on an M3 Ultra 512 GB: 17 to 21 tok/s over 600K tokens (Sofille65).
- Load failure with `mlx-community/Qwen3.8-Flash-Next-4bit` because of PLE shard naming (Teejer).
- The thread also notes that the seed default must be 1234.
- Latest mlx-lm release is v0.31.3 (2026-04-22), so this model is **not** in a released mlx-lm. https://github.com/ml-explore/mlx-lm/releases/tag/v0.31.3
- MLX core is at v0.32.3 (2026-09-29). https://github.com/ml-explore/mlx/releases/tag/v0.32.3

**MLX wired-memory handling**

- `mx.set_wired_limit` docstring: "only useful on macOS 15.0 or higher ... Setting a wired limit larger than system wired limit is an error ... sudo sysctl iogpu.wired_limit_mb=<size>". The default limit is 0. https://github.com/ml-explore/mlx/blob/main/python/src/memory.cpp
- mlx-lm `generate.py` `wired_limit()` raises the wired limit to `max_recommended_working_set_size`. It warns when model bytes exceed 0.9 x that value. https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py

**Hugging Face MLX and GGUF repos** (sizes from the HF API)

| Repo | Size | Downloads / notes |
| --- | ---: | --- |
| mlx-community/Qwen3.8-Flash-Next-4bit | 111.55 GB | 9,678 downloads |
| mlx-community/Qwen3.8-Flash-Next-4bit-mtp | 113.27 GB | |
| garnermccloud/Qwen3.8-Flash-Next-MLX-SSD-Stream | 132 GB | for mlx-serve |
| antirez/qwen3.8-flash-next-gguf | 324 GB | 130,781 downloads |
| Baekpica/Qwen3.8-Flash-Next-Mixed-Quant-SSD-PLE-GGUF | 334 GB | |
| ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 | 106.9 GB | see section 4 |
| Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed | 115 GB | 20,521 downloads |

**LM Studio**

- https://lmstudio.ai/models/qwen/qwen3.8-flash-next lists "Minimum system memory 96 GB".
- It uses lmstudio-community GGUFs (Q4_K_M about 119 GB in 3 shards, Q6_K, Q8_0) built with llama.cpp b10662.
- LM Studio's own **Splash** engine for Mac ships in Bionic 1.1.5 (Sep 19, 2026): "[Mac only] Ultra fast Qwen3.8 inference using a new engine: Splash". Splash requires M3 or later and macOS 26.4+.
  - https://lmstudio.ai/changelog
  - https://lmstudio.ai/blog/splash-engine
- Splash does **not** cover Flash-Next yet (see section 4).

**Ollama** (https://ollama.com/library/qwen3.8-flash-next/tags), 172.9K pulls:

| Tag | Format | Size |
| --- | --- | ---: |
| 125b-mlx / 125b-a6b-nvfp4 | MLX | 105 GB |
| q4_K_M | GGUF | 120 GB |
| q8_0 | GGUF | 189 GB |
| bf16 | GGUF | 355 GB |
| mlx-bf16 | MLX | 360 GB |

### 3.2 Unverified / inferred

- The mlx-community 4-bit repo is 111.55 GB, of which about 30 GiB is PLE. With mlx-vlm `ple_storage` the wired resident set would be about 80 GB, so 96 GB Macs are the practical floor for MLX 4-bit. This is consistent with LM Studio's "96 GB minimum". Inferred.
- None of the MLX-based Flash-Next paths has a sub-4-bit (Q2-class) mainstream build. GGUF Q2 / IQ2 builds (llama.cpp, DwarfStar) are therefore the only verified route onto 64 GB Macs, apart from oMLX and DwarfStar expert offload. Inferred from the repo list above; not exhaustive.

---

## 4. Other Metal engines, expert/SSD offload projects, Strata forks

### 4.1 Verified (with URL)

**DwarfStar (antirez/ds4)**

https://github.com/antirez/ds4 (MIT, about 22,850 stars)

- **Scope:** "Metal, the primary target, on Macs with 96 GB or more". It supports "Qwen3.8 Flash Next (Metal and CUDA)".
- **Model doc:** https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md
  - `qwen38-q2`: 137.10 GiB file, **41.73 GiB resident**. Expert formats: IQ2_XXS for gate/up, and Q2_K for down after padding the 640 columns to 768.
  - `qwen38-q4k`: 165.11 GiB file, **69.74 GiB resident**.
  - The 95.37 GiB BF16 n-gram table stays on disk; the engine "reads selected rows directly from the GGUF, without mapping or preloading".
  - For 64 GB machines it says to "start Q2 ... on a 64 GB machine" with `--ctx 8192 --prefill-chunk 1024`.
  - It also says "Tensor parallelism, pipeline execution and SSD expert streaming are not implemented for this model yet".
  - Supported expert formats: Q8_0, MXFP4, Q4_0, Q4_K, Q2_K, IQ2_XXS.
- **M3 Ultra 512 GB, Q2 IQ2XXS** (speed-bench/qwen38-checkpoints/results.csv in the repo):
  - Prefill 1,331 tok/s at ctx 8192 and 1,318 at 32K.
  - Decode 50.4 to 50.1 tok/s.
  - Generation about 54.7 tok/s without MTP, and 64.8 to 90.5 with MTP.
- **DGX Spark, Q2:** 745 PP / 22.6 TG.
- **docs/PERFORMANCE.md, DeepSeek V4 Flash Q2 on an M5 Max 128 GB** (a reference for M5 Max scaling):

| Context | PP (tok/s) | TG (tok/s) |
| --- | ---: | ---: |
| 2K | 790 | 39.35 |
| 16K | 572.5 | 36.1 |
| 32K | 557 | 34.4 |
| 64K | 398.5 | 27.6 |

- **docs/SSD_STREAMING.md:** "Metal supports streaming for DeepSeek and GLM". On an M5 Max 128 GB:
  - GLM 5.3 Flash Q4_K (177.77 GiB): 121 PP, 11.9 / 14.9 TG.
  - DeepSeek Vision MXFP4 (145 GiB): 300 PP, 11.9 / 19.3 TG.
- **docs/METAL.md:** memory tiers for 64 GB (Flash Q2 with `--ssd-streaming`), 96, 128, 256 and 512 GB.
- **PR #1047** (closed, superseded by **#1056**, open) "Metal: add SSD expert streaming for Qwen3.8 Flash Next". https://github.com/antirez/ds4/pull/1047 and https://github.com/antirez/ds4/pull/1056
  - Runs Q2 on a **32 GiB M1 Max** with a pread expert cache extended to 512 expert IDs and top-10 routing.
  - MTP generation went from 4.55 to **7.86 tok/s**; pread bytes went from 48.40 to 12.02 GiB.
  - PR note: "Expert reads currently finish before routed computation; I/O overlap ... future work". #1056 adds overlap and kernel work, "Up to 40%".
- **Related:** issues #867 and #870 (Baekpica CUDA fork work). maceip's MTPLX PR #380 (closed) reported about 30 tok/s at 1M context. https://github.com/youssofal/MTPLX/pull/380

**ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4** (DwarfStar format)

https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4

- Q4_K imatrix for gate/up, MXFP4 for down, Q8_0 for dense layers.
- Main file 69.7 GiB including MTP. PLE sidecar is Q4_1 at 29.8 GiB, "CPU-only and never enters Metal residency".
- M3 Ultra:

| Context | Prefill (tok/s) | Decode (tok/s) | Decode with MTP |
| --- | ---: | ---: | ---: |
| 4-16K | 1,071-1,171 | 45.4-45.6 | 55-56 |
| 64K | 1,158 | 44.8 | 51.7 |
| 128K | 1,136 | 44.1 | 53.9 |
| 262K | 1,063 | 42.2 | 45.2 |

- Recommends a 128 GB+ Mac.

**MTPLX**

https://github.com/youssofal/MTPLX (Apache-2.0, about 2,496 stars); model card https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed

- **M5 Max 128 GB, MTPLX 2.12.0:**
  - PP 1,453 tok/s on a 4,061-token prompt.
  - PP 1,094 tok/s on 65,502 tokens (1,548 with newer kernels).
  - Decode 63.5 tok/s after a 65K prompt.
- **MTPLX 2.11.3:**
  - OpenCode: 125.8 tok/s with MTP depth 3 and a mostly cached prompt.
  - 9K code prompt: 79.3 tok/s. 109K: 61.8. 200K: 50.3.
  - Plain decode 43.8 vs 73.5 with MTP (1.7x).
- **Footprint:** 4-bit g64 with 8-bit QSA. The n-gram table sidecar (32 GB) streams from SSD. Peak 87 GiB (Bare Speed variant 78 GiB).
- Recommends 128 GB+; the Bare Speed variant is for 96 GB.

**oMLX**

https://github.com/jundot/omlx (Apache-2.0, about 22,445 stars)

- **v0.7.0 (2026-09-30):** https://github.com/jundot/omlx/releases/tag/v0.7.0
  - Faster Flash-Next prefill through "QSA attention on the M5 tensor units" and NAX tiles.
  - Fused decode kernels.
  - "Lightning MTP with expert offload for Qwen3.8-Flash-Next" (#3935).
  - Expert-offload I/O overlap (#4037).
  - "PLE forced to SSD" (#4099).
- **Issue #3614:** on 64 GB M1 / M2 Max the "wired cap ~57.5GB". https://github.com/jundot/omlx/issues/3614
  - Jundot's oQ4e-mtp build is 106.3 GB total, with about 74.3 GB non-PLE and about 32 GB PLE.
  - Expert offload is mutually exclusive with Lightning MTP.
- **Issue #3911**, expert offload on other MoEs: https://github.com/jundot/omlx/issues/3911
  - Qwen3-235B-A22B-4bit on an M5 Max 128 GB with 0.45 of experts resident: 11.7 tok/s.
  - Qwen3-30B-A3B-4bit on an M1 Pro 32 GB: 14.5 to 17.5 tok/s.

**Splash (Inco AI)**

https://github.com/incoai/splash (Apache-2.0, created 2026-09-18)

- Metal-native. Supports only Qwen3.8-27B and Qwen3.6-35B-A3B. Issue #39 "Model request: Qwen3.8 Flash Next" is open.
- **M5 Pro 48 GB:** 35B-A3B decodes at 210 tok/s; prefill at 32K is 2,011 tok/s.
- **Against llama.cpp on M5 Pro (decode, tok/s):** 35B-A3B goes from 69 to 175, and 27B from 16 to 74.
- **Issue #220:** https://github.com/incoai/splash/issues/220
  - "sparse mapping/unmapping exceeded 30000 ms", followed by WindowServer watchdog logouts.
  - Seen on an M5 Max 128 GB with macOS 26.4.1 to 27.0.1, plus a report from an M5 Ultra 96 GB on macOS 27.0.

**ninfer and HyperQwen**

- ninfer: https://github.com/Neroued/ninfer. HyperQwen: https://github.com/syv-ai/HyperQwen.
- Both are CUDA single-GPU or vLLM projects targeting Qwen3.8-27B. Neither describes a Metal path (from their repo descriptions and READMEs).

**LLM in a Flash** (Apple)

- arXiv 2312.11514, "LLM in a flash: Efficient Large Language Model Inference with Limited Memory", submitted 2023-12-12. https://arxiv.org/abs/2312.11514
- It is the conceptual basis for flash/SSD weight streaming: windowing and row-column bundling.

**llama.cpp ExpertCache**

- Discussion #26938, ExpertCache (https://github.com/amos-labs/expertcache): https://github.com/ggml-org/llama.cpp/discussions/26938
  - gpt-oss-120B MXFP4 on a 64 GB M1 Max: prefill 5.75 to 9.80 tok/s, decode about 3 tok/s.
  - 16 GB M1 Pro: decode 0.72 tok/s.
  - Approaches that failed: madvise, copied LRU slots, static prewarm.

**Strata forks**

- All 444 forks were scanned through GraphQL refs (`gh api repos/Niko1221/Strata/forks` plus branch names). Only one is Mac-related: **jinzy0623/Strata-macOS** (branch `macos-port`, created 2026-09-30). https://github.com/jinzy0623/Strata-macOS
  - It is a UI / Python port that uses llama-cpp-python Metal with small Qwen2.5 GGUFs.
  - Its README says SSD expert streaming, the expert cache and MTP are "尚未接入 Metal" (not yet wired to Metal).
  - It also says a 16 GB M4 Air cannot run the 125B model.
- GitHub searches for "strata metal" and "strata apple silicon" returned nothing relevant. The only Strata issue that mentions Mac is #193.

### 4.2 Unverified / inferred

- **DwarfStar is the closest analogue to a Strata Metal port.** It is a single-model-family C/Metal engine with the same feature set: PLE on disk, MTP, Q2-class expert formats, and SSD expert streaming in progress.
  - Its MIT license would allow reusing or porting its Metal kernels (GDN, QSA indexer, HC, IQ2_XXS / Q2_K matvec) into Strata, with attribution.
  - Not checked: kernel-level compatibility with Strata's GSQ/RCO GGUF layouts. DwarfStar uses its own requantized GGUFs (for example, down padded to 768), not ISTA's files.
- **Strata's distinctive value on Mac** would be the "hot experts resident, cold experts fetched" cache, adapted from VRAM/RAM to wired-memory/SSD for 32 to 48 GB Macs. On unified memory the "CPU computes misses" half of Strata's design has no purpose when the expert is already in RAM, because the GPU reads the same memory. It matters only when misses come from SSD. Inferred.
- The SSD-streaming ceiling at Q2 (my arithmetic): about 0.74 GB of routed experts per token on a full miss. At 14.5 GB/s sequential that caps decode at about 20 tok/s with 0% hit rate. With a 70 to 90% hit rate (Strata-style hot set) the ceiling becomes about 65 to 200 tok/s for I/O alone. Real random 1.5 MB preads will run below sequential peak. Not measured.

---

## 5. Apple Silicon lineup (as of 2026-10-02)

### 5.1 Verified (with URL)

**MacBook Pro M5 / M5 Pro / M5 Max**

https://www.apple.com/macbook-pro/specs/ and press release https://www.apple.com/newsroom/2026/03/apple-introduces-macbook-pro-with-all-new-m5-pro-and-m5-max/ (March 3, 2026)

| Chip | GPU cores | Bandwidth | Memory options |
| --- | --- | ---: | --- |
| M5 | not given in excerpt | 153 GB/s | 16 / 24 / 32 GB |
| M5 Pro | 20-core | 307 GB/s | 24 / 48 / 64 GB |
| M5 Max | 32-core | 460 GB/s | 36 GB |
| M5 Max | 40-core | 614 GB/s | 48 / 64 / 128 GB |

- SSD: "up to 2x faster read/write performance ... reaching speeds of up to 14.5GB/s".
- Apple's LLM prompt-processing claims: "up to 4x faster LLM prompt processing than M4 Pro and M4 Max".

**M5 Pro / M5 Max chips**

https://www.apple.com/newsroom/2026/03/apple-debuts-m5-pro-and-m5-max-to-supercharge-the-most-demanding-pro-workflows/

- Built on the "Fusion Architecture", which combines two dies into a single SoC.
- "A Neural Accelerator in each GPU core".
- "Over 4x the peak GPU compute for AI compared to the previous generation".

**Mac Studio M5 Max / M5 Ultra**

https://www.apple.com/mac-studio/specs/ and https://www.apple.com/newsroom/2026/08/apple-introduces-new-mac-studio-with-m5-max-and-m5-ultra/ (August 25, 2026)

| Chip | CPU / GPU | Bandwidth | Memory options |
| --- | --- | ---: | --- |
| M5 Max | 32-core GPU | 460 GB/s | 36 GB |
| M5 Max | 40-core GPU | 614 GB/s | 48 / 64 / 128 GB |
| M5 Ultra | 30-core CPU / 64-core GPU | 1.2 TB/s | 96 GB |
| M5 Ultra | 36-core CPU / 80-core GPU | 1.2 TB/s | 256 / 512 GB |

- "Mac Studio with 512GB of unified memory is coming in late October".
- M5 Ultra: "8x faster LLM prompt processing in LM Studio when compared to Mac Studio with M1 Ultra, and up to 4x faster than M3 Ultra". Next-generation SSD "up to twice as fast".

**M6 and M5 Ultra chips**

https://www.apple.com/newsroom/2026/08/apple-introduces-m6-and-m5-ultra-for-a-big-leap-in-performance-and-ai-compute/ (August 25, 2026)

- M6: 2 nm, 12-core GPU "with a Neural Accelerator in each core", up to 170 GB/s, up to 32 GB.
- M5 Ultra: "first quad-die architecture", up to 80-core GPU, "1.2TB/s of unified memory bandwidth, 50 percent more than M3 Ultra", up to 512 GB.

**Mac mini M6 / M5 Pro**

https://www.apple.com/mac-mini/specs/ and https://www.apple.com/newsroom/2026/08/apple-unveils-a-more-powerful-mac-mini-featuring-the-all-new-m6-and-m5-pro/

| Chip | GPU | Bandwidth | Memory options |
| --- | --- | ---: | --- |
| M6 | 12-core | 153 GB/s at 16 GB; 170 GB/s at 24 / 32 GB | 16 / 24 / 32 GB |
| M5 Pro | 16-core or 20-core | 307 GB/s | 24 / 48 / 64 GB |

**Other current Macs**

- MacBook Air (M5): 153 GB/s, 16 / 24 / 32 GB. Saved copy of https://www.apple.com/macbook-air/specs/.
- iMac page still shows M4: 120 GB/s, up to 32 GB. https://www.apple.com/imac/specs/

**Previous generation, for projections**

- Mac Studio (2025): https://support.apple.com/en-us/122211

| Chip | GPU | Bandwidth | Memory listed on page |
| --- | --- | ---: | --- |
| M4 Max | 32-core | 410 GB/s | 36 GB |
| M4 Max | 40-core | 546 GB/s | 48 / 64 / 128 GB |
| M3 Ultra | 60-core or 80-core | 819 GB/s | 96 GB, configurable to 256 GB |

- Note on M3 Ultra memory: the support page lists only up to 256 GB, although 512 GB M3 Ultra units exist and are used in the benchmarks above.
- MacBook Pro 14" M4 Max: same 410 / 546 GB/s; 36 / 48 / 64 / 128 GB. https://support.apple.com/en-us/121553

**Memory configurations by tier** (from the spec pages above)

| RAM | Macs that offer it |
| --- | --- |
| 32 GB | M5 MacBook Pro / Air, M6 Mac mini |
| 36 GB | M5 Max 32-core GPU |
| 48 GB | M5 Pro, M5 Max 40-core GPU |
| 64 GB | M5 Pro, M5 Max 40-core GPU |
| 96 GB | M5 Ultra 64-core GPU only (no laptop) |
| 128 GB | M5 Max 40-core GPU (MacBook Pro or Mac Studio) |
| 256 / 512 GB | M5 Ultra 80-core GPU (512 GB shipping "late October") |

### 5.2 Unverified / inferred

- The Mac Pro specs page returned only boilerplate, so whether a current Mac Pro exists is **unverified**.
- M5 base-chip GPU core count is not given in my excerpt. I believe it is 10 cores (prior knowledge, unverified).
- SSD speeds for the Mac Studio and Mac mini have no published GB/s figure. Only "up to twice as fast" is verified, for the Mac Studio.

---

## 6. macOS 27 / Xcode 27 / Swift 6.4 / Metal 4.x, and memory APIs

### 6.1 Verified (with URL)

**Releases**

- macOS 27 released 2026-09-14. https://www.apple.com/newsroom/2026/09/major-updates-for-apples-software-platforms-are-now-available/
- Swift 6.4.0 released 2026-09-15. https://github.com/swiftlang/swift/releases/tag/swift-6.4.0-RELEASE
- Xcode 27 adds coding agents. https://developer.apple.com/xcode/whats-new/

**Metal "What's New"** (https://developer.apple.com/metal/whats-new/)

- "New quantized tensor formats and support for scale factors allow you to easily express quantized data."
- "Leverage native support in Metal performance primitives to efficiently compress weights and take advantage of the Neural Accelerators on the M5 Pro and M5 Max."

**WWDC26 session 330, "Optimize custom machine learning operations with Metal tensors"** (https://developer.apple.com/videos/play/wwdc2026/330/)

- TensorOps "automatically uses any available hardware acceleration across all Apple Silicon GPU generations".
- The Neural Accelerator is "a new hardware block in M5, located directly in each shader core ... designed to accelerate dense compute-bound work such as the prefill stage".
- Quantized tensors: 4/8-bit int since macOS 26. **macOS 27 adds 4/8-bit FP and 2-bit int**, plus multi-plane tensors with a scale plane (FP8 E8M0 block scales).
- New programming pieces:
  - `tensor_inline`.
  - Cooperative tensors as direct matmul inputs (`get_left_input_cooperative_tensor`, `is_compatible_as_left`).
  - `reduce_rows`.
- It includes a FlashAttention example and integration with Core AI.

**Other WWDC26 sessions** (index https://developer.apple.com/videos/wwdc2026/)

- 324 "Meet Core AI": a new framework. https://developer.apple.com/videos/play/wwdc2026/324/
- 232 "Run local agentic AI on the Mac using MLX". https://developer.apple.com/videos/play/wwdc2026/232/
  - "Neural Accelerators make matrix multiplication four times faster on M5 compared to M4 ... translates almost exactly to prompt processing speedup".
  - Thunderbolt RDMA has been available since macOS 26.2.
- 233 "Explore distributed inference and training with MLX".
- 328: Swift MLX.
- 241: Foundation Models.
- 339 "Bring an LLM provider to the Foundation Models framework".

**Local probe of this machine** (script /tmp/strata-research/mtlinfo.swift; `sysctl iogpu.wired_limit_mb`)

- Apple M5 Pro, 64 GB, macOS 27.0 (26A428), Xcode 27.0 (27A266a), Metal compiler 32023.921.
- `recommendedMaxWorkingSetSize` = 55,662,788,608 B (81.0% of RAM).
- `maxBufferLength` = 41,747,087,360 B.
- `maxThreadgroupMemoryLength` = 32,768.
- `supportsFamily(apple10)` = true; `supportsFamily(metal4)` = true.
- `iogpu.wired_limit_mb` = 0.

**Default working-set fraction** (https://github.com/waired-ai/waired-agent/issues/857)

- `iogpu.wired_limit_mb` defaults to 0.
- `recommendedMaxWorkingSetSize` measured at 74.0% on an M4 16 GiB (macOS 26.5.1) and 78.0% on an M5 Pro 48 GiB (26.6.2).
- Other reports: oMLX #3614 gives about 57.5 GB on 64 GB M1/M2 Max, and llama.cpp #29465 gives about 96 GiB on a 128 GB M4 Max.

**Metal backend memory handling** in llama.cpp and MLX: see section 2.1 (`newBufferWithBytesNoCopy`, `maxBufferLength` views, residency sets on macOS >= 15) and section 3.1 (`mx.set_wired_limit`).

**Sparse-resource risk** (Splash #220): long sparse map/unmap operations of more than 30 s triggered WindowServer watchdog logouts on macOS 26.4.1 to 27.0.1.

### 6.2 Unverified / inferred

**From prior knowledge, not re-verified against 2026 docs**

- `MTLResidencySet` was introduced at WWDC24 (macOS 15). The llama.cpp `@available(macOS 15.0)` gate is consistent with this.
- Metal 4's `MTL4CommandQueue` / `MTL4CommandBuffer` / argument tables and explicit residency were introduced at WWDC25 (macOS 26).
- I found no WWDC26 session specifically about residency sets, sparse heaps or MTL4 queues. Only session 330 (tensors) was read in full.
- `newBufferWithBytesNoCopy` requires a page-aligned pointer and a page-multiple length. The llama.cpp code aligns both, which is consistent with this, but I did not read Apple's documentation page.
- `iogpu.wired_limit_mb` set with sudo sysctl does not persist across reboots.

**Inferences**

- The observed fractions (74 to 81%; 75% at 128 GB) do not follow a single formula. A port should query `recommendedMaxWorkingSetSize` at runtime rather than hard-code a fraction.
- **Native quantized MTLTensor formats.** 2-bit int with E8M0 block scales maps naturally onto MXFP4-style and simple 2-bit formats (Q2_0-like, if its scale layout can be expressed). Codebook formats (IQ2_XS, IQ2_XXS, IQ3_*) and K-quant super-blocks do not map directly and would need custom dequant-to-cooperative-tensor kernels. ISTA's Q2_0 is likely the most Metal-friendly of the GSQ/RCO files.
- **640-column ffn_down.** The 640 columns are not a multiple of 256, so K-quants need padding: DwarfStar pads to 768, a 20% size and bandwidth overhead on down. A 32-element block format such as MXFP4, Q4_0, Q8_0 or Q2_0 avoids that.

---

## 7. Performance reference points and projections

### 7.1 Verified measured numbers (Qwen3.8-Flash-Next on Mac)

| Machine | Engine / quant | Prefill (tok/s) | Decode plain (tok/s) | Decode with MTP | Source |
| --- | --- | --- | --- | --- | --- |
| M3 Ultra 512 GB (819 GB/s) | DwarfStar Q2 (IQ2_XXS / Q2_K) | 1,331 @8K; 1,318 @32K | 50.1-50.4 (about 54.7 gen) | 64.8-90.5 | https://github.com/antirez/ds4 (speed-bench/qwen38-checkpoints/results.csv) |
| M3 Ultra | DwarfStar DS4-Q4 (Q4_K / MXFP4) | 1,071-1,171 (4-16K); 1,063 @262K | 45.4-45.6; 42.2 @262K | 55-56; 45.2 @262K | https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 |
| M3 Ultra | llama.cpp Q8_0 | n/a | 29.6-32.8 | n/a | https://github.com/ggml-org/llama.cpp/issues/29335 |
| M3 Ultra 512 GB | mlx-lm PR, 8-bit | n/a | 17-21 | n/a | https://github.com/ml-explore/mlx-lm/pull/1788 |
| M5 Max 128 GB (614 GB/s) | MTPLX 4-bit g64 | 1,453 @4K; 1,094-1,548 @65K | 43.8 | 73.5; 63.5 after a 65K prompt | https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed |
| M1 Ultra (800 GB/s) | llama.cpp IQ4_NL (branch) | 548 pp512; 335 @80K | 37.7; 31.2 @80K | n/a | https://github.com/ggml-org/llama.cpp/discussions/27965 |
| M1 Ultra (presumed) | llama.cpp split n-gram, read from SSD | 273 pp512 | 36.1 | n/a | https://github.com/ggml-org/llama.cpp/discussions/27864 |
| M1 Max 32 GiB (400 GB/s) | DwarfStar PR, Q2 + SSD expert streaming | n/a | n/a | 7.86 | https://github.com/antirez/ds4/pull/1047 |
| Mac not stated | mlx-lm PR, 4-bit | n/a | 24.0 (111 GB peak) | n/a | https://github.com/ml-explore/mlx-lm/pull/1788 |
| Reference: RTX 5070 12 GB + 64 GB RAM | Strata Q2_0 | 2,170 @32K | n/a | 93 | https://github.com/Niko1221/Strata |

### 7.2 Verified reference points (other MoE models on Mac)

- **gpt-oss-120b MXFP4 (59.02 GiB), M2 Ultra 192 GB, llama.cpp:** pp2048 1,244.57, tg128 79.68.
- **gpt-oss-20b, M3 Ultra 80-core GPU:** pp2048 2,816.47, tg128 115.52.
- **gpt-oss-20b, M1 Max 64 GB:** pp2048 994.75, tg128 75.15.
- Source for the three gpt-oss rows: https://github.com/ggml-org/llama.cpp/discussions/15396
- **DeepSeek V4 Flash Q2, M5 Max 128 GB, DwarfStar:** 790 PP / 39.35 TG at 2K, falling to 398.5 / 27.6 at 64K. https://github.com/antirez/ds4 (docs/PERFORMANCE.md)
- **Qwen3-235B-A22B-4bit, M5 Max 128 GB, oMLX expert offload (45% of experts resident):** 11.7 tok/s. https://github.com/jundot/omlx/issues/3911
- **Splash on M5 Pro 48 GB:** Qwen3.6-35B-A3B decodes at 210 tok/s (vs 69 in llama.cpp), with 2,011 tok/s prefill at 32K. https://github.com/incoai/splash

### 7.3 Unverified / inferred: projections

**Why decode is not bandwidth-bound**

- At about 2 GB of active bytes per token (Q2), M3 Ultra's measured 50 tok/s uses only about 100 GB/s of its 819 GB/s, roughly 12%.
- MTPLX 4-bit on M5 Max (about 3.4 GB/token x 43.8) uses about 150 GB/s of 614, roughly 24%.
- So decode is limited by launch, synchronization and latency: 48 layers, serial GDN recurrence, top-10 routing, HC ops and the QSA indexer.
- M5-generation GPUs reach similar decode with less bandwidth: M5 Max 4-bit at 43.8 vs M3 Ultra 4-bit at 45.5. Bandwidth matters mostly for SSD-streaming and large-batch cases.
- **Prefill is compute-bound** and benefits directly from the M5 Neural Accelerators.

**Assumptions for the table below**

- The quant is Q2-class with weights resident, using an engine of DwarfStar or MTPLX quality.
- Ranges are wide on purpose. They are not measurements.

| Mac | Fits Q2 resident? | Prefill @8K (tok/s) | Decode plain (tok/s) | Decode with MTP (tok/s) | Basis |
| --- | --- | ---: | ---: | ---: | --- |
| M4 Max 40-core, 64-128 GB (546 GB/s, no NA) | Yes at 64 GB and up | 400-650 | 35-45 | 50-70 | M1 Ultra llama.cpp 548 / 37.7; Apple says M5 Max PP is up to 4x M4 Max; tensor API gives no gain on M4 (llama.cpp comment) |
| M5 Pro 20-core, 64 GB (307 GB/s, NA) | Yes, tight (41.7 GiB in a 51.8 GiB budget) | 600-800 | 28-38 | 45-60 | Half the M5 Max cores and bandwidth; M5 Max MTPLX 1,453 PP; overhead-bound decode scales less than bandwidth |
| M5 Pro 48 GB | No: about 37.4 GiB budget | n/a | 5-15 with expert streaming | n/a | DwarfStar 32 GiB M1 Max 7.86 (MTP); expected to improve with I/O overlap (#1056) and a Strata-style hot set |
| M5 Max 40-core, 128 GB (614 GB/s) | Yes; 4-bit also fits | 1,300-1,600 | 44-55 | 70-90 (up to 125 with prompt cache) | Measured MTPLX 4-bit; Q2 should be slightly faster on decode |
| M3 Ultra (819 GB/s) | Yes | about 1,300 | about 50 | 65-90 | Measured (DwarfStar) |
| M5 Ultra 80-core (1.2 TB/s) | Yes | 2,500-3,500 | 50-70 | 80-110 | 2x M5 Max cores; Apple claims up to 4x M3 Ultra PP; quad-die latency may cap decode |
| M6 / M5 32 GB (153-170 GB/s) | No | n/a | 2-6 with expert streaming | n/a | Low bandwidth plus only about 24 GiB budget; most experts must come from SSD |

**What a Strata Metal port could add**

- Matching DwarfStar or MTPLX decode on 64-128 GB Macs is plausible.
- Beating them substantially there is unlikely: they are already Metal-tuned, with MTP and fused kernels.
- The real gap is 32 to 48 GB Macs with SSD expert streaming plus MTP. There, the best verified number is 7.86 tok/s on an M1 Max. An M5 Pro 48 GB with 14.5 GB/s-class SSD (MacBook Pro claim) and a good hot-expert cache could plausibly reach 10 to 20 tok/s. Inferred, not measured.

---

## 8. Memory-fit matrix (weights resident on GPU, PLE kept off the GPU)

### 8.1 Verified inputs

**Resident weight sizes**

| Build | Resident size | Source |
| --- | ---: | --- |
| ISTA Q2_0 shard 1 | 37.62 GB (35.0 GiB) | https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF |
| ISTA IQ2_XS shard 1 | 39.23 GB (36.5 GiB) | same |
| ISTA IQ3_XXS shard 1 | 47.04 GB (43.8 GiB) | same |
| ISTA IQ3_S shard 1 | 54.82 GB (51.1 GiB) | same |
| Coder IQ1_M shard 1 | 29.61 GB | https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF |
| DwarfStar Q2 | 41.73 GiB | https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md |
| DwarfStar Q4 | 69.74 GiB | same |
| DS4-Q4 main file | 69.7 GiB | https://huggingface.co/ivanfioravanti/Qwen3.8-Flash-Next-DS4-Q4 |
| MTPLX peak | 87 GiB (Bare Speed 78 GiB) | MTPLX model card |
| oMLX oQ4e non-PLE | about 74.3 GB | https://github.com/jundot/omlx/issues/3614 |

**GPU budgets** (`recommendedMaxWorkingSetSize`)

| RAM | Budget | Source |
| --- | ---: | --- |
| 64 GB (M5 Pro, local) | 55.66 GB | measured locally |
| 48 GiB | 78% | waired-agent #857 |
| 16 GiB | 74% | waired-agent #857 |
| 64 GB M1/M2 Max | about 57.5 GB | oMLX #3614 |
| 128 GB M4 Max | about 96 GiB | llama.cpp #29465 |

### 8.2 Inferred fit (weights plus about 3 to 6 GiB for KV and compute at 8 to 32K; my estimate)

**Default budget estimates**

| RAM | Default budget (estimate) |
| --- | --- |
| 32 GB | about 23-25 GiB |
| 36 GB | about 26-28 GiB |
| 48 GB | about 37 GiB |
| 64 GB | about 52 GiB |
| 96 GB | about 72-78 GiB |
| 128 GB | about 96 GiB |
| 256 GB and up | more than 190 GiB |

**What fits at each tier** (OK = fits; Tight = fits only with short context; No = does not fit; Stream = needs SSD expert streaming)

| Build (resident GiB) | 32 GB | 36 GB | 48 GB | 64 GB | 96 GB | 128 GB | 256 GB and up |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Coder IQ1_M (about 27.6) | No / Stream | Tight | OK | OK | OK | OK | OK |
| ISTA Q2_0 (35.0) | Stream | Stream | Tight (raise the wired limit) | OK | OK | OK | OK |
| ISTA IQ2_XS (36.5) | Stream | Stream | Tight / No | OK | OK | OK | OK |
| DwarfStar Q2 (41.73) | Stream (PR #1056) | Stream | Stream | OK (`--ctx 8192` per docs) | OK | OK | OK |
| ISTA IQ3_XXS (43.8) | Stream | Stream | Stream | OK (short context) | OK | OK | OK |
| ISTA IQ3_S (51.1) | Stream | Stream | Stream | No unless the wired limit is raised | OK | OK | OK |
| DwarfStar / DS4 Q4 (about 70) | No | No | No | Stream | Tight | OK | OK |
| MTPLX 4-bit (78-87) | No | No | No | No | Bare Speed only | OK | OK |
| 8-bit (about 190 GB total) | No | No | No | No | No | No | OK |

**Two technical constraints for 64 GB machines** (inferred)

- `maxBufferLength` is 41.7 GB, so any shard over that (IQ3_XXS and IQ3_S shard 1) must be split across several MTLBuffers.
- Raising `iogpu.wired_limit_mb` needs sudo. A shipped app cannot assume it; MLX only raises its limit up to the OS-recommended value.

---

## 9. Risks and implications for a Strata Metal port

All of section 9 is inference from the verified facts above.

### 9.1 Duplication risk

- llama.cpp, DwarfStar, MTPLX, oMLX and mlx-vlm already run this model well on 64 GB+ Macs.
- The value of a port is only clear for:
  1. 32 to 48 GB Macs, via SSD expert streaming with a hot-expert cache. This is Strata's core idea.
  2. Strata-specific UX (one-click installer, OpenAI endpoint), which LM Studio and Ollama already provide on Mac.

### 9.2 Reuse options

- **DwarfStar** (MIT, C + Metal, same model) is the best kernel donor or reference.
- **llama.cpp** (MIT) Metal kernels cover every op.
- **mlx-vlm** (MIT) `ple_storage.py` and `qsa_kernel.py` are useful references.
- **MTPLX and oMLX** are Apache-2.0, which is compatible with Strata's MIT license (attribution and NOTICE required).

### 9.3 Memory-architecture pitfalls (all verified issues)

1. **Keep the PLE out of any mapped range wrapped as an MTLBuffer.** llama.cpp #29465 shows the failure: a no-copy wrapper over a whole mmapped file counts against the working set. Read rows with pread instead, as DwarfStar does ("without mapping"); llama.cpp tarruda found plain fread as good as a custom cache.
2. **`maxBufferLength` (about 41.7 GB on 64 GB) forces several buffers** for larger shards.
3. **Be careful with sparse resources and large map/unmap operations.** Splash #220 shows that on macOS 26.4 to 27.0.1, long sparse mappings stalled WindowServer and logged users out. Prefer fixed-size expert slot pools (pread into preallocated shared buffers) over sparse-heap remapping. DwarfStar #1056 and the ExpertCache lessons point the same way: madvise and prewarm failed, explicit slots worked.
4. **Use residency sets with a keep-alive heartbeat,** as llama.cpp does, so wired weights are not reclaimed between requests.
5. **Batch throughput can collapse at long context** (llama.cpp #29335), and long-context EOS bugs appear with quantized KV (#28805). Validate the QSA indexer and KV paths at 64K and beyond.

### 9.4 Kernel and format implications

- Strata's GGUF formats (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_M) need Metal matvec / matmul kernels.
- Q2_0 is the cheapest to port and fastest on any backend (ISTA's measurements).
- IQ formats depend on lookup tables; on Metal those tables would sit in constant or threadgroup memory (`maxThreadgroupMemoryLength` = 32 KB on M5 Pro).
- The M5 Neural Accelerator path (MPP `matmul2d`, Metal language version 4.0) helps prefill only. It needs either dequant-to-half tiles or native int2/int4 tensors with scale planes (macOS 27).
- Decode on Mac is overhead-bound, so kernel fusion matters more than raw bandwidth. MTPLX, oMLX and Splash all lead with fused decode kernels.

### 9.5 Expected performance envelope

- See section 7.3: about 30 to 55 tok/s plain decode on 64 to 128 GB M5-class Macs, and about 600 to 1,600 tok/s prefill.
- That is below Strata's RTX 5070 Q2_0 numbers (93 decode, 2,170 PP at 32K) for anything below M5 Max or Ultra.

### 9.6 Toolchain baseline

- macOS 27 with Xcode 27 and the Metal 4 / MTL4 APIs.
- M5-or-later tensor paths must be gated; llama.cpp gates on device name and the Metal4 family.
- Thunderbolt 5 RDMA (macOS 26.2+) enables multi-Mac setups, but no Flash-Next engine implements it yet. Per the DwarfStar doc, tensor and pipeline parallelism are not implemented for this model.

---

## Appendix A. Local files produced during research (`/tmp/strata-research/`)

- `mtlinfo.swift`: Metal device probe.
- `llama.cpp/`: clone at fb4b2737.
- `ds4-*.md`: DwarfStar README and docs.
- `omlx-readme.md`, `omlx-070.md`, `splash-readme.md`, `strata-macos-readme.md`.
- `strata-forks.tsv` and `forks-refs.json`: fork scan.
- `disc-*.json`: llama.cpp discussions 15396, 26938, 27668, 27864, 27965.
- `pr-*.txt`: Apple press releases.
- `macbook-pro.txt`, `mac-studio.txt`, `macmini.html`, `sup-122211.txt`, `sup-121553.txt`: Apple specs pages.
- `wwdc26-*.txt`: WWDC26 session transcripts.
- `ista-readme.md`, `qwen-card.md`: model cards.
- `mlxlm-generate.py`.

## Appendix B. Open items not completed

- No local benchmark was run. This M5 Pro 64 GB could run DwarfStar Q2 (41.73 GiB resident) or llama.cpp with ISTA Q2_0 (37.6 GB shard 1 plus 28.8 GB PLE on SSD) to replace the M5 Pro projections with real numbers. That needs about 66 to 140 GB of downloads.
- I did not read the unsloth/Qwen3.8-Flash-Next-GGUF file list or the AtomicChat split GGUF (https://huggingface.co/AtomicChat/Qwen3.8-Flash-Next-GGUF) in detail.
- llama.cpp discussion #27668 (Metal FA tuning across M-chips) was fetched (`disc-27668.json`) but not mined.
