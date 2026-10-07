# 03. Model reference

Status: implementation-ready specification, 2026-10-03.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- [spikes/model-headers/](spikes/model-headers/) and [appendix-a-measurements.md](appendix-a-measurements.md),
  section A.9: the headers of every shipped GGUF file, read by HTTP range requests at the pinned Hugging Face
  revisions, and the sizes from the Hugging Face API;
- [audit/A-model-and-kernels.md](audit/A-model-and-kernels.md), sections A and B: the forward pass, the format table and
  the activation contract;
- [audit/K1-gr-router-shared-rope-cvec.md](audit/K1-gr-router-shared-rope-cvec.md),
  [K2](audit/K2-qsa-kv-gdn.md), [K3](audit/K3-quant-and-matvec.md) and [K4](audit/K4-ple-mtp-head-sampler.md):
  per-kernel arithmetic;
- upstream `tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`, `tools/mtp_rt.py`,
  `src/kernels/cpu/expert_layout.cpp`, `include/strata/kernels/cpu/expert.hpp`, `src/core/layer.cpp:355-392` and
  `setup.py:64-174`.

This document is the single description of the model the Mac engine runs: which files exist, what is in them, how the
bytes are laid out, which formats each file uses, how many bytes a token reads, and the rules of the forward pass that
the Metal kernels must keep. It fixes the facts that [04-metal-kernels.md](04-metal-kernels.md) (kernels),
[05-memory-and-residency.md](05-memory-and-residency.md) (budgets), [06-engine-host-port.md](06-engine-host-port.md)
(loaders) and [09-testing-and-validation.md](09-testing-and-validation.md) (golden files) build on. Kernel-level
choices (threadgroup shapes, tiling) belong to doc 04.

Where this document and an audit file disagree, this document wins. Section 1.3 lists the places where the audits are
superseded.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Scope and conventions

### 1.1 What this document covers

- The model architecture `qwen4exp` (Qwen3.8-Flash-Next) as Strata implements it.
- The seven GGUF files of the Mac catalogue (section 2.2), their metadata, tokeniser, tensor table and byte layout.
- The files that setup derives from them: the native pack (`index.txt`, `dense.bin`, `native_experts.txt`), the MTP
  runtime (`mtp/rt/`), the draft vocabulary, expert profiles and control vectors.
- The per-token byte counts that size the memory plan and the speed estimates.
- The forward-pass rules and their pointers into the audits, which hold the bit-level arithmetic.

It does not cover the CUDA kernels' launch shapes, the PC expert tiers, or the canonical (S-form) packs, which the Mac
engine does not support ([06](06-engine-host-port.md), section 6.1).

### 1.2 Conventions

- Units: GB = 10^9 B, GiB = 2^30 B, MiB = 2^20 B, KiB = 2^10 B. Byte counts are exact unless written "about".
- Shapes use the ggml `ne` order: `ne0` is the fastest dimension and is the input dimension of a weight matrix. A
  weight `[2560, 10240]` maps 2560 inputs to 10240 outputs and is stored as 10240 rows of 2560 elements.
- "Layer" is `blk.<l>`, 0 ≤ l < 48. A layer is QSA when `l % 4 == 3` and GDN otherwise.
- "The Q2_0 file" means the ISTA `Q2_0` release, and likewise for IQ2_XS, IQ3_XXS and IQ3_S. "Coder" is the ISTA Coder
  `IQ1_M` release. "Swift IQ2_XS" and "Swift IQ3_XXS" are the `ukisai` Swift 1.5 releases.
- **must** and **should** are normative. A "check" is something the loader verifies at startup and refuses with exit 1
  if it fails ([06](06-engine-host-port.md), section 3).
- Every number in sections 2-4 was read from the files at the revisions in section 2.2, or computed from their tensor
  tables with the scripts in [spikes/model-headers/](spikes/model-headers/). Numbers taken from the audits say so.

### 1.3 Where the audits are superseded

| Audit statement | Replaced by |
| --- | --- |
| A.1 and B.3 per-layer type counts (`attn_qkv` IQ4_XS ×13, Q3_K ×18, Q2_0 ×1, Q4_K ×4; `ffn_gate_shexp` "Q2_0 on 21 layers"; `ffn_up_shexp` "Q2_0 on 13 layers"), and the same lists in K3 section 1.2, all taken from the comment at `layer.cpp:113-121` | These counts describe the Q2_0 file only. Every file has its own map (section 3.8). The engine reads each tensor's type from the GGUF and must not assume a map. |
| B.3: "Dense tensors are identical across the ISTA Q2_0, IQ2_XS and IQ3_XXS files" | Only the float tensors (BF16, F16, F32) have the same type in every file. The quantised dense tensors differ per file, and `blk.1.ple_key` is Q2_0 in the Q2_0 file and BF16 in every IQ file. setup never passes `iq_pack.py --base`. |
| A.1, B.1 and D.5.2: the head is Q6_K, 521 MB per pass | The head is Q5_K in the Q2_0 and IQ3_XXS files (437,043,200 B), IQ4_XS in IQ2_XS (337,715,200 B) and Q6_K in IQ3_S and Coder (521,472,000 B). |
| B.1, IQ1_M row: "no shipped rows"; "low priority" | The IQ2_XS file (ISTA and Swift) stores the expert gate/up of layers 8, 13 and 37 as IQ1_M. IQ1_M is a Phase-1 expert format. (B.3 is right that Coder, despite its "IQ1_M" name, has no IQ1_M tensor.) |
| B.1, IQ2_XS row: "gate/up of the IQ2_XS file"; B.3: "ISTA IQ2_XS: IQ2_XS, plus other IQ types per layer" | The IQ2_XS file has no IQ2_XS expert tensor. Its gate/up is IQ2_S (34 layers), IQ2_XXS (11) or IQ1_M (3), always with a Q2_0 down. IQ2_XS gate/up appears in 10 layers of the IQ3_XXS file. |
| B.1 and B.3: the dense formats in use are K-quants, IQ4_XS and Q2_0 | IQ3_S is also a dense format: 44 dense tensors of the IQ2_XS file and 55 of the IQ3_XXS file (mostly `attn_gate`, `ssm_out` and the shared-expert gate/up), and the `token_embd` of IQ3_XXS. Section 3.3 has the complete list. |
| B.3: "Swift 1.5: as ISTA, layers 13-47 in shard 2"; the `iq_pack.py` docstring | Swift IQ2_XS puts the experts of layers 13-47 in shard 2, Swift IQ3_XXS those of layers 12-47. One layer's dense tensors straddle the shards, and the PLE table is in shard 1 (section 3.9). |
| B.4.2 and K5 section 2.2: "Layer starts are 16 KiB aligned"; B.4.5 and D.5.2: "one buffer per layer, 16 KiB aligned" | In the GGUF, the only guaranteed alignment of a tensor's file offset is 32 B. A few expert tensors fall on a 16 KiB boundary by chance (3 of 144 in IQ3_XXS, 2 in IQ3_S, none in the other files). Only the layer offsets inside `experts.bin` (low-RAM mode, which the Mac does not use) are aligned, and only because of the dimensions: every blob is a multiple of 1 KiB, so `blob × n_expert` is a multiple of 512 KiB (256 KiB for Coder). Buffers are planned as windows ([05](05-memory-and-residency.md), section 4.2). |
| B 2.4 rule 3 (and [06](06-engine-host-port.md) section 6.1 before this document): `native_experts.txt` is v4; "refuse any version other than 4" | `iq_pack.py` writes v3, with one optional shard column per layer, and writes v4 only when one layer's three roles are in different shards. No Mac-catalogue file needs v4. The reader must accept v3 and v4 and refuse versions above 4 (section 3.5). |
| A.6: "Metal: dispatch exactly 32 threads per token" for the native router, presented as the router of every model | The native router runs only when `n_expert == 512 && k == 10` (`layer.cpp:370`, `verify.cpp:674`). Coder (256 experts) runs the canonical FP64 router on CUDA. The Metal router handles both widths with one kernel (section 6.6). |
| A.1 and B.3: routers `ffn_gate_inp` are BF16 | Swift stores its 96 router tensors as F32 that hold exact BF16 values. The pack narrows them to BF16 without loss (section 3.7). |
| D.5.2: "native IQ3: about 0.9 GB per layer" | 512 experts of a layer are 0.89 GB on average for IQ3_XXS (0.67-1.19 GB) and 1.05 GB for IQ3_S (0.77-1.36 GB). |
| B 2.3: "MTP rt … dense ~0.2 GB" | `mtp/rt/dense.bin` is 116,099,072 B (0.108 GiB); the whole `rt/` folder is 823,887,872 B (section 4.4). |
| A.2 rule 6: "`--dump-layers` writes per layer `2N + 2hc + NH·HD + 5·NKV·HD + 8` floats"; E.3: retained teacher-forced dump format | `--dump-layers` writes the residual ladder, `(48 + 1) × 4 × 2560` floats per position. The 13,840-float layout is `--dump-halves`, and the code writes more slots than A.2 lists. The reader is `tools/metal/tf_compare.py` ([09](09-testing-and-validation.md), section 3.3). |
| A.10, B.1 table (Q5_K row) and K4 section 2.3: the Q5_K head uses `native_q5_k_f32` "with f32 activations" | That function quantises its F32 input to Q8_1 before the Q5_K mat-vec (`native_mmvq.cu:1192-1203`). Every head format meets a Q8_1 activation (sections 3.3 and 5). |
| A.1: norms "stored as 1 + w" | The GGUF stores the final multiplier (the converter has already added 1; `output_hc_norm` ranges 0.63-13.4). The engine multiplies by the stored value and never adds 1 (section 3.7). This restates the audit; it does not change it. |

---

## 2. The model and its files

### 2.1 Identity

| Item | Value |
| --- | --- |
| Model | Qwen3.8-Flash-Next, GGUF architecture `qwen4exp`, `general.size_label` `512x56B` (also in Coder) |
| Parameters, excluding the PLE table | 125,743,653,760 (Coder: 65,314,218,880) |
| Of which routed experts | 120,795,955,200 = 48 × 512 × 3 × 2560 × 640 (Coder: 60,397,977,600) |
| Active per token | 6,671,295,360: everything except `token_embd`, the PLE table and 502 of the 512 experts of each layer. The engine also reads one embedding row and 16 PLE rows per token. |
| PLE table | 51,200,245,760 elements (`[160, 320001536]`), outside the parameter count above |
| Layers | 48: 36 GDN (Gated DeltaNet) and 12 QSA (sparse attention), QSA iff `l % 4 == 3` |
| Residual | 4 hyper-connection streams of 2560 (D = 10240) |
| Experts | 512 routed (Coder 256), top-10, plus one gated shared expert per layer; FF 640 for both |
| Context | `context_length` 262144; longer contexts use YaRN ([07](07-server-setup-port.md), section 4.5.2) |
| Vocabulary | 248,320 |
| Extra layer | One MTP draft layer, not in the GGUF; fetched from the BF16 checkpoint (section 2.3) |
| Vision | A separate BF16 `mmproj` encoder GGUF (section 2.3) |

### 2.2 Families and files

All four repositories are pinned to a commit in `setup.py` `HF_REVISIONS` (#214). The URL of a file is
`https://huggingface.co/<repo>/resolve/<revision>/<path>`.

| Family (setup key) | Repository | Revision (date) | Quants in setup | Mac |
| --- | --- | --- | --- | --- |
| `qwen` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `ed59f92082b1e93c0e96d60a8b11aab089b52f09` (2026-09-29) | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | all four |
| `swift` | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `b22d729eae29b5796f76fb70f91aef549b9fc52c` (2026-09-24) | IQ2_XS, IQ3_XXS | both. Licence: Swift Open License 1.0, which setup prints before the download. |
| `coder` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | `5348543e0147355ac9cbcb031184a3546350988e` (2026-09-29) | IQ1_M | yes |
| `unsloth` | `unsloth/Qwen3.8-Flash-Next-GGUF` | `38bb39ee97821de2c9009abb7e93950eec396e66` (2026-09-30) | UD-Q4_K_XL (experimental) | **no**: refused ([07](07-server-setup-port.md), section 4.1) |

File paths, with `{q}` the quant name and `{i}` the shard number:

| Family | Shard path |
| --- | --- |
| `qwen` | `{q}/Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` |
| `swift` | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` (repository root) |
| `coder` | `{q}/Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf` |
| `unsloth` | `{q}/Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf` |

Files that exist but are out of scope for the Mac:
- **Swift Q2_0.** The repository has it (shards of 39,799,117,984 and 26,750,834,816 B), but it splits one layer's
  experts across the two shards. setup hides it (`MODELS["Q2_0"]["families"] = ("qwen",)`, #171), and so does the Mac
  catalogue.
- **Unsloth UD-Q4_K_XL.** Its 77 GB of experts do not fit any Mac this port targets in full mode, and on a PC it runs
  from the SSD at 7-8.5 tokens/s. It also needs `--compat-bf16` (lossy F32-to-BF16 narrowing). Refused on the Mac.
- **Orca IQ3_XXS and Q4_K_S.** These upstream quantizations are not part of the Mac model catalogue and have not
  been validated by the Metal product. They use Q5_0 expert downs, which the current Metal expert path does not support;
  the Q5_0 PLE variant also remains outside the first release (section 3.3).

### 2.3 Other files

| File | Source | Size (B) | Notes |
| --- | --- | ---: | --- |
| Vision encoder, `qwen` and `coder` | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` at each repository's root | 907,543,008 | Coder's copy is the same file. Unsloth uses the ISTA copy. |
| Vision encoder, `swift` | `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` | 907,543,520 | |
| MTP source | `Qwen/Qwen3.8-Flash-Next`, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` (pinned 2026-09-30; `STRATA_MTP_REVISION` overrides it) | 5,214,301,696 fetched | 31 BF16 `mtp.*` tensors in 28 of the checkpoint's 131 safetensors shards (360 GB in total). `mtp_fetch.py` reads only their byte ranges: every response must be `206` with a matching `Content-Range` (#327), and each tensor's SHA-256 is pinned. |
| MTP runtime | `mtp/rt/` written by `mtp_pack.py` + `mtp_rt.py` | 823,887,872 | Section 4.4 |
| Draft vocabulary | `data/draft_vocab.bin` (`cjk`, default), `draft_vocab_en.bin`, `draft_vocab_cyrillic.bin`, copied to `mtp/rt/draft_vocab.bin` (`setup.py:2411`) | 425,196 / 162,100 / 235,852 | Raw little-endian int32 token ids, **not sorted**: 106,299 / 40,525 / 58,963 ids |
| Expert profile | `data/expert-profile.bin`; `data/expert-profile-coder.bin` for Coder | 196,632 / 98,328 | `STRP` header (24 B) + 48 × E × 8 B ([06](06-engine-host-port.md), section 9.5) |
| Control vector | `data/experimental-speed-projection/Qwen3.8-Flash-Next-experimental-speed-projection.gguf` | 483,520 | Arch `controlvector`, tensors `direction.<l>` F32 `[2560]` for 1 ≤ l < 48 |
| Optional FP8 PLE table | `ple-fp8.gguf` from `tools/ple_fp8_pack.py` | about 51.2 GB | Arch `strata-ple`, I8 tensor holding FP8 E4M3 plus `strata.ple.scale`. Not in the first Mac release. |
| Optional BF16 embedding | `token-embd-bf16.gguf` from `tools/embd_bf16_pack.py` | about 1.27 GB | Arch `strata-embd`, read by `--embd-gguf` |

### 2.4 Shard sizes

From the Hugging Face API at the pinned revisions (`/api/models/<repo>/tree/<revision>?recursive=true`).

| File | Shard 1 (B) | Shard 2 (B) | setup `download_gb` |
| --- | ---: | ---: | ---: |
| ISTA Q2_0 | 37,623,740,192 | 28,800,138,432 | 66.4 |
| ISTA IQ2_XS | 39,225,954,592 | 28,800,138,432 | 68.0 |
| ISTA IQ3_XXS | 47,039,860,096 | 28,800,138,432 | 75.8 |
| ISTA IQ3_S | 54,817,524,224 | 28,800,138,432 | 83.6 |
| Coder IQ1_M | 29,608,446,496 | 28,800,138,432 | 58.4 |
| Swift IQ2_XS | 39,788,473,344 | 28,363,693,824 | 68.0 |
| Swift IQ3_XXS | 39,785,790,560 | 36,180,282,560 | 75.8 |

- The shard 2 of every ISTA file and of Coder is one file (Git LFS object `316b46f3a2dbd68c…`): one tensor,
  `per_layer_token_embd.weight`, IQ4_NL `[160, 320001536]`, 28,800,138,240 B at file offset 192. setup
  hard-links an existing copy instead of downloading it again (`setup.py:3047-3053`), so the engine must treat it as
  read-only and shared.
- Swift's shards hold different things (section 3.9).
- setup's `download_gb` adds the `mmproj` and rounds. `ram_gb` and `arena_gb` in `MODELS` are PC numbers; the Mac plan
  computes its own ([05](05-memory-and-residency.md), section 3).

### 2.5 Metadata

Every shard 1 of the Mac catalogue has these values. The loader **must check** the ones in bold and refuse the file
with a message naming the key if one differs. Upstream reads the same keys in `gguf_reader.hpp` and `generate.cpp`.

| Key | Value | Engine use |
| --- | --- | --- |
| **`general.architecture`** | `qwen4exp` | check |
| `general.alignment` | absent; the default 32 applies | data_start alignment (section 3.4) |
| **`qwen4exp.block_count`** | 48 | check |
| **`qwen4exp.embedding_length`** | 2560 | check |
| **`qwen4exp.attention.head_count`** / **`head_count_kv`** | 24 / 2 | check |
| `qwen4exp.attention.key_length` / `value_length` | 256 / 256 | informational; the engine's constant |
| **`qwen4exp.expert_count`** | 512 (Coder 256) | sets `n_expert`; must equal `ffn_gate_inp` `ne1` of every layer |
| **`qwen4exp.expert_used_count`** | 10 | check |
| `qwen4exp.expert_feed_forward_length`, `expert_shared_feed_forward_length` | 640, 640 | informational |
| `qwen4exp.context_length` | 262144 | the trained context; YaRN factor base ([07](07-server-setup-port.md), section 4.5.2) |
| `qwen4exp.rope.freq_base` | 1.0e7 | read by `generate.cpp`; the engine must use the file's value |
| `qwen4exp.rope.dimension_count` / `dimension_sections` | 64 / `[11, 11, 10, 0]` | informational; the engine's constants |
| `qwen4exp.rope.scaling.{type, factor, original_context_length}` | absent | read if present (YaRN). No shipped file has them. |
| `qwen4exp.attention.layer_norm_rms_epsilon` | 1e-6 | informational; the engine's constant |
| `qwen4exp.full_attention_interval` | 4 | informational; QSA iff `l % 4 == 3` |
| `qwen4exp.ssm.{conv_kernel, state_size, group_count, time_step_rank, inner_size}` | 4, 128, 16, 48, 6144 | informational |
| `qwen4exp.hyper_connection.{count, low_rank}` | 4, 320 | informational |
| `qwen4exp.indexer.{head_count, key_length, top_k}`, `compress_ratios` | 4, 128, 2048, `[0, 0, 0, 4, …]` | informational |
| `qwen4exp.ple.*` | `layers [1]`, `ngram_size 3`, `heads_per_ngram 8`, `conv_kernel 4`, `eos 248044`, `layer_multipliers [23703573157769, 20109073645365, 8052911324071]`, `head_offsets`, `head_vocab_sizes` (primes from 20000003), `embedding_length_per_layer_input 160`, `image_token_id 248056` | informational: the engine's constants in `ngram.hpp` equal them. A test compares them (section 11). |
| `qwen4exp.sampling.{top_k, top_p, temp}` | 20, 0.95, 1.0 | informational; the server's defaults are the same |
| **`split.count`**, **`split.tensors.count`**, `split.no` | 2, 1224, 0 or 1 | check: the shards together list 1224 tensors |
| `general.file_type` | 41 (Q2_0), 20 (IQ2_XS), 23 (IQ3_XXS), 26 (IQ3_S), 31 (Coder) | informational. The engine never derives a tensor's type from it. |
| `general.name`, licence keys, `quantize.imatrix.*` | differ per family (imatrix chunks: 1000 ISTA, 2472 Swift) | informational |
| `qwen4exp.attention.recurrent_layers` | Swift only | informational |

Optional packs add their own keys: `strata.ple.format` and `strata.ple.scale` (`ngram.cpp`), `strata.mtp.*`, and
`strata.embd.source`. `tokenizer.*` is read by the server's `strata_tokenizer.py`, not by the engine.

### 2.6 Tokeniser and special tokens

Verified with `tokcmp.py` and `tokpeek.py` ([spikes/model-headers/](spikes/model-headers/)): the token list, the
merges and the token types are identical in all seven shard 1 files.

| Item | Value |
| --- | --- |
| Model / pre-tokeniser | `gpt2` / `qwen35` |
| Tokens / merges | 248,320 / 247,587 |
| Token types | 248,044 normal (1), 27 control (3), 6 user-defined (4), 243 unused (5: `[PAD248077]` … `[PAD248319]`) |
| EOS | 248046 `<|im_end|>` |
| BOS = padding | 248044 `<|endoftext|>`. `add_bos_token` is false in ISTA and Coder and absent in Swift: no BOS is added. 248044 is also the PLE context cut (section 6.8). |
| Chat template | 9,993 characters in every ISTA file and in Coder (identical); 8,952 characters in both Swift files (identical to each other, different from ISTA). The server must take the template from the installed file, never from a constant. |

Special tokens (types 3 and 4):

| Id | Token | Type |
| ---: | --- | --- |
| 248044 | `<|endoftext|>` | 3 |
| 248045 | `<|im_start|>` | 3 |
| 248046 | `<|im_end|>` | 3 |
| 248047-248048 | `<|object_ref_start|>`, `<|object_ref_end|>` | 3 |
| 248049-248050 | `<|box_start|>`, `<|box_end|>` | 3 |
| 248051-248052 | `<|quad_start|>`, `<|quad_end|>` | 3 |
| 248053-248054 | `<|vision_start|>`, `<|vision_end|>` | 3 |
| 248055 | `<|vision_pad|>` | 3 |
| 248056 | `<|image_pad|>`: rows the engine replaces with image embeddings (section 6.11) | 3 |
| 248057 | `<|video_pad|>` | 3 |
| 248058-248059 | `<tool_call>`, `</tool_call>` | 4 |
| 248060-248063 | `<|fim_prefix|>`, `<|fim_middle|>`, `<|fim_suffix|>`, `<|fim_pad|>` | 3 |
| 248064-248065 | `<|repo_name|>`, `<|file_sep|>` | 3 |
| 248066-248067 | `<tool_response>`, `</tool_response>` | 4 |
| 248068-248069 | `<think>`, `</think>` | 4 |
| 248070-248076 | `<|audio_start|>`, `<|audio_end|>`, `<tts_pad>`, `<tts_text_bos>`, `<tts_text_eod>`, `<tts_text_bos_single>`, `<|audio_pad|>` | 3 |

---

## 3. Tensors and their layout

### 3.1 Constants

These are compile-time constants of the engine (`include/strata/core/layout.hpp` and `mtp.hpp`,
`include/strata/kernels/qsa.hpp`, `gdn.hpp`, `ngram.hpp` and `mrope.hpp`); [audit A.0](audit/A-model-and-kernels.md)
gives the source line of each. Only `n_expert` varies between
the catalogue files, and the engine takes it from the file (section 2.5).

| Constant | Value |
| --- | --- |
| `n_embd` N | 2560 |
| Layers | 48; QSA iff `l % 4 == 3` (layers 3, 7, …, 47) |
| Hyper-connection streams `hc` / residual D | 4 / 10240 |
| GR low rank | 320 |
| Routed experts / top-k / FF | 512 (Coder 256) / 10 / 640; the shared expert's FF is also 640 |
| GDN | 16 q/k heads and 48 v heads of 128; the q/k head of v-head h is `h % 16`; conv kernel 4 over 10240 channels (q `[0, 2048)`, k `[2048, 4096)`, v `[4096, 10240)`); value_dim 6144; state `float[128][48][128]` + conv history `float[10240][3]` = 3.12 MiB per layer, 112 MiB for 36 layers |
| QSA | 24 query heads and 2 KV heads of 256; q head h uses KV head `h / 12`; `attn_q` output 12288 = per head 256 query then 256 gate; attention scale 1/16 |
| RoPE | NEOX on the first 64 dims, pairs (i, i + 32); base 1e7 (from the file); M-RoPE sections {11, 11, 10} = `pair % 3` |
| Indexer | 4 query heads of 128, one shared 128-dim key; block 4 cells; top_k 2048 cells; selection width `min(n_kv, 2051)`; step buffer `int32[4] = {pos, pos + 1, (pos + 1) / 4, width}` |
| RMS epsilon | 1e-6 everywhere |
| Router weight clamp | the sum of the 10 selected probabilities is clamped to ≥ 2^-14 |
| PLE | layer 1 only; 16 heads × 160 = 2560; table 320,001,536 rows; history 9 rows × 10240; EOS 248044 cuts the n-gram context |
| Vocabulary | 248,320 (`output.weight` `ne1`) |
| Verify window | 1 ≤ T ≤ 8 (`kVerifyMaxT`, `kFusedGrMaxT`) |
| MTP | dense attention over the last 32,768 cells (`--mtp-window`); 512 experts |

### 3.2 Tensor inventory

A model has 1224 tensors: 36 GDN layers × 25, 12 QSA layers × 26, 6 PLE tensors in layer 1, and 6 global tensors.
Shapes are ggml `ne`. "Quant" means a block-quantised type that varies per file (section 3.8).

Global:

| Tensor | `ne` | Type |
| --- | --- | --- |
| `token_embd.weight` | [2560, 248320] | quant: Q3_K, IQ3_S or IQ4_XS |
| `output.weight` (head) | [2560, 248320] | quant: Q5_K, IQ4_XS or Q6_K |
| `output_hc_norm.weight` | [10240] | F32 |
| `output_hc_down.weight` | [10240, 320] | BF16 |
| `output_hc_up.weight` | [320, 10240] | BF16 |
| `per_layer_token_embd.weight` (PLE table) | [160, 320001536] | IQ4_NL, 90 B per row |

Every layer (16 tensors):

| Tensor | `ne` | Type |
| --- | --- | --- |
| `hc_attn_norm.weight`, `hc_ffn_norm.weight` | [10240] | F32 |
| `hc_attn_down.weight`, `hc_ffn_down.weight` | [10240, 320] | BF16 |
| `hc_attn_up.weight`, `hc_ffn_up.weight` | [320, 10240] | BF16 |
| `hc_attn_inject.weight`, `hc_ffn_inject.weight` | [10240, 4] | BF16 |
| `ffn_gate_inp.weight` (router) | [2560, n_expert] | BF16 (Swift: F32 holding BF16 values) |
| `ffn_gate_inp_shexp.weight` (shared-expert gate) | [2560] | BF16 (Swift: F32) |
| `ffn_gate_shexp.weight`, `ffn_up_shexp.weight` | [2560, 640] | quant |
| `ffn_down_shexp.weight` | [640, 2560] | quant, 32/64-element blocks only |
| `ffn_gate_exps.weight`, `ffn_up_exps.weight` | [2560, 640, n_expert] | quant, the same type for gate and up |
| `ffn_down_exps.weight` | [640, 2560, n_expert] | quant, 32/64-element blocks only |

GDN layers, 9 more:

| Tensor | `ne` | Type |
| --- | --- | --- |
| `attn_qkv.weight` | [2560, 10240] | quant |
| `attn_gate.weight` (z) | [2560, 6144] | quant |
| `ssm_out.weight` | [6144, 2560] | quant |
| `ssm_conv1d.weight` | [4, 10240] | F32 |
| `ssm_alpha.weight`, `ssm_beta.weight` | [2560, 48] | BF16 |
| `ssm_a`, `ssm_dt.bias` | [48] | F32 (no `.weight` suffix; `ssm_a < 0`) |
| `ssm_norm.weight` | [128] | F32 |

QSA layers, 10 more:

| Tensor | `ne` | Type |
| --- | --- | --- |
| `attn_q.weight` | [2560, 12288] | quant |
| `attn_k.weight`, `attn_v.weight` | [2560, 512] | quant |
| `attn_output.weight` | [6144, 2560] | quant |
| `attn_q_norm.weight`, `attn_k_norm.weight` | [256] | F32 |
| `indexer.q_proj.weight` | [2560, 512] | BF16 |
| `indexer.k_proj.weight` | [2560, 128] | BF16 |
| `indexer.q_norm.weight`, `indexer.k_norm.weight` | [128] | F32 |

Layer 1 only (PLE block):

| Tensor | `ne` | Type |
| --- | --- | --- |
| `ple_key.weight` | [2560, 10240] | Q2_0 in the Q2_0 file, BF16 in every other file |
| `ple_value.weight` | [2560, 2560] | BF16 |
| `ple_norm_key.weight`, `ple_norm_query.weight`, `ple_norm_conv.weight` | [10240] | F32 |
| `ple_conv1d.weight` | [4, 10240] | F16 |

There is no `attn_norm`, `ffn_norm` or `output_norm`: the GR norms take their place.

### 3.3 Formats

The Mac engine must support exactly the formats of the catalogue files. The bit layouts and decode formulas are in
[audit B.1](audit/A-model-and-kernels.md) and [K3 section 1.3](audit/K3-quant-and-matvec.md).

| Format (ggml id) | Block: elements / bytes | Used for, in the Mac catalogue |
| --- | --- | --- |
| Q2_0 (42) | 64 / 18 | Expert gate/up (Q2_0 file) and expert down (all files except where IQ4_NL); dense `attn_qkv`, `attn_gate`, `attn_q`, shared experts and `ple_key` of the Q2_0 file; shared-expert down of the IQ2_XS and IQ3_XXS files |
| IQ1_M (29) | 256 / 56 | Expert gate/up, IQ2_XS layers 8, 13, 37 |
| IQ2_XXS (16) | 256 / 66 | Expert gate/up (IQ2_XS, IQ3_XXS) |
| IQ2_XS (17) | 256 / 74 | Expert gate/up (IQ3_XXS) |
| IQ2_S (22) | 256 / 82 | Expert gate/up (IQ2_XS, IQ3_XXS, IQ3_S, Coder) |
| IQ3_XXS (18) | 256 / 98 | Expert gate/up (IQ3_XXS, IQ3_S, Coder) |
| IQ3_S (21) | 256 / 110 | Expert gate/up (IQ3_XXS, IQ3_S, Coder); dense projections and shared experts (IQ2_XS, IQ3_XXS); `token_embd` (IQ3_XXS) |
| IQ4_XS (23) | 256 / 136 | Expert gate/up (IQ3_S and Coder layer 47); dense projections (all files); head (IQ2_XS); `token_embd` (IQ2_XS, IQ3_S, Coder) |
| IQ4_NL (20) | 32 / 18 | Expert down (IQ3_XXS, IQ3_S, Coder); shared-expert down (all files); the PLE table (CPU only) |
| Q3_K (11) | 256 / 110 | Dense projections and shared experts (Q2_0 file); `token_embd` (Q2_0) |
| Q4_K (12) | 256 / 144 | Dense projections and shared experts (all files except IQ2_XS) |
| Q5_K (13) | 256 / 176 | Dense projections and shared experts (all files except IQ2_XS); head (Q2_0, IQ3_XXS) |
| Q6_K (14) | 256 / 210 | Dense projections and shared experts (all files); head (IQ3_S, Coder) |
| Q4_0 (2), Q5_0 (6) | 32 / 18, 32 / 22 | Shared-expert down (Q2_0 file) |
| Q8_0 (8) | 32 / 34 | Shared-expert down (Q2_0, IQ2_XS, IQ3_S, Coder); MTP dense projections (`mtp/rt`) |
| BF16 (30) | 1 / 2 | Routers, GR mixers, indexer projections, `ssm_alpha/beta`, `ple_value`, `ple_key` (IQ files) |
| F16 (1) | 1 / 2 | `ple_conv1d` |
| F32 (0) | 1 / 4 | Norms, `ssm_a`, `ssm_dt.bias`, `ssm_conv1d`; Swift routers (narrowed to BF16 by the pack) |

So the Phase-1 kernel set needs:
- **dense GEMV (Q8_1 activations):** Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S, IQ4_XS, and for the shared-expert down
  Q2_0, IQ4_NL, Q4_0, Q5_0, Q8_0;
- **expert gate/up:** Q2_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS;
- **expert down:** Q2_0, IQ4_NL;
- **embedding row gather:** Q3_K, IQ3_S, IQ4_XS (and BF16 for `--embd-gguf`);
- **head:** Q5_K, IQ4_XS, Q6_K;
- **BF16 GEMV (F32 activations)** and the F16 conv taps;
- **CPU decode** of IQ4_NL PLE rows;
- **the S2 blob** for the MTP experts (section 3.6).

Formats that appear in no catalogue file (Q5_1, Q2_K, IQ1_S, MXFP4, NVFP4, Q8_K as a weight) need no kernel. A file
that uses one must be refused at load, naming the tensor and the type.

**FF = 640 = 2.5 × 256.** A tensor whose input dimension is 640 (every expert down and every `ffn_down_shexp`)
cannot use a 256-element superblock, so it is always a 32- or 64-element format. The loader **must check** this
(`ne0 % block_elements == 0` for every quantised tensor) rather than trust it.

**The Q5_K head is not an exception to the activation contract.** Upstream's decode head calls `native_q5_k_f32`
for a Q5_K `output.weight`, which audit A.10 and K4 section 2.3 describe as an F32-activation kernel. The function
(`native_mmvq.cu:1192-1203`) takes F32, quantises it with `native_quantize_q8_1` and calls `native_q5_k_mmvq`, and the
verify head quantises to Q8_1 for every type (`verify.cpp:842-843`). Every head is a Q8_1 GEMV (section 5).

Row sizes, which give every slab and blob size in this document:

| Row of 2560 (gate, up, dense with `ne0` 2560) | B | Row of 640 (down) | B |
| --- | ---: | --- | ---: |
| IQ1_M | 560 | Q2_0 | 180 |
| IQ2_XXS | 660 | IQ4_NL, Q4_0 | 360 |
| Q2_0 | 720 | Q5_0 | 440 |
| IQ2_XS | 740 | Q8_0 | 680 |
| IQ2_S | 820 | | |
| IQ3_XXS | 980 | | |
| IQ3_S, Q3_K | 1100 | | |
| IQ4_XS | 1360 | | |
| Q4_K | 1440 | | |
| Q5_K | 1760 | | |
| Q6_K | 2100 | | |

### 3.4 Byte layout of a GGUF shard

- `data_start = align_up(end of the tensor table, general.alignment)`. No catalogue file sets `general.alignment`, so
  it is 32. Tensor offsets in the table are relative to `data_start`.
- Tensors are stored back to back in table order with no padding beyond the 32-B alignment (verified: no gaps in any
  shard). The order is the non-block tensors first (alphabetical), then the layers in ascending order, each layer's
  tensors alphabetical. Swift's shard 2 continues the same order.
- The only alignment a tensor's absolute file offset is guaranteed to have is 32 B. The engine **must not** assume more:
  windows round out to 16 KiB ([05](05-memory-and-residency.md), section 4.2), and tiered-mode copies use `pread`
  at any offset.

| Shard | data_start (B) | Tensors |
| --- | ---: | ---: |
| ISTA Q2_0 / IQ2_XS shard 1 | 11,024,672 | 1223 |
| ISTA IQ3_XXS shard 1 | 11,024,256 | 1223 |
| ISTA IQ3_S shard 1 | 11,024,384 | 1223 |
| Coder shard 1 | 11,024,416 | 1223 |
| ISTA and Coder shard 2 | 192 | 1 (the PLE table) |
| Swift IQ2_XS shard 1 / shard 2 | 10,967,808 / 56,320 | 342 / 882 |
| Swift IQ3_XXS shard 1 / shard 2 | 10,966,240 / 57,920 | 317 / 907 |

**Routed experts.** The three expert tensors of a layer have `ne` = {2560, 640, n_expert} (gate, up) and
{640, 2560, n_expert} (down). The expert index is the slowest dimension, so each expert's part of each role is one
contiguous **slab**:

```
slab_gu(l) = 640  × row_bytes(gu_type(l), 2560)       // gate and up are the same size
slab_d(l)  = 2560 × row_bytes(d_type(l), 640)
blob(l)    = 2 × slab_gu(l) + slab_d(l)
addr(l, e, role) = role_abs(l, role) + e × slab(l, role)   // role_abs: column 6, 7 or 8 of native_experts.txt
```

For Q2_0, each slab is 460,800 B and the blob is 1,382,400 B. One expert is therefore three contiguous byte ranges at
three different file offsets, not one blob. Within a slab, rows are in output order (row r of gate is the r-th
output); a gate/up slab is 640 rows of 2560 elements, and a down slab is 2560 rows of 640 elements.

Every slab is a multiple of 512 B and every blob a multiple of 1 KiB, but slabs inherit only the 32-B alignment of
their tensor's start.

### 3.5 `native_experts.txt`

Written by `iq_pack.py` (`expert_layout`, `tools/iq_pack.py:429-473`), read by `expert_layout_load`
(`src/kernels/cpu/expert_layout.cpp:132-238`). The pack step writes it to a temporary name and renames it after
`index.txt` and `dense.bin` are in place (`tools/iq_pack.py:570-575`), so its presence marks a finished pack (#172).

Header line, v3 (every Mac-catalogue file):

```
# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard] (n_expert %d, total %d; absolute offsets in <src>, or in the named shard beside it)
```

v4, written only when at least one layer has its three roles in different shards (Unsloth's layer 11, whose down is in
shard 2 and gate/up in shard 3; no Mac file):

```
# strata native experts v4: layer gu_type d_type offset blob_bytes gate_off up_off down_off [shard | gate,up,down] (n_expert %d, total %d; absolute offsets in <src>, or in the named shard beside it - per role where the column is gate,up,down)
```

`<src>` is the file name of the `--gguf` shard (shard 1). One line per layer, space-separated:

| Column | Meaning |
| --- | --- |
| `layer` | 0-47 |
| `gu_type`, `d_type` | ggml type ids of gate/up and down |
| `offset` | Cumulative `blob × n_expert` of the earlier layers: the layer's offset in `experts.bin`, if one is written |
| `blob_bytes` | `blob(l)` of section 3.4 |
| `gate_off`, `up_off`, `down_off` | **Absolute** file offsets of the role tensors: that role's shard `data_start` + the tensor's table offset |
| shard (optional) | v3: the file name of the shard that holds all three roles, absent when it is the `--gguf` shard. v4: `gate,up,down`, three names, an empty field meaning the `--gguf` shard. |

The Mac reader **must**:
1. read the version from the `# strata native experts v` tag and refuse versions above 4 with upstream's message;
   accept v3 and v4;
2. take `n_expert` from the header's `(n_expert N` and check it against `qwen4exp.expert_count` and the routers'
   `ne1` (the header wins over the compile-time 512, as upstream);
3. check that `blob_bytes` equals the size its two types give for H = 2560, FF = 640 (upstream `native_fmt`);
4. check that the `offset` column is contiguous from 0 (upstream: "is missing or not contiguous");
5. **require** the three offset columns: the Mac reads experts from the GGUF in place, so a line without them (a
   pre-v2 pack that only has `experts.bin`) is refused, as is a pack without `native_experts.txt` (a canonical pack;
   [06](06-engine-host-port.md), section 6.1);
6. resolve each role's shard by name in the pack's source folder and check that `role_abs + n_expert × slab` lies
   inside that shard;
7. check that `gu_type` and `d_type` equal the types of the GGUF tensors at those offsets (the GGUF is the authority).

`iq_pack.py` itself checks that every layer has the three tensors, that their shapes are `[*, *, n_expert]` of whole
blocks, that gate and up have the same type and size, and that all routers agree on `n_expert`.

`experts.bin` (one blob per expert, gate|up|down back to back, at `offset + e × blob`) and its
`experts.bin.src.json` are written only with `--experts-bin` (PC low-RAM mode). Upstream's engine reads an existing
`experts.bin` instead of the GGUF. The Mac setup never passes the flag, and the Mac engine ignores an `experts.bin`
in a native pack and logs that it does.

### 3.6 Expert blobs

**Native blob.** The logical unit of one expert: gate slab, up slab, down slab, with `up_off = slab_gu`,
`down_off = 2 × slab_gu` and `blob = 2 × slab_gu + slab_d` ([audit B.4.2](audit/A-model-and-kernels.md)). In full mode
the three slabs stay at their three GGUF addresses. In tiered mode a resident slot packs them back to back in this
order ([05](05-memory-and-residency.md), section 5.2). The kernels take three addresses per expert either way.

Blob sizes of every combination in the catalogue:

| gate/up + down | Blob (B) | | gate/up + down | Blob (B) |
| --- | ---: | --- | --- | ---: |
| IQ1_M + Q2_0 | 1,177,600 | | IQ2_XXS + IQ4_NL | 1,766,400 |
| IQ2_XXS + Q2_0 | 1,305,600 | | IQ2_XS + IQ4_NL | 1,868,800 |
| Q2_0 + Q2_0 | 1,382,400 | | IQ2_S + IQ4_NL | 1,971,200 |
| IQ2_XS + Q2_0 | 1,408,000 | | IQ3_XXS + IQ4_NL | 2,176,000 |
| IQ2_S + Q2_0 | 1,510,400 | | IQ3_S + IQ4_NL | 2,329,600 |
| IQ3_XXS + Q2_0 | 1,715,200 | | IQ4_XS + IQ4_NL | 2,662,400 |
| IQ3_S + Q2_0 | 1,868,800 | | | |

**S2 blob** (MTP experts only, `include/strata/kernels/cpu/expert.hpp:44-50`): Strata's canonical Q2_0 form, 1,382,400 B,
with the codes and scales of gate and up interleaved by row.

| Offset (B) | Size (B) | Content |
| ---: | ---: | --- |
| 0 | 819,200 | 1280 rows × 640 B of 2-bit codes; row slot 2r is gate row r, 2r + 1 is up row r |
| 819,200 | 409,600 | 2560 down rows × 160 B of codes |
| 1,228,800 | 102,400 | 1280 rows × 40 fp16 scales (one per 64) |
| 1,331,200 | 51,200 | 2560 rows × 10 fp16 scales |

An interleaving bug once wrote correctly decoded rows to the wrong output slot with plausible-looking results
([audit B.4.1](audit/A-model-and-kernels.md)); `bench/micro/moe_hit_parity.cu` is the check, and its Metal
counterpart must stay in the suite ([09](09-testing-and-validation.md)).

### 3.7 Float tensors: the form the engine reads

The engine reads every float tensor in one fixed form, `iq_pack.py` `FORM` (`tools/iq_pack.py:73-88`):

| Form | Tensors |
| --- | --- |
| BF16 | `ffn_gate_inp`, `ffn_gate_inp_shexp`, `hc_{attn,ffn}_{down,up,inject}`, `output_hc_{down,up}`, `indexer.{k,q}_proj`, `ssm_alpha`, `ssm_beta`, `ple_key` (unless Q2_0 or Q8_0, which are served natively), `ple_value` |
| F16 | `ple_conv1d` |
| F32 | `attn_{q,k}_norm`, `hc_{attn,ffn}_norm`, `output_hc_norm`, `indexer.{q,k}_norm`, `ple_norm_{conv,key,query}`, `ssm_a`, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm` |

`convert()` (`tools/iq_pack.py:226-271`):
- If the stored type equals the form, the raw bytes are copied unchanged.
- F32 to BF16 is done only when the low 16 bits of every value are zero (the values are BF16's) and every value is
  finite; it is then exact. Otherwise it needs `--compat-bf16` (round to nearest even), which the Mac does not use.
- F32 to F16 is written as index kind 3: the F32 bytes go into `dense.bin` and the loader narrows them to F16 (nearest
  even).
- F16 or BF16 to F32 widens exactly.
- A quantised tensor whose form is BF16 (an Unsloth case) needs `--compat-bf16`.

In the catalogue, the only conversion is **Swift's routers**: all 96 `ffn_gate_inp` and `ffn_gate_inp_shexp` tensors
are F32 with BF16 values, narrowed exactly. Every other float tensor of every catalogue file already has its form.

**Norm values.** The F32 norm tensors hold the final multiplier. The converter that made the GGUF already added the 1
of the `(1 + w)` RMSNorm convention: `output_hc_norm` ranges from 0.63 to 13.4 with a mean of 3.75. The engine
multiplies by the stored value and must not add 1. The MTP runtime does the same fold when it converts the checkpoint,
which stores `w` (section 4.4).

### 3.8 Per-file maps

Routed experts (gate/up type + down type, by layer):

| File | Combinations |
| --- | --- |
| Q2_0 | Q2_0 + Q2_0 in all 48 layers |
| IQ2_XS (ISTA and Swift) | IQ2_S + Q2_0 in 34 layers; IQ2_XXS + Q2_0 in 1, 4, 9, 10, 11, 14, 18, 19, 27, 29, 30; IQ1_M + Q2_0 in 8, 13, 37 |
| IQ3_XXS (ISTA and Swift) | IQ2_XS + IQ4_NL: 0, 2, 4, 39. IQ2_XXS + Q2_0: 1, 3, 6, 7, 11, 13, 16, 27. IQ2_XS + Q2_0: 5, 9, 10, 14, 19, 25. IQ3_S + Q2_0: 8, 18, 20, 22, 23, 24, 33. IQ2_S + Q2_0: 12, 15, 17, 21, 26, 40. IQ3_S + IQ4_NL: 28, 30, 32, 34, 37, 43. IQ2_S + IQ4_NL: 29, 38, 42, 46. IQ2_XXS + IQ4_NL: 31. IQ3_XXS + IQ4_NL: 35, 44, 47. IQ3_XXS + Q2_0: 36, 41, 45. |
| IQ3_S and Coder | IQ3_XXS + IQ4_NL: 0, 7, 14, 16, 18, 22, 31, 33, 37, 38, 41, 42, 43, 46. IQ2_S + Q2_0: 1, 8, 10, 13, 36. IQ2_S + IQ4_NL: 2, 3, 4, 5, 6, 9, 11, 15, 19, 23, 24, 26, 32, 34, 39. IQ3_XXS + Q2_0: 12, 20, 30. IQ3_S + IQ4_NL: 17, 25, 27, 28, 29, 35, 40, 44, 45. IQ3_S + Q2_0: 21. IQ4_XS + IQ4_NL: 47. |

| File | Expert bytes, all layers | Blob min / mean / max (B) | Experts read per token (top-10, all layers) |
| --- | ---: | --- | ---: |
| Q2_0 | 33,973,862,400 | 1,382,400 (every layer) | 663,552,000 |
| IQ2_XS | 35,454,976,000 | 1,177,600 / 1,442,666.7 / 1,510,400 | 692,480,000 |
| IQ3_XXS | 42,912,972,800 | 1,305,600 / 1,746,133.3 / 2,329,600 | 838,144,000 |
| IQ3_S | 50,292,326,400 | 1,510,400 / 2,046,400 / 2,662,400 | 982,272,000 |
| Coder (256 experts) | 25,146,163,200 | as IQ3_S | 982,272,000 |

Quantised dense tensors (count of layers per type):

| Tensor | Q2_0 file | IQ2_XS | IQ3_XXS | IQ3_S and Coder |
| --- | --- | --- | --- | --- |
| `attn_qkv` (36) | IQ4_XS 13, Q3_K 18, Q4_K 4, Q2_0 1 | IQ4_XS 35, IQ3_S 1 | Q6_K 14, IQ4_XS 11, Q4_K 8, Q5_K 3 | Q6_K 22, Q5_K 10, Q4_K 2, IQ4_XS 2 |
| `attn_gate` (36) | Q3_K 25, IQ4_XS 4, Q2_0 4, Q4_K 3 | IQ4_XS 22, IQ3_S 11, Q6_K 3 | IQ4_XS 11, IQ3_S 10, Q4_K 7, Q6_K 5, Q5_K 3 | Q6_K 14, Q4_K 10, IQ4_XS 8, Q5_K 4 |
| `ssm_out` (36) | IQ4_XS 17, Q4_K 11, Q5_K 3, Q3_K 3, Q6_K 2 | IQ4_XS 29, IQ3_S 6, Q6_K 1 | Q6_K 19, IQ4_XS 6, Q4_K 4, Q5_K 4, IQ3_S 3 | Q6_K 29, Q5_K 4, Q4_K 3 |
| `attn_q` (12) | Q3_K 6, IQ4_XS 4, Q2_0 2 | IQ4_XS 11, IQ3_S 1 | IQ4_XS 4, Q6_K 4, Q4_K 3, IQ3_S 1 | Q6_K 5, IQ4_XS 3, Q5_K 3, Q4_K 1 |
| `attn_k` (12) | IQ4_XS 4, Q3_K 3, Q4_K 2, Q5_K 2, Q6_K 1 | IQ4_XS 8, Q6_K 3, IQ3_S 1 | Q6_K 6, IQ4_XS 2, Q4_K 2, Q5_K 1, IQ3_S 1 | Q6_K 11, IQ4_XS 1 |
| `attn_v` (12) | Q5_K 3, IQ4_XS 3, Q6_K 3, Q4_K 2, Q3_K 1 | IQ4_XS 7, Q6_K 5 | Q6_K 9, IQ4_XS 2, Q4_K 1 | Q6_K 10, Q5_K 2 |
| `attn_output` (12) | Q4_K 4, Q6_K 4, Q5_K 4 | IQ4_XS 10, IQ3_S 1, Q6_K 1 | Q6_K 10, Q5_K 1, Q4_K 1 | Q6_K 11, Q4_K 1 |
| `ffn_gate_shexp` (48) | Q2_0 21, Q3_K 13, IQ4_XS 6, Q4_K 6, Q6_K 1, Q5_K 1 | IQ4_XS 31, IQ3_S 9, Q6_K 8 | IQ4_XS 21, IQ3_S 13, Q4_K 12, Q5_K 1, Q6_K 1 | Q4_K 16, IQ4_XS 15, Q6_K 13, Q5_K 4 |
| `ffn_up_shexp` (48) | Q3_K 21, Q2_0 13, Q4_K 6, IQ4_XS 5, Q5_K 2, Q6_K 1 | IQ4_XS 26, IQ3_S 14, Q6_K 8 | IQ3_S 27, IQ4_XS 10, Q4_K 7, Q6_K 3, Q5_K 1 | Q4_K 14, IQ4_XS 13, Q6_K 13, Q5_K 8 |
| `ffn_down_shexp` (48) | Q2_0 16, Q4_0 16, Q5_0 7, IQ4_NL 7, Q8_0 2 | IQ4_NL 36, Q8_0 7, Q2_0 5 | IQ4_NL 40, Q2_0 8 | IQ4_NL 47, Q8_0 1 |
| `ple_key` | Q2_0 | BF16 | BF16 | BF16 |
| `token_embd` | Q3_K | IQ4_XS | IQ3_S | IQ4_XS |
| `output` | Q5_K | IQ4_XS | Q5_K | Q6_K |

- Coder has exactly the dense types and expert combinations of IQ3_S; only `n_expert` (256) and the router shapes
  differ.
- Swift IQ2_XS and IQ3_XXS have exactly the types of the ISTA file of the same name, except their routers (F32).
- Every float tensor has the same type in every file, except `ple_key` and Swift's routers.
- `ple_key` as Q2_0 is served natively (`NATIVE_PLE_KEY = {Q2_0, Q8_0}`); as BF16 it is read through the BF16 GEMV.

### 3.9 Swift's shard layout

Swift splits its files by size, not by role. The engine must find every tensor by name across both shards and never
assume which shard holds what.

| File | Shard 1 | Shard 2 |
| --- | --- | --- |
| Swift IQ2_XS | The 5 global tensors, the PLE table (at table offset 350,863,360, right after the head tensors), all of layers 0-12, and `attn_gate` and `attn_qkv` of layer 13 | The other 23 tensors of layer 13 (its experts included) and layers 14-47 |
| Swift IQ3_XXS | The 5 global tensors, the PLE table (at table offset 450,191,360), all of layers 0-11, and `attn_gate` and `attn_qkv` of layer 12 | The other tensors of layer 12 and layers 13-47 |

- So the experts of layers 13-47 (IQ2_XS) and 12-47 (IQ3_XXS) are in shard 2. Their `native_experts.txt` lines carry
  the shard 2 file name. No layer has its experts split, so both packs are v3.
- The PLE table is in shard 1 (upstream `setup.py:3119`: "shard 2 (original) or 1 (Swift)"). The engine finds it by
  the tensor name `per_layer_token_embd.weight`, never by shard number.
- In full mode the plan wraps both shards as windows ([05](05-memory-and-residency.md), section 4.2); the PLE tensor
  has no window mapping in shard 1. Outward 16 KiB alignment may include up to one page of edge padding from the PLE
  range when an adjacent tensor ends off a page boundary.
- Swift's routers are the only tensors that differ from the same-named ISTA file. Swift's metadata adds
  `qwen4exp.attention.recurrent_layers` and licence keys.

---

## 4. Bytes: file totals, per pass and per token

### 4.1 File totals

"Tensor bytes" is the sum of the tensor table's sizes, which is the file size minus the header (`data_start`) and the
alignment padding. Experts are the three `ffn_*_exps` tensors of all 48 layers. "Other" is every tensor except the
experts and the PLE table.

| File | Shard 1 tensor bytes | Experts | Other | Other (GiB) |
| --- | ---: | ---: | ---: | ---: |
| Q2_0 | 37,612,715,520 (35.03 GiB) | 33,973,862,400 (31.64 GiB) | 3,638,853,120 | 3.39 |
| IQ2_XS | 39,214,929,920 (36.52 GiB) | 35,454,976,000 (33.02 GiB) | 3,759,953,920 | 3.50 |
| IQ3_XXS | 47,028,835,840 (43.80 GiB) | 42,912,972,800 (39.97 GiB) | 4,115,863,040 | 3.83 |
| IQ3_S | 54,806,499,840 (51.04 GiB) | 50,292,326,400 (46.84 GiB) | 4,514,173,440 | 4.20 |
| Coder | 29,597,422,080 (27.56 GiB) | 25,146,163,200 (23.42 GiB) | 4,451,258,880 | 4.15 |
| Swift IQ2_XS (both shards, PLE excluded) | 39,341,004,800 | 35,454,976,000 | 3,886,028,800 | 3.62 |
| Swift IQ3_XXS (both shards, PLE excluded) | 47,154,910,720 | 42,912,972,800 | 4,241,937,920 | 3.95 |

The PLE table is 28,800,138,240 B (26.82 GiB) in every file. Swift's "other" is larger than ISTA's by 126,074,880 B,
the F32 routers.

### 4.2 Dense bytes per forward pass

A forward pass reads every non-expert tensor once except `token_embd` and the PLE table, whatever its number of
tokens T: decode (T = 1), a verify window (T ≤ 8) and a prefill chunk all read these bytes once. (Prefill GEMMs may
read them more than once from memory if a kernel tiles the token dimension; doc 04 decides the tiling.)

| Group | Tensors | Q2_0 | IQ2_XS | IQ3_XXS | IQ3_S |
| --- | --- | ---: | ---: | ---: | ---: |
| GDN projections | `attn_qkv`, `attn_gate`, `ssm_out` × 36 | 1,014,661,120 | 1,091,338,240 | 1,338,859,520 | 1,533,829,120 |
| GDN small | `ssm_alpha`, `ssm_beta`, `ssm_conv1d`, `ssm_a`, `ssm_dt.bias`, `ssm_norm` × 36 | 23,625,216 | 23,625,216 | 23,625,216 | 23,625,216 |
| QSA projections | `attn_q`, `attn_k`, `attn_v`, `attn_output` × 12 | 314,777,600 | 320,174,080 | 407,941,120 | 437,585,920 |
| Indexer and QSA norms | `indexer.*`, `attn_{q,k}_norm` × 12 | 39,358,464 | 39,358,464 | 39,358,464 | 39,358,464 |
| GR mixers | `hc_{attn,ffn}_{down,up,inject,norm}` × 48 | 1,270,087,680 | 1,270,087,680 | 1,270,087,680 | 1,270,087,680 |
| Routers | `ffn_gate_inp`, `ffn_gate_inp_shexp` × 48 | 126,074,880 | 126,074,880 | 126,074,880 | 126,074,880 |
| Shared experts | `ffn_{gate,up,down}_shexp` × 48 | 106,240,000 | 134,976,000 | 120,832,000 | 145,536,000 |
| PLE block | layer 1's six `ple_*` tensors | 20,684,800 | 65,740,800 | 65,740,800 | 65,740,800 |
| Head | `output`, `output_hc_{down,up,norm}` | 450,191,360 | 350,863,360 | 450,191,360 | 534,620,160 |
| **Total per pass** | | **3,365,701,120** | **3,422,238,720** | **3,842,711,040** | **4,176,458,240** |

- **Coder:** 4,113,543,680 B, which is IQ3_S's total with routers of 63,160,320 B (256 experts).
- **Swift:** the routers read by the engine are the BF16 copies in `dense.bin` (section 3.7), so a Swift file reads
  exactly the bytes of the ISTA file of the same name. Reading the F32 originals from the GGUF would add
  126,074,880 B per pass.
- **The GR mixers are 38% of the Q2_0 pass** and are BF16 in every file: 26.5 MB per layer, split between its attention and FFN
  mixers. Their GEMV efficiency matters as much as that of the quantised projections
  ([04](04-metal-kernels.md)).
- Embedding and PLE: per token, one `token_embd` row (1100 B for Q3_K and IQ3_S, 1360 B for IQ4_XS) and 16 PLE rows of
  90 B (1440 B). Both are negligible.

### 4.3 Bytes per decoded token

`bytes(T) = dense pass + Σ_l |∪_t selected(l, t)| × blob(l) + T × (embedding row + 1440)`

| File | Dense pass | Top-10 experts (T = 1) | Total per decoded token | 2 × active parameters |
| --- | ---: | ---: | ---: | ---: |
| Q2_0 | 3.366 GB | 0.664 GB | 4.029 GB | 13.3 GFLOP |
| IQ2_XS (and Swift) | 3.422 GB | 0.692 GB | 4.115 GB | 13.3 GFLOP |
| IQ3_XXS (and Swift) | 3.843 GB | 0.838 GB | 4.681 GB | 13.3 GFLOP |
| IQ3_S | 4.176 GB | 0.982 GB | 5.159 GB | 13.3 GFLOP |
| Coder | 4.114 GB | 0.982 GB | 5.096 GB | 13.3 GFLOP |

- Decode is bandwidth-bound: a token costs about 13.3 GFLOP of weight arithmetic plus attention, against 4-5 GB of
  reads, which is about 3 FLOP per byte. The speed ceiling is `memory bandwidth / bytes per token`;
  [appendix-a-measurements.md](appendix-a-measurements.md), section A.7, turns this into tokens/s for each Mac.
- The top-10 column assumes ten distinct experts per layer, which is always the case for one token.
- A verify window of T tokens reads the dense pass once and the union of the experts the T tokens select: at most
  `min(10T, n_expert)` per layer. For Q2_0 the upper bound is 4.69 GB at T = 2, 6.02 GB at T = 4 and 8.67 GB at T = 8;
  the real union is smaller because tokens of one window share experts. Upstream's setup drafts at most 3 tokens
  (T ≤ 4; [K4 section 3.4](audit/K4-ple-mtp-head-sampler.md)).
- Attention reads its KV cache in addition: for QSA, at most 2051 selected cells per layer; for MTP, the last 32,768
  cells. [05](05-memory-and-residency.md), section 3, sizes the caches.

### 4.4 The MTP draft layer's files

The draft layer is not in the GGUF. setup builds it in three steps (`setup.py:3097-3111`):

1. `mtp_fetch.py fetch --out <data>/mtp` fetches the 31 `mtp.*` BF16 tensors (5,214,301,696 B) from 28 shards of
   `Qwen/Qwen3.8-Flash-Next` at the pinned revision. The files stay on disk: setup uses them to check the build
   (`mtp_corrupt`, #327).
2. `mtp_pack.py --experts q2_0` writes `<data>/mtp/mtp-q2_0.gguf` (about 0.9 GB). Its experts are quantised to Q2_0 with
   a per-block scale that minimises the squared error over the grid {-1, 0, 1, 2} × d. The dense tensors stay BF16, and
   the 1-D norms become F32.
3. `mtp_rt.py` writes `<data>/mtp/rt/`:

| File | Bytes | Content |
| --- | ---: | --- |
| `experts.bin` | 707,788,800 | 512 S2 blobs of 1,382,400 B (section 3.6), expert e at `e × 1,382,400`. A lossless relayout of the GGUF's Q2_0 blocks. |
| `dense.bin` | 116,099,072 | 29 tensors at 256-B aligned offsets (every size here is a multiple of 256, so there is no padding) |
| `dense.txt` | small | One line per tensor: `name kind rows cols offset bytes`, kind `q8_0`, `bf16` or `f32`; names without the `mtp.` and `layers.0.` prefixes |
| `draft_vocab.bin` | 425,196 (cjk) | Copied by setup (section 2.3); refreshed on every setup run |

`dense.bin` holds:
- **Q8_0, 10 tensors, 73,809,920 B** (ggml reference rounding): `fc_embedding` and `fc_hidden` [2560 → 2560],
  `self_attn.q_proj` [2560 → 12288], `k_proj` and `v_proj` [2560 → 512], `o_proj` [6144 → 2560],
  `self_attn.indexer.index_qk_proj` [2560 → 640], `mlp.shared_expert.{gate,up}_proj` [2560 → 640] and
  `down_proj` [640 → 2560];
- **BF16, 10 tensors, 42,112,000 B:** the three hyper-connection groups (`hyper_connection_mixer.*`,
  `attn_hyper_connection.*`, `mlp_hyper_connection.*`: `input_mix_weight_down`, `input_mix_weight_up` and, for the
  two block groups, `block_inject_weight`), the router `mlp.gate.weight` [2560 → 512] and `mlp.shared_expert_gate`;
- **F32, 9 tensors, 177,152 B:** the norms, stored as `1 + w` (`mtp_rt.py:85-86`): the three `hc_norm` [10240],
  `pre_fc_norm_embedding` [2560], `pre_fc_norm_hidden` [10240], `self_attn.{q,k}_norm` [256] and
  `self_attn.indexer.{q,k}_layernorm` [128].

The draft layer uses dense attention over the last 32,768 cells and no indexer, so the indexer tensors are loaded
and unused. It always has 512 experts and the main model's head and embedding: with Coder (256 experts) it routes
over its own 512 ([K4 section 3](audit/K4-ple-mtp-head-sampler.md)). A draft step reads `dense.bin` (0.116 GB),
10 experts (0.014 GB), one embedding row, and the head: the full head (0.34-0.52 GB) or, with a draft vocabulary, its
subset (106,299 rows for cjk: 0.187 GB for a Q5_K head).

The fetch must keep upstream's checks: each range response must be `206` with a matching `Content-Range`, and each
tensor's SHA-256 must match the pinned value ([07](07-server-setup-port.md)).

### 4.5 The native pack

`iq_pack.py --gguf <shard 1> --out <pack>` writes the pack (`tools/iq_pack.py:274-376`, `:556-575`). The Mac setup
runs it unchanged, with `--compat-bf16` never passed and `--base` and `--experts-bin` never passed.

| File | Content | Mac use |
| --- | --- | --- |
| `index.txt` | Header `# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from <shard 1>`, then `# align 64 pool <bytes> tensors <n>`, then one row per non-expert tensor (the PLE table excluded) | Read |
| `dense.bin` | Every float tensor in its engine form (section 3.7), 64-B aligned | Read only the rows that `conversions.json` lists (below) |
| `conversions.json` | Schema 1: `tool`, `compat_bf16`, `source_shards` [{`name`, `size`}], `tensors` [{`name`, `shard`, `src_type`, `dst_type`, `method`, `exact`, `max_abs_err`, `source_sha256`}] for every tensor whose stored type differs from its form | Read |
| `native_experts.txt` | Section 3.5; written last | Read; required |
| `tokenizer/` | `vocab.json`, `chat_template.jinja` and the rest, from `strata_tokenizer.py` | Read by the server |
| `extra.bin`, `experts.bin`, `experts.bin.src.json`, `compat-bf16.json` | Only with `--base`, `--experts-bin` or `--compat-bf16` | Absent; ignored if present |

An `index.txt` row has 19 space-separated fields; upstream refuses a row that does not parse to 19
(`src/core/weights.cpp:166-177`):

| # | Field (`IndexRow`, `src/core/weights.cpp:30-41`) | Float row | Native row |
| ---: | --- | --- | --- |
| 1 | `name` | tensor name | tensor name |
| 2 | `file` | 0 (`dense.bin`) | 0 |
| 3 | `kind` | 2 = F32, 3 = F32 to be narrowed to F16, 4 = BF16, 5 = F16 | 0 |
| 4, 5 | `src_off`, `src_bytes` | offset and size in `dense.bin` | 0, 0 |
| 6 | `dst_off` | offset in the pool, set by `write_index` | the running pool offset (a native row adds nothing to it) |
| 7 | `dst_bytes` | size in engine form (kind 3: half of `src_bytes`) | 0 |
| 8, 9 | `ne0`, `ne1` | shape (`ne1` 0 for a vector) | shape |
| 10 | `code_bits` | 0 | 8 |
| 11 | `code_bias` | 0 | 0 |
| 12 | `group_elems` | 1 | 32 |
| 13-19 | `codebook`, `has_offset`, `codes_bytes`, `scales_bytes`, `offset_bytes`, `scales_fp16`, `act_kind` | 0 | 0 |

A native row (`code_bits 8`, `dst_bytes 0`) only says "read this tensor from the GGUF"; its real type comes from the
GGUF tensor table ([06](06-engine-host-port.md), section 6.2). The Mac reader **must** check every field it does not
use against these values and refuse a row that differs: the Mac only accepts native packs.

`dense.bin` sizes: 1,485,606,400 B for Q2_0 (1.38 GiB); 1,538,035,200 B for every IQ file, Swift included (the extra
52,428,800 B are the BF16 `ple_key`); 1,475,120,640 B for Coder (half-size routers). Swift's `dense.bin` is the same
size as ISTA's because its routers are narrowed to BF16.

**Serving rule.** The Mac engine serves a float tensor from the GGUF when `conversions.json` does not list it, because
the GGUF bytes are then identical to the `dense.bin` bytes (a raw copy, section 3.7). It reads `dense.bin` only for
the listed tensors: none for ISTA and Coder files, the 96 routers (126,074,880 B) for Swift. The standalone packer
checks every recorded conversion payload and a deterministic sample of up to 32 unconverted tensors. The model CTest
compares every unconverted float and recomputes every conversion payload for all seven catalogue identities. This
avoids a second copy of 1.4 GiB in memory ([05](05-memory-and-residency.md), section 3.2).

---

## 5. The activation contract

The weights are fixed by the files; the activations that meet them are fixed by the kernels. A kernel that quantises
its activation differently from upstream produces plausible output that is 0.6-1.4% off, which only a parity test
detects ([audit B.2](audit/A-model-and-kernels.md); [K3 section 2](audit/K3-quant-and-matvec.md)). The Mac engine
**must** use, per path, the activation form of upstream's GPU path, so that upstream's CUDA dumps are the oracle:

| Path | Weight formats | Activation | Upstream quantiser |
| --- | --- | --- | --- |
| Dense GEMV, shared experts, `ple_key` (Q2_0), head | every quantised dense format | **Q8_1**: 32 int8 per block, `half2(d, Σx)` with `d = amax / 127`, round half away from zero, `Σx` the sum of the float inputs | `native_quantize_q8_1` (`native_mmvq.cu:141`), `quantize_q8_1_rows` (`iq_kernels.cu:979`) |
| Routed experts of the main model | every expert format, native slabs | **Q8_1** as above | `native_expert_grouped` path |
| MTP routed experts | Q2_0, S2 blob | **Q8_0 bytes with FP32 scales**: `s = amax / 127`, `inv = 1 / s`, `(int)(t + copysign(0.5, t))`, clamp ±127 | `quantize_q8_0_scaled` (`quantize_act.cu:100`, `:245`) |
| MTP dense (Q8_0) | Q8_0 | **Q8_1** | `native_mmvq` (`mtp.cpp:440-508`) |
| BF16 GEMV (routers, GR mixers, indexer, `ssm_alpha/beta`, `ple_value`, `ple_key` as BF16) | BF16 | **F32** | none (`bf16_f32_mmvf`) |

- The legacy CPU quantisers (`act_quant_q8_1`, ActQ), the S-form contracts (Q8_0 by `quantize_q8_0`, Q8_K) and
  ggml-cpu's `from_float` are not used: the Mac has no CPU expert path and no canonical pack.
- Q2_0 meets two contracts: Q8_1 in the main model's experts and dense tensors, Q8_0-scaled in the MTP experts.
  Each kernel takes the contract of its path; the two must not be unified, or the MTP drafts stop matching upstream's.
- The bit-level rules for reproducing these quantisers in MSL (safe math, correctly rounded division, `round` vs
  `rint`, exact integer dots, custom BF16/F16 rounding) are in [audit B.2.4](audit/A-model-and-kernels.md) and are
  normative for [04](04-metal-kernels.md).

---

## 6. The forward pass

This section is the index of the forward pass: per stage, the rules a Metal kernel **must** keep and the place in
the audits that holds the full arithmetic. The audit text is normative where this section points to it. Kernel
shapes, tiling and fusion belong to [04](04-metal-kernels.md).

Two parity contracts apply, as upstream ([audit E.0](audit/A-model-and-kernels.md)):
- **bitwise within the Metal build** where the order is fixed by construction: quantisers, KV quantisation, router
  ids, top-k selection, sampler tokens, decode ≡ verify, multi-token ≡ single-token;
- **tolerance against an FP64/FP32 reference** where the order is free: GEMV, GEMM, attention.

There is no cross-backend bitwise claim: Metal and CUDA outputs are compared with tolerances and KL (section 9).

### 6.1 Token lifecycle

Per token ([audit A.2](audit/A-model-and-kernels.md); `src/core/layer.cpp:1142-1314`):

1. Host: PLE hash and row decode for the token (section 6.8); stage the token, the step buffer and the positions.
2. Embed the token (section 6.9) and copy the row into the 4 streams: `R[c] = e` for c in 0..3.
3. For each layer l in 0..47:
   1. if l == 1: materialise the pending FFN write of layer 0, then run the PLE block on R;
   2. attention read: `mixed, inject = gr_read(hc_attn_*, R)`, folding the pending FFN write of layer l - 1;
   3. block: QSA if `l % 4 == 3`, else GDN → `block_out`;
   4. FFN read: `mixed, inject2 = gr_read(hc_ffn_*, R)`, folding the attention write `(block_out, inject)`;
   5. router → 10 ids and weights; shared expert; 10 routed experts; combine → `block_out`;
   6. FFN write: at l == 47, write now; with a control vector, write then apply the vector; otherwise leave it
      pending for the next layer's step 2.
4. Head mix `gr_read(output_hc_*, R)` without inject, then the head, then argmax or the sampler.

Rules:
- The order inside a layer is fixed: attention read, block, attention write, FFN read, router, shared expert,
  experts, combine, FFN write. Fusion may move a write into the next read, never reorder the arithmetic.
- The fused read refuses `apply && inj_prev == inject_out`: the attention read writes `inject` and folds `inject2`,
  the FFN read writes `inject2` and folds `inject`. **Two inject buffers.**
- Production decode on CUDA requires one configuration (`layer_verify_compatible`, `layer.cpp:448-460`): native BF16
  projections, fused GR, fused native GDN, fast attention, fast selection and the native indexer. The Mac implements
  only this configuration.

### 6.2 GR hyper-connections

[audit A.3](audit/A-model-and-kernels.md); [K1 section 3](audit/K1-gr-router-shared-rope-cvec.md).

```
gr_read(R[4][2560], g = hc_*_norm (stored multiplier), W_down [10240 → 320], W_up [320 → 10240], W_inject [10240 → 4]):
  rs[c]     = 1 / sqrt(mean_d R[c,d]^2 + 1e-6)
  xn[c,d]   = R[c,d] * rs[c] * g[c*2560 + d]
  lo[k]     = silu((W_down[k,:] · xn) / 4)
  gate[i]   = sigmoid(W_up[i,:] · lo)
  mixed[d]  = (1/4) Σ_c xn[c,d] * gate[c*2560 + d]
  inject[c] = W_inject[c,:] · xn            (not for the head)
gr_write(R, block_out, inject): R[c,d] += block_out[d] * 2 * sigmoid(inject[c] / 4)
```

- The `/hc` is inside the SiLU. The norm weight is used as stored (section 3.7).
- Multi-token ≡ single-token bitwise: the 8-element chunk structure of the dot products (eight FMAs from 0, low half
  first, then `acc += chunk`; chunk `j = lane + 32q` ascending, then an xor butterfly) must be kept by any tiling.
- The three write flavours (canonical, native `post`, fused) are not interchangeable bitwise. The control-vector
  write uses the fused flavour so that "vector loaded, off" ≡ stock.

### 6.3 GDN layers

[audit A.4](audit/A-model-and-kernels.md); [K2 section 2](audit/K2-qsa-kv-gdn.md).

- `qkv = attn_qkv · x` [10240]; causal 4-tap conv per channel with the 3-row history, then SiLU; L2 norm of each q and
  k head with **eps added to the sum** of squares.
- `gate = softplus(alpha_h + dt_bias_h) * ssm_a_h` with softplus `v > 20 ? v : log1p(exp(v))`; `beta = sigmoid(beta_h)`;
  `g = exp(gate)`.
- Delta rule per v-head h with q/k head `h % 16` (modulo, not `h / 3`; the GGUF orders the v heads for this):
  `kv = S·k` from the old state; `delta = (v - g·kv)·beta`; `S = fma(g, S, k·delta)`; `o = (S·q) / sqrt(128)`
  (the q scale is applied to the output).
- Output: `y = o * rsqrt(mean(o^2) + 1e-6) * ssm_norm * sigmoid(z)` with `z = attn_gate · x` (eps on the **mean**
  here), then `ssm_out`.
- State `float[128 i][48 h][128 j]` (j fastest) plus the conv history `float[10240][3]`; the reduction
  `kv = red0 + red1 + red2 + red3` is part of the bitwise contract.
- Verify computes T tokens without writing the state; commit replays the kept tokens (section 8).

### 6.4 QSA layers, indexer and KV

[audit A.5](audit/A-model-and-kernels.md); [K2 sections 1 and 3](audit/K2-qsa-kv-gdn.md).

- Projections: `attn_q` gives, per head, 256 query values then 256 gate values; K and V are 2 heads of 256.
- Q and K: RMSNorm with `attn_{q,k}_norm`, then NEOX RoPE on the first 64 dims (section 6.7).
- Indexer: raw key `indexer.k_proj · x` (before any norm), FP16-rounded into a 3-row tail; on block completion the
  pooled key is the FP16-rounded sum `((t0 + t1) + t2) + t3`, then `fma(0.25, sum, 0)`, then RMS (`indexer.k_norm`),
  then RoPE at the block's first cell. The `dead` key is written only at position 0, with angle 0. Query: `indexer.q_proj · x` [4 × 128], RMS (`indexer.q_norm`), RoPE.
- Selection (must be exact, [K2 section 1.3](audit/K2-qsa-kv-gdn.md)): block score `Σ_h relu(key_b · q_h)`, no
  softmax, no scale; the tail block uses the `dead` key plus 1e9 when `n_kv % 4 != 0`; identity when
  `n_kv ≤ 2051`; otherwise a cell-weighted radix select, ties in ascending cell order, ids ascending.
- Attention: 12 query heads per KV head; scale 1/16 before the max; `exp(s - m_chunk)`; unnormalised FMA PV in
  ascending cell order over 64-cell chunks; merge with `exp(m_c - M)` and a final division. The same kernel serves
  decode, verify and the MTP window, so decode ≡ verify.
- Gate: `attn *= sigmoid(q_full[h*512 + 256 + d])`, then `attn_output`.
- KV formats fp16, int8, q4 and k8v4, with their exact encodings in [K2 section 3](audit/K2-qsa-kv-gdn.md). Prefill
  attention and scoring may use different kernels and are accepted at about 1e-4 relative, as on CUDA.

### 6.5 Shared expert, routed experts and combine

[audit A.6](audit/A-model-and-kernels.md); [K1 section 5](audit/K1-gr-router-shared-rope-cvec.md);
[K3](audit/K3-quant-and-matvec.md).

- Shared expert: `y = ffn_down_shexp · (silu(ffn_gate_shexp · x) * (ffn_up_shexp · x))`;
  `g = sigmoid(ffn_gate_inp_shexp · x)` (BF16 GEMV); `shared = g * y`, rounded before the combine.
- Routed expert e: `y_e = down_e · (silu(gate_e · x) * (up_e · x))`, activations per section 5, weights at the
  addresses of section 3.4.
- Combine: `bo = parts_0 * w_0`, then `bo = fma(parts_k, w_k, bo)` for k = 1..9 in rank order, then `bo += shared`.
  The order never depends on where an expert's bytes came from (resident, window or tiered slot).

### 6.6 Router

[audit A.6](audit/A-model-and-kernels.md); [K1 section 4](audit/K1-gr-router-shared-rope-cvec.md).

`logits = ffn_gate_inp · mixed` (BF16 GEMV, F32 out); softmax over `n_expert`; 10 rounds of argmax with ties to the
lower id and NaN as `-FLT_MAX`; `w_k = p_k / max(Σ p_k, 2^-14)`.

- Upstream runs its native router only for `n_expert == 512 && k == 10` and the canonical FP64-normalising router
  otherwise, so Coder (256) takes the canonical path on CUDA. Both give the same ids; the weights differ in the last
  bits.
- **The Mac uses one router kernel for both widths:** one simdgroup per token, lane l holding experts `l + 32i`
  (16 per lane for 512, 8 for 256), the native arithmetic with correctly rounded division, and no FP64. Ids must
  match upstream on tie-free inputs; weights are compared with a tolerance (1e-5 relative). `n_expert` is a runtime
  value from the file (section 2.5).

### 6.7 RoPE, YaRN and M-RoPE

[audit A.7](audit/A-model-and-kernels.md); [K1 section 6](audit/K1-gr-router-shared-rope-cvec.md).

- NEOX partial rotation, pairs (i, i + 32) for i < 32, applied to Q (24 × 256), K (2 × 256), the indexer query
  (4 × 128) and the pooled indexer keys; dims 64 and up are copied.
- `theta = mrope_pos(pos, pair) * base^(-2·pair / 64)`, base from `rope.freq_base`; `mrope_pos = tab[pos*3 + pair % 3]`
  with a vision table, else `pos`.
- **The Mac builds one FP64 cos/sin table on the CPU** (`[cell][32]`, as upstream's canonical path and
  `STRATA_ROPE_TABLE=1`) and every path reads it: decode, verify, prefill, MTP and the indexer. This makes decode and
  prefill agree bit for bit and avoids the 0.02 rad error of fast trigonometry at 256K positions. The table is
  `max_context × 32 × 2` floats (64 MiB at 262,144 cells), shared, rebuilt when the context or YaRN settings change, and from the first image cell on for each vision
  request and the text request after it ([06](06-engine-host-port.md), section 5.4).
- YaRN (`rope_scaling.hpp:65-193`, computed in double on the host) applies when the file or the server's settings
  give scaling keys ([07](07-server-setup-port.md), section 4.5.2). No shipped file has them.

### 6.8 PLE

[audit A.9](audit/A-model-and-kernels.md); [K4 section 1](audit/K4-ple-mtp-head-sampler.md).

- Host: 16 rows per token from a u64 multiply/xor hash of (token, previous, the one before): 8 bigram and 8 trigram
  heads, `row = (u32)(mixed % vocab[h] + offset[h])`. EOS 248044 cuts the context. The constants are in
  `include/strata/kernels/ngram.hpp` and equal the file's `qwen4exp.ple.*` keys (section 2.5).
- Rows are read from `per_layer_token_embd.weight`, found by name (shard 2 for ISTA and Coder, shard 1 for Swift),
  with `pread` and `F_NOCACHE` ([05](05-memory-and-residency.md)). An IQ4_NL row is 5 blocks of {f16 d, 16 B}
  with **split halves**: element j uses the low nibble of byte j, element j + 16 the high nibble. The output is
  head-slowest: head h at `[h*160, (h+1)*160)`.
- GPU block, layer 1 only, before the attention read and after the pending layer-0 write: key, query, value, gate,
  conv and history update as in [audit A.9](audit/A-model-and-kernels.md). Use the native postops arithmetic, not
  the legacy FP64 kernels.
- The hash and the row read stay on the host by design, so a recorded command sequence never replays a stale token's
  rows.

### 6.9 Embedding and head

[audit A.10](audit/A-model-and-kernels.md); [K4 section 2](audit/K4-ple-mtp-head-sampler.md).

- Embedding: dequantise row `tok` of `token_embd.weight` (Q3_K, IQ3_S or IQ4_XS; BF16 with `--embd-gguf`) to F32 with
  the ggml decode of that format, then broadcast to the 4 streams. The table is read in place from the GGUF.
- Head: `mixed = gr_read(R, output_hc_norm, output_hc_down, output_hc_up)` without inject, then 248,320 logits:
  a Q8_1 GEMV for every head format (section 5). The decode entry point for a Q5_K head, `native_q5_k_f32`
  (`native_mmvq.cu:1192`), takes F32 but quantises it with `native_quantize_q8_1` before the Q5_K mat-vec, and the
  verify head (`verify.cpp:842-843`) quantises explicitly, so both are Q8_1.
- Argmax ties go to the lowest index. The MTP drafter requires this native head (`mtp.cpp:383`).

### 6.10 Sampler

[audit A.10](audit/A-model-and-kernels.md); [K4 section 4](audit/K4-ple-mtp-head-sampler.md).

- Defaults top_k 20 (cap 64), top_p 0.95, min_p 0, temperature 1. Chain: penalties, top_k, top_p, min_p, temperature,
  inverse-CDF pick.
- Philox: the non-standard 10-round variant (constants `0x9E3779B9` / `0xBB67AE85`, key (i, 0) per round,
  `u = (c0 >> 8) · 2^-24`), counter `pos0 + t`, so a draw depends only on the absolute position and speculative
  acceptance stays exact under sampling.
- **The ≤ 64-candidate tail runs on the CPU in FP64** (upstream runs it on the GPU in FP64, which Metal lacks). The GPU
  part produces the candidates; the 18 `sampler_parity` fixtures must give their exact tokens.

### 6.11 Vision

[audit A.13](audit/A-model-and-kernels.md).

- The encoder is llama.cpp's `mtmd` with the family's `mmproj` (section 2.3), run through ggml-metal.
- The prompt carries `<|image_pad|>` (248056) once per image row; the engine replaces those embedding rows with the
  encoder's rows and fills the M-RoPE table: t = p, h = p + y, w = p + x, and text after an image continues at
  `p + max(nx, ny)`; every cell after the prompt continues the same way. Without vision the table is the identity.

### 6.12 Control vectors

[audit A.8](audit/A-model-and-kernels.md); [K1 section 7](audit/K1-gr-router-shared-rope-cvec.md).

- `direction.<l>` F32 [2560] for 1 ≤ l < 48; `d = Σ scale · src`. Mode 1 (add, default) `R_c += d`; mode 0
  (project) `R_c -= ||d|| · <R_c, d̂> · d̂`.
- Applied after the layer's FFN write, which the control-vector kernel performs itself in the fused flavour.

---

## 7. The MTP draft layer

[audit A.11](audit/A-model-and-kernels.md); [K4 section 3](audit/K4-ple-mtp-head-sampler.md) (forward:
`mtp.cpp:413-580`). Files: section 4.4.

- At cell c the layer takes the main model's final residual at c (10240) and the embedding of the token at c + 1, and
  predicts the token at c + 2.
- Stages: `pre_fc_norm_embedding` + `fc_embedding`; `pre_fc_norm_hidden` over 10240 + per-stream `fc_hidden`; sum;
  attention GR read (no apply); K, V, Q + gate (Q8_0); norms and RoPE at the cell position; KV append; dense
  attention over the window `[n_kv - 32768, n_kv)`; gate and `o_proj`; MLP GR read (apply); router over 512;
  routed experts (S2 blobs, Q8_0-scaled activations); shared expert; combine; GR write; `hyper_connection_mixer`
  read; the draft head.
- The drafter's KV is plain INT8 even when the main model uses k8v4, and it has no indexer.
- The draft head is the main head, or its rows for the draft vocabulary. Upstream copies those rows at bind
  (`gather_rows`); the Mac reads them in place through the id list ([04](04-metal-kernels.md), section 5.11), with the
  same bits. With a draft vocabulary,
  `row_top_prob` is normalised over the subset only and overestimates the full-vocabulary probability; the 0.5
  threshold applies to that value. Keep this behaviour: acceptance rates are measured against it.
- setup starts the engine with `--spec 4 --spec-min-p 0.5 --mtp rt` (`setup.py:3126`): at most 3 drafts per window,
  stopping at the first draft with probability below 0.5. The engine allows up to 7 (T ≤ 8).
- A K/V-only pass writes the window's T cells; rejected cells are overwritten by the next round before any query
  reads them.

---

## 8. Verify and commit

[audit A.12](audit/A-model-and-kernels.md); [K2 sections 1.5 and 2](audit/K2-qsa-kv-gdn.md);
[K4 section 3.5](audit/K4-ple-mtp-head-sampler.md).

A round verifies a window `[x, d_0, …, d_{T-2}]` of 1 ≤ T ≤ 8 tokens, accepts the leading drafts that equal the
target's own pick (`a` of them), and commits `a + 1` tokens. Per state:

| State | During verify | At commit (`n_keep = a + 1`) |
| --- | --- | --- |
| GDN recurrent state and conv history | not written; the per-token inputs (`qkv`, `h`, `gate`, `beta`) are kept | replay the kept tokens: `conv_commit` + `step_norm_multi(n_keep)`. No snapshot. |
| QSA K/V cells | written for all T cells | nothing: cells past the commit are overwritten before they can be read |
| Indexer tail | snapshot before the window; appended per token | restore the snapshot, replay `append` for the kept tokens |
| PLE history | not written; the window keeps its normalised rows | `strata_ple_history(hist, ple_norm, n_keep)` keeps the last nine rows; the host advances `ple_prev` by `n_keep` |
| MTP K/V | written by the K/V-only pass | nothing (as QSA) |

Invariants:
- Verify never mutates the GDN state, the conv history, the indexer tail or the PLE history.
- Commit replays exactly the kept tokens, in order.
- **Decode ≡ verify bitwise** for every T: the T-token kernels reduce in the same order as the single-token ones.
  The Mac meets this by using the same kernels for T = 1 and T > 1 (decode is a window of one).
- `--spec 0` and `--spec 4` must produce the same token stream for the same prompt and seed.

---

## 9. Reference data for golden comparisons

Golden data comes from two references: llama.cpp at commit `3cf03257` (its per-layer nodes and per-token logits,
written by [09](09-testing-and-validation.md)'s `tools/metal/lc_dump`, gate C1) and upstream Strata's CUDA build of the same pack. CUDA dumps for Mac comparisons are taken with
`STRATA_ROPE_TABLE=1`, so that RoPE matches section 6.7. [09](09-testing-and-validation.md) lists the gates and their
pass criteria.

The Mac `strata generate` **must** keep upstream's teacher-forced per-token loop (`--tokens-file`, `--max-new 1`: the
input at every position is the prompt's own next token, `generate.cpp:5955-5964`) and the dump flags below, with
upstream's file formats, so that upstream's scripts and CUDA dumps are usable unchanged. A dump **must not** change
the arithmetic: the copies are copy kernels recorded in the same command buffers, into a dump arena that no other
kernel reads, and a run with a dump flag produces the same tokens as one without. (Upstream turns fused GR off while
dumping, `generate.cpp:1923-1924`, which is one more reason the cross-backend comparison is a tolerance comparison.
The Mac keeps the fused read and completes the pending FFN write on the dump slot instead:
[04](04-metal-kernels.md), section 9.4.)

### 9.1 Dump files

| Flag | Content | Format (all little-endian) |
| --- | --- | --- |
| `--dump-logits PATH` | logits of every position | `int32 n_vocab, int32 rows`, then `rows × n_vocab` F32. `rows = n_prompt - 1 + max_new` (`generate.cpp:3284-3305`). `--logits-stride N` keeps every Nth row plus the final input and needs `--max-new 1`. Read by `tools/metal/tf_compare.py`. |
| `STRATA_DUMP_FIRST_LOGITS=PATH` | the first verify window's first row, after the batched prompt path | `n_vocab` F32, no header (`generate.cpp:6183-6189`). This is the only logits dump that sees the prompt path of a native pack. |
| `--dump-layers PATH` | the residual ladder (gate C1) | per position `(48 + 1) × 4 × 2560` F32: slot 0 is the embedding broadcast to the 4 streams (llama.cpp's node `hc_init`), slot l + 1 is R after layer l (`l_last-<l>`) (`include/strata/core/session.hpp:344-352`) |
| `--dump-halves PATH` | both halves of every layer | per position `48 × 13,840` F32, layout in section 9.2 |
| `--dump-routing PATH` | routed experts | one record per layer and token: `int32 layer, int32 k, int32 ids[k], F32 weights[k]` (`generate.cpp:620-660`). Records from verify windows carry zero weights, which `tools/make_profile.py` reads as unit weights. This trace is the input of the expert profile ([06](06-engine-host-port.md), section 9.5). |
| `--dump-residual PATH` | the final R | `int32 4, int32 2560`, then `4 × 2560` F32 (`generate.cpp:6364-6377`) |

### 9.2 The halves dump

The stride is `2N + 2hc + NH·HD + 5·NKV·HD + 8` = 13,840 floats (55,360 B) per layer (`dump_stride_floats`,
`src/core/layer.cpp:1118-1131`). The comment there lists fewer slots than the stride holds; the table below is what the
code writes (`layer.cpp:896`, `:915`, `:977`, `:1229-1231`, `:1297`):

| Float offset | Count | Content |
| ---: | ---: | --- |
| 0 | 2560 | Attention half: the block output (GDN or QSA, after `attn_output` / `ssm_out`) |
| 2560 | 2560 | FFN half: the combined MoE output, shared expert included |
| 5120 | 4 | Attention `inject` |
| 5124 | 4 | FFN `inject` |
| 5128 | 6144 | QSA only: attention output before the gate |
| 11,272 | 512 | QSA only: `vcur` as projected |
| 11,784 | 512 | never written (zero) |
| 12,296 | 256 | QSA only: the gathered V scratch, FP16 pairs |
| 12,552 | 4 | QSA only: `ids[0..3]` of the selection (int32 bits) |
| 12,556 | 4 | QSA only: `step[0..3]` (int32 bits) |
| 12,560 | 256 | QSA with FP16 KV only: the first 512 halves of the V pool, FP16 pairs |
| 12,816 | 512 | QSA only: `vcur` after the KV append |
| 13,328 | 512 | QSA only: `kcur` after norm and RoPE |

- With FP16 KV, upstream also copies the first 512 halves of the K pool to offset 12,304 before the attention; the
  V scratch, `ids` and `step` copies overwrite it. The Mac writes only the final values above.
- GDN layers leave every QSA slot zero. The file is a manual bisection aid: no upstream script parses it.

### 9.3 Oracle constants

`src/kernels/ple_oracle_vectors.inc` holds 6 hash cases (window direction, EOS cuts, NULL as EOS, token 0, own EOS,
real ids), the PLE probe rows {0, 12345, 320001535} with 8 head values, 8 tail values and the row's L1 sum (the last
row is all zeros), and two file constants: `kTableDataStart = 192` and `kTableFileSize = 28,800,138,432`. Both
describe the ISTA/Coder shard 2. For Swift the table is a tensor inside shard 1, so the Mac test **must** take the
table's offset from the GGUF tensor table and check the tensor's size (28,800,138,240 B), not the file size. The
probe values hold for Swift as well: rows 0, 12345, 1,000,000, 200,000,000 and 320,001,535 of the Swift IQ2_XS table
are byte-identical to the ISTA rows, and decoding them with the rule of section 6.8 reproduces the oracle's head,
tail and L1 values (`plecmp.sh` in [spikes/model-headers/](spikes/model-headers/)). The
oracle's `pack/full/dense.bin` offsets of `ple_key` and `ple_value` (`kKeyCodesOffset` … `kValueBytes`) describe a
canonical pack and are not used.

---

## 10. Measurements

| ID | Question | How to measure | Provisional default | Decision it drives |
| --- | --- | --- | --- | --- |
| R-1 | Coder (256 experts): do the Metal router's ids equal CUDA's canonical FP64 router's, and how far apart are the weights? | Rebuild the FFN read's `mixed` from a CUDA `--dump-layers` and `--dump-halves` run (1,000 positions of `tf-doc`, [09](09-testing-and-validation.md), section 4.1) and feed it to the Metal router; compare with the CUDA `--dump-routing` trace | Ids identical except at exact ties; weights within 1e-5 relative | The router tolerance in [09](09-testing-and-validation.md); whether Coder needs a CPU FP64 normalisation (not planned) |
| R-2 | The real expert union of a verify window, per layer, for T = 2, 3 and 4 | `--dump-routing` traces of the benchmark prompts on the Mac with `--spec 4`; count distinct ids per (window, layer) | The upper bound of section 4.3 | The verify-window byte estimate; the tiered-mode slot pool size ([05](05-memory-and-residency.md), section 5) |
| R-3 | MTP acceptance per draft position, with and without the draft vocabulary, per file | `strata serve` statistics (`spec_accept` in the DONE line) over the benchmark prompt sets | Upstream's rates, because the arithmetic matches | The `--spec` and `--spec-min-p` defaults on the Mac; whether `cjk` stays the default draft vocabulary |
| R-4 | Are all unlisted float tensors byte-identical between `dense.bin` and the GGUF, in every file? | The full byte compare of work item R5 on all seven packs, once | Yes (section 4.5) | The serving rule; if a tensor differs, it is served from `dense.bin` and listed as an exception |
| R-5 | Per-layer drift of the residual between the Metal and CUDA builds | `--dump-layers` on 64 positions of each benchmark prompt, both builds, same pack; relative L2 per slot | Grows smoothly with depth and stays below the CUDA-vs-llama.cpp drift | The ladder tolerance of gate C1 in [09](09-testing-and-validation.md) |

Items in other documents that change values here: none. The bandwidth numbers that turn section 4.3 into tokens/s are
in [appendix-a-measurements.md](appendix-a-measurements.md), section A.7.

---

## 11. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). The loaders themselves are
[06](06-engine-host-port.md) work item H5; the items here are the reference data and the checks that H5 and the
kernels are tested against.

| # | Item | Done when |
| --- | --- | --- |
| R1 | Header fixtures: the tensor tables and non-tokeniser metadata of the 7 shard 1 files and the shard 2 files, as JSON, from [spikes/model-headers/](spikes/model-headers/) into `tests/fixtures/model-headers/` | The fixtures reproduce `summary.txt` and `swift-summary.txt`, including each shard's actual tensor count; every model has 1,224 tensors across its shards, and the type maps of section 3.8 and sizes of sections 2.4 and 4.1 match |
| R2 | `model_metadata_test`: the checks of section 2.5 | Passes on all seven fixtures; each bold key, changed in a copy, is refused with a message naming it |
| R3 | `model_types_test`: every tensor's ggml type is in the Phase-1 list of section 3.3, and the FF = 640 rule holds | Zero unsupported tensors in the seven fixtures; a synthetic Q5_0 expert down and a synthetic `ne0 % block_elements != 0` tensor are refused with the tensor's name |
| R4 | `ple_constants_test`: the constants in `ngram.hpp` equal the `qwen4exp.ple.*` keys, and EOS, the image token and the head sizes match section 2.5 | Passes on all seven fixtures; a changed multiplier is detected |
| R5 | `dense_vs_gguf` check (setup and test suite): validate required pack forms, conversion payloads and unconverted `dense.bin` bytes against the GGUF (section 4.5) | setup runs a deterministic sample of up to 32 unconverted float tensors after packing and refuses a mismatch; `dense_vs_gguf_test` checks all seven catalogue identities, recomputes recorded conversions and rejects missing or native-marked required-form rows, any `native_experts.txt` table that differs from the source-derived layout, incomplete packs and inconsistent split metadata ([09](09-testing-and-validation.md), section 9.1) |
| R6 | `native_experts_reader_test` (section 3.5) | v3 and v4 files parse; version 5, a missing offset column, a wrong slab size, an unknown shard filename and a truncated file are refused; an `experts.bin` next to the pack is ignored and reported as a warning |
| R7 | `pack_index_test` (section 4.5): 19 fields, native rows, `conversions.json` | Upstream's `index.txt` of each pack parses; a canonical row (code bits other than 8), a row with `compat_bf16: true` and an 18-field row are refused |
| R8 | `expert_address_test` (section 3.4) | For every (layer, role, expert) in sparse 512-expert single-shard, Swift IQ2_XS-like and IQ3_XXS-like two-shard layouts, a synthetic v4 per-role shard split, and 256-expert Coder fixtures, the WeightStore source equals `data_start + tensor offset`, and each planner slice equals that base plus `e × slab` with the correct shard and window mapping. Recheck all seven pinned catalogue packs when T3 provides them; generated fixtures do not establish their real offsets or provenance |
| R9 | PLE oracle on every file (section 9.3) | `ple_oracle_model_test` scans every GGUF under `STRATA_TEST_MODELS` that contains the PLE tensor and checks the frozen head and tail samples plus full-row L1 sums for rows 0, 12345 and 320001535. It requires the canonical `qwen-Q2_0` directory's standalone shard 2 (`split.no = 1`) and the canonical `swift-IQ2_XS` directory's embedded shard 1 (`split.no = 0`), then checks those probe rows are byte-identical between the two sources. The standalone table must match its 192-byte data start and full-file extent; the embedded table must use its parsed tensor-table offset and exact tensor byte size. A host fixture separately checks Direct and mapped reads, including a nonzero tensor offset |
| R10 | MTP runtime reader checks (section 4.4) | `mtp_reader_test`: sizes and names of `dense.txt`, `dense.bin`, `experts.bin` and `draft_vocab.bin` are checked; the required tensor list of `mtp.cpp:196-206` is enforced; a truncated `experts.bin` is refused |
| R11 | Dump flags and formats (section 9.1) | `dump_formats_test`: `tf_compare.py` reads a Mac `--dump-logits` file and `tools/make_profile.py` reads a Mac `--dump-routing` trace; `dump_eq_nodump`: a run with each dump flag gives the same tokens as a run without |
| R12 | CUDA golden set: `--dump-logits`, `--dump-layers`, `--dump-halves` and `--dump-routing` of the benchmark prompts on upstream's CUDA build with `STRATA_ROPE_TABLE=1`, for the Q2_0 and Coder packs, plus `STRATA_DUMP_FIRST_LOGITS` | Stored with the upstream commit, the GPU and the command lines; gate C1 and R-1, R-5 read them |

---

## 12. Where the details are

The audit sections that hold the arithmetic this document points to. Where they disagree with this document,
section 1.3 applies.

| Topic | This document | Audit |
| --- | --- | --- |
| Constants and shapes | 3.1 | [A.0](audit/A-model-and-kernels.md); [K1 section 1](audit/K1-gr-router-shared-rope-cvec.md); [K2 section 0](audit/K2-qsa-kv-gdn.md) |
| Tensor inventory and formats | 3.2-3.3 | [A.1, B.1-B.3](audit/A-model-and-kernels.md); [K3 section 1](audit/K3-quant-and-matvec.md) |
| Activation contract and MSL rounding rules | 5 | [B.2.1-B.2.4](audit/A-model-and-kernels.md); [K3 section 2](audit/K3-quant-and-matvec.md) |
| Token lifecycle | 6.1 | [A.2](audit/A-model-and-kernels.md) |
| GR | 6.2 | [A.3](audit/A-model-and-kernels.md); [K1 section 3](audit/K1-gr-router-shared-rope-cvec.md) |
| GDN | 6.3 | [A.4](audit/A-model-and-kernels.md); [K2 section 2](audit/K2-qsa-kv-gdn.md) |
| QSA, indexer, KV formats | 6.4 | [A.5](audit/A-model-and-kernels.md); [K2 sections 1 and 3](audit/K2-qsa-kv-gdn.md) |
| Router, shared expert, experts, combine | 6.5-6.6 | [A.6](audit/A-model-and-kernels.md); [K1 sections 4-5](audit/K1-gr-router-shared-rope-cvec.md); [K3](audit/K3-quant-and-matvec.md) |
| RoPE, YaRN, M-RoPE | 6.7 | [A.7](audit/A-model-and-kernels.md); [K1 section 6](audit/K1-gr-router-shared-rope-cvec.md) |
| PLE | 6.8 | [A.9](audit/A-model-and-kernels.md); [K4 section 1](audit/K4-ple-mtp-head-sampler.md) |
| Embedding, head | 6.9 | [A.10](audit/A-model-and-kernels.md); [K4 section 2](audit/K4-ple-mtp-head-sampler.md) |
| Sampler | 6.10 | [A.10](audit/A-model-and-kernels.md); [K4 section 4](audit/K4-ple-mtp-head-sampler.md) |
| Vision | 6.11 | [A.13](audit/A-model-and-kernels.md) |
| Control vectors | 6.12 | [A.8](audit/A-model-and-kernels.md); [K1 section 7](audit/K1-gr-router-shared-rope-cvec.md) |
| MTP | 4.4, 7 | [A.11](audit/A-model-and-kernels.md); [K4 section 3](audit/K4-ple-mtp-head-sampler.md) |
| Verify and commit | 8 | [A.12](audit/A-model-and-kernels.md); [K2 sections 1.5 and 2](audit/K2-qsa-kv-gdn.md) |
| Tests, oracles, bring-up order | 9 | [E.0-E.8](audit/A-model-and-kernels.md); [K4 section 8](audit/K4-ple-mtp-head-sampler.md) |
| Numerical subtleties for exact parity | 6 | [K1 section 12](audit/K1-gr-router-shared-rope-cvec.md); [K2 section 7](audit/K2-qsa-kv-gdn.md); [K3 section 8](audit/K3-quant-and-matvec.md) |
| Metal rules | (doc 04) | [F](audit/A-model-and-kernels.md) |
