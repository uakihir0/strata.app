# 05. Memory and residency

Status: implementation-ready specification, 2026-10-03.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- [appendix-a-measurements.md](appendix-a-measurements.md): measured facts on an M5 Pro 64 GB;
- [audit/K5-orchestration-vram-tests.md](audit/K5-orchestration-vram-tests.md), sections 2, 3.5, 4.1-4.3 and 4.9:
  upstream VRAM mechanisms and their Apple replacements;
- [audit/K2-qsa-kv-gdn.md](audit/K2-qsa-kv-gdn.md), section 3: KV formats;
- [audit/B-engine-host.md](audit/B-engine-host.md), section 3: OS memory APIs;
- [03-model-reference.md](03-model-reference.md), sections 2.4, 3.9, 4.1 and 4.5: file, tensor and pack sizes.

This document decides where every byte the engine uses lives on a Mac, how much of the model is GPU-resident, and what
happens to the rest. It covers:
- the memory consumers and their sizes;
- the memory plan, its admission rules and the PLAN line;
- full mode: windows over the model files and residency;
- tiered mode: the arenas, the slot pool, the miss path, prefill, adaptive swaps and lookahead;
- what must never be wrapped;
- buffers, storage modes and alignment;
- example plans for each memory class;
- the interplay with macOS memory management;
- the memory-side measurements, tests and work items.

Other documents cover the rest:
- the host-side control flow of tiered mode (the miss handshake, adaptive swaps):
  [06-engine-host-port.md](06-engine-host-port.md), section 9;
- the kernels that read these buffers: [04-metal-kernels.md](04-metal-kernels.md);
- how setup chooses the flags that feed the plan: [07-server-setup-port.md](07-server-setup-port.md), section 4.4.

Where this document and an audit file disagree, this document wins. Section 1.3 lists the places where the audits are
superseded.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Facts that shape the design

### 1.1 Measured

| Fact | Value (M5 Pro 64 GB, macOS 27.0) | Consequence |
| --- | --- | --- |
| One physical memory | `hasUnifiedMemory` = 1 | No VRAM tier and no PCIe tier. The upstream "expert cache" becomes "which experts are GPU-resident". |
| GPU working-set budget W | `recommendedMaxWorkingSetSize` = 51.84 GiB (81% of RAM) | The sum of all Metal buffers must stay below W − reserve. W varies by machine and is read at startup. |
| Largest single buffer | `maxBufferLength` = 38.88 GiB | A model file is wrapped as several windows (section 4.2) |
| VM page | 16 KiB | Every no-copy pointer and length is 16 KiB-aligned |
| GPU bandwidth | 237-282 GB/s streaming; 228-230 GB/s for random expert-sized gathers | Experts can live anywhere in a buffer; no packing for contiguity |
| CPU + GPU together | about 284 GB/s combined | No CPU expert path (decided in [06](06-engine-host-port.md), section 1.2) |
| **No GPU demand paging** | A kernel touching 1 MiB of a cold 8 GiB no-copy mmap buffer made all 8 GiB resident (1.2 s) | Do not wrap the PLE tensor payload or SSD-tier experts. A full-mode page-aligned window may cover up to 16 KiB of an excluded neighbor at its edge; those padding bytes are not tensor mappings and are counted as `wrapped_unread`. |
| SSD | 12.7-12.9 GB/s at 1.31 MiB random reads with 8+ threads; 4.8 GB/s with 1 thread; 76K-173K IOPS at 16 KiB | SSD-tier expert reads need a queue depth of 8 or more |

### 1.2 Assumed until measured

Each line is a measurement in section 10.

- W on 32, 36 and 48 GB Macs. The estimates used here are 24, 27 and 37 GiB, as in
  [00-feasibility-study.md](00-feasibility-study.md), section 4.1. Measurement M-8.
- How no-copy pages over a file mapping behave under memory pressure while they are in a residency set: wired,
  evictable, or compressed. Measurement M-4.
- Whether `Private` storage reads faster than `Shared` for the state buffers. Measurement M-6.

### 1.3 Where the audits are superseded

| Audit statement | Replaced by |
| --- | --- |
| K5 section 2.7, item 1: "mmap `experts.bin` (or the GGUF shards) once" and wrap it no-copy. K5 table 4.1, row A7: the PLE table "is mmapped and NoCopy-wrapped, and the GPU gathers from it directly". | Correct only for full mode (section 4). The PLE tensor is not mapped to a window and is read through the host path (section 6); only mandatory page-alignment padding from a neighboring tensor can overlap its edge. In tiered mode the GGUF is never wrapped (section 5). |
| K5 section 2.7, item 2: "reinterpret residency as a work assignment" between CPU and GPU | There is no CPU expert path in the first release. Residency means bytes in a GPU buffer. |
| K5 table 4.1, row A10: a helper over `os_proc_available_memory()` | `strata::platform::available_memory()` built on `host_statistics64` ([06](06-engine-host-port.md), section 7.2). `os_proc_available_memory` is not documented for macOS. |
| K5 section 4.9, miss option 1: a GPU kernel page-faulting from a mmapped-file NoCopy buffer | Ruled out by the demand-paging measurement. Option 3, host resolution of misses after routing, is the design. |
| A section D.3 and K5 sections 3.1 and 3.5: "about 680 KB per token" for a prefill chunk, and the chunk rule built on it | The Mac's prompt path needs about 261 KiB per token plus a fixed part ([04](04-metal-kernels.md), section 6.1). Section 2.4 gives the sizes the plan uses. |
| B section 3, the `maxBufferLength` row: "One buffer per shard (the shards are ~10-20 GB)" | Shards are 28-55 GB ([03](03-model-reference.md), section 2.4). A shard is wrapped as one or more windows over the ranges the engine reads (section 4.2). |

---

## 2. Memory consumers

Units: GB = 10^9 B, GiB = 2^30 B, MiB = 2^20 B. The plan computes in bytes.

### 2.1 Model files

Sizes come from the GGUF tensor tables at the pinned revisions ([03](03-model-reference.md), sections 2.4 and 4.1;
[appendix A.9](appendix-a-measurements.md)). The plan reads them from the files or from the geometry file
([07](07-server-setup-port.md), section 4.4.3), never from this table.

Where the PLE table (the n-gram embedding table of [00](00-feasibility-study.md); upstream code calls it PLE) lives:
- **ISTA and Coder:** it is the whole of shard 2, one 28,800,138,432 B file shared by all five (IQ4_NL
  `[160, 320001536]`, 90 B per row, 28,800,138,240 B of row data at file offset 192). Shard 1 holds everything else.
- **Swift:** it is a range of shard 1, between the head tensors and layer 0. Shard 2 holds the later layers, their
  experts included ([03](03-model-reference.md), section 3.9).

The engine finds every tensor by name, in whichever shard holds it. "Non-expert" below means every tensor except the
routed experts and the PLE table that the engine reads from the GGUF: the plan's `nonexpert_bytes`.

| File | Shards (B) | Routed experts | Non-expert tensors | Expert blob (gate + up + down) |
| --- | --- | ---: | ---: | --- |
| Q2_0 | 37,623,740,192 + PLE shard | 33,973,862,400 B = 31.64 GiB | 3,638,853,120 B = 3.39 GiB | 1,382,400 B in every layer |
| IQ2_XS | 39,225,954,592 + PLE shard | 35,454,976,000 B = 33.02 GiB | 3,759,953,920 B = 3.50 GiB | 1,177,600-1,510,400 B, mean 1,442,667 |
| IQ3_XXS | 47,039,860,096 + PLE shard | 42,912,972,800 B = 39.97 GiB | 4,115,863,040 B = 3.83 GiB | 1,305,600-2,329,600 B, mean 1,746,133 |
| IQ3_S | 54,817,524,224 + PLE shard | 50,292,326,400 B = 46.84 GiB | 4,514,173,440 B = 4.20 GiB | 1,510,400-2,662,400 B, mean 2,046,400 |
| Coder IQ1_M (256 experts) | 29,608,446,496 + PLE shard | 25,146,163,200 B = 23.42 GiB | 4,451,258,880 B = 4.15 GiB | as IQ3_S |
| Swift IQ2_XS | 39,788,473,344 + 28,363,693,824 | as IQ2_XS | 3,633,879,040 B, plus 126,074,880 B of routers from `dense.bin`: 3.50 GiB, as IQ2_XS | as IQ2_XS |
| Swift IQ3_XXS | 39,785,790,560 + 36,180,282,560 | as IQ3_XXS | 3,989,788,160 B, plus the same 126,074,880 B: 3.83 GiB, as IQ3_XXS | as IQ3_XXS |

- The per-layer blob size is `native_experts.txt` column `blob`, the sum of the three role slabs
  ([03](03-model-reference.md), section 3.6).
- **Swift's routers.** The GGUF stores the 96 router tensors as F32 (252,149,760 B). The engine reads their exact BF16
  copies from `dense.bin` instead, the only `dense.bin` rows it reads in any catalogue file
  ([03](03-model-reference.md), section 4.5). In full mode the F32 originals lie between tensors the engine reads, so
  the windows wrap them anyway; the plan counts them in `wrapped_unread` (section 3.2).

Within a GGUF, the three role tensors of a layer have ggml `ne` = {n_embd, n_ff, n_expert} (gate, up) and
{n_ff, n_embd, n_expert} (down): `ne0` is the input dimension and the expert index is the slowest
([03](03-model-reference.md), section 3.4). So the slab of one expert in one role is contiguous: 460,800 B for Q2_0.
One expert is therefore three contiguous byte ranges at three offsets.

Other files the GPU reads:

| File | Size | Notes |
| --- | ---: | --- |
| The `dense.bin` rows that `conversions.json` lists | **D_pack**: 0 for ISTA and Coder; 126,074,880 B (0.12 GiB) for Swift | Copied into the dense arena at startup, in both modes (sections 4.1 and 5.2). The rest of `dense.bin` (1.38-1.43 GiB) duplicates GGUF bytes and is never read. |
| `mtp/rt/experts.bin` | 707,788,800 B = 0.66 GiB | 512 S2 blobs ([03](03-model-reference.md), section 4.4) |
| `mtp/rt/dense.bin` | 116,099,072 B = 0.11 GiB | 29 tensors, Q8_0, BF16 and F32. With `experts.bin`: `mtp_bytes` = 823,887,872 B (0.77 GiB). |
| Control vectors | 47 × 2560 × 4 B = 481,280 B each | Copied into a small buffer |

### 2.2 Context-dependent state (per KV cell)

There are 12 QSA layers, each with 2 KV heads of 256 dims ([audit K2 3.1-3.2](audit/K2-qsa-kv-gdn.md)).

| Item | Bytes per cell | Source |
| --- | ---: | --- |
| K + V, fp16 | 12 × 2,048 = 24,576 | K2 table 3.2 |
| K + V, int8 | 12 × 1,056 = 12,672 | |
| K + V, k8v4 | 12 × 816 = 9,792 | |
| K + V, q4_0 | 12 × 576 = 6,912 | |
| Indexer pooled keys (one 128-float key per 4 cells per layer, plus 2 spare rows per layer) | 12 × 128 = 1,536 | sized at `layer.cpp:499`, allocated at `:606-607` |
| RoPE cos/sin table (one copy, n_rot 64: 32 pairs × 4 B × 2) | 256 | `layer.cpp:609-615`; one QSA state owns it, the others and the drafter borrow it (`session.cpp:65-67`, `:106-117`) |
| MTP drafter K/V, one dense-attention layer | fp16 2,048; int8 1,056; k8v4 1,056 (the drafter takes plain int8); q4_0 576. | `mtp.cpp:205-234`; see below |

Main-model total per cell: **fp16 25.75 KiB, int8 14.13 KiB, k8v4 11.31 KiB, q4_0 8.50 KiB**.

The per-cell rates exclude the fixed indexer rows and state. `KvStore` rounds
the configured context to `capacity = round_up_4(max_context)` and allocates:

```
kv_bytes = capacity × (K/V bytes per cell + 1,536 pooled-key bytes per cell + 256 RoPE bytes per cell)
           + 12 × (2×128 + 3×128 + 128 + 1) × 4 B
             // 2 spare rows, 3 tail rows, 1 dead row, 1 int32 block position
```

All fixed-row terms are four bytes per value, so their total is 36,912 bytes.
The memory planner uses this exact allocation size, including capacity rounding.

The drafter attends densely over the last `--mtp-window` cells (default 32,768) and never reads the indexer.
Upstream's generic QSA state still allocates pooled rows for the drafter, even though its MTP attention does not
consume them. The Metal layout omits those unused rows and the indexer tail/dead state. It keeps only the K/V codes
and any scales read by dense attention. MTP borrows the main model's shared RoPE table rather than allocating a
second one. This also removes the planner's former 32 B/cell allowance, which did not match upstream's 128-float
pooled-row width.

Upstream shrinks the drafter KV to a ring of `window + 4 × max_t + 64` cells only when KV streaming is on
(`--kv-resident` > 0). In that case it refills the ring from a host copy of the whole KV after a resume
(`mtp.cpp:208-211`; `layer.cpp:495-513`). Without streaming, it allocates all `max_cells` and the window only
limits reads.

The Mac engine never streams KV, so it allocates the drafter KV for all cells, as upstream does without streaming. A
ring with no full copy behind it cannot be rebuilt after a checkpoint restore: the drafter's K and V come from
main-model hidden states that are not kept. The format follows `--kv`, except that k8v4 becomes plain int8 for the
drafter (`mtp.cpp:212-218`).

```
mtp_capacity = round_up_4(max_context)
mtp_kv_bytes = mtp_capacity × drafter K/V bytes per cell
             // fp16: 0.0625 GiB at 32K, 0.25 GiB at 128K, 0.5 GiB at 262K
```

The four-cell rounding is shared with the main QSA cache. `StepRow.n_kv` remains bounded by the requested logical
context, so the spare rounded cells are not visible to attention. `make_mtp_kv_layout` is the device-free source of
the per-format byte count and region offsets; the planner reserves its exact allocation size. `MtpKvStore` now
allocates those K/V and scale regions in one Context-owned Metal buffer and exposes bounded subviews.
`MtpRuntimeStorage` composes that store with MTP weights and bind scratch under one rollback boundary. It does not
allocate a second RoPE table; the drafter must use the main `KvStore` table. `ModelStorage` allocates that main
QSA KV, indexer and shared RoPE buffer for every accepted plan, validates its exact layout against `kv_bytes`, and
keeps it in the same outer allocation transaction as the model storage and optional MTP runtime.
`--storage-check` reports `main_kv=yes` only after that allocation and the selected storage have opened successfully.
The diagnostic leaves K/V and RoPE values for the session to initialize; it does not initialize the remaining
session state or enable MTP dispatch. For speculative plans, `ModelStorage` passes this `KvStore` to
`MtpRuntimeStorage::open_with_main_kv`, which validates and retains the exact borrowed RoPE subview without
allocating a duplicate. Its `shared_rope()` accessor is empty whenever the MTP owner is invalid. The table's values
remain uninitialized, so a future session must initialize them before binding MTP dispatch.

| Main-model `--max-context` | fp16 | int8 | k8v4 | q4_0 |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 0.20 GiB | 0.11 GiB | 0.09 GiB | 0.07 GiB |
| 32,768 | 0.80 GiB | 0.44 GiB | 0.35 GiB | 0.27 GiB |
| 65,536 | 1.61 GiB | 0.88 GiB | 0.71 GiB | 0.53 GiB |
| 131,072 | 3.22 GiB | 1.77 GiB | 1.41 GiB | 1.06 GiB |
| 262,144 | 6.44 GiB | 3.53 GiB | 2.83 GiB | 2.13 GiB |

Unlike upstream, KV is never streamed (`--kv-resident` is ignored). The main-model QSA K/V, indexer state and
shared RoPE table use one allocation at startup. The drafter KV is a separate allocation when speculation is enabled.

### 2.3 Fixed engine state

| Item | Size | Notes |
| --- | ---: | --- |
| GDN recurrent and convolution state: 36 layers × (48 heads × 128 × 128 + 10240 × 3) × 4 B | 117.7 MB (112.2 MiB) | `SessionStateStorage` allocates these exact regions, one recurrence plus one three-row convolution history per GDN layer. Same bytes as the GDN portion of one conversation checkpoint. |
| PLE history: 9 rows × 10,240 channels × 4 B | 0.37 MB (0.35 MiB) | `SessionStateStorage` owns and zeroes the row-fastest history. The remaining session buffers (MoE and block scratch) are outside this owner. `WindowScratchStorage` and the Mac `KvStore` own their separate exact allocations. |
| Window scratch at T = 8: activations, commit replay inputs, MoE plans and intermediates, QSA/sampler scratch, logits and sink | `make_window_scratch_layout` accounts for every named region and 256-byte inter-region alignment; `WindowScratchStorage` opens its single Shared allocation. For 256 / 512 experts: 47,641,344 / 47,649,536 B at 32,768 cells; 47,903,488 / 47,911,680 B at 65,536 cells; 49,476,352 / 49,484,544 B at 262,144 cells. |
| MTP bind scratch (`bind_bytes`) | about 17 MB at the default `--mtp-window`; about 107 MB with no window at 262,144 cells | The drafter's named scratch regions, carved at bind for T = 8 ([04](04-metal-kernels.md), section 5.11): about 16.3 MB at a 32,768-cell window, 29 MB at 65,536, and 106 MB at 262,144, almost all of it the attention scratch. The geometry includes the device-side draft-vocabulary ID table and adjusts the single `hl` logits row to `n_out` (the subset size, or 248,320 without a subset). `MtpRuntimeStorage` copies the inspected IDs into that table before publishing the owner. A shape-checked reserved tail remains unbound pending ABI review. Only with `STRATA_SPEC_COUPLED=1`: the id → subset map (1 MB), the history ring of (`kCoupledHistCap` 4096 + max_t) int32 and the split scratch (`mtp.cpp:314-339`). Upstream's `bind_bytes` (`mtp.cpp:304-319`) also counts the logits of max_t rows (7.9 MB) and the gathered draft head (0.15-0.22 GB with `cjk`, by head format); the Mac reads the main head rows in place. The plan counts it as `mtp_bind_bytes`, not in F_fixed. |
| MTP parameter slots | 2.25 KiB without GPU prefill; 258 KiB at an 8,192-row chunk; about 1 MiB at 32,768 rows | A separate Shared allocation of 256-byte slots: 8 fixed K/F0-F6 slots and at least one group slot for short-read catch-up. Prefill chunks reserve one slot per group of up to 8 rows. The planner and `MtpRuntimeStorage` use the selected chunk size; slots are zeroed before publication and exposed as bounded views. |
| Metal library, pipelines, residency set | about 50 MB | Inside the separate working-set reserve (default 700 MiB), not `F_fixed`. |

The geometry file stores the MTP bind total at the default 32,768-cell window. When planning, the engine replaces the
window-dependent portion using `cap = round_up_64(min(--mtp-window, --max-context))`; a zero `--mtp-window` means
`cap = round_up_64(--max-context)`. The window-dependent portion is the `cap`-sized `mtp_ids` array plus
`4 × ((cap / 64) × 24 × 258 + 64)` bytes of attention scratch ([04](04-metal-kernels.md), section 5.11). The
geometry also carries the validated `draft_vocab.bin` byte count and the effective `hl` row extent. Most named
buffers stay fixed because the arena is allocated for T = 8 even when `--mtp-max-t` is smaller. `hl` follows
`n_out`, so its extent grows to the full model vocabulary when `draft_vocab.bin` is absent.

The current exact GDN and PLE session-state allocation is 118,038,528 B (112.57 MiB). Together with the T=8
Window scratch at 262,144 cells, the two layouts total 167,514,880 B (159.75 MiB) for 256 experts and
167,523,072 B (159.76 MiB) for 512 experts, below the 256 MiB allowance. The planner reports
`session_state_bytes` and `window_scratch_bytes` from these exact layouts and checks their subtotal against
`F_fixed`; `ModelStorage` rebuilds and validates the same values and opens the Window scratch inside its outer
allocation transaction. Session state remains owned by the session lifecycle. These two fields are a breakdown of
`fixed_bytes`, not additional plan charges. Keep `F_fixed = 0.25 GiB` in admission until every fixed buffer has a
verified exact size.

### 2.4 Prompt path (`--prefill`)

Upstream's planning formula for a chunk of C tokens is `160 MiB + C × 680 KiB` (`generate.cpp:2289-2290`; the same
formula recurs at `:1376`, `:2672` and `:2693`). It covers activations, attention/GDN/MoE scratch, the GEMM scratch
and staging buffers, including the copy ring, the stager ring and the cuBLAS workspace that the Mac does not have.

The Mac `Prefill` reports its own size instead, and the plan calls it **prefill_bytes(C)**:
`bytes_needed(C) = C · P + 8 · ⌈10C / 64⌉ + F` ([04](04-metal-kernels.md), section 6.1).
- P is 267,304 B per token, or 272,424 B when `STRATA_PREFILL_BF16X2` is 1 or 2.
- F is about 153 MiB at 262,144 cells: 64 MiB of dequantisation scratch, the attention partials of one 32-row batch,
  the block scores of one 256-row group (1,024 B × (cells / 4 + 2)), the 48 per-layer plans and slack. Only the
  block scores depend on the context.

| Chunk C | 262,144 cells | 65,536 cells | Upstream's formula |
| ---: | ---: | ---: | ---: |
| 1,024 | 0.405 GiB | 0.358 GiB | 0.82 GiB |
| 2,048 | 0.660 GiB | 0.613 GiB | 1.48 GiB |
| 4,096 | 1.169 GiB | 1.123 GiB | 2.81 GiB |
| 6,144 | 1.679 GiB | 1.632 GiB | 4.14 GiB |
| 8,192 | 2.189 GiB | 2.142 GiB | 5.47 GiB |
| 16,384 | 4.228 GiB | 4.182 GiB | 10.78 GiB |
| 32,768 | 8.307 GiB | 8.260 GiB | 21.41 GiB |

`metal_plan_test` checks that the plan's prefill_bytes equals `Prefill::bytes_needed` for every candidate C, and the
prefill tests check that the allocation `Prefill` makes is exactly that size.

The plan never picks a chunk below 1,024. Upstream switches prefill to its stream-all expert path at 1,024 tokens
(`stream_all_min`, `prefill.cpp:77-84`, env `STRATA_PREFILL_STREAM_MIN`). That is a scheduling threshold, not a limit
on the chunk size. Its comment records why the threshold is not lower: below about 1,000 tokens the Q2_0 output
changed, because a smaller chunk takes other kernels. Chunks of at least 1,024 therefore keep the Mac prompt path
comparable with the upstream path that CUDA reference dumps come from.

### 2.5 Host RAM outside the GPU budget

These live in ordinary process memory. They do not count against W, but they do count against physical RAM.

| Item | Size |
| --- | --- |
| PLE row cache (`--ple-row-cache` 1,048,576 rows × 90 B) | 94 MB |
| PLE DirectFile buffers (`--ple-inflight` 64 × 8 KiB: two pages per slot, `ple_reader.cpp:104`, `:350`) | 512 KiB |
| Prompt-cache checkpoints (`--prompt-cache` 6 × about 118 MB) | about 0.7 GB |
| Conversation parking (`--conversation-cache-mib`) | as configured; default 0 |
| Suffix drafter: capacity `--max-context` + 4096 tokens (`generate.cpp:4665`, `:6119`). 4 B of int32 history per token, plus a hash table of the next power of two ≥ 2 × capacity slots at 32 B each: 64-128 B per token. The upstream header comment's "about 20 B per token" is wrong. | about 33 MiB at 262K |
| Page cache holding cold expert ranges (tiered mode) | whatever the OS gives |
| Server (Python, tokeniser, HTTP) and the vision helper (about 1 GB when loaded) | outside the engine |

---

## 3. The memory plan

### 3.1 Inputs

| Input | Source |
| --- | --- |
| W | `MTLDevice.recommendedMaxWorkingSetSize`, then min(W, `--gpu-budget-mib` or `STRATA_GPU_BUDGET_MIB`). `--plan-w-mib N` replaces the device value (with `--plan-only` only, [06](06-engine-host-port.md), section 3.10). |
| Reserve | `--vram-reserve-mib`, default 700 MiB |
| RAM | `hw.memsize`, and `available_memory()` at startup. `--plan-ram-mib N` replaces `hw.memsize`; the startup available-memory check (section 3.2, step 6, second bullet) is then skipped and `avail_mib=-1`. |
| Pack geometry | shard sizes and tensor tables, `native_experts.txt` (per-layer blob sizes), `conversions.json` (D_pack), MTP sizes. From the files, or from a geometry file (`--geometry FILE`, [07](07-server-setup-port.md), section 4.4.3) that holds exactly these facts. |
| Context | `--max-context`, `--kv` |
| Prompt path | `--prefill` (absent, `auto`, `auto:N` or C) |
| Expert controls | `--expert-cache N\|auto`, `--resident-budget-gib G`, `--expert-cache-per-layer`, `--expert-profile`, `--expert-slots N` |

### 3.2 Algorithm

All arithmetic is in bytes, in this order. `--plan-only` prints the result and exits (section 3.4).

1. **Budget.** `B = W − reserve`.
2. **Always-resident set.**
   ```
   A = nonexpert_bytes   // every non-expert tensor the engine reads from the GGUF shards (section 2.1)
     + D_pack + F_fixed + kv_bytes(max_context, kv)
     + (when --spec > 0: mtp_bytes + mtp_bind_bytes + mtp_params_base_bytes + mtp_kv_bytes)
   ```
   `mtp_params_base_bytes` is 9 × 256 B when `--spec > 0`: eight slots for K and F0-F6 plus one reusable
   short-read catch-up group. For a GPU prefill chunk C, add
   `mtp_params_prefill_bytes(C) = 256 × max(0, ceil(C / 8) − 1)` to P.
   `wrapped_unread` is the number of bytes that full-mode windows cover but the engine does not read: alignment
   padding, the 16 KiB rounding, merged gaps and, for Swift, the F32 routers. The plan computes it from the tensor
   tables with steps 1-3 of the window planning (section 4.2), without the `maxBufferLength` split, so the value does
   not depend on the device. It is 252 MB for Swift and under 1 MiB for the other files. It counts in full mode only.
   If `A > B`, exit 1 with the message from [06](06-engine-host-port.md), section 2.3, step 8:
   `the model does not fit: needs X MiB of GPU memory with no expert resident, Y MiB available`.
   - X counts A, plus P and the empty-set slot pool S(∅) in tiered mode.
   - Y is B.
   - When KV is more than a quarter of A, a second stderr line follows:
     `strata: KV takes K MiB; lower --max-context or use --kv k8v4`.
3. **Mode.**
   - Let `E_all` be the bytes of all routed experts.
   - Let `E_cap` be the resident byte cap: the smallest of `E_all`, the bytes of the first N experts in profile order
     (`--expert-cache N`), and `--resident-budget-gib G`. An explicit `--expert-cache N` also caps the resident count
     at N, as section 3.5 defines.
   - Let `P_min` be the smallest prompt-path allocation the flags allow:
     - 0 without `--prefill`;
     - prefill_bytes(C) with `--prefill C`;
     - prefill_bytes(1024) with `--prefill auto` or `auto:N`.
     Let C_min be the selected minimum chunk (zero without `--prefill`). When speculation is enabled, account for
     additional parameter slots separately as `mtp_params_prefill_bytes(C_min)`.
   - **Full mode** when `E_cap = E_all`, any explicit `--expert-cache N` covers every expert, and
     `A + wrapped_unread + P_min + mtp_params_prefill_bytes(C_min) + E_all ≤ B`. Otherwise **tiered mode**.
4. **Prompt-path buffers P.**
   - Without `--prefill`, P = 0. With `--prefill C`, P = prefill_bytes(C) in both modes.
   - When speculation is enabled, add `mtp_params_prefill_bytes(C)` to P; `mtp_params_bytes` reports the base
     slots in A plus any additional prefill-group slots in P.
   - With `--prefill auto` or `auto:N`:
     - The candidates are {8192, 6144, 4096, 2048, 1024}, plus 16384 and 32768 when N allows them. Each must be
       ≤ N (default 8192) and ≤ `max_context` rounded up to a multiple of 1024.
     - Full mode takes the largest candidate with `A + wrapped_unread + prefill_bytes(C) +
       mtp_params_prefill_bytes(C) + E_all ≤ B`.
     - Tiered mode takes C = 2048 when `B − A ≥ 24 GiB`, and 1024 otherwise (capped by N). On small Macs the
       0.26 GiB that a 2048 chunk adds over a 1024 chunk would displace about 200 Q2_0 experts.
5. **Tiered mode: resident set and slot pool.** The slot pool depends on the resident set, so both are chosen
   together.
   - `slot` = the largest per-layer blob, rounded up to 16 KiB. For Q2_0 that is 1,392,640 B.
   - For a resident set R, with `nr_l(R)` = n_expert − |R ∩ layer l|:
     `S(R) = max(2 × max_l nr_l(R), --expert-slots) × slot`. That is two halves of `H = max_l nr_l` slots
     (section 5.4).
   - Walk the STRP profile ranking of (layer, expert). Admit x when **both** of these hold:
     - `bytes(R ∪ {x}) ≤ E_cap`;
     - when `--expert-cache N` is explicit, `|R| < N`;
     - `A + P + bytes(R ∪ {x}) + S(R ∪ {x}) ≤ B`.
     Otherwise skip x and continue; a later expert of a layer with smaller blobs may still fit. Stop when the explicit
     count cap reaches N. Admitting never raises S, so the walk needs a single pass.
   - With `--expert-cache-per-layer` (the upstream `slots / n_layers` quota, `expert_cache.hpp:109`), first run the
     walk without a quota. Then set quota = floor(|R| / n_layers), and run it again with the extra condition
     |R ∩ layer l| < quota. Every catalogue model has 48 layers.
   - With no profile, admit in round-robin order over layers by expert index and print
     `strata: no expert profile; the resident set is arbitrary and the hit rate will be low`.
   - If even `A + P + S(∅) > B`, exit 1 with the step-2 message. For Q2_0, S(∅) is 1.33 GiB.
6. **Check RAM.**
   - Full mode is counted against RAM as well: the same pages are in the page cache. `wrapped_unread` is excluded
     because those mapped bytes are not read into the page cache.
   - If `A + P + E_res + S > hw.memsize − 6 GiB`, print a warning: `the plan leaves less than 6 GiB of RAM for macOS
     and other apps; expect memory pressure`.
   - If `available_memory()` at startup is below the plan's new anonymous bytes (tiered mode), print the same warning
     with planned and available MiB. These new anonymous bytes are `nonexpert_bytes + D_pack + F_fixed +
     mtp_bind_bytes + mtp_params_bytes + kv_bytes + mtp_kv_bytes + prefill_bytes(C) + E_res + S` (including only the
     MTP terms enabled by `--spec`; parameter slots are counted once).
     The MTP model files are mapped files, not new anonymous allocations. A warning, never an error.

### 3.3 Admission rules

- The profile decides residency order. The upstream profile files are usable as they are. `tools/make_profile.py` can
  also build a Mac profile from `--dump-routing` traces, and setup ships the upstream ones.
- A non-resident expert is never computed on the CPU. A miss always costs a read, either from the page cache or from
  the SSD.
- **Coder (256 experts):** `nr_l` = 256 − resident; everything else is identical.
- **MTP experts** (512 for the draft layer) are always resident when `--spec > 0`. They are part of A. They are small
  (0.66 GiB), and a draft miss would stall every round.

### 3.4 The PLAN line (`--plan-only`)

`--plan-only` prints one stdout line of `key=value` pairs and exits 0 when the plan fits, 3 when it does not. Setup
and the Monitor parse it, so the keys are stable:

```
PLAN fits=1 mode=full|tiered gpu=Apple_M5_Pro w_mib=53084 reserve_mib=700 ram_mib=65536 avail_mib=41210
 dense_mib=3471 unread_mib=1 pack_mib=0 mtp_mib=786 mtp_bind_mib=17 fixed_mib=256 kv_mib=1648 mtp_kv_mib=130
 kv=fp16 context=65536 prefill_chunk=8192 prefill_mib=2194 experts_total=24576 experts_total_mib=32400
 experts_resident=24576 experts_resident_mib=32400 slots=0 slot_mib=0 ssd_experts=0
 total_mib=40904 free_mib=11480 ram_warning=0
 w_full_mib=... mtp_params_mib=1 session_state_mib=113 window_scratch_mib=46
```

The line wraps here only for readability; the real output is one line. The `mib` values are rounded up. The example
is Q2_0 at 65,536 fp16 cells, with 17 MiB of MTP bind scratch (the default drafter window). `dense_mib` is `nonexpert_bytes`,
`unread_mib` is `wrapped_unread` (0 in tiered mode) and `pack_mib` is D_pack. `total_mib` sums the top-level
allocation fields; `session_state_mib` and `window_scratch_mib` are informational breakdowns inside `fixed_mib` and
are not added again. `free_mib` is `w_mib − reserve_mib − total_mib`. The `mtp_params_mib` key follows `w_full_mib`
or the optional `reason`, and includes fixed and prefill-group slots. The session and Window scratch keys follow it.
The fit decision itself is made in bytes.
Because each component is rounded up separately, `free_mib` can be negative by a few MiB even when `fits=1`; the
byte-accurate fit decision and `free_bytes` are authoritative.

The remaining fields are:
- `w_full_mib`: the W, in MiB, at which this plan would be full mode,
  `ceil((reserve + A + wrapped_unread + P_min + mtp_params_prefill_bytes(C_min) + E_all) / MiB)` when speculation
  is enabled, omitting the parameter term otherwise. It is printed in both modes (in full mode it is at most
  `w_mib`). This is the memory threshold assuming the residency flags allow every expert; an explicit
  `--expert-cache N` below the total still forces tiered mode at larger W. Setup's wired-limit advice reads it
  ([07](07-server-setup-port.md), section 4.5.6).
- `reason`, on `fits=0` lines only. Its value has spaces replaced by `_`. The reasons:

| Reason (before the replacement) | When |
| --- | --- |
| `the model does not fit: needs X MiB of GPU memory with no expert resident, Y MiB available` | Section 3.2, steps 2 and 5 (an explicit `--prefill C` that does not fit ends here too) |
| `unsupported_type_<T>` (no replacement needed; `<T>` is the GGUF type name, for example `q4_K`) | A tensor the engine loads has a type with no Metal kernel ([04](04-metal-kernels.md)). Checked first, from the files or the geometry file's `types=`. |

- `mtp_params_mib`: the planned fixed and prefill-group MTP parameter slots, appended after `w_full_mib` or the
  optional `reason`.
- `session_state_mib`: the exact GDN and PLE session-state allocation, rounded up to MiB for display; this is
  included in the conservative `fixed_mib` allowance.
- `window_scratch_mib`: the exact T=8 Window scratch allocation at the QSA layout's rounded cell capacity,
  rounded up to MiB for display; this is included in `fixed_mib` and opened by `ModelStorage`.

The keys are stable. A new key is appended at the end of the line; a key is never renamed or removed without a
`metal_rev` bump ([06](06-engine-host-port.md), section 5.2).

---

## 4. Full mode

### 4.1 Layout

| Data | Placement |
| --- | --- |
| GGUF shards | The upstream reader's `mmap(PROT_READ, MAP_PRIVATE)` of each whole shard (`gguf_reader.hpp:404-405`), wrapped as no-copy **windows** over the ranges the engine reads (section 4.2), all in the residency set. The PLE tensor has no window mapping; page-alignment padding can cover up to 16 KiB at a neighboring tensor boundary (section 6). The ISTA and Coder shard 2 therefore has no window. Wrapping a `MAP_PRIVATE` mapping is verified ([appendix A.3](appendix-a-measurements.md)). |
| The `dense.bin` rows of `conversions.json` (Swift's routers) | Exact rows are `pread` from the locked `dense.bin` descriptor into a separate anonymous `Shared` arena at startup. Tensor addresses are 256-B-aligned and the arena length is rounded up to 16 KiB; the page-rounding slack is charged to the plan's free bytes. The rest of `dense.bin` is not read. A Mac pack has no `extra.bin`; one found in the pack is ignored and logged. |
| `mtp/rt/experts.bin`, `mtp/rt/dense.bin` | Read-only mappings split into 16 KiB-aligned no-copy windows no larger than `maxBufferLength`; expert blobs and dense tensors stay intact. The files are warmed before open succeeds, and the mappings plus generation lock live until Context releases the Metal buffers. Each opened descriptor must match the inspected device, inode and size, and the lock path must still name its inspected regular file. |
| KV, GDN state, session, window, MTP scratch, prefill buffers | Anonymous `Shared` buffers from the bump arena (section 7) |
| Expert address table: 48 × 512 × 3 `uint64` GPU addresses | A 590 KB `Shared` buffer, built at load: `window.gpuAddress + offset` for each role of each expert |
| Resident table: 48 × 512 `int32` | All entries "resident". The plan kernel still reads it, so full and tiered modes run the same kernels. |

### 4.2 Windows over a file

- A window is one `newBufferWithBytesNoCopy:length:options:MTLResourceStorageModeShared deallocator:nil` over
  `[base + start, base + end)` of the mapping.
- `start` and `end` are multiples of 16 KiB, and `end − start ≤ maxBufferLength`.
- The last window may extend past the exact file size to the end of its final mapped VM page. Tensor payload ranges
  remain bounded by the exact file size; the padding is only the zero-filled tail of that mmap page.

Planning the windows:

1. Collect the byte ranges of every tensor the engine reads from the file, sorted by offset.
2. Merge neighbours closer than 256 MiB. The unread bytes between them are wrapped too and counted in the plan.
3. Round each merged range out to 16 KiB.
4. If a range is longer than `maxBufferLength`, split it at a tensor boundary.
   - For an expert tensor, split at an expert-slab boundary: a split inside a 3D tensor is allowed between experts.
   - A single tensor slab larger than `maxBufferLength` is fatal. No current model has one.
   - Preserve the entire pre-split merged range. If a merged unread gap is itself longer than the limit, cover it
     with additional bounded windows; a split must not drop bytes counted by `wrapped_unread`.
5. Each tensor or expert slab maps to (window, offset). Its GPU address is `window.gpuAddress + offset`.

Examples on a 64 GB M5 Pro (`maxBufferLength` 38.88 GiB):
- Q2_0: shard 1 is 35.04 GiB, so it is one window. Shard 2, the PLE table, has none.
- IQ3_XXS (43.81 GiB): two windows, split at an expert-slab boundary near the middle of the expert tensors.
- Swift IQ2_XS: shard 1 gives two windows, because the PLE table (26.82 GiB) separates the global tensors (about
  0.33 GiB) from layers 0-13 (about 9.9 GiB). Shard 2 (26.42 GiB) gives one.

`maxBufferLength` on 96 and 128 GB Macs is not measured (M-8). The algorithm does not depend on it.

**Rounding out.** Rounding to 16 KiB may cover up to 16 KiB of the neighbouring bytes on each side. That is safe in
full mode, because the whole file is resident anyway. macOS 27.0 accepted unaligned pointers and lengths in the spike
([appendix A.3](appendix-a-measurements.md)), but the engine follows the documented page-alignment rule.

### 4.3 Making it resident

1. Create one `MTLResidencySet` (`MTLResidencySetDescriptor`, initial capacity 64).
2. Add every window, the converted-router arena and every other long-lived buffer, call `commit`, then `requestResidency`.
3. Attach the set to the command queue with `addResidencySet:`. No per-encoder `useResource` calls are needed.
4. **Warm-up.** Before `READY`, page the windows in with 8 threads. Each thread takes 64 MiB chunks, calls
   `madvise(MADV_WILLNEED)` on the chunk, and reads one byte per 16 KiB page. Experts are warmed in profile order, so
   with a cold page cache the hot experts land first. Then run one GPU pass that reads one 16-byte word per window
   page and converted-router arena page, to complete GPU residency. Print
   `strata: warmed N GiB in S s (X GB/s)`.
5. The first `READY` happens only after the warm-up. Expected time: 35 GiB at 7-12 GB/s is 3-5 s from a cold SSD, and
   under 1 s when the page cache is warm. Measurement M-5.

### 4.4 Under memory pressure

The page cache pages of the windows are clean and file-backed. If macOS evicted them while they were in the residency
set, the next window would fault them back in, at SSD speed. Measurement M-4 records:
- what macOS does with them under pressure;
- whether `requestResidency` wires them;
- the effect on the next window's time.

If eviction happens and costs more than 10% of decode speed under a realistic load (a browser plus an IDE), the
fallback is tiered mode's copy layout with every expert resident. Anonymous memory compresses instead of being dropped.

---

## 5. Tiered mode

### 5.1 Why the layout differs

Wrapping a shard would make all of the window resident on first use (section 1.1), so no GGUF shard is wrapped in
tiered mode. The engine copies what must be resident into anonymous memory and reads the rest on demand.

### 5.2 Layout

| Data | Placement | How it is filled |
| --- | --- | --- |
| Non-expert tensors of every shard, and the `dense.bin` rows of `conversions.json` | **Dense arena**: anonymous memory, 16 KiB-aligned, wrapped no-copy, in the residency set | `pread` through an `F_NOCACHE` descriptor, 8 threads, at startup. Tensor alignment inside the arena is 256 B. |
| Resident experts R | **Resident arena**: anonymous; one 256-B-aligned slot per resident expert, sized by its layer's blob. The three roles are packed back to back: gate, up, down. | `pread` of the three slabs per expert, through the `F_NOCACHE` descriptor, 8+ threads, profile order |
| `mtp/rt/experts.bin`, `mtp/rt/dense.bin` | Bounded no-copy windows, warmed before open succeeds; these files are always read in full | as full mode |
| Cold experts | Stay in the GGUF on disk. They are never mapped as Metal memory. | On a miss, `pread` through a second, **cached** descriptor of the shard that holds the expert. The page cache is the RAM tier. |
| Slot pool | **Slot arena**: anonymous, 2H slots of `max_blob` rounded to 16 KiB | Miss reads (section 5.4) |
| Everything else | as full mode | |

Two descriptors per shard:
- **Bulk descriptor (`F_NOCACHE`).** Startup copies and adaptive swaps use it, so a 30 GiB copy does not also fill the
  page cache.
- **Miss descriptor (cached; `F_RDAHEAD` off).** A cold expert read twice in a session comes from RAM the second time.

Startup copy time: 31 GiB at 10-12.7 GB/s is about 3 s cold. On a warm restart of a large-RAM machine the page cache
may already hold the file; `F_NOCACHE` reads are still served from cache pages that exist.

### 5.3 Address and resident tables

- The resident buffer reserves 48 × 512 `int32` entries; the address buffer reserves 48 × 512 × 3
  `uint64` entries. Active experts are packed with row stride `n_expert`: entry `l * n_expert + e`, and role address
  `((l * n_expert + e) * 3 + role)`, matching the kernel ABI in [04](04-metal-kernels.md), section 4.3. The unused
  tail of each maximum-capacity buffer is zero. In tiered mode, an active resident entry holds its slot index and a
  non-resident entry is −1; in full mode every active resident entry is 1.
  - For a resident expert the entry points into the resident arena. It is written at load and by adaptive swaps.
  - For a non-resident expert the `MissResolver` writes the slot's addresses during the layer's handshake, before it
    signals ([06](06-engine-host-port.md), section 9.3). An entry left over from an earlier window is never read,
    because the plan kernel decides residency from `res`, and every miss is rewritten.
- The **plan buffer** `plan[48]` holds, per layer, the distinct routed experts and their resident flags. The plan
  kernel writes it; the `MissResolver` reads it.
- The expert kernels read the addresses for each (row, k) from `addr`. They compute on bytes that are either in the
  resident arena or in a slot. The arithmetic is identical; gate C4 checks this.

### 5.4 Slot pool and the per-layer miss path

The pool is split into two halves of H slots. Layer l uses half `l % 2`.

**Why one half per layer is enough.** Layer l's handshake signal (`encodeSignalEvent` after the router and plan
kernel) fires only after every command encoded before it has completed. Those commands include the expert kernels of
layer l − 2, the last earlier reader of half `l % 2` in this window, and all of the previous window, which
completed before this one was committed. The resolver retires the completed layer's lease set before reusing that
half; the command-buffer completion callback releases any pins that remain when the window ends.

**Why a second half.** The lookahead may prefetch layer l + 1's predicted misses into half `(l + 1) % 2` while
layer l computes. That half was last read by layer l − 1, which has finished.

**Capacity.** One half holds every miss of one layer:
- prefill: at most H = max_l nr_l;
- decode: at most min(80, H) for T = 8 × top-10.

That is why H is defined from `nr_l` and not from T (section 3.2, step 5).

**Slot cache.** A slot keeps its (l, e) tag after use. A later miss of the same (l, e) in the same half, whose slot
has not been overwritten, skips the read. This counts as `ram_blobs` in DONE.

**Read sizes.** A miss is three `pread`s (gate, up, down slabs) of 460,800 B each for Q2_0, issued to the fetch pool.
All misses of a layer are issued at once, so the queue depth is 3 × the miss count. A single miss (3 reads) already
gives 3 in flight. Prefetch of l + 1 adds more.

**Example cost (Q2_0, decode, T = 4).**
- A window makes 4 × 48 × 10 = 1,920 routings.
- At a 97-99% hit rate (to be measured, M-9), that is about 20-60 misses, or 0.4-1.2 per layer. Fewer than that need
  reads, because rows of one window share experts.
- Two misses in one layer are 6 reads, 2.6 MiB. That takes about 0.3-0.5 ms from the SSD (between the 4.8 GB/s
  single-thread rate and 12.7 GB/s at QD ≥ 8), and about 0.05 ms from the page cache.
- So a window pays about 10-30 ms of SSD reads in the worst case, against about 30 ms of compute at T = 4.
- A layer with no misses still pays the handshake latency (M-1 in [06](06-engine-host-port.md)).

Gate G2 decides whether this is fast enough ([09-testing-and-validation.md](09-testing-and-validation.md)).

**Skipping the handshake.** If the plan kernel finds no misses in layer l, it writes `skip = 1`. The `MissResolver`
still signals `base + 2l + 2` without reading anything. Removing the wait from such layers would need GPU-side control
flow over `encodeWaitForEvent`, which Metal does not offer. An ICB-based variant could skip encoding the wait when the
whole window is a hit. That is a later optimisation, listed in section 11.

### 5.5 Prefill in tiered mode

A prefill chunk of C ≥ 1024 tokens touches nearly every expert of a layer: C × 10 routings over 512 experts. It
uses the same per-layer handshake, with the plan kernel listing the distinct experts of all C rows. A half holds
H ≥ nr_l slots, so a whole layer's misses always fit.

**Prefetch, whole layer.** Routing for layer l + 1 is not known while layer l computes. But with C ≥ 1024 nearly all
of layer l + 1's non-resident experts will be used. So as soon as layer l's handshake is released, the resolver
reads **all** nr_(l+1) non-resident experts of layer l + 1 into half `(l + 1) % 2`. At layer l + 1's handshake it
reads only the stragglers, usually none.

| Case | Chunk | Non-resident per layer | Bytes per layer | Read at 12.7 GB/s | Compute per layer (estimate) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q2_0, 48 GB Mac, 90% resident | 2048 | about 51 | 67 MiB | about 5.5 ms | about 20-40 ms |
| Q2_0, 32 GB Mac, 51% resident | 1024 | about 250 | 330 MiB | about 27 ms | about 10-20 ms |

- **48 GB Macs.** The reads hide behind compute.
- **32 GB Macs.** Prefill becomes read-bound, at about 1.3 s per 1024-token chunk (about 800 tok/s), unless the page
  cache already holds the cold experts.
- **Without the prefetch**, the reads would be serial: about 0.26 s extra per chunk on the 48 GB Mac and 1.3 s on the
  32 GB Mac.

Measurement M-10 checks these estimates.

### 5.6 Adaptive swaps

The policy and constants are in [06](06-engine-host-port.md), section 9.4. Memory side:

- A swap moves an expert **into** the resident arena by overwriting the victim's place. Both are in the same layer,
  and their blob sizes are equal because the size is per layer.
- Ordering ([06](06-engine-host-port.md), section 9.4):
  1. Between windows, while no command buffer is in flight, set `res[victim] = −1`. From then on a routing to the
     victim takes the miss path.
  2. The adapt thread `pread`s the incoming blob through the bulk (`F_NOCACHE`) descriptor, directly into the
     victim's place. No window reads that place any more, so no staging copy is needed.
  3. Between windows, after the read has landed, set `addr[in]` to the place and `res[in]` to its index.
- Usage counters: `usage[48][512]` float, written by the plan kernel (+1 per (row, k) routing), read and decayed
  (× 0.7) by the adapt thread between windows.

### 5.7 Lookahead

The upstream `RouterLookahead` predicts layer l + 1's experts from layer l's hidden state
(`expert_source.hpp:152`; `STRATA_LOOKAHEAD`, `STRATA_LOOKAHEAD_K`). On a Mac, for each predicted non-resident
expert, the resolver does one of two things:

1. If the expert's bytes are already in the page cache (`mincore` on its range), do nothing; the read will be fast.
2. Otherwise issue `pread`s into the other slot half, tagged "speculative". If layer l + 1 really misses it, the slot
   is used as is.

Wrong predictions waste only SSD bandwidth, which is otherwise idle during compute.

### 5.8 DONE accounting

| Counter | Rule |
| --- | --- |
| `hits` | (row, k) routings served by the resident arena |
| `lookups` | all (row, k) routings |
| `ram_blobs` | misses served by a slot-cache hit, or a speculative prefetch that had landed, or pages found in the page cache by `mincore` |
| `file_blobs` | misses that needed SSD reads |
| `file_MB` | bytes read for `file_blobs` |

`mincore` is checked before the read, on the expert's three ranges, rounded to the system page size (16 KiB on
supported Apple silicon). It reflects the state at the moment of the check, which is accurate enough for statistics.
If the query fails, the read still proceeds and the blob is conservatively counted as a file read.

---

## 6. Things that must never be wrapped

1. **The PLE tensor payload** (28.8 GB: shard 2 of the ISTA and Coder files, a range of shard 1 in the Swift files; or the
   51.2 GB FP8 table) is never assigned a Metal window or exposed as a GPU tensor binding. Outward page alignment may
   cover up to 16 KiB of its edge when the preceding or following read tensor shares that page; those bytes are
   `wrapped_unread` padding and the PLE kernel never reads them through the window. Rows are read with `pread` by
   `DirectFile` (`--ple-io direct`, `F_NOCACHE`) or copied from a plain mapping (`--ple-io mmap`). Full mode can share
   the already-live `GgufModel` mapping through `PleTable::open_mapped`; Direct reads must wait until no mapping of that
   shard remains live. Rows are then written into the window's PLE staging buffer. The PLE kernel never sees the table.
2. **Any GGUF shard in tiered mode.**
3. **Anything the plan does not count.** In full mode, the loader re-plans the complete source inventory and checks
   that the union of all windows equals the page-aligned ranges in the memory plan (section 3.2, `wrapped_unread`,
   plus the tensors read). Every GGUF tensor must occur once as a window source or as an explicit PLE/conversion
   exclusion, and every converted exclusion must match exactly one validated `dense.bin` row and one arena binding.
   Unlisted pack bytes are never read. Windows split at `maxBufferLength` may overlap by up to 16 KiB at the split;
   the union counts those bytes once. The tiered loader validates the complete source inventory before allocating
   and uses anonymous no-copy arenas; `metal_tiered_copy_test` checks the resulting Metal allocation set contains no
   GGUF-backed buffer.

Violating rule 1 or 2 would, on the first command buffer, page the whole file into memory and push the machine into
heavy memory pressure. `metal_never_wrap_test` checks the full-mode loader's PLE and union rules on a live synthetic
mmap. `metal_tiered_copy_test` checks the backing region names of all six tiered buffers against the known GGUF
mapping and verifies that the bulk descriptor copies only inspected tensor ranges. The miss descriptor is opened
separately with read-ahead disabled; U6 implements its runtime reads.

---

## 7. Buffers, storage modes and alignment

| Rule | Detail |
| --- | --- |
| Bump arena | Long-lived anonymous buffers are carved from a few large `newBufferWithLength:options:` allocations, one per role: state, scratch, prefill, slots, resident, dense. Carving happens once at startup, so addresses never change (needed for residency sets and ICBs). No buffer is allocated per request or per token. |
| Storage mode | `MTLResourceStorageModeShared` everywhere by default, so the host can read and write `contents` directly. That covers snapshots, PLE staging, inputs and outputs. `MTLResourceStorageModePrivate` only for buffers the CPU never touches, and only where measurement M-6 shows a gain. A Private buffer needs a blit for snapshots. |
| Hazard tracking | `MTLResourceHazardTrackingModeUntracked` for every long-lived buffer. Metal then orders nothing by itself: every compute encoder waits for the queue's `MTLFence` and updates it, and a command buffer that reads what an earlier one wrote is wrong without it (20 of 20 runs, [appendix A.11](appendix-a-measurements.md), test 4). The rules are [04](04-metal-kernels.md), section 8, and [06](06-engine-host-port.md), section 8.2, rule 7. |
| CPU cache mode | `MTLResourceCPUCacheModeDefaultCache`. Write-combined mode only for upload-only staging, if a measurement shows a gain. |
| Alignment | 16 KiB for no-copy bases and lengths and for the full-mode converted-router arena length. 256 B for tensors and slots in arenas. GGUF in-place tensors (full mode) are only 32-B aligned (`general.alignment`), so expert and dense kernels must accept 32-B-aligned blob addresses. |
| Poison | Opt-in (`STRATA_METAL_POISON=1`, [06](06-engine-host-port.md) section 4.4; the tests set it): fill each new allocation with `0xFF` once, at creation. Owners that require zero-initialized state clear their regions afterward. Unwritten float reads become NaNs; integer and byte reads see all-ones values. Upstream has a similar opt-in, `DeviceArena(poison = true)` (`device.cu:22-25`, `:197-218`), used only by `device_main.cpp:85`. |
| Zeroing | `session_zero` uses `fillBuffer:range:value:0` on a blit encoder, or `memset` on `contents` when the queue is idle |
| Labels | Every buffer gets a label (`strata dense`, `strata kv`, `strata slots`, ...) for captures and Instruments |

---

## 8. Example plans

These are estimates from the formulas above. They assume:
- D_pack 0 (Swift 0.12 GiB), MTP 0.77 GiB, and F_fixed with `mtp_bind_bytes` 0.25 GiB: 1.02 GiB together (Swift
  1.13 GiB). The reserve is 0.68 GiB.
- `wrapped_unread` is 0, except 0.23 GiB for Swift in full mode (its F32 routers).
- The KV figure is the main KV plus the drafter KV (section 2.2).
- In tiered mode, H = 1.15 × the mean non-resident count per layer, for profile skew between layers, and resident
  experts have the file's mean blob size.

W for 32, 36 and 48 GB Macs is an estimate (M-8). `metal_plan_test` keeps these rows as golden cases, and
`--plan-only` on real machines replaces them.

### 8.1 64 GB (W = 51.84 GiB measured, B = 51.16 GiB)

| Pack | Context, KV (GiB) | Prefill chunk (GiB) | Total (GiB) | Mode | Free (GiB) |
| --- | --- | ---: | --- | --- | ---: |
| Q2_0 | 262,144 fp16 (6.44 + 0.51) | 8192 (2.19) | 3.39 + 1.02 + 6.95 + 2.19 + 31.64 = 45.18 | full | 5.98 |
| Q2_0 | 65,536 fp16 (1.61 + 0.13) | 8192 (2.14) | 39.93 | full | 11.23 |
| IQ2_XS | 131,072 fp16 (3.22 + 0.25) | 8192 (2.16) | 3.50 + 1.02 + 3.47 + 2.16 + 33.02 = 43.17 | full | 7.99 |
| Swift IQ2_XS | 131,072 fp16 (3.22 + 0.25) | 8192 (2.16) | 3.50 + 0.23 + 1.13 + 3.47 + 2.16 + 33.02 = 43.52 | full | 7.64 |
| IQ3_XXS | 131,072 k8v4 (1.41 + 0.13) | 8192 (2.16) | 3.83 + 1.02 + 1.55 + 2.16 + 39.97 = 48.52 | full | 2.64 |
| IQ3_XXS | 262,144 k8v4 (2.83 + 0.27) | 8192 (2.19) | 50.10 | full | 1.06 |
| IQ3_S | 65,536 k8v4 (0.71 + 0.07) | 2048 (0.61) | 4.20 + 1.02 + 0.77 + 0.61 + slots 0.15 + resident 44.40 | tiered: 95% of experts resident, about 1,300 on SSD | 0 |
| Coder IQ1_M | 262,144 fp16 (6.44 + 0.51) | 8192 (2.19) | 4.15 + 1.02 + 6.95 + 2.19 + 23.42 = 37.72 | full | 13.44 |

### 8.2 48 GB (W ≈ 37 GiB estimated, B ≈ 36.32 GiB)

The ISTA and Swift files run in tiered mode:

| Pack | Context, KV (GiB) | Prefill | A + P + S (GiB) | Resident experts (GiB) | On SSD |
| --- | --- | ---: | ---: | --- | ---: |
| Q2_0 | 32,768 fp16 (0.80 + 0.06) | 2048 | 3.39 + 1.02 + 0.87 + 0.61 + 0.06 = 5.94 | 30.38 of 31.64 (96%) | 4%, about 1,000 experts |
| Q2_0 | 65,536 k8v4 (0.71 + 0.07) | 2048 | 5.85 | 30.47 (96%) | 4% |
| Q2_0 | 131,072 k8v4 (1.41 + 0.13) | 2048 | 6.68 | 29.64 (94%) | 6% |
| IQ2_XS | 32,768 k8v4 (0.35 + 0.03) | 2048 | 5.63 | 30.69 of 33.02 (93%) | 7% |

Coder IQ1_M runs in full mode: at 65,536 k8v4 cells (0.71 + 0.07) with an 8192 chunk the total is
4.15 + 1.02 + 0.77 + 2.14 + 23.42 = 31.50 GiB, which leaves 4.82 GiB.

### 8.3 36 GB (W ≈ 27 GiB, B ≈ 26.32 GiB) and 32 GB (W ≈ 24 GiB, B ≈ 23.32 GiB), estimated W

All rows run in tiered mode. `B − A` is under 24 GiB, so the prefill chunk is 1024.

| Machine, pack | Context, KV (GiB) | A + P + S (GiB) | Resident experts (GiB) | On SSD |
| --- | --- | ---: | --- | ---: |
| 36 GB, Q2_0 | 32,768 k8v4 (0.35 + 0.03) | 3.39 + 1.02 + 0.39 + 0.35 + 0.53 = 5.68 | 20.64 of 31.64 (65%) | 35%, about 8,500 experts |
| 32 GB, Q2_0 | 32,768 k8v4 (0.35 + 0.03) | 5.83 (slots 0.69) | 17.49 (55%) | 45% |
| 32 GB, Q2_0 | 8,192 k8v4 (0.09 + 0.01) | 5.52 | 17.80 (56%) | 44% |
| 32 GB, Coder IQ1_M | 32,768 k8v4 (0.35 + 0.03) | 4.15 + 1.02 + 0.39 + 0.35 + 0.40 = 6.30 | 17.01 of 23.42 (73%) | 27% |

On 32-36 GB Macs the slot pool (0.4-0.7 GiB) is noticeable: every 2 × H slots it reserves would otherwise hold
experts. Splitting a prefill layer's misses into batches would cut the pool to 2 × 80 slots (about 0.2 GiB). That
change is listed in section 11.

What the hit rate means: with a frequency profile, the resident share of *routings* is far above the resident share
of *experts*. Upstream measured h = 0.6447 with 4,105 slots (17% of experts) in a ten-fold leave-one-out test
([audit K5 2.1](audit/K5-orchestration-vram-tests.md)). The upstream comment cites `tools/routing_kfold.py`
(`expert_cache.hpp:11`), but that script is not in the repository. The Mac plans above hold
55-96% of experts, where the curve should give well over 90% hits. That is an extrapolation, not a measurement. Measurement M-9 records the real
curve on a Mac with a new script, `tools/hit_curve.py` (section 10.1). It is the number gate G2 depends on.

### 8.4 96 GB and up

Everything is in full mode, IQ3_S included, with a long context. UD-Q4_K_XL is not supported on any Mac
([03](03-model-reference.md), section 2.2). These machines are supported but not tuned ([00](00-feasibility-study.md), section 4.1).

### 8.5 Raising W

macOS lets an administrator raise the GPU wired limit with `sudo sysctl iogpu.wired_limit_mb=<MiB>`. The value resets
at reboot. Metal then reports a higher `recommendedMaxWorkingSetSize` (verify in M-8). The engine never changes the
setting. Setup may print it as advice when the plan is tiered and RAM would allow more:
`raising iogpu.wired_limit_mb to N would keep all experts resident`. The documentation must warn that the OS and other
apps then have less memory.

---

## 9. Interplay with macOS memory management

| Topic | Rule |
| --- | --- |
| Memory pressure | The engine installs a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source and frees conversation parking and non-root checkpoints at `CRITICAL` ([06](06-engine-host-port.md), section 11.4). It never shrinks the GPU arenas while running. |
| Swap and compression | Anonymous arenas can be compressed or swapped by macOS if they are not wired. The plan's RAM warning (section 3.2, step 6) is the only defence; measurement M-4 tells how bad this is. |
| `mlock` | Off by default. `STRATA_ARENA_LOCK=1` tries it on the resident arena and warns on failure ([06](06-engine-host-port.md), section 7.5). |
| App Nap / timer coalescing | `beginActivityWithOptions` during requests ([06](06-engine-host-port.md), section 2.2) |
| Several engines | Each engine process plans against the whole W, so two can each fit alone but not together. Decided in [07](07-server-setup-port.md), section 6.8: one engine per Mac is the supported setup, and setup warns when another engine runs. An expert can still split W by hand with `--gpu-budget-mib` set to what the first engine leaves (its PLAN line shows `free_mib`). |

---

## 10. Measurements (memory side)

| ID | What | How | Drives |
| --- | --- | --- | --- |
| M-3 | The reserve that keeps a 64K session stable with other apps running | Run `serve` with 64K context and a browser plus an IDE open; watch `currentAllocatedSize`, memory pressure and GPU errors | `--vram-reserve-mib` default |
| M-4 | No-copy file pages and anonymous arenas in a residency set under memory pressure: wired, evicted, compressed? | Fill memory with a second process; watch `vm_stat`, `footprint`, `mincore` on windows, decode speed | Full mode stays no-copy, or moves to copy |
| M-5 | Warm-up time, cold and warm page cache | Section 4.3 step 4, on the reference Mac | Startup UX; whether setup shows a progress bar |
| M-6 | `Private` vs `Shared` for KV and state | A/B on decode and prefill | Storage mode per buffer |
| M-7 | `mlock` limits on macOS 27 | Try 1, 4, 16 and 32 GiB | Whether `STRATA_ARENA_LOCK` is useful |
| M-8 | W and `maxBufferLength` on 32, 36 and 48 GB Macs; effect of `iogpu.wired_limit_mb` | `strata --list-devices` on each machine, or the published values from other engines' logs | Section 8 tables; product claims |
| M-9 | Expert hit-rate curve vs resident share, with the shipped profile | `tools/hit_curve.py` (section 10.1) on Mac traces of the benchmark prompt sets | Gate G2 forecast; default `--prefill` chunk in tiered mode |
| M-10 | Whole-layer prefill prefetch in tiered mode (section 5.5): read time vs compute per layer | A 32K prompt with `--gpu-budget-mib` set for 90% and 50% residency; prefetch on and off | Whether the 32 GB class needs batched prefill misses (section 11) |

### 10.1 `tools/hit_curve.py` (new)

Upstream does not ship the script behind its 0.6447 figure, so the port adds one. It needs no GPU and runs on any host.

```
python tools/hit_curve.py TRACE ... [--profile data/expert-profile.bin] [--n-expert 512]
                          [--resident 2048,4105,8192,12288,16384,20480,24576] [--folds 10] [--per-layer]
```

- Input traces are `--dump-routing` files. Each record is `int32 layer, int32 k`, then k int32 expert ids, then k
  float32 weights. `tools/make_profile.py` already has the reader (`read_trace`); import it.
- For each R in `--resident`, the resident set is the first R pairs of the profile. Profile files are `STRP`: version,
  n_layers, n_expert, slots and n_ranked as five uint32 after the magic, then n_ranked × (uint16 layer, uint16 expert),
  then a slot table that the reader ignores. `make_profile.read_profile` reads them.
- Output, one line per R: `R share h_expert h_layer misses_per_token`.
  - `h_expert` is resident routings over all routings.
  - `h_layer` is the share of (layer, token) pairs whose routed experts are all resident.
- `misses_per_token` is the mean number of non-resident route occurrences per token over all 48 layers. It is a
  routing-miss metric, not an SSD read count: it does not model per-layer route deduplication, slot-cache hits,
  page-cache hits or prefetch.
- `--folds K` repeats the run as a leave-one-out over K trace groups. Fold i builds a profile with
  `make_profile.py --no-base` from the other trace files and measures that held-out file. Each file is one group;
  exactly K files are required. It prints the mean and range for each metric, as the upstream comment does for
  0.6447 [0.6110, 0.6750].
- Every trace file must contain the same nonzero number of routing records for each of the 48 layers. This makes the
  per-token route-miss count well-defined even when a prefill trace is layer-major. Fold inputs must identify distinct
  regular files and distinct routing records regardless of record order or unused weights; this catches aliases and
  copies. Fold setup opens each file once, snapshots it into a private temporary directory, checks for changes while
  copying, then uses only the snapshots for fingerprints, training and evaluation. Non-regular inputs are refused
  after a nonblocking open where supported, so FIFOs cannot hang the run.
- `--per-layer` also prints H = max over layers of (n_expert − resident experts in that layer) for the first R
  profile pairs. This is the slot-pool bound of section 5.4, so the 1.15 skew factor of section 8 can be replaced by
  the exact value for the measured resident prefix.
- Routing records are checked against the producer contract: 1-64 unique expert ids, in-range ids, complete F32
  weights, and finite non-negative weight values. The metric ranks each route equally; weight values are validated
  but do not weight the frequency count.
- Unit tests: `tools/test_make_profile.py` checks runtime-compatible STRP headers, ranking pairs and routing records;
  `tools/test_hit_curve.py` checks exact hit, layer-hit, miss, H and leave-one-group-out values on synthetic traces.

---

## 11. Deferred optimisations

| Item | Gain | Why later |
| --- | --- | --- |
| Lend resident experts to prefill on small Macs: drop the coldest resident experts to fit a larger chunk, then re-read them | Faster long prompts in tiered mode | Upstream's slot lending, simpler here because a refill is a `pread`. Needs the base path first. |
| Batch a prefill layer's misses through a small pool (2 × 80 slots) instead of a whole-layer half | Frees 0.2-0.5 GiB for experts on 32-36 GB Macs | More handshakes per prefill layer; only worth it if M-10 shows prefill is not read-bound anyway |
| Skip the handshake for all-hit windows with ICBs | Removes 48 waits from most windows in tiered mode | Needs ICB encoding (`STRATA_METAL_ICB`) |
| Per-conversation KV arenas switched by pointer | Instant conversation switching | [06](06-engine-host-port.md), section 11.3 |
| Wire the resident set | Protects tiered mode from swap | Depends on M-4 and M-7 |

---

## 12. Upstream mechanisms deleted by this design

| Mechanism | Upstream location | Reason |
| --- | --- | --- |
| VRAM expert cache: slot arena, `fill_slot*`, `verify_slot`, `admit`, sized slots | `expert_cache.cpp:72-372`; `generate.cpp:2666-2880` | Replaced by the resident set (full: all; tiered: profile-ranked, copied once) |
| `--expert-cache auto` VRAM formula | `generate.cpp:2672-2680` (block `:2664-2686`) | Replaced by the plan (section 3) |
| Slot lending to prefill and refill | `generate.cpp:3714-3770`, `:5655-5737` | Deferred (section 11) |
| Exchange buffers, `resident_stage_swaps`, `commit_exchanges` | `resident_stage_swaps` at `generate.cpp:122-158`; `reserve_exchanges`, `stage_exchange`, `commit_exchanges` at `expert_source.cpp:1341-1403` | No CPU compute of experts |
| Partial pinning, hugetlb, large pages, `cudaHostRegister` | `pinned.cu:32-34` (`MAP_HUGE_2MB`), `:54-213` (the arena), `:306-361` (`cudaHostRegister`) | Unified memory |
| KV streaming (`--kv-resident`, `kv_stream.cu`, host mirror) | `kv_plan` at `layer.cpp:495-513`; host mirror at `:627-657`; flag parsed at `generate.cpp:1063` | KV is resident |
| PCIe staging rings, copy streams and stager threads in prefill | `prefill.cpp:162-286`, `:541-597`, `:1179-1251` | The GPU reads weights in place |
| 48 host syncs per prefill chunk for grouping tables | the sync at `prefill.cpp:1592` (block `:1583-1615`), once per layer | GPU counting sort, `strata_pf_sort_*` ([04](04-metal-kernels.md), section 6.6) |
| WDDM page-file and commit-limit handling | `generate.cpp:2749-2815`; commit-limit logic at `:2768-2793` | Windows only |

---

## 13. Tests for this area

| Test | Checks |
| --- | --- |
| `metal_plan_test` | Section 3 on synthetic inputs: full/tiered choice, prefill chunk choice, slot pool iteration, error messages, the PLAN line keys. Includes every row of section 8 as a golden case with fixed W. |
| `metal_window_test` | Window planning (section 4.2): 16 KiB rounding, merge rule, splits at expert-slab boundaries when `maxBufferLength` is forced small (test hook), and preservation of the merged-range union across an oversized unread gap. On the header fixtures of [03](03-model-reference.md) work item R1: one window for Q2_0 shard 1, none for the PLE shard, two for Swift shard 1 with no PLE tensor mapping (page-alignment padding may cover up to one page at an adjacent tensor boundary), and `wrapped_unread` of 252,149,760 B plus padding for Swift. |
| `metal_window_storage_test` | Sections 4.1 and 4.3: a nine-expert synthetic GGUF spanning multiple VM pages is mmaped, wrapped in page-rounded no-copy MTLBuffer, and mapped to the exact expert address table; checks the full-mode shared-mapping IQ4 PLE reader through `ModelStorage`, tiered-mode reader absence, exact converted-router arena bytes, immutable inspection-time row mapping, 16 KiB padding, malformed/overlapping ranges, layer bounds, rollback and warm-up, plus the shared maximum-capacity address/resident buffers and mmap lifetime. `metal_window_test` checks profile order across wave boundaries, multiple converted-router row bindings, pack-range overlap and `maxBufferLength`, and compact row indexing with a two-layer, 256-expert Coder layout. |
| `metal_never_wrap_test` | Section 6: the live full-mode loader requires the shard inventory and `wrapped_unread` total to match the memory plan, accounts for every GGUF tensor exactly once as included or intentionally excluded, excludes the PLE tensor from sources, slices and addresses, resolves the converted F32 router only to its BF16 arena binding, and allows at most one 16 KiB page of overlap at the PLE boundary. |
| `metal_tiered_copy_test` | On a small synthetic GGUF: the dense and resident arenas hold exact source bytes (byte compare), including converted `dense.bin` router rows; the tiered opener configures both descriptors and address/resident/plan tables match the layout; region-name queries prove no tiered buffer is backed by the GGUF and fail closed if a reported path cannot be compared; same-size source replacements, malformed dense-pack ranges/sizes and pending-generation markers fail before allocation. `metal_weight_store_test` checks rejection of held source-file/source-set/pack locks and pending generations, and that returned leases keep blocking writers through storage opening. |
| `metal_slot_pool_test` | Ping-pong halves, Loading-to-Ready publication, duplicate-load refusal, abandoned-load retry, LRU tags, and no slot reuse until the layer-complete event; the command-buffer callback releases remaining leases, and the test gates the GPU with an arrival/release event pair. |
| `metal_full_vs_tiered` (gate C4) | The same prompts, greedy and seeded, give byte-identical tokens and DONE counts (except the tier counters) in full mode and in tiered mode with `--gpu-budget-mib` forcing 50% and 90% residency ([09-testing-and-validation.md](09-testing-and-validation.md)) |
| `metal_memory_pressure_smoke` | The M-4 procedure; records its numbers. Runs weekly, and on demand ([09-testing-and-validation.md](09-testing-and-validation.md), section 5.5). |

---

## 14. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). An item is done when its
"Done when" holds on the reference Mac. The host control flow of tiered mode is [06](06-engine-host-port.md) work
item H10, and the kernels that read these buffers are [04](04-metal-kernels.md) work items L7 and L12.

| # | Item | Done when |
| --- | --- | --- |
| U1 | `Geometry` and `Plan` (`src/metal/engine/geometry.cpp`, `plan.cpp`, [06](06-engine-host-port.md), section 1.5): the inputs, the algorithm, the admission rules and the PLAN line (section 3), with [06](06-engine-host-port.md) H5 for the light path | `metal_plan_test` passes, with every row of section 8 as a golden case; `--plan-only` on the reference Mac prints a PLAN line with every key of section 3.4 for each model, and a plan that does not fit exits 3 with its `reason` |
| U2 | Window planning (section 4.2) | `metal_window_test` passes on the header fixtures of [03](03-model-reference.md) work item R1 |
| U3 | Full mode (sections 4.1 and 4.3): the file windows, the address and resident tables, the residency set, the warm-up | Q2_0 at 65,536 fp16 cells starts on the reference Mac in full mode with the plan of section 8.1 and prints the warm-up line; M-5 is in appendix A |
| U4 | The never-wrap rules and the loader's union check (section 6) | `metal_never_wrap_test` and the tiered no-GGUF assertion in `metal_tiered_copy_test` pass |
| U5 | The tiered layout (sections 5.2 and 5.3): the dense and resident arenas filled through the bulk descriptor, the miss descriptor, `res`, `addr` and `plan` | `metal_tiered_copy_test` passes |
| U6 | The slot pool, its halves and the slot cache; miss reads; prefill prefetch; lookahead; DONE accounting (sections 5.4, 5.5, 5.7 and 5.8), with [06](06-engine-host-port.md) H10 | `metal_slot_pool_test` and synthetic ExpertStore coverage in `metal_tiered_copy_test` pass; production layer-handshake parity and `metal_full_vs_tiered` remain open under REV-024 |
| U7 | Adaptive swaps, memory side (section 5.6), with [06](06-engine-host-port.md) H10 | Gate C4 passes, including the arm of `accept.py c4` in which experts are swapped during the run ([09](09-testing-and-validation.md), section 4.4) |
| U8 | macOS interplay (section 9): the memory-pressure source with [06](06-engine-host-port.md) H9, `STRATA_ARENA_LOCK`, the RAM warnings of section 3.2, step 6 | `metal_memory_pressure_smoke` passes ([09](09-testing-and-validation.md), section 5.5, with work item T13); with `STRATA_ARENA_LOCK=1` and a resident arena above the `mlock` limit, the engine warns and runs |
| U9 | `tools/hit_curve.py` and `tools/test_hit_curve.py` (section 10.1) | `test_hit_curve.py` passes on Linux; the tool prints the curve for Mac traces of the benchmark prompt sets |
| U10 | Measurements M-3 to M-10 (section 10). M-1 and M-2 are [06](06-engine-host-port.md), section 15. | Each result is in appendix A, and the default it drives is updated here |

---

## 15. Where the details are

| Topic | This document | Audit and measurements |
| --- | --- | --- |
| Measured facts | 1.1 | [appendix A.1-A.4](appendix-a-measurements.md) |
| Memory consumers and their sizes | 2 | [03](03-model-reference.md), sections 2.4, 4.1 and 4.5; [K2 section 3](audit/K2-qsa-kv-gdn.md) |
| The plan and the PLAN line | 3 | [K5 sections 2 and 4.1-4.3](audit/K5-orchestration-vram-tests.md) |
| Full mode | 4 | [appendix A.3](appendix-a-measurements.md) |
| Tiered mode | 5 | [K5 sections 2.1, 2.7 and 4.9](audit/K5-orchestration-vram-tests.md); [appendix A.4](appendix-a-measurements.md) |
| What is never wrapped | 6 | [appendix A.3](appendix-a-measurements.md) |
| Buffers, storage modes, ordering | 7 | [appendix A.11](appendix-a-measurements.md) |
| Example plans | 8 | [00](00-feasibility-study.md), section 4.1 |
| macOS memory management | 9 | [B section 3](audit/B-engine-host.md) |
| Upstream mechanisms deleted | 12 | [K5 section 3.5](audit/K5-orchestration-vram-tests.md) |
