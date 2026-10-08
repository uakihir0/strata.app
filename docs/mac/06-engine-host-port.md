# 06. Engine host port: the `strata` binary on macOS

Status: implementation-ready specification, 2026-10-07.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Source audits: [audit/B-engine-host.md](audit/B-engine-host.md) (host inventory) and
[audit/K5-orchestration-vram-tests.md](audit/K5-orchestration-vram-tests.md) (decode orchestration, VRAM assumptions).
Every `file:line` reference is relative to the upstream repository root.

This document covers everything in the engine process that is not GPU kernel math:
- the driver and command line;
- environment variables;
- the serve protocol;
- artifacts and loaders;
- OS memory and file APIs;
- the CUDA host runtime and its Metal replacement;
- the host side of expert residency;
- speculative decoding;
- the conversation cache;
- platform sites;
- telemetry and the watchdog.

Kernel contracts are in [04-metal-kernels.md](04-metal-kernels.md), and memory placement is in
[05-memory-and-residency.md](05-memory-and-residency.md). The overall structure is in
[02-architecture.md](02-architecture.md).

Normative words: **must** means required for conformance with the upstream engine contract. **Should** means the
recommended default, which a measurement may overturn.

---

## 1. Scope and strategy

### 1.1 What the upstream host does, in one paragraph

`src/program/generate.cpp` (6,487 lines) parses options, loads the model, and either generates once or runs `--serve`.

**Token graph.** A token is decoded by one captured CUDA graph over all 48 layers. Inside it, each layer's router
publishes its top-10 experts to mapped host memory and rings a doorbell. A device kernel then spins on a host-written
flag while the host CPU pool computes the experts that are not in VRAM.

**Verify windows.** With `--spec`, the same scheme runs T ≤ 8 tokens at once in a verify window. It adds:
- two more flags (`flagA`, the GPU plan; `flagB`, PCIe DMA landed);
- an asynchronous commit;
- an MTP drafter;
- a prompt-lookup drafter.

**Policy around it.** Expert tiers (VRAM, pinned RAM, page cache, SSD), PCIe share tuning, multi-GPU layer split,
Windows driver workarounds, the conversation cache and a watchdog.

Details: [audit/B-engine-host.md](audit/B-engine-host.md) section 0.4 and
[audit/K5-orchestration-vram-tests.md](audit/K5-orchestration-vram-tests.md) section 1.

### 1.2 What changes on a Mac

| Upstream mechanism | Mac decision | Why |
| --- | --- | --- |
| CPU expert pool computes VRAM misses | **Deleted in the first release.** All routed experts run on the GPU. | CPU and GPU share one DRAM limit, about 284 GB/s measured ([appendix A.2](appendix-a-measurements.md)). A CPU path adds no bandwidth, and it makes output depend on residency. |
| Doorbell / `flagA` / `flagB` / device spin on host flags | **Deleted.** Never spin a GPU thread on CPU-written memory. | Metal gives no forward-progress guarantee for that pattern. A long command buffer can also be killed by the GPU watchdog ([audit B 4.10](audit/B-engine-host.md)). |
| Separate token graph (T = 1) and verify graphs (T = 2..8) | **One decode path:** the verify window, with T = 1 for plain decode | Bit-exactness between speculative and plain decode becomes true by construction ([audit B 5.6](audit/B-engine-host.md), point 1) |
| CUDA graph per T | Re-encode one command buffer per window. An indirect command buffer per T is an optional later optimisation. | Measured 0.78 µs per dispatch and about 0.1 ms CPU encode per 1,000 dispatches ([appendix A.5](appendix-a-measurements.md)) |
| VRAM expert cache plus adaptive swaps | Replaced by the **resident expert set** in unified memory, with two modes: *full* (all experts resident) and *tiered* (hot set resident, cold experts read from SSD into a slot pool) | [05-memory-and-residency.md](05-memory-and-residency.md) |
| PCIe share, DMA staging, `probe_pcie_h2d_gbps` | Deleted | No PCIe link |
| Multi-GPU (RemoteExperts, layer split, `OnDevice`) | Deleted | One GPU |
| WDDM flushes, `CUDA_MODULE_LOADING=EAGER` | Deleted; all pipelines are built at startup instead | Windows/CUDA only |
| Serve protocol, prompt cache policy, spec policy, conversation cache, watchdog model | **Kept, byte-compatible** | `serve/server.py` must run unchanged against either engine |

### 1.3 Strategy: a slim Metal driver, not a translation of `generate.cpp`

Do **not** port `generate.cpp` line by line. About half of it is policy for multi-GPU, PCIe, VRAM sizing and Windows
([audit B 10.3](audit/B-engine-host.md), item 7).

Instead, the project adds a new Metal driver under `src/metal/program/`. It copies or reuses these parts of the upstream
driver, keeping upstream names and behaviour:

| Upstream part | Lines | How the Mac driver uses it |
| --- | --- | --- |
| `struct Options` | `generate.cpp:160-415` | Copied into `src/metal/program/options.hpp`. Every field is kept, so flag defaults stay identical. Fields with no Mac meaning remain and are ignored. |
| Argument parser | `generate.cpp:999-1255` | Copied into `src/metal/program/options.cpp`, with the dispositions in section 3 |
| Serve loop: reader thread, request parse, BusyScope, watchdog start, INFO/READY | `generate.cpp:4510-4741` | Re-implemented with the same structure and the same output strings |
| Prompt-cache policy: checkpoint cut points, `checkpoint_at`, eviction | `generate.cpp:4318-4410`, `:5212-5240` | Copied; it is plain C++ over `conv_cache.hpp` |
| Resume selection | `generate.cpp:4858-5009` | Copied |
| Spec round loop | `generate.cpp:5322-5430` (serve), `:6128-6232` (one-shot) | Copied; calls the Metal `Window` and `Mtp` with the upstream method names |
| Request finish, DONE line and stderr summary | `generate.cpp:5560-5605` | Copied; field meanings remapped in section 5.6 |
| `--stats` block | `generate.cpp:6236-6470` | Speculation lines kept verbatim; GPU lines replaced (section 13) |

Copied code is marked with a header comment: `// synced from upstream generate.cpp:<range> @ <commit>`.

During an upstream compatibility review ([10-implementation-plan.md](10-implementation-plan.md), section 4), run
`tools/metal/check_drift.py`. CI must fail when any of the following changes upstream:
- a new `--flag` string inside the parser range;
- a new `STRATA_*` `getenv` anywhere in `src/` or `include/`;
- a new `INFO` key, `ERR` string or `DONE` field;
- a change in a copied range's hash;
- a new or changed CUDA kernel, a model added to `MODELS`, or a changed pinned Hugging Face revision.

Each failure is resolved by classifying the item in this document's tables and updating the copy.

### 1.4 Upstream code reused without copying

These compile on macOS arm64 as they are, or with the small `__APPLE__` patches in section 7. All of them are verified
in [audit B 7.2](audit/B-engine-host.md).

| Code | Status on macOS | Used for |
| --- | --- | --- |
| `include/strata/artifact/*` (`gguf_reader.hpp`, `gguf_split.hpp`), `src/artifact/dequant.cpp` | builds, tests pass | GGUF parsing, `strata-gguf`, `strata-dequant` |
| `src/spec/*` (suffix drafter, draft policy, controller) | builds, tests pass | Draft selection; used unchanged |
| `include/strata/core/coupled_draft.hpp`, `include/strata/kernels/sampler.hpp` (host-callable sampler chain) | header-only, CUDA-free | Coupled drafts; host reference for sampler parity tests |
| `include/strata/program/conv_cache.hpp`, `logits_selection.hpp` | header-only | Checkpoint eviction; logits selection |
| `include/strata/core/conversation_cache.hpp`, `conversation_buffer.hpp`, `progress.hpp` | header-only, tests pass | Conversation parking, progress beats |
| `src/ngram/ple_reader.cpp`, `src/kernels/ngram.cpp` | compile; `ngram.cpp:263` hardcodes a 4096 page | PLE row reads |
| `src/platform/direct_file.cpp` | **fails** at `:308` (`O_DIRECT`) | Unbuffered PLE reads; needs the patch in section 7.4 |
| `src/platform/memory.cpp` | compiles | `lock_resident`, total RAM; extended in section 7.2 |
| `src/core/expert_cache.cpp:12-68` `read_expert_profile` | the TU includes `<cuda_runtime.h>` at `:4`, but the function is CUDA-free | Copied into `src/metal/engine/expert_profile.cpp` (synced-from header) |

Everything else under `src/core`, `src/prefill`, `src/kernels/cuda` and `src/program/generate.cpp` is **not compiled**
on macOS. It is replaced by `src/metal/**`.

### 1.5 Mac host components

The names below are used throughout the documentation set. The namespace is `strata::metal`; headers live in
`include/strata/metal/`.

| Component | File | Responsibility | Replaces upstream |
| --- | --- | --- | --- |
| `Context` | `src/metal/host/context.mm` | Device selection and capability checks; the command queue and its initialized `MTLFence`; loading `strata.metallib`; building every pipeline at startup; the residency set; command buffer creation with labels and error options | `device.cu`, `on_device.hpp`, `graph.cpp` |
| `CommandBuffer` | `src/metal/host/context.mm`, `include/strata/metal/context.hpp` | Serializes encoding against wrapper submission per command buffer; records cross-command-buffer producer dependencies and rejects a consumer until its producer has completed successfully | CUDA stream ordering |
| `Buffers` | `src/metal/host/buffers.mm` | Shared buffers; no-copy windows over mapped files (16 KiB-aligned, each ≤ `maxBufferLength`); a bump arena with fixed addresses | `pinned.cu`, every `cudaHostAlloc` / `cudaMalloc` site |
| Generic Encoder | `include/strata/metal/encoder.hpp`, `src/metal/host/encoder.mm` | Creates one labeled compute encoder per segment under the Context fence; validates manifest geometry, pipeline limits, Context-owned buffer views, and exact resource slots from pipeline binding reflection; clears stale slots and caps inline bytes at 4 KiB per segment. `EncoderReadOnlyBufferBinding` carries `ReadOnlyBufferRef` to MSL `const` inputs and rejects overlap with writable views through handle, CPU-backing, or GPU-address ranges. `STRATA_METAL_DISPATCH` selects serial or concurrent encoding; concurrent dispatches get a buffer barrier after each singleton or declared independent group. A failed body or validation discards the unsubmitted command buffer. Typed coordinators encode the T=1 Window input stem, layer-0 attention/GDN/FFN read and router→route→Q8_1 prefix; MoE plan has a validated standalone encoder, while a validated T=1..8 routed/shared MoE FFN, combine and GR-write compositor exists separately; production Window scheduling, later layers, MTP and Prefill remain pending. | CUDA launch sites and stream capture |
| MoE route and plan encoders | `include/strata/metal/moe_dispatch.hpp`, `src/metal/engine/moe_dispatch.cpp` | Validates top-10 routing for 256/512 experts and plan grouping for T=1..8 with active expert strides from 1 to 512, each with standalone transactional or caller-owned Encoder forms. Router logits and plan IDs/residency bind read-only; the plan validates layer-strided table extents and rejects every writable overlap. Window selects multiply-by-reciprocal route weights and Prefill selects division. `encode_window_t1_moe_router_prefix` orders the read-only BF16 router, Window route and Q8_1 input row after layer 0. Plan dispatch has its own oracle tests and an event-gated GPU-plan-to-resolver fixture. Routed expert FFN and routed/shared FFN→combine→GR-write compositors preflight active cross-stage ranges before dispatch; tiered consumers still require successful `ExpertStore` resolution. Production Window scheduling and the `MissResolver` remain unintegrated | `native_router.cu`, `native_moe.cu` |
| `Events` | `src/metal/host/events.mm` | Monotonic `MTLSharedEvent` values for the tiered layer handshake; `release_gpu_waits(timeout_ms)` satisfies queued waits and permanently retires a failed event but does not cancel commands after those waits | `doorbell_*`, `wait_flag_ge`, `cudaEvent*` |
| `PipelineCache` | `src/metal/host/pipeline_cache.mm` | The `MTLBinaryArchive` of compiled pipelines in `~/Library/Caches/strata/`: its name, load, update and pruning ([08](08-build-and-distribution.md), section 10.1) | `CUDA_MODULE_LOADING=EAGER` and the driver's JIT cache |
| `Capture` | `src/metal/host/capture.mm` | `--metal-capture`: a GPU trace of the first decode window ([08](08-build-and-distribution.md), section 10.2) | none (Nsight is external) |
| `WeightStore` | `src/metal/engine/weights.cpp` | Opens GGUF shards and the native pack (`index.txt` v3, `native_experts.txt` v3 or v4, `conversions.json`); maps tensors to (buffer, offset) | `weights.cpp`, `native_dense.cpp`, `native_head.cpp` |
| `Geometry` | `src/metal/engine/geometry.cpp` | The facts the memory plan needs about a model (shard sizes, tensor tables, per-layer blob sizes, D_pack, MTP sizes, tensor types): built by `WeightStore` from the files, or read from a geometry file with `--geometry`; written with `--write-geometry` ([07](07-server-setup-port.md), section 4.4.3) | none |
| `Plan` | `src/metal/engine/plan.cpp` | The memory plan of [05](05-memory-and-residency.md), section 3: full or tiered mode, the resident set, the slot pool, the prefill chunk; the window planning of [05](05-memory-and-residency.md), section 4.2; the PLAN line. Plain C++ with no Metal calls: it computes over `Geometry`, W, RAM and the flags, so `metal_plan_test` and `metal_window_test` need no GPU and the light `--plan-only` path (section 2.3) needs no pipelines. | the `--expert-cache auto` VRAM formula (`generate.cpp:2664-2686`) and setup's fit formulas |
| `ExpertStore` | `src/metal/engine/experts.cpp` | Residency mode (full or tiered); the resident set; the slot pool; `MissResolver`; plan-event-driven lease retirement; adaptive swaps; lookahead | `expert_source.cpp`, `expert_cache.cpp`, `remote_experts.cpp`, `pool.cpp` |
| `KvStore` | `src/metal/engine/kv.cpp` | QSA K/V in fp16 / int8 / q4_0 / k8v4, indexer state and the single shared RoPE table. `ModelStorage` verifies the exact planner byte count and opens it under its outer allocation transaction for every accepted plan; callers still initialize K/V and RoPE values | KV parts of `layer.cpp` |
| `SessionStateStorage` | `include/strata/metal/session_state.hpp`, `src/metal/engine/session_state_layout.cpp`, `src/metal/engine/session_state.cpp` | One Context-owned Shared allocation for qwen4exp's 36 GDN recurrences, 36 convolution histories and row-fastest PLE history; exact fixed-shape layout, zero-on-open, and reset guarded by tracked command-buffer completion. QSA KV/indexer state remains with `KvStore`; this owner remains attached to the session lifecycle, not `ModelStorage` | `SessionState` state carve and `session_zero` |
| `WindowScratchLayout` / `WindowScratchStorage` | `include/strata/metal/window_scratch.hpp`, `src/metal/engine/window_scratch_layout.cpp`, `src/metal/engine/window_scratch_storage.cpp` | Device-free checked T=8 layout plus one Context-owned Shared allocation for activations, per-layer commit replay, PLE, QSA, MoE, head, sampler and sink regions. The planner reports its exact size inside `fixed_bytes`; `ModelStorage` validates the planned capacity, checks maxBufferLength and opens it under the outer allocation transaction. Storage exposes bounded named-region views; host PLE staging now writes its active prefix into the PLE region, while Window dispatch and queue-use integration remain pending | `Verifier` arena carve |
| `WindowPleRows` | `include/strata/metal/window_ple_rows.hpp`, `src/metal/engine/window_ple_rows.cpp` | Device-free validation and token-major PLE row-index construction for T=1..8 from committed history and current tokens, restricted to supported qwen4exp runtime geometry and using the artifact `ngram_rows` hash. Exposes both the contiguous active batch and each token's sixteen rows. EOS handling stays in the shared hash implementation | PLE token-history preparation |
| Window PLE input staging | `include/strata/metal/window_ple_stage.hpp`, `src/metal/engine/window_ple_stage.cpp` | Validates a prepared T=1..8 row set, an open artifact-sized table and the named Shared scratch range; calls `PleTableReader::gather_batch` and returns a bounded view covering exactly T × 2560 floats. This is synchronous CPU staging and submits no GPU work; callers must wait until prior GPU use of the shared region is complete. Tiered Direct integration and production Window dispatch remain pending | PLE input rows |
| `PleTable` shared-mapping open | `include/strata/kernels/ngram.hpp`, `src/kernels/ngram.cpp` | Mmap-only open over a validated shard already owned by `GgufModel`; checks the PLE tensor's shard and extent, reads FP8 format metadata from the model's first shard, uses the existing payload mapping and retains its shared owner. `FullModeStorage` opens and owns it; `ModelStorage::ple_reader()` exposes a row-access-only `PleTableReader` in full mode and returns null in tiered mode. The view cannot reopen or close the storage-owned table. Full-mode host staging uses `gather_batch`; tiered Direct mode and issue/collect scheduling remain pending | PLE table reader |
| `MtpKvStore` / `MtpRuntimeStorage` | `include/strata/metal/mtp_kv.hpp`, `src/metal/engine/mtp_kv.cpp`, `src/metal/engine/mtp_kv_layout.cpp`, `include/strata/metal/mtp_bind_layout.hpp`, `src/metal/engine/mtp_bind_layout.cpp`, `include/strata/metal/mtp_params_layout.hpp`, `src/metal/engine/mtp_params_layout.cpp`, `include/strata/metal/mtp_runtime_storage.hpp`, `src/metal/engine/mtp_runtime_storage.cpp` | One MTP owner verifies planned weight-file, bind-layout, parameter-slot and K/V byte counts, then opens the MTP file windows, named bind views, a separate zeroed Shared array of bounded 256-byte parameter slots, and drafter K/V under a wider `Buffers::Transaction`. It initializes the window IDs to identity and copies the inspected draft-vocabulary IDs into their bounded device-side region before publication. Speculative `ModelStorage` passes its main `KvStore` to `open_with_main_kv`, which validates and retains the exact borrowed RoPE subview without a second allocation; `shared_rope()` is empty when the MTP owner is invalid. `open()` remains unbound for standalone storage use. The shape-dependent reserved bind tail has no kernel consumer, and the main RoPE values are not initialized or dispatched here | MTP weights and KV allocation |
| Full-mode layout and tensor views | `src/metal/engine/window.cpp`, `src/metal/engine/window_storage.cpp`, `include/strata/metal/window.hpp` | Builds and validates full-mode tensor slices, file-window extents and expert address layouts. `FullModeStorage::dense_tensor_view` resolves named whole dense weights to checked buffer, GGML type and shape metadata, including BF16-converted router rows; it rejects expert slabs, PLE and stale views. `ModelStorage::full_mode_storage()` publishes a borrowed const resolver with dense weights and expert address/resident tables only in full mode; every public full-mode model view uses `ReadOnlyBufferRef`, whose CPU mapping is `const` and whose generation token remains checked. `ModelStorage::moe_usage_table()` exposes the mutable, zeroed per-model usage table in either mode and retains it for the owner lifetime. Tiered mode returns null for the full-mode resolver. Future Window binding must represent model weights and address/resident inputs as read-only, and pass the usage table to the plan encoder. `FullModeStorage` also owns a CPU-only PLE table reader over its shared GGUF mapping. It does not encode a decode window or submit inference work | layout portions of `verify.cpp`, `session.cpp` |
| Window parameter builder and ring | `src/metal/engine/window_params.cpp`, `include/strata/metal/window_params.hpp` | Builds the ABI-compatible T=1..8 `WindowParams`, position-derived `StepRow`s and per-row Philox counters; `WindowParamsRing` owns two fixed-address Shared slots with 256-byte-aligned strides, publishes only built views and validates Context allocation lifetime. The caller waits for prior GPU use before reusing a slot; the T=1 input-stem coordinator consumes the published token slice and encodes gather → GR broadcast; the layer-0 coordinator orders attention GR read, speculative GDN, FFN GR read, BF16 router, top-10 route and Q8_1 quantization in one segment. The router's live-row count is a read-only `nc` slice; GDN `xq` scratch is reused after its final GDN read. The validated MoePlan encoder is tested independently, while expert execution, tiered miss resolution and later layers/head/sampler remain pending | `verify.cpp` parameter setup |
| Sampler encoders | `src/metal/engine/sampler_dispatch.mm`, `include/strata/metal/sampler_dispatch.hpp` | Validates buffers and encodes greedy, split top-k, merge and coupled-penalty kernels through standalone command buffers or caller-owned Encoder segments. Split→merge and penalty→sample dependencies use ordered dispatches; dependent consumers reject an open concurrent group. The inference Window does not call these encoders yet | CUDA sampler launch sites |
| GR read normalizer | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Validates T, manifest geometry, six buffer views, Context ownership and alias rules, then encodes `strata_gr_norm`; supports standalone command buffers and caller-owned Encoder segments. The T=1 layer-0 coordinator calls it through `encode_gr_read` for both attention and FFN using distinct layer-0 weight tensors; later Window layers and Prefill remain pending | CUDA fused GR read |
| GR down and injection encoder | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Standalone and caller-owned `Encoder` forms validate T, BF16 weight alignment, Context ownership, five buffer views and dynamic threadgroup memory, then encode the 41-group projection; the T=1 layer-0 attention/GDN/FFN coordinator calls it through both ordered `encode_gr_read` stages; later Window layers and Prefill remain pending | CUDA fused GR read |
| GR up projection and mixer | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Standalone and caller-owned `Encoder` forms validate T/apply, packed BF16 alignment, Context ownership, eight buffer views and static threadgroup-memory limits, then encode 160 groups; the T=1 layer-0 attention/GDN/FFN coordinator calls it through both ordered `encode_gr_read` stages; later Window layers and Prefill remain pending | CUDA fused GR read |
| GR post write | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Validates T, four buffer views and Context ownership; permits only exact residual/output aliasing, then encodes 40 groups per row. Supports standalone command buffers and caller-owned Encoder segments; Window and Prefill do not call it yet | CUDA native post |
| GR down projection SiLU | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Standalone and caller-owned `Encoder` forms derive `T × 320` values, validate the in-place float view and Context ownership, bind the count and dispatch the required tail-safe 256-thread groups; production GR/Window paths do not call it yet | CUDA GR down read |
| GR head-mixer pre-gated encoder | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Standalone and caller-owned `Encoder` forms validate T, fixed grid, three buffer ranges and Context ownership; permits overlap between read-only `xn` and `gate`, while keeping output disjoint; Window and head-mixer production do not call it yet | CUDA head-mixer pre-gated |
| Weighted RMS norm encoder | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Validates rows, documented width/gamma pairs, positive finite epsilon, three buffer ranges, Context ownership and exact in-place alias rules; supports both a caller-owned `Encoder` segment and a standalone wrapper routed through `Encoder::encode`, dispatching one 1024-thread group per row; Window, Prefill and MTP do not call it yet | CUDA weighted GR/QSA norm |
| Q8_1 and dense mat-vec encoders | `src/metal/engine/mv_dispatch.mm`, `include/strata/metal/mv_dispatch.hpp` | Validates the existing Q8_1 and dense projection contracts through standalone `CommandBuffer` wrappers or caller-owned `Encoder` segments; the read-only BF16 path accepts const F32 activations, BF16 weights and active-row metadata, allowing full-mode model weights to remain `ReadOnlyBufferRef` through dispatch. Read-only inputs may overlap each other; outputs and sink reject overlap with every input and each other. Standalone failure discards the command buffer, and callers of the shared form must propagate failure. The indexed draft-head wrapper remains standalone; router-to-route-to-Q8_1 and production Window/Prefill call sites remain pending | CUDA Q8_1 quantizer and dense mat-vecs |
| Embedding row-gather encoder | `src/metal/engine/embed_dispatch.mm`, `include/strata/metal/embed_dispatch.hpp` | Selects Q3_K, IQ3_S, IQ4_XS or BF16 pipelines; validates vocabulary-sized table views, CPU-visible token ids, image flags, shader-safe rows, output ranges and Context ownership; allows read-only table/token overlap and rejects output/input overlap; supports standalone command buffers and caller-owned Encoder segments; the T=1 Window input stem composes read-only table/token gather and GR broadcast, while full Window decode and Prefill production paths remain pending | CUDA embedding gather |
| Image embedding encoder | `src/metal/engine/embed_dispatch.mm`, `include/strata/metal/embed_dispatch.hpp` | Validates positive prefill rows, fixed `(10, T) × 256` geometry, token/image/output buffer sizes and alignment, allocation bounds and Context ownership; allows read-only token/image overlap and rejects output/input overlap; supports standalone command buffers and caller-owned Encoder segments; the production `Prefill::run` call site remains pending | CUDA image embedding copy |
| GR stream-broadcast encoder | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Validates shader-safe rows, the fixed `(40, T) × 256` grid, embedding/output ranges and CPU backing metadata; rejects all input/output overlap; supports standalone command buffers and caller-owned Encoder segments; the Prefill embedding-stage coordinator uses its lower-level validated chain, while Window and `Prefill::run` call sites remain pending | CUDA stream broadcast |
| Prefill embedding-stage coordinator | `src/metal/engine/prefill_dispatch.mm`, `include/strata/metal/prefill_dispatch.hpp` | Verifies chunk metadata and CPU-visible token consistency, validates every needed stage, then encodes gather, optional image copy and GR broadcast through standalone or caller-owned Encoder forms in one ordered segment; rejects an open concurrent group; callers propagate failure to discard earlier segment work; Prefill and Window do not call the coordinator yet | CUDA prefill embedding setup |
| Prefill GR normalization encoder | `src/metal/engine/prefill_gr_norm_dispatch.mm`, `include/strata/metal/prefill_gr_norm_dispatch.hpp` | Validates shader-safe rows, mode-specific BF16 remainder output, Context ownership, buffer extents/alignment and writable aliases; dispatches the Prefill-specific two-stage butterfly through standalone or caller-owned Encoder forms. Mode 1 selects the dedicated remainder pipeline; concurrent groups are rejected. The embedding-to-normalization test composes both stages in one segment; `Prefill::run` remains pending | CUDA `gr_norm_rs` |
| PLE value-broadcast encoder | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Validates T, the fixed `(40, T) × 256` grid, three buffer ranges and Context ownership; permits overlapping read-only inputs and rejects output/input overlap. It supports both a caller-owned `Encoder` segment and a standalone wrapper; PLE production dispatch does not call it yet | CUDA PLE value broadcast |
| PLE history encoder | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Validates the signed row count, `n` norm rows, fixed nine-row history, disjoint views and Context ownership; binds `n` at buffer 2 and dispatches 40×256, preserving the empty-norm n=0 no-op. It supports a caller-owned `Encoder` segment and a standalone wrapper; PLE commit and write-through do not call it yet | CUDA PLE history advancement |
| PLE gate encoder | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Validates T, the fixed `(4T) × 512` grid, the three buffer ranges and Context ownership; derives and binds `1 / sqrt(2560)` at buffer 3, permits overlapping read-only key/query inputs and rejects gate/input overlap. It supports both a caller-owned `Encoder` segment and a standalone wrapper; production PLE does not call it yet | CUDA PLE gate reduction |
| PLE conv-residual encoder | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Validates T, the fixed `(40, T) × 256` grid, F32/F16 buffer ranges and alignment, allocation bounds and Context ownership; permits read-only input overlap and rejects residual overlap with any input. It supports both a caller-owned `Encoder` segment and a standalone wrapper; production PLE does not call it yet | CUDA PLE convolution and residual |
| PLE post-projection coordinator | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Composes three weighted RMS norms, PLE gate/broadcast and convolution-residual after the caller produces `key` and `value`; an explicit history policy either advances all rows for Prefill/write-through or defers mutation for a rollback-capable Window whose commit later advances `n_keep`. Supports disjoint query/ple_norm views or their exact documented scratch alias and rejects other cross-role overlap. The shared-Encoder form requires failure propagation to discard the enclosing segment; the standalone command-buffer form discards on failure. Production Window and Prefill call sites remain pending | CUDA PLE post-ops and history |
| PLE decode-block coordinator | `src/metal/engine/ple_dispatch.mm`, `include/strata/metal/ple_dispatch.hpp` | Composes Q8_1 plus Q2_0/Q8_0 key mat-vec or direct-F32 BF16 key mat-vec, F32-activation BF16 value mat-vec and the post-projection coordinator in one segment. Reuses one sink serially and rejects writable cross-stage aliases before dispatch. Supports an explicit history policy and standalone discard-on-failure wrapper; model-backed Window/Prefill dispatch remains pending | CUDA PLE key/value projections and post-ops |
| Window PLE-entry coordinator | `src/metal/engine/window_ple_dispatch.mm`, `include/strata/metal/window_ple_dispatch.hpp` | For the L9 layer-1 PLE entry, consumes the caller-bound pending layer-0 MoE output as `previous_block_out`, pairs it with the layer-0 FFN read's `inject_out`, writes in place to that read's residual, then composes the validated PLE block in the same ordered segment. Supports caller-owned and transactional command-buffer forms and rejects an open concurrent dispatch group. Production `Window::run` and model-backed decode remain pending | `verify.cpp` window sequence |
| Window GDN block and accepted-prefix coordinators | `src/metal/engine/window_gdn_dispatch.mm`, `include/strata/metal/window_gdn_dispatch.hpp` | The block coordinator composes mixed Q8_1, qkv projection, convolution/L2, alpha/beta gates, gate projection, window recurrence/norm, output Q8_1 and output projection. The commit coordinator advances one layer's history for `n_keep`, then replays that prefix into recurrent state in one ordered segment. Both derive row contracts, reject cross-stage view overlap and provide caller-owned and transactional command-buffer forms; open concurrent groups are rejected. A T=1 layer-0 coordinator orders attention GR read, speculative GDN, and FFN GR read, reusing exact mixed and GDN block-output views plus the swapped injection views; it does not commit state. The full-mode layer-0 coordinator extends this through router, route, plan and routed/shared combine, leaving the combined output pending for PLE or the next layer's attention read. Production `Window::run`, multi-layer commit scheduling and model-backed decode remain pending | `verify.cpp` window sequence |
| GR control-vector encoder | `src/metal/engine/gr_dispatch.mm`, `include/strata/metal/gr_dispatch.hpp` | Validates the cvec shape, layer/mode, Context ownership, buffer sizes/alignment/overlap and command-buffer state, then encodes fused write/project-or-add work; supports standalone command buffers and caller-owned Encoder segments. Window and Prefill do not call it yet | CUDA `cvec_apply` and fused GR write |
| GDN decode and commit encoders | `src/metal/engine/gdn_dispatch.mm`, `include/strata/metal/gdn_dispatch.hpp` | Validates the fixed GDN views, T=1..8, mode, Context and command-buffer ownership, native buffer metadata, byte extents and output/history alias rules; encodes convolution/L2, alpha-beta gates, recurrent output/state modes and convolution-history commit. All four GDN operations support standalone command buffers and caller-owned Encoder segments. Step norm retains its four reflected threadgroup-memory bindings, and step/history commit preserve their empty no-op behavior. The fixture Window GDN block coordinator composes conv/L2, alpha-beta and step-norm; its accepted-prefix coordinator composes history commit then state replay. Production `Window::run` multi-layer scheduling and Prefill call sites remain pending | CUDA fused GDN verify/decode and commit |
| QSA norm/RoPE encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates T=1..8, K/Q/indexer-Q view extents and alignment, Context ownership, CPU/GPU metadata, RoPE-table positions and output alias rules; encodes one 256-thread group per head with exact in-place K/indexer-Q support and disjoint strided-Q input/output. It supports both standalone command buffers and caller-owned Encoder segments; Window, Prefill and MTP do not call it yet | CUDA QSA norm and RoPE |
| QSA FWHT256 encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates positive row count, overflow-safe byte extent, float alignment, allocation bounds, Context and command-buffer ownership, and native CPU/GPU metadata; encodes `ceil(rows/8) × 256` in-place work, one simdgroup per row; supports standalone command buffers and caller-owned Encoder segments; Window and Prefill do not call it yet | CUDA `fwht256` for O-class KV rotation |
| QSA indexer snapshot encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Copies all 12 QSA layers' 384-float tails and block positions into packed outputs; accepts independent per-layer strides in float/int32 element counts, preserves raw tail bits, checks overflow-safe extents, Context ownership, native CPU/GPU metadata and exact strided input/output aliases; supports standalone command buffers and caller-owned Encoder segments; Window and commit do not call it yet | CUDA indexer state snapshot |
| QSA indexer append encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Encodes up to eight contiguous rows for one QSA layer through the sequential fp16-tail, pooled RMS/RoPE transition; commit mode restores the indexed layer's tail/block position from the packed snapshot before replaying the accepted prefix. Validates the folded RoPE table extent, cell/pool extents, snapshot selection, ownership, native metadata and all output aliases; supports standalone command buffers and caller-owned Encoder segments, where snapshot→window append→commit replay can share one ordered segment; Window and commit do not call it yet | CUDA native QSA indexer append |
| QSA selection-score encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Scores each row's blocks using its validated `StepRow`; checks T, cell capacity, all extents/alignment, exact `pos`/`n_kv`/`n_bid`/`width` relationships, output/input disjointness, Context ownership and native metadata; supports standalone command buffers and caller-owned Encoder segments, dispatching at most 256 threadgroups with the documented block stride; Window and Prefill do not call it yet | CUDA block score kernel |
| QSA selection top-k encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Selects `min(n_kv, 2051)` cells by the documented ordered score key; standalone and caller-owned Encoder forms share StepRow, score/id extent, alignment, Context ownership, native metadata and id-output disjointness validation; chooses the 1024-thread register kernel through 33,792 blocks and the 256-thread memory kernel above it. Shared score→top-k dispatches can occupy one ordered Encoder segment; Window and Prefill do not call it yet | CUDA block top-k kernels |
| QSA attention-gate encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Standalone and caller-owned Encoder forms share T=1..8, attention/query/output float extents, alignment, Context ownership, native metadata and alias validation; permits overlapping read-only inputs and rejects output/input overlap; dispatches one 256-thread group per head and row; Window and Prefill do not call it yet | CUDA QSA sigmoid gate |
| QSA sparse-attention chunk/merge encoders | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Supports fp16, int8, q4_0 and k8v4; standalone and caller-owned Encoder forms share validation for T=1..8, cell capacity, ID-row stride, `StepRow` consistency, overflow-safe input/scratch/output extents, mode-specific alignment, Context ownership, native metadata and writable aliases; dispatches `(ceil(id_row_stride/64), 2, T) × 256` chunks then `(24, T) × 256` merge; the shared form composes chunk→merge in order and requires failure propagation to discard the segment; valid live-prefix IDs are a caller/producer precondition; Window and Prefill do not call these encoders yet | CUDA QSA sparse-attention chunk and merge kernels |
| QSA selection-attention coordinator | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Composes selection scores→top-k→attention chunks→merge in one ordered Encoder segment; preflights all component extents and StepRows, requires a 2051-element ID stride while allowing cache capacities below 2051, enforces shared score/StepRow/ID/scratch views, rejects cross-stage aliases except producer→consumer views, and refuses an open concurrent dispatch group so automatic buffer barriers remain active; GPU-produced live IDs remain a producer precondition under REV-032; no production Window/Prefill call site yet | CUDA QSA selection and sparse attention chain |
| QSA fp16 KV append encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates T=1..8, a four-cell-aligned capacity addressable by `StepRow`, unique in-range positions, current/pool extents and alignment, pool/input disjointness, Context ownership and native CPU/GPU metadata; encodes one 256-thread group per K/V head-row into the interleaved cache; standalone and caller-owned Encoder forms share preflight validation; Window and Prefill do not call it yet | CUDA fp16 KV append |
| QSA int8 KV append encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates T=1..8, a four-cell-aligned capacity addressable by `StepRow`, unique in-range positions, input/code/scale extents and alignment, all output/input and output/output ranges, Context ownership and native CPU/GPU metadata; encodes one 256-thread group per K/V head-row with four fp16 scales per row; disjoint code/scale views may share one allocation, and callers must provide finite K/V values; standalone and caller-owned Encoder forms share preflight validation; Window and Prefill do not call it yet | CUDA int8 KV append |
| QSA q4_0 KV append encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates T=1..8, a four-cell-aligned capacity addressable by `StepRow`, unique in-range positions, finite-reciprocal q4 scale preconditions, input/pool extents and alignment, pool/input and pool/pool disjointness, Context ownership and native CPU/GPU metadata; encodes one 256-thread group per K/V head-row into eight packed 18-byte blocks after caller-encoded FWHT256; standalone and caller-owned Encoder forms share preflight validation, and both K/V FWHT dispatches can precede append in one ordered segment; Window and Prefill do not call it yet | CUDA q4_0 KV append |
| QSA k8v4 KV append encoder | `src/metal/engine/qsa_dispatch.mm`, `include/strata/metal/qsa_dispatch.hpp` | Validates T=1..8, aligned `StepRow`-addressable capacity, unique positions, K-code/K-scale/V-block extents, native CPU/GPU metadata, Context ownership and command-buffer state; rejects overlapping inputs because V is FWHT-transformed in place, and rejects output/input and output/output aliases while allowing three disjoint views in one allocation; one 256-thread group writes unrotated int8 K and FWHT256-rotated q4_0 V after same-command-buffer V-only FWHT; finite input, fp16-scale and q4 reciprocal preconditions are caller-owned; standalone and caller-owned Encoder forms share preflight validation, and V-only FWHT can precede append in one ordered segment; Window and Prefill do not call it yet | CUDA hybrid K8V4 KV append |
| Sampler tail | `src/metal/engine/sampler_tail.cpp` | `sampled_tail`: the serial CPU tail of sampled decoding (top-p, min-p, temperature and the Philox draw) over the candidates the sampler kernel returns ([04](04-metal-kernels.md), section 5.12) | the tail of `sampled_tail_warp` (`sampler.cu`) |
| `Window` | `src/metal/engine/window.cpp` layout helpers; `src/metal/engine/window_dispatch.mm` T=1 input stem and MoE stages | The input stem reads the published T=1 token from `WindowParams`, gathers its embedding from a read-only table, then broadcasts the row into GR streams in one ordered transactional segment. A second T=1 coordinator orders layer-0 attention GR read, speculative GDN and FFN GR read, reusing named mixed/GR scratch and injection regions with distinct attention/FFN weights. A full-mode layer-0 coordinator takes `FullModeStorage`, binds its expert address/resident/usage tables, and composes router/route/plan production with routed expert FFN and shared combine in one ordered transaction. It leaves the combined output pending; PLE or the next layer's attention read applies the write according to `pending(1)`. It checks declared producer/consumer handoffs and rejects other consumer-input overlaps with producer storage. Tiered mode still ends at plan production; callers need successful plan readiness, `ExpertStore` resolution and retained expert leases before calling the lower-level consumer. No production scheduler performs that sequence, and later-layer scheduling, head, sampler and state commit remain unimplemented | `verify.cpp`, `session.cpp` token graph |
| `Mtp` | `include/strata/metal/mtp.hpp`, `include/strata/metal/mtp_storage.hpp`, `include/strata/metal/mtp_bind_layout.hpp`, `include/strata/metal/mtp_params_layout.hpp`, `include/strata/metal/mtp_runtime_storage.hpp`, `src/metal/engine/mtp.cpp`, `src/metal/engine/mtp_storage.cpp`, `src/metal/engine/mtp_bind_layout.cpp`, `src/metal/engine/mtp_params_layout.cpp`, `src/metal/engine/mtp_runtime_storage.cpp` | `WeightStore` returns an immutable catalog of all 29 dense rows, source file sizes and identities, and draft-vocabulary IDs in source order, retained with the artifact-generation lock. The device-free planner keeps expert blobs and dense tensors intact inside page-aligned `maxBufferLength` windows. `MtpStorage` maps and warms those windows; `MtpRuntimeStorage` transactionally composes the windows, vocabulary-aware bind regions, planned parameter slots and drafter K/V. Production startup composition, drafter kernels and the upstream-compatible MTP API remain unimplemented | `mtp.cpp` |
| `Prefill` | `src/metal/engine/prefill.cpp`, `include/strata/metal/prefill.hpp`, `src/metal/engine/prefill_dispatch.mm`, `src/metal/engine/prefill_gr_norm_dispatch.mm` | Builds and validates host metadata for one chunk (`PrefillParams`, query slots, tokens, `StepRow`s and image-row presence); provides composable embedding and GR-normalization stages. Remaining chunk kernels, production submission/wait and `Prefill::run` are unimplemented | `src/prefill/*` |
| `Snapshot` | `src/metal/engine/snapshot.cpp` | Checkpoint save/restore and conversation parking | `conversation_snapshot.cpp`, `conversation_state.cpp` |
| Driver | `src/metal/program/{main,options,serve,oneshot,stats,watchdog,version}.cpp` | The `strata` executable. `version.cpp` holds the version, `METAL_REV` and the metallib hash ([08](08-build-and-distribution.md), section 4.5). | `generate.cpp` |

---

## 2. Process model

### 2.1 Executables built on macOS

| Target | Mac status | Notes |
| --- | --- | --- |
| `strata` | **Mac-only Metal build** (Metal is always enabled by the root CMake project) | Same name and CLI, so `serve/server.py` and the run config point at it unchanged. Ships with `strata.metallib` next to it ([08-build-and-distribution.md](08-build-and-distribution.md)). |
| `strata-gguf`, `strata-dequant` | Built unchanged | Already portable |
| `strata-vision` | Built against the pinned llama.cpp revision with Metal | Mac-only arm64 build. CPU mode defaults to the reported performance-core count; `--gpu` means Metal. |
| `strata-device`, `strata-load`, `strata-concurrent`, `strata-overlap`, `tune_hipblaslt` | Not built | CUDA / HIP / PCIe only. `strata --list-devices` replaces `strata-device`. |

### 2.2 Threads

| Thread | QoS (`pthread_set_qos_class_self_np`) | Role |
| --- | --- | --- |
| Main | `QOS_CLASS_USER_INTERACTIVE` | Serve loop or one-shot loop: request parse, prompt cache, prefill, decode rounds, encode and commit of command buffers |
| stdin reader | default | `read(2)` on fd 0 (upstream `generate.cpp:4514-4560`). Strips `\r`. `STOP` sets `stop_req` immediately; every other line is queued; EOF sets `in_eof`. |
| Watchdog | default | One check per second (section 14) |
| PLE I/O workers | `QOS_CLASS_USER_INITIATED` | `DirectFile` pread workers (`STRATA_IO_THREADS`) |
| Expert fetch workers (tiered mode only) | `QOS_CLASS_USER_INTERACTIVE` | Persistent `STRATA_FETCH_THREADS` pool, default 8 and clamped to 1-64; each cold expert queues its three exact slab reads. 8 or more reads must be in flight to reach 12-13 GB/s ([appendix A.4](appendix-a-measurements.md)). |
| `MissResolver` (tiered mode only) | `QOS_CLASS_USER_INTERACTIVE` | The layer handshake (section 9.3) |
| Adapt (tiered mode only) | `QOS_CLASS_UTILITY` | Resident-set swaps every `--adapt-every` rounds (section 9.4) |
| Metal completion handlers | Metal-owned | Release retained slot leases and signal completion state; never perform file I/O or encode GPU work |

While a request is running, hold `[[NSProcessInfo processInfo] beginActivityWithOptions:NSActivityUserInitiated |
NSActivityLatencyCritical reason:@"strata request"]`. This stops timer coalescing and idle sleep from stretching a
generation. End the activity when the request finishes. Measure its effect on wake latency (risk H-8 in section 15).

### 2.3 Startup sequence

Each step's failure is listed. Exit code 2 is for argument errors (as upstream). Exit code 1 is for every other fatal
startup error. Messages go to stderr and are prefixed `strata: `.

| # | Step | Failure |
| --- | --- | --- |
| 1 | `setvbuf(stdout, nullptr, _IONBF, 0)` (upstream `:985`). Install signal handling (section 2.4). | none |
| 2 | Parse arguments (section 3). Print one warning per accepted-but-ignored flag. | exit 2: `unknown argument: X` plus usage, `X needs a value`, or a validation message |
| 3 | `--version` or `--list-devices`: print and exit 0 (section 3.10). | none |
| 4 | `Context::open`. The checks are listed after this table. | exit 1 with a specific message |
| 5 | Read the memory facts: `W = recommendedMaxWorkingSetSize`, or `--gpu-budget-mib` if smaller; `hw.memsize`; available memory (section 7.2). | none |
| 6 | Open the GGUF shards (mmap, `PROT_READ`, `MAP_PRIVATE`, as upstream; Metal wraps such mappings, [appendix A.3](appendix-a-measurements.md)) and the native pack. Validate the format rules in section 6.2. | exit 1: missing file, bad magic, a `native_experts.txt` version above 4 (`kExpertLayoutVersion`), any other rule of section 6.2 |
| 7 | Load the MTP runtime from `--mtp` (about 0.9 GB). It must load before the expert budget is computed, as upstream (`mtp.hpp:45`). | exit 1 |
| 8 | Memory plan ([05-memory-and-residency.md](05-memory-and-residency.md)): dense + KV(`--max-context`, `--kv`) + scratch + MTP + prefill buffers + slot pool + resident experts ≤ W − reserve. This chooses full or tiered mode and the resident expert count. Print the plan on stderr. With `--plan-only`, print it on stdout and exit (section 3.10). | exit 1: `the model does not fit: needs X MiB of GPU memory with no expert resident, Y MiB available`. With `--plan-only`, `PLAN fits=0 ...` on stdout and exit 3. |
| 9 | Create the buffers. Full mode: no-copy windows over the shards. Tiered mode: copy arenas. Add every long-lived buffer to the residency set and `requestResidency`. Warm the resident set (one GPU pass over it) so that the first request does not pay page-in. | exit 1 on allocation failure, with the plan printed |
| 10 | Open the PLE table (`--ple-gguf`, `--ple-io`). Load control vectors. Read the expert profile (`--expert-profile`). | exit 1 |
| 11 | Zero the session state (`session_zero`). | none |
| 12 | Serve mode: print the memory line, `INFO`, start the watchdog, print `READY` (section 5.2). One-shot mode: run. | none |

`Context::open` checks, each with its own error message:
- `MTLCreateSystemDefaultDevice` is not nil.
- `hasUnifiedMemory`.
- `supportsFamily(MTLGPUFamilyApple7)` (simdgroup matrix) and `MTLGPUFamilyMetal4`.
- Capture the immutable `DeviceFacts` snapshot once: IORegistry GPU cores, Metal working-set and allocation limits,
  host RAM, macOS version and supported families. Reuse it for later planning and dispatch validation; do not repeat
  IORegistry or sysctl probes in the kernel-dispatch path.
- `strata.metallib` is found next to the executable (via `_NSGetExecutablePath`) or at `--metallib`.
- Its SHA-256 equals the hash compiled into the engine, unless `--metallib` is given. A mismatch exits 1 with
  `strata.metallib does not belong to this engine ...` ([08](08-build-and-distribution.md), section 5.6).
- Every kernel function in the manifest is present. This is the replacement for G9 `device_code_error`.
- Every pipeline has `threadExecutionWidth == 32`, and `maxTotalThreadsPerThreadgroup` is at least the kernel's
  declared need.

Building all pipelines at startup replaces `CUDA_MODULE_LOADING=EAGER`. A binary archive (`MTLBinaryArchive`) cached
in `~/Library/Caches/strata/` cuts the second start. A mismatched or invalid archive is ignored, never fatal. The
archive's name, the steps and the stats line are in [08](08-build-and-distribution.md), section 10.1.

**The `--plan-only` path is light.** Setup calls it for every menu line, so it must not do the work of a real start:
- Step 4 runs only the device checks (device, unified memory, families). It does not look for `strata.metallib`,
  check the manifest or build pipelines. With both `--plan-w-mib` and `--plan-ram-mib`, it does not open the Metal
  device at all, and the PLAN line says `gpu=simulated`.
- Steps 6 and 7 read only what the plan needs: the GGUF headers and tensor tables of every shard, `index.txt`,
  `native_experts.txt` and `conversions.json` of the pack, and the MTP runtime's sizes plus its small draft-vocabulary
  ID list. No GGUF or packed tensor payload is read, and nothing is wrapped as a Metal buffer. With `--geometry FILE`
  they read the geometry file instead and open no model file.
- For the Qwen catalogue, the loader checks the complete 1,224-name and shape inventory and permits only the
  tensor-role/type combinations present in the seven frozen model headers. Conversion records are limited to exact,
  zero-error F32-to-BF16 router conversions.
- Step 8 prints the PLAN line on stdout ([05](05-memory-and-residency.md), section 3.4) and exits 0 or 3. Steps 9-12
  never run.
- It must take under 200 ms with a geometry file and under 1 s with real files on the reference Mac (measurement X-3
  in [07](07-server-setup-port.md), section 14).

`--write-geometry FILE` takes the same light path through steps 6 and 7 and needs no Metal device. It writes the
geometry file format of [07](07-server-setup-port.md), section 4.4.3, to a unique temporary file beside the target,
syncs it and atomically replaces the target, then exits 0. Its `types=` line lists the type of every tensor the engine
would load, whether or not a kernel exists for it.

### 2.4 Signals, parent death and exit

| Event | Upstream | Mac engine |
| --- | --- | --- |
| stdin EOF | `in_eof`; the main loop exits | Same. Also set `stop_req`, so a request in progress ends at the next round. With EOF, no process reads stdout any more. Exit 0. |
| `QUIT` | Ends the loop | Same, exit 0 |
| SIGINT | Default action (unspecified in the audit) | In `--serve` mode, **ignore SIGINT** (`signal(SIGINT, SIG_IGN)`). The engine is spawned without a new session, so a terminal Ctrl+C reaches it too. The server then QUITs it cleanly ([07-server-setup-port.md](07-server-setup-port.md)). One-shot mode keeps the default action. |
| SIGHUP | Default (terminate) | In `--serve` mode, **ignore SIGHUP**, as SIGINT. Closing the Terminal window sends it to the whole foreground process group; the server catches it and QUITs the engine ([07-server-setup-port.md](07-server-setup-port.md), section 6.2). One-shot mode keeps the default action. |
| SIGTERM | Default | Default (terminate) |
| SIGPIPE on stdout | Default | Default. The server is gone, so dying is correct. |
| Parent killed with SIGKILL | stdin EOF | Same; there is no `PR_SET_PDEATHSIG` on macOS, and stdin EOF is sufficient |
| Watchdog | `std::abort()` | `std::abort()`. macOS writes a crash report with every thread's stack (section 14). |
| GPU command buffer error (status `MTLCommandBufferStatusError`) | n/a (CUDA errors are checked per launch) | Print `strata: GPU error in "<label>": <error domain/code/description>` on stderr. In serve mode, reply `ERR gpu error: <description>`, then exit 1, because the model state can no longer be trusted. The server restarts the engine on the next request. |

---

## 3. Command line

### 3.1 Dispositions

| Code | Meaning |
| --- | --- |
| **KEEP** | Same flag, same default, same meaning. |
| **MAC** | Same flag; its meaning is mapped to unified memory as described. |
| **IGNORE** | Accepted, so old configs and scripts still start. Prints one stderr line `strata: --X has no effect on the Metal engine` and does nothing. |
| **NEW** | Mac-only flag. |

Error handling is unchanged. An unknown argument prints `unknown argument: X` plus usage and exits 2. A missing value
prints `X needs a value` and exits 2. A validation failure exits 2 (upstream `:1248-1252`).
The upstream `--help` and `-h` flags remain KEEP and print the Metal driver's usage.

### 3.2 Model and pack

| Flag | Upstream default | Mac |
| --- | --- | --- |
| `--pack DIR` | `pack/full` | KEEP. The native pack directory: `index.txt`, `dense.bin`, `extra.bin`, `native_experts.txt`. |
| `--native SHARD1` | off | MAC: **required**. The Metal engine implements only the native-pack path, because the canonical pack is produced only on AVX-512 hosts (`setup.py:3077`). Without `--native`, exit 2 with `the Metal engine needs --native <shard 1 of the GGUF>`. |
| `--native-bf16`, `--native-bf16-extra`, `--native-ple-key`, `--native-moe-combine`, `--native-gdn`, `--native-flash-attn-short`, `--native-qsa-indexer`, `--native-qsa`, `--native-rope`, `--native-ple-postops`, `--native-router` | off | IGNORE, **without** a warning. Every native arm is always on in the Metal engine. |
| `--native-head-gguf F`, `--embd-gguf F`, `--native-dense-gguf F` (repeatable) | - | KEEP |
| `--ple-gguf F` | - | KEEP |
| `--no-ple` | - | KEEP (diagnostic) |
| `--ple-io direct\|mmap\|ram` | `direct` | MAC. `direct` = `pread` with `F_NOCACHE` (section 7.4). `mmap` = rows are memcpy'd from a mapping; the mapping is **never** wrapped as a Metal buffer ([appendix A.3](appendix-a-measurements.md)). `ram` is allowed but warns `--ple-io ram wires 28.8 GB; macOS will usually refuse, and the engine then only pre-touches the pages`. |
| `--ple-row-cache N` | `1<<20` rows | KEEP |
| `--ple-inflight N` | 64 (1..1024) | KEEP |
| `--cpu-oracle-q8-0`, `--gr-fp32-activations`, `--gr-native-mmvf` | off | IGNORE. `--gr-native-mmvf` and `--gr-fp32-activations` without a warning, because their contract is always on; `--cpu-oracle-q8-0` with one ([04](04-metal-kernels.md), section 9.2) |

### 3.3 Context, KV and RoPE

| Flag | Upstream default | Mac |
| --- | --- | --- |
| `--max-context N` | 4096 | KEEP |
| `--kv fp16\|int8\|q4_0\|k8v4` (`q4` alias) | fp16 | KEEP. Anything else exits 2. |
| `--kv-resident N` | 0 | IGNORE. All KV is in unified memory; `INFO kv_resident=0`. |
| `--rope-scaling none\|linear\|yarn`, `--rope-scale`, `--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`, `--yarn-ext-factor` (-1), `--yarn-attn-factor` (1), `--yarn-beta-fast` (32), `--yarn-beta-slow` (1) | none | KEEP |

### 3.4 Sampling (one-shot mode; serve takes these per request)

| Flag | Default | Mac |
| --- | --- | --- |
| `--max-new N` | 16 | KEEP |
| `--greedy` | true | KEEP |
| `--seed S` | - (setting it turns sampling on) | KEEP |
| `--top-k` | 20 | KEEP |
| `--top-p` | 0.95 | KEEP |
| `--temperature` | 1.0 | KEEP |
| `--stop-eos`, `--eos-ids a,b` | `{248044, 248046}`; `--eos-ids` implies `--stop-eos` | KEEP |
| `--tokens a,b,...` / `--tokens-file F` | one of the two; the serve dummy prompt is `{248045}` | KEEP |

### 3.5 Experts and residency

| Flag | Upstream default | Mac |
| --- | --- | --- |
| `--expert-cache N\|auto` | 0 (`auto` = -1) | MAC. The number of routed experts in the **resident set**. The Mac default is **`auto`**: as many as the memory plan allows, capped at all 24,576 (12,288 for Coder). If all experts fit, the engine runs in full mode; otherwise in tiered mode. An explicit `N` forces tiered mode when N is below the total. `0` is diagnostic: every expert goes through the miss path. |
| `--expert-cache-per-layer` | off | MAC. Per-layer quota of the resident set (`slots / n_layers`), as `expert_cache.hpp:109` |
| `--expert-profile F` | - | KEEP. The STRP ranking that fills the resident set in tiered mode. In full mode it only orders the warm-up. |
| `--adapt-every N` | 4 | MAC. Tiered mode: rounds between resident-set swaps. Full mode: no effect, and no warning. |
| `--adapt-swaps N` | 96 | MAC. Tiered mode: maximum swaps per adapt pass |
| `--vram-reserve-mib N` | 700 | MAC. MiB of the GPU working-set budget W left unallocated. The default stays 700; this is a measurement item (M-3 in [05-memory-and-residency.md](05-memory-and-residency.md)). |
| `--resident-budget-gib G` | - | MAC. Caps the bytes of the resident set; `0` forces no residents, while omitting the flag leaves this cap unset. An alternative to `--expert-cache N`; if both are given, the smaller wins. |
| `--resident-experts`, `--resident-cpu-experts`, `--mmap-experts` | off | IGNORE, without a warning. The Mac engine always works this way. |
| `--expert-cache-device1..3 N`, `--expert-cache-remote-placement` | 0 / stripe | IGNORE |
| `--expert-cache-cpu-order`, `--no-hit-poke` | off | IGNORE |
| `--shared-expert-arena F` | - | IGNORE. Read-only mappings already share the page cache between processes. |
| `--pool-workers N`, `--pool-affinity`, `--no-host-worker` | 0 / all / off | IGNORE in the first release (no CPU pool; `INFO pool_workers=0`). An optional CPU helper behind a flag restores `--pool-workers` (section 12). |
| `--pcie-frac F`, `--pcie-mode` | auto | IGNORE. `INFO pcie_frac=0.00`. |

### 3.6 Prompt path

| Flag | Upstream default | Mac |
| --- | --- | --- |
| `--prefill CHUNK\|auto\|auto:16384\|auto:32768` | - | MAC. `auto` picks the largest chunk from {8192, 6144, 4096, 2048, 1024} whose buffers fit the memory plan; the floor is 1024. `auto:N` raises the cap to N. An explicit `CHUNK` is used as given, and exits 1 if it does not fit. ([05-memory-and-residency.md](05-memory-and-residency.md)). Without `--prefill`, the prompt goes through decode windows, as upstream. |
| `--prefill-until N`, `--dump-final-r F` | - | KEEP (diagnostic) |
| `--no-split-rows`, `--no-prefill-borrow` | - | IGNORE. There is no VRAM/CPU row split and no slot lending. |

### 3.7 Speculation

| Flag | Default | Mac |
| --- | --- | --- |
| `--spec N` | 0 | KEEP. Window size S (setup uses 4). With 0, plain decode is a T = 1 window. |
| `--mtp DIR` | - | KEEP (`mtp/rt`) |
| `--mtp-window N` | 32768 | KEEP. The planner scales MTP bind scratch with `round_up_64(min(N, max_context))`; `N=0` or `N >= max_context` uses `max_context`. |
| `--mtp-max-t N` | 0 | KEEP. The bind arena is allocated for T = 8 regardless, so this limits draft work but does not reduce planned bind bytes. |
| `--spec-min-p P` | 0 (setup uses 0.5) | KEEP |
| `--suffix-draft N` | 3 | KEEP |
| `--coupled-draft` / `--no-coupled-draft` | env `STRATA_SPEC_COUPLED` | KEEP |
| `--spec-oracle F`, `--spec-corrupt N` | - | KEEP. Gate C2 needs them ([09-testing-and-validation.md](09-testing-and-validation.md)). |
| `--spec-split` / `--no-spec-split` | off | IGNORE. The row-group split exists to overlap the CPU pool with the GPU. |

### 3.8 Multi-GPU

| Flag | Mac |
| --- | --- |
| `--layer-split K\|K1,K2\|auto`, `--split-device a,b`, `--split-skip-if-fits` | IGNORE. One GPU. Upstream exits 2 for the combination of `--layer-split` and conversation parking; on a Mac the split is ignored first, so that combination is accepted. |

### 3.9 Serve, conversation cache, control vectors, diagnostics

| Flag | Default | Mac |
| --- | --- | --- |
| `--serve` | off | KEEP (section 5) |
| `--vision` | off | KEEP |
| `--prompt-cache N` | 6 | KEEP |
| `--prompt-cache-every N` | 16384 | KEEP |
| `--prompt-cache-root N` | 2048 | KEEP |
| `--turn-token ID` | 248045 | KEEP |
| `--short-read N` | 64 | KEEP |
| `--conversation-cache-mib N` | 0 | KEEP. It works on a Mac only after the patch in section 7.3. |
| `--conversation-cache-slots N` | 4 | KEEP |
| `--conversation-cache-min-free-mib N` | 2560 | KEEP |
| `--control-vector F`, `--control-vector-scaled F:S[,F:S]` (split at the last colon) | - | KEEP |
| `--control-vector-layer-range A B` | (setup uses 4 44) | KEEP |
| `--cvec-mode add\|project` | add | KEEP |
| `--cvec-dir per-layer\|single:L` | per-layer | KEEP |
| `--dump-logits F`, `--logits-stride`, `--dump-residual`, `--dump-mixed`, `--dump-layers`, `--dump-halves`, `--dump-routing F` | - | KEEP, with **byte-identical file formats**. Gate C1 compares them against llama.cpp. The `--dump-routing` record is `int32 layer, int32 k, k × int32 ids, k × f32 weights`, the input of `tools/make_profile.py`. |
| `--stats` | off | MAC (section 13) |
| `--stage-timing` | off | MAC: per-layer GPU timestamps from counter sample buffers (section 13) |
| `--sync-every-layer` | off | MAC: one command buffer per layer, each waited on. A debugging aid; slow. |
| `--no-capture` | off | MAC: forces re-encoding even when indirect command buffers are implemented. A no-op before that. |
| `--no-pool`, `--gpu-only-full`, `--graph-only`, `--gpu-stages` | off | IGNORE, without a warning. Always GPU-only. |
| `--shared-late`, `--keep-canonical`, `--no-token-graph`, `--no-fused-gr`, `--no-fast-attn`, `--no-publish-kernel`, `--no-fused-gdn`, `--no-fast-select`, `--stream-token` | off | IGNORE. `--stream-token` without a warning (always on); the others with one. Each row is in [04](04-metal-kernels.md), section 9.2. |
| `--check-logits` | off | KEEP: the host scans row 0 of the window's logits after the wait; upstream's message and exit code ([04](04-metal-kernels.md), section 9.2) |

### 3.10 New flags

| Flag | Meaning |
| --- | --- |
| `--version` | Print `strata <x.y.z> metal <metal_rev>` and exit 0. It needs no GPU. The binary must also contain the literal `engine=<x.y.z>\n`, which `setup.py:engine_version()` scans for, and `metallib_sha256=<64 hex>\n`, the hash of the library it was built with ([08](08-build-and-distribution.md), sections 4.5 and 5.6). |
| `--list-devices` | Print the device facts and exit 0: name, GPU core count (from IORegistry), W, `maxBufferLength`, `maxThreadgroupMemoryLength`, families, `hw.memsize`, macOS version. It builds no pipeline and opens no pipeline archive. It ignores `STRATA_METAL_CAPTURE`; an explicit `--metal-capture` is an error because device listing has no decode window. Used by setup's smoke test. |
| `--self-test` | Development-only GPU smoke test: build the pipelines, dispatch the two context self-test kernels and verify their output. It takes no model files. An empty invocation does not run the self-test implicitly. |
| `--storage-check` | Diagnostic storage-only startup: build pipelines, inspect real model inputs, compute the normal memory plan, then open the selected Full/Tiered weight storage, planned main QSA KV/indexer/shared-RoPE allocation and planned Window scratch. Full mode also opens its CPU-only PLE reader over the retained GGUF mapping, reads only the validated converted-router rows into its page-aligned anonymous arena and runs CPU/GPU warm-up; tiered mode performs startup copies and publishes no PLE reader. When speculation is planned, it also opens the complete MTP runtime from its inspected catalog under the same outer allocation transaction. Print `STORAGE_CHECK ... main_kv=yes ple_table=<yes|no> window_scratch=yes inference=unavailable` only after success, with `ple_table=yes` for full mode and `mtp_runtime=yes` when that owner is open. K/V and RoPE values remain for a session to initialize. Failed opens roll back provisional Metal residency allocations. The owner verifies inspected geometry against GGUF payload sizes and requires the source and artifact-set inspection locks to remain held. This path does not collect PLE rows, initialize recurrent session state or PLE history, and never prints `READY`. Requires `--native`; cannot be combined with `--plan-only`, `--geometry`, `--write-geometry`, `--list-devices`, `--self-test` or Metal capture. |
| `--plan-only` | Run startup steps 1-8 on the light path of section 2.3: no metallib, no pipelines, no tensor data, no allocation. Print the plan as one `PLAN key=value ...` line on stdout ([05](05-memory-and-residency.md), section 3.4) and exit. Exit 0 if it fits, 3 if it does not. A tensor type with no Metal kernel gives `fits=0 reason=unsupported_type_<T>` and exit 3, from real files and from a geometry file alike. When `--spec > 0`, the inspected input must include a validated MTP runtime; missing MTP geometry is an argument error (exit 2). Setup uses this for its memory-fit decision ([07-server-setup-port.md](07-server-setup-port.md), section 4.4), so the fit logic exists only once, in C++. |
| `--geometry FILE` | Plan from a geometry file instead of the model files ([07-server-setup-port.md](07-server-setup-port.md), section 4.4.3). Valid only with `--plan-only`, and never together with `--pack`, `--native`, `--ple-gguf` or `--mtp`: either error exits 2. A geometry without MTP facts cannot be used with `--spec > 0`. An unknown format version exits 2 with `unsupported geometry file version N`. |
| `--write-geometry FILE` | Write the geometry file of the given real files (`--pack`, `--native`, `--ple-gguf`, `--mtp`) and exit 0 (section 2.3). Used by `tools/make_geometry.py`. Export accepts `--geometry-family F` and `--geometry-model M` to set safe, non-empty identity atoms. It creates a unique temporary file beside the target and atomically replaces the target. |
| `--geometry-family F`, `--geometry-model M` | Set the geometry identity fields when used with `--write-geometry`. Values must be non-empty valid UTF-8 and contain no whitespace, `=` or `#`; invalid values exit 2. |
| `--plan-w-mib N`, `--plan-ram-mib N` | Hidden (not in usage). Replace W and `hw.memsize` for the plan, so a plan for another Mac can be recorded on any Mac ([07-server-setup-port.md](07-server-setup-port.md), section 10.2). Valid only with `--plan-only`, otherwise exit 2. With both, the Metal device is not opened. `--gpu-budget-mib` still applies after `--plan-w-mib`. |
| `--gpu-budget-mib N` | Use min(W, N) as the GPU budget. This simulates a smaller Mac on a larger one: the 64 GB form of gate G2 runs with `--gpu-budget-mib 37888` and a 16 GiB balloon ([09](09-testing-and-validation.md), section 6.2). |
| `--expert-slots N` | Tiered mode: the slot pool size. The default and minimum is 2 × H, where H is the largest per-layer count of non-resident experts: two halves that each hold one layer's misses, decode or prefill. A larger N adds slot-cache capacity. The plan reserves the pool before choosing resident experts ([05-memory-and-residency.md](05-memory-and-residency.md), sections 3.2 and 5.4). |
| `--metallib PATH` | Use this library instead of the one next to the executable (development). Its hash is not checked, and the engine warns `strata: --metallib: <path> is not checked against this engine` ([08](08-build-and-distribution.md), section 5.6). |
| `--metal-capture F` | Write a GPU trace of the first decode window to `F.gputrace` (`.gputrace` is appended when missing). The engine sets `MTL_CAPTURE_ENABLED=1` itself, before Metal starts. An existing trace exits 2. It cannot be combined with `--list-devices` or `--plan-only`, because neither path runs a decode window. Development only ([08](08-build-and-distribution.md), section 10.2). |

---

## 4. Environment variables

### 4.1 Rule

Every upstream `STRATA_*` variable keeps its name. On a Mac it is either honoured with the same meaning, or ignored
silently. Variables are not warned about, because the server passes the user's `env` dictionary through. The tables
below give the disposition of every variable in [audit B 1.3](audit/B-engine-host.md).

### 4.2 Honoured with the same meaning

| Variable | Meaning |
| --- | --- |
| `STRATA_TRACE` | `strata trace:` progress lines. `mem_mark` prints the GPU working-set free space instead of VRAM. |
| `STRATA_PREFILL_AUTO_MAX` | Largest auto prefill chunk (8192) |
| `STRATA_RESIDENT_HEADROOM_GIB` | RAM headroom kept free when the memory plan is computed |
| `STRATA_KV_ROT` | Hadamard rotation for int8 KV (a kernel variant) |
| `STRATA_SSD_KEEPALIVE`, `STRATA_SSD_KEEPALIVE_WINDOW` | Periodic reads that keep the SSD awake. Keep them; their value on Apple SSDs is a measurement item. |
| `STRATA_FETCH_THREADS` | Tiered mode: expert fetch threads (default 8) |
| `STRATA_LOOKAHEAD`, `STRATA_LOOKAHEAD_K` | Tiered mode: router lookahead prefetch |
| `STRATA_TEST_CACHE_FAIL` | Test hook: the first N long-lived buffers of startup step 9 (section 2.3) fail as an allocation does; `metal_alloc_fail_test` ([09](09-testing-and-validation.md), section 9.1) |
| `STRATA_SNAPSHOT_VERIFY`, `STRATA_SNAPSHOT_FULL_CAPTURE` | Verify snapshots after save and restore |
| `STRATA_WATCHDOG_S` | Watchdog limit (default 60 s, 0 = off) |
| `STRATA_CKPT_REREAD` | Re-read the prompt instead of restoring a checkpoint |
| `STRATA_LOGPOS`, `STRATA_LOGPOS_EXTRA`, `STRATA_LOGPOS_TOPK` | Position logging |
| `STRATA_DECODE_TIMING` | Per-request `strata decode timing:` line |
| `STRATA_PREFILL_TIMING` | Prompt-path GPU time by phase (`prefill.cpp:948-1007`); on a Mac it comes from counter sample buffers |
| `STRATA_PREFILL_BF16X2` | The prompt path's BF16-weight GEMMs also multiply the activations' BF16 remainder (`prefill.cpp:108-121`, read with `atoi`): 1, every BF16 projection, the hyper-connection's included; any other non-zero value, all but the hyper-connection's; 0 or unset, off ([04](04-metal-kernels.md), section 6.2) |
| `STRATA_PREFILL_DUMP_R` | A file path. After each chunk the host appends `int64 pos` and the 10,240 floats of R for every position that is a multiple of 64, read from the prefill arena; the file is opened `wb` at a call's first chunk and `ab` after ([04](04-metal-kernels.md), section 6.1) |
| `STRATA_STATE_HASH`, `STRATA_STATE_HASH_GDN` | State fingerprints, the same hash function over the same bytes (gate C3) |
| `STRATA_DUMP_FIRST_LOGITS` | Diagnostic dump |
| `STRATA_VERIFY_DEBUG`, `STRATA_VERIFY_PROFILE`, `STRATA_VERIFY_NODES` | Window diagnostics. `VERIFY_NODES` prints the dispatch count of the encoded window. |
| `STRATA_DBG_NAN` | Non-finite diagnostics, any value. **Window** (`verify.cpp:1183-1197`): after a window's wait the host scans its T rows of logits in window scratch. For the first row with a non-finite value, once per process, it prints `strata dbg: verify window at position <pos0>, row <t>: <bad> of <n_vocab> logits non-finite (token out <id>)` (the `printf` format of upstream, `%lld` and `%d` as there). **Prompt path:** upstream's MoE layer line and end-of-prompt line, verbatim ([04](04-metal-kernels.md), sections 6.6 and 6.1). |
| `STRATA_TEST_VERIFY_STALL` | Test hook: simulate a stalled window to exercise the watchdog |
| `STRATA_COMMIT_SYNC` | Wait for every commit before continuing |
| `STRATA_MTP_PREFILL_SYNC` | Diagnostic |
| `STRATA_POOL_TRACE` | Becomes `miss trace: layer L what n` in tiered mode |
| `STRATA_IO_THREADS` | DirectFile worker threads |
| `STRATA_SPEC_COUPLED` | Coupled draft sampling |
| `STRATA_IQ_MT_MIN` | Used only by the optional CPU helper (section 12). The CUDA reference runs set it to 1 for the IQ formats, because of upstream issue #152 ([09-testing-and-validation.md](09-testing-and-validation.md), section 2.4). |
| `STRATA_POOL_SPIN_US` | Used only by the optional CPU helper |
| `STRATA_ARENA_LOCK` | Optional `mlock` of the resident set. Off by default; it warns when the lock fails (section 7.5). |

### 4.3 Ignored on a Mac

These are discrete-GPU, Windows, x86, PCIe or multi-GPU only:
- `STRATA_REFILL_BLOCKING`, `STRATA_REFILL_SERIAL`, `STRATA_RESIDENT_PIN`, `STRATA_PREFILL_LEND_PCT`
- `STRATA_SPLIT_OWN`, `STRATA_SPLIT_OWN_PLACE`, `STRATA_SPLIT_MISS_MS`, `STRATA_SPLIT_RING`, `STRATA_SPLIT_OWN_BUFFERS`,
  `STRATA_SPLIT_SMALL_OWN`, `STRATA_SPLIT_SMALL_MAX`, `STRATA_SPLIT_TIMING`
- `STRATA_EARLY_REMOTE_CONTEXTS`, `STRATA_REMOTE_SPIN`, `STRATA_REMOTE_ZEROCOPY`
- `STRATA_VERIFY_DEVICE_PLAN` (always on, in generalised form), `STRATA_DEC_BATCH`
- `STRATA_TG_FLUSH_US`, `STRATA_KV_RING_OFF`, `STRATA_KV_MAIN_OFF`, `STRATA_FETCH_PVM`
- `STRATA_PARTIAL_PIN`, `STRATA_PARTIAL_PIN_GIB`, `STRATA_ARENA_PIN_GIB`, `STRATA_NO_LARGEPAGES`
- `STRATA_CPU_YMM`, `STRATA_FORCE_AVX2`, `STRATA_IQ_GATHER`, `STRATA_IQ_PREFETCH`, `STRATA_NO_IQ512`, `STRATA_KQ256`,
  `STRATA_NO_IQ4NL`, `STRATA_NO_IQ256` (the Metal IQ kernels have one form, [04](04-metal-kernels.md), section 5.10)
- `STRATA_EMULATE_CC`: every Metal kernel is written for 32 KiB of threadgroup memory, the floor of the supported GPUs
  ([04](04-metal-kernels.md), section 2.1), so there is no smaller-device variant to emulate
- `CUDA_MODULE_LOADING`, `CUDA_VISIBLE_DEVICES`

The kernel-scope variables of [audit B 1.3](audit/B-engine-host.md), the prompt path's included, each have a row in
[04](04-metal-kernels.md), section 9.3. Those honoured are in section 4.2 (`STRATA_KV_ROT`, `STRATA_PREFILL_BF16X2`,
`STRATA_PREFILL_TIMING`, `STRATA_PREFILL_DUMP_R`, `STRATA_DBG_NAN`). The rest are ignored:
- CUDA and HIP A/B and tuning arms: `STRATA_GR_UNFUSED`, `STRATA_GR_V3`, `STRATA_HC_SPLIT`, `STRATA_PREFILL_MMQ`,
  `STRATA_PLE_BATCH`, `STRATA_GDN_CONV_SERIAL`, `STRATA_GDN_REC_HEADS`, `STRATA_GDN_PIPELINE`,
  `STRATA_INDEXER_PER_TOKEN`, `STRATA_MTP_BATCH`, `STRATA_MTP_BATCH_F16`, `STRATA_ROPE_TABLE`, `STRATA_QSA_WARP`,
  `STRATA_PROMPT_ATTN_V1`, `STRATA_SCORES_MULTI`, `STRATA_SELECT_WMMA`, `STRATA_TOPK_OLD`, `STRATA_OLD_GROUPED`,
  `STRATA_GROUPED_PAIR_MIN_HITS`, `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK`, `STRATA_OLD_IQ_MMVQ`,
  `STRATA_HIP_ROUTER_OLD`, `STRATA_HIP_WMMA`, `HIPBLASLT_*`. The Mac runs one arm of each. Three of them fix the arm
  the CUDA references must run: `STRATA_ROPE_TABLE=1` ([03](03-model-reference.md), work item R12),
  `STRATA_PREFILL_MMQ=0`, and `STRATA_PLE_BATCH` unset ([09](09-testing-and-validation.md), section 2.4).
- Staging of the discrete-GPU prompt path: `STRATA_GROUP_COPY`, `STRATA_KV_STAGE_OWN`, `STRATA_PREFILL_ISSUER`,
  `STRATA_PREFILL_RING`, `STRATA_PREFILL_STREAM_MIN`, `STRATA_STAGER_RING`, `STRATA_STAGER_THREADS`.
- Debug statistics of the prompt selection: `STRATA_SEL_OVERLAP`, `STRATA_IDX_FP16_CHECK`.
- `STRATA_PROMPT_ATTN_OLD` and `STRATA_SELECT_OLD`, which select the arm Phase 1 always runs. They are honoured once
  the Phase-2 kernels exist ([04](04-metal-kernels.md), section 2.6).
- `STRATA_QSA_DUMP` and `STRATA_QSA_DUMP_LAST`, in Phase 1 ([04](04-metal-kernels.md), work item L23).

A Metal kernel variant switch that corresponds to one of them reuses its name and is documented in
[04-metal-kernels.md](04-metal-kernels.md).

### 4.4 New variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `STRATA_METAL_WAIT` | `block` | How the host waits for a window: `block` (`waitUntilCompleted`) or `spin` (spin on a `MTLSharedEvent`'s `signaledValue` with `__builtin_arm_yield()`). The default is chosen by measurement M-1 (section 15). |
| `STRATA_METAL_ERRORS` | 0 in release, 1 in debug | 1 sets `MTLCommandBufferDescriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus`, so errors name the failing encoder |
| `STRATA_METAL_SMALL_TILES` | 0 | Reserved and ignored. It would select 32 KiB variants if a later kernel uses more threadgroup memory; no Phase-1 kernel does. |
| `STRATA_METAL_ICB` | 0 | 1 uses indirect command buffers per T, once implemented ([04](04-metal-kernels.md), section 8) |
| `STRATA_METAL_DISPATCH` | `serial` | Unset, empty or `serial`: every segment uses `MTLDispatchTypeSerial`; `concurrent`: `MTLDispatchTypeConcurrent` with a buffer-scope memory barrier after each singleton or declared group of independent dispatches. Other values fail the generic Encoder call and discard its unsubmitted command buffer. The caller is responsible for declaring only groups with no write/read or write/write dependency. The standalone `metal_encoder_test` checks the generic policy; production bitwise parity across both modes remains the L-14 `metal_dispatch_modes_test` gate. The default stays `serial` unless L-14 shows a gain ([04](04-metal-kernels.md), section 8.2). |
| `STRATA_METAL_GEMM` | `mpp` on Apple10 (M5), `simd` elsewhere | The prefill GEMM: `mpp` uses Metal Performance Primitives' `matmul2d` on F16 tiles, `simd` uses `simdgroup_matrix` tiles ([04](04-metal-kernels.md), section 6.2). `mpp` on an older GPU is allowed if L-6 finds it supported. Prefill is KL-gated, so the two give close but not equal outputs. |
| `STRATA_GPU_BUDGET_MIB` | - | Same as `--gpu-budget-mib` |
| `STRATA_METAL_POISON` | 0 | 1 fills newly allocated buffers with `0xFF` once, at creation. Owners that require zero-initialized state clear their regions afterward. Unwritten float scratch reads become NaNs; integer and byte reads see all-ones values. The Metal tests set it and use type-appropriate expectations. Upstream has the same switch only as a constructor argument (`DeviceArena(poison)`, `device.cu:197`). |
| `STRATA_METAL_ARCHIVE` | unset | The pipeline archive's folder: unset, empty or `1` is `~/Library/Caches/strata/`; `0` turns the archive off; anything else is a folder ([08](08-build-and-distribution.md), section 10.1). The tests and CI point it at their own folder. |
| `STRATA_METAL_CAPTURE` | unset | Same as `--metal-capture` ([08](08-build-and-distribution.md), section 10.2) |

---

## 5. The serve protocol (normative)

The Metal engine must be interchangeable with the CUDA engine behind `serve/server.py`. Everything in this section is
**byte-compatible** with upstream (`generate.cpp:4510-5610`, `serve/server.py:246-466`) unless it says otherwise.

### 5.1 Transport

- The server starts `Popen([exe, "--serve", *args], text=True, bufsize=1)` (`server.py:246`).
- The protocol is one request per line on stdin and one response per line on stdout.
- Human-readable diagnostics go to stderr, which the server writes to its log.
- stdout is unbuffered, and every response line ends with `\n`.
- The stdin reader is a dedicated thread using `read(2)`, not `std::cin` (section 2.2).

### 5.2 Startup lines

1. stderr: load progress and warnings.
2. stderr: `strata serve: N MiB of the GPU working set free ...`. Below 256 MiB, print the `LOW ...` variant suggesting
   `--vram-reserve-mib` (as `:4577-4592`, with "VRAM" replaced by "the GPU working set").
3. stdout: one `INFO` line of `key=value` pairs. The upstream keys are listed below in upstream order; the new keys go
   after `engine=`. The server parses INFO into a dict, so extra keys are safe.
4. The watchdog thread starts.
5. stdout: `READY <max_context> stop`. The literal `stop` advertises STOP support and must be present.

**INFO keys, in upstream order** (`:4613-4628`):

| Key | Upstream meaning | Mac value |
| --- | --- | --- |
| `context` | `--max-context` | same |
| `kv` | KV format | same |
| `kv_resident` | KV cells kept in VRAM | `0` |
| `expert_slots` | VRAM expert slots, all GPUs | Number of experts in the resident set (24,576 in full mode) |
| `expert_cache_mib` | VRAM expert MiB, all GPUs | MiB of the resident set |
| `expert_slots_primary` | slots on CUDA0 | Same as `expert_slots` |
| `expert_cache_primary_mib` | MiB on CUDA0 | Same as `expert_cache_mib` |
| `spec` | S | same |
| `mtp_max` | `--mtp-max-t` | same |
| `lookup` | `--suffix-draft` | same |
| `vram_free_mib` | free VRAM after load | `(W − currentAllocatedSize) >> 20` after load |
| `cvec` | control-vector summary | same format |
| `arena_mib` | resident or total expert bytes in host RAM | MiB of routed experts the engine keeps in memory: everything in full mode; resident set plus slot pool in tiered mode. `server.py:warn_tight_ram` reads it. |
| `pool_workers` | CPU pool size | `0` (or the CPU helper's size, section 12) |
| `pcie_frac` | `%.2f` | `0.00` |
| `spec_min_p` | `%.2f` | same |
| `conversation_cache_mib`, `conversation_cache_slots`, `conversation_cache_min_free_mib` | | same |
| `engine` | version `x.y.z` | The project's CMake project version (section 3.10). It must be the last upstream key. |

**New INFO keys**, appended after `engine=`:

| Key | Value |
| --- | --- |
| `backend` | `metal` |
| `metal_rev` | Integer from `METAL_REV`, raised by one in every Mac release and never reset ([08](08-build-and-distribution.md), section 11.1). Setup and the server compare it with `MIN_ENGINE_METAL` ([07-server-setup-port.md](07-server-setup-port.md), section 4.1), which is raised only when they need a newer engine. |
| `gpu` | Device name with spaces replaced by `_`, for example `Apple_M5_Pro` |
| `gpu_budget_mib` | W in MiB (after `--gpu-budget-mib`) |
| `residency` | `full` or `tiered` |
| `ssd_experts` | Number of experts outside the resident set (0 in full mode) |
| `plan_mib` | The plan's `total_mib` ([05](05-memory-and-residency.md), section 3.4): the memory this engine planned to keep resident. `server.py:warn_tight_ram` reads it on a Mac ([07-server-setup-port.md](07-server-setup-port.md), section 6.4). |

### 5.3 Requests (server to engine)

```
GEN  <max_new>[ key=value ...] <id,id,...>
GENI <max_new>[ key=value ...] <embeddings_file> <id,id,...>
STOP
QUIT
```

- Each request runs under a `BusyScope` for the watchdog. `stop_req` is cleared at the start of the request.
- `GENI` without `--vision` replies `ERR this engine was started without --vision`.
- **Path robustness (Mac improvement, backward-compatible).** The last space-separated field is the id list. For
  `GENI`, the embeddings path is everything between the last `key=value` token and the id list. This allows a path with
  spaces, which is likely under `~/Library/Application Support` or `/Volumes/...`. A server that never sends spaces is
  unaffected.
- Unknown keys are skipped silently.

| Key | Type / clamp | Default |
| --- | --- | --- |
| `cvec` | control-vector selector | |
| `temperature` | float | 0 (greedy) |
| `top_p` | float | 1 |
| `top_k` | int; must be 1..64 when sampling | 20 |
| `min_p` | float | 0 |
| `penalty_last_n` | int (coupled history cap 4096) | 0 |
| `penalty_repeat`, `penalty_freq`, `penalty_present` | float | neutral |
| `seed` | uint64 | |
| `pcie_frac` | parsed and clamped to 0..1, then ignored on a Mac | engine value |
| `spec_min_p` | clamped 0..1 | engine value |

### 5.4 Image embeddings for GENI

The rules match upstream `:4745-4804`.

**File format.**
- The file is a concatenation of `SVE1` records.
- Each record is `int32 {0x31455653, n_tokens, nx, ny, n_embd}` followed by `float32[n_tokens][n_embd]`.
- `n_tokens == nx*ny`, and `n_embd == 2560`.

**Matching records to the prompt.** Each run of `<|image_pad|>` (248056) tokens consumes the next record, in order.

**M-RoPE positions.**
- Text token: (p, p, p).
- Image row j: (p, p + j/nx, p + j%nx).
- After an image, p advances by max(nx, ny).

**The position table and the RoPE table.** Upstream keeps an int32 table `[cells][3]` of (t, h, w) per cell and
rebuilds it for every cell on each GENI request, before the request's first chunk (`:4745-4832`):
- prompt cells as above, with p starting at 0;
- every cell c from n (the prompt length) to the end: (p + c − n) three times, where p is its value after the
  prompt;
- on a text request that follows a GENI request, and after any error below, the identity (c, c, c) for every cell.

The Mac has no position table on the GPU. The kernels read the folded table `rope[cell][32]`
([04](04-metal-kernels.md), section 4.9), whose pair i at cell c is the cos and sin of the position
`tab[c][i % 3]`. So the host:
- parses the records and builds the position table on the CPU, as upstream, before it encodes anything for the
  request;
- waits until the queue is idle (`wait_commit()` and the last command buffer completed), because every window,
  commit and chunk reads the RoPE table;
- recomputes the RoPE rows from the first image cell to the last cell, in double with the YaRN settings of
  [03](03-model-reference.md), section 6.7, split across the performance cores with `dispatch_apply`. Rows before
  the first image cell are (c, c, c), the identity. If an earlier GENI request left a lower first rewritten cell
  (below), the recompute starts there instead. A full 262,144-cell rebuild took 7 ms on the M5 Pro used for this
  study (88 ms on one core).
- records the first rewritten cell. A text request that follows, or an error, recomputes the rows from that cell as
  the identity and clears the record. A text request after a text request touches nothing.

**Cache keys.** Each image is fingerprinted with FNV-1a over the grid and the floats, for the conversation cache.

**Staging for the GPU.** A part of the prompt with image rows always goes through `Prefill`, never through decode
windows (section 11.2, step 8). For each chunk, the host:
- writes `0x80000000u` as the token word `tokens[t]` of every image row t, and the token id of every other row;
- copies image row j's 2,560 floats from the records into `img_in[t]` of the prefill arena
  ([04](04-metal-kernels.md), section 6.1). Only marked rows are read;
- leaves the chunk's step rows as for text: a row's position is its cell. The vision positions reach the kernels
  only through the RoPE table (below).

`strata_embed_image` then copies the marked rows into the embedding ([04](04-metal-kernels.md), section 6.7). The
token check of section 5.7 (`ERR a token id is outside the vocabulary`) runs on the prompt's ids before the flags
are set, so bit 31 never marks a real token: every id is below 248,320.

**Errors**, each sent as `ERR <message>`, verbatim. Each one resets the tables to the identity first:
- `cannot open <path>`
- `bad embeddings file (expected strata-vision records of width 2560)`
- `short embeddings file`
- `the prompt has more images than the embeddings file`
- `image k has n rows but fewer <|image_pad|> tokens`
- `the embeddings file has more images than the prompt`
- `the prompt cannot end in an image`

The Metal host also rejects a request before allocation when its combined SVE1 payload exceeds 268,435,456 bytes.

### 5.5 Responses (engine to server), in order for one request

| Line | Meaning |
| --- | --- |
| `RESUME <resume>` | Prompt tokens reused from the live session, a checkpoint or a parked conversation |
| `PP <done> <total> <ms %.0f> <fresh tok/s %.1f>` | Prompt progress, from the batched path and the short-read path. The server reads field 4 as tok/s. |
| `REUSED <resume>` | Printed once the prompt is fully read |
| `T <id>` | One emitted token per line, in order |
| `DONE <generated> <prompt_n> <prompt_ms %.1f> <decode_ms %.1f> <finish> <drafts_accepted> <drafts_offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_MB %.1f>` | Exactly 13 fields. `finish` is `stop` (EOS), `cancel` (STOP received) or `length`. |
| `ERR <message>` | The request failed. The engine stays alive unless the error is fatal (section 2.4). |

### 5.6 DONE fields on a Mac

| Field | Mac meaning |
| --- | --- |
| `hits` | Routed-expert lookups served from the resident set |
| `lookups` | All routed-expert lookups (rows × 10 per layer, summed). In full mode, `hits == lookups`. |
| `ram_blobs` | Tiered mode: misses whose bytes were already in memory, either in a slot (slot cache hit) or in the page cache (checked with `mincore`). Full mode: 0. |
| `file_blobs` | Tiered mode: misses read from the SSD. Full mode: 0. |
| `file_MB` | MB read for `file_blobs` |

All other fields keep their upstream meaning.

### 5.7 ERR messages that must stay verbatim

- `ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>`
- `ERR bad request: %s`
- `ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)`
- `ERR a token id is outside the vocabulary`
- restore failures (upstream `:4909`, `:4916`, `:4989`)
- `ERR an adaptive refill failed`: tiered mode, when a swap read fails
- state-fingerprint mismatches (`:5460`, `:5555`)

These messages are deleted with their mechanism: `ERR refilling a lent slot failed: ...` (slot lending does not exist).

These messages are new:
- `ERR gpu error: <description>` (section 2.4)
- `ERR expert read failed: layer L expert E: <strerror>` (tiered mode, fatal)

### 5.8 stderr after each DONE

The four upstream lines are kept, with the same text and Mac meanings:

```
strata serve: prompt N tokens = R reused + F read in X ms (tok/s), G generated in Y ms (tok/s), drafts accepted a of o, K checkpoints[ (cancelled)]
strata serve: decode expert cache hit rate: %.1f%% (h hits / l lookups)
strata serve: resident RAM: %.2f GiB of experts in RAM, N exchanged with the VRAM tier, M blob reads from the file
strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs %.1f MB read ...
```

"Exchanged with the VRAM tier" counts tiered-mode swaps into the resident set. Keep the wording so log parsers keep
working.

### 5.9 Example exchange (Mac, full mode)

```
<- (stderr) strata serve: 9876 MiB of the GPU working set free ...
<- INFO context=65536 kv=fp16 kv_resident=0 expert_slots=24576 ... pcie_frac=0.00 ... engine=0.1.34 backend=metal metal_rev=1 gpu=Apple_M5_Pro gpu_budget_mib=53084 residency=full ssd_experts=0
<- READY 65536 stop
-> GEN 512 temperature=0.7 top_p=0.95 top_k=20 seed=42 248045,872,198,...
<- RESUME 1830
<- PP 2048 4096 812 2521.3
<- PP 4095 4096 1601 2556.8
<- REUSED 1830
<- T 1234
<- T 99
...
<- DONE 57 4096 1620.1 2200.4 stop 31 48 1830 13680 13680 0 0 0.0
-> QUIT
```

### 5.10 Vision helper (separate process)

`strata-vision` keeps its protocol (`tools/vision/strata_vision.cpp`).

**Exchange.**
1. The helper prints `READY <n_embd>` after a warmup encode at 2048×2048.
2. The server sends `ENC <image path> <out path>`.
3. The helper replies `OK <n_tokens> <nx> <ny> <ms>` or `ERR <msg>`.
4. `QUIT` exits the helper.

**Parsing.** The image path ends at the last space before the output path. The output path must have no spaces;
`tempfile.mkdtemp` paths have none.

**Mac change.** Only the thread count changes (section 2.1).

---

## 6. Artifacts and loaders

### 6.1 What the Mac engine reads

All formats are little-endian, and so is Apple Silicon, so no byte swapping is needed. Every producing tool is Python +
numpy and runs on macOS. The full map is [audit B 2.1](audit/B-engine-host.md).

| Artifact | Producer | Mac engine use |
| --- | --- | --- |
| GGUF shards `data/models/<tag>/*.gguf` (Q2_0 = type 42, 18 B per 64 weights) | Hugging Face download | Mapped read-only. Full mode: wrapped as no-copy windows. Tiered mode: dense tensors and hot experts are copied out ([05-memory-and-residency.md](05-memory-and-residency.md)). |
| Native `index.txt` v3, `conversions.json`, `dense.bin` | `tools/iq_pack.py` | Read ([03](03-model-reference.md), section 4.5). Rows with `code_bits 8, dst_bytes 0` carry only a shape; their real type comes from the GGUF. A float tensor is served from the GGUF unless `conversions.json` lists it; conversion records must name a router tensor and declare the exact packer method for zero-error F32-to-BF16 conversion, and only those rows are read from `dense.bin`. The loader checks source shard names and sizes, source type, method, exact flag and error value, and SHA-256 syntax. The light path does not read tensor payloads, so it does not recompute the declared source SHA-256 or independently prove the converted values are exact. `extra.bin` exists only with `--base` and is ignored. |
| `native_experts.txt` v3 or v4 | `iq_pack.py` | Read ([03](03-model-reference.md), section 3.5). Per layer: `layer gu_type d_type offset blob gate_off up_off down_off [shard \| gate,up,down]`, offsets absolute within each role's own shard. Every catalogue pack is v3; v4 is accepted; a version above 4 (`kExpertLayoutVersion`) is refused, and so is a line without the three offset columns. An `experts.bin` in the pack is ignored and logged: experts are always read from the GGUF. |
| `packs/<tag>/tokenizer/*` | `strata_tokenizer.py` | Server only |
| `mtp/rt/{experts.bin, dense.bin, dense.txt}` | `tools/mtp_rt.py` | Read ([03](03-model-reference.md), section 4.4). `experts.bin` holds 512 S2 blobs, Strata's canonical Q2_0 form (707,788,800 B). `dense.bin` tensors are 256-B aligned; F32 norms are stored as 1+w; large projections are Q8_0 and the rest BF16. `dense.txt` rows: `short kind rows cols offset bytes`. |
| `mtp/rt/draft_vocab.bin` | copied by setup | Raw int32, **not sorted**. Build an id-to-subset map. |
| `data/expert-profile.bin` (196,632 B), `expert-profile-coder.bin` (98,328 B) | `tools/make_profile.py` | STRP reader (section 1.4) |
| Control vectors `data/experimental-speed-projection/*.gguf` | shipped | GGUF arch `controlvector`, tensors `direction.<l>` F32[2560], 1 ≤ l < 48 |
| Optional `token-embd-bf16.gguf` (arch `strata-embd`) | `tools/embd_bf16_pack.py` | `--embd-gguf` |
| PLE table `per_layer_token_embd.weight` | download | Found by name: shard 2 of the ISTA and Coder files (at file offset 192), a range of shard 1 in the Swift files. IQ4_NL, [160, 320001536], 90 B/row, 28,800,138,240 B. Read per token with `pread` (section 7.4). |
| Optional `ple-fp8.gguf` (arch `strata-ple`, FP8 E4M3 in I8, plus `strata.ple.scale`) | `tools/ple_fp8_pack.py` | Supported later; needs an FP8 decode in the PLE kernel |
| Canonical pack (`experts.bin`, canonical `dense.bin`) | `strata_pack.py` on AVX-512 hosts only | **Not supported** on the Metal engine |
| `strata-<tag>.json` run config | setup | Rewritten for the Mac by setup ([07-server-setup-port.md](07-server-setup-port.md)) |

### 6.2 Format rules the loader must honour

These are the rules in [audit B 2.4](audit/B-engine-host.md). Each one has a loader unit test.

1. GGUF `data_start = align_up(end of the tensor table, general.alignment or 32)`, and tensor offsets are relative to
   it. `newBufferWithBytesNoCopy` needs a 16 KiB-aligned pointer and a length that is a multiple of 16 KiB. So map the
   whole file, wrap page-aligned windows of it, and address each tensor as (window, offset). Never wrap per tensor.
   The mapping owner keeps the source pages mapped until its `Context` and wrapped `MTLBuffer`s are destroyed;
   `Buffers::wrap_no_copy` does not take ownership or install a deallocator.
2. Native `index.txt` rows with `code_bits 8, dst_bytes 0` need the GGUF shards opened to learn their type (upstream
   `NativeDense::served_names`, `generate.cpp:1868-1883`).
3. The shard column of `native_experts.txt` names one shard per layer in v3 and one per role in v4, and the offsets
   are absolute in the named shard. Never apply shard 1's `data_start` to another shard. Swift's experts of layers
   13-47 (IQ2_XS) and 12-47 (IQ3_XXS) are in shard 2 ([03](03-model-reference.md), section 3.9).
4. `draft_vocab.bin` is unsorted (`sampler.cu:670-674`).
5. `mtp_rt` F32 norm tensors already include the +1 (GemmaRMSNorm fold). Do not add it again.
6. Canonical `dense.bin` plane alignment is relative to each tensor's start. This matters only for the MTP runtime's
   blobs.

Additional Mac rules:

7. **Window size.** A no-copy window must be at most `maxBufferLength` (38.88 GiB measured on a 64 GB M5 Pro, and
   smaller on smaller Macs). Windows cover the ranges the engine reads and never map the PLE tensor; outward
   16 KiB alignment may include up to one page of its edge as padding at an adjacent tensor boundary. Windows are
   planned by the algorithm of [05](05-memory-and-residency.md), section 4.2: merge, round to 16 KiB, split at a
   tensor boundary or, inside an expert tensor, between experts. If one expert slab or non-expert tensor is larger than
   `maxBufferLength`, which none is, exit 1.
8. **Expert addressing.** In the GPU, an expert blob is (window index, byte offset) for each of gate, up and down. The
   loader builds this table once, as a `uint64` GPU address per role, using `MTLBuffer.gpuAddress + offset`. The GPU
   kernels read through the table, so full and tiered modes share one kernel.
9. **Routers.** `n_expert` is 512, or 256 for Coder. It comes from the `native_experts.txt` header and must equal
   `qwen4exp.expert_count` and the `ne1` of every router. Every router the engine reads is BF16: from the GGUF for
   ISTA and Coder, from `dense.bin` for Swift, whose GGUF stores them as F32 with BF16 values (listed in
   `conversions.json`, narrowed exactly). One router kernel serves both widths ([04](04-metal-kernels.md)).

---

## 7. Platform layer: OS memory and file APIs

Each row in [audit B 3.2](audit/B-engine-host.md) and [8.2](audit/B-engine-host.md) is accounted for here or marked as
belonging to code that is not compiled on a Mac.

### 7.1 Page size

macOS on Apple Silicon uses **16 KiB** pages. Every VM page computation must use `getpagesize()` or `vm_page_size`,
never 4096.

| Site | Problem | Fix |
| --- | --- | --- |
| `src/kernels/ngram.cpp:263` (`page = 4096`) | `madvise` on a range that is not 16 KiB-aligned returns EINVAL, and the result is ignored | `getpagesize()` |
| `src/core/expert_source.cpp:840-841` | same | Not compiled on a Mac. The Mac `ExpertStore` uses `getpagesize()`. |
| `src/ngram/ple_reader.cpp:339` (`PAGE = 4096`) | It is a DirectFile sector granularity, not a VM page | Keep |
| Every no-copy wrap | Needs 16 KiB alignment and length | `Buffers` asserts both |

### 7.2 Memory probes: macOS

`include/strata/platform/memory_probe.hpp` and `src/platform/memory_probe.cpp` provide the memory readings used by
the Mac planner and diagnostics. `available_memory()` uses `host_statistics64(HOST_VM_INFO64)` and multiplies
free, inactive, purgeable and speculative pages by `vm_page_size`. `total_memory()` reads
`sysctlbyname("hw.memsize")`. `process_memory()` combines `task_info(TASK_VM_INFO)`,
`getrusage(RUSAGE_SELF).ru_majflt` and `sysctlbyname("vm.swapusage")`; it returns no sample if any source
fails.

The GPU budget is `Context::gpu_budget()`: `recommendedMaxWorkingSetSize`, capped by `--gpu-budget-mib`, minus
`currentAllocatedSize`.


Callers that need these functions:

| Upstream site | Upstream API | Mac |
| --- | --- | --- |
| `expert_source.cpp:219-283` `available_memory_bytes` | `/proc/meminfo`, cgroup v2, `sysconf(_SC_AVPHYS_PAGES)` (absent from the macOS SDK, so a compile error) | Not compiled. The Mac memory plan calls `available_memory()`. |
| `generate.cpp:716-754` `mem_sample` | Windows / `/proc` | The Mac stall report calls `process_memory()` and `available_memory()` (section 14) |
| `memory.cpp:133` | `sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE` | Works; `total_memory()` uses `hw.memsize` |
| `memory.cpp:60-105` DXGI budget | Windows | n/a |

### 7.3 Withdrawn: legacy conversation-memory probe

The old helper queried Windows `GlobalMemoryStatusEx` or Linux `/proc/meminfo`, returned unknown on macOS, and was used only by the unbuilt PC generation program. It has been removed from the Mac target under D-IMPL-238. Conversation-cache admission remains a planned Metal-engine responsibility; it must use the Mac planner and its tests rather than revive this cross-platform helper.

### 7.4 Patch: `src/platform/direct_file.cpp`

`open(O_RDONLY|O_DIRECT)` at `:308` fails to compile because macOS has no `O_DIRECT`. Under `__APPLE__`:
- `open(path, O_RDONLY)`;
- then `fcntl(fd, F_NOCACHE, 1)` (no alignment requirement);
- then `fcntl(fd, F_RDAHEAD, 0)`.

Keep the `posix_memalign` buffers (`:36`) and the `pread` worker threads (`:284`). After this patch, enable
`direct_file_async_test` on Apple (`CMakeLists.txt:1015`).

### 7.5 Locking memory

`mlock` exists on macOS, but `RLIMIT_MEMLOCK` and the kernel wire limit bound it. Wiring 30+ GB usually fails.

Policy:
- Never lock by default.
- `STRATA_ARENA_LOCK=1` tries `mlock` on the resident set and prints one warning on failure.
- `--ple-io ram` tries `mlock` on the PLE table, then falls back to touching the pages, as upstream `ngram.cpp:270`
  does, with the warning in section 3.2.

Whether a residency set keeps buffers out of compression and swap is measurement M-4 in
[05-memory-and-residency.md](05-memory-and-residency.md).

### 7.6 Other OS sites

| Upstream site | Mac |
| --- | --- |
| `expert_source.cpp:1274` `posix_fadvise(POSIX_FADV_DONTNEED)` (absent on macOS) | Not compiled. The Mac engine never needs to drop page-cache ranges. To release memory it unmaps or uses `madvise(MADV_FREE_REUSABLE)` on anonymous memory. |
| `pinned.cu` (hugetlb, `MAP_HUGE_2MB`, large pages, `cudaHostRegister`, shared arena file) | Not compiled. Anonymous arenas use `mmap(MAP_ANON\|MAP_PRIVATE)` with 16 KiB alignment, wrapped no-copy. |
| `generate.cpp:104-112` `under_wddm` (`/dev/dxg`) | Not compiled |
| `generate.cpp:787-806` minidump | Replaced by the crash report (section 14) |
| `generate.cpp:989-995` `CUDA_MODULE_LOADING` | Not compiled |
| `generate.cpp:1463` `--ple-io ram` refused on Windows | Allowed with a warning (section 3.2) |
| `generate.cpp:4514-4560` stdin reader with `read(0, ...)` | Portable; copied |
| `gguf_reader.hpp:404-405` `mmap(PROT_READ, MAP_PRIVATE)` | Portable |
| `weights.cpp:78-82` `fseeko` | Not compiled. The Mac `WeightStore` uses `pread`. |

---

## 8. CUDA host runtime to Metal

### 8.1 API map

| CUDA | Metal / Apple |
| --- | --- |
| `cudaMalloc` / `cudaFree` | `newBufferWithLength:options:` from the bump arena. Shared storage is the default; Private only where a measurement shows a gain. |
| `cudaHostAlloc(Mapped\|Portable)` + `cudaHostGetDevicePointer` | One `MTLResourceStorageModeShared` buffer. The CPU uses `contents` and the GPU binds the buffer; they are the same bytes. |
| `cudaHostRegister` / `Unregister` | `newBufferWithBytesNoCopy:length:options:deallocator:` over 16 KiB-aligned memory / release |
| `cudaMemcpy[Async]` H2D / D2H / D2D | Mostly deleted. The CPU writes `contents` directly. A blit encoder is used only for Private buffers. |
| `cudaMemset[Async]` | `memset` on `contents`, or `fillBuffer:range:value:` |
| `cudaStreamCreate` (compute, copy, MTP, adapt) | One `MTLCommandQueue` for compute. A second queue only if measurement M-2 shows that overlapping the MTP drafter with the commit pays. |
| `cudaEventRecord` / `Query` / `Synchronize` | `MTLSharedEvent` signal / `signaledValue` / `waitUntilSignaledValue:timeoutMS:`, or `addCompletedHandler:` |
| `cudaStreamWaitEvent` | `encodeWaitForEvent:value:` |
| `cudaLaunchHostFunc` | `addCompletedHandler:`, or an `MTLSharedEventListener` notification block |
| `cudaStreamBeginCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` | Re-encode one command buffer per window. `MTLIndirectCommandBuffer` per T is optional later (`STRATA_METAL_ICB`). |
| `cudaStreamQuery` as a WDDM flush | Nothing; `commit` submits immediately |
| `cudaMemGetInfo` | `Context::gpu_budget()` |
| `cudaDeviceSynchronize` / `cudaStreamSynchronize` | `waitUntilCompleted` on the last committed command buffer of the queue |
| `cudaSetDevice`, `OnDevice` | Deleted |
| `cudaGetDeviceProperties` | `MTLDevice` name, `hasUnifiedMemory`, `maxBufferLength`, `maxThreadgroupMemoryLength`, `supportsFamily`, `recommendedMaxWorkingSetSize` |
| `cudaGetLastError` after each launch | Command buffer `status` / `error` in the completion handler; `errorOptions` in debug (`STRATA_METAL_ERRORS`) |
| `%globaltimer` / `gpu_stamp` | `MTLCounterSampleBuffer` with `MTLCommonCounterSetTimestamp` at stage boundaries; command buffer `GPUStartTime` / `GPUEndTime` |

### 8.2 Rules

1. **No GPU thread may wait on memory the CPU writes while the command buffer runs.** Host-to-GPU ordering inside a
   command buffer uses `encodeWaitForEvent` only. Device atomics in MSL are relaxed-only, and CPU-to-GPU visibility
   during a running command buffer is not a supported contract ([audit K5 4.2 B16](audit/K5-orchestration-vram-tests.md)).
2. **Per-step values live in buffers, not in kernel arguments.** This includes token, position, step, n_keep, T, the
   row counts and the sampler parameters. It keeps the encoded command sequence valid for reuse and for a later ICB
   path. It is the same rule as CUDA graphs (`graph.hpp`).
3. **Fixed addresses.** Every long-lived buffer comes from the bump arena at startup. No allocation happens per token.
   This is required for residency sets and ICBs.
4. **Ordering on the CPU side.** Replace `_mm_sfence()` and seq_cst fences with
   `std::atomic_thread_fence(std::memory_order_release)` before signalling an event. Read GPU results only after
   `waitUntilCompleted` or an observed event value. Both order the GPU's writes before the CPU's reads.
5. **Monotonic event values.** A 64-bit counter per queue, never reset. This gives the same `>= ring` semantics as
   upstream's monotonic CAS `raise_flag` (`verify.cpp:1225-1238`). Host reservation, host signalling and GPU signal
   encoding share a mutex, so a reservation cannot race ahead of a signal's counter update. Reservations start above
   both the observed event value and the last GPU-encoded signal. An event belongs to one command queue. Encode all
   work before enqueueing or committing its command buffer; signals for one event must be encoded as contiguous
   command-buffer ranges and submitted in increasing value order. A higher signal cannot be submitted until every
   lower signal from another command buffer has been committed. A GPU wait cannot be submitted ahead of any pending
   lower GPU signal it depends on, and a lower signal cannot be added behind an already enqueued wait. This prevents
   a wait at the head of the single queue from blocking its own producer.
6. **Labels.** Every command buffer and encoder gets a label, for example `window T=4 pos=1234`, `commit`, `mtp`,
   `prefill chunk 3`, or `layer 17`. These appear in error messages, GPU captures and Instruments.
7. **One queue, one fence.** Every buffer is `HazardTrackingModeUntracked` ([05](05-memory-and-residency.md),
   section 7), so Metal orders nothing by itself. `Context::open` creates one `MTLFence` for the queue and seeds it
   with an update-only blit encoder, then waits for that command buffer before returning. This is the only encoder
   that does not wait first. Every later compute or blit encoder uses the common begin/end helpers: wait for the
   fence first and update it last. For every cross-command-buffer producer/consumer edge, call
   `consumer.depends_on(producer)` before submission. The wrapper checks same-queue ownership and rejects consumer
   enqueue/commit until the earlier-created producer has completed successfully; use `wait_until_completed` to
   establish that result. A queue fence preserves execution order but does not propagate GPU failures to downstream
   buffers. The wrapper cannot infer buffer access dependencies, so dispatch owners must register them. Without the
   fence a second command buffer read stale data in 20 of 20 runs ([appendix A.11](appendix-a-measurements.md),
   test 4). The encoding rules are
   [04-metal-kernels.md](04-metal-kernels.md), section 8.

---

## 9. Experts: the host side

The memory side (sizes, budgets, layout, warm-up) is in [05-memory-and-residency.md](05-memory-and-residency.md). This
section specifies what the host does per window.

### 9.1 Modes

| Mode | When | Host work per window |
| --- | --- | --- |
| **Full** | Every routed expert fits in the plan (64 GB+ Macs with Q2_0, IQ2_XS and IQ3_XXS, Swift included; Coder from 48 GB) | None. The router, grouping (the generalised E-6 plan on the GPU), the experts and the combine all run inside the window's command buffer. The host writes inputs, commits and waits. |
| **Tiered** | Some experts do not fit (32-48 GB Macs, IQ3_S on 64 GB, or `--expert-cache N`, or `--gpu-budget-mib`) | One `MissResolver` handshake per layer (section 9.3), plus swaps every `--adapt-every` rounds |

Output must not depend on the mode. Both modes run the same expert kernels on the same bytes; only the address of the
bytes differs. Gate C4 in [09-testing-and-validation.md](09-testing-and-validation.md) checks that the full and tiered
modes give bit-identical tokens.

### 9.2 What is deleted from upstream

The following are deleted:
- The CPU pool and its kernels.
- The doorbell, `flagA`/`flagB` and `y_miss`.
- The PCIe share and its three modes.
- The `GpuPlanSink` PCIe half.
- `RemoteExperts`.
- Exchange buffers, partial pinning, slot lending to prefill, and the `--expert-cache auto` VRAM formula.
- The VRAM-tier refill on lend.

Each item is listed in [audit B 10.2](audit/B-engine-host.md).

### 9.3 Tiered mode: the layer handshake

A window command buffer encodes, for each layer l:

1. The layer's pre-expert work, ending with the router and a GPU plan kernel. The plan kernel writes the distinct
   expert ids of the T rows, and for each one whether it is resident, into a shared buffer `plan[l]`.
2. `encodeSignalEvent(ev, base + 2l + 1)`.
3. `encodeWaitForEvent(ev, base + 2l + 2)`.
4. The grouped expert kernels. They read each expert's (gate, up, down) addresses from `addr[l]`, which is the resident
   table entry or a slot address.
5. The combine and the rest of the layer.

`base` advances by 2 × 48 + 2 per window.

Before enqueueing the window, the host calls `ExpertStore::prepare_command_buffer`
to register one completion callback and obtain its `ExpertLeaseSet`. After the
GPU signals that a plan is ready, the resolver adds the newly acquired slot
leases to that set before signaling the layer-ready event. This keeps slot
addresses pinned even though plan resolution happens after command-buffer
submission.

`ExpertStore::resolve_layer_after_plan` implements the host bridge for a tiered decode plan. It waits for the plan-ready value, acquires the shared plan buffer, resolves cold experts through the existing slot cache and fetch workers, and signals the immediately following layer-ready value on that same `Events` object only after address publication succeeds. It reads only the four-word header when `skip` is set, and rejects a skipped plan that reports misses. On failure it leaves the layer-ready value unsignaled. `Events::release_gpu_waits(timeout_ms, error)` satisfies waits already encoded for that event; it does not cancel commands after those waits. The caller must fail the request and ensure each waiting command buffer has no unsafe downstream consumer, or that every such consumer is guarded against request failure, before releasing a wait. The current single-Window-command-buffer outline does not meet that failure-safety condition, so production failure cleanup remains open. When safe to release, the method validates its drain timeout, advances the event to `UINT64_MAX`, drains committed or enqueued command buffers that contain its waits, and permanently disables event operations. After release, `signaled_value()` returns no value and `wait()` refuses to treat the sentinel as completion. `ModelStorage::tiered_storage()` supplies the borrowed tiered storage needed to create `ExpertStore` and remains valid only for the owner's lifetime.

This helper is not yet a production Window scheduler. Command-buffer layer encoding and lookahead scheduling remain open. The fixtures encode GPU plan kernels into one synthetic command buffer and verify that a later plan-ready event retires leases from prior layers.

After `resolve_layer_after_plan` observes layer l's plan-ready event, it retires leases for every earlier layer. The Window command buffer is serial, so this event proves that their expert reads have finished. The Window's completion callback releases any leases that remain for the current or later layers. A cache entry stays Ready after retirement, but its slot may be evicted to serve a later layer.
The `MissResolver` thread, for each layer:

1. Wait for `base + 2l + 1`, through a `MTLSharedEventListener` block or a spin on `signaledValue` (`STRATA_METAL_WAIT`).
2. Read `plan[l]`. For each non-resident expert, look it up in the slot cache of half `l % 2` (an LRU keyed by
   (l, e)). On a miss, take a slot of that half and queue a `pread` of its gate, up and down byte ranges on the fetch
   workers. 8+ reads are in flight; the reads go through the page cache, which is the RAM tier.
3. Wait for its reads. Write `addr[l]` for the missed experts. Issue a release fence. Set
   `ev.signaledValue = base + 2l + 2`.
4. Start a lookahead for layer l+1 into half `(l + 1) % 2`.
   - Decode: the experts predicted by the upstream `RouterLookahead` algorithm (`expert_source.hpp:152`;
     `STRATA_LOOKAHEAD`, `_K`).
   - Prefill: every non-resident expert of layer l+1.
   - Details are in [05](05-memory-and-residency.md), sections 5.5 and 5.7.

In a prefill chunk the plan is a `PfPlan`, written by `strata_pf_sort_scan` ([04](04-metal-kernels.md), section 6.6).
It starts with the same four fields as `MoePlan` (`n_groups`, `n_entries`, `n_miss`, `skip`), but its arrays hold 512
groups instead of 80, so the resolver reads `grp_e` and `grp_res` at `PfPlan`'s offsets. Both structures are in
`common/types.h`, and the host includes that file, so the offsets come from the same declaration. When `skip` is 1 the
resolver signals `base + 2l + 2` without reading the arrays.

Rules:
- The pool has two halves of at least H slots, where H is the largest per-layer count of non-resident experts. Layer l
  uses half `l % 2`.
- Layer l's signal fires only after every earlier command has completed. Those include layer l − 2, the last earlier
  reader of half `l % 2`, and the whole previous window, because only one expert-reading command buffer is in flight.
  So at that signal the resolver may overwrite any slot of half `l % 2`.
- The lookahead into half `(l + 1) % 2` starts only after layer l's signal, when layer l − 1 has completed.
- Before resolving a new plan, `resolve_layer_after_plan` retires every earlier layer whose reads precede the
  plan-ready event. `metal_tiered_copy_test` runs `metal_window_storage_test --tiered-copy`, which resolves
  GPU-produced plans for layers 0 and 2 in one command buffer, dispatches a real IQ2_XS gate consumer for
  layer 0, fills that parity half, then proves layer 2's plan event frees capacity while the Window command
  buffer is still in flight.
- If a later change lets two windows overlap, slots must instead be pinned per window and released in the completion
  handler.
- Timeout: if layer l's signal does not arrive within 20 s, as upstream `verify.cpp:1089-1093`, fail the request.
  Do not call `release_gpu_waits` (section 14) while a queued command buffer can run unguarded expert reads after
  that wait; production failure cleanup must first provide a per-layer command-buffer boundary or a GPU-visible
  failure guard.
- Cost model: 48 hops per window. The latency of one hop is measurement M-1. At 20 µs a hop costs about 1 ms per
  window, and the misses' I/O costs more. Gate G2 decides whether that is acceptable.

### 9.4 Tiered mode: adaptive swaps

Keep upstream's policy from `generate.cpp:4459-4512`. Keep its constants:

| Constant | Value |
| --- | --- |
| Interval | every `--adapt-every` (4) rounds |
| Candidate threshold | usage ≥ 2.0 |
| Pair margin | candidate usage ≥ victim usage + 1.5 |
| Cap | `--adapt-swaps` (96) per pass, sorted by gain |
| Decay | 0.7 |

Change the mechanism: a swap is a `pread` from the GGUF into the victim's slot of the resident arena, done by the adapt
thread. There is no GPU copy and no exchange buffer.

Ordering:
1. Set `res[out] = not resident` before the read.
2. Read the incoming blob.
3. Set `res[in] = slot` when the read has landed **and** every window that might have read the victim slot has
   completed.

The resident table is a shared buffer that the plan kernel reads, so these updates must happen between windows. Apply
them in the main thread just before encoding the next window. That is the upstream `apply_pending` hook.

### 9.5 Expert profile

The STRP profile is unchanged:

| Part | Layout |
| --- | --- |
| Header | `"STRP"` plus five `u32`: version (1, but the reader ignores it), n_layers, n_expert, slots, n_ranked |
| Ranking | n_ranked × (u16 layer, u16 expert) |
| Slot table | ignored by the reader |

Use: tiered mode fills the resident set in ranking order. Full mode uses the ranking only to order the warm-up, so hot
experts are paged in first.

`--dump-routing` traces are byte-compatible, so `tools/make_profile.py` can build Mac-measured profiles.

---

## 10. Speculative decoding: the host side

### 10.1 `Window` API (the Mac `Verifier`)

These keep the upstream names and semantics from `verify.hpp`:

| Method | Upstream line | Mac |
| --- | --- | --- |
| `run(T, tokens, pos0, out, err)` | 76 | One command buffer. Row t samples with Philox(seed, pos0 + t), and penalties use one history row per window row (84-93). The upstream `pool` and `user` arguments are dropped. |
| `commit(n_keep)` | 121 | Writes `n_keep` and `pos0` into the shared commit buffer and encodes the commit kernels, in their own command buffer or at the head of the next one. Commit replays GDN from the stored per-row inputs for the kept rows. It restores the indexer tail from its snapshot and re-appends the kept keys, and rebuilds the PLE history from the kept rows' normalised vectors. K/V cells past the accepted prefix stay in place and are overwritten later. |
| `set_commit_async(bool)` | 126 | Kept |
| `wait_commit()` | 129 | Waits for the commit command buffer, through its completion or an event value |
| `diag()` | 62 | Last window: T, pos0, the layer the `MissResolver` reached, event values |
| `release_gpu_waits(timeout_ms)` | 68 | Sets the event to `UINT64_MAX`, satisfying its queued waits but not cancelling later commands; callers must ensure downstream work is absent or guarded before release. It then waits up to the timeout for committed or enqueued command buffers containing its waits to complete and retires the event permanently (section 14) |

Dropped: `set_stage`, `set_next`, `handoff_floats` (layer split); `set_split` (`--spec-split`); `set_pcie_mode`.

### 10.2 One round (serve `generate.cpp:5322-5430`, one-shot `:6128-6232`)

This is unchanged except for the steps marked **Mac**.

1. T = S_mtp = min(`--mtp-max-t`, S) when `mtp_max_t > 0`, else S. If `spec_min_p > 0`, T grows only while
   `dprob[T-1] >= min_p`. The first window is T = 1.
2. Suffix drafter: `k = sfx.propose(S-1)`. If `sbuf[0] == drafts[0]`, then `pk = policy.choose(T, k, match)`. If
   `pk.lookup`, T = pk.t and the lookup tokens fill the window.
3. Window = [x, drafts...]. Stage the penalty rows.
4. **Mac:** read the PLE rows for the T tokens (pread on the PLE workers) and write them into the window's staging
   buffer.
5. **Mac:** write the inputs into the shared staging buffers. Encode and commit the window command buffer. Wait
   (`STRATA_METAL_WAIT`).
6. **Acceptance (exact match):** `a = 0; while (a < T-1 && window[a+1] == outv[a]) ++a;`.
7. Every `--adapt-every` rounds in tiered mode, run the adapt pass (section 9.4).
8. `commit(a + 1)`.
9. Emit `T <id>` for `outv[0..a]` while produced < max_new and no EOS appears. The finish reason is `stop`, `cancel`
   or `length`.
10. `mtp.draft(T, outv, p, a, drafts, err, dprob, min_p)`. **Mac:** the commit and the drafter's first draft are
    encoded into one command buffer on the same queue (queue order replaces stream order). The host waits for it
    before building the next window, because the next window needs the drafts. Each further draft is one more
    command buffer and wait, as upstream's one graph launch and sync per draft step. In coupled mode each draft also
    runs the CPU sampler tail after its wait ([04](04-metal-kernels.md), sections 5.11 and 5.12).
11. `policy.observe(from_sfx, T, a, match, round_ms)`, then `sfx.append(accepted)`.
12. x = outv[a]; p += a + 1.

That gives two host round trips per round, the window and then commit plus the first draft, and one more per further
draft. Each costs about 11 µs
([appendix A.5](appendix-a-measurements.md)), which is noise in a 25 ms window. Two optimisations may follow once the
basic loop passes the gates:
- moving acceptance to the GPU, so that window, commit and draft share one command buffer;
- encoding the next window speculatively.

### 10.3 MTP API (`mtp.hpp`)

These keep the upstream names:

| Call | Mac notes |
| --- | --- |
| `MtpRuntimeStorage::open(context, geometry, sources, plan, ...)` / `open_with_main_kv(...)` | Both validate the plan's MTP file, named bind-layout, parameter-slot and K/V byte counts against inspected geometry, then map and warm about 0.9 GB through bounded read-only no-copy windows, allocate the exact bind-scratch extent, a zeroed 256-byte-slot parameter array and drafter K/V, copy the inspected draft-vocabulary IDs into their device view, and publish only after all nested opens succeed. `open_with_main_kv` additionally validates the live main QSA layout and retains its exact borrowed RoPE subview before opening MTP resources; the caller keeps the Context and main KV allocation live. `open()` remains unbound. `bind_region()`, `mtp_params_slot()` and `shared_rope()` return bounded views only while the MTP owner remains valid; after rollback they become empty. `ModelStorage` owns the main `KvStore` longer than its MTP runtime. This composition does not initialize RoPE values or submit MTP work. |
| `bind_bytes()` | Bytes the plan must reserve |
| `bind(weights, head, window_R)` | |
| `prefill(R_rows, next_tokens, n, cell0)` | K/V-only passes in groups of up to 8 rows, skipping groups below `first_needed()`, in one command buffer per call, waited before it returns ([04](04-metal-kernels.md), section 5.11). Upstream's batched path for one GPU is a work item. |
| `draft(T, tokens, p, a, drafts, err, probs, min_p, n_drafts)` | Writes the drafter's K/V for the window's T cells from p, runs the full layer from accepted row a, then one step per further draft while j < min(S − 1, max_drafts) and the last probability is at least `spec_min_p`. `probs` returns the draft probabilities. |
| `draft_first()` | First round after a prompt: the residual of the first window (T = 1, the prompt's last token) copied into every row ([04](04-metal-kernels.md), section 5.11) |
| `kv_restore(upto)` | A no-op on the Mac: upstream refills the drafter's ring from its host copy under KV streaming, which is not ported. Conversation parking copies the drafter's cells (section 11.3). |
| `set_draft_sampling()`, `set_draft_history()` | Coupled mode |
| `first_needed()` | `prompt_len − window − 64`. MTP needs only the last `--mtp-window` cells. |

The MTP model is:
- e = fc_embedding(rms(embed(tok))·(1+w));
- h = fc_hidden per stream (rms_10240(R)·(1+w));
- R = h + e;
- then the attention hyper-connection, QSA attention over its own K/V, the MLP hyper-connection, MoE (512 experts,
  top-10, plus the shared expert), the inject, the final mixer, and the head over the draft vocabulary subset.

Three simplifications affect draft quality only, never the output:
- Q8_0 projections;
- dense attention over the last `--mtp-window` cells (32,768 by default), which is identical to the sparse selection
  below 2,051 cells;
- all 512 MTP experts are always resident (708 MB).

Its experts are S2 blobs, Strata's own Q2_0 form ([03](03-model-reference.md), section 3.6), and its dense weights are
Q8_0 and BF16. The kernels are [04-metal-kernels.md](04-metal-kernels.md), section 5.11.

### 10.4 Draft policy and drafters

`src/spec/*` is compiled unchanged, with the constants from `draft_policy.cpp`:

| Constant | Value |
| --- | --- |
| `kMaxT` | 8 |
| `kBuckets` | 4: match length < 6, < 12, < 24, else |
| margin | 0.03 |
| `kShape` | {0, 1, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6} |
| `kCostAlpha` | 0.1 |
| `kTokAlpha` | 0.05 |
| `kDecay` | 0.97 |
| `kPriorQ` | {0.75, 0.88, 0.93, 0.96}, with `kPriorN` 4 |
| `kProbes` | 3 |
| Prior for unseen MTP tokens | 1 + 0.7(t−1) |

How the policy chooses:
- Lookup value: E(k) = 1 + q + ... + q^k, divided by cost(k+1).
- MTP value: mtp_tokens(t) / cost(t).
- Lookup wins when its best E/cost beats MTP by more than the margin.
- Probe: try the full size when q ≥ 0.85 and that size has fewer than 3 samples.

The policy learns its costs from observed round times, so it adapts to Mac timings with no change.

`SuffixDrafter(max(1, suffix_draft), 64, max_context + 4096)`:
- trigram hash, 4 ways;
- min_match 3, max_match 64;
- about 20 B per history token.

`controller.cpp` is test-only and not wired in. Its `distinct_ratio` table {1, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35,
4.80, 5.2} models how the expert union grows with T. [05-memory-and-residency.md](05-memory-and-residency.md) uses it
for window traffic.

### 10.5 Exactness contract (must hold)

1. **Greedy.** Every emitted token is the window's argmax for its row. Row t of a window must be bit-identical to the
   same position decoded at T = 1. Every T-row kernel reduces each row in the same order as T = 1, with no
   batch-dependent reduction ([04-metal-kernels.md](04-metal-kernels.md)). Because plain decode is a T = 1 window, this
   is one code path.
2. **Sampling.** Row t is sampled with Philox(seed, pos0 + t) over the chain: penalties, top_k, top_p, min_p,
   temperature, inverse-CDF pick. A draft is kept only if it **equals** that pick. The output is a pure function of the
   seed, whatever the drafts were. Penalty history row t = the last h tokens of (consumed, x, drafts 0..t−1).
3. **Coupled draft.** The MTP samples its draft j with the target's chain and the same uniform, counter = cell + 1
   (`coupled_draft_counter`). Draft j after a window at p with a accepted runs at cell p + a + j. The device penalty
   ring is `kCoupledHistCap` 4096 + max_t.
4. **Rollback.** Main model: commit replay. MTP: the next round's K/V-only pass rewrites its speculative cells before
   any query reads them ([04](04-metal-kernels.md), section 5.11), and conversation parking saves and restores its
   cells (section 11.3). The suffix drafter only sees accepted tokens.
5. **Fingerprints.** `STRATA_STATE_HASH` / `_GDN` over the same state bytes as upstream. This lets a Mac state be
   compared with a CUDA state for the same prompt at the level of individual layers. Bitwise equality across backends is
   not expected (gate C1 uses KL, not equality).

---

## 11. Conversation and prompt cache

The policy is copied verbatim. Only the snapshot transfer changes.

### 11.1 Checkpoints

A checkpoint is a `ConversationCheckpoint` (`conversation_cache.hpp:23-40`). It holds:
- ids and image keys;
- GDN state for 36 layers;
- the PLE history;
- QSA indexer tails;
- `dead`, `block_pos` and `used`.

Each is about 118 MB. `stage_parts` is always empty on a Mac.

KV is positional, so checkpoints are prefixes of the live KV chain.

Cut points, in order: `{reread_to, root_at, turn_at, n−1}`.
- `root_at` is the first `<|im_start|>`, taken only when it is at least `--prompt-cache-root` (2048) tokens in. The root
  is pinned by the retention policy.
- `turn_at` is the last `<|im_start|>` after the resume point.
- A checkpoint is also taken every `--prompt-cache-every` (16384) freshly read tokens.

`checkpoint_at(L)`:
1. If a checkpoint at L exists, bump its `used` stamp.
2. Otherwise wait for the queue to idle (the `cudaDeviceSynchronize` equivalent) and save.
3. If the count exceeds `--prompt-cache` (6), `conv_cache::eviction_victim` evicts. The root is pinned; the rest go LRU.

### 11.2 Resume selection (copied from `generate.cpp:4858-5009`)

1. `starts_with(candidate, prompt)`: the candidate length L ≤ n − 1, the ids are equal, and the image keys below L are
   equal.
2. Candidates, in order: the live session; checkpoints (the longest wins); parked conversations via
   `conversations.best`.
3. If a parked conversation wins, `park_current` saves the outgoing live state if admitted, then the snapshot is
   restored. The results are `ConversationRestore restored | invalid | transfer_failed`.
4. A control-vector mismatch clears checkpoints and the live session.
5. Checkpoints longer than the resume point, or not on its path, are erased.
6. If resume = 0, run `session_zero`. Otherwise restore the checkpoint and bump its stamp (`STRATA_CKPT_REREAD`
   re-reads instead).
7. `mtp.kv_restore(resume)`, then `RESUME <resume>`.
8. Short text parts (≤ `--short-read`, 64 tokens) after the resume point go through decode windows, S tokens at a
   time. A part goes through windows only if it has at most `--short-read` tokens, holds no image row, and
   `STRATA_CKPT_REREAD` is unset (upstream `windows_ok`, `generate.cpp:5012-5025`: "Picture rows need the batched
   path"; the re-read keeps every read batched). Every other part goes through `Prefill`. These windows keep every row,
   so they are write-through ([04](04-metal-kernels.md), section 8.5).

### 11.3 Snapshot transfer on unified memory

- All state buffers are Shared, so save and restore are `memcpy` between the state buffers and host-RAM checkpoint
  buffers.
- They run only when the queue is idle: after `wait_commit` and after the last command buffer has completed.
- Private buffers, if any are adopted, use a blit encoder.
- Parking copies the KV (main layers, then the draft layer) into `ConversationBuffer` segments (16 MiB, geometric
  growth), exactly as upstream.
- On a Mac the copy costs RAM like anything else. Admission therefore uses `available_memory()` (section 7.3) against
  `--conversation-cache-min-free-mib` (2560).
- A later optimisation can give each conversation slot its own KV arena and restore by switching buffer pointers
  ([audit B 6.5](audit/B-engine-host.md)). It is not needed for parity.

### 11.4 Memory pressure (Mac addition)

Install a `DISPATCH_SOURCE_TYPE_MEMORYPRESSURE` source:

| Level | Action |
| --- | --- |
| `WARN` | Log `strata: memory pressure warning` on stderr |
| `CRITICAL` | Drop all parked conversations, and all checkpoints except the root, at the next request boundary. Log what was freed. |

Never act during a window. This addition is not in upstream. It keeps a Mac usable when other apps grow, because the
engine cannot be swapped cheaply.

---

## 12. Optional CPU helper (not in the first release)

Upstream's CPU pool is deleted in the first release (section 1.2). Revive it as a "CPU share" behind a flag only if a
measurement shows unused memory bandwidth while the GPU computes experts
([audit K5 4.8](audit/K5-orchestration-vram-tests.md)).

If revived, it must follow these rules:

| Concern | Rule |
| --- | --- |
| Handshake | Per layer, through the same `MTLSharedEvent` pattern as section 9.3. Never a GPU spin. |
| Arithmetic | NEON kernels whose operation order matches the Metal expert kernels bit for bit, as `moe_hit_grouped_s2_cpu_order` does on CUDA (`s2_expert_grouped.cu:386-460`). Otherwise output depends on the CPU share. |
| Kernels | NEON `sdot`/`i8mm`. ggml-cpu's `ggml_vec_dot_q2_0_q8_0` (`ggml/src/ggml-cpu/arch/arm/quants.c:222` at the pinned `3cf03257f`) is a reference. `src/kernels/cpu/native_expert.cpp` already compiles on arm64. |
| Topology | `hw.nperflevels`; `hw.perflevel0.physicalcpu` for performance cores; `hw.perflevel1.physicalcpu`; `hw.physicalcpu`. No SMT. |
| Affinity | None exists on macOS. Use `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)`. `THREAD_AFFINITY_POLICY` is ignored on Apple Silicon. |
| Spinning | `__builtin_arm_yield()` instead of `_mm_pause`. For long waits use `os_sync_wait_on_address` / `os_sync_wake_by_address_any` (macOS 14.4+). |
| False sharing | 128-B padding: the Apple M-series cache line is 128 B, and `std::hardware_destructive_interference_size` is 128 on arm64 Apple clang. Upstream uses `alignas(64)` in `pool.hpp:236-264`. |
| Build | `pool.cpp` needs guards for `immintrin.h` (`:9`), `cpu_set_t` / `sched_getaffinity` (`:165-172`) and `pthread_*affinity_np` (`:267-311`). The AVX TUs are excluded on arm64 (section 8.1 of [audit B](audit/B-engine-host.md)). |

---

## 13. Telemetry

### 13.1 `--stats` (one-shot)

The block is printed on stdout with a `%-24s` label column, as upstream `generate.cpp:6236-6470`.

**Kept verbatim**, so they compare directly across platforms:
- `N rounds of S, drafts accepted a of o (rate), x tokens per round`
- `window sizes T1:n T2:n ... (min draft probability p)`
- `suffix drafts W windows, drafts accepted a of o`
- `accepted per round 0:n 1:n ...`
- `mtp %.3f ms/round drafting (N rounds), MTP prompt %.1f ms, %.0f MiB of VRAM`
- `per token %.3f ms/token (WALL CLOCK: ...)`
- prefill `(time to first token)`
- the dump summaries

**Removed:**
- `wait for rings`, `pool`, `rows phases` and `dispatch` lines;
- `pcie experts`;
- `R4 overlap`;
- `token graph: ... flushes`;
- `rings seen MID-GRAPH`;
- `ring latency`;
- per-device `CUDAn experts`.

**Replaced or added:**

| Line (label, then format) | Source |
| --- | --- |
| `gpu window` `%.3f ms/round (encode %.3f, gpu %.3f, host wait %.3f)` | CPU clock around encoding; `GPUEndTime − GPUStartTime`; time from commit to wake |
| `commit+draft` `%.3f ms/round` | the second command buffer |
| `dispatches` `%d per window at T=1, %d at T=4` | the encoder count |
| `gpu bandwidth` `%.1f GB/s over the window` | bytes read (computed from the plan) / GPU time |
| `expert tiers` `resident %lld, slot hits %lld, page cache %lld, ssd %lld blobs, %.1f MB read` | `ExpertStore` counters (tiered mode) |
| `miss handshake` `%.3f ms/round over 48 layers (max hop %.3f ms)` | `MissResolver` (tiered mode) |
| `adaptive tier` `N experts swapped into the resident set (every k rounds, ms/round)` | the adapt pass (tiered mode) |
| `routing prefetch` (upstream text) | lookahead (tiered mode) |
| `layers` `%.3f ms/layer (QSA %.3f, GDN %.3f, experts %.3f)` | `--stage-timing` only: counter sample buffers at encoder boundaries, so each layer gets its own compute encoder in that mode |

### 13.2 Signposts

Emit `os_signpost` intervals under `OSLog` subsystem `com.strata.engine` for:
- category `window`: one interval per window;
- category `layer`: one per layer, only with `--stage-timing`;
- category `io`: one per miss read and per PLE read batch;
- category `request`: one per serve request.

Instruments' Metal System Trace then lines CPU and GPU up on one timeline.

### 13.3 Serve-side telemetry

- The INFO line and the post-DONE stderr lines (section 5).
- `STRATA_DECODE_TIMING`, `STRATA_TRACE`, `STRATA_VERIFY_PROFILE`, `STRATA_VERIFY_NODES` and `STRATA_PREFILL_TIMING`.
- `--dump-routing`, flushed after every DONE.

The Monitor's hardware cards are fed by the server's own `_Apple` telemetry reader (ioreg), not by the engine
([07-server-setup-port.md](07-server-setup-port.md)).

---

## 14. Watchdog and stall reports

### 14.1 Progress model (unchanged, `include/strata/core/progress.hpp:17-57`)

- `beats` advance per token, prompt chunk or window (`progress_beat`).
- `ticks` count layers served. On a Mac a tick advances per layer resolved by `MissResolver` in tiered mode, and per
  completed command buffer in full mode.
- `busy`, `where`, `detail`, `chunk` and `since_ms` describe the current stage.
- `BusyScope` brackets each request.

### 14.2 Watchdog thread (unchanged behaviour)

`STRATA_WATCHDOG_S` defaults to 60; 0 turns the watchdog off. It checks once per second.

If the engine is `busy` and `beats` has not changed for the limit, the watchdog does three things.

1. It prints this line **verbatim**, because the server's `death_note` looks for "issue #29":
   ```
   strata serve: no progress for N s during a request (<stage>) - stopping the engine so the server starts it again (issue #29)
   ```
2. It calls `stall_report`. The header is
   `strata serve: stall report (engine V): stage "..." for N s; L layers served since the last finished step (0 = stopped, more = slow)`.
   It makes two passes, 2 s apart. Each pass prints:
   - `Window::diag()`;
   - the `MissResolver` state (tiered mode), which replaces the pool diag;
   - the memory line `memory: R MiB resident, S MiB in swap, A MiB RAM available; F major page faults so far`, filled
     from `process_memory()` and `available_memory()`. Upstream prints zeros here on macOS.
   - Mac addition: `gpu: last command buffer "<label>" status <status>, W MiB budget, A MiB allocated`.
3. It calls `release_gpu_waits` only for events whose waiting command buffers have no unguarded GPU consumer after the
   wait. Each event advances to `UINT64_MAX`, which satisfies its encoded `encodeWaitForEvent` operations but does not
   cancel later commands, then waits up to the supplied timeout for committed command buffers containing those waits to
   complete. If a window command buffer contains an unguarded expert consumer after the wait, the watchdog must abort
   without releasing that event until the scheduler provides a safe boundary or GPU-visible failure guard. Released
   events are permanently unusable, and `wait()` does not provide completion evidence. In full mode nothing is waiting
   on the host. Then the watchdog calls `std::abort()`.

### 14.3 Crash report instead of a minidump

`abort()` raises SIGABRT, and macOS writes a crash report with every thread's stack to
`~/Library/Logs/DiagnosticReports/strata-<date>.ips`. Before aborting, the stall report prints:

```
strata serve: a crash report with every thread will be in ~/Library/Logs/DiagnosticReports/
```

No in-process stack walking is required.

### 14.4 Other timeouts

| Timeout | Mac |
| --- | --- |
| Upstream verify host loop, 20 s without a ring | `MissResolver` waits at most 20 s for each layer signal. Full-mode window waits use `waitUntilSignaledValue:timeoutMS:` with 20 s, or a timed wait on the completion. On timeout, fail the request. Release GPU waits only after proving that queued commands after each wait are absent or guarded; the current single-command-buffer tiered outline does not satisfy that condition. |
| Pool self-watchdog, `kStall` 60 s | Deleted with the pool |
| GPU-side timeout | A command buffer that runs too long is ended by the system with `MTLCommandBufferErrorTimeout` and handled as a GPU error (section 2.4). The Metal-only test in [09-testing-and-validation.md](09-testing-and-validation.md) checks that a 33 ms window, and a 32K prefill chunk, never come close. L-12 ([04](04-metal-kernels.md), section 11) finds the limit; if a chunk at C = 32,768 comes within half of it, the prompt path encodes each layer as its own command buffer ([04](04-metal-kernels.md), section 6.1). |
| `STRATA_TEST_VERIFY_STALL` | Kept. `STRATA_TEST_VERIFY_STALL=N` withholds the last layer's signal of the N-th window, so `MissResolver` (tiered) or the window wait (full) stalls. With the default `STRATA_WATCHDOG_S`, the 20 s timeout of this table ends the engine first; with a limit below 20 s, the watchdog of section 14.2 does. `metal_watchdog_test` runs both cases. |

---

## 15. Host-side risks and measurements

| ID | Item | How to resolve | Decision it drives |
| --- | --- | --- | --- |
| M-1 | Wake latency of `waitUntilCompleted`, `MTLSharedEventListener` and a spin on `signaledValue`; event hop latency inside a command buffer | Micro-benchmark in `tests/metal/metal_shared_event_handoff` | Default of `STRATA_METAL_WAIT`; whether tiered mode's 48 hops fit gate G2 |
| M-2 | Does a second queue (MTP drafting overlapped with the commit) help? | A/B on the decode benchmark | One queue or two |
| M-3 | The right `--vram-reserve-mib` default on a Mac | Specified, with its method, in [05](05-memory-and-residency.md), section 10; listed here because the engine reads the reserve | The default reserve |
| H-4 | CPU encode time of a T = 8 window (48 layers × about 30 dispatches) | `gpu window` stats line | Whether to implement ICBs (`STRATA_METAL_ICB`) |
| H-5 | A command buffer stalled in `encodeWaitForEvent` might count toward a GPU timeout | Metal-only test: stall `MissResolver` for 5 s, 15 s and 30 s | The `MissResolver` timeout; whether tiered mode needs per-layer command buffers instead |
| H-6 | Copied driver code drifts from upstream | `tools/metal/check_drift.py` in CI (section 1.3) | Effort per compatibility review; no fixed cadence |
| H-7 | Paths with spaces in GENI / ENC | The engine-side parse rule (section 5.3); server-side quoting ([07](07-server-setup-port.md)) | none |
| H-8 | Timer coalescing / App Nap stretching waits | Measure with and without `beginActivityWithOptions` | Whether the activity token is needed |
| H-9 | Legacy Windows/Linux memory probes and fixed 4096-byte pages | Removed unsupported probes; Metal memory reads use Mach/sysctl and page size is queried from the OS | None |

---

## 16. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md).

| # | Item | Done when |
| --- | --- | --- |
| H1 | Mac-only CMake graph: Metal is always enabled, non-Darwin targets fail before compiler detection, and the engine requires arm64 with the macOS 27 Metal toolchain. | The `mac-dev` configure/build passes and the host and Metal CTest presets pass on the reference Mac; no Linux build-graph comparison is part of this repository. |
| H2 | Mac platform: `direct_file.cpp`, `ngram.cpp` page size, and `memory_probe` | Direct-file and memory probe tests pass on macOS |
| H3 | `Context`, `Buffers`, `Events`: device checks, metallib and its hash, pipelines, binary archive (with [08](08-build-and-distribution.md) B6), residency set, labels, error reporting | `metal_list_devices_test`, `metal_context_test`, `metal_buffers_test`, `metal_events_test` and `metal_fault_test` pass; the event test proves higher signals and dependent GPU waits cannot be submitted before their producers, a GPU signal after an unsignaled host gate remains pending, and Context teardown drains committed work; the fault test submits a deliberately invalid GPU read, verifies the section 2.4 diagnostic and proves a following command buffer completes; a metallib from another build exits 1 with the message of section 2.3 |
| H4 | `options.cpp` copy with all section 3 dispositions, plus every flag of section 3.10 | `options_dispositions_test` parses every upstream flag, checks its disposition, and preserves values beginning with `--`; it tests the exit-2 combinations of `--geometry`, `--plan-w-mib` and `--plan-ram-mib`. `metal_cli_modes_test` checks empty and conflicting modes; `metal_self_test` exercises the explicit CLI smoke-test path on the GPU. |
| H5 | `WeightStore` with format rules 1-9, and the light `--plan-only` / `--write-geometry` path (section 2.3) over `Geometry` and `Plan` ([05](05-memory-and-residency.md) work item U1) | Host fixtures cover the complete 1,224-tensor inventory, role/type constraints, split metadata, the Swift-style 342/882 shard layout, native expert v3/v4 rows, index/conversion/MTP validation and CLI export. Acceptance still requires every catalogue model's live `--plan-only` time and PLAN parity against its geometry file ([07](07-server-setup-port.md), section 4.4.3); no real model pack is present in this checkout. |
| H6 | Serve loop and protocol (section 5), first with a fake `Window` | `serve_transcript_test` passes byte-for-byte protocol fixtures, bounded SVE1 loading and M-RoPE checks, callback/EOS invariants, bounded input, STOP/EOF ordering and signal cases. Ready input, EOF finalization and the request-finish snapshot share one mutex; request completion drains at most 64 MiB before failing closed; the host asks the backend to stop after EOS and rejects later token callbacks. The host-only `ServeBackend` seam is ready for L9/H8's real Window adapter. The production `strata --serve` path and upstream `serve/test_*` parity remain pending until model initialization and a real request backend are connected. |
| H7 | One-shot loop and `--stats` | Output format matches upstream line by line for the kept lines |
| H8 | `Window` + spec round + MTP host side (section 10) | Gate C2 (spec equals greedy) passes |
| H9 | Prompt cache, resume, snapshots, parking, memory pressure (section 11) | Gate C3 (conversation cache) passes |
| H10 | `ExpertStore` tiered mode: `MissResolver`, slot pool, adapt, lookahead (section 9) | Gate C4 (full equals tiered) passes; gate G2 throughput |
| H11 | Watchdog, stall report, crash-report hint, window timeout (section 14) | `metal_watchdog_test` ([09](09-testing-and-validation.md), section 5.5): with `STRATA_TEST_VERIFY_STALL=3` alone, the engine prints `ERR gpu error: window timeout` and exits 1 within 20 s + 5 s; with `STRATA_WATCHDOG_S=5` as well, it prints the issue #29 line, the stall report and the crash-report hint, and ends by SIGABRT within 5 s + 5 s, leaving a crash report |
| H12 | `check_drift.py` | `tools/metal/test_check_drift.py` ([09](09-testing-and-validation.md), section 9.3): `check_drift.py` fails on each kind of change of [10](10-implementation-plan.md), section 4.2, planted one at a time |
| H13 | `strata-vision` thread-count fix | Uses all performance cores on a Mac |
