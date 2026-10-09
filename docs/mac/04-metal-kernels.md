# 04. Metal kernels

Status: implementation-ready specification, updated 2026-10-09.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- [audit/A-model-and-kernels.md](audit/A-model-and-kernels.md), sections C-F: the kernel inventory, the decode and
  prefill dataflow, the parity ladder and the Metal rules;
- [audit/K1-gr-router-shared-rope-cvec.md](audit/K1-gr-router-shared-rope-cvec.md),
  [K2](audit/K2-qsa-kv-gdn.md), [K3](audit/K3-quant-and-matvec.md), [K4](audit/K4-ple-mtp-head-sampler.md) and
  [K5](audit/K5-orchestration-vram-tests.md): every upstream `__global__` with its launch shape and arithmetic;
- [appendix-a-measurements.md](appendix-a-measurements.md), sections A.5 (dispatch cost), A.6 (matrix units),
  A.7 (projections), A.10 (numerics) and A.11 (binding and ordering), with the programs in
  [spikes/metal-feasibility/](spikes/metal-feasibility/), [spikes/metal-numerics/](spikes/metal-numerics/) and
  [spikes/metal-binding/](spikes/metal-binding/);
- upstream `src/kernels/cuda/*.cu`, `src/prefill/kernels.cu`, `src/core/layer.cpp`, `src/core/verify.cpp`,
  `src/core/mtp.cpp` and `src/kernels/cuda/native_mmvq.cu:135-205`, read for this document.

This document specifies every Metal kernel the Mac engine runs: its name, its launch shape, the order of its
arithmetic, what it reads and writes, the numerics contract that all kernels share, how the host encodes them into
command buffers, and the tests that hold each kernel to its oracle. It builds on
[03-model-reference.md](03-model-reference.md), which fixes the model, the formats, the activation contract and the
rules of the forward pass, and on [05-memory-and-residency.md](05-memory-and-residency.md), which places the buffers.
The host code that calls the kernels is [06-engine-host-port.md](06-engine-host-port.md). The build of
`strata.metallib` is [08-build-and-distribution.md](08-build-and-distribution.md), section 5.

Where this document and an audit file disagree, this document wins. Section 1.3 lists the places where the audits are
superseded.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Scope and conventions

### 1.1 What this document covers

- The Phase-1 kernel set: decode, verify and commit (one code path, section 5.1), the MTP drafter, the head and the
  sampler (section 5), and the prompt path (section 6). Phase-2 kernels are named but not specified (section 2.6).
- The numerics contract: the two math modes, contraction, reductions, conversions and transcendental functions
  (section 3).
- The data the kernels see: parameter blocks, the step buffer, scratch, the KV cache, the indexer, the GDN state, the
  PLE history and the RoPE table (section 4).
- The threadgroup-memory budget (section 7).
- The structure of the command buffers of a window, a commit, an MTP round and a prefill chunk (section 8).
- What each upstream kernel flag and kernel-scope variable becomes (section 9).
- The kernel-level tests, measurements and work items (sections 10-12). Section 13 says where each detail lives.

It does not cover:
- the vision encoder, which is llama.cpp's ggml-metal inside `strata-vision` ([06](06-engine-host-port.md),
  section 5.10);
- CPU code: the PLE hash and IQ4_NL row decode ([03](03-model-reference.md), section 6.8; the read is
  [06](06-engine-host-port.md), section 7.4), the RoPE table builder and
  the sampler tail (section 5.12), which this document only specifies as far as the kernels depend on them;
- the canonical (S-form) packs and the PC expert tiers, which the Mac does not support
  ([06](06-engine-host-port.md), section 6.1).

### 1.2 Conventions

- **T** is the window width, 1 ≤ T ≤ 8. "Row t" is token t of a window. "Cell" is a KV position. In section 6, T is
  a prefill chunk's row count instead, 1 ≤ T ≤ C.
- **lane** is `thread_index_in_simdgroup`, **sg** is `simdgroup_index_in_threadgroup`, **tid** is
  `thread_position_in_threadgroup` (x), **tg** is `threadgroup_position_in_grid`. A "warp" is a simdgroup of 32 lanes.
  Kernels take lane and sg from these attributes and never derive them from tid ([K5 section 5.4](audit/K5-orchestration-vram-tests.md)).
- A launch shape `(gx, gy[, gz]) × tx` means `dispatchThreadgroups:MTLSizeMake(gx, gy, gz)
  threadsPerThreadgroup:MTLSizeMake(tx, 1, 1)`. `(tx, ty)` threads means a 2-D threadgroup of tx·ty threads.
- **Bitwise** means equal as bit patterns. **ulp** distances are measured on float32. A **twin** is a test that runs
  a Metal kernel and its oracle on the same inputs (section 10).
- The **oracle** of a kernel is a C++ function in `tests/metal/oracles/`, written from the upstream formula, compiled
  with `-ffp-contract=off` and using `std::fma` exactly where the kernel uses `fma` (section 3.3).
- Kernel names are the MSL `[[host_name]]` names. They are the function names of the manifest
  ([08](08-build-and-distribution.md), section 5.2).
- **O-class** marks a kernel that only an optional feature dispatches (audit A's class O): `q4` and `k8v4` KV,
  control vectors and coupled draft sampling. These kernels are in the Phase-1 set and specified like any other,
  because the options that reach them are kept ([06](06-engine-host-port.md), section 3). The default configuration
  never runs them, so their twins and the feature runs of section 10 are what test them.
- Upstream `file:line` references are to commit `1678de33`.

### 1.3 Where the audits are superseded

| Audit statement | Replaced by |
| --- | --- |
| F.1 and K2 section 8.4: two libraries, fast math for the native kernels, "set `mathMode` per library" | One library, two folders. Every Phase-1 kernel is in `safe/`. `fast/` holds only `strata_selftest_exp` until a kernel passes the rule of section 3.2. |
| F.2 and K1 section 13: `fast::exp` for the native and fused kernels | `exp` in safe mode (≤ 1 ulp, A.10). A safe file that calls a `fast::` function fails the build ([08](08-build-and-distribution.md), section 5.4). |
| F.2 and B.2.4: division needs `precise::divide`; "confirm safe-mode `/` is correctly rounded" | In safe mode `/`, `sqrt` and `rsqrt` are correctly rounded (A.10). Safe files use the operators. `precise::divide` is needed only in a fast file. |
| B.2.4 and F.3: "check whether `#pragma METAL fp contract(off)` is available" | Verified: the pragma and `-ffp-contract=off` both stop contraction, and `fma()` always fuses (A.10). Section 3.3 gives the contraction rule. |
| F.3 and B.2.4: reproduce `quantize_q8_0`'s double-precision rounding with an fma residual | Not ported. `quantize_q8_0` serves only the S-form dense GEMV, which the Mac does not run. The MTP experts use `quantize_q8_0_scaled` (section 5.11), which needs no double. |
| F.4 and K5 section 5.4: `simd_sum` order "unspecified, unverified" | Measured on M5: an ascending pairwise tree (xor offsets 1, 2, 4, 8, 16), not CUDA's descending butterfly. Every reduction that a twin compares bitwise uses an explicit butterfly (section 3.4). L-3 rechecks M1-M4. |
| F.4: "the native router dispatches exactly 32 threads and serves 512 experts only" | One router kernel for 512 and 256 experts (section 5.10). |
| F.5 and F.6, K3 section 3.4: copy IQ codebooks into threadgroup memory | Program-scope `constant` arrays. They are the fastest placement at every table size (A.10). L-4 rechecks inside the real kernels. |
| A.0 and F.5: "32 KiB of threadgroup memory, unverified on M3-M5" | Verified on M5: `maxThreadgroupMemoryLength` is 32,768 B and 32,784 B fails pipeline creation (A.10). Section 7 lists the kernels re-tiled for it. |
| F.6: one `MTLBuffer` per layer for the weights | Weights are bound through the 16 KiB-aligned windows of [05](05-memory-and-residency.md), section 4.2, by offset, and experts through the `addr` table (section 4.3). |
| F.7 and D.1.2: re-encode every window, or record an ICB | Re-encode every window (section 8). An ICB is optional, decided by H-4 in [06](06-engine-host-port.md). |
| F.8: half conversion and subnormals "unverified" | Verified: round to nearest even, subnormal halves kept. Two exceptions: every NaN becomes `0x7e00`, and a float32 denormal input gives +0 (A.10). Oracles copy both (section 3.5). |
| F.8: Q5_0 expert downs need a kernel | No catalogue file has one ([03](03-model-reference.md), section 3.3). A Q5_0 down is refused at load ([05](05-memory-and-residency.md), `unsupported_type_q5_0`). |
| (new) MSL `log1p` | MSL has none. Safe files use `log1p_k` (section 3.6). |
| B.2.4 and K1 section 13: "use `fma` where nvcc would contract" | The six-step contraction rule of section 3.3. |
| C.4: "115 kernels to port"; [00](00-feasibility-study.md): "about 45 kernels" | The kernel set is the manifest (section 2.4): about 150 host names from about 85 kernel templates, counted with their format and width specialisations. |
| C.4: `to_bf16` on the decode path | Not on the production decode path: upstream `--native` reads the PLE value with the BF16 mat-vec on F32 activations (`ple.cu:322-324`), as the Mac does (section 5.3). `to_bf16` survives as `strata_util_to_bf16` for prefill. |
| C.4: the single-token `gr_down` / `gr_up`, `gr_down_staged`, `gr_down_v3`, `sampler_one_block` | Not ported. The multi-token kernels at T = 1 are the single-token path (section 5.1). |
| C.4: `prompt_attn` and `block_scores_tc` in Phase 1 | Phase 2 (section 2.6). Phase-1 prefill uses the decode attention and scoring kernels in query batches (section 6.5). |
| D.3.4: prefill GEMMs as float `simdgroup_matrix` with float activations | Dense: dequantise to F16 scratch, then an MPP `matmul2d` on F16 (A.6). Experts: dequantise inside the tile. BF16: bfloat `matmul2d` if L-6 allows, else float tiles. Section 6. |
| A.12 and K4 section 1.7: the PLE history is mutated during the window and restored at commit | A window never writes the history, unless it is write-through (section 8.5); commit rebuilds it from the window's normalised rows with the prefill kernel (section 5.3). |
| D.1.2: "each dispatch boundary is a full memory barrier" | Kernels of one segment run in one serial compute encoder; segments are ordered by one fence (section 8.1). |
| K4 section 3.3: the MTP final mixer is the canonical `gr_read` | Under `--native`, which the drafter requires, `gr_read` takes the native path, the head mixer's arithmetic. The Mac runs the head mixer kernels (sections 5.4 and 5.11), bitwise. |
| A.10, B.1 and K4 section 2.3: the Q5_K head takes F32 activations | Q8_1, like every head ([03](03-model-reference.md), section 3.3). |

---

## 2. The kernel set

### 2.1 Target

| Item | Requirement | Checked |
| --- | --- | --- |
| OS and language | macOS 27 or later, MSL 4.0 (`-std=metal4.0`) | Build ([08](08-build-and-distribution.md), section 5.3) |
| GPU | `supportsFamily(MTLGPUFamilyApple7)` and `supportsFamily(MTLGPUFamilyMetal4)`: M1 and later | `Context` at startup; exit 1 with `strata: this Mac's GPU does not support Metal 4` ([06](06-engine-host-port.md), section 2.3) |
| SIMD width | `threadExecutionWidth == 32` for every pipeline | `Context`, per pipeline |
| Threadgroup memory | `maxThreadgroupMemoryLength >= 32768`. Every kernel is written for 32,768 B. | `Context` |
| Threads | `maxTotalThreadsPerThreadgroup` of each pipeline ≥ its manifest value | `Context` ([08](08-build-and-distribution.md), section 5.2) |
| FP64 | Not used. MSL has no `double` on these GPUs. | — |
| Tuning | Shapes are tuned on M5. M1-M4 must give the same results; their tile defaults come from L-2, L-5 and L-7. | The `metal` label runs on each family in the family runs of [09](09-testing-and-validation.md) |

### 2.2 Naming and specialisation

- A kernel's name is `strata_<stage>_<op>[_<fmt>][_n<NC>][_x]`. `_x` marks a mat-vec that reads its weight rows
  through an index list (the MTP draft head, section 5.8).
  - `<stage>` is one of `embed`, `ple`, `gr`, `gdn`, `qsa`, `idx`, `sel`, `attn`, `q`, `mv`, `moe`, `shx`, `mtp`,
    `util`, `samp`, `pf` (prefill) and `selftest`.
  - `<fmt>` is the lower-case ggml type name (`q2_0`, `iq2_xxs`, `q4_k`, `bf16`) or a KV mode (`f16`, `i8`, `q4`,
    `k8v4`).
  - `_n<NC>` is the number of columns (tokens) a mat-vec processes, NC ∈ {1, 2, 4, 8}. A window of T rows uses the
    smallest NC ≥ T. The padding columns read row 0's activation and write to a scratch sink (section 4.4), so no
    kernel has a per-T branch.
- Specialisations are C++ templates instantiated with `template [[host_name("...")]]`. The library then exports
  only the host names, with no mangled `_Z` symbol, so the manifest check passes (verified with `metal-nm` on
  2026-10-03).
- Helpers in `common/` are `static inline`. Codebooks and constant tables are program-scope `constant` arrays in
  `common/codebooks.h`.
- Each kernel **should** carry `[[max_total_threads_per_threadgroup(N)]]` with N equal to its manifest value, so the
  compiler allocates registers for that size.
- No function constants. Every variant a host can select is a separate host name, created at startup with the other
  pipelines ([06](06-engine-host-port.md), section 2.3). L-8 measures the pipeline count and creation time.

### 2.3 Files

| File | Contents |
| --- | --- |
| `common/types.h` | Block structs of every format (2-B aligned, natural member types), `Q81` and logical `Q80S` activation blocks, `WindowParams`, `PrefillParams`, `MtpParams`, the step layout |
| `common/math.h` | `sigmoid_k`, `silu_k`, `softplus_k`, `log1p_k`, `bf16_to_f32`, `f32_to_bf16_rne`, `half_sat`, `round_half_away` (section 3.6) |
| `common/reduce.h` | Explicit butterflies (`bfly_sum`, `bfly_max`), fixed-order threadgroup partial sums (section 3.4) |
| `common/dot.h` | Integer dot helpers (`dot4_i8`), per-format `load`/`acc` decoders for the mat-vecs (section 5.8) |
| `common/codebooks.h` | `iq2xxs_grid`, `iq2xs_grid`, `iq2s_grid`, `iq3xxs_grid`, `iq3s_grid`, `iq1s_grid_gpu`, `ksigns_iq2xs`, `kmask_iq2xs`, `kvalues_iq4nl`, copied from ggml-common.h at llama.cpp `3cf03257f` |
| `common/params.h` | Architecture constants of [03](03-model-reference.md), section 3.1 (`N_EMBD`, `HC`, `N_LAYER`, ...) as `constant constexpr` values; model metadata such as `rope.freq_base` remains runtime data |
| `safe/embed.metal` | Q3_K, IQ3_S, IQ4_XS and BF16 token gathers, bit-exact image-row copies and GR stream broadcast; the validated embedding encoders and Prefill embedding coordinator support caller-owned Encoder segments with concurrent-mode barriers. The FullMode T=1 step maps native `token_embd.weight` types Q3_K/IQ3_S/IQ4_XS and composes gather→broadcast→layer-0 MoE→one pending consumer; it checks scratch liveness and allows only exact reuse as `shared.output`. Caller-owned Encoder and standalone transactional CommandBuffer forms share this sequence, with the standalone form discarding all encoded work on validation failure. Its BF16 gather remains available to standalone callers, but this coordinator does not resolve the optional BF16 embedding artifact. A separate layer-0 coordinator composes attention GR read→GDN→FFN GR read. Full Window decode and Prefill production dispatch remain unimplemented |
| `safe/prefill_gr.metal` | Prefill-specific GR read normalization with the documented two-stage butterfly, fused GR write/next-read normalization, low-rank SiLU and four-stream mix; the isolated encoders support mode-specific BF16 remainder outputs, while `Prefill::run` remains unimplemented |
| `safe/prefill_gdn.metal` | Isolated Prefill post-projection GDN gate/beta transforms, 64-row tiled convolution and ordered convolution-history update with validated standalone and caller-owned Encoder APIs; Prefill GEMM and chunk-loop integration remain pending |
| `safe/ple.metal` | `strata_ple_gate`, `strata_ple_broadcast`, `strata_ple_conv_residual` and `strata_ple_history` are implemented; validated gate, value-broadcast, conv-residual and history encoders can join a caller-owned `Encoder` segment (the Window path must use its documented serial mode); production PLE dispatch remains unimplemented |
| `safe/gr.metal` | Weighted RMS norm, fused `strata_gr_norm`, tiled `strata_gr_down`, fused `strata_gr_up`, O-class `strata_gr_cvec`, native `strata_gr_post`, `strata_gr_down_silu` and `strata_gr_pre_gated` are implemented; validated weighted-norm, norm/down/up/cvec/post/down-SiLU/pre-gated encoders exist, and weighted RMS norm, GR norm, GR down, GR up, GR cvec, GR post, GR down-SiLU and GR pre-gated can join a caller-owned `Encoder` segment; the T=1 Window layer-0 coordinator calls attention and FFN GR reads with their distinct `hc_attn_*` and `hc_ffn_*` weights; later Window layers and Prefill production dispatch remain unimplemented |
| `safe/gdn.metal` | GDN conv, gates, step and commit; all four decode kernels are implemented and covered by validated encoders and `metal_gdn_test`; all four also support caller-owned Encoder segments; the T=1 layer-0 coordinator orders the speculative GDN block between attention and FFN GR reads, while router/MoE, full `Window::run`, commit scheduling and Prefill dispatch remain open |
| `safe/qsa.metal` | QSA norm/RoPE, O-class FWHT256, raw-bit indexer snapshot, window/commit indexer append, fp16/int8/q4_0/k8v4 KV append encoders, block selection scoring, register/memory top-k and attention gate are implemented; all four KV append encoders, norm/RoPE, FWHT256, indexer snapshot, indexer append, selection scoring, register/memory top-k and attention gate also support caller-owned Encoder segments; a host coordinator composes selection through sparse attention |
| `safe/attention.metal` | Standalone sparse QSA attention chunk and ordered-merge kernels for fp16, int8, q4_0 and k8v4; the host coordinator can feed GPU-produced selection IDs through chunk, merge, O-class FWHT and gate in one ordered segment; production Window/Prefill integration remains open under L6 |
| `safe/quantize_q8_1.metal` | Q8_1 activation quantiser |
| `safe/mmvq_q8_0.metal` | Native Q8_0-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q4_0.metal` | Native Q4_0-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q5_0.metal` | Native Q5_0-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q2_0.metal` | Native Q2_0-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_iq4_nl.metal` | Native IQ4_NL-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_iq4_xs.metal` | Native IQ4_XS-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_iq3_s.metal` | Native IQ3_S-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_iq2_xxs.metal` | Standalone IQ2_XXS row-dot twin for the grouped expert path |
| `safe/mmvq_iq2_xs.metal` | Standalone IQ2_XS row-dot twin for the grouped expert path |
| `safe/mmvq_iq2_s.metal` | Standalone IQ2_S row-dot twin for the grouped expert path |
| `safe/mmvq_iq3_xxs.metal` | Standalone IQ3_XXS row-dot twin for the grouped expert path |
| `safe/mmvq_iq1_m.metal` | Standalone IQ1_M row-dot twin for the grouped expert path |
| `safe/mmvq_q4_k.metal` | Native Q4_K-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q5_k.metal` | Native Q5_K-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q3_k.metal` | Native Q3_K-weight/Q8_1-activation mat-vecs |
| `safe/mmvq_q6_k.metal` | Native Q6_K-weight/Q8_1-activation mat-vecs |
| `safe/mmvf.metal` | BF16 mat-vecs with F32 activations |
| `safe/moe.metal` | Router (`r16`/`r8`), its test-only weight-mode probe, MoE plan, routed-expert and shared-expert SwiGLU, grouped Q2_0/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS gate/up kernels, grouped Q2_0/IQ4_NL down and the shared-gated combine implemented; validated route and plan encoders plus a grouped gate/up encoder now bind all eight expert formats, while production Window scheduling, shared-expert dispatch and model-backed parity remain open under L7 |
| `safe/mtp.metal` | MTP-only kernels (S2 experts, `q8_0_scaled`, draft helpers) |
| `safe/sampler.metal` | Greedy argmax, candidate selection, coupled penalties |
| `safe/util.metal` | Copies and conversions |
| `safe/prefill_gr.metal`, `prefill_gdn.metal`, `prefill_qsa.metal`, `prefill_moe.metal`, `prefill_gemm.metal` | Prefill kernels (section 6) |
| `safe/selftest.metal`, `fast/selftest.metal` | Safe arithmetic, layout, conversion, reduction, dot and helper kernels plus `strata_selftest_add`; `strata_selftest_exp` (fast) remains the build's smoke test ([08](08-build-and-distribution.md), section 5.2) |

`THIRD_PARTY` lists ggml-common.h's codebooks and every kernel structure adapted from ggml-metal (MIT,
[08](08-build-and-distribution.md), section 5.1).

### 2.4 Phase-1 inventory

Each row is one kernel template; its host names are the template with each listed specialisation. The section
column points to the specification. "Threads" is the manifest value.

**Embedding and streams**

| Kernel | Specialisations | Shape | Threads | Section |
| --- | --- | --- | --- | --- |
| `strata_embed_rows_<fmt>` | `q3_k`, `iq3_s`, `iq4_xs`, `bf16` | (10, n_tok) × 256 | 256 | 5.2 |
| `strata_gr_broadcast` | | (40, n_tok) × 256 | 256 | 5.2 |
| `strata_embed_image` (prefill only) | | (10, T) × 256 | 256 | 6.7 |

**PLE (layer 1 only)**

| Kernel | Shape | Threads | Section |
| --- | --- | --- | --- |
| `strata_ple_gate` | (T·4) × 512 | 512 | 5.3 |
| `strata_ple_broadcast` | (40, T) × 256 | 256 | 5.3 |
| `strata_ple_conv_residual` | (40, T) × 256 | 256 | 5.3 |
| `strata_ple_history` | 40 × 256 | 256 | 5.3 |

**GR hyper-connections**

| Kernel | Shape | Threads | Section |
| --- | --- | --- | --- |
| `strata_gr_norm` | (T, 4) × 256 | 256 | 5.4 |
| `strata_gr_down` | 41 × 256 | 256 | 5.4 |
| `strata_gr_up` | 160 × 256 | 256 | 5.4 |
| `strata_gr_post` | (40, T) × 256 | 256 | 5.4 |
| `strata_rms_norm_w` | rows × 1024 | 1024 | 5.4 |
| `strata_gr_down_silu` | ⌈T·320/256⌉ × 256 | 256 | 5.4 |
| `strata_gr_pre_gated` | (10, T) × 256 | 256 | 5.4 |
| `strata_gr_cvec` (O-class) | (4, T) × 256 | 256 | 5.4 |

**GDN**

| Kernel | Shape | Threads | Section |
| --- | --- | --- | --- |
| `strata_gdn_conv_l2` | (80, T) × 128 | 128 | 5.5 |
| `strata_gdn_ab` | 12 × 256 | 256 | 5.5 |
| `strata_gdn_step_norm` | 48 × 512 | 512 | 5.5 |
| `strata_gdn_conv_commit` | 80 × 128 | 128 | 5.5 |

**QSA, indexer, selection, attention**

| Kernel | Specialisations | Shape | Threads | Section |
| --- | --- | --- | --- | --- |
| `strata_qsa_norm_rope` | `k` (256-dim, `attn_k_norm`), `q` (256, `attn_q_norm`, reads `q_full` at stride 512, writes `qcur` at stride 256), `iq` (128, `indexer.q_norm`) | rows × 256 | 256 | 5.6 |
| `strata_qsa_kv_append_<mode>` | `f16`, `i8`, `q4`, `k8v4` (O-class) | (2, T) × 256 | 256 | 5.6 |
| `strata_qsa_fwht256` (O-class) | | ⌈rows/8⌉ × 256 | 256 | 5.6 |
| `strata_idx_snapshot` | Raw-bit snapshot of the 12 QSA layers; source strides are float and int32 element counts | 1 × 384 | 384 | 5.6 |
| `strata_idx_append` | `window`, `commit` | 1 × 128 | 128 | 5.6 |
| `strata_sel_scores` | | min(256, ⌈(top+1)/8⌉) × 256 | 256 | 5.6 |
| `strata_sel_topk_reg` | | T × 1024 | 1024 | 5.6 |
| `strata_sel_topk_mem` | | T × 256 | 256 | 5.6 |
| `strata_attn_chunk_<mode>` | `f16`, `i8`, `q4`, `k8v4` (O-class) | (⌈id_row_stride/64⌉, 2, T) × 256 | 256 | 5.7 |
| `strata_attn_merge` | | (24, T) × 256 | 256 | 5.7 |
| `strata_qsa_gate` | | (24, T) × 256 | 256 | 5.6 |

**Mat-vec**

| Kernel | Specialisations | Shape | Threads | Section |
| --- | --- | --- | --- | --- |
| `strata_q_q8_1` | | (⌈n/256⌉, T) × 256 | 256 | 5.8 |
| `strata_mv_<fmt>_n<NC>` | 11 Phase-1 dense formats × NC 1, 2, 4, 8, plus the IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS and IQ1_M expert row-dot twins | ⌈n_out/R⌉ × 128 | 128 | 5.8 |
| `strata_mv_<fmt>_n1_x` | the head formats Q5_K, Q6_K, IQ4_XS | as `strata_mv_<fmt>_n1` | 128 | 5.8 |
| `strata_mv_bf16_b<BS>_n<NC>` | BS 256, 160; NC 1, 2, 4, 8 | n_out × BS | BS | 5.9 |

**MoE**

| Kernel | Specialisations | Shape | Threads | Section |
| --- | --- | --- | --- | --- |
| `strata_moe_route` | `r16` (512 experts), `r8` (256) | T × 32 | 32 | 5.10 |
| `strata_moe_plan` | | 1 × 128 | 128 | 5.10 |
| `strata_moe_gu_<fmt>` | `q2_0`, `iq1_m`, `iq2_xxs`, `iq2_xs`, `iq2_s`, `iq3_xxs`, `iq3_s`, `iq4_xs` implemented | (160, T·10) × 256 | 256 | 5.10 |
| `strata_moe_swiglu` | | (3, T·10) × 256 | 256 | 5.10 |
| `strata_moe_down_<fmt>` | `q2_0`, `iq4_nl` implemented | (320, T·10) × 256 | 256 | 5.10 |
| `strata_shx_swiglu` | shared-expert SwiGLU | (3, T) × 256 | 256 | 5.10 |
| `strata_moe_combine` | | (10, T) × 256 | 256 | 5.10 |

**MTP drafter**

| Kernel | Shape | Threads | Section |
| --- | --- | --- | --- |
| `strata_mtp_add_streams` | (40, T) × 256 | 256 | 5.11 |
| `strata_q_q8_0_scaled` | ⌈n/4096⌉ × 128 | 128 | 5.11 |
| `strata_mtp_gu_s2` | (40, 10) × 256 | 256 | 5.11 |
| `strata_mtp_swiglu_s2` | 25 × 256 | 256 | 5.11 |
| `strata_mtp_down_s2` | (40, 10) × 256 | 256 | 5.11 |
| `strata_mtp_window_ids` | 8 × 256 | 256 | 5.11 |
| `strata_mtp_row_top_prob` | 1 × 1024 | 1024 | 5.11 |
| `strata_mtp_map_ids` | 1 × 32 | 32 | 5.11 |
| `strata_mtp_select` | 16 × 256 | 256 | 5.11 |

**Sampler and utilities**

| Kernel | Shape | Threads | Section |
| --- | --- | --- | --- |
| `strata_samp_greedy` | rows × 1024 | 1024 | 5.12 |
| `strata_samp_split_part` | (61, rows) × 128 | 128 | 5.12 |
| `strata_samp_split_merge` | rows × 32 | 32 | 5.12 |
| `strata_samp_coupled_penalize` (O-class) | 1 × 1024 | 1024 | 5.12 |
| `strata_util_copy_strided` | ⌈rows·cols/256⌉ × 256 | 256 | 5.13 |
| `strata_util_to_f16` | ⌈rows·cols/256⌉ × 256 | 256 | 5.13 |
| `strata_util_to_bf16` | ⌈rows·cols/256⌉ × 256 | 256 | 5.13 |

**Prefill** (section 6)

| Kernel | Specialisations | Section |
| --- | --- | --- |
| `strata_pf_gr_norm`, `strata_pf_gr_norm_lo`, `strata_pf_gr_mix`, `strata_pf_gr_mix_lo`, `strata_pf_gr_write_norm`, `strata_pf_gr_write_norm_lo`, `strata_pf_gr_silu`, `strata_pf_gr_silu_lo` | | 6.3 |
| `strata_pf_gdn_gates`, `strata_pf_gdn_conv_tiled`, `strata_pf_gdn_conv_hist`, `strata_pf_gdn_l2`, `strata_pf_gdn_rec`, `strata_pf_gdn_out_norm` | | 6.4 |
| `strata_pf_gate_attn` | | 6.5 |
| `strata_idx_append_first`, `strata_idx_append_blocks`, `strata_idx_append_tail` | | 6.5 |
| `strata_pf_dequant_<fmt>` | the 11 dense formats | 6.2 |
| `strata_pf_gemm_f16` | `mpp`, `simd` | 6.2 |
| `strata_pf_gemm_bf16` | `mpp`, `simd` | 6.2 |
| `strata_pf_sort_count`, `strata_pf_sort_scan`, `strata_pf_sort_scatter`, `strata_pf_gather_f16` | | 6.6 |
| `strata_pf_moe_gu_<fmt>`, `strata_pf_moe_down_<fmt>` | the 8 gate/up and 2 down formats; `mpp`, `simd`. The gate/up tile applies the SwiGLU in its epilogue. | 6.6 |
| `strata_pf_swiglu_pair` (the shared expert's SwiGLU) | | 6.6 |
| `strata_pf_combine` | | 6.6 |

Prefill also uses these decode kernels, with T up to the chunk length unless a batch is named:
- `strata_embed_rows_<fmt>` and `strata_gr_broadcast` (section 6.7);
- `strata_gr_post` and `strata_gr_cvec` (section 6.3);
- `strata_rms_norm_w` and the four `strata_ple_*` kernels (section 6.7);
- `strata_qsa_norm_rope`, `strata_qsa_kv_append_<mode>` and `strata_qsa_fwht256` (section 6.5);
- `strata_sel_scores` and `strata_sel_topk` in groups of 256 rows, and `strata_attn_chunk_<mode>` and
  `strata_attn_merge` in batches of 32 rows (section 6.5);
- `strata_moe_route` with its `div` argument set (section 6.6);
- `strata_util_to_f16` and `strata_util_to_bf16` (sections 6.2 and 6.7).

### 2.5 What is not ported

| Upstream | Why |
| --- | --- |
| Canonical kernels (`gr.cu`, `router_top10.cu`, `gdn.cu`, `qsa.cu` attend and index, `s2_gemv*`, `s_gemv`, `gemv_quantized`, `quantize_q8_0`, `quantize_q8_K`) | The Mac implements only the production configuration ([03](03-model-reference.md), section 6.1). The S2 blob kernels survive only for the MTP (section 5.11). |
| Native non-fused kernels (`native_gdn*`, `native_qsa_score`, `native_flash_attn`) | Not on the production path ([K2 section 6](audit/K2-qsa-kv-gdn.md)) |
| Single-token `gr_down` / `gr_up`, `gr_down_staged`, `gr_down_v3`, `fused_gr_read` | The multi-token kernels at T = 1 replace them (section 5.4) |
| `sampler_one_block`, the old sampler, `coupled_stage` | The split sampler serves every row count; the coupled tail runs on the CPU (section 5.12) |
| `copy_indexed` | Its one use restores the PLE history from a per-row snapshot (`verify.cpp:996`); the Mac rebuilds the history at commit instead (section 4.8) |
| `hit_select*`, `add_hits`, the per-hit S2 kernels, `cpu_order_*`, `activation_correction` | Unified memory makes every expert a hit; the grouped kernels serve every case ([K3 section 5.5](audit/K3-quant-and-matvec.md)) |
| KV streaming (`kv_stream.cu`), `kv_gather*`, `ring_kernel`, mapped copies | Unified memory ([K2 section 3.5](audit/K2-qsa-kv-gdn.md)) |
| `block_topk_kernel` with a global-memory key array for `max_blocks > 33,792` as a separate code path | Kept as the `mem` specialisation of `strata_sel_topk` (section 5.6) |
| Q5_0 and Q5_1 expert downs, UD-Q4_K_XL's formats, Q2_K, IQ1_S, MXFP4, NVFP4 | No catalogue file uses them ([03](03-model-reference.md), section 3.3). Adding UD-Q4_K_XL later means adding `strata_moe_gu_q4_k`, `_q5_k` and `strata_moe_down_q5_1` ([07](07-server-setup-port.md), section 4.1). |
| `--gr-fp32-activations`, `--cpu-oracle-q8-0`, `--gr-native-mmvf` arms | Diagnostic arms with no Metal variant (section 9) |

A tensor whose type has no kernel is refused before any buffer is planned, with the reason code
`unsupported_type_<T>` ([05](05-memory-and-residency.md), section 3.4). The check reads the kernel tables of this
section, generated from the manifest: a format is supported for a role exactly when its kernel exists.

### 2.6 Phase 2

| Kernel | What it replaces | Condition |
| --- | --- | --- |
| `strata_pf_prompt_attn_<mode>` | Prefill attention through `strata_attn_chunk` in batches of 32 queries | The prefill-speed criterion PF-1 fails ([09](09-testing-and-validation.md)), or L-13 finds prefill attention above 25% of a chunk (work item L19) |
| `strata_pf_sel_scores_mma` | Prefill block scores through `strata_sel_scores` | L-13 finds selection above 25% of a chunk (work item L20) |
| `strata_mtp_round_device` | The host acceptance step between the window and the MTP command buffer ([06](06-engine-host-port.md), section 10.2) | After gates C1-C4 pass (work item L21) |

Phase-2 kernels are not bitwise equal to the decode kernels. They are held to the prefill tolerance of section 10.

---

## 3. Numerics

### 3.1 The two contracts

[03](03-model-reference.md), section 6, sets two parity contracts. This section gives the rules that make the
kernels meet them.

| Contract | Holds for | Checked by |
| --- | --- | --- |
| **Bitwise within the Mac build** | Quantisers (Q8_1, Q8_0-scaled, KV int8 and q4), router ids, selection ids, the embedding gather, the PLE history, sampler candidates, every integer dot; decode ≡ verify for every T; NC-column mat-vecs ≡ one column; tiered ≡ full experts | Twins with tolerance class B, and the property tests of section 10.4 |
| **Bounded against an oracle** | Every kernel whose order this document fixes but which calls `exp`, `log`, `sqrt` or `rsqrt`. The oracle uses the same order, with each library call in double rounded to float. | Twins with tolerance class U(n) or R(ε) (section 10.2) |
| **Tolerance against an FP32 or FP64 reference** | The prompt path's GEMMs, where the order is the tile's: the dense projections (section 6.2) and the expert tiles (section 6.6); the Phase-2 prompt attention (section 2.6) | Twins with classes U(2K) and R(2^-16) (section 10.3) |
| **KL against CUDA** | The whole model | Gate C1 ([09](09-testing-and-validation.md)) |

No kernel's output is required to equal a CUDA output bit for bit (section 3.9).

### 3.2 Math modes and the fast-folder rule

There is one library and two source folders ([08](08-build-and-distribution.md), section 5.3):

| Folder | Flags | Holds |
| --- | --- | --- |
| `safe/` | `-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off` | Every Phase-1 kernel |
| `fast/` | `-fmetal-math-mode=fast` | `strata_selftest_exp` only |

What safe mode gives, measured on M5 (appendix A.10):

| Operation | Safe mode |
| --- | --- |
| `/`, `1/x`, `sqrt`, `rsqrt` | Correctly rounded |
| `exp` | ≤ 1 ulp. `precise::exp` gives the same bits. |
| `rint` | Half to even |
| `round` | Half away from zero. It returns +0 for inputs in (−0.5, −0]. |
| `trunc` | As `truncf` |
| `a*b + c` | Two roundings (no contraction) |
| `fma(a, b, c)` | One rounding |
| `log` | Not measured on its own. MSL has no `log1p` (section 3.6). |

Rules:
- A safe file **must not** call a `fast::` function. The build refuses it ([08](08-build-and-distribution.md),
  section 5.4).
- No kernel calls `sin`, `cos`, `tan`, `pow` or `powf`. RoPE angles come from the host table (section 4.9).
- A kernel moves to `fast/` only when all of these hold, and the move is recorded in section 9:
  1. an A/B on M5 and on one of M1-M4 shows at least a 5% gain in the kernel's GPU time in a release-build window;
  2. the kernel has no bitwise contract (section 3.1) and tests no NaN or infinity (fast mode assumes neither);
  3. every division in the kernel that must be correctly rounded uses `precise::divide`;
  4. its twin passes with the tolerance it had in `safe/`, and gate C1 still passes.

### 3.3 The contraction rule

nvcc contracts `a*b + c` into an FMA by default (`--fmad=true`) in every `.cu` file, with or without fast math
([K1 conventions](audit/K1-gr-router-shared-rope-cvec.md)). The Metal build never contracts implicitly. Each kernel
writes its FMAs out, in six steps:

1. Every file is compiled with `-ffp-contract=off`. `a*b + c` is a multiply and an add, each rounded.
2. Where upstream writes `fmaf`, `__fmaf_rn` or `fma`, the kernel writes `fma`.
3. Where upstream writes `__fmul_rn` and `__fadd_rn` (the embedding gather, `pre_gated`, the PLE gate products and
   `conv_residual`, the `cpu_order` quantiser), the kernel writes plain `*` and `+` in the same order.
4. Where upstream writes a plain product feeding an add or subtract (`a*b + c`, `c += a*b`, `a*b - c`, `c - a*b`),
   the kernel writes `fma(a, b, c)`, `fma(a, b, -c)` or `fma(-a, b, c)`.
5. A sum of products is evaluated left to right. The first product is rounded and each later product is fused into
   the running sum: `a*b + c*d + e*f` becomes `fma(e, f, fma(c, d, a*b))`. L-9 confirms this against the SASS of the
   pinned upstream build and corrects the oracles and kernels where nvcc chose differently.
6. The oracle is compiled with `-ffp-contract=off` and calls `std::fma` exactly where the kernel calls `fma`
   ([08](08-build-and-distribution.md), section 6.5).

The places where step 4 or 5 applies:

| Upstream expression | Kernel | Section |
| --- | --- | --- |
| `sq = r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w` (GR norm) | `fma(r.w, r.w, fma(r.z, r.z, fma(r.y, r.y, r.x*r.x)))` | 5.4 |
| `R[i] = R[i] + block_out[d]*w[c]` (GR write) | `fma(block_out[d], w[c], R[i])` | 5.4 |
| `x0*c - x1*s`, `x0*s + x1*c` (RoPE) | `fma(-x1, s, x0*c)`, `fma(x1, c, x0*s)` | 5.6 |
| `k.x*q.x + k.y*q.y + k.z*q.z + k.w*q.w` (block score) | `fma(k.w, q.w, fma(k.z, q.z, fma(k.y, q.y, k.x*q.x)))` | 5.6 |
| `k8[0]*qa.x + ... + k8[7]*qb.w` (attention score) | the same chain over eight terms | 5.7 |
| `x*id + 8.5` (q4_0 append) | `fma(x, id, 8.5f)` | 5.6 |
| `s + shared*sigm` (prefill shared gate) | `fma(shared, sigm, s)` | 6.6 |
| `h0*w0 + h1*w1 + h2*w2 + x*w3` (GDN conv) | `fma(x, w3, fma(h2, w2, fma(h1, w1, h0*w0)))` | 5.5 |
| `v - g*kv_col` (GDN delta) | `fma(-g, kv_col, v)` | 5.5 |

Section 5 states the FMA structure of each kernel. A kernel spec that writes `a*b + c` means two roundings.

### 3.4 Reductions

- **Bitwise reductions** use the explicit butterfly in CUDA's order:
  `for (ushort o = 16; o > 0; o >>= 1) v += simd_shuffle_xor(v, o);`. Every lane holds the result. This applies to
  every value that a twin compares bitwise, and to every reduction this document gives an order for.
- **`simd_sum`** adds in the ascending order o = 1, 2, 4, 8, 16 on M5 (A.10), which is not CUDA's. It is allowed only
  where the result is held to a tolerance. L-3 records the order on M1-M4; no bitwise contract depends on it.
- **Maxima** use `simd_max` or `fmax`, which do not depend on the order. Argmax shuffles (value, index) pairs; the
  tie rule is the kernel's own (larger value, then lower index, unless section 5 says otherwise).
- **Threadgroup partials.** Each simdgroup writes its partial to a `threadgroup` array, then a
  `threadgroup_barrier(mem_flags::mem_threadgroup)`, then one simdgroup combines the partials in the fixed order that
  section 5 gives for the kernel (serial ascending `((p0 + p1) + p2) + ...` unless stated).
- **No floating-point atomics.** Integer atomics on threadgroup memory are allowed (the selection histograms), since
  integer addition does not depend on the order.
- **Mat-vec dots** keep a fixed map from lanes to elements, then the butterfly. The map never depends on T or NC, which
  is what makes NC columns equal one column (section 5.8).

### 3.5 Conversions

| Conversion | Kernel | Oracle |
| --- | --- | --- |
| float → half | `half(x)`: round to nearest even, subnormal halves kept, ≥ 65,520 gives infinity | `f16_from_f32_metal`: IEEE RNE, plus two rules: every NaN gives `0x7e00`, and a float32 denormal input gives +0 |
| half → float | `float(h)`, exact | exact |
| bf16 → float | `as_type<float>(uint(u) << 16)` | the same |
| float → bf16 | `f32_to_bf16_rne` (section 3.6). Never `bfloat(x)`. | the same integer code |
| float → int, half to even (upstream `__float2int_rn`, `rintf`) | `int(rint(x))` after the clamp | `std::nearbyint` in the default rounding mode |
| float → int, half away (upstream `roundf`) | `int(round(x))` | `std::round` |
| float → int, toward zero (upstream `__float2int_rz`) | `int(trunc(x))` or `int(x)` | `std::trunc` |

Float32 denormal *arithmetic* on the GPU is not measured. The oracles assume IEEE behaviour; L-10 checks whether the
GPU flushes denormal operands or results, and if it does, the oracles flush too.

### 3.6 Helpers (`common/math.h`)

```c++
static inline float sigmoid_k(float x)  { return 1.0f / (1.0f + exp(-x)); }
static inline float silu_k(float x)     { return x / (1.0f + exp(-x)); }
static inline float log1p_k(float x)    { float u = 1.0f + x; return u == 1.0f ? x : log(u) * (x / (u - 1.0f)); }
static inline float softplus_k(float v) { return v > 20.0f ? v : log1p_k(exp(v)); }
static inline float bf16_to_f32(ushort u) { return as_type<float>(uint(u) << 16); }
static inline ushort f32_to_bf16_rne(float x) {
    uint i = as_type<uint>(x);
    return isnan(x) ? ushort((i >> 16) | 64u) : ushort((i + ((i >> 16) & 1u) + 0x7fffu) >> 16);
}
static inline half half_sat(float x) {
    uint bits = as_type<uint>(x);
    if ((bits & 0x7f800000u) == 0u && (bits & 0x007fffffu) != 0u) return as_type<half>(ushort(0u));
    if (isnan(x)) return as_type<half>(ushort(0x7e00u));
    return half(clamp(x, -65504.0f, 65504.0f));
}
```

| Helper | Bound in safe mode (A.10) | Upstream form |
| --- | --- | --- |
| `sigmoid_k` | ≤ 4 ulp on [−30, 30] | `1/(1+__expf(-x))` or `1/(1+expf(-x))` |
| `silu_k` | not separately measured; L1 accepts U(8), by analogy with `sigmoid_k` | `s/(1+__expf(-s))` |
| `log1p_k` | ≤ 3 ulp on (−0.999, 100) | `log1pf` |
| `softplus_k` | ≤ 5 ulp on [−30, 30], against double and against `log1pf(expf(v))` | `v > 20 ? v : log1pf(__expf(v))` |
| `f32_to_bf16_rne` | exact | the NaN-quieting RNE of `native_bf16.cu`; prefill's `bf()` has no NaN branch, and the Mac uses this one function everywhere |
| `half_sat` | exact | prefill's `hf_sat` (`prefill/kernels.cu:43`): clamp to ±65,504, keep a NaN, then round to half (RNE) |

`log1p_k` only works in safe mode: fast mode folds `u - 1` to `x` and it becomes `log(1 + x)` (A.10). The GR write
gate `2·sigmoid(inject/4)` has three upstream flavours; section 5.4 says which kernel uses which.

### 3.7 Codebooks

- The tables are program-scope `constant` arrays in `common/codebooks.h`, copied from ggml-common.h at llama.cpp
  `3cf03257f`:

  | Table | Size |
  | --- | ---: |
  | `iq2xxs_grid` (256 × u64) | 2 KiB |
  | `iq2xs_grid` (512 × u64) | 4 KiB |
  | `iq2s_grid` (1024 × u64) | 8 KiB |
  | `iq3xxs_grid` (256 × u32) | 1 KiB |
  | `iq3s_grid` (512 × u32) | 2 KiB |
  | `iq1s_grid_gpu` (2048 × u32) | 8 KiB |
  | `ksigns_iq2xs` (128 × u8), `kmask_iq2xs` (8 × u8), `kvalues_iq4nl` (16 × i8) | 152 B |

- Program-scope `constant` was the fastest placement at every size, and a threadgroup copy the slowest at 4 and
  8 KiB (A.10). L-4 repeats the comparison inside the real mat-vec and expert kernels.
- `codebooks.h` is valid as both MSL and C++. A macro `STRATA_CONSTANT` expands to `constant` in MSL and to `const`
  in C++, so the oracles include the same file. The test `codebooks` compares the bytes with the copy in upstream
  `src/kernels/cuda/` and with ggml-common.h.

### 3.8 Integer dots are exact

- `dp4a` becomes `dot4_i8(a, b)`: unpack two `char4`, multiply and add in `int`. `__byte_perm` and `__vsubss4` become
  explicit `char4` operations. Integer sums cannot overflow at these block sizes: a 32-element block is at most
  32 · 127 · 127 = 516,128, and every block sum is converted to float before it is scaled.
- Integer arithmetic does not depend on the order, so the integer part of every quantised dot is bitwise for free.
  The float part (`d_w · d_x · sumi`, the Q8_1 `Σx` correction, the per-block accumulation) follows the order of
  section 5.8 exactly.

### 3.9 Deviations from CUDA

These are the places where the Mac computes something different from CUDA on purpose. Each changes only the last
bits, which is why gate C1 compares the Mac with CUDA by KL and the ladder, not by equality.

| Upstream | Mac | Effect |
| --- | --- | --- |
| `__expf` (`ex2.approx`) in the native, fused, verify and grouped-S2 kernels, and in the prompt path's `prefill/kernels.cu` (`sigm`, the two SwiGLUs, the GR and GDN activations, the decay); the accurate `expf` of the prompt path's router and `gate_attn` | `exp`, ≤ 1 ulp | Last bits of sigmoid, SiLU, softplus, softmax, the GDN decay, the drafter's expert SwiGLU and draft probability, on both paths |
| `rsqrtf` (MUFU, not correctly rounded) | correctly rounded `rsqrt` | Last bits of every RMS and L2 norm |
| Fast-math division in `native_*.cu` | correctly rounded `/` | Last bits of norms and router weights |
| Fast math's flush of FP32 denormals to zero (`--ftz=true`) in `native_*.cu` | Denormals as L-10 finds them (section 3.5) | Values below 2^-126 only |
| `native_quantize_q8_1` (fast-math `amax/127` and `x/d`) for the dense projections, the head, the shared expert and the MTP | One precise quantiser for all paths (section 5.8) | Codes can differ near half-integer boundaries; subnormal scales follow the documented deterministic policy where CUDA conversion has no portable result |
| Decode RoPE angles from fast `powf`/`__cosf`/`__sinf`; prefill angles from accurate `powf`/`cosf`/`sinf` | One FP64 host table for every path ([03](03-model-reference.md), section 6.7) | Decode ≡ prefill on the Mac; up to 2e-2 rad closer to the exact angle than CUDA decode at 262K |
| Canonical FP64-normalising router for 256 experts, in decode | One FP32 router for both widths (section 5.10) | Router weights in the last bits; ids equal on tie-free inputs. The prompt path matches upstream's `route_kernel<8>` (section 6.6). |
| `log1pf` | `log1p_k` | ≤ 3 ulp |
| The GPU FP64 sampler tail | The same arithmetic on the CPU in FP64 (section 5.12) | The order of operations is the same. Only libm's `exp` and `log` against CUDA's (each within an ulp of the true value) can differ, so a sampled pick can change only where its uniform lies within about 1e-16 of a cumulative boundary. `sampler_parity` holds the tail to exact tokens on its cases. |
| The drafter's prompt K/V through the batched E-9 path (one GPU, `kv_mode` 0) | `mtp.prefill`'s per-row arithmetic (section 5.11) | The drafter's K/V bits, so drafts only |
| Prefill `bf()` without the NaN branch | `f32_to_bf16_rne` | NaN inputs only |
| TF32 prefill scorer and MMA prompt attention | The decode kernels in query batches (Phase 1, section 6.5) | Prefill selection may differ near ties, as CUDA's own prefill and decode differ |
| cuBLAS `cublasGemmEx`, with the algorithm cuBLAS chooses, for every prompt-path projection | The Mac's tiles (sections 6.2 and 6.6) | Last bits of every prompt-path GEMM. Both are F32 sums of exact products in an order neither build fixes. |
| MMQ on Q8_1 activations, upstream's default for the prompt path's routed experts (`moe_mmq.cu`) | F16 tiles on F16 activations, upstream's `STRATA_PREFILL_MMQ=0` path (section 6.6) | The routed experts' outputs on the prompt path. CUDA references of the prompt path are taken with `STRATA_PREFILL_MMQ=0` ([09](09-testing-and-validation.md)). |

---

## 4. Data and binding

### 4.1 Argument conventions

- `[[buffer(0)]]` is always the parameter block of the pass: `constant WindowParams&`, `constant PrefillParams&`
  or `constant MtpParams&`. A window and a commit have one block per command buffer; a prefill chunk has its block
  and six batch slots (section 6.1); a drafter command buffer has one block per pass (section 5.11). The host writes
  the blocks into a Shared buffer before it commits the command buffer.
- Every parameter block starts with `uint T`, its row count, at offset 0. A kernel shared by the window, prefill and
  the drafter reads only `T` from `buffer(0)`, so any of the three blocks can be bound there. Per-row data comes
  through arguments bound at offsets inside the block: `steps` at its `step`, `rows` at its `T` (or at another
  count, section 5.11), `tokens` at its `tokens`, `samp` at its `samp`.
- **In prefill** the per-row arrays are arena buffers of their own (section 6.1). `steps` is `pf_steps`, bound at
  `row0 · sizeof(StepRow)` for a query batch whose first row is `row0`, and `tokens` is the chunk's `tokens`. A kernel
  that runs over one batch binds that batch's slot at `buffer(0)`; one that runs over the chunk binds
  `PrefillParams`.
- **Per-step values are never kernel arguments** ([06](06-engine-host-port.md), section 8.2, rule 2): token,
  position, T, n_keep, row counts and sampler settings are read from the parameter block. `setBytes` carries only
  values fixed for the life of the engine (a layer index, a dimension, an epsilon, a tensor offset).
- `build_window_params` is the host constructor for this block. It accepts 1..8 token rows and one resolved sampler
  row per token, chooses the smallest `nc` in 1/2/4/8 that covers T, derives `StepRow` and the Philox counter from
  `pos0`, and leaves inactive fixed-capacity rows zeroed. It accepts `n_keep = 0` for an ordinary window and
  `1..T` for commit or write-through metadata; the caller owns sampler-history buffer bounds.
- Grid sizes are set at encode time. Where a grid depends on a per-step value (the number of indexer blocks), the
  host sizes it from the same value it writes into the block, and the kernel exits early past the row's own limit.
- Pointers are `device const T*` for inputs and `device T*` for outputs, with natural member types (section 2.3).
  No kernel reads through a pointer type wider than the alignment the data guarantees: block loads are `ushort` and
  `uchar` (2-B granularity), and wider loads are used only where section 4.3 guarantees 16-B alignment.

### 4.2 Parameter blocks (`common/types.h`)

```c++
struct StepRow {        // the step layout of 03 section 3.1, one per window row
    int pos;            // cell of this row = pos0 + t
    int n_kv;           // pos + 1: cells this row attends to
    int n_bid;          // (pos + 1) / 4: completed indexer blocks
    int width;          // min(n_kv, 2051): selected cells
};

struct SampRow {        // upstream SamplerParams (sampler.hpp:15-28) for one row
    float temperature;  // <= 0 means greedy
    float top_p, min_p;
    float penalty_repeat, penalty_freq, penalty_present;
    int   top_k;        // 1..63 as given; 0 or >= 64 means 64
    int   min_keep;
    int   greedy;       // 1: argmax (also when temperature <= 0)
    uint  counter;      // Philox counter of this row = pos0 + t
    uint  hist_off;     // this row's penalty history in the history buffer (penalty_rows)
    uint  hist_len;     // min(penalty_last_n, tokens available)
};

struct WindowParams {
    uint    T;          // rows, 1..8
    uint    nc;         // mat-vec columns: the smallest of 1, 2, 4, 8 that is >= T
    uint    pos0;
    uint    n_keep;     // rows kept, 1..T: read by commit and by a write-through window (section 8); 0 in other windows
    uint    seed_lo, seed_hi;
    uint    tokens[8];
    StepRow step[8];
    SampRow samp[8];
};
```

- `PrefillParams` (16 B: `T`, `p0`, `idx_f`, `idx_e`) and the six batch slots that follow it are specified in
  section 6.1. A chunk's `StepRow`s are in `pf_steps`, not in the block.
- `MtpParams` is one pass of the drafter. Its blocks are an array of 256-B slots of their own, and section 5.11
  gives the values the host writes:

  ```c++
  struct MtpParams {      // one pass of the MTP drafter (section 5.11)
      uint    T;          // rows: 1..8 for a K/V-only pass, 1 for the full layer
      uint    hid[4];     // fc_hidden pass k: min(8, 4T - 8k) columns, 0 past the last pass
      uint    a;          // K/V-only pass of a round: the accepted row the full layer continues from
      uint    j;          // full layer: the draft slot it writes, 0..6
      uint    tokens[8];  // K/V-only pass: row t's token, the one at cell step[t].pos + 1
      StepRow step[8];    // the cells of the rows
      SampRow samp;       // full layer: the sampler settings of its draft
  };                      // 236 B; static_assert(sizeof(MtpParams) <= 256)
  ```
- The parameter buffer is a ring of two slots, `slot = round % 2`. A window and the commit that follows it read the
  same slot. Phase 1 never has two rounds in flight; the second slot is for the Phase-2 encoding of the next window
  ([06](06-engine-host-port.md), section 10.2). `WindowParamsRing` owns one fixed-address Shared allocation for
  these slots, with each slot stride rounded up to 256 bytes. It builds the canonical parameter block before
  exposing the slot view. The caller must wait for a slot's previous command-buffer use to finish before rebuilding
  it.
- The structs are shared by MSL and C++ (`common/types.h` compiles as both). A `static_assert` on each side pins the
  size and every offset.
- `Q80S` is a logical decode value (`float d` plus 32 signed codes), not the physical `strata_q_q8_0_scaled`
  output: that quantiser writes the scale and code planes separately (section 5.11).

### 4.3 Weights

- **Dense tensors** are bound with `setBuffer:offset:atIndex:`, where the buffer is the window of
  [05](05-memory-and-residency.md), section 4.2, that covers the tensor, and the offset is the tensor's offset inside
  it. GGUF tensor data is 32-B aligned and windows are 16 KiB aligned, so every offset is 32-B aligned. A.11 showed a
  32-B-aligned, non-256-B-aligned offset binds correctly.
- **Routed experts** are read through the address table:

  ```c++
  struct ExpertAddr { device const uchar* role[3]; };   // 0 gate, 1 up, 2 down
  // device const ExpertAddr* addr: addr[l * n_expert + e]
  ```

  Each entry is `buffer.gpuAddress + offset`. The host writes the table at load in full mode and on every slot fill
  in tiered mode ([05](05-memory-and-residency.md), section 5.3). Expert slabs are 32-B aligned
  ([03](03-model-reference.md), section 3.6). Every buffer the table points into is in the engine's residency set
  ([05](05-memory-and-residency.md), section 4.3), so no encoder calls `useResource`; A.11 test 1 verified this.
- The MTP experts are S2 blobs at `mtp_blob + e · 1,382,400` (section 5.11), bound by buffer and offset.
- BF16 tensors are `device const ushort*`; F32 tensors (norm weights, `ssm_a`, `dt_bias`) are `device const float*`;
  F16 tensors (the PLE conv weight) are `device const half*`.

### 4.4 Scratch and the sink

- Every activation of a window lives in the window scratch arena ([05](05-memory-and-residency.md), section 2.3). The
  host lays it out once at startup for T = 8; a window of T rows uses the first T rows of each region. Each region
  starts on a 256-B boundary.
- The arena holds, per window: the residual streams `R[T][4][2560]`, the two inject buffers, the GR intermediates
  (`xn`, `lo`, `gate`), the output-head mixer regions `head_normalized[T][4][2560]`,
  `head_low_rank[T][4][320]`, and `head_gate[T][4][2560]` (680 KiB total at T=8),
  the mixed rows `head_mixed[T][2560]`, the Q8_1 activations, the dedicated head Q8_1 rows
  `head_q8_1[8][80]` of 36-byte blocks, and the per-layer GDN inputs that commit replays (`qkv`, `h`, `gate`,
  `beta` for each of the 36 GDN layers), the QSA intermediates, the router plan, the expert intermediates, the
  logits `[T][248,320]`, the PLE normalised rows `ple_norm[8][10240]` (section 4.8), the indexer tail and block
  position snapshots (section 4.6), and the sampler's history rows, candidate lists, selected lists and greedy
  outputs (section 5.12). `WindowScratchLayout` names and sizes these regions;
  `WindowScratchStorage` opens one Shared allocation for T = 8 and exposes bounded views of its named regions.
  The window uses the first T rows of row-shaped regions.
- **The sink.** A mat-vec with NC > T computes NC − T padding columns. A padding column reads row 0's activation (so
  it never reads unwritten bytes) and writes `sink + (t − T) · n_out`. The sink holds 3 · 248,320 floats (2.84 MiB):
  NC − T is at most 3 and the head is the widest output. Nothing reads the sink.
- The layout omits CUDA-only PCIe staging and CPU-expert hit buffers. Mac experts are read through the full or
  tiered Metal path; neither buffer is consumed by the Window kernels.
- The prefill arena is a separate allocation with its own layout (section 6.1). A window never carries image rows,
  so the window arena has no `img_in` (section 6.7).
- With `STRATA_METAL_POISON=1` the arena is filled with `0xFF` at creation ([06](06-engine-host-port.md), section 4.4).
  Unwritten float reads become NaNs; integer and byte reads see all-ones values. Tests use type-appropriate poison
  expectations.

### 4.5 KV cache

The layout is upstream's with the page table removed (`page = cell / 4`, [K2 section 3.1](audit/K2-qsa-kv-gdn.md)):

```
row(cell, kvh) = ((cell / 4) * 2 + kvh) * 4 + cell % 4        // kvh in {0, 1}
```

Each of the 12 QSA layers has a K pool and a V pool, each `max_cells × 2` rows. Per mode:

| Mode | K row | V row | Scales |
| --- | --- | --- | --- |
| `f16` | `half[256]` at `row · 256` | the same | none |
| `i8` | `char[256]` at `row · 256` | the same | `ushort` (half bits) `scale[row · 4 + g]`, one per 64-value group g, in separate K and V scale arrays |
| `q4` (O-class) | 8 × `block_q4_0` (18 B) at `row · 144` | the same | inside the block |
| `k8v4` (O-class) | as `i8`, unrotated | as `q4`, FWHT-rotated | K scales are separate; V scales are inside each q4 block |

- For `i8`, the four arrays of a layer (K codes, K scales, V codes, V scales) are regions of the one KV
  allocation. For `k8v4`, K codes, K scales and V blocks occupy three regions; each V scale is stored inside
  its `block_q4_0`. The host computes these offsets from `max_cells` ([05](05-memory-and-residency.md), section 2.2).
- The main-model `KvStore` places all 12 layers' K/V regions, indexer state (section 4.6) and the shared RoPE
  table (section 4.9) in one Context-owned allocation. Its capacity is `round_up_4(--max-context)` so every
  four-cell-interleaved row is addressable; the planner reserves the exact resulting byte count.
- The MTP drafter has one K/V layer with `round_up_4(max_cells)` cells, in the main model's `--kv` mode except
  that `k8v4` becomes `i8`. Its dense attention has no indexer, so the Metal allocation omits indexer rows and
  pooled keys. MTP borrows the main `KvStore`'s shared RoPE table; see [05](05-memory-and-residency.md), section 2.2.
- Rejected cells keep whatever the window wrote; the next window overwrites them before any query reads them
  ([03](03-model-reference.md), section 8).

### 4.6 Indexer buffers

Per QSA layer, as upstream's `QsaIndexerBuffers` ([K2 section 3.4](audit/K2-qsa-kv-gdn.md)):

| Buffer | Type | Contents |
| --- | --- | --- |
| `tail` | `float[3][128]` | Raw keys of the current block's first three cells, rounded through half |
| `dead` | `float[128]` | The spare key, written only at position 0 |
| `pooled` | `float[max_cells / 4 + 2][128]` | Pooled, normed, rotated block keys; row `b + 1` holds a copy of `dead` |
| `block_pos` | `int` | RoPE position of the last completed block |

A rollback-capable window snapshots `tail` and `block_pos` of all 12 QSA layers before its first append
(`strata_idx_snapshot`, 12 · (384 + 1) values). The standalone and caller-owned-Encoder forms accept per-layer
strides in float and int32 elements and write tightly packed snapshots in QSA-layer order. They copy tail words as
raw uint32 bits and validate overflow-safe source extents, Context ownership and disjoint source/output regions. The
shared form can be encoded in an ordered Encoder segment. The caller-owned-Encoder forms of snapshot and append
can compose in one ordered segment: snapshot, window append, then commit replay. In concurrent dispatch mode,
dependent dispatches receive barriers unless the caller explicitly groups them; snapshot and append must not be
placed in the same concurrent group. A write-through window does not snapshot because it never rolls back indexer
state (section 8.5). Commit restores the snapshot and replays the kept tokens (section 5.6).

### 4.7 GDN state

Per GDN layer, one region of `48 · 128 · 128 + 10240 · 3` floats (3.12 MiB):

- the recurrent state, element (i, h, j) at `(i · 48 + h) · 128 + j`, j fastest (i is the k row, j the v column);
- the conv history right after it, channel c's three rows at `c · 3 + r`, r = 0 the oldest.

The 36 regions are contiguous in layer order, which makes the conversation checkpoint one copy
([06](06-engine-host-port.md), section 11.3). A window never writes them. Commit (section 5.5), a write-through
window (section 8.5) and a prefill chunk (section 6.4) do.

### 4.8 PLE history

- `hist`: `float[10240 · 9]`, channel c's nine rows at `c · 9 + r`, r = 0 the oldest. 368,640 B.
- `hist` is written only by commit, by a write-through window (section 8.5) and by a prefill chunk (section 6.7), each
  with `strata_ple_history` (section 5.3). Any other window reads it and never writes it.
- The window keeps its normalised rows `ple_norm[T][10240]` in the arena until the commit that follows it, which
  rebuilds `hist` from the first `n_keep` of them.
- The host's PLE row-index builder also needs the last two committed token IDs. `WindowPleTokenHistory`
  (D-IMPL-301) retains only that oldest-first suffix: constructing rows is read-only, and a speculative window
  advances it only after the caller confirms that the GPU state commit succeeded, using the accepted prefix.
  `SessionStateStorage` owns this CPU-only suffix with the sequence lifecycle; a successful state reset clears it
  after tracked GPU uses are complete without adding bytes to the Metal allocation.
  Seed it only from token IDs already applied to model state. In the prompt path, the last prompt token is the
  first T=1 decode window ([section 6.1](#61-the-chunk-loop-and-the-arena)); do not include it in the initial
  suffix until that window commits. `prepare_window_t1_input` uses the session-owned suffix to stage the PLE
  rows with the T=1 parameters, without advancing token history. This token-ID suffix is separate from the
  floating-point `hist` above; production `Window::run` still does not consume the prepared input.

### 4.9 RoPE table

- `float2 rope[max_cells][32]`: `(cos θ, sin θ)` of pair i at cell p, built on the CPU in double and rounded to float
  ([03](03-model-reference.md), section 6.7). 256 B per cell, 64 MiB at 262,144 cells.
- With M-RoPE (vision), the host builds the table from the position table `[cells][3]`, so a kernel always reads
  `rope[p][i]` and never sees the vision positions.
- The QSA, indexer, prefill and MTP RoPE kernels all read this table. The host rebuilds it in three cases, always
  while the queue is idle: when the context length or the YaRN settings change; before the first chunk of each
  GENI request, from the first image cell to the last cell; and, from that cell again, back to the identity for the
  next text request or after an error ([06](06-engine-host-port.md), section 5.4).

---

## 5. Window kernels

Each subsection gives, per kernel, the grid and threadgroup, what each thread computes, the order of every reduction
(section 3.4) and the FMA structure (section 3.3). "Upstream" is the production `--native` path at the pinned commit,
and file references are to `src/kernels/cuda/` unless they name another directory. Apart from the deviations of
section 3.9, each kernel computes upstream's arithmetic in upstream's order, and its oracle (section 10) transcribes
the upstream code with those deviations applied. Where the Mac changes a shape, the subsection says why the bits do
not change.

Notation:
- `T`, `nc`, `tokens[]` and `step[]` come from `WindowParams` (section 4.2). A mat-vec host name's `NC` equals `nc`.
- `R[t][c][d]` is the residual of row t, stream c, dimension d. `i = c·2560 + d` indexes the 10,240 values of a row.
- `butterfly(v)` is the xor butterfly of section 3.4. A *serial* sum is `((p0 + p1) + p2) + ...`.
- `sigmoid_k`, `silu_k`, `softplus_k` and `log1p_k` are the helpers of section 3.6.
- `tg` is `threadgroup_position_in_grid`, `tid` is `thread_index_in_threadgroup`, `lane = tid % 32` and
  `warp = tid / 32`.
- Every kernel serves T = 1..8 with the same per-row arithmetic; T changes only loop bounds and grid sizes. This is
  what makes decode ≡ verify (section 10.4).

### 5.1 The window, in order

Decode is a window of T = 1; there is no separate decode path ([06](06-engine-host-port.md), section 10.1). Before
the window is committed, the host has written `WindowParams` and the PLE input rows `ple_in[T][2560]` into Shared
buffers ([06](06-engine-host-port.md), section 7.4). The window's command buffer then encodes:

| Step | Kernels | Section |
| --- | --- | --- |
| 1 | `strata_embed_rows_<fmt>`, `strata_gr_broadcast`; `strata_idx_snapshot` for rollback-capable windows only | 5.2, 5.6 |
| | **For each layer l = 0..47:** | |
| 2 | l = 1 with the PLE on: `strata_gr_post` (layer 0's FFN write), then the PLE block | 5.3 |
| 3 | Attention read: `strata_gr_norm`, `strata_gr_down`, `strata_gr_up` with `apply = pending(l)`, `inj_prev = inject2`, `inj_out = inject` → `mixed` | 5.4 |
| 4 | Block: GDN when `l % 4 != 3`, otherwise QSA → `bo` | 5.5-5.7 |
| 5 | FFN read: the same three kernels with `apply = 1`, `inj_prev = inject`, `inj_out = inject2`. This folds the attention write `(bo, inject)` into R. → `mixed` | 5.4 |
| 6 | Router: `strata_mv_bf16_b256_n<NC>` (`ffn_gate_inp`, 2560 → n_expert), `strata_moe_route` | 5.9, 5.10 |
| 7 | `strata_q_q8_1(mixed)` → `xq`, read by steps 8 and 9 | 5.8 |
| 8 | Shared expert: `strata_mv_<fmt>_n<NC>` (gate, up), `strata_shx_swiglu`, `strata_q_q8_1`, `strata_mv_<fmt>_n<NC>` (down), `strata_mv_bf16_b256_n<NC>` (`ffn_gate_inp_shexp`, n_out 1) | 5.10 |
| 9 | Routed experts: `strata_moe_plan`, `strata_moe_gu_<fmt>`, `strata_moe_swiglu`, `strata_q_q8_1`, `strata_moe_down_<fmt>` | 5.10 |
| 10 | `strata_moe_combine` (with the shared gate and scale) → `bo` | 5.10 |
| 11 | FFN write (below) | 5.4 |
| | **After layer 47:** | |
| 12 | Head mixer: `strata_rms_norm_w` (`output_hc_norm`), `strata_mv_bf16_b256_n<NC>` (`output_hc_down`, 10240 → 320), `strata_gr_down_silu`, `strata_mv_bf16_b160_n<NC>` (`output_hc_up`, 320 → 10240), `strata_gr_pre_gated` → `mixed` | 5.4 |
| 13 | `strata_q_q8_1(mixed)`, `strata_mv_<fmt>_n<NC>` (`output`, 2560 → 248,320) → logits | 5.8 |
| 14 | Sampler | 5.12 |

- Upstream quantises `mixed` twice per layer: once in `shared_expert_multi` (`native_quantize_q8_1`) and once for
  the experts (`quantize_q8_1_rows`, `verify.cpp:717`). The first copy uses fast-math division; the second does
  not. The Mac quantises once (step 7) with one correctly rounded quantiser. For ordinary inputs where the precise
  CUDA conversion is defined, this is bitwise equal to `quantize_q8_1_rows`; denormal scales use the deterministic
  behavior in section 5.8. Using the precise quantiser for the shared expert, dense projections and head can differ
  from their fast-math CUDA outputs near half-integer boundaries.
- `pending(l)` is 1 when layer l − 1's FFN write has not been applied yet:
  - `pending(0) = 0`;
  - `pending(1) = !ple_on`;
  - `pending(l) = !covers(l − 1)` for l ≥ 2.
- When the full-mode layer-0 MoE coordinator leaves its result pending,
  `encode_window_moe_ple_entry` binds the PLE entry's `previous_block_out` to
  `ffn.shared.output`, `previous_inject` to the layer-0 FFN read's `inject_out`, and requires
  `ple.post.residual` to be the same view as that read's `residual`. The no-PLE layer-1 attention
  read uses those same output and injection views as `block_out` and `inject_prev` with
  `apply = true`. The Window scheduler must encode exactly the selected consumer for the window.
- The FFN write of layer l (step 11):
  - l = 47: `strata_gr_post`, then `strata_gr_cvec` with `write = 0` if `covers(47)`;
  - l < 47 and `covers(l)`: `strata_gr_cvec` with `write = 1`, which performs the write and then steers;
  - otherwise nothing. The write stays pending, and layer l + 1 folds it in step 2 (l = 0 with the PLE on) or in
    step 3.
- `covers(l)` means a control vector is loaded with a non-zero scale for layer l, for 1 ≤ l < 48 (section 5.4). It is
  known when the window is encoded. Whether the vector is on is a device word that the kernel reads, as upstream.
- `ple_on` is false only with `--no-ple` ([06](06-engine-host-port.md), section 3). Without the PLE, layer 1's
  attention read folds layer 0's write like any other layer's.
- The window writes the KV cells of its T rows, the indexer tail (after `strata_idx_snapshot`), the logits and the
  arena. Unless it is write-through (section 8.5), it never writes the GDN state, the conv history or the PLE
  history.
- Commit runs in its own command buffer (section 8). It replays the kept rows into the GDN state
  (`strata_gdn_conv_commit` and `strata_gdn_step_norm`, section 5.5). It restores the indexer tail and replays the
  appends (section 5.6). It rebuilds the PLE history with `strata_ple_history` (section 5.3).

### 5.2 Embedding and stream broadcast

Upstream: `iq_embed_rows` (`iq_kernels.cu:1384`), with grid (n_embd / 256, n_tok) × 32, then `broadcast_streams`.

**`strata_embed_rows_<fmt>`**, grid (10, n_tok) × 256, one value per thread:

```c++
t = tg.y;  b = tg.x;  v = tid;                  // super-block b, value v of it; e = 256·b + v
token = tokens[t];
if ((token & 0x80000000u) && !tokens_are_gpu_written) return; // image row
if (token >= n_vocab) { emb[t][256*b + v] = quiet_NaN; return; }
row = table + tokens[t] * row_bytes;            // the row is read in place from the GGUF window
emb[t][256*b + v] = decode_<fmt>(row + b * block_bytes, v);
```

`tokens` is a `device const uint*`; `n_vocab` and `tokens_are_gpu_written`
are inline constants. A window binds the parameter buffer at the offset of
`WindowParams::tokens`; prefill binds its token array, in which bit 31 marks an
image row (section 6.7). A window never sets that bit. Host-visible tokens are
checked before dispatch. An ordered caller may explicitly mark a token view as
GPU-written when an earlier dispatch in the same command buffer supplies it;
that path never reads the stale CPU mapping and writes a quiet NaN for an
out-of-vocabulary value instead of indexing past the embedding table.

| Format | Block (bytes) | Row bytes |
| --- | --- | ---: |
| `q3_k` | `uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d;` (110) | 1,100 |
| `iq3_s` | `half d; uchar qs[64]; uchar qh[8]; uchar signs[32]; uchar scales[4];` (110) | 1,100 |
| `iq4_xs` | `half d; ushort scales_h; uchar scales_l[4]; uchar qs[128];` (136) | 1,360 |
| `bf16` | none | 5,120 |

```c++
// q3_k: value v = 128n + 32j + l
n = v / 128;  j = (v % 128) / 32;  l = v % 32;
is = 8*n + 2*j + l / 16;  shift = 2*j;  m = 1 << (4*n + j);
us = is <  4 ? (sc[is]   & 0xF) | ((sc[is+8]        & 3) << 4)
   : is <  8 ? (sc[is]   & 0xF) | (((sc[is+4] >> 2) & 3) << 4)
   : is < 12 ? (sc[is-8] >> 4)  | (((sc[is]   >> 4) & 3) << 4)
   :           (sc[is-8] >> 4)  | (((sc[is-4] >> 6) & 3) << 4);
y = (float(d) * float(us - 32)) * float(int((qs[32*n + l] >> shift) & 3) - ((hmask[l] & m) ? 0 : 4));

// iq3_s: value v = 32ib + 8il + 4g + j
ib = v / 32;  il = (v % 32) / 8;  pos = v % 8;  g = pos / 4;  j = pos % 4;
idx = qs[8*ib + 2*il + g] | ((qh[ib] << (8 - 2*il - g)) & 256);
dl  = float(d) * float(1 + 2*((scales[ib/2] >> (4*(ib%2))) & 0xf));
y   = dl * float((iq3s_grid[idx] >> (8*j)) & 0xff) * ((signs[4*ib + il] & kmask_iq2xs[pos]) ? -1.0f : 1.0f);

// iq4_xs: value v = 32ib + e
ib = v / 32;  e = v % 32;
nib = e < 16 ? (qs[16*ib + e] & 0xf) : (qs[16*ib + e - 16] >> 4);
ls  = ((scales_l[ib/2] >> (4*(ib%2))) & 0xf) | (((scales_h >> (2*ib)) & 3) << 4);
y   = (float(d) * float(ls - 32)) * float(kvalues_iq4nl[nib]);

// bf16
y = bf16_to_f32(row[v + 256*b]);
```

- These are ggml-cuda's `dequantize_block_q3_K`, `_iq3_s` and `_iq4_xs` with one value per thread.
- Every product is exact. Each factor is a half or an integer of at most 7 bits, and every product fits in 24 bits.
  So the order of the multiplications does not matter, and the kernel is bitwise equal to upstream and to the
  oracle (contract B).
- `iq3s_grid` entries are read as four little-endian bytes.

**`strata_gr_broadcast`**, grid (40, n_tok) × 256: `i = 256·tg.x + tid; R[t][i] = emb[t][i % 2560]`. In prefill,
the rows of an image's `<|image_pad|>` tokens are replaced by the encoder's rows before the broadcast
(`strata_embed_image`, section 6.7; [03](03-model-reference.md), section 6.11).

### 5.3 PLE

Upstream: `ple_block` with the native postops (`ple.cu:306-346`, `native_ple_postops.cu`,
[K4 section 1.6](audit/K4-ple-mtp-head-sampler.md)). Constants: N = 2560, H = 4 streams, D = 10240, HISTORY = 9,
eps = 1e-6.

**Inputs and buffers**
- `ple_in[T][2560]`: the host-decoded rows, F32, in a Shared buffer ([06](06-engine-host-port.md), section 7.4).
- R: at layer 1, after `strata_gr_post` has applied layer 0's FFN write.
- `hist`: section 4.8.
- Weights of `blk.1`:
  - `ple_key` [2560 → 10240]: Q2_0 in the Q2_0 file, BF16 in the IQ files. Q8_0 is also served natively
    ([03](03-model-reference.md), section 3.8).
  - `ple_value`: BF16 [2560 → 2560].
  - `ple_norm_key`, `ple_norm_query` and `ple_norm_conv`: F32[10240], one 2560-value gamma row per stream.
  - `ple_conv1d`: F16, with tap k of channel ch at `ch·4 + k`.
- Arena regions: `key[T][10240]`, `value[T][2560]`, `query[T][10240]`, `gate[T][4]`, `gated[T][10240]` and
  `ple_norm[T][10240]` (section 4.4). `ple_norm` must survive until the commit that follows the window.

**Sequence**

| # | Kernel | Computes |
| --- | --- | --- |
| 1 | `strata_q_q8_1(ple_in)`, then `strata_mv_q2_0_n<NC>` (or `_q8_0`). For a BF16 key: `strata_mv_bf16_b256_n<NC>`. | `key = ple_key · ple_in` |
| 2 | `strata_mv_bf16_b256_n<NC>` | `value = ple_value · ple_in`. The activation is the F32 `ple_in`, not rounded to BF16 (upstream `ple.cu:322-324` under `--native-bf16`, which `--native` sets). |
| 3 | `strata_rms_norm_w`, T·4 rows, gamma `ple_norm_key` | key, normalised in place |
| 4 | `strata_rms_norm_w`, T·4 rows of R, gamma `ple_norm_query` | `query` |
| 5 | `strata_ple_gate` | `gate[T][4]` |
| 6 | `strata_ple_broadcast` | `gated` |
| 7 | `strata_rms_norm_w`, T·4 rows, gamma `ple_norm_conv` | `ple_norm` |
| 8 | `strata_ple_conv_residual` | R, in place |

`strata_rms_norm_w` is specified in section 5.4.

`metal_ple_test` composes the three weighted norms, gate, broadcast and
convolution-residual for T=1, 2 and 8. Its explicit history policy advances all
rows for Prefill and write-through, or defers history mutation for a
rollback-capable Window. The deferred path leaves committed history bitwise
unchanged; a later commit advances only its accepted `n_keep` prefix. The
distinct-buffer reference chain runs through `encode_ple_post_projection` in a
caller-owned `Encoder`, with each downstream oracle consuming the preceding
GPU output. An instrumented chain reuses `query` as `ple_norm` and snapshots
the query norm before overwrite; its results are checked bitwise against the
reference. The coordinator also succeeds with that exact scratch alias through
its standalone command-buffer wrapper. The FullMode T=1 PLE handoff resolves
the key, value, gamma and convolution tensors through `FullModeStorage` and
passes their checked `ReadOnlyBufferRef` views into the block. Its integration
fixture clears the caller-provided model views and deliberately declares the
wrong key format, then verifies that the step resolves the mapped tensor
metadata and bindings. The sparse GGUF fixture uses deterministic values in the
production types and shapes; it tests the mapped-view path but does not provide
real-model parity.

**`strata_ple_gate`**, grid (T·4) × 512. Threadgroup r handles row `t = r / 4`, stream `c = r % 4`. `scale` is the host
constant `1.0f / std::sqrt(2560.0f)`, passed with `setBytes`.

```c++
float sums[8];
for (j = 0; j < 8; ++j) { d = tid + 512*j; sums[j] = d < 2560 ? key[t][c*2560 + d] * query[t][c*2560 + d] : 0.0f; }
sum = 0.0f;  for (j = 0; j < 8; ++j) sum = sum + sums[j];          // serial; j = 5..7 add zeros
sum = butterfly(sum);  if (lane == 0) partials[warp] = sum;        // 16 simdgroups
barrier;
if (warp == 0) {
    v = lane < 16 ? partials[lane] : 0.0f;  v = butterfly(v);
    if (lane == 0) {
        s    = fma(scale, v, 0.0f);
        mag  = sqrt(fmax(fabs(s), 1e-6f));
        sign = float((s > 0.0f) - (s < 0.0f));
        gate[t][c] = sigmoid_k(sign * mag);
    }
}
```

The products are plain products, as upstream's `__fmul_rn`.

**`strata_ple_broadcast`**, grid (40, T) × 256:
`i = 256·tg.x + tid; gated[t][i] = value[t][i % 2560] * gate[t][i / 2560]`.

**`strata_ple_conv_residual`**, grid (40, T) × 256, one channel `ch = 256·tg.x + tid` of row t per thread:

```c++
for (k = 0; k < 4; ++k) {
    p    = t - 9 + 3*k;                                      // the tap's row; p < 0 is history
    x    = p >= 0 ? ple_norm[p][ch] : hist[ch*9 + 9 + p];
    term = x * float(conv1d[ch*4 + k]);
    sum  = k == 0 ? term : sum + term;                       // no FMA (upstream __fmul_rn / __fadd_rn)
}
act = silu_k(sum);
R[t][ch] = R[t][ch] + (gated[t][ch] + act);
```

- Upstream runs the block once per row and advances the live history after each row (`ple_history_advance`), so
  row t sees the history advanced t times. Its row r is `hist[ch·9 + r + t]` while `r + t < 9`, and
  `ple_norm[r + t − 9]` after that. The taps are rows 0, 3 and 6 of that history plus the row's own `ple_norm`.
  With r = 3k this is the formula above. So the batch kernel reads exactly the values the sequential form reads,
  and computes the same bits. The test `ple_batch_eq_sequential` checks this (section 10.4).
- A window never writes `hist`, unless it is write-through (section 8.5). Upstream copies the whole history after every row (`hist_snap_`, 2.95 MB at T = 8)
  and copies one snapshot back at commit. The Mac rebuilds the history at commit from `ple_norm`, which the window
  keeps anyway.

**`strata_ple_history`**, grid 40 × 256, one channel per thread. n is `WindowParams.n_keep` at commit, or the chunk
length in prefill (from `PrefillParams`):

```c++
for (r = 0; r < 9; ++r) { p = n - 9 + r; h[r] = p >= 0 ? ple_norm[p][ch] : hist[ch*9 + n + r]; }
for (r = 0; r < 9; ++r) hist[ch*9 + r] = h[r];               // all reads before any write
```

This is upstream's `history_batch` (`native_ple_postops.cu:205-220`). Its result equals n sequential advances
bitwise, because it only copies values. Prefill computes the key and the value with its own GEMMs, then runs steps
3-8 over the chunk and `strata_ple_history` with n = chunk length (section 6.7).

`encode_ple_post_projection` composes the post-projection sequence for T=1..8:
the three weighted norms, gate, broadcast and convolution-residual update. Its
inputs are the already projected `key` and `value` plus `R`; projection GEMMs
remain the caller's responsibility. The caller must explicitly choose its
`PLEHistoryUpdate` policy. `advance_rows` also advances all T history rows and
is for Prefill chunks and write-through windows. `defer` leaves history
unchanged and is for rollback-capable windows; their commit separately calls
`encode_ple_history` with `n_keep` and the retained `ple_norm` rows. Advancing
all speculative rows would corrupt committed session state. `query` and
`ple_norm` may be disjoint or exact aliases, matching the documented scratch
lifetime; every other buffer role must be disjoint. The shared-Encoder overload
can return false after earlier dispatches have been encoded, so callers must
propagate failure from the enclosing `Encoder::encode` body. The
command-buffer overload discards on failure.

`encode_ple_block` composes the full decode sequence in one caller-owned
segment. It accepts Q2_0, Q8_0 or BF16 key weights, quantizes `ple_in` to Q8_1
only for the quantized key formats, and always feeds the original F32
`ple_in` to the BF16 value projection. It then invokes the post-projection
coordinator with the caller's explicit history policy. The key and value
projections reuse one mat-vec sink serially. The coordinator rejects writable
cross-stage aliases before encoding and permits read-only overlap plus the
documented exact query/`ple_norm` scratch alias. Synthetic fixtures exercise
all three key formats, padded compiled columns, deferred history and
late-failure discard. Production Window and Prefill callers remain open.

**Deviations** (section 3.9): `exp`, `sqrt`, `rsqrt` and `/` are correctly rounded or ≤ 1 ulp where upstream's
fast-math build uses `__expf`, approximate square roots and approximate division.

**Without the PLE** (`--no-ple`), none of this runs. Layer 1 folds layer 0's write in its attention read
(section 5.1).

### 5.4 GR hyper-connections

Upstream: `fused_gr_read_multi` (`fused_gr.cu`, [K1 section 3.6](audit/K1-gr-router-shared-rope-cvec.md)) for the
attention and FFN reads, and the native op-level read (`native_gr_norm.cu`, `native_gr_postops.cu`, `native_bf16.cu`,
[K1 section 3.4](audit/K1-gr-router-shared-rope-cvec.md)) for the head. Constants: N = 2560, HC = 4, D = 10240,
LR = 320.

Weights per read (`hc_attn_*` or `hc_ffn_*` of the layer):
- `g = hc_*_norm`: F32[10240], used as stored ([03](03-model-reference.md), section 3.7).
- `hc_*_down`: BF16, 320 rows of 10240.
- `hc_*_inject`: BF16, 4 rows of 10240.
- `hc_*_up`: BF16, 10240 rows of 320.

`dot8(w, x)` is the 8-element BF16 chunk, the same in every GR kernel:

```c++
static inline float dot8(uint4 w, threadgroup const float* x) {
    float acc = 0.0f;
    for (int j = 0; j < 4; ++j) {
        acc = fma(as_type<float>(w[j] << 16),         x[2*j],     acc);   // low half first
        acc = fma(as_type<float>(w[j] & 0xffff0000u), x[2*j + 1], acc);
    }
    return acc;
}
```

**The write gate.** The GR write `R += bo · 2·sigmoid(inject/4)` has three upstream flavours, and upstream's
flavours are not interchangeable bitwise ([K1 section 3.8](audit/K1-gr-router-shared-rope-cvec.md)):

| Flavour | Upstream | Mac kernel |
| --- | --- | --- |
| Canonical (`gr.cu`) | not on the production path | not ported |
| Native `post`: `fma(bo, fma(2, sigmoid(fma(0.25, inj, 0)), 0), R)` | layer 0's write at layer 1 (PLE on), layer 47's write | `strata_gr_post` |
| Fused: `fma(bo, 2·sigmoidf_(inj/4), R)` | the fold inside the next read, and the control-vector write | `strata_gr_norm`, `strata_gr_up`, `strata_gr_cvec` |

On the Mac every sigmoid is `sigmoid_k`. `fma(0.25, x, 0)` and `x / 4` are the same correctly rounded value, and so
are `fma(2, s, 0)` and `2·s`. So the native and fused flavours give the same bits on the Mac. Each kernel is still
tested against an oracle of its own upstream flavour.

**`strata_gr_norm`**, grid (T, 4) × 256. Upstream uses one block per row. On the Mac, threadgroup (t, c) handles
stream c of row t:

The Metal ABI binds `R[T][10240]`, `bo[T][2560]`, `g[10240]`, and `inj_prev[T][4]` at buffers 0–3, then writes
`xn[T][10240]` and `rs_out[T][4]` at buffers 4–5. Buffer 6 is a `uint32_t apply` value passed with `setBytes`;
`group.x` selects row t and `group.y` selects stream c, matching the `(T, 4)` grid. `bo` and `inj_prev` are not read when
`apply` is zero. The host validates the grid and byte lengths before dispatch; the input and output buffers are disjoint.

`encode_gr_norm` in `include/strata/metal/gr_dispatch.hpp` checks T=1..8, the manifest's 256-thread geometry,
all six buffer ranges and alignments, pairwise disjointness and command-buffer state before binding this ABI.
It rejects buffers and command buffers from a different Context. Window and Prefill do not call it yet.

```c++
gw = apply ? 2.0f * sigmoid_k(inj_prev[t][c] / 4.0f) : 0.0f;
ss = 0.0f;
for (k = 0; k < 10; ++k) {                       // float4 index q = tid + 256k, ascending
    q = tid + 256*k;  if (q / 640 != c) continue;    // only the float4s of stream c
    r = R4[t][q];                                    // 4 values, i = 4q
    if (apply) r = fma(bo4[t][q % 640], float4(gw), r);
    sq = fma(r.w, r.w, fma(r.z, r.z, fma(r.y, r.y, r.x * r.x)));
    ss = ss + sq;
    xn4[t][q] = r * g4[q];
}
ss = butterfly(ss);  if (lane == 0) part[warp] = ss;
barrier;
s = 0.0f;  for (w = 0; w < 8; ++w) s = s + part[w];     // every thread, serial
rs = rsqrt(s / 2560.0f + 1e-6f);
if (tid == 0) rs_out[t][c] = rs;
for (each q handled above) xn4[t][q] = xn4[t][q] * rs;
```

Upstream's thread `tid` visits the same float4s in the same order and keeps one partial per stream. So the
(t, c) layout performs the same additions per thread, and the butterfly and the serial combine are those of
upstream. The result is bitwise equal. `xn = (r·g)·rs` in that order. The kernel does not write R; `strata_gr_up`
does.

**`strata_gr_down`**, grid 41 × 256. Threadgroup b < 40 computes `lo` rows `8b + warp`. Threadgroup 40 computes the
four inject rows, one per simdgroup 0-3; simdgroups 4-7 only stage the tiles.

The Metal ABI binds `xn[T][10240]`, `hc_down[320][10240]`, and
`hc_inject[4][10240]` as BF16 at buffers 0–2, then writes `lo[T][320]` and
`inj_out[T][4]` at buffers 3–4. Buffer 5 is the `uint32_t T` passed with
`setBytes`. Threadgroup memory 0 is `T × 512` F32 values, for a size of
`T * 512 * sizeof(float)` bytes; `T` must be 1..8. `TILEV` is 512 in this
implementation. All simdgroups stage the input tile at each barrier;
simdgroups 4-7 in the injection group do not read weights or write outputs.

```c++
W = tg.x == 40 ? hc_inject : hc_down;   row = tg.x == 40 ? warp : 8*tg.x + warp;
active = tg.x < 40 || warp < 4;
w4 = (device const uint4*)(W + row * 10240);              // 1,280 chunks of 8
acc[k] = 0 for k < T;
for (base = 0; base < 10240; base += TILEV) {
    if (active) for (q = 0; q < TILEV/256; ++q) wv[q] = w4[base/8 + lane + 32*q];
    barrier;  tile[k][0 .. TILEV) = xn[t = k][base .. base + TILEV) for k < T;  barrier;   // all 256 threads stage
    if (active) for (q ...) for (k < T) acc[k] = acc[k] + dot8(wv[q], &tile[k][(lane + 32*q) * 8]);
}
if (active) for (k < T) {
    s = butterfly(acc[k]);
    if (lane == k) {
        if (tg.x == 40) inj_out[k][row] = s;
        else { x = s / 4.0f; lo[k][row] = silu_k(x); }
    }
}
```

- Lane `lane` adds the chunks `lane + 32m` for m = 0..39 in ascending order whatever TILEV is. So TILEV does not
  change the bits.
- TILEV is 512 (`tile` is 16 KiB at T = 8). It may be 256 or 1024 (32 KiB at T = 8, the whole budget). L-2 measures
  the three; a value that wins by ≥ 5% on a family becomes a specialisation `strata_gr_down_tv<TILEV>`.

`encode_gr_down` in `include/strata/metal/gr_dispatch.hpp` validates T, the manifest's 256-thread geometry,
all five buffer ranges and pairwise disjointness, four-byte float views and 16-byte BF16 weight offsets, plus the
`T × 512 × sizeof(float)` threadgroup-memory size against the device limit before dispatching 41 groups. Standalone
and caller-owned `Encoder` forms share these checks; the shared form binds T at slot 5 and dynamic memory at
threadgroup-memory slot 0 through the manifest-validated dispatch facade. Propagate a failed call from the
enclosing `Encoder::encode` body so earlier dispatches are discarded. Window and Prefill do not call it yet.
It rejects views and command buffers owned by another Context.

**`strata_gr_up`**, grid 160 × 256, 16 output columns `d0 = 16·tg.x` of each stream per threadgroup. Threadgroup
memory: `lo_s[8][320]` and `g_s[8][4][16]` (12 KiB).

The Metal ABI binds in-place `R[T][10240]`, `bo[T][2560]`, `inj_prev[T][4]`,
`g[10240]`, `rs[T][4]`, `lo[T][320]`, BF16 `hc_up[10240][320]`, and
`mixed[T][2560]` at buffers 0–7. Buffer 8 is a two-word `GRUpParams`
(`uint32_t token_count`, `uint32_t apply`) passed with `setBytes`; T must be
1..8 and `apply` must be 0 or 1. Each group covers 16 adjacent output columns
across all four streams. With `apply = 0`, R is left untouched; with
`apply = 1`, the pending write is folded into R in place before reconstructing
`xn = (R * g) * rs`. The host validates the fixed 160-group grid, parameter
values, buffer lengths and read/write overlap before dispatch.

```c++
lo_s[k][j] = lo[k][j] for k < T;  barrier;
for (r = warp; r < 64; r += 8) {
    c = r / 16;  dd = r % 16;  i = c*2560 + d0 + dd;
    w4 = (device const uint4*)(hc_up + i * 320);                 // 40 chunks
    wa = w4[lane];  wb = lane < 8 ? w4[32 + lane] : uint4(0);
    for (k < T) {
        a = dot8(wa, &lo_s[k][lane * 8]);
        if (lane < 8) a = a + dot8(wb, &lo_s[k][(32 + lane) * 8]);
        a = butterfly(a);
        if (lane == k) mine = a;
    }
    if (lane < T) {                                               // lane plays row t = lane
        rv = R[lane][i];
        if (apply) { rv = fma(bo[lane][d0 + dd], 2.0f * sigmoid_k(inj_prev[lane][c] / 4.0f), rv); R[lane][i] = rv; }
        x = (rv * g[i]) * rs[lane][c];                            // equals xn[lane][i] bitwise
        g_s[lane][c][dd] = x * sigmoid_k(mine);
    }
}
barrier;
if (tid < 16*T) { k = tid / 16;  col = tid % 16;
    mixed[k][d0 + col] = ((((0.0f + g_s[k][0][col]) + g_s[k][1][col]) + g_s[k][2][col]) + g_s[k][3][col]) / 4.0f; }
```

`strata_gr_up` is the only GR read kernel that writes R. With `apply`, it recomputes the fold that `strata_gr_norm`
computed and stores it.

`encode_gr_up` in `include/strata/metal/gr_dispatch.hpp` validates T, the 160-group manifest geometry,
the two-word T/apply ABI, the eight buffer ranges and pairwise disjointness, 16-byte BF16 weight alignment, static
threadgroup-memory limits and Context ownership before dispatch. Standalone command-buffer and caller-owned
`Encoder` forms share these checks; the latter binds the parameter block at slot 8 through the manifest-validated
dispatch facade. A caller must propagate a `false` result from its enclosing `Encoder::encode` body so earlier
dispatches in the same segment are discarded.

`encode_gr_read` composes `strata_gr_norm`, `strata_gr_down`, and `strata_gr_up` in that order for T=1..8. It wires
`normalized` and `rs` from norm to the later stages, `lo` from down to up, and returns the computed `inject_out` to
the caller for the next read. All twelve supplied buffer views must be pairwise disjoint, including read-only roles;
this keeps the in-place residual fold and the old/new injection values separate across dispatches. The coordinator
rejects an open concurrent dispatch group because automatic buffer barriers are suppressed inside one. Its
caller-owned `Encoder` form inserts the dependencies between dispatches when called outside a group. The
`CommandBuffer` form wraps the whole chain in one `Encoder::encode` segment, so a late stage validation failure
discards earlier dispatches. In a larger caller-owned segment, the caller must propagate `false` from the enclosing
body to get the same rollback. `metal_gr_test` exercises T=1 without a pending write and T=8 with a write, checks
all stage outputs against their scalar references, rejects cross-stage aliases and an open concurrent group, and
verifies late-failure discard. The T=1 layer-0 Window coordinator uses this ordered GR read for both attention and FFN, with distinct `hc_attn_*` and `hc_ffn_*` tensors and shared activation scratch; Prefill remains unconnected.

**`strata_gr_post`**, grid (40, T) × 256. This is the native `post`.

```c++
i = 256*tg.x + tid;  c = i / 2560;  d = i % 2560;
weight = fma(2.0f, sigmoid_k(fma(0.25f, inj[t][c], 0.0f)), 0.0f);
R[t][i] = fma(bo[t][d], weight, R[t][i]);
```

`encode_gr_post` in `include/strata/metal/gr_dispatch.hpp` validates T, the 40-group manifest geometry, all four buffer
ranges and Context ownership before encoding. Residual and output may be the exact same view for the in-place write;
partial overlap and every other overlap are rejected. The command buffer must belong to the same Context and remain
unsubmitted. Window and Prefill do not call this encoder yet.

**`strata_rms_norm_w`**, grid rows × 1024. Upstream: `native_gr_rms_norm_weighted` (`native_gr_norm.cu:85`), which
is llama.cpp's `rms_norm_f32<1024>` with the multiply fused, and `native_qsa_rms_norm_weighted` (`native_qsa.cu:108`)
for rows of 1024 values or more, whose `norm<1024>` is the same arithmetic. Row r has `n_cols` values and its gamma
row is `w + (r % gp)·n_cols`. `n_cols` and the gamma period `gp` are `setBytes` constants.

```c++
partial = 0.0f;
for (col = tid; col < n_cols; col += 1024) partial = fma(in[col], in[col], partial);
partial = butterfly(partial);  if (lane == 0) sums[warp] = partial;     // 32 simdgroups
barrier;
partial = butterfly(sums[lane]);                                         // every simdgroup, so every thread holds it
mean  = partial / float(n_cols);
scale = rsqrt(mean + eps);
for (col = tid; col < n_cols; col += 1024) out[col] = (scale * in[col]) * gamma[col];
```

| Use | Rows | `n_cols` | `gp` |
| --- | --- | ---: | ---: |
| The PLE's three norms (section 5.3), the head mixer, the MTP final mixer | T·4 | 2560 | 4 |
| MTP stage 2, `pre_fc_norm_embedding` (section 5.11) | T | 2560 | 1 |
| MTP stage 4, `pre_fc_norm_hidden` | T | 10240 | 1 |

Upstream's 256-thread forms are used only for rows shorter than 1024, which none of these are, so the Mac has only
the 1024-thread form. The drafter's 256-wide K and Q norms are `strata_qsa_norm_rope` (section 5.6).

The twin compares against the scalar reduction-order oracle with **U(64)** (section 10.2). The reference M5 Pro
matches that oracle bitwise on the current test cases, but the tolerance keeps the test valid across supported
Apple GPU generations. The M5 Pro's safe-mode `rsqrt` also matched `1/sqrt` rounded from double on all 4,194,304
measured inputs ([A.10](appendix-a-measurements.md), [spike results](spikes/metal-numerics/results.txt)); that
measurement is not a cross-generation bitwise guarantee. CUDA's approximate `rsqrtf` is a separate reference and
must use its own tolerance; Mac inference remains Metal-only.

Both `encode_weighted_rms_norm` overloads validate positive rows, the documented `(n_cols, gp)` pairs
`(2560, 1)`, `(2560, 4)` and `(10240, 1)`, finite positive epsilon, parameter and buffer sizes, float alignment,
allocation ranges, Context ownership and the 1024-thread manifest geometry. The `Encoder&` overload appends one
dispatch to a caller-owned compute segment; the standalone `CommandBuffer&` overload routes through
`Encoder::encode`. Exact input/output view alias is supported for in-place normalization; partial input/output and
every output/gamma overlap are rejected. The read-only input and gamma views may overlap. The FullMode T=1 PLE
branch uses the read-only-gamma overload; other Window, Prefill and MTP call sites remain pending.

**`strata_gr_down_silu`**, grid ⌈n_values / 256⌉ × 256, in place on `lo[T][320]`, with `n_values = 320·T`
passed as a `setBytes` constant:
`j = 256·tg.x + tid; if (j < n_values) { x = fma(0.25f, lo[j], 0.0f); lo[j] = silu_k(x); }`.

`encode_gr_down_silu` validates T=1..8, derives `n_values = T * HC_LOW_RANK`, checks the 256-thread manifest
geometry and the full in-place float view range, alignment and Context ownership, then binds `n_values` at buffer 1
and dispatches ⌈n_values/256⌉ groups. Its caller-owned `Encoder` overload applies the same validation and
manifest-checked bindings. The Window head-mixer composition below calls this stage after its down projection. In
either form, the caller's command buffer must belong to the same Context and remain unsubmitted.

**`strata_gr_pre_gated`**, grid (10, T) × 256. This is the unfused form (upstream `pre_gated<false>`), the only one
the Mac needs: the head has no inject. `xn` is the output of `strata_rms_norm_w` and `gate` is the up mat-vec's
output.

```c++
d = 256*tg.x + tid;
for (c = 0; c < 4; ++c) {
    i = c*2560 + d;
    product = xn[t][i] * sigmoid_k(gate[t][i]);
    sum = c == 0 ? product : sum + product;
}
mixed[t][d] = fma(0.25f, sum, 0.0f);
```

Upstream also stores `product` back into `gate`. Nothing reads it, so the Mac does not.
The Metal kernel uses the fixed `(10, T)` grid and does not accept a runtime
shape parameter; the host must bind complete `T × 10240` inputs and a
`T × 2560` output. The four products are accumulated in stream order, then
scaled with the documented fused operation.

`encode_gr_pre_gated` validates T, the fixed `(10, T) × 256` dispatch, all three buffer ranges, float alignment and
Context ownership. The read-only `xn` and `gate` views may overlap; the writable `mixed` view must be disjoint from
both. Its caller-owned `Encoder` overload uses the same checks and manifest-validated dispatch. The command buffer
must belong to the same Context and remain unsubmitted. The Window head-mixer composition below calls this stage;
production `Window::run` remains unimplemented.

**`strata_gr_cvec`** (O-class), grid (4, T) × 256, threadgroup (c, t). Upstream: `cvec.cu`
([K1 section 7](audit/K1-gr-router-shared-rope-cvec.md)).
- The host builds, per layer, `v[l][2560]` (the scaled direction in add mode, the unit direction in project mode) and
  `s_l[l]` (0 where the layer has no vector).
- `mode` (0 project, 1 add) and `write` are `setBytes` constants. `on` is a device word.

```c++
steer = on[0] != 0 && s_l[layer] != 0.0f;
if (!steer && !write) return;
w = write ? 2.0f * sigmoid_k(inj[t][c] / 4.0f) : 0.0f;
dot = 0.0f;
for (k = 0; k < 10; ++k) { d = tid + 256*k;
    xv = R[t][c][d];  if (write) xv = fma(bo[t][d], w, xv);  x[k] = xv;
    if (steer && mode == 0) dot = fma(xv, v[layer][d], dot); }
if (steer && mode == 0) {
    dot = butterfly(dot);  if (lane == 0) part[warp] = dot;  barrier;
    if (warp == 0) { p = lane < 8 ? part[lane] : 0.0f;  p = butterfly(p);  if (lane == 0) part[0] = p; }
    barrier;  dot = part[0] * s_l[layer];
}
for (k = 0; k < 10; ++k) { d = tid + 256*k;
    R[t][c][d] = steer ? (mode == 0 ? fma(-dot, v[layer][d], x[k]) : x[k] + v[layer][d]) : x[k]; }
```

The Metal ABI binds `R`, `v`, `s_l`, `on`, `bo` and `inj` at buffers 0–5. Buffer 6 carries four `uint32_t`
values in order: `layer`, `token_count`, `mode` (0 project, 1 add) and `write` (0 or 1). The host launches
`(4, T) × 256` with stream on x and token on y. `v` contains the unit direction in project mode and the
already-scaled direction in add mode. `s_l[layer]` scales only the projection dot. The kernel trusts the host
to provide valid T, layer, mode and buffer views; `encode_gr_cvec` below validates those inputs before dispatch.

`encode_gr_cvec` in `include/strata/metal/gr_dispatch.hpp` validates T, layer, mode, manifest geometry,
buffer sizes/alignment and pairwise overlap before encoding. It supports standalone command buffers and
caller-owned Encoder segments, and rejects buffers and command buffers from a different Context, as well as
already-submitted command buffers. The inference Window and Prefill paths do not call this encoder yet.

With the vector loaded and off, the kernel performs exactly the fused write. That write is what the next read would
have folded, so "loaded, off" ≡ stock bitwise. The test `cvec_off_eq_stock` checks this (section 10.4).

**The head mixer** (section 5.1, step 12) is upstream's native `gr_read` without inject (`lm_head_mix`):
1. `strata_rms_norm_w` of the T·4 rows of R with `output_hc_norm` → `xn`;
2. `strata_mv_bf16_b256_n<NC>` (`output_hc_down`, 10240 → 320) → `lo`;
3. `strata_gr_down_silu`;
4. `strata_mv_bf16_b160_n<NC>` (`output_hc_up`, 320 → 10240) → `gate`;
5. `strata_gr_pre_gated` → `mixed[T][2560]`.

`encode_window_head_mixer` composes these stages in this order for T=1..8. It derives the smallest supported `NC`
from T, requires the active-row word to match T, and rejects overlapping scratch outputs or scratch that aliases
model inputs; exact in-place aliasing of residual and normalized output is allowed. The `Encoder&` form must be
called outside an open concurrent group so dependent dispatches receive barriers; the `CommandBuffer&` form
discards the whole segment on failure. Callers provide the validated model-weight and scratch views and keep
read-only inputs unchanged until completion. This composition is covered by a synthetic scalar-oracle test in
serial and concurrent dispatch modes.

`encode_window_output_projection` resolves `output.weight` through the live
`FullModeStorage`, requires shape `[2560, 248320]`, and maps GGML Q5_K, Q6_K or
IQ4_XS to the matching immutable dense mat-vec. It quantizes `mixed[T][2560]`
into the dedicated `head_q8_1` scratch, then writes `logits[T][248320]` and
padded columns to `matvec_sink` in the same ordered Encoder segment. The
active-row word must equal T; writable scratch may not overlap the mixed input,
mapped model weight or row-count input. The `Encoder&` form must be called
outside an open concurrent group and its failure must escape the enclosing
transaction. The standalone form discards the command buffer on failure.
Keep the FullMode owner and every scratch view alive through completion. This
completes the low-level vocabulary projection composition; `Window::run`,
production scheduling, accepted-prefix state commit and model-backed decode
remain unimplemented.

`encode_window_output_head` resolves the live mixer tensors with exact shapes
and types: F32 `[10240]` for `output_hc_norm.weight`, BF16 `[10240, 320]` for
`output_hc_down.weight`, and BF16 `[320, 10240]` for `output_hc_up.weight`.
It composes the weighted RMS/down/SiLU/up/pre-gated mixer with the vocabulary
projection in the same ordered encoder. Named T=8 scratch keeps normalized,
low-rank and gate values in separate regions; the composite rejects residual
overlap with later Q8_1/logits writes. Its standalone form discards the whole
transaction on failure; callers using the Encoder form must propagate failure.
The sparse FullMode test validates the mixer output and logits against scalar
oracles and checks rollback when `output.weight` is missing. Production
`Window::run`, accepted-prefix state commit and shipped-model parity remain
open; sampling is available through `encode_window_output_sampling` but is not
connected to the production Window scheduler.

The MTP drafter uses `strata_gr_norm`, `strata_gr_down` and `strata_gr_up` for its two reads, and the head mixer
kernels for its final mixer (section 5.11).

### 5.5 GDN block

Upstream: `verify.cpp:509-545` (window) and `verify.cpp:972-981` (commit), with the kernels of `verify_kernels.cu`.
That file transcribes the single-token `fused_gdn.cu` kernels in the same operation order, so one kernel set serves
decode and verify, as on the Mac. Dimensions ([03](03-model-reference.md), section 6.3):

| Symbol | Value | Meaning |
| --- | --- | --- |
| C | 10,240 | conv channels: q `[0, 2048)`, k `[2048, 4096)`, v `[4096, 10240)` |
| HK | 16 | q/k heads of 128 |
| HV | 48 | v heads of 128; the q/k head of v-head h is `h % 16` |
| ZV | 6,144 | value dim, `HV · 128` |
| S | 128 | head size, and the state is `S × HV × S` |
| eps | 1e-6 | the model's RMS eps, used for both norms |

`gi` is the layer's index among the 36 GDN layers. `qkv_L[gi]`, `h_L[gi]`, `gate_L[gi]` and `beta_L[gi]` are its
regions in the arena (section 4.4), `[8][10240]`, `[8][10240]`, `[8][48]` and `[8][48]`. `z[8][6144]` and
`y[8][6144]` are shared by all layers. `state` and `hist` are the layer's region of section 4.7.

**The window, per GDN layer:**

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_q_q8_1` (n 2560) | `mixed` → `xq` |
| 2 | `strata_mv_<fmt>_n<NC>` (`attn_qkv`, 2560 → 10240) | `xq` → `qkv_L[gi]` |
| 3 | `strata_gdn_conv_l2` | `hist`, `qkv_L[gi]`, `ssm_conv1d` → `h_L[gi]` |
| 4 | `strata_gdn_ab` | `mixed`, `ssm_alpha`, `ssm_beta`, `ssm_dt.bias`, `ssm_a` → `gate_L[gi]`, `beta_L[gi]` |
| 5 | `strata_mv_<fmt>_n<NC>` (`attn_gate`, 2560 → 6144) | `xq` → `z` |
| 6 | `strata_gdn_step_norm`, mode 0 | `state`, `h_L[gi]`, `gate_L[gi]`, `beta_L[gi]`, `z`, `ssm_norm` → `y` |
| 7 | `strata_q_q8_1` (n 6144), `strata_mv_<fmt>_n<NC>` (`ssm_out`, 6144 → 2560) | `y` → `bo` |

Step 1 is the block's own quantisation of the attention read's `mixed`. It is not step 7 of section 5.1, which
quantises the FFN read's `mixed`. Step 4 reads `mixed` in F32 with the BF16 weights, as upstream.

`encode_window_gdn_block` encodes these seven table rows in one ordered segment, including both Q8_1 quantizers and
all three mat-vectors. It chooses compiled columns 1, 2, 4 or 8 from T and requires quantized weight formats for the
Q8_1 activations. Its supplied input, weight, state and scratch views must be Context-owned and pairwise disjoint;
the shared mat-vec sink and active-row metadata are reused serially. Window mode leaves both `state` and `hist`
unchanged for the later accepted-prefix commit. The caller-owned `Encoder` overload requires the caller to propagate
failure from its enclosing encode body, while the command-buffer overload discards the whole segment on a late
validation failure. `metal_gdn_test` runs the fixture chain at T=1, T=2 and T=3, including the padded-sink boundary at T=3. It compares every intermediate with its scalar oracle using the previous GPU stage's actual output, rejects a mixed/history cross-stage alias, and checks output/state guards plus both rollback paths. Its
focused concurrent CTest also checks the open-group boundary. This is a reusable block coordinator; production
Window assembly, commit scheduling and model-backed decode remain open.

**Commit, per GDN layer**, in layer order, `n = WindowParams.n_keep`:
1. `strata_gdn_conv_commit` advances `hist` by n rows of `qkv_L[gi]`;
2. `strata_gdn_step_norm` in mode 1 replays rows 0..n−1 from the untouched `state` and writes it back.

`encode_window_gdn_commit` composes these two operations for one layer and the accepted `n_keep` prefix in a
single ordered segment: convolution history first, recurrent state replay second. `encode_window_gdn_commit_batch`
binds the four fixed-capacity T=8 replay regions to the 36 `SessionStateStorage` GDN ordinals and their gamma views,
validates all extents and cross-layer aliases before encoding, then appends the per-layer commits in ordinal order.
It leaves history and state untouched for `n_keep = 0`; the batch caller-owned Encoder failure must propagate so a
later-layer validation error discards every earlier encoded layer. Its shared `z` and output scratch bindings are
required by the primitive step kernel but are not read or written in commit mode. The batch API does not submit the
command or track SessionStateStorage use; the Window owner still has to register the command-buffer use and run the
batch after successful verification and before another window reads committed state. Commit must not overwrite
`hist` before step 3 of the window has read it. That holds because commit runs in the next command buffer and after a
fence (section 8).

`encode_window_gdn_ple_commit` composes that all-layer batch with
`strata_ple_history` over the first `n_keep` normalized rows in the same ordered transaction. It rejects PLE views
that alias GDN replay/state views, exceed their active extents or violate float alignment before adding dispatches.
After the command buffer completes successfully, the session owner advances its CPU PLE token suffix with
`SessionStateStorage::commit_accepted_ple_tokens`. QSA indexer snapshot restoration/replay is still a separate
commit stage; this helper does not complete the Window state-commit protocol.

**`strata_gdn_conv_l2`**, grid (80, T) × 128, one channel per thread: `c = 128·tg.x + tid`, row `t = tg.y`.
Threadgroup b < 32 is a q or k head (b < 16 q, 16 ≤ b < 32 k) and normalises it.

```c++
// the last four of [hist[c][0], hist[c][1], hist[c][2], qkv[0][c], ..., qkv[t][c]]
for (j = 0; j < 3; ++j) { src = t + j; v[j] = src < 3 ? hist[c*3 + src] : qkv[src - 3][c]; }
x   = qkv[t][c];
sum = fma(x, w[c*4 + 3], fma(v[2], w[c*4 + 2], fma(v[1], w[c*4 + 1], v[0] * w[c*4])));
y   = silu_k(sum);
if (tg.x < 32) {
    sq = butterfly(y * y);  if (lane == 0) part[warp] = sq;  barrier;
    ss = ((part[0] + part[1]) + part[2]) + part[3];
    y  = y * rsqrt(ss + eps);                     // eps on the sum of squares
}
h[t][c] = y;
```

`w` is `ssm_conv1d` as F32 `[10240][4]`, tap 3 the newest. The branch on `tg.x` is uniform per threadgroup, so the
barrier is legal.

**`strata_gdn_conv_commit`**, grid 80 × 128, one channel per thread. The shape is free: each thread copies.

```c++
c = 128*tg.x + tid;  n = params.n_keep;  if (n == 0) return;
for (j = 0; j < 3; ++j) { src = n + j; seq[j] = src < 3 ? hist[c*3 + src] : qkv[src - 3][c]; }
for (j = 0; j < 3; ++j) hist[c*3 + j] = seq[j];   // after all three reads
```

**`strata_gdn_ab`**, grid 12 × 256. Simdgroup `warp` of threadgroup b handles row `r8 = 8b + warp` of 96: rows
0..47 are alpha rows, rows 48..95 beta rows. `wrow` is row `r8 % 48` of `ssm_alpha` or `ssm_beta`, BF16 `[48][2560]`,
read as `uint4` (eight values).

```c++
acc[0..T) = 0.0f;
for (j = lane; j < 320; j += 32) {
    uint4 wv = wrow[j];
    for (t = 0; t < T; ++t) {
        float4 xa = x[t][8j .. 8j+3], xb = x[t][8j+4 .. 8j+7];
        a = acc[t];
        a = fma(lo(wv.x), xa.x, a);  a = fma(hi(wv.x), xa.y, a);
        a = fma(lo(wv.y), xa.z, a);  a = fma(hi(wv.y), xa.w, a);
        a = fma(lo(wv.z), xb.x, a);  a = fma(hi(wv.z), xb.y, a);
        a = fma(lo(wv.w), xb.z, a);  a = fma(hi(wv.w), xb.w, a);
        acc[t] = a;
    }
}
for (t = 0; t < T; ++t) {
    a = butterfly(acc[t]);
    if (lane != 0) continue;
    h = r8 % 48;
    if (r8 >= 48) beta[t][h] = sigmoid_k(a);
    else          gate[t][h] = softplus_k(a + dt[h]) * ssm_a[h];
}
```

`lo(u) = as_type<float>(u << 16)` and `hi(u) = as_type<float>(u & 0xffff0000u)`. `a + dt[h]` is a plain add.

**`strata_gdn_step_norm`**, grid 48 × 512, one threadgroup per v-head `head = tg.x`. `col = tid % 128` is the v
column and `rg = tid / 128` the row group: thread (rg, col) holds rows `32·rg .. 32·rg + 31` of column `col` in
registers. The launch is 1-D so that `warp = tid / 32` is the simdgroup index without assumptions about how Metal
linearises a 2-D threadgroup. Threadgroup memory: `sk[128]`, `sq[128]`, `red[4][128]`, `wsum[16]` (3,136 B).

`mode` is a `setBytes` constant: 0 window, 1 commit, 2 write-through (section 8).

```c++
qh = head % 16;  n = mode == 1 ? params.n_keep : T;
emit = mode != 1;                                  // commit computes no output
stride = 48 * 128;                                 // between state rows i and i+1
device float* base = state + ((32*rg) * 48 + head) * 128 + col;
for (r = 0; r < 32; ++r) s[r] = base[r * stride];
for (t = 0; t < n; ++t) {
    barrier;                                       // the previous row is done with sk, sq, red, wsum
    if (tid < 128) { sk[tid] = h[t][2048 + qh*128 + tid];  sq[tid] = h[t][qh*128 + tid]; }
    barrier;
    g = exp(gate[t][head]);
    kv = 0.0f;
    for (r = 0; r < 32; ++r) kv = fma(s[r], sk[32*rg + r], kv);
    red[rg][col] = kv;  barrier;
    kv_col = ((red[0][col] + red[1][col]) + red[2][col]) + red[3][col];
    delta  = fma(-g, kv_col, h[t][4096 + head*128 + col]) * beta[t][head];
    o = 0.0f;
    for (r = 0; r < 32; ++r) {
        s[r] = fma(g, s[r], sk[32*rg + r] * delta);
        o    = fma(s[r], sq[32*rg + r], o);
    }
    barrier;
    red[rg][col] = o;  barrier;
    oc = 0.0f;  sq_part = 0.0f;
    if (rg == 0) {
        oc = (((red[0][col] + red[1][col]) + red[2][col]) + red[3][col]) * rsqrt(128.0f);
        sq_part = oc * oc;
    }
    if (!emit) continue;                           // uniform; skips the norm's barrier for every thread
    sq_part = butterfly(sq_part);
    if (lane == 0) wsum[warp] = sq_part;
    barrier;
    if (rg == 0) {
        ss    = ((wsum[0] + wsum[1]) + wsum[2]) + wsum[3];   // warps 0-3 are rg 0; 4-15 hold zeros
        scale = rsqrt(ss / 128.0f + eps);                    // eps on the mean
        y[t][head*128 + col] = ((oc * scale) * gamma[col]) * sigmoid_k(z[t][head*128 + col]);
    }
}
if (mode != 0 && n > 0) for (r = 0; r < 32; ++r) base[r * stride] = s[r];
```

- **Bits.** The row partition, the ascending r loops, the serial `red` and `wsum` sums and the FMA forms are
  upstream's. Upstream's `delta = (v - g*kv_col) * beta` is contracted by nvcc (section 3.3, step 4).
  `ss / 128.0f` is exact, so it does not matter whether a compiler turns it into a multiply and fuses the `+ eps`.
- **Deviations** (section 3.9): `exp` for `__expf` in g, SiLU and both sigmoids; `rsqrt` for `rsqrtf`, including
  the constant `rsqrt(128.0f)`; `log1p_k` in `softplus_k`.
- **Commit skips the output.** Upstream's commit computes y into a dummy buffer (`y_dummy_`) with `t_out_begin = 0`.
  The state update does not depend on y, so mode 1 skips it. The state bits are those of the window's mode 0 over
  the same rows. `state_after_commit_eq_decode` (section 10.4) checks this: replaying n kept rows gives the same
  state bits as n decode windows of T = 1 each followed by a commit of 1.
- **No partial replay.** Upstream's `t_out_begin` serves its grouped verify, where a later group replays the earlier
  groups' rows without output. The Mac encodes the window as one group (section 8), so the window's mode 0 emits
  every row and `t_out_begin` does not exist.
- **Registers.** 32 state floats plus about 16 more per thread at 512 threads. The pipeline's
  `maxTotalThreadsPerThreadgroup` must be at least 512; the `Context` check of section 2.1 enforces it and L-8
  records it.
- **The conv history and the window.** Window row t reads its taps from `hist` and from `qkv_L[gi]` rows < t. A
  window with T rows therefore needs every row's `qkv` before any row's conv. The serial encoder gives this: step 2
  finishes before step 3 starts (section 8).

### 5.6 QSA block: projections, KV append, indexer and selection

Upstream: `verify.cpp:545-664` (window) and `verify.cpp:982-992` (commit); [K2 sections 1 and 3](audit/K2-qsa-kv-gdn.md).
Dimensions: 24 query heads and 2 KV heads of 256 (query head h uses KV head `h / 12`), `n_rot` 64, the indexer's
4 query heads and one key head of 128, blocks of 4 cells, `cap = 2051` selected cells per row,
`max_blocks = max_cells / 4 + 2`. The ID row stride stays 2051 even when the configured KV capacity is smaller;
`StepRow.width` limits the live selection and attention cells. `qi` is the layer's index among the 12 QSA layers.

**The window, per QSA layer:**

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_q_q8_1` (n 2560) | `mixed` → `xq` |
| 2 | `strata_mv_bf16_b256_n<NC>` (`indexer.k_proj`, 2560 → 128) | `mixed` → `idx_raw_L[qi][T][128]` |
| 3 | `strata_mv_<fmt>_n<NC>` (`attn_k`, `attn_v`, 2560 → 512 each) | `xq` → `kcur[T][2][256]`, `vcur[T][2][256]` |
| 4 | `strata_qsa_norm_rope` `k` (2T rows, in place) | `kcur`, `attn_k_norm` → `kcur` |
| 5 | O-class: `strata_qsa_fwht256` on K and V (`q4`, `STRATA_KV_ROT`) or V only (`k8v4`) | in place |
| 6 | `strata_qsa_kv_append_<mode>` | `kcur`, `vcur` → the K and V pools at cell `step[t].pos` |
| 7 | `strata_idx_append`, window form | `idx_raw_L[qi]`, `indexer.k_norm` → `tail`, `dead`, `pooled`, `block_pos` |
| 8 | `strata_mv_<fmt>_n<NC>` (`attn_q`, 2560 → 12288) | `xq` → `q_full[T][24][512]` |
| 9 | `strata_qsa_norm_rope` `q` (24T rows, in `q_full` at stride 512, out `qcur` at stride 256) | → `qcur[T][24][256]` |
| 10 | O-class: `strata_qsa_fwht256` on `qcur` (`q4`, `STRATA_KV_ROT`) | in place |
| 11 | `strata_mv_bf16_b256_n<NC>` (`indexer.q_proj`, 2560 → 512), then `strata_qsa_norm_rope` `iq` (4T rows, in place) | `mixed` → `qidx[T][4][128]` |
| 12 | `strata_sel_scores`, `strata_sel_topk` | `pooled`, `dead`, `qidx` → `scores[T][max_blocks]` → `ids[T][2051]` |
| 13 | `strata_attn_chunk_<mode>`, `strata_attn_merge` | `qcur`, the pools, `ids` → `attn[T][24][256]` (section 5.7) |
| 14 | O-class: `strata_qsa_fwht256` on `attn` (`q4`, `k8v4`, `STRATA_KV_ROT`) | in place |
| 15 | `strata_qsa_gate` | `attn`, `q_full` → `attn32[T][24][256]` |
| 16 | `strata_q_q8_1` (n 6144), `strata_mv_<fmt>_n<NC>` (`attn_output`, 6144 → 2560) | `attn32` → `bo` |

- All T rows' K, V and indexer keys are appended (steps 6 and 7) before any row selects (step 12). Causality comes
  from each row's own `n_kv`, `n_bid` and `width` in `step[t]`; there is no mask.
- Upstream runs steps 4, 9 and 11 as one launch over all rows when `T > 1` and token by token otherwise. The
  arithmetic is per row either way.
- Upstream splits `q_full` with a strided `cudaMemcpy2DAsync`. The Mac's step 9 reads with the stride instead, which
  copies the same bits.
- Upstream snapshots the indexer tail before the first append of each layer. The Mac snapshots all 12 layers once,
  at the start of the window (`strata_idx_snapshot`, with step 1 of section 5.1), because nothing writes a tail
  before its layer's step 7.
- `idx_raw_L` is per QSA layer in the arena because commit replays it.

**Commit, per QSA layer:** `strata_idx_append` in its commit form restores `tail` and `block_pos` from the snapshot
and replays rows 0..n_keep−1 from `idx_raw_L[qi]`. Nothing else is needed:
- the KV cells of rejected rows are overwritten by the next window before any row can select them;
- the replay rewrites accepted complete blocks; rejected pooled rows after that prefix are ignored. In particular,
  `pooled[n_bid]` may remain stale because selection uses `dead` for the tail block.

Upstream does not restore `block_pos`. No kernel reads it; it is saved in conversation checkpoints. The Mac restores
it, so a checkpoint holds the value that decoding the kept rows one at a time would have left.

`encode_qsa_indexer_commit_batch` applies this replay to all 12 QSA layers in
ordinal order. Every layer is preflighted before the first dispatch, including
its raw-key and state extents plus cross-layer writable/output aliases. The
batch requires identical StepRow, RoPE and packed snapshot views for every
layer; raw keys, gamma and mutable indexer state remain per-layer. Use its
standalone transactional wrapper or propagate caller-owned Encoder failure so
the whole commit segment is discarded. This is the QSA portion of accepted
state only; composing it with GDN/PLE state and production Window scheduling
remains open.

**`strata_qsa_norm_rope`**, grid `rows × 256`, one row per threadgroup. It fuses upstream's `native_qsa.cu` `norm<256>`
with `native_rope.cu` `apply`; both are per row, so fusing them changes no bits. `W` is 256 for `k` and `q`, 128
for `iq`. Row r belongs to window row `t = r / H` (H = 2, 24, 4) and has position `p = steps[t].pos`.

```c++
x = tid < W ? in[r*in_stride + tid] : 0.0f;
part = x * x;                                     // upstream: partial += x*x from 0
part = butterfly(part);  if (lane == 0) sums[warp] = part;  barrier;
part = butterfly(lane < 8 ? sums[lane] : 0.0f);
scale = rsqrt(part / float(W) + eps);             // / W is exact
v = (scale * x) * gamma[tid];                     // tid < W
vals[tid] = v;  barrier;
if (tid < 32) {
    float2 cs = rope[p][tid];
    a = vals[tid];  b = vals[tid + 32];
    out[r*out_stride + tid]      = fma(-b, cs.y, a * cs.x);
    out[r*out_stride + tid + 32] = fma(b, cs.x, a * cs.y);
} else if (tid >= 64 && tid < W) {
    out[r*out_stride + tid] = v;
}
```

- In place (`in == out`) is safe: every thread reads its input before the first barrier and writes after the second.
- For `iq`, threads 128..255 contribute zeros, as upstream's 256-thread `norm` does for 128-wide rows.
- `steps` is a `device const StepRow*` bound at the offset of `WindowParams::step` (or of the MTP rows,
  section 5.11). The RoPE table is section 4.9.
- Deviations: `rsqrt` and the FP64 RoPE table (section 3.9).

`encode_qsa_norm_rope` provides both a standalone command-buffer overload and
a caller-owned Encoder overload. Both validate the same mode-specific extents,
step positions, RoPE rows and alias rules. The shared overload binds the five
buffer views at slots 0–4, `{token_rows, mode}` at slot 5, and dynamic
threadgroup-memory slots 0 and 1 at 8 and 256 floats. It dispatches one
256-thread group for each head row. Window, Prefill and MTP wiring remain open.

**`strata_qsa_kv_append_<mode>`**, grid (2, T) × 256, threadgroup (kvh, t). `row = row(step[t].pos, kvh)` of
section 4.5, `d = tid`. The kernel writes K, then V, from `kcur[t][kvh]` and `vcur[t][kvh]`.

- `f16`: `pool[row*256 + d] = half(x)`.
- `i8` (upstream `kv_q8.cu:30-66`): group `g = d / 64` is simdgroups 2g and 2g + 1. K/V inputs must be finite; non-finite values are unsupported because their scale and integer conversion results are undefined.

  ```c++
  a = bfly_max(fabs(x));  if (lane == 0) wmax[warp] = a;  barrier;
  amax  = fmax(wmax[2*g], wmax[2*g + 1]);
  sbits = as_type<ushort>(half(amax / 127.0f));
  sf    = float(as_type<half>(sbits));                 // quantize against the stored scale
  q     = sf > 0.0f ? clamp(int(rint(x / sf)), -127, 127) : 0;
  codes[row*256 + d] = char(q);  if (d % 64 == 0) scales[row*4 + g] = sbits;
  ```

  `amax / 127.0f` and `x / sf` are correctly rounded divisions, as upstream's (this file is not fast-math).
- `q4` and `k8v4` (O-class): one 32-value group per simdgroup, group `warp` of the row, ggml's `q4_0`
  (upstream `kv_q4.cu:75-92`). `q4` consumes K/V after FWHT256; each nonzero block's fp32 reciprocal and stored
  fp16 scale must remain finite. `k8v4` leaves K unrotated and writes it with the `i8` body; V must be finite
  after FWHT256 in the same command buffer, and each nonzero V block's reciprocal and stored fp16 scale must
  remain finite. The K int8 fp16 scales must also remain finite. The encoder does not scan GPU-produced values.

  ```c++
  amax = fabs(x);  mval = x;
  for (o = 16; o > 0; o >>= 1) {
      a = simd_shuffle_xor(amax, o);  v = simd_shuffle_xor(mval, o);
      if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }   // ties: the larger value, in every lane
  }
  dq = mval / -8.0f;  id = dq != 0.0f ? 1.0f / dq : 0.0f;
  q  = clamp(int(trunc(fma(x, id, 8.5f))), 0, 15);
  hi = simd_shuffle_down(q, 16);
  if (lane < 16) blk[warp].qs[lane] = uchar(q | (hi << 4));
  if (lane == 0) blk[warp].d = half(dq);
  ```

**`strata_qsa_fwht256`** (O-class), grid ⌈rows/8⌉ × 256, one row per simdgroup. Upstream `kv_q4.cu:25-67`.

```c++
for (i = 0; i < 8; ++i) reg[i] = x[i*32 + lane] * 0.0625f;          // the 1/16 scale on load
for (h = 1; h < 32; h *= 2)
    for (j = 0; j < 8; ++j) { v = reg[j]; v2 = simd_shuffle_xor(v, h); reg[j] = (lane & h) == 0 ? v + v2 : v2 - v; }
for (h = 32; h < 256; h *= 2) { s = h / 32;
    for (j = 0; j < 8; j += 2*s) for (k = 0; k < s; ++k) {
        a = reg[j + k];  b = reg[j + k + s];  reg[j + k] = a + b;  reg[j + k + s] = a - b; } }
for (i = 0; i < 8; ++i) x[i*32 + lane] = reg[i];
```

Row `r = 8·tg.x + sg`; a simdgroup with `r >= rows` returns. Upstream launches (32, rows_per_block) threads; the
arithmetic per row is the same. It is an O-class kernel: the products and sums are plain, with no FMA to place.
`encode_qsa_fwht256` supports a standalone command buffer and a caller-owned
Encoder segment. Both validate positive rows, overflow-safe byte extents,
float alignment, Context ownership and pipeline geometry; the shared form
binds values at slot 0 and `rows` at slot 1, then dispatches
`ceil(rows/8) × 256`. It preserves exact in-place operation.

**`strata_idx_snapshot`**, grid 1 × 384, at the start of a rollback-capable window. Thread i copies raw uint32 word i
of each of the 12 QSA layers' `tail` into the tightly packed `tail_snap[12][384]`; thread 0 also copies the 12
`block_pos` values into `block_pos_snap[12]`. Buffer slots 0–3 hold tails, block positions and the two packed
outputs; bytes slot 4 receives the source layer strides in float and int32 element counts. Both standalone and
caller-owned-Encoder forms validate that source views cover the last strided layer and that outputs are disjoint
from each other and every source region. The Encoder form dispatches through the manifest-checked 1 × 384
threadgroup. `INDEXER_SNAPSHOT_LAYERS` is 12, distinct from the model's 48 transformer layers.

**`strata_idx_append`**, grid 1 × 128, `d = tid`. Upstream: `native_qsa_indexer.cu:46-118`, one launch per row.
The Mac loops over the rows in one threadgroup, in order:
- window form: rows t = 0..T−1, `pos = step[t].pos`;
- commit form: first `tail = tail_snap[qi]` and `block_pos = block_pos_snap[qi]`, then rows t = 0..n_keep−1 at
  `pos = pos0 + t`.

```c++
for (t = 0; t < n_rows; ++t) {
    barrier;                                      // the previous row is done with vals and part
    pos = ...;  slot = pos % 4;
    incoming = float(half(raw[t][d]));            // upstream: "SET_ROWS stores F16"
    if (slot < 3) tail[slot][d] = incoming;
    if (pos != 0 && slot != 3) continue;          // uniform
    sum = pos == 0 ? incoming : tail[0][d];
    for (j = 1; j < 4; ++j) sum = sum + (pos == 0 || j == 3 ? incoming : tail[j][d]);
    mean = fma(0.25f, sum, 0.0f);
    ss = butterfly(mean * mean);  if (lane == 0) part[warp] = ss;  barrier;
    ss = butterfly(lane < 4 ? part[lane] : 0.0f);
    scale = rsqrt(ss / 128.0f + eps);
    vals[d] = (scale * mean) * gamma[d];  barrier;
    b = pos / 4;  rp = pos == 0 ? 0 : 4*b;        // the block's first cell
    y = vals[d];
    if (d < 64) {
        i = d % 32;  float2 cs = pos == 0 ? rope0 : rope[rp][i];
        a = vals[i];  z = vals[i + 32];
        y = d < 32 ? fma(-z, cs.y, a * cs.x) : fma(z, cs.x, a * cs.y);
    }
    pooled[b][d] = y;
    if (pos == 0) dead[d] = y;  else pooled[b + 1][d] = dead[d];
    if (d == 0 && pos != 0) block_pos = rp;
}
```

- Upstream runs 256 threads of which 128 work, so its second reduction reads eight partials of which four are zero.
  The Mac's four partials padded with zeros are the same 32 lane values, so the butterfly gives the same bits.
- Each thread reads only the `tail` and `dead` elements it wrote itself, so device memory needs no barrier between
  rows.
- `rope0` is the RoPE table's value at angle 0, `(m, 0)` with m the YaRN magnitude (1 without YaRN), passed with
  `setBytes`. The spare key at position 0 uses it, as upstream's zero angle does.
- Standalone command-buffer and caller-owned-Encoder forms share the same validation. The Encoder form binds
  buffers at slots 0–9 and `IndexerAppendParams` at bytes slot 10, requests 16 bytes at threadgroup-memory slot 0
  and 512 bytes at slot 1, then dispatches the manifest-checked 1 × 128 group. In a shared segment, commit reads
  the packed snapshot produced by an earlier snapshot dispatch. Keep dependent snapshot/append dispatches out of
  one concurrent group so the Encoder can order them.

**`strata_sel_scores`**, grid `min(256, ⌈(top + 1)/8⌉) × 256`, where `top` is the largest `n_bid` of the window's
rows (the host knows it when it encodes). Upstream: `block_scores_multi_kernel` (`qsa_select.cu:606`) with a fixed
grid of 256. Each simdgroup takes blocks `b = 8·tg.x + warp, b += 8·grid` up to `top`, and reads each key once for
all rows. The rows' `qidx` (16 KiB) and steps are staged in threadgroup memory first.

```c++
for (b = ...; b <= top && b < max_blocks; b += 8*grid) {
    kp = pooled[b][4*lane .. 4*lane+3];  kd = dead[4*lane .. 4*lane+3];
    for (t = 0; t < T; ++t) {
        if (b > n_bid[t]) continue;
        k = b == n_bid[t] ? kd : kp;
        score = 0.0f;
        for (h = 0; h < 4; ++h) {
            q = qs[t][h][4*lane .. 4*lane+3];
            dt = fma(k.w, q.w, fma(k.z, q.z, fma(k.y, q.y, k.x * q.x)));
            dt = butterfly(dt);
            score = score + (dt > 0.0f ? dt : 0.0f);
        }
        if (lane == 0) scores[t][b] = (b == n_bid[t] && n_kv[t] % 4 != 0) ? score + 1e9f : score;
    }
}
```

The grid size does not change any score: each (block, row) pair is computed by one simdgroup in a fixed order.
The standalone and caller-owned-Encoder forms accept T = 1..8 and a positive cell capacity divisible by four. They
share validation: each `StepRow` must match `pos`, `n_kv = pos + 1`, `n_bid = n_kv / 4` and
`width = min(n_kv, 2051)`; row extents use `max_cells / 4 + 2`; scores after each row's `n_bid` stay untouched.
Both reject output/input overlap and validate Context ownership, native buffer metadata and alignment. The Encoder
form binds pooled keys, dead key, queries, steps and scores at slots 0–4, params at bytes slot 5, and staged queries
and steps in threadgroup-memory slots 0 and 1. It dispatches the validated `min(256, ceil((top + 1)/8)) × 256`
grid, so a later encode failure can discard the score dispatch with the rest of its segment.

**`strata_sel_topk_reg` and `strata_sel_topk_mem`**, one threadgroup per row. The standalone command-buffer and
caller-owned-Encoder forms of `encode_qsa_selection_topk` share the same validation and dispatch preparation. They
choose the register kernel through 33,792 `max_blocks` and the memory kernel above that limit. Each returns exactly
`width` cell ids in ascending order, and the ids are a function of the scores alone
([03](03-model-reference.md), section 6.4; [K2 section 1.3](audit/K2-qsa-kv-gdn.md)):

1. If `n_kv <= width`, `ids = 0..n_kv−1`.
2. `order_key(s)`: `v = s + 0.0f` (−0 → +0); NaN → 0; otherwise the float bits with the sign bit set for positive
   values and all bits flipped for negative ones.
3. Block b weighs 4 for `b < n_bid` and `n_kv − 4·n_bid` (0..3) for the tail block `n_bid`.
4. Four 8-bit radix passes, high digit first, find the largest key `thr` such that the weight of blocks with key ≥ thr
   is at least `width`. Each pass builds a 256-bin histogram of block weights with integer threadgroup atomics, and
   thread 0 walks it from digit 255 down.
5. `eq_budget = width − (weight of blocks with key > thr)`.
6. Walking blocks in ascending order, emit every cell of a block with key > thr, and the cells of blocks with key ==
   thr in ascending order until `eq_budget` runs out. A threshold block can be emitted in part.

Thread i owns the consecutive blocks `[i·per, (i+1)·per)`, `per = ⌈(n_bid + 1)/threads⌉`. Exclusive prefix sums of
the per-thread counts give each thread's first output slot and its share of `eq_budget`. Because every step is
integer arithmetic on a fixed partition, any thread count gives the same ids. There are two forms:

| Form | Threads | Keys | Used when | Threadgroup memory |
| --- | --- | --- | --- | --- |
| `reg` (upstream `block_topk_reg_kernel<33>`) | 1024 | 33 per thread, in registers | `max_blocks ≤ 33,792` (about 135K cells) | `hist[8][256]` int (8 KiB) + scans |
| `mem` (upstream `block_topk_kernel`) | 256 | re-read from `scores` on each pass | otherwise | `hist[256]`, `s_a[256]`, `s_b[256]` (3 KiB) |

Upstream's `reg` form keeps one histogram per warp, `hist[32][256]` (32 KiB), which leaves no room for its scan
arrays under the 32 KiB limit. The Mac shares one histogram between four simdgroups (`hist[warp / 4]`). Integer
atomics do not depend on the order, so the counts are the same. Like upstream's CUDA path, the choice depends only
on `max_blocks`, not on the position.

The encoder validates every `StepRow`, score and id extent, alignment, Context ownership, native view metadata and
command-buffer or Encoder state. It rejects id-output overlap with either input and leaves each row's ids after
`width` untouched. The Encoder form binds scores, steps and ids at buffer slots 0–2 and `SelectionTopkParams` at
bytes slot 3. It requests `histogram_banks × 256 × sizeof(uint32_t)` bytes at threadgroup-memory slot 0 and one
`threads × sizeof(uint32_t)` scan array at each of slots 1 and 2, then dispatches `token_rows × 1 × 1` groups using
the selected form's thread count. If a later encode fails, the enclosing `Encoder::encode` body must return false
to discard earlier work in the segment.

`metal_qsa_selection_test` checks standalone and shared-Encoder forms against an independent sorted-cell oracle,
including tied threshold blocks, special float order keys, register/memory equivalence and T=8 shared dispatches of
both kernel forms in serial and concurrent modes. It also encodes scoring followed by top-k in one shared segment
and verifies both outputs after completion.

**`strata_qsa_gate`**, grid (24, T) × 256: `attn32[t][h][d] = attn[t][h][d] * sigmoid_k(q_full[t][h][256 + d])`.
The standalone and caller-owned-Encoder forms of `encode_qsa_gate` share validation for T=1..8 and all three float
views; they reject output overlap with either input and permit the two read-only inputs to overlap. They dispatch
one 256-thread group per head and token row. Upstream `native_qsa.cu:69` is a fast-math file. The Mac uses safe-mode
`sigmoid_k` (section 3.6), without CUDA's fast-math build; it is still an approximation with a documented bound of
4 ULP on [−30, 30], not an exact sigmoid.

### 5.7 Sparse attention: chunks and merge

Upstream: `qsa_decode_attn.cu:83-207`. The same two kernels serve the window, prefill (in query batches, section
6.5) and the MTP drafter (section 5.11). `n_chunks = ⌈cap / 64⌉`, where `cap` is the per-row selected-ID capacity
(`id_row_stride` in the Metal encoder): 33 for QSA's 2,051-cell selection. The drafter's `cap` is its window rounded
up to 64, or `max_cells` with no window, so its `n_chunks` is 512 at the default window and at most 4096.

The scratch per row is `part_acc[2·n_chunks][12][256]`, then `part_m[2·n_chunks][12]` and `part_l[2·n_chunks][12]`
(upstream's layout, `n_chunks · 24 · 258 + 64` floats). Slot `s = kvh · n_chunks + chunk`.

**`strata_attn_chunk_<mode>`**, grid (n_chunks, 2, T) × 256, threadgroup (chunk, kvh, t). It handles 64 selected
cells of one KV head for its 12 query heads. Threadgroup memory: `sq[12][256]` (12 KiB), `sp[12][64]` (3 KiB) and
`srow[64]` as `uint` (256 B), 15,616 B in all. Upstream's `srow` is `long long`; a `uint` row index covers
2^31 cells per head.

```c++
n_ids = step[t].width;  c0 = 64*chunk;  n_here = min(64, n_ids - c0);
if (n_here <= 0) { if (tid < 12) { m[s][tid] = -FLT_MAX; l[s][tid] = 0.0f; } return; }
for (i = tid; i < 12*256; i += 256) sq[i / 256][i % 256] = q[t][12*kvh + i / 256][i % 256];
if (tid < 64 && tid < n_here) srow[tid] = row(ids[t][c0 + tid], kvh);    // section 4.5
barrier;
for (c = warp; c < 64; c += 8) {                          // scores: simdgroup per cell, 8 dims per lane
    if (c >= n_here) { if (lane < 12) sp[lane][c] = -FLT_MAX; continue; }
    k8 = load8_<mode>(K, srow[c], 8*lane);
    for (h = 0; h < 12; ++h) {
        sc = k8[0] * sq[h][8*lane];
        for (j = 1; j < 8; ++j) sc = fma(k8[j], sq[h][8*lane + j], sc);
        sc = butterfly(sc);
        if (lane == 0) sp[h][c] = sc * 0.0625f;          // 1/sqrt(256), exact
    }
}
barrier;
for (h = warp; h < 12; h += 8) {                          // simdgroup w: heads w and w + 8
    a = sp[h][lane];  b = sp[h][lane + 32];
    mx = bfly_max(fmax(a, b));
    ea = lane < n_here ? exp(a - mx) : 0.0f;
    eb = lane + 32 < n_here ? exp(b - mx) : 0.0f;
    sp[h][lane] = ea;  sp[h][lane + 32] = eb;
    sum = butterfly(ea + eb);
    if (lane == 0) { m[s][h] = mx; l[s][h] = sum; }
}
barrier;
acc[0..12) = 0.0f;                                        // thread tid owns dimension tid
for (c = 0; c < n_here; ++c) {
    v = load1_<mode>(V, srow[c], tid);
    for (h = 0; h < 12; ++h) acc[h] = fma(sp[h][c], v, acc[h]);
}
for (h = 0; h < 12; ++h) part_acc[s][h][tid] = acc[h];
```

The loads, per mode (upstream `load8_*`):

| Mode | `load8` (K, 8 dims from d0) | `load1` (V, dim d) |
| --- | --- | --- |
| `f16` | `float(half)` of 8 values (one 16-B load) | `float(V[row·256 + d])` |
| `i8` | `float(code[j]) * sc`, `sc = float(as_type<half>(scale[row·4 + d0/64]))` | `float(code) * sc` with the scale of group `d / 64` |
| `q4` (O-class) | block `d0/32`, nibbles `(byte & 15) − 8` for `d0 % 32 < 16`, else `(byte >> 4) − 8`, bytes from `(d0 % 16)`, times `float(d)` | the same for one dim |
| `k8v4` (O-class) | the `i8` K | the `q4` V |

- The products `float(code) * sc` are rounded before they enter the dot, as upstream's are (they are stored in
  `k8[]` first).
- Upstream masks cells whose page is not resident (`srow < 0`). The Mac has no page table, so this residency mask is
  omitted. Independently, the kernel masks IDs outside `[0, min(cell_capacity, step.n_kv))` before deriving a cache
  address; the chunk also limits work to `c < n_here`.
- `exp` for `__expf` is a deviation (section 3.9).

**`strata_attn_merge`**, grid (24, T) × 256, threadgroup (h, t), `kvh = h / 12`, `hl = h % 12`, `d = tid`:

```c++
M = -FLT_MAX;
for (c = 0; c < n_chunks; ++c) M = fmax(M, m[kvh*n_chunks + c][hl]);
L = 0.0f;  acc = 0.0f;
for (c = 0; c < n_chunks; ++c) {                       // ascending, serial
    s = kvh*n_chunks + c;  mc = m[s][hl];
    if (mc == -FLT_MAX) continue;
    w = exp(mc - M);
    L = fma(l[s][hl], w, L);
    acc = fma(part_acc[s][hl][d], w, acc);
}
attn[t][h][d] = L > 0.0f ? acc / L : 0.0f;
```

- The merge loops over all `n_chunks` (33, or 512 for the MTP) and skips empty chunks by their `-FLT_MAX`. A row with
  fewer selected cells than `cap` therefore gives the same bits as upstream's, which also launches every chunk.
- Run to run the result is bitwise: no atomics, every order fixed.

The standalone and caller-owned-Encoder forms of
`encode_qsa_attention_chunk` and `encode_qsa_attention_merge` share validation
for T=1..8, the cache capacity (a positive multiple of four,
up to 262,144 cells), ID-row stride, overflow-safe input/scratch extents,
mode-specific cache layouts and alignment, `StepRow` consistency, Context
ownership, native buffer metadata and command-buffer state. Each row must have
`0 < width <= min(id_row_stride, n_kv)`; every ID in that live prefix must
name a cell below both `cell_capacity` and `n_kv`. The encoder checks the
StepRow bounds on the host but cannot validate ID values when an earlier GPU
dispatch produces them in the same command buffer. The kernel masks negative,
out-of-capacity and future-cell IDs before any cache access; callers must still
produce valid IDs for complete attention results. Read-only inputs may overlap;
the three scratch outputs and merge output must be
disjoint from their respective inputs and each other. The chunk encoder's
`id_row_stride` sets the number of chunks. Callers can encode chunk followed by
merge in one ordered Encoder segment; in concurrent mode, do not place the
dependent dispatches in the same concurrent group. The segment body must
propagate either encoder's failure so previously encoded work is discarded.
Production Window/Prefill wiring and model-backed parity remain open.

### 5.8 Q8_1 and quantised mat-vecs

Upstream: `native_mmvq.cu` (a fast-math file) and `iq_kernels.cu:448-560, 780-850` (not fast-math);
[K3 sections 1.3, 2 and 3.4-3.5](audit/K3-quant-and-matvec.md). Every quantised dense tensor of
[03](03-model-reference.md), section 3.3 (11 formats), the head and the shared expert multiply Q8_1 activations with
the kernels of this section. The routed experts use the same per-format dots inside the grouped kernels of
section 5.10. Prefill uses the GEMMs of section 6.2 instead.

**`strata_q_q8_1`**, grid (⌈n/256⌉, rows) × 256. One simdgroup quantises one 32-value block. `rows` is the grid's
y size: T for the window's activations, 10·T for the expert intermediates (section 5.10). `n` (2560, 6144 or 640) is a
multiple of 32 and comes with `setBytes`. The output block is `struct Q81 { half2 ds; char qs[32]; }` (36 B), row r
at `y + r·(n/32)`.

```c++
i = 256*tg.x + tid;  r = tg.y;
if (i >= n) return;                               // whole simdgroups return
xi = x[r*n + i];
amax = fabs(xi);  sum = xi;
for (o = 16; o > 0; o >>= 1) {
    amax = fmax(amax, simd_shuffle_xor(amax, o));
    sum  = sum + simd_shuffle_xor(sum, o);
}
d = amax / 127.0f;                                // correctly rounded (section 3.2)
q = amax == 0.0f ? 0 :
    (d == 0 ? exact_round_ratio(xi, amax, 127)
            : d is subnormal ? round(xi / exact_subnormal_d)
                             : int(round(xi / d)));
blk = y + r*(n/32) + i/32;
blk->qs[i % 32] = char(q);
if (i % 32 == 0) blk->ds = half2(half(d), half(sum));
```

- `sum` is the butterfly sum of the unrounded inputs, not of the codes. The Q4_0 and Q5_0 dots read it.
- `amax` is reduced as unsigned absolute float32 bit patterns so a hardware flush-to-zero cannot hide an input
  subnormal. For exponent fields through 7, the correctly rounded `d = amax/127` is reconstructed in units of the
  smallest float32 subnormal. If `d` rounds to zero, integer significand arithmetic rounds the normalized ratio
  `127·xi/amax`; this defines useful Q8_1 codes where CUDA's `xi/0` conversion has no portable result. If `d` is a
  nonzero subnormal, both operands are rescaled from their exact significands before division, reproducing the
  correctly rounded `xi/d` without hardware denormal arithmetic. That exceptional path bounds codes to [-127, 127].
  If `d` is normal but `xi` is subnormal, the significands are divided as normal floats when `d` has exponent field
  1; when its exponent field is greater than 1, `xi/d < 0.5` and the rounded code is zero. Other inputs with normal
  `d` use `round(xi/d)` directly.
- Upstream has two copies of this kernel. `quantize_q8_1_rows` (`iq_kernels.cu:979`, precise) quantises the expert
  inputs and intermediates. `native_quantize_q8_1` (`native_mmvq.cu:141`, fast math, so `amax / 127.0f` and `xi / d`
  are approximate) quantises the inputs of the dense projections, the head, the shared expert and the MTP. The Mac
  has one quantiser. For finite inputs where the precise CUDA scale and float-to-integer conversion are defined,
  it matches `quantize_q8_1_rows`; zero and out-of-range subnormal cases follow the explicit local policy above.
  Using the precise quantiser instead of `native_quantize_q8_1` can change a code near a half-integer boundary
  (section 3.9).

**The mat-vecs.** `strata_mv_<fmt>_n<NC>` computes `y_j[r] = Σ_b dot(W[r], block b; x_j)` for every output row r
and every column j < NC. Its arguments:

| Argument | Binding |
| --- | --- |
| `w` | the tensor, by buffer and offset (section 4.3). Row r starts at `r · row_bytes`, `row_bytes = (n_in / DIV) · block_bytes`. |
| `xq` | the Q8_1 activations. Column j starts at block `j · (n_in / 32)`. |
| `y`, `sink` | the output, column j at `y + j · n_out`, and the sink of section 4.4 |
| `n_in`, `n_out` | `setBytes`, fixed per tensor |
| `rows` | the live column count, a `device const uint*` bound at the offset of `WindowParams::T` (or of the MTP's row count, section 5.11) |

Column j reads `xcol(j) = j < rows ? j : 0` and writes `ycol(j) = j < rows ? y + j·n_out : sink + (j − rows)·n_out`.
Weight words are assembled from two `ushort` loads, as upstream's `get_int_b2` / `load_int_b2` do, because most
block sizes leave blocks only 2-B aligned. Activation words are `int` loads: Q8_1 blocks are 36 B in 256-B-aligned
regions.

There are two families, as upstream's dispatch (`native_mmvq.cu:1477-1493`) has:

| Family | Formats | Threadgroup | Upstream |
| --- | --- | --- | --- |
| Native | Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_XS, Q4_0, Q5_0, Q8_0, IQ4_NL | 128 threads (4 simdgroups) for R rows; grid ⌈n_out/R⌉ | `native_<fmt>_mmvq_kernel`, `native_small_mmvq_kernel`, and `native_mmvq_multi_kernel` in its exact layout (`g_multi_exact = true`, the default) |
| Row | IQ3_S | 128 threads, one simdgroup per row, 4 rows; grid ⌈n_out/4⌉ | `mmvq_kernel<21>`, `mmvq_multi_kernel<21, NC>` (`iq_kernels.cu:500, 832`) |

**Native family.** Each format F has the traits of the table below. `F::load` decodes one weight block's share for
the thread (codes, integer scales, the block scale as a float), and `F::acc` applies it to one column and returns the
new accumulator, with the FMA form of the table.

```c++
row0 = R*tg.x;  bpr = n_in / F::DIV;
float tmp[NC][R] = {};
for (kbx = tid / F::T; kbx < bpr; kbx += F::BPI) {
    kby = kbx * F::KBY;  kqs = F::kqs(tid);
    for (i = 0; i < R && row0 + i < n_out; ++i) {
        F::W wv = F::load(w + (row0 + i)*row_bytes, kbx, kqs);            // once per (row, block)
        for (j = 0; j < NC; ++j)
            tmp[j][i] = F::acc(wv, xq + xcol(j)*(n_in/32) + kby, kqs, tmp[j][i]);
    }
}
if (warp > 0) for (j, i) partial[warp - 1][j][i][lane] = tmp[j][i];
barrier;
if (warp > 0) return;
for (j = 0; j < NC; ++j) for (i = 0; i < R && row0 + i < n_out; ++i) {
    v = tmp[j][i];
    v = v + partial[0][j][i][lane];  v = v + partial[1][j][i][lane];  v = v + partial[2][j][i][lane];
    v = butterfly(v);
    if (lane == i) ycol(j)[row0 + i] = v;
}
```

`partial` is `float[3][NC][R][32]`, 384·NC·R B: 12 KiB at NC = 8 and R = 4.

| Format (id) | DIV / block B | T | KBY | BPI | `kqs(tid)` | Upstream dot | `acc`: float tail and accumulation |
| --- | --- | ---: | ---: | ---: | --- | --- | --- |
| Q4_K (12) | 256 / 144 | 16 | 8 | 8 | `2·(tid % 16)` | `q4_q8_dot` (514), impl 495 | for i < 2: `sd = fma(d8[i], float(dot1·sc[i]), sd)`, `sm = fma(d8[i], float(dot2·m[i]), sm)`; `v = fma(-dm.y, sm, dm.x * sd)`; `acc + v` |
| Q5_K (13) | 256 / 176 | 16 | 8 | 8 | `2·(tid % 16)` | `q5_q8_dot` (178), impl 155 | as Q4_K |
| Q6_K (14) | 256 / 210 | 32 | 8 | 4 | `tid % 32` | `q6_q8_dot` (608), impl 593 | for i < 2: `sf = fma(d8[i], float(dp·sc), sf)`; `fma(d, sf, acc)` |
| Q3_K (11) | 256 / 110 | 16 | 8 | 8 | `tid % 16` | `q3_q8_dot` (357), impl 334 | for i < 4: `sf = fma(d8[i], float(dp·sc), sf)`; `fma(d3, sf, acc)` |
| IQ4_XS (23) | 256 / 136 | 8 | 8 | 16 | `4·(tid % 8)` | `iq4_xs_q8_dot` (436) | `sumi *= ls − 32`; `fma(dw * d8, float(sumi), acc)` |
| Q2_0 (42) | 64 / 18 | 2 | 2 | 64 | `tid % 2` | `q2_q8_dot` (262) | `fma(d2 * d8, float(sumi), acc)` |
| Q4_0 (2) | 32 / 18 | 2 | 1 | 64 | `2·(tid % 2)` | `small_q8_dot` (667) | `fma(d, fma(-4.0f, ds.y, float(sumi) * ds.x), acc)` |
| Q5_0 (6) | 32 / 22 | 2 | 1 | 64 | `2·(tid % 2)` | `small_q8_dot` (683) | `fma(d, fma(-8.0f, ds.y, float(sumi) * ds.x), acc)` |
| IQ4_NL (20) | 32 / 18 | 2 | 1 | 64 | `2·(tid % 2)` | `small_q8_dot` (722) | `fma(dw * d8, float(sumi), acc)` |
| Q8_0 (8) | 32 / 34 | 4 | 1 | 32 | `2·(tid % 4)` | `small_q8_dot` (708) | `fma(d0 * d1, float(sumi), acc)` |

In the table, `d8` and `ds.x` are the activation block's `float(ds.x)`, `ds.y` its `float(ds.y)`, and `dw`, `d`,
`d2`, `d3`, `d0` and `dm` the weight block's scales as floats. The accumulators `sd`, `sm` and `sf` start at 0.0f.
The integer parts (`sumi`, `dot1`, `dot2`, `dp`, the scale unpacking) are upstream's, transcribed with the
emulations below.

**Row family (IQ3_S here; the routed experts in section 5.10).** One simdgroup per row, lanes strided over the
calls of the row, then the butterfly (upstream `row_dot`, `iq_kernels.cu:490`, and `row_dot_multi`, 812):

```c++
row = 4*tg.x + warp;  if (row >= n_out) return;
wr = w + row*row_bytes;  nb = n_in / F::qk;
float s[NC] = {};
for (k = lane; k < nb*F::ipb; k += 32) {
    kbx = k / F::ipb;  iqs = F::step * (k % F::ipb);
    F::W wv = F::load(wr, kbx, iqs);
    for (j = 0; j < NC; ++j) s[j] = F::acc(wv, xq + xcol(j)*(n_in/32) + kbx*(F::qk/32), iqs, s[j]);
}
for (j = 0; j < NC; ++j) { v = butterfly(s[j]); if (lane == j) ycol(j)[row] = v; }
```

| Format (id) | qk / block B | ipb | step | Upstream dot | Integer step after the dot | `acc` |
| --- | --- | ---: | ---: | --- | --- | --- |
| IQ2_XXS (16) | 256 / 66 | 8 | 2 | `vec_dot_iq2_xxs_q8_1` (80) | `ls = aux32 >> 27 | 1; sumi = sumi*ls / 8` | `fma(dw * d8, float(sumi), acc)` |
| IQ2_XS (17) | 256 / 74 | 8 | 2 | `vec_dot_iq2_xs_q8_1` (106) | `sumi = (sumi0*ls0 + sumi1*ls1 + (sumi0 + sumi1)/2) / 4` | as above |
| IQ2_S (22) | 256 / 82 | 8 | 2 | `vec_dot_iq2_s_q8_1` (137) | as IQ2_XS | as above |
| IQ3_XXS (18) | 256 / 98 | 8 | 2 | `vec_dot_iq3_xxs_q8_1` (170) | `ls = aux32 >> 28; sumi = (ls*sumi + sumi/2) / 2` | as above |
| IQ3_S (21) | 256 / 110 | 8 | 2 | `vec_dot_iq3_s_q8_1` (196) | `sumi *= 1 + 2*((scales[iqs/4] >> ((iqs << 1) & 4)) & 15)` | as above |
| IQ1_M (29) | 256 / 56 | 8 | 1 | `vec_dot_iq1_m_q8_1` (223) | `sumf[k] = fma(delta, float(sumy), sumf[k])`, all exact | `fma(dw * d8, X, acc)`, `X = (float(sumi0) + sumf0)·sc0 + (float(sumi1) + sumf1)·sc1`, exact in any order |
| IQ4_XS (23) | 256 / 136 | 8 | 4 | `vec_dot_iq4_xs_q8_1` (275) | `sumi *= ls − 32` | `fma(dw * d8, float(sumi), acc)` |
| IQ4_NL (20) | 32 / 18 | 2 | 2 | `vec_dot_iq4_nl_q8_1` (256) | none | as above |
| Q2_0 (42) | 64 / 18 | 2 | 1 | `vec_dot_q2_0_q8_1` (57) | none | `fma(d2 * d8, float(sumi), acc)` |

`strata_mv_iq2_xxs_n<NC>` is an additional standalone row-dot twin for the IQ2_XXS
gate/up dot used by section 5.10. It has the row family's 128-thread shape and NC
specialisations, independently checks the format's packed decode and arithmetic,
and shares its fixed block with the grouped expert tests. Dense tensor dispatch
does not select it; the implementation choice is recorded as D-IMPL-086.

`strata_mv_iq2_xs_n<NC>` is the corresponding standalone row-dot twin for the
IQ2_XS gate/up format. It preserves the row-family's packed 9-bit grid indices,
sign lookup, two per-half scale nibbles and integer divisions so those operations
can be verified independently of grouped scheduling, and shares its fixed block
with the grouped expert tests. Dense tensor dispatch does not select it; the
implementation choice is recorded as D-IMPL-087.

`strata_mv_iq2_s_n<NC>` is the standalone row-dot twin for the IQ2_S gate/up
format. It checks the 1024-entry grid index assembled from `qs` and `qh`,
per-component sign bytes, two per-half scale nibbles and signed integer
divisions independently of grouped scheduling, and shares its fixed packed
block with the grouped expert tests. Dense tensor dispatch does not select it;
the implementation choice is recorded as D-IMPL-088.

`strata_mv_iq3_xxs_n<NC>` is the standalone row-dot twin for the IQ3_XXS
gate/up format. It checks the eight-part 98-byte block layout, each pair of
four-value grid indices, 7-bit sign-index parity, scale nibble and signed
integer divisions as a decoder-focused check alongside grouped expert
integration. Dense tensor dispatch does not select it; the implementation
choice is recorded as D-IMPL-089.

`strata_mv_iq1_m_n<NC>` is the standalone row-dot twin for the IQ1_M gate/up
format. It checks the 56-byte block layout, 11-bit codebook indices, packed
delta selectors, both odd local scales and the reconstructed fp16 super-scale
as a decoder-focused check alongside grouped expert integration. Its 32-lane
reduction covers the eight Q8_1 parts of each 256-value block. Dense tensor
dispatch does not select it; the implementation choice is recorded as
D-IMPL-090.

- Integer division in these steps truncates toward zero, as C's does. MSL's `/` on `int` does the same; a shift
  must not replace it, since `sumi` can be negative.
- **Indexed rows (`_x`).** The `_x` variants take `device const int* rmap` and read weight row `rmap[r]` for output
  row r: `F::load(w + rmap[row0 + i]*row_bytes, ...)` in the R-row family, `wr = w + rmap[row]*row_bytes` in the row
  family. Nothing else changes, so output row r is bitwise the plain kernel's row r over a matrix whose row r is
  weight row `rmap[r]`, which is upstream's gathered draft head (section 5.11). They exist for the head formats at
  NC = 1 only.
- Upstream's `Split<TY>` load/apply forms (`iq_kernels.cu:523-810`) are bitwise equal to the plain dots; the Mac
  writes only the split form.

**Emulating the CUDA integer intrinsics.** All are exact integer operations (section 3.8):

| CUDA | Mac |
| --- | --- |
| `__dp4a(a, b, c)` | `dot4_i8(a, b) + c`: the four signed byte products, summed in `int` |
| `__vsub4(a, b)` | per-byte subtraction modulo 256: `as_type<int>(as_type<uchar4>(a) - as_type<uchar4>(b))` |
| `__vcmpne4(a, 0)` | per byte `0xFF` where the byte is non-zero, else `0x00` |
| `__vsubss4(a, b)` | per-byte saturating subtraction. In Q3_K (`0..3 − {0, 4}`) and Q6_K (`0..63 − 32`) it never saturates, so it is the plain `char4` subtraction. |
| `__popc(v)` | `popcount(v)` |
| `__byte_perm(0x020100ff, 0x020100ff, q)` (Q2_0) | each 2-bit code c becomes the signed byte `c − 1` ∈ {−1, 0, 1, 2}. CUDA's selector uses three bits per nibble, so bit 2 only chooses the duplicate copy of the table. |
| `get_int_from_table_16` (IQ4_NL, IQ4_XS) | `v.x` = `kvalues_iq4nl[byte & 15]`, `v.y` = `kvalues_iq4nl[byte >> 4]` for the four bytes of the word |

The test `mv_decode` compares each decoder's signed bytes with a C++ transcription of the upstream intrinsic code,
for every value of the input word (all 2^16 for Q2_0's `int16`, every byte for the tables and signs).

**Why the bits do not depend on NC or R:**
- The map from threads to (block, call) depends only on `tid` (native) or `lane` (row). Neither NC, R nor T changes
  it.
- `F::load` produces the same values for every column. `F::acc` applies them to each column with the same operations
  in the same order as an NC = 1 call. So column j of an NC-wide call is bitwise equal to an NC = 1 call on column j
  (`mv_multi_eq_single`, section 10.4), and a window row is bitwise equal to the same row decoded alone.
- R only changes which rows a threadgroup takes. Upstream uses R = 4 when `n_in / DIV < BPI` and R = 1 otherwise. The
  Mac's R is a per-format constant in `common/params.h`, default 4, chosen by L-2; R ≤ 8 keeps `partial` within
  24 KiB at NC = 8. If L-2 finds that one family of Macs needs a different R, it becomes a host-name suffix `_r<R>`.
- Every integer sum converts to float exactly: the largest, IQ4_XS's `sumi·(ls − 32)`, is at most
  516,128 · 32 < 2^24.

**FMA forms (L-9).** The `acc` column applies section 3.3 to the inlined expression `tmp += dot(...)`. nvcc fuses
the dot's last product into the outer add, which gives `fma(d, float(sumi), acc)`. In Q4_K and Q5_K the dot ends
in a subtraction of two products, so the outer add stays a plain add. In Q4_0 and Q5_0, step 5 fuses the second
product of `sumi*ds.x − 4*ds.y` into the first. L-9 checks every row of the two tables against the SASS of
`native_mmvq.cu` and `iq_kernels.cu`, including the alternative `fma(dm.x, sd, -(dm.y * sm))` for the K-quants.
Where the SASS differs, L-9 corrects the kernel and its oracle together.

### 5.9 BF16 mat-vecs

Upstream: `native_bf16.cu:44-133` (`bf16_f32_mmvf_kernel<BS>`, `bf16_f32_mmvf_multi_kernel<BS, NT>`,
`mmvf_block_size`), an adaptation of llama.cpp's `mmvf.cu`; [K1 K20-K21](audit/K1-gr-router-shared-rope-cvec.md) and
[K3 section 3.6](audit/K3-quant-and-matvec.md). Every BF16 tensor of [03](03-model-reference.md), section 3.3,
multiplies F32 activations (not rounded to BF16) with this kernel: the routers, the shared-expert gate, the head
mixer, the indexer projections, `ssm_alpha` and `ssm_beta`, `ple_value`, a BF16 `ple_key`, and the draft layer's
BF16 tensors. The layer GR mixers use the fused kernels of section 5.4 instead.

**`strata_mv_bf16_b<BS>_n<NC>`**, grid n_out × BS. One threadgroup per output row. BS is upstream's
`mmvf_block_size(n_in)`: of 32, 64, ..., 256 threads, the smallest with the fewest iterations
`⌈n_in / (2·BS)⌉`. It is 256 for n_in = 2560 and 10240, and 160 for n_in = 320 (`output_hc_up`), the only widths in
the catalogue. The host computes it with the same function. A BF16 tensor whose BS has no variant fails at load.

```c++
row = tg.x;  wr = (device const uint*)(w + row*n_in*2);              // two BF16 per word
float acc[NC] = {};
for (p = tid; p < n_in/2; p += BS) {
    word = wr[p];
    w0 = as_type<float>(word << 16);  w1 = as_type<float>(word & 0xffff0000u);
    for (j = 0; j < NC; ++j) {
        x2 = ((device const float2*)(x + xcol(j)*n_in))[p];
        acc[j] = fma(w0, x2.x, acc[j]);
        acc[j] = fma(w1, x2.y, acc[j]);                               // two ordered FMAs, not a pair sum
    }
}
for (j) acc[j] = butterfly(acc[j]);
if (BS > 32) {
    if (tid < 32) for (j) partials[j][tid] = 0.0f;
    barrier;
    if (lane == 0) for (j) partials[j][warp] = acc[j];
    barrier;
    if (warp == 0) for (j) acc[j] = butterfly(partials[j][lane]);     // the zero slots are summed too
}
if (tid == 0) for (j) ycol(j)[row] = acc[j];
```

- The arguments and the column map (`rows`, `xcol`, `ycol`, the sink) are those of section 5.8, with `x` the F32
  activations, column j at `x + j·n_in`.
- `partials` is `float[NC][32]` (1 KiB at NC = 8). The zeroing must come before the stores, with a barrier between,
  as upstream's does. Upstream zeroes before the loop; the order of the two is irrelevant to the result.
- The second butterfly adds 32 values, of which 32 − BS/32 are +0.0. Adding +0.0 is exact except that it turns
  −0.0 into +0.0, so the Mac sums all 32 slots, as upstream does, and the sign of a zero result matches.
- BF16 widening is a 16-bit shift and exact for every input, NaN payloads included.
- Every column is bitwise equal to an NC = 1 call (upstream states this for its multi kernel, NT = 4 or 8), so the
  Mac's NC does not need to match upstream's NT. `mv_multi_eq_single` (section 10.4) covers this kernel too.
- Weight words are `uint` loads: tensors start 32-B aligned and n_in is even. Activations are `float2` loads:
  `x` is 256-B aligned and n_in is even, so column j starts 8-B aligned.
- No FMA form is left to the compiler: the two FMAs are written out, and the butterflies are adds.

The engine exposes Q8_1 quantization and dense quantized/BF16 mat-vecs through both standalone
`CommandBuffer` wrappers and caller-owned `Encoder` overloads. A dependent projection can therefore
encode Q8_1 and its mat-vec in the same serial segment, with the wrapper discarding that segment if a
later-stage validation fails. `encode_bf16_matvec` additionally accepts F32 activations, BF16 weights
and the active-row count as `ReadOnlyBufferRef`; read-only inputs may overlap each other, while both
outputs must be disjoint from every input and one another. This supports the const full-mode router
weight view without changing the existing mutable mat-vec API. `encode_q8_0_matvec` provides the
same read-only binding contract for Q8_1 activations, Q8_0 weights and live-row metadata, with
F32 output and padded-column sink views. It supports the shared-expert projections without
converting full-mode model views back to mutable `BufferRef`.
`encode_q2_0_matvec` provides the same read-only contract for Q8_1 activations,
Q2_0 weights and live-row metadata, including the FullMode T=1 PLE key
projection; its scalar-oracle test covers rows with 65 Q2_0 blocks.
`encode_read_only_matvec` generalizes immutable activation, weight and row-count
binding to every safe dense mat-vec format. This covers the Q5_K, Q6_K and
IQ4_XS formats accepted by the inspected `output.weight` role while keeping
the existing mutable `encode_matvec` interface available to other callers.
All inputs remain unchanged until command completion even if another mutable
view can reach the same backing bytes. Its standalone wrapper is transactional;
the caller-owned form must propagate failure out of its enclosing Encoder body.
The Window output-projection coordinator resolves the actual `output.weight`
view and composes Q8_1 activation quantization with this read-only mat-vec for
the fixed `[2560, 248320]` head. `encode_window_output_head` now resolves the
FullMode `output_hc_norm`, `output_hc_down` and `output_hc_up` tensors and
composes the final mixer through logits in one transaction, using dedicated
named intermediate scratch. `encode_window_output_sampling` appends the
penalty-aware split top-k and merge to that same transaction using the
`WindowParams` sampler rows and named history, candidate and selection scratch.
It rejects non-finite settings, out-of-range top-p/min-p, non-positive repeat
penalties, and unsupported top-k/min-keep/greedy values before encoding.
After successful command completion, `sample_window_output_tail` waits and
runs the existing CPU FP64 sampler tail for every active row. History contents
and all owners remain caller-managed through completion. The tail rejects
non-finite candidate logits or invalid probabilities without publishing a
partial result. This is a standalone composition API; production `Window::run`,
accepted-prefix state commit and model-backed parity remain open.
The PLE post-projection path accepts read-only gamma and F16 convolution weights; its
convolution-residual dispatch binds normalized rows,
history, convolution weights and gated activations as read-only inputs and
keeps only the residual writable. The validated
`encode_moe_shared_ffn` coordinator now composes the shared-expert chain and
combine. The full-mode layer-0 coordinator reaches those stages through the
routed/shared composition so it can bind the routed parts in the same transaction.
The T=1 Window router prefix orders
the layer-0 attention/GDN/FFN read, 256/512-expert BF16 router, multiply-by-reciprocal top-10 route
and Q8_1 quantization of `mixed` in one segment. It slices the live-row count from `WindowParams.nc`
and reuses GDN's `xq` only after the GDN reads finish. `encode_window_t1_moe_plan_prefix` extends
that segment with the layer-0 GPU plan kernel, using the produced route IDs and disjoint resident,
usage and plan views. `encode_window_t1_moe_layer0_full_mode` takes the `FullModeStorage` owner,
uses its expert address/resident/usage tables, and composes the actual router/route/plan prefix
with routed expert FFN, shared expert and combine in one ordered transaction. It leaves the
combined output pending in `ffn.shared.output`; it does not write the GR residual.
The separate `encode_window_moe_layer0_full_mode` producer accepts T=1..8 from validated
`WindowParams`, requires `n_keep == 0`, uses `WindowParams.T` for active router, route, Q8_1,
plan and shared-FFN rows, and treats `WindowParams.nc` only as the padded compiled-column width.
It preserves the same FullMode table ownership and pending-output policy. This producer can be
followed by the generic pending-consumer selector in a caller-owned Encoder segment, but does not
compose token embedding or GR broadcast. The legacy integrated FullMode token step remains T=1;
`encode_window_moe_full_mode_input_step` is its speculative T=1..8 sibling and composes row-wise
embedding gather, four-stream GR broadcast, the generic FullMode producer and exactly one pending
consumer. It derives `T` and token rows from validated `WindowParams`, requires `n_keep == 0`, and
retains the T=1 exact `shared.output` scratch reuse rule. This layer-0 input composition still does
not schedule later layers or commit speculative state.
The generic speculative router prefix supports T=1..8 and uses `WindowParams.T` for active router,
route and Q8_1 rows while `nc` determines padded BF16 mat-vec columns. Its plan-prefix companion
continues through the layer-0 `strata_moe_plan` kernel for either full or tiered mode. Full mode
does not update usage; tiered mode adds one usage count for every routed entry, including speculative
rows, and stops before expert execution or accepted-prefix state updates.
`encode_window_moe_ple_entry` takes the routed/shared result and layer-0 FFN GR-read structures,
binds that output as `previous_block_out` and the GR read's `inject_out` as `previous_inject`, and
requires the PLE residual to reuse the GR read's residual view. Its caller-owned Encoder form runs
after the producer in the same ordered segment; the standalone form requires the producer command
buffer dependency to be registered with `consumer.depends_on(producer, error)`, and the producer
to complete successfully before the consumer command buffer is submitted.
For the no-PLE branch, `encode_window_moe_attention_entry` binds those same output and injection
views as the next attention read's `block_out` and `inject_prev`, requires the exact prior residual
view, and encodes that read with `apply=true`. Its standalone form has the same command-buffer
dependency and producer-completion requirement. The Window scheduler must select exactly one
consumer. The full-mode MoE coordinator validates the consumer's address/resident tables, Q8_1
activations, plan, route weights and mixed input against the producer views, including the same
`WindowParams.nc`. Its generic speculative path binds active rows to `WindowParams.T`; the existing
T=1 path preserves its `WindowParams.nc` view. Consumer
reads may overlap producer storage only in the specific input roles declared for each handoff.
Router weights and the resident table are checked as producer inputs, and consumer outputs and
sinks cannot overlap them. In tiered mode the plan prefix ends at plan production; a caller must
establish plan readiness, resolve misses through `ExpertStore`, retain expert leases through GPU
completion, and then call the no-write routed/shared consumer for layer 0.
There is no production scheduler that performs this sequence. Later Window layers, the indexed
draft-head API, and Prefill dispatch remain pending.
`metal_fullmode_handoff_test` exercises the full-mode producer and the no-PLE attention consumer
in one ordered segment using a sparse 256-expert GGUF fixture. It checks the produced route,
nonzero routed/shared output, consumer outputs against a direct GR-read reference, and exactly-once
residual application. Its generic FullMode layer-0 producer cases cover T=1, T=2, T=3 with
`nc=4`, and T=8. The generic producer-to-consumer step runs as a standalone T=2 transaction and
T=3 transaction, and as a caller-owned T=8 transaction. Independent router, route, Q2_0 routed-expert, Q8_0
shared-expert, combine, pending-output and GR-consumer oracles check the active outputs. T=3 cases
verify active rows are distinct from compiled columns and reject undersized padded sinks. Late
shared-FFN validation failures in caller-owned T=2 and standalone T=3 forms discard earlier
producer work; late consumer failures in standalone T=2 and caller-owned T=3 steps discard the
producer and consumer dispatches and preserve every writable view. A consumer output alias to the
FullMode usage table is rejected before dispatch. The fixture checks active-row output guards and
unchanged FullMode usage. These synthetic multirow fixtures validate composition and transaction
behavior; they do not establish real-model parity or production scheduling.
`metal_fullmode_ple_handoff_test` exercises the T=1 producer with the typed PLE entry in one ordered
segment, then compares its PLE outputs and residual against a direct GR-post plus PLE-entry
reference. It also runs generic producer-to-PLE steps at T=1 and T=3 (`nc=4`); T=1 matches the
token-to-consumer step, and T=3 uses distinct synthetic PLE input rows to match every output and
the residual against a direct T=3 PLE-entry reference, while checking padded output guards and
deferred PLE history. The compact fixture has a one-token vocabulary, so the T=3 row-sensitive
oracle supplies synthetic input values after staging its table allocation. Speculative PLE
composition rejects `advance_rows` and a PLE table-input alias to the pending MoE output before
encoding. The speculative input stem compares T=3 and T=8 embedding gather and four-stream
broadcast against direct row-wise encoder calls, checks nonzero T=8 reference data and untouched
embedding/stream tails, and rejects an invalid later token and nonzero `n_keep` before dispatch.
The T=3 FullMode-to-PLE integration now uses `encode_window_moe_full_mode_input_step`; standalone
and caller-owned late PLE output failures verify transaction rollback, with the caller-owned test
propagating false from its enclosing `Encoder::encode` body. `prepare_window_t1_input`
builds the legacy T=1 `WindowParams` ring slot and, for PLE, stages the sixteen IQ4_NL rows
derived from the test token and committed history. An uninitialized session is rejected without
changing the ring slot or PLE scratch. The test compares staged values against a direct table gather
and verifies preparation leaves token history unchanged. The new
`prepare_window_speculative_input` prepares T=3 PLE parameters and session-derived rows and T=8
no-PLE parameters; an invalid final token preserves both the ring slot and staged-input scratch,
successful T=3 staging matches a direct batch table gather, and token/sampler spans backed by the
PLE output scratch retain their original values in the published parameters. It resolves the PLE projection, normalization and
convolution weights from deterministic, correctly typed tensors in that sparse
GGUF fixture and binds them through `ReadOnlyBufferRef`. These branch-specific
tests do not select a production scheduler branch or establish parity against a
real model artifact.
Both tests route through `encode_window_moe_pending_consumer`, which requires the selected
`ple_enabled` tag to match a variant containing exactly one consumer payload. The mismatch case is
rejected before dispatch. They now call `encode_window_t1_full_mode_step`, which resolves the
read-only native `token_embd.weight` view from `FullModeStorage`, maps GGML Q3_K, IQ3_S or IQ4_XS
to its gather format, and orders embedding gather, GR broadcast, the FullMode layer-0 producer,
and the selected PLE or next-attention consumer in one segment. (The optional BF16 embedding
artifact is not resolved by this coordinator.) The embedding scratch is dead after broadcast and
is checked against all later live views. It may be reused as `shared.output` only when the views
are exactly equal. A standalone `CommandBuffer` overload wraps the caller-owned encoder and
discards the entire segment on validation failure. This synthetic T=1 composition still does not
wire `Window::run`, later layers, commit, head, sampler, tiered scheduling, or model parity.

### 5.10 MoE: router, plan, experts, shared expert and combine

Upstream: `native_router.cu` (`route`, fast math), `iq_kernels.cu:853-976` (`native_gu_multi_kernel`,
`swiglu_entries_kernel`, `native_down_multi_kernel`), `shared_expert.cu:172-219` (`shared_expert_multi`),
`native_moe.cu` (`combine`, fast math), and `s2_expert_grouped.cu:1044-1087` (`group_resident_kernel`, the device
grouping that the Mac's plan kernel follows); [K1 sections 4-5](audit/K1-gr-router-shared-rope-cvec.md) and
[K3 section 3.5](audit/K3-quant-and-matvec.md). Constants: k = 10 experts per row, n_ff = 640 per expert,
n_expert = 512 (256 for Coder), n_embd = 2560.

The FFN of a layer is steps 6-10 of section 5.1. Its buffers in the arena (section 4.4):

| Buffer | Shape | Written by |
| --- | --- | --- |
| `logits_r` | F32 [T][n_expert] | router mat-vec |
| `ids`, `wts` | int32 [T][10], F32 [T][10] | `strata_moe_route` |
| `plan[l]` | `MoePlan` (below), one per layer, in a Shared buffer | `strata_moe_plan` |
| `xq` | Q81 [T][80] | `strata_q_q8_1(mixed)` (step 7) |
| `eg`, `eu` | F32 [T·10][640], indexed by entry | `strata_moe_gu_<fmt>` |
| `eh`, `ehq` | F32 [T·10][640], Q81 [T·10][20] | `strata_moe_swiglu`, `strata_q_q8_1` |
| `parts` | F32 [T][10][2560], indexed by (row, rank) | `strata_moe_down_<fmt>` |
| `sg`, `su`, `sh`, `shq`, `shx`, `gl` | the shared expert's gate, up, h, its Q8_1, its output, the gate logit | step 8 |

**`strata_moe_route_r<NV>`**, grid T × 32, one simdgroup per row. NV = n_expert / 32: `r16` for 512 experts,
`r8` for 256. Expert e is value `e / 32` of lane `e % 32`.

```c++
t = tg.x;  l = logits_r + t*n_expert;
float v[NV];
for (i = 0; i < NV; ++i) v[i] = l[lane + 32*i];
m = -INFINITY;
for (i) m = fmax(m, v[i]);
m = butterfly_max(m);                                  // xor 16 -> 1 with fmax
s = 0.0f;
for (i) { v[i] = exp(v[i] - m); s = s + v[i]; }        // serial per lane
r = 1.0f / butterfly(s);
for (i) { v[i] = v[i] * r; if (isnan(v[i])) v[i] = -FLT_MAX; }
sel = 0.0f;  ssum = 0.0f;
for (rank = 0; rank < 10; ++rank) {
    best = v[0];  e = lane;
    for (i = 1; i < NV; ++i) if (v[i] > best) { best = v[i]; e = lane + 32*i; }
    for (o = 16; o > 0; o >>= 1) {
        ob = simd_shuffle_xor(best, o);  oe = simd_shuffle_xor(e, o);
        if (ob > best || (ob == best && oe < e)) { best = ob; e = oe; }
    }
    if (e % 32 == lane) { v[e / 32] = -INFINITY; ids[t][rank] = e; ssum = ssum + best; }
    if (rank == lane) sel = best;
}
ssum = fmax(butterfly(ssum), 6.103515625e-5f);         // 2^-14
inv = 1.0f / ssum;
if (lane < 10) wts[t][lane] = div ? sel / ssum : sel * inv;
```

`div` is a `setBytes` uint: 0 for the window and the drafter, 1 for the prompt path (section 6.6).

- `ssum` is accumulated by the lane that owns the winner, in selection order, and then summed by the butterfly, as
  upstream does (its comment: "Deliberately accumulate by WINNING EXPERT lane, not output rank"). The weights are a
  multiply by the reciprocal, not a division.
- Upstream's `expf`, `1.0f / x` and the fast-math reciprocal become `exp` and correctly rounded `/` (section 3.9).
- 256 experts (Coder): upstream's decode does not use this kernel. Its canonical `router_top10` sums the
  exponentials in FP64 and normalises in FP64. The Mac uses `r8`, a deviation (section 3.9): the ids are the same
  unless two probabilities tie in FP32 but not in FP64, and the weights differ in the last bits. On the prompt path
  upstream uses `route_kernel<8>`, which `r8` with `div` = 1 matches (section 6.6), so the deviation is decode's
  only.
- The draft layer always has 512 experts and uses `r16` (section 5.11).

**`strata_moe_plan`**, grid 1 × 128. It groups the window's T·10 (row, rank) entries by expert and, in tiered mode,
records which experts are not resident ([05](05-memory-and-residency.md), sections 5.3-5.4;
[06](06-engine-host-port.md), section 9.3). The layout, in `common/types.h`:

```c++
struct MoePlan {
    uint n_groups;          // distinct experts of the window
    uint n_entries;         // T * 10
    uint n_miss;            // groups whose expert is not resident (tiered mode; 0 in full mode)
    uint skip;              // n_miss == 0
    int  grp_e[80];         // expert id of group g, in first-appearance order
    int  grp_res[80];       // res[l][grp_e[g]] >= 0
    int  grp_start[81];     // entries of group g are grp_start[g] .. grp_start[g+1]-1
    int  ent_tok[80];       // row of entry a
    int  ent_dst[80];       // (row, rank) index t*10 + rank of entry a
};
```

```c++
n = T*10;  i = tid;
e = i < n ? ids_flat[i] : -1;  e_s[i] = e;                        // ids_flat[t*10 + rank]
barrier;
first = i;  rank = 0;  size = 0;
if (i < n) {
    for (j = 0; j < i; ++j) if (e_s[j] == e) { if (first == i) first = j; ++rank; }
    if (first == i) for (j = i; j < n; ++j) size += e_s[j] == e;
}
first_s[i] = first;  size_s[i] = (i < n && first == i) ? size : 0;
barrier;
if (i == 0) {
    gi = 0;  acc = 0;  miss = 0;
    for (j = 0; j < n; ++j) if (first_s[j] == j) {
        gidx_s[j] = gi;  gstart_s[gi] = acc;
        P.grp_e[gi] = e_s[j];  P.grp_start[gi] = acc;
        r = !tiered || res[l*n_expert + e_s[j]] >= 0;  P.grp_res[gi] = r;  miss += !r;
        acc += size_s[j];  ++gi;
    }
    P.grp_start[gi] = acc;  P.n_groups = gi;  P.n_entries = n;  P.n_miss = miss;  P.skip = miss == 0;
    if (tiered) for (j = 0; j < n; ++j) usage[l*n_expert + e_s[j]] += 1.0f;   // in entry order
}
barrier;
if (i < n) { a = gstart_s[gidx_s[first]] + rank;  P.ent_dst[a] = i;  P.ent_tok[a] = i / 10; }
```

- `tiered` and `l` come with `setBytes`; T comes from the parameter block. In full mode `res` is all resident and
  the plan is used only for the grouping.
- The plan only decides which entries share a weight read. Every output value depends only on its (expert, row)
  pair, so the grouping and its order cannot change any bit. The upstream production path builds its groups
  elsewhere; the Mac uses the device grouping of `group_resident_kernel`, which upstream uses for the draft layer.
  The Mac's draft layer itself does not group: with T = 1 its ten ids are distinct (section 5.11).
- The `MissResolver` reads `n_miss`, `grp_e` and `grp_res` after the layer's first event
  ([06](06-engine-host-port.md), section 9.3). The plan is in a Shared buffer for that reason.
- Prefill chunks group their C·10 entries with a GPU counting sort, `strata_pf_sort_*` (section 6.6).

**`strata_moe_gu_<fmt>`**, grid (160, T·10) × 256. One simdgroup per weight row: 8 rows per threadgroup, rows
0..639 of gate and 640..1279 of up, so 160 threadgroups cover both. The y index is the group; threadgroups past
`n_groups` exit.

```c++
g = tg.y;  if (g >= P.n_groups) return;
row = 8*tg.x + warp;  up = row >= 640;  r = up ? row - 640 : row;
wr = addr[l*n_expert + P.grp_e[g]].role[up ? 1 : 0] + r*gu_row;      // read after the handshake
for (a0 = P.grp_start[g]; a0 < P.grp_start[g + 1]; a0 += 4) {
    n = min(4, P.grp_start[g + 1] - a0);
    for (c = 0; c < 4; ++c) col[c] = xq + P.ent_tok[a0 + min(c, n - 1)]*(2560/32);
    s[0..3] = row_dot_multi<fmt, 4>(wr, col, 2560);                  // the row family of section 5.8
    for (c = 0; c < n; ++c) if (lane == c) (up ? eu : eg)[(a0 + c)*640 + r] = s[c];
}
```

The validated engine `encode_moe_experts` binds this dispatch for T=1..8,
1..512 active experts and all eight listed gate/up formats. It checks Context
ownership, extents, alignment and input/output aliasing, and requires an
ordered dispatch boundary. Each kernel checks the plan's row/group bounds,
expert IDs and token indices before reading weights or activations. The plan
must be produced for the same rows, active expert stride and layer by
`encode_moe_plan`; tiered callers queue the consumer only after `ExpertStore`
successfully resolves the plan. The partition and selected role addresses must
still come from trusted engine outputs and remain valid until command
completion. The production Window scheduler remains open, so this API and its
kernel twins do not establish end-to-end MoE inference.

`encode_moe_swiglu` validates read-only gate/up F32 views and a disjoint output
for routed expert entries (`rows * 10`) or shared-expert rows, with `rows` in
1..8. It dispatches `strata_moe_swiglu` or `strata_shx_swiglu` using three
256-thread groups to cover all 640 columns. Keep it ordered after the gate/up
projection and before Q8_1 quantization of the resulting rows.

The IQ4_XS specialization uses ten 136-byte blocks per row. Lanes traverse
`k = lane; k < 80; k += 32`, with `block = k / 8` and `part = k % 8`.
Each part decodes its 6-bit scale from `scales_l` and `scales_h`, decodes the
low and high nibbles through `kvalues_iq4nl`, multiplies the integer dot by
`scale_code - 32`, then applies `fma(float(d) * ds.x, float(scaled_dot), sum)`.
Its standalone mat-vec twin and grouped gate/up twin share one literal packed
block and a position-41 one-hot answer of 215265, isolating nibble order,
subblock scale assembly and Q8_1 lane mapping.

**`strata_moe_down_<fmt>`**, grid (320, T·10) × 256, the same structure over the 2560 down rows: the weight row is
`addr[...].role[2] + r·d_row`, the columns are `ehq + (a0 + c)·(640/32)`, and entry a writes
`parts[P.ent_dst[a]·2560 + r]`.

The validated engine `encode_moe_down` binds the Q2_0 or IQ4_NL variant for
T=1..8 and 1..512 active experts. It checks Context ownership, extents,
alignment and input/output overlap, and requires an ordered dispatch boundary.
The GPU checks the plan's row and group bounds, expert IDs, and every grouped
output destination before reading or writing. The plan and selected down-weight
addresses must still come from trusted engine outputs and remain valid through
command completion. Keep this dispatch ordered after the routed SwiGLU output
has been quantized as Q8_1. The production Window scheduler is still open, so
this encoder does not establish end-to-end MoE inference.

The Q2_0 down specialization has ten 64-value blocks per row and two Q8_1
parts per block. It traverses 20 calls with `k = lane; k < 20; k += 32`,
decodes Q2_0 values as the two-bit code minus one, applies
`fma(d2 * ds.x, float(dot), sum)`, then uses the lane butterfly. It reads
activations in grouped-entry order and writes every result through `P.ent_dst`
back to the original row/rank slot. Its 80-group fixture permutes expert IDs,
uses distinct per-expert scales, and pins the position-0 one-hot answer to 127.

The IQ4_NL down specialization has 20 32-value blocks per row and two
16-value Q8_1 dot parts per block. It traverses 40 calls with
`k = lane; k < 40; k += 32`, where `block = k / 2` and `part = k % 2`;
each part decodes the low/high IQ4_NL nibbles through `kvalues_iq4nl`,
applies its own `fma(d * ds.x, dot, sum)`, then participates in the lane
butterfly. The activation remains in grouped-entry order and every result is
restored through `P.ent_dst`. Its 80-entry fixture permutes expert IDs, gives
the experts distinct half scales, exercises both partial dots with values
`-228600` and `131445`, and checks the GPU's fixed answers exactly. At least
one fixture scale distinguishes the required split-FMA result from one FMA
over the combined integer dot.

- `row_dot_multi` is the row family of section 5.8 with the column map replaced by the entry list: lanes are strided
  over the calls of the row, every column gets its own accumulator, and the butterfly follows. The passes of four
  are upstream's `GRP_NC = 4`. Each column is bitwise equal to a one-column `row_dot` (upstream
  `iq_multi_parity.cpp`; the Mac's `mv_multi_eq_single` includes the expert kernels).
- `gu_row = (2560 / qk) · block_bytes` and `d_row = (640 / qk) · block_bytes` come with `setBytes`. Gate and up of a
  layer share one format; down has its own (Q2_0 or IQ4_NL). The host picks the two kernels per layer from the
  layer's types.
- For a Q2_0 down row, `nb·ipb` = 10 · 2 = 20 calls, so lanes 20-31 contribute zeros, as upstream's do.
- No threadgroup memory.

**`strata_moe_swiglu`**, grid (3, T·10) × 256: `i = 256·tg.x + tid`, `a = tg.y`; if `i < 640`,
`eh[a][i] = silu_k(eg[a][i]) * eu[a][i]`. Upstream `swiglu_entries_kernel` computes
`(g / (1.0f + __expf(-g))) * up`, so only `__expf` changes (section 3.9). `strata_q_q8_1` then quantises the T·10
rows of `eh` into `ehq` (n = 640), matching upstream's `quantize_q8_1_rows` where its conversion is defined;
exceptional denormal cases follow the explicit policy in section 5.8.

**The shared expert** (step 8), upstream `shared_expert_multi`:

| Order | Kernel | Computes |
| --- | --- | --- |
| 1 | `strata_mv_<fmt>_n<NC>` (`ffn_gate_shexp`, 2560 → 640) on `xq` | `sg` |
| 2 | `strata_mv_<fmt>_n<NC>` (`ffn_up_shexp`, 2560 → 640) on `xq` | `su` |
| 3 | `strata_shx_swiglu`, grid (3, T) × 256 | `sh[t][i] = silu_k(sg[t][i]) * su[t][i]` |
| 4 | `strata_q_q8_1(sh)`, n = 640 | `shq` |
| 5 | `strata_mv_<fmt>_n<NC>` (`ffn_down_shexp`, 640 → 2560) on `shq` | `shx` |
| 6 | `strata_mv_bf16_b256_n<NC>` (`ffn_gate_inp_shexp`, 2560 → 1) on `mixed` (F32) | `gl[t]` |

- Upstream's swiglu here is `__fdividef(g, 1.0f + __expf(-g)) * up` and its gate is
  `__fdividef(1.0f, 1.0f + __expf(-g))`. Both become the helpers of section 3.6.
- Upstream quantises `mixed` again for the shared expert with `native_quantize_q8_1`. The Mac reuses step 7's `xq`
  (section 5.1, first note under the table).
- Upstream then scales the shared output in place (`scale_rows_kernel`: `out[i] *= g[t]`). The Mac does that
  multiply inside the combine, rounded the same way.
- `strata_shx_swiglu` and `strata_moe_swiglu` are one function body with two host names; they differ only in the
  buffers bound.

**`strata_moe_combine`**, grid (10, T) × 256, one output value per thread (upstream `native_moe_combine_multi`):

```c++
t = tg.y;  d = 256*tg.x + tid;                       // 10 * 256 = 2560, no tail
p = parts + t*10*2560;  w = wts + t*10;
s = p[d] * w[0];
for (k = 1; k < 10; ++k) s = fma(p[k*2560 + d], w[k], s);
g = sigmoid_k(gl[t]);
sh = shx[t*2560 + d] * g;                            // upstream scale_rows_kernel
bo[t*2560 + d] = s + sh;
```

- The combine runs in a fast-math file upstream, where nvcc contracts `sum += parts·w` into
  `fma(parts, w, sum)` and keeps the first product and the final `+ shared` separate. L-9 checks this against the
  SASS of `native_moe.cu`.
- `g` is computed by every thread from the same `gl[t]`, so all threads use the same value.
- `bo` is the FFN block output that the FFN write of section 5.4 adds to the residual.

The validated engine `encode_moe_combine` binds the routed parts, route weights,
shared-expert output and gate logits as read-only inputs for T=1..8. It validates
Context ownership, extents, alignment and output/input overlap, and requires an
ordered dispatch boundary. The typed encoder and scalar oracle cover the
standalone combine and the composed shared-expert chain; production Window
scheduling remains unfinished.

`encode_moe_shared_ffn` composes the shared path in one ordered Encoder segment:
read-only Q8_0 gate/up projections, shared SwiGLU, Q8_1 quantization, the
read-only Q8_0 down projection, the read-only BF16 gate projection, then routed
and shared combine. It accepts the Q8_1 mixed activation and F32 `mixed`
views, the shared weight views, and already available routed parts and weights.
It derives the smallest supported compiled column width for T=1..8. Before
encoding, it validates Context ownership, extents, alignment, the live-row
metadata, and pairwise separation of every writable scratch, output and
padded-column sink from all inputs and each other. Read-only inputs may alias.
The standalone form discards the whole unsubmitted command buffer on failure;
the caller-owned form must propagate false from its enclosing encode body.
Tiered callers must wait for successful plan resolution and routed expert
production before combining. This coordinator does not build the route or
expert plan, resolve misses, write the FFN residual, or wire the production
Window scheduler, so it does not establish end-to-end inference.

`encode_moe_shared_ffn_write` extends this composition with the layer's
`strata_gr_post` FFN write. It uses the combine output as `block_out` and
updates the four-stream residual in place using the supplied FFN injection.
Residual and injection views must be disjoint from every shared-FFN input,
scratch, output and sink before the first dispatch. The coordinator still
assumes route weights and routed parts are ready, and does not add the Window
layer scheduler or tiered resolver. The caller-owned form must propagate
`false` from its enclosing Encoder callback so an error after an earlier stage
discards the whole segment. Keep the read-only FFN inputs, injection and
initial residual contents unchanged until command completion, and do not access
the written scratch or outputs before completion.

`encode_moe_routed_ffn` composes the routed path for T=1..8 in this order:
grouped gate/up, routed SwiGLU, Q8_1 quantization of the 640-wide hidden rows,
then grouped down projection into original route order. It validates every
scratch extent and alignment, Context ownership, and scratch/input and
scratch/scratch disjointness before encoding, and requires the address,
activation and plan input views to be mutually disjoint. Full-mode callers may
place the plan dispatch and this composition in the same ordered Encoder
segment. A tiered caller must wait for successful `ExpertStore` resolution
before submitting this consumer; it must not queue the consumer behind an
unresolved plan. Propagate caller-owned failures through the enclosing
`Encoder::encode` callback so prior dispatches are discarded. This composition
requires the plan partition and selected expert addresses to come from trusted
engine outputs and remain valid until GPU completion; it does not read back or
fully validate GPU-produced plan arrays. It does not include the shared
expert, combine, later Window layers or state commit, and does not establish
end-to-end MoE inference.

`encode_moe_routed_shared_ffn_write` composes the routed expert FFN, shared
expert FFN, routed/shared combine and GR residual write in one ordered
T=1..8 Encoder segment. It preflights both component chains and their
cross-stage aliases before the first dispatch, comparing the active byte
extents consumed or written at each stage. The shared Q8_1 input and routed
parts must start at the routed producer's corresponding views; other routed
outputs must be disjoint from active shared inputs, outputs, residual and
injection ranges. Full-mode callers may encode plan production before this
composition in the same ordered segment. Tiered callers must wait for
successful `ExpertStore` resolution before submitting it. The compositor does
not build routes or plans, resolve misses, or integrate the production Window
scheduler, and its synthetic tests do not establish end-to-end inference.

`encode_moe_routed_shared_ffn` uses the same validated routed/shared stages
and stops after combine, leaving the result in `shared.output`. Layer 0 uses
this form because its FFN write remains pending: the PLE entry applies it
when PLE is enabled, and otherwise the next layer's attention GR read applies
it. `encode_moe_routed_shared_ffn_write` remains available for paths whose
writeback is due at the current stage. The routed/shared GPU fixture checks
that the pending form matches the combine oracle and leaves residual and
injection bytes unchanged for T=1..8.

**What the FFN reads per layer, T = 1:** the router (2.6 MB for 512 experts), 10 experts (13.8 MB for Q2_0, three
slabs of 460,800 B each), and the shared expert. For T > 1, rows that share an expert share its read, which is why
the groups exist.

### 5.11 The MTP drafter

Upstream: `mtp.cpp:245-300` (buffers), `:413-580` (`record_forward`), `:621-671` (`capture_round`, `capture_step`),
`:682-770` (`prefill`) and `:771-838` (`draft`, `draft_first`); `verify_kernels.cu:233-380`;
`quantize_act.cu:100-128` (`quantize_q8_0_scaled_kernel`); `s2_expert_grouped.cu:64-99` (`row_dot_s2_q8`),
`:750-990` (the grouped S2 kernels) and `:1044-1132` (`group_resident_kernel`, `moe_grouped_s2`);
`shared_expert.cu:210-314`; [K4 section 3](audit/K4-ple-mtp-head-sampler.md). The model is
[03](03-model-reference.md), section 7, and its files are section 4.4 there. Constants: N = 2560, HC = 4, 24 query
heads and 2 KV heads of 256, k = 10, n_ff = 640, and 512 experts for every model.

The draft layer runs in two forms, both upstream's `record_forward`:
- the **K/V-only pass** over T = 1..8 rows, stages 1-9 of the table below. It writes the drafter's K and V at T
  cells and stops.
- the **full layer** over one row, all stages. It writes one cell's K and V, attends over the window, and produces
  one draft token and its probability.

At cell c a row takes the main model's final residual at c and the token at c + 1, and predicts the token at c + 2
([03](03-model-reference.md), section 7). Most stages reuse the kernels of sections 5.2-5.10 with the drafter's
weights. Nine kernels are new; they follow the stage table.

**Buffers.** The drafter has its own scratch region, carved once at bind for T = 8 and kept apart from the window
arena (section 4.4). It is upstream's `arena_` (`mtp.cpp:252-286`) without the grouping buffers. Every named region
starts on a 256-B boundary. `MtpBindLayout` exposes bounded views into this allocation. Runtime-bound
ordered smoke coverage now reaches stages 1–20; stages 21–22 have runtime view bindings and isolated GPU tests.
Production scheduler integration remains open.

| Buffer | Shape | Written by |
| --- | --- | --- |
| `mtp_rin` | F32 [8][4][2560] | the round's copy, `strata_mtp_select` |
| `mtp_tok` | int32 [1] | `strata_mtp_select`, or the CPU tail in coupled mode |
| `emb`, `en`, `e2` | F32 [8][2560] | stages 1-3 |
| `hn`, `h2` | F32 [32][2560]: row 4t + c is stream c of row t | stages 4-5 |
| `xq` | Q81 [32][80] | every `strata_q_q8_1` of the layer |
| `R` (`residual` region) | F32 [8][4][2560] | stage 6, the MLP read (with `apply`), stage 19 |
| `xn`, `lo`, `rs`, `inj`, `inj2`, `mixed` | F32 [8][10240], F32 [8][320], F32 [8][4], F32 [8][4], F32 [8][4], F32 [8][2560] | stages 7, 15 and 20 |
| `gate` | F32 [1][10240] | stage 20 |
| `kcur`, `vcur` | F32 [8][2][256] | stage 8 |
| `q_full`; `qcur`, `attn`, `attn32` | F32 [24][512]; F32 [24][256] each | stages 10-14 |
| `mtp_ids` | int32 [cap] | bind initializes the identity mapping; stage 12 refreshes the active prefix on every pass, writing either the finite suffix or full identity mapping |
| `att_scratch` | `n_chunks · 24 · 258 + 64` floats (section 5.7) | `strata_attn_chunk_<mode>` |
| `bo`, `y` | F32 [2560] each | `o_proj`; the combine |
| `logits_r`, `ids`, `wts` | F32 [512], int32 [10], F32 [10] | stage 16 |
| `mq`, `ms` | char [2560], F32 [80] | `strata_q_q8_0_scaled(mixed)` |
| `eg`, `eu`, `eh` | F32 [10][640] each, indexed by rank | `strata_mtp_gu_s2`, `strata_mtp_swiglu_s2` |
| `hq`, `hs` | char [6400], F32 [200] | `strata_q_q8_0_scaled(eh)` |
| `parts` | F32 [10][2560] | `strata_mtp_down_s2` |
| `sg`, `su`, `sh`, `shq`, `shx`, `gl` | as section 5.10, one row | stage 18 |
| `sample` | F32 [2560] | the final mixer |
| `hl` | F32 [n_out] | the draft head |
| `out_ids`, `probs` | int32 [1], F32 [1] | stage 22 |
| `mtp_out`, `mtp_prob` | int32 [8], F32 [8], in a Shared buffer the host reads | `strata_mtp_select`, or the CPU tail |
| `cand`, `ring` | coupled mode only (section 5.12) | |

The named normal-mode regions follow this table's order, with the device-side `draft_vocab_ids` table immediately
after `mtp_ids`. The `hl` region has `n_out` logits: the draft-vocabulary count when present, otherwise all
248,320 tokens. `MtpRuntimeStorage` initializes `mtp_ids` to the identity mapping and copies the inspected
vocabulary IDs into their device-side region before it publishes the owner. The arena baseline is 15,874,804 bytes
at `cap = 32,768` and 106,299 logits, before the
`draft_vocab_ids` table; geometry adjusts that baseline by the logits-row size, the vocabulary table and the
capacity-dependent `mtp_ids` plus `att_scratch` bytes. The layout builder enforces the shape-dependent reserved
tail exactly; it is 128,192 bytes at the 32,768-cell / 106,299-ID geometry, with only the 256-byte padding after
the ID table, logits and attention scratch changing it for other supported shapes. The tail remains reserved and
has no kernel binding. Resolve its purpose against the complete MTP ABI before enabling any drafter dispatch.

`cap` is upstream's `cap_ = round_up_64(window > 0 ? window : max_cells)`, where the drafter's `window` is
`--mtp-window` if it is below `max_cells` and 0 otherwise. `n_chunks = cap / 64`: 512 at the default 32,768. The
region is about 16 MB at that window, and 106 MB with no window at 262,144 cells, almost all of it `att_scratch`
([05](05-memory-and-residency.md), section 2.3).

The drafter's persistent K/V allocation reserves `round_up_4(--max-context)` cells for the shared four-cell row
mapping. Per-step `n_kv` remains bounded by the requested logical context, so the rounded tail is spare capacity only.
Unlike upstream's generic `QsaState` arena, the Metal drafter has no indexer consumer; it does not allocate that
arena's pooled-key buffer or indexer rows. It uses the RoPE table owned by the main `KvStore`.

**Parameter blocks.** The drafter's `MtpParams` (section 4.2) live in their own Shared array of 256-B slots:
slot 0 is the round's K/V-only pass (block K), slots 1-7 are the full-layer passes F0-F6, and one slot per
prefill group of up to eight rows follows. The planner reserves `8 + max(1, ceil(prefill_chunk / 8))` slots; with no
GPU prefill chunk it still reserves one group slot because MTP catch-up runs after short-read decode windows. The
runtime owner allocates the array separately from bind scratch and exposes one bounded slot view at a time. The
complete array is zero-initialized before publication.
Every kernel of a pass binds `buffer(0)` at its block's slot. `steps` is bound at the
block's `step`, `rows` at its `T` (or at `hid[k]`, stage 5), and `tokens` at its `tokens` for the K/V-only pass or at
`mtp_tok` for the full layer. For a round at position p with a accepted and window tokens `outv`, the host
writes, with `row(c) = {c, c + 1, (c + 1) / 4, window > 0 ? min(c + 1, window) : c + 1}`:

| Block | `T` | Other fields |
| --- | --- | --- |
| K | the window's T | `hid[k] = min(8, 4T − 8k)` for k < ⌈T/2⌉, else 0; `a`; `tokens[t] = outv[t]`, the token at cell p + t + 1; `step[t] = row(p + t)` |
| Fj | 1 | `j`; `step[0] = row(p + a + j)`; `samp`: in coupled mode the target row that will verify this draft (section 5.12), in greedy mode `greedy = 1` and `hist_len = 0` |

`n_bid` is unused (the drafter has no indexer) and is written as upstream writes it.

**The stages** (upstream `record_forward`; the K/V-only pass stops after stage 9):

| Stage | Kernels | Weights | Reads → writes |
| --- | --- | --- | --- |
| 1 | `strata_embed_rows_<fmt>` | the main model's `token_embd` | `tokens` → `emb` |
| 2 | `strata_rms_norm_w`, `n_cols` 2560, gp 1 | `pre_fc_norm_embedding` | `emb` → `en` |
| 3 | `strata_q_q8_1` (n 2560), `strata_mv_q8_0_n<NC>` (2560 → 2560) | `fc_embedding` | `en` → `xq` → `e2` |
| 4 | `strata_rms_norm_w`, `n_cols` 10240, gp 1 | `pre_fc_norm_hidden` | `mtp_rin` → `hn` |
| 5 | `strata_q_q8_1` (n 2560, 4T rows), then ⌈T/2⌉ passes of `strata_mv_q8_0_n<hid[k]>` | `fc_hidden` | `hn` → `xq` → `h2` |
| 6 | `strata_mtp_add_streams` | | `h2`, `e2` → `R` |
| 7 | `strata_gr_norm`, `strata_gr_down`, `strata_gr_up`, apply 0 | `attn_hyper_connection.*` | `R` → `inj`, `mixed` |
| 8 | `strata_q_q8_1` (n 2560), `strata_mv_q8_0_n<NC>` (2560 → 512) twice | `self_attn.k_proj`, `v_proj` | `mixed` → `xq` → `kcur`, `vcur` |
| 9 | `strata_qsa_norm_rope` `k` (2T rows); O-class: `strata_qsa_fwht256` on K and V if `kv_rot`; `strata_qsa_kv_append_<mode>` | `self_attn.k_norm` | → the drafter's K and V at `step[t].pos` |
| 10 | `strata_mv_q8_0_n1` (2560 → 12288) on stage 8's `xq` | `self_attn.q_proj` | → `q_full` |
| 11 | `strata_qsa_norm_rope` `q` (24 rows, stride 512 in, 256 out); O-class: `strata_qsa_fwht256` if `kv_rot` | `self_attn.q_norm` | `q_full` → `qcur` |
| 12 | `strata_mtp_window_ids`, every attention pass | | → `mtp_ids` |
| 13 | `strata_attn_chunk_<mode>` (n_chunks, 2, 1), `strata_attn_merge` (24, 1); O-class: `strata_qsa_fwht256` on `attn` if `kv_rot` | | `qcur`, the pools, `mtp_ids` → `attn` |
| 14 | `strata_qsa_gate`; `strata_q_q8_1` (n 6144); `strata_mv_q8_0_n1` (6144 → 2560) | `self_attn.o_proj` | `attn`, `q_full` → `attn32` → `xq` → `bo` |
| 15 | `strata_gr_norm`, `strata_gr_down`, `strata_gr_up`, apply 1 with `bo` and `inj` | `mlp_hyper_connection.*` | `R` → `R`, `inj2`, `mixed` |
| 16 | `strata_mv_bf16_b256_n1` (2560 → 512), `strata_moe_route_r16` | `mlp.gate` | `mixed` → `logits_r` → `ids`, `wts` |
| 17 | `strata_q_q8_0_scaled` (n 2560), `strata_mtp_gu_s2`, `strata_mtp_swiglu_s2`, `strata_q_q8_0_scaled` (n 6400), `strata_mtp_down_s2` | the S2 blobs | `mixed` → `mq`, `ms` → `eg`, `eu` → `eh` → `hq`, `hs` → `parts` |
| 18 | `strata_q_q8_1(mixed)`, then the shared expert of section 5.10 | `mlp.shared_expert.{gate,up,down}_proj` (Q8_0), `mlp.shared_expert_gate` (BF16) | `mixed` → `xq` → `sg`, `su` → `sh` → `shq` → `shx`; `mixed` → `gl` |
| 19 | `strata_moe_combine` (10, 1) into `y`; `strata_gr_post` with `bo = y`, `inj = inj2` | | `parts`, `wts`, `shx`, `gl` → `y`; the write into `R` |
| 20 | the head mixer of section 5.4: `strata_rms_norm_w` (gp 4), `strata_mv_bf16_b256_n1`, `strata_gr_down_silu`, `strata_mv_bf16_b160_n1`, `strata_gr_pre_gated` | `hyper_connection_mixer.{hc_norm, input_mix_weight_down, input_mix_weight_up}` | `R` → `sample` |
| 21 | `strata_q_q8_1` (n 2560); `strata_mv_<headfmt>_n1_x` over `dvocab` with a draft vocabulary, else `strata_mv_<headfmt>_n1` | the main head (the draft head, below) | `sample` → `xq` → `hl` |
| 22 | greedy: `strata_samp_greedy` (one row), `strata_mtp_row_top_prob`, then `strata_mtp_map_ids` with a draft vocabulary; coupled: section 5.12 | | `hl` → `out_ids`, `probs` |

- The mat-vecs of stages 3 and 8 take NC = the smallest of 1, 2, 4, 8 that is ≥ T, chosen at encode time; the
  padding columns write the sink (section 4.4). The full layer has T = 1 and NC = 1 throughout.
- Stage 5: upstream quantises and multiplies 8 rows at a time (`c0 += 8`, `nc = min(8, 4T − c0)`). The Mac
  quantises the 4T rows once (quantisation is per row) and then runs pass k on `xq + 8k` rows into `h2 + 8k` rows,
  with NC = `hid[k]`, 8 or 4. Every column is bitwise an NC = 1 call (section 5.8), so the passes only bound NC.
- `encode_mtp_input_projection` composes stages 1–5. The K/V-only form gathers
  T token IDs from `MtpParams::tokens`; the full-layer form takes one selected
  token and requires T=1. It validates `hid[k] = min(8, 4T − 8k)` (or zero
  after the active rows), uses one Q8_1 scratch view first for stage 3 and then
  for stage 5, and permits only exact in-place aliasing between `emb` and `en`.
  K/V-only token IDs must not carry the reserved image-row bit, and a full-layer
  selected-token view must not overlap its `MtpParams` block.
  Stage 4 normalizes T rows of width 10240 before stage 5 quantizes the same
  data as 4T rows of width 2560. Serial/concurrent twins cover T=1, 2, 3 and 8,
  the selected-token route, the four mat-vec widths and hidden-pass tails,
  scalar norm/quantization/mat-vec oracles, guards, invalid IDs/metadata,
  output aliases, caller-owned rollback and group-boundary refusal. Production
  `MtpRuntimeStorage` binding now covers stages 1–22, while scheduler
  integration and inspected-model drafter parity remain open. The synthetic
  runtime-bound ordered chain covers stages 1–20; stage-21/22 bindings are
  checked against runtime-owned views and their GPU kernels have isolated
  numerical tests in `metal_mtp_test`.
- Stage 9 reads the RoPE position from `steps[t].pos`; upstream reads the same cell from its `pos_` array.
- `encode_mtp_attention_kv_append` composes stage 9 from a live `MtpParams`
  slot, the stage-8 K/V rows, the MTP K norm, and the main model's shared RoPE
  view. It validates the canonical MTP KV layout and its four views before
  dispatch, requires unique in-capacity positions, and uses ordered QSA
  norm/RoPE, optional FWHT and format-specific append encoders. `kv_rot` is
  explicit at the encoder boundary; `bind_attention_kv_append` resolves the
  owned cache, norm and RoPE views using the pass mode. Runtime-bound stages
  9–14 use `bind_attention_pass` to return one immutable bundle sharing
  parameters, scratch handoffs and the rotation mode across the pass. The caller
  keeps `MtpParams`, its active
  StepRows, current K/V, K norm and RoPE alive and unchanged through GPU
  completion; the encoder's in-place K and optional V transforms are the
  writes made by this stage.
- Stage 10 reuses stage 8's `xq`, as upstream does: nothing writes `xq` between them.
- `encode_mtp_attention_query_projection` implements only stage 10's full-layer
  T=1 pass. It consumes stage 8's existing Q8_1 `xq` and the Q8_0
  `self_attn.q_proj` matrix without requantizing, writes the full 24-head ×
  512-value `q_full` tensor, and accepts the first float of the `bo` region
  for the mat-vec ABI's required sink. Keep `MtpParams`, `xq` and the weights
  alive and unchanged until GPU completion, and keep every view including
  `q_full` and the sink alive through completion; stage 14 later reuses `bo`.
  `bind_attention_query_projection` resolves these views from runtime storage;
  production drafter scheduling and model-backed parity remain open.
- Stage 11's `encode_mtp_attention_query_norm` reads the full `q_full`
  tensor, the MTP `self_attn.q_norm`, active `StepRow` and shared RoPE table,
  then writes the compact 24×256 `qcur` view. It requires T=1 and a valid
  RoPE position. O-class callers set the explicit `kv_rot` flag to append
  `strata_qsa_fwht256` after query normalization/RoPE in the same ordered
  Encoder segment. Every view must remain alive through GPU completion;
  `MtpParams`, `q_full`, `q_norm` and RoPE remain unchanged.
  Runtime-bound callers use `bind_attention_pass` and
  `encode_mtp_attention_pass` to validate shared parameter views, scratch
  handoffs and rotation mode, then encode stages 9–14 in order. The standalone
  stage-11 buffer remains available to low-level callers, who own mode
  consistency when composing stages manually. Production scheduling and
  model-backed parity remain open.
- Stage 12's `encode_mtp_window_ids` refreshes IDs on every attention pass.
  It requires T=1, positive `step[0].n_kv`, and a host-published width equal
  to `n_kv` when `window == 0`, or `min(n_kv, window)` otherwise. A zero
  window writes the full identity prefix; a positive finite window must fit
  the ID region and writes absolute KV positions
  `[max(0, n_kv - window), n_kv)`. The inactive suffix is untouched. The
  active prefix must fit the region; generated positions may exceed the
  finite window because they index the drafter's larger KV history.
  Refreshing the identity prefix after a finite-window pass prevents
  stale IDs from affecting a later unwindowed pass. Keep params and the ID
  view alive through completion, and keep params unchanged while the kernel
  reads it. The runtime-bound attention pass always runs stage 12.
- Stage 13's `encode_mtp_attention` composes the QSA chunk pass, fixed-order
  merge, and optional output FWHT in one ordered Encoder segment. It requires
  T=1, a canonical MTP KV layout and its exact views, and the bind-layout ID
  stride (a positive multiple of 64). The ID stride may exceed the rounded KV
  cell capacity by the bind layout's padding; `StepRow.width`, rather than
  either capacity, bounds the live IDs. QSA mode follows the MTP store's
  **storage** format: fp16, int8, or q4_0; a K8v4 request uses MTP's int8
  storage mapping and therefore dispatches int8 attention. `kv_rot` must agree
  with stages 9 and 11: fp16 forbids it, q4_0 requires it, and int8 accepts
  either choice. The named `att_scratch` view must cover
  `ceil(id_row_stride / 64) * 24 * 258 + 64` floats. Its active prefix is
  split into chunk accumulators, maxima, and denominators; the final 64 floats
  remain reserved and untouched. The encoder rejects unsafe aliases and an
  open concurrent group so chunk→merge→FWHT dependencies retain their
  barriers. The runtime-bound attention pass supplies this encoder with the
  planned ID stride, KV views and scratch. Its synthetic dispatch does not
  establish production scheduling or model-backed attention parity.
- Stage 14's `encode_mtp_attention_output_projection` composes the QSA gate,
  Q8_1 quantization of the 6144-wide gated attention row, and the Q8_0
  6144→2560 O projection in one ordered Encoder segment. It requires T=1 and
  uses `MtpParams.T` as the projection's active-row metadata. The caller
  supplies stage 13's attention output, stage 10's `q_full`, the named Q8_1
  scratch, O-projection weights, `bo`, and the required sink. The wrapper
  validates extents, Context ownership and aliases before dispatch, and
  refuses an open concurrent group. The runtime-bound attention pass resolves
  the O-projection views and encodes this stage after attention. Its synthetic
  dispatch does not establish production scheduling or model-backed parity.
- Stage 15's `encode_mtp_feed_forward_read` composes the GR normalization,
  down/injection and up/mix stages with `apply=true`. It reads the residual
  `R`, stage-14 `bo`, stage-7 `inj`, normalization and GR weights, then writes
  the updated `R`, `inject_out` and `mixed` views. It requires T=1, keeps the
  MTP parameter block disjoint from every active GR view, and preserves
  normalize→down→up ordering at a closed concurrent-group boundary.
  `bind_feed_forward_read` resolves stage 14's output, stage 7's injection,
  residual and scratch/weights from runtime storage. This synthetic binding
  does not establish production scheduler integration or model-backed parity.
- Stage 16's `encode_mtp_router` composes the BF16 2560→512 router projection
  with the stable top-10 route selection. It reads the stage-15 `mixed` row,
  the BF16 `mlp.gate` matrix and T=1 parameters, then writes `logits_r`, `ids`
  and normalized route weights. The MTP wrapper checks all active extents and
  cross-stage aliases before dispatch, uses the full 512-expert route with
  `divide_weights=false`, and requires an ordered concurrent-group boundary.
  `bind_router` resolves the mixed row, BF16 router matrix and route outputs
  from runtime storage. This synthetic binding does not establish production
  scheduler integration or model-backed parity.
- Stage 17's `encode_mtp_routed_experts` quantizes stage 15's `mixed` row
  with scaled Q8_0, computes gate/up projections for the ten stage-16 route
  IDs from S2 expert weights, applies SwiGLU, quantizes its ten rows, and
  computes each down projection into `parts`. It accepts either a contiguous
  full image or ordered expert windows that cover all 512 IDs. For each
  window, the S2 kernels compare device-produced IDs with its expert range and
  address local blobs; the encoder validates exact coverage, active extents
  and aliases before dispatch. Quantization, per-window gate/up dispatch,
  SwiGLU, hidden quantization and per-window down dispatch remain ordered.
  `MtpRuntimeStorage::bind_routed_experts` supplies the owner's bounded
  windows and named scratch. The serial/concurrent twin compares full-image
  and split-window outputs with independent scalar references, and the
  runtime storage smoke chain dispatches through stage 17. These synthetic
  checks do not establish production scheduling or inspected-model parity.
- Stages 18–19's `encode_mtp_shared_ffn_write` quantizes the `mixed` row to
  Q8_1, runs the shared expert, combines its sigmoid-gated output with stage
  17's `parts` and stage 16's route weights, then applies the combined vector
  to the MTP GR residual. It requires T=1, keeps the full parameter block
  disjoint from every active view, rejects the Q8_1 output overlapping any
  later read input or writable scratch, and delegates the ordered shared-FFN,
  combine and GR-post chain to the validated MoE encoder. The caller keeps
  every view alive and does not modify inputs, including residual and
  injection, until GPU completion; the stage reads and then writes residual
  in place. The serial and concurrent twin checks Q8_1 against the scalar
  oracle, output parity against the direct shared-FFN/write path, T and alias
  rejection, concurrent open-group refusal and caller-owned rollback.
  Production runtime-view binding and inspected-model parity remain open.
- Stage 20's `encode_mtp_final_mixer` composes the validated Window head mixer
  with `MtpParams.T` as its active-row source and requires T=1. It reads the
  stage-19 residual and the three mixer weights, then writes `sample`; the
  caller keeps every view alive and does not access or reuse any active view
  until GPU completion. The serial/concurrent GR twins compare against the
  scalar head-mixer oracle, preserve the residual and model inputs, and cover
  parameter/residual and residual/weight aliases, caller-owned rollback, and
  open-group refusal in concurrent mode. This isolated wrapper does not bind
  `MtpRuntimeStorage` or establish model-backed parity.
- Stage 21's `encode_mtp_draft_head` quantizes the T=1 `sample` row to Q8_1
  and projects through the read-only output head to one logit per draft-vocab
  row. An optional immutable row map selects an unsorted subset of the full
  vocabulary; without it, rows are projected in full-vocabulary order. The
  host validates map range and uniqueness once and returns a typed token
  reused by stages 21 and 22, avoiding a full map scan and sort per draft
  token. The indexed Q5_K, Q6_K and IQ4_XS mat-vec Encoder overload reuses the
  existing quantizer and format-specific dot kernels. Active buffers are
  range- and alias-checked before dispatch, and the caller keeps all views
  alive and avoids CPU access or reuse until GPU completion.
  Serial/concurrent fixtures compare full and indexed logits for all three
  formats against independent Q8_1 plus scalar mat-vec oracles, check guards
  and immutability, and exercise late rollback. Production `MtpRuntimeStorage`
  binding and inspected-model parity remain open.
- Stage 22's `encode_mtp_draft_sampling` consumes the stage-21 logits and
  requires a T=1 greedy sampler with zero-offset, empty history. It computes
  the selected row's softmax probability over the active head rows, then
  optionally maps the subset row to its full token ID through the same
  validated vocabulary token. The standalone wrapper is transactional; the
  caller-owned Encoder form requires a closed concurrent group and propagated
  failure. Serial/concurrent fixtures verify full and subset IDs, probability
  parity, map-before/after ordering, immutable inputs, invalid sampler state,
  aliases, rollback and group-boundary refusal. This isolated sampling step
  does not bind `MtpRuntimeStorage`, implement the drafter scheduler, or
  establish inspected-model parity.
- Stages 2 and 4 are upstream's `native_qsa_rms_norm_weighted`, whose `norm<1024>` (for rows of 1024 or more) is
  the same arithmetic as `native_gr_rms_norm_weighted`. `strata_rms_norm_w` serves both with `n_cols` and the gamma
  period `gp` as `setBytes` constants (section 5.4).
- Stage 15 folds the attention write (`bo`, `inj`) into `R` and reads it, as the main layers' FFN read does.
- Stage 18: upstream's `shared_expert` quantises `mixed` into its Q8_1 scratch itself; the Mac's `strata_q_q8_1`
  is that step. Its gate is `native_scalar_sigmoid_kernel` and then `scale_kernel` (`out[i] *= g`), the same
  operations as the main model's `scale_rows_kernel`, which the Mac's combine performs.
- Stage 20, the final mixer, is upstream's native `gr_read` without inject, the arithmetic of the head mixer. It is
  bitwise, not a deviation.
- Upstream builds the expert groups on the device (`moe_group_resident`). In the full layer T = 1, so the ten ids
  are distinct, every group has one entry, and the first-appearance order is the rank order. The Mac's S2 kernels
  index by rank and skip the grouping; the bits are the same.

**`strata_mtp_add_streams`**, grid (40, T) × 256 (upstream `add_streams_broadcast_kernel`):

```c++
t = tg.y;  i = 256*tg.x + tid;                      // i < 10240
R[t][i] = h2[t][i] + e2[t][i % 2560];               // h2[t] is the 4 streams of row t
```

The validated `encode_mtp_add_streams` host entry binds `MtpParams`, `h2`,
`e2` and `R` at indices 0..3. It reads `T` from the CPU-visible 256-byte
parameter slot, validates `1..8`, sizes the active extents, and dispatches
`(40, T) × 256`. Its standalone command-buffer form is transactional; callers
of the Encoder form must propagate failure from their enclosing
`Encoder::encode`. `encode_mtp_attention_read` composes stage 7's GR
normalize/down/up read with `apply=false`, keeping `R` unchanged while writing
`inj` and `mixed`; it rejects parameter aliases with any active GR view before
delegating to the GR chain. `encode_mtp_attention_kv_projection` reads T from
the same parameter block, quantizes `mixed` once to Q8_1, then encodes the
ordered Q8_0 key and value projections to `kcur` and `vcur`. It selects the
smallest supported NC at least T, shares the padded-column sink across those
ordered dispatches, and rejects parameter, input, output and sink aliases that
could corrupt later reads. These are isolated stage encoders only;
runtime-storage wiring, stages 1–22 as one production pass, and the complete
drafter remain open.

**`strata_q_q8_0_scaled`**, grid ⌈n/4096⌉ × 128, one 32-value block per thread. Upstream
`quantize_q8_0_scaled_kernel`, in a file without fast math. The Mac output is the codes `char[n]` and the FP32 scales
`float[n/32]`; upstream also writes a half `d` into each 34-B block, which the S2 kernels ignore when FP32 scales are
given, so the Mac does not store it.

```c++
b = 128*tg.x + tid;  if (b >= n/32) return;
xb = x + 32*b;
amax = 0.0f;
for (i = 0; i < 32; ++i) amax = fmax(amax, fabs(xb[i]));
s   = amax > 0.0f ? amax / 127.0f : 0.0f;           // correctly rounded
inv = s > 0.0f ? 1.0f / s : 0.0f;
scales[b] = s;
for (i = 0; i < 32; ++i) {
    t = xb[i] * inv;
    r = t + (t >= 0.0f ? 0.5f : -0.5f);             // rounded product, then the add (L-9)
    codes[32*b + i] = char(clamp(int(r), -127, 127)); // int() truncates toward zero
}
```

`n` is 2560 (`mixed` → `mq`, `ms`) or 6400 (the ten rows of `eh` → `hq`, `hs`), as a `setBytes` constant.

**The S2 blob** of expert e is at `mtp_blob + e · 1,382,400` in the `experts.bin` window, bound by buffer and offset
(section 4.3). Its layout ([03](03-model-reference.md), section 3.6):

| Part | Offset | Size | Row |
| --- | ---: | ---: | --- |
| gate/up codes | 0 | 819,200 | 1,280 rows of 640 B; slot i is gate row i / 2 if i is even, up row i / 2 if odd |
| down codes | 819,200 | 409,600 | 2,560 rows of 160 B |
| gate/up scales | 1,228,800 | 102,400 | 80 B per slot: one half per 64 values |
| down scales | 1,331,200 | 51,200 | 20 B per row |

A code byte holds four 2-bit codes, low bits first; a code q stands for q − 1 ∈ {−1, 0, 1, 2}. The chunk dot, the
body of upstream `row_dot_s2_q8` and `chunk_dot`, for 32 codes `cb[0..8)` against 32 activations `xw[0..8)` (words):

```c++
static inline float s2_chunk(device const uchar* cb, threadgroup const int* xw, float dw, float dx, float acc) {
    int s = 0, hx = 0;
    for (int j = 0; j < 8; ++j) {
        uint b = cb[j];
        int cw = int((b & 3u) | (((b >> 2) & 3u) << 8) | (((b >> 4) & 3u) << 16) | (((b >> 6) & 3u) << 24));
        s  = dot4_i8(cw, xw[j]) + s;
        hx = dot4_i8(0x01010101, xw[j]) + hx;          // sum of the activations
    }
    return fma(dw * dx, float(s - hx), acc);           // upstream acc += dw*dx*(float)(s - hx); L-9
}
```

`s − hx` is the exact integer Σ (q − 1)·x. `dw` is the row's half scale for the chunk pair, `dw = half at
scales + 2·(c >> 1)`, and `dx` is the activation's FP32 scale of chunk c.

**`strata_mtp_gu_s2`**, grid (40, 10) × 256, threadgroup (b, rank). Upstream `gu_grouped_kernel`
(`GU_ROWS` = 32, 4 rows per simdgroup). Threadgroup memory: `xs_q[640]` as `int` and `xs_d[80]`, 2,880 B.

```c++
rank = tg.y;  e = ids[rank];  blob = mtp_blob + e*1382400;
for (i = tid; i < 640; i += 256) xs_q[i] = ((device const int*) mq)[i];     // word 8c + w of chunk c
for (i = tid; i < 80; i += 256) xs_d[i] = ms[i];
barrier;
for (rr = warp; rr < 32; rr += 8) {
    i = 32*tg.x + rr;                                   // slot 0..1279
    codes = blob + i*640;  sc = blob + 1228800 + i*80;
    acc = 0.0f;
    for (q = 0; q < 3; ++q) {
        c = lane + 32*q;  if (c >= 80) break;           // lanes 0-15 take 3 chunks, 16-31 take 2
        acc = s2_chunk(codes + 8*c, &xs_q[8*c], half_at(sc + 2*(c >> 1)), xs_d[c], acc);
    }
    acc = butterfly(acc);                               // lane 0 equals upstream's shuffle-down warp_sum
    if (lane == 0) ((i & 1) ? eu : eg)[rank][i >> 1] = acc;
}
```

**`strata_mtp_swiglu_s2`**, grid 25 × 256: `i = 256·tg.x + tid` (< 6400); `eh[i] = silu_k(eg[i]) * eu[i]` over
the ten rows. Upstream `swiglu_kernel` computes `(g / (1.0f + __expf(-g))) * u` in a file without fast math, so only
`__expf` changes (section 3.9).

**`strata_mtp_down_s2`**, grid (40, 10) × 256, threadgroup (b, rank). Upstream `down_grouped_kernel` (`D_ROWS` = 64,
8 rows per simdgroup, one chunk per lane for lanes 0-19). Threadgroup memory: `hs_q[160]` and `hs_d[20]`.

```c++
rank = tg.y;  e = ids[rank];  blob = mtp_blob + e*1382400;
for (i = tid; i < 160; i += 256) hs_q[i] = ((device const int*)(hq + 640*rank))[i];
for (i = tid; i < 20; i += 256) hs_d[i] = hs[20*rank + i];
barrier;
for (rr = warp; rr < 64; rr += 8) {
    r = 64*tg.x + rr;  c = lane;
    acc = 0.0f;
    if (c < 20) acc = s2_chunk(blob + 819200 + r*160 + 8*c, &hs_q[8*c],
                               half_at(blob + 1331200 + r*20 + 2*(c >> 1)), hs_d[c], 0.0f);
    acc = butterfly(acc);                               // lanes 20-31 add +0.0, as upstream's
    if (lane == 0) parts[rank][r] = acc;
}
```

- Upstream's entry e of the group writes `out[ent_dst[e] · 2560 + r]`; with one entry per group `ent_dst` is the
  rank, so `parts[rank]` is the same place.
- Upstream also has `gu_grouped_t_kernel` and `down_grouped_t_kernel` (bank-conflict-free staging, two rows per
  warp). Their comments and parity tests state they are bitwise the plain ones; the Mac follows the plain form.

The validated `encode_mtp_routed_experts` host entry composes the two scaled
Q8_0 activation quantizers, S2 gate/up and down kernels, and SwiGLU in one
ordered Encoder segment. Its input contract is the full contiguous
512-expert file and ten in-range IDs from stage 16, including expert 511 at
the file's final blob. The serial and concurrent twins compare quantized
codes/scales and gate/up/SwiGLU/down values with an independent CPU oracle,
and check alias/extent/ownership rejection, output guards, read-only inputs,
open-group refusal, standalone parity, and rollback.
The test's complete synthetic expert allocation is about 675 MiB. Production
runtime-view composition and model-backed parity are still open.

**`strata_mtp_window_ids`**, grid 8 × 256, encoded for every runtime-bound attention pass. Upstream
`window_ids_kernel`, `window` a `setBytes` constant:

```c++
n_kv = steps[0].n_kv;
width = window == 0 ? n_kv : min(n_kv, window);
start = n_kv - width;
for (j = 256*tg.x + tid; j < width; j += 2048) mtp_ids[j] = start + j;
```

Upstream's block 0 also stores `width` into the step record, which the attention then reads. The Mac's host has
already written the same value into `step[0].width` (the `row(c)` rule above), so no kernel writes a parameter
block. A zero window rewrites the active identity prefix `0..n_kv−1`; bind-time initialization of the full
capacity remains a startup safeguard. Rewriting on each pass prevents a previous finite window from leaving
stale IDs when the runtime returns to full history.

The attention kernels of section 5.7 run with the drafter's `cap` and `n_chunks` as `setBytes` constants and one
query row. A chunk past `width` exits after writing its empty marker, so a short context costs little more than
its own cells.

**`strata_mtp_row_top_prob`**, grid 1 × 1024. Upstream `row_top_prob_kernel` (`<<<rows, 1024>>>`), in a file without
fast math. `n_out` is the head's row count, a `setBytes` constant.

```c++
m = hl[out_ids[0]];
s = 0.0f;
for (i = tid; i < n_out; i += 1024) s = s + exp(hl[i] - m);
s = butterfly(s);  if (lane == 0) part[warp] = s;
barrier;
if (tid == 0) { tot = 0.0f;  for (w = 0; w < 32; ++w) tot = tot + part[w];  probs[0] = 1.0f / tot; }
```

With a draft vocabulary the sum covers the subset only, so the value overestimates the full-vocabulary probability;
`--spec-min-p` is applied to that value ([03](03-model-reference.md), section 7).

**`strata_mtp_map_ids`**, grid 1 × 32, only with a draft vocabulary, after `strata_mtp_row_top_prob` (which indexes
`hl` by the subset index): `if (tid == 0) out_ids[0] = dvocab[out_ids[0]]`. Upstream `map_ids_kernel`.

**`strata_mtp_select`**, grid 16 × 256, `mode` a `setBytes` constant. Upstream `mtp_select_kernel`.

```c++
row = mode == 0 ? P.a : 0;
src = mode == 0 ? mtp_rin : R;
for (i = 256*tg.x + tid; i < 10240; i += 4096) mtp_rin[0][i] = src[row][i];
if (tg.x == 0 && tid == 0) {
    if (mode == 0) mtp_tok[0] = P.tokens[P.a];
    if (mode == 1) { tok = out_ids[0];  mtp_tok[0] = tok;  mtp_out[P.j] = tok;  mtp_prob[P.j] = probs[0]; }
}
```

| Mode | After | Block | Effect |
| --- | --- | --- | --- |
| 0 | the K/V-only pass | K | the full layer continues from row a: its residual and the token at cell p + a + 1 |
| 1 | a greedy full layer | Fj | the next step continues from this layer's residual and draft; draft j and its probability are published |
| 2 | a coupled full layer | Fj | the residual only; the CPU tail writes `mtp_tok`, `mtp_out[j]` and `mtp_prob[j]` (section 5.12) |

In mode 0 with a = 0 every thread copies an element onto itself; for a > 0 the two rows do not overlap.

**Command buffers** (the encoding rules are section 8):

| Command buffer | Contents | Then the host |
| --- | --- | --- |
| Commit + round | the commit of section 8; `strata_util_copy_strided` (the window arena's `R` rows 0..T−1 → `mtp_rin`, upstream's copy of `window_R_` into `Rin_`); the K/V-only pass (K); `strata_mtp_select` mode 0 (K); the full layer (F0, cell p + a); `strata_mtp_select` mode 1 (F0) | waits; reads `mtp_out[0]`, `mtp_prob[0]` |
| Step j | the full layer (Fj, cell p + a + j); `strata_mtp_select` mode 1 (Fj) | waits; reads `mtp_out[j]`, `mtp_prob[j]` |
| Prefill | one K/V-only pass per group (below) | waits |

- Steps run for j = 1, 2, … while j < min(S − 1, max_drafts) and the last probability is ≥ `--spec-min-p`, as
  upstream's loop (`mtp.cpp:804`). Each costs one host round trip (about 11 µs, appendix A.5), as upstream's graph
  launch and sync per step. The host zeroes the drafts it did not run.
- In coupled mode each full layer ends with the kernels of section 5.12 and `strata_mtp_select` mode 2 instead of
  stage 22 and mode 1, and the CPU tail runs after the wait.
- Every drafter command buffer waits for and updates the queue fence, like every other (A.11 test 4, section 8).

**`draft_first`.** The first round after a prompt is upstream's `draft_first(S, R_row, x, p − 1)`: the residual of
the prompt's last row is copied into all S rows of the window residual, and `draft` runs with S copies of x and
a = 0. Row 0 is the real pair. The other rows write cells p..p + S − 2, which the next round's K/V-only pass
rewrites before any query reads them. The Mac encodes the same with `strata_util_copy_strided` at a source row stride
of 0 from `R_row` into `mtp_rin`, in place of the window copy. `R_row` is row 0 of the window `R` of the first window
after the prompt, the T = 1 window that runs the prompt's last token (section 6.1). That is upstream's `ss.R` after
the token-path pass of that token (`generate.cpp:6108-6113`), not a prefill row: prefill stops one token short.

**Prefill.** `prefill(R_rows, next_tokens, n, cell0)` writes the drafter's K and V for n prompt cells
([06](06-engine-host-port.md), section 10.3). Upstream calls it after every prompt chunk and after every short-read
window's commit (`generate.cpp:4364-4369`, `:5065`). The Mac encodes it as K/V-only passes:
- groups of up to 8 rows, each with its own block (`T`, `hid`, `tokens = next_tokens[c..]`,
  `step[t] = row(cell0 + c + t)`);
- stage 4 reads `R_rows` at the group's offset directly (no copy into `mtp_rin`);
- a group with `cell0 + c + T ≤ first_needed` is skipped, where
  `first_needed = window > 0 && prompt_len > 0 ? prompt_len − window − 64 : 0` (upstream `mtp.cpp:687`);
- one command buffer per call, committed after the chunk's own command buffer and waited before the call returns,
  as upstream's single sync at the end (E-4).

Upstream groups by S rows (its graphs are captured per T ≤ S). The Mac's groups of 8 write the same bits into every
cell: each K/V-only row is independent of T (the mat-vecs, norms and GR reads are per row). Only cells below
`first_needed`, which no query reads, may be written or skipped differently. The property test
`mtp_kv_t_independent` checks the per-row claim (section 10.4).

A group of 8 rows reads about 51 MB of weights (four `fc_hidden` passes, `fc_embedding`, `k_proj`, `v_proj` and the
attention GR read), 6.4 MB per prompt token. `first_needed` limits the work to the last 32,832 cells, so the drafter
adds at most about 0.75 s to a long prompt at A.2's 280 GB/s, against 40-80 s for the prompt itself.

When upstream runs one GPU with `kv_mode` 0, it takes the batched E-9 path instead (`sp.draft_kv`, chunks of at
least 64 rows through the prompt GEMMs), whose K and V bits differ from `mtp.prefill`'s. The Mac's are those of
`mtp.prefill`. This changes drafts only (deviations, below). A batched drafter prefill is work item L16 (section 12).

**The draft head.** With a draft vocabulary (`draft_vocab.bin`, default `cjk`, 106,299 unsorted ids), upstream's
bind gathers the head rows of those ids, in file order, into a buffer of their own (`dhead`, `gather_rows`,
`mtp.cpp:390-405`), 0.187 GB for a Q5_K head. The Mac does not copy them. Bind loads `dvocab` (subset index → token
id) into a Shared buffer and, in coupled mode, `dinv` (token id → subset index, or −1). The head mat-vec is then
`strata_mv_<headfmt>_n1_x`, which reads weight row `dvocab[r]` of the main head in place for output row r
(section 5.8), with `n_out = 106,299`. Each output row is bitwise upstream's row r of `dhead`. Without a vocabulary
the plain `strata_mv_<headfmt>_n1` reads the main head with `n_out = 248,320`. The stderr line `strata mtp: draft head
over N tokens (X MiB)` is kept, with X the size of the rows read. The drafter requires a head format with a native
kernel (upstream `mtp.cpp:383`); every head format of the catalogue has one (Q5_K, Q6_K, IQ4_XS).

The rows are scattered over the head (1,360-2,100 B each), so a draft reads them at row granularity rather than as
one stream. L-15 measures that read against the contiguous head mat-vec. If it falls below 90% of it, bind gathers
the rows as upstream does, and the plan adds the copy to `mtp_bind_bytes`.

**The KV.** One QSA-shaped layer of section 4.5, in the main model's `--kv` mode, except that `k8v4` becomes `i8`
(upstream `mtp.cpp:205-234`). `kv_rot` is set for `q4`, and for `i8` with `STRATA_KV_ROT=1`, as upstream's
`st.kv_rot`. The layer has all `max_cells` cells and no ring: upstream's ring (`window + 4·S + 64` cells) exists
only with KV streaming (`kv_mode` 2), which the Mac does not port, and `kv_restore` is a no-op without it.
Conversation parking copies these cells after the main layers' ([06](06-engine-host-port.md), section 11.3).

Speculative cells are harmless. Draft j writes cell p + a + j from the drafter's own residual. The next round
starts at p' = p + a + 1, and its K/V-only pass rewrites cells p'..p' + T' − 1 from the target's residuals before
its full layer, at cell p' + a', attends to any of them. The steps then attend only to cells their own full layers
have written.

**Cost per draft** (cjk subset, Q5_K head, the default window, `f16` KV): `dense.bin` 0.116 GB, 10 experts
0.014 GB, the head subset's rows 0.187 GB and up to 32,768 cells of KV 0.067 GB, about 0.38 GB or 1.4 ms at A.2's
280 GB/s. The K/V-only pass reads about 37 MB at T = 4 (0.13 ms). A round of three drafts costs about 4.5 ms.

**Deviations** (in addition to section 3.9's `native_quantize_q8_1`, the norms' `rsqrt` and division, and `exp`
in the router and the attention):
- `__expf` becomes `exp` in `strata_mtp_swiglu_s2` and `strata_mtp_row_top_prob`.
- Prefill K and V follow `mtp.prefill`, not upstream's batched E-9 path (above).

None of them can change the output: a draft is only a proposal, kept only if it equals the target's own pick
([06](06-engine-host-port.md), section 10.5). They can change which drafts are offered and so the speed; R-3 measures
acceptance against upstream's ([03](03-model-reference.md), section 10).

**L-9 items:** the form of `acc += dw·dx·(float)(s − hx)` in `s2_expert_grouped.cu` (expected
`fma(dw·dx, float(s − hx), acc)`), and the ±0.5 add of `quantize_q8_0_scaled_kernel` (expected a rounded product
and a separate add; if the SASS shows `fma(x, inv, ±0.5)`, the Mac kernel and its oracle use that).

### 5.12 The sampler

Upstream: `sampler.cu:39-79` (Philox, penalties), `:100-167` (`sampler_greedy_kernel`), `:326-424` (`sampled_k`,
`take_first`, `warp_first`, `sampled_tail_warp`), `:505-676` (the split top_k), `:678-750` (coupled), `:836-931`
(`sample_tokens`, `coupled_draft_sample`); `sampler.hpp:15-28` (`SamplerParams`) and `:73-84` (`penalty_rows`);
`coupled_draft.hpp:50-88`; the host references in `sampler_parity.cpp:134-345`; [K4 section 4](audit/K4-ple-mtp-head-sampler.md).
The chain is penalties, top_k, top_p, min_p, temperature, then one inverse-CDF pick
([03](03-model-reference.md), section 6.10).

The sampler kernels and CPU tail are implemented and exercised by synthetic parity tests, including deterministic
upstream-derived cases 16 and 17. These cases preserve the tie, sentinel, penalty-history, parameter and draw-counter
shapes while using a portable integer-generated background instead of upstream's implementation-dependent
`std::normal_distribution` vectors; they are semantic ports, not byte-identical copies of upstream's generated data.
The Metal vocabulary ceiling excludes fixture 16's 262,145-entry fallback case, and fixture 17's two draw-counter
variants use the same Metal queue. These scope limits and the remaining fixtures are listed under REV-028. The
kernels and CPU tail are not yet called from a production `Window`, head or MTP dispatch. Until REV-028 is cleared,
the paths and buffer ownership below specify the intended integration contract, not current inference behavior.

The GPU selects; the CPU picks. Upstream's tail (`sampled_tail_warp`) runs in FP64 on one warp; Metal has no
FP64, so the intended Mac path runs the same arithmetic on the CPU after the command buffer's wait. Everything
before the tail, including the penalties and the ordered top_k list, is intended to match upstream bitwise.

**Planned paths.** The request will decide the path when its window is encoded:

| Request | Kernels at the end of the command buffer | After the wait |
| --- | --- | --- |
| Greedy (`greedy`, or `temperature ≤ 0`), with or without penalties | `strata_samp_greedy` over the T rows | the host reads `samp_out[0..T)` |
| Sampled | `strata_samp_split_part`, `strata_samp_split_merge` | the CPU tail runs once per row on `sel[t]` |
| A window whose picks are discarded (upstream `head_sampling_` off) | `strata_samp_greedy` with `hist_len = 0` | as greedy |
| MTP draft, greedy | stage 22 of section 5.11 | section 5.11 |
| MTP draft, coupled (O-class) | `strata_samp_coupled_penalize`, `strata_samp_split_part`, `strata_samp_split_merge` (one row), `strata_mtp_select` mode 2 | the CPU tail, then the host publishes the draft (below) |

Upstream's verify graph records a greedy argmax without penalties, and a sampled or penalised request runs
`sample_tokens` a second time after the replay (`verify.cpp:1164-1180`), because a captured sampler would replay
the same parameters forever. The planned Mac path encodes every window afresh (section 8) and reads the settings
from the parameter block, so it encodes the request's own sampler once. The picks should be the same.

**Buffers.**

| Buffer | Shape | Storage | Written by |
| --- | --- | --- | --- |
| `pen_hist` | int32 [8][4096] (128 KiB) | Shared | the host, `penalty_rows` for the window, before the commit (`generate.cpp:5352-5361`) |
| `cand` | int2 [8][64][64] (256 KiB): row t, block b, entry i at `(t·n_blocks + b)·64 + i`, as (id, float bits) | window arena | `strata_samp_split_part` |
| `sel` | int2 [8][64] (4 KiB): row t's top_k list in the selection order | Shared | `strata_samp_split_merge` |
| `samp_out` | int32 [8] | Shared | `strata_samp_greedy` |

The window's history row length is `h = min(penalty_last_n, 4096)`, upstream's `kPenaltyWindowCap`
(`generate.cpp:4127`). The host writes row t of `penalty_rows` at `pen_hist[t]` and sets `samp[t].hist_off =
t·4096 + (h − take)` and `samp[t].hist_len = take`, with `take = min(h, n_consumed + t + 1)`: the kernels skip
upstream's −1 front padding, which matches no id. `hist_len = 0` when penalties are off. The drafter has its own
`cand` [64][64], `sel` [64] and `ring` int32 [4096 + 8] (Shared) in its region, only in coupled mode (section 5.11).

**Constants** (`setBytes`, fixed for the engine):

```c++
struct SampArgs {
    uint n_vocab;    // logits per row: 248,320, or the drafter head's row count nv
    uint n_blocks;   // ceil(n_vocab / 4096); at most 64
    uint kpart;      // 0: each list is k = sampled_k(top_k, n_vocab) long; else this length (coupled: min(nv, 64))
    uint penalize;   // 1: the selection applies the row's penalties (window); 0: none (coupled applies them first)
};
```

`sampled_k(top_k, n) = min(n, 1 ≤ top_k < 64 ? top_k : 64)`. At load the engine refuses a vocabulary with more than
64 blocks: `strata: a vocabulary of <n> tokens exceeds the sampler's 262,144`. The model's 248,320 is 61 blocks.

**Shared functions** (`safe/sampler.metal`):

```c++
// the first of two candidates in the selection order: larger value, then lower id
inline void take_first(thread float& bv, thread int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}
// upstream warp_first: an xor butterfly over the simdgroup; every lane ends with the same pair
inline void simd_first(thread float& bv, thread int& bi) {
    for (ushort off = 16; off > 0; off >>= 1) take_first(bv, bi, simd_shuffle_xor(bv, off), simd_shuffle_xor(bi, off));
}
inline int history_count(device const int* h, int n, int v) {
    int c = 0;  for (int i = 0; i < n; ++i) c += h[i] == v;  return c;
}
inline float penalize(float x, int c, constant SampRow& p) {     // upstream apply_penalties
    if (c <= 0) return x;
    x = x <= 0.0f ? x * p.penalty_repeat : x / p.penalty_repeat;
    return x - fma(float(c), p.penalty_freq, p.penalty_present);  // (float)c*freq + 1.0f*present, contracted (L-9)
}
```

The selection order is a strict total order (an id is in one place at most, and two sentinels are the same pair), so
`simd_first` and every reduction below give the same pair in any order. Ties go to the lower id; a NaN never wins;
the sentinel is (−inf, `n_vocab`). The division is IEEE in `safe` (A.10).

**`strata_samp_greedy`**, grid rows × 1024, one threadgroup per row. Upstream `sampler_greedy_kernel`. It scans
chunks of 131,072 tokens, retaining each thread's best candidate across chunks. The dynamic threadgroup scratch
contains `⌈min(n_vocab,131072)/32⌉` bitmap words and 64 words for the 32 simdgroup value/id pairs. Its length is
`round_up_16(4·⌈min(n_vocab,131072)/32⌉ + 256)`, at most 16,640 B. The bitmap is cleared and reused for each chunk;
the reduction words share the same allocation after the scan.

```c++
row = samp[t];  l = logits + t*n_vocab;  h = hist + row.hist_off;  n = row.hist_len;
bv = -INFINITY;  bi = n_vocab;
for (base = 0; base < n_vocab; base += 131072) {
    end = min(base + 131072, n_vocab);  words = ceil((end - base)/32);
    if (n > 0) {
        for (w = tid; w < words; w += 1024) bits[w] = 0;
        barrier;
        for (i = tid; i < n; i += 1024) { v = h[i]; if (base <= v && v < end) atomic_fetch_or(&bits[(v-base) >> 5], 1u << ((v-base) & 31)); }
        barrier;
    }
    for (local = tid; local < end - base; local += 1024) {
        v = base + local;  s = l[v];
        if (n > 0 && (bits[local >> 5] >> (local & 31) & 1)) s = penalize(s, history_count(h, n, v), row);
        if (s > bv || (s == bv && bi < n_vocab && v < bi)) { bv = s; bi = v; }
    }
    barrier;  // all bitmap readers finish before the next chunk reuses it
}
simd_first(bv, bi);
if (lane == 0) { atomic_store(&bits[max_words + sg], as_type<uint>(bv)); atomic_store(&bits[max_words + 32 + sg], as_type<uint>(bi)); }
barrier;
if (sg == 0) { bv = as_type<float>(atomic_load(&bits[max_words + lane])); bi = as_type<int>(atomic_load(&bits[max_words + 32 + lane])); simd_first(bv, bi); if (lane == 0) out[t] = bi < n_vocab ? bi : 0; }
```

The bitmap and reduction scratch use relaxed `threadgroup atomic_uint` operations. The window writes `samp_out`;
the drafter's stage 22 writes `out_ids` with one row, `hist_len = 0`.

**`strata_samp_split_part`**, grid (n_blocks, rows) × 128, threadgroup (b, t). Upstream `sampler_split_part_kernel`,
line for line:
- Simdgroup w of block b covers ids `4096·b + 1024·w + 32·j + lane`, j = 0..31, held in a `float s[32]` per lane.
  Ids past `n_vocab` hold −inf.
- With `penalize` and `hist_len > 0`, the block's 4,096-bit bitmap (`uint bits[128]`, 512 B) marks the history ids
  inside the block. A marked logit becomes `penalize(s[j], history_count(h, n, v), row)`; the count is over the
  row's whole history.
- `kp = kpart ? kpart : sampled_k(row.top_k, n_vocab)` rounds. Round i keeps the logits strictly after the
  previous pick in the selection order (`x < prev_v || (x == prev_v && v > prev_i)`, starting from (+inf, −1)) and
  takes their first: two chains (even and odd j) with a strict `>`, `take_first` of the two, then `simd_first`. A
  round that finds nothing ends the list; lane 0 writes `wl[w][i] = (id, as_type<int>(value))`, and the rest of the
  list is sentinels.
- Simdgroup 0 merges the four lists (`warp_merge_lists` below, stride 64) into the first `kp` entries of
  `cand[(t·n_blocks + b)·64 ...]`.

Threadgroup memory: `bits[128]` (512 B) and `wl[4][64]` (2,048 B), 2,560 B total. Each lane loads 32 floats once,
so the 61 × T threadgroups read the logits once.

**`strata_samp_split_merge`**, grid rows × 32, one simdgroup per row. Upstream `sampler_split_merge_kernel` and the
merge half of `coupled_merge_kernel`. `kp` as above; `k = sampled_k(row.top_k, n_vocab)`, which is at most `kp`.

```c++
for (e = lane; e < n_blocks*kp; e += 32) lists[e] = cand[(t*n_blocks + e/kp)*64 + e%kp];
simdgroup_barrier;
warp_merge_lists(lists, n_blocks, kp, k, [&](i, v, id) { if (lane == 0) sel[t*64 + i] = int2(id < n_vocab ? id : 0, as_type<int>(v)); });
```

`warp_merge_lists` is upstream's (`sampler.cu:522-561`): lane L holds the heads of lists L and L + 32; each of the
k rounds takes `simd_first` of all heads, passes it to the sink, and advances the list it came from (exactly one
head matches, because the lists cover disjoint ids). Threadgroup memory: `lists`, `n_blocks · kp · 8` B, set at
encode with kp = 64: 31,232 B for 61 blocks, and exactly the 32 KiB limit at 64 blocks, so the kernel has no other
threadgroup array.

**The CPU tail** (`src/metal/engine/sampler_tail.cpp`, compiled with `-ffp-contract=off`,
[08](08-build-and-distribution.md), section 4.3). A serial transcription of `sampled_tail_warp`, the same as
upstream's host mirror `mirror_pick` (`sampler_parity.cpp:309-345`) plus the coupled probability:

```c++
struct TailPick { int32_t index; float prob; };
TailPick sampled_tail(const int32_t (*sel)[2], int k, const SamplerParams& p, uint64_t counter) {
    if (p.greedy || p.temperature <= 0.0f) return {sel[0][0], 1.0f};
    float l[64]; for (int i = 0; i < k; ++i) l[i] = bit_cast<float>(sel[i][1]);
    const float inv_t = p.temperature > 0.0f ? 1.0f / p.temperature : 0.0f;
    double ex[64]; int n_keep = k;
    float mx = l[0]; for (int i = 1; i < k; ++i) mx = std::fmax(mx, l[i]);
    if (p.top_p < 1.0f) {
        for (int i = 0; i < k; ++i) ex[i] = std::exp((double) l[i] - (double) mx);
        double sum = 0.0; for (int i = 0; i < k; ++i) sum += ex[i];
        for (int i = 0; i < k; ++i) ex[i] = ex[i] / sum;
        int cut = k; double cum = 0.0;
        for (int i = 0; i < k; ++i) { cum += ex[i]; if (cum >= (double) p.top_p) { cut = i + 1; break; } }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = l[0] + std::log(p.min_p);                 // float logf, as logf on the GPU
        for (int i = 0; i < n_keep; ++i) if (l[i] < thresh) { n_keep = i; break; }
    }
    float smx = l[0] * inv_t; for (int i = 1; i < n_keep; ++i) smx = std::fmax(smx, l[i] * inv_t);
    for (int i = 0; i < n_keep; ++i) ex[i] = std::exp((double) (l[i] * inv_t) - (double) smx);
    double sum = 0.0; for (int i = 0; i < n_keep; ++i) sum += ex[i];
    for (int i = 0; i < n_keep; ++i) ex[i] = ex[i] / sum;
    const float u = philox_uniform(p.seed, counter);
    int pi = n_keep > 0 ? n_keep - 1 : 0; double cum = 0.0;
    for (int i = 0; i < n_keep; ++i) { cum += ex[i]; if ((double) u < cum) { pi = i; break; } }
    return { sel[pi][0], n_keep > 0 ? (float) ex[pi] : 1.0f };
}
```

`philox_uniform` is upstream's `host_philox_uniform` (`sampler_parity.cpp:134-159`): ten rounds of the non-standard
Philox 4×32 with key (i, 0), the high halves as `uint32_t((uint64_t) a * b >> 32)`, and `u = (c0 >> 8) · 2^-24`.
The counter is `pos0 + t` for window row t (`verify.cpp:1175`) and `coupled_draft_counter(p + a + j)` =
`p + a + j + 1` for coupled draft j (`coupled_draft.hpp:70-73`). The window's seed and counter are not read by any
kernel; `WindowParams.seed_lo`/`seed_hi` stay in the block so that a Phase-2 GPU tail needs no layout change.

The tail runs at most 128 `exp` calls and 128 divisions per row, microseconds next to the round trip (A.5). Its
only deviation from upstream is libm's `exp` and `logf` against CUDA's (section 3.9): a pick can differ only when
its uniform lies within about 1e-16 of a cumulative boundary, or `logf(min_p)` lands on a logit.

**Coupled drafts** (O-class; `STRATA_SPEC_COUPLED=1` or `--coupled-draft`, [06](06-engine-host-port.md),
section 10.5). Upstream: `coupled_draft_sample` and its kernels, captured in the drafter's graphs, reading their
parameters from device memory. The Mac:
- Before the round's command buffer, the host writes the ring base `ring[cap − h, cap)` with
  `coupled_hist_base(consumed, n, x', h, ...)`, `cap = kCoupledHistCap` = 4096 and
  `h = coupled_hist_len(penalty_last_n, cap)`. This replaces `coupled_stage_kernel`. The full layer's `samp` is
  the request's settings with `hist_off = cap + j − h` and `hist_len = h`, and `ring` is bound as the history.
- Stage 21 writes the draft logits `hl` (nv entries: the draft vocabulary, or the whole vocabulary).
- `strata_samp_coupled_penalize` applies the penalties in place (below).
- `strata_samp_split_part` with `SampArgs {nv, ⌈nv/4096⌉, min(nv, 64), 0}`, grid (n_blocks, 1); then
  `strata_samp_split_merge` (one row) with k = `sampled_k(top_k, nv)` into the drafter's `sel`.
- `strata_mtp_select` mode 2 continues the residual.
- After the wait the CPU runs `sampled_tail` with the counter above (for a greedy request, defensively, the first
  entry with probability 1, as `coupled_merge_kernel`), maps the subset index to its token id through `dvocab`, and
  writes `mtp_tok[0]`, `mtp_out[j]`, `mtp_prob[j]` and `ring[cap + j]`, all Shared, before it commits the next step.
  Each draft already has its own wait (section 5.11), so this adds no round trip.

**`strata_samp_coupled_penalize`**, grid 1 × 1024. Upstream `coupled_penalize_kernel`. `has_map` (`setBytes`)
selects the `dinv` map (`id_to_sub`, n_vocab int32, −1 outside the draft vocabulary, Shared, built at bind) or
none. Threadgroup memory is `round_up_16(4·⌈nv/32⌉)` bytes; the kernel indexes only the bitmap words it needs.

```c++
row = samp;  n = row.hist_len;  if (n <= 0) return;
h = ring + row.hist_off;
for (w = tid; w < ceil(nv/32); w += 1024) seen[w] = 0;
barrier;
for (i = tid; i < n; i += 1024) {
    v = h[i];  if (v < 0 || v >= n_vocab) continue;
    s = has_map ? dinv[v] : v;  if (s < 0 || s >= nv) continue;
    bit = 1u << (s & 31);
    if (atomic_fetch_or(&seen[s >> 5], bit) & bit) continue;    // each distinct token once
    hl[s] = penalize(hl[s], history_count(h, n, v), row);
}
```

The target applies the same `penalize` to the same token, with the same count over the same window, before its own
selection, which is why the coupled draft agrees with the target's pick as often as it does.

**Deviations:** the FP64 tail on the CPU (section 3.9). The one-block kernel, the old sampler and
`coupled_stage_kernel` are not ported (section 2.5).

**L-9 item:** the penalty's `(float) count * freq + present` (expected `fma(float(c), freq, present)`).

**Tests** (section 10): `metal_sampler_parity_test` and
`metal_sampler_parity_test_concurrent` run random, edge and fixture sets for T = 1..8 and check model and maximum
vocabularies. Standalone and shared Encoder paths cover greedy, split-part, split-merge, composite sampled,
coupled penalty and CPU-tail behavior; the concurrent test also checks that dependent merge/sample stages reject
an open concurrent group and discard the segment. Some upstream `sampler_parity` fixtures remain unported, and production
vocabulary refusal is not yet tested.
`coupled_draft_test` (host) runs unchanged. `samp_coupled_eq_host` checks the Metal coupled penalty/selection
chain against host arithmetic; the host mirror is not the CUDA-only `coupled_draft_sample` entry point. Shared
composite tests also cover 248,320 and 262,144 vocabulary entries, exercising the 32 KiB merge scratch in one
Encoder segment. REV-028 tracks the remaining fixture and production integration work.

### 5.13 Utilities

`safe/util.metal`. Each kernel takes its row count from the parameter block (`rows`, bound at the block's `T`) and
its widths as `setBytes` constants. The host sizes the grid from the same T (section 4.1), and threads past the end
return.

**`strata_util_copy_strided`**, grid ⌈rows·cols/256⌉ × 256. Upstream: the copy of `window_R_` into the drafter's
`Rin_` (`mtp.cpp:634`) and `draft_first`'s S copies of one residual row (`mtp.cpp:830-836`).

```c++
struct CopyArgs { uint cols; uint src_stride; uint dst_stride; };   // in floats; src_stride 0 repeats one row
i = 256*tg + tid;  r = i / cols;  c = i - r*cols;
if (r >= rows[0]) return;
dst[r*dst_stride + c] = src[r*src_stride + c];
```

| Use | rows | cols | src_stride | dst_stride |
| --- | --- | --- | --- | --- |
| The MTP round: window `R` rows 0..T−1 → `mtp_rin` (section 5.11) | the K/V-only block's T | 10,240 | 10,240 | 10,240 |
| `draft_first`: the last prompt row's `R` → S rows of `mtp_rin` | S | 10,240 | 0 | 10,240 |

**`strata_util_to_f16`**, grid ⌈rows·cols/256⌉ × 256. Upstream prefill `to_f16` (`prefill/kernels.cu:645-648`,
`hf`: `__float2half_rn`). `y[i] = as_type<ushort>(half(x[i]))`, which rounds to nearest even with subnormal halves
kept (A.10). A NaN becomes `0x7e00` whatever its sign and payload, and a float32 denormal becomes zero (A.10).
Neither reaches a GEMM of a healthy prompt.

**`strata_util_to_bf16`**, grid ⌈rows·cols/256⌉ × 256. Upstream prefill `to_bf16` (`prefill/kernels.cu:653-660`).
`has_lo` is a `setBytes` constant:

```c++
hi = f32_to_bf16_rne(x[i]);  y[i] = hi;
if (has_lo) ylo[i] = f32_to_bf16_rne(x[i] - bf16_to_f32(hi));   // upstream bf_lo: what hi left out, in BF16
```

Upstream's `bf()` has no NaN branch; `f32_to_bf16_rne` quiets a NaN instead (section 3.9). Both are used by the
prefill GEMMs for their activation images (section 6.2).

---

## 6. The prompt path (prefill)

Upstream: `Prefill::run` (`src/prefill/prefill.cpp:994-1973`), the kernels of `src/prefill/kernels.cu`, the GEMMs of
`src/prefill/gemm.cu` and `src/prefill/moe_mmq.cu`, and the prompt attention of `src/kernels/cuda/qsa_prompt_attn.cu`.
In this section a file reference without a directory is to `src/prefill/`. A pair such as "84/694" gives a kernel's
line and its launcher's line in `kernels.cu`. T is the chunk's row count, 1 ≤ T ≤ C, not a window width.

The prompt path has no bitwise contract with decode ([03](03-model-reference.md), section 6). As upstream's do, its
GEMMs sum in the tile's order and take F16 or BF16 activations. Section 10.3 gives the tolerances. Two properties
hold anyway:

- **Deterministic.** No value is accumulated with atomics and every order is fixed. The same prompt, chunk size, GEMM
  path and GPU family give the same bits on every run.
- **Independent of the chunk size under `simd`.**
  - Every GEMM element sums its K terms in an order the tile fixes, and that order does not depend on where the
    element sits in the tile.
  - Rows past T are zero-padded and never mix with live rows.
  - Every other prefill kernel works per row, or walks the rows in order.
  - So the state after a prompt does not depend on C. `prefill_chunk_independent` (section 10.4) runs C = 1,024,
    2,048 and 4,096 and compares the state bitwise. L-6 tests whether `mpp` has the same property.
  - Upstream does not have this property. Below about 1,024 tokens its output changes, because a smaller chunk takes
    other kernels ([05](05-memory-and-residency.md), section 2.4).

### 6.1 The chunk loop and the arena

**Entry.** `Prefill::run(tokens, n, pos0)` reads n tokens into cells pos0 .. pos0 + n − 1.
- **One-shot** (`generate.cpp:5654`, `:5712`): n = n_prompt − 1, or `--prefill-until N` if that is smaller, and
  pos0 = 0.
- **Serve** (`generate.cpp:5248`): one call per prompt segment `[at, to)` between the cut points of
  [06](06-engine-host-port.md), section 11.1, with pos0 = at. Text parts of at most `--short-read` tokens go through
  decode windows instead ([06](06-engine-host-port.md), section 11.2, step 8).
- **Without `--prefill`**, the whole prompt goes through windows, as upstream's does.
- **The prompt's last token is never prefilled.** It is the first decode window, with T = 1
  (`generate.cpp:5749-5753`), and that window's `R` row 0 feeds `draft_first` ([06](06-engine-host-port.md),
  section 10.3).
- **Prefill runs no output norm and no head.**

Chunk k covers rows from `c0 = k·C`, with `T = min(C, n − c0)` and `p0 = pos0 + c0`. A chunk is never padded to C:
every grid is sized from T at encode time.

**Inputs the host writes**, all in the prefill arena (a Shared buffer, [05](05-memory-and-residency.md), section 7):

| Buffer | Contents |
| --- | --- |
| `tokens` uint [C] | The chunk's token ids. Bit 31 (`0x80000000u`) marks a row whose embedding is an image row (section 6.7). |
| `pf_steps` StepRow [C] | `pos = p0 + t`, `n_kv = pos + 1`, `n_bid = (pos + 1) / 4`, `width = min(n_kv, 2051)` (section 4.2) |
| `img_in` F32 [C][2560] | The vision encoder's rows ([03](03-model-reference.md), section 6.11), each at its token's row; only marked rows are read |
| `ple_in` F32 [2][C][2560] | The host-decoded PLE rows, read by the PLE workers ([06](06-engine-host-port.md), section 7.4), double-buffered: while chunk k runs on slot k % 2, a host thread decodes chunk k + 1 into slot (k + 1) % 2 |

**`PrefillParams`** (`common/types.h`) opens the prefill parameter buffer:

```c++
struct PrefillParams {
    uint T;        // rows of the chunk, 1..C
    uint p0;       // cell of row 0
    uint idx_f;    // indexer: p0 == 0 ? 4 : roundup4(p0), the first cell the block kernel may start at (section 6.5)
    uint idx_e;    // indexer: 4 * floor((p0 + T) / 4), the end of the chunk's last complete block
};                 // 16 B; static_assert on both sides
```

**Batch slots.** Six 256-B slots follow the block, each a `{uint T}`. They hold the row counts of the query batches of
section 6.5, in this order:

| Slot | Row count |
| --- | --- |
| 0 | 8 |
| 1 | T % 8 |
| 2 | 256 |
| 3 | T % 256 |
| 4 | 32 |
| 5 | T % 32 |

- When a remainder is 0, the last batch is full and uses the full slot.
- A window kernel that runs over one batch of the chunk binds a slot at `buffer(0)` as its parameter block, and its
  data at `row0 · stride`. These kernels read only `T` from the block, and every position from their `StepRow`s, so a
  slot needs nothing else.
- Kernels that run over the whole chunk bind `PrefillParams` itself, whose first field is T.

`build_prefill_chunk_metadata` in `src/metal/engine/prefill.cpp` now constructs these host inputs for one chunk:
it copies the token words, derives the six slot row counts and `PrefillParams`, builds one signed `StepRow` per token,
and records whether any token has the image bit set. It rejects empty chunks, rows beyond the GR broadcast
32-bit-index limit, and positions whose `StepRow` values do not fit `int`; a failed build leaves its output unchanged.
The helper does not encode or execute the prompt path, which remains the work of `Prefill::run`.

**Per chunk**, the host:
1. waits for chunk k's PLE rows, and starts decoding chunk k + 1's;
2. writes `PrefillParams`, the batch slots, `tokens`, `pf_steps` and, if the chunk has image rows, `img_in`;
3. encodes one command buffer, labelled `prefill chunk <k>` (section 8.3). It contains:
   - the embedding (section 6.7);
   - then, for each of the 48 layers, the GR read and write of each half (section 6.3) around the layer's mixer
     (section 6.4 or 6.5) and its MoE (section 6.6), with the PLE at layer 1 (section 6.7).

   In full mode the command buffer is one segment. In tiered mode it is 97 segments, with the handshake of
   [06](06-engine-host-port.md), section 9.3, at each layer:
   - the prefill lookahead is every non-resident expert of layer l + 1;
   - the 20 s timeout applies to each layer;
4. commits it and waits.

The previous chunk has completed before the host writes the next chunk's inputs. So every input needs one copy,
except the PLE rows, which need two because their decode overlaps the GPU.

**Long chunks.** macOS ends a command buffer that runs too long, with `MTLCommandBufferErrorTimeout`
([06](06-engine-host-port.md), section 14.4). If L-12 sees that error, or a chunk time close to the limit, at
C = 32,768, the host encodes each layer as its own command buffer: 48 per chunk, committed back to back, with only
the last one waited on. The arena persists between them and the fence orders them (section 8.1), so no bit changes.

**After the wait**, in order:
1. **`STRATA_PREFILL_DUMP_R`**, in upstream's format (1886-1898):
   - the file is opened with `wb` for the call's first chunk (c0 = 0) and with `ab` after that, so in serve each
     segment's call starts the file again;
   - for `t = (64 − p0 % 64) % 64; t < T; t += 64`, the host writes `int64 pos = p0 + t`, then the 10,240 floats of
     `R[t]`, read from the arena.
2. **With MTP:** `mtp.prefill(R rows, next, T, p0)`, where `next[t]` is the token at p0 + t + 1, which is always
   inside the prompt ([06](06-engine-host-port.md), section 10.3). The call skips the groups below `first_needed()`.
   Upstream's batched drafter prefill (`draft_kv`, E-9, `prefill.cpp:737`) is work item L16 (section 12).
3. **Serve only**, as upstream's `on_chunk` does (`generate.cpp:4364-4400`):
   - the line `PP <p0 + T> <total> <ms> <tok/s>` ([06](06-engine-host-port.md), section 5.5);
   - the progress beat;
   - a checkpoint when p0 + T reaches the next `--prompt-cache-every` boundary ([06](06-engine-host-port.md),
     section 11.1).

**After the last chunk:**
- **`ple_prev`.** Let e = pos0 + n. The host sets `ple_prev = {token[e − 2], token[e − 1]}`, using −1 for any index
  below 0 (`generate.cpp:5741-5742`).
- **`STRATA_PREFILL_TIMING=1`.** The host prints upstream's two lines (1942-1958):

  ```
  strata prefill timing: %lld tokens, GPU timeline %.0f ms, wall %.0f ms, host staging %.0f ms:%s
  strata prefill timing: host: chunk setup (PLE rows, the expert stream plan) %.0f ms, waiting for each chunk %.0f ms, after each chunk (the draft layer, progress) %.0f ms, PLE %.0f ms
  ```

  - The `%s` is each phase in turn, printed as ` %s %.0f (%.1f%%)`. The phases are upstream's PfTimer marks
    (959-990), and phases with no time are left out, except three that do not exist on a Mac: "host grouping", "wait copy" and "kv stage".
  - Apple GPUs sample counters only at stage boundaries (`MTLCounterSamplingPointAtStageBoundary`). So with TIMING
    on, the host starts a new compute encoder at each mark and attaches a timestamp sample buffer to it
    (`sampleBufferAttachments`, start and end of encoder).
  - The time from one encoder's start sample to the next one's is charged to the earlier encoder's phase, as upstream
    charges the gap between two marks.
  - GPU timestamps are converted to nanoseconds with `sampleTimestamps:gpuTimestamp:` pairs, taken before and after
    the chunk.
  - The extra encoders change no bits: the serial dispatch and the fence keep the same order.
  - In tiered mode the phase encoders nest inside the segments.
- **`STRATA_STATE_HASH_GDN=1`.** The host prints `strata prefill: GDN_HASH %s` (1959-1973). For each GDN layer, in layer order,
  `%s` holds `%04llx ` of `x & 0xffff`, where x is the FNV-1a 64 hash of the layer's `gdn_floats · 4` bytes:
  - offset basis 1469598103934665603, prime 1099511628211;
  - the bytes are the state, then the conv history, which is the region of section 4.7. That region has upstream's
    layout, so the line format is comparable.
- **`STRATA_DBG_NAN=1`.** The host prints upstream's end-of-prompt line (1915-1931):
  - it begins `strata dbg: prompt end: last residual row`, followed by ` %lld non-finite (max |x| %.3g)` for the
    10,240 floats of `R` row `(n − 1) % C`;
  - then `; PLE history` and the same pair for the 92,160 floats of `hist` (section 4.8), if the PLE is on;
  - then `; GDN state 0` and the pair for the first 65,536 floats of the GDN state (section 4.7);
  - then `\n`.

  The maximum is over the finite values only. The layer report of section 6.6 is separate.

**The arena.** `Prefill` asks for `bytes_needed(C) = C · P + 8 · ⌈10C / 64⌉ + F`. This replaces the planning
formula of [05](05-memory-and-residency.md), section 2.4. P is the per-token size of the regions below:
- P = 267,304 B in BF16X2 mode 0;
- P = 272,424 B in modes 1 and 2, because `mixed_bf_lo` adds 5,120 B to `base`.

That is `base` (81,972 B) plus one **region** of 185,332 B, which is the largest set, `moe`.
- The sets that share the region are live at different times.
- Each buffer starts on a 256-B boundary.

| Set | B per token | Buffers (B per token) | Live |
| --- | ---: | --- | --- |
| `base` | 81,972 | `R` 40,960; `rs` 16; `inj` 16; `mixed_bf` 5,120; `mixed_h` 5,120; `bo` 10,240; `ple_in` × 2, 20,480; `pf_steps` 16; `tokens` 4 | the whole chunk |
| `gr` | 63,360 | `xn16` 20,480; `lo` 1,280; `lo16` 640; `gated` 40,960. Mode 1 adds `xn16_lo` 20,480 and `lo16_lo` 640. | a GR read, and the fused write before it |
| `gdn` | 144,128 | `qkv` 40,960; `z` 24,576; `ab` 384; `gate` and `beta` 384; `h` 40,960; `oc` 24,576; `y_h` 12,288 | a GDN mixer |
| `qsa` | 125,452 | `Kc` 2,048; `Vc` 2,048; `Qf` 49,152; `qcur` 24,576; `idx_raw` 512; `q_idx` 2,048; `ids` 8,204; `attn` 24,576; `attn_h` 12,288 | a QSA mixer |
| `moe` | 185,332 | `logits` 2,048; `ids` 40; `wts` 40; `sg` and `su` 5,120; `sh_h` 1,280; `shared` 10,240; `gl` 4; `src` and `dst` 80; `hist` 80; `eh` 12,800; `parts` 102,400; `Xs` 51,200 | an MoE |
| `ple` | 138,256 | `key` 40,960; `query` 40,960 (`ple_norm` reuses it); `value` 10,240; `gate` 16; `gated` 40,960; `e16` 5,120. Modes 1 and 2 add `e16_lo`, 5,120. | the PLE at layer 1 |
| `emb` | 20,480 | `emb` 10,240; `img_in` 10,240 | the embedding |

F is the fixed part:

| Item | Bytes |
| --- | ---: |
| Dequantisation scratch (section 6.2) | 64 MiB (32M halves) |
| Attention partials for one 32-row batch (section 6.5) | 32 · 817,600 |
| Block scores for one 256-row group (`max_blocks = max_cells / 4 + 2`, section 5.6) | 1,024 · max_blocks |
| `PfPlan`, one per layer (section 6.6) | 48 · 6,400 |
| The tile list's fixed part (its C-dependent part is the `8 · ⌈10C / 64⌉` term) | 8 · 512 |
| Rounding slack: each buffer's 256-B alignment, and the last partial tile of `hist` (section 6.6) | 65,536 |
| `PrefillParams` and the batch slots | 4,096 |

At 262,144 cells, F = 160,763,904 B. The totals, in GiB, in mode 0:

| Cells | C = 1,024 | 2,048 | 4,096 | 6,144 | 8,192 | 16,384 | 32,768 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 262,144 | 0.405 | 0.660 | 1.169 | 1.679 | 2.189 | 4.228 | 8.307 |
| 131,072 | 0.373 | 0.628 | 1.138 | 1.648 | 2.158 | 4.197 | 8.276 |
| 65,536 | 0.358 | 0.613 | 1.123 | 1.632 | 2.142 | 4.182 | 8.260 |
| 32,768 | 0.350 | 0.605 | 1.115 | 1.625 | 2.134 | 4.174 | 8.253 |

Upstream's formula gives 5.47 GiB at C = 8,192.

**Lifetimes.**
- A GR read ends with `mixed_bf` and `mixed_h`, both in `base`, before its mixer or MoE starts. The mixer or MoE ends
  with `bo`, which is also in `base`. So the `gr` set never overlaps a mixer set in time.
- `inj` and `rs` are in `base`, because the write that follows the mixer reads them.
- The fused write writes the next read's `rs` and `xn16`, after the MoE's or the mixer's buffers are dead.
- The expert tiles of section 6.6 never read past a group's last entry: a tile row past it loads the last entry
  again, and its output is not stored.

### 6.2 GEMMs

Upstream: `Gemm::bf16`, `Gemm::f16` and `Gemm::native` (`gemm.cu:375-430`), `native_proj` and `bf16_proj`
(931-943), `dequant_f16` (`src/kernels/cuda/dequant_bf16.cu:242`, with the per-format `group32` at 37-188).

Every dense projection of the prompt path computes `Y[t][ldy·…] (+)= Σ_k X[t][k] · W[n][k]`, with F32 output and
F32 accumulation. Upstream calls `cublasGemmEx` with `CUBLAS_COMPUTE_32F` and lets cuBLAS choose the algorithm. The
Mac runs its own tiles. The projections:

| Projection | n × K | Path | X | Y |
| --- | --- | --- | --- | --- |
| `hc_*_down` | 320 × 10240 | bf16 | `xn16` | `lo` |
| `hc_*_up` | 10240 × 320 | bf16 | `lo16` | `gated` |
| `hc_*_inject` | 4 × 10240 | bf16 | `xn16` | `inj` (ldy 4) |
| `attn_qkv` (GDN) | 10240 × 2560 | native | `mixed_h` | `qkv` |
| `attn_gate` (GDN z) | 6144 × 2560 | native | `mixed_h` | `z` |
| `ssm_alpha`, `ssm_beta` | 48 × 2560 each | bf16 | `mixed_bf` | `ab`, ldy 96, beta at column 48 |
| `ssm_out` | 2560 × 6144 | native | `y_h` | `bo` |
| `attn_q` (QSA, query and gate) | 12288 × 2560 | native | `mixed_h` | `Qf` |
| `attn_k`, `attn_v` | 512 × 2560 each | native | `mixed_h` | `Kc`, `Vc` |
| `attn_output` | 2560 × 6144 | native | `attn_h` | `bo` |
| `indexer.k_proj` | 128 × 2560 | bf16 | `mixed_bf` | `idx_raw` |
| `indexer.q_proj` | 512 × 2560 | bf16 | `mixed_bf` | `q_idx` |
| `ffn_gate_inp` (router) | 512 × 2560 | bf16 | `mixed_bf` | `logits` |
| `ffn_gate_shexp`, `ffn_up_shexp` | 640 × 2560 each | native | `mixed_h` | `sg`, `su` |
| `ffn_down_shexp` | 2560 × 640 | native | `sh_h` | `shared` |
| `ffn_gate_inp_shexp` | 1 × 2560 | bf16 | `mixed_bf` | `gl` (ldy 1) |
| PLE key | 10240 × 2560 | bf16, or native for a Q2_0 or Q8_0 key | `e16` | `key` |
| PLE value | 2560 × 2560 | bf16 | `e16` | `value` |

- **bf16**: the weight is a BF16 tensor in file layout, `W[n][K]` ([03](03-model-reference.md), section 3.7). The
  activation is the BF16 image of an F32 row. `strata_pf_gemm_bf16` reads both in place.
- **native**: the weight is a quantised dense tensor in one of the 11 formats of section 5.8.
  `strata_pf_dequant_<fmt>` expands it to F16, then `strata_pf_gemm_f16` multiplies the F16 activation by it. As
  upstream does, this happens again for every layer of every chunk.
- **The routed experts** do not use these kernels. They dequantise inside their own tiles (section 6.6).

**`strata_pf_dequant_<fmt>`**, one pipeline per format of section 5.8. Grid (⌈n_out / 256⌉, K / 32) × 256: thread
n = 256·tg.x + tid decodes group g = tg.y of row n, values 32g .. 32g + 31.

```c++
n = 256*tg.x + tid;  g = tg.y;
if (n >= n_out) return;
row = W + n * row_bytes;                               // the tensor in place, section 4.3
for (j = 0; j < 32; ++j)
    scratch[(32*g + j) * n_out + n] = half(value_<fmt>(row, 32*g + j));   // K-major: scratch[k][n]
```

- `value_<fmt>` is upstream's `group32` expression for the format, written per value:
  - Q3_K, IQ3_S and IQ4_XS use the decodes of section 5.2;
  - Q2_0: `float(q − 1) · d`;
  - Q4_0, Q5_0 and Q8_0: `float(q − 8)`, `float(q − 16)` and `float(q)`, each times `d`;
  - IQ4_NL: `d · float(kvalues_iq4nl[nib])`;
  - Q6_K: `(d · float(sc)) · float(q − 32)`;
  - Q4_K and Q5_K: `d1 · float(q) − m1`, where `d1 = d · float(sc)` and `m1 = dmin · float(m)`, and where the
    scale and min come from `scale_min_k4`.

  Section 5.8's block structs give the byte layouts.
- **Bits.** Every product above is exact in F32. Each factor is a half or an integer of at most 8 bits, and every
  product fits in 24 bits. So the only F32 rounding is Q4_K's and Q5_K's subtraction, and an FMA there would give the
  same bits as section 3.3's rule. The kernel then rounds once to half (RNE; an overflow gives infinity, as
  `__float2half_rn` does). The output is B against upstream's `dequant_f16` and against its oracle.
- **The scratch is K-major** (`scratch[k][n]`), unlike upstream's row-major `[n][k]`. This is the right-operand layout
  that A.6 measured `matmul2d` with. The scratch holds 32M halves (64 MiB, section 6.1). The largest dense weight,
  `attn_q`, is 12288 × 2560 = 31.46M, so no weight is split. If a dense native weight has n · K > 32M, `Prefill::init`
  fails with `prefill: <name> needs more than 32M halves of dequantization scratch`, where upstream would split it
  into row slices. No supported model has such a weight.

**`strata_pf_gemm_f16`** and **`strata_pf_gemm_bf16`** compute `Y[t][col0 + c] = Σ_k X[t][k] · W(k, c)` for
t < T and c < n.
- With `beta` = 1 (a function constant), each result is added to the value already in Y: `y = acc + y`, one rounding.
- The arguments:
  - X: `[T][K]`, F16 or BF16;
  - W: the K-major scratch (F16) or the file-layout tensor (BF16);
  - Y: F32, row stride `ldy`;
  - n, K, ldy, `col0`: `setBytes`;
  - T: the `PrefillParams` or batch slot at `buffer(0)`.

  K is a multiple of 32 for every projection: 320, 640, 2560, 6144 or 10240.

Each kernel has two forms. `STRATA_METAL_GEMM` ([06](06-engine-host-port.md), section 4.4) chooses between them. The
default is `mpp` on Apple10 (M5) and `simd` elsewhere. A projection with n < 64 always takes `simd`: the inject (4),
`ssm_alpha` and `ssm_beta` (48) and `ffn_gate_inp_shexp` (1).

**`mpp`** (MSL 4.0, Metal Performance Primitives). Grid (⌈n / 64⌉, ⌈T / 64⌉) × 128.

```c++
using namespace mpp::tensor_ops;
auto tX = tensor<device const half,  dextents<int32_t, 2>, tensor_inline>(X, dextents<int32_t, 2>(K, T));
auto tW = tensor<device const half,  dextents<int32_t, 2>, tensor_inline>(Wk, dextents<int32_t, 2>(n, K));
auto tY = tensor<device float, dextents<int32_t, 2>, tensor_inline>(Y + col0, dextents<int32_t, 2>(n, T),
                                                                     array<int32_t, 2>{1, ldy});
constexpr auto desc = matmul2d_descriptor(64, 64, dynamic_extent, false, false, false,
                                          beta ? matmul2d_descriptor::mode::multiply_accumulate
                                               : matmul2d_descriptor::mode::multiply);
matmul2d<desc, execution_simdgroups<4>> op;
op.run(tX.slice(0, 64*tg.y), tW.slice(64*tg.x, 0), tY.slice(64*tg.x, 64*tg.y));
```

- This is the `mm64` kernel of A.6 (22-29 TFLOPS of half on an M5 Pro), with a row stride and an accumulate mode
  added.
- The BF16 form binds `bfloat` tensors. Its weight is in file layout, `dextents(K, n)`, with `transpose_right = true`.
- The slices at the T edge and the n edge are shorter than 64. `matmul2d` takes their extents from the slice and
  writes nothing outside them.
- L-6 checks four things on each family: that MSL 4.0 and `bfloat` tensors compile; that edge slices (T = 1, 63, 65)
  write only live elements; that `multiply_accumulate` adds to Y; and whether the result is independent of the chunk
  size (`prefill_chunk_independent` under `mpp`). If any check fails, that family defaults to `simd`.

**`simd`** (`simdgroup_matrix`, every family). Grid (⌈n / 64⌉, ⌈T / 32⌉) × 128. The output tile is 32 rows × 64
columns. Simdgroup sg computes rows `16·(sg / 2)` .. +15 and columns `32·(sg % 2)` .. +31 of it, as 2 × 4 blocks of
8×8.

```c++
threadgroup half  sx[32][32];      // X tile, t × k       (BF16 form: float, the exact widening)
threadgroup half  sw[32][64];      // W tile, k × n       (BF16 form: float)
threadgroup float so[32][64];      // output staging, 8 KiB
simdgroup_float8x8 acc[2][4];      // all zero
t0 = 32*tg.y;  n0 = 64*tg.x;
for (k0 = 0; k0 < K; k0 += 32) {
    // 128 threads stage both tiles; rows t0 + i ≥ T and columns n0 + c ≥ n are staged as 0
    sx[i][j] = t0 + i < T ? X[(t0 + i)*K + k0 + j] : 0;
    sw[j][c] = n0 + c < n ? W(k0 + j, n0 + c)     : 0;   // F16: Wk[(k0+j)*n + n0+c]; BF16: W[(n0+c)*K + k0+j]
    barrier;
    for (kk = 0; kk < 32; kk += 8) {
        simdgroup_load(a[i], &sx[16*(sg/2) + 8*i][kk], 32);          // i < 2
        simdgroup_load(b[c], &sw[kk][32*(sg%2) + 8*c], 64);          // c < 4
        for (i < 2) for (c < 4) simdgroup_multiply_accumulate(acc[i][c], a[i], b[c], acc[i][c]);
    }
    barrier;
}
simdgroup_store(acc[i][c], &so[16*(sg/2) + 8*i][32*(sg%2) + 8*c], 64);  barrier;
for (each (i, c) of the tile with t0 + i < T and n0 + c < n)
    Y[(t0 + i)*ldy + col0 + n0 + c] = beta ? so[i][c] + Y[...] : so[i][c];
```

- The staging is 6 KiB in the F16 form and 12 KiB in the BF16 form, plus 8 KiB of output (section 7).
- A half or bfloat product is exact in F32. Each element adds its products in a fixed order: k0 ascending, then kk
  ascending, then the hardware's order inside one 8×8 multiply-accumulate. That order does not depend on where the
  element sits in the tile, or on T. So `simd` gives the same bits for a row whatever the chunk size.
- L-7 measures the tile on M1-M5. It may change the tile shape, but only to one that keeps that property.

**BF16X2** (`STRATA_PREFILL_BF16X2`, `prefill.cpp:108-121`).
- Mode 1 or 2 adds each BF16 activation's remainder as a second GEMM with beta = 1: `Y = X_hi · W`, then
  `Y += X_lo · W`. Here `X_lo = f32_to_bf16_rne(x − bf16_to_f32(X_hi))`, as upstream's `bf_lo` computes it.
- Mode 1 covers every bf16 projection.
- Mode 2 covers all of them except the hyper-connection's down, inject and up.
- The remainders are written by the kernels that write the BF16 images: `mixed_bf_lo` by `strata_pf_gr_mix`,
  `xn16_lo` by the GR norms, `lo16_lo` by `strata_pf_gr_silu`, and `e16_lo` by `strata_util_to_bf16`.

**Tolerance.** U(2K) and R(2^-16) against the FP64 sum of the exact products (section 10.3).

**Deviations** (section 3.9).
- cuBLAS's algorithm becomes the Mac tiles. Both are F32 sums of exact products in an order neither contract fixes.
- Upstream's dequant scratch is row-major; the Mac's is K-major.
- Upstream splits a weight above 32M halves into row slices; the Mac refuses it at load.

**Upstream references:**
- `Gemm::bf16` (`gemm.cu:375`), `Gemm::f16` (395), `Gemm::native` (414);
- `bf16_proj`'s lo GEMM (`prefill.cpp:937-943`);
- the projection call sites (`prefill.cpp:1263-1857`).

### 6.3 The hyper-connection read and write

Upstream: the read of each half (`prefill.cpp:1322-1340`) and its write (1838-1857); `block_sum` (`kernels.cu:44-57`),
`gr_norm_rs` (84/694), `gr_mix_r` (102/699), `gr_write_norm_rs` (128/705), `gr_silu` (160/711) and `gr_write`
(191/720). The decode form is section 5.4, which has the same weights and the same dataflow; the prefill kernels
keep upstream's prefill reductions instead.

**The read of half h** (h = 0 before the mixer, with `hc_attn_*`; h = 1 before the MoE, with `hc_ffn_*`):

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_pf_gr_norm`, unless the write before it was fused (below) | `R`, `*_norm` → `rs[T][4]`, `xn16[T][4][2560]` (mode 1: `xn16_lo`) |
| 2 | `strata_pf_gemm_bf16` `*_down` (section 6.2) | `xn16` → `lo[T][320]` |
| 3 | `strata_pf_gr_silu` | `lo` → `lo16` (mode 1: `lo16_lo`) |
| 4 | `strata_pf_gemm_bf16` `*_up` | `lo16` → `gated[T][4][2560]` |
| 5 | `strata_pf_gemm_bf16` `*_inject` | `xn16` → `inj[T][4]` |
| 6 | `strata_pf_gr_mix` | `R`, `rs`, `*_norm`, `gated` → `mixed_bf` (modes 1 and 2: `mixed_bf_lo`), `mixed_h` |

"Mode" is the BF16X2 mode of section 6.2. The mixer and the MoE read only `mixed_bf` and `mixed_h`.

**`block_sum`**, upstream's 256-thread block reduction, used by the two norm kernels:

```c++
threadgroup float part[8];
float block_sum(float v) {
    v = butterfly(v);
    if (lane == 0) part[warp] = v;
    barrier;
    v = lane < 8 ? part[lane] : 0.0f;    // as upstream's warp 0: lanes 8..31 add zeros
    return butterfly(v);                 // every simdgroup gets the same bits; upstream broadcasts warp 0's
}
```

The second stage is a butterfly over eight partials padded with zeros, not section 3.4's serial combine. That is
upstream's prefill order, and it differs from decode's `strata_gr_norm` (section 5.4).

**`strata_pf_gr_norm`**, grid (T·4) × 256. Threadgroup `row = 4t + c` handles stream c of row t, as upstream's
block does.

```c++
c = row % 4;  r = R + row*2560;                      // R[t][c][.]
ss = 0.0f;
for (d = tid; d < 2560; d += 256) ss = fma(r[d], r[d], ss);
rs = rsqrt(block_sum(ss) / 2560.0f + 1e-6f);
if (tid == 0) rs_out[row] = rs;
for (d = tid; d < 2560; d += 256) {
    v = (r[d] * rs) * w[c*2560 + d];
    h = f32_to_bf16_rne(v);  xn16[row*2560 + d] = h;
    if (lo_on) xn16_lo[row*2560 + d] = f32_to_bf16_rne(v - bf16_to_f32(h));
}
```

- `strata_pf_gr_norm_lo` writes the remainder and is selected only in BF16X2 mode 1. Modes 0 and 2 use
  `strata_pf_gr_norm`, which has no remainder output binding. This follows section 2.2's separate-host-name rule for
  pipeline variants.
- **Bits.** B against upstream's `gr_norm_rs`, given section 3.3 step 4 for `ss += r*r` (L-9) and the deviations
  for `rsqrtf` and `bf()` (section 3.9).

**`strata_pf_gr_silu`**, grid ⌈320T / 256⌉ × 256:

```c++
i = 256*tg + tid;  if (i >= 320*T) return;
x = lo[i] / 4.0f;
v = silu_k(x);
h = f32_to_bf16_rne(v);  lo16[i] = h;
if (lo_on) lo16_lo[i] = f32_to_bf16_rne(v - bf16_to_f32(h));
```

**Bits.** B, apart from `__expf` (section 3.9).

The isolated `strata_pf_gr_silu` and `_lo` kernels and their validated
`encode_prefill_gr_silu` host entry point are implemented. The encoder accepts
a Context-owned read-only low-rank view, validates the row range and output
views, and selects the remainder pipeline only for BF16X2 mode 1. The serial
and concurrent Prefill tests cover their scalar BF16 oracle, tails and buffer
contracts. The kernel is not yet connected to the Prefill GEMM stages or a
production chunk loop; `Prefill::run` remains pending.

**`strata_pf_gr_mix`**, grid (10, T) × 256. Thread `d = 256·tg.x + tid` of threadgroup (·, t) computes element d of
row t:

```c++
s = 0.0f;
for (c = 0; c < 4; ++c) {
    j = t*10240 + c*2560 + d;
    x = (R[j] * rs[4*t + c]) * w[c*2560 + d];     // strata_pf_gr_norm's v, bit for bit
    s = fma(x, sigmoid_k(gated[j]), s);
}
s = s / 4.0f;
h = f32_to_bf16_rne(s);  mixed_bf[t][d] = h;
if (lo2_on) mixed_bf_lo[t][d] = f32_to_bf16_rne(s - bf16_to_f32(h));   // modes 1 and 2
mixed_h[t][d] = half(s);                                                 // not saturated
```

- Upstream also writes an F32 `mixed`, which its MMQ quantiser reads (`prefill.cpp:1628`). The Mac's tiles take
  `mixed_h` (section 6.6), so nothing reads an F32 copy and the kernel does not write one.
- **Bits.** B, apart from `__expf` and `bf()`.

The isolated `strata_pf_gr_mix` and `_lo` kernels and their validated
`encode_prefill_gr_mix` host entry point are implemented. The encoder binds
the four F32 inputs read-only, validates the row range, view extents,
alignment, Context ownership and output aliases, then selects the remainder
pipeline for BF16X2 modes 1 and 2. Serial and concurrent Prefill tests cover
rows 1/3/8, all three modes, scalar output checks, BF16X2 reconstruction,
guards, immutable inputs, invalid views, rollback and ordered-group refusal.
The test compares `mixed_bf + mixed_bf_lo` against the scalar sum within
`2^-15` of the sum of absolute terms; this checks the two-part representation
without treating tiny residual ULP changes from the documented `exp`
deviation as large value errors. Exact binary-input midpoint fixtures also
assert high BF16 and positive/negative low-remainder RNE bits in modes 1 and
2, so the reconstruction tolerance cannot hide truncation. The mixer is not
connected to the Prefill GEMM stages or production chunk loop; `Prefill::run`
remains pending.

**`strata_pf_gr_write_norm`**, grid (T·4) × 256, upstream's `gr_write_norm_rs`. Threadgroup `row = 4t + c`:

```c++
sc = 2.0f * sigmoid_k(inj[4*t + c] / 4.0f);
ss = 0.0f;
for (k = 0, d = tid; d < 2560; d += 256, ++k) {
    x = fma(bo[t*2560 + d], sc, r[d]);           // r = R + row*2560
    r[d] = x;  v[k] = x;                         // v[10] in registers
    ss = fma(x, x, ss);
}
rs = rsqrt(block_sum(ss) / 2560.0f + 1e-6f);
if (tid == 0) rs_out[row] = rs;
for (k = 0, d = tid; d < 2560; d += 256, ++k) {
    y = (v[k] * rs) * w_next[c*2560 + d];          // the next half's *_norm
    h = f32_to_bf16_rne(y);  xn16[row*2560 + d] = h;
    if (mode == 1) xn16_lo[row*2560 + d] = f32_to_bf16_rne(y - bf16_to_f32(h));
}
```

It is the write followed by `strata_pf_gr_norm` with the next half's weight. The same thread holds the same elements
in the same order, so `rs` and `xn16` are the bits that kernel would produce from the written `R`. The next read skips
its step 1.

**Which write.** Let nl be the layer of the next half: l after half 0, l + 1 after half 1. The write is fused unless
one of these holds:
1. nl = 48: the last write of the prompt;
2. half 1 of layer 0 with the PLE on, because the PLE block reads and changes `R` before layer 1's first read
   (section 6.7);
3. half 1 of a layer l with `covers(l)` (a control vector, section 5.1; [06](06-engine-host-port.md), section 3.9),
   because the steering changes `R` after the write.

So half 0 is always fused. An unfused write uses decode's kernels, which are already sized by T:

| Case | Kernels | Then |
| --- | --- | --- |
| Not covered (cases 1 and 2) | `strata_gr_post` (section 5.4) | The next read runs `strata_pf_gr_norm`. |
| Covered, l < 47 | `strata_gr_cvec` with `write` = 1 (section 5.4): the write and the steering in one pass | as above |
| Covered, l = 47 | `strata_gr_post`, then `strata_gr_cvec` with `write` = 0 | nothing reads `R` after it |

- Upstream's unfused write is `gr_write` (`fmaf(bo, 2·sigm(inj/4), R)`), then `cvec_apply` if covered. Section 5.4
  shows that `strata_gr_post`'s `2·sigmoid_k(0.25·inj)` and `gr_write`'s gate are the same correctly rounded values on
  the Mac. So every write flavour gives the same `R`.
- `STRATA_GR_UNFUSED` is a CUDA tuning variable ([06](06-engine-host-port.md), section 4.3) and is not ported: on
  CUDA it computes the same bits through an F32 `xn`.

**Tolerance.** These kernels are B against their upstream kernels, apart from the deviations named. The GEMMs between
them carry section 6.2's tolerance.

The isolated `strata_pf_gr_write_norm` and `_lo` kernels and validated
`encode_prefill_gr_write_norm` entry point are implemented. The encoder binds
the residual in place, treats block output, injection and next-half norm as
read-only, validates their extents, alignment, ownership and output aliases,
and requires an ordered dispatch boundary. Mode 1 selects the low-remainder
pipeline; modes 0 and 2 omit that output. Serial and concurrent tests cover
T=1/3/8 and every BF16X2 mode. The fused residual, scale, high BF16 and
mode-1 low BF16 bits are compared with `strata_gr_post` followed by
`strata_pf_gr_norm` on the same inputs. This kernel is not yet connected to
Prefill GEMMs, layer scheduling or the production chunk loop.

### 6.4 GDN mixer

Upstream: `prefill.cpp:1342-1364`; `gdn_gates` (206/728), `gdn_conv_tiled` (235), `gdn_conv_hist` (253) and `gdn_l2`
(263), launched by `gdn_conv` (732), and `gdn_rec_cols_pipe` (380) and `gdn_out_norm` (436), launched by
`gdn_recurrence` (744). The dimensions are section 5.5's. `state` and `hist` are the layer's region of section 4.7,
written in place: prefill is the one writer of that region besides commit.

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_pf_gemm_f16` `attn_qkv` and `attn_gate` (section 6.2) | `mixed_h` → `qkv[T][10240]`, `z[T][6144]` |
| 2 | `strata_pf_gemm_bf16` `ssm_alpha` and `ssm_beta` | `mixed_bf` → `ab[T][96]` (alpha at 0, beta at 48) |
| 3 | `strata_pf_gdn_gates` | `ab`, `ssm_dt.bias`, `ssm_a` → `gate[T][48]`, `beta[T][48]` |
| 4 | `strata_pf_gdn_conv_tiled` | `hist`, `qkv`, `ssm_conv1d` → `h[T][10240]` |
| 5 | `strata_pf_gdn_conv_hist` | `hist`, `qkv` → `hist` |
| 6 | `strata_pf_gdn_l2` | `h` → `h` (q and k heads) |
| 7 | `strata_pf_gdn_rec` | `state`, `h`, `gate`, `beta` → `state`, `oc[T][6144]` |
| 8 | `strata_pf_gdn_out_norm` | `oc`, `z`, `ssm_norm` → `y_h[T][6144]` |
| 9 | `strata_pf_gemm_f16` `ssm_out` | `y_h` → `bo` |

Step 5 must follow step 4, which reads the old history, and the serial encoder orders them (section 8.1).

**`strata_pf_gdn_gates`**, grid ⌈48T / 256⌉ × 256:

```c++
i = 256*tg + tid;  if (i >= 48*T) return;
t = i / 48;  h = i % 48;
gate[i] = softplus_k(ab[t*96 + h] + dt[h]) * ssm_a[h];
beta[i] = sigmoid_k(ab[t*96 + 48 + h]);
```

The arithmetic after the dot is `strata_gdn_ab`'s (section 5.5). The inputs differ: decode dots the F32 `mixed`,
prefill takes a BF16 GEMM of `mixed_bf`.

The isolated `strata_pf_gdn_gates` kernel and validated
`encode_prefill_gdn_gates` entry point are implemented. They read the
T×96 F32 alpha/beta GEMM output and the 48-element `dt_bias`/`A` vectors,
write disjoint T×48 gate and beta outputs, and validate shader-safe rows,
extents, alignment, Context ownership and output aliases. Serial and
concurrent Prefill tests exercise both standalone and caller-owned Encoder
forms at T=1/3/8, including tail lanes, the softplus linear branch, negative
inputs, immutable read-only inputs, rollback and rejection inside an open
concurrent group. The stage is not yet connected to the Prefill BF16 GEMM or
chunk loop.

**`strata_pf_gdn_conv_tiled`**, grid (80, ⌈T / 64⌉) × 128. Thread `c = 128·tg.x + tid` walks rows
`t0 = 64·tg.y` .. `min(t0 + 64, T) − 1` of channel c:

```c++
in(t) = t >= 0 ? qkv[t][c] : hist[c*3 + t + 3];     // rows before the chunk come from the history
v0 = in(t0 - 3);  v1 = in(t0 - 2);  v2 = in(t0 - 1);
w0 = w[c*4];  w1 = w[c*4 + 1];  w2 = w[c*4 + 2];  w3 = w[c*4 + 3];
for (t = t0; t < t1; ++t) {
    x = qkv[t][c];
    s = fma(x, w3, fma(v2, w2, fma(v1, w1, v0 * w0)));
    h[t][c] = silu_k(s);
    v0 = v1;  v1 = v2;  v2 = x;
}
```

- The taps are the conv's inputs, not its outputs, so the tiles are independent. Every element has the expression
  of `strata_gdn_conv_l2` (section 5.5), without the L2 norm, which step 6 applies.
- Upstream walks the whole chunk in one thread per channel (`gdn_conv`, 217) when T ≤ 64. That kernel computes the
  same expression on the same inputs and is not ported.

The isolated `strata_pf_gdn_conv_tiled` kernel and
`encode_prefill_gdn_conv_tiled` host API are implemented. The encoder reads
the previous 3×10240 F32 history, T×10240 QKV rows and 10240×4 convolution
weights, and writes the unnormalized SiLU convolution to a disjoint output.
It validates shader-safe row counts, extents, alignment, Context ownership
and output/input aliases; read-only input views may overlap. It rejects an
open concurrent dispatch group so callers preserve ordered history
consumption. Serial and concurrent tests exercise standalone and caller-owned
Encoder forms at T=1/3/64/65/129 against a scalar oracle, including both sides
of 64-row tile boundaries, input immutability, guards, invalid views,
rollback, open-group rejection and a cancellation-sensitive FMA-order
negative control using non-dyadic values. It does not update history; step 5
is implemented by the separate `strata_pf_gdn_conv_hist` kernel and
`encode_prefill_gdn_conv_history` API below. Neither encoder is connected to
Prefill GEMMs or the chunk loop.

**`strata_pf_gdn_conv_hist`**, grid 80 × 128, one channel per thread:

```c++
c = 128*tg + tid;
for (k = 0; k < 3; ++k) { t = T - 3 + k;  v[k] = t >= 0 ? qkv[t][c] : hist[c*3 + t + 3]; }
for (k = 0; k < 3; ++k) hist[c*3 + k] = v[k];       // after all three reads
```

For T ≥ 3 this is the chunk's last three inputs. A shorter chunk keeps the newest 3 − T rows of the old history, as
`strata_gdn_conv_commit` does for n < 3.

The isolated `strata_pf_gdn_conv_hist` kernel and
`encode_prefill_gdn_conv_history` API are implemented. The kernel loads all
three next-history values before writing any slot, preserving the newest
`3 − T` old rows for T=1/2 and selecting the final three QKV rows for T≥3.
The validated standalone and caller-owned Encoder forms reject zero or
shader-unsafe row counts, undersized/misaligned/foreign views, history/QKV
overlap and calls in an open concurrent group. Serial and concurrent tests
cover T=1/2/3/8/65; same-Encoder convolution→history compositions at T=1/2/65
verify convolution consumes the old history before it is advanced. Validation
and rollback cover both API forms. The step remains isolated from the Prefill
GEMMs and chunk loop.

**`strata_pf_gdn_l2`**, grid (32, T) × 128. Threadgroup (b, t) normalises q/k head b of row t (b < 16 q, else k):

```c++
x = h[t][128*b + tid];
sq = butterfly(x * x);  if (lane == 0) part[warp] = sq;  barrier;
ss = ((part[0] + part[1]) + part[2]) + part[3];
h[t][128*b + tid] = x * rsqrt(ss + 1e-6f);
```

This is `strata_gdn_conv_l2`'s norm on the same value, so steps 4 and 6 together give that kernel's `h` bits.

**`strata_pf_gdn_rec`**, grid 192 × 128, 1-D. Threadgroup b handles v-head `head = b / 4` and the 32 columns
`col = 32·(b % 4) + c`, where `c = tid % 32`. Thread (rg, c), with `rg = tid / 32`, holds state rows
`32·rg` .. `32·rg + 31` of its column in registers. Threadgroup memory: `sk[128]`, `sq[128]`, `red[4][32]`
(1,536 B).

```c++
qh = head % 16;
device float* base = state + ((32*rg) * 48 + head) * 128 + col;
for (r = 0; r < 32; ++r) s[r] = base[r * 6144];
for (t = 0; t < T; ++t) {
    barrier;                                          // the previous row is done with sk, sq and red
    sk[tid] = h[t][2048 + qh*128 + tid];  sq[tid] = h[t][qh*128 + tid];
    barrier;
    g = exp(gate[t][head]);
    kv = 0.0f;
    for (r = 0; r < 32; ++r) kv = fma(s[r], sk[32*rg + r], kv);
    red[rg][c] = kv;  barrier;
    kv_col = ((red[0][c] + red[1][c]) + red[2][c]) + red[3][c];
    delta  = fma(-g, kv_col, h[t][4096 + head*128 + col]) * beta[t][head];
    o = 0.0f;
    for (r = 0; r < 32; ++r) {
        s[r] = fma(g, s[r], sk[32*rg + r] * delta);
        o    = fma(s[r], sq[32*rg + r], o);
    }
    barrier;
    red[rg][c] = o;  barrier;
    if (rg == 0) oc[t][head*128 + col] = (((red[0][c] + red[1][c]) + red[2][c]) + red[3][c]) * rsqrt(128.0f);
}
for (r = 0; r < 32; ++r) base[r * 6144] = s[r];
```

- Upstream launches `dim3(32, 4)` threads and loads the next row's inputs into registers during the current row
  (software pipelining). Neither changes an operation: the Mac's 1-D index is `32·rg + c`, the same thread for the
  same state rows, and it loads at the top of each row.
- The state arithmetic, the partition into four row groups and the serial `red` sums are `strata_gdn_step_norm`'s
  (section 5.5). Splitting a head's 128 columns over four threadgroups changes nothing, because no value crosses
  columns before the output norm.
- **Registers.** 32 state floats per thread at 128 threads.

**`strata_pf_gdn_out_norm`**, grid (T, 48) × 128. Threadgroup (t, head), thread `col = tid`:

```c++
x = oc[t][head*128 + col];
sp = butterfly(x * x);  if (lane == 0) wsum[warp] = sp;  barrier;
ss = ((wsum[0] + wsum[1]) + wsum[2]) + wsum[3];
v  = ((x * rsqrt(ss / 128.0f + 1e-6f)) * gamma[col]) * sigmoid_k(z[t][head*128 + col]);
y_h[t][head*128 + col] = half(v);                    // not saturated
```

- This is `strata_gdn_step_norm`'s norm: the same butterfly per simdgroup, the same serial sum of four partials and
  the same products.
- Upstream also writes the F32 `y` in place over `oc`. Nothing reads it (`ssm_out` takes `y_h`), so the Mac writes
  only `y_h`.

**Bits.**
- Given the same `qkv`, `gate`, `beta` and `z`, steps 4-8 give the bits of decode's GDN kernels over the same rows:
  the state and the history that write-through windows (section 8.5) would leave, and `y_h = half(y)` for decode's
  F32 `y`. `pf_gdn_eq_step_norm` (section 10.4) checks this for T = 1, 3, 64, 65 and 300. It runs the prefill
  kernels over T rows, and `strata_gdn_conv_l2`, `strata_gdn_step_norm` in mode 2 and `strata_gdn_conv_commit` in
  consecutive windows of up to 8 rows, and gives both paths the same `gate` and `beta`.
- Against upstream each kernel is B, apart from section 3.9's `exp`, `rsqrt` and `log1p_k`.
- `STRATA_GDN_CONV_SERIAL`, `STRATA_GDN_REC_HEADS` and `STRATA_GDN_PIPELINE` choose between upstream variants that
  give the same bits. They are CUDA tuning variables ([06](06-engine-host-port.md), section 4.3) and are not ported.

### 6.5 QSA mixer

Upstream: `prefill.cpp:1366-1560`; `rms_rows` 553/807, `rope` 566/812, `split_q` 586/833, `gate_attn` 592/837 and
`kv_append` 602/664 in `kernels.cu`; the batched indexer `native_qsa_indexer.cu:145-288` (`append_first`,
`append_blocks`, `append_tail`, `native_qsa_indexer_append_batch`); `qsa_block_scores` and `qsa_block_topk`
(`src/kernels/cuda/qsa_select.cu:606-748`); `qsa_decode_attn_batch` (`src/kernels/cuda/qsa_decode_attn.cu:211`). The
dimensions are section 5.6's. `qi` is the layer's index among the 12 QSA layers.

**Per QSA layer:**

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | F16 GEMMs `attn_k`, `attn_v`, `attn_q` (section 6.2) | `mixed_h` → `Kc[T][2][256]`, `Vc[T][2][256]`, `Qf[T][24][512]` |
| 2 | BF16 GEMMs `indexer.k_proj`, `indexer.q_proj` | `mixed_bf` (and `mixed_bf_lo`) → `idx_raw[T][128]`, `q_idx[T][4][128]` |
| 3 | `strata_qsa_norm_rope` `k` (2T rows, in place) | `Kc`, `attn_k_norm` → `Kc` |
| 4 | O-class: `strata_qsa_fwht256`, as decode step 5 | in place on `Kc` and `Vc` |
| 5 | `strata_qsa_kv_append_<mode>`, grid (2, T) | `Kc`, `Vc` → the K and V pools at cells `p0 .. p0 + T − 1` |
| 6 | `strata_qsa_norm_rope` `q` (24T rows, `Qf` at stride 512, out at stride 256) | `Qf`, `attn_q_norm` → `qcur[T][24][256]` |
| 7 | O-class: `strata_qsa_fwht256` on `qcur`, as decode step 10 | in place |
| 8 | `strata_qsa_norm_rope` `iq` (4T rows, in place) | `q_idx`, `indexer.q_norm` → `q_idx` |
| 9 | `strata_idx_append_first`, `strata_idx_append_blocks`, `strata_idx_append_tail` | `idx_raw`, `indexer.k_norm` → `tail`, `dead`, `pooled`, `block_pos` of layer `qi` |
| 10 | Selection: `strata_sel_scores` and `strata_sel_topk` per group of 256 rows (below) | `pooled`, `dead`, `q_idx` → `ids[T][2051]` |
| 11 | Attention: `strata_attn_chunk_<mode>` and `strata_attn_merge` per batch of 32 rows (below) | `qcur`, the pools, `ids` → `attn[T][24][256]` |
| 12 | O-class: `strata_qsa_fwht256` on `attn`, as decode step 14 | in place |
| 13 | `strata_pf_gate_attn` | `attn`, `Qf` → `attn_h[T][6144]` (F16) |
| 14 | F16 GEMM `attn_output` | `attn_h` → `bo` |

- Steps 3, 6 and 8 bind `pf_steps` as their `steps` and `PrefillParams` as their block. Upstream's `rms_rows` and
  `rope` are two launches. Its norm is the decode kernel's arithmetic: 256 threads, one product per thread
  (`ss += r*r` from 0), `block_sum`'s butterfly over eight partials, `rsqrtf(ss / W + eps)` and `(s*r)*w`. Fusing it
  with the rotation changes nothing, so only the RoPE angles differ, and those are section 3.9's angle row.
- Step 6 replaces upstream's `split_q`, which copies the first 256 of every 512 `Qf` values into `q`. Reading with the
  stride copies the same bits.
- Step 5 writes every row's K and V before any row selects (step 10), and step 9 appends every row's indexer key.
  Section 5.6 explains why that is safe; causality comes from each row's `n_kv`, `n_bid` and `width` in `pf_steps`.
- Upstream's `kv_mode` 1 stages the KV pools in device memory before step 5 (`prefill.cpp:1386-1412`). Unified memory
  makes the pools resident, so the Mac has no staging ([K2 section 3.5](audit/K2-qsa-kv-gdn.md)).

**The indexer, step 9.** `PrefillParams` gives two cells (section 6.1):
- `idx_f = p0 == 0 ? 4 : roundup4(p0)`: rows below it go through the single-row body, because they include cell 0 or
  complete a block that began in an earlier chunk;
- `idx_e = 4·⌊(p0 + T)/4⌋`: the end of the chunk's last completed block.

The three kernels run in this order. Each encodes only when it has work, and the host knows which from p0 and T.

**`strata_idx_append_first`**, grid 1 × 128, encoded when `p0 < idx_f`. It runs `strata_idx_append`'s window body
(section 5.6) over rows `t = 0 .. n_rows − 1` at `pos = p0 + t`, where `n_rows = min(idx_f, p0 + T) − p0` (at most 4).
These rows read `tail` slots written by an earlier chunk, write their own, and complete at most one block. At p0 = 0
they also write the spare `dead` and `pooled[0]`.

**`strata_idx_append_blocks`**, grid `(idx_e − idx_f)/4` × 128, encoded when `idx_e > idx_f`. Threadgroup `tg` takes
block `b = idx_f/4 + tg`, whose four cells all lie in the chunk; `d = tid`.

```c++
c = 4*b - p0;                                            // the block's first row in idx_raw
sum = float(half(idx_raw[c][d]));
for (j = 1; j < 4; ++j) sum = sum + float(half(idx_raw[c + j][d]));
mean = fma(0.25f, sum, 0.0f);
ss = butterfly(mean * mean);  if (lane == 0) part[warp] = ss;  barrier;
ss = butterfly(lane < 4 ? part[lane] : 0.0f);
scale = rsqrt(ss / 128.0f + eps);
vals[d] = (scale * mean) * gamma[d];  barrier;
y = vals[d];
if (d < 64) {
    i = d % 32;  float2 cs = rope[4*b][i];
    a = vals[i];  z = vals[i + 32];
    y = d < 32 ? fma(-z, cs.y, a * cs.x) : fma(z, cs.x, a * cs.y);
}
pooled[b][d] = y;
if (4*b + 4 == idx_e) { pooled[b + 1][d] = dead[d]; if (d == 0) block_pos = 4*b; }
```

- `b ≥ 1` always, so the block never needs `rope0`, and it never reads `tail`.
- Only the last block writes the spare row after it. Each earlier block's spare row would be overwritten by the next
  block, which is what the row-by-row form leaves.

**`strata_idx_append_tail`**, grid 3 × 128, always encoded. Threadgroup `s` is a tail slot:

```c++
last = p0 + T - 1;
cell = last - ((last % 4) - s + 4) % 4;                  // the last cell <= last with cell % 4 == s
if (cell >= p0) tail[s][d] = float(half(idx_raw[cell - p0][d]));
```

The cell can lie inside a completed block, as upstream's does: the row-by-row form writes a tail slot for every row
whose slot is below 3, completed block or not, and the last such write wins.

**Bits.** The three kernels leave `tail`, `dead`, `pooled[0 .. n_bid + 1]` and `block_pos` bitwise as `T` calls of the
row-by-row body would:
- every completed block sums the same four F16-rounded keys in cell order, with the same norm and rotation;
- the first kernel is the row-by-row body itself;
- the last write of every location is the one the row-by-row form makes last.

Upstream splits the work differently: its `append_first` runs only at p0 = 0, for cell 0, and its `append_blocks`
takes every block whose last cell lies in the chunk, reading the earlier chunk's `tail` for cells below p0. The
values are the same. `idx_batch_eq_sequential` (section 10.4) checks the Mac's three kernels against the window form
in windows of up to 8 rows, for p0 ∈ {0, 1, 3, 4, 6} and T ∈ {1, 2, 3, 4, 5, 9, 300}.
`STRATA_INDEXER_PER_TOKEN`, upstream's switch to one call per row, is not ported ([06](06-engine-host-port.md),
section 4.3).

**Selection, step 10.** The rows go in groups of 256, `g = 0 .. ⌈T/256⌉ − 1`, which is upstream's `sel_batch`. For
group g with rows `256g .. 256g + n − 1`:

1. For each batch of 8 rows starting at row `r0` (the last one shorter when T % 8 ≠ 0), `strata_sel_scores` with:
   - batch slot 0 (8 rows) or 1 (T % 8) as its block;
   - `q_idx + r0` and `pf_steps + r0`;
   - scores at `scores + (r0 − 256g) · max_blocks`;
   - grid `min(256, ⌈(top + 1)/8⌉)`, where `top = (p0 + r0 + rows)/4` is the `n_bid` of the batch's last row.
2. `strata_sel_topk` (`reg` or `mem`, by `max_blocks`, as in decode) with slot 2 (256 rows) or 3 (T % 256), grid n,
   `pf_steps + 256g`, the scores, and ids at `ids + 256g · 2051`.

- The scores buffer holds 256 rows (`1,024 · max_blocks` bytes, in F of section 6.1). Its groups reuse it in order;
  the serial encoder orders the reuse.
- Every row reads only blocks below its own `n_bid`, which are complete and final, and `dead` for its tail block. So
  appending the whole chunk in step 9 first gives every row the selection it would get in a window.
- Upstream with `STRATA_SELECT_OLD=1` runs `block_scores_kernel` over the 256 queries at once; its per-(block, row)
  arithmetic is the order of section 5.6's kernel, so the scores are the same bits. Its default is a 3×TF32
  tensor-core scorer (`qsa_block_scores_tc`), which is not bitwise. Section 3.9 lists the difference.

**The cost of selection.** Each 8-row batch reads every pooled key up to its `top` once, 512 B per block. Over a
prompt of N tokens that is about `N/8 · N/8 · 512 = 8N²` bytes per QSA layer, `96N²` for the 12 layers:
- That is 0.37 s at N = 32K, 5.9 s at 128K and 23.5 s at 256K at A.2's 280 GB/s, before any cache helps. A
  layer's pooled keys are 4 MiB at 32K and 32 MiB at 256K, so the system-level cache may absorb part of the reads.
- The arithmetic is about 64 lane operations per lane for each (block, row) pair: 16 FMAs, 20 shuffles and their
  adds, and the clamps. There are about `N²/8` pairs per layer, so at 128K the 12 layers need about 5·10^13 lane
  operations, several seconds on a 40-core GPU.

L-13 measures selection's share of a chunk. If it passes the 25% of section 2.6, Phase 2 adds
`strata_pf_sel_scores_mma` (work item L20).

**Attention, step 11.** The rows go in batches of 32, upstream's `attn_batch`. For the batch at `r0` with `n` rows:
- `strata_attn_chunk_<mode>`, grid (33, 2, n) × 256, then `strata_attn_merge`, grid (24, n) × 256;
- both take batch slot 4 (32 rows) or 5 (T % 32) as their block;
- the chunk kernel binds `qcur + r0`, `ids + r0 · 2051` and `pf_steps + r0`;
- the partials go to the F scratch for 32 rows (`32 · 817,600` bytes, section 6.1), reused by each batch;
- the output goes to `attn + r0`.

This is upstream's path under `STRATA_PROMPT_ATTN_OLD=1`, which calls `qsa_decode_attn_batch` 32 queries at a time.
Its default, `qsa_prompt_attn_batch`, runs the chunk on tensor cores and is not bitwise. Section 3.9 lists the
difference, and `strata_pf_prompt_attn_<mode>` (section 2.6) is the Phase-2 counterpart.

**`strata_pf_gate_attn`**, grid ⌈6144T/256⌉ × 256, one value per thread. `i = 256·tg.x + tid`, `t = i / 6144`,
`h = (i % 6144) / 256`, `d = i % 256`:

```c++
if (i < 6144*T) attn_h[i] = half(attn[i] * sigmoid_k(Qf[t*12288 + h*512 + 256 + d]));    // not saturated
```

Upstream computes `a * (1.0f / (1.0f + expf(-g)))` in the same order. Its `expf` is section 3.9's exp row. Decode's
`strata_qsa_gate` writes F32 for `strata_q_q8_1`; prefill writes F16 for its GEMM.

**Not ported.** These upstream debug variables are not ported ([06](06-engine-host-port.md), section 4.3):
- `STRATA_SEL_OVERLAP` (the overlap of neighbouring rows' selections);
- `STRATA_IDX_FP16_CHECK` (selection with F16-rounded keys, against the F32 one).

`STRATA_QSA_DUMP` and `STRATA_QSA_DUMP_LAST` write each QSA layer's selected cells to a file:
- a header `{qsa layer, pos0, T, cap}`;
- then `T · cap` cells.

The Mac's `ids` hold the same cells in the same layout, but each QSA layer overwrites them. So the dump needs a host
read of `ids` after every QSA layer, which needs the per-layer command buffers of section 6.1 and a wait after each
QSA layer. Phase 1 ignores the variables (section 9.3); honouring them is work item L23 (section 12).

### 6.6 MoE

Upstream: the MoE block of `Prefill::run` (`prefill.cpp:1563-1838`). The kernels in `kernels.cu` are `route`
(454/759), `swiglu_il` (518/776), `swiglu_pair` (525/781), `gather_rows16` (532/797) and `moe_combine` (540/802).
Outside that file: `iq_dequant_gu_f16` and `iq_dequant_f16` (`src/kernels/cuda/iq_kernels.cu:1408`, `:1377`, with
the per-format `dq_<fmt>` at 1002-1140), and the MMQ path of `moe_mmq.cu`. Constants are those of section 5.10:
k = 10, n_ff = 640, n_embd = 2560, and n_expert = 512 (256 for Coder).

The prefill MoE computes the decode MoE's function over T rows (section 5.10). It changes three things:
- Upstream groups the chunk's 10·T (row, rank) entries on the host. The Mac groups them with a counting sort on the
  GPU.
- Each expert's weights are decoded to F16 inside the GEMM tiles, so no dequant scratch is written.
- The products are F16 GEMMs (upstream's FP16 path) instead of Q8_1 dots.

| Step | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_pf_gemm_bf16` `ffn_gate_inp`, with its lo GEMM in BF16X2 modes 1 and 2 (section 6.2) | `mixed_bf` → `logits[T][n_expert]` |
| 2 | `strata_moe_route_r<NV>` with `div` = 1 (below), grid T × 32 | `logits` → `ids[T][10]`, `wts[T][10]` |
| 3 | `strata_pf_sort_count` | `ids` → `hist` |
| 4 | `strata_pf_sort_scan` | `hist`, `res` → `hist` (as offsets), `plan[l]`, `tiles` |
| | tiered mode: `encodeSignalEvent(ev, base + 2l + 1)` | |
| 5 | `strata_pf_sort_scatter` | `ids`, `hist` → `src`, `dst` |
| 6 | `strata_pf_gather_f16` | `mixed_h`, `src` → `Xs[10T][2560]` |
| 7 | `strata_pf_dequant_<fmt>` and `strata_pf_gemm_f16`, `ffn_gate_shexp` and then `ffn_up_shexp` | `mixed_h` → `sg[T][640]`, `su[T][640]` |
| 8 | `strata_pf_swiglu_pair` | `sg`, `su` → `sh_h[T][640]` |
| 9 | `strata_pf_dequant_<fmt>` and `strata_pf_gemm_f16`, `ffn_down_shexp` | `sh_h` → `shared[T][2560]` |
| 10 | `strata_pf_gemm_bf16` `ffn_gate_inp_shexp` (n = 1, so `simd`), with its lo GEMM in modes 1 and 2 | `mixed_bf` → `gl[T]` |
| | tiered mode: `encodeWaitForEvent(ev, base + 2l + 2)` | |
| 11 | `strata_pf_moe_gu_<fmt>` | `Xs`, `plan[l]`, `tiles`, `addr[l]` → `eh[10T][640]` |
| 12 | `strata_pf_moe_down_<fmt>` | `eh`, `plan[l]`, `tiles`, `dst`, `addr[l]` → `parts[T][10][2560]` |
| 13 | `strata_pf_combine` | `parts`, `wts`, `shared`, `gl` → `bo` |

- Upstream runs the shared expert and its gate before the grouping. The Mac runs the grouping first, so the shared
  expert overlaps the `MissResolver` in tiered mode, as in decode (section 8.3). No step reads another's output
  except along the arrows of the table, so the order changes no bit.
- In full mode the two event calls are left out, and the order is unchanged.
- `ids` and `wts` here are the `moe` set's buffers (section 6.1), not the QSA `ids`.

**The router, steps 1-2.** Upstream's prefill `route_kernel<REG>` is decode's router algorithm (section 5.10) with
three differences:
- it is built without fast math, so it calls the accurate `expf`;
- its reciprocal `1.0f / warp_sum(sum)` is correctly rounded;
- it ends with a division, `wout = selected / selected_sum`, where decode multiplies by `1.0f / ssum`.

It runs one warp per row, eight rows per block of 256 threads. It takes REG = 16 for 512 experts and REG = 8 for
256. Other widths go to `router_top10`, which no supported model uses.

The Mac runs `strata_moe_route_r<NV>` from section 5.10. Its new `setBytes` argument `div` chooses the last line:
- `div` = 1, prefill: `wts[t][lane] = sel / ssum`;
- `div` = 0, the window and the drafter: `sel * inv`.

So the prompt path's router matches upstream's except for `exp` (section 3.9). That includes 256 experts: on the
prompt path upstream uses `route_kernel<8>` for them, not the FP64 `router_top10`. Upstream writes `ids` from lane 0
and the Mac from the winner's lane. After the butterfly every lane holds the same winner, so the ids are the same.

**The grouping, steps 3-5.** The entries are `i = 10t + k` for t < T and k < 10, with expert `ids_flat[i]`. The three
kernels sort them by expert and keep entry order within an expert: a stable counting sort. Entry a of the sorted
order is upstream's slot `p`. In upstream's host loop (1583-1615):
- `off` is the exclusive prefix of the per-expert counts, in id order;
- the stable fill gives `slot[i] = p` and `src[p] = i / 10`.

The Mac writes `src[a] = i / 10` and `dst[a] = i`, which is the inverse of `slot`. So `Xs` and the expert outputs
hold the same rows as upstream's, in the same order.

`hist` is `uint [⌈10T/256⌉][512]`: one row per 256-entry tile, one column per expert. The 512 columns serve both
widths.

**`strata_pf_sort_count`**, grid ⌈10T/256⌉ × 256. Threadgroup b counts tile b. There are no atomics, so the counts
need no argument about order:

```c++
b = tg.x;  i = 256*b + tid;
threadgroup int e_s[256];
e_s[tid] = i < 10*T ? ids_flat[i] : -1;
barrier;
for (e = tid; e < 512; e += 256) {
    c = 0;
    for (j = 0; j < 256; ++j) c += e_s[j] == e;          // broadcast reads
    hist[b*512 + e] = c;
}
```

**`strata_pf_sort_scan`**, grid 1 × 512. Thread e owns expert e. It turns its `hist` column into slot offsets and
writes the layer's plan and the tile list:

```c++
e = tid;  nt = (10*T + 255) / 256;
c = 0;
if (e < n_expert) for (b = 0; b < nt; ++b) { h = hist[b*512 + e]; hist[b*512 + e] = c; c += h; }
cnt_s[e] = e < n_expert ? c : 0;
barrier;
if (tid == 0) {                                          // 512 serial steps, in id order
    gi = 0;  acc = 0;  ti = 0;  miss = 0;
    for (x = 0; x < n_expert; ++x) {
        off_s[x] = acc;  gix_s[x] = -1;  t0_s[x] = ti;
        if (cnt_s[x] > 0) {
            r = !tiered || res[l*n_expert + x] >= 0;
            P.grp_e[gi] = x;  P.grp_res[gi] = r;  P.grp_start[gi] = acc;  miss += !r;
            gix_s[x] = gi++;  ti += (cnt_s[x] + 63) / 64;
        }
        acc += cnt_s[x];
    }
    P.grp_start[gi] = acc;  P.n_groups = gi;  P.n_entries = 10*T;
    P.n_miss = miss;  P.skip = miss == 0;  P.n_tiles = ti;  P.bad = 10*T - acc;
}
barrier;
if (e < n_expert) {
    for (b = 0; b < nt; ++b) hist[b*512 + e] += off_s[e];      // first slot of e's entries in tile b
    for (j = 0; j < (cnt_s[e] + 63) / 64; ++j) tiles[t0_s[e] + j] = uint2(gix_s[e], off_s[e] + 64*j);
}
```

- `tiered` and `l` come with `setBytes`; T comes from `PrefillParams`. In full mode every expert is resident and
  `n_miss` is 0.
- `tiles` is `uint2 [⌈10C/64⌉ + 512]`, one `{group, first entry}` per tile of up to 64 entries. A group of n entries
  has ⌈n/64⌉ tiles. The sum over groups of ⌈n/64⌉ is at most ⌈10T/64⌉ + n_groups, so the list fits (section 6.1).
  It is rewritten for each layer. The layer's expert kernels read it before the next layer's scan, and the serial
  order of the encoder makes that safe.
- **The plan.** One `PfPlan` per layer, 6,400 B apart in a Shared buffer (section 6.1), in `common/types.h`:

  ```c++
  struct PfPlan {
      uint n_groups;          // distinct experts of the chunk
      uint n_entries;         // T * 10
      uint n_miss;            // groups whose expert is not resident (tiered mode; 0 in full mode)
      uint skip;              // n_miss == 0
      uint n_tiles;           // entries of tiles[]
      uint bad;               // entries whose id is outside [0, n_expert)
      uint pad[2];
      int  grp_e[512];        // expert id of group g, in id order
      int  grp_res[512];      // res[l][grp_e[g]] >= 0
      int  grp_start[513];    // entries of group g are grp_start[g] .. grp_start[g+1]-1
  };                          // 6,180 B; static_assert on both sides
  ```

  Its first four fields are those of `MoePlan` (section 5.10). In a prefill chunk the `MissResolver` reads
  `n_miss`, `grp_e` and `grp_res` at `PfPlan`'s offsets ([06](06-engine-host-port.md), section 9.3).
- **Range check.** Upstream fails with `prefill: routed id out of range` if an id lies outside [0, n_expert). The
  router cannot produce one:
  - a NaN probability becomes −FLT_MAX;
  - a chosen one becomes −∞;
  - so every rank finds a value above −∞ among the experts not yet chosen, and the 10 ids are distinct and in
    range.

  The Mac keeps the check anyway, at no cost. `count` ignores ids outside [0, 512), and `scan` ignores columns at or
  past n_expert, so a bad id is missing from `acc` and shows up in `bad`. After the chunk, the host reads each
  layer's `bad`. If any is non-zero, the call fails with upstream's message ([06](06-engine-host-port.md),
  section 2.4).
- **No usage counts.** In tiered mode the window's `strata_moe_plan` adds 1 to `usage[l][e]` for each entry. Upstream
  counts usage only on its verify path (`expert_source.cpp:1619-1621`), so a prompt does not move the adaptive swaps
  ([06](06-engine-host-port.md), section 9.4). The scan counts nothing either.

**`strata_pf_sort_scatter`**, grid ⌈10T/256⌉ × 256:

```c++
b = tg.x;  i = 256*b + tid;
e_s[tid] = i < 10*T ? ids_flat[i] : -1;
barrier;
e = e_s[tid];
if (i >= 10*T || e < 0 || e >= n_expert) return;
rank = 0;
for (j = 0; j < tid; ++j) rank += e_s[j] == e;           // earlier entries of e in this tile
a = hist[b*512 + e] + rank;
src[a] = i / 10;  dst[a] = i;
```

**`strata_pf_gather_f16`**, grid ⌈3200T/256⌉ × 256, one `uint4` (8 halves) per thread, as upstream's
`gather_rows16`:

```c++
j = 256*tg.x + tid;  a = j / 320;  q = j % 320;
if (a >= P.grp_start[P.n_groups]) return;                // 10T unless P.bad
((device uint4*) Xs)[j] = ((device const uint4*) mixed_h)[src[a]*320 + q];
```

**The expert tiles, steps 11-12.** Both kernels compute a 64-entry × 64-column tile in F32, from F16 operands, over
K in steps of 32:
- The **X tile** is 64 rows of the input, for entries `a0 .. a0 + 63`. A row past the group's end `a1` loads row
  `a1 − 1` again, and its output is not stored.
- The **W tile** is decoded from the expert's GGUF blocks in place. Each value is `half(value_<fmt>(row, k))`.

Grid (x, `⌈10T/64⌉ + min(n_expert, 10T)`) × 128. Threadgroup y takes tile y of the list, and threadgroups with
`y ≥ P.n_tiles` exit. Step 11 has x < 20 and step 12 has x < 40.

```c++
if (tg.y >= P.n_tiles) return;
uint2 tl = tiles[tg.y];  g = tl.x;  a0 = tl.y;  a1 = min(a0 + 64, P.grp_start[g + 1]);
ex = addr[l*n_expert + P.grp_e[g]];                      // read after the handshake (tiered mode)
threadgroup half  sx[64][32];                            // X tile, entry x k, 4 KiB
threadgroup half  sw[32][64];                            // W tile, k x column, 4 KiB
threadgroup float so[64][64];                            // output staging, 16 KiB, aliasing sx and sw
simdgroup_float8x8 acc[2][8];                            // simdgroup sg: entries 16*sg .. +15, all 64 columns; zero
for (k0 = 0; k0 < K; k0 += 32) {
    // X: 128 threads copy 64 rows x 32 halves as 256 uint4s
    sx[r][j] = X[min(a0 + r, a1 - 1)*K + k0 + j];
    // W: thread i decodes values k0 + 16*(i % 2) .. +15 of the tile's weight row i / 2 (below)
    sw[16*(i % 2) + j][i / 2] = half(value_<fmt>(wrow(i / 2), k0 + 16*(i % 2) + j));      // j < 16
    barrier;
    for (kk = 0; kk < 32; kk += 8) {
        simdgroup_load(x[i], &sx[16*sg + 8*i][kk], 32);                // i < 2
        simdgroup_load(w[c], &sw[kk][8*c], 64);                        // c < 8
        for (i < 2) for (c < 8) simdgroup_multiply_accumulate(acc[i][c], x[i], w[c], acc[i][c]);
    }
    barrier;
}
simdgroup_store(acc[i][c], &so[16*sg + 8*i][8*c], 64);  barrier;
epilogue (below), for the rows r with a0 + r < a1
```

| | `strata_pf_moe_gu_<fmt>` | `strata_pf_moe_down_<fmt>` |
| --- | --- | --- |
| X, K | `Xs`, K = 2560 | `eh`, K = 640 |
| Tile column c < 32 | gate row `32x + c`: `ex.role[0] + (32x + c)·gu_row` | down row `64x + c`: `ex.role[2] + (64x + c)·d_row` |
| Tile column c ≥ 32 | up row `32x + c − 32`: `ex.role[1] + (32x + c − 32)·gu_row` | down row `64x + c` |
| Epilogue | `eh[(a0 + r)·640 + 32x + c] = half_sat(silu_k(so[r][c]) * so[r][c + 32])` for c < 32 | `parts[dst[a0 + r]·2560 + 64x + c] = so[r][c]` for c < 64 |

- `gu_row = (2560 / qk) · block_bytes` and `d_row = (640 / qk) · block_bytes` come with `setBytes`, as in
  section 5.10. The host picks the two kernels of a layer from its gate/up type and its down type.
- K = 32 per step is one IQ4_NL block, half a Q2_0 block, or one of the eight 32-value sub-blocks of a 256-value
  super-block. So a step never splits a scale.
- The threadgroup memory is 16 KiB: `so` aliases the two tiles, which are dead after the loop's last barrier.
- The `simd` form is shown. The `mpp` form has the same staging and epilogue.
  - Each step runs `op.run(tX, tW, cT)`, where tX and tW are threadgroup tensors over `sx` and `sw`, and cT is the
    `cooperative_tensor` destination of a `matmul2d_descriptor(64, 64, 32)` with `execution_simdgroups<4>`.
  - After the loop, `cT.store` writes `so`.
  - This is llama.cpp's tensor path of `kernel_mul_mm`.
  - L-6 covers it as it covers section 6.2's `mpp`. `STRATA_METAL_GEMM` chooses the form for both sections.

**`value_<fmt>`** for the expert formats is upstream's `dq_<fmt>` (`iq_kernels.cu:1002-1140`, llama.cpp's
`dequantize_block_<fmt>`), written per value. It takes the block structs of section 5.8 and the codebooks of
section 3.7. For the value at position j of a 32-value sub-block:

| Format | Upstream | Value |
| --- | --- | --- |
| IQ2_XXS | `dq_iq2_xxs` (1002) | `(d · (0.5f + ls) · 0.25f) · grid[j] · ±1`, with `ls = aux32 >> 28` |
| IQ2_XS | `dq_iq2_xs` (1015) | as IQ2_XXS, with ls the sub-block's 4-bit scale for the value's 16-value half |
| IQ2_S | `dq_iq2_s` (1026) | as IQ2_XS. The grid index takes two high bits from `qh`, and the signs come from the block. |
| IQ3_XXS | `dq_iq3_xxs` (1036) | `(d · (0.5f + ls) · 0.5f) · grid[j] · ±1`, with `ls = aux32 >> 28` |
| IQ3_S | `dq_iq3_s` (1053) | `(d · (1 + 2s)) · grid[j] · ±1` (section 5.2) |
| IQ1_M | `dq_iq1_m` (1068) | `(d · (2s + 1)) · (q[j] + delta)`, where `delta = −1 ∓ 1/8` by the `qh` bit and q[j] ∈ {0, 1, 2} |
| IQ4_XS | `dq_iq4_xs` (1120) | `(d · (ls − 32)) · kvalues_iq4nl[nib]` (section 5.2) |
| IQ4_NL | `dq_iq4_nl` (1086) | `d · kvalues_iq4nl[nib]` |
| Q2_0 | `dq_q2_0` (1132) | `d · float(code − 1)` |

**Bits.**
- **The decoded weights are B** against upstream's `iq_dequant_*_f16`. Every product in the table is exact in F32:
  - the scale is a half times an integer of at most 6 bits;
  - the grid, `kvalues` and `q + delta` factors have at most 7 significant bits;
  - IQ4_XS, the largest case, needs 11 + 6 + 7 = 24 bits.

  Both sides then round once to half (RNE; an overflow gives infinity, as `__float2half` does).
- **The products are exact.** Each output sums them in the order of section 6.2's `simd` tile, and that order does
  not depend on the entry's place in the tile. So an entry's output depends only on its expert and its row, and the
  property of section 6 holds: the result does not depend on the chunk size.
- **Upstream's gate/up product** comes from one GEMM over interleaved gate and up rows (`dequant_gu_kernel`: row 2r is
  gate row r and row 2r + 1 is up row r). The Mac uses separate tile columns. Neither placement changes a column's
  sum.
- **The epilogue** is upstream's `swiglu_il_kernel`, `hf_sat(g / (1.0f + __expf(-g)) * u)`, with `exp` in place of
  `__expf` (section 3.9). `half_sat` is section 3.6's.
- **Tolerance.** The GEMM sums take the tolerance of section 6.2: U(2K) and R(2^-16), for K = 2560 and K = 640.

**What a tile costs.** Each tile decodes the whole of its expert's gate and up, or its down, so an expert with n
entries in the chunk is decoded ⌈n/64⌉ times. That is one value decode per 64 multiply-adds. Upstream instead
decodes each expert once into a ring of F16 buffers and runs two cuBLAS GEMMs per expert. On a Mac that would be
more than 1,500 dispatches per layer. L-13 measures the decode's share of a chunk; above 25%, work item L22 decodes
each expert once instead.

**The shared expert, steps 7-10.** Upstream (1575-1581):
- `native_proj` for gate and up from `mixed_h`;
- `swiglu_pair`;
- `native_proj` for down;
- `ffn_gate_inp_shexp` as a BF16 GEMM with n = 1, plus its lo GEMM with beta = 1.

Upstream fails with `prefill: shared gate is not BF16` unless that weight is BF16. `Prefill::init` makes the same
check, with the same message.

**`strata_pf_swiglu_pair`**, grid ⌈640T/256⌉ × 256:

```c++
i = 256*tg.x + tid;
if (i < 640*T) sh_h[i] = half_sat(silu_k(sg[i]) * su[i]);
```

Upstream's `swiglu_pair_kernel` computes `hf_sat(a / (1.0f + __expf(-a)) * u[i])`.

**`strata_pf_combine`**, grid (10, T) × 256, one output value per thread:

```c++
t = tg.y;  d = 256*tg.x + tid;
s = 0.0f;
for (k = 0; k < 10; ++k) s = fma(wts[t*10 + k], parts[(t*10 + k)*2560 + d], s);
bo[t*2560 + d] = fma(shared[t*2560 + d], sigmoid_k(gl[t]), s);
```

- This is upstream's `moe_combine_kernel`. Its loop is `s = fmaf(w, Dm[slot[t·10 + k]·N + d], s)` from s = 0, and
  its last line is `s + shared[i] * sigm(sg[t])`.
- `kernels.cu` is built without fast math, but with nvcc's default `-fmad=true`, which contracts that line into one
  FMA. L-9 checks the SASS. If the line is not contracted, the Mac writes `s + shared · g` instead.
- Upstream reads the expert outputs in sorted order through `slot`. The Mac's down kernel has already put each entry
  at its (row, rank) place, so the values are the same.
- It is not decode's `strata_moe_combine`. That kernel starts from the first product and adds the scaled shared
  output with a separate rounding (section 5.10).

**Tiered mode.** The layer's handshake is the window's ([06](06-engine-host-port.md), section 9.3):
- The signal follows the scan, so the `MissResolver` reads a complete `plan[l]` while the GPU runs steps 5-10.
- The routed experts read `addr[l]` only after the wait.
- The lookahead for layer l + 1 is every expert of that layer that is not resident, read into half `(l + 1) % 2`.
  So a half needs H = max_l nr_l slots ([05](05-memory-and-residency.md), section 5.4).
- The 20 s timeout applies per layer.

Upstream's prompt-path expert machinery is not ported:
- the host grouping and its 48 synchronisations per chunk;
- the staging ring;
- the `Stager` threads;
- the stream-all walk and its threshold `STRATA_PREFILL_STREAM_MIN`;
- the Strata-blob path (`blob_dequant_f16`). The Mac only accepts native packs ([03](03-model-reference.md),
  section 4.5).

The expert bytes are in unified memory, either resident or in a slot, so a tile reads them where they are.

**MMQ.** Upstream's default prompt path quantises `mixed` to Q8_1 and runs its MMQ kernels on 16 experts per launch
(`moe_mmq.cu`; `mmq_plan`, `prefill.cpp:461-490`). It takes the FP16 path above only under `STRATA_PREFILL_MMQ=0`,
or when it is built without MMQ. The Mac has only the FP16 path. `STRATA_PREFILL_MMQ` is ignored
([06](06-engine-host-port.md), section 4.3), and section 3.9 lists the difference. CUDA reference dumps of the prompt
path should be taken with `STRATA_PREFILL_MMQ=0` ([09](09-testing-and-validation.md)).

**`STRATA_DBG_NAN=1`** keeps upstream's layer report (1814-1836):
- The chunk is encoded as one command buffer per layer, as under "Long chunks" (section 6.1), and the host waits
  after each.
- After layer l it counts the non-finite values of `eh` (10T · 640), `parts` (10T · 2560) and `bo` (2560T).
- On the first such layer of each chunk, it prints upstream's line:

  ```
  strata dbg: layer %lld (mmq 0, types %d/%d, %zu experts): non-finite GU %lld H -1 Dm %lld bo %lld of T %lld
  ```

  - The types are the layer's gate/up and down GGML type ids, and the expert count is `n_groups`.
  - The GU count is over `eh`, because the Mac keeps no F32 gate/up rows. `half_sat` turns infinities into ±65504,
    so a non-finite `eh` value is a NaN.
  - Upstream's FP16 path also prints `H -1`.

**Deviations** (section 3.9):
- MMQ, upstream's default, becomes the FP16 tiles.
- cuBLAS's per-expert GEMMs become the grouped tiles.
- `__expf` in `sigm` and in the two SwiGLUs, and the router's `expf`, become `exp`.

### 6.7 Embedding and PLE

Upstream: the embedding at `prefill.cpp:1058-1111`; the PLE rows at 1112-1131; the PLE block at 1266-1321;
`native_ple_postops_batch` (`native_ple_postops.cu:205-220`).

**The embedding** is the first work of every chunk:

| # | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | `strata_embed_rows_<fmt>`, grid (10, T) × 256 (section 5.2) | `tokens`, `token_embd` → `emb[T][2560]` |
| 2 | `strata_embed_image`, grid (10, T) × 256, encoded only when the chunk has image rows | `tokens`, `img_in` → `emb` |
| 3 | `strata_gr_broadcast`, grid (40, T) × 256 (section 5.2) | `emb` → `R` |

- **Image rows.** The host writes `0x80000000u` into `tokens[t]` for each row whose embedding is an image row, and
  copies the encoder's row into `img_in[t]` ([06](06-engine-host-port.md), section 5.4). `strata_embed_rows_<fmt>`
  skips a row with bit 31 set:

  ```c++
  if (tokens[t] & 0x80000000u) return;                       // an image row: strata_embed_image writes it
  ```

  **`strata_embed_image`**, grid (10, T) × 256:

  ```c++
  t = tg.y;  v = 256*tg.x + tid;
  if (tokens[t] & 0x80000000u) emb[t][v] = img_in[t][v];
  ```

  `encode_embed_image` validates a positive prefill row count that keeps the
  shader's 32-bit `2560 * row + value` index in range, the manifest's
  256-thread geometry, complete token/image/output views, float and token
  alignment, Metal allocation bounds, Context ownership and an unsubmitted
  command buffer from the same queue. It verifies each CPU-visible view
  pointer against its underlying `MTLBuffer` and offset, so distinct no-copy
  handles over one backing range remain detectable. It allows token and image
  inputs to overlap because both are read-only, and rejects any output
  overlap. `encode_embed_rows` checks each host-visible token before dispatch:
  high-bit image markers skip vocabulary lookup, and every other id must be
  below the supplied `n_vocab`. For a token produced by an earlier GPU
  dispatch in the same ordered command buffer, the caller opts into the
  GPU-written path; the CPU does not inspect the mapped bytes and the gather
  shader checks the vocabulary bound before reading the table. An invalid GPU
  token writes a quiet NaN to the embedding row. After the producer runs, the
  caller keeps the token view unchanged until command completion.

`encode_prefill_embedding` has a caller-owned `Encoder` form and a standalone `CommandBuffer` wrapper. It checks
chunk metadata, token contents, the optional image view and GR output before appending the row gather, optional
image-row copy and GR broadcast to one ordered segment. The stages use the shared encoder APIs, which add the
required barriers in concurrent-dispatch mode; an open concurrent group is rejected because GR broadcast reads
rows written by earlier passes. A caller must propagate any failure so the whole segment, including earlier work,
is discarded. The standalone wrapper routes through `Encoder::encode` and discards a fresh command buffer on
failure. `Prefill::run` and the decode `Window` still need to call this coordinator. Every compute and blit encoder
also waits on and updates the `Context` queue fence (section 8.1; [06](06-engine-host-port.md), section 8.2).

  A decode window never carries an image row, because upstream sends any part of the prompt with image rows down the
  batched path (`generate.cpp:5017-5025`, "Picture rows need the batched path"). So the window form of
  `strata_embed_rows_<fmt>` never sees bit 31, and the window arena has no `img_in`.
- **Token ids.** The host rejects a token id outside `0 .. n_vocab − 1` before the prompt reaches the engine, with
  `ERR a token id is outside the vocabulary` ([06](06-engine-host-port.md), section 5.7). The gather shader also
  checks the bound before indexing the table; its GPU-written-token path writes a quiet NaN for an invalid value.
- **Bits.** Upstream gathers the whole chunk in one launch (`gather_dev`), and falls back to `core::embed_row` a token
  at a time when the chunk has image rows or a token outside the table. Its comment "C-4" states that the two paths
  have the same per-element arithmetic. The image rows are copies. So the Mac's single path gives upstream's bits
  for every chunk, with or without images (contract B).
- **Not ported:**
  - the layer split's hand-off upload (`hand_in_`), which replaces the embedding in a later stage of a multi-GPU
    split ([06](06-engine-host-port.md), section 4.3);
  - the codebook embedding gather (`embedding_gather_dev`) for Strata-format tables. The Mac only accepts native
    packs ([03](03-model-reference.md), section 4.5).

**The PLE rows.** The host decodes the chunk's PLE rows into `ple_in[k % 2]` on a thread, one chunk ahead, as
upstream's `ple_gather` does on its own thread. The rows are read from the PLE table with `pread`
([06](06-engine-host-port.md), section 7.4), never through a Metal buffer ([05](05-memory-and-residency.md),
section 6). Each row depends only on
the token and the two before it. The first chunk's rows are decoded before its command buffer is encoded. Each later
chunk's rows are decoded while the chunk before it runs.

**The PLE block** runs at layer 1, after `strata_gr_post` has applied layer 0's FFN write and before layer 1's
attention read, as in decode (section 5.1). It runs when the PLE is on: `ss.ple.ready()`, which `--no-ple` turns off.
It is encoded over the whole chunk:

| # | Kernel | Reads → writes |
| --- | --- | --- |
| 1 | BF16 key: `strata_util_to_bf16` (section 5.13), with `has_lo` in BF16X2 modes 1 and 2 | `ple_in[k % 2]` → `e16` (and `e16_lo`) |
| 2 | BF16 key: `strata_pf_gemm_bf16` `ple_key`, with its lo GEMM in modes 1 and 2 (section 6.2) | `e16` → `key[T][10240]` |
| 1' | Q2_0 or Q8_0 key: `strata_util_to_f16` | `ple_in[k % 2]` → `e16`, as halves |
| 2' | Q2_0 or Q8_0 key: `strata_pf_dequant_<fmt>` of `ple_key`, then `strata_pf_gemm_f16` | `e16` → `key` |
| 2'' | Q2_0 or Q8_0 key: `strata_util_to_bf16`, with `has_lo` in modes 1 and 2 | `ple_in[k % 2]` → `e16` (and `e16_lo`) |
| 3 | `strata_pf_gemm_bf16` `ple_value`, with its lo GEMM in modes 1 and 2 | `e16` → `value[T][2560]` |
| 4 | `strata_rms_norm_w`, 4T rows, gamma `ple_norm_key` | `key`, in place |
| 5 | `strata_rms_norm_w`, 4T rows of `R`, gamma `ple_norm_query` | `query` |
| 6 | `strata_ple_gate`, grid 4T × 512 | `key`, `query` → `gate[T][4]` |
| 7 | `strata_ple_broadcast`, grid (40, T) × 256 | `value`, `gate` → `gated` |
| 8 | `strata_rms_norm_w`, 4T rows, gamma `ple_norm_conv` | `gated` → `ple_norm` (the `query` buffer) |
| 9 | `strata_ple_conv_residual`, grid (40, T) × 256 | `ple_norm`, `hist`, `gated` → `R`, in place |
| 10 | `strata_ple_history`, n = `PrefillParams::T` | `ple_norm`, `hist` → `hist` |

- Rows 1-3 are upstream's three call shapes (1291-1300): `to_bf16` and two BF16 GEMMs; or `to_f16`, the native
  GEMM, then `to_bf16` over the same scratch. The serial encoder orders the overwrite of `e16` after the GEMM that
  reads it.
- Steps 4-10 are section 5.3's steps 3-8 and its `strata_ple_history`, with T = the chunk length. They are the same
  pipelines as the window's. Upstream's batch kernels (`rms_rep`, `gate`, `broadcast_batch`, `conv_residual_batch`,
  `history_batch`) have the same per-element arithmetic as its per-token ones, and section 5.3 specifies them.
  `encode_ple_post_projection` composes these steps after the caller has produced `key` and `value`; the Prefill and
  Window call sites remain open.
- `ple_norm` reuses `query` (section 6.1), as upstream reuses `query_norm`: `query` is dead once step 6 has run.
- **The activations differ from decode.** The prompt path rounds `ple_in` to BF16 for the value and for a BF16
  key, and to F16 for a native key. Decode reads the value with F32 activations, and a native key with Q8_1
  activations (section 5.3). The Mac follows each path as upstream has it, so a prompt row and a decode row of the
  same token differ in the PLE by these roundings, as they do upstream.
- **Sub-batches.** Upstream runs the block in sub-batches of `SB = min(T, region_bytes / per_token)` rows, carved
  from its idle scratch. The Mac's `ple` set is part of P (section 6.1), so the whole chunk runs at once. This
  changes no bit:
  - the GEMMs and steps 4-8 compute each row on its own;
  - upstream's sub-batch s reads the history after sub-batch s − 1's `history_batch`, which holds copies of the same
    `ple_norm` rows that the whole-chunk step 9 reads directly.
- **Tolerance.** Steps 1, 1', 2'' and 4-10 are B against upstream's batch path. The GEMMs of steps 2, 2' and 3 are
  U(2K) and R(2^-16) (section 6.2), and everything after them inherits that.
- **Timing.** The block starts upstream's `ple` phase mark (`kPfPle`). The host line's `PLE %.0f ms` (section 6.1)
  is the host time spent waiting for the chunk's PLE rows plus the time spent encoding the block. Upstream's
  `ms_ple` counts the same two things: the wait for its gather and the time spent issuing the block.
- **Not ported:** upstream's per-token fallback (1306-1321: `ple_block` and `ple_history_advance` for each row). It
  runs under `STRATA_PLE_BATCH=0`, without the native postops, or when the scratch region holds fewer than 64 rows.
  None of these occurs on a Mac. `STRATA_PLE_BATCH` is ignored ([06](06-engine-host-port.md), section 4.3). CUDA
  references of the prompt path must be taken with it unset ([09](09-testing-and-validation.md)).

**Deviations** (section 3.9): those of section 5.3, and cuBLAS's algorithm for the three GEMMs (section 6.2).

---

## 7. Threadgroup memory

### 7.1 The rule

Every kernel is written for 32,768 B of threadgroup memory, the `maxThreadgroupMemoryLength` of every supported GPU
(section 2.1; A.10 measured 32,768 B on M5 and saw 32,784 B fail pipeline creation). The budget counts the static
`threadgroup` arrays and any length set with `setThreadgroupMemoryLength:atIndex:`.

- Only two kernels set a length at encode time, both from the vocabulary: `strata_samp_greedy` (the penalty bitmap)
  and `strata_samp_split_merge` (the candidate lists) (section 5.12). Every other kernel declares its arrays
  statically. A set length is rounded up to a multiple of 16 B, as Metal requires.
- `Context` checks each pipeline after building it: `staticThreadgroupMemoryLength` plus the kernel's largest set
  length must not exceed the device's `maxThreadgroupMemoryLength`. The set lengths are known only after the model is
  loaded, so the two sampler kernels are checked again by the loader, which also refuses a vocabulary above 262,144
  (section 5.12). The error is
  `strata: kernel <name> needs <n> B of threadgroup memory; this GPU allows <m> B`, exit 1.
- `metal_context_test` ([08](08-build-and-distribution.md), section 6.4), which builds every pipeline, also fails if
  any pipeline's `staticThreadgroupMemoryLength` exceeds 32,768 B, whatever the device reports. A kernel edit that
  grows an array past the budget fails there, not on a user's Mac.
- `STRATA_METAL_SMALL_TILES` stays reserved ([06](06-engine-host-port.md), section 4.4): no Phase-1 kernel has a
  large-tile variant.

### 7.2 Upstream kernels that exceed the budget

Upstream asks for more than 48 KiB of dynamic shared memory at eight sites ([K5 section 5.3](audit/K5-orchestration-vram-tests.md))
and uses about 32 KiB or more in a few more kernels ([A section F.5](audit/A-model-and-kernels.md)). Each one, and
what the Mac does instead:

| Upstream kernel | Upstream footprint | Mac | Bits |
| --- | --- | --- | --- |
| `gr_down_multi_kernel<1280>`, `<2560>` (`fused_gr.cu:726-728`) | T · tile · 4 B: 40 KiB or 80 KiB at T = 8 | `strata_gr_down` stages `xn` in tiles of TILEV = 512 values per row, 16 KiB at T = 8 (section 5.4). The norm writes `xn` to device memory first, as upstream's `gr_norm_multi` does. | Unchanged: each lane adds its chunks `lane + 32m` in ascending order whatever TILEV is |
| `gr_down_kernel` (static `xn[10240]`), `gr_down_staged_kernel`, `gr_down_v3_kernel` | 40 KiB; `2·8·H_TILE·4` B; 80 KiB | Not ported (section 2.5): the multi-row kernel at T = 1 is the single-token path | — |
| `block_topk_reg_kernel<33>` | `hist[32][256]` int, 32 KiB, plus its scans | `strata_sel_topk` `reg`: one histogram per four simdgroups, `hist[8][256]` (8 KiB) plus the scans (section 5.6) | Unchanged: integer atomics, any order |
| `sampler_split_merge_kernel`, the coupled merge | `lists[64·64]` int2 + 1 KiB: 33,792 B | `strata_samp_split_merge` sizes `lists` to `n_blocks · 64 · 8` B: 31,232 B at 248,320 tokens, 32,768 B at the 262,144-token limit, with no other array (section 5.12). The coupled draft uses the same kernel at one row. | Unchanged: the lists are the same values in the same order |
| `block_scores_tc_kernel` (`qsa_select.cu:717`) | `(QT·QS + NB·KS) · 4` B | Phase 2 (section 2.6). Phase 1 scores prompt rows with `strata_sel_scores` in query batches (section 6.5). | Phase 1: the decode kernel's bits |
| `prompt_attn_kernel<mode>`, `prompt_attn_i8_kernel` (`qsa_prompt_attn.cu:639`, `:670`) | `sizeof(Smem)`, above 48 KiB | Phase 2 (section 2.6). Phase 1 runs `strata_attn_chunk_<mode>` and `strata_attn_merge` in query batches (section 6.5). | Phase 1: the decode kernel's bits |
| `qsa_attend_kernel`, canonical (`qsa.cu:684`) | grows with `max_ids`; `exit(1)` above the limit | Not ported: the canonical path is not run on a Mac | — |
| cuBLAS prefill GEMMs | library-chosen | The prefill GEMM tiles of sections 6.2 and 6.6, each within the budget | Prefill tolerance (section 10.3) |

### 7.3 The largest Mac footprints

Every kernel not listed uses less than 8 KiB.

| Kernel | Threadgroup memory | Section |
| --- | --- | --- |
| `strata_samp_split_merge` | `n_blocks · 512` B: 31,232 B at 61 blocks, 32,768 B at 64 | 5.12 |
| `strata_samp_greedy` | bitmap `⌈min(n_vocab,131072)/32⌉ · 4` B plus 256 B, rounded to 16 B: at most 16,640 B | 5.12 |
| `strata_gr_down` | `tile[T][TILEV]` floats: 16 KiB at T = 8 and TILEV = 512 (32 KiB at TILEV = 1024, the L-2 maximum) | 5.4 |
| `strata_sel_scores` | the rows' `qidx` (16 KiB at T = 8) and steps | 5.6 |
| `strata_attn_chunk_<mode>` | `sq`, `sp`, `srow`: 15,616 B | 5.7 |
| `strata_mv_<fmt>_n<NC>` | `partial[3][NC][R][32]` floats: 12 KiB at NC = 8 and R = 4, 24 KiB at R = 8, the largest R allowed | 5.8 |
| `strata_gr_up` | `lo_s[8][320]` and `g_s[8][4][16]`: 12 KiB | 5.4 |
| `strata_sel_topk` `reg` | `hist[8][256]` int (8 KiB) plus the scans | 5.6 |
| The prefill GEMMs | sections 6.2 and 6.6 | 6 |

A tuning measurement (L-2, L-5, L-7) may change a tile size only within these limits. A tile size that would exceed
32,768 B at T = 8 is not a candidate.

---

## 8. Encoding

### 8.1 Segments, encoders and the fence

Every buffer is `MTLResourceHazardTrackingModeUntracked` ([05](05-memory-and-residency.md), section 7), so Metal
orders nothing by itself. The engine orders work with exactly two mechanisms:

1. **Inside a compute encoder**, the dispatch type. An encoder of `MTLDispatchTypeSerial` runs its dispatches in
   the order they were encoded, and each one sees the writes of the ones before it (`MTLCommandBuffer.h`; A.11
   test 3 found every policy exact). Kernels never synchronise across threadgroups inside one dispatch.
2. **Between encoders**, one `MTLFence` per queue, created by `Context::open`. Every encoder, compute or blit, calls
   `waitForFence:` first and `updateFence:` last. This orders encoders within a command buffer and across command
   buffers on the queue. Without it, a second command buffer read the first one's output too early in 20 of 20 runs
   (A.11 test 4).

`Context::open` initializes the fence with one update-only blit encoder, commits that seed command buffer and waits
for it before exposing the Context. This is the only encoder that does not wait on the fence first. Between command
buffers, callers must register every producer/consumer dependency with
`consumer.depends_on(producer)` before submission. The wrapper checks that both buffers use the same queue, that the
producer was created first, and that the producer completed successfully before the consumer is submitted.
`wait_until_completed` provides that checked wait. A queue fence orders encoders but does not propagate a producer's
GPU execution failure, so the consumer must remain unsubmitted until success is known. The wrapper does not infer
buffer reads and writes; production dispatch must register each cross-command-buffer dependency.

A **segment** is the run of dispatches between two command-buffer-level operations: the start of the command
buffer, an `encodeSignalEvent`, an `encodeWaitForEvent`, a blit encoder, or the end. Each segment is one compute
encoder. The rules:

- A command buffer never contains two compute encoders back to back with nothing between them: one segment, one
  encoder. Splitting an encoder costs an encoder boundary and a fence wait and buys nothing.
- No kernel argument changes between the encodes of two windows except through the parameter block
  ([06](06-engine-host-port.md), section 8.2, rule 2). Buffers are bound by fixed address and offset.
- The host never writes a buffer that a committed, unfinished command buffer reads or writes, except through
  `MTLSharedEvent` handshakes that the command buffer waits for (tiered mode, section 8.3). The parameter ring of
  section 4.2 has two slots for this reason.
- Every command buffer and every encoder carries a label ([06](06-engine-host-port.md), section 8.2, rule 6):
  `window T=<T> pos=<pos0>`, `commit n=<n_keep>`, `mtp round`, `mtp step <j>`, `mtp prefill`,
  `prefill chunk <k>`. In tiered mode the encoders are labelled `layer <l>` and `layer <l> shared` (section 8.3);
  otherwise an encoder takes its command buffer's label.
- `STRATA_METAL_ERRORS=1` creates command buffers from a descriptor with
  `MTLCommandBufferErrorOptionEncoderExecutionStatus` ([06](06-engine-host-port.md), section 4.4). The completion
  handler of every command buffer checks `status` and `error` and reports a failure as
  [06](06-engine-host-port.md), section 2.4, specifies.

### 8.2 Dispatch policy (`STRATA_METAL_DISPATCH`)

| Value | Encoder | Ordering inside a segment |
| --- | --- | --- |
| `serial` (default) | `computeCommandEncoderWithDispatchType:MTLDispatchTypeSerial` | Dispatch order |
| `concurrent` | `MTLDispatchTypeConcurrent` | The host declares groups of independent dispatches and encodes `memoryBarrierWithScope:MTLBarrierScopeBuffers` after each group. A dispatch outside a declared group is a group of one. |

The common `Encoder` in `include/strata/metal/encoder.hpp` and `src/metal/host/encoder.mm`
([06](06-engine-host-port.md), section 1.5) creates one labeled, fence-ordered compute encoder per segment. Its
`dispatch(kernel, grid, threads, bindings...)` validates that the kernel and threadgroup match the manifest, that
bound buffers belong to the Context and cover their declared extents/alignment, and that the supplied buffer and
dynamic threadgroup-memory slots exactly match the active slots from pipeline binding reflection. It clears stale
slots when a later dispatch uses a different pipeline. Inline-byte payloads are limited to 4 KiB per segment;
dynamic threadgroup-memory bindings must fit the pipeline/device limits. Unset, empty or `serial` selects
`MTLDispatchTypeSerial`; `concurrent` selects `MTLDispatchTypeConcurrent` and emits a buffer-scope barrier after
each dispatch outside a group or after each group. `group_begin()` / `group_end()` are no-ops in serial mode; in
concurrent mode groups must be balanced and non-nested. The caller must ensure group members have no write/read or
write/write dependency; the wrapper does not infer resource aliases between dispatches. An invalid mode or failed
body/dispatch discards the unsubmitted command buffer. `metal_encoder_test` covers this generic wrapper. Text/image
embedding, GR broadcast, GR norm/post/cvec, weighted RMS norm, mat-vec and PLE expose typed caller-owned Encoder overloads;
the remaining typed dispatchers are still standalone. Production parity remains gated by
`metal_dispatch_modes_test`.

The groups the window and the commit declare:

| Where | Group (dispatches that may run together) |
| --- | --- |
| PLE block (section 5.3) | the `ple_key` mat-vec (after its quantisation), the `ple_value` mat-vec, the query norm (step 4) |
| GDN block (section 5.5) | steps 2, 4 and 5: `attn_qkv`, `strata_gdn_ab`, `attn_gate` |
| QSA block (section 5.6) | steps 2, 3, 8 and the mat-vec of step 11: `indexer.k_proj`, `attn_k`, `attn_v`, `attn_q`, `indexer.q_proj` |
| QSA block | the three `strata_qsa_norm_rope` calls (steps 4, 9, 11) |
| QSA block | steps 6 and 7: `strata_qsa_kv_append_<mode>`, `strata_idx_append` |
| FFN (section 5.10) | the shared expert's gate and up mat-vecs and `ffn_gate_inp_shexp` |
| Commit (section 8.4) | all of it: one group |

- A group's members must not write anything another member reads or writes. The groups above satisfy this by
  construction; `metal_dispatch_modes_test` (section 10.4) runs windows of T = 1..8, commits, MTP rounds and a
  prefill chunk in both modes and compares every output and state buffer bitwise.
- The cost (A.11 test 5): a dependent dispatch costs 0.87 µs in a serial encoder and 1.11-1.14 µs with a barrier
  after each in a concurrent one, and two independent half-size dispatches per barrier saved about 10%. So
  `concurrent` pays only where groups are wide. L-14 measures a window of each T and a commit in both modes. The
  default becomes `concurrent` only if it is at least 3% faster on the decode benchmark of every measured family.

### 8.3 The command buffers

One `MTLCommandQueue` carries all work ([06](06-engine-host-port.md), section 8.1). At most one command buffer that
reads expert slots is in flight ([06](06-engine-host-port.md), section 9.3).

| Command buffer | Contents | Segments | After commit, the host |
| --- | --- | --- | --- |
| Window | section 5.1, steps 1-14 | 1 in full mode; 97 in tiered mode (below) | waits (`STRATA_METAL_WAIT`), then reads `samp_out` or runs the CPU tail (section 5.12) |
| Commit | section 8.4 | 1 | does not wait; `wait_commit()` waits for it ([06](06-engine-host-port.md), section 10.1) |
| Commit + round | the commit, then the MTP round of section 5.11 | 1 | waits; reads the first draft |
| MTP step j | section 5.11 | 1 | waits; reads draft j |
| Write-through window | a window that is also its own commit (section 8.5) | as a window | as a window |
| MTP prefill | section 5.11 | 1 | waits |
| Prefill chunk | section 6.1 | 1 in full mode; 97 in tiered mode | section 6.1 |
| Session zero, conversation copies | a blit encoder (`fillBuffer`), or nothing when the queue is idle and the CPU copies ([06](06-engine-host-port.md), section 11.3) | 1 | waits |

**Tiered mode.** Each layer's FFN is encoded as:

```
router mat-vec, strata_moe_route, strata_moe_plan          (end of segment l)
encodeSignalEvent(ev, base + 2l + 1)
strata_q_q8_1, the shared expert (section 5.1, steps 7-8)  (start of segment l + 1)
encodeWaitForEvent(ev, base + 2l + 2)
routed experts, strata_moe_combine, the FFN write, layer l + 1 ... strata_moe_plan of layer l + 1
```

The plan is encoded right after the route, before the quantisation and the shared expert, so the shared expert can
run on the GPU while the `MissResolver` reads the plan and fills slots. M-1 measures how much of the hop this hides. The shared expert reads only `xq` and dense
weights. The routed experts read `addr[l]` only after the wait. Moving the shared expert does not change any bit,
because no kernel between the route and the combine reads another's output except along the chains of section 5.1.
Full mode encodes the same order without the two event calls. The window's 48 signals and 48 waits give 97
segments; `base` advances by 2 · 48 + 2 per window ([06](06-engine-host-port.md), section 9.3).

The signal for layer l fires only when every command encoded before it has completed. Those include the reads of
half `l % 2` by layer l − 2 and the whole previous window, which is what makes the slot rule of
[06](06-engine-host-port.md), section 9.3, hold. The shared-expert encoder after the signal does not read slots.

**Re-encoding.** The host encodes every command buffer afresh from the same encode function, with the shape-level
choices as arguments: T and `nc`, full or tiered, write-through, `ple_on`, `covers(l)`, the KV mode and the
sampler path (section 5.12). A window of T = 8 is about 1,500 dispatches; at A.5's 0.1 ms per 1,000 the encode
costs about 0.15 ms of CPU time per window, against a window of about 25 ms. H-4 ([06](06-engine-host-port.md),
section 15) measures it.

**Indirect command buffers.** `STRATA_METAL_ICB=1` is reserved for an ICB per T and per mode combination, recorded at
startup and executed with `executeCommandsInBuffer:withRange:`. Phase 1 does not implement it: the variable prints
`strata: STRATA_METAL_ICB is not implemented; encoding each window` once and is ignored. It becomes work item L17
(section 12) only if H-4 measures an encode time above 1 ms. Tiered mode's event calls cannot live inside an ICB,
so an ICB would hold one range per segment.

### 8.4 Commit

`commit(n_keep)` writes `n_keep` into the window's parameter slot (section 4.2) and encodes, in layer order:

| Layer kind | Kernels | Section |
| --- | --- | --- |
| GDN (36) | `strata_gdn_conv_commit`; `strata_gdn_step_norm` mode 1 | 5.5 |
| QSA (12) | `strata_idx_append`, commit form | 5.6 |
| Once, with the PLE on | `strata_ple_history` with n = `n_keep` | 5.3 |

- The commit reads the window arena's per-layer regions (`qkv_L`, `h_L`, `gate_L`, `beta_L`, `idx_raw_L`,
  `ple_norm`), the indexer snapshot and the parameter slot. The next window overwrites the arena, so the commit must
  be encoded before the next window on the same queue. The fence orders the two.
- No commit dispatch reads what another commit dispatch writes: each layer's state, history and indexer buffers are
  its own, and `strata_gdn_step_norm` reads `h_L`, not the conv history. In `concurrent` mode the whole commit is one
  group.
- The commit writes nothing the window's outputs depend on, so the host may read the window's results before the
  commit runs, as upstream's asynchronous commit allows (`set_commit_async`).
- Rejected KV cells are not touched; the next window overwrites them before any row selects them (section 4.5).

### 8.5 Write-through windows

A window is **write-through** when the host knows at encode time that every row will be kept:
- plain decode with `--spec 0`, where every window is one token and nothing drafts;
- the short reads after a resume point, which feed prompt tokens through decode windows
  ([06](06-engine-host-port.md), section 11.2, step 8). Every row of such a window is kept.

A write-through window sets `n_keep = T` in its parameter block and folds the commit into the window:

| Commit work | In a write-through window |
| --- | --- |
| GDN state | `strata_gdn_step_norm` runs in mode 2: it emits every row, as mode 0, and writes the state back at the end, as mode 1 |
| GDN conv history | `strata_gdn_conv_commit` right after the layer's `strata_gdn_step_norm`. The layer's `strata_gdn_conv_l2`, the only reader of the history, ran before it in the same serial segment. |
| Indexer | Nothing: the window form of `strata_idx_append` has appended all T rows, which is the state the commit form's restore and replay of T rows would rebuild. `strata_idx_snapshot` is not encoded. |
| PLE history | `strata_ple_history` with n = T right after `strata_ple_conv_residual` at layer 1, the history's only reader |

`encode_window_gdn_write_through_block` implements the GDN state/history ordering for one layer and validates
`T == n_keep == rows` plus the matching `nc` before encoding. The model-wide GDN/QSA/PLE sequence, write-through
Window scheduler and rollback-free call sites remain open.

- The result is bitwise equal to the window followed by `commit(T)`: mode 2 performs mode 0's arithmetic on the same
  rows and mode 1's write, and every other step is a copy. `write_through_eq_commit` (section 10.4) checks every
  output and state buffer for T = 1..8.
- The saving is the commit's command buffer and its replay. The 36 GDN states (113 MiB) are read and written once
  instead of being read by the window and then read and written again by the commit: about 0.4 ms per token at
  A.2's 280 GB/s.
- A write-through window can never be rolled back, so the host uses it only in the two cases above. A drafter that
  reads the window afterwards (the MTP prefill after a short read, section 5.11) still finds the window's arena
  intact.

### 8.6 Arenas and their lifetimes

| Arena ([05](05-memory-and-residency.md), section 2.3) | Written by | Read later by | Valid until |
| --- | --- | --- | --- |
| Window scratch | the window | the commit (per-layer replay inputs); the MTP round (`R` rows 0..T−1), or `draft_first` (`R` row 0) after the prompt's T = 1 window; the CPU tail (`sel`, in Shared memory) | the next window starts |
| Parameter slot `round % 2` | the host, before the window | the window and its commit | the round after next reuses the slot |
| Prefill scratch | a prefill chunk | the MTP prefill (`R` rows of the chunk); the host's `STRATA_PREFILL_DUMP_R` read | the next chunk starts |
| MTP scratch | the drafter | the next step of the same round | the next round |

Only one round is in flight in Phase 1, and every later reader is encoded on the same queue before the arena's next
writer, so the fence is the only ordering needed. A Phase-2 overlap of the next window with the commit and the round
([06](06-engine-host-port.md), section 10.2) would need a second window arena and is work item L18 (section 12).

---

## 9. Dispositions: flags, variables and the fast folder

### 9.1 Rule

- Each Metal kernel has one form. A switch between two forms exists only where this section, or [06](06-engine-host-port.md),
  section 4.4, lists it.
- The Mac implements the arm that upstream runs by default under `--native`, unless this document says otherwise. The
  exceptions are three prompt-path arms of section 9.3:
  - prefill attention and prefill block scores, where the Mac runs upstream's older arm until the Phase-2 kernels
    exist (section 2.6);
  - the routed experts, where the Mac runs upstream's `STRATA_PREFILL_MMQ=0` arm, the FP16 GEMMs, instead of its
    default MMQ (section 6.6).
- A flag that selects another upstream arm is IGNORE ([06](06-engine-host-port.md), section 3.1), and prints
  `strata: --X has no effect on the Metal engine`. A flag whose arm is always on is IGNORE **without** a warning, like the
  native arms ([06](06-engine-host-port.md), section 3.2).
- A kernel-scope variable is either honoured with upstream's meaning or ignored silently
  ([06](06-engine-host-port.md), section 4.1). New switches are named `STRATA_METAL_*`.

### 9.2 Kernel flags

Upstream lines are in `src/program/generate.cpp`.

| Flag | Upstream | Mac |
| --- | --- | --- |
| `--gr-native-mmvf` | Pins the projection reduction tree and FP32 inputs (183, 1067). `--native` sets it (1520), and the fused GR read requires it (1923). | Always on: the GR kernels of section 5.4 are this contract. IGNORE without a warning. |
| `--gr-fp32-activations` | FP32 activations in the canonical `gr.cu` kernels (182, 1066). `gr.cu:363` already uses FP32 when `--gr-native-mmvf` is set. | Always on in effect. IGNORE without a warning. |
| `--cpu-oracle-q8-0` | Pinned x86 activation scales for the CPU expert kernels (209, 1079) | IGNORE: the Mac has no CPU expert path. |
| `--stream-token` | Orders the token's work on the session stream instead of synchronising the device (180). `--native` sets it (1519). | Always on: one queue, and the host waits only for the window ([06](06-engine-host-port.md), section 8). IGNORE without a warning. |
| `--check-logits` | Reads the logits back each token and fails on a non-finite value (181, 5875-5887) | **KEEP.** After the window's wait the host scans row 0 of the window's logits, which are in window scratch, a Shared buffer ([05](05-memory-and-residency.md), sections 2.3 and 7), with no copy. The message and exit code are upstream's: `strata generate: <n> of <n_vocab> logits are not finite at position <pos>`, exit 1. |
| `--shared-late` | Plan v0.3 A/B: the shared expert inside `post[l]`, the old order (314, 1497) | IGNORE. The order is fixed (section 8.3). |
| `--keep-canonical` | Loads canonical copies of natively served tensors (315, 1848-1869) | IGNORE. The Mac loads no canonical tensor ([06](06-engine-host-port.md), section 6.1). |
| `--no-token-graph` | Two CUDA graphs per layer instead of one per token (316, 3617) | IGNORE. The Mac has no graphs; `--no-capture` is its own switch ([06](06-engine-host-port.md), section 3.9). |
| `--no-fused-gr` | The six-kernel native GR read and a separate write (317, 1923) | IGNORE. The Mac has only the fused read. Upstream also turns the fused read off for `--dump-layers`, `--dump-halves` and `--stage-timing` (1923-1924); the Mac does not, and dumps R as in section 9.4. |
| `--no-fast-attn` | Gather plus one threadgroup per head for QSA attention (318, 1919) | IGNORE. Section 5.7 is the fast form. |
| `--no-publish-kernel` | Memcpy nodes for the doorbell and the QSA step (319, 1920) | IGNORE. There is no doorbell ([06](06-engine-host-port.md), section 1.2), and per-step values come from the parameter block (section 8.1). |
| `--no-fused-gdn` | The llama.cpp-layout GDN step and a separate output norm (320, 1921) | IGNORE. Section 5.5 is the fused form. |
| `--no-fast-select` | FP64 row scores and a bit-serial cell top-k (321, 1922) | IGNORE. Section 5.6 is the fast form; Metal has no FP64. |

[06](06-engine-host-port.md), sections 3.2 and 3.9, point here for these rows.

### 9.3 Kernel-scope variables

Every variable of [audit B 1.3](audit/B-engine-host.md)'s kernel-scope list. "Ignored" means read by nothing.

| Variable (upstream line) | Upstream meaning | Mac |
| --- | --- | --- |
| `STRATA_KV_ROT` | Hadamard rotation of int8 KV | **Honoured** (sections 5.6 and 5.11) |
| `STRATA_PREFILL_BF16X2` (`prefill.cpp:108-121`) | 1: each BF16-weight prompt GEMM also multiplies the activation's BF16 remainder, hyper-connection projections included; 2: all but those; 0 (default): off | **Honoured**, same modes (section 6.2) |
| `STRATA_PREFILL_TIMING` (960) | Prompt-path GPU time by phase | **Honoured**, from counter sample buffers ([06](06-engine-host-port.md), section 4.2) |
| `STRATA_PREFILL_DUMP_R` (1886-1897) | After each chunk, appends `int64 pos` and the D = 10,240 floats of R for every position that is a multiple of 64 (the file is truncated at chunk 0) | **Honoured**, same format. The host reads the rows from the prefill arena after the chunk's wait (section 6.1). |
| `STRATA_PROMPT_ATTN_OLD` (1549) | Prefill attention through the decode kernel, 32 queries at a time | Phase 1 always runs this arm (section 6.5). Once `strata_pf_prompt_attn_*` exists (section 2.6), honoured with upstream's meaning. |
| `STRATA_SELECT_OLD` (1448) | Prefill block scores through the warp kernel instead of tensor cores | As `STRATA_PROMPT_ATTN_OLD`, for `strata_pf_sel_scores_mma` |
| `STRATA_ROPE_TABLE` (`native_rope.cu:97`) | RoPE from the host table instead of fast-math angles | Ignored: every Mac path reads the table ([03](03-model-reference.md), section 6.7) |
| `STRATA_MTP_BATCH` (740), `STRATA_MTP_BATCH_F16` (772) | `MTP_BATCH=0` turns the batched E-9 drafter prefill off; `_F16=1` makes it F16 GEMMs | Ignored. The Mac always runs the drafter prefill as `mtp.prefill`, upstream's `MTP_BATCH=0` arm (section 5.11). |
| `STRATA_PLE_BATCH` (1026) | 0 keeps the per-token PLE block in prefill | Ignored: the Mac runs only the batched block, upstream's default (section 6.7). Upstream's per-token arm takes F32 activations for the value, so its bits differ; CUDA references are taken with the variable unset. |
| `STRATA_DBG_NAN` (`prefill.cpp:1814-1836`, 1915-1931; `verify.cpp:1183-1197`) | Non-finite counts: the first MoE layer of a chunk with any, the state the prompt leaves, and the first window row with non-finite logits | **Honoured**, all three lines verbatim. The prompt path's two are sections 6.1 and 6.6; the window's is [06](06-engine-host-port.md), section 4.2. |
| `STRATA_INDEXER_PER_TOKEN` (1427) | The old per-token indexer append | Ignored: `strata_idx_append_*` (section 6.5) |
| `STRATA_GR_UNFUSED` (125) | Keeps an FP32 copy of the normalised rows in the prompt GR (the F-1 A/B arm) | Ignored: one prompt GR form (section 6.3) |
| `STRATA_PREFILL_MMQ` (476) | 0 turns off the integer (Q8_1 × quantised) expert GEMMs of the prompt path, upstream's default | Ignored: the Mac's expert GEMM is the FP16 arm, which upstream runs under `=0` (section 6.6). CUDA references of the prompt path are taken with `STRATA_PREFILL_MMQ=0` ([09](09-testing-and-validation.md)). |
| `STRATA_GDN_CONV_SERIAL` (`prefill/kernels.cu:733`), `STRATA_GDN_REC_HEADS` (746), `STRATA_GDN_PIPELINE` (750) | Prompt GDN A/B arms: the old convolution walk, one block per head, the pipelined recurrence | Ignored: one prompt GDN form (section 6.4) |
| `STRATA_GROUP_COPY` (569), `STRATA_KV_STAGE_OWN` (375), `STRATA_PREFILL_ISSUER` (1207), `STRATA_PREFILL_RING` (92), `STRATA_PREFILL_STREAM_MIN` (82), `STRATA_STAGER_RING` (188, 562), `STRATA_STAGER_THREADS` (554) | Pinned host buffers, KV staging, the expert-copy issuer thread and the ring and stager of streamed experts | Ignored: unified memory has no staging; the tiered prompt path uses the slot pool ([05](05-memory-and-residency.md), sections 5.4 and 5.5) |
| `STRATA_QSA_DUMP`, `STRATA_QSA_DUMP_LAST` (1527-1529) | Per QSA layer, the selected cells of the last 4,096 prompt positions, for `tools/qsa_locality.py` | Ignored in Phase 1. Honouring it needs a read of the selection after every QSA layer of a chunk, a per-layer wait (section 6.5; work item L23). |
| `STRATA_SEL_OVERLAP` (1459), `STRATA_IDX_FP16_CHECK` (1481) | Debug statistics of the prompt selection | Ignored |
| `STRATA_GR_V3` (`fused_gr.cu:824`), `STRATA_HC_SPLIT` (791, 1042) | The v3 GR kernels, the stream split of the GR read | Ignored (section 2.5) |
| `STRATA_OLD_GROUPED` (`s2_expert_grouped.cu:495`), `STRATA_GROUPED_PAIR_MIN_HITS` (508) | The old grouped expert kernel, the pairing threshold | Ignored: one grouped expert form (section 5.10) |
| `STRATA_PROMPT_ATTN_V1` (`qsa_prompt_attn.cu:1024`), `STRATA_QSA_WARP` (994; `qsa_select.cu:707`) | Tensor-core prompt attention version, warps per block | Ignored. A Phase-2 kernel that has such a choice reuses the name (section 2.6). |
| `STRATA_SCORES_MULTI` (`qsa_select.cu:657`), `STRATA_SELECT_WMMA` (681), `STRATA_TOPK_OLD` (752) | Selection A/B arms | Ignored: one selection form (section 5.6) |
| `STRATA_OLD_SAMPLER`, `STRATA_SAMPLER_ONE_BLOCK` (`sampler.cu:763-764`) | The old sampler, the one-block sampler | Ignored: one sampler form (section 5.12) |
| `STRATA_OLD_IQ_MMVQ` (`iq_kernels.cu:1295`) | The old IQ mat-vec kernels | Ignored (sections 5.8 and 5.10) |
| `STRATA_HIP_ROUTER_OLD` (`router_top10.cu:349`), `STRATA_HIP_WMMA` (`prefill.cpp:97`), `HIPBLASLT_TUNING`, `HIPBLASLT_VERBOSE`, `HIPBLASLT_TUNING_V1` | HIP only | Ignored |

The Mac's own switches are in [06](06-engine-host-port.md), section 4.4: `STRATA_METAL_GEMM` (section 6.2),
`STRATA_METAL_DISPATCH` (section 8.2), `STRATA_METAL_ICB` (section 8.3) and `STRATA_METAL_SMALL_TILES` (section 7.1).

### 9.4 Dumps with the fused read

Upstream turns the fused GR read off while it dumps, because a layer's FFN write is still pending in R when the layer
ends (section 5.1, `pending(l)`). The Mac keeps the fused read and makes the dump instead:
- the dump arena is layer-major, `float [49][T][10240]`, so each slot holds T rows at R's own stride. The host
  writes the file in upstream's position-major order after the window's wait;
- slot l + 1 of `--dump-layers` is written by `strata_util_copy_strided` (rows T, cols 10,240, both strides 10,240)
  from R at the end of layer l;
- when layer l's FFN write is pending, `strata_gr_post` then runs on the dump slot with layer l's `block_out` and
  `inject2`, so the slot holds R after layer l. The fused fold and `strata_gr_post` give the same bits on the Mac
  (section 5.4), so the slot equals the R that layer l + 1 reads;
- `--dump-halves` reads only buffers that the window writes anyway (`block_out`, `inject`, `mixed` and the rest of
  [03](03-model-reference.md), section 9.2), with the same copy kernel.

The dump arena is allocated in window scratch only when a dump flag is set. Neither step writes R, so a run with a dump flag gives the same tokens as one
without, which [03](03-model-reference.md), section 9, requires. R11 checks it.

### 9.5 The fast folder

`fast/` holds `strata_selftest_exp` only. A kernel moves there only by the rule of section 3.2, and each move adds a
row here:

| Kernel | Date | A/B (M5; M1-M4 GPU) | Tolerance in `safe/` and `fast/` | Gate C1 |
| --- | --- | --- | --- | --- |
| (none) | | | | |

L-11 is the A/B that would justify the first move.

---

## 10. Tests

This section covers the kernel-level tests and the properties that tie kernels into windows. The model-level gates
(C1-C4, G1-G3) and the release acceptance runs are in [09](09-testing-and-validation.md). Every test here is a
`strata_metal_test` with one label ([08](08-build-and-distribution.md), sections 6.2 and 6.5).

### 10.1 The twin pattern

**Files.** `tests/metal/<area>_test.cpp` builds `metal_<area>_test`. Its oracles are in
`tests/metal/oracles/<area>.cpp`, part of `strata_metal_oracles`. The harness `tests/metal/twin.{hpp,cpp}` opens a
`Context` on the `STRATA_METALLIB` it is given, allocates buffers, encodes one kernel the way the engine encodes it,
waits, and compares. A twin uses the engine's own parameter-block structures from `common/types.h` and the manifest's
threadgroup shape, so a test cannot drift from the engine's encoding.

**One twin, step by step:**
1. Build the inputs on the CPU from `std::mt19937_64`, seeded by `--seed N` (default 1). A failure prints the seed,
   the input set, T, the first failing index and both values in hex. Run the oracle on those same input bytes.
2. Before dispatch, initialize each declared output element to a value that its comparison class is guaranteed to
   reject against the oracle: flip one bit for B, use a quiet NaN for finite U(n)/R(ε) references, and use finite
   zero for non-finite references. An in-place output keeps its required input value, which must differ from the
   oracle result. Initialize guards and scratch poison with `0xFF`. Every guarded buffer has 4 KiB on each side.
3. Encode the kernel with the engine's grid mode, grid dimensions, manifest threadgroup shape and parameter block,
   in one serial encoder of one command buffer, and wait. The harness supports both exact thread grids and
   threadgroup-count grids in three dimensions.
4. Compare the declared output range in the kernel's tolerance class (section 10.2), and check that every other byte
   of every output and scratch buffer still matches its pre-dispatch contents. A stray write fails the twin even
   when the values match.
5. Repeat for T = 1..8 where the kernel has a T, and for every specialisation (format, NC, mode). Every T must also
   give row t the same bits as T = 1 does, the kernel-level form of decode ≡ verify (`row_t_independent`).

No output bit pattern is reserved. The comparison-class initializer detects an unwritten result without restricting
valid integer encodings or NaN payloads: it chooses a distinct initial value for every expected element, including
non-finite references.

**Input sets.** Each twin runs all four that apply:

| Set | Content | Label |
| --- | --- | --- |
| `random` | The real shapes ([03](03-model-reference.md), section 2). Activations N(0, 1) scaled to the RMS the model has at that point (from R12's dumps; 1 where unknown). Quantised weights are random valid blocks: F16 scales drawn log-uniformly from [2^-12, 2^-4], codes uniform over the format's range, IQ grid indices uniform. BF16 weights N(0, 0.02²). | `metal` |
| `edge` | All zeros; one row whose amax is 2^10 times the rest; quantiser inputs placed exactly on a rounding tie (x / d = k + 0.5); −0; ±65,504 on F16 paths; the largest and smallest positions, cells and expert ids the kernel accepts; NaN and ±∞ where the kernel's section specifies their handling | `metal` |
| `fixture` | Upstream's vectors and fixtures, unchanged: `ple_oracle_vectors.inc` and the IQ fixtures of `tools/iq_fixture.py` | `metal` |
| `real` | The first and last 64 rows of every tensor of each format in the Q2_0 and Coder packs, with activations from a CUDA `--dump-halves` run (R12) | `model` |

The sampler twin's `fixture` input combines deterministic synthetic coverage with semantic ports of upstream
`sampler_parity` cases 16 and 17. Their background logits come from a portable integer-only generator with
known-answer checks, not `std::normal_distribution`; the structural tie, sentinel, penalty, parameter and draw
counter cases are preserved, but byte-identical upstream random vectors are not claimed. Fixture 16 covers
vocabularies 248,320, 100,003, 262,144 and 1,000. Its 262,145-entry one-block fallback is outside Metal's declared
vocabulary limit; the twin checks that the shape is refused. Fixture 17 covers vocabularies 248,320 and 512, all 17
source rows, penalties on and off, the top-k/top-p/min-p matrix and both upstream draw-counter variants. Its two
variants share the Metal test queue, so they do not validate distinct CUDA stream behavior. Host fixture 15 separately
ports `penalty_rows` across 420 boundary cases. Literal upstream fixtures 1-14 and 18, byte-for-byte parity against
upstream-generated vectors, and production sampler integration remain open under REV-028.

**Inputs without near-ties.** Where a kernel's output is a choice (router ids, selection ids, sampler candidates),
`random` inputs are redrawn until the choice is separated: the k-th and (k+1)-th values differ by more than 2^-16
times `max(1, abs(k-th), abs((k+1)-th))`. Choices are then compared exactly. The `edge` set adds exact ties,
where the kernel's tie rule (the larger value, then the lower index, section 3.4, unless its section says otherwise)
is compared exactly.

**Validation.** Every `metal` twin also passes with `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`
([08](08-build-and-distribution.md), section 13), which catches out-of-bounds reads that land inside a buffer the
poison check cannot see. The nightly hardware job runs the label once that way ([09](09-testing-and-validation.md)).

### 10.2 Tolerance classes

Every twin compares in exactly one class per output. `ŷ` is the oracle's value, `y` the kernel's.

| Class | Rule |
| --- | --- |
| **B** | Every element equal as bits, including the sign of zero. Integers and ids equal. |
| **U(n)** | For every element, \|y − ŷ\| ≤ n · 2^-24 · m + 2^-126, where m is the element's *magnitude*. NaN and ±∞ appear at the same places in y and ŷ. |
| **R(ε)** | For every output row (the last dimension), ‖y − ŷ‖₂ ≤ ε · ‖ŷ‖₂, computed in double, with ŷ from an FP64 reference of the mathematical definition on the same rounded inputs. Also no element is non-finite unless ŷ's is. |

**The magnitude m.** The oracles are written over a small type, `twin::tf`, that carries a float value and a double
magnitude. Every operation computes the value exactly as the kernel does (float, `std::fma` where the kernel calls
`fma`, `-ffp-contract=off`) and the magnitude by the same operation on absolute values with every subtraction
turned into an addition. For a library function f the magnitude is |f(value)|. m is then the classic running error
scale: a sum that cancels to a small result keeps the magnitude of its terms, so a few ulps of error in an `exp`
inside the sum are measured against what they were added to, not against the small result. For class B the
magnitude is ignored.

**Where n comes from.** For the transcendental helpers, `exp` (≤ 1 ulp) and `log` can differ between the GPU and the
oracle (section 3.2; the oracle rounds each library call computed in double to float). n is 16 per helper on the
element's path
(`exp`, `sigmoid_k`, `silu_k`, `softplus_k`, `log1p_k`), counting helpers inside a loop once per loop. That covers
A.10's measured bounds (≤ 5 ulp for a helper) with the rounding of what follows. Weighted RMS norm uses U(64) for
its fixed reduction: up to ten strided FMA steps, two five-stage butterflies, mean and epsilon, `rsqrt`, and two
output multiplies. Conservative error propagation for the tested finite input range is about 42 units, rounded up
to 64. The reference M5 Pro was bitwise equal on the current RMS norm cases and on 4,194,304 standalone safe-mode
`rsqrt` measurements, but exact host/GPU agreement is not assumed across generations. Each twin prints the largest
n it observed; a twin whose observed n exceeds half its class's n fails in the `real` set, so a kernel close to its
bound is looked at rather than passed. This RMS norm bound does not claim coverage of extreme or subnormal inputs
outside the documented test range.

For R(ε), the comparison uses the oracle's FP64 mathematical reference directly; it must not first round that
reference to F32.

### 10.3 The prefill tolerance

The prompt path has no bitwise contract with decode ([03](03-model-reference.md), section 6): its GEMMs sum in the
tile's order and take F16 or BF16 activations.

| Kernels | Class | Reference |
| --- | --- | --- |
| `strata_pf_dequant_<fmt>`, `strata_util_to_f16`, `strata_util_to_bf16` | B | Upstream's `group32` expression per value, whose products are exact in F32, rounded once to F16 (RNE); the BF16 helper of section 3.6 |
| `strata_pf_gemm_f16`, `strata_pf_gemm_bf16` (`mpp` and `simd`) | U(2K) and R(2^-16) | FP64 sum of the exact products of the F16 or BF16 images. A half or bfloat product is exact in F32, so K additions in F32 in any order stay within 2K·2^-24·Σ\|a·b\| (U(2K)); R(2^-16) catches an accumulator narrower than F32. |
| BF16X2 modes 1 and 2 | As above, with K doubled | FP64 of `W · (hi + lo)` |
| `strata_pf_moe_gu_<fmt>`, `strata_pf_moe_down_<fmt>` | As the GEMMs; the gate/up epilogue adds U(16) for `silu_k` | The tile's in-place decode must give the bits of upstream's `dq_<fmt>` (`iq_kernels.cu:1002-1140`) rounded to F16, checked B through a debug output of the `simd` form; then FP64 of the exact products |
| `strata_pf_sort_*`, `strata_pf_gather_f16` | B | The stable counting sort of section 6.6: entry order within an expert, experts in id order, and the `PfPlan` fields |
| `strata_pf_gr_*`, `strata_pf_gdn_*`, `strata_pf_gate_attn`, `strata_idx_append_*`, `strata_pf_swiglu_pair`, `strata_pf_combine`, `strata_embed_image` | B or U(n), as decode kernels | Their order is fixed by section 6, so they are twins like the window kernels |
| Prefill attention and block scores (Phase 1: the decode kernels in query batches) | The decode kernels' classes | Each query of a batch must equal the same query run alone (B) |
| Phase-2 `strata_pf_prompt_attn_*`, `strata_pf_sel_scores_mma` | R(ε), ε set when they are specified | The decode kernels' outputs on the same inputs |

If `mpp` fails U(2K) or R(2^-16) on a GPU where `simd` passes, L-6 records it and `STRATA_METAL_GEMM` defaults to
`simd` on that family. The end-to-end prompt-path check (prompt through `Prefill` against the same prompt through
windows) is a model-level criterion of [09](09-testing-and-validation.md).

### 10.4 Properties

Properties compare two Mac runs, always in class B. "Synthetic" means random weights in the real shapes, built by
`tests/metal/synth.cpp` for the layers a property needs; "pack" means the Q2_0 pack (and the Coder pack where noted).

| Property | Label | What runs | Must hold |
| --- | --- | --- | --- |
| `row_t_independent` | metal | Every twin with a T (section 10.1, step 5) | Row t of a T-row dispatch equals the row at T = 1 |
| `mv_multi_eq_single` | metal | Every `strata_mv_<fmt>_n<NC>`, `strata_mv_bf16_b<BS>_n<NC>`, `strata_moe_gu_<fmt>` and `strata_moe_down_<fmt>`, NC = 2, 4, 8 | Each column equals the NC = 1 result. A negative control flips one weight bit in a copy and must fail, as upstream's `iq_multi_parity` does. |
| `ple_batch_eq_sequential` | metal | The PLE kernels over n = 1..64 rows against n single-row runs (section 5.3) | Every output and the history equal |
| `idx_batch_eq_sequential` | metal | `strata_idx_append_first`, `_blocks` and `_tail` against the window form of `strata_idx_append` in windows of up to 8 rows, for p0 ∈ {0, 1, 3, 4, 6} and T ∈ {1, 2, 3, 4, 5, 9, 300} (section 6.5) | `tail`, `dead`, `pooled` and `block_pos` equal |
| `pf_gdn_eq_step_norm` | metal | The prefill GDN kernels over T = 1, 3, 64, 65 and 300 rows against `strata_gdn_conv_l2`, `strata_gdn_step_norm` mode 2 and `strata_gdn_conv_commit` in consecutive windows of up to 8 rows, with the same `gate` and `beta` (section 6.4) | The state, the conv history, and `y_h` against `half(y)` |
| `pf_sort_eq_host_grouping` | metal | `strata_pf_sort_*` against upstream's host grouping loop (`prefill.cpp:1583-1615`, transcribed in the oracle) for T = 1, 7, 64, 300 and 4,096, with random ids, all rows on one expert, and every expert used (section 6.6) | `src`, `dst` (the inverse of `slot`), the `PfPlan` groups and the tile list |
| `samp_coupled_eq_host` | metal | Coupled penalty/selection GPU chain against the independent host arithmetic mirror (section 5.12); the CUDA-only `coupled_draft_sample` function is not called on macOS | Candidate ids/logit bits and tail tokens/probabilities equal |
| `window_eq_decode` | model | 32 positions of each benchmark prompt as windows of T = 1..8, against the same tokens one at a time | Logits of every row, every state buffer after commit |
| `state_after_commit_eq_decode` | model | A window of T rows with commit(n_keep), T = 2..8 and n_keep = 1..T, against n_keep single-token windows (section 5.5) | GDN state and conv history, indexer, PLE history, KV cells below `pos0 + n_keep`; cells at and above it untouched |
| `write_through_eq_commit` | model | A write-through window against a window plus commit(T), T = 1..8 (section 8.5) | Every state buffer and the logits |
| `cvec_off_eq_stock` | model | A control vector loaded with scale 0, and one whose range covers no layer, against no control vector (section 5.4) | Logits for 64 tokens |
| `mtp_kv_t_independent` | model | The drafter prefill of 256 prompt rows in groups of 8 and in groups of 1 (section 5.11) | Every drafter K/V cell at or above `first_needed` |
| `tiered_eq_full` | model | 256 decode tokens with a 37 GiB budget (tiered) and unbudgeted (full), Q2_0 and Coder | Logits and tokens. This is gate C4's kernel side. |
| `dump_eq_nodump` | model | 64 tokens with each dump flag and without (section 9.4) | Tokens and logits (R11) |
| `metal_dispatch_modes_test` | model | One window of each T, a commit, an MTP round and a prefill chunk, `serial` against `concurrent` (section 8.2) | Every output and state buffer |
| `prefill_chunk_independent` | model | A 6,000-token prompt through `Prefill` with C = 1,024, 2,048 and 4,096, `simd`, then under `mpp` where L-6 enables it (section 6.1) | The state after the prompt: KV cells, indexer, GDN state and history, PLE history, and the last row of `R` |

The synthetic model is used where a property needs no real weights: it has one GDN layer and one QSA layer with the
real shapes and a 4,096-row head, so the `metal` label stays under a minute.

### 10.5 The test list

The `metal_ple_test` fixture executes the documented weighted-norm, gate,
broadcast, convolution-residual and history chain at T=1, 2 and 8. The
distinct-buffer reference chain uses `encode_ple_post_projection` and checks
each stage; downstream oracles consume the preceding GPU output. An
instrumented exact-alias chain snapshots query norm before overwrite and
compares all outputs bitwise to the reference. The coordinator is also run
successfully with the exact alias in a separate command buffer. Both history
policies are tested: `defer` preserves committed history during a speculative
Window, then a separate commit encoder advances only an accepted prefix;
invalid policies are rejected. A late short history and a partial
query/`ple_norm` overlap verify failure discard and alias rejection; outputs
and guards are checked. The fixture also composes `encode_ple_block` for
Q2_0, Q8_0 and BF16 key projections, including the F32-input BF16 value
projection, padded columns, per-stage scalar oracles and a late-failure check
after projection dispatches. `encode_window_ple_entry` composes the layer-0
GR post write in place with the PLE block at T=1 and T=2. Its query-norm
oracle consumes the GPU output from a separate GR-post probe, so the PLE
comparison follows the rounded value produced by the preceding stage. A short
PLE history rejects the later stage and discards the preceding GR dispatch;
the concurrent fixture also rejects a call made inside an open dispatch
group. Both serial and concurrent dispatch modes run these fixtures. The
embedding fixture composes text gather, optional
image copy and GR broadcast through caller-owned Encoder overloads. GR cvec
fixtures also run through a shared Encoder at T=1 and T=8 in project/add and
write/no-write modes, checking residual oracles and guards. GR norm fixtures
run through a shared Encoder segment at T=1 and T=8 with apply on and off,
checking normalized values and reciprocal scales against their scalar oracles.
The post fixture also runs through a shared Encoder segment and compares the
native write against its scalar oracle. The pre-gated fixture runs through a
shared Encoder at T=1 and T=8, checks its U(16) oracle and guards, and verifies
that overlapping read-only inputs remain accepted while writable aliases are
rejected. Down-SiLU also runs through a shared Encoder at T=1 and T=8, checking
the in-place U(16) oracle, tail handling and guards. GR up runs through a
shared Encoder at T=1 and T=8 with apply on and off, checking both outputs
against their scalar oracles and verifying the parameter binding.
GR down/injection runs through a shared Encoder at T=1 and T=8, checking U(16)
low-rank outputs, bit-exact injection outputs, guards and T-sized dynamic
threadgroup memory.
GDN convolution/L2 also runs through a shared Encoder at T=1 and T=8, matches
the U(128) oracle, preserves its read-only inputs and output guards, rejects
invalid and undersized views, misalignment, out-of-range views, writable
aliases and foreign Context views, and accepts overlapping read-only inputs.
The alpha/beta gate projections also run through a shared Encoder at T=1 and
T=8, match both scalar oracles, preserve input bytes and output guards, and
reject invalid rows, short or one-row T=2 views, misalignment, out-of-range
views, every writable/input and output/output alias, and foreign Context views.
The recurrent step norm runs through a shared Encoder at T=1 and T=8 in
window, commit and write-through modes, matches the output/state oracles,
preserves state in window mode and output in commit mode, verifies all four
threadgroup-memory regions, and accepts the zero-row commit no-op.
Convolution-history commit also runs through a shared Encoder at T=1 and T=8,
matches the exact three-tap history oracle, checks guards and read-only qkv,
accepts the zero-row no-op, and rejects invalid rows, short or one-row T=2
views, misalignment, out-of-range views, exact and partial history/qkv aliases,
and foreign Context ownership.

| Test | Label | Kernels | Classes |
| --- | --- | --- | --- |
| `metal_embed_test_concurrent` | metal | `strata_embed_rows_{q3_k,iq3_s,iq4_xs,bf16}`, `strata_embed_image`, `strata_gr_broadcast` | Runs the embedding test binary with concurrent dispatch; its shared Encoder gather → optional image copy → broadcast chain matches the bitwise stage oracles |
| `metal_prefill_test_concurrent` | metal | `strata_pf_gr_norm`, `strata_pf_gr_norm_lo`, `strata_pf_gr_write_norm`, `strata_pf_gr_write_norm_lo`, `strata_pf_gdn_gates`, `strata_pf_gdn_conv_tiled`, `strata_pf_gdn_conv_hist`, `strata_pf_gr_silu`, `strata_pf_gr_silu_lo`, `strata_pf_gr_mix`, `strata_pf_gr_mix_lo`, and the kernels listed for `metal_embed_test` | Runs Prefill GR normalization, fused GR write/next-read normalization, GDN gates, tiled convolution and history update, SiLU and mix with concurrent dispatch, plus embedding → broadcast → normalization composition; checks stage-chain bit parity, scalar gate and convolution oracles including tile-boundary and FMA-order controls, history oracles at T=1/2/3/8/65, old-history convolution→update ordering and rollback for both APIs, reduction and conversion oracles, BF16X2 ties, invalid-group rejection and discard after late validation failure |
| `metal_mv_test_concurrent` | metal | `strata_q_q8_1`, `strata_mv_<fmt>_n<NC>` (Q5_K, Q6_K and IQ4_XS read-only head formats), Q2_0 and BF16 mat-vec | Runs the mat-vec test binary with concurrent dispatch; the shared Encoder Q8_1 → Q2_0/BF16 projection checks retain bitwise oracle parity, and immutable Q5_K, Q6_K and IQ4_XS mat-vecs match the scalar oracles |
| `metal_math_test` | metal | The helpers of section 3.6 over dense grids (2^24 points on their ranges), `butterfly`, the conversions of section 3.5, `strata_selftest_*` | Helpers U(n), with n = 2 × A.10's bound in ulps where measured; `silu_k` uses U(8) by analogy with `sigmoid_k`. Butterfly and conversions B |
| `codebooks` | host | `common/codebooks.h` against upstream's copy and ggml-common.h (section 3.7) | Bytes equal |
| `metal_embed_test` | metal | `strata_embed_rows_{q3_k,iq3_s,iq4_xs,bf16}` (including the image-token no-write case), `strata_embed_image`, `strata_gr_broadcast`, `strata_selftest_copy_u32_one` | Text gathers run through `encode_embed_rows` for T=1/7 in all four formats against ggml CPU, preserving untouched image rows and exact decoded output bits. The encoder rejects zero/overflowing rows, empty or flag-colliding vocabularies, unsupported formats, undersized/misaligned/out-of-allocation table/token/output views, insufficient table rows, non-image ids outside the vocabulary, contained and partial output overlaps with each input, valid same-handle aliases, distinct no-copy handles over shared backing, forged CPU/GPU metadata, foreign Context resources and enqueued/committed command buffers; it accepts overlapping read-only table/token views and skips vocabulary lookup for image flags. A same-segment GPU token producer verifies that stale CPU bytes are not inspected and that an invalid GPU-produced ID writes a quiet NaN without reading past the table. Image-row copies run through `encode_embed_image`, preserve every payload bit and leave text rows unchanged for T=1..8 under row-zero and after-text flag patterns, including T=1 with and without an image. Encoder checks reject zero rows, shader-index overflow, short T=2 token/image/output views, misalignment, out-of-allocation views, exact and partial output overlaps with each input, valid same-handle aliases, distinct Metal handles over a shared backing range, forged CPU/GPU metadata, foreign Context resources and enqueued/committed command buffers; read-only token/image overlap is accepted. GR broadcast runs through `encode_gr_broadcast`, preserves every stream bit for T=1/3/8 on random, edge and fixture inputs, and rejects invalid or shader-overflowing row counts, undersized/misaligned/out-of-allocation views, valid same-handle and cross-handle input/output aliases, forged CPU/GPU metadata, foreign Context resources and enqueued/committed command buffers. The T=1 Window input-stem coordinator gathers the published token through a read-only WindowParams slice and read-only embedding table, broadcasts the exact BF16 row into four streams, rejects T=2 and out-of-vocabulary input, and discards the earlier gather when late broadcast validation fails. |
| `metal_ple_test` | metal | `strata_rms_norm_w`, `strata_ple_gate`, `strata_ple_broadcast`, `strata_ple_conv_residual`, `strata_ple_history`, `strata_selftest_copy_u32` | Gate runs through `encode_ple_gate` and is checked at U(64) against the documented ordered lane and 16-simdgroup butterfly oracle for T=1..8 on random, edge and fixture inputs; two reduction-order negative controls, first-row batch independence, read-only inputs and guards. The encoder rejects invalid T, short T=2 key/query/output views, misalignment, out-of-allocation ranges, exact and partial output/input overlaps, foreign Context resources and submitted command buffers, while permitting overlapping read-only key/query inputs; valid same-handle subviews are checked for overlap, and forged CPU/GPU metadata is separately rejected as unowned. Broadcast runs through `encode_ple_broadcast`, with bit-exact scalar parity for T=1..8, including signed-zero products, per-row guards and row independence; rejects invalid T, short T=2 views, misalignment, out-of-allocation views, exact and partial output/input overlaps, foreign Context resources and submitted command buffers, while accepting overlapping read-only inputs. Conv-residual runs through `encode_ple_conv_residual` and is checked at U(16) against sequential history advancement for T=1..8 with F16 weights; exact controls reject fused accumulation, reordered taps and residual-parenthesization changes. Encoder checks reject invalid T, one-row T=2 `ple_norm`/`gated`/`residual` views, short history and F16 weight views, misalignment, out-of-allocation ranges, exact and partial residual/input overlaps with each input, foreign Context/buffers and enqueued/committed command buffers; valid overlapping read-only views are covered and forged CPU/GPU metadata is separately rejected as unowned. In-place residual guards, input immutability and first-row batch independence are checked. History runs through `encode_ple_history` and bit-exactly matches sequential advances for n=0/1/2/8/9/10/17, including the history/norm boundary; rejects signed-index overflow, a short history, insufficient norm rows (including 16 rows for n=17), misalignment, out-of-allocation views, history/norm overlap, foreign Context resources and submitted command buffers; accepts an empty zero-row norm view, including a zero-length alias into history. Valid same-handle subviews are checked for overlap; forged CPU/GPU metadata is separately rejected as unowned. Input immutability and guards checked; the T=2 and T=8 pending-consumer PLE fixtures verify selector dispatch and late caller-owned rollback |
| `metal_ple_test_concurrent` | metal | `strata_rms_norm_w`, `strata_ple_gate`, `strata_ple_broadcast`, `strata_ple_conv_residual`, `strata_ple_history` | Runs the PLE test binary with concurrent dispatch; checks the shared Encoder projection and post-op dependency chain against the same oracles; the pending-consumer PLE branch runs at T=2 and T=8 and verifies late caller-owned rollback |
| `metal_gr_test_concurrent` | metal | `strata_gr_norm`, `strata_gr_down`, `strata_gr_up`, `strata_gr_cvec`, `strata_gr_post`, `strata_gr_down_silu`, `strata_gr_pre_gated` and the kernels listed for `metal_gr_test` | Runs the GR test binary with concurrent dispatch; shared Encoder normalization, down/injection, up projection/mixer, cvec, post-write, down-SiLU and pre-gated T=1/T=8 fixtures match their scalar oracles; the no-PLE pending-MoE selector matches its direct GR oracle for T=1..8 and rejects T=0/T=9 |
| `metal_gr_test` | metal | `strata_rms_norm_w`, `strata_gr_norm`, `strata_gr_down`, `strata_gr_up`, `strata_gr_cvec`, `strata_gr_post`, `strata_gr_down_silu`, `strata_gr_pre_gated` | Weighted RMS norm U(64); fused `strata_gr_norm` T=1..8 with apply on/off, exact no-apply output and U(64) apply output against the ordered per-thread float4/FMA, butterfly and serial-partial oracle; BF16 `strata_gr_down` T=1..8 with U(16) low-rank SiLU outputs and bit-exact injection outputs against an ordered lane/dot8 oracle; shared Encoder GR down/injection at T=1/T=8 matches both output oracles, checks guards and dynamic threadgroup-memory sizes, and rejects invalid T, short and one-row T=2 views, misalignment, out-of-range views, input/input, input/output and output/output aliases, and foreign Context views; `strata_gr_up` T=1..8 with apply on/off, U(64) mixed output, U(16) folded residual write, exact no-apply residual and fixture bit checks for `(R*g)*rs` and ascending stream accumulation with reordered-operation negative controls; shared Encoder GR up T=1/T=8 with apply on/off matches both output oracles and checks all eight short views, BF16/float misalignment, out-of-range views, input/output and input/input aliases and foreign Context ownership; `strata_gr_cvec` T=1..8 in project/add modes with write on/off, disabled flag, zero-scale and uncovered-layer paths against the ordered per-thread/FMA and two-stage butterfly oracle, including active layer-3 cases with distinct direction/scale inputs and a fixture FMA-versus-separate-multiply/add negative control; native post U(16), including rows=1/8 and exact residual/output alias; down SiLU U(16), in place for T=1..8 including partial-threadgroup tails, plus exact negative-zero sign checks for the scale FMA; shared Encoder down-SiLU at T=1/T=8 matches the in-place U(16) oracle and checks the partial-group tail, input/output guards and invalid T, short, misaligned, out-of-range and foreign Context views; pre-gated U(16) for T=1..8 against the ordered four-stream scalar oracle, with exact negative-zero sum/FMA checks and edge negative controls that reject fused product/add and reordered streams; shared Encoder pre-gated T=1/T=8 fixture outputs match the U(16) oracle and reject invalid T, short T=1/T=2 views, misalignment, out-of-range views, exact/partial writable aliases and foreign Context buffers while accepting overlapping read-only inputs; read-only inputs, output ranges and guards are checked; the generic pending-MoE no-PLE selector matches its direct apply=true GR oracle for T=1..8 and rejects T=0/T=9 before dispatch |
| `metal_gdn_test_concurrent` | metal | Same GDN kernels and Window block/commit coordinators as `metal_gdn_test` | Runs the fixture set with `STRATA_METAL_DISPATCH=concurrent`; T=1..8 also covers the generic router→top-10 route→Q8_1 prefix for 512- and 256-expert routers, checking BF16 logits, route IDs/weights, padded-column sink values and Q8_1 rows against scalar oracles, reciprocal routing and output alias rejection; T=3 rejects an undersized padded sink, and T=2 caller-owned encoding discards earlier attention work after a late GDN active-row failure. T=1..8 generic router→plan composition matches full/tiered MoePlan and usage scalar oracles; T=3 also rejects an undersized plan output before dispatch; T=2 caller-owned encoding then forces a later failure and checks rollback of route/plan outputs, usage and layer state. The generic speculative attention→GDN and attention→GDN→FFN-read coordinators run against component chains, check state/history preservation and reject malformed parameters and cross-stage aliases. At T=2, undersized GDN active-row and FFN up-weight views prove standalone and caller-owned rollback after earlier stages encode, preserving outputs, residual, injection and persistent state. T=1 also compares the dedicated attention GR-read→GDN→FFN GR-read coordinator with its component chain, verifies exact scratch and injection handoffs, distinct attention/FFN weights and the residual fold, and checks late FFN rollback; the router prefix checks standard 512-expert and Coder 256-expert BF16 logits against the bitwise scalar oracle, route IDs and U(16) weights against the route oracle, and caller-owned segment discard. T=2 exercises caller-owned block and accepted-prefix commit, T=3 covers padded sink writes, cross-stage aliases are rejected, late block/commit validation discards prior dispatches, and open concurrent groups are rejected |
| `metal_gdn_test` | metal | `strata_gdn_conv_l2`, `strata_gdn_ab`, `strata_gdn_step_norm` (modes 0, 1, 2), `strata_gdn_conv_commit` | T=1..8 over random, edge and fixture inputs; the generic router→top-10 route→Q8_1 prefix covers T=1..8 for 512- and 256-expert routers, checking BF16 logits, route IDs/weights, padded-column sink values and Q8_1 rows against scalar oracles, reciprocal routing and output alias rejection; T=3 rejects an undersized padded sink, and T=2 caller-owned encoding discards earlier attention work after a late GDN active-row failure; T=1..8 generic router→plan composition matches full/tiered MoePlan and usage scalar oracles, and T=3 rejects an undersized plan output before dispatch; the speculative attention→GDN and attention→GDN→FFN-read layer-0 coordinators cover T=1..8 in standalone and caller-owned forms, match the ordered component chains byte-for-byte, keep GDN state/history speculative, and reject malformed T/nc/n_keep/tokens/StepRows plus attention/GDN and attention/FFN aliases; T=2 late GDN and FFN failures verify standalone and caller-owned rollback after earlier stages have encoded, including output, residual, injection and persistent-state preservation. The separate T=1 attention GR-read→GDN→FFN GR-read layer-0 coordinator matches the ordered component chain byte-for-byte, verifies the residual fold and swapped inject buffers, requires shared mixed/GR scratch and GDN block-output views, rejects attention/FFN weight aliasing, and preserves speculative GDN state/history; a late FFN-up failure discards all prior dispatches through standalone and caller-owned forms. The separate attention→GDN fixture uses a nonzero GR block output to distinguish apply=false from apply=true, checks exact mixed handoff, rejects split handoff, T=2, invalid nc, nonzero n_keep, out-of-vocabulary tokens, inconsistent StepRows and cross-stage aliases, and discards late failures through both overloads; the speculative Window GDN chain runs T=1..8 with scalar-oracle output checks; its write-through variant runs T=1..8, matches speculative y/block outputs bitwise, checks mode-2 state and exact convolution-history oracles, and rejects malformed WindowParams T/n_keep/nc plus parameter overlap with writable state before encoding; an undersized final Q8_1 output proves both wrappers discard failures after state/history dispatch encoding and preserve every buffer; standalone and caller-owned forms are exercised, with shared Encoder at T=2/T=8 and open-group refusal; it rejects mixed/history cross-stage overlap; accepted-prefix commit covers T=1..8, checks exact history advancement and scalar state replay, rejects history/state aliasing, and verifies command-buffer and caller-owned rollback after late gamma failure; edge adds all-zero buffers and an exact 2^10 row-amax outlier. U(128) convolution outputs, U(256) gates, U(512) recurrent outputs/state, exact state equality between commit and write-through, sequential convolution-history advancement, input immutability and guards. Shared Encoder convolution/L2 at T=1/T=8 matches the same oracle and checks read-only overlap acceptance, invalid rows, short and one-row T=2 views, misalignment, out-of-range views, writable aliases and foreign Context ownership. Shared Encoder alpha/beta gates at T=1/T=8 match both scalar oracles and reject invalid rows, short and one-row T=2 views, misalignment, out-of-range views, all writable/input and output/output aliases, and foreign Context ownership; read-only overlap is accepted. Shared Encoder recurrent step norm at T=1/T=8 checks window/commit/write-through output and state oracles, preserves the mode-specific no-write regions, accepts zero-row commit, and validates its four threadgroup-memory regions and view/alias rejection rules. Shared Encoder convolution-history commit at T=1/T=8 matches the exact three-tap history oracle, checks guards and read-only qkv, accepts the zero-row no-op, and rejects invalid rows, short or one-row T=2 views, misalignment, out-of-range views, exact and partial history/qkv aliases, and foreign Context ownership. The standalone encoders reject invalid rows/modes, short or misaligned views, output/input aliases, foreign Context resources, forged CPU/GPU metadata, and foreign or submitted command buffers. |
| `metal_qsa_test` | metal | `strata_qsa_norm_rope` (`k`, `q`, `iq`), `strata_qsa_fwht256`, `strata_idx_snapshot`, `strata_idx_append` window/commit modes, QSA KV append encoders and `strata_qsa_gate` with scalar oracles | Norm/RoPE: T=1..8 random, edge and fixture inputs against ordered butterfly/FMA (U(64)), including row independence, outliers and nonidentity position-zero RoPE. Fixture T=1/T=8 cases for all three modes also use a shared Encoder and match the same oracle. Shared-path validation rejects invalid rows/modes, short and one-row T=2 views, misalignment, incomplete RoPE rows, invalid positions, out-of-allocation and foreign views, forged CPU/GPU metadata and unsafe aliases. FWHT: rows=1/2/7/8/9/15/16/17/191/192 against the bitwise scalar oracle, row independence, impulse goldens and a fixed dense digest. Indexer snapshot: all 12 QSA layers with packed/padded strides through standalone and shared Encoder forms, raw-bit equality, untouched padding, guards, invalid extents, aliases including shared backing, forged metadata and foreign/submitted resources; the shared path also verifies that a late validation failure discards earlier encoded work. Indexer append: window T=1..8 across cell 0, partial blocks and completion boundaries for random, edge and fixture data; half-rounded tails and block positions exact, dead/pooled rows U(64), nonidentity angle-zero RoPE, untouched rows and guards. Standalone and shared Encoder forms are checked against the same state oracle; shared fixtures cover window mode and snapshot→window append→accepted-prefix restore/replay in one ordered segment. Commit cases use keep counts 1/2/4/7; validation rejects invalid modes, dimensions, layers, noncontiguous/out-of-range steps, short views, missing snapshots, aliases and foreign/submitted resources, and a late failure must discard earlier shared work. F16, int8, q4_0 and k8v4 KV appends cover T=1..8, capacity tails, block boundaries, random/edge/fixture data, scalar byte/bit oracles, same-command-buffer FWHT where required, untouched cache rows and guards; shared Encoder fixtures also cover T=1 and T=8 for every mode, with Q4 K/V FWHT and K8V4 V-only FWHT ordered before append in the same segment. The gate covers T=1..8 on random, edge and fixture inputs against an accurate sigmoid oracle (U(8), including shared-Encoder T=1/T=8 fixtures); ±30 sigmoid inputs, batch-row independence, input immutability and output guards are checked. Shared and standalone validation cover invalid rows, short/misaligned views, output/input aliases, read-only overlap, foreign Context views and forged metadata; a late failure discards earlier encoded gate work. Rejection tests cover invalid rows/capacity/positions, short/misaligned/out-of-allocation views, output/input and output/output aliases (including distinct no-copy handles), forged metadata, foreign Context resources and submitted command buffers. GPU-produced inputs must satisfy each encoder's documented finite/representable-scale preconditions. CUDA-captured goldens and model-backed `real` inputs remain unavailable under REV-002/025; indexer append, FWHT and KV append use specification-derived oracles. Standalone selection scoring, top-k, the attention gate and sparse-attention chunk/merge are implemented and tested under L6; production Window/Prefill integration and external parity remain open under REV-002/025/032. |
| `metal_qsa_test_concurrent` | metal | Same binary and fixtures as `metal_qsa_test` | Runs the full QSA suite with `STRATA_METAL_DISPATCH=concurrent`, including shared attention-gate T=1/T=8 fixtures, validation and late-failure discard; results are checked after command-buffer completion |
| `metal_qsa_indexer_commit_batch_test` and `metal_qsa_indexer_commit_batch_test_concurrent` | metal | `strata_idx_snapshot`, `strata_idx_append` window/commit modes | Runs T=8 from position 1 with `n_keep=3` across all 12 layers through snapshot, speculative append and accepted-prefix replay in one Encoder segment. The accepted row at position 3 completes block 0; rejected position 7 writes pooled row 1 at the accepted `n_bid` and spare row 2 beyond it, while replay restores the visible prefix. Each layer has distinct raw keys/gamma and an independent scalar oracle; exact tails/block positions, pooled/spare rows and read-only inputs are checked. Rejects a short layer list, distinct valid StepRow/RoPE/snapshot views, cross-layer output/input alias and malformed final-layer input. It permits outputs to reuse inactive StepRow and RoPE suffixes, and caller-owned failure discards an earlier encoded replay. The two CTests exercise serial and concurrent dispatch modes |
| `metal_qsa_selection_test` | metal | `strata_sel_scores`, `strata_sel_topk_reg`, `strata_sel_topk_mem` through their validated encoders and independent scalar oracles | Scoring: T=1..8 random, edge and fixture queries; exact ordered FMA/butterfly score bits; per-head ReLU fixture; complete and partial tail blocks; untouched score columns; one row at 131,072 cells exercises the 256-group stride. Shared Encoder scoring fixtures cover T=1/T=8 and match the same oracle; shared validation rejects invalid rows/capacity, short views, inconsistent StepRows, output/input aliases, foreign Context buffers and forged CPU/GPU metadata; a late failure verifies an encoded score dispatch is discarded. Top-k: standalone and shared-Encoder forms; direct and radix paths; register form through `max_blocks` 33,792 and memory form above it; tied threshold blocks; NaN key 0 ranked below −∞ at the selection boundary; signed-zero tie at the selection boundary prefers lower IDs; highest-scoring tail cells of widths 1..3 selected and zero-width tails ignored; a high-id fixture verifies every selected id exceeds 65,535; ±infinity and finite scores; exact register/memory equivalence; shared score→top-k in one segment. Shared validation checks invalid rows/capacities, short views, inconsistent StepRows, id/input aliases, foreign Context buffers and forged CPU/GPU metadata; a late failure verifies an encoded top-k dispatch is discarded. Both CTests run the same fixtures in serial and concurrent modes. Inputs, output suffixes and guards are checked; invalid rows/capacities/StepRows, short or misaligned views, output/input aliases, foreign Context resources/command buffers and submitted command buffers are rejected. |
| `metal_qsa_selection_test_concurrent` | metal | Same binary and fixtures as `metal_qsa_selection_test` | Runs all fixtures with `STRATA_METAL_DISPATCH=concurrent`, including shared score dispatch and validation, and checks outputs against their oracles after command-buffer completion |
| `metal_attn_test` | metal | `strata_attn_chunk_<mode>`, `strata_attn_merge`, fp16/int8/q4_0/k8v4 KV modes | All four modes: standalone random T=1..8 and edge/fixture T=8 against an ordered scalar oracle at U(32); shared-Encoder chunk→merge fixture cases at T=1 and T=8 (one and two chunks) against the same oracle; fixture row independence for T=8 versus isolated rows; ID strides 1/63/64/2051 across one- and many-chunk cases; negative, capacity-boundary and future-cell IDs are masked, including exact-zero empty chunks; shared and standalone validation cover inconsistent StepRows, short views, aliases, foreign Context and forged metadata; a late merge failure discards the earlier chunk dispatch and leaves output untouched; input immutability and guards; shared-Encoder selection scores→top-k→attention T=1/T=8 cases pass GPU-produced IDs directly to attention for all four KV modes, then apply O-class FWHT for q4_0/k8v4 and the gate; a varied-prefix T=8 fixture also covers live widths 1/2/4/5/64/65/2048/2051 and verifies untouched ID suffixes; a separate eight-cell cache exercises live widths 1..8 with the fixed 2051-ID stride; scalar score/ID/attention/gate oracles and guards are checked |
| `metal_attn_test_concurrent` | metal | Same binary and fixtures as `metal_attn_test` | Runs shared chunk→merge composition, validation and late-failure discard with `STRATA_METAL_DISPATCH=concurrent`; the dependent passes remain ordered by the Encoder barrier; the selection→attention coordinator also rejects calls inside an open concurrent dispatch group |
| `metal_mv_test` | metal | `strata_q_q8_1`, every `strata_mv_<fmt>_n<NC>` and `_n1_x`, `strata_mv_bf16_b<BS>_n<NC>` | B. The weights' dequantisation is also checked against ggml's. Engine encoders quantize Q8_1 against the scalar oracle and check nonzero engine results for every dense quantized format and BF16 against their scalar mat-vec oracles. Engine routing covers NC=1/2/4/8, large-K Q4_0 NC=2 (R=1), BF16 NC=8 (256 threads), padded output and sink handling, and a same-command-buffer Q8_1→Q2_0 chain. The indexed draft-head encoder checks nonzero Q5_K, Q6_K and IQ4_XS results at n_in=2560 against their scalar oracles with an unsorted, non-contiguous row map; it rejects negative and out-of-range indices, invalid row counts, signed and shader-index overflow, unsupported formats, a wrong live-row value, short or misaligned maps, and output/sink overlap with the map. Guards and read-only input preservation are checked. Dense encoder rejection cases cover forged CPU/GPU-view metadata, mismatched row count, invalid NC, Q8_1 input/output overlap, and mat-vec output/sink overlap with activations, weights, active-row metadata, and each other. Both same-handle subviews and distinct no-copy handles sharing a CPU backing are covered; an adjacent Q8_1 view is accepted. A read-only Q2_0-by-Q8_1 path checks overlapping read-only views, disjoint writable output/sink ranges and a 65-block input row against its scalar oracle. The format-general read-only path also checks Q5_K, Q6_K and IQ4_XS against scalar results; Q6_K checks caller-owned failure rollback after an earlier dispatch. MTP stage 21 composes this indexed read path with draft-head projection; the L-15 indexed-read measurement and production MTP runtime-view binding remain open. |
| `metal_moe_test`, `metal_moe_test_concurrent` | metal | `strata_moe_route` (`r16`, `r8`), exact weight-mode probe, `strata_moe_plan` (T=1..8, full/tiered), `strata_moe_swiglu` and `strata_shx_swiglu` (T=1..8), `strata_moe_combine` (T=1..8), grouped `strata_moe_gu_q2_0`, `strata_moe_gu_iq1_m`, `strata_moe_gu_iq2_xxs`, `strata_moe_gu_iq2_xs`, `strata_moe_gu_iq2_s`, `strata_moe_gu_iq3_xxs`, `strata_moe_gu_iq3_s`, `strata_moe_gu_iq4_xs`, `strata_moe_down_q2_0` and `strata_moe_down_iq4_nl` (256/512 experts, T=1..8) across random, edge and fixture inputs; a synthetic shared-expert GPU chain feeds Q8_1 input quantization through Q8_0 gate/up, shared SwiGLU, Q8_1 hidden requantization, Q8_0 down, BF16 gate and combine for T=1..8, checking each intermediate against scalar references and checking partial-batch sinks and guards; both SwiGLU entry points are checked against the scalar oracle, including the 128-value tail in the shared-expert grid; combine pins fused-rank accumulation with an analytic golden and checks the final shared multiply/add bitwise against a same-dispatch sigmoid probe; IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS check 80 unique gate/up expert groups; both down kernels check 80 permuted expert IDs and `ent_dst` restoration; IQ4_NL down also checks exact two-part FMA-sensitive answers, grouped-versus-single-entry bit parity and a weight-bit negative control; remaining L7: production shared-expert dispatch and inference integration | Exact route IDs and plan bytes, independent tie/order fixtures, route weights, routed/shared SwiGLU U(16), combine U(16) plus contraction checks, grouped gate/up and down U(16), fixed IQ4_NL two-part codebook goldens, 80-group endpoints, singleton parity, negative-control sensitivity and `ent_dst` permutation, plus address-table/guard checks; exact multiply/divide mode bits. Explicit `--input-set real` is reported as skipped (CTest exit 77) until REV-002's model pack and CUDA dump fixture are available; route oracle cases use the validated route encoder, every grouped gate/up format runs through the validated engine encoder, and validation rejects bad shapes/formats, short or misaligned views, input/output aliases and caller-owned late-failure rollback; the concurrent CTest also checks open-group refusal |
| `metal_mtp_test` | metal | `strata_embed_rows_bf16`, `strata_rms_norm_w`, `strata_q_q8_1`, `strata_mv_q8_0_n<NC>`, MTP add-streams, indexed output-head mat-vec kernels, `strata_samp_greedy`, `strata_mtp_row_top_prob`, `strata_mtp_map_ids` | MTP stages 1–5 are composed and compared with independent BF16 gather, weighted RMS, Q8_1 and Q8_0 mat-vec oracles for K/V-only T=1,2,3,8 and full-layer selected-token T=1; checks NC=1/2/4/8, hidden-pass tails, exact in-place embedding norm, immutable inputs, active extents, guards, invalid hid/OOV/image-marker tokens, selected-token/parameter overlap, other aliases, caller-owned rollback and concurrent-group refusal. Stage 6 T=1..8 matches the independent scalar addition oracle bit for bit. Stage 21 checks full-vocabulary and mapped draft-head logits for Q5_K, Q6_K and IQ4_XS against independent Q8_1/mat-vec oracles, including one-time range/uniqueness map validation, output guards, immutable inputs, aliases and caller-owned rollback. Stage 22 checks greedy full-vocabulary and strict-subset token IDs and probabilities against independent softmax expectations, excluding logits outside the mapped rows; also checks malformed sampler/history, aliases, guards, immutable inputs, caller-owned rollback and open-group refusal in concurrent mode. Runtime bindings through stage 22 and ordered synthetic smoke coverage for stages 1–20 are covered separately by `metal_mtp_storage_test`; scheduler integration and full drafter parity remain open |
| `metal_mtp_routed_experts_test` | metal | `strata_q_q8_0_scaled`, `strata_mtp_gu_s2`, `strata_mtp_swiglu_s2`, `strata_mtp_down_s2` | MTP stage 17 compares one full expert image with three ordered GPU-buffer windows covering experts 0–511, including route IDs at every boundary; validates complete nonoverlapping coverage, context ownership, scratch aliases, output guards and late caller-owned rollback. Serial and concurrent outputs match byte for byte. |
| `metal_mtp_storage_test` | metal | Runtime-bound MTP stages 1–20 and the stage-21/22 bind views | Checks planned weights, named scratch regions, shared parameters, draft-vocabulary identity, routing windows, MLP versus attention hyperconnection weights and exact inter-stage view handoffs. Dispatches the synthetic ordered chain through stage 20. Stage 21/22 bindings reuse the caller-owned main head and the same once-validated vocabulary token. These bindings and synthetic dispatch checks do not establish model parity, production scheduling or end-to-end inference. |
| `metal_mtp_gr_test` | metal | `strata_gr_norm`, `strata_gr_down`, `strata_gr_up` | MTP stage 7 composes the existing ordered GR read with `apply=false` for T=1..8 and matches the direct GR chain bit for bit; verifies finite populated outputs, residual and params preservation, active-row output bounds, malformed T, dead-h2 scratch size/alignment, foreign params, parameter/input and parameter/output alias rejection, and standalone/caller-owned rollback after late failures. `metal_mtp_storage_test` separately verifies the runtime mappings and ordered chain; these synthetic tests do not establish inspected-model drafter parity |
| `metal_mtp_gr_test_concurrent` | metal | Same GR kernels as `metal_mtp_gr_test` | Repeats the MTP stage-7 fixture with concurrent dispatch enabled and verifies that the ordered GR chain rejects an open concurrent group without changing any buffers |
| `metal_mtp_kv_projection_test` | metal | `strata_q_q8_1`, `strata_mv_q8_0_n<NC>` | MTP stage 8 covers T=1..8, compares the quantized rows and both 2560→512 Q8_0 projections with independent scalar oracles, checks padded sink values, active output bounds and guards, immutable inputs, malformed T, short weights/sink, foreign weights, parameter/output aliases and overlapping K/V outputs, and verifies caller-owned rollback after a later invalid stage-8 call. `metal_mtp_storage_test` separately verifies runtime mappings and ordered integration; inspected-model drafter parity remains open |
| `metal_mtp_kv_projection_test_concurrent` | metal | Same Q8_1 and Q8_0 matvec kernels as `metal_mtp_kv_projection_test` | Repeats the stage-8 fixture under concurrent dispatch and verifies rejection inside an open group without changing any buffers |
| `metal_mtp_kv_append_test` | metal | `strata_qsa_norm_rope`, `strata_qsa_fwht256`, `strata_qsa_kv_append_<mode>` | MTP stage 9 covers T=1..8 for fp16, int8 without and with `kv_rot`, rotated q4_0, and the K8v4-requested→int8 MTP mapping. It compares K norm/RoPE and FWHT against scalar oracles, then compares every cache byte with scalar format-specific append oracles. It checks untouched cache cells and tails, immutable parameters/norm/RoPE, malformed T, duplicate and out-of-capacity positions, forged layout, short and foreign cache views, invalid rotation/format combinations, caller-owned late rollback and open-group rejection. `metal_mtp_storage_test` separately binds and dispatches stage 9 in an int8 pass with the shared rotation mode enabled; production configuration and inspected-model parity remain open |
| `metal_mtp_kv_append_test_concurrent` | metal | Same QSA kernels as `metal_mtp_kv_append_test` | Repeats the stage-9 formats with concurrent dispatch and verifies rejection inside an open group without changing fixture buffers |
| `metal_mtp_query_projection_test` | metal | `strata_mv_q8_0_n1` | MTP stage 10 requires T=1 and reuses stage 8's Q8_1 activation row for the 2560→12288 Q8_0 query projection. Checks exact scalar-oracle output across the full 24×512 q_full tensor, finite values, allocation guards, untouched sink and immutable inputs, invalid T, short weights/sink, foreign weights, q_full/sink and partial q_full/xq plus sink/xq aliases, caller-owned rollback and open-group rejection. `metal_mtp_storage_test` also verifies the runtime mapping in its full ordered chain; inspected-model parity remains open |
| `metal_mtp_query_projection_test_concurrent` | metal | Same Q8_0 mat-vec kernel as `metal_mtp_query_projection_test` | Repeats stage 10 with concurrent dispatch and verifies ordered-boundary rejection without changing fixture buffers |
| `metal_mtp_query_norm_test` | metal | `strata_qsa_norm_rope`, O-class `strata_qsa_fwht256` | MTP stage 11 requires T=1 and compares qcur against scalar QSA normalization/RoPE and optional FWHT oracles in standalone and caller-owned segments. Checks finite outputs, guards, immutable inputs, invalid T/position, short and foreign views, q_full/qcur and params/input aliases, ordered-group rejection, and rollback after a later stage validation failure. `metal_mtp_storage_test` also checks runtime binding with the same explicit rotation mode as stage 9; production configuration and inspected-model parity remain open |
| `metal_mtp_query_norm_test_concurrent` | metal | Same QSA kernels as `metal_mtp_query_norm_test` | Repeats stage 11 with concurrent dispatch and verifies ordered-boundary rejection without changing fixture buffers |
| `metal_mtp_window_ids_test` | metal | `strata_mtp_window_ids` | MTP stage 12 compares active IDs for short, exact, truncated and full-history finite windows against absolute-cell expectations, including IDs greater than the ID-region capacity and a 4,097-ID window that crosses threadgroup and 2,048-thread grid-stride boundaries. A zero window writes full-history identity IDs; a finite→unwindowed reuse case verifies stale finite IDs are replaced. Verifies inactive-tail sentinels, guards, immutable params, signed-range edges, malformed T/window/n_kv/StepRow width, short/misaligned/foreign views, active output/params alias rejection, caller-owned rollback and open-group rejection |
| `metal_mtp_window_ids_test_concurrent` | metal | Same kernel as `metal_mtp_window_ids_test` | Repeats stage 12 with concurrent dispatch and verifies ordered-boundary rejection without changing fixture buffers |
| `metal_sampler_parity_test`, `metal_sampler_parity_test_concurrent` | metal | Section 5.12's greedy, split-part, split-merge, composite sampled, coupled-penalty and CPU-tail twin on random, edge and fixture inputs for T=1..8 and model/maximum vocabularies; standalone and shared Encoder paths match the scalar references; concurrent tests reject dependent merge/sample calls in an open group and verify segment discard, including discard of a prior split-part dispatch; shared coupled-penalty→sampled composition and 248,320/262,144 vocabulary merge scratch are checked in one Encoder segment; fixtures 16 and 17 are deterministic semantic ports, with host fixture 15 in `sampler_tail_test`; literal upstream fixtures 1-14 and 18 and production sampler integration remain open under REV-028 | Candidate IDs/logit bits and CPU-tail tokens/probabilities exact |
| `metal_util_test` | metal | `strata_util_*` | B |
| `metal_prefill_test` | metal | `strata_pf_gr_norm`, `strata_pf_gr_norm_lo`, `strata_pf_gr_write_norm`, `strata_pf_gr_write_norm_lo`, `strata_pf_gdn_gates`, `strata_pf_gdn_conv_tiled`, `strata_pf_gdn_conv_hist`, `strata_pf_gr_silu`, `strata_pf_gr_silu_lo`, `strata_pf_gr_mix`, `strata_pf_gr_mix_lo`, `strata_embed_rows_{q3_k,iq3_s,iq4_xs,bf16}`, `strata_gr_broadcast` | Prefill GR normalization T=1/3/8 covers BF16X2 modes 0/1/2 and random, edge (including an order-sensitive reduction sentinel) and fixture data against a scalar oracle that reproduces 256-thread FMA accumulation and both XOR reduction stages. Scale is U(4), high BF16 is U(1), and the low remainder is bitwise checked from the GPU scale/high outputs. GR write-norm covers T=1/3/8 in all three modes; the fused residual, scale, high BF16 and mode-1 low BF16 outputs match the separate GR-post then GR-normalization GPU chain bitwise. Prefill GDN gates cover T=1/3/8 with softplus gate U(256) and sigmoid beta U(128) against scalar transforms; fixtures cover branch and tail lanes, readonly input overlap, guards, alias/extent/ownership rejection and rollback. Prefill GDN tiled convolution covers T=1/3/64/65/129 against a scalar four-tap/FMA/SiLU oracle with a cancellation-sensitive non-dyadic FMA-order negative control; history/QKV/weight immutability, guards, validation, rollback and open-group rejection are checked. Convolution history covers T=1/2/3/8/65 in standalone and caller-owned Encoder forms against an exact three-row oracle; T=1/2/65 same-segment convolution→history cases verify old-history consumption. History/QKV aliases, short/misaligned/foreign views, row bounds, both-overload rollback and open-group rejection are checked. GR SiLU T=1/3/8 covers modes 0/1/2 with positive, negative, zero and tail values; high BF16 is U(1) and the BF16 remainder is U(2) against the scalar SiLU oracle. Exact high-output BF16 halfway cases check RNE bits in every mode; mode 1 also checks positive and negative remainder halfway cases against exact even-result bits. GR mix covers T=1/3/8 × BF16X2 mode 0/1/2, checks high BF16 and F16 against scalar oracles, and checks `high + low` reconstruction within `2^-15` of the absolute-term magnitude in modes 1/2 while requiring the low part to improve more than a quarter of high-only outputs. Exact binary-input midpoint fixtures assert high BF16 and positive/negative low-remainder RNE bits in modes 1/2. Standalone and shared Encoder forms check read-only inputs, guards, invalid mode/extent/alignment/alias/ownership, and late-failure discard. The embedding gather → GR broadcast → Prefill normalization chain runs in one segment; a concurrent-group fixture verifies rejection. Prefill GEMM and chunk-loop integration remain open. |
| `metal_dispatch_modes_test` and the `model` properties | model | Section 10.4 | B |
| `metal_window_gdn_commit_batch_test` | metal | `strata_gdn_conv_commit`, `strata_gdn_step_norm` mode 1, `strata_ple_history` | T=2 accepted-prefix replay across all 36 GDN state ordinals, followed by PLE history advancement in the same segment. Per-layer convolution history uses distinct qkv rows; every recurrent state matches an independent per-layer scalar oracle; PLE history matches the two accepted normalized rows. Rejects a short final gamma, short PLE rows, a repeated GDN state and a PLE/GDN alias before dispatch, verifies caller-owned rollback after an earlier encoded layer, and preserves replay inputs and unused scratch |
| `metal_window_gdn_commit_batch_test_concurrent` | metal | Same GDN/PLE commit kernels and batch fixture | Runs the same GDN and PLE accepted-prefix state fixture with `STRATA_METAL_DISPATCH=concurrent`, preserving serial layer ordering and all oracle, alias and rollback checks |

The indexed-head encoder-specific rejection checks also cover row maps from a
foreign Context, forged CPU/GPU metadata, and foreign, enqueued or committed
command buffers.

`coupled_draft_test` (host) runs unchanged. Every twin also has its `real` set under the `model` label, as
`metal_<area>_model_test`.

---

## 11. Measurements

Every L item has a provisional default in this document, so implementation does not wait for it. The measurement
either replaces the default or confirms it. Results go to [appendix-a-measurements.md](appendix-a-measurements.md),
and the default they drive is updated in the section named in the last column.

**How a timing item is measured.**
- The programs are in `spikes/metal-tuning/`. They link the engine's `strata_metal` library and the twin harness of
  section 10.1, so each kernel runs with the engine's grid, threadgroup size and parameter block.
- A kernel's GPU time is `GPUEndTime − GPUStartTime` of a command buffer that repeats the dispatch 100 times on real
  weights from the Q2_0 pack. The result is the median of 20 such command buffers.
- Runs use a release build, on mains power, with no other GPU work.
- "Each family" means the reference Mac (M5) and one Mac of each of M1-M4: the family runs of
  [09](09-testing-and-validation.md).
- A "win" is at least 5% less GPU time. Smaller gains do not change a default.

| ID | Question | How to measure | Provisional default | Decision it drives |
| --- | --- | --- | --- | --- |
| L-2 | Which `strata_gr_down` tile width TILEV is fastest? How many rows R should a mat-vec threadgroup take, per format? | TILEV 256, 512 and 1,024 in windows of T = 1, 4 and 8. R = 1, 2, 4 and 8 for every `strata_mv_<fmt>_n<NC>` at NC = 1 and 8. Run on each family. | TILEV 512; R = 4 for every format | A TILEV that wins on a family becomes `strata_gr_down_tv<TILEV>` there (section 5.4). On M5, the fastest R for each format becomes its `common/params.h` value. On another family, an R that wins becomes the `_r<R>` host name (section 5.8). Neither choice changes bits. |
| L-3 | In what order does `simd_sum` add on M1-M4? | The A.10 probe of [spikes/metal-numerics/](spikes/metal-numerics/), on each family | Ascending, as on M5: offsets 1, 2, 4, 8, 16 | No bitwise contract depends on it (section 3.4). The order is recorded. Where it differs from M5's, the twins that hold a `simd_sum` result to U(n) are rerun on that family. |
| L-4 | Is program-scope `constant` still the fastest place for the codebooks inside the real kernels? | Every mat-vec and expert kernel whose format has a codebook (section 3.7), at T = 1 and 8. Compare the tables as `constant` arrays against a threadgroup copy, on each family. | `constant` (A.10) | A threadgroup copy that wins for a format on a family becomes that format's placement there, within section 7's budget. The bits do not change. |
| L-5 | Do the expert kernels' shapes suit M1-M4? | The rows per threadgroup of `strata_moe_gu_<fmt>` and `strata_moe_down_<fmt>`, at T = 1 and 8. The prefill expert tile of section 6.6 (64 entries × 64 columns, k step 32) on a 4,096-row chunk. Run on each family. | The M5 shapes | A shape that wins on a family becomes its default there. It must fit section 7's limits and keep each output's order of summation (sections 5.10 and 6.6). |
| L-6 | Is `mpp` usable on each family? | Five checks on each family, M5 included: (1) MSL 4.0 and `bfloat` tensors compile; (2) the edge slices at T = 1, 63 and 65 write only live elements; (3) `multiply_accumulate` adds to Y; (4) `prefill_chunk_independent` passes under `mpp`; (5) the twins of section 10.3 pass U(2K) and R(2^-16). | `mpp` on Apple10 (M5), `simd` elsewhere ([06](06-engine-host-port.md), section 4.4) | The family's `STRATA_METAL_GEMM` default, for the GEMMs of section 6.2 and the expert tiles of section 6.6. A failed check makes it `simd`. If check 4 fails, `mpp` gives up the chunk-size independence of section 6.1, so it is not used even where it is faster. |
| L-7 | Which `simd` GEMM tile is fastest? | Output tiles of 32 × 64, 64 × 64 and 32 × 32, with k step 32, at T = 512 and 4,096, for every dense projection shape of section 6.2. Run on M1-M5. | 32 × 64 | A tile that wins on a family becomes its default there. Only tiles that keep each element's order of section 6.2 are eligible. |
| L-8 | How many pipelines does the engine create, and how long does that take? Does every pipeline get its thread count? | Time `Context::open` on each family: cold (`STRATA_METAL_ARCHIVE=0`, after the OS shader cache is deleted as in [08](08-build-and-distribution.md), section 14, B-1) and warm. Count the manifest's host names. Read every pipeline's `maxTotalThreadsPerThreadgroup` and compare it with its manifest value. | All pipelines at startup ([06](06-engine-host-port.md), section 2.3) | If a cold `Context::open` takes more than 10 s on a family, the host creates only the pipelines the loaded model's formats use; the manifest check stays complete. A pipeline below its manifest thread count fails `Context::open` (512 for `strata_gdn_step_norm`, section 5.5); that family then needs a smaller variant before it is supported. |
| L-9 | Does nvcc contract the lines this document says it contracts? | Read the SASS of upstream's release build for the architecture of the CUDA reference GPU ([09](09-testing-and-validation.md)) with `cuobjdump -sass`. Check each listed line for FFMA against FMUL followed by FADD. The lines: the mat-vec FMA tables (section 5.8); the MoE combine (section 5.10); the S2 expert accumulation and the MTP round's rounding (section 5.11); the penalty (section 5.12); `ss += r*r` in `gr_norm_rs` (section 6.3); the prefill combine (section 6.6). | The forms the sections give | Where the SASS differs, the kernel and its oracle change together, in one commit. |
| L-10 | Does the GPU flush FP32 denormal operands or results? | A `safe/` self-test kernel that adds, multiplies, fmas and converts denormal inputs and inputs whose results are denormal. Compare with IEEE results in `metal_math_test`, on each family. | IEEE: no flush (section 3.5) | If the GPU flushes, the oracles flush too, and the denormal row of section 3.9 is updated. |
| L-11 | Would any kernel gain from `fast/`? | For each candidate, the same kernel built in `safe/` and in `fast/`, timed in a release-build window on M5 and on one of M1-M4. Candidates are kernels with no bitwise contract, taken in order of their share of a window's GPU time. | No kernel in `fast/` | A move under the four rules of section 3.2, recorded in section 9.5 |
| L-12 | How long does the largest prefill chunk run, and where is the GPU timeout? | (1) A synthetic kernel that loops for 2, 5, 10, 20 and 40 s gives the limit at which macOS ends a command buffer with `MTLCommandBufferErrorTimeout`. (2) A 32,768-token prompt at C = 32,768 on each family that can hold it: the chunk's command-buffer GPU time. | One command buffer per chunk (section 6.1) | If (2) fails with the timeout, or takes more than half of the limit (1) finds, the host encodes each layer as its own command buffer (section 6.1). The bits do not change. |
| L-13 | What share of a chunk do attention, block selection and the expert decode take? | A GPU capture (`--metal-capture`, [08](08-build-and-distribution.md), section 10.2) of one chunk of C = 4,096 at p0 = 0, 32,768 and 131,072, with Xcode's per-dispatch GPU time summed per step of sections 6.5 and 6.6. Q2_0 pack on M5 and on the slowest family. | The Phase-1 kernels | Attention above 25% of a chunk adds `strata_pf_prompt_attn_<mode>` (section 2.6, work item L19). Selection above 25% adds `strata_pf_sel_scores_mma` (L20). Expert decode above 25% opens work item L22. |
| L-14 | Is `concurrent` dispatch faster than `serial`? | The decode benchmark of [09](09-testing-and-validation.md) in both modes, plus a window of each T = 1..8 and a commit, timed on each family | `serial` (section 8.2) | The default becomes `concurrent` only if it is at least 3% faster on the decode benchmark of every measured family. |
| L-15 | Is the draft head's indexed read as fast as a contiguous read? | GPU time of the draft-vocabulary head mat-vec (106,299 rows read by index, section 5.11) against the contiguous head mat-vec over the same number of bytes, on M5 and on the slowest family | Indexed read in place | Below 90% of the contiguous bandwidth, the bind gathers the rows as upstream does, and the plan adds the copy to `mtp_bind_bytes` (section 5.11; [05](05-memory-and-residency.md)). |

L-1 is retired: its question, `serial` or `concurrent` dispatch, is L-14.

Items in other documents that change values here:

| Item | Document | Values here that it changes |
| --- | --- | --- |
| H-4, the CPU encode time of a T = 8 window | [06](06-engine-host-port.md), section 15 | Above 1 ms: work item L17, indirect command buffers (section 8.3) |
| M-1, wake and event-hop latency | [06](06-engine-host-port.md), section 15 | How much of a tiered layer's hop the shared expert hides (section 8.3). If the hop is not hidden, nothing here changes, but gate G2 may fail. |
| M-2, a second queue | [06](06-engine-host-port.md), section 15 | A gain makes work item L18, a second window arena (section 8.6), a Phase-2 item |
| R-1, the 256-expert router | [03](03-model-reference.md), section 10 | The decode router's deviation for Coder (section 5.10) and its tolerance |
| R-3, MTP acceptance | [03](03-model-reference.md), section 10 | Whether work item L16, the batched drafter prefill, is worth its deviation (section 5.11) |
| PF-1, the prefill-speed criterion | [09](09-testing-and-validation.md) | A failure adds `strata_pf_prompt_attn_<mode>` (section 2.6, L19) |

---

## 12. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). An item is done when its
"Done when" holds on the reference Mac, and the `metal` label stays green on every family in the family runs of
[09](09-testing-and-validation.md). The host code that encodes the kernels is [06](06-engine-host-port.md) work
items H3, H8 and H10. The library build is [08](08-build-and-distribution.md) work item B3.

### 12.1 Phase 1

The order follows the bring-up of audit E.8: twins first, then T = 1, then windows, then MTP and prefill.

| # | Item | Done when |
| --- | --- | --- |
| L1 | `common/` headers: `types.h`, `math.h`, `reduce.h`, `dot.h`, `codebooks.h`, `params.h`; the `strata_selftest_*` kernels (sections 2.3, 3 and 4.2) | `metal_math_test` and `codebooks` pass. Every parameter block of section 4.2 has the same size and member offsets in MSL and in C++, checked by a kernel that writes its `offsetof` values. |
| L2 | The twin harness and the oracle library (sections 10.1-10.2): `tests/metal/twin.{hpp,cpp}`, `strata_metal_oracles`, the four input sets, the poison check, `row_t_independent`, `tests/metal/synth.cpp` | Three planted faults each fail with the seed and the first failing index: a kernel that writes one byte past its range; one that is one ulp off in a B twin; and one whose row t depends on T. They stay as `metal_twin_selftest` ([09](09-testing-and-validation.md), section 9.1). |
| L3 | Q8_1 quantiser and the quantised and BF16 mat-vecs (sections 5.8-5.9) | `metal_mv_test` passes for every Phase-1 format, NC and R; `mv_multi_eq_single` passes with its negative control; `mv_decode` covers every input word |
| L4 | Embedding, PLE and GR (sections 5.2-5.4) | `metal_embed_test`, `metal_ple_test` and `metal_gr_test` pass; `ple_batch_eq_sequential` passes; the fused fold equals `strata_gr_post` bitwise |
| L5 | GDN (section 5.5) | `metal_gdn_test` passes for modes 0, 1 and 2 |
| L6 | QSA, KV, indexer, selection and attention (sections 5.6-5.7) | `metal_qsa_test` and `metal_attn_test` pass for every `--kv` mode; `reg` and `mem` give the same ids. Sparse-attention chunk and merge now have a tested selection→top-k→attention→merge shared-Encoder coordinator, including O-class FWHT and gate for q4_0/k8v4; production Window/Prefill integration and external parity remain open under REV-002/025/032 |
| L7 | MoE: router, plan, experts, shared expert, combine (section 5.10) | `metal_moe_test` passes with both plan tables (full and tiered), both router sizes, and the validated grouped gate/up encoder for all eight formats; Window scheduling, shared-expert production dispatch and model-backed inference remain open |
| L8 | Head and sampler (section 5.12) | The sampler twins pass synthetic coverage, upstream host fixture 15, and semantic ports of upstream cases 16-17; `samp_coupled_eq_host` passes; `coupled_draft_test` passes unchanged. L8 remains open until literal upstream cases 1-14 and 18, external generated-vector parity, and production window/head/MTP integration are completed under REV-028 |
| L9 | The T = 1 window of section 5.1 and its encode in full mode (sections 8.1-8.3), with the dump hooks of section 9.4 | The benchmark prompts decode at T = 1 and pass gate C1 against the R12 golden set ([03](03-model-reference.md), section 11); `dump_eq_nodump` passes |
| L10 | Windows of T = 2..8, commit and write-through windows (sections 5.1, 8.4 and 8.5), with [06](06-engine-host-port.md) H8 | `window_eq_decode`, `state_after_commit_eq_decode` and `write_through_eq_commit` pass; gate C2 passes |
| L11 | The MTP drafter (section 5.11) | `metal_mtp_test` and `mtp_kv_t_independent` pass; a recorded window replayed on the CPU oracle and on Metal gives the same draft ids, `row_top_prob` and drafter K/V |
| L12 | Tiered encoding: the per-layer signal and wait, the shared expert between them, and the plan's miss table (section 8.3), with [06](06-engine-host-port.md) H10 | `tiered_eq_full` passes for Q2_0 and Coder; gate C4 passes |
| L13 | The prompt path (section 6): the chunk loop, GEMMs, the GR read and write, GDN, QSA, MoE, embedding and PLE | `metal_prefill_test` passes in the classes of section 10.3; `idx_batch_eq_sequential`, `pf_gdn_eq_step_norm`, `pf_sort_eq_host_grouping` and `prefill_chunk_independent` pass; the prompt-path criterion of [09](09-testing-and-validation.md) passes |
| L14 | Dispositions (section 9): every kernel flag and kernel-scope variable, `STRATA_METAL_DISPATCH` and `concurrent` groups, the `STRATA_METAL_ICB` notice | `kernel_env_dispositions_test`, a table-driven test, sets each variable of sections 9.2 and 9.3 and checks its disposition: honoured, warned, or ignored with its line; `metal_dispatch_modes_test` passes |
| L15 | Measurements L-2 to L-15 (section 11) | Each result is in appendix A, and the default it drives is updated here and in the manifest |

### 12.2 Conditional and Phase-2 items

Each of these starts only when its condition holds. Until then, the Phase-1 behaviour this document specifies stands.

| # | Item | Condition | Done when |
| --- | --- | --- | --- |
| L16 | Batched drafter prefill: upstream's `draft_kv` (E-9, `prefill.cpp:737`) through the prompt GEMMs, in chunks of at least 64 rows | The drafter prefill exceeds 5% of a long prompt's time, or R-3 shows that the batched K and V bits accept better | Drafter K/V within the prefill tolerance of section 10.3; acceptance on the R-3 sets not lower than with `mtp.prefill` |
| L17 | Indirect command buffers (`STRATA_METAL_ICB=1`): one per T and per mode combination, one range per segment (section 8.3) | H-4 measures an encode time above 1 ms | Outputs bitwise equal to the encoded windows for every T in full and tiered mode; H-4 rerun shows the saving |
| L18 | A second window arena, so the next window can overlap the commit and the MTP round (section 8.6; [06](06-engine-host-port.md), section 10.2) | M-2 shows a gain, after gates C1-C4 pass | `window_eq_decode` and `state_after_commit_eq_decode` pass with the overlap; the decode benchmark improves |
| L19 | `strata_pf_prompt_attn_<mode>` (section 2.6) | PF-1 fails, or L-13 finds attention above 25% of a chunk | Specified here with its ε; twins in R(ε) against the decode kernels' outputs (section 10.3); PF-1 passes |
| L20 | `strata_pf_sel_scores_mma` (section 2.6) | L-13 finds selection above 25% of a chunk | As L19, for block scores, with the selected ids equal to the decode kernels' wherever the scores are separated |
| L21 | `strata_mtp_round_device` (section 2.6) | Gates C1-C4 pass | The round's acceptance equals the host step's on every window of the benchmark prompts; gate C2 passes |
| L22 | Decode each expert once per chunk into F16 scratch, as upstream's ring does, in place of the decode inside each tile (section 6.6) | L-13 finds the expert decode above 25% of a chunk | `metal_prefill_test` passes in the expert tiles' classes; the chunk time improves on the measured families |
| L23 | `STRATA_QSA_DUMP` and `STRATA_QSA_DUMP_LAST` (section 6.5): a host read of `ids` after each QSA layer, which needs the per-layer command buffers of section 6.1 | A user of `tools/qsa_locality.py` asks for it | The file is byte-identical in layout to upstream's and `tools/qsa_locality.py` reads it; `dump_eq_nodump` covers the flag |

---

## 13. Where the details are

The audit sections and measurements behind each part of this document. Where they disagree with this document,
section 1.3 applies.

| Topic | This document | Audit and measurements |
| --- | --- | --- |
| Kernel inventory and what is not ported | 2.4-2.5 | [C.0-C.4](audit/A-model-and-kernels.md); the catalogues of [K1 section 9](audit/K1-gr-router-shared-rope-cvec.md), [K2 section 4](audit/K2-qsa-kv-gdn.md), [K3 section 3](audit/K3-quant-and-matvec.md) and [K4 section 5](audit/K4-ple-mtp-head-sampler.md) |
| Numerics: math modes, contraction, reductions, conversions | 3 | [F.1-F.4, F.8](audit/A-model-and-kernels.md); [K1 section 2](audit/K1-gr-router-shared-rope-cvec.md); [appendix A.10](appendix-a-measurements.md) |
| Codebooks and integer dots | 3.7-3.8, 5.8 | [B.1](audit/A-model-and-kernels.md); [K3 sections 1-2](audit/K3-quant-and-matvec.md); A.10 |
| Deviations from CUDA | 3.9 | [K1 section 12](audit/K1-gr-router-shared-rope-cvec.md); [K2 section 7](audit/K2-qsa-kv-gdn.md); [K3 section 8](audit/K3-quant-and-matvec.md) |
| Binding, parameter blocks and buffers | 4 | [F.6-F.7](audit/A-model-and-kernels.md); [A.11](appendix-a-measurements.md); [05](05-memory-and-residency.md) |
| The decode window | 5.1 | [A.2, D.1](audit/A-model-and-kernels.md); [K5 section 1](audit/K5-orchestration-vram-tests.md) |
| Embedding, PLE | 5.2-5.3, 6.7 | [A.9-A.10](audit/A-model-and-kernels.md); [K4 sections 1-2](audit/K4-ple-mtp-head-sampler.md) |
| GR hyper-connections, control vectors | 5.4, 6.3 | [A.3, A.8](audit/A-model-and-kernels.md); [K1 sections 3 and 7](audit/K1-gr-router-shared-rope-cvec.md) |
| GDN | 5.5, 6.4 | [A.4](audit/A-model-and-kernels.md); [K2 section 2](audit/K2-qsa-kv-gdn.md) |
| QSA, KV, indexer, selection, attention | 5.6-5.7, 6.5 | [A.5](audit/A-model-and-kernels.md); [K2 sections 1 and 3](audit/K2-qsa-kv-gdn.md) |
| RoPE | 4.9 | [A.7](audit/A-model-and-kernels.md); [K1 section 6](audit/K1-gr-router-shared-rope-cvec.md) |
| Quantised and BF16 mat-vecs | 5.8-5.9 | [B.2](audit/A-model-and-kernels.md); [K3 sections 2-3](audit/K3-quant-and-matvec.md) |
| MoE: router, plan, experts, shared expert, combine | 5.10, 6.6 | [A.6, B.4](audit/A-model-and-kernels.md); [K1 sections 4-5](audit/K1-gr-router-shared-rope-cvec.md); [K3 section 5](audit/K3-quant-and-matvec.md) |
| MTP drafter | 5.11 | [A.11](audit/A-model-and-kernels.md); [K4 section 3](audit/K4-ple-mtp-head-sampler.md) |
| Sampler | 5.12 | [A.10, A.12](audit/A-model-and-kernels.md); [K4 section 4](audit/K4-ple-mtp-head-sampler.md) |
| The prompt path | 6 | [D.3](audit/A-model-and-kernels.md); [K5 section 3](audit/K5-orchestration-vram-tests.md); [A.6-A.7](appendix-a-measurements.md) |
| Threadgroup memory | 7 | [F.5](audit/A-model-and-kernels.md); A.10 |
| Encoding, commit, write-through | 8 | [A.12, F.7, F.9](audit/A-model-and-kernels.md); [A.5, A.11](appendix-a-measurements.md); [06](06-engine-host-port.md), sections 8-10 |
| Flags and variables | 9 | [B section 1.3](audit/B-engine-host.md); [06](06-engine-host-port.md), section 4 |
| Tests, oracles, bring-up order | 10, 12 | [E.0-E.8](audit/A-model-and-kernels.md); [K4 section 8](audit/K4-ple-mtp-head-sampler.md); [K5 section 6](audit/K5-orchestration-vram-tests.md); [09](09-testing-and-validation.md) |
