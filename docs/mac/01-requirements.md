# 01. Requirements: what the Mac port must do

Status: implementation-ready specification, 2026-10-07.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- [00-feasibility-study.md](00-feasibility-study.md), sections 4, 5 and 7: the target Macs, the scope decision and
  the gates;
- [03-model-reference.md](03-model-reference.md) to [08-build-and-distribution.md](08-build-and-distribution.md): the
  specification behind each requirement;
- [09-testing-and-validation.md](09-testing-and-validation.md): the checks that each requirement names;
- [appendix-a-measurements.md](appendix-a-measurements.md): the measured facts of the reference Mac.

This document lists what a conforming port must do, each item with an ID and the checks that show it holds. Gate G3
([00](00-feasibility-study.md), section 7) passes when every requirement here holds. It covers:
- how to read a requirement, its checks and when it holds;
- the scope: the target Macs, the models, and what is out of scope;
- the requirements, by area: platform, models, engine features, memory, correctness, performance, server and
  product, robustness, build and release, maintainability, and the user documents;
- the targets that are reported but not required.

Other documents cover the rest:
- how each requirement is met: the document and section in its "Specified in" column;
- how each check runs, its thresholds and its schedule: [09-testing-and-validation.md](09-testing-and-validation.md);
- the order of the work: [10-implementation-plan.md](10-implementation-plan.md);
- what could stop a requirement from being met: [11-risks-and-open-questions.md](11-risks-and-open-questions.md).

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. How to read this document

### 1.1 Requirement IDs

A requirement ID is `REQ-<AREA>-<NN>`, for example `REQ-MEM-04`. It is the first cell of a row of the requirement
tables of sections 3-13, and only those rows start with `REQ-`. `tools/metal/test_map.py` finds the IDs with the
pattern `^\| (REQ-[A-Z]{3}-[0-9]{2}) \|` ([09](09-testing-and-validation.md), section 9.4, check 3).

| Area | Section | Covers |
| --- | --- | --- |
| PLT | 3 | Platform: OS, hardware, Python, setup's refusals |
| MOD | 4 | Models: the files, their formats and the formats that are refused |
| FUN | 5 | Engine features: the protocol, the flags, context, sampling, speculation, the prompt path and the caches |
| MEM | 6 | Memory: the plan, full and tiered mode |
| COR | 7 | Correctness: C1-C4, the prompt path, needles, determinism |
| PRF | 8 | Performance: decode, prefill, regressions |
| SRV | 9 | Server and product: setup, the server, the APIs, telemetry, MCP |
| ROB | 10 | Robustness: GPU errors, stalls, memory pressure, signals |
| BLD | 10 | Build and release: the asset, signing, reproducibility and licences |
| MNT | 12 | Maintainability: following upstream, the test map |
| DOC | 13 | User documents |

The rules:
- **IDs are never reused.** A dropped requirement keeps its row: the Requirement cell becomes
  `Withdrawn <yyyy-mm-dd>: <reason>`, the "Verified by" cell `withdrawn`, and `requirements.json` maps the ID to
  `"withdrawn"`. A new requirement takes the next free number of its area.
- **A requirement changes in one commit with `tests/metal/requirements.json`**, which lists every ID with its checks
  ([09](09-testing-and-validation.md), section 9.4).
- **A threshold changes by the rule of [09](09-testing-and-validation.md), section 10.3**, in the same commit as the
  row here and `tools/metal/accept/criteria.py`.

### 1.2 Checks, and when a requirement holds

The "Verified by" column names checks in the form `requirements.json` uses:

| Check | Meaning | Defined in |
| --- | --- | --- |
| `ctest:<name>` | A CTest test | [09](09-testing-and-validation.md), section 9.1 |
| `accept:<ID>` | A criterion of `accept.py` | [09](09-testing-and-validation.md), section 3.2 |
| `python:<path>` | A Python test module | [09](09-testing-and-validation.md), section 9.3 |
| `ci:<workflow>/<job>` | A job of a workflow in `.github/workflows/`, for example `ci:mac-hardware.yml/mac-host` | [08](08-build-and-distribution.md), section 12; [09](09-testing-and-validation.md), section 6.3 |
| `review:<id>` | A review by a person, with its record | [09](09-testing-and-validation.md), section 7.6 |

A requirement **holds** when every check it names has passed on the candidate commit, in the sense of
[09](09-testing-and-validation.md), section 6.3: on a commit whose engine sources and whose `serve/`, `tools/` and
`setup.py` equal the candidate's. A requirement whose only checks are reviews holds when each review's record names
such a commit.

G3 also needs two things that no single requirement names:
- **The release verification.** `verify_release.py` passes on a release-candidate tag
  ([08](08-build-and-distribution.md), section 11.8). Its step 6 is the release run of
  [09](09-testing-and-validation.md), section 8, and it is the `ci:ci.yml/mac-host` check.
- **The family runs**, one for each of M1-M4, on the candidate commit ([09](09-testing-and-validation.md),
  section 7.4).

Some requirements gate earlier, at G1 or G2. Each gate is its own run of `accept.py`
([09](09-testing-and-validation.md), section 6):

| Gate | End of | Requirements it checks first |
| --- | --- | --- |
| G1 | Phase 2 | Every `model`-label CTest (`ctest --preset mac-model`, step 1 of [09](09-testing-and-validation.md), section 6.1), REQ-COR-01 (C1.K; C1.4 on the models; C1.1 and C1.5 NOT RUN until a model-level reference is named), REQ-COR-03 (NOT RUN, same reason), REQ-COR-04 (the nightly arms), REQ-PRF-02 |
| G2 | Phase 4 | REQ-PRF-05, REQ-COR-08 |

A gate that fails follows its own rule in [09](09-testing-and-validation.md), section 6: the gate decides whether the
work goes on, not whether these requirements stand.

### 1.3 Terms

| Term | Meaning |
| --- | --- |
| The reference Mac | An Apple M5 Pro, 20-core GPU, 64 GB, macOS 27.0, the self-hosted runner `strata-ref` ([08](08-build-and-distribution.md), section 12.4). W = 51.84 GiB. Every number here that names no Mac is measured there. |
| A target Mac | A Mac of section 2.1 whose status is not "Not a target" |
| W | The GPU budget: `recommendedMaxWorkingSetSize`, or `--gpu-budget-mib` if smaller ([05](05-memory-and-residency.md), section 3.1) |
| Full mode, tiered mode | Every expert resident in GPU memory; or a resident set plus a slot pool fed from the SSD ([05](05-memory-and-residency.md), sections 4 and 5) |
| The catalogue | The model files of section 2.2, which setup offers on a Mac |
| Upstream | [Niko1221/Strata](https://github.com/Niko1221/Strata), at the commit in `UPSTREAM_SYNC`. Its CUDA build on the CUDA reference PC is the numerical reference ([09](09-testing-and-validation.md), section 2.4). |
| Setup | `setup.sh` and `setup.py`, the installer ([07](07-server-setup-port.md), sections 4 and 5) |

---

## 2. Scope

### 2.1 Target Macs

| Memory | Status | How the catalogue runs ([05](05-memory-and-residency.md), section 8) |
| --- | --- | --- |
| 64 GB | Primary | Full mode, except IQ3_S (tiered). Q2_0 and Coder IQ1_M hold 262,144 cells. |
| 48 GB | Primary | The ISTA and Swift files in tiered mode; Coder IQ1_M in full mode |
| 36 GB | Next | Tiered |
| 32 GB | Stretch | Tiered |
| 96 GB and more | Supported, not tuned | Full mode, with a long context |
| 16-24 GB | Not a target | Setup asks for consent (REQ-PLT-05) |

Every Mac with Apple silicon (M1 or newer) and macOS 27 is in scope; Intel Macs are not. The M5 generation (the
Apple10 GPU family) is the tuned one. M1-M4 run the same engine, with the `simd` form of the prompt-path GEMMs and
expert tiles where M5 uses `mpp` ([04](04-metal-kernels.md), section 11, L-6), and must pass the same tests
(REQ-PLT-03).

### 2.2 Models

The catalogue is the seven files of [03](03-model-reference.md), section 2.2:

| Family (setup key) | Repository | Quants |
| --- | --- | --- |
| `qwen` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | Q2_0, IQ2_XS, IQ3_XXS, IQ3_S |
| `swift` | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | IQ2_XS, IQ3_XXS. Licence: Swift Open License 1.0, which setup prints before the download. |
| `coder` | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | IQ1_M (256 experts per layer) |

With them come the vision encoders, the MTP draft layer, the draft vocabularies, the expert profiles and the control
vector of [03](03-model-reference.md), section 2.3. Which files each test uses is
[09](09-testing-and-validation.md), section 2.3: Q2_0 and Coder gate, the IQ formats run weekly.

### 2.3 Out of scope

Not in the first release. Each is refused or hidden with a message, never run with a wrong result:
- **Model files:** Unsloth UD-Q4_K_XL (refused), Swift Q2_0 (hidden), the Orca files (their Q5_0 expert downs), and
  the optional FP8 PLE table ([03](03-model-reference.md), sections 2.2 and 2.3).
- **PC features** ([07](07-server-setup-port.md), section 1.1): NVIDIA and AMD card selection, multi-GPU, the CUDA
  wheels, the AVX-512 Q2_0 repack, the low-RAM `experts.bin`, KV streaming, `--calibrate`, Docker.
- **The optional CPU expert helper** ([06](06-engine-host-port.md), section 12).
- **Packaging:** a `.pkg`, a `.dmg`, Homebrew, and a release asset downloaded with a browser and unpacked by hand,
  which macOS quarantines ([08](08-build-and-distribution.md), section 11.5). Setup installs the engine itself.

### 2.4 Supported platform

This repository targets Apple Silicon Macs running macOS 27 or later. Windows, Linux, Intel Macs and other graphics
backends are outside the product scope.

---

## 3. Platform (PLT)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-PLT-01 | `strata` and `strata-vision` must be arm64-only executables for macOS 27.0 and later: `lipo -archs` is `arm64` and `vtool -show-build` shows `minos 27.0`. BUILD.json says `arch` `arm64` and `macos_min` `"27.0"`. | `python:tools/test_metal_build.py`, `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), sections 1.1 (P4), 4.1 and 11.7 (checks 2, 3 and 9) |
| REQ-PLT-02 | The engine must run only on a GPU with unified memory, `MTLGPUFamilyApple7` and `MTLGPUFamilyMetal4`, a `threadExecutionWidth` of 32 for every pipeline, and at least 32,768 B of threadgroup memory. On any other device it exits 1 at startup with a `strata: ` message that names what is missing, and setup refuses the device before it downloads anything. | `ctest:metal_context_test`, `python:tools/test_setup_golden_mac.py` | [06](06-engine-host-port.md), section 2.3; [04](04-metal-kernels.md), section 2.1; [07](07-server-setup-port.md), section 3.2 |
| REQ-PLT-03 | M1, M2, M3 and M4 Macs must run the same release asset, and each family must pass the `host` and `metal` tests and the family run's acceptance steps | `review:family` | [04](04-metal-kernels.md), sections 2.1 and 11; [09](09-testing-and-validation.md), section 7.4 |
| REQ-PLT-04 | Setup must refuse, with exit 1 and the exact texts of [07](07-server-setup-port.md), section 4.3: an Intel Mac, a Python under Rosetta, macOS below 27, a device that `device_problem` rejects, `--backend cuda` or `hip`, `--gpus` or `--layer-split`, and `--gpu` other than 0 | `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), section 4.3 |
| REQ-PLT-05 | Below 32 GB of RAM, setup must ask for consent (`confirm_risk`) before it installs. `--yes` alone stops there; an explicit `--model` with `--yes` proceeds. Setup recommends and never forces (P7). | `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), sections 1.2 (P7) and 4.3 (check 6) |
| REQ-PLT-06 | Until the Mac installer is implemented, `setup.sh` must accept only Darwin on Apple silicon with macOS 27 or newer, then exit 2 before creating `.venv` or changing files. `setup.py` CLI and direct `main()` calls must also fail closed. | `python:tools/test_setup_shell.py` | [07](07-server-setup-port.md), section 5.1 |
| REQ-PLT-07 | The Mac probes in `serve/macos.py` must import only the standard library (no pyobjc, no psutil), import and run under Python 3.9, call tools by absolute path, and never raise: off macOS, and when a probe fails, each function returns its documented empty value | `python:serve/test_macos.py` | [07](07-server-setup-port.md), sections 1.2 (P4) and 3.1 |

---

## 4. Models (MOD)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-MOD-01 | The engine must load each of the seven catalogue files with its pack, as upstream's setup prepares them: the metadata checks, the tensor table, `native_experts.txt` versions 3 and 4, `index.txt`, `conversions.json`, and the expert addresses, Swift's shard-straddling layer included | `ctest:model_metadata_test`, `ctest:model_types_test`, `ctest:gguf_reader_test`, `ctest:gguf_split_test`, `ctest:native_experts_reader_test`, `ctest:pack_index_test`, `ctest:expert_address_test`, `ctest:metal_weight_store_test`, `ctest:metal_weight_store_cli_test` | [03](03-model-reference.md), sections 2.5, 3.4-3.6, 3.9 and 4.5; [06](06-engine-host-port.md), section 6 |
| REQ-MOD-02 | The engine must read the published GGUF files and the Strata pack format, and the unlisted float tensors in `dense.bin` must equal the GGUF bytes | `ctest:dense_vs_gguf_test`, `python:tools/test_iq_pack.py`, `python:tools/metal/test_dense_vs_gguf.py` | [03](03-model-reference.md), sections 3.7 and 4.5 |
| REQ-MOD-03 | The engine must load the MTP runtime that upstream's `mtp_fetch.py` and `mtp_pack.py` write, with its required tensors and the draft vocabulary, while setup prevents mixed or interrupted artifact generations from being reused | `ctest:mtp_reader_test`, `ctest:mtp_reader_vocabulary_test`, `python:tools/test_artifact_set_lock.py`, `python:tools/test_mtp_fetch.py`, `python:tools/test_mtp_pack.py`, `python:tools/test_mtp_vocab.py`, `ctest:metal_weight_store_test`, `ctest:metal_mtp_storage_test` | [03](03-model-reference.md), sections 4.4 and 7; [06](06-engine-host-port.md), section 6; [05](05-memory-and-residency.md), section 5.3 |
| REQ-MOD-04 | The PLE table must be found through each file's tensor table, and its constants must equal the file's `qwen4exp.ple.*` keys | `ctest:ple_constants_test`, `ctest:ple_reader_selftest`, `ctest:ple_oracle_layout_test`, `ctest:ple_oracle_model_test` | [03](03-model-reference.md), sections 6.8 and 9.3 |
| REQ-MOD-05 | Every Phase-1 format of [03](03-model-reference.md), section 3.3, must have its kernel: expert gate/up in Q2_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS; expert down in Q2_0 and IQ4_NL; dense mat-vecs in Q2_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ3_S and IQ4_XS; shared-expert downs in Q2_0, IQ4_NL, Q4_0, Q5_0 and Q8_0; the embedding gather in Q3_K, IQ3_S and IQ4_XS; the head in Q5_K, IQ4_XS and Q6_K; BF16, F16 and F32; and the S2 blob of the MTP experts | `ctest:model_types_test`, `ctest:codebooks`, `ctest:metal_mv_test`, `ctest:metal_moe_test`, `ctest:metal_embed_test` | [03](03-model-reference.md), section 3.3; [04](04-metal-kernels.md), section 2.4 |
| REQ-MOD-06 | A file with a tensor type that has no Metal kernel must be refused before any buffer is planned, with `unsupported_type_<T>`; setup must refuse the Unsloth family and hide Swift Q2_0 | `ctest:model_types_test`, `ctest:metal_plan_test`, `python:tools/test_setup_golden_mac.py` | [03](03-model-reference.md), section 2.2; [04](04-metal-kernels.md), section 2.5; [05](05-memory-and-residency.md), section 3.4; [07](07-server-setup-port.md), section 4.1 |
| REQ-MOD-07 | `--dump-logits`, `--dump-layers`, `--dump-halves` and `--dump-routing` must write upstream's formats, which upstream's tools read, and a dump must not change the output | `ctest:dump_formats_test`, `ctest:dump_eq_nodump` | [03](03-model-reference.md), section 9.1; [04](04-metal-kernels.md), section 9.4 |

---

## 5. Engine features (FUN)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-FUN-01 | The serve protocol must be byte-compatible with upstream: the requests `GEN`, `GENI`, `STOP` and `QUIT`; the lines `INFO`, `READY`, `RESUME`, `PP`, `REUSED`, `T`, `DONE` and `ERR`, in upstream's order and with its verbatim `ERR` texts. New `INFO` keys are additive. | `ctest:serve_transcript_test`, `python:serve/test_server.py`, `python:serve/test_lifecycle.py` | [06](06-engine-host-port.md), section 5; [07](07-server-setup-port.md), section 1.2 (P5) |
| REQ-FUN-02 | Every upstream flag must have a disposition of [06](06-engine-host-port.md), section 3.1 (kept, mapped, or ignored, with or without a warning), and every upstream `STRATA_*` variable must be honoured or ignored silently (section 4.1 there). A new flag or variable upstream must fail CI until it has one. | `ctest:options_dispositions_test`, `ctest:kernel_env_dispositions_test`, `ci:ci.yml/mac-host` | [06](06-engine-host-port.md), sections 3 and 4; [04](04-metal-kernels.md), section 9 |
| REQ-FUN-03 | The new flags must behave as specified: `--version`, `--list-devices`, `--plan-only`, `--storage-check`, `--geometry`, `--write-geometry`, the hidden `--plan-w-mib` and `--plan-ram-mib`, `--gpu-budget-mib`, `--expert-slots`, `--metallib` and `--metal-capture` | `ctest:options_dispositions_test`, `ctest:metal_cli_modes_test`, `ctest:strata_version`, `ctest:metal_geometry_plan_cli_test`, `ctest:metal_plan_test`, `ctest:metal_weight_store_cli_test`, `python:serve/test_macos.py` | [06](06-engine-host-port.md), section 3.10 |
| REQ-FUN-04 | The engine must hold contexts up to 262,144 cells, with every `--kv` mode (`fp16`, `int8`, `q4_0`, `k8v4`), wherever the plan admits them | `ctest:metal_qsa_test`, `ctest:metal_attn_test`, `accept:NEEDLE.262k` | [06](06-engine-host-port.md), section 3.3; [04](04-metal-kernels.md), sections 4.5 and 5.6-5.7 |
| REQ-FUN-05 | The RoPE and YaRN flags must be kept with upstream's meaning, and setup must write upstream's RoPE arguments for a context above 262,144. No Mac test runs a context above 262,144; until one does, the release notes say so ([11](11-risks-and-open-questions.md), OQ-12). | `ctest:options_dispositions_test`, `python:tools/test_setup_rope.py` | [03](03-model-reference.md), section 6.7; [07](07-server-setup-port.md), section 4.5.2 |
| REQ-FUN-06 | Greedy and sampled decoding must follow upstream's sampler: temperature, top-k, top-p, min-p and the Philox draw, with coupled drafting | `ctest:metal_sampler_parity_test`, `ctest:samp_coupled_eq_host`, `ctest:coupled_draft_test` | [04](04-metal-kernels.md), section 5.12; [06](06-engine-host-port.md), section 3.4 |
| REQ-FUN-07 | Speculative decoding must work as upstream's: MTP drafts, the suffix drafter, the draft policy and windows of T = 1..8 | `ctest:metal_mtp_test`, `ctest:metal_mtp_kv_store_test`, `ctest:mtp_kv_t_independent`, `ctest:suffix_drafter_test`, `ctest:draft_policy_test`, `ctest:controller_test`, `accept:C2.d.accept` | [06](06-engine-host-port.md), section 10; [04](04-metal-kernels.md), section 5.11 |
| REQ-FUN-08 | The prompt path (`--prefill`) must process a prompt in chunks on the GPU, and its result must not depend on the chunk size | `ctest:metal_prefill_test`, `ctest:prefill_chunk_independent`, `accept:PP.top1`, `accept:PP.chunk` | [04](04-metal-kernels.md), section 6; [05](05-memory-and-residency.md), section 2.4 |
| REQ-FUN-09 | The prompt cache must keep upstream's behaviour: checkpoints, resume selection, snapshots, parking under memory pressure and admission | `ctest:conversation_cache_test`, `ctest:conv_cache_test`, `accept:C3.reuse`, `accept:C3.exchange`, `accept:C3.admission` | [06](06-engine-host-port.md), section 11 |
| REQ-FUN-10 | Control vectors must apply as upstream's, and with none loaded the output must equal the stock engine's | `ctest:cvec_off_eq_stock`, `ctest:metal_gr_test` | [03](03-model-reference.md), section 6.12 |
| REQ-FUN-11 | Image input must work: `strata-vision` encodes on the GPU through Metal, and `GENI` takes its embeddings | `accept:S.image` | [06](06-engine-host-port.md), sections 5.4 and 5.10; [08](08-build-and-distribution.md), section 9 |
| REQ-FUN-12 | One-shot mode and `--stats` must print upstream's lines for every kept line, in upstream's format | `python:tools/metal/test_accept.py` | [06](06-engine-host-port.md), section 13.1 |

---

## 6. Memory (MEM)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-MEM-01 | The fit decision must exist once, in the engine: setup, `--check`, `start()` and the MCP read the `PLAN` line of `strata --plan-only` and compute no fit of their own (P3) | `ctest:metal_plan_test`, `ctest:platform_memory_test`, `python:serve/test_macos.py`, `python:tools/test_setup_golden_mac.py` | [05](05-memory-and-residency.md), section 3; [07](07-server-setup-port.md), sections 1.2 (P3) and 4.4 |
| REQ-MEM-02 | `--plan-only` must answer within 200 ms from a geometry file and within 1 s from real files, on the reference Mac, without building pipelines, reading tensor data or allocating buffers | `accept:PLAN.models`, `python:serve/test_macos.py` | [06](06-engine-host-port.md), section 2.3; [07](07-server-setup-port.md), section 14 (X-3) |
| REQ-MEM-03 | A plan from a geometry file must equal the plan from the real files, key for key | `python:serve/test_macos.py` | [07](07-server-setup-port.md), section 4.4.3 |
| REQ-MEM-04 | On the reference Mac, the plan of every file in [05](05-memory-and-residency.md), section 8.1, must match that table: in particular, every catalogue file in full mode except IQ3_S, and Q2_0 and Coder at 262,144 cells | `ctest:metal_plan_test`, `ctest:metal_mtp_bind_layout_test`, `accept:PLAN.models` | [05](05-memory-and-residency.md), sections 3 and 8.1 |
| REQ-MEM-05 | Where the model does not fit in full mode, tiered mode must run it: a resident set chosen by the expert profile, a slot pool fed from the SSD, adaptive swaps and lookahead | `ctest:metal_slot_pool_test`, `ctest:metal_tiered_copy_test`, `ctest:direct_file_async_test`, `ctest:metal_plan_test`, `ctest:metal_full_vs_tiered`, `accept:C4.swaps` | [05](05-memory-and-residency.md), section 5; [06](06-engine-host-port.md), section 9 |
| REQ-MEM-06 | Nothing that must not be resident may be wrapped as a Metal buffer: the PLE table, any GGUF shard in tiered mode, and any byte the plan does not count | `ctest:metal_never_wrap_test`, `ctest:metal_tiered_copy_test` | [05](05-memory-and-residency.md), section 6 |
| REQ-MEM-07 | The plan must fit W less the reserve, `--vram-reserve-mib` (default 700 MiB). The engine must never change `iogpu.wired_limit_mb`; setup may only print the advice. | `ctest:metal_plan_test`, `python:tools/test_setup_golden_mac.py` | [05](05-memory-and-residency.md), sections 3.1, 3.2 and 8.5; [07](07-server-setup-port.md), section 4.5.6 |
| REQ-MEM-08 | Full mode must wrap the GGUF and MTP model files in windows that respect `maxBufferLength` and the 16 KiB page; expert slabs, MTP expert blobs and dense tensors must remain intact and map to their exact GPU addresses | `ctest:metal_window_test`, `ctest:metal_window_storage_test`, `ctest:metal_fullmode_handoff_test`, `ctest:metal_fullmode_ple_handoff_test`, `ctest:metal_model_storage_owner_test`, `ctest:metal_mtp_storage_test` | [05](05-memory-and-residency.md), sections 4.1 and 4.2 |
| REQ-MEM-09 | `tools/hit_curve.py` must give the hit-rate curve of a resident set from a routing trace | `python:tools/test_hit_curve.py`, `python:tools/test_make_profile.py` | [05](05-memory-and-residency.md), section 10.1 |

---

## 7. Correctness (COR)

The thresholds of these criteria are in [09](09-testing-and-validation.md), section 4. Several are provisional until
a T measurement records them ([09](09-testing-and-validation.md), section 11). The requirement is the criterion, at
the value last set by the rule of [09](09-testing-and-validation.md), section 10.3.

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-COR-01 | The Mac must compute the same model as upstream's CUDA build, within the references' own spread (C1): teacher-forced logits, the router replay, and the first token after the prompt path. The CUDA build is out of project scope (D-IMPL-629), and per-kernel FP64 oracles replace it (D-IMPL-671): the kernels are verified by C1.K, the router on the model's own rows by C1.4, and the whole-model rows (C1.1, C1.5) are NOT RUN until a model-level reference is named. C1.2 and C1.3 are retired (D-IMPL-960). | `accept:C1.K`, `accept:C1.1.top1`, `accept:C1.1.kl`, `accept:C1.1.safe`, `accept:C1.1.band`, `accept:C1.1.ppl`, `accept:C1.4.ids`, `accept:C1.4.weights`, `accept:C1.5`, `ctest:metal_router_replay` | [09](09-testing-and-validation.md), section 4.1 |
| REQ-COR-02 | Every kernel must pass its twin against its CPU oracle, in its tolerance class, on synthetic inputs and on real model rows | `ctest:dequant_oracle_test`, `ctest:bf16_bits_test`, `ctest:metal_math_test`, `ctest:metal_embed_test`, `ctest:metal_ple_test`, `ctest:metal_gdn_test`, `ctest:metal_util_test`, `ctest:metal_twin_selftest`, `ctest:metal_embed_model_test`, `ctest:metal_ple_model_test`, `ctest:metal_gr_model_test`, `ctest:metal_gdn_model_test`, `ctest:metal_qsa_model_test`, `ctest:metal_attn_model_test`, `ctest:metal_mv_model_test`, `ctest:metal_moe_model_test`, `ctest:metal_mtp_model_test`, `ctest:metal_prefill_model_test` | [04](04-metal-kernels.md), section 10 |
| REQ-COR-03 | Greedy output on the ten `g1-*` prompts must match upstream's CUDA output over 256 tokens, as G1.tf and G1.free define "match". The CUDA output is out of project scope (D-IMPL-629): until a model-level reference is named, G1.tf and G1.free are NOT RUN (D-IMPL-671) | `accept:G1.tf`, `accept:G1.free` | [09](09-testing-and-validation.md), section 6.1 |
| REQ-COR-04 | Speculation and every Mac path must give the plain-decode tokens bit for bit (C2): every greedy arm equals arm (a), and each sampled arm equals the sampled `--spec 0` arm with the same seed | `accept:C2.b`, `accept:C2.c`, `accept:C2.d`, `accept:C2.e`, `accept:C2.f`, `accept:C2.g`, `accept:C2.h`, `accept:C2.i`, `accept:C2.j`, `ctest:window_eq_decode`, `ctest:state_after_commit_eq_decode`, `ctest:write_through_eq_commit` | [09](09-testing-and-validation.md), section 4.2; [06](06-engine-host-port.md), section 10.5 |
| REQ-COR-05 | The conversation cache must change nothing but the time (C3): upstream's six tools pass unchanged, a re-read equals a restore, and a repeated request gives one answer | `accept:C3.reuse`, `accept:C3.pressure`, `accept:C3.oversized`, `accept:C3.exchange`, `accept:C3.admission`, `accept:C3.growth`, `accept:C3.isolation`, `accept:C3.http`, `accept:C3.disabled`, `accept:C3.reread`, `accept:C3.repeat`, `python:tools/test_conversation_cache_parity.py` | [09](09-testing-and-validation.md), section 4.3 |
| REQ-COR-06 | The prompt path must stay within the prompt-path tolerance of upstream's CUDA build on a 32K prompt | `accept:PP.kl`, `accept:PP.ppl` | [09](09-testing-and-validation.md), section 4.5 |
| REQ-COR-07 | Every needle must be found at 32K, 128K and, on a Mac whose plan holds it, 262K | `accept:NEEDLE.32k`, `accept:NEEDLE.128k`, `accept:NEEDLE.262k` | [09](09-testing-and-validation.md), section 4.6 |
| REQ-COR-08 | Tiered mode must give the bits of full mode, in tokens and in every DONE counter but the tier counters, with and without adaptive swaps (C4) | `accept:C4.tiered`, `accept:C4.swaps`, `ctest:tiered_eq_full`, `ctest:metal_full_vs_tiered` | [09](09-testing-and-validation.md), section 4.4; [05](05-memory-and-residency.md), section 13 |
| REQ-COR-09 | Every kernel must be deterministic: no floating-point atomics, fixed orders, and the same bits for the same inputs across batch sizes, dispatch modes and the pipeline archive | `ctest:mv_multi_eq_single`, `ctest:ple_batch_eq_sequential`, `ctest:idx_batch_eq_sequential`, `ctest:pf_gdn_eq_step_norm`, `ctest:pf_sort_eq_host_grouping`, `ctest:metal_dispatch_modes_test`, `accept:C2.h` | [04](04-metal-kernels.md), sections 3.4 and 10.4; [02](02-architecture.md), section 8 (I11) |

---

## 8. Performance (PRF)

Each number is measured by the method of [09](09-testing-and-validation.md), section 5. Unless a row says otherwise,
that is on the reference Mac, with Q2_0 in full mode and the args setup writes: the median of three fresh processes
(of six when their spread is above 5%), after a warm-up, on AC power.

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-PRF-01 | Decode with MTP on `ca-4k` must reach at least 50 tok/s | `accept:BENCH.mtp` | [09](09-testing-and-validation.md), section 5.1 |
| REQ-PRF-02 | Plain decode (`--spec 0 --suffix-draft 0`) on `ca-4k` must reach at least 20 tok/s | `accept:BENCH.plain` | [09](09-testing-and-validation.md), section 5.1 |
| REQ-PRF-03 | Prefill of `ca-32k` must reach at least 450 tok/s | `accept:PF-2` | [09](09-testing-and-validation.md), section 5.2 |
| REQ-PRF-04 | Prefill of `ca-128k` must keep at least 70% of the `ca-32k` rate | `accept:PF-1` | [09](09-testing-and-validation.md), section 5.2 |
| REQ-PRF-05 | On a 48 GB Mac (or the reference Mac with `--gpu-budget-mib 37888` and a 16 GiB balloon), Q2_0 with MTP in tiered mode at an 8,192 context must decode `ca-4k` at 20 tok/s or more | `accept:G2.speed` | [09](09-testing-and-validation.md), section 6.2 |
| REQ-PRF-06 | A release must not be slower than the previous release on the same Mac: decode with MTP on `ca-1k`, `ca-4k` and `ca-32k`, and prefill on `ca-32k`, at least the bar of [09](09-testing-and-validation.md), section 5.4, times the previous release's. The bar is 0.95, or 1 − 2.5 × T-4's coefficient of variation when that is above 2%. A skipped or noisy result does not fail it. | `accept:REG.decode`, `accept:REG.prefill` | [09](09-testing-and-validation.md), section 5.4 |

---

## 9. Server and product (SRV)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-SRV-01 | The HTTP APIs must work on a Mac: models, status, metrics, OpenAI chat completions (streaming and not), tool calls, the Anthropic messages and token count, cancel, unload and load | `accept:S.models`, `accept:S.status`, `accept:S.openai`, `accept:S.stream`, `accept:S.tool`, `accept:S.anthropic`, `accept:S.cancel`, `accept:S.unload` | [09](09-testing-and-validation.md), section 4.7 |
| REQ-SRV-02 | Setup must run the Mac step order: check the Mac, the Python packages, the engine, the choices, the download, the pack and the start script | `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), section 4.3 |
| REQ-SRV-03 | The model and context menus must show the engine's plan class for each entry, recommend by the rules of [07](07-server-setup-port.md), section 4.5, and refuse an entry that does not fit | `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), section 4.5 |
| REQ-SRV-04 | The config and args setup writes on a Mac must be the Mac's: `backend` `metal`, no GPU choice, no PC-only flag, and a `plan` object. Calibration is neither offered nor run. | `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), sections 4.9 and 4.16 |
| REQ-SRV-05 | Setup must install the release asset whose tag matches the checkout, check BUILD.json, strip quarantine and pass the smoke test (`--version`, `--list-devices`); `--build` must build the same engine locally with Xcode 27 | `python:tools/test_metal_build.py`, `ci:ci.yml/mac-host` | [07](07-server-setup-port.md), section 4.7 |
| REQ-SRV-06 | The run scripts and `.command` launchers must start the model from any folder name, and a double-click in Finder must work | `python:serve/test_macos.py`, `review:S20` | [07](07-server-setup-port.md), sections 4.10 and 5.2 |
| REQ-SRV-07 | The data folder must default to `~/Strata-data` when the checkout is in a protected or synced folder, and be excluded from Spotlight and Time Machine | `python:serve/test_macos.py` | [07](07-server-setup-port.md), section 4.12 |
| REQ-SRV-08 | The server must probe its port as specified, shut down cleanly on SIGHUP (closing the Terminal window) and SIGINT, explain why an engine died with the Mac texts, and keep the Mac awake while a model is loaded when `keep_awake` is true | `python:serve/test_macos.py` | [07](07-server-setup-port.md), sections 4.11, 6.1, 6.2, 6.5 and 6.9 |
| REQ-SRV-09 | `/metrics`, `/v1/status` and the Monitor and About tabs must show the Mac's readings: GPU, unified memory, the plan, the tiers. Off macOS their keys stay upstream's. | `accept:S.status`, `python:serve/test_monitor.py`, `review:S14` | [07](07-server-setup-port.md), section 7 |
| REQ-SRV-10 | The MCP server and hub must work on a Mac: `strata_models` returns engine plans, install, start and stop work, and upstream's MCP tests pass with no change | `python:tools/test_strata_mcp.py`, `python:serve/test_mcp.py`, `python:tools/test_setup_golden_mac.py` | [07](07-server-setup-port.md), section 8 |
| REQ-SRV-11 | Withdrawn 2026-10-07: the inherited PC CUDA calibration tool is not part of the macOS product | `withdrawn` | D-IMPL-236; [07](07-server-setup-port.md), section 9 |
| REQ-SRV-12 | Every retained Python test module must pass on the supported Mac target | The Python modules listed in [09](09-testing-and-validation.md), section 9.3 | [07](07-server-setup-port.md), section 10.1 |


---

## 10. Robustness (ROB)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-ROB-01 | No command buffer may hit the GPU's time limit: every command buffer below half of the limit L-12 records, and a T = 8 window below 1 s | `ctest:metal_gpu_time_test` | [09](09-testing-and-validation.md), section 5.5; [04](04-metal-kernels.md), section 11 (L-12) |
| REQ-ROB-02 | A stalled window must end the engine within 20 s + 5 s, in full and in tiered mode, with `ERR gpu error: window timeout` and exit 1. A request with no progress for `STRATA_WATCHDOG_S` must end the engine with the issue #29 line and the stall report, and macOS keeps the crash report. | `ctest:metal_watchdog_test` | [06](06-engine-host-port.md), sections 14.2-14.4 |
| REQ-ROB-03 | A GPU command-buffer error must give `ERR gpu error: <description>` in serve mode and exit 1; an allocation failure must exit 1 with the plan printed | `ctest:serve_transcript_test`, `ctest:metal_alloc_fail_test`, `ctest:metal_fault_test` | [06](06-engine-host-port.md), sections 2.3 and 2.4 |
| REQ-ROB-04 | Decoding must go on while another process fills memory, with no GPU error and no engine exit | `accept:PRESSURE.smoke` | [05](05-memory-and-residency.md), sections 4.4 and 10 (M-4) |
| REQ-ROB-05 | The engine must survive upstream's conversation-cache soak at a 131,072 context | `accept:C3.soak` | [09](09-testing-and-validation.md), section 4.3 |
| REQ-ROB-06 | In serve mode the engine must ignore SIGINT and SIGHUP, so that the server ends it with `QUIT`; stdin EOF must end it with exit 0 | `ctest:serve_transcript_test`, `python:serve/test_macos.py` | [06](06-engine-host-port.md), section 2.4; [07](07-server-setup-port.md), section 6.2 |

---

## 11. Build and release (BLD)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-BLD-01 | The release asset `strata-macos-arm64.zip` must hold exactly `strata`, `strata.metallib`, `strata-vision`, `BUILD.json` and `LICENSES.txt`, and pass the twelve checks of `check_asset.py` | `python:tools/test_metal_build.py`, `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), sections 11.2 and 11.7 |
| REQ-BLD-02 | The engine must run only with its own kernel library: the SHA-256 of `strata.metallib` is compiled into `strata`, and every manifest kernel is in the library | `ctest:metal_context_test`, `ctest:strata_version`, `ctest:metal_manifest_test` | [08](08-build-and-distribution.md), sections 4.5, 5.2 and 5.6 |
| REQ-BLD-03 | The executables must link only system libraries under `/System/Library/` and `/usr/lib/`, with no `LC_RPATH`, and must contain no path of the build machine | `python:tools/test_metal_build.py`, `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), section 11.7 (checks 4 and 6) |
| REQ-BLD-04 | The same commit, built with the same Xcode and Metal toolchain, must give the same members of the asset, in any folder and on any Mac | `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), section 11.4 |
| REQ-BLD-05 | Releases must be signed ad hoc by default, or with Developer ID and the hardened runtime when the owner sets it up, and never with an entitlement | `python:tools/test_metal_build.py`, `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), section 11.5 |
| REQ-BLD-06 | `LICENSES.txt` must hold the notice of every component in the asset | `review:B20` | [08](08-build-and-distribution.md), section 11.9 |
| REQ-BLD-07 | Releases must follow the policy of [08](08-build-and-distribution.md), section 11.1: the tag `v{version}-metal.{metal_rev}`, `METAL_REV` only growing, and main always pointing at a published release | `ci:ci.yml/mac-host`, `ci:ci.yml/mac-host` | [08](08-build-and-distribution.md), section 11.1 |
| REQ-BLD-08 | The build must refuse every non-Darwin target before compiler detection and configure only arm64 macOS with Metal enabled; CUDA and HIP must remain disabled in the CPU oracle build | `python:tools/test_metal_build.py`, `ci:mac-hardware.yml/mac-host` | [08](08-build-and-distribution.md), section 3.4; [02](02-architecture.md), section 8 (I10) |
| REQ-BLD-09 | The self-hosted runner must keep the rules of [08](08-build-and-distribution.md), section 12.4: no pull-request trigger, a dedicated non-admin user, read-only models | `review:B21` | [08](08-build-and-distribution.md), section 12.4 |

---

## 12. Maintainability (MNT)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-MNT-01 | Every edit in an upstream file must carry a `// MAC:` or `# MAC:` marker, and every copied range its `synced from` header; `check_drift.py` must fail on each kind of change of [10](10-implementation-plan.md), section 4.2: a marker count, a copied range, a new flag, variable, INFO key, ERR string or DONE field, a Python change next to a marker or a new platform branch, a changed or new CUDA kernel, and a new model or pinned revision | `python:tools/metal/test_check_drift.py`, `ci:ci.yml/mac-host` | [02](02-architecture.md), section 9; [06](06-engine-host-port.md), section 1.3 |
| REQ-MNT-02 | Every test must be in the test map, with its label, and every requirement here must name at least one check that exists | `ctest:test_map`, `python:tools/metal/test_map.py` | [09](09-testing-and-validation.md), section 9.4 |
| REQ-MNT-03 | The benchmark prompt sets must be built by `make_prompts.py` and match their manifest | `ctest:prompts_manifest_test`, `python:tools/metal/test_make_prompts.py` | [09](09-testing-and-validation.md), section 2.2 |
| REQ-MNT-04 | The acceptance tools must have their own tests, each criterion passing on equal inputs and failing on its planted difference | `python:tools/metal/test_accept.py`, `python:tools/metal/test_tf_compare.py`, `python:tools/metal/test_ladder_compare.py` | [09](09-testing-and-validation.md), sections 3.2 and 3.3 |
| REQ-MNT-05 | The geometry files must match the model files' published sizes | `ci:ci.yml/mac-host` | [07](07-server-setup-port.md), section 4.4.3 |

---

## 13. User documents (DOC)

| ID | Requirement | Verified by | Specified in |
| --- | --- | --- | --- |
| REQ-DOC-01 | User-facing repository documents must describe only the supported Apple Silicon macOS target and must identify setup as in progress until the native installer passes its acceptance checks | `review:documentation` | [07](07-server-setup-port.md), section 11 |
| REQ-DOC-02 | Product documentation must not recommend Linux or Windows setup, Docker deployment, or CUDA/HIP configuration | `review:documentation` | [07](07-server-setup-port.md), sections 1 and 11 |
| REQ-DOC-03 | The macOS setup guide must state the verified macOS, Apple Silicon, Python and memory requirements before claiming installation is ready | `review:documentation` | [07](07-server-setup-port.md), section 11 |

---

## 14. Targets, reported but not required

These are measured and published with every release, but a miss does not fail G3.

| Criterion | Target | Why it is not a requirement |
| --- | --- | --- |
| `BENCH.mtp.target` | Decode with MTP on `ca-4k` at least 60 tok/s on the reference Mac | It is the top of the 45-60 tok/s projected for an engine of DwarfStar's quality on an M5 Pro, the reason the port exists ([00](00-feasibility-study.md), section 3.3). REQ-PRF-01 is the floor. |
| `BENCH.other` | Every other model, prompt and arm | No upstream number is a reference ([09](09-testing-and-validation.md), section 2.6); the regression check compares them |
| `REG.trend` | No change of more than 3% night to night | A trend, not a gate ([09](09-testing-and-validation.md), section 5.4) |
| Time to first token | Reported with PF-2 | Prefill speed covers it |
| `C1.1.p99` | p99 KL within 2 × the floor's of the model-level reference (NOT RUN until one is named) | A tail statistic, reported beside the gating median ([09](09-testing-and-validation.md), section 4.1) |

---

## 15. Where the details are

| Topic | Document |
| --- | --- |
| The decision and the gates | [00-feasibility-study.md](00-feasibility-study.md), sections 5 and 7 |
| The design and its invariants | [02-architecture.md](02-architecture.md), sections 7 and 8 |
| The model, the files and the formats | [03-model-reference.md](03-model-reference.md) |
| The kernels and their tests | [04-metal-kernels.md](04-metal-kernels.md), sections 2 and 10 |
| The memory plan | [05-memory-and-residency.md](05-memory-and-residency.md), sections 3-8 |
| The engine host, the protocol and the flags | [06-engine-host-port.md](06-engine-host-port.md), sections 2-5 |
| Setup, the server and the user documents | [07-server-setup-port.md](07-server-setup-port.md) |
| The build, the release and CI | [08-build-and-distribution.md](08-build-and-distribution.md), sections 11 and 12 |
| Every check named here | [09-testing-and-validation.md](09-testing-and-validation.md), sections 3-9 |
| The work, phase by phase | [10-implementation-plan.md](10-implementation-plan.md) |
| What could stop a requirement | [11-risks-and-open-questions.md](11-risks-and-open-questions.md) |
