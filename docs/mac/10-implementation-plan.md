# 10. Implementation plan

Status: implementation-ready specification, updated 2026-10-09.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- the work items of [03](03-model-reference.md) (R1-R12), [04](04-metal-kernels.md) (L1-L23),
  [05](05-memory-and-residency.md) (U1-U10), [06](06-engine-host-port.md) (H1-H13),
  [07](07-server-setup-port.md) (S1-S21), [08](08-build-and-distribution.md) (B1-B22) and
  [09](09-testing-and-validation.md) (T1-T19);
- the gates of [00-feasibility-study.md](00-feasibility-study.md), section 7, as
  [09](09-testing-and-validation.md), section 6, makes them exact;
- the effort estimate of [00-feasibility-study.md](00-feasibility-study.md), section 3.4.

This document orders the work. Every work item of documents 03-09 is one task here, in exactly one phase. It covers:
- how the plan is organised, and the rule for a test whose work item is not done yet;
- the seven phases, 0-6, each with its tasks, measurements and exit criteria, and the three gates;
- upstream compatibility reviews;
- how a release is made, and the first release;
- the schedule of every measurement, with the default that stands until it is recorded;
- the machines and accounts the work needs.

Other documents cover the rest:
- what each task must do, and its "Done when": the document its ID names;
- what the finished port must do: [01-requirements.md](01-requirements.md);
- what could go wrong, and the decisions still open: [11-risks-and-open-questions.md](11-risks-and-open-questions.md).

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. How the plan works

### 1.1 Tasks, phases and gates

- **A task is a work item.** Its ID names the document that specifies it: R ([03](03-model-reference.md),
  section 11), L ([04](04-metal-kernels.md), section 12), U ([05](05-memory-and-residency.md), section 14), H
  ([06](06-engine-host-port.md), section 16), S ([07](07-server-setup-port.md), section 15), B
  ([08](08-build-and-distribution.md), section 15) and T ([09](09-testing-and-validation.md), section 12). Its
  "Done when" is in that table, and this document does not restate it.
- **A task is done** when its "Done when" holds on the reference Mac and its
  pull request is merged to main with the result in the pull request. A task that names another ("with H10") is done
  when both are.
- **A task is listed in the phase in which it must be done.** It may start earlier, once the tasks it comes after
  are done. Where an earlier phase needs part of a task, that part is listed there as "(part)", with what the part
  is, and the task is done in the later phase.
- **A phase ends** when every task listed in it is done and its exit criteria (section 3) hold.
- **A gate** is a run of `accept.py` at the end of phases 2, 4 and 6 ([09](09-testing-and-validation.md),
  section 6). Its result and the decision go into `docs/mac/gates/G<n>.md`. Phase 3 does not start until G1 passes
  ([09](09-testing-and-validation.md), section 6.1). G2's failure changes the plan of phases 5 and 6, not their
  order (section 3.5).
- **The conditional work items** L16-L23 are not in any phase. Each starts when its condition holds (section 3.8).
- **"Phase 1" and "Phase 2" in [04](04-metal-kernels.md)** name the kernel set of the first release and the
  conditional items after it (section 12.2 there), not phases of this plan. Here, the phases are always written with a
  lowercase "phase" and a number from 0 to 6.

### 1.2 Estimates

The figures are planning figures for one senior engineer who knows GPU programming, the assumption of
[00](00-feasibility-study.md), section 3.4. They add up to 29 weeks, inside its 5-7 months. A second engineer
shortens phases 1, 3 and 5 the most, because their tasks are in independent tracks (section 2). The estimate of the
next phase is redone at each gate, from the measured pace of the phases before it.

### 1.3 Tests and checks that do not exist yet

[09](09-testing-and-validation.md), section 9, lists every test the documents define, and `test_map.py` checks that
CTest, `tests/metal/test_map.json`, that list and `tests/metal/requirements.json` agree
([09](09-testing-and-validation.md), section 9.4). During the phases, most tests do not exist yet. So:
- **A test is in the map from phase 0, and in CTest from its task.** Until its task registers it, its entry in
  `test_map.json` carries `"planned": "<task ID>"`, for example
  `"tiered_eq_full": {"label": "model", "origin": "04 10.4", "planned": "L12"}`. The commit that registers the test
  in CMake removes the key.
- **A requirement gets its checks as they come to exist.** `requirements.json` lists every requirement ID of
  [01-requirements.md](01-requirements.md) from phase 0, at first with an empty list. A check is added in the commit
  that creates it: a `ctest:` check may name a planned test, an `accept:` check needs its criterion in `criteria.py`,
  and a `python:` check needs its module.
- **`test_map.py` reports what is missing, and `--final` fails on it.** Without `--final`, a planned test and a
  requirement with no check each print a note. `accept.py g3` runs `test_map.py --final`, which fails on both. So G3
  cannot pass while anything is planned.
- **A gate runs what is registered.** G1's `ctest --preset mac-model` runs the model tests registered by the end of
  phase 2 (section 3.3 lists them), and G2 those of phase 4.
- **The nightly run is red until its parts exist.** `ctest --preset mac-model` now includes `dense_vs_gguf_test`
  and fails closed until `STRATA_TEST_MODELS` contains all seven completed catalogue packs. The broader nightly
  acceptance step still needs `accept.py` (T4) and the runs of T14. Until T14 is done (phase 3), a nightly failure
  caused by a missing test or tool is expected and is not triaged ([09](09-testing-and-validation.md), section 10).
  The push runs must be green from phase 0 on.

---

## 2. The phases at a glance

| Phase | Goal | Ends with | Weeks | Main tasks |
| --- | --- | --- | ---: | --- |
| 0 | The project repository, Metal build and CI are ready, with the source reference recorded | Exit criteria (section 3.1) | 2 | B1-B5, B14-B17, B19, B21, H1, H2, H12, R1, S17, T3, T17 |
| 1 | The engine without a model: host, loaders, the plan and every kernel against its oracle | Exit criteria (section 3.2) | 7 | H3-H5, B6, B7, U1, U2, R2-R10, R12, T1, T2, T4, T5, L1-L8, S1, S4 |
| 2 | First token, windows, MTP and the prompt path, in full mode | **G1** | 6 | U3, U4, L9-L11, L13, L14, H6-H8, H11, R11, T6, T8, T11; parts of T7 and T13 |
| 3 | The Mac product on a 64 GB Mac: setup, server, prompt cache, vision | Exit criteria (section 3.4) | 4 | H9, H13, B8-B11, S2, S3, S5-S13, S15, S16, S18, S19, T9, T12, T14, T18 |
| 4 | Tiered mode: the SSD tier for 32-48 GB Macs | **G2** | 4 | U5-U9, L12, H10, T10, T13; part of T7 |
| 5 | Release engineering, and the M1-M4 families | Exit criteria (section 3.6) | 3 | B12, B13, B18, B20, T15, T16, S14 |
| 6 | Every catalogue file, the user documents, the measurements closed | **G3** and the first release | 3 | S20, S21, L15, U10, B22, T7, T19 |

The tracks that can run side by side, by a second person or a coding agent:
- **Kernels** (L) need only the twin harness (L2) and the oracles until L9.
- **Setup and server** (S) need a fake engine and the plan's PLAN line (S4). S1-S6 can start in phase 1, and the
  Mac golden set (S18) can be recorded once `--plan-only` works (H5, U1). The geometry files of S5 can be written
  then too; S5 is done when the nightly parity run of T18 passes.
- **Build and release** (B) after phase 0 depend only on a buildable engine.
- **Test tools** (T) of phases 1 and 2 run on macOS against `fake_engine.py`
  ([09](09-testing-and-validation.md), section 3.2).

---

## 3. The phases

Each table lists the tasks of the phase in a workable order. "After" names the tasks that must be done first, other
than those of earlier phases.

### 3.1 Phase 0: the repository, the build and CI

The working tree already contains the project docs, source baseline reference, build skeleton and local test maps.
Phase 0 still requires:
- **Set the publication identity** when the repository is hosted: configure `{REPOSITORY}` and
  `setup.RELEASE_REPOSITORY` consistently with the checked-out GitHub repository. Until then, keep release-history
  checks local and do not publish release artifacts.
- **Keep specifications beside the implementation.** `test_map.py` reads
  `docs/mac/01-requirements.md` and `docs/mac/09-testing-and-validation.md`; work items take their reference inputs
  from `docs/mac/spikes/`. Changes to a specification land with the code or test they describe.
- **Keep the provenance reference** in `UPSTREAM_SYNC` and the Metal release revision in `METAL_REV` (`1`). The
  upstream SHA is a source and compatibility reference, not a parent to merge into the project history.
- **Prepare the machines** of section 8: the reference Mac as the runner `strata-ref`, with the model disk; the CUDA
  reference PC.

| Task | What | After |
| --- | --- | --- |
| B1 | Root edits A-G, `direct_file.cpp`, `.gitattributes`, `.gitignore` | |
| B2 | `cmake/metal/backend.cmake` | B1 |
| B3 | The metallib build and the manifest, at first with the `strata_selftest_*` kernels only | B2 |
| B4 | `tests.cmake`, `check_engine.cmake` | B2 |
| B5 | `CMakePresets.json`, `tools/test_metal_build.py` | B2 |
| H1 | Done when B1-B5 are | B1-B5 |
| H2 | The platform patches, with H-9 | B1 |
| B14 | `run_python_tests.py` | |
| S17 | The upstream test fixes: `normalize()`, `test_prebuilt_hip_zip`, the 13 `setUpModule` lines | B14 |
| H12 | `check_drift.py` and `drift.json` (section 4.3) | B1 |
| T17 | `test_map.py` with `--final`; `test_map.json` with every test of [09](09-testing-and-validation.md), section 9.1, those not yet in CTest planned; `requirements.json` with every requirement ID and no check yet; an empty `reviews.json` | B4 |
| B16 | `ci.yml` | B14, H12, T17 |
| B17 | `mac-hardware.yml`, the push and nightly host and metal parts | B4, B5 |
| B19 | Action pinning, `dependabot.yml` | B16, B17 |
| B21 | The `strata-ref` runner rules, checked by a second person before the first workflow runs on it | B17 |
| R1 | The header fixtures from `spikes/model-headers/` | |
| T3 | `models.json`, `test_models.py`, the model disk of the reference Mac | |

Measurements: X-8 and B-3, if a hosted macOS 27 image exists; H-9 with H2; H-6 from the first compatibility review onward.

Exit criteria:
- The retained macOS checks are green on the configured Mac runner; the workflow does not build or test Windows/Linux product targets.
- A push to main runs `mac-hardware.yml` green on `strata-ref`: `ctest --preset mac-host` runs the 19 host tests
  of [08](08-build-and-distribution.md), section 6.4, and `ctest --preset mac-metal` runs the six registered
  bootstrap Metal tests, including `metal_context_test` and the explicit `metal_self_test` CLI path.
- `check_drift.py` passes at `UPSTREAM_SYNC`. `test_map.py` passes, with notes for every planned test and every
  requirement.
- The rules of [08](08-build-and-distribution.md), section 12.4, hold on `strata-ref` (B21).

### 3.2 Phase 1: the engine without a model

The engine starts, reads every catalogue file, plans it, and every kernel passes its twin. Nothing decodes yet.

| Task | What | After |
| --- | --- | --- |
| H3 + B6 | `Context`, `Buffers`, `Events`; the pipeline cache | H1 |
| B7 | GPU capture | H3 |
| H4 | `options.cpp` and every disposition | H1 |
| R2-R4, R6-R8, R10 | The loader checks on the fixtures of R1 | R1 |
| H5 | `WeightStore`, the light `--plan-only` and `--write-geometry` path | H4, R2-R8 |
| U1 | `Geometry` and `Plan`: the PLAN line | H5 |
| U2 | Window planning | U1 |
| R5 | The `dense.bin` compare, with R-4 | H5 |
| R9 | The PLE oracle on every file | H5 |
| L1, L2 | `common/` headers; the twin harness and oracles | H3 |
| L3-L8 | Mat-vecs, embedding, PLE, GR, GDN, QSA, attention, MoE, head and sampler: each against its twin | L1, L2 |
| T1 | The prompt sets | |
| R12 + T2 | The CUDA golden set and `record_golden.py`, with T-7 | T1, T3 |
| T4 | The `accept.py` framework | |
| T5 | `tf_compare.py`, `lc_dump` | T4 |
| S1 | `serve/macos.py` | |
| S4 | `metal_plan`, `plan_class` | U1 |

Measurements: R-4; L-6 (checks 1-3 and 5), L-8, L-9 and L-10 on the reference Mac; M-1; X-3; T-1, T-2 and T-3
once the golden set exists; T-7 with the golden set.

Exit criteria:
- `ctest --preset mac-metal` passes, with the twins of L1-L8 registered ([04](04-metal-kernels.md), section 10).
- `ctest --preset mac-model` passes, with the model tests of R5, R9 and R10 registered.
- `strata --plan-only` prints a PLAN line for each of the seven catalogue files, within the time limits of
  [06](06-engine-host-port.md), section 2.3, and the plan of [05](05-memory-and-residency.md), section 8.1, on the
  reference Mac.
- The golden set of Q2_0 and Coder is on the reference Mac, with its manifest in git.
- `tools/metal/test_accept.py` passes on the reference Mac with its documented fixture inputs.

### 3.3 Phase 2: first token, speculation and the prompt path

Q2_0 decodes on the reference Mac, in full mode, with every path G1 checks: T = 1, windows of T = 2..8, MTP, the
suffix drafter, sampling and the prompt path.

| Task | What | After |
| --- | --- | --- |
| U3 | Full mode: file windows, address and resident tables, residency set, warm-up; with M-5 | U2 |
| U4 | The never-wrap rules | U3 |
| H6 | The serve loop and protocol, first with a fake `Window` | H4 |
| H7 | One-shot mode and `--stats` | H6 |
| L9 | The T = 1 window and its encode, with the dump hooks | L3-L8, U3, D-IMPL-177 (fixed-address WindowParams ring), D-IMPL-178 (generic Metal Encoder), D-IMPL-180 (session-state owner), D-IMPL-181 (named full-mode dense tensor views), D-IMPL-182 (exact T=8 Window scratch layout), D-IMPL-183 (Context-owned Window scratch storage), D-IMPL-184 (ModelStorage and planner composition), D-IMPL-185 (validated Window PLE row indices), D-IMPL-186 (full-mode PLE shared-mapping reader), D-IMPL-187 (ModelStorage PLE reader composition), D-IMPL-188 (Window PLE row staging), D-IMPL-261 (ordered GR read composition), D-IMPL-262 (layer-0 write and PLE entry composition), D-IMPL-263 (ordered Window GDN block composition), D-IMPL-265 (full-mode tensor-view access through ModelStorage), D-IMPL-266 (read-only Encoder input binding), D-IMPL-267 (T=1 Window input stem), D-IMPL-268 (T=1 layer-0 attention/GDN coordinator), D-IMPL-269 (layer-0 attention/GDN/FFN GR chain), D-IMPL-270 (validated MoE route dispatch), D-IMPL-271 (read-only BF16 mat-vec inputs), D-IMPL-272 (T=1 router→route→Q8_1 prefix), D-IMPL-285 (routed expert FFN composition), D-IMPL-289 (routed/shared FFN and GR write composition), D-IMPL-290 (full-mode T=1 Window MoE layer composition), D-IMPL-291 (preserve the layer-0 pending-write policy; tiered scheduling remains open), D-IMPL-294 (same-segment FullMode-to-no-PLE handoff test), D-IMPL-295 (same-segment FullMode-to-PLE handoff test), D-IMPL-296 (T=1 typed pending-consumer selector), D-IMPL-297 (FullMode T=1 token-to-consumer composition; production Window wiring and model parity remain pending), D-IMPL-298 (standalone transactional CommandBuffer wrapper for the FullMode T=1 step), D-IMPL-299 (read-only FullMode PLE model-weight binding in the T=1 branch), D-IMPL-300 (FullMode PLE native key-format handoff coverage), D-IMPL-301 (committed PLE token suffix and accepted-prefix state), D-IMPL-302 (bind PLE token history to the SessionStateStorage reset lifecycle), D-IMPL-303 (prepare T=1 parameters and session-owned PLE input rows), D-IMPL-304 (ordered GR head-mixer composition), D-IMPL-305 (read-only dense mat-vec for output-weight formats), D-IMPL-306 (FullMode output.weight resolution and Q8_1-to-logits projection), D-IMPL-307 (FullMode output_hc tensor resolution and mixer→logits composition), D-IMPL-308 (FullMode output head→GPU top-k→CPU tail composition; production Window wiring, accepted-prefix scheduling and model parity remain open), D-IMPL-411 (synchronous accepted-prefix GPU state commit with success-only session PLE suffix advancement; production Window wiring remains open) |
| R11 | The dump flags and formats | L9 |
| L10 + H8 | Windows of T = 2..8, commit and write-through; the spec round | L9, D-IMPL-264 (per-layer accepted-prefix GDN commit), D-IMPL-309 (all-layer GDN batch), D-IMPL-310 (GDN/PLE composition), D-IMPL-311 (12-layer QSA indexer commit batch), D-IMPL-312 (GDN/PLE/QSA full state-commit transaction), D-IMPL-313 (validated per-layer GDN write-through block; model-wide scheduler remains open), D-IMPL-314 (speculative layer-0 attention/GDN/FFN-read coordinators for T=1..8; full Window scheduler remains open), D-IMPL-315 (speculative layer-0 router/route/Q8_1 prefix for T=1..8), D-IMPL-316 (generic speculative router/route/plan prefix for T=1..8; expert scheduling remains open), D-IMPL-317 (generic 1..8 pending-MoE consumer selector), D-IMPL-318 (generic FullMode layer-0 MoE producer for T=1..8), D-IMPL-319 (compose the generic FullMode producer with the typed PLE/no-PLE pending consumer for T=1..8, including speculative history/alias checks; token gathering, GR stream initialization, production Window scheduling and model parity remain open), D-IMPL-320 (compose speculative T=1..8 token gather and GR broadcast with the FullMode producer/consumer; proposal preparation, accepted-prefix scheduling, production Window integration and model parity remain open), D-IMPL-321 (prepare T=1..8 speculative WindowParams and optional committed-history PLE rows; proposal generation and production integration remain open) |
| L11 | The MTP drafter; isolated encoders and runtime view binding for stages 1–22 are implemented, and the synthetic runtime chain reaches stage 20; the full item remains open | L10, D-IMPL-174 (storage-owner composition), D-IMPL-175 (main KV/RoPE allocation), D-IMPL-176 (bind MTP storage to the main RoPE view), D-IMPL-322/323/334/335/336/337/338/339/340/341/342/343/344/345/346 (validated stages 1–22 and runtime binding; production scheduler integration and full drafter parity remain open; REV-136/137/138/139/140/141/142/143 record hostile reviews) |
| L13 | The prompt path | L9 |
| L14 | The kernel dispositions, and `metal_dispatch_modes_test` | L10 |
| H11 | The watchdog and the stall report | H6, L10 |
| T6 | C1: `ladder_compare.py`, `metal_router_replay`, `metal_layer_replay` | T5, L9 |
| T8 | C2, every arm | L11, H8 |
| T11 | The benchmark, PF-1, PF-2 and the regression check | T4, L13 |
| T13 (part) | `metal_gpu_time_test` and `metal_watchdog_test`, with their full-mode cases; the tiered cases come in phase 4 | L13, H11 |
| T7 (part) | `accept.py g1` and `docs/mac/gates/` | T6, T8, T11 |

The model tests registered at G1: `dense_vs_gguf_test`, `ple_oracle_model_test`, `mtp_reader_test`,
`dump_formats_test`, the ten `metal_<area>_model_test`, `window_eq_decode`, `state_after_commit_eq_decode`,
`write_through_eq_commit`, `cvec_off_eq_stock`, `mtp_kv_t_independent`, `dump_eq_nodump`,
`metal_dispatch_modes_test`, `prefill_chunk_independent`, `metal_router_replay`, and `metal_gpu_time_test` and
`metal_watchdog_test` with their full-mode cases: 25 of the 28. `tiered_eq_full`, `metal_full_vs_tiered` and
`metal_alloc_fail_test` stay planned until phase 4.

Measurements: R-1 and R-5 (G1 step 7); R-2; L-2, L-4, L-12, L-13, L-14 and L-15; L-6 check 4; H-4; H-7; M-5;
M-6; T-4 and T-5; B-1 and B-10.

**G1**, `accept.py g1` ([09](09-testing-and-validation.md), section 6.1). If it fails on correctness, the bisection
of [09](09-testing-and-validation.md), section 10.1, comes first, and phase 3 waits. Between 15 and 20 tok/s, the
profile (H-4, L-13, L-14) and its fixes come first, which may start conditional items (section 3.8), and G1 runs
again. Below 15 tok/s with no clear cause, the stop rule of [00](00-feasibility-study.md), section 7, applies.

### 3.4 Phase 3: the Mac product on a 64 GB Mac

A person with a 64 GB Mac can install Strata with `./setup.sh`, run it, and use every API and the Monitor.

| Task | What | After |
| --- | --- | --- |
| H9 | The prompt cache, resume, snapshots, parking and memory pressure | H8 |
| T9 | C3, with T-9 | H9 |
| H13 + B9 | `strata-vision` on a Mac | H1 |
| S2 | `setup.sh` and the launchers | S1 |
| S3 | Setup's constants, probes and step 1 | S1 |
| S6 | Menus, consent, `--check`, `--low-ram`, `--resident-budget-gib`, the wired-limit advice | S4 |
| B10, B11 | `make_release.py`, `check_asset.py` | B3, B9 |
| S5 | The geometry files; done when T18's nightly parity run passes | S4 |
| S7 | Engine install from a release asset; until a release exists, an asset from `make_release.py` served by `python3 -m http.server` and named by `STRATA_PREBUILT_URL`, as step 5 of `verify_release.py` does | B10, B11 |
| S8 + B8 | `build_engine_metal`, the build definitions; with B-9 | B5 |
| S9 | Steps 6 and 7: pack, MTP, disk, args, the final plan, the config | S6, S7 |
| S10 | Run scripts, `.command` files, quarantine | S9 |
| S11 | `start()` on a Mac, the data folder, sleep, flags and texts | S9 |
| S12 | `serve/server.py`: port probe, SIGHUP, the death texts, `keep_awake` | S1 |
| S13 | `serve/telemetry.py` | S1 |
| S15 | `tools/strata_mcp.py` | S4 |
| S16 | The tools | |
| S18 | The Mac golden set and `record_mac_plans.py` | S6, S9 |
| S19 | `serve/test_macos.py` | S2, S10-S12 |
| T12 | The needles and the server run, with S.image | H9, H13 |
| T14 | The nightly job's five steps and the weekly job | T7-T9, T11, T12 |
| T18 | The Python layer in the nightly run | T14, S5 |

Measurements: R-3; M-3; H-8; X-1, X-2 and X-4 to X-7; B-4 and B-9; T-6 over the two weeks after T14.

Exit criteria:
- On the reference Mac, a fresh checkout runs `./setup.sh`, installs Q2_0 with the engine asset of S7, and starts
  the server from `run-<model>.command`.
- `accept.py server` and `accept.py c3` pass on Q2_0. `accept.py nightly` passes.
- The Mac golden set passes on the reference Mac.

### 3.5 Phase 4: tiered mode

| Task | What | After |
| --- | --- | --- |
| U5 | The tiered layout: dense and resident arenas, the descriptors, `res`, `addr`, `plan` | U3 |
| U6 + H10 | The slot pool and slot cache, miss reads, prefill prefetch, lookahead, DONE accounting; `ExpertStore` | U5 |
| L12 | The tiered encoding: per-layer signal and wait, the miss table | U6 |
| U7 | Adaptive swaps | L12 |
| U8 | macOS interplay: the memory-pressure source with H9, `STRATA_ARENA_LOCK`, the RAM warnings | U6 |
| U9 | `tools/hit_curve.py` | U6 |
| T10 | C4 | L12, U7 |
| T13 | `metal_gpu_time_test` and `metal_watchdog_test` in tiered mode, `metal_alloc_fail_test`, the pressure smoke step | L12, U8 |
| T7 (part) | `accept.py g2`, `balloon.py` | T10 |

Measurements: H-5; M-4, M-7, M-9 and M-10. R-2, recorded in phase 2, sizes the slot pool.

**G2**, `accept.py g2` ([09](09-testing-and-validation.md), section 6.2), on a 48 GB Mac, or on the reference Mac
with `--gpu-budget-mib 37888` and the 16 GiB balloon. If it fails, the fallback of
[00](00-feasibility-study.md), section 7, applies: the 48 GB class is marked beta in the user documents and the
release notes, and phases 5 and 6 go on with product parity and long context on 64 GB first. Tiered mode stays in
the release, and C4 still gates it.

### 3.6 Phase 5: release engineering and the M1-M4 families

| Task | What | After |
| --- | --- | --- |
| B13 | `check_release_policy.py` | |
| B12 | `verify_release.py` | B10, B11 |
| T15 | `accept.py release` and step 6 of `verify_release.py` | B12, T14 |
| B18 | `mac-release.yml`, tried in a test repository | B12, B13 |
| B20 | The licence review | B10 |
| S14 | The Monitor and About texts, reviewed | S13 |
| T16 | `family_run.sh` and the first family run on each of M1-M4 | T14 |

Measurements: B-2, B-5, B-6, B-7, B-8 and B-11; M-8 on the 32, 36 and 48 GB Macs that the family runs reach;
L-2 to L-8, L-10, L-12 and L-14 on M1-M4, and L-13 and L-15 on the slowest of them.

Exit criteria:
- A test tag in a test repository is built, verified and published by `mac-release.yml`, and a planted failure
  leaves only a draft (B18, T15).
- Each of M1-M4 has a family file ([09](09-testing-and-validation.md), section 7.4).

### 3.7 Phase 6: parity and the first release

| Task | What | After |
| --- | --- | --- |
| L15, U10, S21, B22, T19 | Every measurement recorded in appendix A, and the default it drives updated where it is specified | |
| S20 | The Mac user documents, reviewed by a person who follows the documented macOS setup flow | S2-S13 |
| `requirements.json` | Every requirement of [01-requirements.md](01-requirements.md) with its checks; `test_map.py --final` passes | T17 |
| T7 | `accept.py g3` | T15, T16 |
| The candidate | The first release's candidate (section 5.2): its release run, the family runs on its commit, and the reviews of [09](09-testing-and-validation.md), section 7.6, on a commit with its sources in `reviews.json` | All of the above |

Measurements: L-11; M-2; anything still open from earlier phases. A measurement that cannot be made, for lack of a
Mac, keeps its default, and the gap is listed in the release notes and in [11](11-risks-and-open-questions.md),
section 2.

**G3**, `accept.py g3` ([09](09-testing-and-validation.md), section 6.3), on the first release's candidate
(section 5.2). If it fails, the first release goes out as a beta, with each failing requirement and its check in the
notes.

### 3.8 Conditional work items

L16-L23 ([04](04-metal-kernels.md), section 12.2) are done only when their conditions hold. Each is then added to
the phase in which its condition is found, and that phase's estimate is redone.

| Task | Its condition is found by | Earliest phase |
| --- | --- | ---: |
| L16, batched drafter prefill | The drafter prefill's share of a long prompt; R-3 | 3 |
| L17, indirect command buffers | H-4 above 1 ms | 2 |
| L18, a second window arena | M-2, after gates C1-C4 pass | 6 |
| L19, the prompt attention kernel | PF-1 failing, or L-13 above 25% for attention | 2 |
| L20, the selection kernel | L-13 above 25% for selection | 2 |
| L21, the device MTP round | Gates C1-C4 passing | 4 |
| L22, the expert decode once per chunk | L-13 above 25% for the expert decode | 2 |
| L23, `STRATA_QSA_DUMP` | A user of `tools/qsa_locality.py` asking for it | Any |

One that has not started by the first release stays open, and [11](11-risks-and-open-questions.md) lists it.

---

## 4. Upstream compatibility reviews

`UPSTREAM_SYNC` records the source revision used to establish provenance and compatibility expectations. It is not a
Git parent. The project does not merge upstream history on a schedule. When upstream changes could affect the model,
protocol, build or supported platforms, review the change and selectively port only the fixes this project needs.
Keep the invariants in [02](02-architecture.md), section 9; run the checks in
[09](09-testing-and-validation.md), section 7.3.

### 4.1 The procedure

1. **Choose a reference revision.** Start with the latest upstream release or commit under review. Verify the
   reference revision's own tests when practical, and record its full SHA before comparing it with this project.
2. **Review without merging history.** Use a temporary checkout of the reference revision. Do not merge or rebase its
   commits into this repository. If a change is needed, implement or cherry-pick the smallest suitable patch onto a
   project commit, preserving its authorship, notices and license requirements.
3. **Run `tools/metal/check_drift.py`.** Classify each changed marker, copied range, kernel, model or pinned revision
   using section 4.2. Update its implementation, test and specification together. Advance `UPSTREAM_SYNC` only when
   intentionally changing the project's comparison baseline, then run `check_drift.py --update` (section 4.3).
4. **Review golden sets.** Apply [09](09-testing-and-validation.md), section 2.5. Follow `check_drift.py`'s
   path-based verdict: every changed `src/` or `include/` path requires a new CUDA golden set unless it is explicitly
   listed in `no_arithmetic` (currently only `src/platform/**`). Retain the existing validity record only when the
   verdict is `still valid`.
5. **Test the local change.** Run the applicable steps in [09](09-testing-and-validation.md), section 7.3, and the
   Mac hardware workflow where available. Land passing changes as ordinary project commits with English messages.
6. **Log material reviews** in `docs/mac/sync-log.md`: date, reference SHA, changes considered and ported, golden-set
   verdict, validation results and time spent. This remains measurement H-6 ([06](06-engine-host-port.md), section 15).

Urgent upstream fixes may be reviewed immediately. An update to the reference SHA alone does not import upstream Git
history or imply that every upstream change has been adopted.

### 4.2 Resolving `check_drift.py` failures

| Failure | Resolution | Specified in |
| --- | --- | --- |
| A new `--flag` in the parser range | Give it a disposition (KEEP, MAC, IGNORE, [06](06-engine-host-port.md), section 3.1) in the copy of `options.cpp` and in the table; `options_dispositions_test` covers it | [06](06-engine-host-port.md), section 3 |
| A new `STRATA_*` variable in `src/` or `include/` | Classify it as host or kernel scope, and give it a disposition; `kernel_env_dispositions_test` covers a kernel-scope one | [06](06-engine-host-port.md), section 4; [04](04-metal-kernels.md), section 9 |
| A new `INFO` key, `ERR` string or `DONE` field | Emit it, or record why the Mac omits it; `serve_transcript_test`'s golden transcript is updated | [06](06-engine-host-port.md), section 5 |
| A copied range's hash | Port the change into the copy, and update the range and commit in its `synced from` header | [06](06-engine-host-port.md), section 1.3 |
| A marker count in a patched file | Re-check every marked edit in the resulting project file, then record the new count | [02](02-architecture.md), section 9 |
| A Python change next to a `# MAC:` marker, or a new `subprocess`, `os.name` or `sys.platform` branch in a patched Python file | Re-check the Mac branch next to the marker; give the new branch its Mac behaviour in the section of [07](07-server-setup-port.md) that patches the file | [07](07-server-setup-port.md), section 9 |
| A changed CUDA kernel, or a new one | Port the change to its Metal kernel, with its twin and oracle | [04](04-metal-kernels.md), sections 2 and 10 |
| A new model in upstream's `MODELS`, or a new pinned revision | `make_geometry.py --check-remote` fails in CI until its geometry file is written. A model with a tensor type that has no Metal kernel plans as `unsupported` ("not yet available on Mac") until [03](03-model-reference.md) describes it and its kernels exist. | [07](07-server-setup-port.md), sections 4.4.2 and 4.4.3 |

### 4.3 `tools/metal/drift.json`

The file records what the last compatibility review classified. `check_drift.py` recomputes each value at `UPSTREAM_SYNC` and fails
on every difference, with one line per item:

```json
{
  "schema": 1,
  "classified_at": "<40 hex: the UPSTREAM_SYNC these values were classified at>",
  "markers": {"CMakeLists.txt": 22, "setup.py": 0, "serve/server.py": 0, "src/platform/direct_file.cpp": 0},
  "copies": {
    "src/metal/program/options.cpp": {"from": "src/program/generate.cpp", "range": "999-1255", "sha256": "<hex>"}
  },
  "flags": ["--adapt-every", "..."],
  "env": ["STRATA_ARENA_LOCK", "..."],
  "info_keys": ["..."],
  "err_strings": ["..."],
  "done_fields": ["..."],
  "cuda_kernels": {"src/kernels/cuda/example.cu": "<sha256>"},
  "models": ["..."],
  "revisions": {"owner/model": "<40-hex revision>"},
  "no_arithmetic": ["src/platform/**"]
}
```

- The marker counts are counted in the project; `0` stands for a count that work items set when they patch the file.
- A copy's `sha256` is over the upstream lines of `range` at `UPSTREAM_SYNC`, with line endings normalised to LF.
- `cuda_kernels` records the SHA-256 of each upstream `.cu` and `.cuh` file under `src/kernels/`; adding, removing
  or changing one fails the check. `models` records the keys of `setup.py`'s `MODELS`; `revisions` records the
  repository-to-commit mapping in `HF_REVISIONS`.
- `no_arithmetic` holds reviewed globs of upstream files whose changes cannot affect numerical outputs through
  any supported execution path. It currently contains only `src/platform/**`; every other `src/` or `include/` change
  conservatively requires a new golden set. Add a path only after review, with a regression test and the reason recorded
  in the change.
- `check_drift.py --update` rewrites the file from the current values and sets `classified_at` to `UPSTREAM_SYNC`,
  after the reviewer has resolved each failure. The pull request's diff of `drift.json` is the list the
  reviewer checks against the classifications.
- `tools/metal/test_check_drift.py` ([09](09-testing-and-validation.md), section 9.3) plants each kind of change in
  a temporary copy of the tree and checks that `check_drift.py` names it.

### 4.4 Compatibility reviews and releases

- **Before the first release**, commit selected compatibility changes on `main` as ordinary project changes. Advance
  `UPSTREAM_SYNC` only after the reference revision has been reviewed; it remains metadata, not a Git parent.
- **After the first release**, if a review adopts a newer upstream CMake project version, make that change as a project
  commit on `release/<new version>-metal.<METAL_REV + 1>` ([08](08-build-and-distribution.md), section 11.1,
  rule 2), publish the release from that branch, and merge the project branch to `main` after publication.
- A compatibility review that does not change the CMake version follows the ordinary project release rules; an
  engine-only change needs no release under rule 4 of section 11.1.

---

## 5. Releases

### 5.1 Making a release

The steps are those of [08](08-build-and-distribution.md), section 11.1, here for `v0.1.34-metal.2`:
1. `git switch -c release/0.1.34-metal.2 main`, or the project branch of section 4.4.
2. Raise `METAL_REV` by one, and `MIN_ENGINE_METAL` when setup or the server needs the new engine. Push; `ci.yml`
   runs on `release/**`.
3. Before tagging, run the weekly slices that have not passed on the branch's engine sources (`workflow_dispatch`
   with `day`), so that step 10 of the release run finds them ([09](09-testing-and-validation.md), section 8).
4. `git tag -a v0.1.34-metal.2 -m "..."` on the branch head, with a `Numerics: changed (<reason>)` line when the
   kernels' rounding changed since the previous release. Push the tag.
5. `mac-release.yml` builds a draft, `verify_release.py` verifies it on `strata-ref`, and a person approves the
   `release` environment to publish it ([08](08-build-and-distribution.md), section 12.3).
6. Merge the branch into main with a pull request. `check_release_policy.py --remote` passes once the release is
   published.

**The release candidate.** The tag of step 4 is the candidate while its draft is unpublished. When `build` or
`verify` fails, the fix goes onto the release branch. Then `gh release delete <tag> --yes --cleanup-tag` removes the
draft and the tag, and step 4 runs again on the new head. A tag that was ever published is never deleted or moved.

**When to release.** After the first release: when setup or the server needs a newer engine
(`MIN_ENGINE_METAL`), when a compatibility review adopts a newer upstream version (section 4.4), or when an engine fix matters to users.
An engine change without a setup change goes to main without a release (rule 4 of
[08](08-build-and-distribution.md), section 11.1).

### 5.2 The first release

The first release is `v{version}-metal.1`, at the upstream version the project uses for its current compatibility baseline. Its candidate is made
at the end of phase 6:
1. Steps 1-4 of section 5.1, with `METAL_REV` left at 1. `check_release_policy.py --remote` passes with
   `no release yet` ([08](08-build-and-distribution.md), section 11.1).
2. `mac-release.yml` builds the draft, and its `verify` job runs the release run of
   [09](09-testing-and-validation.md), section 8.
3. Run `tools/metal/family_run.sh <tag's commit> <the draft's asset>` on one Mac of each of M1-M4
   ([09](09-testing-and-validation.md), section 7.4), and attach the files to the release-tracking issue.
4. Record the reviews of [09](09-testing-and-validation.md), section 7.6, in `reviews.json`, on a commit whose
   sources are the tag's.
5. Before the publication is approved, run `accept.py g3` on the reference Mac and record the decision in
   `docs/mac/gates/G3.md`. The person who approves the publication reads it.
6. If G3 passes, approve the publication. If it fails, publish it as a beta: mark the release a pre-release
   (`gh release edit <tag> --prerelease`), and list in its notes each failing requirement with its check
   ([09](09-testing-and-validation.md), section 6.3). G2's outcome is in the notes either way.

A fix found in steps 2-5 goes onto the release branch, and the candidate is made again (section 5.1).

The first release run has no previous release. REG.decode and REG.prefill are skipped with the note
`no previous release` ([09](09-testing-and-validation.md), section 5.4), and C3.disabled runs the engine under test
on both sides ([09](09-testing-and-validation.md), section 4.3).

---

## 6. Gates and their records

| Gate | End of | Command | Record | If it fails |
| --- | --- | --- | --- | --- |
| G1 | Phase 2 | `accept.py g1` | `docs/mac/gates/G1.md` | Section 3.3 |
| G2 | Phase 4 | `accept.py g2` | `docs/mac/gates/G2.md` | Section 3.5 |
| G3 | Phase 6 | `accept.py g3` | `docs/mac/gates/G3.md` | Section 5.2 |

Each record names the result folder, the commit, the Mac, pass or fail, and what was decided. When G2 runs on the
reference Mac with the balloon, its record says so, and a later run on a 48 GB Mac replaces it
([09](09-testing-and-validation.md), section 6.2).

---

## 7. Measurements

Every measurement is listed in the document that defines it, with its method and the decision it drives. This table
schedules them. Until a measurement is recorded in [appendix-a-measurements.md](appendix-a-measurements.md), its
default stands, and the specification is written for that default.

| ID | Measures | Phase | Default until recorded | Defined in |
| --- | --- | ---: | --- | --- |
| R-1 | Coder's router ids and weights against CUDA's FP64 router | 2 | Ids identical except at exact ties; weights within 1e-5 relative | [03](03-model-reference.md), section 10 |
| R-2 | The expert union of a verify window, T = 2..4 | 2 | The upper bound of [03](03-model-reference.md), section 4.3 | [03](03-model-reference.md), section 10 |
| R-3 | MTP acceptance per draft position, with and without the draft vocabulary | 3 | Upstream's rates, because the arithmetic matches | [03](03-model-reference.md), section 10 |
| R-4 | Unlisted float tensors byte-identical between `dense.bin` and the GGUF | 1 | Yes | [03](03-model-reference.md), section 10 |
| R-5 | Per-layer residual drift between the Metal and CUDA builds | 2 | Smooth growth, below the CUDA-to-llama.cpp drift | [03](03-model-reference.md), section 10 |
| L-2 | Tile width of `strata_gr_down`; rows per mat-vec threadgroup | 2, 5 | TILEV 512; R = 4 | [04](04-metal-kernels.md), section 11 |
| L-3 | `simd_sum` order on M1-M4 | 5 | Ascending, as on M5 | [04](04-metal-kernels.md), section 11 |
| L-4 | Codebook placement in the real kernels | 2, 5 | Program-scope `constant` | [04](04-metal-kernels.md), section 11 |
| L-5 | Expert kernel shapes on M1-M4 | 5 | The M5 shapes | [04](04-metal-kernels.md), section 11 |
| L-6 | `mpp` usable per family | 1, 2, 5 | `mpp` on M5, `simd` elsewhere | [04](04-metal-kernels.md), section 11 |
| L-7 | The fastest `simd` GEMM tile | 5 | 32 × 64 | [04](04-metal-kernels.md), section 11 |
| L-8 | Pipeline count and creation time | 1, 5 | Every pipeline at startup | [04](04-metal-kernels.md), section 11 |
| L-9 | nvcc's contractions in the SASS of the reference GPU | 1 | The forms [04](04-metal-kernels.md) gives | [04](04-metal-kernels.md), section 11 |
| L-10 | FP32 denormal flushing | 1, 5 | IEEE, no flush | [04](04-metal-kernels.md), section 11 |
| L-11 | Gains from `fast/` | 6 | No kernel in `fast/` | [04](04-metal-kernels.md), section 11 |
| L-12 | The longest prefill chunk, and the GPU time limit | 2, 5 | One command buffer per chunk | [04](04-metal-kernels.md), section 11 |
| L-13 | Shares of attention, selection and expert decode in a chunk | 2, 5 | The kernels of the first release | [04](04-metal-kernels.md), section 11 |
| L-14 | `concurrent` against `serial` dispatch | 2, 5 | `serial` | [04](04-metal-kernels.md), section 11 |
| L-15 | The draft head's indexed read | 2, 5 | Indexed read in place | [04](04-metal-kernels.md), section 11 |
| M-1 | Wake latency of the wait modes; event hops | 1 | `STRATA_METAL_WAIT=block` | [06](06-engine-host-port.md), section 15 |
| M-2 | A second queue for MTP overlap | 6 | One queue | [06](06-engine-host-port.md), section 15 |
| M-3 | The reserve that keeps a 64K session stable | 3 | `--vram-reserve-mib` 700 | [05](05-memory-and-residency.md), section 10 |
| M-4 | No-copy pages and arenas under memory pressure | 4 | Full mode stays no-copy | [05](05-memory-and-residency.md), section 10 |
| M-5 | Warm-up time, cold and warm | 2 | No progress bar | [05](05-memory-and-residency.md), section 10 |
| M-6 | `Private` against `Shared` for KV and state | 2 | The storage modes of [05](05-memory-and-residency.md), section 7 | [05](05-memory-and-residency.md), section 10 |
| M-7 | `mlock` limits on macOS 27 | 4 | `STRATA_ARENA_LOCK` off; the balloon without `--lock` | [05](05-memory-and-residency.md), section 10 |
| M-8 | W and `maxBufferLength` on 32, 36 and 48 GB Macs | 5 | The values of [05](05-memory-and-residency.md), section 8 | [05](05-memory-and-residency.md), section 10 |
| M-9 | The expert hit-rate curve against the resident share | 4 | [05](05-memory-and-residency.md)'s forecast | [05](05-memory-and-residency.md), section 10 |
| M-10 | Whole-layer prefill prefetch in tiered mode | 4 | Prefetch on | [05](05-memory-and-residency.md), section 10 |
| H-4 | CPU encode time of a T = 8 window | 2 | No indirect command buffers | [06](06-engine-host-port.md), section 15 |
| H-5 | A stalled `encodeWaitForEvent` and the GPU timeout | 4 | Tiered waits inside one command buffer per window | [06](06-engine-host-port.md), section 15 |
| H-6 | Effort per upstream compatibility review | 0 on | Review each upstream release and material change; no fixed cadence | [06](06-engine-host-port.md), section 15 |
| H-7 | Paths with spaces in `GENI` and `ENC` | 2 | The parse rule of [06](06-engine-host-port.md), section 5.3 | [06](06-engine-host-port.md), section 15 |
| H-8 | Timer coalescing and App Nap stretching waits | 3 | The activity held during requests ([06](06-engine-host-port.md), section 2.2) | [06](06-engine-host-port.md), section 15 |
| H-9 | Silent macOS gaps in the reused code | 0 | The patches of [06](06-engine-host-port.md), section 7 | [06](06-engine-host-port.md), section 15 |
| X-1 | GPU memory of `strata-vision` with `--vision gpu` | 3 | 2,100 MiB reserve | [07](07-server-setup-port.md), section 14 |
| X-2 | `ioreg` "In use system memory" against the engine's allocation | 3 | `free = min(W − in_use, available RAM)` | [07](07-server-setup-port.md), section 14 |
| X-3 | `--plan-only` time | 1 | Below 200 ms (geometry) and 1 s (files) | [07](07-server-setup-port.md), section 14 |
| X-4 | `caffeinate -i -w` during a long download | 3 | caffeinate during install runs only | [07](07-server-setup-port.md), section 14 |
| X-5 | App Nap and background throttling of a hidden server | 3 | No activity in Python; `ProcessType` `Interactive` | [07](07-server-setup-port.md), section 14 |
| X-6 | `setup.sh` from a Rosetta shell | 3 | An arm64 run, or the Rosetta refusal | [07](07-server-setup-port.md), section 14 |
| X-7 | The Local Network privacy prompt | 3 | No prompt for a listening socket | [07](07-server-setup-port.md), section 14 |
| X-8 | A hosted macOS runner and `--list-devices` | 0 on | Engine tests on the reference Mac only | [07](07-server-setup-port.md), section 14 |
| B-1 | The pipeline archive against the OS shader cache | 2 | Keep the archive | [08](08-build-and-distribution.md), section 14 |
| B-2 | A hosted runner's build against the reference Mac's | 5 | Same bytes when the toolchains match | [08](08-build-and-distribution.md), section 14 |
| B-3 | The Metal toolchain download on a hosted runner | 0 on | Download on every run | [08](08-build-and-distribution.md), section 14 |
| B-4 | `strata-vision`'s first start | 3 | Setup does not warm it | [08](08-build-and-distribution.md), section 14 |
| B-5 | The hardened runtime and Metal | 5 | No entitlements | [08](08-build-and-distribution.md), section 14 |
| B-6 | Developer ID in practice | 5 | Ad hoc releases | [08](08-build-and-distribution.md), section 14 |
| B-7 | Zip bytes with another Python or zlib | 5 | Compare members, not zips | [08](08-build-and-distribution.md), section 14 |
| B-8 | Asset sizes | 5 | None needed | [08](08-build-and-distribution.md), section 14 |
| B-9 | Local build time | 3, 5 | "5-10 minutes" | [08](08-build-and-distribution.md), section 14 |
| B-10 | GPU trace size with a real model | 2 | Warn above 8 GiB | [08](08-build-and-distribution.md), section 14 |
| B-11 | Symbolication with the dSYM and `.metallibsym` | 5 | Ship both | [08](08-build-and-distribution.md), section 14 |
| T-1 | The spread of upstream's own CUDA paths | 1 | Upstream's Q2_0 floor; Y_pp 89.8% and 0.33 | [09](09-testing-and-validation.md), section 11 |
| T-2 | The margin above which the floor never flips | 1 | m₁ = 1.0 logit | [09](09-testing-and-validation.md), section 11 |
| T-3 | CUDA against llama.cpp | 1 | C1.2 and C1.3 not gating | [09](09-testing-and-validation.md), section 11 |
| T-4 | Run-to-run variation of a benchmark | 2 | CV ≤ 2%; bar 0.95 | [09](09-testing-and-validation.md), section 11 |
| T-5 | Slowdown over back-to-back runs | 2 | No pause | [09](09-testing-and-validation.md), section 11 |
| T-6 | The time of each run | 3 | The split of [09](09-testing-and-validation.md), section 7.2 | [09](09-testing-and-validation.md), section 11 |
| T-7 | Size and recording time of the golden set | 1 | About 40 GB, about 2 h per model | [09](09-testing-and-validation.md), section 11 |
| T-8 | A replacement reference GPU | When a card is replaced | Run only then | [09](09-testing-and-validation.md), section 11 |
| T-9 | The cache budget that gives the pressure scenario its pressure | 3 | None: C3.pressure and C3.exchange wait for it | [09](09-testing-and-validation.md), section 11 |

A recorded value changes the default where it is specified, in the same pull request as the appendix row. A
threshold of [09](09-testing-and-validation.md) changes by its section 10.3.

---

## 8. Machines and accounts

| Resource | Needed from | Use | If missing |
| --- | --- | --- | --- |
| The reference Mac: M5 Pro, 64 GB, 20-core GPU, 2 TB SSD for models and results | Phase 0 | `strata-ref`: every hardware run, every gate, every release verification | The plan cannot run; it is the one machine nothing replaces |
| The CUDA reference PC: an sm_120 card with 12 GB or more, 64 GB of RAM ([09](09-testing-and-validation.md), section 2.4) | Phase 1 | The golden set, at the start and for every compatibility change whose verdict is `golden: record again` (all changed `src/` or `include/` paths except reviewed `no_arithmetic` paths; currently only `src/platform/**`) | C1 has no reference, and G1 cannot pass ([11](11-risks-and-open-questions.md), section 2) |
| A 48 GB Apple10 Mac | Phase 4 | The authoritative form of G2; M-8 | G2 runs on the reference Mac with the balloon |
| Macs of the M1, M2, M3 and M4 generations, for a few hours each; 32 GB or more for `accept.py family` | Phase 5, and again for each release candidate | The family runs; the family parts of L-2 to L-8 and L-10 to L-15; M-8 on 32 and 36 GB | REQ-PLT-03 cannot hold, and the release notes say "tested on M5 only" |
| A GitHub repository for the project, and a test repository for the release pipeline | Phase 0, phase 5 | CI, releases; B12 and B18 | — |
| A GitHub-hosted macOS 27 arm64 image | When GitHub offers one | `mac-host` on pull requests | The macOS workflow remains the only product build and test path ([08](08-build-and-distribution.md), section 14) |
| An Apple Developer account | Optional | Developer ID signing and notarisation | Ad hoc releases ([08](08-build-and-distribution.md), section 11.5) |

---

## 9. Where the details are

| Topic | Document |
| --- | --- |
| Why the port, and the gates | [00-feasibility-study.md](00-feasibility-study.md), sections 5-7 |
| What the port must do | [01-requirements.md](01-requirements.md) |
| The design, and how it follows upstream | [02-architecture.md](02-architecture.md), sections 8 and 9 |
| The work items | Section 11 of [03](03-model-reference.md); 12 of [04](04-metal-kernels.md); 14 of [05](05-memory-and-residency.md); 16 of [06](06-engine-host-port.md); 15 of [07](07-server-setup-port.md) and [08](08-build-and-distribution.md); 12 of [09](09-testing-and-validation.md) |
| The gates' steps | [09-testing-and-validation.md](09-testing-and-validation.md), section 6 |
| The tests for an upstream compatibility review | [09-testing-and-validation.md](09-testing-and-validation.md), section 7.3 |
| The release policy and CI | [08-build-and-distribution.md](08-build-and-distribution.md), sections 11 and 12 |
| Measured results | [appendix-a-measurements.md](appendix-a-measurements.md) |
| Risks and open decisions | [11-risks-and-open-questions.md](11-risks-and-open-questions.md) |
