# Strata for macOS: the documentation set

Status: implementation in progress, 2026-10-07. The active CMake graph builds only the macOS arm64 Metal product; Windows and Linux are not supported targets. On Mac17,9 (64 GB, macOS 27.0.1), the full CTest run passed 82 tests, with one expected skip because real model inputs were unavailable (41 host, 41 Metal and one model test are registered). The eight affected loader tests (`gguf_reader_test`, `gguf_split_test`, `metal_weight_store_test`, `model_metadata_test`, `model_types_test`, `native_experts_reader_test`, `pack_index_test` and `ple_constants_test`) passed. `.venv/bin/python tools/metal/run_python_tests.py` passed 35/35 modules; the test map and upstream drift check also passed. `dense_vs_gguf_test` passed against synthetic seven-pack fixtures; no full catalogue checkpoint was available. The `mac-model` preset requires `STRATA_TEST_MODELS`. The native installer and production inference path are not ready: `setup.sh` validates the supported Mac, then exits before creating an environment or changing files. Phase 0 remains incomplete pending its reference-machine and CI exit criteria.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.

Implementation choices, constraints, and required reviews are tracked in
[IMPLEMENTATION-DECISIONS.md](IMPLEMENTATION-DECISIONS.md).

This folder specifies the native Metal backend for a separately maintained Strata project on Apple Silicon Macs with macOS 27. The source baseline is Strata v0.1.34; separate Git history does not make the code clean-room or remove its license and attribution obligations. See [PROJECT_ORIGIN.md](../PROJECT_ORIGIN.md).
It is written for the engineers who build the port and for the owner of the project. A reader who follows it should
not need the research behind it, but that research is kept here too, in `audit/` and `spikes/`.

This file covers:
- the answer in short;
- what to read first, by role;
- what each document answers;
- the research input in `audit/` and `spikes/`;
- the conventions every document follows: normative words, IDs, references, units and spelling;
- how to start the work, and how to keep the set current.

---

## 1. The answer in short

[00-feasibility-study.md](00-feasibility-study.md) gives the evidence. In brief:

| Question | Answer |
| --- | --- |
| Can Strata run on a Mac? | Yes. Every operation the model needs already runs on Metal in other engines, and the measurements on the reference Mac show no hardware blocker ([appendix-a-measurements.md](appendix-a-measurements.md)). |
| Is it worth doing? | Yes, for 36-64 GB Apple Silicon Macs, where the Metal engine, server, Monitor and MCP can form one native product. Not as one more engine for 96 GB+ Macs. |
| Decision | **Go**: a native Metal engine inside a separately maintained Strata project based on the documented v0.1.34 source baseline, macOS 27 only, Apple M5-class first, Q2_0 first. |
| Shape of the port | A native Metal engine and a simpler memory manager, not a translation of the CUDA code. Every expert runs on the GPU. In full mode every weight is resident. In tiered mode, for smaller Macs, the experts stream from the SSD into a fixed slot pool. The Python setup and server layers are also being narrowed to macOS ([02-architecture.md](02-architecture.md), section 1). |
| Effort | 29 weeks in seven phases for one senior engineer who knows GPU programming ([10-implementation-plan.md](10-implementation-plan.md), section 2) |
| Checkpoints | Gates G1 (first token, end of phase 2), G2 (tiered mode, end of phase 4) and G3 (parity, end of phase 6). Each gate is a command, `accept.py g1`, `g2` or `g3` ([09-testing-and-validation.md](09-testing-and-validation.md), section 6), and each has a stated action if it fails ([00](00-feasibility-study.md), section 7). |
| Main risks | Upstream moves fast; competitors may close the 36-48 GB gap; decode may stay overhead-bound; macOS memory behaviour is less documented than CUDA's ([11-risks-and-open-questions.md](11-risks-and-open-questions.md)) |

---

## 2. What to read first

Everyone should read [00](00-feasibility-study.md), [02](02-architecture.md), sections 1 and 10 (the design in
brief and the glossary), and [10](10-implementation-plan.md), section 2 (the phases at a glance). Then:

| Role | Read | Then use as reference |
| --- | --- | --- |
| The project's owner | [01](01-requirements.md), sections 1 and 2; [10](10-implementation-plan.md), sections 1, 4-6 and 8; [11](11-risks-and-open-questions.md), sections 1-3 | [08](08-build-and-distribution.md), sections 11 and 12 (release policy and CI) |
| Kernel implementer | [03](03-model-reference.md); [04](04-metal-kernels.md), sections 1-4, then the section for each kernel family as its work item comes up | [appendix-a-measurements.md](appendix-a-measurements.md), sections A.2, A.5, A.6, A.10 and A.11; `spikes/metal-numerics/`, `spikes/metal-binding/` |
| Engine host and memory | [05](05-memory-and-residency.md); [06](06-engine-host-port.md) | [02](02-architecture.md), sections 3-5 and 8; appendix A, sections A.3 and A.4 |
| Python layer: setup, server, MCP | [07](07-server-setup-port.md) | `spikes/serve-macos/macos.py`, which becomes `serve/macos.py` |
| Build, release and CI | [08](08-build-and-distribution.md) | `spikes/metal-build/`; appendix A, section A.8 |
| Testing | [09](09-testing-and-validation.md) | [01](01-requirements.md), whose rows name the checks; [10](10-implementation-plan.md), section 1.3 |

Every implementer also needs [10](10-implementation-plan.md), section 4 (upstream compatibility reviews) and
[02](02-architecture.md), section 8 (the invariants every change must keep).

---

## 3. The documents

| Document | What it answers |
| --- | --- |
| [00-feasibility-study.md](00-feasibility-study.md) | Can it be done, is it worth it, what is the scope, and where are the decision gates? |
| [01-requirements.md](01-requirements.md) | What must the finished port do? Each requirement `REQ-<AREA>-<NN>` names where it is specified and which checks verify it. |
| [02-architecture.md](02-architecture.md) | How does the port fit together? Processes, layers, components, threads, the main flows, the memory budget, the source tree, decisions D1-D15, invariants I1-I11, following upstream, and the glossary. |
| [03-model-reference.md](03-model-reference.md) | What is the model? Its files, tensors, formats and forward pass, the MTP layer, the PLE table, and the golden set (R1-R12). |
| [04-metal-kernels.md](04-metal-kernels.md) | What does each Metal kernel compute, in what order, with what numerics, and how is it encoded and tested? (L1-L15; conditional L16-L23) |
| [05-memory-and-residency.md](05-memory-and-residency.md) | Where does every byte live? The budget W, the plan, full and tiered mode, the slot pool, wrapping files, memory pressure (U1-U10). |
| [06-engine-host-port.md](06-engine-host-port.md) | What changes in the engine process apart from the kernels? Flags and variables, the platform layer, the Metal context, decode and prefill orchestration, the serve protocol, errors and the watchdog (H1-H13). |
| [07-server-setup-port.md](07-server-setup-port.md) | What changes in the Python layer? `serve/macos.py`, the Mac setup flow, server, telemetry, MCP and tools (S1-S21). |
| [08-build-and-distribution.md](08-build-and-distribution.md) | How is the macOS product built, packaged, signed, released and checked in CI? (B1-B22) |
| [09-testing-and-validation.md](09-testing-and-validation.md) | How is it tested? Prompt sets, test models, the CUDA reference, criteria C1-C4, performance criteria, the gates, schedules, the test map, and what to do when a test fails (T1-T19). |
| [10-implementation-plan.md](10-implementation-plan.md) | In what order is the work done? Phases 0-6, every work item in one phase, upstream compatibility reviews, releases, the schedule of every measurement, and the machines needed. |
| [11-risks-and-open-questions.md](11-risks-and-open-questions.md) | What could go wrong, and what is still undecided? Open questions OQ-01 to OQ-12, risks RSK-01 to RSK-23, retired risks, and the external sources. |
| [appendix-a-measurements.md](appendix-a-measurements.md) | What was measured on the reference Mac, and what it implies? Bandwidth, mapping, SSD, dispatch overhead, matrix throughput, projections, the build, model headers, numerics, binding and host contraction. |

Each of documents 01-11 opens with its status, the upstream baseline and its sources, then says what it covers and
which documents cover the rest.

---

## 4. Research input

### 4.1 `audit/`

Reports from the research phase, each headed "Audit report: research input, kept verbatim". They read the upstream
source at `1678de33` and the external projects. They are evidence, not specification: where a document and an audit
disagree, the document wins, and [03](03-model-reference.md) and [04](04-metal-kernels.md), section 1.3, list the
places where the audits are wrong or superseded.

| File | Covers |
| --- | --- |
| [A-model-and-kernels.md](audit/A-model-and-kernels.md) | The model and the CUDA kernels; its appendices are the files K1-K5 |
| [B-engine-host.md](audit/B-engine-host.md) | The engine process: binaries, flags, the serve protocol, artifacts, memory APIs, expert tiers, speculation, the caches, the build and tests on macOS, telemetry |
| [C-python-setup-serve.md](audit/C-python-setup-serve.md) | `setup.py` and its fit formulas, the server, the Monitor tab, MCP, the Python tests, packaging |
| [D-external.md](audit/D-external.md) | The model's files, llama.cpp, MLX, LM Studio, Ollama and other Metal engines, the Apple Silicon lineup, macOS 27 and its memory APIs |
| [K1-gr-router-shared-rope-cvec.md](audit/K1-gr-router-shared-rope-cvec.md) | Hyper-connections, the router, the shared expert, RoPE, control vectors |
| [K2-qsa-kv-gdn.md](audit/K2-qsa-kv-gdn.md) | Sparse attention, the KV cache, Gated DeltaNet |
| [K3-quant-and-matvec.md](audit/K3-quant-and-matvec.md) | Quantisation formats and mat-vecs |
| [K4-ple-mtp-head-sampler.md](audit/K4-ple-mtp-head-sampler.md) | The PLE table, MTP, the output head, the sampler |
| [K5-orchestration-vram-tests.md](audit/K5-orchestration-vram-tests.md) | Decode and prefill orchestration, the VRAM tier, every discrete-GPU assumption, the test strategy |

### 4.2 `spikes/`

Small programs run on the reference Mac. Their results are in [appendix-a-measurements.md](appendix-a-measurements.md),
and some of their files are inputs to work items. Each folder except `serve-macos/` has a README with its files and
how to run them.

| Folder | What it holds | Results |
| --- | --- | --- |
| [metal-feasibility/](spikes/metal-feasibility/) | Bandwidth, concurrency, mapping flags, matrix multiply and SSD micro-benchmarks | Appendix A, sections A.2-A.6 |
| [metal-build/](spikes/metal-build/) | The build experiments, and the project skeleton in `mac-port/tree/` that work items B1-B5 start from | Section A.8 |
| [model-headers/](spikes/model-headers/) | Header and tensor inventories of the model files, read by HTTP range requests; the fixtures of work item R1 | Section A.9 |
| [metal-numerics/](spikes/metal-numerics/) | Contraction, rounding, transcendentals, simdgroup order and codebook placement in MSL | Section A.10 |
| [metal-binding/](spikes/metal-binding/) | Buffer binding and ordering on untracked shared buffers | Section A.11 |
| [host-contraction/](spikes/host-contraction/) | Whether Apple clang fuses `a*b + c` in host C++ | Section A.12 |
| [serve-macos/](spikes/serve-macos/) | `macos.py`, the reference code for `serve/macos.py`, verified on the reference Mac | [07](07-server-setup-port.md), sections 3 and 4 |

---

## 5. Conventions

### 5.1 Normative words

**Must** means required for a conforming port. **Should** means the recommended default, which a measurement or
review may overturn. Each of documents 01-11 states this in its header.

### 5.2 IDs

Requirement, open-question and risk IDs are never reused: a withdrawn requirement keeps its row with the reason
([01](01-requirements.md), section 1.1), and a settled question or a retired risk moves to its own section
([11](11-risks-and-open-questions.md), section 1.1).

| ID | What it names | Defined in |
| --- | --- | --- |
| `REQ-<AREA>-<NN>` | A requirement. Areas: PLT, MOD, FUN, MEM, COR, PRF, SRV, ROB, BLD, MNT, DOC | [01](01-requirements.md), section 1.1 |
| D1-D15 | The design decisions | [02](02-architecture.md), section 7.1 |
| I1-I11 | The invariants | [02](02-architecture.md), section 8 |
| G1-G3 | The decision gates | [00](00-feasibility-study.md), section 7; exact steps in [09](09-testing-and-validation.md), section 6 |
| C1-C4 | The correctness criteria | [09](09-testing-and-validation.md), section 4 |
| R1-R12 | Work items: model and golden set | [03](03-model-reference.md), section 11 |
| L1-L23 | Work items: kernels (L16-L23 conditional) | [04](04-metal-kernels.md), section 12 |
| U1-U10 | Work items: memory and residency | [05](05-memory-and-residency.md), section 14 |
| H1-H13 | Work items: engine host | [06](06-engine-host-port.md), section 16 |
| S1-S21 | Work items: Python layer | [07](07-server-setup-port.md), section 15 |
| B1-B22 | Work items: build, release and CI | [08](08-build-and-distribution.md), section 15 |
| T1-T19 | Work items: tests | [09](09-testing-and-validation.md), section 12 |
| R-n, L-n, M-n, H-n, X-n, B-n, T-n | Measurements still to make, each with the default that stands until it is recorded. M-n is shared by [05](05-memory-and-residency.md) and [06](06-engine-host-port.md). | The measurement section of each document ([03](03-model-reference.md), 10; [04](04-metal-kernels.md), 11; [05](05-memory-and-residency.md), 10; [06](06-engine-host-port.md), 15; [07](07-server-setup-port.md), 14; [08](08-build-and-distribution.md), 14; [09](09-testing-and-validation.md), 11); scheduled in [10](10-implementation-plan.md), section 7 |
| OQ-01 to OQ-12 | Open questions and missing resources | [11](11-risks-and-open-questions.md), section 2 |
| RSK-01 to RSK-23 | Risks | [11](11-risks-and-open-questions.md), section 3 |

A work item is a task with a "Done when" column, and its letter names the document that specifies it. A measurement
ID has a hyphen (L-6); a work item does not (L6).

### 5.3 References

- **Between documents:** `[06](06-engine-host-port.md), section 14.2`. Appendix sections are written A.1 to A.12.
- **Into upstream:** `file:line`, relative to the upstream repository root at commit `1678de33`, for example
  `src/program/generate.cpp:160-415`. They stay valid at that commit; across syncs, `check_drift.py` watches the
  upstream ranges the project copies ([10](10-implementation-plan.md), section 4.3).
- **The project:** `{REPOSITORY}` stands for the eventual GitHub `owner/repository`; no destination is configured yet
  ([10](10-implementation-plan.md), section 3.1).
- **The reference Mac:** the M5 Pro 64 GB (20-core GPU) on macOS 27.0 (26A428) with Xcode 27.0 (27A266a) where every
  measurement was taken. It is also the self-hosted CI runner `strata-ref` ([appendix
  A.1](appendix-a-measurements.md)).
- **Precedence:** a detailed document wins over [02](02-architecture.md), and every document wins over `audit/`.
  Any disagreement is a documentation bug, to be fixed in both places.

### 5.4 Units, dates and spelling

- GB = 10^9 B, GiB = 2^30 B, MiB = 2^20 B, KiB = 2^10 B; bandwidths are GB/s. The exception is what the product
  shows a user: memory sizes on a Mac print as GB = 2^30 B, to match the RAM size Apple advertises
  ([07](07-server-setup-port.md), section 4.15).
- Dates are `yyyy-mm-dd`.
- Prose uses British spelling: behaviour, colour, optimise, licence (noun). Code, file names, API names and quoted
  output keep their own spelling: `normalize()`, `MTLResidencySet`, `LICENSE`.

---

## 6. Project layout and current status

The macOS port specification, audit reports, and spikes live in `docs/mac/`. The repository keeps the Mac build,
Metal, setup and validation documentation; retired PC installation and backend guides are not part of the product docs.

This local checkout contains the documented source baseline and the Phase 0 Metal build skeleton. The current CTest and Python results are recorded above. The publication repository and release-test repository remain unconfigured under OQ-01; the model disk and CUDA reference PC are unavailable. Phase 0 is therefore still open. See [IMPLEMENTATION-DECISIONS.md](IMPLEMENTATION-DECISIONS.md).

The repository, documentation set and source reference are already present in this working tree. The remaining Phase 0 preflight in [10](10-implementation-plan.md), section 3.1, is to configure a publication repository when one exists and prepare the reference machines described in section 8. `UPSTREAM_SYNC` records the source baseline for provenance and compatibility checks; Git history is independent of the upstream repository. The current checkout has no configured remote, so no publication destination is claimed.

The B1-B5 Metal build skeleton (root edits, CMake backend and metallib build) is already present. Phase 0 ends only when its remaining exit criteria hold, including CI on the selected publication repository, `mac-hardware.yml` on `strata-ref`, `check_drift.py`, and `test_map.py`.

---

## 7. Keeping the set current

- **In the project, the specification changes with the code.** From phase 0 on, a change to `docs/mac/` is made in the
  commit that changes the code or the test it describes ([10](10-implementation-plan.md), section 3.1).
- **A requirement changes in one commit with `tests/metal/requirements.json`**, and a threshold changes by the rule
  of [09](09-testing-and-validation.md), section 10.3 ([01](01-requirements.md), section 1.1).
- **A measurement is recorded in appendix A** when it is made. Until then its default stands
  ([10](10-implementation-plan.md), section 7). The decision it drives changes the document that defines it, in the
  same pull request.
- **The register is reviewed** at each gate, at each release candidate and when a risk's early sign shows
  ([11](11-risks-and-open-questions.md), section 1.4). A settled question moves to section 2.4 of that document, and
  a retired risk to section 4.
- **At upstream compatibility reviews,** each `check_drift.py` failure is resolved in one commit that changes the code, its
  test and the paragraph of `docs/mac/` that specifies it ([10](10-implementation-plan.md), section 4.1).
- **The Status line** of documents 01-11 takes the date of the document's last change. [00](00-feasibility-study.md)
  records the decision of 2026-10-02; later gate decisions are written to `docs/mac/gates/G<n>.md`
  ([10](10-implementation-plan.md), section 6).

---

## 8. Where the details are

| Topic | Document |
| --- | --- |
| The decision and the gates | [00-feasibility-study.md](00-feasibility-study.md), sections 1, 5 and 7 |
| The requirements | [01-requirements.md](01-requirements.md) |
| The design and the glossary | [02-architecture.md](02-architecture.md), sections 1 and 10 |
| The plan, the phases and upstream compatibility reviews | [10-implementation-plan.md](10-implementation-plan.md) |
| The open questions and risks | [11-risks-and-open-questions.md](11-risks-and-open-questions.md) |
| The measurements behind the numbers | [appendix-a-measurements.md](appendix-a-measurements.md) |
