# 11. Risks and open questions

Status: implementation-ready specification, 2026-10-07.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources:
- the main risks and the gates of [00-feasibility-study.md](00-feasibility-study.md), sections 6 and 7;
- the decisions of [02-architecture.md](02-architecture.md), section 7;
- the measurement and open-item tables of documents 03-09, and the machines and accounts of
  [10-implementation-plan.md](10-implementation-plan.md), section 8;
- the external research of 2026-10-02, kept verbatim in [audit/D-external.md](audit/D-external.md).

This document lists what could stop the port, or a requirement of [01-requirements.md](01-requirements.md), from
being met, and what is still undecided. It covers:
- how the register is kept, scored and owned;
- the open questions and missing resources: what each blocks, who settles it, by when, and what holds until then;
- the measurements that may never be made, and the conditional work items;
- the risk register, with each risk's early signs, mitigation and plan if it happens;
- the risks that the measurements of appendix A have already retired;
- the external sources that [00-feasibility-study.md](00-feasibility-study.md), section 3.1, and this document cite.

Other documents cover the rest:
- what each measurement is and how it is made: the table its ID names ([10](10-implementation-plan.md), section 7,
  lists them all);
- what happens when a gate fails: [00](00-feasibility-study.md), section 7, and
  [10](10-implementation-plan.md), sections 3.3, 3.5 and 3.7;
- the design decisions, and what is deliberately left out of the first release:
  [02-architecture.md](02-architecture.md), section 7.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. How the register works

### 1.1 Entries

- **An open question** (OQ-nn, section 2) is a decision or a resource that the documents cannot settle: a person
  must decide or provide it. Each names what it blocks, who settles it, the latest point at which it must be settled,
  and the default that holds until then. Every open question has a default, so none stops work before its
  "needed by" point.
- **A risk** (RSK-nn, section 3) is something that may happen and would cost time, a requirement, or the port's
  reason to exist. Each names its early signs, the mitigation already in the design or the plan, and what is done if
  it happens.
- **Measurements are not restated here.** A measurement whose result could change the plan is named in the risk it
  watches. The schedule and default of every measurement are in [10](10-implementation-plan.md), section 7.
- **IDs are never reused.** A settled question moves to section 2.4, and a retired risk to section 4, each with
  the date, the reason, and the pull request that recorded it.

### 1.2 Scores

Likelihood is judged for the time until the first release:

| Likelihood | Meaning |
| --- | --- |
| Low | Not expected: under about one chance in five |
| Medium | About one chance in five to one in two |
| High | More likely than not |

Impact is judged by what happens if the risk comes true:

| Impact | Meaning |
| --- | --- |
| Low | A default, a text or a date moves, by less than two weeks |
| Medium | A phase overruns by two weeks or more, or a requirement holds only on some Macs or only with a note in the release |
| High | A gate fails, the first release goes out as a beta, or the decision of [00](00-feasibility-study.md), section 1, must be taken again |

### 1.3 Owners

| Role | Who | Owns |
| --- | --- | --- |
| The project's owner | The person or organisation that owns `{REPOSITORY}` | The open questions, the accounts, the gate decisions ([10](10-implementation-plan.md), section 6), and the fallbacks of the project risks |
| The implementer | The engineer, or engineers, doing the tasks of [10](10-implementation-plan.md) | The technical risks, the measurements that watch them, and upstream compatibility reviews |
| The CUDA PC owner | Whoever runs the CUDA reference PC ([09](09-testing-and-validation.md), section 2.4) | The golden sets |
| The release reviewer | The reviewer of the `release` environment ([08](08-build-and-distribution.md), section 12.3) | The family results of each release ([09](09-testing-and-validation.md), section 7.6) |

One person may hold several roles. The owner of a risk watches its early signs and starts its "if it happens" plan;
a plan that changes a gate decision, a requirement or the scope is the project's owner's decision.

### 1.4 When the register is reviewed

- **At each gate,** before the decision is written into `docs/mac/gates/G<n>.md`: every score is judged again,
  a risk the gate retired moves to section 4, and the sources that the decision depends on (section 5) are checked
  again and dated.
- **At each release candidate** ([10](10-implementation-plan.md), section 5.1): every open question still open,
  every measurement of section 2.2 not made, and every conditional item of section 2.3 not started goes into the
  release notes.
- **When an early sign shows,** in the pull request that records it: a sync log row, a measurement, a nightly
  result.

This document lives at `docs/mac/11-risks-and-open-questions.md` in the project. Update it by pull request,
alongside the relevant code and plan changes.

---

## 2. Open questions and missing resources

### 2.1 Decisions and resources

| ID | Question or resource | Blocks | Settled by | Needed by | Until then |
| --- | --- | --- | --- | --- | --- |
| OQ-01 | The GitHub publication slug (`owner/repository`) and the release-test repository. The local working tree is named `strataapp`; `strata.app` remains a planned public slug, not a configured destination. | Release URLs, issue links, user documents and release-policy checks; B12 and B18 need a test repository | The project owner | Before publishing a release | The checkout has no configured remote. Do not infer a GitHub destination or publish artifacts; configure `setup.RELEASE_REPOSITORY` only after the destination is established. |
| OQ-02 | The CUDA reference PC, and who owns and runs it: an sm_120 card with 12 GB or more, 64 GB of RAM ([09](09-testing-and-validation.md), section 2.4) | The golden set (T2, R12), so gate C1 and G1; R-1, R-5, T-1 to T-3 and L-9; every compatibility review whose verdict is `golden: record again` | The project's owner | Phase 1 | Mac-against-Mac tests run; C1 has no reference and G1 cannot pass ([10](10-implementation-plan.md), section 8). A rented machine meets the requirement if it meets section 2.4 there; its card is recorded, and T-8 applies whenever the card changes. |
| OQ-03 | Macs of the M1, M2, M3 and M4 generations, a few hours each, 32 GB or more for `accept.py family` | REQ-PLT-03; the family runs of each release candidate ([09](09-testing-and-validation.md), section 7.4); L-2 to L-8 and L-10 to L-15 on M1-M4; M-8 on 32 and 36 GB | The project's owner | Phase 5, and again for each release candidate | REQ-PLT-03 fails, so G3 makes the first release a beta, and the release notes say "tested on M5 only" ([10](10-implementation-plan.md), section 8). Family Macs need no runner and no secrets, so a contributor's or a rented Mac can serve. |
| OQ-04 | A 48 GB Apple10 Mac | The authoritative form of G2; M-8 at 48 GB | The project's owner | Phase 4 | G2 runs on the reference Mac with `--gpu-budget-mib 37888` and the 16 GiB balloon, and `G2.md` says so; a later run on a 48 GB Mac replaces it ([10](10-implementation-plan.md), section 6) |
| OQ-05 | A GitHub-hosted macOS 27 arm64 image | `mac-host` on main/release pushes; X-8, B-2, B-3 and B-7 | GitHub; checked at phase 0 and during compatibility reviews | None: the plan works without it | While `MAC_HOSTED_RUNNER` is empty, the hosted CI job is skipped. Hardware acceptance runs on `strata-ref`. Release builds do not yet have an active workflow ([08](08-build-and-distribution.md), sections 12 and 14). Pull-request CI needs a dedicated runner that safely executes unreviewed code; until one is configured, developers run the Mac presets locally. An image older than macOS 27 cannot stand in. |
| OQ-06 | Developer ID signing and notarisation | B-5 and B-6; the signing decision | The project's owner, with an Apple Developer account | None: optional | Ad hoc releases ([08](08-build-and-distribution.md), section 11.5); setup strips the quarantine of the asset it installs ([07](07-server-setup-port.md), section 4.10) |
| OQ-08 | The licence review (work item B20) | REQ-BLD-06; the first release | A reviewer ([09](09-testing-and-validation.md), section 7.6) | Before the first release | `LICENSES.txt` as in [08](08-build-and-distribution.md), section 11.9 |
| OQ-09 | A `.pkg`, a `.dmg` or a Homebrew formula | Nothing | The project's owner | After the first release, if users ask | Setup is the only installer ([02](02-architecture.md), section 7.2) |
| OQ-10 | Whether to contribute the `normalize()` and `test_prebuilt_hip_zip` fixes ([07](07-server-setup-port.md), section 10.1) upstream | Two fewer `# MAC:` markers after each upstream contribution | The project's owner | Before the first release | They live in the project with `# MAC:` markers |
| OQ-11 | Whether to offer the Metal backend to upstream. Upstream closed Strata issue #193 with "Not planned for now ... a Metal backend would be a new port" (section 5). | The long-term cost of upstream compatibility reviews (RSK-01) | The project's owner, with upstream's owner | After the first release | A project with selective compatibility reviews and no upstream history merges (D14). The backend stays in its own folders (`src/metal/`, `cmake/metal/`, `tools/metal/`), so it can be offered whole. |
| OQ-12 | Contexts above 262,144 on a Mac. The engine keeps the RoPE and YaRN flags, the table is built on the host with upstream's YaRN code ([03](03-model-reference.md), section 6.7), and setup writes upstream's RoPE arguments ([07](07-server-setup-port.md), section 4.5.2). But no Mac test runs past 262,144 cells, and `test_setup_rope` runs with `MAC = False`. | REQ-FUN-05 beyond its flags; whether the Mac setup guide may offer such contexts | The project's owner, from the reference Mac's `--plan-only` result for Q2_0 at 524,288 cells | Phase 6 | Setup may offer a context above 262,144 wherever the plan admits it. The Mac setup guide must say these contexts remain untested until the reference Mac check passes. If the reference Mac's plan admits 524,288 cells, the proposed check is a weekly needle at 524,288 with `yarn`, in the Saturday slot of [09](09-testing-and-validation.md), section 7.2. |

### 2.2 Measurements that may never be made

[10](10-implementation-plan.md), section 3.7: a measurement that cannot be made, for lack of a machine, keeps its
default, and the gap is listed here and in the release notes.

| Measurement | Needs | Default that stands | What is lost without it |
| --- | --- | --- | --- |
| L-2 to L-8 and L-10 to L-15 on M1-M4 ([04](04-metal-kernels.md), section 11) | OQ-03 | The M5 values (ascending `simd_sum`, `constant` codebooks, the M5 shapes and tiles, IEEE denormals, no `fast/` kernel, one command buffer per chunk, `serial` dispatch, the indexed head read), with `simd` in place of `mpp` | Per-family tuning and the family evidence of REQ-PLT-03 |
| M-8 on 32, 36 and 48 GB ([05](05-memory-and-residency.md), section 10) | OQ-03, OQ-04 | W as 0.67 of RAM at 24 GB and 0.75 above, from published values ([07](07-server-setup-port.md), section 10.2) | The plan classes that setup shows on those Macs are forecasts |
| B-4 on M1; B-9 on M1 with 8 and 16 GB ([08](08-build-and-distribution.md), section 14) | An M1 Mac of each size, for a build only | Accept the vision helper's first-start cost; the text "5-10 minutes" | The build-time text on small Macs |
| X-8, B-2, B-3, B-7 | OQ-05 | Engine tests on the reference Mac only; releases built there | Hosted CI for engine tests and release builds |
| B-5, B-6 | OQ-06 | Ad hoc releases | Developer ID |
| R-1, R-5, T-1 to T-3, L-9 ([03](03-model-reference.md), section 10; [09](09-testing-and-validation.md), section 11) | OQ-02 | None that lets G1 pass: these define C1's tolerances | Gate C1, so G1 and G3 |
| T-8 | A replacement card | Run only when a card is replaced | None |

### 2.3 Conditional work items

L16-L23 ([04](04-metal-kernels.md), section 12.2) start only when their condition holds
([10](10-implementation-plan.md), section 3.8). Status at 2026-10-03: none started. At the first release, each item
not started is listed in the release notes with its condition.

| Item | Condition | Watched by |
| --- | --- | --- |
| L16, batched drafter prefill | The drafter prefill exceeds 5% of a long prompt's time, or R-3 shows that batched K and V accept better | R-3 |
| L17, indirect command buffers | H-4 above 1 ms | H-4 |
| L18, a second window arena | M-2 shows a gain, after C1-C4 pass | M-2 |
| L19, the prompt attention kernel | PF-1 fails, or L-13 shows attention above 25% of a chunk | L-13, PF-1 |
| L20, the selection kernel | L-13 shows selection above 25% | L-13 |
| L21, the device MTP round | Gates C1-C4 pass | The gates |
| L22, the expert decode once per chunk | L-13 shows the expert decode above 25% | L-13 |
| L23, `STRATA_QSA_DUMP` | A user of `tools/qsa_locality.py` asks for it | Issues |

The other items of [02](02-architecture.md), section 7.2, are outside the first release by decision, not by
condition, and are not repeated here. L17 and L18 appear there too, with their conditions.

### 2.4 Settled questions

None yet. A settled question moves here with its date, its decision and the pull request that recorded it.

---

## 3. The risk register

### 3.1 Summary

| ID | Risk | Likelihood | Impact | Owner | Section |
| --- | --- | --- | --- | --- | --- |
| RSK-01 | Upstream outpaces the project | High | Medium | The implementer; the project's owner for the fallback | 3.2 |
| RSK-02 | Upstream changes what the Mac reads or runs | High | Medium | The implementer | 3.2 |
| RSK-03 | Competitors close the 36-48 GB gap first | Medium | High | The project's owner | 3.2 |
| RSK-04 | The effort estimate is too low | Medium | Medium | The project's owner | 3.2 |
| RSK-05 | A key machine is lost | Low | High | The project's owner | 3.2 |
| RSK-06 | The Mac does not compute the same model as CUDA | Medium | High | The implementer | 3.3 |
| RSK-07 | Kernels behave differently on M1-M4 | Medium | Medium | The implementer | 3.3 |
| RSK-08 | Decode stays overhead-bound | Medium | Medium | The implementer | 3.4 |
| RSK-09 | The tensor ops (`mpp`) are unusable or slow | Low | High | The implementer | 3.4 |
| RSK-10 | Tiered mode misses G2 | Medium | High | The implementer | 3.4 |
| RSK-11 | A command buffer reaches the GPU time limit | Medium | Medium | The implementer | 3.5 |
| RSK-12 | The plan does not hold at run time | Medium | Medium | The implementer | 3.5 |
| RSK-13 | Memory pressure evicts or kills | Medium | Medium | The implementer | 3.5 |
| RSK-14 | Sparse resources or large mappings stall the system | Low | High | The implementer | 3.5 |
| RSK-15 | Small Macs give a poor first experience | High | Low | The project's owner | 3.5 |
| RSK-16 | A macOS or Xcode update changes behaviour | Medium | Medium | The implementer | 3.6 |
| RSK-17 | Power management slows the engine or the install | Medium | Low | The implementer | 3.6 |
| RSK-18 | macOS privacy and security get in the user's way | Medium | Low | The implementer | 3.6 |
| RSK-19 | Model files change or disappear | Low | Medium | The project's owner | 3.6 |
| RSK-20 | Mac regressions are found only after merging | High | Low | The implementer | 3.7 |
| RSK-21 | The self-hosted runner is compromised | Low | High | The project's owner | 3.7 |
| RSK-22 | The golden set cannot follow upstream | Medium | Medium | The CUDA PC owner | 3.7 |
| RSK-23 | A licence obligation is missed | Low | Medium | The release reviewer | 3.7 |

### 3.2 Project and ecosystem

**RSK-01. Upstream outpaces the project.** Likelihood high, impact medium.
- *What could happen.* Upstream made 532 commits in the 9 days to the baseline, about 140 a day lately
  ([00](00-feasibility-study.md), section 6, risk 1). Compatibility reviews may take longer than the project can
  sustain, leaving changes that affect the model, protocol, build or supported platforms unexamined. Mac users can then
  miss fixes, and the claim that Mac and PC users run the same product lapses.
- *Early signs.* The hours and classified items per review in `docs/mac/sync-log.md` (measurement H-6), unresolved
  drift findings, and the time since the last material compatibility review ([10](10-implementation-plan.md), section 4.1).
- *Mitigation.* A slim Metal driver with `synced from` markers (D9) and minimal, marked upstream edits with selective compatibility reviews (D14,
  [02](02-architecture.md), section 9). `check_drift.py` names each kind of change and its resolution
  ([10](10-implementation-plan.md), sections 4.2 and 4.3). The golden verdict requires a new recording for every
  `src/` or `include/` change except paths explicitly reviewed into `no_arithmetic` (currently only
  `src/platform/**`). Reviews are scoped to material changes; upstream history is not merged into this project.
- *If it happens.* If reviews average more than two working days over four weeks, or a material drift finding stays
  unresolved for two weeks, the project's owner assigns the review to a second person or narrows the comparison to
  upstream release tags. Release notes name the source revision used for compatibility. OQ-11 is then reconsidered.

**RSK-02. Upstream changes what the Mac reads or runs.** Likelihood high, impact medium.
- *What could happen.* A new pack version (`native_experts.txt` version 5, a new `conversions.json` field), a new
  catalogue file or tensor type, a new kernel path the model needs, or a changed protocol line.
- *Early signs.* `check_drift.py` during the compatibility review ([10](10-implementation-plan.md), section 4.2: flags, variables,
  protocol strings, copied ranges, CUDA kernel changes, models and revisions); the readers' refusal of an unknown
  version; `model_metadata_test` on a new file.
- *Mitigation.* Readers refuse what they do not know, with a message, rather than guess
  ([03](03-model-reference.md)). A file with a type that has no Metal kernel is refused before any buffer is planned
  (`unsupported_type_<T>`, [04](04-metal-kernels.md), section 2.5). New `INFO` keys are additive (D8).
- *If it happens.* The compatibility change adds the reader, kernel or string in a project pull request. Split larger work into
  separate project changes: setup keeps a new file hidden on Mac until its Metal kernel is ready, as Swift Q2_0 is today ([03](03-model-reference.md),
  section 2.2), and the kernel follows as its own task.

**RSK-03. Competitors close the 36-48 GB gap first.** Likelihood medium, impact high.
- *What could happen.* The port's distinct value is on 36-64 GB Macs ([00](00-feasibility-study.md), section 4.1).
  DwarfStar's PR #1047 reached 7.86 tok/s on a 32 GiB M1 Max with SSD expert streaming, and its open successor #1056
  adds I/O overlap, "up to 40%". oMLX v0.7.0 has expert offload; its issue #3614 says offload excludes MTP, but the
  release lists "Lightning MTP with expert offload" (#3935), so G2 checks again. LM Studio's Splash engine has an open
  request for this model. Any of them could ship a good 48 GB story before G2 (risk 2 of
  [00](00-feasibility-study.md), section 6).
- *Early signs.* The rows of section 5 checked again at each gate; a release note of one of these projects that
  names this model with expert streaming on 48 GB or less.
- *Mitigation.* G2 tests the 48 GB case at the end of phase 4, after about 23 of the 29 weeks
  ([10](10-implementation-plan.md), sections 1.2 and 2). The port keeps value that no general engine has: the same
  files as PC users, the n-gram table never in GPU memory, and the Strata product ([00](00-feasibility-study.md),
  section 4.2).
- *If it happens.* The fallback of G2 ([00](00-feasibility-study.md), section 7;
  [10](10-implementation-plan.md), section 3.5): the 48 GB class is marked beta, and phases 5 and 6 put product
  parity and long context on 64 GB first.

**RSK-04. The effort estimate is too low.** Likelihood medium, impact medium.
- *What could happen.* The plan's 29 weeks for one senior engineer ([10](10-implementation-plan.md), section 1.2)
  overrun. The kernel rewrite (16,895 lines of CUDA in 44 files, about 45 kernels to port) is the largest single
  job, and each kernel also needs its twin and oracle ([04](04-metal-kernels.md), section 10).
- *Early signs.* Phase 1's pace on L1-L8; the estimate redone at each gate from the measured pace.
- *Mitigation.* G1 comes after phase 2, so the riskiest work is measured first. Phases 1, 3 and 5 have independent
  tracks that a second engineer shortens ([10](10-implementation-plan.md), section 2). Conditional items stay out
  unless their condition holds (section 2.3).
- *If it happens.* The project's owner adds an engineer or moves the release date. Requirements are not dropped
  quietly: what is not done by G3 fails its check, and the first release goes out as a beta with the gaps listed
  ([10](10-implementation-plan.md), section 3.7).

**RSK-05. A key machine is lost.** Likelihood low, impact high.
- *What could happen.* The reference Mac fails or is withdrawn. It runs every hardware test, gate and release
  verification, and nothing else replaces it ([10](10-implementation-plan.md), section 8). The CUDA PC is lost
  (OQ-02).
- *Early signs.* None in advance.
- *Mitigation.* The runner's rules and setup are written down ([08](08-build-and-distribution.md), section 12.4), and
  the results folders and golden sets are copies, not the only record. The regression check runs the previous release
  and the engine under test in the same run ([09](09-testing-and-validation.md), section 5.4), so it carries over to
  another Mac unchanged.
- *If it happens.* The replacement is the same model (M5 Pro, 64 GB, 20-core GPU, 2 TB). T-4 and T-5 run again on
  it before its first gate or release, and the gate record names the new Mac. A new CUDA card is measurement T-8.

### 3.3 Correctness

**RSK-06. The Mac does not compute the same model as CUDA.** Likelihood medium, impact high.
- *What could happen.* Gate C1 (invariant I5) fails beyond the references' own spread: contraction or rounding that
  differs from nvcc's, a transcendental, the router of the 256-expert Coder file (R-1), drift that grows with depth
  (R-5), or a long-context failure like llama.cpp's open issue #28805 (EOS after one token at long context with
  quantised KV).
- *Early signs.* The twins of [04](04-metal-kernels.md), section 10, on real model rows; the ladder of C1.3 at G1;
  the weekly needles at 128K and 262K.
- *Mitigation.* Kernels compute upstream's arithmetic in upstream's order (D11). The numerics spike settled the
  tools for it: `fp contract(off)` works, `/` is correctly rounded in `safe/`, and `simd_sum` is deterministic
  ([appendix A.10](appendix-a-measurements.md)). L-9 reads nvcc's SASS, so the Mac contracts exactly the lines CUDA
  contracts. The host is built with `-ffp-contract=off` ([appendix A.12](appendix-a-measurements.md)).
- *If it happens.* The bisection of [09](09-testing-and-validation.md), section 10.1, finds the first state that
  differs, and phase 3 waits ([10](10-implementation-plan.md), section 3.3). A difference that cannot be removed
  becomes a listed deviation with its tolerance ([04](04-metal-kernels.md), section 3.9).

**RSK-07. Kernels behave differently on M1-M4.** Likelihood medium, impact medium.
- *What could happen.* A family adds `simd_sum` lanes in another order (L-3), flushes denormals (L-10), or prefers
  other shapes or codebook placement (L-4, L-5, L-7), so twins fail or decode is slow there. M1-M4 have no Neural
  Accelerators, so prefill uses the `simd` tile path (D13).
- *Early signs.* The family runs ([09](09-testing-and-validation.md), section 7.4), made before each release and
  after kernel changes.
- *Mitigation.* No bitwise contract depends on the `simd_sum` order ([04](04-metal-kernels.md), section 3.4). The
  family runs before a release run the draft's asset itself ([09](09-testing-and-validation.md), section 7.4). A shape
  that wins on a family becomes its default there.
- *If it happens.* The fix goes into the next release candidate. Until then the release notes name the family, and
  REQ-PLT-03 fails, which makes G3 a beta. Without family Macs at all, this risk cannot be watched (OQ-03).

### 3.4 Performance

**RSK-08. Decode stays overhead-bound.** Likelihood medium, impact medium.
- *What could happen.* Like the existing Mac engines, which use 12-24% of the bandwidth in decode, the port lands at
  30-45 tok/s (risk 3 of [00](00-feasibility-study.md), section 6), below the 50-75 tok/s with MTP that section 3.3
  of [00](00-feasibility-study.md) projects. That is still useful on 48 GB but misses REQ-PRF-01 (50 tok/s) and
  weakens the 64 GB case.
- *Early signs.* G1's plain decode (REQ-PRF-02, 20 tok/s); the `gpu window` stats line; H-4, L-13 and L-14.
- *Mitigation.* One decode path and one command buffer per window (D2, D3); 0.78 µs per dispatch and a serial
  encoder measured as the cheapest correct ordering ([appendix A.5](appendix-a-measurements.md),
  [A.11](appendix-a-measurements.md)); fused kernels; L17, L18 and M-2 ready if the profile points at them.
- *If it happens.* The rule of a slow G1 ([10](10-implementation-plan.md), section 3.3): between 15 and 20 tok/s,
  profile and fix first, which may start conditional items; below 15 tok/s with no clear cause, the stop rule of
  [00](00-feasibility-study.md), section 7. Above 20 tok/s but below the targets, the release ships with the measured
  numbers, and the failing requirement makes G3 a beta.

**RSK-09. The tensor ops (`mpp`) are unusable or slow.** Likelihood low, impact high.
- *What could happen.* The Metal Performance Primitives tensor ops that the M5 prefill uses are new. llama.cpp's open
  issue #27473 shows how they can be silently inert, there because the language version was not set. Without them,
  prefill on M5 falls back to `simd` tiles. WWDC26 session 232 puts the Neural Accelerators' matmul gain at about
  four times, so REQ-PRF-03 (450 tok/s on `ca-32k`) would likely fail.
- *Early signs.* L-6's five checks on each family, M5 included; `prefill_chunk_independent` under `mpp`; the
  prefill benchmark against [appendix A.6](appendix-a-measurements.md) (22-29 TFLOPS of half matmul already measured).
- *Mitigation.* The library is built with MSL 4.0 explicitly ([08](08-build-and-distribution.md), section 5.3), and
  the `simd` path exists for M1-M4 anyway, so a fallback costs no new code.
- *If it happens.* The fallback path runs on the failing family, the case goes to Apple through Feedback Assistant,
  and REQ-PRF-03 is reported with the measured rate.

**RSK-10. Tiered mode misses G2.** Likelihood medium, impact high.
- *What could happen.* On 48 GB, Q2_0 with MTP stays below 20 tok/s at 8K (REQ-PRF-05). Causes: a hit rate below the
  forecast with the shipped profile on Mac traces (M-9), a larger expert union per window than expected (R-2), event
  hops that cost more than measured (M-1), or prefill misses that the SSD cannot hide (M-10).
- *Early signs.* M-9's hit curve, recorded before G2 runs; R-2 in phase 2; M-1 in phase 1.
- *Mitigation.* The resident set comes from upstream's measured expert profile and adapts with swaps; the slot pool
  is sized from R-2; reads run on 8 or more threads, which gave 12.7-12.9 GB/s
  ([appendix A.4](appendix-a-measurements.md)). Tiered mode equals full mode bit for bit (I3), so its speed is the
  only question.
- *If it happens.* The fallback of G2 (RSK-03). The deferred optimisations of [05](05-memory-and-residency.md),
  section 11, become candidates for the next release.

### 3.5 Memory and the GPU

**RSK-11. A command buffer reaches the GPU time limit.** Likelihood medium, impact medium.
- *What could happen.* macOS ends a command buffer that runs too long (`MTLCommandBufferErrorTimeout`), and the limit
  is not documented. A 32,768-token prefill chunk may come close to it, and so may a tiered window that waits in
  `encodeWaitForEvent` for an SSD read, if waiting counts.
- *Early signs.* L-12 (the limit, and the largest chunk's time); H-5 (whether a wait counts); the times that
  `metal_gpu_time_test` reports.
- *Mitigation.* REQ-ROB-01: every command buffer below half the limit. The prompt path can encode each layer as its
  own command buffer ([04](04-metal-kernels.md), section 6.1). No GPU thread spins on CPU-written memory (D6), and a
  stalled window ends the engine within a bounded time ([06](06-engine-host-port.md), section 14).
- *If it happens.* A smaller default chunk, or per-layer command buffers. For tiered mode, per-layer command buffers
  cost about 48 × 11.2 µs, about 0.5 ms per window ([appendix A.5](appendix-a-measurements.md)), which fits.

**RSK-12. The plan does not hold at run time.** Likelihood medium, impact medium.
- *What could happen.* W (`recommendedMaxWorkingSetSize`) differs between Macs: 74-81% of RAM measured or reported;
  oMLX issue #3614 reports a wired cap of about 57.5 GB on 64 GB M1 and M2 Max. Users set `iogpu.wired_limit_mb`, and
  other apps use GPU memory. A plan made at setup then fails to allocate, or runs into GPU errors under load.
- *Early signs.* M-3 (the reserve with a browser and an IDE open); M-8 on other sizes; X-2 (whether `ioreg` sees
  other apps' GPU memory); `metal_alloc_fail_test`.
- *Mitigation.* The engine plans on the device (D4). The plan keeps a reserve of 700 MiB, and the engine never
  changes the wired limit (REQ-MEM-07). An allocation failure exits 1 with the plan printed (REQ-ROB-03), and the
  server explains the exit ([07](07-server-setup-port.md), section 6.5).
- *If it happens.* M-3 raises the default reserve. Setup's allocation guidance and [server failure notes](07-server-setup-port.md), section 6.5, cover the case.

**RSK-13. Memory pressure evicts or kills.** Likelihood medium, impact medium.
- *What could happen.* Under pressure, macOS compresses or evicts the no-copy file pages of full mode, or the
  anonymous arenas of tiered mode, and decode slows; or the engine is killed for memory (REQ-ROB-04).
- *Early signs.* M-4; the weekly `metal_memory_pressure_smoke` ([09](09-testing-and-validation.md), section 5.5);
  the server's jetsam text in user reports.
- *Mitigation.* M-4 decides between no-copy and copied weights in full mode; `STRATA_ARENA_LOCK` and M-7 for the
  arenas; one engine per Mac is the supported setup ([07](07-server-setup-port.md), section 6.8).
- *If it happens.* Full mode moves to copied weights, and wiring the resident set ([05](05-memory-and-residency.md),
  section 11) moves up.

**RSK-14. Sparse resources or large mappings stall the system.** Likelihood low, impact high.
- *What could happen.* Splash issue #220 reports sparse mapping and unmapping that took more than 30,000 ms, then
  WindowServer watchdog logouts, on macOS 26.4.1 to 27.0.1. A user logged out loses work in other apps. A related,
  measured effect: a GPU touch of a no-copy mapped buffer makes the whole buffer resident, 8 GiB in 1.2 s
  ([appendix A.3](appendix-a-measurements.md)).
- *Early signs.* A long stall at load or at the first window; a report of a logout while the engine runs.
- *Mitigation.* No sparse resources and no remapping: fixed slot pools filled with `pread`, and buffers made once at
  load (D5). Nothing that must not be resident is wrapped (I7, `metal_never_wrap_test`) (risk 4 of
  [00](00-feasibility-study.md), section 6).
- *If it happens.* The engine's buffer windows get smaller, the warm-up touches pages in smaller steps, and the case
  goes to Apple through Feedback Assistant.

**RSK-15. Small Macs give a poor first experience.** Likelihood high, impact low.
- *What could happen.* On 32-36 GB Macs, a third to a half of the experts come from the SSD and prompt processing is
  SSD-bound ([00](00-feasibility-study.md), section 4.1). Users judge the port by it.
- *Early signs.* M-8 and M-9 on those Macs; user reports.
- *Mitigation.* 36 GB is the next target and 32 GB a stretch target, by decision ([00](00-feasibility-study.md),
  section 5; [01](01-requirements.md), section 2.1). Setup shows the engine's plan class, asks for consent before it
  installs a slow or very slow class ([07](07-server-setup-port.md), section 4.5.1; REQ-SRV-03), and the planned
  Mac setup guide must state what each size gets (REQ-DOC-03).
- *If it happens.* As designed. The smaller slot pools of [05](05-memory-and-residency.md), section 11, come after the
  first release.

### 3.6 Platform and product

**RSK-16. A macOS or Xcode update changes behaviour.** Likelihood medium, impact medium.
- *What could happen.* A macOS 27 point release changes the GPU compiler that turns the library into GPU code, the
  timeout, or memory behaviour; an Xcode or Metal toolchain update changes the library's bytes.
- *Early signs.* A nightly change on both engines at once: the regression check runs the previous release beside the
  engine under test ([09](09-testing-and-validation.md), section 5.4), so a shift in both points at the OS. B-1 after
  an OS update.
- *Mitigation.* The runner's toolchain is updated on purpose, never automatically ([08](08-build-and-distribution.md),
  section 12.4). `BUILD.json` records `xcode` and `metal_toolchain`, and REQ-BLD-04 holds per toolchain. Each result
  file records the macOS build.
- *If it happens.* The nightly results are compared across the OS versions on the reference Mac; a regression in the
  OS goes to Apple, and the release notes name the affected macOS build.

**RSK-17. Power management slows the engine or the install.** Likelihood medium, impact low.
- *What could happen.* App Nap, timer coalescing or background throttling stretch the engine's waits (H-8) or slow a
  server in a hidden window (X-5); a MacBook sleeps during a two-hour download (X-4); a laptop throttles over long
  runs (T-5).
- *Early signs.* H-8, X-4, X-5 and T-5.
- *Mitigation.* The engine holds an activity during requests ([06](06-engine-host-port.md), section 2.2); setup runs
  `caffeinate` during installs ([07](07-server-setup-port.md), section 4.13); the benchmarks refuse to run off mains
  power or in Low Power Mode, and pause between runs only if T-5 finds drift ([09](09-testing-and-validation.md),
  section 5.1).
- *If it happens.* The server holds an activity too, or setup adds `caffeinate -s` on AC, as X-4 and X-5 decide.

**RSK-18. macOS privacy and security get in the user's way.** Likelihood medium, impact low.
- *What could happen.* A Local Network prompt when the server listens on all interfaces (X-7); Gatekeeper on a ZIP
  downloaded in a browser; a checkout in a protected or synced folder; an organisation's policy that blocks ad hoc
  signed code.
- *Early signs.* X-7; the first-run review of the Mac setup guide (S20); user reports.
- *Mitigation.* Setup strips the quarantine of what it installs, and `git clone` sets none
  ([07](07-server-setup-port.md), section 4.10). The data folder moves to `~/Strata-data` out of protected and synced
  folders (REQ-SRV-07). The firewall and quarantine messages are specified in
  [07](07-server-setup-port.md), sections 4.10 and 6.3.
- *If it happens.* The texts are updated. For managed Macs, Developer ID (OQ-06) is the answer.

**RSK-19. Model files change or disappear.** Likelihood low, impact medium.
- *What could happen.* A Hugging Face repository of the catalogue is changed, gated or removed. The research could
  not read one related repository, Swift's base model `ukisai/Swift-Qwen3.8-Flash-Next`, through the Hugging Face
  API; it is probably gated ([audit D](audit/D-external.md), section 1.1).
- *Early signs.* The CI Content-Length check of the geometry files against the pinned URLs
  ([07](07-server-setup-port.md), section 4.4.3); setup's download errors.
- *Mitigation.* Upstream pins each file by revision (`HF_REVISIONS`), and the Mac uses the same pins; the test models
  are checked by SHA-256 (`tests/metal/models.json`). This risk is shared with upstream's PC users.
- *If it happens.* A compatibility review adopts the relevant upstream catalogue change. Until then, setup hides a file it cannot download.

### 3.7 Build, CI and release

**RSK-20. Mac regressions can reach main before hardware CI finds them.** Likelihood high, impact low.
- *What could happen.* The CI workflow does not run on pull requests. A failure in the Mac build or tests may first
  appear in `mac-hardware.yml` after the change reaches main.
- *Early signs.* A red `mac-hardware.yml` run on a push to main, or a local Mac preset failure before review.
- *Mitigation.* Developers run the `mac-host` and `mac-metal` presets before review
  ([08](08-build-and-distribution.md), section 13). Configure a dedicated runner that safely executes unreviewed
  pull-request code before enabling PR CI; `mac-hardware.yml` validates main on the reference Mac.
- *If it happens.* Fix or revert the breaking commit. Require the hosted Mac job before merge once safe PR CI is enabled.

**RSK-21. The self-hosted runner is compromised.** Likelihood low, impact high.
- *What could happen.* `strata-ref` runs workflow code with its user's rights, and it builds releases while no hosted
  runner exists, with the signing secrets if they are set.
- *Early signs.* A workflow run that no one started; a change to the runner user's files outside a run.
- *Mitigation.* The rules of [08](08-build-and-distribution.md), section 12.4 (REQ-BLD-09): no pull-request
  trigger, a dedicated non-admin user, read-only models, approval for outside collaborators. A person approves each
  publication, and `verify_release.py` rebuilds and compares the asset ([08](08-build-and-distribution.md),
  section 11.8).
- *If it happens.* Take the runner offline; revoke its registration, the repository tokens and any signing
  certificate; rebuild the Mac's user from scratch; run `verify_release.py` again on each published release; and tell
  users in a release note.

**RSK-22. The golden set cannot follow upstream.** Likelihood medium, impact medium.
- *What could happen.* A compatibility review whose verdict is `golden: record again` needs the CUDA PC (about two hours per model,
  T-7), and the PC or its owner is not available that week. A change of the card or of upstream's CUDA build changes
  the reference itself.
- *Early signs.* The golden verdict in each compatibility pull request ([10](10-implementation-plan.md), section 4.1,
  step 4).
- *Mitigation.* The set is recorded again for every `src/` or `include/` change except paths reviewed into
  `no_arithmetic` ([09](09-testing-and-validation.md), section 2.5). The current exception is only
  `src/platform/**`; `valid_for` carries the existing set across other compatibility updates. T-8 and L-9 cover a new card.
- *If it happens.* The project pull request waits for the set and is not merged to `main` until the required golden run passes. If
  the CUDA PC is unavailable, keep the review open and arrange a qualified reference run; see RSK-01. Repeated, this is RSK-01.

**RSK-23. A licence obligation is missed.** Likelihood low, impact medium.
- *What could happen.* Kernels adapt ggml-metal's codebooks and kernel structures (MIT), DwarfStar (MIT) is a
  reference, and `strata-vision` contains llama.cpp and its vendored files. If code from DwarfStar, MTPLX or oMLX
  (Apache-2.0, which needs attribution and a NOTICE) is ever adapted, its notice is needed too. A missing notice
  means a release must be replaced.
- *Early signs.* The licence review (B20, OQ-08); a new vendored file adopted in a compatibility update.
- *Mitigation.* `LICENSES.txt` holds every notice (REQ-BLD-06), put together from the licence files and the
  excerpts in `tools/metal/licenses/` ([08](08-build-and-distribution.md), section 11.9). Adapted kernel code is
  recorded with its source, commit and licence in `src/metal/kernels/THIRD_PARTY`
  ([08](08-build-and-distribution.md), section 5.1).
- *If it happens.* A new release with the notice. The published tag is not reused ([10](10-implementation-plan.md),
  section 5.1).

---

## 4. Risks retired by measurement

The spikes of appendix A, run on 2026-10-02 and 2026-10-03 (A.9 by HTTP range requests, the rest on the reference
Mac), settled these before the first task. Each result is already a decision in the design.

| Risk | Result | Where | What it decided |
| --- | --- | --- | --- |
| The GPU cannot read experts fast enough in decode | 238-282 GB/s; random 1.31 MiB gathers at 228-230 GB/s | [A.2](appendix-a-measurements.md) | Experts are read in any order (D1) |
| A CPU expert path is needed for bandwidth | About 284 GB/s for CPU and GPU together | [A.2](appendix-a-measurements.md) | No CPU expert path (D1) |
| The n-gram table needs GPU memory | 76K-173K IOPS of random 16 KiB reads | [A.4](appendix-a-measurements.md) | `pread` of 16 rows per token (D10) |
| The SSD cannot feed an expert tier | 12.7-12.9 GB/s at 1.31 MiB with 8 or more threads | [A.4](appendix-a-measurements.md) | Tiered mode (D5) |
| Wrapping a mapped file is cheap | A GPU touch makes the whole no-copy buffer resident: 8 GiB in 1.2 s | [A.3](appendix-a-measurements.md) | The never-wrap rules (D10, I7) |
| One buffer can hold the weights | `maxBufferLength` 38.88 GiB | [A.1](appendix-a-measurements.md) | Windows over the files (REQ-MEM-08) |
| Launch overhead needs a CUDA-graph equivalent | 0.78 µs per dependent dispatch; 11.2 µs per round trip | [A.5](appendix-a-measurements.md) | One command buffer per window, re-encoded (D3) |
| Prefill lacks compute | 22-29 TFLOPS of half matmul | [A.6](appendix-a-measurements.md) | The `mpp` prefill path |
| The toolchain cannot give a reproducible asset | Identical bytes from two checkouts and two build folders | [A.8](appendix-a-measurements.md) | The build of [08](08-build-and-distribution.md) |
| The files are not what the documents assume | Headers, shards, tensor types and PLE rows read from the pinned revisions | [A.9](appendix-a-measurements.md) | [03](03-model-reference.md), sections 2-4 |
| Metal cannot follow upstream's arithmetic | `fp contract(off)` works; `/` correctly rounded in `safe/`; deterministic `simd_sum` | [A.10](appendix-a-measurements.md) | Upstream's arithmetic in upstream's order (D11) |
| Untracked buffers on one queue are ordered | Wrong in 20 of 20 runs without a fence | [A.11](appendix-a-measurements.md) | One queue with a fence (D7) |
| The host compiler keeps upstream's arithmetic | Apple clang fuses `a*b + c` unless told not to | [A.12](appendix-a-measurements.md) | `-ffp-contract=off` for host code and oracles ([08](08-build-and-distribution.md), sections 4.3 and 6.5) |

---

## 5. Sources

Verified on 2026-10-02 unless a row says otherwise. Numbers are as their sources publish them, not measured again
here. The full research notes are [audit/D-external.md](audit/D-external.md). Rows that a gate decision depends on
are checked again at that gate (section 1.4).

### 5.1 Engines that run the model on a Mac

| Source | What it supports | Cited in |
| --- | --- | --- |
| llama.cpp PR #27742, "model: add Qwen3.8-Flash-Next (qwen4exp)", merged 2026-08-27, first in build b10660: https://github.com/ggml-org/llama.cpp/pull/27742 | Official support | [00](00-feasibility-study.md), section 3.1 |
| llama.cpp v0.5.0, 2026-09-23: https://github.com/ggml-org/llama.cpp/releases/tag/v0.5.0 | The stable release with the model's Metal ops | [00](00-feasibility-study.md), section 3.1 |
| llama.cpp issue #29465 (open): the per-file mapping counts the lazy-read table against the GPU working set, OOM on an M4 Max 128 GB: https://github.com/ggml-org/llama.cpp/issues/29465 | The never-wrap rule; the 64 GB comparison | [00](00-feasibility-study.md), sections 4.1 and 4.2 |
| llama.cpp issue #28805 (open): EOS after one token at long context on Metal: https://github.com/ggml-org/llama.cpp/issues/28805 | RSK-06 | Section 3.3 |
| llama.cpp issue #27473 (open): the Metal tensor API silently inert on M5: https://github.com/ggml-org/llama.cpp/issues/27473 | RSK-09 | Section 3.4 |
| DwarfStar (antirez/ds4, MIT): https://github.com/antirez/ds4, and its model notes https://github.com/antirez/ds4/blob/main/docs/QWEN38_FLASH_NEXT.md | Q2 at 41.73 GiB resident; Metal as the primary target, for 96 GB or more; M3 Ultra prefill 1,331 tok/s and MTP decode 64.8-90.5 tok/s | [00](00-feasibility-study.md), sections 3.1, 3.3 and 4.1 |
| DwarfStar PR #1047 (closed, superseded by #1056): https://github.com/antirez/ds4/pull/1047 | SSD expert streaming on a 32 GiB M1 Max: MTP generation from 4.55 to 7.86 tok/s | [00](00-feasibility-study.md), section 4.1; RSK-03 |
| DwarfStar PR #1056 (open): https://github.com/antirez/ds4/pull/1056 | I/O overlap and kernel work, "up to 40%" | [00](00-feasibility-study.md), sections 3.1 and 6; RSK-03 |
| MTPLX (Apache-2.0): https://github.com/youssofal/MTPLX, and its model card https://huggingface.co/Youssofal/Qwen3.8-Flash-Next-MTPLX-Optimized-Speed | M5 Max 128 GB: 73.5 tok/s with MTP, 43.8 plain, prefill 1,453 tok/s; 128 GB or more recommended | [00](00-feasibility-study.md), sections 3.1 and 3.3 |
| oMLX (Apache-2.0): https://github.com/jundot/omlx; v0.7.0, 2026-09-30: https://github.com/jundot/omlx/releases/tag/v0.7.0 | MTP, expert offload and the n-gram table on the SSD; "Lightning MTP with expert offload" (#3935), which issue #3614 contradicts | [00](00-feasibility-study.md), section 3.1; RSK-03 |
| oMLX issue #3614: https://github.com/jundot/omlx/issues/3614 | A wired cap of about 57.5 GB on 64 GB M1 and M2 Max | RSK-12 |
| mlx-vlm (MIT) v0.6.17, 2026-08-26, PR #2032: https://github.com/Blaizzy/mlx-vlm/releases/tag/v0.6.17 | Official support; `ple_storage.py` reads rows on demand | [00](00-feasibility-study.md), section 3.1 |
| mlx-lm PR #1788 (open): https://github.com/ml-explore/mlx-lm/pull/1788 | The model is not in a released mlx-lm | [audit D](audit/D-external.md), section 3.1 |
| LM Studio: https://lmstudio.ai/models/qwen/qwen3.8-flash-next | "Minimum system memory 96 GB" | [00](00-feasibility-study.md), sections 3.1 and 4.2 |
| Ollama: https://ollama.com/library/qwen3.8-flash-next/tags | The MLX build is 105 GB | [00](00-feasibility-study.md), section 3.1 |

### 5.2 Other projects

| Source | What it supports | Cited in |
| --- | --- | --- |
| Splash (Inco AI): https://github.com/incoai/splash; issue #220: https://github.com/incoai/splash/issues/220 | Sparse mapping above 30,000 ms and WindowServer logouts on macOS 26.4.1 to 27.0.1; an open model request (issue #39) for this model | [00](00-feasibility-study.md), section 6; RSK-03, RSK-14 |
| LM Studio's Splash engine: https://lmstudio.ai/blog/splash-engine and https://lmstudio.ai/changelog | Splash ships in LM Studio for M3 and later | RSK-03 |
| jinzy0623/Strata-macOS, branch `macos-port`, created 2026-09-30: https://github.com/jinzy0623/Strata-macOS | The only Mac-related project of 444: a UI and Python port on llama-cpp-python, with SSD streaming, the expert cache and MTP not wired to Metal | [00](00-feasibility-study.md), section 4.4 |
| Strata issue #193, closed by the owner: https://github.com/Niko1221/Strata/issues/193#issuecomment-5908103145 | "Not planned for now ... a Metal backend would be a new port" | OQ-11 |
| llama.cpp discussion #26938 and ExpertCache: https://github.com/ggml-org/llama.cpp/discussions/26938 | `madvise`, copied LRU slots and static prewarm failed; explicit slots work | [audit D](audit/D-external.md), sections 4.1 and 9.3 |

### 5.3 Apple

| Source | What it supports | Cited in |
| --- | --- | --- |
| macOS 27, released 2026-09-14: https://www.apple.com/newsroom/2026/09/major-updates-for-apples-software-platforms-are-now-available/ | The target OS | [00](00-feasibility-study.md), section 5 |
| Metal, what's new: https://developer.apple.com/metal/whats-new/ | Quantised tensor formats; Metal Performance Primitives on the M5 Neural Accelerators | [audit D](audit/D-external.md), section 6.1 |
| WWDC26 session 330, "Optimize custom machine learning operations with Metal tensors": https://developer.apple.com/videos/play/wwdc2026/330/ | Tensor ops on every Apple GPU generation; the Neural Accelerator in each M5 shader core | [audit D](audit/D-external.md), section 6.1 |
| WWDC26 session 232, "Run local agentic AI on the Mac using MLX": https://developer.apple.com/videos/play/wwdc2026/232/ | Matmul about four times faster on M5 than M4 | RSK-09 |
| waired-agent issue #857: https://github.com/waired-ai/waired-agent/issues/857 | `recommendedMaxWorkingSetSize` at 74.0% of RAM on an M4 16 GiB and 78.0% on an M5 Pro 48 GiB | [00](00-feasibility-study.md), section 4.1; RSK-12 |

The model files and their pinned revisions are in [03](03-model-reference.md), section 2.2, and upstream's own numbers
in `bench/results/2026-09-29-speed-0126` at the baseline ([09](09-testing-and-validation.md), section 5.3).

---

## 6. Where the details are

| Topic | This document | Elsewhere |
| --- | --- | --- |
| How the register is kept | 1 | — |
| Open questions and missing resources | 2.1 | [07](07-server-setup-port.md), section 14; [08](08-build-and-distribution.md), section 14; [10](10-implementation-plan.md), section 8 |
| Measurements that may not be made | 2.2 | [10](10-implementation-plan.md), section 7 |
| Conditional work items | 2.3 | [04](04-metal-kernels.md), section 12.2; [10](10-implementation-plan.md), section 3.8 |
| What the first release leaves out | — | [02](02-architecture.md), section 7.2 |
| Project and ecosystem risks | 3.2 | [00](00-feasibility-study.md), sections 4 and 6 |
| Correctness risks | 3.3 | [09](09-testing-and-validation.md), sections 4 and 10 |
| Performance risks | 3.4 | [00](00-feasibility-study.md), section 3.3; [appendix A.7](appendix-a-measurements.md) |
| Memory and GPU risks | 3.5 | [05](05-memory-and-residency.md), sections 9-11; [06](06-engine-host-port.md), sections 14 and 15 |
| Platform and product risks | 3.6 | [07](07-server-setup-port.md), section 14 |
| Build, CI and release risks | 3.7 | [08](08-build-and-distribution.md), sections 11, 12 and 14 |
| What the measurements settled | 4 | [appendix-a-measurements.md](appendix-a-measurements.md) |
| External sources | 5 | [audit/D-external.md](audit/D-external.md) |
| The gates and their fallbacks | — | [00](00-feasibility-study.md), section 7; [10](10-implementation-plan.md), sections 3 and 6 |
