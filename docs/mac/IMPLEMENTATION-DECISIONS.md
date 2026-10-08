# Implementation decisions and review items

Status: working log, 2026-10-08.

This file records choices made while implementing the plan. A recorded choice does not
make a hardware gate, a human review, or an external resource available. Update the
relevant specification in the same commit when a decision changes a normative
requirement.

## Decisions

### D-IMPL-001: Start from the documented upstream commit

- **Decision:** Use upstream Strata `v0.1.34`, commit
  `1678de333d0e0711bc414ad992b640e1a37dd814`, as the source baseline.
- **Reason:** This is the baseline named throughout the specification, and the
  included Metal build skeleton is explicitly verified against it. Starting from
  upstream `main` would introduce unclassified source drift before Phase 0.
- **Review:** Revisit when an upstream compatibility review intentionally advances
  `UPSTREAM_SYNC`.

### D-IMPL-002: Keep the fork local until its owner is known

- **Status:** Superseded by D-IMPL-225 on 2026-10-07. The project has an independent repository identity; its
  publication destination and release-test repository remain open under OQ-01.
- **Decision:** Work in this local repository on the requested `codex` branch. Use
  `strata.app` as the planned repository name, but leave `{FORK}` placeholders in
  published-facing specifications and do not invent a GitHub owner or test
  repository.
- **Reason:** The plan records the slug `strata.app` as planned but leaves its owner
  and the release test repository open (OQ-01). This checkout has no Git remote.
  Inventing either value would create misleading clone, issue, workflow, and release
  URLs.
- **Plan status:** The user requested local implementation to proceed without waiting
  for these human choices. Work is therefore being done as a local bootstrap despite
  the formal Phase 0 precondition. This checkout does not claim Phase 0 completion or
  a conforming published fork.
- **Review required:** The repository owner must confirm the GitHub `owner/repository`
  and test repository before Phase 0 can be accepted or those URLs and release
  workflows are used externally.

### D-IMPL-003: Treat local hardware checks as evidence, not as gate completion

- **Decision:** Use this host for reference-Mac build and Metal checks. Do not claim
  CUDA-golden, family, or release acceptance until their separate machine and model
  requirements are met.
- **Reason:** `system_profiler` confirms an M5 Pro, 64 GB, 20-core GPU, macOS 27.0
  build `26A428`, and Xcode 27.0, matching the reference Mac's key compute
  specifications. The mounted system volume has about 287 GiB free, no external
  model disk is mounted, and no CUDA reference PC is available. CMake 4.3.2 is
  installed globally, while Python 3.12.13 is available and Ninja is not; use the
  Python 3.12 interpreter and install pinned build tools in the project venv instead
  of changing global tools.
- **Review required:** Confirm model assets and result storage before model runs, and
  provide the CUDA reference. Without the required CUDA reference, C1 and G1 remain
  blocked and must not be waived or recorded as passing.

### D-IMPL-004: Keep self-hosted runner activation behind its documented review

- **Decision:** Runner configuration and workflow files may be prepared locally, but
  no first run on a self-hosted `strata-ref` runner is considered approved by this
  implementation.
- **Reason:** B21 explicitly requires a second person to check the runner rules
  before the first workflow runs on that machine. This checkout has no GitHub remote
  or configured runner. Manual hardware runs are restricted to `main` and PRs
  cannot target the configurable Mac runner, but actor permissions and runner-group
  workflow restrictions still need owner configuration.
- **Review required:** A second person must review the runner restrictions in
  `docs/mac/08-build-and-distribution.md` section 12.4 before the first hardware
  workflow is enabled.

### D-IMPL-005: Let Phase 0 tests patch the not-yet-implemented Mac setting

- **Decision:** The 13 upstream test modules patch `setup.MAC` with
  `mock.patch.object(..., create=True)` for now.
- **Reason:** Upstream `v0.1.34` has no `setup.MAC`; the Mac setup constant is added
  by S3 in Phase 3. Without `create=True`, the Phase 0 tests fail before running.
  The temporary attribute changes only the test process and keeps those existing
  tests on their documented PC path.
- **Review required:** Remove `create=True` in the same change that implements S3,
  so the tests fail if the production setting is ever removed.

### D-IMPL-006: Fix the baseline MCP failures instead of hiding them

- **Decision:** Bring forward the MCP process-identity and data-folder fixes from
  S15 so the full Python test runner can execute every module without skips or
  exclusions for known failures.
- **Reason:** The first full run showed the documented zombie and temporary-folder
  failures. Marking those tests as expected failures would let real regressions
  pass unnoticed. `serve/macos.py` comes from the plan's reviewed reference spike;
  the MCP server now uses it for darwin process identity and applies the specified
  system-folder exceptions.
- **Status:** The full Python runner now exits successfully for all 37 discovered
  modules on this Mac. This closes the test failures, but does not complete S15:
  Mac hardware reporting and engine-driven model recommendations remain.

### D-IMPL-007: Correct the R1 tensor-count acceptance from the frozen headers

- **Decision:** Treat 1,224 as the total tensor count for each model across all
  shards, not as the count in every shard file. Preserve the actual per-shard
  counts in the fixture documentation and acceptance test.
- **Reason:** The frozen headers show 1,223 tensors in each ISTA/Coder shard 1
  plus the shared one-tensor shard 2; the Swift pairs contain 342 + 882 and
  317 + 907 tensors. The two summary outputs reproduce from these fixtures.
  Counting 1,224 in each file would reject the supplied evidence and misstate
  the model format.
- **Review required:** Confirm the frozen headers still correspond to the
  intended reference model packs before changing or replacing them.

### D-IMPL-008: Resolve macOS path spelling only when filesystem identity agrees

- **Decision:** Keep path comparisons case-sensitive unless `samefile()` proves
  that alternate component spellings name the same object.
- **Reason:** Blind lowercasing would merge distinct directories on
  case-sensitive APFS and could incorrectly admit an external data directory.
  The check also preserves equivalent spellings on the usual
  case-insensitive volume.
- **Review required:** Keep the case-sensitive sibling regression test when
  changing path normalization; verify it on both case-sensitive and
  case-insensitive APFS.

### D-IMPL-009: Avoid signaling a process group after its leader has exited

- **Decision:** Retain the start-time identity check for a live process leader.
  Once code observes that leader has exited, only probe whether its old
  process-group ID still exists; do not send a later cleanup signal to that
  group. Retry identity probes for new children and stop an untracked session's
  process group while its `Popen` leader is still alive and unreaped, rather
  than persisting it with an empty identity. When a leader exits before
  tracking, report a remaining group without signaling it.
- **Reason:** The POSIX APIs used by this Python 3.10+ server accept a PID or
  process-group ID for signaling; they do not accept the observed process
  start-time token. A process can exit and the identifier can be reused between
  the check and signal, including between the check and `killpg()` in
  `proc_kill_tree()`. The post-exit path no longer signals the old group, but
  the live-leader check-to-signal race remains. In particular, after the
  session leader exits there is no reliable way to prove that an extant group
  with the same numeric ID is still its former group. Leaving such a group
  untouched avoids signaling an unrelated process, at the cost of reporting
  possible leftovers for manual inspection.
- **Review required:** Assess an OS-specific stable process-handle mechanism
  before making a stronger safety claim. Keep this limitation visible until
  such a mechanism is implemented and tested.

### D-IMPL-010: Normalize only complete path tokens in build graphs

- **Decision:** Replace a configured source or build root only when its
  surrounding text identifies a complete path token or an explicitly supported
  CMake/compiler path argument.
- **Reason:** A suffix match could turn a distinct external path such as
  `/deps/tmp/upstream/include` into the same token as `/tmp/upstream/include`,
  hiding a real build-graph difference. Spaces and prefix flags need explicit
  handling so valid Ninja and compile-command paths still normalize, including
  paths embedded in `-ffile-prefix-map` values.
- **Review required:** Retain the embedded-parent, sibling, whitespace and
  attached `-I`/CMake-flag and `-ffile-prefix-map` regression cases.

### D-IMPL-011: Keep upstream drift snapshots conservative

- **Decision:** Snapshot the upstream parser flags, literal `STRATA_*` reads,
  emitted protocol strings, copied ranges, CUDA kernel files, model names and
  Hugging Face revisions. Treat `src/platform/**` as non-arithmetic for the
  golden-set verdict.
- **Reason:** These snapshots cover the changes that can silently invalidate a
  port or its frozen numerical inputs. The platform files select file and
  memory operations; they do not change the model arithmetic that the golden
  logits compare. All other `src/` and `include/` changes conservatively call
  for a new golden set.
- **Review required:** Reassess the `src/platform/**` classification if a
  platform change starts changing tensor values or execution order. Keep the
  per-file kernel hashes and the model/revision inventory in `drift.json`.

### D-IMPL-012: Require the Metal engine plan for Mac recommendations

- **Decision:** Report Apple RAM and Metal device facts directly. On macOS,
  show a model recommendation or fit only when the installed engine and setup
  expose the Metal planner; otherwise report that no safe recommendation is
  available. Do not substitute the PC RAM/VRAM formulas or equate the Metal
  working set with a model fit. The Mac model table omits PC RAM fit values,
  and the legacy install-plan `ram` field is null; the general `fit` field
  carries the selected Metal PLAN result. Exclude Unsloth from Mac
  recommendations and installation because its kernels are unavailable in
  this port. Show the 32K menu PLAN, and check 8K when 32K is `none` or
  `unsupported`; allow a model that fits only at 8K as a last-resort choice.
  Across eligible families, preserve `FAMILIES` insertion order; within a
  family, prefer models that fit at 32K, then fall back to the first model
  that fits at 8K.
- **Reason:** Unified memory and the model's expert residency are evaluated by
  the engine's PLAN output. This checkout's upstream `setup.py` baseline has no
  `metal_plan()` yet, so guessing from total RAM or `recommendedMaxWorkingSetSize`
  could recommend a model that does not fit. The plan's context menu explicitly
  recommends 8K when 32K does not fit but 8K does, so the model menu and MCP
  must preserve that smaller-context fallback instead of making the documented
  context unreachable. The MCP checks both eligibility and the exact requested
  context. Contexts at or below 8K use FP16 KV regardless of a requested
  higher-context KV mode; the preview and command use that effective setting.
- The specification ranks models within a family but does not define how the
  MCP should rank families against each other. Retaining setup's family-table
  order keeps the existing Qwen-first default and is deterministic.
- **Review required:** Integrate the MCP path with S3-S5's setup planner and
  geometry files, then verify its recommendation and every model-table entry
  against `setup.py --check` on the reference Mac, especially the 8K-only
  fallback.

### D-IMPL-013: Check the complete release history and publishing target

- **Decision:** The B13 remote check retrieves all published release pages,
  verifies that `setup.RELEASE_REPOSITORY`, the checkout's fetch and push URLs, and
  `GITHUB_REPOSITORY` identify the same repository. It accepts only annotated
  release tags on the current release-branch head, requires `METAL_REV` to
  increase by exactly one, rejects Metal prereleases, and permits a
  post-publication release PR only when the tag is latest, annotated, has the
  engine asset and targets the PR event's actual head SHA. Main also verifies
  the tag against the retained release branch head. It rejects unpublished
  release PRs, missing or lightweight published tags, malformed published tags
  containing `-metal`, and invalid release-history responses. The documented
  create and publish commands name their target repository explicitly.
- **Reason:** `gh release list`'s default window can omit an older release with
  a higher Metal revision. Checking one repository while publishing to another
  can reuse a revision; checking only that a revision is higher can skip values;
  silently ignoring malformed Metal-looking tags can reuse a revision;
  release PRs run after publication; a base repository's same-named branch does
  not prove a fork PR's head commit; and a lightweight or stale tag does not
  meet the release procedure's `git tag -a` on branch head requirement. Retaining
  the current release branch lets main recheck that the published tag was not
  retargeted after the tag workflow.
- **Review required:** Keep regression coverage for more than 30 releases with
  an older higher revision, a mismatched publishing repository or push URL, a
  revision gap, a prerelease, malformed or empty Metal tags, null GitHub API
  responses, missing or lightweight release tags, a stale tag on main, an
  unpublished release PR and a post-publication PR whose head differs from the
  base branch. Retain the current release branch until its successor is merged.
  Resolve OQ-01 before enabling remote release checks.

### D-IMPL-014: Defer remote release-history checks until the repository is known

- **Decision:** Always run local release-version validation in CI. Run the
  published GitHub release-history check only when `setup.RELEASE_REPOSITORY` is configured.
  When unset, ordinary `main` development checks emit a notice and defer remote
  history; release branches and tags fail closed.
- **Reason:** OQ-01 is unresolved and this checkout has no remote. Failing every
  Linux CI run because the publishing repository is intentionally unset would
  block unrelated implementation work, while guessing a repository could check
  or publish against the wrong release history. Release preparation cannot
  proceed without checking the destination's revision history.
- **Review required:** Set and verify the intended `setup.RELEASE_REPOSITORY`, then require
  `check_release_policy.py --remote` in CI before treating release-history
  validation as active.

### D-IMPL-015: Keep hardware acceptance fail-closed and recover queued runs

- **Decision:** A scheduled run and a manual dispatch with a selected `day`
  require model assets, model-labelled tests, the Metal validation-layer run,
  the vision build and acceptance tooling. Both serialized workflows use
  `queue: max`. If a hardware or release run is canceled at the pending-run
  limit, wait for capacity and rerun the missed weekly slice or original tag
  workflow within 30 days.
- **Reason:** Marking a run green when a planned validation layer is absent
  hides missing coverage. The concurrency queue protects a running hardware
  job from replacement, but the platform cancels new runs once its pending
  limit is reached; waiting for queue capacity before rerunning the original
  work within the platform's rerun period preserves evidence without
  weakening serialization.
- **Review required:** Keep the missing-prerequisite and day-range workflow
  assertions. Before enabling the runner, verify the external runner-group
  restriction to the reviewed `main` workflows and trusted dispatch actors.

### D-IMPL-016: Keep no-copy mapping ownership with the loader

- **Decision:** `Buffers::wrap_no_copy` wraps an existing page-aligned mapping
  without taking ownership or installing a deallocator. The loader retains the
  mapping until the owning `Context` has drained its queue and released its
  Metal buffers.
- **Reason:** The wrapped `MTLBuffer` points directly at caller-provided pages;
  unmapping them while the GPU can still access them would invalidate the
  buffer. Keeping mapping lifetime in the loader makes file-descriptor and
  mapping cleanup explicit. `Context` drains committed queue work before
  `Buffers` removes the residency set or releases its Metal objects.
- **Review required:** When H5 implements `WeightStore`, verify its teardown
  order keeps every mapping alive until its `Context` has released all wrapped
  buffers. Keep the no-copy test's `Context`-before-`munmap` order.

### D-IMPL-017: Reject potentially synced model folders in MCP

- **Decision:** The MCP install tool rejects explicit and configured/default
  data folders inside iCloud Drive, `~/Library/CloudStorage`, `~/Desktop`,
  `~/Documents`, or `~/Downloads`. This does not change the direct setup CLI's
  documented confirmation flow for explicit cloud-storage paths.
- **Reason:** Desktop & Documents can be included in iCloud Drive sync, and
  these user folders also have macOS privacy restrictions. A generic
  `confirm=true` approves the install plan and download; it does not clearly
  confirm that the user intends to sync 70–120 GB of model files. Failing
  closed directs MCP installs away from the known system-managed roots. macOS
  does not provide a general signal for every arbitrary folder a third-party
  sync client may have enrolled.
- **Review required:** Keep tests for every root, configured/default paths,
  and symlink aliases. Keep the limitation visible: user-selected sync folders
  outside the guarded roots are not detected. Reassess if the install
  confirmation protocol gains an explicit, clearly displayed cloud-sync
  consent.

### D-IMPL-018: Skip pipeline archives under Metal validation

- **Decision:** When either `MTL_DEBUG_LAYER=1` or
  `MTL_SHADER_VALIDATION=1`, compile pipelines without reading or writing an
  `MTLBinaryArchive`. Keep cold, warm and damaged archive coverage in the
  non-validation `metal_context_test`.
- **Reason:** The first validation crash was traced with LLDB to
  `Context::run_selftest` bridging a command buffer whose borrowed Objective-C
  reference had already been released; an owning C++ handle fixed that bug.
  Further checks showed validation-instrumented pipelines do not produce warm
  archive hits on the reference M5 Pro. A fault-test child opening the normal
  cached archive under validation terminated with SIGSEGV during context
  initialization. Skipping the archive avoids both a nonfunctional cache path
  and this driver failure.
- **Review required:** Recheck archive use under validation after an Xcode or
  macOS update; keep archive cache coverage in ordinary `metal_context_test`.

### D-IMPL-019: Report Metal validation logs even when a command buffer completes

- **Decision:** Inspect `MTLCommandBuffer.logs` in the completion handler and
  print validation diagnostics with the command-buffer label. The fault test
  treats an out-of-bounds shader validation log as the expected diagnostic;
  it accepts either `MTLCommandBufferStatusCompleted` or
  `MTLCommandBufferStatusError` when that diagnostic is present.
- **Reason:** On the reference M5 Pro, the shader validator detects a 1 TiB
  out-of-bounds load and includes the access, buffer, and shader stack in
  `MTLFunctionLog`, while Metal reports the command buffer as completed and
  leaves `NSError` unset. Other GPU families may report `Error`; the validation
  log is the portable assertion. Relying only on `status == Error` and `error`
  drops this actionable diagnostic. A following normal command buffer
  completes and writes its result.
- **Review required:** Preserve validation-log assertions on Apple GPU
  hardware, require it to identify the bad offset and kernel, and verify the
  status and diagnostic behavior when Xcode or macOS changes. Keep the separate
  `MTLCommandBufferStatusError` / `NSError` reporting path for actual
  command-buffer errors.

### D-IMPL-020: Drain committed work before Context teardown

- **Decision:** `Context::~Context()` enqueues and waits for a queue fence before
  removing its residency set or releasing its buffers. If Metal cannot create
  that fence, terminate the process with exit code 1 before returning from the
  destructor. Callers must commit or discard command buffers before destroying
  their `Context`.
- **Reason:** A loader's no-copy mappings and Context-owned `MTLBuffer`s must
  remain valid until every committed GPU access has finished. A command buffer
  may still be held by a caller after its `Context` is destroyed, so the
  destruction contract explicitly forbids committing it afterward. If Metal
  refuses a drain command buffer, Context cannot ensure that caller-owned
  no-copy mappings remain mapped after the destructor returns. Terminating
  before returning is safer than allowing the caller to unmap memory that
  queued GPU work may still reference.
- **Review required:** Keep the blocked-event teardown test and confirm the
  drain-failure termination path, and confirm the loader destroys `Context`
  before unmapping weights when H5 implements `WeightStore`.

### D-IMPL-021: Serialize shared-event value allocation and signalling

- **Decision:** Protect host reservations, host signals and GPU signal encoding
  with one mutex. Reservations advance beyond both the observed event value and
  the last GPU-encoded signal. Track in-flight GPU signal values until their
  event value is observed or their command buffer completes; cancel pending
  values when a `NotEnqueued` command buffer is discarded. Metal cannot cancel
  an `Enqueued` command buffer, so its wrapper commits it when discarded. Reject
  a host signal that would overtake the lowest pending GPU signal.
- **Reason:** Independent atomics allowed a reservation to observe stale
  signal state between the GPU signal counter update and reservation advance.
  That could return a value below an already encoded signal; the caller could
  later attempt to signal the event backwards. A single lock makes those
  updates linearizable. A discarded, uncommitted command buffer has no
  completion callback, so its destructor must cancel the reservation. An
  enqueued command buffer cannot be canceled without stalling its queue. The
  event test covers both paths and first waits for a GPU arrival marker before
  keeping a later signal behind an unsignaled GPU wait.
- **Review required:** Keep signal/reservation updates serialized if the event
  API changes, retain the discard test, and keep the pending-wait test so a
  missing GPU wait cannot pass silently.

### D-IMPL-023: Submit shared-event signals in value order

- **Decision:** Bind each `Events` instance to the command queue that created
  it. Encode all command-buffer work before enqueue or commit. Require signal
  ranges from different command buffers to be contiguous in encoding order,
  and reject enqueue or commit of a higher event value until every lower
  pending value from another command buffer has been committed. Serialize
  wrapper submissions with event encoding so validation and queue insertion
  form one ordering transaction. A failed validation leaves the command buffer
  unsubmitted so the caller can submit the lower value and retry.
- **Reason:** Increasing values at encode time do not guarantee increasing
  execution order: separately encoded command buffers can be submitted in the
  reverse order. If a higher value reaches the shared event first, a waiter for
  a lower value wakes before its associated GPU work has run. Queue binding and
  submission validation make the event's `>=` wait semantics match the
  execution order; contiguous signal ranges prevent a single command buffer
  from placing a later value ahead of an intervening command buffer's signal.
- **Review required:** Keep the reverse-submission test, preserve wrapper-only
  command-buffer submission, and verify event ordering if support for multiple
  queues or encoding after enqueue is introduced.

### D-IMPL-024: Submit GPU waits after their pending producers

- **Decision:** Before enqueueing or committing a command buffer with an event
  wait, require every unobserved GPU signal from another command buffer to be
  committed. Once an unsignaled wait from another command buffer has been
  enqueued, reject any new GPU signal on that event. Also reject a
  same-command-buffer signal at or below an earlier unsignaled wait. Keep
  host-released waits valid when the host can signal below all pending GPU
  signal values.
- **Reason:** A consumer wait committed first can sit at the head of the only
  queue and prevent its producer from running. Lower pending signals also
  prevent the host from advancing the event to the wait value, while higher
  signals can be the GPU operation that satisfies the wait's `>=` condition.
  If the producer has not been encoded yet, the converse race can still arise
  unless future signals are checked against already enqueued waits. The
  submission lock and pending wait records close both orderings while
  preserving explicit host gates such as the layer handshake.
- **Review required:** Keep consumer-before-producer, late-producer-after-wait,
  same-buffer self-dependency and host-release tests. Recheck the rules before
  adding another queue or changing the handshake.

### D-IMPL-025: Name raw Metal access as an unsafe escape hatch

- **Decision:** Do not expose the `MTLSharedEvent` handle. Name the raw
  `MTLCommandBuffer` accessor `unsafe_native_handle()` and document that it is
  for Objective-C++ encoding and diagnostics only; native enqueue or commit
  bypasses Strata's ordering checks. Normal submission remains through the
  wrapper.
- **Reason:** Metal encoder APIs require the native command-buffer object, so
  the host port needs an Objective-C++ interop path. Exposing the event too
  would let callers signal around both monotonic value tracking and queue
  ownership. The explicit unsafe name makes the remaining command-buffer
  escape visible at each call site.
- **Review required:** Keep every production call site limited to encoding or
  diagnostics, and do not add a raw shared-event accessor. If the host API
  becomes externally consumed, replace the escape hatch with restricted
  encoder wrappers.

### D-IMPL-026: Validate geometry identically at every planner boundary

- **Decision:** Use one strict validator for geometry files and directly
  constructed `Geometry` values. Require unique shard names, complete
  consecutive shard records, bounded model dimensions, unique tensor types,
  canonical decimal shard IDs, and every expert tensor type to appear in the
  type inventory. Preserve an unknown GGML type ID as `ggml_type_N` instead
  of collapsing it to `?`.
- **Reason:** The planner is also a public host-only API; validating only the
  text parser would let tests or future callers bypass allocation-safety and
  type-inventory checks. Rejecting non-canonical shard IDs prevents aliases
  such as `shard01_name` from being silently ignored beside `shard1_name`.
  Retaining the numeric ID makes unsupported-type diagnostics actionable.
- **Review required:** Keep the malformed-file and direct-Geometry tests
  together. If the geometry format starts escaping delimiters or adding model
  families, revise the shared validator before relaxing either entry point.

### D-IMPL-027: Apply both documented meanings of `--expert-cache N`

- **Decision:** Limit the resident set to at most N entries and at most the
  combined bytes of the first N entries in profile order. Apply both limits
  during each admission pass; `auto` has no explicit count restriction.
- **Reason:** The CLI table defines N as the number of routed experts in the
  resident set, while the memory algorithm defines the byte cap using the
  first N profile entries. Enforcing both preserves each statement, including
  when later layers have smaller expert blobs. The memory algorithm now states
  the count condition explicitly.
- **Review required:** Keep a varied-blob test where an early large expert is
  skipped and later small experts would otherwise exceed N.

### D-IMPL-028: Count MTP memory only when speculation is enabled

- **Decision:** Include MTP weights, bind scratch and MTP KV in A only when
  `--spec > 0`. For the tiered available-memory warning, compare against
  anonymous allocations and exclude the file-backed MTP model windows.
- **Reason:** The setup contract explicitly gates MTP plan terms on `--spec`,
  while the memory table describes MTP weights as mapped files. The revised
  memory-plan formula also excludes `wrapped_unread` from physical RAM use and
  reports both planned anonymous MiB and available MiB when startup availability
  is low.
- **Review required:** Keep the no-spec, full-mode unread-range, and tiered
  available-memory warning tests. Reconcile these fields with `WeightStore`
  and real-file planning when H5 is connected.

### D-IMPL-029: Preserve the specified zero per-layer quota

- **Decision:** Use `floor(first_pass_residents / n_layers)` literally. If
  fewer residents than layers fit in the first pass, the second pass has quota
  zero and admits none.
- **Reason:** This follows the explicit two-pass formula in 05 §3.2. Clamping
  to one would be a silent behavior change and can overstate the intended
  per-layer fairness when the total cache is smaller than the layer count.
- **Review required:** Decide whether the zero-resident edge case is the
  intended product behavior or whether the memory-plan specification should
  define a minimum quota. Until reviewed, the implementation and regression
  test preserve the literal formula.

### D-IMPL-030: Keep PLAN fit byte-accurate and MiB fields specification-stable

- **Decision:** Decide fit and `free_bytes` from exact bytes; keep `total_mib`
  as the sum of each displayed component rounded up, and keep `free_mib` as
  `w_mib - reserve_mib - total_mib`. Document that a byte-fitting plan can
  therefore display a slightly negative `free_mib`. Treat `w_full_mib` as the
  memory threshold assuming the residency flags permit all experts; a smaller
  explicit `--expert-cache N` continues to force tiered mode above that W.
- **Reason:** The PLAN contract explicitly requires both the rounded component
  sum and byte-level fit. Deriving `free_mib` from exact free bytes would make
  it disagree with the displayed values. Separating the memory threshold from
  the residency cap keeps the stable key useful as wired-limit advice.
- **Review required:** Keep the exact-byte boundary case and the explicit-cache
  `w_full_mib` case. Revisit only with a PLAN format revision and matching
  setup/parser changes.

### D-IMPL-031: Distinguish an explicit zero resident budget from no cap

- **Decision:** Add an explicit presence bit to `PlanOptions`. When the byte
  cap is present with value zero, admit no routed experts; when it is absent,
  it imposes no cap.
- **Reason:** A zero-GiB resident budget naturally means zero resident bytes.
  Using zero as both the default sentinel and a user value silently ignored the
  strictest possible cap. The CLI specification now states the distinction.
- **Review required:** Keep both absent-cap and explicit-zero tests, and ensure
  the CLI adapter sets the presence bit whenever `--resident-budget-gib` is
  supplied.

### D-IMPL-032: Keep memory probes portable across the existing host test target

- **Status:** Superseded by D-IMPL-233 and D-IMPL-238. This entry records the earlier multi-platform design; it is
  not a current build decision.
- **Decision:** Compile the probe implementation into both the Metal platform
  library and the existing GPU platform library. Keep Linux cgroup-v2
  availability capped by the tightest visible ancestor, and let Windows report
  an unavailable process sample when it cannot provide the complete Mac-shaped
  record.
- **Reason:** The shared `platform_memory_test` runs on non-Metal builds too.
  Linking only the Mac implementation would break that target, and requiring
  the Windows process record would fail even though Windows has no matching
  source for swap usage and major faults. Linux must not overstate container
  headroom by using host `MemAvailable` alone.
- **Review required:** Verify this target wiring on CUDA/HIP builds and check
  the Darwin available-memory calculation against an independent reading under
  normal and high memory pressure.

### D-IMPL-033: Reject plan-only inputs that cannot produce the requested result

- **Decision:** Keep geometry planning device-free when both memory overrides
  are supplied; reject `--metal-capture` with `--plan-only`; validate the
  context ceiling and automatic-prefill floor before invoking the planner.
- **Reason:** A geometry plan opens no decode window and the planner's semantic
  input failures are not user argument errors. Catching them at the CLI
  boundary preserves exit 2 for invalid arguments and exit 3 for a valid
  no-fit plan.
- **Review required:** Keep the CLI regression cases aligned with sections
  2.3 and 3.10. H5 now applies MTP validation to the real-file plan path too;
  keep both geometry-file and real-file cases aligned if the CLI changes.

### D-IMPL-022: Keep the complete CLI inventory in a device-free registry

- **Decision:** Store every upstream and Metal flag's arity, disposition and
  ignored-warning policy in one registry. Preserve supplied values and
  occurrences generically so later engine steps can map them into their
  runtime options. Keep parser tests in a host-only executable. Require
  `--native` for model planning/export except `--plan-only --geometry`. Until
  planning and inference are connected, the driver rejects those modes instead
  of silently running its self-test. Keep the development smoke test available
  only through explicit `--self-test`; an empty invocation prints usage and
  exits 2.
- **Reason:** The upstream `Options` structure contains CUDA-only state and is
  not a safe copy into the Metal driver. A device-free registry makes flag
  acceptance and dispositions testable without a GPU while providing a single
  inventory for future host-side work. The temporary driver guard prevents
  accepted model options from appearing to take effect before H5-H7 implement
  them. The explicit `--native` validation preserves the documented model
  requirement even while those actions remain unavailable.
- **Review required:** When H5-H7 consume the registry, verify value parsing
  and defaults against the disposition tables in section 3; replace generic
  values with validated runtime fields before using them for allocation or
  inference. Keep empty invocation fail-closed and preserve explicit
  `--self-test` as the only route to the scaffold smoke test.

### D-IMPL-034: Match the loader to the frozen Qwen tensor catalogue

- **Decision:** Require the exact 1,224 tensor names and shapes and constrain
  each tensor role to the union of GGML types found for that role in the seven
  checked-in frozen model headers. Accept only the catalogue's 256-expert
  Coder or 512-expert forms. Validate the same inventory across all shards
  before constructing geometry.
- **Reason:** The planner's supported-type set is global and does not prove
  that a particular kernel can consume a particular tensor's encoding. The
  seven frozen headers share one architecture inventory but vary quantization
  by role. An unexpected, missing, or wrongly encoded tensor could otherwise
  produce a plausible plan for a model the runtime cannot load.
- **Review required:** Recompute the inventory and role/type matrix whenever
  a catalogue model or quantization is added. Review against the intended
  upstream model packs before widening any allowed set.

### D-IMPL-035: Accept only catalogued exact router conversions

- **Decision:** Accept conversion records only for router tensor roles and
  only when they describe exact F32-to-BF16 conversion with zero maximum
  absolute error. The source type and shard must match the GGUF tensor.
- **Reason:** The checked-in catalogue records this conversion for Swift
  routers; it does not define lossy conversion as a supported engine input.
  Allowing `exact=false` or another role/type could create geometry for data
  whose runtime representation is not established by the model contract.
- **Review required:** Recheck the conversion contract against `iq_pack.py`
  outputs when the packer or catalogue adds a conversion method or role.

### D-IMPL-036: Make real-file speculative planning require validated MTP data

- **Decision:** When `--spec > 0`, require and inspect the MTP runtime for
  real-file plans, just as geometry-file plans require MTP byte facts.
- **Reason:** A speculative PLAN that omits MTP weights or binding scratch
  understates its working set. Requiring MTP files at the real-file boundary
  keeps both planning paths based on complete inputs.
- **Review required:** Keep the missing-runtime, malformed-index, projection
  shape and geometry-versus-real MTP accounting cases together.

### D-IMPL-037: Replace exported geometry atomically through a unique file

- **Decision:** Write geometry to a unique same-directory `mkstemp` file,
  sync its contents, then rename it over the requested target. Remove the
  temporary file on write or rename failure.
- **Reason:** A shared `FILE.tmp` name can collide across concurrent exports
  or be redirected through a symlink. A same-directory unique file allows
  atomic replacement without exposing a partially written geometry.
- **Review required:** Keep concurrent-safe naming and existing-target /
  temporary-path regression cases if the export implementation changes.

### D-IMPL-038: Reject invalid UTF-8 geometry identities during argument parsing

- **Decision:** Require `--geometry-family` and `--geometry-model` values to
  be non-empty valid UTF-8 atoms without whitespace, `=` or `#`; report
  invalid values as CLI argument errors.
- **Reason:** Geometry files are UTF-8 text and the parser rejects malformed
  encoding. Deferring that failure until serialization would turn a malformed
  command line into a runtime error with a different exit code.
- **Review required:** Keep malformed UTF-8 argument coverage and align this
  validation with the geometry format if its encoding contract changes.

### D-IMPL-039: Exercise the Swift split through WeightStore, not just GGUF parsing

- **Decision:** Build a sparse synthetic 1,224-tensor Swift-style pair with
  342 tensors in shard 1 and 882 in shard 2, including the PLE table and
  cross-shard expert-role lookups, and inspect it with both expert-table
  versions 3 and 4.
- **Reason:** A standalone split-header test cannot prove that WeightStore
  resolves the PLE tensor, index rows, native expert offsets and geometry
  across the real catalogue's shard layout.
- **Review required:** Preserve the split counts and placement when changing
  the fixture; validate against real Swift headers before claiming model-pack
  parity.

### D-IMPL-040: Scale MTP bind scratch with the effective MTP window

- **Decision:** Keep geometry's `mtp_bind_bytes` as the inspected vocabulary's
  default-window baseline. The layout includes the vocabulary ID table and
  sizes `hl` to the actual output count: the subset length when
  `draft_vocab.bin` exists, otherwise 248,320 logits. The planner replaces
  only the capacity-dependent attention scratch and `mtp_ids` bytes for
  `round_up_64(min(mtp_window, max_context))`. A zero window uses the full
  planned context. The remaining T=8 buffers do not shrink with
  `--mtp-max-t`.
- **Reason:** `--mtp-window` changes the drafter's attention cap, while
  `--max-context` bounds that cap. The optional draft vocabulary also changes
  the one-row logits extent, which must be reflected in geometry even though
  the default vocabulary table remains an optional file. Recording the actual
  vocabulary-dependent base and replacing only the window-dependent portion
  keeps real-file and geometry-file plans aligned.
- **Review required:** Keep the 32,768, 65,536, full-context, zero-window and
  context-cap tests, plus cases with no vocabulary and each supported subset
  size. Recheck the formula against the allocation table in 04 §5.11 if arena
  sizes change.

### D-IMPL-041: Validate draft vocabulary IDs before planning

- **Decision:** Read the optional little-endian int32 `draft_vocab.bin` during
  MTP inspection and require every ID to be within `[0, 248320)` and unique.
  Distinguish a clean EOF from a read error after the expected bytes. Include
  its exact ID-table bytes in `mtp_bind_bytes`, and size the single logits row
  to the subset count or the full 248,320-token vocabulary. The default CJK
  plan remains 16,300,000 bytes at the 32,768-cell window.
- **Reason:** The file feeds the draft head's token-row lookup. A negative,
  out-of-range or repeated ID would make a geometry-valid MTP runtime produce
  an invalid or ambiguous vocabulary subset. When the file is absent, using
  the subset-sized logits allocation would make the plan smaller than the
  full-head kernel output.
- **Review required:** Keep out-of-range, duplicate and truncated-file cases;
  compare against all shipped draft-vocabulary assets when model files are
  available.

### D-IMPL-042: Protect WeightStore inputs from geometry export replacement

- **Decision:** Before exporting, compare an existing output target against
  every opened GGUF shard, the PLE shard, the effective pack files read by
  WeightStore (including the default `pack/full`) and the MTP runtime files.
  Refuse if `filesystem::equivalent` finds the target is any of those inputs.
- **Reason:** Atomic rename deliberately replaces an existing target. Without
  an identity check, a valid export command could replace its own GGUF,
  pack index or MTP input, including through a symlink or hard link.
- **Review required:** Keep same-file, symlink-alias and pack-index collision
  tests. Include any new WeightStore input in the protected set.

### D-IMPL-043: Treat conversion metadata as an attestation on the light path

- **Decision:** Match router conversion records to the exact `iq_pack.py`
  method string and require the F32 source type, BF16 destination type,
  `exact=true` and zero declared error. Validate SHA-256 syntax, but do not
  read GGUF tensor payloads or recompute the digest during PLAN or geometry
  export.
- **Reason:** H5's light path is explicitly limited to headers, indices and
  file sizes. Recomputing the digest would read converted tensor payloads and
  break the no-tensor-data contract and its timing bound. The metadata is
  checked for consistency but remains a packer attestation, not a payload
  integrity proof.
- **Review required:** Decide whether release acceptance needs a separate
  opt-in payload audit that checks hashes and exact conversion values against
  the real catalogue packs.

### D-IMPL-044: Keep the serve protocol testable before the Metal request backend exists

- **Decision:** Put the line protocol behind a host-only `ServeBackend` interface and
  test it with a deterministic fake Window. Keep `main.cpp` from advertising
  `READY` or connecting `--serve` until model initialization and a real request
  backend are available.
- **Reason:** `src/metal/engine/window.cpp` is still a skeleton and `main.cpp`
  currently rejects inference modes. A fake wired into the shipped executable
  would return plausible protocol output without running the model. The H6 test
  now exercises byte-stable startup and response lines, request errors, GPU
  failure, SVE1 parsing, image-pad matching, M-RoPE positions, asynchronous
  STOP, EOF cancellation and signal handling.
- **Review required:** Connect the protocol to real model startup, Window,
  Prefill, sampling, live INFO/DONE counters and H11 watchdog; then run the
  upstream `serve/test_*` suite before accepting H6.

### D-IMPL-045: Reject malformed numeric request settings before they reach sampling

- **Decision:** Parse recognized numeric `GEN`/`GENI` values completely, reject
  overflow and non-finite floats, keep unknown `key=value` options silent, and
  apply the documented clamps to `pcie_frac`, `spec_min_p` and
  `penalty_last_n`.
- **Reason:** Prefix conversions can silently turn malformed values into zero,
  accept trailing junk or pass NaN/Infinity into sampling. Valid protocol
  settings retain their documented values and default behavior.
- **Review required:** Compare real server-generated request values and error
  handling with the retained upstream parser when the production adapter is
  connected.

### D-IMPL-046: Bound serve request memory and linearize cancellation

- **Decision:** Cap parsed prompts and effective serve context at 262,144
  tokens, and reject startup metadata above that ceiling. Before allocating
  SVE1 payloads, match each record to a contiguous image-pad run, enforce a
  256 MiB aggregate image-payload ceiling and verify the complete payload is
  present in the file. Limit an input line to 4 MiB and the queued commands to
  1,024 lines / 8 MiB; discard excess bytes of an overlong line and fail
  closed on queue overflow. Parse ready input bytes while holding the reader
  mutex. At request completion, drain currently available stdin under that
  same mutex before snapshotting STOP, up to a 64 MiB drain ceiling; exceeding
  the ceiling cancels the active request and fails the input stream closed.
  At EOF, flush a final CR-terminated line, cancel an active request and
  discard queued commands, or mark the earliest queued request for
  cancellation before dequeue. Validate prompt progress, generated-token
  counts and EOS outcomes at the host boundary; once a configured EOS token
  is emitted, request backend stop and reject later token callbacks.
- **Reason:** SVE1 headers are untrusted input and their 32-bit dimensions
  could otherwise request multi-terabyte allocations. Prompt matching, the
  context and image-payload bounds, file-length checks and allocation error
  handling keep memory use finite. Input bounds prevent a producer from
  growing a partial line or command queue without limit. EOF closes the
  command stream, so cancellation must not depend on whether the main thread
  or reader thread wins a scheduling race. Parsing, EOF finalization and the
  completion snapshot share the same mutex, so STOP or EOF already readable
  at completion cannot be attributed to a later request. PP totals and EOS
  finish values are checked against the request contract before a successful
  terminal line. The completion-drain ceiling bounds time spent synchronously
  draining an adversarial input stream at the request boundary.
- **Review required:** Confirm the 262,144-token ceiling and 256 MiB image
  payload limit against supported YaRN contexts and the vision encoder's
  configured maximum output; confirm the 64 MiB completion-drain ceiling
  against expected server input; retain the EOF, completion-order,
  input-bound, oversized-header and allocation-bound tests.

### D-IMPL-047: Use Darwin performance-core topology for CPU vision encoding

- **Decision:** When `strata-vision` runs on CPU without an explicit
  `--threads`, query `hw.perflevel0.physicalcpu` and use that count. If Darwin
  does not return a positive count, fall back to `hardware_concurrency()`;
  preserve the existing half-count default on non-Apple platforms.
- **Reason:** Dividing the reported CPU count by two assumes SMT and does not
  identify the performance cores on Apple silicon. The sysctl supplies the
  intended worker count on supported Macs. Using all reported concurrency as
  a fallback avoids repeating the unsupported SMT assumption if the key is
  unavailable.
- **Review required:** Verify the emitted CPU thread count against the P-core
  count with a model-backed `strata-vision` launch on the reference Mac; the
  fallback path is a portability guard and is not the H13 target.

### D-IMPL-048: Keep one-shot output formatting separate from unavailable inference

- **Decision:** Implement the H7 output formatter and its host golden test as a
  reusable seam, but keep the production CLI's inference path unavailable until
  the real `Window` and `Prefill` backend exists. Do not connect a fake generator
  to `strata`.
- **Reason:** `oneshot.cpp` and `stats.cpp` were empty, while the Metal
  `Window` and `Prefill` remain skeletons. A formatter can be checked now
  against the specified line shapes; emitting synthetic tokens from the
  production binary would falsely imply model inference works.
- **Review required:** Integrate the formatter with real one-shot execution,
  compare every retained line against captured upstream output, and close H7
  only after model-backed checks pass.

### D-IMPL-049: Do not count GPU execution twice in the window wall time

- **Decision:** Compute the `gpu window` wall-time field from CPU encode time
  plus the host's commit-to-wake wait. Report `GPUEndTime - GPUStartTime` as
  its own component, without adding it again to the wall-time total.
- **Reason:** The documented wait begins at commit and ends when the host wakes
  after completion, so it contains the GPU execution interval. Adding all
  three measurements would double-count GPU time.
- **Review required:** Verify the encode, completion-wait and GPU timestamp
  boundaries against a real `MTLCommandBuffer` timeline before using the line
  for performance decisions.

### D-IMPL-050: Keep window source ranges transient and device free

- **Decision:** Keep raw tensor offsets and expert-slab boundaries in a
  transient `WindowSources` value returned optionally by `WeightStore::inspect`.
  Keep `Geometry` limited to the portable, serializable aggregate facts it
  already owns. Both `wrapped_unread` and the bounded `WindowPlan` use the same
  validated source ranges.
- **Reason:** Geometry files must remain usable without access to model files
  and should not serialize tensor-by-tensor mappings. The window planner needs
  exact file offsets and expert boundaries, while metadata inspection must stay
  payload-free and device-independent.
- **Review required:** Verify that the production full-mode loader uses these
  mappings with the correct mapped-file and `MTLBuffer` lifetimes, and compare
  its address table against the source tensor ranges before claiming U3.

### D-IMPL-051: Permit only page padding at an excluded tensor boundary

- **Decision:** Exclude the PLE tensor from all `WindowSlice` mappings and
  `wrapped_unread` read bytes. Permit a page-aligned window to cover up to one
  16 KiB page of its edge bytes when an adjacent read tensor ends at an
  unaligned offset.
- **Reason:** The specification requires both complete tensor coverage and
  outward 16 KiB alignment, which can make a small part of an excluded
  neighboring tensor physically fall inside the page-rounded buffer range.
  Skipping that page would leave the adjacent tensor unmapped. This follows
  section 4.2's explicit allowance for neighboring-byte padding while keeping
  the PLE payload out of the engine's tensor mappings.
- **Review required:** Confirm that no-copy Metal buffers may cover these
  page-padding bytes without the PLE tensor being read through the window, and
  keep the test's one-page overlap bound.

### D-IMPL-052: Encode active full-mode residency entries as resident

- **Decision:** Initialize each active `int32` entry in the full-mode resident
  table to `1`; leave the unused maximum-capacity tail zero. Store expert GPU
  addresses independently in gate/up/down order.
- **Reason:** Section 4.1 requires every expert to be marked resident, but the
  not-yet-implemented plan kernel has no integer encoding contract. `1` is the
  direct true value for the documented all-resident state and avoids inventing
  tiered slot indices for experts that remain in their GGUF windows.
- **Review required:** Confirm the encoding against the first production plan
  kernel before connecting or consuming the resident table.

### D-IMPL-053: Allow the final page-rounded mmap tail in a file window

- **Decision:** Validate a no-copy window's upper bound against the file size
  rounded up to the 16 KiB VM page, while continuing to validate every tensor
  payload against the exact file size.
- **Reason:** The window planner must round tensor ranges outward for Metal.
  A tensor ending in the final partial mmap page therefore needs the zero-filled
  tail of that page in its page-aligned buffer. Rejecting those padding bytes
  would make valid catalogue shards fail full-mode loading at EOF.
- **Review required:** Keep the live `metal_buffers_test` case that maps a short
  file, wraps its final page and checks the file bytes plus zero-filled tail.

### D-IMPL-054: Warm profile-ranked experts in eight-expert waves

- **Decision:** Warm expert slabs in profile order in waves of eight experts.
  The eight CPU workers touch each wave concurrently and wait at a barrier
  before starting the next wave; remaining window pages are warmed in ascending
  file order in chunks no larger than 64 MiB.
- **Reason:** The specified eight workers provide SSD read concurrency, while a
  wave barrier prevents lower-ranked experts from being scheduled before the
  current hot group has been touched.
- **Review required:** Measure cold and warm startup on the reference Q2_0
  model and confirm profile-ranked pages are loaded ahead of the remaining
  windows (M-5). A pure helper test verifies the profile order across wave
  boundaries; the live synthetic test exercises a multi-page mapping.

### D-IMPL-055: Reserve maximum table capacity with a compact active prefix

- **Decision:** Allocate address and resident buffers for 48 layers and 512
  experts, but pack active layer rows with stride `n_expert`, as the kernel ABI
  in `docs/mac/04-metal-kernels.md` section 4.3 specifies. Zero the unused
  capacity tail.
- **Reason:** `docs/mac/05-memory-and-residency.md` specifies maximum buffer
  capacity, while the kernel binding contract specifies a compact active row
  stride. This retains the documented 590 KB maximum address-buffer allocation
  without making 256-expert Coder addresses disagree with kernel indexing.
- **Review required:** Confirm the first production plan kernel consumes the
  active prefix and agrees with the full-mode resident value before connecting
  these buffers.

### D-IMPL-056: Revalidate the complete window plan before wrapping

- **Decision:** `FullModeStorage::open` reruns the window planner with the live
  device's `maxBufferLength` and compares every file range, slice and
  `wrapped_unread` total before allocating no-copy Metal buffers.
- **Reason:** A stale or partially edited plan could omit part of a range the
  memory planner counted, or could include bytes the loader did not validate.
  Replanning enforces the exact planned union before any file range becomes a
  GPU resource.
- **Review required:** Apply the equivalent no-GGUF-mapping check to the
  tiered loader; U5 now exercises it through `metal_tiered_copy_test`, while
  production startup integration remains open under REV-017.

### D-IMPL-057: Commit the GPU warm-up checksum into residency

- **Decision:** After allocating and clearing the checksum buffer, commit and
  request residency again before submitting the GPU page-touch command.
- **Reason:** The checksum allocation is added after the storage tables and
  windows have already been committed. The command queue's residency set must
  include that output buffer before the kernel writes it.
- **Review required:** Keep the live Metal warm-up test and run it with Metal
  API validation enabled when changing the warm-up pipeline.

### D-IMPL-058: Match full-mode windows to the memory-plan accounting

- **Decision:** Before allocating any no-copy buffer, require the reopened
  shard names and sizes to match the geometry, require every GGUF tensor to
  occur exactly once in the included source inventory or the explicit
  PLE/converted-router exclusion inventory, reject overlapping payload ranges,
  and require `wrapped_unread_bytes` and converted-router bytes to match the
  geometry. Validate that PLE has no source or slice and overlaps any
  full-mode window by at most one 16 KiB page.
- **Reason:** Replanning catches omitted or extra window ranges, and the
  per-tensor inventory catches omissions whose aligned payload would add zero
  unread bytes. Requiring converted entries to be supported F32 routers and
  matching their packed bytes prevents ordinary dense tensors from being
  silently classified as absent. Matching the geometry prevents a correct
  plan for a different source set from consuming memory admitted for another
  model. The PLE edge exception preserves page-aligned mapping without
  exposing its payload through an address-table entry.
- **Review required:** Apply the corresponding no-GGUF-window assertion to the
  tiered loader; U5 now exercises the anonymous-arena path on a synthetic
  model. The test does not prove the reference Swift model's live PLE boundary.

### D-IMPL-059: Carry converted dense-pack ranges and file identities into the tiered loader

- **Decision:** Extend the inspected `WindowSources` result with the dense pack
  path, size and file identity plus each converted tensor's `dense.bin` source
  offset, source byte count, destination byte count and index kind. Capture the
  device and object identity of each GGUF shard as well. Keep
  `WeightStore::inspect` payload-free; the tiered loader rechecks file size,
  identity and every range before reading the selected conversion rows.
  `WeightStore::inspect` acquires shared advisory locks before opening the GGUF
  mappings or dense-pack metadata, and returns lock ownership in `WindowSources`
  so the locks stay held while storage takes its own descriptors.
- **Reason:** U5 must copy only the engine-form Swift router rows into the dense
  arena. Reopening `index.txt` in the runtime loader would duplicate its
  validation and could let the copy layout diverge from the metadata used by
  planning. Size alone cannot reject a same-size path replacement between
  inspection and the tiered `pread`s. The source/destination sizes are distinct
  in the pack format, so the loader needs both even though the currently
  supported router conversion stores BF16 bytes directly.
- **Review required:** The tiered-copy test must byte-compare each converted
  row and prove that bad offsets, lengths, pack sizes or same-size path
  replacements fail before allocating Metal buffers. `metal_tiered_copy_test`
  now covers those cases. Shared `flock` locks reject cooperating exclusive
  writers, but do not stop writers that ignore advisory locks. File identity
  does not detect in-place writes to the same file; payload hashes remain
  outside the payload-free PLAN contract under REV-010.

### D-IMPL-060: Account for page-rounded tiered arena padding

- **Decision:** Round the dense anonymous arena to 16 KiB before comparing its
  size with `PlanResult::free_bytes`; keep each tensor payload 256-byte aligned
  and include inter-tensor padding in the same budget check.
- **Reason:** `wrap_no_copy` requires a page-aligned backing range, so the last
  partial page is an actual allocation cost even when no tensor uses its tail.
  Accepting only the unrounded tensor total could exceed the memory plan.
- **Review required:** Revisit the anonymous-arena allocation and the
  `alignment_slack` accounting if the allocator alignment or Metal page size
  changes; connect the allocation to production startup before claiming tiered
  model startup is complete.

### D-IMPL-061: Lock and publish GGUF source sets and their derived packs as generations

- **Decision:** Use persistent sibling lock files for the source-set directory
  and pack directory. On POSIX, readers take nonblocking shared locks in this
  order: source set, pack, then individual files. The Windows byte-range lock
  backend has no shared mode, so all acquisitions serialize safely. Setup holds
  exclusive source-set and pack locks across shard validation/download and
  pack publication;
  `iq_pack.py` takes a shared source lock and exclusive output-pack lock, plus
  a shared base-pack lock when `--base` is used. POSIX setup children inherit
  the lock descriptors, including the nested tokenizer child launched by
  `iq_pack.py`. Windows setup runs `iq_pack.py`, `strata_pack.py`,
  `pack_index.py` and `strata_tokenizer.py` in-process so no pack writer can
  outlive the process-local byte-range locks. `WeightStore::inspect` acquires
  the set locks before
  parsing, including metadata-only inspections, and refuses a pack directory
  containing `.strata-generation-pending`. Tiered storage reopens and retains
  the matching shared set locks through its lifetime, along with the existing
  source-file descriptors.
  Setup and direct `iq_pack.py` runs write a pending marker before mutation.
  POSIX runs fsync the marker and its containing directory. On Windows the
  marker file is flushed, but the CRT cannot flush the containing directory;
  the protocol recovers process interruption and does not promise power-loss
  durability because generated pack contents are not fsynced as a transaction.
  Setup clears it only after checking shards and completing pack preparation;
  a restart that finds the marker forces a pack rebuild even when the old
  completion files exist. A direct pack run clears its marker only on success.
- **Reason:** Per-file locks acquired one by one allow an updater to publish a
  new pack between shard locks, producing a mixed source generation. Atomic
  replacement of individual files also cannot make the shard set and pack
  change atomically. The directory-level locks close that cooperating-writer
  race, and the pending marker keeps readers closed after process exit releases
  locks during an interrupted update.
- **Review required:** `metal_weight_store_test` rejects held source-set and
  pack locks and a pending marker. `metal_tiered_copy_test` verifies both
  generation locks stay held while storage opens and refuses a pending pack.
  `test_artifact_set_lock.py` checks platform-appropriate exclusion, inherited
  locks, symlink aliases, and marker lifetime. `test_iq_pack.py` checks
  interrupted pack recovery, pending base-pack refusal, nested POSIX child lock
  inheritance, Windows in-process dispatch for every setup pack helper, and
  in-process tokenizer execution. Setup crash injection and native Windows
  execution remain open under REV-023. Advisory locks cannot stop writers that
  ignore this protocol; that residual remains in REV-010.

### D-IMPL-062: Publish slot-cache entries only after their reads complete

- **Decision:** Represent each tiered slot as Empty, Loading or Ready. A miss
  lease owns the Loading entry and must publish it after a successful read.
  Duplicate requests for a Loading tag fail explicitly; dropping an
  unpublished lease returns the slot to Empty. Ready cache hits and miss
  leases pin their slots until reset, preventing LRU eviction while GPU work
  still references the contents.
- **Reason:** A cache tag alone does not prove that the requested bytes have
  finished arriving. Treating a Loading tag as a hit could expose stale or
  partially written weights. A separate publication point makes read
  completion part of the cache contract.
- **Review required:** `metal_slot_pool_test` covers duplicate Loading
  requests, cancellation and retry, explicit publication, ping-pong halves,
  LRU selection, and a GPU event gate that confirms the slot is pinned through
  the wrapper's completion callback. Synthetic ExpertStore coverage now tests
  miss reads and cache reuse; production layer handshakes and expert kernels
  remain open under REV-024.

### D-IMPL-063: Publish MTP runtime files as a locked generation

- **Decision:** Setup holds exclusive sibling locks for both the MTP source
  directory and `rt/`; the pending marker lives in `rt/`, the same generation
  root used by `WeightStore`. Direct `mtp_fetch.py` and `mtp_pack.py` runs lock
  the source directory, while `mtp_rt.py` takes a shared source lock and an
  exclusive `rt/` lock. Setup children borrow inherited locks on POSIX and
  Windows. Setup reuses runtime files only when both binary sizes and all 29
  dense tensor names, kinds, dimensions and byte ranges match the engine
  schema. The inventory records the requested ref, resolved URL and immutable
  Hub commit; symbolic refs without a resolved commit fail closed. Per-tensor
  sidecars bind completed or resumable range files to that identity, so a moved
  branch or changed request cannot reuse stale bytes after an inventory update
  or interrupted fetch. Known tensor hashes apply only when the resolved commit
  is the hashed pin. The runtime stamp records the same full source identity.
  A standalone runtime writer validates and fsyncs all outputs before clearing
  its pending marker, and removes any old source stamp because its input GGUF
  was not attested by setup. Draft-vocabulary replacement remains atomic.
- **Reason:** Setups for different main models share the same MTP directory.
  An interrupted or concurrent `mtp_rt.py` run can otherwise truncate the
  shared files while an existing `experts.bin` still makes setup skip repair.
  A branch name such as `main` can also resolve to new tensor bytes without
  changing its URL; the immutable commit and per-tensor identity prevent stale
  runtimes and partial downloads from being treated as current.
- **Review required:** Python coverage checks valid, malformed, truncated and
  interrupted runtime decisions, requested and resolved revision changes,
  main fallback, partial-file resume, standalone output validation and
  invalidation, direct-writer exclusion, lock inheritance and Windows helper
  dispatch; `metal_weight_store_test` checks reader exclusion, pending-marker
  refusal and lock lifetime. Native Windows and process-kill recovery remain
  open under REV-023.

### D-IMPL-064: Resolve cold experts through exact cached file reads

- **Decision:** `ExpertStore` validates the complete decoded plan before
  acquiring slots, queues each cold expert's three exact role ranges on a
  persistent worker pool (`STRATA_FETCH_THREADS`, default 8, clamped to 1-64),
  and publishes Ready entries only after every requested slab read succeeds.
  Page-cache classification uses `mincore` over temporary page-rounded
  mappings; a query failure is treated as a file read and never suppresses the
  actual `pread`. Prefetch always fills a reusable slot even when the source
  pages are already resident. The resolver writes all three slot addresses and
  a release fence before returning success; its caller must not signal the GPU
  on error.
- **Reason:** The grouped plan is GPU-produced data, so bounds, group
  cardinality, uniqueness, residency flags and `n_miss` must be checked before
  indexing host tables or reading files. Temporary mappings let `mincore`
  inspect the shared file cache without retaining GGUF mappings or wrapping a
  shard as a Metal buffer. Conservative accounting keeps telemetry from
  claiming a RAM hit when the OS query is unavailable.
- **Review required:** `metal_tiered_copy_test` checks mixed resident/cold
  plans, exact role bytes and addresses, routing counters, slot-cache hits,
  cached-page prefetch, and parity-half reuse after a layer-complete event on a
  synthetic GGUF. Real GPU plan production, per-layer handshake behavior,
  decode lookahead predictions, and full-versus-tiered parity remain open
  under REV-024.

### D-IMPL-065: Retire slot leases after their last GPU read

- **Decision:** Register one `ExpertLeaseSet` completion callback before
  enqueue/commit. The set accepts leases after the submitted command buffer
  exposes each layer's plan. The host retires a layer only after its GPU expert
  kernels have completed reading those slots; the command-buffer callback
  releases every lease still retained at native completion. A discarded
  NotEnqueued buffer reports failure immediately; an Enqueued buffer that is
  discarded is committed and releases its remaining leases only after native
  completion.
- **Reason:** A cache entry can remain Ready while its slot is being reused.
  The plan is produced by GPU work in the submitted buffer, so registering the
  callback inside the later resolver would be rejected as too late. Preparing
  the lease set first lets the host add newly read slots while the GPU waits.
  Layer retirement then lets a later layer of the same parity reuse capacity
  without releasing addresses that an earlier GPU layer still reads.
- **Review required:** `metal_events_test` covers successful and discarded
  callbacks, `metal_slot_pool_test` proves a delayed GPU reader pins a slot,
  and `metal_tiered_copy_test` exhausts a parity half, retires a layer after a
  GPU event, and reuses a slot while its command buffer remains in flight.

### D-IMPL-067: Validate every grouped plan route index before resolving misses

- **Decision:** Require decoded plans to contain a whole number of ten-route
  rows. Before acquiring a slot, verify each `ent_tok` is within the row count,
  each `ent_dst` is within `n_entries`, `ent_dst / 10 == ent_tok`, and
  destinations form a complete permutation.
- **Reason:** The grouped plan comes from GPU memory and the expert kernels use
  `ent_tok` and `ent_dst` directly as input and output indices. Correct group
  boundaries and expert ids do not make those independent indices safe.
- **Review required:** Synthetic resolver tests reject out-of-range token and
  destination indices and duplicate destinations before any file read.

### D-IMPL-066: Publish manually selected draft vocabularies through the MTP lock

- **Decision:** Replace the documented raw `cp` with `tools/mtp_vocab.py`.
  The helper validates little-endian int32 IDs for range and uniqueness, takes
  the runtime's exclusive generation lock, writes and fsyncs a unique temporary
  file, atomically replaces `draft_vocab.bin`, and clears its pending marker
  only after success. When setup already owns the runtime transaction, the
  helper borrows that lock and leaves marker cleanup to setup.
- **Reason:** A raw file copy bypasses WeightStore's shared runtime lock and
  could replace the vocabulary while a model is loading. An explicit source
  path preserves the user's selected vocabulary while keeping manual setup
  within the same publication protocol.
- **Review required:** `tools/test_mtp_vocab.py` covers valid and invalid
  vocabularies, atomic replacement, reader exclusion and nested marker
  ownership; the hostile transaction review checks lock and failure paths.

### D-IMPL-068: Measure resident hits from complete routing traces

- **Decision:** Treat each trace file as one group for `--folds K`, require
  exactly K files, and build each fold's profile from the other files with
  `make_profile`'s deterministic frequency ranking and interleaved fill.
  Require distinct underlying files and distinct layer/ID routing records for
  held-out groups and balanced nonzero routing-record counts across all 48
  layers before reporting per-token route misses. Fingerprint held-out routing
  groups as order-independent multisets of layer/expert-ID-set records,
  sorting ids within each record. Permit runtime-compatible partial STRP
  rankings when they cover the requested resident prefix; unranked experts
  count as nonresident. `--per-layer` reports the exact H bound of the chosen
  resident prefix. Open each held-out input once, copy it into a private
  temporary snapshot, and use only those snapshots for fingerprinting,
  training folds, and evaluation. Open with `O_NONBLOCK` where available
  before rejecting non-regular files, so a FIFO cannot stall the preflight.
- **Reason:** The trace format has layer and route IDs but no token index.
  Equal per-layer record counts allow correct token-normalized totals for both
  token-major decode traces and layer-major prefill traces. A held-out trace
  file per fold prevents routing data from a measured group leaking into its
  profile, including when callers pass aliases, duplicate copies, or the same
  route IDs with different weights or ordering. The preflight hashes an
  order-independent multiset of layer/expert-ID-set records from the exact
  snapshots later used for ranking and metrics. This prevents a trace replaced
  after preflight from changing the evaluated group or leaking into training.
  The trace weights are checked for valid
  finite values but are not used because this metric ranks every routed expert
  equally. `misses_per_token` counts nonresident route occurrences; it cannot
  predict disk reads without modeling deduplication, prefetch and the
  slot/page caches.
- **Review required:** `test_make_profile.py` rejects malformed profiles and
  traces. `test_hit_curve.py` checks exact metrics, held-out ranking, snapshot
  stability and nonblocking FIFO refusal on synthetic data; real
  benchmark-prompt traces remain unavailable here.

### D-IMPL-069: Pack only a verified complete MTP source generation

- **Decision:** Before writing a draft GGUF, compare the selected requested
  revision, immutable inventory identity, exact 31-name BF16 tensor set,
  exact source tensor shapes, complete manifest, each tensor's source sidecar,
  byte length, and SHA-256. A pending runtime generation may be packed only by
  setup while it holds inherited source and runtime locks. Validate packed
  expert geometry with explicit errors that remain active under `python -O`.
  Open runtime files read-write before `fsync` so Windows `_commit` can
  durably publish them.
- **Reason:** An interrupted or partial fetch can leave a new inventory, old
  manifest, mixed tensor files, and a previously valid draft GGUF together.
  Structural runtime validation cannot identify that provenance mismatch.
  Full content hashing adds one sequential read of the fetched tensors before
  the packer's conversion read, trading setup I/O for a verifiable generation.
- **Review required:** `test_mtp_pack.py` covers complete and incomplete
  metadata, the frozen 31 tensor shape catalog, same-element-count transposed
  shapes, mismatched sidecars, same-size corruption, revision changes, and
  direct-versus-inherited pending-marker behavior. `test_iq_pack.py` verifies
  wrong packed expert geometry is rejected under optimized Python and checks
  writable descriptors under the Windows sync branch; native Windows
  execution remains open under REV-023.

### D-IMPL-070: Keep Metal numeric helpers and layouts aligned with their oracles

- **Decision:** Share the parameter structs between MSL and C++, and execute a
  Metal kernel that reports every documented parameter-block size and member
  offset. Represent `Q80S` as a logical `{float scale, 32 signed codes}`
  decode value; the production MTP quantiser's backing data remains separate
  scale and code planes. Keep `rope.freq_base` out of compile-time
  architecture constants because each model file supplies that value.
  Implement `half_sat` with the documented oracle rules: clamp finite values to
  ±65,504, canonicalize every NaN to `0x7e00`, and map float32 denormals to
  positive half zero.
- **Reason:** C++ and MSL must agree on the parameter ABI before later kernels
  bind these structs. `Q80S` provides a convenient local decode value without
  claiming that the model's MTP buffers use an interleaved block. A shipped
  default `rope.freq_base` is not a safe architecture constant when the model
  metadata is authoritative. The Metal conversion oracle explicitly defines
  NaN and float32-denormal behavior; an executed boundary test caught negative
  float32 denormals otherwise becoming negative half zero. `silu_k` has no
  separate A.10 measurement, so its U(8) test limit is chosen by analogy with
  the measured sigmoid structure and is stated separately in the kernel plan.
- **Review required:** `metal_math_test` exercises the helpers and conversions
  on the reference Apple GPU at twice the documented A.10 helper error bound,
  while `Context::run_selftest` compares MSL and C++ parameter layouts.
  The `silu_k` limit is an implementation acceptance choice pending a separate
  A.10 measurement. Recheck the logical `Q80S` decode representation against
  the first S2 mat-vec implementation; the SoA storage contract must remain
  authoritative.

### D-IMPL-071: Make twin guard and dispatch rules executable before kernel bring-up

- **Decision:** Put shared allocation, dispatch, poison checking, B/U/R
  comparison and row-T checks in `tests/metal/twin.{hpp,cpp}`. Compile the
  `.cpp` implementation as Objective-C++ because it must encode through the
  native Metal API exposed by `Context`. Support exact thread grids and
  threadgroup-count grids in three dimensions. Record all three threadgroup
  axes in the manifest; for production kernels, use that exact shape and reject
  a conflicting test value. Compare each row in `row_t_independent` to its own
  T=1 baseline. Use an FP64 reference in R(ε) comparisons and reject invalid
  tolerances. Initialize each output against its oracle using a value guaranteed
  to fail its B/U/R class, while reserving no output encoding. Build
  `strata_metal_oracles` with `-ffp-contract=off` and exercise its contraction
  probe. Give every guarded buffer 4 KiB of poison on both sides.
- **Reason:** The first real kernel twins must share encoding and comparison
  behavior rather than repeat hand-written Metal calls that can drift from the
  manifest. Exact axis checks prevent a same-product threadgroup from silently
  changing shader behavior. Oracle-derived initialization detects unwritten
  bytes and floats even when the expected output is all `0xFF` or a NaN, without
  banning a valid output bit pattern. The runtime-compiled negative controls
  show that an out-of-range write, a one-ULP B-class error, and output that
  varies with T are rejected with a reproducible seed and first failing index.
- **Review required:** `synth.cpp` currently creates generic float inputs for
  the harness selftest. It does not claim to implement upstream's
  area-specific fixtures, valid quantized weight generators, or the model
  `real` input set. Integrate those as the corresponding kernel twins arrive;
  model-backed data is unavailable under REV-002 and the remaining harness
  coverage is tracked by REV-025.

### D-IMPL-072: Match Q8_1 quantization with an explicit SIMD butterfly

- **Decision:** Implement the Metal Q8_1 quantizer with the same five XOR
  butterfly stages as the pinned CUDA path for both `amax` and the original
  float sum. Divide by 127 in safe math, round codes half away from zero, and
  write the rounded scale and unquantized sum as the two half values in each
  36-byte block. The `metal_mv_test` first slice covers n = 640, 2560 and 6144,
  and row counts 1, 3 and 8.
- **Reason:** The Q8_1 scale, codes and sum are consumed by every native
  quantized mat-vec. Preserving the reduction order and storing the sum of the
  original inputs avoids a numerically plausible but incompatible quantizer.
- **Review required:** L3 now includes `strata_q_q8_1` and the BF16 mat-vec
  family. Quantized weight decoders and mat-vecs, exhaustive `mv_decode`,
  the `mv_multi_eq_single` property and its negative control, production
  dispatch wiring, and model-backed `real` coverage remain unimplemented or
  unavailable. These are tracked by REV-025 and REV-026.

### D-IMPL-073: Define Q8_1 codes for subnormal scales

- **Decision:** Find `amax` with an unsigned butterfly over the absolute
  float32 bit patterns. For `amax` exponent fields through 7, reconstruct the
  correctly rounded `d = amax/127` in subnormal units using integer
  round-to-nearest-even. Branch on `d` itself: if it is zero, compute codes by
  exact integer rounding of `127·xi/amax`, half away from zero; if it is a
  nonzero subnormal, rescale exact significands to normal float32 operands and
  evaluate `round(xi/d)`, bounding codes to [-127, 127]; otherwise divide
  directly unless `xi` is subnormal. For a subnormal `xi`, divide its exact
  significand by the normal `d` significand when `d` has exponent field 1,
  and return zero when `d` has exponent field greater than 1, where the ratio
  is below 0.5. Keep the stored half scale at zero.
- **Reason:** The first subnormal test showed this Apple GPU flushes
  float32-denormal operands in the floating maximum path, hiding their
  magnitudes. Converting the infinity from `xi/0` to an integer has no
  portable result. Using hardware division for a nonzero subnormal `d`, or
  with a subnormal `xi`, can depend on GPU flush-to-zero behavior. The
  significands are rescaled without changing the ratio; where a tiny `xi`
  cannot reach half a code, returning zero is exact. The explicit bound keeps
  subnormal-scale outputs within Q8_1's code range.
- **Review required:** The Metal fixtures pin zero-scale, nonzero subnormal
  scale, normal scale with a subnormal input, and the minimum-normal-scale
  boundary on the current Apple GPU. The zero-scale integer ratio
  intentionally supplies a deterministic result where CUDA's float-to-
  integer conversion is not portable. For a nonzero subnormal scale, the
  portable Q8_1 bound can differ from an out-of-range CUDA float-to-int8
  conversion; bitwise parity is claimed only where that CUDA conversion is
  defined and the policy agrees. L-10 still determines general float32
  denormal arithmetic on each supported GPU family; this path avoids
  depending on that behavior.

### D-IMPL-074: Preserve the BF16 mat-vec FMA and reduction order

- **Decision:** Add `strata_mv_bf16_b<BS>_n<NC>` for BS = 160 and 256 and
  NC = 1, 2, 4 and 8. Each thread keeps a separate F32 accumulator, applies
  the two BF16 products as ordered FMAs for each packed weight word, then
  reduces first within each SIMD group and then across zero-padded group
  partials. Map inactive columns to input row 0 and the sink buffer, matching
  section 5.9.
- **Reason:** Changing either the two-FMA sequence or the two-level
  butterfly can change output bits. The catalogue's BF16 widths select only
  BS 160 or 256; matching upstream's iteration and reduction layout keeps the
  port deterministic and supports all specified NC variants. The NC property
  uses dense, exactly representable BF16 weight patterns and F32 input
  patterns across every K position, with distinct columns. A 128-to-129
  one-bit weight mutation at a nonzero product deterministically exercises the
  negative control.
- **Review required:** `metal_mv_test` compares random, edge and fixture
  inputs at n_in = 320, 2560 and 10240 for all four NC variants and every
  live-row count from 1 through NC, including each inactive-column sink and
  the all-live case. A separate analytic fixture checks fixed BF16 weights and
  inputs against literal expected output values before GPU comparison.
  `mv_multi_eq_single` also compares every NC=2/4/8 BF16 column with its own
  NC=1 Metal dispatch at all three widths and verifies that flipping one weight
  bit is detected. The property across quantized and expert families,
  quantized formats, production binding and model-backed `real` inputs remain
  open under REV-025 / REV-026.

### D-IMPL-075: Match the native Q8_0 mat-vec reduction

- **Decision:** Add `strata_mv_q8_0_n<NC>` for NC = 1, 2, 4 and 8. Each
  thread handles two 32-bit code words selected by
  `kqs = 2 * (tid % 4)`, computes the exact signed-byte dot, and applies the
  rounded `d_weight * d_activation` scale with the specified FMA. Preserve
  CUDA's four-warp partial-add order and butterfly. Use four output rows per
  threadgroup when `n_in / 32 < 32`, and one otherwise. Load weight codes as
  individual signed bytes because the 34-byte block stride guarantees only
  two-byte alignment.
- **Reason:** Reinterpreting each packed Q8_0 row as aligned 32-bit words is
  invalid for general 34-byte blocks. Scalar signed-byte loads keep the
  integer dot exact without requiring stronger alignment, while the fixed
  thread and reduction mapping preserves output bits across NC. The property
  fixture uses dense K-varying codes and activations; its negative control
  changes the first weight code from 1 to 0 while the matching Q8_1 code and
  scale are 127 and 1, guaranteeing a detectable output change.
- **Review required:** `metal_mv_test` covers random, edge and fixture inputs
  at n_in = 640, 992, 1024, 2560 and 6144, NC = 1, 2, 4 and 8, and every
  live-row count. The 992/1024 widths test both sides of the 32-block
  four-row/one-row dispatch boundary; the other widths cover production shapes.
  An analytic n_in = 640 fixture checks the Q8_1 scale, integer dot and
  inactive-column sink. The Q8_0 `mv_multi_eq_single` subset compares each
  NC=2/4/8 result with single-column Metal dispatches across all five widths
  and has a code-bit negative control. Other quantized formats, exhaustive
  decoders, property coverage for the remaining quantized and expert families,
  production binding and model-backed `real` inputs remain open under
  REV-025 / REV-026.

### D-IMPL-076: Match the native Q4_0 mat-vec decoder and reduction

- **Decision:** Add `strata_mv_q4_0_n<NC>` for NC = 1, 2, 4 and 8. Decode
  Q4_0 low nibbles against Q8_1 codes 0..15 and high nibbles against codes
  16..31. Each thread handles the two 32-bit words selected by
  `kqs = 2 * (tid % 2)`, then applies
  `fma(d_weight, fma(-4, dsum, float(dot) * dscale), acc)`.
  Preserve CUDA's four-warp partial-add order and butterfly. Use four output
  rows per threadgroup when `n_in / 32 < 64`, and one otherwise. Read packed
  nibbles as bytes because Q4_0 blocks are 18 bytes and only two-byte aligned.
- **Reason:** This follows the native `small_q8_dot` Q4_0 path and keeps
  accesses valid across consecutive 18-byte blocks. The exact integer dot,
  nested FMA sequence and fixed reduction mapping make the scalar oracle
  independent of floating-point dequantization and preserve output bits.
- **Review required:** `metal_mv_test` covers random, edge and fixture inputs
  at n_in = 640, 2016, 2048, 2560 and 6144, NC = 1, 2, 4 and 8, and every
  live-row count. The 2016/2048 widths test both sides of the 64-block
  four-row/one-row dispatch boundary. An analytic n_in = 640 fixture checks
  Q4_0 nibble order, the Q8_1 sum correction, column layout and inactive
  output sink. `mv_multi_eq_single` compares each multi-column output with
  separate NC = 1 Metal dispatches for NC = 2, 4 and 8 at all five widths;
  its negative control changes one Q4_0 code bit. Exhaustive decoder
  coverage, production binding and model-backed `real` inputs remain open
  under REV-025 / REV-026.

### D-IMPL-077: Match the native Q5_0 mat-vec high-bit decoder

- **Decision:** Add `strata_mv_q5_0_n<NC>` for NC = 1, 2, 4 and 8. For each
  Q5_0 code index k, combine the matching low or high nibble in `qs` with
  bit k in the four-byte `qh` plane. The logical decoded value is code −16;
  the integer dot itself uses the unsigned 0..31 code, with the affine
  correction applied as `-8 * dsum` in each of the two per-thread partial
  dots. Each thread handles the two packed words selected by
  `kqs = 2 * (tid % 2)` and applies the nested FMA sequence
  `fma(d_weight, fma(-8, dsum, float(dot) * dscale), acc)`. Preserve CUDA's
  four-warp partial-add order and butterfly. Use four rows per threadgroup
  when `n_in / 32 < 64`, and one otherwise. Read qh and qs as bytes because
  22-byte Q5_0 blocks are only two-byte aligned.
- **Reason:** The Q5_0 high bits are scattered across the entire qh plane;
  deriving each code by logical index directly matches the pinned decoder
  while avoiding aligned 32-bit reads from packed rows. Applying the original
  Q8_1 sum correction per thread and preserving the reduction sequence keeps
  the native arithmetic reproducible.
- **Review required:** `metal_mv_test` covers random, edge and fixture inputs
  at n_in = 640, 2016, 2048, 2560 and 6144, NC = 1, 2, 4 and 8, and every
  live-row count. The 2016/2048 widths test both sides of the 64-block
  four-row/one-row dispatch boundary. An analytic n_in = 640 fixture activates
  code indices 0, 17 and 31 with distinct codes 31, 17 and 14 and checks the
  literal answer `1778 * (row + 1) * (column + 1)`, including the inactive
  column sink. A separate raw-byte golden uses qh bytes
  `a5 3c 5a c3` and qs bytes `00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff`;
  the complementary qh bits make all 32 decoded codes unique, and literal
  one-hot answers check every code position. An n_in = 2080 two-block-per-thread fixture
  checks the fused accumulation answer `-11.19607448577880859375`, which
  differs by one float32 bit if the outer FMA is split into multiply and add.
  `mv_multi_eq_single` compares NC = 2, 4 and 8 with separate NC = 1 Metal
  dispatches across all five widths and flips one qh bit as its negative
  control. Exhaustive decoder coverage, production binding and model-backed
  `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-078: Keep both Q8_1 scales across a Q2_0 block

- **Decision:** Add `strata_mv_q2_0_n<NC>` for NC = 1, 2, 4 and 8. Each
  64-value Q2_0 block contains sixteen packed bytes and maps to two distinct
  32-value Q8_1 activation blocks. A thread processes one half, selected by
  `tid % 2`, while `kbx = tid / 2` advances through weight blocks in strides
  of 64. Decode each 2-bit code as `code - 1`; accumulate
  `fma(d_weight * d_activation, float(dot), acc)` and preserve the three
  cross-warp adds followed by the butterfly. Use four rows per threadgroup
  below 64 weight blocks per row, and one row at or above that boundary.
  Load packed codes by byte because Q2_0 blocks are only two-byte aligned.
- **Reason:** One weight block crosses two independently quantized activation
  blocks. Applying one activation scale to both halves would silently alter
  the native result. Byte decoding also avoids depending on wider alignment
  than the packed row format guarantees.
- **Review required:** `metal_mv_test` covers widths 64, 640, 2560, 4032,
  4096, 4160, 6144 and 10240, NC = 1, 2, 4 and 8, and every live-row count.
  The 4032/4096 widths straddle the 64-block row-group threshold; 4160 tests
  the next stride iterations, with independent 524256.0 and 1044448.0
  goldens for 65 and 129 blocks. The final weight and activation scales
  differ from earlier blocks so an incorrect second- or third-iteration
  address changes the result. A literal n_in = 64 fixture uses different
  activation scales for the two halves and checks padded-column output.
  Three raw-byte patterns make each of the 64 code positions unique across
  their combined signatures and check every position with one-hot inputs.
  `mv_multi_eq_single` compares every NC = 2/4/8 column to a separate NC = 1
  dispatch over all widths and flips a packed code bit as its negative
  control. Production binding, exhaustive `mv_decode` coverage and
  model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-079: Use the pinned nonlinear table for IQ4_NL mat-vecs

- **Decision:** Add `strata_mv_iq4_nl_n<NC>` for NC = 1, 2, 4 and 8.
  Each 32-value weight block is 18 bytes. Decode each byte's low nibble as
  element j and its high nibble as element j + 16 through the pinned
  `kvalues_iq4nl` table. A thread handles 16 values, with
  `first_word = 2 * (tid % 2)`; weight blocks advance in strides of 64.
  Accumulate `fma(d_weight * d_activation, float(dot), acc)`, then keep the
  three cross-warp adds and butterfly. Use four rows per threadgroup below
  64 weight blocks per row and one row at or above that boundary. Read packed
  nibbles by byte.
- **Reason:** IQ4_NL's values are a nonlinear codebook, and its split-half
  nibble order differs from a linear Q4 decoder. A direct byte decoder keeps
  the signed lookup values and logical weight positions explicit while
  preserving the native small-mat-vec layout.
- **Review required:** `metal_mv_test` covers widths 32, 640, 2016, 2048,
  2080, 2560 and 6144; NC = 1, 2, 4 and 8; and every live-row count. The
  2016/2048/2080 widths cover 63/64/65 weight blocks. Two literal raw-byte
  patterns with one-hot activations exercise all 16 table values at all 32
  positions; compile-time offset checks pin the 18-byte block layout. The
  analytic NC = 4 fixture checks padded-column output. The 65- and
  129-block fixtures check the literal fused result
  `-40.222751617431640625`, which differs by one float32 bit if the
  accumulation FMA is split.
  `mv_multi_eq_single` compares every NC = 2/4/8 result to separate NC = 1
  dispatches and flips a packed code bit as a negative control. Production
  binding, exhaustive `mv_decode` coverage and model-backed `real` inputs
  remain open under REV-025 / REV-026.

### D-IMPL-080: Match the native IQ4_XS mat-vec layout

- **Decision:** Add `strata_mv_iq4_xs_n<NC>` for NC = 1, 2, 4 and 8. Each
  256-value weight block is 136 bytes and contains eight 32-value subblocks.
  Decode the low nibble as the first half and the high nibble as the second
  half of each subblock through the pinned `kvalues_iq4nl` table. Reconstruct
  each six-bit subblock scale from `scales_l` and `scales_h`, multiply the
  integer dot by `scale_code - 32`, then accumulate
  `fma(d_weight * d_activation, float(dot), acc)`. Map `tid / 8` to weight
  blocks, `tid % 8` to subblocks, and advance blocks by 16. Use four rows per
  threadgroup below 16 weight blocks per row and one row at or above that
  boundary, matching the native launch condition.
- **Reason:** IQ4_XS is used by dense projections, the head and token
  embeddings in the documented model catalogue. Keeping its packed six-bit
  scale and nonlinear nibble interpretation explicit preserves the native
  Q8_1 mat-vec arithmetic and launch mapping.
- **Review required:** `metal_mv_test` covers widths 256, 2560, 3840, 4096,
  6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4 and 8; and every
  live-row count. The 3840/4096 widths exercise the 15/16-block row-group
  boundary, and 16128/16384/16640 exercise 63/64/65 blocks. Two raw nibble
  patterns with one-hot activations cover all 256 logical positions and all
  16 codebook values in each nibble half. Independently specified raw
  `scales_l` and `scales_h` bytes cover every low-bit position and all four
  high-bit combinations. The literal NC = 4 fixture checks padded-column
  output. The 65- and 129-block fixtures check the fused literal
  `0.64050829410552978515625`; splitting the FMA yields
  `0.6405084133148193359375`. `mv_multi_eq_single` compares every NC = 2/4/8
  result to separate NC = 1 dispatches across all widths and flips a packed
  code bit as a negative control. Production binding, exhaustive `mv_decode`
  coverage and model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-081: Match the native Q6_K mat-vec layout

- **Decision:** Add `strata_mv_q6_k_n<NC>` for NC = 1, 2, 4 and 8. Each
  256-value weight block is 210 bytes: 128 bytes of low nibbles, 64 bytes of
  high bits, 16 signed subblock scales and a half-precision block scale.
  Reconstruct each six-bit code as `low | (high << 4)`, then subtract 32.
  Each lane accumulates two Q8_1 dot groups using the corresponding signed
  Q6_K subblock scale; preserve `fma(d8, float(dot * scale), sum)` and the
  outer `fma(d_weight, sum, accumulator)`. Assign one row per threadgroup
  when a row has at least four Q6_K blocks and four rows below that threshold,
  matching the native CUDA launch.
- **Reason:** Q6_K is used by dense projections, shared experts and the model
  head in the documented catalogue. Its split six-bit packing and scale
  ordering require a format-specific decoder, and preserving its two-level
  FMA order and native row grouping keeps this kernel aligned with the
  reference arithmetic and dispatch shape.
- **Review required:** `metal_mv_test` covers widths 256, 512, 768, 1024,
  1280, 2560, 6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4 and 8; and
  every live-row count. Widths 768/1024 cover the 3/4-block row-group
  threshold; 16128/16384/16640 cover 63/64/65 blocks. A setter-based
  one-hot sweep uses four code patterns to cover all 256 logical positions
  and the 64 six-bit codes. A separate raw-layout fixture writes literal 128-byte
  `ql` and 64-byte `qh` arrays and checks independently derived answers at all
  256 positions; its code pattern varies between positions 64 values apart.
  Signed-scale coverage includes negative, zero and positive values. The
  analytic NC = 4 fixture checks distinct columns and padded-column output.
  The 5- and 9-block fixtures check the literal fused result
  `-266.151519775390625`; splitting the inner FMA changes the result.
  Independent long-K goldens check all 63, 64 and 65 blocks against `-32004`,
  `-30988` and `-30480`. `mv_multi_eq_single` compares every NC = 2/4/8 result
  to separate NC = 1 dispatches across all widths and flips a packed code bit
  as a negative control. Production binding, exhaustive `mv_decode` coverage
  and model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-082: Match the native Q3_K mat-vec layout

- **Decision:** Add `strata_mv_q3_k_n<NC>` for NC = 1, 2, 4 and 8. Each
  256-value block is 110 bytes: 32 high-mask bytes, 64 packed low-bit bytes,
  12 packed bytes for sixteen signed six-bit scales and one half-precision
  block scale. Decode a weight as the packed two-bit value minus four when
  its high-mask bit is clear; otherwise keep the value nonnegative. Rebuild
  each signed scale from its low nibble and two high bits, then subtract 32.
  Each lane consumes four Q8_1 dot groups, applying
  `fma(d8, float(dot * scale), sum)` followed by
  `fma(d_weight, sum, accumulator)`. Use a four-row threadgroup below eight
  Q3_K blocks per row and one row at or above that boundary, matching CUDA.
- **Reason:** Q3_K appears in dense projections and token embeddings in the
  documented catalogue. Its high-mask sign bit and non-linear packing of
  sixteen signed scales make a format-specific decoder necessary; preserving
  the native eight-block stride, launch threshold and FMA order follows the
  pinned Q8_1 mat-vec contract.
- **Review required:** `metal_mv_test` covers widths 256, 512, 768, 1792,
  2048, 2304, 2560, 6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4 and
  8; and every live-row count. Widths 1792/2048/2304 exercise the 7/8/9-block
  row-group threshold; 16128/16384/16640 cover 63/64/65 blocks. A literal
  `hmask`/`qs`/`scales` block with one-hot inputs checks independently listed
  answers at all 256 positions, all eight signed Q3 codes and signed scale
  extremes; each position pair sharing a scale group has distinct raw codes
  and answers. The analytic NC = 4 fixture checks distinct columns, a
  zero-scale row and padded output. The 9- and 13-block fixtures check
  FMA-sensitive literal results; independent long-K goldens check all 63, 64
  and 65 blocks. `mv_multi_eq_single` compares every NC = 2/4/8 result to
  separate NC = 1 dispatches across all widths and flips a packed code bit as
  a negative control. Production binding, exhaustive `mv_decode` coverage and
  model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-083: Match the native Q4_K mat-vec layout

- **Decision:** Add `strata_mv_q4_k_n<NC>` for NC = 1, 2, 4 and 8. A 256-value
  Q4_K block is 144 bytes: half-precision `dm`, twelve packed scale/min bytes
  and 128 nibble bytes. Decode the eight 32-value groups with the native
  six-bit scale/min layout. Each of 128 threads starts at `tid / 16`, strides
  by eight weight blocks and uses `2 * (tid % 16)` as `iqs`; the four Q8 words
  per group also select the matching four-byte activation lane offset. Below
  eight blocks per row use four rows per threadgroup; at eight or more use one.
  Preserve the two inner `fma(d8, dot * scale, sd)` /
  `fma(d8, sum_q8 * min, sm)` updates, then
  `fma(-dm.y, sm, dm.x * sd)` and the plain `acc + v`.
- **Reason:** Q4_K occurs in the documented dense projection and shared-expert
  formats. Its split six-bit scale/min packing and Q8 lane layout need a
  dedicated decoder; following the pinned native thread mapping and arithmetic
  order keeps the Metal implementation aligned with the documented Q8_1 path.
- **Review required:** `metal_mv_test` covers widths 256, 512, 768, 1792, 2048,
  2304, 2560, 6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4 and 8; and
  every live-row count. Widths 1792/2048/2304 exercise the 7/8/9-block
  threadgroup threshold, and 16128/16384/16640 exercise 63/64/65 blocks.
  Literal packed bytes and one-hot inputs check all 256 logical positions,
  all sixteen Q4 codes, and representative low/high six-bit scales and minima.
  The analytic NC = 4 fixture checks distinct columns and padded output;
  independently derived scalar lane/warp results pin the 9- and 13-block FMA
  paths (bits `0x44bf9107` and `0x4506f316`). Long-K fixtures pin 63/64/65-block
  results. The NC = 2/4/8 property compares each output against separate NC = 1
  dispatches across every width and includes a one-bit comparator negative
  control. Production Metal
  inference binding, exhaustive `mv_decode` coverage, target CUDA SASS
  confirmation of the final fused expression and model-backed `real` inputs
  remain open under REV-025 / REV-026.

### D-IMPL-084: Match the native Q5_K mat-vec layout

- **Decision:** Add `strata_mv_q5_k_n<NC>` for NC = 1, 2, 4 and 8. A 256-value
  Q5_K block is 176 bytes: half-precision `dm`, twelve packed scale/min bytes,
  32 high-bit bytes and 128 low-nibble bytes. For group `g` and local position
  `l`, decode `q = (qs[32 * (g / 2) + l] >> (4 * (g % 2)) & 15) |
  (((qh[l] >> g) & 1) << 4)`. Preserve the native Q8 lane offset in both QH
  (`qh[4 * q8_word + local]`) and activation reads. The scale/min decode,
  `tid / 16` start, eight-block stride, four-row threshold below eight blocks,
  FMA sequence and reduction match Q4_K's native layout.
- **Reason:** Q5_K appears in documented dense projections, shared experts and
  a Q5_K decode head. The high-bit plane is packed independently of the low
  nibbles and must be indexed alongside each Q8 lane. A dedicated kernel keeps
  these reads explicit while reusing the pinned K-quant scale/min and Q8_1
  arithmetic contract.
- **Review required:** `metal_mv_test` covers widths 256, 512, 768, 1792, 2048,
  2304, 2560, 6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4 and 8; and
  every live-row count. The 7/8/9-block and 63/64/65-block thresholds are
  checked. Literal `qh`, `qs` and scale bytes plus one-hot inputs check all 32
  Q5 codes at all 256 positions, both high-bit values in each group and
  representative six-bit scales/minima. The analytic NC = 4 fixture checks
  padded output; independent scalar lane/warp answers pin the 9- and 13-block
  FMA paths (bits `0x455ecdce` and `0x4597a411`); analytical long-K answers pin
  63/64/65 blocks. The NC = 2/4/8 property checks parity against separate
  NC = 1 runs using identical weights and has a one-bit comparator negative
  control. Production Metal inference binding, exhaustive `mv_decode`
  coverage, target CUDA SASS confirmation of the fused expression and
  model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-085: Match the native IQ3_S row-family mat-vec

- **Decision:** Add `strata_mv_iq3_s_n<NC>` for NC = 1, 2, 4 and 8 with one
  simdgroup per output row. Decode each 9-bit codebook index from `qs` and the
  matching `qh` bit, apply the eight sign bits in each `signs` byte to the
  little-endian `iq3s_grid` components, and accumulate eight signed byte-dot
  products. The sign-byte base is `4 * (iqs / 2)`, matching the four-byte
  `get_int_b2` load. Apply the per-32-value scale multiplier
  `1 + 2 * nibble`, then accumulate with `fma(d * d8, float(sumi), acc)` in
  CUDA lane-stride order and reduce with the explicit 32-lane butterfly.
- **Reason:** IQ3_S is used by documented routed-expert gate/up tensors. Its
  110-byte block and Q8_1 call mapping differ from the native K-quant kernels;
  a dedicated row-family kernel follows `row_dot` and `row_dot_multi` while
  preserving the codebook, sign, scale and reduction operations.
- **Review required:** `metal_mv_test` covers n_in 256, 512, 768, 1024, 1280,
  1792, 2048, 2304, 2560, 6144, 10240, 16128, 16384 and 16640; NC = 1, 2, 4
  and 8; and every live-row count. The 3/4/5-, 8/9- and 63/64/65-block
  lane-stride boundaries are checked. Literal raw bytes and one-hot inputs
  cover all 512 codebook indices, all qh bit positions, both signs at every
  value position and all 16 scale nibbles. Analytic padded-column output,
  independent scalar lane/warp FMA answers for 9/13 blocks (bits `0x48a33700`
  and `0x48b6e750`), analytical 63/64/65-block answers, NC parity and a
  one-bit negative control pass. Production Metal inference binding,
  exhaustive `mv_decode`, CUDA SASS confirmation of the final fused expression
  and model-backed `real` inputs remain open under REV-025 / REV-026.

### D-IMPL-086: Test the IQ2_XXS expert row dot before grouped integration

- **Decision:** Add `strata_mv_iq2_xxs_n<NC>` for NC = 1, 2, 4 and 8 as a
  standalone parity twin of the IQ2_XXS row dot. Keep it in the Metal library
  and manifest so the packed decoder, integer scaling, lane order and Q8_1
  mapping can be tested independently before `strata_moe_gu_iq2_xxs` is
  implemented. Dense tensor dispatch must not select this kernel.
- **Reason:** IQ2_XXS appears in the catalogue as routed-expert gate/up weights,
  and section 5.8 specifies its row-dot arithmetic for the grouped expert path.
  Testing it through a dedicated row kernel isolates format errors from the
  much larger MoE grouping implementation and gives that later kernel a
  byte-exact twin for reuse.
- **Review required:** Reassess the extra four manifest entries and startup
  pipeline cost when L7 integrates grouped experts. Keep the standalone twin
  if it remains useful for decoder isolation; otherwise move its exhaustive
  format checks to the grouped kernel before removing it.

### D-IMPL-087: Test the IQ2_XS expert row dot before grouped integration

- **Decision:** Add `strata_mv_iq2_xs_n<NC>` for NC = 1, 2, 4 and 8 as a
  standalone parity twin for the IQ2_XS row dot. Keep it in the Metal library
  and manifest so the 9-bit grid index, sign lookup, half scales and signed
  integer divisions can be checked before `strata_moe_gu_iq2_xs` is
  implemented. Dense tensor dispatch must not select this kernel.
- **Reason:** IQ2_XS is used by routed experts, and its two half-scale paths
  and nested signed divisions add arithmetic cases beyond the IQ2_XXS twin.
  Isolating its row dot makes packed decode and truncation failures visible
  before they are combined with grouped MoE scheduling.
- **Review required:** Reassess the additional four manifest entries and
  startup pipeline cost when L7 integrates grouped experts. Retain the twin
  while it provides useful decoder isolation; otherwise move its format
  checks to the grouped kernel before removing it.

### D-IMPL-088: Test the IQ2_S expert row dot before grouped integration

- **Decision:** Add `strata_mv_iq2_s_n<NC>` for NC = 1, 2, 4 and 8 as a
  standalone parity twin for IQ2_S expert gate/up. Register the four
  specialisations for packed decoder tests before `strata_moe_gu_iq2_s` is
  implemented. Dense tensor dispatch must not select this kernel.
- **Reason:** IQ2_S is used by routed experts and differs from IQ2_XS in its
  1024-entry grid assembled from low `qs` bytes and `qh` high bits, plus its
  explicit per-component sign plane. Testing those fields outside grouped MoE
  scheduling makes packed-index, sign and scale-half errors easier to isolate.
- **Review required:** Reassess the additional four manifest entries and
  startup pipeline cost when L7 integrates grouped experts. Retain the twin
  while it provides useful decoder isolation; otherwise move its format
  checks to the grouped kernel before removing it.

### D-IMPL-089: Test the IQ3_XXS expert row dot before grouped integration

- **Decision:** Add `strata_mv_iq3_xxs_n<NC>` for NC = 1, 2, 4 and 8 as a
  standalone parity twin for IQ3_XXS expert gate/up. Register the four
  specialisations for packed decoder tests before `strata_moe_gu_iq3_xxs` is
  implemented. Dense tensor dispatch must not select this kernel.
- **Reason:** IQ3_XXS uses a 98-byte block with eight 32-value parts, two grid
  indices per four-value code, parity-completed 7-bit sign indices and a
  high-nibble scale. Its integer dot also applies
  `(scale * sumi + sumi / 2) / 2`, where signed division truncates toward zero.
  Testing these together with the Q8_1 lane mapping before grouped MoE
  scheduling isolates packed-byte, sign, scale and rounding errors.
- **Review required:** Reassess the additional four manifest entries and
  startup pipeline cost when L7 integrates grouped experts. Confirm the
  explicit FMA form against upstream CUDA SASS under L-9; the reference CUDA
  machine is unavailable under REV-002. Retain the twin while it provides
  useful decoder isolation; otherwise move its format checks to the grouped
  kernel before removing it.

### D-IMPL-090: Test the IQ1_M expert row dot before grouped integration

- **Decision:** Add `strata_mv_iq1_m_n<NC>` for NC = 1, 2, 4 and 8 as a
  standalone parity twin for IQ1_M expert gate/up. Register the four
  specialisations for packed-decoder tests before implementing grouped MoE.
  Dense tensor dispatch must not select this kernel.
- **Reason:** IQ1_M's 56-byte block packs four 11-bit codebook indices and
  delta selectors per Q8_1 part, two odd local scales, and a half-precision
  super-scale across four 16-bit words. The standalone sweep can verify those
  layouts and the FMA/lane order without conflating them with expert grouping.
- **Review required:** Reassess the additional four manifest entries and
  startup pipeline cost when L7 integrates grouped experts. Confirm the
  explicit FMA form against the pinned CUDA implementation and SASS under
  L-9; the reference CUDA machine is unavailable under REV-002. Retain this
  twin while it provides useful decoder isolation; otherwise move its format
  checks to the grouped kernel before removing it.

### D-IMPL-091: Implement the MoE router as the first L7 kernel slice

- **Decision:** Add and test the standalone `r16` and `r8` Metal router kernels
  and their scalar oracle. Add a fixed-input weight-mode probe generated from
  the same kernel body, while leaving plan, expert, shared-expert, combine
  and production dispatch work open.
- **Reason:** The router has a self-contained buffer contract and exact
  lane/reduction algorithm in section 5.10. Its IDs and row independence can
  be checked before grouping or model execution exists; keeping this slice
  separate makes it possible to resolve router parity issues before those
  kernels depend on its outputs. General route outputs can differ by only one
  ULP between reciprocal multiplication and division, so they cannot reliably
  prove which mode the `divide_weights` flag selects. A fixed `0.1f / 0.3f`
  probe checks distinct result bits through the same shared-body call site;
  this avoids dependence on platform-specific `exp` results.
- **Review required:** Finish L7 integration and validate the router against
  CUDA captures/model-backed inputs when the reference assets and GPU are
  available. This standalone twin is not evidence that MoE inference is wired
  into the engine.

### D-IMPL-092: Implement the MoE plan kernel against the tiered host ABI

- **Decision:** Add `strata_moe_plan` as a standalone kernel that emits the
  existing `TieredMoePlan` layout, groups entries in first-appearance order,
  and handles both full and tiered residency in one dispatch.
- **Reason:** The engine already allocates this shared plan structure and its
  resolver validates `grp_res` as a 0/1 resident flag. Keeping the plan kernel
  independent lets the grouping, active-layer offsets, usage increments, and
  exact output layout be verified before grouped expert reads or handshakes
  are connected. The twin covers all T values from 1 through 8, both supported
  router widths, and all resident/cold mixtures.
- **Review required:** Connect the kernel to production layer dispatch and the
  `ExpertStore` miss handshake. Preserve the tested full-mode rule
  (`grp_res=1`, no usage update) and tiered entry-order usage increments.

### D-IMPL-093: Verify routed-expert SwiGLU as a standalone kernel

- **Decision:** Add `strata_moe_swiglu` for the T·10 routed rows before
  grouped expert mat-vecs are connected.
- **Reason:** The elementwise gate/up transform has a fixed 640-value row
  contract and no dependency on weight formats or residency. Testing all
  supported T values with random, edge and fixed fixture inputs isolates its
  math, row indexing and tail handling from expert weight decoding.
- **Review required:** Connect the result to the routed-expert pipeline and
  compare the complete expert path against CUDA captures/model-backed inputs
  when those assets are available.

### D-IMPL-094: Keep the macOS compute backend Metal-only

- **Decision:** Do not add CUDA runtime, CUDA kernels, or a CUDA backend to the
  macOS application. Implement GPU execution with Metal; retain CUDA sources
  and captures only as external references for formulas, packed formats, and
  parity checks where a CUDA reference machine is available.
- **Reason:** CUDA is not a macOS execution target for this project. A second
  GPU backend would add build, runtime, and maintenance paths without helping
  Mac users. The documented Metal-versus-CUDA parity gates can still use
  externally produced CUDA results.
- **Review required:** Keep CUDA mentions in the implementation plan scoped to
  reference behavior, source provenance, and cross-machine validation. Revisit
  only if the project explicitly adds a non-macOS CUDA product target.

### D-IMPL-095: Add grouped MoE formats in isolated kernel slices

- **Decision:** Implement grouped expert kernels one format at a time, starting
  with Q2_0 gate/up. The kernel uses the documented `TieredMoePlan` grouping,
  the existing layer/expert/role GPU-address table, and the CUDA reference
  row-dot four-entry batching contract. CUDA remains an external reference;
  the macOS implementation is Metal-only under D-IMPL-094.
- **Reason:** Grouped reads combine a pointer-table ABI, entry ordering, and
  quantized row arithmetic. Establishing those independently with Q2_0's
  existing block layout and row-dot oracle makes faults attributable before
  adding the more complex IQ decoders.
- **Review required:** Before production dispatch, validate both full and
  tiered address-table updates, and ensure every cold expert is resolved and
  its lease retained before any grouped kernel reads it. Each later format
  needs its own decoder fixtures and grouped-versus-single property coverage.

### D-IMPL-096: Add the grouped IQ1_M gate/up slice

- **Decision:** Add `strata_moe_gu_iq1_m` using the same plan, address table,
  four-entry batching and output layout as Q2_0, with IQ1_M's eight 32-value
  parts per 256-value block. Keep a separate scalar oracle that carries
  `twin::tf` magnitudes through IQ1_M's inner FMA and lane reduction.
- **Reason:** IQ1_M decoding already has a standalone Metal mat-vec and oracle.
  Reusing that established arithmetic lets grouped scheduling be checked
  independently, including arbitrary T values and the 80-entry plan limit.
- **Fixture and scale policy:** Random half scales use the documented
  log-uniform range [2^-12, 2^-4]. The grouped fixture embeds raw blocks
  generated by `tools/iq_fixture.py`; its IQ1_M scale repair is checked against
  the pinned ggml decoder and emits the 1.0 half scale (0x3c00), while
  preserving and covering all eight encoded local-scale values.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration is
  implemented and model-backed parity is available.

### D-IMPL-097: Add the grouped IQ2_XXS gate/up slice

- **Decision:** Add `strata_moe_gu_iq2_xxs` with the established plan,
  address-table, four-entry batching and gate/up output contract. Preserve the
  IQ2_XXS row dot's sign parity, integer `/8` truncation and FMA/reduction
  order. Use a TF-aware scalar oracle and share the standalone twin's fixed
  packed block and one-hot answers.
- **Reason:** IQ2_XXS already has an independent Metal row-dot twin and raw
  packed goldens. The grouped kernel can therefore exercise scheduling and
  expert addressing while the fixed one-hot case separately pins its decoder
  arithmetic. Testing 80 distinct groups also covers the plan's maximum
  endpoint.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs include zero, subnormal and ordinary finite half
  values.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration,
  external CUDA/model-backed parity and end-to-end inference are available.

### D-IMPL-098: Add the grouped IQ2_XS gate/up slice

- **Decision:** Add `strata_moe_gu_iq2_xs` with the established plan,
  address-table, four-entry batching and gate/up output contract. Preserve the
  9-bit grid and 7-bit sign lookup, two 4-bit scale paths, signed `/2` and
  `/4` truncation, and FMA/reduction order. Use a TF-aware scalar oracle and
  share the standalone twin's fixed packed block and eight one-hot answers.
- **Reason:** IQ2_XS already has a standalone Metal row-dot twin with packed
  goldens. Reusing that fixture isolates decoder parity while the grouped
  tests exercise plan traversal, expert addresses and output ownership.
  Eighty distinct groups cover the plan's maximum endpoint and use separate
  gate/up scales so an incorrect slot mapping cannot alias identical answers.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs include zero, subnormal and ordinary finite half
  values, along with varied local scale bytes.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration,
  external CUDA/model-backed parity and end-to-end inference are available.

### D-IMPL-099: Add the grouped IQ2_S gate/up slice

- **Decision:** Add `strata_moe_gu_iq2_s` with the established plan,
  address-table, four-entry batching and gate/up output contract. Preserve the
  `qh` high bits in the 10-bit grid index, explicit sign bytes, two 4-bit scale
  paths, signed `/2` and `/4` truncation, and FMA/reduction order. Use a
  TF-aware scalar oracle and share the standalone twin's fixed packed block,
  high-code bytes, scales and one-hot answers.
- **Reason:** IQ2_S differs from IQ2_XS in how it forms the grid index and
  stores component signs. Its standalone Metal row-dot twin isolates that
  decoder, while the grouped kernel tests expert addressing and plan traversal.
  The 80-group fixture uses separate slots and gate/up scales so address
  aliasing produces observably wrong answers.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs cover grid high bits, sign bytes, varied local
  scales, zero, subnormal and ordinary finite half values.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration,
  external CUDA/model-backed parity and end-to-end inference are available.

### D-IMPL-100: Add the grouped IQ3_XXS gate/up slice

- **Decision:** Add `strata_moe_gu_iq3_xxs` using the established grouped plan,
  expert-address, four-entry batching and gate/up output contract. Decode the
  256-value block's eight 32-value parts, restore the parity-coded sign byte,
  use the packed scale nibble with signed `/2` truncation, and preserve the
  standalone row-dot's FMA and butterfly order. Add a TF-aware scalar oracle
  and share a fixed packed block plus its one-hot answers with the standalone
  IQ3_XXS twin.
- **Reason:** IQ3_XXS's sign restoration and packed auxiliary word are
  format-specific. Integrating only after its standalone row-dot twin exists
  lets the grouped test check both decoder parity and the distinct expert
  address mapping. The 80-group case gives every gate/up expert a distinct
  scale and checks the maximum plan endpoint against a fixed one-hot answer.
  This is a Metal implementation; the CUDA row-dot behavior is reference
  material under D-IMPL-094, not a Mac execution target.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs include zero, subnormal and ordinary finite half
  values.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration,
  external CUDA/model-backed parity and end-to-end inference are available.

### D-IMPL-101: Add the grouped IQ3_S gate/up slice

- **Decision:** Add `strata_moe_gu_iq3_s` with the established grouped plan,
  expert-address, four-entry batching and gate/up output contract. Decode each
  9-bit grid index from `qs` plus its `qh` bit, restore component signs from
  the packed sign nibbles, multiply each 32-value dot by `1 + 2*nibble`, and
  preserve the standalone row-dot's FMA and butterfly order. Add a TF-aware
  scalar oracle and share one fixed packed block and one-hot answer with the
  standalone IQ3_S mat-vec fixture.
- **Reason:** IQ3_S is a documented routed-expert format whose high index bit,
  sign packing and scale nibble differ from IQ3_XXS. A separate grouped slice
  uses the existing standalone row oracle while testing grouped address and
  entry mapping. The 80-group fixture assigns distinct gate/up scales and
  validates its known one-hot result at the maximum plan endpoint. This is a
  Metal implementation; CUDA remains external reference material under
  D-IMPL-094.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs include zero, subnormal and ordinary finite half
  values and exercise packed code, high-bit, sign and scale bytes.
- **Review required:** Keep full/tiered production address updates and the
  cold-expert resolver/lease handshake open until dispatch integration,
  external CUDA/model-backed parity and end-to-end inference are available.

### D-IMPL-102: Add the grouped IQ4_XS gate/up slice

- **Decision:** Add `strata_moe_gu_iq4_xs` using the grouped plan,
  expert-address, four-entry batching and gate/up output contract. Decode the
  eight 32-value parts per 256-value block, reconstruct each 6-bit scale from
  `scales_l` and `scales_h`, use the IQ4_NL codebook for both packed nibbles,
  multiply the integer dot by `scale_code - 32`, and preserve the row kernel's
  per-lane FMA and butterfly order. Add a TF-aware row oracle and grouped
  oracle. Share one literal packed block and its position-41 one-hot answer
  between the standalone mat-vec and grouped twins.
- **Reason:** IQ4_XS is the final documented gate/up format. Its split scale
  plane and low/high code nibbles deserve a decoder fixture independent of the
  grouped address traversal. The 80-group fixture assigns distinct gate/up
  block scales to every active expert and checks each one-hot answer against
  the analytic `fma(scale, 215265, 0)` result. This is a Metal implementation;
  CUDA remains external reference material under D-IMPL-094.
- **Scale policy:** Random block scales use the log-uniform range
  [2^-12, 2^-4]; edge inputs cover split scale bits, code nibbles, zero,
  subnormal and ordinary finite half values.
- **Review required:** Keep down-format kernels, shared expert, combine,
  production dispatch, full/tiered address updates and the cold-expert
  resolver/lease handshake open until their own synthetic twins,
  dispatch integration, external CUDA/model-backed parity and end-to-end
  inference are available.

### D-IMPL-103: Add the grouped Q2_0 MoE down slice

- **Decision:** Add `strata_moe_down_q2_0` for the documented Q2_0 expert
  down shape: 2,560 output rows, ten 64-value Q2_0 blocks per row and 20
  Q8_1 dot calls. Use role 2 of the expert address table, the four-entry
  grouping loop and the row-family FMA/butterfly order. Read activations in
  grouped-entry order and restore each result with `plan.ent_dst`. Add a
  TF-aware scalar oracle and synthetic random, edge and fixture twins for
  256/512 experts and T=1..8.
- **Reason:** Down results are consumed in original token/rank order, while
  the plan groups entries by expert. `ent_dst` therefore has to be tested as
  part of the output contract, not treated as incidental bookkeeping. The
  maximum-group fixture permutes 80 expert IDs, gives all 80 weight slots
  distinct half scales, and checks the fixed position-0 one-hot result
  `fma(scale, 127, 0)`.
- **Validation-time decision:** Raise `metal_moe_test`'s CTest timeout from
  120 to 180 seconds. The all-input-set ASan run takes 127.30 seconds after
  adding this down kernel, so the old limit would reject a passing run.
- **Review required:** Keep shared-expert execution, production dispatch,
  full/tiered address updates and the cold-expert resolver/lease handshake
  open until their own twins and end-to-end integration are tested.

### D-IMPL-104: Add the grouped IQ4_NL MoE down slice

- **Decision:** Add `strata_moe_down_iq4_nl` for the documented 2,560-row
  expert down matrix with 20 IQ4_NL blocks per row. Each Q8_1 activation block
  contributes two 16-value integer dots; lanes traverse 40 calls with
  `block = k / 2`, `part = k % 2`, and apply an FMA for each partial before
  the butterfly reduction. Read grouped entries from the 640-wide activation
  and write results back through `plan.ent_dst`.
- **Oracle and fixture:** Decode IQ4_NL nibbles by logical value position and
  compute a scalar row reference without copying the kernel's lane assignment
  or butterfly. Test random, edge and fixture inputs for 256/512 experts and
  T=1..8. The 80-entry permuted fixture assigns every expert a distinct half
  scale; its two active partial dots are `-228600` and `131445`, and at least
  one scale makes the exact split-FMA result differ from one FMA over the
  combined dot. Check all 80 GPU answers bitwise, compare grouped entries
  against a forced singleton plan bitwise, and flip one packed code bit in a
  copied weight buffer as a negative control against an independently
  calculated changed answer. Seed the negative-control output with the
  unmodified GPU output so a missing write cannot masquerade as a detected
  change.
- **Reason:** The two halves of a 32-value IQ4_NL block are separate upstream
  FMA contributions. Combining their integer dots changes floating-point
  rounding, and reproducing the same nibble and lane arithmetic in the CPU
  oracle could hide a shared indexing error. The logical-position oracle,
  multi-lane analytic fixture and forced-single dispatch check these
  contracts independently.
- **Review required:** Keep shared-expert execution, production dispatch,
  full/tiered address updates and the cold-expert resolver/lease handshake
  open until their own twins and end-to-end integration are tested.

### D-IMPL-105: Add a fixed-order MoE combine with rounding-sensitive goldens

- **Decision:** Implement `strata_moe_combine` for 2,560-wide rows and top-k
  10. Preserve the plain first product, explicit FMA for ranks 1 through 9,
  the shared-output multiply and the final add as separate float operations.
  Add a row-major scalar oracle and random, edge and fixture twins for T=1..8.
  Pin the fused routed-rank accumulation with an exact bitwise golden. For the
  final shared multiply/add, use a same-dispatch coordinate to observe the
  device's sigmoid result, calculate the separately rounded expected value,
  and require it to differ from the FMA counterfactual. For an explicit
  `--input-set real`, return 77 and configure a host CTest to report a skip
  until the model pack and CUDA dump fixture are available; do not report an
  unrun real-input twin as passed.
- **Reason:** The first and final operations have different contraction rules
  from ranks 1 through 9, and a tolerance-only comparison can hide an unwanted
  FMA at either boundary. The planted rank fixture has an analytic float
  answer. The same-dispatch sigmoid probe makes the final-add check independent
  of implementation-specific `exp` rounding while still comparing the GPU
  output bitwise. The model-backed tensors required by REV-002 are absent, so
  an explicit CTest skip accurately represents the unavailable real-input
  check.
- **Validation:** The focused `metal_moe_test` passes for random, edge and
  fixture inputs. The full build and CTest suite pass (42 passed, one
  model-input test skipped). The ASan `metal_mv_test` and `metal_moe_test`
  pass, and `tools/metal/test_map.py` reports all checks passed.
- **Review required:** Shared-expert execution, production dispatch,
  full/tiered address updates, the cold-expert resolver/lease handshake, and
  external CUDA/model-backed parity remain open.

### D-IMPL-106: Add the shared-expert SwiGLU entry point

- **Decision:** Export `strata_shx_swiglu` with the documented grid
  `(3, T) × 256` and reuse the same device helper as routed
  `strata_moe_swiglu`, whose grid is `(3, T·10) × 256`. Exercise the two
  entry points through the existing scalar SwiGLU oracle with random, edge and
  fixture inputs for T=1..8.
- **Reason:** The shared expert has one 640-value row per token, while routed
  experts have ten rows per token. A separate manifest entry is needed for the
  shared grid even though both kernels must perform identical per-element
  arithmetic. The third threadgroup writes only 128 values, so the dedicated
  twin also checks its tail boundary and output guards.
- **Validation:** Focused `metal_moe_test` passes with both SwiGLU entry points
  across random, edge and fixture inputs. The full build and CTest suite pass
  (42 passed, one model-input test skipped), and the ASan `metal_moe_test`
  passes.
- **Review required:** Shared-expert gate/up/down execution, production
  dispatch, full/tiered address updates, the cold-expert resolver/lease
  handshake, and external CUDA/model-backed parity remain open.

### D-IMPL-107: Chain the synthetic shared-expert GPU stages

- **Decision:** Add a composed scalar oracle and a GPU integration twin for
  Q8_1 input quantization, Q8_0 gate/up, shared-expert SwiGLU, Q8_1 hidden
  quantization, Q8_0 down, the BF16 gate and the shared-gated combine. Run
  T=1..8 with the smallest supported NC that contains T, zero-pad inactive
  rows, and check every partial-batch sink and buffer guard. Feed each GPU
  stage's actual output into the next stage. Use exact-byte checks for
  quantization, Q8_0 and BF16; use U(16) for SwiGLU and combine. The combine
  uses synthetic routed parts and weights while consuming the GPU-produced
  shared output and gate logits.
- **Reason:** The documented model pack and CUDA dump inputs are unavailable
  under REV-002. A composed synthetic twin still checks the real Metal ABI,
  row layouts, quantization boundaries, partial batches and stage ordering,
  and catches integration faults that independent kernel twins cannot. Each
  stage also has its own scalar comparison, so the composed test does not
  depend only on a final output that could hide compensating errors.
- **Synthetic data:** Deterministic random, edge and fixture mixed activations;
  Q8_0 matrices and BF16 gate weights are seeded per input set. The routed
  combine branch remains synthetic and is not evidence of model-backed MoE
  parity.
- **Validation:** `metal_moe_test --input-set random`, `edge` and `fixture`
  pass for T=1..8. The ASan CTest run passes in 124.25 seconds with its
  existing 180-second timeout. Full CTest passes (42 tests pass, and the
  explicit real-input check is skipped); the CTest `test_map` also passes.
- **Review required:** Production dispatch, full/tiered miss-handshake
  integration, actual model weights and external CUDA parity remain open.

### D-IMPL-108: Bring up sampler kernels as Metal-only primitives

- **Decision:** Implement greedy selection, split top-k selection/merge,
  coupled penalty application and the deterministic FP64 CPU sampling tail as
  Metal-path primitives. Keep the sampler bounds at T = 1..8, 64 candidates
  and 262,144 vocabulary entries. Scan greedy logits in 131,072-token chunks
  and reuse the bitmap allocation for SIMD partial results so dynamic and
  static threadgroup memory stays below the 32 KiB device limit. Round dynamic
  bitmap allocations up to a 16-byte multiple while indexing only the required
  uint words. Make greedy CPU-tail calls return the first candidate with
  probability 1. Seed and input-set options drive the synthetic matrices; the
  default parity test runs random, edge and fixture sets. Use upstream CUDA
  code as a reference; do not add CUDA runtime or kernels to the macOS
  application, consistent with D-IMPL-094. Port host fixture 15 and semantic
  cases 16-17 with deterministic integer-generated background logits; keep
  literal fixtures 1-14 and 18 and production inference wiring open under
  REV-028.
- **Reason:** Metal has no FP64 arithmetic for the specified sampled tail, so
  matching its ordered top-p/min-p/temperature arithmetic on the CPU avoids an
  unsupported GPU operation. Chunking keeps the maximum greedy bitmap and its
  reduction values within 16,640 B. Metal's validation layer requires dynamic
  threadgroup-memory lengths to be multiples of 16 bytes; a two-word bitmap
  therefore still needs a 16-byte allocation. Greedy sampling must remain an
  argmax with probability 1 even when more than one candidate is available.
- **Validation:** The kernel twin passes random, edge and fixture inputs for
  T = 1..8, including history-penalized greedy at 262,144 vocabulary entries.
  It also exercises the coupled path with nonzero `kpart`, nonempty history
  and penalty bypass after the GPU penalty stage. The parity twin passes with
  `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1`; `sampler_tail_test` pins
  strict inverse-CDF, filter ordering, Philox and multi-candidate greedy
  behavior and upstream fixture 15's `penalty_rows` history layout across 420
  cases. After adding fixture 16/17, both pass with `--seed 77 --input-set
  fixture` normally and under `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`;
  the integer generator known answers pass. The post-change full host/Metal
  CTest run passes (45 passed, one expected real-input skip), including
  `test_map`; an explicit `test_map.py` run with `STRATA_BUILD_DIR=build-dev`
  and `git diff --check` also pass.
- **Review required:** REV-028 tracks the complete upstream fixture set and
  production window/head/MTP dispatch. The hostile review also found the
  maximum-vocabulary scratch overrun, nonzero-kpart/bypass coverage gap,
  ignored test input options and multi-candidate greedy-tail behavior; the
  implementation and tests address these findings. External CUDA and
  model-backed parity remain unavailable under REV-002.

### D-IMPL-109: Expose sampler kernels through Metal command-buffer encoders

- **Decision:** Add a Metal-only host API for encoding greedy selection,
  split top-k, split merge, sampled selection and coupled-draft penalties.
  Validate sampler shapes, buffer bounds and alignment, row-history ranges,
  disjoint bound views, `kpart` against each row's requested top-k, coupled
  vocabulary-subset bounds, manifest threadgroup sizes and device
  threadgroup-memory limits before encoding. Require buffers and the command
  buffer to belong to the same Context, and require exclusive single-thread
  use of each command buffer. Share sampler parameter structs between C++ and
  Metal. Keep command-buffer ownership with the caller; if a later stage
  fails after an earlier stage was encoded, the caller must discard the whole
  command buffer.
- **Reason:** The Metal kernels and their parity fixtures existed without a
  production host entry point. This API exercises the same dispatch contract
  production inference can use while keeping unfinished Window, head, MTP and
  Prefill integration from being presented as a working model path. CUDA
  remains external reference material under D-IMPL-094.
- **Validation:** `metal_sampler_parity_test` now exercises the public
  encoders for greedy, split-part, split-merge and coupled-penalty work, as
  well as the composed sampled encoder. Its negative checks cover unsupported
  row counts, insufficient `kpart`, undersized and misaligned views, and an
  already-enqueued command buffer, as well as overlapping views and invalid
  coupled vocabulary sizes. The fixture run passes with Metal shader and
  debug validation enabled. The full build passes; host/Metal CTest passes
  (45 passed, one expected real-input skip), including `test_map`. An explicit
  `test_map.py` run and `git diff --check` also pass.
- **Review required:** Connect the encoders to a real Window only after its
  layer/head dispatch and sampler scratch lifetime are implemented. Add
  parameter and scratch lifetime to the production contract; retain
  vocabulary fallback/refusal, upstream fixture and coupled-round publication
  gaps under REV-028.

### D-IMPL-110: Port embedding gather and GR stream broadcast to Metal

- **Decision:** Add Metal gather kernels for Q3_K, IQ3_S, IQ4_XS and BF16
  embeddings using the documented 10-by-256 grid. A token with bit 31 set
  returns without writing the embedding row, leaving the later image path to
  supply it. Add the 40-by-256 GR broadcast as an integer-bit copy so every
  source bit pattern is preserved. Compare gather outputs with the pinned
  ggml CPU type-trait decoder.
- **Reason:** The Mac runtime is Metal-only; upstream CUDA remains numerical
  reference material under D-IMPL-094. The gather uses the model's fixed
  2,560-value row and the four formats specified in section 5.2. The image
  gather, PLE, the other GR operations and production Window/Prefill wiring
  are separate unfinished work and are not implied by these primitives.
- **Validation:** The `metal_embed_test` CTest passes for random, edge and
  fixture inputs. It checks the four gather formats at T=1 and T=7 against
  ggml CPU, verifies image-token rows remain untouched, and checks broadcast
  bits for T=1, 3 and 8. The fixture run also passes with Metal API and shader
  validation enabled. After synchronizing the engine and metallib, the full
  host/Metal CTest run passes 46 tests with one expected real-input skip,
  including `test_map`; the explicit `test_map.py` run and `git diff --check`
  also pass.
- **Review required:** `encode_embed_rows` now checks every non-image token
  id against the supplied vocabulary size; production Window/Prefill still
  need to pass the model's vocabulary and keep token storage unchanged until
  command completion. `strata_embed_image`, the PLE and the rest of GR remain
  open under REV-029; do not treat the registered partial twin as completion
  of L4.

### D-IMPL-111: Add the shared weighted RMS norm kernel

- **Decision:** Implement `strata_rms_norm_w` with a 1,024-thread threadgroup
  for the documented 2,560- and 10,240-column rows. Each lane accumulates its
  strided squares with explicit FMA, reduces its 32-lane simdgroup with the
  documented butterfly, writes one partial per simdgroup, then all
  simdgroups reduce those 32 partials in the same butterfly order. Apply the
  correctly rounded reciprocal square root and multiply by the row selected
  with `row % gp`. The scalar twin propagates the documented reduction
  magnitude and checks U(64), while reporting the observed maximum U(n) and
  exact-bit count. Support aliased input/output, as the PLE uses in-place
  normalization.
- **Reason:** PLE, the head mixer and MTP all need the same weighted
  normalization. The fixed reduction has up to ten strided FMAs, two
  five-stage butterflies, mean and epsilon operations, `rsqrt` and two output
  multiplies. Conservative propagation for the tested finite range is about
  42 units, rounded up to U(64). The reference M5 Pro currently matches the
  host oracle bitwise, but the project does not guarantee that exact host/GPU
  result across all supported Apple GPU generations. This tolerance is for
  the Metal oracle; upstream CUDA remains separate reference material under
  D-IMPL-094, and Mac execution stays Metal-only.
- **Validation:** `metal_gr_test` passes random, edge and fixture inputs for
  2,560 columns with gamma periods 1 and 4, and 10,240 columns with period 1.
  It checks one and batched rows, U(64), separate and in-place output, and
  guarded buffers. The M5 Pro's safe-mode `rsqrt` matched a double-derived
  FP32 reference on all 4,194,304 inputs in the existing numerics measurement;
  the current test cases also match the oracle bitwise on that machine. The
  fixture run passes with Metal API and shader validation enabled. The synced
  host/Metal CTest suite passes 47 of 48 tests; `metal_moe_real_inputs_unavailable`
  is the expected skip because model inputs are unavailable. `test_map.py`
  and `git diff --check` pass.
- **Review required:** `encode_gr_norm` validates the fixed GR norm host
  contract under D-IMPL-125, and `encode_weighted_rms_norm` validates the
  shared weighted normalization contract under D-IMPL-131. PLE host
  validation and production Window/Prefill integration remain open under
  REV-029. U(64) covers the documented finite test range, not extreme or
  subnormal inputs.

### D-IMPL-112: Add the native GR post write

- **Decision:** Implement `strata_gr_post` as the documented `(40, T) × 256`
  elementwise kernel using `N_EMBD` and `HC` from the shared parameter header.
  Preserve the native order: `fma(0.25, inject, 0)`, `sigmoid_k`, `fma(2,
  sigmoid, 0)`, then `fma(block_out, weight, residual)`. Exact residual/output
  alias is supported; other overlap rules remain a host responsibility.
- **Reason:** This write is needed after layer 0 before the PLE block and after
  layer 47. Keeping the native post arithmetic preserves the documented
  flavour independently from the fused GR read/write path. CUDA is reference
  material only; this is a Metal kernel and adds no CUDA Mac path.
- **Validation:** `metal_gr_test` exercises `T=1` and `T=8` with random, edge
  and fixture inputs, checks separate output and exact residual/output alias,
  and compares against an independent scalar oracle at U(16). It verifies
  read-only inputs and buffer guards. The reference M5 Pro reported a maximum
  of U=3.91 for the current fixtures. Targeted CTest and the fixture run with
  Metal API and shader validation enabled pass. The synced full build and
  host/Metal CTest pass 47 of 48 tests; `metal_moe_real_inputs_unavailable`
  is the expected skip because model inputs are unavailable. `test_map.py`
  and `git diff --check` pass.
- **Review required:** The kernel assumes the documented fixed dimensions.
  Host-side bounds validation and the overlap policy are implemented under
  D-IMPL-128, including exact residual/output aliasing only. Production
  Window/Prefill integration remains open under REV-029; weighted norm and
  head-mixer host encoders remain incomplete.

### D-IMPL-113: Add the in-place GR down SiLU

- **Decision:** Implement `strata_gr_down_silu` as a one-dimensional
  256-thread kernel over `n_values = T * HC_LOW_RANK`, with `n_values` passed
  as a `setBytes` constant. Each value uses `fma(0.25, lo, 0)` followed by
  `silu_k`, and is written in place.
- **Reason:** The GR down projection produces `T × 320` values that need the
  documented quarter-scale and SiLU before the next stage. A scalar element
  count makes the final partially populated threadgroup safe for every
  supported T without coupling the launch to a fixed batch width.
- **Validation:** `metal_gr_test` covers T=1..8 on random, edge and fixture
  inputs, including the partial-threadgroup tails, guarded in-place writes,
  and an independent scalar oracle at U(16). Edge inputs explicitly check
  that every negative zero scaled by FMA produces positive zero; per T this
  covers 27..214 such values. The fixture run passes with Metal API and shader
  validation enabled. Targeted CTest passes. The synced full build and
  host/Metal CTest pass 47 of 48 tests; `metal_moe_real_inputs_unavailable`
  is the expected skip because model inputs are unavailable. `test_map.py`
  and `git diff --check` pass.
- **Review required:** Host dispatch must validate `T`, `n_values`, the
  corresponding buffer length and the in-place span; these checks are
  implemented under D-IMPL-129. GR production dispatch and Window/Prefill
  integration remain open under REV-029.

### D-IMPL-114: Add the head mixer's unfused GR pre-gated kernel

- **Decision:** Implement `strata_gr_pre_gated` with a fixed `(10, T)` grid
  and 256 threads per group. Each output column reads four `xn`/`gate`
  stream values, multiplies `xn` by `sigmoid_k(gate)`, adds products in
  stream order, and writes `fma(0.25, sum, 0)` to `mixed`.
- **Reason:** The head mixer has no inject, so it uses upstream's
  `pre_gated<false>` form. The fixed ten groups cover 2,560 output columns;
  this standalone stage completes the final operation after the head mixer's
  down-SiLU and up projection. Upstream's write-back of each product into
  `gate` is omitted because this pipeline does not consume those values.
- **Validation:** `metal_gr_test` covers T=1..8 on random, edge and fixture
  inputs. Its independent scalar oracle checks U(16), ordered accumulation,
  the exact negative-zero sum/FMA result, input immutability and output guards.
  Edge negative controls demonstrate that the exact checks reject a fused
  product/add and a reordered stream reduction. The fused control derives its
  exact separate-product result from a runtime Metal sigmoid probe, so it
  does not assume the GPU and high-accuracy CPU sigmoid round identically.
  The targeted CTest, edge run, and Metal API/Shader Validation fixture pass.
  The full build and host/Metal CTest pass: 47 tests passed, zero failed, and
  `metal_moe_real_inputs_unavailable` was the expected skip because model
  inputs are unavailable. `test_map.py` and `git diff --check` pass.
- **Review required:** Host validation of T, the `(10, T)` grid and the `xn`,
  `gate` and `mixed` views is implemented under D-IMPL-130. Production
  head-mixer, Window and Prefill integration remains open under REV-029.

### D-IMPL-115: Add the PLE value broadcast kernel

- **Decision:** Implement `strata_ple_broadcast` with the documented
  `(40, T)` grid and 256 threads per group. Each thread multiplies the value
  at `index % N_EMBD` by the gate for `index / N_EMBD` and writes its unique
  `gated[T][HC_DIM]` output.
- **Reason:** This is the standalone expansion between the PLE gate and
  convolution stages, requires no model weights, and is bitwise testable
  independently for every supported batch size. The arithmetic is a single
  F32 multiply; no reduction or runtime parameters are needed.
- **Validation:** `metal_ple_test` checks T=1..8 against the scalar bitwise
  oracle on random, edge and fixture inputs, verifies signed-zero products,
  read-only inputs and output guards, and compares single-row output with a
  replicated eight-row batch. Targeted CTest, direct edge and Metal API/Shader
  Validation fixture runs pass. The full build and host/Metal CTest pass:
  48 passed, zero failed, and `metal_moe_real_inputs_unavailable` is the
  expected skip because model inputs are unavailable. `test_map.py` and
  `git diff --check` pass.
- **Review required:** Host dispatch and buffer validation are implemented
  under D-IMPL-132. PLE production integration remains open under REV-029.

### D-IMPL-116: Add batch PLE history advancement

- **Decision:** Implement `strata_ple_history` with 256 threads in each of
  40 threadgroups. Each thread stages all nine next-history values in private
  storage, then writes its channel's `hist` row after all old-history reads
  are complete. `n` is a `setBytes` constant; nonnegative source rows come
  from `ple_norm`, and the remaining prefix comes from the old history.
- **Reason:** This reproduces `n` sequential history advances bitwise while
  writing the live history in place. Staging nine values prevents an early
  write from overwriting an old row that a later output still needs. It also
  handles n=0 as a no-op and n≥9 by taking only the final nine norm rows.
- **Validation:** `metal_ple_test` compares with a sequential-update oracle
  for n=0/1/2/8/9/10/17 on random, edge and fixture inputs, checks all
  92,160 output bits, verifies `ple_norm` immutability and history guards.
  Targeted CTest, direct edge and Metal API/Shader Validation fixture runs
  pass. The full build and host/Metal CTest pass: 48 passed, zero failed,
  and `metal_moe_real_inputs_unavailable` is the expected skip because model
  inputs are unavailable. `test_map.py` and `git diff --check` pass.
- **Review required:** Host validation and the fixed history grid are
  implemented under D-IMPL-133. Production PLE commit, write-through and
  Prefill integration remain open under REV-029.

### D-IMPL-117: Implement the PLE gate with the documented two-stage butterfly

- **Decision:** Implement `strata_ple_gate` with 512 threads per group and one
  group per row/stream. Each thread forms eight ordered products, using zeros
  beyond the 2,560 columns, then performs the 32-lane XOR butterfly. The 16
  simdgroup partials are stored in threadgroup memory; simdgroup zero reduces
  them with a second butterfly. Apply the host-provided scale with `fma`, then
  `sqrt(fmax(fabs(sum), 1e-6))`, the sign and `sigmoid_k`.
- **Reason:** This preserves section 5.3's reduction and native arithmetic
  order while computing four gate values for each token row. The scalar
  oracle mirrors the two butterflies but is implemented independently of
  Metal SIMD operations. U(64) bounds the ordered floating-point reduction,
  scale, square root and sigmoid path; crafted controls ensure both reduction
  stages reject serial accumulation.
- **Validation:** `metal_ple_test` covers T=1..8 on random, edge and fixture
  inputs; verifies U(64), read-only key/query inputs, output guards and
  first-row batch independence. The two negative controls place
  `2^24, 1, -2^24` at columns 0/1/16 and 0/32/256, respectively, and reject
  linear thread-lane and serial simdgroup-partial reductions outside U(64).
  The full build and host/Metal CTest pass 48 tests, with zero failures and
  `metal_moe_real_inputs_unavailable` as the expected skip. After adding the
  second-stage negative control, the rebuilt targeted CTest and the fixture
  with Metal API/GPU Validation pass; the seed-77 edge run also passes.
  `test_map.py` and `git diff --check` pass.
- **Review required:** Host validation of T, the `(4T) × 512` grid, derived
  scale binding and key/query/gate buffers is implemented under D-IMPL-134.
  PLE conv-residual and production Window/Prefill integration remain open
  under REV-029.

### D-IMPL-118: Add batched PLE convolution and residual

- **Decision:** Implement `strata_ple_conv_residual` with one channel per
  thread in the documented `(40, T) × 256` grid. For tap k, read
  `ple_norm[t - 9 + 3k]` when that row is nonnegative, otherwise read history
  row `9 + (t - 9 + 3k)`. Convert the stored half weight to float, round each
  product separately, add the four taps in order, apply `silu_k`, then store
  `R + (gated + activation)` in place.
- **Reason:** The batched source-row formula must reproduce the values seen
  when upstream advances its nine-row live history after each token. The
  sequential test oracle models those advances directly, so it exercises the
  equivalence without reusing the kernel's batch-index formula. Exact
  arithmetic probes pin product/add separation, tap order and residual
  parenthesization; general finite inputs use U(16) for the four-term sum,
  SiLU and residual path.
- **Validation:** `metal_ple_test` covers T=1..8 on random, edge and fixture
  inputs, including F16 weights, read-only inputs, in-place residual output,
  guards and first-row batch independence. Exact probes verify a +0 golden
  that changes under fused accumulation, reordered taps, or reassociation of
  `R + (gated + act)`. The full host/Metal CTest passes 48 tests with zero
  failures and `metal_moe_real_inputs_unavailable` as the expected skip.
  After adding the residual-order probe, the rebuilt targeted CTest and the
  fixture run with Metal API/GPU Validation pass; the seed-77 edge run also
  passes. `test_map.py` and `git diff --check` pass.
- **Review required:** Host dispatch shape, grid, allocation, alias and
  ownership validation is implemented under D-IMPL-135. Production
  Window/Prefill integration remains open under REV-029.

### D-IMPL-119: Copy flagged image embeddings as raw words

- **Decision:** Implement `strata_embed_image` in the documented 10-by-T grid
  with 256 threads per threadgroup. Bind `tokens`, `img_in` and `emb` at
  buffers 0, 1 and 2. Only a token with bit 31 set writes its row, using a
  32-bit integer load/store to preserve each F32 payload bit exactly.
- **Reason:** Image embeddings replace the skipped token-table gather only on
  flagged prefill rows. A raw word copy preserves signed zero, infinities and
  NaN payloads without introducing arithmetic or conversion. Unflagged rows
  must retain the preceding text gather.
- **Validation:** `metal_embed_test` checks T=1..8 with row-zero and
  after-text image-flag patterns on random, edge and fixture inputs, including
  T=1 with and without an image. It compares all output bits, checks that
  unflagged rows are unchanged, verifies token and image inputs remain
  read-only, and checks buffer guards. The focused CTest passes all three
  sets; fixture and edge runs with Metal API and GPU Validation enabled also
  pass. The full build and host/Metal CTest pass 48 tests with zero failures
  and `metal_moe_real_inputs_unavailable` as the expected skip. `test_map.py`
  and `git diff --check` pass after hostile review.
- **Review required:** Encoder-side shape, grid, allocation, alias and
  ownership validation is implemented under D-IMPL-136. The token vocabulary
  precondition and production Window/Prefill integration remain open under
  REV-029.

### D-IMPL-120: Add the fused GR normalization read

- **Decision:** Implement `strata_gr_norm` with one 256-thread group per
  `(row, stream)` in a `(T, 4)` grid (`group.x` is row, `group.y` is stream).
  Follow the documented ascending
  float4-per-thread order, fused inject, nested square FMA, SIMD butterfly,
  serial eight-partial reduction, `rsqrt`, and `(r * g) * rs` scaling. Bind
  `R`, `bo`, `g`, `inj_prev`, `xn`, and `rs_out` at buffers 0–5; pass `apply`
  as a `uint32_t` at buffer 6. When `apply` is zero, do not read `bo` or
  `inj_prev`.
- **Reason:** The attention/FFN read folds an optional pending write into the
  residual before normalization. Preserving upstream's per-thread sequence and
  reduction order keeps the result compatible with the documented fused
  flavour. A single explicit `apply` value avoids separate shader
  specializations while keeping the disabled path free of those reads.
- **Validation:** `metal_gr_test` checks T=1..8 with apply on and off over
  random, edge and fixture inputs. No-apply output is bit-exact against an
  ordered scalar twin; apply output and per-stream reciprocal scales use
  U(64). Inputs, all output values, read-only buffers and guards are checked.
  The focused CTest and an edge run with Metal API and GPU Validation pass.
- **Review required:** The production host must validate T, the `(T, 4)` grid,
  the six buffer lengths and the `uint32_t` apply binding, and keep `R`, `bo`,
  `g`, `inj_prev`, `xn` and `rs_out` disjoint where their read/write roles
  require it. Production Window/MTP dispatch remains open under REV-029.

### D-IMPL-121: Add the tiled GR down and inject projection

- **Decision:** Implement `strata_gr_down` with 512 F32 input values per tile,
  one 256-thread group per block in a 41-block grid, and dynamic threadgroup
  storage of `T × 512` floats. Bind `xn`, BF16 `hc_down`, BF16 `hc_inject`,
  `lo`, and `inj_out` at buffers 0–4; pass T as a `uint32_t` at buffer 5 and
  bind the tile at threadgroup-memory index 0. Preserve the documented
  low-half/high-half BF16 dot8 FMA order, lane accumulation, butterfly, and
  down projection's quarter-scale SiLU.
- **Reason:** The first 40 blocks produce the 320 low-rank values; block 40
  produces four injection values. A 512-value tile uses at most 16 KiB at
  T=8 and lets inactive injection simdgroups continue staging and reach every
  barrier. This follows the documented L-2 baseline; later tile-size tuning
  remains a measurement task.
- **Validation:** `metal_gr_test` checks all outputs for T=1..8 on random,
  edge and fixture inputs. The low-rank outputs use U(16), injection outputs
  are bit-exact, and inputs, output guards, BF16 packing and the dynamic
  threadgroup-memory binding are checked. The focused CTest and edge run with
  Metal API and GPU Validation pass.
- **Review required:** `encode_gr_down` validates T, the 41-block grid,
  buffer sizes/ranges/overlap, BF16 alignment, the `uint32_t` T binding, and
  `T × 512 × sizeof(float)` bytes of threadgroup memory under D-IMPL-126.
  Production Window/MTP wiring remains open under REV-029.

### D-IMPL-122: Add the fused GR up projection and mixer

- **Decision:** Implement `strata_gr_up` with the specified 160 groups of 256
  threads, each covering 16 columns across all four streams. Stage the complete
  `T × 320` low-rank input in threadgroup memory, then preserve the packed
  BF16 `lane` / `32 + lane` dot8 FMA order, butterfly reduction, pending-write
  fold, `(R * g) * rs` scaling, sigmoid gate and ascending four-stream mixer.
  Bind `R`, `bo`, `inj_prev`, `g`, `rs`, `lo`, BF16 `hc_up` and `mixed` at
  buffers 0–7. Buffer 8 carries a two-word `GRUpParams` (`token_count`,
  `apply`) so both runtime values are set in one encoder binding.
- **Reason:** A static `lo_s[8][320]` and `g_s[8][4][16]` consume 12 KiB and
  let each group retain the documented per-token order without exceeding the
  Metal threadgroup-memory limit. A paired parameter block makes the fixed
  T/apply ABI explicit and fits the existing single-`setBytes` dispatch helper.
  The no-apply path must preserve R exactly; the apply path folds the pending
  write into R before rebuilding the normalized value.
- **Validation:** `metal_gr_test` covers T=1..8 with apply on and off over
  random, edge and fixture inputs. It checks U(64) mixed outputs, U(16)
  residual writes, exact no-apply residuals, read-only inputs and guards.
  Fixture-only bit checks require the documented `(R * g) * rs` order and
  ascending stream sum; negative controls prove that reversed norm
  multiplication and reordered stream sums produce different bits. The
  focused CTest passes with Metal API and GPU Validation enabled. The full
  build and host/Metal suite pass 49/49, with 48 passes and one expected
  model-input skip; `test_map.py` and `git diff --check` pass.
- **Review required:** `encode_gr_up` validates T, apply, the 160-group grid,
  buffer sizes/alignment/ranges/overlap, static threadgroup memory and Context
  ownership under D-IMPL-127. Production Window/MTP wiring remains open under
  REV-029.

### D-IMPL-123: Add the O-class control-vector kernel

- **Decision:** Implement `strata_gr_cvec` on the documented `(4, T) × 256`
  grid with stream on x and token on y. Bind `R`, direction, per-layer scale,
  device `on`, `bo` and `inj` at buffers 0–5. Buffer 6 carries four
  `uint32_t` values: `layer`, `token_count`, `mode` (0 project, 1 add), and
  `write`. Fuse the optional `fma(bo, 2*sigmoid(inj/4), R)` before projection
  or addition. Project mode accumulates ten per-thread values, applies the
  documented two-stage butterfly and layer scale, then uses `fma(-dot, v, x)`.
  Add mode adds the already-scaled direction. Skip the pass when steering and
  writing are both disabled.
- **Reason:** 2,560 columns divide exactly into ten values per lane for a
  256-thread group. The specified eight-simdgroup reduction fits in eight
  threadgroup partials and preserves the upstream projection order. Fusing the
  pending GR write avoids a separate residual pass while preserving stock
  behavior when steering is disabled.
- **Validation:** `metal_gr_test` exercises T=1..8 in project and add modes
  with write enabled and disabled, including active project/add cases at layer
  3 with a nonzero direction and a project scale distinct from layer 7.
  Disabled add and uncovered add use nonzero directions. A fixture-only probe
  pins the fused write with a bit-sensitive FMA input and separate
  multiply/add negative control. Random, edge and fixture inputs also cover
  disabled-vector, zero-scale and uncovered-layer paths. The focused CTest
  passes. The full test executable passes with Metal API Validation and GPU
  Validation enabled. `test_map.py` and `git diff --check` pass. The full
  build and host/Metal CTest pass 49/49, with 48 passes and one expected
  model-input skip.
- **Review required:** Production dispatch must validate T=1..8, layer < 48,
  mode ∈ {0,1}, write ∈ {0,1}, the `(4, T)` grid, six buffer lengths plus the
  parameter block, and alias/overlap rules for in-place `R`. Production
  Window/Prefill integration remains open under REV-029.

### D-IMPL-124: Add a validated GR control-vector encoder

- **Decision:** Add `encode_gr_cvec` to the Metal engine. It accepts T, layer,
  project/add mode, write flag and the six buffer views; validates T=1..8,
  layer < 48, mode, the manifest's 256-thread group, every required buffer
  length/alignment and pairwise overlap; then binds the 16-byte parameter block
  and dispatches `(4, T) × 256` into a caller-owned command buffer. Reject
  command buffers that have already been submitted.
- **Reason:** A typed encoder keeps the four-word shader ABI and its grid
  contract in one production entry point. It catches malformed shapes, views
  and aliases before encoding, while leaving command-buffer composition to the
  future Window and Prefill schedulers.
- **Validation:** `metal_gr_test` now exercises cvec through `encode_gr_cvec`
  for T=1..8 and the project/add, write, off, zero-scale and uncovered cases.
  Encoder rejection tests cover T, layer, mode, all six short buffers,
  misalignment, same-handle and distinct-handle no-copy overlap, and both
  enqueued and committed command buffers. The full build passes, focused CTest
  and the fixture run with Metal API and GPU Validation pass, and the complete
  host/Metal suite passes 49/49 (one expected model-input skip). The regular
  `test_map.py` check and `git diff --check` pass. Its `--final` gate remains
  open because the broader project still contains planned tests and
  requirements without checks.
- **Review required:** The Window and Prefill paths must call this API only
  after their layer buffers and command-buffer schedule are ready. Context
  must outlive the encoded work. The encoder now verifies buffer-pool and
  command-queue ownership. GR up, post and other kernels still need equivalent
  host validation before production dispatch.
  The repository-wide final acceptance gate also remains open.

### D-IMPL-125: Add a validated GR norm encoder

- **Decision:** Add `encode_gr_norm` for `strata_gr_norm`. It accepts T and
  `apply`, validates T=1..8 and the manifest's 256-thread group, checks the
  required sizes, four-byte offsets, Metal allocation ranges and pairwise
  disjointness of all six buffers, then binds the one-word parameter and
  dispatches `(T, 4) × 256` into a caller-owned command buffer. Reject
  enqueued or committed command buffers.
- **Reason:** The normalization kernel's grid and six-buffer ABI are fixed,
  but malformed rows and views could previously reach raw manifest
  dispatch. A typed encoder makes the documented host preconditions
  executable and exercises the same production-facing entry point in the
  parity test.
- **Validation:** The focused `metal_gr_test` passes, including T=1..8,
  `apply` on/off, invalid T, all six undersized buffers, misalignment,
  allocation-range overflow, overlap and enqueued/committed command-buffer
  checks. Its fixture run passes with Metal API and GPU Validation enabled.
  The full build passes; all 49 host/Metal CTest tests pass, with one expected
  model-input skip. `test_map.py` and `git diff --check` pass.
- **Review required:** Residual and output views must remain disjoint.
  `block_out` and `inject_prev` are required views even when `apply` is false,
  although the shader does not read them in that mode. The encoder verifies
  buffer-pool and command-queue ownership; the Context must outlive encoded
  work. Window and Prefill integration remains open under REV-029.

### D-IMPL-126: Add a validated GR down encoder

- **Decision:** Add `encode_gr_down` for `strata_gr_down`. It accepts T,
  validates T=1..8 and the manifest's 256-thread group, checks the five
  required buffer sizes, Metal allocation ranges, pairwise disjointness,
  four-byte float offsets and 16-byte BF16 weight offsets, and ensures
  `T × 512 × sizeof(float)` fits the device's threadgroup-memory limit. It
  verifies the buffers belong to the Context's pool and the command buffer to
  its queue, then binds T, sets threadgroup memory at index 0 and dispatches
  41 groups of 256 threads. Reject enqueued or committed command buffers.
- **Reason:** `strata_gr_down` has a fixed 41-group grid but a T-sized dynamic
  tile and packed `uint4` reads from BF16 weights. A typed encoder checks
  those host preconditions before encoding and lets the existing independent
  scalar twin exercise the production-facing ABI.
- **Validation:** The focused `metal_gr_test` passes for T=1..8, including
  invalid T, short/misaligned/out-of-allocation views, overlap,
  enqueued/committed buffers and foreign-context buffers and command buffers.
  Its fixture run with Metal API and GPU Validation passes. The full build,
  all 49 host/Metal CTest tests, `test_map.py` and `git diff --check` pass;
  one model-input test is the expected skip.
- **Review required:** All five views must remain disjoint and belong to the
  same Context as the command buffer; the encoder enforces this ownership.
  Keep the Context alive through submission. Window and Prefill scheduling
  and production use remain open under REV-029.

### D-IMPL-127: Add a validated GR up encoder

- **Decision:** Add `encode_gr_up` for `strata_gr_up`. It accepts T and
  `apply`, validates T=1..8 and the manifest's 256-thread group, checks the
  two-word parameter block, all eight buffer sizes/ranges and pairwise
  disjointness, four-byte float offsets, 16-byte BF16 weight alignment and
  the pipeline's static threadgroup-memory size against the device limit.
  Verify pool and queue ownership, then dispatch the fixed 160 groups of 256
  threads into a caller-owned command buffer.
- **Reason:** The up kernel folds a pending write into R in place and also
  reads seven other views while producing a separate mixed output. Its packed
  BF16 `uint4` reads, fixed threadgroup state and T/apply constants need a
  host entry point that validates the full binding contract before encoding.
- **Validation:** The focused `metal_gr_test` passes for T=1..8 with apply
  on/off and checks invalid T, all eight short buffers, BF16 alignment,
  allocation range, overlap and command-buffer state. The fixture run with
  Metal API and GPU Validation passes. The full build and all 49 host/Metal
  CTest tests pass, with one expected model-input skip. `test_map.py` and
  `git diff --check` pass.
- **Review required:** All eight views must be disjoint and owned by the same
  Context as the command buffer; the encoder checks both conditions. Context
  lifetime must include command submission. Window and MTP integration
  remains open under REV-029.

### D-IMPL-128: Add a validated GR post encoder

- **Decision:** Add `encode_gr_post` for `strata_gr_post`. It validates
  T=1..8, the manifest's 256-thread geometry, all four buffer lengths,
  alignment and allocation ranges, and buffer-pool and command-queue
  ownership. The four views must be disjoint except that residual and output
  may be the exact same handle, offset and length. Reject all other overlaps
  and command buffers that have already been enqueued or submitted; dispatch
  the fixed 40×T groups into the caller-owned command buffer.
- **Reason:** Native post updates R and must support the documented in-place
  form while detecting unsafe cross-role and partial aliases before GPU work
  is encoded. The arithmetic and launch remain the native Metal contract.
- **Validation:** The focused `metal_gr_test` passes, including invalid T,
  each undersized view, misalignment, cross-role overlap, partial
  residual/output overlap, out-of-allocation range, foreign-context resources,
  enqueued/committed command buffers, and separate and exact in-place oracle
  cases. The fixture run with Metal API and GPU Validation passes. The full
  build and all 49 host/Metal CTest cases pass, with the expected
  `metal_moe_real_inputs_unavailable` skip. `test_map.py` and
  `git diff --check` pass.
- **Review required:** The exact-view alias rule and fixed shape are host
  preconditions for this kernel; production Window/Prefill integration
  remains open under REV-029.

### D-IMPL-129: Add a validated GR down-SiLU encoder

- **Decision:** Add `encode_gr_down_silu` for `strata_gr_down_silu`. It accepts
  T=1..8, derives `n_values = T * HC_LOW_RANK`, validates the 256-thread
  manifest geometry and the in-place float view's size, alignment, allocation
  range and Context ownership, then binds the derived element count and
  dispatches ⌈n_values/256⌉ groups into a same-Context command buffer that has
  not been enqueued or submitted.
- **Reason:** Deriving the scalar bound from T prevents callers from supplying
  a count inconsistent with the `T × 320` low-rank allocation. The kernel's
  final partial group needs the exact count for its tail guard.
- **Validation:** The focused `metal_gr_test` passes T=1..8 through
  `encode_gr_down_silu` on random, edge and fixture inputs. Validation rejects
  invalid T, foreign resources, short storage (including one-row storage for
  T=2), a misaligned view, an out-of-allocation view, and enqueued/submitted
  command buffers. The fixture run with Metal API and GPU Validation passes;
  the full build and all 49 host/Metal CTest cases pass, with one expected
  model-input skip. `test_map.py` and `git diff --check` pass. The rejection
  cases reuse one unsubmitted command buffer to stay below Metal's
  pending-buffer limit.
- **Review required:** The encoder enforces the fixed `T × 320` in-place
  contract. Production GR and head-mixer wiring remains open under REV-029.

### D-IMPL-130: Add a validated GR pre-gated encoder

- **Decision:** Add `encode_gr_pre_gated` for `strata_gr_pre_gated`. Validate
  T=1..8, the manifest's 256-thread geometry and fixed `(10, T)` grid, and
  the sizes, four-byte alignment, allocation ranges and Context ownership of
  all three views. The read-only `xn` and `gate` views may overlap, but the
  `mixed` output must be disjoint from each. Reject foreign or already
  enqueued/submitted command buffers; encode into a caller-owned buffer.
- **Reason:** The fixed grid writes all 2,560 output columns per row. Output
  aliasing an input can race with reads by another threadgroup, while overlap
  between the two read-only inputs is safe and need not be rejected.
- **Validation:** The focused `metal_gr_test` exercises T=1..8 on random,
  edge and fixture data through the production encoder. It rejects invalid
  T, each short view, one-row storage for each individual view at T=2,
  misalignment, output overlap with either input, out-of-allocation views,
  foreign-context resources and command-buffer lifecycle violations; it
  accepts aliased read-only inputs. The fixture run with Metal API and GPU
  Validation passes. The full build and all 49 host/Metal CTest cases pass,
  with one expected model-input skip. `test_map.py` and `git diff --check`
  pass.
- **Review required:** Input/output bounds and the fixed grid are enforced by
  the encoder. Production head-mixer, Window and Prefill integration remains
  open under REV-029.

### D-IMPL-131: Add a validated weighted RMS norm encoder

- **Decision:** Add `encode_weighted_rms_norm` for `strata_rms_norm_w`.
  Require positive rows, one of the documented `(n_cols, gamma_period)`
  pairs `(2560, 1)`, `(2560, 4)` or `(10240, 1)`, and finite positive
  epsilon. Check byte-size arithmetic, the 1024-thread manifest geometry,
  static threadgroup memory, buffer bounds/alignment and Context ownership.
  Support exact input/output view alias for in-place normalization; reject
  partial input/output overlap and every output/gamma overlap. Permit
  read-only input/gamma overlap. Validate command-buffer ownership and
  require it to remain unsubmitted, then bind the 12-byte parameter block and
  dispatch one threadgroup per row.
- **Reason:** PLE and GR/head-mixer/MTP paths use several widths and gamma
  periods, and the kernel also supports the documented in-place normalization
  path. Deriving bounds from the validated shape and limiting writes to the
  exact input alias prevent unsafe cross-row and gamma races.
- **Validation:** `metal_gr_test` now runs every weighted RMS norm oracle case
  through the encoder, including separate and exact in-place output. Encoder
  tests reject zero rows/columns/gamma period, unsupported width/gamma pairs,
  invalid epsilon, undersized and misaligned buffers, partial input/output
  overlap, output/gamma overlap, out-of-allocation views, foreign resources
  and submitted/enqueued command buffers; read-only input/gamma overlap is
  accepted. The focused test and Metal API/GPU Validation fixture pass.
  The full build and all 49 host/Metal CTest cases pass, with one expected
  model-input skip. `test_map.py` and `git diff --check` pass.
- **Review required:** The encoder implements the documented weighted RMS
  norm host bounds. Window, Prefill and MTP production dispatch remain open
  under REV-029; U(64) does not cover extreme or subnormal inputs.

### D-IMPL-132: Add a validated PLE value-broadcast encoder

- **Decision:** Add `encode_ple_broadcast` for `strata_ple_broadcast`.
  Validate T=1..8, the 256-thread manifest geometry and fixed `(40, T)`
  grid, the value/gate/output byte ranges, float alignment, Metal allocation
  bounds, Context pool ownership and command-queue ownership. Permit overlap
  between the read-only value and gate inputs, reject any output/input
  overlap, and encode into a caller-owned command buffer before it is
  enqueued or committed.
- **Reason:** The broadcast has no runtime parameters and each output has one
  fixed F32 multiply. A typed host entry point makes the batch shape, row
  bounds and no-output-alias contract executable while retaining safe
  read-only input sharing.
- **Validation:** `metal_ple_test` runs the random, edge and fixture broadcast
  cases for T=1..8 through the production encoder and checks bit-exact scalar
  results, signed zeros, row independence and guards. Encoder checks reject
  invalid T, each individual one-row view at T=2, misalignment, allocation
  overflow, exact and nonzero-offset partial output/value and output/gate
  overlap, foreign buffers and command buffers, and enqueued/committed
  command buffers; overlapping read-only inputs are accepted. Same-handle
  overlap is also rejected when CPU/GPU address metadata is absent. The
  focused test passes, as does the fixture with Metal API and GPU Validation
  enabled. The full build and all 49 host/Metal CTest cases pass, with one
  expected model-input skip. `test_map.py` and `git diff --check` pass.
- **Review required:** The encoder enforces buffer and queue ownership.
  Production PLE Window/Prefill wiring and commit/write-through use remain
  open under REV-029.

### D-IMPL-133: Add a validated PLE history encoder

- **Decision:** Add `encode_ple_history` for `strata_ple_history`. Reject `n`
  above `INT32_MAX`, because the shader uses signed row indices, and check
  the byte count for `n` rows of `ple_norm`. Require a fixed
  `HC_DIM × PLE_HISTORY_ROWS` history view, float alignment, allocation
  bounds, Context ownership, disjoint history/norm views and a same-Context
  unsubmitted command buffer. Bind `n` at buffer 2 and dispatch the fixed
  40×256 history grid. A zero-length norm view is valid for n=0.
- **Reason:** For n>0 the shader's largest norm index is n−1, even though it
  reads only the final nine rows when n≥9. Requiring all n rows prevents
  out-of-bounds indexing. n=0 reads no norm values and leaves history
  unchanged; an empty norm view accurately represents that no-read case.
- **Validation:** `metal_ple_test` routes the sequential oracle cases
  n=0/1/2/8/9/10/17 through the encoder, bitwise checks all 92,160 history
  values, and checks input immutability and guards. Encoder tests reject
  signed-index overflow, short history, one norm row for n=2 and 16 rows for
  n=17, misalignment, allocation overflow, exact and partial history/norm
  overlap, foreign resources and enqueued/committed command buffers. It also
  rejects same-handle overlap when CPU/GPU address metadata is absent. A
  zero-length norm subview inside the history allocation is accepted for
  n=0. Focused CTest, the fixture with Metal API/GPU Validation, the full
  build and all 49 host/Metal CTest cases pass; one model-input skip is
  expected. `test_map.py` and `git diff --check` pass.
- **Review required:** The encoder bounds the signed index and every buffer
  view. Production history advancement at PLE commit/write-through and
  Prefill remains open under REV-029.

### D-IMPL-134: Add a validated PLE gate encoder

- **Decision:** Add `encode_ple_gate` for `strata_ple_gate`. Validate T=1..8,
  the manifest's 512-thread geometry and `(T × HC, 1, 1)` grid, sizes and
  four-byte alignment for the `T × HC_DIM` key/query and `T × HC` gate
  views, Metal allocation bounds, Context ownership and command-buffer
  ownership/state. Permit key/query input overlap, reject output overlap with
  either input, derive `1 / sqrt(N_EMBD)` on the host, bind it at buffer 3,
  and encode into a caller-owned command buffer.
- **Reason:** The gate scale is a fixed model constant, not a model weight or
  caller option. Deriving it prevents caller drift while retaining the
  documented single-precision value and two-stage butterfly kernel. The
  output is written by row/stream groups and cannot alias either input;
  shared read-only inputs are safe.
- **Validation:** `metal_ple_test` runs T=1..8 and both reduction-order
  negative controls through the encoder, checking U(64), row independence,
  input immutability and guards. Encoder tests reject invalid T, each
  one-row key/query/output view at T=2, misalignment, allocation overflow,
  exact and partial output/key and output/query overlap, foreign resources,
  and enqueued/committed command buffers; key/query read-only overlap is
  accepted. Same-handle alias rejection also passes when CPU/GPU address
  metadata is absent. The focused test and fixture with Metal API/GPU
  Validation pass. The full build and all 49 host/Metal CTest cases pass,
  with one expected model-input skip. `test_map.py` and `git diff --check`
  pass.
- **Review required:** The gate's host shape, scale and buffer contracts are
  validated. PLE conv-residual and production Window/Prefill dispatch remain
  open under REV-029.

### D-IMPL-135: Add a validated PLE conv-residual encoder

- **Decision:** Add `encode_ple_conv_residual` for
  `strata_ple_conv_residual`. Validate T=1..8 and the manifest's 256-thread
  `(40, T)` grid. Check the F32 `ple_norm`, `gated` and in-place `residual`
  views for `T × HC_DIM` values, the nine-row F32 history, and the
  `HC_DIM × 4` F16 convolution weights. Validate each view's element
  alignment, allocation bounds, Context pool ownership, command queue and
  unsubmitted command-buffer state. Allow overlap among read-only inputs;
  reject every overlap between the residual output and any input.
- **Reason:** The kernel indexes token norms, rolling history, half-precision
  taps, gates and residuals with different shapes. Validate each role against
  its actual shader access pattern and keep output disjoint from all inputs,
  while allowing safe sharing among read-only views.
- **Validation:** `metal_ple_test` routes T=1..8 random, edge and fixture
  cases through the encoder and checks U(16) against sequential history
  advancement with F16 weights. It rejects invalid T, one-row T=2
  `ple_norm`/`gated`/`residual` views, short history and weight views,
  misalignment, out-of-allocation ranges, exact and partial residual overlap
  with each of the four inputs, foreign-context buffers/commands and
  enqueued/committed commands. Same-handle overlap is checked without CPU/GPU
  address metadata; overlapping read-only inputs are accepted. The focused
  test, Metal API/GPU Validation fixture, full build and all 49 host/Metal
  CTest cases pass, with one expected model-input skip. `test_map.py` and
  `git diff --check` pass.
- **Review required:** Encoder contracts are validated. Production PLE
  Window/Prefill dispatch and history write-through remain open under
  REV-029.

### D-IMPL-136: Add a validated image embedding encoder

- **Decision:** Add `encode_embed_image` for `strata_embed_image`. Accept
  positive prefill row counts beyond the decode-only eight-row limit, up to
  `floor(2^32 / N_EMBD)` rows so the shader's 32-bit
  `N_EMBD * row + value` index cannot wrap. Check overflow-safe byte sizes for
  `T` tokens and `T × N_EMBD` F32 image/output rows, the fixed
  `(10, T) × 256` manifest geometry, token/F32 alignment, Metal allocation
  bounds, Context pool ownership and consistency between the CPU-visible
  `BufferRef` pointer and its actual `MTLBuffer` plus offset. Permit overlap
  between read-only tokens and image input; reject output overlap with either
  input, including distinct no-copy handles over one backing range. Require a
  same-Context, unsubmitted command buffer and bind tokens/image/output at
  indices 0/1/2.
- **Reason:** Prefill chunk sizes are configuration- and memory-plan-driven
  and may exceed decode T=8. The image kernel's row guard handles text rows,
  so dispatch can cover the complete chunk while copying only flagged image
  rows. The uint-index bound prevents wrap in the shader's `2560 * row +
  value` calculation even if a caller provides unusually large buffers. Token
  vocabulary remains a separate host responsibility because the shader reads
  token flags but does not perform vocabulary lookup.
- **Validation:** `metal_embed_test` routes image copies for T=1..8 and both
  row-zero/after-text patterns through the encoder, preserving all payload
  bits and leaving unflagged rows unchanged. It rejects zero and shader-index
  overflowing row counts, each
  one-row input/output view at T=2, misaligned views, an out-of-allocation
  view, token/output and image/output exact/partial overlaps, same-handle
  overlap without CPU/GPU address metadata, exact overlap through distinct
  no-copy handles over the same allocation, missing or inconsistent CPU
  backing pointers, each foreign-context buffer, foreign command buffers and
  enqueued/committed commands. Overlapping read-only token/image views are
  accepted and submitted. Focused CTest, the fixture with Metal API/GPU
  Validation, the full build and all 49 host/Metal CTest cases pass, with one
  expected model-input skip. `test_map.py` and `git diff --check` pass.
- **Review required:** Image buffer geometry and command ownership are
  validated. Vocabulary bounds are enforced by `encode_embed_rows`
  (D-IMPL-138); production Prefill wiring remains open under REV-029.

### D-IMPL-137: Add a validated GR stream-broadcast encoder

- **Decision:** Add `encode_gr_broadcast` for `strata_gr_broadcast` with the
  documented `(40, T) × 256` grid and buffers 0/1 for the embedding and
  four-stream output. Accept positive row counts up to
  `floor(2^32 / HC_DIM) = 419,430`, keeping the shader's 32-bit
  `HC_DIM * row + value` index in range. Derive overflow-safe byte ranges for
  `T × N_EMBD` and `T × HC_DIM` 32-bit values, require 4-byte alignment,
  Context ownership, allocation bounds and CPU-visible pointers consistent
  with each `MTLBuffer` view, and reject every input/output overlap including
  distinct no-copy handles over shared backing. Require a same-Context,
  unsubmitted command buffer.
- **Reason:** The broadcast expands each 2,560-value embedding into four
  streams using raw integer copies, so the output must be disjoint from the
  source and preserve all floating-point payload bits. The shader computes
  its output index in 32-bit unsigned arithmetic; the derived row bound
  prevents wrap even for callers outside the decode-only T=1..8 range.
- **Validation:** `metal_embed_test` routes T=1/3/8 random, edge and fixture
  broadcasts through the encoder and compares every output bit. It rejects
  zero and overflowing row counts, short/misaligned/out-of-allocation views,
  contained and partial same-handle aliases, same-handle aliases without GPU
  addresses, cross-handle aliases over one mmap allocation, missing CPU
  backing metadata, foreign-context buffers and commands, and enqueued or
  committed commands. The focused test, fixture with Metal API/GPU
  Validation, full build and all 49 host/Metal CTest cases pass, with one
  expected model-input skip. `test_map.py` and `git diff --check` pass.
- **Review required:** Host geometry, alias and command validation are
  implemented. Production embedding/GR dispatch in Window and Prefill remains
  open under REV-029.

### D-IMPL-138: Add a validated token embedding-row encoder

- **Decision:** Add `encode_embed_rows` for Q3_K, IQ3_S, IQ4_XS and BF16 with
  the documented `(10, T) × 256` grid and table/tokens/output at buffers
  0/1/2. Use row strides 1,100, 1,100, 1,360 and 5,120 bytes. Accept
  `1..floor(2^32 / N_EMBD) = 1,677,721` rows and vocabulary sizes up to
  `2^31`, reserving token bit 31 for image rows. Derive overflow-safe buffer
  sizes from the selected format and vocabulary, validate allocation bounds,
  alignment, Context ownership and CPU-visible backing pointers, then scan
  token ids before encoding: image-marked rows are skipped, and every other
  id must be below `vocabulary_size`. Permit table/token read-only overlap,
  reject output overlap with either input, and require an unsubmitted
  same-Context command buffer. Callers keep input views alive and token words
  unchanged until GPU completion.
- **Reason:** The gather kernels index table rows from device token values and
  do not guard vocabulary bounds. Validating the CPU-visible token view before
  dispatch prevents an invalid id from reading past the quantized or BF16
  table. The image flag is outside the vocabulary namespace and must continue
  to bypass the table lookup. Requiring stable token contents through command
  completion preserves that check for the asynchronous GPU read.
- **Validation:** `metal_embed_test` routes T=1/7 gathers for all four
  formats through the encoder and compares with ggml CPU on random, edge and
  fixture inputs, including untouched image rows. It rejects zero and
  shader-index-overflowing rows, empty or flag-colliding vocabularies,
  unsupported formats, undersized/misaligned/out-of-allocation views,
  insufficient table rows, invalid non-image token ids, contained and
  partial output overlap with both inputs, missing metadata, same-handle
  aliases without GPU addresses, cross-handle shared backing, foreign
  Context resources and submitted command buffers. Read-only table/token
  overlap and high-bit image markers are accepted. The focused test, fixture
  with Metal API/GPU Validation, full build, `test_map.py` and `git diff
  --check` pass. All 49 host/Metal CTest cases pass with one expected
  model-input skip. The hostile review found no implementation defect and
  its P3 coverage finding is closed by a targeted partial token/output
  overlap case.
- **Review required:** Production Window and Prefill must route the token
  gather through this encoder with the model's `n_vocab`; production
  embedding/GR dispatch remains open under REV-029.

### D-IMPL-139: Build bounded Metal Prefill chunk metadata

- **Decision:** Add a device-free `build_prefill_chunk_metadata` helper that
  copies one non-empty token span and derives `PrefillParams`, the six query
  batch sizes, signed `StepRow`s and image-row presence. Cap rows at
  `floor(2^32 / HC_DIM) = 419,430`, the stricter flattened-index limit of the
  embedding-to-GR-broadcast stage. Require the exclusive position end to fit
  the signed `StepRow::n_kv` field. Build into a temporary value and replace
  the caller's output only after successful validation.
- **Reason:** The prompt specification gives the exact chunk fields and
  per-row equations, while the shared Metal `StepRow` stores signed 32-bit
  fields and the broadcast shader flattens `HC_DIM` values per row. Using the
  strictest first-stage shader limit prevents a host-valid chunk from
  overflowing before later prompt kernels run. The helper is kept device-free
  so the metadata contract can be tested without opening a Metal Context.
- **Validation:** `prefill_chunk_test` checks first-chunk and indexer boundary
  values, `width` clamping, full and remainder batch slots, copied image flags,
  the exact 419,430-row limit and its rejected successor, signed position
  bounds, and unchanged output after invalid requests. The planner accepts the
  exact row limit and rejects its successor; the embedding shaders derive
  their row strides from shared model-dimension constants, and compile-time
  assertions tie the planner and metadata-builder limit to `HC_DIM`. The
  full host/Metal rerun passed 49 of 50 tests, with the model-input-dependent
  `metal_moe_real_inputs_unavailable` test skipped as expected. The focused
  three tests, `test_map.py`, and `git diff --check` pass; `metal_embed_test`
  also passes with Metal shader validation and the debug layer enabled.
- **Review required:** Confirm the row and position bounds against all prompt
  kernels before `Prefill::run` accepts chunks or the planner raises its
  current chunk limits. Production prompt execution remains open under
  REV-030.

### D-IMPL-140: Encode the Prefill embedding stage as one ordered unit

- **Decision:** Add `encode_prefill_embedding` to verify the chunk metadata and
  CPU-visible token words, prepare all required stage bindings, then encode
  text gather, optional image copy and GR broadcast in one
  `MTLDispatchTypeSerial` compute encoder on the caller's fresh command buffer.
  The coordinator performs its validation and encoding through
  `CommandBuffer::encode_unsubmitted`, which serializes that operation against
  enqueue/commit and atomically discards on failure.
- **Reason:** Section 6.7 specifies this exact order and the image pass is
  conditional. All buffers use untracked hazard tracking. One serial encoder
  gives each dispatch visibility of earlier writes in the same segment; queue
  fences order any other encoder boundaries. The shared token view also
  determines the image-row choice on the host and in the shader; verifying
  identical CPU-visible words prevents those decisions from diverging.
- **Validation:** `metal_embed_test` exercises text-only and image chunks
  through the coordinator, compares gathered and copied embeddings plus all
  four GR streams bitwise, rejects GPU token words that differ from the
  metadata, and confirms that invalid image and broadcast stages discard the
  command buffer without changing outputs. It also covers an enqueued-buffer
  rejection and races a failed encoding callback against `enqueue` to verify
  that the wrapper discards before releasing its submission lock. The focused
  CTest and fixture run with Metal API/GPU Validation pass; the full host/Metal
  CTest passes 50/50 with one expected unavailable-model-input skip.
- **Review required:** Connect this coordinator from `Prefill::run`, compose
  subsequent Prefill kernels into the same segment, register every
  cross-command-buffer producer dependency, and preserve the fresh-buffer and
  stable-input requirements (REV-029, REV-031).

### D-IMPL-141: Keep the Prefill embedding dependency inside one Metal segment

- **Decision:** Split the embedding gather, image-copy and GR-broadcast
  encoders into validation/preparation and native-encoder dispatch helpers.
  Reuse those helpers in both their standalone test APIs and the Prefill
  coordinator; have the coordinator validate every required pass before
  opening its single serial compute encoder.
- **Reason:** An adversarial review found that the initial coordinator created
  three consecutive compute encoders while every buffer was untracked. That
  violated the section 8.1 ordering contract and could let GR broadcast read
  stale or incomplete embedding rows. Folding the passes into one serial
  encoder resolves this intra-segment dependency without adding a fence
  between compute encoders.
- **Validation:** The focused `metal_embed_test` CTest and an explicit fixture
  run with Metal API and GPU Validation pass. The test checks exact gathered
  embedding, image-row and four-stream output bits for text-only and image
  chunks. The full host/Metal CTest also passes 50/50 with one expected
  unavailable-model-input skip.
- **Review required:** Preserve the one-segment encoder rule in future
  production composition and register cross-command-buffer dependencies
  before submission (REV-031).

### D-IMPL-142: Fence untracked Metal encoders and require producer submission first

- **Decision:** Create and seed one `MTLFence` per Context queue before returning
  from `Context::open`; route every engine and twin-test compute/blit encoder
  through begin/end helpers that wait and update it. Provide
  `CommandBuffer::depends_on` to record cross-command-buffer dependencies and
  reject consumer enqueue/commit until its earlier-created producer has
  completed successfully. Keep `encode_sampler_sampled`'s split and merge
  passes in one serial encoder. The fence seed is the sole update-only encoder
  and completes before the Context is exposed.
- **Reason:** All Metal buffers use untracked hazard tracking, and the fence
  carries execution ordering only after command buffers enter the queue in
  dependency order; it cannot propagate a producer's GPU execution failure.
  Explicit dependency registration plus a successful completion check prevents
  a consumer from reading incomplete or failed producer output. A recursive
  per-command-buffer encoding lock protects each full engine operation from
  concurrent submission while the process-wide mutex remains available for
  short queue/event state changes.
- **Validation:** `metal_embed_test` covers the atomic encode/enqueue race,
  rejects consumer submission both before producer submission and before
  producer success, and checks image production followed by broadcast
  consumption across two command buffers. `metal_sampler_parity_test` verifies
  sampled parity after the two kernels share one encoder and rejects
  cross-Context command buffers and buffers. `metal_twin_selftest` uses the
  fence helpers. The full build completed; all 50 host/Metal CTest cases pass,
  with one expected model-input skip. `test_map.py` and `git diff --check`
  pass.
- **Review required:** Dependency registration is explicit and cannot infer
  buffer accesses. Each future production producer/consumer edge must call
  `depends_on`; audit those call sites and queue failure/recovery before
  production Prefill or Window submission (REV-031). Successful-completion
  gating is conservative and may add host waits; review its cost before
  enabling asynchronous production overlap.

### D-IMPL-143: Serialize encoders per command buffer and verify sampler views

- **Decision:** Hold a recursive encoding mutex on one `CommandBuffer` from
  engine validation through `endEncoding`, before any wrapper enqueue, commit
  or discard can proceed. Keep the process-wide submission mutex out of the
  encoder callback. Require sampler command buffers and every referenced
  buffer to belong to the supplied `Context`, and verify that each view's CPU
  pointer and GPU address match its native Metal handle and offset.
- **Reason:** A submission mutex around long prefill validation would block
  unrelated work on other command buffers and Contexts. Per-buffer locking
  closes the status-check/submit race without imposing that global stall.
  Sampler ownership and view-metadata checks prevent foreign resources or
  forged host/GPU addresses from bypassing validation or alias checks.
- **Validation:** `metal_embed_test` deterministically holds an encoding
  callback while a competing enqueue attempts submission. `metal_sampler_parity_test`
  covers foreign command buffers and buffers, plus forged CPU and GPU view
  metadata. The full build completed; 49 of 50 host/Metal CTest cases pass,
  with one expected model-input skip. `test_map.py` and `git diff --check`
  pass.
  The hostile follow-up found no remaining P1/P2/P3 issue.
- **Review required:** Keep every new encoder path under the per-buffer guard,
  and preserve the event lock order: the command-buffer guard precedes the
  shared submission mutex.

### D-IMPL-144: Add engine encoders for Q8_1 and dense mat-vecs

- **Decision:** Expose `encode_q8_1` and `encode_matvec` as validated engine
  operations. Support the eleven documented dense quantized formats and BF16,
  compiled NC widths 1/2/4/8, and the non-indexed forms. Derive block sizes,
  row grouping and pipeline names from one format table that mirrors the safe
  Metal kernels. Validate Context ownership, native CPU/GPU view metadata,
  byte ranges, alignment, output aliases, active-row metadata, dimensions and
  flattened 32-bit shader indices before encoding. Keep indexed-row draft-head
  dispatch separate until the MTP stage.
- **Reason:** L9's T=1 Window needs the same dense projections and output head
  entry points as later batched windows. Putting shape and binding checks in
  engine encoders lets Window assemble operations without duplicating kernel
  ABI rules, and makes Q8_1 production quantization composable with mat-vecs
  in one fenced command buffer.
- **Validation:** `metal_mv_test` checks `encode_q8_1` against its scalar
  oracle and gives every dense quantized format and BF16 a nonzero engine
  encoder result checked against its scalar oracle. The fixtures exercise
  compiled NC widths 1/2/4/8, large-K Q4_0 NC=2 (R=1), and BF16 NC=8
  (256 threads), plus padded output and sink handling. Rejection tests cover
  forged CPU/GPU view metadata, active-row mismatch, invalid NC, Q8_1
  input/output overlap, and mat-vec output or sink overlap with activations,
  weights, active-row metadata, and each other. Same-handle subviews and
  distinct no-copy handles over one CPU backing are rejected; an adjacent
  Q8_1 view succeeds. The focused Metal twin passes normally and with shader
  validation/debug layer enabled; the full host/Metal CTest run passes 49
  tests with one expected model-input skip.
- **Review required:** Connect the operations to Window and wire the indexed
  head encoder into MTP stage 21. Keep the 16,640 input-width ceiling aligned
  with the L3 fixture catalogue; revisit it when model dimensions are added.
  Model-backed parity remains open.

### D-IMPL-145: Keep macOS inference on Metal

- **Decision:** Do not add a CUDA runtime, CUDA device path, or CUDA toolchain
  requirement to the macOS build or inference path. Use Metal on supported Macs.
  CUDA references remain useful only in the separate PC reference workflow when
  the implementation plan requires numerical comparisons.
- **Reason:** CUDA is not the execution backend for Apple GPUs, and the project
  already has a Metal implementation plan for macOS. Making CUDA a Mac feature
  would add an unrelated backend and build dependency without helping Mac users.
- **Review required:** Keep Mac build and release checks independent of CUDA
  availability; retain CUDA parity requirements only for the separately
  specified reference-PC checks.

### D-IMPL-146: Add indexed draft-head mat-vec encoders

- **Decision:** Add `encode_indexed_head_matvec` for Q5_K, Q6_K and IQ4_XS
  with one active column. The signed int32 row map translates each output row
  to a row in the full head, so the Metal path reads `dvocab[r]` in place and
  does not require a gathered draft-head copy. Validate row counts and every
  map entry, full-head shader indexing and byte extents, Context ownership,
  native CPU/GPU view metadata, alignment, output/map overlap and command-buffer
  submission state before encoding. Keep the map and inputs immutable through
  command completion.
- **Reason:** MTP stage 21 specifies in-place indexed reads for an unsorted
  draft vocabulary. A dedicated single-column API makes the weight-row extent
  distinct from the output count and validates the exact map contract shared
  by all three supported head formats.
- **Validation:** The three indexed kernels compile into the Metal library and
  appear in the safe-kernel manifest. `metal_mv_test` compares nonzero engine
  results for all three formats against scalar oracles with an unsorted,
  non-contiguous map. It rejects negative and out-of-range entries, invalid
  row counts, signed-index and flattened shader-index overflow, unsupported
  formats, a wrong live-row value, short or misaligned maps, output/sink map
  overlap, foreign-Context maps, forged CPU/GPU metadata, and foreign,
  enqueued or committed command buffers. The focused test passes normally and
  with Metal API/shader validation enabled; 49 of 50 CTests pass and the
  model-input-dependent case is skipped as expected. The hostile follow-up
  confirms the initial P2 test-coverage finding is resolved and reports no
  remaining actionable finding.
- **Review required:** Connect the encoder to MTP stage 21 and measure the
  indexed read against a contiguous head as L-15 requires. Validate the map
  against shipped draft-vocabulary assets when model files are available.

### D-IMPL-147: Add standalone GDN decode and commit encoders

- **Decision:** Implement the four documented GDN kernels as validated
  caller-owned-command-buffer encoders: convolution with q/k L2 normalization,
  alpha/beta projections, recurrent step/output normalization with window,
  commit and write-through modes, and convolution-history commit. Preserve the
  documented 512-thread recurrent mapping and 3,136-byte dynamic threadgroup
  memory layout. Keep GDN production Window and Prefill wiring for their
  integration steps.
- **Reason:** GDN is L5 and precedes QSA in the implementation plan. Completing
  its standalone kernels, scalar oracle, and host validation first gives later
  decode and Prefill integration independently testable operation boundaries.
  The Mac path remains Metal-only; CUDA is used only by the separate reference
  workflow.
- **Validation:** `metal_gdn_test` passes for T=1..8, random/edge/fixture
  inputs and state modes 0/1/2. Edge coverage includes all-zero buffers and a
  row whose amax is exactly 2^10 times the other rows. The focused CTest passes
  with Metal API and shader validation enabled. The full CTest suite passes
  (50 passed, one model-input-dependent test skipped). Coverage includes
  scalar-oracle comparisons, exact commit/write-through state equality,
  history advancement, guards, input immutability and encoder rejection
  cases. The hostile review's two P2 findings were fixed and its read-only
  follow-up found no remaining actionable issue.
- **Review required:** Wire decode and commit into Window and Prefill only
  after their state ownership and command-buffer sequencing are established.
  Add model-backed `state_after_commit_eq_decode` and `pf_gdn_eq_step_norm`
  coverage, and confirm supported Apple GPU families allow the required
  512-thread pipeline.

### D-IMPL-148: Implement the first standalone QSA kernel

- **Decision:** Implement `strata_qsa_norm_rope` for K, Q and indexer-Q as a
  validated caller-owned-command-buffer encoder with a scalar oracle and
  standalone Metal twin. Preserve the documented 256-thread butterfly order,
  RMS normalization and NEOX RoPE FMA expressions. Allow exact in-place K and
  indexer-Q operation; require Q's strided input and compact output to be
  disjoint. Validate every step position against the supplied 32-float2-per-
  position RoPE table. Keep Window, Prefill and MTP integration, and the other
  L6 kernels, open.
- **Reason:** This is the smallest self-contained L6 kernel: it has a complete
  arithmetic contract and can be checked independently without KV storage,
  indexer append state, selection scratch or attention intermediates. The Mac
  implementation remains Metal-only; CUDA is only a separate reference path.
- **Validation:** `metal_qsa_test` passes for K/Q/indexer-Q at T=1..8 against
  the scalar oracle on random, edge and fixture inputs, including all-zero and
  2^10 outlier profiles, non-identity position-zero RoPE values, mixed
  positions, guards, read-only inputs and row-t independence. The encoder
  rejects invalid T/modes, short and out-of-allocation views, incomplete RoPE
  rows, misalignment, invalid positions, unsafe aliases, foreign Context
  resources, and forged CPU/GPU metadata. The focused CTest passes both
  normally and with Metal API and shader validation enabled. The target has no
  model-backed `real` input set and returns CTest skip code 77 for that option;
  real-input parity remains open under REV-002/025. Full CTest passes 52 tests
  with 51 passing and the model-input-dependent test skipped. Hostile-review
  findings and follow-up are recorded below.
- **Review required:** Add model-backed real-input parity when the model and
  CUDA reference fixtures are available under REV-002/025; integrate the
  encoder, FWHT and KV append into Window/Prefill/MTP; and implement and test
  indexer, selection, gate and attention before closing L6. Confirm the full
  QSA dispatch and RoPE table extents against live model/MTP buffers.

### D-IMPL-149: Add standalone O-class QSA FWHT256

- **Decision:** Implement `strata_qsa_fwht256` as a standalone in-place
  encoder over a generic positive number of 256-float rows. Use one
  256-thread group for up to eight independent simdgroups/rows, preserve the
  documented 1/16 load scale and exact lane/register butterfly ordering, and
  keep production Window/Prefill dispatch open.
- **Reason:** FWHT is the smallest remaining QSA operation with a complete
  independent contract. Implementing it separately allows O-class KV rotation
  and inverse-attention rotation to be tested without depending on KV storage,
  indexer selection or attention.
- **Validation:** `metal_qsa_test` passes normally and with Metal API/shader
  validation. FWHT rows=1/2/7/8/9/15/16/17/191/192 match an independent
  contiguous scalar oracle bitwise, including cross-dispatch row independence,
  eight lane/register-boundary impulse goldens and a dense float32
  scalar-spec output digest. Full CTest passes 52 tests with 51 passing and
  the model-input-dependent test skipped. The hostile follow-up found no
  remaining implementation issue. A CUDA-captured FWHT golden is unavailable
  and remains open under REV-025.
- **Review required:** Integrate the encoder under the documented
  q4/k8v4/`STRATA_KV_ROT` policy, including V-only rotation for k8v4; retain
  the bitwise impulse, dense digest and T-independent fixtures, and add
  model/CUDA reference parity when available under REV-002/025.

### D-IMPL-150: Add fp16 QSA KV append as the first storage mode

- **Decision:** Implement `strata_qsa_kv_append_f16` as a standalone validated
  encoder for T=1..8. One 256-thread group handles each token/KV-head pair and
  writes `half(K/V)` to the documented four-cell-interleaved pool row.
  Validate unique in-range `StepRow` positions, four-cell capacity, pool
  extents, alignment, aliases, Context ownership and command-buffer state.
  Keep int8, q4_0 and k8v4 append modes, and production
  Window/Prefill/MTP dispatch, open for later steps.
- **Reason:** Fp16 is the direct-storage baseline and has no scale or packing
  reduction that could obscure the cache address contract. Implementing it
  first gives the row mapping and encoder boundary an independent test before
  adding the quantized formats. Repeated positions would make separate
  threadgroups write the same cache rows concurrently, so the encoder rejects
  them before dispatch. This does not change the documented target of
  supporting all configured KV modes.
- **Validation:** The focused `metal_qsa_test` passes normally and with Metal
  API/shader validation enabled. Cases cover T=1..8, positions at cell 0,
  across four-cell row groups and at the capacity tail, random/edge/fixture
  values that distinguish each head and token, half subnormal/tie/overflow
  cases, bit-exact scalar-oracle output, untouched pool rows, guards,
  read-only inputs and invalid extents, capacities, positions, duplicate
  positions, exact/partial/cross-handle aliases, ownership and command-buffer
  state. The full host/Metal CTest passed 51 tests with one expected
  model-input test skipped (52 registered); after a final test-fixture
  metadata correction, the focused QSA test passed again normally and with
  Metal API/shader validation enabled.
- **Review required:** Connect fp16, int8, q4_0 and k8v4 append and rotation
  to Window/Prefill/MTP; add CUDA/model-backed comparison when the reference
  assets become available under REV-002/025.

### D-IMPL-151: Add int8 QSA KV append with 64-value scales

- **Decision:** Implement `strata_qsa_kv_append_i8` for T=1..8. Each
  256-thread group handles one token/KV-head row, reduces each adjacent pair
  of simdgroups to one maximum over 64 values, rounds the scale to fp16, and
  quantizes against that stored scale with round-to-nearest-even. Validate
  unique in-range positions, capacity and buffer extents/alignment, disjoint
  input/output ranges, Context ownership, native metadata and command-buffer
  state. Permit the four disjoint code/scale views to share one allocation.
  Require finite K/V values as an API precondition; do not scan them on the
  CPU because earlier GPU work in the same command buffer may produce them.
  Keep q4_0, k8v4 and production Window/Prefill/MTP dispatch open.
- **Reason:** This follows the documented cache layout and the safe-kernel
  rounding contract while preserving a single-allocation KV pool layout.
  Unique positions prevent separate threadgroups from racing on code or scale
  stores. Non-finite values have no defined quantization result in the
  documented scale/code equations, and host scanning would read stale data
  when K/V values are generated by an earlier GPU encoder in the same
  command buffer.
- **Validation:** The focused `metal_qsa_test` passes normally and with Metal
  API/shader validation enabled. The full host/Metal CTest passes 51 tests
  with one expected model-input skip (52 registered). `test_map.py` and
  `git diff --check` pass. Cases cover T=1..8 on random, edge and fixture
  values; cell zero, four-cell boundaries and capacity tails; exact code
  bytes and scale half bits; one shared allocation for all four output views;
  output poison that makes a no-store kernel fail; untouched rows, guards and
  read-only inputs; zero scales and explicit ties-to-even values; and encoder
  rejection cases.
- **Review required:** Connect QSA append to production dispatch and add
  external CUDA/model parity when reference assets are available under
  REV-002/025.

### D-IMPL-152: Add q4_0 QSA KV append after explicit FWHT encoding

- **Decision:** Implement a standalone `strata_qsa_kv_append_q4` encoder for
  T=1..8. The encoder consumes FWHT256-transformed K/V rows and writes eight
  packed 18-byte `block_q4_0` blocks per cache row. For each 32-value group,
  select the maximum-magnitude value with ties resolved to the larger value,
  compute the fp32 q4 scale and reciprocal, quantize with fused
  `fma(x, id, 8.5)` followed by truncation/clamping, store the finite fp16
  scale, and pack elements j and j+16 into the low/high nibbles. Callers
  encode the existing in-place FWHT kernel for K and V before append in the
  same command buffer. Validate unique in-range positions, capacity and
  buffer extents/alignment, disjoint input/output ranges, Context ownership,
  native metadata and command-buffer state. Require finite source values,
  finite stored fp16 scales and a finite fp32 reciprocal for each nonzero
  32-value group's scale; do not scan GPU-produced values on the CPU.
- **Reason:** This preserves the documented upstream operation order and
  q4_0 byte layout while keeping FWHT independently testable and reusable.
  Encoding FWHT and append in one command buffer gives explicit producer to
  consumer ordering. Tiny nonzero groups can make the fp32 reciprocal
  infinite, after which the documented float-to-integer conversion has no
  portable result. A scale outside fp16's finite range also decodes to an
  infinity and can produce NaN for zero codes. The caller precondition makes
  both boundaries explicit without reading potentially unfinished GPU
  outputs on the host.
- **Validation:** The focused `metal_qsa_test` passes normally and with Metal
  API/shader validation enabled. The full host/Metal CTest passes 51 tests
  with one expected model-input skip (52 registered). `test_map.py` and
  `git diff --check` pass. Cases exercise T=1..8, capacity tail and cell-group
  boundaries, same-command-buffer FWHT-to-q4 ordering, exact block bytes
  against an independent scalar oracle, max-magnitude sign ties, nibble
  order, zero-scale negative-zero bits, shared backing for K/V pools,
  cross-handle no-copy overlap rejection, no-store poison, guards, untouched
  rows, transformed input bits and validation failures.
- **Review required:** Connect all KV append modes to QSA production dispatch
  and add external CUDA/model parity when reference assets are available
  under REV-002/025.

### D-IMPL-153: Add combined K8V4 QSA KV append

- **Decision:** Implement `strata_qsa_kv_append_k8v4` as a standalone
  T=1..8 encoder. One 256-thread group handles each token/KV-head row and
  writes unrotated K as signed int8 with four 64-value fp16 scales, while
  writing V as eight packed 18-byte q4_0 blocks after FWHT256. The caller
  must encode FWHT256 on V only, in the same command buffer, before append.
  Validate unique positions, capacity and byte extents, alignment, native
  metadata, Context/command-buffer ownership, pairwise-disjoint K/V/step
  inputs, and output/input and output/output disjointness. Permit the three
  output views to share one backing allocation. Require finite K and
  transformed V values, finite stored fp16 scales and finite fp32
  reciprocals for nonzero V groups; do not scan GPU-produced inputs.
- **Reason:** K8V4's attention path consumes K without rotation and consumes
  q4 V after rotation. A dedicated dispatch keeps both cache writes in one
  operation while preserving those different contracts. Since V FWHT is
  in-place, overlapping it with K or `StepRow` would corrupt data needed by
  append; reject those aliases even though both are logically input views.
  Metal cannot safely inspect GPU-produced values on the host, so representable
  scales and finite transformed values are documented caller preconditions.
- **Validation:** `cmake --build build-dev -j4` passes. The focused
  `metal_qsa_test` passes normally and with Metal API/shader validation
  enabled. Full host/Metal CTest passes 51 tests with the one expected
  model-input test skipped (52 registered); `test_map.py` and
  `git diff --check` pass. Tests cover T=1..8 on random, edge and fixture
  inputs, cell zero/four-cell/capacity-tail positions, exact scalar-oracle K
  codes/scales and V blocks, same-command-buffer V-only FWHT, unchanged K,
  shared three-view output storage, poison, guards, all-zero and sign-tie
  cases, the largest finite fp16 scales, and invalid extents, aliases,
  ownership, metadata and command-buffer state. The first hostile review
  findings were fixed, and the read-only follow-up found no remaining
  actionable issue.
- **Review required:** Integrate the encoder with QSA Window/Prefill after
  ownership and command-buffer sequencing are established. Add CUDA/model
  parity when reference assets are available under REV-002/025.

### D-IMPL-154: Define the standalone QSA indexer snapshot ABI

- **Decision:** Implement `strata_idx_snapshot` for exactly 12 QSA layers,
  distinct from the model's 48 transformer layers. Read each layer's 384-float
  tail and `block_pos` through base views with independent strides measured in
  float and int32 elements. Write packed per-layer snapshots and copy tail
  words as raw uint32 bits. Reject overflow, short or misaligned views, foreign
  resources, submitted command buffers and every output overlap with another
  output or any strided source region. Run the snapshot once before appends in
  rollback-capable windows; write-through windows do not need a snapshot.
- **Reason:** The kernel plan specifies fixed layer strides but did not define
  their units or the host state allocation. Element-count strides make the
  source layout explicit while allowing the future Metal state arena to insert
  padding. Packed outputs make commit restore straightforward. Raw-bit copies
  preserve NaN payloads, subnormals and signed zero without floating-point
  arithmetic. Limiting this encoder to the 12 QSA layers avoids confusing
  them with all 48 transformer layers.
- **Validation:** The standalone twin seeds all layers with unique words under
  both packed and padded strides, checks exact output bits including NaN
  payloads/subnormals/signed zero, and checks untouched padding. It rejects
  undersized, misaligned and out-of-allocation views for all four buffers;
  exact and partial aliases in layers 0 and 11, including two no-copy handles
  over shared backing; and invalid strides, forged metadata, foreign resources
  and foreign/submitted command buffers. `cmake --build build-dev -j4`, normal
  and Metal-validation `metal_qsa_test`, the full 52-test host/Metal CTest
  suite (51 passed, one expected model-input skip), `test_map.py` and
  `git diff --check` pass. The encoder is not wired into Window or commit yet.
- **Review required:** Make the Metal `KvStore` state allocation expose these
  same per-layer strides, then verify snapshot-before-append ordering and
  restore/replay against accepted-prefix decode when Window dispatch is
  connected. Confirm rollback-capable versus write-through window scheduling.

### D-IMPL-155: Implement QSA indexer append and commit replay

- **Decision:** Implement one 128-thread `strata_idx_append` kernel for both
  window and commit modes. Process up to eight contiguous `StepRow::pos` cells
  in order, fp16-round each raw key before storing it in the three-slot tail,
  and preserve the specified serial sum, FMA mean, ordered RMS reduction and
  NEOX rotation. Window mode appends to current state. Commit mode restores
  only the selected QSA layer's tail and block position from the packed
  12-layer snapshot using raw uint32 tail copies, then replays the accepted
  prefix. The host encoder checks state, input and snapshot extents, table
  coverage, contiguous in-range positions, ownership, command-buffer state and
  all writable/input aliases. Allocate pooled state for
  `cell_capacity / 4 + 2` rows, matching the Metal `KvStore` plan.
- **Reason:** Sharing the row transition between tentative windows and commit
  replay makes accepted-prefix reconstruction use the exact same pooling path.
  `StepRow::pos` is the Metal cache-cell coordinate, and the Metal RoPE table
  already folds text or M-RoPE coordinates into each cell row; block `b`
  therefore reads table row `4*b`. The angle-zero pair is explicit so YaRN's
  magnitude is retained for the spare key. The current Metal ABI has no
  nonzero state-origin/`pos_base` field, so importing CUDA's separate
  `pos_base` into this encoder would misindex the folded table. If nonzero
  origins are needed, add separate state-cell and RoPE-table row mappings to
  the shared Metal position ABI.
- **Validation:** The independent scalar twin matches fp16-rounded tails and
  block positions exactly and checks pooled/dead values at U(64). Window cases
  cover T=1..8 at cells 0, 3, 1, 4 and 6 with random, edge and fixture inputs;
  edge raw keys include signed zero, fp16 subnormals, ties-to-even and the
  largest finite fp16 value. Tests also cover the spare's nonidentity
  angle-zero pair, completed and partial blocks, untouched rows and guards.
  Commit tests encode snapshot, full window append and prefix replay in one
  command buffer, keeping 1, 2, 4 or 7 rows across block boundaries. Rejection
  tests cover invalid rows/mode/capacity/
  layer, noncontiguous and out-of-range steps, short views, missing commit
  snapshots, exact/partial commit-snapshot overlap, disjoint snapshot views
  in one allocation, input/output and output/output aliases, foreign
  resources and submitted command buffers. Commit replay selects layers 0, 5
  and 11. The build, focused normal and Metal-validation `metal_qsa_test`,
  full host/Metal CTest (52 registered, 51 passed, one expected model-input
  skip), `test_map.py` and `git diff --check` pass.
- **Review required:** Connect the encoder to Window and commit, expose the
  real per-layer QSA state views from `KvStore`, and validate replay against
  accepted-prefix decode. Add nonzero state and folded-table origins only as a
  shared ABI extension. CUDA-captured/model-backed parity remains open under
  REV-002/025.

### D-IMPL-156: Implement standalone sparse QSA attention chunk and merge

- **Decision:** Add safe-mode Metal chunk kernels for fp16, int8, q4_0 and
  k8v4 KV layouts, plus a fixed-order merge kernel and validated standalone
  encoders. Derive scratch size and chunk count from T and `id_row_stride`;
  cap cache capacity at 262,144 cells and require a positive capacity divisible
  by four. Validate `StepRow` position/count/width relationships on the host,
  mask invalid IDs in the kernel before cache addressing, and keep valid
  live-prefix IDs as an explicit caller/producer precondition for complete
  attention results.
  Keep Window/Prefill integration outside this change.
- **Reason:** The documented CUDA-compatible algorithm has a fixed 64-cell
  chunk, an ordered online-softmax merge, and four KV storage modes. The
  bounded host encoder prevents malformed dimensions and writable aliases
  from producing invalid GPU accesses; the kernel also masks negative,
  out-of-capacity and future-cell IDs before loading cache data. Host-reading
  IDs would break the supported producer-to-consumer sequence when a preceding
  GPU encoder writes them, so validity remains the producer's contract. The
  262,144-cell ceiling follows the existing Metal context bound; it is a
  conservative implementation limit pending hardware/model validation.
- **Validation:** All four modes match an independent ordered scalar oracle at
  U(32), with random T=1..8 and edge/fixture T=8, fixture row-independence
  checks, one- and many-chunk strides, negative/capacity/future ID masking,
  exact-zero empty chunks, input/guard checks and encoder rejection tests. The
  focused CTest passes both normally and with Metal shader/debug validation
  enabled. Production inference and CUDA/model-backed parity remain open.
- **Review required:** Confirm the 262,144-cell encoder ceiling against the
  intended model/runtime configurations. During Window/Prefill integration,
  prove every live ID is in range and that producer/consumer ordering protects
  the ID and cache views; add production-path and CUDA/model-backed parity
  coverage.

### D-IMPL-157: Define the contiguous main-model QSA state layout

- **Decision:** Represent the 12 QSA layers' K/V pools, indexer tail/dead/
  pooled/block-position state, and one shared RoPE table as explicit regions
  in one allocation. Round requested capacities up to a four-cell boundary,
  retain the two documented spare pooled rows per layer, and reject capacity
  above 262,144 or unknown formats.
- **Reason:** The cache row mapping groups four cells, and the append/attention
  encoders require four-cell-aligned capacity. The CLI and planner accept
  other positive context values, so rounding preserves those inputs while
  giving every encoded row a valid backing region. A single address map makes
  each encoder's K/V and indexer views traceable and supports the documented
  shared RoPE table without per-layer Metal allocations. The drafter's
  distinct one-layer KV layout remains separate work.
- **Validation:** `metal_kv_layout_test` covers all four modes at capacities
  around group boundaries and at the 262,144-cell ceiling, checks every region
  extent and pairwise non-overlap, and rejects invalid capacities and formats.
  `metal_kv_store_test` verifies one Context-owned allocation, subview
  addresses, zeroed indexer startup state and invalid layer refusal on Apple
  GPU hardware. `metal_plan_test` checks all four formats with speculation at
  context 65 and confirms both main and drafter plans reserve capacity 68.
  The full host/Metal CTest run passes 55 tests with one expected skip for
  unavailable model inputs; after the MTP rounding fix, the focused layout,
  store and planner tests pass.
- **Review required:** Confirm this exact allocation size against reference
  model configurations and Metal memory limits, and validate MTP's eventual
  K/V writes and shared RoPE use in production Window/MTP sequencing
  (REV-002/025/033).

### D-IMPL-158: Construct decode-window parameters on the host

- **Decision:** Build the fixed `WindowParams` block from 1..8 tokens and one
  already-resolved sampler-settings row per token. Derive the smallest
  supported `nc`, all `StepRow` fields and each Philox counter from `pos0`;
  zero unused rows and preserve the ABI layout shared with Metal. Accept
  `n_keep = 0..T` because the same block is used for ordinary windows,
  write-through windows and commit replay.
- **Reason:** The block's layout and row arithmetic are normative in
  `docs/mac/04-metal-kernels.md` section 4.2. Centralizing construction
  removes duplicated position calculations from T=1 decode and T=2..8 verify
  callers. Sampler policy and history-buffer bounds remain owned by the
  existing options/session layer; this builder copies those resolved values
  and derives only the position-dependent counter.
- **Validation:** `window_params_test` covers T=1..8, NC rounding, the indexer
  and width boundaries, complete sampler-row copying, row zeroing, write-
  through metadata, the final valid signed position and transactional failure.
  Compile-time assertions compare the complete parameter, sampler and step
  layouts with the shared C/MSL declarations.
- **Review required:** Review sampler-setting ownership and the `n_keep`
  contract before connecting this builder to Window/commit dispatch. Model
  parity and production buffer-bound checks remain open under REV-002,
  REV-025, REV-028 and REV-031.

### D-IMPL-159: Add an explicit storage-only startup owner

- **Decision:** Add `ModelStorage` to select and own the planned Full/Tiered
  weight storage. Full mode recomputes the device-bounded window plan, opens
  the exact GGUF union and address/resident tables, then completes CPU/GPU
  warm-up before returning. Tiered mode delegates to the existing checked
  dense/resident copy owner. Reject an active MTP plan and reject full mode
  when converted dense-pack tensors are present, before allocating buffers.
  Add `--storage-check` as an explicit diagnostic path that inspects real
  inputs, prints the normal `PLAN`, and prints `STORAGE_CHECK ...` only after
  the selected weight storage succeeds. It never claims READY or inference.
- **History:** The initial implementation rejected full mode when converted
  dense-pack rows were present. D-IMPL-163 supersedes that part of the decision.
  The initial rejection of speculative plans was superseded by D-IMPL-174,
  which composes the MTP owner under the same storage transaction.
- **Reason:** The existing storage components had no common owner or driver
  path. Full mode cannot silently omit converted router rows, and neither mode
  owns the MTP runtime, PLE runtime, KV/session state or request dispatch.
  Failing closed on planned-but-unowned storage prevents a successful message
  from being mistaken for a complete model startup. A storage-only path gives
  the live full warm-up and tiered copy setup a reusable startup boundary
  without claiming inference support.
- **Validation (at initial landing):** `metal_model_storage_owner_test` runs full-mode open and
  warm-up plus tiered converted-router copies through the owner. It checks
  plan/geometry byte mismatches, source-vs-geometry dense and expert byte
  undercounts, missing and forged GGUF/artifact-set inspection locks, duplicate
  residents, active MTP weights and MTP KV, an unlisted non-empty `dense.bin`,
  and the then-current full-mode dense-pack rejection before allocation. `metal_buffers_test`
  checks rollback after provisional residency is committed. `options_dispositions_test` and
  `metal_cli_modes_test` cover option inventory, missing `--native` and
  incompatible planning mode. Failed Full/Tiered opens and failed full warm-up
  now remove their provisional Context-owned buffers before the Context can be
  reused. The focused storage and CLI/planning CTests pass. Real model startup
  and M-5 remain unverified because model assets are unavailable in this
  checkout.
- **Current status:** The dense-pack rejection was superseded by D-IMPL-163;
  converted-router full-mode weight storage and speculative MTP storage
  diagnostics are implemented and covered. Production model startup, main
  KV/session state, PLE, MTP dispatch and inference remain open.
- **Review required:** Review supported asset combinations before routing
  normal inference startup through this owner. PLE runtime loading, control
  vectors, main KV/session allocation, the production kernel ABI and
  model-backed M-5 remain open under REV-017, REV-019 and REV-020.

### D-IMPL-160: Keep the embedded engine tag at a binary string boundary

- **Decision:** Prefix the embedded `engine=<version>` tag with a NUL byte.
- **Reason:** The full Metal link placed a neighbouring printable byte directly
  before `engine=`, so CMake's anchored binary-string scan omitted the tag even
  though `setup.py` could still find its substring. A NUL boundary makes both
  build validation and the external version scanner independent of linker
  string coalescing.
- **Validation:** After the change, `strata_version` passes against the linked
  engine and adjacent metallib hash.
- **Review required:** Keep the executable tag check when changing link flags
  or the version-tag representation.

### D-IMPL-161: Roll back provisional Metal buffers when storage opening fails

- **Decision:** Add a scoped transaction to the Context buffer registry.
  Transactions serialize buffer-pool mutations, remove newly registered
  allocations from the residency set on failure, and release their retained
  no-copy mappings. Full and tiered storage opens use nested transactions;
  `ModelStorage` keeps an outer transaction through full-mode GPU warm-up.
- **Reason:** Destroying a storage candidate alone did not release MTLBuffers:
  the Context intentionally owns them for GPU lifetime safety. A failed copy,
  table allocation, residency request or warm-up could therefore leave memory
  registered after `open()` returned null, making a retry exceed its plan.
  Buffer-pool rollback preserves Context ownership on success while restoring
  the pre-open allocation set on failure. The buffer mutex makes each rollback
  apply only to the transaction's allocations even if another thread tries to
  use the same pool.
- **Validation:** `metal_buffers_test` commits temporary allocations inside a
  transaction, then verifies that rollback removes their residency entries and
  releases a no-copy mapping only after its Metal handle is removed.
- **Review required:** Verify nested rollback and Metal residency-set updates
  on the minimum supported macOS runtime, including a failed GPU warm-up. A
  regression now opens FullModeStorage inside an outer buffer transaction and
  confirms that an uncommitted outer transaction removes the inner
  transaction's committed buffers.

### D-IMPL-162: Bind inspection-lock claims to an owned descriptor

- **Decision:** Replace the public opaque `shared_ptr<void>` lifetime with a
  typed `WindowSourceLockLease`. The lease can only be created by opening the
  requested path, acquiring a nonblocking shared `flock`, and capturing that
  descriptor's file identity. Full and tiered storage verify the descriptor
  identity, the caller-visible snapshot, and the current path before mapping
  or copying. Production lease issuance is private to `WeightStore`. Synthetic
  storage fixtures use a separate engine target compiled with
  `STRATA_METAL_TEST_LOCK_ACCESS`; the production target has no test friend.
- **Reason:** A non-null `shared_ptr<void>` did not prove that WeightStore held
  any lock. Callers could forge matching path metadata while a cooperating
  writer replaced or mutated the source during storage opening. A typed lease
  ties the claim to a real descriptor kept alive for the full operation.
- **Validation:** `metal_window_storage_test` supplies exact source metadata
  without a lease and verifies full-mode opening fails before allocating.
  A compile-time check ensures the public lease type does not expose an
  acquisition method. `metal_weight_store_test`, `metal_window_storage_test`
  and `metal_tiered_copy_test` pass with descriptor-backed leases; both full
  and tiered paths use the same validation contract. The test-only issuer is
  compiled solely into the isolated storage-test engine.
- **Review required:** Confirm the internal issuer preserves the inspection
  lifetime and the supported filesystems honour the advisory `flock` contract.
  A noncooperating writer remains outside that guarantee under REV-010.

### D-IMPL-163: Load converted router rows in full mode

- **Decision:** Supersede D-IMPL-159's full-mode dense-pack rejection. Require
  every converted F32 router exclusion to match exactly one kind-4
  `dense.bin` row with the expected BF16 byte count. Validate unique names,
  canonical in-range layer IDs, file bounds, non-overlapping source ranges and
  supported router names before allocating. Bind the returned rows to an
  immutable catalog on the descriptor lease at WeightStore inspection, so
  storage rejects caller edits that remap a name to another valid byte range.
  Copy only those row ranges with `pread` from the retained inspection-lock
  descriptor into a zero-initialized anonymous shared arena.
  Align rows to 256 bytes, round the arena length to 16 KiB, bind each
  converted router to its BF16 arena address, and include the arena in GPU
  warm-up. Check the rounded arena against `maxBufferLength` and charge its
  padding to the memory plan's recomputed free budget. When there are no
  converted rows, an unlisted non-empty `dense.bin` remains unused.
- **Reason:** The F32 router is intentionally excluded from GGUF windows, but
  its validated BF16 conversion is part of the full-mode model layout. Rejecting
  all converted rows left that supported model family without a full-mode
  storage path. Loading the indexed rows preserves the no-wrap rule and keeps
  unrelated pack bytes out of memory.
- **Validation:** `metal_window_test` checks arena addresses and the
  `maxBufferLength` boundary. `metal_window_storage_test`,
  `metal_never_wrap_test` and `metal_model_storage_owner_test` verify exact
  row contents, immutable index-row binding in both storage modes, 16 KiB zero
  padding, ignored prefix/tail bytes, malformed and overlapping inventory,
  out-of-range router layers in full and tiered storage, lock rejection, memory-plan padding,
  outer-transaction rollback, and full-mode warm-up. Host/Metal CTest passed
  58/58; the model-input-only test was skipped because those assets are
  unavailable. This completes
  converted-router weight storage only, not U3, normal model startup or
  inference.
- **Review required:** Review the dense-pack identity and row inventory against
  real Swift packs, and measure full-mode warm-up and page-cache behaviour on
  the reference Mac. Noncooperating writers remain outside advisory-lock
  guarantees under REV-010; PLE/MTP runtime ownership and production startup
  remain open under REV-017.

### D-IMPL-164: Preserve validated MTP runtime inventory

- **Decision:** `WeightStore::inspect` may return an immutable
  `MtpRuntimeCatalog` containing the validated MTP runtime paths, sizes and
  device/inode identities, all 29 dense tensor names/kinds/shapes/offsets/byte
  counts, and draft-vocab IDs in their original file order. The two mapped files
  must be regular files opened without following a final symlink. If the
  optional draft-vocabulary file exists, open it once with nonblocking,
  no-follow flags, require a regular file, and read and validate IDs from that
  descriptor. `MtpRuntimeSources` retains the MTP generation-lock lease and
  shares the catalog with that lease; the lease matches only that exact catalog
  identity. Resolve the runtime directory once and use its canonical path for
  the lock, pending marker and catalog reader. Publish the output only after
  all model and geometry validation succeeds. Storage must retain this source
  object and verify the opened file identities before mapping.
- **Reason:** The existing inspector already validates these facts for the
  planner, but discarded them. Returning the exact validated inventory avoids
  a second parser and prevents a later storage layer from trusting a caller-
  substituted row catalog while also keeping the generation locked.
- **Validation:** `metal_weight_store_test` compares every returned dense row
  against the expected 29-row fixture, checks original ordering for
  non-monotonic draft-vocabulary IDs, verifies that a substituted catalog does
  not match the lock lease, confirms failed or disabled inspection leaves no
  published catalog, and proves a symlink alias retarget cannot redirect the
  canonical inventory or its retained lock. It rejects FIFO files for the
  experts, dense index and optional draft vocabulary without blocking.
  `metal_mtp_storage_test` checks the captured file identities against
  same-size replacements and symlinks.
- **Review required:** Verify the catalog against real generated MTP runtime
  assets when available. Model-backed MTP execution and inference remain open
  under REV-017.

### D-IMPL-165: Plan bounded MTP weight windows

- **Decision:** Plan the MTP `experts.bin` and `dense.bin` mappings as
  page-aligned windows no larger than the live `maxBufferLength`. Keep every
  S2 expert blob whole by ending expert windows at eight-blob boundaries, the
  smallest boundary shared by the 1,382,400-byte blob size and 16 KiB pages.
  Keep each dense tensor whole in one window and group adjacent rows while
  they fit. Reject the layout if any row or one aligned expert group cannot
  fit; require the immutable catalog to match its live generation-lock lease.
- **Reason:** Metal no-copy buffers require page-aligned ranges and have a
  device-specific maximum size. Engine kernels need each tensor/expert payload
  to remain a single contiguous binding, so a generic byte split would produce
  unusable views.
- **Validation:** `metal_weight_store_test` builds a layout from a real
  WeightStore-inspected synthetic catalog, checks all 512 expert views and all
  29 dense tensor views against their source offsets, verifies page alignment
  and the 64 MiB bound, rejects a 24 MiB limit that cannot hold the largest
  dense row, rejects a catalog substituted after inspection, and confirms a
  failed rebuild clears all windows, bindings and byte counters from an
  already-populated layout. The full build succeeded;
  host/Metal CTest passed 58/58 with one unavailable model-input test skipped.
- **Review required:** Confirm the eight-blob boundary and window sizes against
  `maxBufferLength` on each supported Mac family. Model-backed MTP use remains
  open under REV-017.

### D-IMPL-166: Map and retain the inspected MTP generation

- **Decision:** `MtpStorage::open` accepts only the immutable WeightStore
  catalog with its live generation-lock lease. It opens `experts.bin` and
  `dense.bin` read-only without following symlinks, checks that the generation
  lock path still names its inspected regular file, and checks each data
  descriptor's regular-file type, size, device and inode. It then maps the files
  into the planner's bounded no-copy windows and warms those windows before
  succeeding. The owner exposes each of the 512 expert blobs, all 29 dense
  tensors and the inspected draft-vocabulary IDs. Context retains the mappings
  and generation lock until its Metal buffers are released. A failed open rolls
  back provisional buffer registrations.
- **Reason:** The MTP files are part of the enabled speculation working set and
  must be resident before use. Keeping them in page-aligned file mappings avoids
  a second anonymous copy, while checking device/inode as well as size prevents
  a same-size replacement between inspection and mapping. The generation lease
  must outlive the buffers, not just the C++ view object, because Context owns
  Metal resources independently.
- **Validation:** `metal_mtp_storage_test` maps and warms the synthetic runtime,
  checks all expert/tensor views and vocabulary order, refuses a same-size
  replacement, data-file symlink and replaced generation-lock path, verifies
  that partial no-copy registrations roll back, and proves Context retains the
  generation lock after `MtpStorage` is destroyed. The fixture tests cover
  FIFO refusal, path replacement before/after pinning, ACL refusal before
  marker creation, and descriptor closure after ownership-marker/ACL failure.
  After review, fixtures use unique per-run directories without cleanup or
  reuse. The targeted build succeeded; four focused CTests and `test_map.py`
  passed; the final `ctest --test-dir build-dev -L 'host|metal'
  --output-on-failure` passed all 60 registered tests with one unavailable
  real-model-input test skipped. `git diff --check` passed.
- **Review required:** Validate MTP file mapping and warm-up against generated
  model assets and measure its page-cache and unified-memory cost on supported
  Mac families. Connect this owner to the plan/startup flow only when bind
  scratch and MTP KV ownership can be charged and rolled back together.

### D-IMPL-167: Give each MTP storage test a unique fixture directory

- **Decision:** Generate a random per-run fixture name under the build's
  dedicated `metal` directory, create it with mode `0700`, and never reuse or
  delete it. The helper accepts only a fixture parent owned by the current
  user, not writable by group or others, and without an extended ACL. It pins
  the fixture directory, requires current-user ownership, mode `0700`, no
  extended ACL, and checks its device and inode. It refuses pre-existing
  entries before creating the ownership marker, then passes that same
  descriptor and identity directly to both the fixture generator and storage
  test. Each child rechecks the descriptor with `fstat` and changes into the
  pinned directory before using relative paths. There is no pathname handoff
  between setup and the Metal test. A failed fixture setup leaves its unique
  artifact for inspection.
- **Reason:** A fixed reusable fixture requires cleanup. Checking an inode and
  then unlinking or removing by path leaves a race in which a replacement can
  be deleted. Passing the descriptor avoids redirecting either child if the
  build-tree entry is renamed or replaced after pinning. Requiring a private
  parent without ACL entries prevents other users from replacing a newly
  created directory before it is pinned. Per-run names and no cleanup remove
  the destructive cleanup path; the fixture data is sparse and confined to
  the build tree.
- **Validation:** The fixture self-test replaces the directory before and
  after descriptor pinning, checks non-empty and empty replacement behavior,
  refuses group/world-writable and ACL-enabled parents without creating
  entries, refuses an ACL-enabled fixture directory, and confirms a later
  setup leaves sentinel data untouched. It also injects marker `fstat` failure
  and verifies the descriptor closes. The storage test checks the inherited
  descriptor's device and inode before changing directory; the fixture pin
  test renames its pinned directory and replaces its old name with a symlink
  before writing through the descriptor.
- **Review required:** Per-run fixture directories accumulate under the build
  tree. Add cleanup only after an API can bind removal to the exact created
  directory throughout the operation; do not restore fixed-path reuse or
  check-then-remove cleanup. POSIX directory creation does not return a
  descriptor atomically, so a concurrent process running as the same user can
  still race the `mkdir`/`open` pair in the trusted parent. Reassess this
  boundary if same-UID processes are considered untrusted or if a platform API
  with atomic directory-handle creation becomes available.

### D-IMPL-168: Account only for the Metal drafter's live K/V regions

- **Decision:** Give the MTP drafter a device-free `MtpKvLayout` for one dense
  K/V layer. Round capacity to four cells, lay out the K/V and int8 scale
  regions, and map `k8v4` to plain int8. Do not allocate indexer rows, pooled
  keys, or RoPE state for this Metal path. Replace the planner's former
  2,080/1,088/608 B-per-cell figures with the exact K/V totals:
  2,048/1,056/1,056/576 B per cell for fp16/int8/k8v4/q4_0.
- **Reason:** The Metal MTP contract uses dense attention and has no indexer
  consumer. Upstream's generic QSA state still allocates pooled rows, but MTP
  does not read them; its 128-float rows also do not match the former
  32-B-per-cell planner allowance. Reserving only regions consumed by this
  Metal path keeps the byte layout exact and avoids carrying dead CUDA-side
  state into the Metal allocation.
- **Validation:** `metal_kv_layout_test` checks every region, scale extent,
  format mapping, four-cell rounding, maximum capacity, disjoint ranges and
  transactional refusal. `metal_plan_test` checks that the planner reserves
  exactly the layout size for all formats, including a non-four-aligned
  context.
- **Review required:** Before connecting MTP append/attention to production,
  verify from the final Metal encoders that no pooled or indexer region is
  consumed, and bind MTP RoPE reads to the main `KvStore`'s shared table rather
  than allocating a duplicate. Model-backed execution and external CUDA parity
  remain open under REV-017 and REV-025.

### D-IMPL-169: Allocate MTP K/V as a separate Context-owned Metal store

- **Decision:** Implement `MtpKvStore` as one Context-owned Metal allocation
  described by `MtpKvLayout`, with bounded K/V and scale subviews. Open it in a
  nested `Buffers::Transaction` so a wider owner can roll the allocation back.
- **History:** The store was initially kept separate from `ModelStorage` until
  MTP weights, bind scratch and K/V could be charged and rolled back together.
  D-IMPL-174 ends that separation for MTP storage diagnostics. D-IMPL-175
  adds main KV allocation to that owner; session initialization and inference
  startup remain open.
- **Reason:** The layout and memory plan already agree on exact live-region
  bytes, but at the time the model owner deliberately rejected plans containing
  MTP buffers. Composing only K/V would have exposed incomplete storage and
  made accounting or rollback misleading. The Metal drafter borrows the main
  `KvStore` RoPE table; this allocation must not duplicate it.
- **Validation:** `metal_mtp_kv_store_test` opens all four requested formats,
  checks one Context-owned allocation and exact K/V/scale subviews, writes edge
  markers through each present view, verifies invalid input allocates nothing,
  proves a wider transaction rolls back the nested committed allocation, and
  verifies that a later allocation cannot revive the rolled-back store's
  generation or make it return stale views.
- **Review required:** Before production use, verify that the session and
  encoder keep the store alive through every submitted command buffer, that
  publication occurs only after the wider startup transaction commits, and
  that the drafter binds the main `KvStore` RoPE view. Production session
  composition, append/attention and model-backed execution remain open under
  REV-017, REV-025 and REV-033.

### D-IMPL-170: Group planned MTP weights, bind scratch and K/V

- **Decision:** Add `MtpRuntimeStorage` as the transaction boundary for all
  planned MTP resources: the inspected no-copy weight windows, exactly
  `PlanResult::mtp_bind_bytes` of bind scratch with a device-free
  `MtpBindLayout`, and the exact `MtpKvLayout` allocation. Validate the plan's
  file bytes and bind regions against inspected geometry and its effective
  rounded bind-window capacity; validate K/V bytes against the exact layout.
  Expose only named bounded bind views; keep the allocation root private so
  consumers cannot slice into the reserved tail through this owner.
  Give every Context
  buffer a shared validity token: copies of a `BufferRef` become false and fail
  `Buffers::owns` after rollback or Context destruction. Since the public view
  metadata can be edited by callers, snapshot each issued view's handle,
  pointers, offset and maximum length. A view may be shortened, but widening
  or changing its backing metadata invalidates it; derive bounded views only
  through `BufferRef::slice`. A missing GPU address remains zero through
  slicing. Keep the MTP generation lock only through the Context's mapped-window
  backing lifetime, so rollback releases it even when an invalid child owner
  remains alive. At initial landing, this owner remained separate from
  `ModelStorage` while startup composition was pending.
- **Reason:** Startup needs a single rollback boundary for the three MTP memory
  classes before it can safely publish a drafter. `MtpStorage` and `MtpKvStore`
  already support nested transactions, but copied raw views also needed shared
  invalidation and the MTP lock needed to follow the mapped backing lifetime,
  not the child object's lifetime. Allocation-wide validity alone did not
  prove a public `BufferRef` still described its originally issued subrange:
  callers could rewrite offset, length and pointers together to address
  adjacent bytes in the same allocation. Per-view snapshots close that gap,
  while `slice` preserves supported subview creation and permits safe
  shortening. A missing GPU address is a sentinel, not a base address, so
  slices preserve zero instead of synthesizing an offset as an address. A byte
  count alone was not enough to prove the bind arena matched the planner;
  `PlanResult` now carries the rounded window capacity so the owner can
  recompute and verify the geometry-derived extent. `MtpBindLayout` exposes
  named aligned subviews, sizes the draft logits row to the inspected
  vocabulary, initializes window IDs to identity and copies inspected draft
  IDs into their device-side view before publication. The planner baseline
  retains a shape-checked reserved tail that is not exposed to kernels.
- **History:** D-IMPL-174 supersedes keeping this owner separate from
  `ModelStorage` for storage diagnostics. Production startup/session
  composition remains open.
- **Validation:** `metal_mtp_storage_test` opens a fixture-backed plan, checks
  its mapped weights, vocabulary, bind arena and K/V views, refuses a
  plan/layout byte mismatch before allocating, injects a deterministic
  post-weight-open failure to verify inner rollback, and then rolls back a
  successful nested owner while checking both owner accessors and previously
  copied `BufferRef`s. It also confirms the artifact lock is available while
  the invalid owner remains alive. `metal_buffers_test` verifies coherent
  slice widening is rejected, root and bounded views become invalid after
  Context destruction, and the GPU-address offset helper preserves the
  unavailable-address sentinel and rejects overflow. `metal_plan_test` checks
  the effective rounded MTP bind capacity against its byte count and geometry.
  `metal_mtp_bind_layout_test` verifies named region sizes, 256-byte start
  alignment, vocabulary-dependent logits, exact reserved-tail accounting,
  capacity arithmetic and refusal cases. `metal_mtp_storage_test` checks
  every exposed subview against the owned arena, byte-compares the initialized
  device vocabulary IDs with inspection and checks every initial window ID.
  The fixture test retains committed-storage generation-lock lifetime checks.
- **Review required:** Reconcile the reserved tail with the complete MTP ABI
  and keep it unexposed until then. Allocate and plan the separate `MtpParams`
  array before dispatch. The storage-diagnostic composition with `ModelStorage`
  is implemented under D-IMPL-174. Before production inference, compose the
  owner with main `KvStore`, session lifetime and startup, and verify the
  drafter borrows the main RoPE table; model-backed execution and external
  reference parity remain open under REV-017, REV-025 and REV-033.

- **Adversarial review status:** Harvey the 2nd's first pass found four issues:
  previously returned refs stayed non-empty after rollback; an undersized
  hand-built bind byte count could be accepted; an invalid owner could keep the
  artifact lock alive; and no deterministic test covered failure after the
  weight windows had opened. The current implementation adds shared
  `BufferValidity` tokens, geometry-based bind-byte validation, backing-only
  lock retention and a test-engine-only post-weight failure injection. Dirac
  the 2nd later found that public metadata could widen a view within its
  allocation and that Context destruction lacked a retained-view regression;
  per-view snapshots, `slice`, and `metal_buffers_test` now cover both.
  Banach the 2nd found that slicing a buffer with no GPU address synthesized
  a nonzero address from its offset. Its follow-up found that sampler and
  mat-vec validators independently reconstructed a nonzero address for those
  slices. The shared address helper now preserves the zero sentinel through
  slices and both validators; overlap and arena-alignment checks also treat
  zero as unavailable. Full-mode and tiered raw-address table builders now
  refuse a missing GPU base instead of publishing the window/arena offset as a
  pointer. Direct tests cover zero, normal offsets, overflow and both builder
  rejection paths. The final hostile follow-up found no additional actionable
  issue.

### D-IMPL-171: Enforce and reserve the MTP bind tail

- **Decision:** Derive named MTP bind regions from the normal-mode contract and
  expose only those bounded views. Check the exact shape-dependent remaining
  bytes against a fixed reference reservation adjusted only for known 256-byte
  alignment padding. Keep that tail within the planned allocation but
  unassigned and unavailable to encoders until the complete bind ABI explains
  it.
- **Reason:** The documented baseline and per-region dimensions do not account
  for the full allocation. Merely accepting any positive remainder could hide
  a later omitted or undersized region. The exact reservation invariant makes
  such layout drift fail closed while keeping the unresolved bytes inaccessible.
- **Validation:** `metal_mtp_bind_layout_test` checks region extents, start
  alignment, vocabulary-dependent logits sizing, exact reserved-tail bytes,
  capacity arithmetic and invalid-input refusal. The storage integration
  test checks every exposed view against its owning allocation.
- **Review required:** Reconcile the reserved tail against the complete MTP
  ABI before assigning or exposing it.

### D-IMPL-172: Keep the Mac GPU backend Metal-only

- **Decision:** Implement the Mac GPU path with Metal. Do not add CUDA runtime,
  kernels or a CUDA backend to the Mac build. Retain CUDA only where separate
  PC implementations or external reference data are required by the project.
- **Reason:** The target Mac hardware uses Apple's GPU stack; a CUDA backend
  does not provide a runnable Mac implementation. Keeping CUDA references
  outside the Mac backend avoids an unsupported dependency and build path.
- **Review:** Revisit only if the project adds a distinct CUDA-capable target;
  that would not change the Mac backend.

### D-IMPL-173: Plan and allocate MTP parameter slots

- **Decision:** Give the MTP drafter's `MtpParams` blocks their own Shared
  allocation of 256-byte slots. Reserve eight fixed slots for K and F0-F6
  plus at least one reusable group slot for short-read catch-up; a selected
  prefill chunk reserves one slot per group of up to eight rows. Include the
  base slots in always-resident memory and additional group slots in prefill
  admission; report their sum as `mtp_params_mib`, appended after the existing
  PLAN keys. Zero the allocation before publication and expose only one
  bounded slot at a time through `MtpRuntimeStorage`.
- **Reason:** Every recorded MTP kernel binds one pass-specific block, and
  prefill may encode several groups in one call. Omitting either set could
  let a nominally fitting plan exceed its actual allocation or reuse a block
  while the command buffer still reads it. Keeping the slots in a separate
  array matches the documented ABI and keeps the bind arena's reserved tail
  untouched. Short-read decode windows also trigger MTP catch-up when GPU
  prefill is disabled, so that path still needs a group slot.
- **Validation:** `metal_plan_test` checks group rounding, total slot bytes,
  no-spec omission, PLAN reporting and rejection of a forged byte count.
  `metal_mtp_storage_test` checks the planned allocation, zero initialization,
  bounded fixed/prefill slot views and transaction rollback invalidation.
  Full/tiered memory admission includes the correct fixed and variable terms.
- **Review required:** Confirm the final drafter encoder uses at most one
  parameter slot per eight-row group in a single prefill call and retains
  every slot until that call's command buffer completes. Production
  startup/session composition and dispatch remain open under REV-017 and
  REV-034.

### D-IMPL-174: Compose speculative MTP storage under ModelStorage

- **Decision:** When a fitting plan contains MTP weight, bind, parameter-slot,
  or drafter K/V bytes, require the inspected MTP catalog and open the complete
  `MtpRuntimeStorage` under the same outer `Buffers::Transaction` as the
  selected Full/Tiered model storage. Pass the retained MTP inspection sources
  through `--storage-check` and report `mtp_runtime=yes` only after both
  storage owners have opened successfully. Plans without MTP bytes do not open
  an MTP runtime, even if inspection found an optional MTP directory.
- **Reason:** Opening just the model weights for a speculative plan could
  report success while omitting nearly a gigabyte of planned MTP mappings and
  allocations. Nesting both owners under one transaction guarantees that a
  failure in either storage path rolls back the allocations and retained MTP
  file generation lease together. At this stage, the diagnostic still did not
  initialize per-session state or PLE, bind the shared RoPE table, or submit
  MTP work. D-IMPL-175 adds the main KV allocation without initializing its
  values or claiming inference readiness.
- **Validation:** `metal_mtp_storage_test` exercises MTP plan-byte validation,
  missing-source refusal, MTP allocation rollback when the later model-storage
  open fails, and a successful aggregate owner whose MTP runtime remains
  valid until Context teardown. Its CLI fixture verifies speculative Full and
  Tiered storage checks plus a non-speculative check with `--mtp` supplied;
  all three also confirm `window_scratch=yes`.
- **Review required:** The hostile review's common-transaction, source,
  validation, mode-coverage and scope findings are addressed and recorded
  below. Production session must still borrow the main `KvStore` RoPE table
  before MTP dispatch. Production startup, session lifetime, PLE, and inference
  remain open under REV-017 and REV-034.

### D-IMPL-175: Compose main QSA KV storage under ModelStorage

- **Decision:** For every accepted plan, recompute `QsaKvLayout` from the
  planned context and format, require its exact allocation size to match
  `PlanResult::kv_bytes`, and reject a layout larger than the live device's
  `maxBufferLength` before opening model or MTP storage. Open `KvStore` first
  inside the existing outer `Buffers::Transaction`, then open optional MTP
  runtime and Full/Tiered model storage. Expose the owned store through
  `ModelStorage::kv_store()`; keep MTP K/V separate and use the main store as
  the future shared RoPE source. `--storage-check` prints `main_kv=yes` only
  after successful allocation and storage startup. D-IMPL-184 also composes
  the planned Window scratch under this transaction.
- **Reason:** The planner already charges main KV, indexer and RoPE bytes.
  Omitting the allocation let storage diagnostics accept a plan whose charged
  state did not exist. Owning it with the model ensures byte accounting,
  rollback and lifetime match the plan. The diagnostic allocates the buffer
  but does not populate K/V or RoPE values and does not imply session readiness.
- **Validation:** `metal_window_storage_test` builds small valid main-KV plans,
  checks exact allocation and bounded shared-RoPE ownership, rejects forged
  `kv_bytes` before allocation, checks Full/Tiered rollback after a later open
  failure, and proves stale KV/RoPE views expire with an outer rollback.
  `metal_mtp_storage_test` covers exact plan rejection, the live
  `maxBufferLength` preflight through `ModelStorage::open`, and success with
  MTP under the same owner. The fixture runner checks `main_kv=yes` and
  `window_scratch=yes` for speculative Full/Tiered and ordinary storage-check
  paths.
- **Review required:** Confirm the session initializes K/V and RoPE values
  before use and that the MTP dispatcher consumes the runtime's borrowed main
  RoPE view. Session state, PLE and request dispatch remain open under
  REV-017/034.

### D-IMPL-176: Bind speculative MTP storage to the main QSA RoPE view

- **Decision:** Speculative `ModelStorage` opens `MtpRuntimeStorage` through
  `open_with_main_kv`. Before any MTP resource is opened, it validates the
  main store against the plan's context, KV format, exact allocation size,
  Context ownership, shared allocation identity, and exact RoPE subview.
  The MTP owner retains that bounded `BufferRef`; it does not allocate or
  initialize another RoPE table. `shared_rope()` returns an empty view unless
  the whole runtime and the borrowed allocation remain valid. Preserve
  `open()` as an unbound entry point for standalone storage use. Callers of
  `open_with_main_kv` keep the Context and main KV allocation live for the
  runtime's lifetime; `ModelStorage` enforces that order by destroying its
  MTP member before its main KV member.
- **Reason:** A borrowed view ties MTP to the exact main allocation that was
  planned, prevents duplicate RoPE storage, and keeps MTP resource allocation
  behind a no-cost compatibility check. Keeping standalone `open()` avoids
  making a main KV prerequisite implicit in isolated MTP storage operations.
  The storage binding does not establish initialized values, session state,
  or inference readiness.
- **Validation:** `metal_mtp_storage_test` verifies the borrowed view's
  handle, address, range and allocation identity; rejects live context-
  mismatched and rolled-back main KV before MTP allocates; proves standalone
  `open()` remains unbound; and checks that an MTP-only rollback empties the
  runtime accessor while the committed main KV and its original view remain
  owned. The aggregate `ModelStorage` case checks the same allocation identity.
  The test target builds successfully, and `ctest --test-dir build-dev -L
  'host|metal' --output-on-failure -j8` passes 61 tests; one model-real-input
  test is skipped because its assets are unavailable.
- **Review required:** Confirm session initialization writes the main RoPE
  values before any consumer, and that MTP dispatch binds this accessor only
  after the owning runtime is valid. Kernel wiring and inference acceptance
  remain open under REV-017/034.

### D-IMPL-177: Add fixed-address WindowParams ring storage

- **Decision:** Own the two window parameter slots in one Context-managed
  Shared buffer. Round each slot stride up to 256 bytes, initialize a slot
  through the canonical `build_window_params()` function, and expose a view
  only after the complete block has been copied. Reject stale owners and
  allocation sizes above the live `maxBufferLength` before publication.
  Callers select only a slot whose previous command-buffer use has completed;
  the ring does not track GPU submissions.
- **Reason:** The Window contract binds per-step values from fixed shared
  buffers and reuses two addresses across rounds. A single ring removes
  per-window parameter allocations while preserving a full-size slot for the
  command buffer and its following commit. The ring is part of the existing
  fixed engine-state reserve; its exact allocation is 2 ×
  `align_up(sizeof(WindowParams), 256)`.
- **Validation:** `window_params_ring_test` checks T=1 and T=8 blocks against
  the canonical builder, both stable slot addresses, failure-without-write,
  out-of-range rejection, outer-transaction rollback and a live
  `maxBufferLength` cap that refuses before allocation. `window_params_test`
  continues to cover the device-free parameter ABI and bounds. Both focused
  tests pass, as do all 63 registered host/Metal CTests; the real-input MoE
  test is skipped because its model assets are unavailable. On this reference
  host, the 1,536-byte ring increases `MTLResidencySet.allocatedSize` by
  131,072 bytes, so the test checks the reported size increases and the
  implementation checks the requested `BufferRef.length` exactly.
- **Review required:** Confirm Window/commit scheduling waits for the prior
  reader before reusing each slot, and include the ring in the final exact
  session/window memory accounting. This is an L9 prerequisite only; it does
  not encode a window or complete L9.

### D-IMPL-178: Add a shared Metal compute Encoder

- **Decision:** Add a C++ `Encoder` facade over one labeled Metal compute
  encoder per segment. It waits on and updates the Context hazard fence,
  validates every kernel against the manifest and its pipeline, validates
  Context-owned buffer generations/ranges/alignment and unique argument
  indices, and requires the active buffer and threadgroup-memory slots to
  match the pipeline's Metal binding reflection exactly. Clear stale slots
  when dispatches switch pipelines, and cap inline-byte payloads at 4 KiB per
  segment. Capture the immutable `DeviceFacts` once when the Context opens so
  dispatch-time limit checks do not repeat IORegistry and sysctl probes.
  `STRATA_METAL_DISPATCH` accepts unset/empty/`serial` or `concurrent`; other
  values fail closed. In concurrent mode, each ungrouped dispatch and each
  explicitly grouped sequence is followed by a buffer-scope barrier. Group
  declarations are no-ops in serial mode. Any body/dispatch failure or
  exception discards the unsubmitted command buffer.
- **Reason:** The implementation plan and section 8.2 define one encoder per
  segment and require an explicit policy for independent dispatch groups.
  Central validation and lifecycle handling make those contracts reusable
  when Window, commit, MTP and Prefill orchestration is added. Metal encoder
  resource bindings persist between dispatches, so reflected exact-slot checks
  prevent a missing binding from silently reusing an earlier resource. The
  inline-byte cap bounds `setBytes` storage for the segment.
- **Validation:** `metal_encoder_test` compares serial, concurrent and default
  output bitwise; exercises inline-byte and dynamic threadgroup-memory
  bindings and slot transitions; accepts exactly 4 KiB of inline bytes across
  dispatches and rejects individually valid payloads whose segment total is
  4,097 bytes; checks manifest, resource ownership/range/alignment, missing
  and duplicate slots, argument indices, threadgroup-memory limits, group
  balance and invalid mode; and verifies callback failure, an exception after
  dispatch, later command-buffer recovery and reused error strings.
  `metal_context_test` checks stable device-fact snapshots and validates both
  required and missing slots on a pipeline loaded from the binary archive.
  The focused Metal tests and full host/Metal CTest suite pass on the reference
  Mac; the one unavailable real-input MoE test is skipped.
- **Review required:** Existing typed dispatch encoders remain on their own
  lifecycle and have not been migrated to this facade, except PLE gate's new
  caller-owned-segment overload. Migrate the remaining operations and wire
  production Window, commit, MTP and Prefill segments through it; prove each
  declared concurrent group has no data dependency, and pass
  `metal_dispatch_modes_test` before claiming L14 or production
  serial/concurrent parity.

### D-IMPL-179: Keep the CUDA backend off macOS

- **Decision:** Reject `STRATA_ENABLE_CUDA=ON` when `CMAKE_SYSTEM_NAME` is
  `Darwin`, with a message directing Apple-silicon builds to Metal. Keep the
  CUDA backend available for supported NVIDIA systems and the macOS backend
  Metal-only.
- **Reason:** The Mac port is a native Metal backend, not a CUDA port. The
  CUDA reference PC is used to record comparison goldens; that does not require
  compiling or running CUDA on the Mac. The explicit guard prevents an
  unsupported toolkit probe and makes the documented platform boundary clear.
- **Validation:** Configuring on the reference Mac with
  `-DSTRATA_ENABLE_CUDA=ON` fails at the explicit macOS guard before CMake
  probes or enables a CUDA compiler. The upstream root-edit helper and patch
  produce byte-identical CMake files with 22 `# MAC:` markers. The host/Metal
  CTest run passes 64/64, with the unavailable real-input MoE case skipped.
- **Review required:** Confirm a Darwin cross-configuration fails before CUDA
  language/toolkit detection, and that Linux and Windows CUDA configure paths
  remain unchanged.

### D-IMPL-180: Allocate sequence recurrence state as one Shared owner

- **Decision:** Represent qwen4exp's 36 GDN recurrent states,
  36 three-row convolution histories and nine-row PLE history in one exact,
  256-byte-aligned Context-owned Shared allocation. Zero it on open. Before
  encoding a command buffer that uses a state view, register it with
  `track_use()`; reset refuses while any registered buffer is pending and
  clears only after each completes or is discarded. Keep QSA K/V and indexer
  state in `KvStore`.
- **Reason:** These mutable regions have fixed model geometry, are per-sequence
  state, and must be ready before the first token. A single allocation gives
  stable bounded views. Requiring a supported Qwen catalogue family plus 48 layers,
  12 QSA layers, two KV heads, 256-wide heads, embedding width 2560 and
  vocabulary 248320 rejects geometries outside the ABI this layout assumes.
- **Validation:** `metal_session_state_test` checks the exact byte layout,
  alignment, Context ownership, initial zero state, tracked command-buffer
  completion/discard before reset, rejected completion-handler registration,
  outer-transaction rollback, and fail-closed geometry/layer/context checks.
  The exact GDN+PLE state is 118,038,528 B
  (112.57 MiB), inside the 256 MiB `F_fixed` allowance. The planner now
  reports this exact size together with the capacity-specific Window scratch
  subtotal. Session allocation and per-command-buffer tracking remain a
  session-lifecycle responsibility. Keep the 0.25 GiB admission allowance
  until every fixed buffer reports a verified exact size. The separate ~50 MB
  pipelines/residency estimate is in the default 700 MiB working-set reserve,
  not `F_fixed`.
- **Review required:** Connect this owner to the Window/session lifecycle and
  call `track_use()` for every command buffer that accesses its views.

### D-IMPL-181: Resolve named dense tensors to checked full-mode views

- **Decision:** Expose whole dense GGUF tensors from `FullModeStorage` as
  `FullModeTensorView` values containing a bounded Context-owned `BufferRef`,
  GGML runtime type and shape. Resolve ordinary tensors through their planned
  file window and converted F32 routers through their BF16 arena row. Reject
  per-expert slabs, excluded tensors such as PLE, unknown names and views made
  stale by an enclosing allocation rollback.
- **Reason:** L9's dense dispatch needs the exact model type, shape and buffer
  extent at the point it binds a weight. Centralizing the resolution keeps
  callers from rebuilding GPU addresses or assuming that converted router
  rows retain their source GGUF type. Expert tables and the PLE owner have
  different access contracts and are not whole dense views.
- **Validation:** `metal_window_storage_test` checks ordinary mapped weights,
  converted BF16 gate and shared-expert router rows including a nonzero
  second-row offset, metadata and every byte, unknown/expert/PLE rejection and
  stale-view rejection after rollback. `metal_window_storage_test` passes;
  the complete host/Metal suite passes 65/65 with the unavailable real-input
  MoE case skipped.
- **Review required:** Use these views in the T=1 Window dispatch and verify
  every bound format, shape and extent against model-backed parity fixtures.

### D-IMPL-182: Calculate the complete T=8 Window scratch layout

- **Decision:** Use one checked, device-free layout for the fixed-capacity
  T=8 Window scratch allocation. Name every residual/activation, PLE, GDN
  replay, QSA, MoE, head, sampler and mat-vec-sink region; align each region
  start to 256 bytes. Parameterize only the rounded QSA cell capacity and the
  supported 256- or 512-expert model geometry. Keep mutable recurrent state
  and the main K/V/indexer/RoPE allocation in their existing owners.
- **Reason:** A single exact map provides stable bounded views for both a
  T=1 decode and T=2..8 verify window, sizes QSA scoring scratch from the
  actual cache capacity, and makes the fixed-memory subtotal auditable before
  production dispatch is connected. Mac has no PCIe miss path, so CUDA-only
  host staging and expert-hit scratch do not belong in the Metal allocation.
- **Validation:** `metal_window_scratch_layout_test` pins non-overlapping,
  256-byte-aligned regions and exact totals for both expert counts: 256 / 512
  experts use 47,641,344 / 47,649,536 B at 32,768 cells, 47,903,488 /
  47,911,680 B at 65,536 cells and 49,476,352 / 49,484,544 B at 262,144
  cells. It rejects unsupported model geometry and cell capacities. The full
  host/Metal CTest suite passes 66/66, with the unavailable real-input MoE
  test skipped. Planner reporting and ModelStorage composition are recorded
  in D-IMPL-184; production Window dispatch remains separate.
- **Review required:** Connect bounded views to the final Window, commit and
  draft consumers.

### D-IMPL-183: Own the Context-allocated Window scratch

- **Decision:** Allocate the checked D-IMPL-182 layout in one Context-owned
  Shared buffer. Reject unsupported geometry, invalid cell capacity and
  allocations above the device's `maxBufferLength` before publication. Expose
  generation-checked bounded views only for exact named layout regions, and
  let the existing `STRATA_METAL_POISON` allocator behavior initialize the
  scratch when requested. The caller owns command-buffer ordering when it
  reuses scratch; Context lifetime must cover the owner and every consumer.
- **Reason:** The layout alone cannot provide stable GPU addresses or enforce
  Context ownership. A single transaction-aware allocation keeps all kernel
  views at fixed addresses, lets outer model-open rollback invalidate every
  view, and preserves poisoned bytes for write-before-read diagnostics.
- **Validation:** `metal_window_scratch_storage_test` checks the exact logical
  allocation length, one-allocation ownership, every named region view,
  256-byte starts, `0xFF` poison in every named region, refusal before
  allocation for invalid geometry/capacity and a restrictive
  `maxBufferLength`, forged-region rejection, and outer-transaction rollback.
  The focused test and full host/Metal CTest selection pass 67/67; the
  unavailable real-input MoE test is skipped. `test_map.py` exits successfully
  with its existing planned tests and missing requirement checks reported as
  notes. ModelStorage/planner composition is recorded in D-IMPL-184;
  production Window/commit/draft consumers remain open.
- **Review required:** Connect bounded views to production Window, commit and
  draft consumers, and verify every queue-use lifetime.

### D-IMPL-184: Compose exact fixed-state planning with ModelStorage

- **Decision:** Keep `fixed_bytes = 256 MiB` as the single conservative
  admission charge. For supported Qwen catalogue geometries, compute exact
  `session_state_bytes` and capacity-specific `window_scratch_bytes` from the
  device-free layouts, check their sum fits `fixed_bytes`, and append both as
  informational PLAN fields nested under `fixed_mib`. `ModelStorage` rebuilds
  and validates those exact values and the rounded QSA cell capacity, then
  opens `WindowScratchStorage` inside its outer allocation transaction.
  Session state remains owned by the session lifecycle. Reject a plan with
  forged subcomponents, an undersized fixed allowance or scratch above the
  live `maxBufferLength` before allocating.
- **Reason:** The geometry `family` normally names the catalogue package
  (`qwen`, `swift` or `coder`); direct `WeightStore` inspection defaults it to
  `qwen4exp` when no package label is supplied. All four labels use the same
  qwen4exp runtime ABI. Exact architecture dimensions plus the fixed 256/512
  expert layouts let the planner work without Metal while keeping the display
  family meaningful.
  Charging the exact portions inside the existing allowance avoids counting
  them twice and keeps planning conservative until every fixed allocation is
  accounted for.
- **Validation:** `metal_plan_test` pins the 118,038,528 B session state and
  the exact 256/512-expert Window scratch totals at planned capacities; it
  confirms the subtotal stays inside the unchanged 256 MiB charge and that
  nested PLAN fields do not change `total_bytes`. `metal_mtp_storage_test`
  checks the planner/owner match, bounded owner views, forged and underfunded
  plan refusal, pre-allocation `maxBufferLength` refusal, and outer rollback
  during later storage failure. The storage-check fixture verifies the
  `window_scratch=yes` result in full, tiered and non-speculative startup.
- **Review required:** Reconcile the remaining fixed buffers before reducing
  `F_fixed`, then wire Window/commit/draft consumers and their queue-use
  lifetimes.

### D-IMPL-185: Build validated Window PLE row indices

- **Decision:** Add a device-free `WindowPleRows` builder for T=1..8. It
  validates supported qwen4exp runtime geometry, candidate token IDs and up
  to two committed predecessors in oldest-first order, then delegates hashing
  and EOS behavior to the artifact-backed `ngram_rows` implementation.
  Publish token-major row spans only after a complete successful build;
  invalid input preserves the prior result. Keep PLE reads, collection and
  Metal dispatch out of this component.
- **Reason:** Window must assemble the per-token predecessor matrix before
  requesting the sixteen PLE table rows. Reusing `ngram_rows` avoids a second
  implementation of the artifact-sensitive XOR, modulo and EOS rules.
  Explicit committed history keeps speculative rows separate from session
  history, so rejected candidates cannot advance persistent state. PLE direct
  I/O cannot safely be attached here while full-mode GGUF mappings may remain
  live on the same shard.
- **Validation:** The device-free host test covers T=1..8, empty and partial
  history, token ID zero, oldest-first predecessor matrices, EOS in committed
  and candidate history, pinned artifact hash output, fixed model geometry and
  vocabulary, invalid IDs, bounded row access, and unchanged output after
  rejection. The focused PLE row and reader tests pass 2/2; the full host/Metal
  CTest run passes 68/68 with one known real-input test skipped. The production
  `strata` target and focused test target build successfully. `test_map.py`
  exits 0 with the repository's existing planned tests and missing requirement
  checks reported as notes.
- **Review required:** Connect the generated indices to PLE issue/collect and
  resolve direct-I/O versus live GGUF mapping ownership before production
  storage startup.

### D-IMPL-186: Reuse the full-mode GGUF mapping for PLE reads

- **Decision:** Add an Mmap-only `PleTable::open_mapped` path that takes a
  shared `GgufModel` and the shard containing `per_layer_token_embd.weight`.
  Validate the shard, format and table extent using the existing reader
  contract; read FP8 format and scale from the model's first metadata shard.
  Point at the tensor bytes in the already-open mapping and retain the
  `GgufModel` until `PleTable::close`. Do not create a Direct reader from this
  path or expose the PLE payload as a Metal window/binding.
- **Reason:** Full mode keeps every GGUF shard mapped while its no-copy Metal
  windows live. `DirectFile` opens the same shard with `F_NOCACHE`, and the
  reader contract forbids direct reads while any mapping of that shard remains
  live. Sharing the validated CPU mapping is safe for Full mode and avoids a
  second long-lived map. Tiered Direct I/O requires a separate owner path after
  its transient GGUF mapping has closed.
- **Validation:** The synthetic sparse Qwen host fixtures check Mmap mode and
  exact PLE row count, reject an out-of-range shard, gather sparse rows after
  the caller releases its model reference, and verify `close` releases the
  retained mapping owner. A two-shard FP8 fixture places format/scale metadata
  in the first shard and the PLE payload in the second, then checks open and
  row gathering. F64 scale fixtures also check rejection when conversion to
  float overflows to infinity or underflows to zero, and a negative signed
  integer scale is rejected without being reinterpreted as unsigned.
  `metal_weight_store_test` exercises these alongside its existing inventory
  and lock checks.
- **Review required:** Define Tiered Direct I/O over an identity-pinned
  descriptor after shard mappings close, then connect issue/collect to Window
  staging and dispatch.

### D-IMPL-187: Compose the PLE table reader into full-mode storage

- **Decision:** `FullModeStorage` opens and owns `PleTable` through
  `open_mapped` using its existing `GgufModel`. `ModelStorage::ple_reader()`
  exposes a row-access-only `PleTableReader` in full mode and returns null in
  tiered mode. The view does not expose `open` or `close`, so callers cannot
  switch the mapped reader to Direct I/O or invalidate storage ownership. Add
  `ple_table=yes|no` to `--storage-check` output. Do not add PLE GPU bindings
  or start row collection/dispatch here.
- **Reason:** Full mode already retains the validated shard mappings for
  no-copy windows, so the CPU reader can share that owner and have the same
  lifetime as the storage. Tiered startup releases its transient mappings and
  will need a distinct Direct reader over the verified PLE file identity.
  Publishing null in tiered mode avoids accidental access to a closed mapping.
  Returning an operation-only view prevents callers from reopening the shared
  table as Direct or closing the storage-owned mapping.
- **Validation:** The live synthetic full-mode fixture now uses valid IQ4_NL
  PLE rows and checks that `ModelStorage` exposes an open Mmap reader with the
  exact row count. A compile-time capability check confirms the exposed reader
  has no public `open` or `close`; the tiered owner checks that no reader is
  published. The focused storage/fixture tests pass 7/7 and the full host/Metal
  CTest run passes 68/68 with one known real-input test skipped.
  `test_map.py` exits 0 with existing unassigned requirements reported as
  notes.
- **Review required:** Connect staged PLE input to the validated PLE encoders
  and the Window command sequence. Add Tiered Direct mode only after its PLE
  descriptor identity and mapping close order are enforced.

### D-IMPL-188: Stage Window PLE rows in shared scratch

- **Decision:** Add `stage_window_ple_input` to validate a prepared T=1..8
  `WindowPleRows`, an open artifact-sized `PleTableReader`, and the named
  `ple_input` scratch region. Gather the token-major batch with
  `PleTableReader::gather_batch` into Shared memory, then publish a bounded
  `BufferRef` covering exactly T × 2560 floats. Do not submit GPU work, change
  session history, or publish a view after any failed validation or read.
- **Reason:** Full mode retains the PLE shard mmap, and its reader cannot
  safely switch to Direct I/O while that mapping is live. `gather_batch`
  preserves token-major row order; its mapped implementation decodes rows
  synchronously. The exact table row count and per-row bounds reject truncated
  or mismatched artifacts; the active-length view prevents a later encoder
  from consuming stale rows from the unused T=8 suffix. The helper does not
  synchronize GPU use: the future Window scheduler must wait before reusing
  this single shared scratch region.
- **Validation:** A sparse full-size GGUF fixture writes a unique FP16 scale
  per row and a column-varying IQ4_NL code pattern for the artifact-generated
  indices. The live test uses an independent codebook oracle to check T=1..8
  decoded values, token/head ordering, split-half column order, exact active
  view lengths, untouched inactive rows, rejection of the existing 728-row
  truncated fixture, and rejection of an uninitialized row set. The focused
  `metal_model_storage_owner_test` passes 1/1, and the full host/Metal CTest
  selection passes 68/68 with the known unavailable real-input MoE case
  skipped. `test_map` also passes.
- **Review required:** Connect the staged input to the layer-1 PLE dispatch and
  tracked command-buffer use; add the Tiered Direct reader only with an
  identity-pinned descriptor and closed transient mappings.

### D-IMPL-189: Add PLE gate dispatch to a shared Encoder segment

- **Decision:** Add an `encode_ple_gate` overload for a caller-owned `Encoder`
  segment. Preserve the existing typed checks for T, manifest shape, buffer
  extents, Context ownership and output/input aliases, then submit the gate
  through `Encoder::dispatch` with the derived scale bound at argument 3.
  Route the standalone `CommandBuffer` overload through `Encoder::encode`.
- **Reason:** The Window specification places dependent kernels in one serial
  compute-encoder segment. The prior gate implementation always opened and
  ended its own Metal encoder, so a later PLE coordinator could not compose it
  into that segment. The reusable overload keeps the current validation while
  using the common manifest, binding and dispatch-policy checks. This step
  migrates only PLE gate; it does not force a dispatch mode, implement the
  rest of the PLE block or encode a Window command sequence. The future Window
  coordinator must create its documented serial segment.
- **Validation:** `metal_ple_test` exercises the existing T=1..8 parity and
  rejection cases through the standalone wrapper. Its T=1 fixture also
  dispatches the gate twice through one caller-owned segment and compares both
  outputs to the oracle and to each other. `metal_ple_test` and
  `metal_encoder_test` pass 1/1 each. `test_map.py` exits 0 with its existing
  planned-test and requirement-gap notes, including REQ-MEM-04's missing
  `accept:PLAN.models` and `ctest:metal_plan_test` mappings.
- **Review required:** Migrate PLE broadcast, conv-residual and history to
  caller-owned segments, then compose Q8_1, the PLE key/value projections,
  norms and post-ops before wiring the Window's command-buffer lifetime.

### D-IMPL-190: Add PLE value-broadcast to a shared Encoder segment

- **Decision:** Add an `encode_ple_broadcast` overload for a caller-owned
  `Encoder` segment, preserving its T, buffer, Context and alias checks before
  dispatching through the common manifest-validated facade. Route the existing
  standalone `CommandBuffer` wrapper through `Encoder::encode`.
- **Reason:** PLE gate and value-broadcast are dependent operations in the
  documented Window segment. Moving both to the shared facade allows the
  coordinator to encode them without opening a separate compute encoder for
  each operation. The wrapper keeps the existing independent-input overlap
  contract and remains available to focused standalone callers. The
  production PLE block and Window scheduler remain unimplemented.
- **Validation:** `metal_ple_test` checks bitwise T=1..8 output through the
  standalone wrapper and dispatches the T=1 broadcast twice through one
  caller-owned segment, comparing each result to the oracle and each other.
  The wrapper validation also rejects enqueued and committed command buffers.
  The existing `metal_embed_test` command-buffer concurrency test holds an
  encode callback while enqueue attempts and verifies serialization and
  failure handling. `metal_ple_test` and `metal_encoder_test` pass 2/2, and
  `metal_ple_test` passes in concurrent dispatch mode 1/1. The full suite
  passes 68/68 with one expected unavailable-model-input skip. `test_map.py`
  exits 0 with the baseline planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`).
- **Review required:** Compose gate-to-broadcast and the full PLE sequence in
  one caller-owned segment.

### D-IMPL-191: Add PLE conv-residual to a shared Encoder segment

- **Decision:** Add an `encode_ple_conv_residual` overload for a caller-owned
  `Encoder` segment. Preserve the T, pipeline, typed-buffer, allocation,
  Context and residual-alias checks, then dispatch the manifest-validated
  kernel through the common facade. Route the standalone `CommandBuffer`
  wrapper through `Encoder::encode`.
- **Reason:** Conv-residual is part of the documented Window compute segment.
  A caller-owned overload lets the future coordinator append it after the
  preceding PLE operations while preserving the fixed `(40, T) × 256` grid
  and the in-place residual contract. Production PLE and Window dispatch
  remain unimplemented.
- **Validation:** `metal_ple_test` checks T=1..8 outputs against the U(16)
  sequential-history oracle with F16 weights, including arithmetic-order
  controls. Its T=1 fixture dispatches twice through one caller-owned segment,
  checks each result against the U(16) oracle, and requires the two GPU
  outputs to match bitwise. The focused test passes 1/1 in normal and
  concurrent dispatch modes. The full suite passes 68/68 with one expected
  unavailable-model-input skip. `test_map.py` exits 0 with the baseline
  planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
- **Review required:** Compose gate, broadcast, conv-residual and history in
  one caller-owned segment.

### D-IMPL-192: Add PLE history to a shared Encoder segment

- **Decision:** Add an `encode_ple_history` overload for a caller-owned
  `Encoder` segment. Preserve the signed-index cap, fixed history extent,
  per-row norm extent, Context and overlap checks, and dispatch through the
  common manifest-validated facade. Route the standalone wrapper through
  `Encoder::encode`.
- **Reason:** PLE history is the remaining standalone kernel in the documented
  Window compute segment. The shared overload enables the future coordinator
  to append history work without opening another compute encoder. For `n=0`,
  validate the caller's empty norm view, then bind the valid history view in
  the unused norm slot: the shader reads no norm elements at zero rows, while
  the common dispatcher requires non-empty buffer bindings. This preserves
  the existing no-op behavior without weakening generic binding validation.
- **Validation:** `metal_ple_test` checks bitwise sequential-history parity
  for n=0/1/2/8/9/10/17, signed-index overflow rejection, bounds, aliases and
  zero-row empty-norm handling through both wrappers. Its T=1 fixture rebinds
  n from 1 to 2 in one caller-owned segment while advancing the same history
  buffer, then verifies the three-row result; a third n=1 dispatch verifies
  an independent one-row result. The focused test passes 1/1 in normal and
  concurrent dispatch modes. The full suite passes 68/68 with one expected
  unavailable-model-input skip. `test_map.py` exits 0 with baseline planned
  task notes and REQ-MEM-04 missing mappings (`accept:PLAN.models`,
  `ctest:metal_plan_test`); `git diff --check` passes.
- **Review required:** Compose gate, broadcast, conv-residual and history in
  one caller-owned segment, then connect the segment to Window PLE dispatch.

### D-IMPL-193: Add weighted RMS norm to shared Encoder segments

- **Decision:** Add an `encode_weighted_rms_norm` overload for a
  caller-owned `Encoder` segment. Retain the documented shapes, epsilon,
  buffer-range, Context, threadgroup-limit and alias validation. Route the
  standalone `CommandBuffer&` overload through `Encoder::encode`, and preserve
  its validation errors and exact in-place behavior.
- **Reason:** The PLE sequence normalizes key, query and convolution rows
  between PLE dispatches. A caller-owned overload lets a future coordinator
  encode these operations in one compute segment while retaining the existing
  standalone API. This work remains in the macOS Metal backend; D-IMPL-179
  keeps CUDA unsupported on macOS, so no CUDA runtime or kernels belong in
  this change.
- **Validation:** `metal_gr_test` checks weighted RMS norm against the U(64)
  oracle on the documented shapes; its `(rows=4, n_cols=2560, gp=4)` fixture
  dispatches twice through one caller-owned segment, checks both outputs
  against the oracle, requires bitwise equality between GPU outputs, and
  checks input/gamma immutability and output guards. The focused test passes
  in normal and concurrent dispatch modes. The full CTest suite passes 68/68
  with one expected unavailable-model-input skip. `test_map.py` exits 0 with
  its existing planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
  Mendel the 3rd's hostile review found no actionable issues.
- **Review required:** Compose the shared weighted norms with the PLE
  gate/broadcast/conv-residual/history sequence, then connect that segment to
  Window PLE dispatch.

### D-IMPL-194: Verify the documented PLE chain in one Encoder segment

- **Decision:** Add a T=1 fixture to `metal_ple_test` that composes the three
  weighted RMS norms, PLE gate, value broadcast, convolution-residual and
  history update twice in one caller-owned `Encoder` segment. The first chain
  uses distinct query and `ple_norm` buffers for stage-by-stage inspection;
  the second reuses query for `ple_norm`, matching the documented scratch
  lifetime, and must match the first chain's outputs bitwise. Before the
  reused query buffer is overwritten, a test-only
  `strata_selftest_copy_u32` dispatch snapshots its full query-norm result for
  direct U(64) and bitwise comparison. Keep this as a kernel-integration
  regression; it does not claim production Window wiring.
- **Reason:** Per-kernel twins verify each arithmetic contract separately but
  cannot catch a wrong dispatch order, missing inter-dispatch dependency or
  incorrect buffer rebinding across the documented sequence. Feeding each
  downstream scalar oracle with the preceding GPU output avoids accumulating
  independent kernel tolerances into false failures; initial key/query norm
  expectations are still derived from the CPU fixture. The full query snapshot
  ensures the reuse chain's query norm is checked directly instead of only
  indirectly through four gate results. The paired chain also checks that the
  documented scratch reuse preserves the same result.
- **Validation:** The all-target build succeeds. Focused `metal_ple_test`
  passes in normal and concurrent dispatch modes. The full CTest suite passes
  all 68 tests, with one expected unavailable-model-input skip.
  `test_map.py` exits 0 with its existing planned-task notes and REQ-MEM-04
  missing mappings (`accept:PLAN.models`, `ctest:metal_plan_test`);
  `git diff --check` passes.
- **Review required:** Run the sequence with model-backed inputs and connect
  it to production Window/Prefill dispatch.

### D-IMPL-195: Compose the PLE post-projection sequence

- **Decision:** Add `encode_ple_post_projection` overloads for a caller-owned
  `Encoder` and a standalone `CommandBuffer`. The coordinator accepts
  projected `key` and `value`, `R`, the norm parameters and the PLE state
  buffers, then encodes the three weighted norms, gate, broadcast,
  convolution-residual update and history advancement for T=1..8. Permit
  `query` and `ple_norm` to be disjoint or exact aliases, and reject overlap
  between every other buffer role. Keep projection GEMMs outside this helper.
- **Reason:** The individual encoders now have a directly tested documented
  sequence but Window and Prefill still need one reusable post-projection
  entry point. Exact query/norm aliasing implements the documented scratch
  lifetime; rejecting other cross-stage aliases prevents earlier outputs
  from clobbering values that later stages still read. The caller-owned form
  fits a larger serial segment, while its caller must propagate failure so a
  late validation error discards any earlier encoded dispatches. The
  standalone form provides that discard guarantee itself.
- **Validation:** All-target build succeeds. Focused `metal_ple_test` passes
  for T=1,2,8 in normal and concurrent modes. Full CTest passes all 68 tests
  with one expected unavailable-model-input skip. `test_map.py` exits 0 with its
  existing planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
  Final hostile review confirms its findings are closed.
- **Review required:** Wire the coordinator after the PLE projection stages
  in production Window and Prefill dispatch.

### D-IMPL-196: Make PLE history advancement explicit

- **Decision:** Require every caller of `encode_ple_post_projection` to pass
  `PLEHistoryUpdate::advance_rows` or `PLEHistoryUpdate::defer`. The former
  advances all T rows after the post-projection work and is for Prefill and
  write-through windows. The latter leaves committed history unchanged; a
  rollback-capable Window's later commit calls `encode_ple_history` for only
  its accepted `n_keep` rows.
- **Reason:** A speculative Window retains `ple_norm` for commit and must not
  mutate PLE history while evaluating drafts. Unconditionally advancing all
  rows would make rejected draft rows part of live session state and corrupt
  subsequent windows. Making the policy mandatory keeps this state boundary
  visible at each future integration site.
- **Validation:** `metal_ple_test` passes in normal and
  `STRATA_METAL_DISPATCH=concurrent` modes for T=1,2,8. The all-target build
  succeeds; full CTest passes 68/68 with one expected
  `metal_moe_real_inputs_unavailable` skip. `test_map.py` exits 0 with its
  existing planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
- **Review required:** Verify the production Window selects `defer` except
  when every row is guaranteed to be kept, and Prefill selects
  `advance_rows`; complete model-backed accepted-prefix parity with L10.

### D-IMPL-197: Add shared-Encoder dense projection dispatch

- **Decision:** Add caller-owned `Encoder` overloads for Q8_1 quantization and
  the regular dense quantized/BF16 mat-vec dispatch. Keep existing standalone
  `CommandBuffer` wrappers and kernel arithmetic unchanged; leave the indexed
  draft-head wrapper standalone.
- **Reason:** A dense projection consumes the Q8_1 rows produced immediately
  before it. Encoding both operations in one serial segment gives later
  projection coordinators an explicit dependency boundary and lets the
  enclosing command buffer discard the entire segment if validation of a
  later dispatch fails. This is a Metal-only Mac change and does not expand
  CUDA support on macOS; D-IMPL-179 keeps that backend boundary explicit.
- **Validation:** Focused `metal_mv_test` and
  `metal_mv_test_concurrent` both pass. The all-target build succeeds; CTest
  passes 69/69 with one expected `metal_moe_real_inputs_unavailable` skip.
  `test_map.py` passes with its existing planned-task notes and REQ-MEM-04
  missing mappings (`accept:PLAN.models`, `ctest:metal_plan_test`);
  `git diff --check` passes.
- **Review required:** Confirm exact kernel bindings and geometry in the
  shared form, serial dependency behavior, and that late failure leaves all
  segment outputs untouched. The next PLE projection stage must compose the
  correct key format and F32-activation BF16 value projection with the
  post-projection coordinator; production Window/Prefill integration remains
  open under REV-029.

### D-IMPL-198: Compose the complete decode PLE block

- **Decision:** Add caller-owned-Encoder and standalone-command-buffer
  `encode_ple_block` overloads. For T=1..8, the coordinator accepts Q2_0,
  Q8_0 or BF16 key weights, uses Q8_1 activations only for Q2_0/Q8_0, always
  feeds the original F32 PLE input to the BF16 value projection, reuses one
  mat-vec sink for the serial key and value projections, then invokes the
  existing post-projection coordinator with the caller's explicit history
  policy. Reject writable cross-stage aliases before encoding while allowing
  read-only overlap and the exact documented query/`ple_norm` scratch alias.
- **Reason:** The docs define one dependent PLE sequence whose key format
  varies by model file and whose BF16 value projection consumes F32 input.
  Keeping projection order, scratch reuse, alias checks and history policy in
  one encoder boundary reduces the chance that future Window and Prefill
  callers quantize the wrong input, use an unsupported key format, or commit
  speculative history. A standalone wrapper provides all-or-nothing discard;
  the shared form lets a caller compose the block into a larger fresh
  command-buffer segment.
- **Validation:** `metal_ple_test` builds and passes in both explicitly
  pinned serial and concurrent modes (2/2 focused CTest cases). The full
  CTest suite passes 70/70, with the expected
  `metal_moe_real_inputs_unavailable` skip. `test_map.py` passes with the
  baseline planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
- **Adversarial review:** The first review found a Q8_1 test-fixture
  overrun: a four-column oracle copy exceeded the two-row output allocation.
  The fixture now copies only live rows and checks guards captured before
  host initialization. Follow-up review confirmed that fix and found that
  the serial CTest case did not pin its dispatch mode; CTest now sets
  `STRATA_METAL_DISPATCH=serial` explicitly. Final hostile review found no
  remaining P1-P3 findings.
- **Review required:** Connect full-mode tensor views and staged PLE input to
  production Window and Prefill dispatch. Add Tiered Direct reads only after
  its descriptor identity and mapping lifetime are enforced; model-backed
  accepted-prefix parity with L10 remains open under REV-029.

### D-IMPL-199: Make the embedding prefix composable in a shared Encoder segment

- **Decision:** Add caller-owned-Encoder overloads for text embedding rows,
  image-row copies and GR stream broadcast. Route each shared form through
  the manifest-validated `Encoder::dispatch` facade, while preserving the
  existing standalone wrappers' validation and command-buffer behavior.
  Register the embedding fixture in explicitly pinned serial and concurrent
  modes.
- **Reason:** Window and Prefill each begin with dependent embedding work,
  and production must encode later kernels in the same segment when required
  by the untracked-resource ordering rules. Shared overloads let a future
  coordinator join these operations without opening a second compute encoder.
  Keeping the existing standalone wrappers' pre-dispatch failure behavior
  avoids changing callers that reuse an unsubmitted command buffer after
  validation errors.
- **Validation:** The `metal_embed_test` target builds; both explicitly
  pinned serial and concurrent cases pass (2/2). The full build succeeds and
  the rebuilt full CTest suite passes 71/71, with the expected
  `metal_moe_real_inputs_unavailable` skip. `test_map.py` passes with the
  baseline planned-task notes and REQ-MEM-04 missing mappings
  (`accept:PLAN.models`, `ctest:metal_plan_test`); `git diff --check` passes.
  The first focused run caught a compatibility regression when standalone
  validation failures began discarding a reusable command buffer. Restoring
  the original standalone path fixed it; the shared Encoder chain passes in
  both modes.
- **Adversarial review:** Archimedes the 3rd found no P1-P3 issue. The review
  checked buffer bindings, manifest geometry, standalone API behavior,
  output oracles and guards, concurrent barriers, and CTest/test-map/docs
  consistency.
- **Review required:** Apply the same typed shared-Encoder contract to the
  remaining GR, GDN, QSA, MoE and sampler dispatchers before the Window
  coordinator can encode its full dependency chain.

### D-IMPL-200: Make the GR post write composable in a shared Encoder segment

- **Decision:** Add a caller-owned-Encoder overload for `encode_gr_post`,
  using the manifest-validated `Encoder::dispatch` facade. Preserve the
  existing extent, alignment, Context ownership and exact residual/output
  alias rules. Register the GR fixture in explicitly pinned serial and
  concurrent modes.
- **Reason:** GR post is the native residual write used after the first
  layer's FFN and at the final layer. A shared overload lets Window append
  this operation to the layer's existing dependency segment while keeping
  the in-place residual contract explicit.
- **Validation:** Focused serial/concurrent tests passed, the full build
  succeeded, all 72 CTests passed except the expected skip for unavailable MoE
  real inputs, `test_map.py` passed, `git diff --check` passed, and hostile
  review found no P1-P3 issues.
- **Review required:** Add shared-Encoder overloads for the remaining GR
  read, projection, control-vector and head-mixer operations before wiring
  the complete Window layer loop.

### D-IMPL-201: Make fused GR read normalization composable

- **Decision:** Add a caller-owned-Encoder overload for `encode_gr_norm`.
  Preserve T=1..8, six-view extent/alignment/ownership checks and pairwise
  disjointness; bind the `apply` parameter at slot 6 and dispatch the
  manifest-validated `(rows, HC)` grid. Keep the standalone command-buffer
  overload unchanged.
- **Reason:** GR normalization begins each layer's read path. The shared
  overload lets a future Window layer append it to the same serial Encoder
  segment as its projection and residual operations.
- **Validation:** The updated `metal_gr_test` target builds; serial and
  concurrent GR tests pass (2/2); the full build succeeds; all 72 CTests pass
  except the expected skip for unavailable MoE real inputs; `test_map.py`
  passes with its baseline planned-task and REQ-MEM-04 notes; and
  `git diff --check` passes.
- **Review:** Galileo the 3rd confirmed shared and standalone binding slots,
  apply behavior, geometry, rejection coverage and serial/concurrent oracle
  parity. Follow-up found no P1-P3 issues.

### D-IMPL-202: Make GR control-vector work composable

- **Decision:** Add a caller-owned-Encoder overload for `encode_gr_cvec`.
  Preserve T=1..8, layer/mode validation, six-view extent/alignment/ownership
  checks and pairwise disjointness. Bind the four-word `GRCvecParams` block
  at slot 6 and dispatch the manifest-validated `(HC, rows)` grid. Keep the
  standalone command-buffer overload unchanged.
- **Reason:** `strata_gr_cvec` performs both the optional control-vector
  steering and the folded residual write in the GR read path. A shared
  overload lets Window place it in the same serial layer segment while
  retaining the operation's ordered projection and write semantics.
- **Validation:** The updated `metal_gr_test` target builds; serial and
  concurrent GR tests pass (2/2); the full build succeeds; all 72 CTests pass
  except the expected skip for unavailable MoE real inputs; `test_map.py`
  passes with its baseline planned-task and REQ-MEM-04 notes; and
  `git diff --check` passes.
- **Review:** Galileo the 3rd confirmed the four parameter words, binding
  slots, geometry, validation coverage and serial/concurrent oracle parity.
  No P1-P3 issues were found.

### D-IMPL-203: Make GR pre-gated projection composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_gr_pre_gated`, dispatched through the manifest-validated
  `Encoder::dispatch` facade. Preserve T=1..8, the fixed `(10, rows)` grid,
  three-view extent/alignment/Context checks and the alias rule that permits
  overlap between read-only `xn` and `gate` while requiring `mixed` to be
  disjoint from both. Keep the standalone command-buffer overload unchanged.
- **Reason:** The GR head mixer runs after its low-rank projection and needs
  its output before later head operations. A shared overload lets the future
  Window coordinator append it to the same dependency segment without
  changing the fixed four-stream reduction order or the accepted input alias.
- **Validation:** Focused serial and concurrent `metal_gr_test` pass (2/2);
  the full build succeeds; all 72 CTests pass except the expected skip for
  unavailable MoE real inputs; `test_map.py` passes with the baseline planned
  task and REQ-MEM-04 notes; and `git diff --check` passes. The initial hostile
  review found one P3 documentation omission in the §10.5 overview; it was
  fixed and follow-up review confirmed closure with no remaining P1-P3 issues.
- **Review required:** Before production Window/head-mixer wiring, confirm the
  coordinator preserves the fixed grid, output/input alias rule and serial
  dependency ordering through the full head-mixer chain.

### D-IMPL-204: Make GR down-SiLU composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_gr_down_silu`. Preserve T=1..8, derive `T × HC_LOW_RANK` from rows,
  validate the in-place float view and Context ownership, bind the derived
  count at slot 1 and dispatch the manifest-validated tail-safe grid. Keep the
  standalone command-buffer overload unchanged.
- **Reason:** GR down-SiLU runs between the down and up projections, including
  the head-mixer chain. A shared overload lets both sequences remain inside
  one serial Encoder segment without changing the in-place lifetime or tail
  behavior.
- **Validation:** Focused serial and concurrent `metal_gr_test` pass (2/2);
  the full build succeeds; all 72 CTests pass except the expected skip for
  unavailable MoE real inputs; `test_map.py` passes with the baseline planned
  task and REQ-MEM-04 notes; and `git diff --check` passes. Hostile review found
  no actionable P1-P3 issues.
- **Review required:** Verify the derived count, inline binding, grid and
  partial-group bounds when this operation is composed with its production
  projection consumers.

### D-IMPL-205: Make the fused GR up projection composable

- **Decision:** Add a caller-owned-Encoder overload for `encode_gr_up`.
  Preserve T=1..8, the two-word `{rows, apply}` parameter block at slot 8,
  the fixed 160 × 256 dispatch, eight-view extent/alignment/Context checks,
  pairwise disjointness and the static threadgroup-memory limit. Keep the
  standalone command-buffer overload unchanged.
- **Reason:** The fused GR up projection consumes the low-rank activation
  after down-SiLU, optionally folds the pending residual write, and produces
  the next block activation. A shared overload lets the GR read sequence keep
  this state transition inside its serial Encoder segment.
- **Validation:** Focused serial and concurrent `metal_gr_test` pass (2/2);
  the full build succeeds; all 72 CTests pass except the expected skip for
  unavailable MoE real inputs; `test_map.py` passes with the baseline planned
  task and REQ-MEM-04 notes; and `git diff --check` passes. The initial hostile
  review found one P3 documentation omission about failure propagation; it
  was documented, and follow-up review confirmed closure with no remaining
  P1-P3 issues.
- **Review required:** Confirm production GR coordinators bind the same
  `apply` policy and preserve the pending residual write ordering when
  composing GR norm, down, up and cvec. Propagate every failed shared encode
  through `Encoder::encode` to discard the complete segment.

### D-IMPL-206: Make GR down and injection projections composable

- **Decision:** Add a caller-owned-Encoder overload for `encode_gr_down`.
  Preserve T=1..8, the five disjoint view checks and float/BF16 alignment,
  bind the row count at slot 5, bind `T × 512 × sizeof(float)` dynamic
  threadgroup memory at index 0, and dispatch the fixed 41 × 256 grid through
  the manifest-validated facade. Keep the standalone wrapper unchanged.
- **Reason:** GR down produces the low-rank input used by down-SiLU and the
  per-stream injection outputs consumed by GR up. Composing it in the same
  serial segment makes those producer/consumer edges explicit and validates
  the T-sized scratch requirement at the dispatch boundary.
- **Validation:** Focused serial and concurrent `metal_gr_test` pass (2/2);
  the full build succeeds; all 72 CTests pass except the expected skip for
  unavailable MoE real inputs; `test_map.py` passes with the baseline planned
  task and REQ-MEM-04 notes; and `git diff --check` passes. The initial hostile
  review found one P3 status-record omission; this entry now records the
  completed checks, and follow-up review confirmed the record is current with
  no new documentation inconsistencies.
- **Review required:** Ensure production coordinators pass down outputs to the
  matching down-SiLU/up stages in order and propagate shared-encode failures
  so the whole segment is discarded.

### D-IMPL-207: Make GDN convolution and q/k normalization composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_gdn_conv_l2`. Preserve the standalone contract for T=1..8,
  float-aligned Context-owned views, read-only input overlap and output/input
  disjointness; dispatch the manifest-checked 80×T grid with 128 threads and
  bind only the four reflected buffer slots. Keep the standalone
  command-buffer overload.
- **Reason:** This operation reads committed convolution history and produces
  the normalized q/k rows consumed by later GDN stages. A shared overload
  allows the production layer coordinator to keep that dependency in its
  ordered compute segment without changing the standalone API. Metal
  reflection omits the shader's unused row-count argument, so the shared
  dispatch leaves that slot unbound; the requested row count is already
  represented by the grid's y dimension.
- **Validation:** Focused `metal_gdn_test` passes (1/1); the full build
  succeeds; all 72 CTests pass except the expected skip for unavailable MoE
  real inputs; `test_map.py` passes with the baseline planned-task and
  REQ-MEM-04 notes; and `git diff --check` passes. Ramanujan the 3rd's
  hostile review found no P1-P3 issues.
- **Review required:** Verify the eventual production coordinator preserves
  convolution-history read ordering and propagates a failed shared encode so
  the complete segment is discarded.

### D-IMPL-208: Make GDN alpha/beta gates composable

- **Decision:** Add a caller-owned-Encoder overload for `encode_gdn_ab`.
  Preserve T=1..8, the seven typed view extents, 16-byte BF16 weight
  alignment, Context ownership, and disjoint gate/beta outputs from each
  other and every input. Bind `rows` at slot 7 and dispatch the fixed 12×256
  grid through manifest/reflection validation; retain the standalone wrapper.
- **Reason:** The gate and beta projections are part of every GDN layer's
  window and consume the current mixed activation. A shared overload lets the
  production layer append them to the same validated compute segment while
  preserving their separate outputs and exact binding ABI.
- **Validation:** Focused `metal_gdn_test` passes (1/1); the full build
  succeeds; all 72 CTests pass except the expected skip for unavailable MoE
  real inputs; `test_map.py` passes with the baseline planned-task and
  REQ-MEM-04 notes; and `git diff --check` passes. Boyle the 3rd's hostile
  review found no P1-P3 issues.
- **Review required:** Verify that Window supplies the current mixed
  activation and matching alpha/beta weights and carries both gate outputs to
  `strata_gdn_step_norm` in the same row order.

### D-IMPL-209: Make GDN recurrent step norm composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_gdn_step_norm`. Preserve the three state modes, rows=0 commit
  no-op, T/mode, buffer, Context, alias and state-disjointness contracts.
  Bind `{rows, mode}` at slot 7, provide reflected threadgroup-memory slots
  0..3 with 128, 128, 512 and 16 floats, and dispatch 48×512 through the
  manifest/reflection-validated facade.
- **Reason:** The recurrent kernel emits window outputs without mutating
  state, replays accepted rows for commit, or performs both for write-through.
  Sharing one Encoder segment preserves those ordering boundaries and lets
  the coordinator keep the GDN recurrence on its layer's compute path.
- **Validation:** Focused `metal_gdn_test` passes (1/1); the full build
  succeeds; all 72 CTests pass except the expected skip for unavailable MoE
  real inputs; `test_map.py` passes with the baseline planned-task and
  REQ-MEM-04 notes; and `git diff --check` passes. Hostile review found that
  validation errors labeled every view as `state`; diagnostics now name the
  actual view, and a test checks the short-`h` case. Follow-up confirmed the
  P3 is closed with no new actionable issue.
- **Review required:** Confirm Window selects `window` for speculation,
  `commit` for the accepted prefix and `write_through` only when all rows are
  retained; preserve recurrence-state ownership and propagate segment failure.

### D-IMPL-210: Make GDN convolution-history commit composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_gdn_conv_commit`. Keep rows=0 as a no-op; for rows 1..8 preserve
  the standalone extent, float-alignment, Context-ownership and history/qkv
  disjointness checks. Bind history/qkv at slots 0/1, `rows` at slot 2, and
  dispatch the manifest-validated 80×128 kernel. Retain the standalone
  command-buffer API.
- **Reason:** A speculative GDN window must read the old convolution history
  before commit advances it by the accepted prefix. The shared overload lets
  Window's later ordered commit segment compose history advancement with
  recurrent-state replay while preserving the existing no-op and alias rules.
- **Validation:** The focused `metal_gdn_test` passes; the full `build-dev`
  build succeeds; all 72 CTests pass except the expected skip for unavailable
  MoE real inputs; `test_map.py` passes with the baseline planned-task and
  REQ-MEM-04 notes; and `git diff --check` passes. The first hostile review
  found only that this decision record was missing. It is now added; the
  read-only follow-up found no implementation issue and its stale future-tense
  status wording has been corrected.
- **Review required:** When Window commit is wired, verify that convolution
  history advances only after the window's convolution readers finish, that
  it uses `n_keep`, and that a failed shared encode discards the full commit
  segment together with recurrent-state replay.

### D-IMPL-211: Make QSA norm/RoPE composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_norm_rope`, sharing the standalone validation path for all three
  modes, T=1..8, view extents/alignment, Context ownership, step positions,
  exact in-place K/indexer-query, and disjoint query input/output. Bind the
  five views at slots 0..4, `{token_rows, mode}` at slot 5, and threadgroup
  memory at slots 0 and 1 (8 and 256 floats). Preserve the standalone API.
- **Reason:** K, Q and indexer-Q normalization are separated by projection,
  quantization and other QSA stages in Window. A shared overload lets the
  eventual coordinator preserve their order in one validated compute
  segment. The Mac path remains Metal-only; CUDA remains a separate
  numerical-reference workflow.
- **Validation:** The `metal_qsa_test` target builds and its focused CTest
  passes. T=1/T=8 fixture cases for all three modes execute through the
  shared overload and compare against the same scalar oracle. The full
  `build-dev` build succeeds; all 72 CTests pass with the expected skip for
  unavailable MoE real inputs. `test_map.py` passes with the baseline
  planned-task and REQ-MEM-04 notes, and `git diff --check` passes. Dirac's
  hostile review found one stale review-status record, corrected below, and
  no implementation issues.
- **Review required:** Confirm Window, Prefill and MTP supply the correct
  mode-specific input strides and output views, retain exact in-place
  K/indexer-query only, and keep step positions within the live RoPE table.
  Propagate shared-encode failure so the containing segment is discarded.

### D-IMPL-212: Make QSA FWHT256 composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_fwht256`. Share the standalone positive-row, overflow-safe byte
  extent, float-alignment and Context-view checks. Bind the in-place values
  at slot 0, `rows` at slot 1, and dispatch `ceil(rows/8) × 256` through the
  manifest-validated Encoder. Keep the standalone command-buffer API.
- **Reason:** O-class QSA layers apply FWHT256 to K, V, Q and attention
  outputs, sometimes before a dependent KV append or attention gate. A shared
  overload lets those ordered operations use one validated compute segment.
  The Mac runtime remains Metal-only; CUDA is numerical reference material.
- **Validation:** Focused `metal_qsa_test` passes; shared-Encoder fixtures at
  rows 1, 8, 9 and 192 match the scalar oracle bitwise and preserve the
  guards. Shared validation rejects zero rows, oversized extents, short and
  misaligned views, foreign Context buffers, and forged CPU/GPU metadata.
  A late-failure case encodes one valid FWHT dispatch, then rejects a short
  view and verifies the enclosing command buffer cannot be committed.
  The full build succeeds; all 72 CTests pass with the expected skip for
  unavailable MoE real inputs. `test_map.py` passes with the baseline planned
  task and REQ-MEM-04 notes, and `git diff --check` passes. Lagrange's hostile
  review found one P3 coverage gap, addressed below. The read-only follow-up
  confirmed closure with no new implementation or test issues.
- **Review required:** When the QSA layer coordinator is added, verify each
  O-class path transforms only the documented rows and views, orders FWHT
  before its consumer, and discards the complete segment on a later failure.

### D-IMPL-213: Make QSA indexer snapshots composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_indexer_snapshot`. Share stride conversion, overflow-safe
  extents, Context-owned view validation and source/output overlap checks
  between standalone and shared forms. Bind source tails, source block
  positions, packed tail output and packed block-position output at slots 0–3;
  bind the original element strides at bytes slot 4; dispatch one
  manifest-validated 384-thread group. Keep the standalone command-buffer API.
- **Reason:** A rollback-capable Window needs this snapshot before any
  per-layer indexer append, while commit later restores and replays from that
  packed state. The shared overload lets the coordinator include the
  snapshot in an Encoder segment. At this step append still had only its
  standalone API; D-IMPL-214 adds the missing shared form so snapshot and
  append can compose in one ordered segment. The Mac runtime remains
  Metal-only; CUDA is numerical reference material.
- **Validation:** The focused `metal_qsa_test` passes. Packed and padded
  stride fixtures match standalone and shared Encoder outputs bitwise,
  including signaling NaNs, subnormals, negative zero, block positions,
  untouched output padding and guards. Shared validation rejects undersized
  and zero strides, overflowing extents, undersized and misaligned views,
  source/output aliases, forged CPU/GPU metadata and foreign Context views.
  A late-failure case confirms the first snapshot dispatch encoded, then
  rejects a second invalid snapshot, preserves both output sentinels and
  refuses command-buffer commit as discarded. The full build succeeds; all
  72 CTests pass with the expected skip for unavailable MoE real inputs.
  `test_map.py` passes with the baseline planned-task and REQ-MEM-04 notes,
  and `git diff --check` passes. Cicero's hostile review found an
  overstatement about same-segment snapshot/append support and a late-failure
  assertion gap; the docs and assertion were corrected. Its follow-up found
  no new implementation or test issues and one stale validation status,
  corrected here.
- **Review required:** Confirm Window encodes exactly one snapshot before the
  first indexer append, uses the packed snapshots for commit replay, and
  propagates every shared-encode failure so the whole segment is discarded.
  Production Window/commit wiring remains open.

### D-IMPL-214: Compose QSA indexer append in a shared Encoder

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_indexer_append` and share its validation preparation with the
  standalone command-buffer form. Preserve the existing shader ABI: buffers
  at slots 0–9, `IndexerAppendParams` at bytes slot 10, and dynamic
  threadgroup memory of 16 and 512 bytes at slots 0 and 1. Keep the
  manifest-validated 1 × 128 dispatch. The shared overload supports ordered
  snapshot→window append→commit replay in one Encoder segment; callers must
  propagate a failed append from the `Encoder::encode` body, and dependent
  dispatches must not be placed in one concurrent group.
- **Reason:** The rollback protocol needs a snapshot, the speculative window
  append, and accepted-prefix restore/replay to operate on the same state in
  a known order. Reusing one validation path avoids different acceptance,
  extent, and alias rules between the existing standalone API and composed
  dispatch. Caller-owned segments allow this ordering without changing the
  kernel or its arithmetic. The Mac runtime remains Metal-only; CUDA is
  numerical reference material.
- **Validation:** Focused `metal_qsa_test` passes with default serial
  dispatch and with `STRATA_METAL_DISPATCH=concurrent`. Shared fixtures cover
  window mode and snapshot→window append→commit replay against the scalar
  state oracle. Shared validation rejects invalid rows/modes, noncontiguous
  steps, short views, missing commit snapshots, overlapping outputs, foreign
  Context buffers and forged CPU/GPU metadata. A late-failure fixture
  verifies that an already encoded append is discarded, leaves output
  sentinels untouched, and prevents command-buffer commit. The full build
  succeeds; all 72 CTests pass with the expected skip for unavailable MoE real
  inputs. `test_map.py` passes with the baseline planned-task and REQ-MEM-04
  notes, and `git diff --check` passes. Gibbs's hostile review found no
  actionable issue and ran no tests.
- **Review required:** Wire the shared overload into production Window and
  commit scheduling, verify the snapshot is encoded once before the first
  append and that commit uses its packed state, and preserve whole-segment
  discard on every encode failure. Keep dependent dispatches ordered through
  serial mode or Encoder barriers.

### D-IMPL-215: Make QSA selection scores composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_selection_scores` and route the standalone command-buffer
  overload through one shared validation/preparation path. Preserve bindings
  at buffer slots 0–4, `SelectionScoresParams` at bytes slot 5, and the two
  dynamic threadgroup-memory bindings for staged queries and `StepRow`s.
  Derive the same capped grid from validated StepRows and keep the existing
  `min(256, ceil((top + 1)/8)) × 256` dispatch.
- **Reason:** QSA selection consumes the scores immediately in top-k. The
  caller-owned Encoder form lets the coordinator compose scoring with
  selection in one segment while preserving standalone callers, exact
  validation rules and score arithmetic. D-IMPL-215 added score encoding;
  D-IMPL-216 adds the corresponding top-k overload.
- **Validation:** Focused `metal_qsa_selection_test` passes in default serial
  dispatch and with `STRATA_METAL_DISPATCH=concurrent`. Shared T=1/T=8
  fixtures match the ordered scalar oracle. Shared validation rejects invalid
  rows/capacity, short score views, inconsistent StepRows, output/input
  overlap, foreign Context buffers and forged CPU/GPU metadata. A late
  validation failure discards an earlier score dispatch, leaves the output
  sentinel unchanged and prevents command-buffer commit. The full build
  succeeds; all 73 CTests pass, with the expected skip for unavailable MoE
  real inputs. `test_map.py` passes with the baseline planned-task and
  REQ-MEM-04 notes, and `git diff --check` passes. Hooke's hostile review
  found no implementation defect; its concurrent-CTest finding was fixed
  and confirmed closed on follow-up.
- **Review required:** D-IMPL-216 completes shared top-k support. Before
  production selection is wired, verify score→top-k ordering, the per-row
  score suffix, and failure propagation across the whole segment.

### D-IMPL-216: Make QSA selection top-k composable

- **Decision:** Add a caller-owned-Encoder overload for
  `encode_qsa_selection_topk` and move the standalone path's validation and
  dispatch preparation into one helper. Keep register-form selection through
  33,792 max-blocks (1,024 threads and eight histogram banks) and memory-form
  selection above it (256 threads and one bank). In the shared overload,
  bind scores, steps and ids at buffer slots 0–2, params at bytes slot 3,
  histogram storage at threadgroup-memory slot 0 and the two scan arrays at
  slots 1 and 2; dispatch one group per token row.
- **Reason:** Selection scoring must feed top-k before either is consumed by
  attention. A shared overload permits scores→top-k in one ordered Encoder
  segment and keeps the existing standalone entry point while using the same
  validation contract. The caller must propagate a failed encode from the
  segment body so the earlier score/top-k dispatch is discarded.
- **Validation:** The target test binary builds. Focused serial and
  concurrent CTests pass 2/2. Fixtures check shared register and memory
  kernels against the sorted-cell oracle, including T=8 for both forms;
  same-segment score→top-k results; shared validation for invalid
  rows/capacity, short views, inconsistent StepRows, aliases, foreign Context
  and forged CPU/GPU metadata; and late failure that leaves IDs untouched and
  refuses command-buffer submission. The full build succeeds. All 73 CTests
  pass or have the expected unavailable-real-input skip (72 passed, one
  skipped); `test_map.py` exits successfully with the baseline planned-task
  and REQ-MEM-04 notes, and `git diff --check` passes. The generated CTest
  fixture directories were removed from `build-dev`, reducing it from 1.7 GB
  to 272 MB. The hostile-review findings were closed and its final follow-up
  found no remaining actionable issue. After the final test-only coverage
  additions, the focused serial/concurrent pair passes 2/2.
- **Review required:** Before production Window/Prefill wiring, verify the
  ordered score→top-k→sparse-attention chain in both dispatch modes, preserve
  untouched score/ID suffixes, and propagate every failure through the
  enclosing Encoder segment.

### D-IMPL-217: Compose sparse QSA attention chunk and merge

- **Decision:** Add caller-owned-Encoder overloads for sparse-attention chunk
  and merge, and route the existing standalone encoders through the same
  validation and dispatch-preparation helpers. Keep the established buffer
  slots, parameter layouts, grids and threadgroup-memory sizes. For shared
  chunk dispatch, bind only scale slots used by that mode's reflected
  pipeline; keep standalone fallback bindings for the existing entry points.
  Encode chunk before merge in one ordered segment, and make the caller
  propagate a late failure so `Encoder::encode` discards the segment.
- **Reason:** Chunk partials are consumed by merge, and selection IDs feed
  chunk attention. Shared overloads allow these dependent stages to compose
  without duplicating their validation. Metal reflection omits scale
  arguments unused by a specialized KV mode, so shared dispatch must match
  each pipeline's active resource slots exactly. Host validation still cannot
  inspect IDs produced by an earlier GPU dispatch; their live-prefix validity
  remains a producer precondition under REV-032.
- **Validation:** The full build and the final `metal_attn_test` target build
  succeed. Shared T=1 one-chunk and T=8 two-chunk fixtures for fp16, int8,
  q4_0 and k8v4 match the ordered scalar oracle. Shared validation rejects
  invalid rows/modes, short views, inconsistent StepRows, writable aliases,
  foreign Context buffers and forged CPU/GPU metadata. A late merge failure
  leaves output unchanged and prevents command-buffer commit. Focused serial
  and concurrent CTests pass 2/2. All 74 registered CTests pass or have the
  expected unavailable-real-input skip (73 passed, one skipped);
  `test_map.py` passes with the existing planned-task and REQ-MEM-04 notes,
  and `git diff --check` passes. The two uniquely named, ignored 758 MiB
  fixture trees created by CTest were removed from `build-dev`. Euler's
  read-only hostile review found no actionable implementation issue.
- **Review required:** Before production Window/Prefill wiring, validate the
  configured attention capacity and live IDs from the actual producer, keep
  chunk→merge ordered in serial and concurrent modes, and propagate failure
  through the full selection→attention segment. Retain REV-032 until
  model-backed configuration checks, production ordering and external parity
  are verified.

### D-IMPL-218: Compose the QSA attention gate

- **Decision:** Add a caller-owned-Encoder overload for `encode_qsa_gate` and
  route both overloads through the same extent, ownership, alignment and alias
  validation. Keep the existing buffer slots, parameter layout and
  `(N_HEAD_Q, token_rows, 1) × 256` dispatch. Cover shared T=1/T=8 execution
  for random, edge and fixture inputs, and exercise validation, read-only
  input overlap and late-failure discard. Scope Objective-C temporaries from
  `Context::make_command_buffer` to a local autorelease pool while the wrapper
  retains the returned command buffer.
- **Reason:** The gate output feeds sparse attention and must be composable
  with its producer in one ordered Encoder segment. Shared and standalone
  callers must keep the same memory contract. During testing, the extended
  QSA binary waited in AGX while creating a command buffer; adding the local
  autorelease pool made the full QSA fixture binary complete. The precise
  driver-level cause of that wait is unconfirmed. Keep the Mac gate on the
  documented safe-mode sigmoid approximation; this does not add Mac CUDA
  support or change CUDA builds.
- **Validation:** The full build succeeds. Direct
  `metal_qsa_test --input-set fixture` reports zero failures after the
  autorelease-pool change. The focused serial/concurrent QSA CTests pass 2/2.
  All 75 registered CTests pass or have the expected unavailable-real-input
  skip (74 passed, one skipped); `test_map.py` and `git diff --check` pass,
  with the existing planned-task and REQ-MEM-04 notes. The two uniquely named,
  ignored 758 MiB fixture trees created by CTest were removed from
  `build-dev`. Mendel's read-only hostile review found no actionable issue.
- **Review required:** Before production Window/Prefill wiring, verify the
  gate-to-selection-to-attention producer/consumer ordering in both dispatch
  modes, retain failure propagation through the complete segment, and test
  with model-backed inputs. Keep REV-032 open until the configured capacity,
  live IDs and external parity are verified.


### D-IMPL-219: Compose QSA selection and sparse attention

- **Decision:** Add `QsaSelectionAttentionBuffers` and
  `encode_qsa_selection_attention` to encode selection scores, top-k, sparse
  attention chunks and merge as one ordered Encoder chain. Preflight every
  component before encoding; require the fixed 2051-ID row stride and a cell
  capacity of at least 2051; enforce shared score, StepRow, ID and scratch
  views; and reject stage-output aliases with unrelated stage inputs or
  outputs. Expose whether an Encoder is inside an open concurrent dispatch
  group and reject this dependent chain there, where automatic buffer barriers
  are suppressed. Extend `metal_attn_test` to feed the GPU-produced top-k IDs
  directly into attention, then run FWHT for q4_0/k8v4 and the gate.
- **Reason:** Individual encoder twins did not prove that GPU selection output
  can safely feed the full attention path. A shared synthetic chain checks the
  producer/consumer views, dispatch order, fixed-width ID contract and
  quantized-mode post-processing in one command segment. It does not establish
  production Window/Prefill wiring, model-backed capacity choices, or external
  parity; keep those gates open under REV-032.
- **Validation:** The `metal_attn_test` target builds, and focused serial and
  concurrent CTests pass 2/2. Both T=1 and T=8 cases for fp16, int8, q4_0 and
  k8v4 match the independent selection score, sorted-ID, attention and gate
  oracles; q4_0/k8v4 include O-class FWHT. Read-only buffers and output guards
  pass. Preflight tests reject a non-fixed stride, a capacity below 2051, a
  mismatched ID view, a cross-stage score/query alias, and an open concurrent
  dispatch group. The full build succeeds and all 75 registered CTests pass
  or have the expected unavailable-real-input skip (74 passed, one skipped);
  `test_map.py` passes with the existing planned-task and REQ-MEM-04 notes,
  and `git diff --check` passes. The ignored MTP and WeightStore CTest fixture
  trees were removed after checking their generated paths, the MTP ownership
  marker, the nested WeightStore input ownership marker, and both ignore rules;
  `build-dev` is 276 MiB.
- **Review required:** Connect the coordinator from production Window/Prefill,
  confirm its capacity against the model configuration, and validate live IDs
  from the actual GPU producer. Keep REV-032 open until production ordering,
  configured capacity and external parity are verified.

### D-IMPL-220: Exercise QSA attention with short selection prefixes

- **Decision:** Extend the shared selection→attention T=8 fixture with
  increasing row positions whose live widths are 1, 2, 4, 5, 64, 65, 2048 and
  2051 while keeping the fixed 2051-ID row pitch. Compare the complete GPU
  score and ID arrays against independent oracles, including untouched IDs
  after each short live prefix, and feed the GPU-produced ID rows through
  attention, merge, mode-specific FWHT and gate for all four KV formats.
- **Reason:** The initial D219 integration fixture exercised only rows wider
  than the top-k limit. The varied-prefix fixture checks that `StepRow.width`
  controls the live IDs while `id_row_stride` controls storage and chunk
  geometry, including early positions, chunk boundaries and the fixed-width
  selection limit.
- **Validation:** The full build succeeds. Focused serial/concurrent
  attention CTests pass 2/2. The varied-prefix cases pass for fp16, int8,
  q4_0 and k8v4 against the score, ID, attention and gate oracles; complete ID
  equality confirms the untouched suffixes. All 75 registered CTests pass or
  have the expected unavailable-real-input skip (74 passed, one skipped), and
  `test_map.py` passes with its existing planned-task and REQ-MEM-04 notes.
  `git diff --check` passes. The ignored 758 MiB MTP and WeightStore CTest
  fixture trees were removed after checking their generated paths, the MTP
  ownership marker, the nested WeightStore input ownership marker, and both
  ignore rules.
  Poincare's read-only hostile review found no actionable issue.
- **Review required:** Keep this edge fixture in the CTest binary when the
  coordinator is connected to production. Production producer validity and
  model-backed capacity remain open under REV-032.

### D-IMPL-221: Allow QSA selection with a short cache capacity

- **Decision:** Keep the selection and attention ID row stride fixed at 2051,
  but allow the configured cache capacity to be smaller. Require only the
  component encoders' positive, four-cell-aligned capacity contract. Add a
  T=8 end-to-end shared-Encoder fixture with an eight-cell cache for all KV
  formats, while retaining the 2051-element ID rows. Replace the old rejection
  of capacities below 2051 with rejection of a non-interleaved capacity.
  Allow the attention ID stride to exceed the cache capacity; attention reads
  only the live prefix selected by `StepRow.width`.
- **Reason:** The planner and KV layout accept short contexts and round storage
  to a four-cell boundary. Selection and attention use `StepRow.width` to bound
  live cells and already size their pools, scores and KV reads from the actual
  cache capacity. A fixed maximum ID stride therefore does not require a cache
  with 2051 cells; the old coordinator and chunk-encoder checks rejected valid
  short-context configurations.
- **Validation:** The target and full project build succeed. Focused serial and
  concurrent `metal_attn_test` CTests pass 2/2; the eight-cell T=8 cases pass
  for fp16, int8, q4_0 and k8v4 against the score, ID, attention and gate
  oracles, including untouched ID suffixes. All 75 registered CTests pass or
  have the expected unavailable-real-input skip (74 passed, one skipped).
  `test_map.py` passes with its existing planned-task and REQ-MEM-04 notes, and
  `git diff --check` passes. The ignored 758 MiB MTP and WeightStore CTest
  fixture trees were removed after checking their generated paths, the MTP
  ownership marker, the nested WeightStore input ownership marker, and both
  ignore rules. Raman's read-only hostile review found no actionable issue.
- **Review required:** Keep the short-cache fixture when production Window and
  Prefill dispatch are connected. Validate the actual model capacity and
  producer-generated IDs under REV-032.

### D-IMPL-222: Add shared-Encoder QSA KV append paths

- **Decision:** Add caller-owned Encoder overloads for fp16, int8, q4_0 and
  k8v4 KV append. Keep one validation implementation for standalone and shared
  dispatches; the shared overload changes only where the validated kernel is
  encoded. Preserve each format's existing bindings and dynamic threadgroup
  memory. Compose both K/V FWHTs before q4_0 append and V-only FWHT before
  k8v4 append when callers use a shared segment.
- **Reason:** Window construction needs KV writes to share a command-encoder
  segment with preceding projections and transforms. Duplicating the large
  extent, position, alias and Context checks would let standalone and shared
  APIs drift. A shared preflight keeps their contracts identical while
  `Encoder::dispatch` owns manifest binding, barrier and segment-failure rules.
- **Validation:** `cmake --build build-dev -j 8` succeeds. The focused
  `metal_qsa_test` and `metal_qsa_test_concurrent` cases pass 2/2. Every KV mode
  runs T=1 and T=8 shared-Encoder fixtures for random, edge and fixture inputs
  against the existing exact scalar oracles. Q4 K/V FWHT and k8v4 V-only FWHT
  execute before append in the same segment; output bytes, untouched rows,
  input immutability and guards are checked. Q4 and K8V4 append reject an open
  concurrent group after their FWHT, and the test verifies the partial segment
  is discarded. `tools/metal/test_map.py` passes
  with `STRATA_BUILD_DIR=build-dev`; its existing planned model requirements
  and REQ-MEM-04 measurement checks remain outstanding. Full CTest passes 74
  tests with one expected model-input-unavailable skip (75 registered).
  `git diff --check` passes.
- **Review:** The first hostile review found a P2 API-contract gap: the new
  Q4/K8V4 overload comments omitted their FWHT requirement and the concurrent
  group ordering boundary. The comments now state the transform and barrier
  requirements. Both overloads reject an open concurrent group, and the
  concurrent test verifies the rejection discards the partial segment. A
  hostile follow-up found no remaining actionable issue. Production
  Window/Prefill dispatch and model-backed validation remain open under
  REV-032.

### D-IMPL-223: Compose sampler dispatches in shared Encoder segments

- **Decision:** Add caller-owned Encoder overloads for greedy, split-part,
  split-merge, composite sampled and coupled-penalty dispatches. Reuse the
  existing sampler shape, row, history, buffer, kernel and scratch validation.
  Reject dependent sampled and split-merge consumers while a concurrent
  dispatch group is open, because Encoder suppresses automatic buffer barriers
  inside a group. A failed shared callback must return false so Encoder discards
  all earlier work in the segment.
- **Reason:** The Window needs to place sampler work at the end of a decode
  segment. Shared APIs enable composition while keeping the standalone sampler
  kernels and bindings. Merge and penalty consumers require an explicit
  ordered boundary in concurrent dispatch mode.
- **Validation:** The sampler serial CTest, its concurrent-mode CTest and
  `samp_coupled_eq_host` pass 3/3. Shared greedy rejects undersized and
  misaligned ranges, overlapping views, forged backing metadata and foreign
  Context buffers, and verifies that each failed callback discards its segment.
  The coupled penalty and sampled composite now run in the same Encoder segment
  and compare against the post-penalty selection oracle. The concurrent test
  encodes split-part before rejecting merge and checks that the earlier dispatch
  is discarded. Shared composite cases cover 248,320 and 262,144 vocabulary
  entries, including the 32 KiB merge scratch. The full suite had 75 passes,
  one expected model-input-unavailable skip and zero failures. The full
  `build-dev` build, `test_map.py` and `git diff --check` pass. Original upstream
  fixtures and production Window integration remain open under REV-028/032.
- **Review required:** Production Window integration and complete upstream
  fixture parity remain open under REV-028/032.

### D-IMPL-224: Make the Prefill embedding coordinator composable

- **Decision:** Add a caller-owned `Encoder` overload for the Prefill embedding
  coordinator. Validate chunk metadata, CPU-visible token words and every
  required stage before appending text gather, optional image copy and GR
  broadcast through the shared encoder APIs. Reject calls inside an open
  concurrent dispatch group, where the broadcast dependency would lack an
  automatic barrier. Route the standalone command-buffer overload through
  `Encoder::encode`; the caller supplies a fresh command buffer, which the
  wrapper discards if validation or encoding fails.
- **Reason:** Later Prefill work needs to append to the same compute segment as
  embedding, while all three passes use untracked buffers and depend on ordered
  writes. The new overload preserves the coordinator's complete preflight and
  lets its caller discard earlier segment work if any stage fails.
- **Validation:** `metal_embed_test` exercises text-only and image chunks
  through standalone, individual shared-stage and shared-coordinator paths,
  comparing embedding and all four broadcast streams bitwise. Concurrent-mode
  tests reject an open group and verify that token-metadata failure discards a
  previously encoded broadcast. Existing enqueued-buffer and standalone
  failure checks remain active. Focused serial and concurrent tests pass 2/2.
  The full 76-test suite passes with one expected unavailable-model-input skip;
  `test_map.py`, the full build and `git diff --check` pass.
- **Review required:** Connect this coordinator from `Prefill::run`, compose
  subsequent Prefill kernels into the same segment and verify production
  producer dependencies under REV-029/031.

### D-IMPL-225: Separate repository identity from source provenance

- **Status:** Superseded by D-IMPL-227 and D-IMPL-239 for supported platforms and source-tree scope.

- **Decision:** Give the project its own Git history and repository identity. Preserve its Strata v0.1.34 source
  provenance, copyright notices and licenses; rewriting history does not make copied code clean-room work. Keep tracked
  engine, setup, test, fixture, audit and build-plan files required by the cross-platform product or Metal plan. Rename
  the prototype directory to `docs/mac/spikes/metal-build/mac-port/` and the setting to
  `setup.RELEASE_REPOSITORY`. Remove ignored `build-dev` outputs after implementation tests finish.
- **Reason:** The checkout has no configured remote or GitHub publication identity, so local history can be prepared
  but hosting metadata cannot be changed. The cleanup audit found no safe tracked-source deletion: platform code and
  fixtures remain necessary for supported systems and validation. Removing them would drop product support or test
  coverage. Only ignored generated `build-dev` outputs are cleanup candidates. Provenance avoids implying that copied
  source was independently authored.
- **Validation:** The Python runner passed all 42/42 modules; after the review fix, the focused drift-policy suite passed 15/15 and loads the production `drift.json`; release-policy tests passed 33/33; workflow tests passed 11/11; macOS path test passed 1/1; `test_map.py` and `git diff --check` passed. The full CTest run for D-IMPL-224 registered 76 tests, with 75 passing and one expected model-input skip on Mac17,9 (64 GB, macOS 27.0.1). Before rewriting the local `codex` history, verify every retained project tree, a linear history with no Strata commit ancestry, `main` unchanged, and a clean worktree. Remove ignored build outputs after the final implementation test run.
- **Review required:** Confirm the publication repository slug before configuring release URLs or changing remote/GitHub metadata. No remote is configured in this checkout. Keep the golden `no_arithmetic` list conservative; expanding it requires a numerical-impact review, a regression test and a recorded reason.
- **Adversarial-review resolution:** Descartes found that sections 09-11 described server/options/logging as exempt while `tools/metal/drift.json` exempted only `src/platform/**`. Preserve the narrower executable policy: only the platform glob is exempt today, and every other `src/` or `include/` change requires a new golden recording. Added direct verdict tests tied to the production drift configuration for platform, server, options, headers and non-engine paths; the compatibility procedure and risk summaries now state the same rule.

### D-IMPL-226: Validate the Prefill BF16 remainder against the device scale

- **Decision:** Compare the GR scale to the ordered CPU oracle within four F32 ULPs, compare the high BF16 image within one BF16 ULP, and verify the low remainder bitwise using the scale and high image written by the GPU.
- **Reason:** The documented `rsqrt` deviation is tiny in the normalized value but can move a near-zero BF16 remainder by many ULPs after subtraction. Using the device-produced scale/high value isolates the remainder arithmetic and RNE conversion without weakening the independent scale/high checks.
- **Review required:** Keep the scale and high-image comparisons independent. If the Metal math contract changes, re-evaluate the limits and the scale-conditioned remainder oracle together.

### D-IMPL-227: Support macOS Apple Silicon with Metal only

- **Status:** Supersedes D-IMPL-225's decision to retain Windows/Linux product paths.
- **Decision:** Make the repository's supported product, setup, build, CI, release and test paths macOS 27+ on Apple Silicon using Metal. Remove Windows/Linux launchers and product backends, CUDA/HIP build and installer paths, and their platform-specific packaging. Keep upstream CUDA code or captures only where a documented Metal oracle or provenance check still consumes them; do not present them as supported backends.
- **Reason:** The user explicitly chose a simple macOS-only repository and said Windows/Linux support can be removed. CUDA and HIP are not targets for Mac; Metal is the only product backend. Retaining product installers and CI for other platforms would contradict that scope.
- **Review complete:** D-IMPL-252 and REV-048 record the follow-up audit. Supported product, setup, build, CI, release and test paths are macOS-only; unsupported-host rejection checks and third-party portability inputs remain solely as tests or dependency code.

### D-IMPL-228: Remove non-Mac build graphs

- **Decision:** Make the root CMake project configure only on macOS and always include the Metal backend. Remove CUDA, HIP, x86 expert-kernel, and cross-platform build switches from the active CMake graph and presets. Keep shared artifact, planning, and host-test targets that are used by the Mac product or its tests.
- **Reason:** D-IMPL-227 selects macOS Apple Silicon as the only supported product. Keeping dormant backend branches in the primary build file makes the repository harder to understand and allows unsupported builds to drift.
- **Validation:** macOS `mac-dev` and `mac-release` configure/builds passed; host CTests passed 37/37 and Metal CTests passed 41/41; all 42 Python test modules passed; Linux, Windows and iOS cross-targets are rejected before compiler detection. Injected `GGML_CUDA=ON` and `GGML_HIP=ON` cache values are forced OFF in the ggml CPU oracle, and the empty `mac-model` preset fails closed as specified.
- **Review complete:** REV-048 records the audit of root targets, host gates, CI and project-owned runtime sources. No Windows/Linux product path remains supported; unsupported-host checks and third-party portability code remain as validation inputs.
- **Adversarial review and resolution:** Dirac found no missing engine link dependency. The review caught that an Apple non-macOS target could pass the platform guard and that caller cache values could enable CUDA/HIP in the test-only ggml oracle; the root now rejects non-Darwin targets before compiler detection, and the oracle forces both backends off. The review also noted that `mac-model` has no registered tests. Section 7.1 already specifies that this preset fails closed until those tests are implemented, so it remains; the README and contributor instructions now state that limitation instead of presenting the preset as a runnable model test. Removed CMake options were corrected in the active engine and setup instructions.

### D-IMPL-229: Restrict Mac CI to trusted Mac runs

- **Decision:** Run the optional CI job only on pushes to `main` and `release/**`, and run reference-hardware jobs only on `main`. Do not schedule CI for pull requests until a runner can safely execute untrusted code. Keep manual hardware dispatch restricted to `main` and validate its requested acceptance day before checkout.
- **Reason:** The repository is macOS-only and has no safe hosted PR runner configured. Cross-platform Linux jobs were removed with D-IMPL-227. The reference runner contains persistent model and result data, so PR branches must never execute there. Runner-group restrictions are still required because a manually selected ref can supply a different workflow definition.
- **Validation:** Workflow configuration tests passed 11/11 after the trigger and runner-condition changes; CMake host and Metal suites passed 37/37 and 41/41. No remote CI run was possible because this checkout has no Git remote.
- **Review required:** Configure the repository's trusted Mac runner and runner-group policy before enabling these workflows; the local YAML checks do not verify GitHub-side runner permissions.

## Adversarial review findings

- **2026-10-07, Descartes, Prefill GR normalization:** The reviewer found no concrete implementation flaw, but identified two proof gaps. Added a scale-separated 4096²-plus-31 fixture that proves the ordered butterfly differs from a linear sum by more than the scale tolerance, and made the late-failure test assert the command buffer's discarded state. The changes pass the targeted serial/concurrent CTests and are covered by the test-map check.
- **2026-10-07, Descartes, D-IMPL-225 hostile review:** Found three documentation issues: the README described independent history before the rewrite; prompt-source instructions read the baseline commit from project history; and D-IMPL-001 retained a stale first-sync review trigger. Follow-up review found that sections 09-11 described server/options/logging as golden-exempt while `drift.json` exempted only `src/platform/**`; the procedure step 4 and RSK-22 also said re-recording depended only on arithmetic changes. Kept the executable policy conservative, aligned the documents, and added direct golden-verdict regression tests. A further hostile pass found two remaining hardware-effort references tied only to arithmetic changes; both now follow the `golden: record again` path verdict. Final read-only verification confirmed the machine requirement, T-7 estimate, all three policy documents, checker configuration and production-config regression test agree. A final D225 documentation pass also caught an early claim that the pending history rewrite was already complete and a stale weekly upstream-review cadence; both were corrected and the reviewer confirmed them. No actionable D225 findings remain. The reviewer made no edits and ran no tests; local validation is recorded under D-IMPL-225.
- **2026-10-07, Descartes, D-IMPL-224 first hostile review:** No dispatch
  correctness finding. The reviewer identified two documentation issues: the
  header incorrectly said the standalone wrapper owns its caller-supplied
  command buffer, and the kernel inventory still described the coordinator as
  single-serial-encoder only. Both statements are corrected. The follow-up
  confirmed both fixes and found no actionable issue. The reviewer made no
  edits and ran no tests; validation is recorded under D-IMPL-224.
- **2026-10-07, Descartes, D-IMPL-224 hostile follow-up:** Confirmed that the
  header accurately describes the caller-supplied fresh command buffer and
  discard behavior, and that the kernel inventory reflects shared Encoder
  support with concurrent-mode barriers. No actionable issues remain. No tests
  were run; validation is recorded under D-IMPL-224.
- **2026-10-07, Descartes, D-IMPL-223 first hostile review:** No concrete
  dispatch correctness issue was found. The read-only reviewer identified four
  P2 coverage gaps: shared-path rejection tests, a same-segment
  penalty-to-sampled dependency, discard after a prior split-part dispatch,
  and shared merge scratch coverage at model/maximum vocabulary sizes. All four
  are now covered. The follow-up confirmed the fixes and found no remaining
  issues. Neither review ran tests; validation is recorded under D-IMPL-223.
- **2026-10-07, Descartes, D-IMPL-223 hostile follow-up:** The reviewer
  confirmed shared validation/discard coverage, same-segment penalty→sampled
  ordering, partial-segment discard and large-vocabulary scratch checks. No
  actionable issues remain. The reviewer ran no tests; validation is recorded
  under D-IMPL-223.
- **2026-10-07, Descartes, D-IMPL-222 hostile follow-up:** No remaining
  actionable issue. Q4/K8V4 document and enforce the concurrent-group boundary;
  the concurrent test verifies the append rejection and segment discard. The
  reviewer made no edits and ran no tests.
- **2026-10-07, Descartes, D-IMPL-222 first hostile review:** No blocking
  correctness, lifetime, or binding issue. It found a P2 API-contract gap:
  Q4/K8V4 shared overload comments omitted the required FWHT inputs and the
  concurrent-group ordering boundary. These were documented, enforced, and
  covered by a concurrent rejection test; the follow-up confirmed the fix.
  The reviewer made no edits and ran no tests.
- **2026-10-07, Raman, D-IMPL-221 short-cache QSA attention hostile review:**
  The read-only review checked short cache capacities with fixed-stride ID
  rows, live-prefix bounds, kernel masking, producer-consumer flow and stage
  preflight/alias checks. It found no actionable issue; the reviewer made no
  edits and ran no tests.
- **2026-10-07, Poincare, D-IMPL-220 varied-prefix QSA attention hostile review:**
  The read-only review checked the live-width fixture, untouched ID suffixes,
  and flow of GPU-produced IDs into attention. It found no actionable issue;
  the reviewer made no edits and ran no tests.
- **2026-10-07, Singer, D-IMPL-219 QSA selection-attention hostile review:**
  The first read-only pass found a P1 cross-stage output/read alias gap, a P2
  missing concurrent-group ordering precondition, and a P2 late capacity
  rejection. The coordinator now preflights global stage aliases, refuses an
  open concurrent group, and rejects capacity below 2051 before any dispatch.
  The follow-up confirmed all three fixes and found no new actionable issue.
  The reviewer made no edits and ran no tests; the focused serial/concurrent
  and full CTest suites passed independently.
- **2026-10-07, Euler, D-IMPL-217 sparse QSA attention hostile review:** The
  read-only review checked standalone/shared preparation parity, reflected
  bindings, mode-specific scale resources, dynamic threadgroup memory,
  dispatch ordering, failure discard, tests and docs. It found no actionable
  implementation issue. It noted the missing D217 decision entry; this entry
  now records the rationale and remaining review gate. The reviewer made no
  edits and ran no tests.
- **2026-10-07, Mendel, D-IMPL-218 QSA attention-gate hostile review:** The
  read-only review found no actionable implementation, test, CMake, map or
  documentation issue. It could not attribute the observed AGX semaphore
  wait to a D218 resource leak and could not establish the driver's root
  cause. The local autorelease-pool change was validated separately by the
  direct QSA fixture run and CTest; the reviewer made no edits and ran no
  tests.
- **2026-10-07, Erdos, D-IMPL-216 QSA selection top-k hostile review:** The
  first read-only pass found no kernel binding or validation defect. It
  identified missing shared-top-k documentation and a missing T=8 shared
  fixture; both register and memory forms now have T=8 oracle cases. The
  follow-up found the QSA overview still omitted the shared overload and the
  shared validation table overstated short-view coverage. The overview was
  corrected and shared short score, StepRow and ID cases were added. The
  D215 note was also updated to show that D216 completed its pending shared
  top-k work. Final follow-up confirmed these findings are closed and found
  no remaining actionable issue. The focused serial/concurrent CTests pass
  2/2; the reviewer made no edits and ran no tests.
- **2026-10-07, Hooke, D-IMPL-215 QSA selection-score hostile review:** The
  review found no implementation correctness issue. It identified a P3
  coverage gap: shared score dispatch had only been exercised in a manual
  concurrent run, with no persistent concurrent CTest. Added
  `metal_qsa_selection_test_concurrent` with
  `STRATA_METAL_DISPATCH=concurrent` and documented it. The registered
  serial/concurrent pair passes 2/2, and the full 73-test CTest run passes
  with the expected unavailable-real-input skip. The follow-up confirmed the
  P3 was closed and caught one inaccurate sentence implying barriers
  preserve GPU output before command submission; the docs now say results are
  checked after command-buffer completion. The reviewer made no edits and
  ran no tests.
- **2026-10-07, Gibbs, D-IMPL-214 QSA indexer append hostile review:** The
  read-only review checked shared/standalone validation parity, Metal buffer
  and parameter bindings, threadgroup memory, dispatch geometry, parameter
  lifetime, serial/concurrent ordering, and combined snapshot/window/
  commit plus late-failure tests. It found no actionable correctness issue.
  The reviewer made no edits and ran no tests.
- **2026-10-07, Cicero, D-IMPL-213 QSA indexer snapshot hostile review:** The
  initial review found a P2 documentation mismatch: indexer append is
  standalone, so the snapshot and append could not be placed in one shared
  Encoder segment as the docs implied. The docs now state this limitation and
  require caller-owned-Encoder support for append before claiming that
  composition. It also found a P3 test gap: the late-failure assertion did
  not prove the first snapshot dispatch had encoded. The test now records and
  requires that success before accepting the second dispatch's validation
  failure. The read-only follow-up confirmed both fixes and found one stale
  “Pending implementation and review” status, updated above; it found no
  further implementation or test issues. The final read-only follow-up
  confirmed the stale status was closed and found no new doc-record issue.
  The reviewer made no edits and ran no tests.
- **2026-10-07, Lagrange, D-IMPL-212 QSA FWHT256 hostile review:** The
  initial review found a P3 coverage gap: tests did not prove that a valid
  FWHT already encoded in a shared segment is discarded after a later
  validation failure. Added a segment that first encodes valid work and then
  rejects an undersized view; the test verifies failure propagation and that
  committing the command buffer is refused as discarded. Focused
  `metal_qsa_test` passes. The read-only follow-up confirmed the P3 is closed
  and found no new implementation or test issue. The reviewer made no edits
  and ran no tests.
- **2026-10-06, Dirac, QSA norm/RoPE shared Encoder hostile review:** The
  review found one P3 documentation issue: D-IMPL-211 still said full-suite
  validation and hostile review were pending. Both had completed; this entry
  now records the 72-test CTest result and review. The read-only review
  confirmed standalone/shared validation parity, mode-specific extents,
  position and alias checks, reflected bindings and threadgroup memory,
  manifest geometry, error propagation, and T=1/T=8 fixture coverage for all
  three modes. No implementation or test-coverage findings remain. The
  reviewer made no edits and ran no tests. The read-only follow-up confirmed
  the P3 is closed and found no stale or inaccurate documentation.
- **2026-10-06, Zeno the 3rd, D-IMPL-210 hostile review and follow-up:**
  The initial review found a P3 documentation omission: D-IMPL-210 was not
  yet recorded in this log. The decision, rationale, validation and remaining
  Window-integration review requirement were added. The read-only follow-up
  confirmed the record and found only stale future-tense review wording,
  which was corrected. No implementation findings remain. The reviewer made
  no edits and ran no tests.
- **2026-10-06, Maxwell the 3rd, D-IMPL-209 hostile review and follow-up:**
  The first review found a P3 diagnostic issue: all seven view-validation
  failures were labeled `state`. The shared overload now supplies the
  field-specific names and a regression check confirms that an undersized
  `h` view reports `GDN h buffer`. The read-only follow-up confirmed the fix
  and found no new issue. Reviewers ran no tests; validation is recorded under
  D-IMPL-209.
- **2026-10-06, Boyle the 3rd, D-IMPL-208 hostile review:** No P1-P3
  findings. Confirmed the T=1..8 bounds, checked sizes, BF16 alignment,
  Context checks, input/output alias rules, slot 7 parameter binding,
  12×256 grid, T=1/T=8 oracle and guard coverage, and failure propagation.
  The range test rejects the invalid slice as unowned before reaching the
  lower-level Metal allocation-boundary check; valid public views cannot
  carry that out-of-range geometry, so this was not considered actionable.
  Review was read-only and ran no tests; validation is recorded under
  D-IMPL-208.
- **2026-10-06, Ramanujan the 3rd, D-IMPL-207 hostile review:** No P1-P3
  findings. Confirmed the shared overload preserves T/range/alignment,
  ownership and writable-alias checks, allows read-only input overlap, matches
  reflected buffer slots 0-3, and propagates encoder failure. The reviewer
  confirmed the shader's unused row-count argument is omitted from reflection.
  Related docs match the current production-integration status. Review was
  read-only and ran no tests; validation is recorded under D-IMPL-207.
- **2026-10-06, Lovelace the 3rd, D-IMPL-206 hostile review:** Found a P3
  decision-log status error: focused serial/concurrent GR tests were marked
  pending despite having passed. Updated the validation entry with the
  focused results and completed full build, CTest, `test_map.py` and diff
  checks. Follow-up review confirmed the stale status is corrected and found
  no new documentation inconsistency. Reviewers made no edits and ran no
  tests; validation is recorded under D-IMPL-206.
- **2026-10-06, Mencius the 3rd, D-IMPL-205 hostile review:** Found a P3 API
  documentation gap: a shared GR-up validation failure must be returned from
  the enclosing `Encoder::encode` callback, or earlier encoded dispatches in
  the segment could still be submitted. Documented this propagation rule in
  the public header and kernel reference. Follow-up review confirmed the
  warning is explicit and found no new actionable inconsistencies. Reviewers
  made no edits and ran no tests; validation is recorded under D-IMPL-205.
- **2026-10-06, Sartre the 3rd, D-IMPL-204 hostile review:** No actionable
  P1-P3 findings. Confirmed the count is bound at slot 1, the dispatch grid
  matches the T × 320 manifest contract, the in-place view checks extent,
  alignment, range and Context ownership, and the shared T=1/T=8 fixtures
  cover oracle parity, the partial-group tail and guards. Documentation
  matches the implementation. Review was read-only; validation is recorded
  under D-IMPL-204.
- **2026-10-06, Faraday the 3rd, D-IMPL-203 hostile review and follow-up:**
  The initial review found a P3 documentation gap: the §10.5 narrative listed
  shared GR cvec, norm and post fixtures but omitted the new pre-gated T=1/T=8
  fixture. Added its U(16) oracle, guard and alias coverage. The follow-up
  confirmed the omission is closed, the test list agrees, and no new P1-P3
  documentation inconsistencies remain. Reviewers made no edits and ran no
  tests; validation is recorded under D-IMPL-203.
- **2026-10-06, Galileo the 3rd, D-IMPL-202 hostile review:** No actionable
  P1-P3 findings. Confirmed the shader-matching bindings and parameter block,
  `(HC, rows)` dispatch geometry, T=1/T=8 project/add and write/no-write
  fixtures with scalar oracles and guards, shared rejection coverage, and
  decision/documentation consistency.
- **2026-10-06, Galileo the 3rd, D-IMPL-201 hostile review and follow-up:**
  The initial review found missing T>1 shared-Encoder geometry coverage,
  incomplete shared-path rejection cases and a duplicated word in the kernel
  overview. Added T=1 and T=8 apply-on/off oracle cases, shared-path checks for
  all six short views, misalignment, out-of-range, foreign ownership and exact
  and partial output aliases; corrected the documentation. The follow-up
  confirmed these changes and found no remaining P1-P3 issues.
- **2026-10-06, Arendt the 3rd, D-IMPL-200 hostile review:** No P1-P3
  findings. The shared Encoder overload preserves the standalone validation
  and alias contract; the fixture checks its scalar oracle and guards; serial
  and concurrent CTest registration, test map, and documentation agree.
- **2026-10-06, Archimedes the 3rd, D-IMPL-199 hostile review:** No
  actionable P1-P3 findings. Confirmed the new shared-Encoder buffer
  bindings and manifest geometry, preservation of standalone validation
  behavior, bitwise embedding/broadcast oracles and output guards, barriers
  between dependent dispatches in concurrent mode, and the serial/concurrent
  CTest and test-map entries. The reviewer made no edits and ran no tests;
  focused and full validation are recorded under D-IMPL-199.
- **2026-10-06, D-IMPL-198 hostile review and follow-ups:** Leibniz the 3rd
  found that the T=2 Q8_1 fixture copied the complete four-column oracle into
  a two-row allocation. Fixed the copy extent to live rows and moved the
  guard snapshot before host initialization so fixture writes are checked
  too. Curie the 3rd confirmed the bounds and guard fix, then found that
  inherited `STRATA_METAL_DISPATCH=concurrent` could make the nominal serial
  CTest case run concurrently. Pinned the serial CTest environment. Mill the
  3rd's final read-only review found no P1-P3 issues and confirmed alias
  validation, failure discard, dispatch-mode registration and docs/test-map
  counts. Reviewers made no edits and ran no tests; the final focused and
  full-suite results are recorded under D-IMPL-198.
- **2026-10-06, Laplace the 3rd, D-IMPL-197 hostile review and follow-up:**
  The production encoder overloads preserve existing validation and bindings;
  the first review found no implementation defect. It identified three test
  gaps: CTest did not force the dependent chain through concurrent mode, the
  shared BF16 path lacked the 160-thread specialization, and the late-failure
  fixture checked only the non-null return from `memcpy` instead of the
  restored row value. Added explicit serial/concurrent CTest cases, nonzero
  scalar-oracle shared BF16 coverage for both 256- and 160-thread variants,
  and a read-back check that the discarded encode leaves intentionally
  invalid active-row metadata unchanged. Follow-up confirmed each fix and
  found no actionable P1-P3 issue or regression. The reviewer made no edits
  and ran no tests; the focused and full test results are recorded under
  D-IMPL-197.
- **2026-10-06, Pascal the 3rd, D-IMPL-196 hostile review:** No actionable
  findings. Confirmed that invalid history policies are rejected and
  propagated through the standalone wrapper, `defer` preserves committed
  history while the T=1/2/8 commit fixtures update only the accepted prefix,
  and the alias, poison and guard checks match the documented buffer
  lifetimes. The review also found the `04`, `06` and `09` descriptions
  consistent. The reviewer ran no tests; validation is recorded under
  D-IMPL-196.
- **2026-10-06, Euclid the 3rd, D-IMPL-195 hostile review and follow-ups:**
  The first follow-up found that the new coordinator's exact
  `query`/`ple_norm` alias success path and T>1 composition were not directly
  covered. Added successful coordinator runs at T=1,2,8, including the exact
  alias, with row-indexed norm/gate/broadcast/convolution/history oracles. A
  second follow-up found that prior correct outputs could mask missing writes
  by the coordinator; the test now re-poisons its scratch, gate and broadcast
  outputs immediately before the exact-alias call. The final read-only review
  confirmed the alias and multirow coverage, the poison catches stale output,
  and found no remaining actionable issue. The reviewer ran no tests;
  validation is recorded under D-IMPL-195.
- **2026-10-06, Euclid the 3rd, D-IMPL-194 hostile review and follow-up:**
  The first pass found three P3 issues: the reused-query chain did not inspect
  its full query norm directly, the test catalogs omitted
  `strata_rms_norm_w`, and the oracle wording could imply that initial key and
  query references came from GPU outputs. Added a test-only
  `strata_selftest_copy_u32` dispatch before overwrite, then checked all
  10,240 values against U(64) and bitwise against the reference chain; listed
  the weighted norm in both catalogs; and clarified that only downstream
  oracles consume prior GPU outputs. The read-only follow-up confirmed all
  three findings closed and found no new actionable issue. It also caught the
  stale “review pending” status, which this entry and the validation record
  correct. The reviewer ran no tests; validation is recorded under
  D-IMPL-194.
- **2026-10-06, Mendel the 3rd, D-IMPL-193 hostile review:** No actionable
  findings or false-positive risks. Confirmed the second output is poisoned
  before dispatch and independently checked against the oracle, and that this
  change does not expand CUDA support on macOS: CMake rejects CUDA on Darwin
  and D-IMPL-179 records the Metal-only Mac boundary. The reviewer ran no
  tests; validation is recorded under D-IMPL-193.
- **2026-10-06, Epicurus the 3rd, D-IMPL-192 hostile review:** Found a P3:
  the shared-segment history test used the same n twice on separate histories,
  so it did not verify changing the inline n binding while advancing one
  history in order. The fixture now dispatches n=1 then n=2 against the same
  history using consecutive norm slices, followed by n=1 against a second
  history; independent bitwise oracles verify the three-row and one-row
  results. The read-only follow-up confirmed the P3 is closed and found no
  other actionable issue. The reviewer ran no tests; validation is recorded
  under D-IMPL-192.
- **2026-10-06, Meitner the 3rd, D-IMPL-191 hostile review and follow-up:**
  The review found that the test-catalog wording could imply bitwise
  comparison against the oracle, while the test uses U(16) tolerance for each
  oracle comparison and bitwise comparison only between the two GPU outputs.
  Updated both the test catalog and decision record to state those checks
  separately. The read-only follow-up confirmed the P3 is closed; no other
  actionable issue was reported. The reviewer ran no tests; validation is
  recorded under D-IMPL-191.
- **2026-10-06, Hegel the 3rd, D-IMPL-190 hostile review:** The first pass
  found that both standalone PLE wrappers inspected command-buffer status
  before `Encoder::encode` acquired the command-buffer encoding lock. Removed
  those unsynchronized preflight calls; the shared encoder now checks
  ownership and submission state through `encode_unsubmitted` while holding
  the per-buffer lock.
- **2026-10-06, James the 3rd, D-IMPL-190 follow-up reviews:** The first
  follow-up confirmed the synchronization fix and found a P3: a
  simultaneous-start wrapper/enqueue test did not guarantee the problematic
  interleaving. Removed the timing-dependent test; deterministic wrapper
  rejection cases and the existing locked encode/enqueue serialization test
  cover the lifecycle boundary. The final read-only review found no actionable
  issues. Reviewers ran no tests; validation is recorded under D-IMPL-190.
- **2026-10-06, Jason the 3rd, D-IMPL-189 hostile review:** No actionable
  findings. The review confirmed the standalone wrapper revalidates within
  `Encoder::encode`, the shared-segment overload preserves typed buffer and
  alias checks, and both routes use the common dispatch facade. Review was
  read-only and ran no tests; focused validation is recorded under D-IMPL-189.
- **2026-10-06, Averroes the 3rd, D-IMPL-188 hostile review and follow-up:**
  The first pass found that the fixture reused only sixteen row markers and
  repeated each marker across all IQ4_NL columns, allowing row collisions and
  hiding within-row column-order bugs. The fixture now assigns a unique
  representable FP16 scale to every row and varies IQ4_NL codes by column;
  the oracle uses a test-local codebook and checks each decoded column. The
  follow-up confirmed that the P3 is closed and found no additional
  actionable issue. Review was read-only and ran no tests; the focused and
  full validation results are recorded under D-IMPL-188.
- **2026-10-06, Averroes the 3rd, D-IMPL-187 hostile review:** The first
  pass found that returning mutable `PleTable*` let callers reopen the
  storage-owned table in Direct mode or close it while the GGUF mapping was
  retained. Replaced it with a non-copyable row-access-only `PleTableReader`
  view that exposes no lifecycle methods; a compile-time test guards that
  boundary. The follow-up confirmed the P2 is closed and found no additional
  actionable issue. Review was read-only and ran no tests.
- **2026-10-06, Dalton the 2nd, D-IMPL-186 hostile review and follow-up:**
  The first review found that mapped FP8 PLE opened its payload shard and then
  looked for format/scale metadata there; split GGUF stores model metadata in
  shard 1. The reader now validates FP8 metadata from `GgufModel::meta()`, and
  a two-shard fixture keeps those keys only in the first shard while putting
  the PLE tensor in the second. The follow-up confirmed that fix, then found
  that range validation happened after narrowing the scale to float. The
  reader now checks the finite positive double against float's representable
  positive range before conversion; F64 overflow and underflow fixtures cover
  both cases. A further review found signed integer metadata could appear
  positive through `MetaValue::num()`'s unsigned conversion. The reader now
  interprets signed integer scales without losing their sign, and a negative
  I32 fixture verifies rejection. The final re-review found no additional
  actionable issues. Reviewers ran no tests. The full host/Metal suite passed
  68/68 before the signed-scale guard; after that guard, `metal_weight_store_test`
  and `ple_reader_selftest` passed 2/2.

- **2026-10-06, Jason the 2nd and Bohr the 2nd, WindowScratchLayout hostile
  review and follow-ups:** The first pass found that `head_inject` was not
  consumed by the documented head-mixer ABI; its region and append were
  removed, reducing the aligned T=8 allocation by 256 B. The follow-up found
  that the published totals did not distinguish 256- and 512-expert models;
  all six capacity/model combinations are now pinned in the test and listed
  in the memory table. It also found an unused `head_inj_` member and 16 B
  carve in the CUDA verifier; repository search found only its declaration
  and carve, so both were removed. The final follow-up found that the test
  catalogue overstated rejection coverage; its description now distinguishes
  unsupported geometry/capacity rejection from the checked overlap/bounds
  invariants. Reviewers made no edits and ran no tests. The full host/Metal
  CTest selection passes 66/66, with the unavailable real-input MoE case
  skipped. `test_map.py` exits successfully; its existing planned tests and
  missing requirement checks remain reported as notes.

- **2026-10-06, Jason the 2nd, WindowScratchStorage hostile review and
  follow-ups:** The first pass found a P2 escape hatch: a public accessor
  returned the full-arena `BufferRef`, bypassing named-region bounds. It was
  removed; storage now publishes only generation-checked views matching exact
  named regions. The review also found poison documentation that claimed all
  unwritten reads become NaN. Sections 04, 05 and 06 now state that allocation
  fills bytes with `0xFF`, floating-point reads become NaN, integer/byte reads
  are all ones, and zero-state owners clear their regions afterward. The final
  follow-up confirmed the access restriction and found no stale Metal poison
  claims. Review was read-only and ran no tests; validation is recorded under
  D-IMPL-183.

- **2026-10-06, Mendel the 2nd, full-mode tensor-view hostile review and
  follow-up:** The first review found a P3 coverage gap because only the first
  converted BF16 row was tested. A two-row fixture now checks a gate router
  and a shared-expert router at a nonzero arena offset, including metadata,
  CPU/GPU address deltas and every byte. The follow-up found no remaining
  actionable issue. The review was read-only and ran no tests.

- **2026-10-06, Pasteur the 2nd, SessionStateStorage hostile review:** The
  review found a reset/use race, an insufficient model-ABI guard, owner
  construction after transaction commit, missing outer-rollback coverage and
  inconsistent L9/test-map attribution. Reset now refuses while registered
  command buffers remain pending and receives completion notifications on
  completion or discard; the geometry guard checks qwen4exp and every
  applicable documented architecture dimension; the owner is created before
  transaction commit; and rollback tests invalidate allocation, views and
  reset. L9 dependencies and test attribution now name this as a prerequisite.
  The review also questioned the 256 MiB fixed allowance. The 50 MB pipeline
  estimate is explicitly assigned to the separate default 700 MiB working-set
  reserve; the approximate GDN/PLE, T=8 scratch and other fixed-buffer subtotal
  is below 256 MiB. Exact reconciliation remains open until the Window scratch
  layout exists. The initial reviewer ran no tests and made no edits. It also
  found a P2: an exception from completion-handler registration could leave
  the tracked-use count elevated. A scope guard now rolls back the count on
  both false returns and exceptions, and `metal_session_state_test` checks the
  false-return path. The hostile follow-up confirmed the guard covers the
  exception path by inspection; no exception-injection test was added. Focused
  `metal_session_state_test` and `metal_events_test` pass 2/2, and the complete
  host/Metal suite passes 65/65 with the unavailable real-input MoE case
  skipped. `test_map.py` exits successfully; its existing planned tests and
  missing requirement checks remain reported as notes. The follow-up ran no
  tests and made no edits.

- **2026-10-06, Averroes the 2nd, WindowParams ring hostile review:** The
  read-only review found no actionable findings across allocation lifetime,
  transaction rollback, fixed slot layout, caller-owned GPU synchronization,
  input bounds, tests and documentation. The review ran no tests; validation
  is recorded under D-IMPL-177.

- **2026-10-06, Averroes the 2nd, main RoPE binding hostile review and
  follow-up:** The first pass found that validation computed a nonzero
  `base + offset` even when Metal's GPU address was unavailable and correctly
  represented as zero by `BufferRef::slice`; it now uses the zero-aware
  `detail::gpu_address_with_offset` helper. It also requested a live
  context-mismatched KV regression in addition to stale-view coverage; the
  test now proves rejection without additional MTP allocations. The
  follow-up found no remaining actionable issues. Review was read-only and
  ran no tests; validation is recorded under D-IMPL-176.

- **2026-10-06, Banach the 2nd, MTP storage-composition hostile reviews:** The
  design review required exact inspected MTP sources,
  `validate_mtp_memory_plan()` before allocation, one outer transaction for
  both owners, and composed rollback coverage. `ModelStorage` validates and
  opens the MTP owner before Full/Tiered storage inside that transaction;
  `--storage-check` passes both source objects from one `WeightStore::inspect()`
  call. The synthetic fixture checks a later model-storage failure rolls back
  MTP allocations and releases the generation lock, then verifies a successful
  aggregate owner. Its CLI cases cover speculative Full and Tiered storage
  plus non-speculative storage with `--mtp` supplied. The follow-up
  implementation review found one P3 documentation conflict: D-IMPL-169 and
  D-IMPL-170 still described the owner as separate from `ModelStorage`. Those
  decisions now mark that separation as historical for storage diagnostics
  while keeping production main-KV/session composition open. D-IMPL-175 later
  adds the main KV allocation to storage diagnostics, and D-IMPL-176 binds
  speculative MTP storage to its borrowed RoPE view. This remains storage-only:
  K/V and RoPE values, session state, PLE and dispatch remain outside these
  changes. After the documentation correction, the follow-up review found no
  remaining actionable findings.

- **2026-10-06, Averroes the 2nd, main QSA KV composition hostile reviews:**
  The design review required exact `kv_bytes` validation, main-KV-first
  allocation inside the outer transaction, a bounded shared RoPE view and
  rollback coverage. The first implementation review found that the device's
  `maxBufferLength` refusal lacked a deterministic `ModelStorage::open` test
  and that rollback coverage was attributed too broadly. Added a
  hardware-clamped `ContextOptions::max_buffer_length_cap` that reaches both
  `DeviceFacts` and `Buffers`, then tested a valid speculative plan with a cap
  one byte below its KV allocation; `ModelStorage::open` returns its preflight
  error with zero allocations. Full and Tiered later-open rollback are both
  covered, with MTP aggregate rollback in the MTP storage test. The read-only
  final follow-up found no actionable safety or API issues and ran no tests.

- **2026-10-06, Kepler the 2nd, MTP KV allocator hostile review:** The first
  pass found that a store surviving an enclosing transaction rollback could
  still return its stale `BufferRef`; a follow-up noted that a new allocation
  could reuse the same raw handle. `Buffers` now assigns a monotonically
  increasing identity to every owned allocation, and `MtpKvStore` checks both
  handle and identity before exposing it. The regression allocates a new
  buffer after rollback and confirms the stale store remains invalid. Failed
  view lookup also clears its outputs. The review identified stale bootstrap
  counts; the docs now distinguish the 28-test bootstrap subset from the
  current 34-host/27-Metal preset selection. The requirement map remains
  intentionally incomplete for REQ-FUN-07 as a whole: its
  `accept:C2.d.accept` item has no registered criterion definition, and the
  other checks exercise draft behavior beyond this allocator. The new test is
  documented as one piece of evidence without marking the full requirement
  complete. The final hostile follow-up found no remaining findings.

- **2026-10-06, Confucius the 2nd, MTP fixture hostile review and
  follow-ups:** The first pass found a P2 race where a same-name empty
  directory could be substituted between `mkdir` and the first open. The
  helper now accepts only current-user-owned parents that are not group/world
  writable and have no extended ACL; the opened fixture must also be
  current-user-owned, mode `0700`, and have no extended ACL. A follow-up found
  that mode-bit checks alone do not reject ACL grants. The helper now checks
  `ACL_TYPE_EXTENDED` through `acl_get_fd_np` and fails closed if ACL state
  cannot be read. Final regressions exercise the actual creation path: an ACL
  parent is refused before creating an entry, and an ACL added immediately
  after the fixture FD is opened is refused before writing the owner marker;
  the fixture FD must close. The final read-only review found no P1/P2/P3
  issue. Reviewers ran no tests; the same-UID `mkdir`/`open` race remains an
  explicit trust boundary under D-IMPL-167.

- **2026-10-06, Ampere the 2nd, MTP KV sizing audit:** The read-only audit
  found that the documented 2,080/1,088/608 B-per-cell values did not match
  the K/V row formats, while upstream's generic drafter state allocates
  128-float pooled rows that Metal dense attention does not consume. The
  device-free MTP layout and planner now account for only the K/V and scale
  regions used by the Metal drafter, with k8v4 mapped to int8. The audit ran
  no tests; production encoder confirmation remains open under D-IMPL-168.

- **2026-10-06, Ohm the 2nd, MTP KV layout hostile review:** The review found
  that the nearby context-size table displayed main-model KV bytes without
  identifying them as such, the D-IMPL-168 review criterion mischaracterized
  the shared RoPE table as unused, and the test map omitted the new MTP
  coverage origin while attributing planner comparison to the wrong test.
  Labeled the table as main-model values, documented borrowing the main
  `KvStore` RoPE table, and corrected the test descriptions and registry.
  Arithmetic, overflow checks, format mapping, disabled-spec behavior and
  CMake registration had no findings. A read-only follow-up found two stale
  references: REV-033 could imply that shared RoPE is unused, and the build
  guide omitted MTP KV layout coverage. REV-033 now distinguishes the omitted
  MTP-owned copy from the borrowed main table, and the build-guide inventory
  includes the MTP layout. The final hostile follow-up found no remaining
  actionable P1/P2/P3 issues. Reviewers ran no tests.

- **2026-10-05, Darwin the 2nd, converted-router storage hostile review and
  follow-up:** The first review found that a caller could change a converted
  row's source offset to another valid in-bounds range while retaining the
  inspection lock, and that router block numbers were not bounded by model
  geometry. WeightStore now binds the parsed row inventory to its descriptor
  lease; full and tiered storage compare supplied rows against that immutable
  catalog. Canonical router layer IDs are checked against layer count in both
  modes. Added regressions for same-size offset remapping, out-of-range layers,
  overlapping source ranges and malformed owner input; corrected stale status
  text in the docs. Follow-up review confirmed the fixes, and Aquinas the 2nd
  found no additional actionable issue in the final diff. Reviewers ran no
  tests; the full host/Metal CTest passed 58/58 with one unavailable
  model-input test skipped.

- **2026-10-05, Aquinas the 2nd, MTP inventory hostile review and
  follow-ups:** The first pass found that failure could leave a partially
  published catalog, the vocabulary-order test used sorted IDs, and middle
  dense rows were not fully asserted. The output is now held locally until all
  validation passes; tests use non-monotonic IDs and compare all 29 rows. A
  follow-up found that a retargeted `mtp_dir` symlink could make the retained
  lock and catalog paths refer to different directories. Inspection now
  canonicalizes the directory once for locking, marker checks, parsing and
  catalog paths, with a symlink-retarget regression. The review also caught a
  fixture reset before its forged-catalog assertion; the reset now follows the
  assertion. The final read-only follow-up found no remaining actionable
  issue. Reviewers ran no tests.

- **2026-10-05, Hooke the 2nd, MTP window-layout hostile review and
  follow-up:** The first review found a P3 test gap: the failure-output check
  started with an empty result and would not catch stale windows after a prior
  success. The test now seeds the output from a valid layout before forcing a
  too-small `maxBufferLength` failure, then checks that all window and binding
  vectors and the mapped-byte count are cleared. The implementation already
  reset output at entry. Review covered page alignment, eight-blob expert
  boundaries, dense-row mapping, bounds and generation-lock matching. The
  final follow-up confirmed every layout field is covered and found no
  remaining actionable issue. Reviewers ran no tests.

- **2026-10-05, Halley the 2nd, sparse-attention hostile review and
  follow-up:** The first pass found that negative, out-of-capacity and
  future-cell IDs could be converted to cache addresses without validation.
  The chunk kernel now maps these IDs to an invalid sentinel and excludes them
  before key/value loads or softmax; tests cover each invalid class, mixed
  valid/invalid rows and all-invalid chunks returning exact positive zero in
  every KV mode. The review also found stale L6 status text and an inaccurate
  statement that the residency mask was the only cell mask; both docs were
  corrected. Final read-only review found no remaining actionable issue.
  Reviewers ran no tests; validation is recorded under D-IMPL-156.

- **2026-10-05, Kant the 2nd, QSA state layout hostile review and follow-up:**
  The first pass found a P2 undercount: main KV rounded capacity to four cells,
  but speculative MTP KV still used raw `max_context`. The planner now sizes
  both from the rounded main layout capacity; docs separate spare allocation
  from logical `StepRow.n_kv`, and `metal_plan_test` checks context 65 in all
  four modes with speculation enabled. Follow-up confirmed the finding is
  fixed and found no new actionable P1/P2 issue. The review was read-only and
  ran no tests; validation is recorded under D-IMPL-157.

- **2026-10-05, Cicero the 2nd, next-step planning audit:** The audit found
  that L9 is not a safe immediate integration target: its full decode path
  depends on unfinished production dispatch work and requires one shared
  serial compute encoder, while current per-kernel encoders open their own.
  It recommended connecting the already-tested U3/U4 full-mode storage
  components through a production startup owner first, without claiming
  `READY` or inference support. The audit also identified that MTP runtime
  residency and converted Swift routers need explicit startup handling
  (REV-017). The audit was read-only and ran no tests.

- **2026-10-05, Noether the 2nd, WindowParams hostile review:** The first
  review found that `WindowSampRow` defaults (`top_p=1`, `min_keep=1`,
  `greedy=1`) leaked into unused fixed-capacity rows, contrary to the block
  contract. The ABI mirror now leaves sampler members without nonzero member
  defaults so value initialization produces all-zero inactive rows; the host
  test checks every field of every inactive sampler row. The follow-up review
  confirmed the issue is resolved and found no new actionable issue. The
  reviewer was read-only and ran no tests; validation is recorded under
  D-IMPL-158.

- **2026-10-05, Poincare the 2nd, K8V4 append hostile review and
  follow-ups:** The first pass found that V's in-place FWHT could corrupt
  aliased K or `StepRow` inputs, the finite fp16-scale contract was missing,
  K8V4 had no dedicated twin, and the three-region output layout was
  ambiguous in the docs. The encoder now rejects pairwise input overlap,
  including distinct no-copy handles; the API/docs state finite stored-scale
  and q4-reciprocal preconditions; tests chain V-only FWHT and append in one
  command buffer, compare raw K and rotated V against independent scalar
  oracles, and exercise the largest finite fp16 scales; the layout docs
  identify K codes, K scales and V blocks as the three regions. A second pass
  found that zero-initialized expected buffers could miss stray zero writes
  to untouched rows. Expected K codes/scales and V blocks now start with
  nonzero sentinels; the final read-only follow-up found no remaining
  actionable issue. Focused normal and Metal API/shader-validation CTests and
  the full host/Metal CTest passed after the fixes. Reviewers ran no tests.
- **2026-10-05, Euclid the 2nd, QSA indexer snapshot hostile review and
  follow-up:** The first pass found P2 coverage gaps for aliases in later
  strided layers and across separate no-copy handles, and for short, misaligned
  or out-of-allocation views among the four buffers. It also found a P3
  ambiguity about snapshotting write-through windows. Added exact and partial
  alias cases in layer 11 including shared backing across no-copy handles;
  invalid extent/alignment/allocation cases for each view; and clarified that
  orchestration snapshots only rollback-capable windows. The follow-up found
  no remaining actionable issue. Reviewers were read-only and ran no tests;
  validation is recorded under D-IMPL-154.

- **2026-10-05, Avicenna the 2nd, QSA indexer append hostile review and
  follow-up:** The first pass found two P2 gaps: commit tail and
  block-position snapshots could overlap, and functional commit cases selected
  only QSA layer 11. The encoder now rejects exact or partial snapshot overlap
  while accepting disjoint subviews in one allocation; validation covers all
  three cases. Commit replay tests select layers 0, 5 and 11 with
  per-layer-distinct state. Follow-up confirmed both findings resolved and
  found no other actionable issue. Reviewers did not edit files or run tests;
  validation is recorded under D-IMPL-155.

- **2026-10-05, McClintock the 2nd, GDN hostile review and follow-ups:** The
  initial read-only review found two P2 issues: the softplus oracle's
  accumulated magnitude did not follow §10.2's library-function rule, and the
  edge inputs lacked all-zero buffers and a 2^10 row-amax outlier. The oracle
  now resets magnitude to `|f(value)|` in both softplus branches, and GDN tests
  explicitly exercise all-zero and exact-ratio outlier inputs across the
  kernels. Follow-up confirmed both findings resolved and found no other
  actionable issue. Reviewers ran no tests.
- **2026-10-05, Planck the 2nd, indexed draft-head encoder hostile review and
  follow-up:** The first pass found a P2 test-coverage gap: no direct indexed
  API checks for foreign row-map ownership, forged CPU/GPU metadata, or
  foreign/enqueued/committed command buffers. Those rejection cases were added
  and passed in the focused Metal test. The follow-up confirmed the P2 was
  resolved, reviewed row-map bounds, flattened indexing, byte extents, aliases,
  kernel bindings and unsorted-map oracle coverage, and found no new actionable
  issue. The reviewer was read-only and ran no tests.
- **2026-10-05, Boyle the 2nd, Q8_1/mat-vec encoder hostile review and
  follow-ups:** The first pass found no encoder safety or arithmetic defect,
  but identified missing nonzero/large-K/BF16-256 and buffer-contract coverage.
  Follow-ups also required nonzero engine-oracle cases for every quantized
  format, the remaining output/sink alias predicates, and distinct-handle
  backing overlap. Tests now cover all eleven quantized formats plus BF16,
  NC=1/2/4/8, Q4_0 R=1, BF16's 256-thread path, same- and cross-handle
  overlapping views, every output/sink versus read-buffer class, and adjacent
  Q8_1 views. The final read-only review found no actionable P1/P2/P3 issue
  and ran no tests. The complete build succeeded; focused CTest passed 1/1
  normally and 1/1 with Metal shader validation/debug enabled; the full
  host/Metal suite passed 49 tests with one expected model-input skip.
- **2026-10-05, Turing the 2nd, Prefill bounds hostile review:** Confirmed
  419,430 is the inclusive limit for the current embedding/broadcast shaders:
  GR broadcast's last flattened index is 4,294,963,199, while the next row
  count wraps 32-bit indexing. Found that fixed-mode planning accepted larger
  chunks even though `build_prefill_chunk_metadata` rejected them. The planner
  now rejects fixed chunks above the shared limit, and embedding shader strides
  use shared `N_EMBD`/`HC_DIM` constants. The exact-limit metadata test already
  exists; planner boundary tests were added. Future QSA/GEMM/MoE prompt kernels
  and Prefill's production PLE path are not present, so REV-030 remains open.
  The reviewer was read-only and ran no tests. Focused validation and the full
  49-of-50 host/Metal CTest run passed, with one expected model-input skip.
  Faraday the 2nd performed a read-only hostile follow-up and found no
  actionable P1/P2/P3 findings.
- **2026-10-05, Mencius the 2nd, sampler concurrency hostile review:** Confirmed
  the per-command-buffer guards, event lock order, dependency completion gate,
  and Context ownership checks. Found a P2: a public `BufferRef` could pair an
  owned Metal handle with forged CPU or GPU address metadata; sampler host
  validation dereferenced the CPU pointer and overlap checks trusted both
  addresses. Sampler view validation now checks the CPU pointer and GPU address
  against the native handle and offset, with regressions for each forged field.
  Updated the sampler API comment to match its ownership checks. The reviewer
  was read-only and ran no tests; focused CTest passed, and the hostile
  follow-up found no remaining P1/P2/P3 issue.
- **2026-10-05, Copernicus the 2nd, token embedding-row encoder hostile
  review:** The read-only review found no implementation defect, but reported
  a P3 test gap: token/output overlap had only been tested with one range
  containing the other while the documented claim included partial overlap.
  Added valid aligned views `[0, 8)` and `[4, 20,484)` over a 20,496-byte
  allocation; they intersect only over four bytes. Follow-up confirmed the
  partial case reaches the intended overlap error and is in bounds. The
  reviewer made no edits and ran no tests; validation is recorded under
  D-IMPL-138.
- **2026-10-05, Ramanujan the 2nd, GR broadcast encoder hostile review:** The
  initial review found a P1 test expectation mismatch: the same-handle
  metadata-less case cleared the CPU pointer and was rejected by metadata
  validation before reaching the overlap check. The test now retains the
  verified CPU pointer and clears only the GPU address; a separate
  cross-handle test still checks missing CPU metadata. Follow-up confirmed the
  P1 is closed and found no remaining actionable issue. The review was
  read-only and ran no tests; validation is recorded under D-IMPL-137.
- **2026-10-05, Hilbert the 2nd, GR up hostile review:** The initial P2
  found that U(64) on mixed outputs and U(16) on the folded residual did not
  force the exact `(R * g) * rs` or ascending stream-mix order, while `rs`
  came from the CPU norm oracle. Added a fixture with zero BF16 up rows and
  exact bit checks at two outputs: the required normalization product order
  differs from the reversed order by one ULP, and the ascending stream mix
  yields 0.75 while pair reordering yields 1.5. Follow-up confirmed both
  negative controls distinguish the expected output and found no P1/P2 or
  remaining indexing, ABI, synchronization, oracle or guard issue. The review
  was read-only and did not rerun tests; validation is recorded under
  D-IMPL-122. Production T/apply/buffer validation remains open.
- **2026-10-05, Gibbs the 2nd, GR cvec kernel hostile review:** Found one P3
  documentation mismatch: the file inventory said control vectors were
  unimplemented although `strata_gr_cvec` was added. The inventory now
  separates the implemented kernel from pending production dispatch. The
  reviewer found no indexing, synchronization, memory-access, arithmetic-order
  or manifest defect. The review was read-only and did not run tests.
- **2026-10-05, Descartes the 2nd, GR cvec test hostile review:** Found three
  P2 test gaps: the disabled-write fixture used multiplier 1 and could not
  distinguish fused FMA from multiply/add; active steering only used layer 7;
  and add-mode off/uncovered cases used a zero direction. Added a GPU
  sigmoid-derived FMA-sensitive fixture with a bit-exact output and split
  multiply/add negative control, active layer-3 project/add cases with a
  distinct project scale, and add-mode off/uncovered cases with nonzero
  directions. Follow-up confirmed all three are resolved and found no new
  issue. Both reviews were read-only and did not run tests; validation is
  recorded under D-IMPL-123.
- **2026-10-05, Lorentz the 2nd, GR cvec encoder hostile review:** The initial
  review found three gaps: distinct no-copy Metal handles over the same CPU
  backing were not checked for overlap; committed and enqueued command-buffer
  states were not tested separately; and the header omitted the caller's
  same-Context ownership precondition. Added the alias fixture and separate
  state checks, and documented that Metal handle ownership is a caller
  contract. Follow-up confirmed all findings are resolved and found no
  remaining actionable issue. The review was read-only and ran no tests;
  implementation validation is recorded under D-IMPL-124.
- **2026-10-05, Anscombe the 2nd, GR down encoder hostile review:** The initial
  review found a P2: the encoder checked command-buffer state but did not
  verify that the command buffer came from the supplied Context's queue.
  Added `CommandBuffer::belongs_to`, tracked retained buffer handles in each
  `Buffers` pool, and made all three GR encoders reject foreign buffers and
  command buffers. A second pipelines-disabled Context now supplies explicit
  foreign-resource regression cases. Follow-up confirmed queue/pool ownership
  checks and the new tests are sound, with no remaining actionable issue.
  Reviews were read-only and did not run tests; validation is recorded under
  D-IMPL-126.
- **2026-10-05, Zeno the 2nd, GR up encoder hostile review:** Found no
  actionable issue. The read-only review verified the 160×256 grid, T/apply
  binding order, static 12 KiB threadgroup-memory check, all eight buffer
  sizes/alignment/ranges, overlap and Context ownership checks, and T=1..8
  apply on/off oracle coverage. It noted that the explicit foreign-Context
  regression runs through GR down, while GR up uses the same shared ownership
  checks; no correctness issue was found. No tests were run by the reviewer.
  Validation is recorded under D-IMPL-127.
- **2026-10-05, Zeno the 2nd, GR post encoder hostile review:** The initial
  review found a P3 test gap: GR post did not directly reject a foreign
  buffer or command buffer. Added both cases using the second Context.
  Follow-up confirmed the gap is resolved and found no new actionable issue.
  Reviews were read-only and ran no tests; validation is recorded under
  D-IMPL-128.
- **2026-10-05, Zeno the 2nd, GR down-SiLU encoder hostile review:** Found a
  P2 size-coverage gap: the short-buffer case used T=1 only, so it would not
  catch a regression that sized the view for one row regardless of T. Added a
  T=2 case with exactly one row of storage; follow-up confirmed it now rejects
  the view and found no further issue. The first focused run also exposed a
  test-harness resource limit because each rejected input created another
  unsubmitted command buffer; the rejection cases now reuse one such buffer.
  Validation is recorded under D-IMPL-129.
- **2026-10-05, Zeno the 2nd, GR pre-gated encoder hostile review:** Found a
  P2 size-coverage gap: the initial short-view cases only used T=1. Added
  separate T=2 checks for `xn`, `gate` and `mixed`, each with only its target
  view shortened to one row while the other two remain large enough.
  Follow-up confirmed each reaches the intended binding's size check and
  found no new issue. Reviews were read-only and ran no tests; validation is
  recorded under D-IMPL-130.
- **2026-10-05, Zeno the 2nd, weighted RMS norm encoder hostile review:**
  Found no actionable issue. The read-only review confirmed that the T=2
  short-view cases reach the intended input, gamma and output checks, and
  reviewed the 12-byte parameter block, fixed 1024-thread row dispatch,
  alias policy, bounds, epsilon and command-buffer ownership checks. The
  reviewer made no edits and ran no tests; validation is recorded under
  D-IMPL-131.
- **2026-10-05, Zeno the 2nd, PLE broadcast encoder hostile review:** The
  initial review found a P3 test gap because exact output/input overlap was
  covered only at offset zero. Added separate nonzero-offset partial overlap
  cases for value and gate, advancing each view's offset, CPU pointer and GPU
  address consistently. Follow-up confirmed both ranges overlap the output
  while remaining aligned and in bounds, and found no further actionable
  issue. Reviews were read-only and did not run tests; validation is recorded
  under D-IMPL-132.
- **2026-10-05, Zeno the 2nd, PLE history encoder hostile review:** The
  initial review found a P2 gap because norm-size coverage used only n=2;
  added an n=17 view with 16 rows to verify the full-n source bound. It also
  found a P3 zero-length-range issue in overlap detection that rejected an
  empty n=0 norm view inside history. The range predicate now treats either
  zero-byte range as empty, and the test uses a correctly offset, zero-length
  aliased view. Follow-up confirmed both fixes and found no further
  actionable issue. Reviews were read-only and did not run tests; validation
  is recorded under D-IMPL-133.
- **2026-10-05, Zeno the 2nd, PLE gate encoder hostile review:** The initial
  review found a P3 overlap-bypass case: same-handle buffer offsets can be
  zero, but the shared range helper treated address zero as empty. Removed
  that sentinel from range comparisons; CPU pointers and GPU addresses are
  now checked for presence separately, while same-handle aliases always use
  offset and length. Added gate, broadcast and history regressions that keep
  valid handles/ranges and clear optional address metadata. Follow-up
  confirmed the fix and found no further actionable issue. Reviews were
  read-only and did not run tests; validation is recorded under D-IMPL-134.
- **2026-10-05, Mendel the 2nd, GR norm encoder hostile review:** Found no
  actionable issue. Static review confirmed the `(T, 4) × 256` ABI, row
  bounds, buffer sizes/alignment/allocation ranges and overlap checks, and
  command-buffer state validation. It also confirmed the tests cover the
  encoder path, guarded views and T=1..8 with `apply` on and off. The review
  was read-only and did not run tests; validation is recorded under
  D-IMPL-125.
- **2026-10-05, Singer the 2nd, GR down hostile review:** Found no
  actionable issue. The read-only review confirmed the 41-block down/inject
  mapping, BF16 low/high-half order, dot8 FMA and lane accumulation sequence,
  dynamic threadgroup-memory size, inactive-inject warp barriers, independent
  elementwise BF16 oracle, T=1..8 output/guard coverage, and tolerance split
  (bit-exact inject, U(16) down). The reviewer made no edits and did not rerun
  tests; focused CTest and edge Metal API/GPU Validation are recorded under
  D-IMPL-121. Production host validation remains open.
- **2026-10-05, Meitner the 2nd, GR norm hostile review:** Found a P1
  `(T, 4)` grid/ABI disagreement: the first kernel and test version treated
  group x as stream although the documented grid places T on x. The kernel,
  test launch and ABI notes now map x to row and y to stream. The review also
  found that the first scalar oracle repeated the shader's q division/modulo
  mapping. It now iterates logical vectors within each stream and separately
  assigns each vector to its simulated thread. Follow-up confirmed both
  findings are resolved and found no further issue. Focused GR CTest and
  seed-77 edge with Metal API/GPU Validation pass; `test_map.py` and
  `git diff --check` pass. The review was read-only and did not run tests.
- **2026-10-05, Laplace the 2nd, image embedding hostile review:** Found
  that the initial fixtures never marked batch row 0 as an image, so T=1
  exercised only the no-write path and a row-zero indexing defect could pass.
  The test now runs row-zero and after-text flag patterns for every T=1..8,
  including T=1 both with and without an image; expected bytes and allowed
  write ranges are derived from the token's image bit. Follow-up confirmed
  the finding is resolved and found no remaining test, kernel or documentation
  issue. The rebuilt focused CTest, fixture and edge runs with Metal API/GPU
  Validation, `test_map.py` and `git diff --check` pass. The review was
  read-only and ran no tests.
- **2026-10-05, Sagan the 2nd, GR pre-gated hostile review:** The first
  review found that the U(16) gate alone would not catch a fused product/add
  or reordered four-stream sum. Added edge inputs with exact known outputs
  and negative controls: the ordered and reordered sums yield 0.25 and 0.5;
  a Metal sigmoid probe drives a product whose fused accumulation differs
  from the separate product/sum. The reviewer also noted that the initial
  fused control depended on the CPU high-accuracy sigmoid matching Metal;
  the separate-path expected value now comes from the runtime-probed
  rounded product, while the independent scalar oracle continues to cover
  general inputs at U(16). Stream-tagged values reduce shared-layout oracle
  risk, and the test label and test-matrix wording were corrected. Final
  follow-up found no remaining concrete issue. The review was static and
  read-only; implementation validation is recorded under D-IMPL-114.
- **2026-10-05, Raman the 2nd, PLE broadcast hostile review:** Found no
  actionable kernel, indexing, exact-arithmetic, guard, row-independence,
  manifest, CMake or documentation issue against §5.3. The review was
  read-only; implementation validation is recorded under D-IMPL-115.
- **2026-10-05, Russell the 2nd, PLE history hostile review:** Found no
  kernel or test defect. An initial P2 said REV-029 still listed history as
  unimplemented; that referred to a stale document snapshot. The reviewer
  checked the current text, confirmed history is marked implemented and the
  PLE-history host n/grid/buffer checks remain open, then withdrew the P2.
  The review was read-only; implementation validation is recorded under
  D-IMPL-116.
- **2026-10-05, Euler the 2nd, PLE gate hostile review:** Found that the
  initial negative control missed serial accumulation of the 16 simdgroup
  partials. Added a second-stage control at columns 0/32/256; the follow-up
  confirmed it reduces to one under the specified butterfly, to zero under
  serial accumulation, and is rejected outside U(64). No remaining issue was
  found. The review was read-only; validation is recorded under D-IMPL-117.
- **2026-10-05, Zeno the 2nd, PLE conv-residual encoder hostile review:**
  Found no actionable issue. The read-only review checked all five buffer
  roles, T=1..8 sizing, float/half alignment, fixed grid and bindings,
  allocation and Context/queue ownership, command-buffer lifecycle, output
  alias rejection, and the policy allowing read-only input overlap. Validation
  is recorded under D-IMPL-135.
- **2026-10-05, Zeno the 2nd, image embedding encoder hostile review:** The
  initial review confirmed the 32-bit shader-index row cap
  `floor(2^32 / 2560) = 1,677,721` and found a P2: cross-handle backing
  overlap could escape detection when caller-supplied CPU/GPU metadata was
  absent. The encoder now rejects missing or inconsistent CPU pointers by
  checking each view against `MTLBuffer.contents + offset`. Tests wrap one
  anonymous backing allocation in distinct no-copy handles, check exact
  cross-handle overlap and reject cleared metadata. The read-only follow-up
  confirmed the finding is closed and found no other actionable issue; it
  checked the shared mapping lifetime, row cap, grid, bindings, range checks,
  ownership and command-buffer lifecycle. Validation is recorded under
  D-IMPL-136.
- **2026-10-05, Carver the 2nd, PLE conv-residual hostile review:** Found
  that the initial exact probes did not distinguish `R + (gated + act)` from
  a reassociated residual expression. Added a channel where SiLU(1.2) is
  absorbed into `-2^25` before cancellation with `+2^25`, while the
  reassociated oracle stays nonzero; the GPU result is required to match +0
  bitwise. Follow-up confirmed the finding is resolved and found no other
  actionable issue. The review was read-only; validation is recorded under
  D-IMPL-118.
- **2026-10-04, Erdos the 2nd, embedding/broadcast hostile review:** Found
  that the existing `metal_embed_test` inventory incorrectly claimed coverage
  of `strata_embed_image`, which was not implemented or in the manifest. The
  kernel formulas, GGML parity setup, thread indexing, guard checks and
  CMake/test-map registration had no actionable findings. The inventory now
  lists the image gather as unfinished under REV-029. The reviewer made no
  edits and ran no tests; the implementation tests are recorded under
  D-IMPL-110.
- **2026-10-04, Huygens the 2nd, weighted RMS norm hostile review:** The
  read-only review found no actionable reduction-order, FMA, registration or
  documentation defect. It confirmed that the scalar oracle reproduces the
  documented native reduction but does not claim bitwise parity with ggml
  CPU's FP64 accumulation. Host validation for `n_cols`, `gp`, buffer bounds
  and alias rules remains a prerequisite to production dispatch; the current
  twin covers only documented widths/periods, with in-place behavior on the
  fixture set. Those scope limits are recorded under REV-029. The reviewer
  made no edits and ran no tests.
- **2026-10-04, Huygens the 2nd, RMS norm tolerance follow-up:** Found that
  requiring exact host-oracle bits would overstate cross-generation behavior:
  the M5 Pro measurements show safe-mode `rsqrt` matching a double-derived
  FP32 reference, but do not guarantee that result on every supported GPU.
  The test now propagates scalar-oracle magnitudes and uses U(64), while
  retaining exact-bit counts as diagnostics. The tolerance is derived for the
  tested finite range; subnormal/extreme inputs remain open. The review also
  confirmed that the oracle is not a ggml CPU FP64 parity test. A second pass
  verified the `twin::tf` magnitude chain, the U(64) error estimate, and that
  the reported maximum U(n) uses the same bound as the pass/fail comparison;
  it found no actionable issues. Both passes were read-only and ran no tests.
- **2026-10-05, Epicurus the 2nd, native GR post hostile review:** Found no
  actionable implementation defects. The review confirmed the native
  sigmoid/FMA order, the `(40, T) × 256` indexing, and the scalar U(16)
  magnitude accounting. It noted that only exact residual/output alias is
  covered, and that fixed-grid/buffer checks, overlap policy, production
  Window/Prefill wiring, and broader T/model inputs remain open under
  REV-029. The reviewer made no edits and ran no tests.
- **2026-10-05, Popper the 2nd, down-SiLU hostile review:** Found that the
  U(16) comparison did not enforce the sign of zero, so replacing the
  quarter-scale FMA with multiplication could pass despite changing `+0` to
  `-0`. Added exact sign-bit assertions for every edge-set negative-zero
  input, and renamed the twin's summary to include down SiLU. Targeted and
  full tests pass. The review also confirmed the `n_values` ABI, T=1..8 tail
  coverage, in-place indexing and scalar magnitude accounting. Its follow-up
  verified the exact sign assertion and found only this documentation
  direction typo; all other scope and coverage claims matched. The reviewer
  made no edits and ran no tests.
- **2026-10-04, Socrates the 2nd, sampler hostile review:** Found that maximum
  vocabulary greedy scratch exceeded the 32 KiB threadgroup limit, the
  coupled-path twin did not use nonzero `kpart` or nonempty-history penalty
  bypass, `--seed` and `--input-set` were ignored, and the CPU tail could
  randomly choose among multiple candidates for greedy requests. Greedy now
  scans two 131,072-token chunks and reuses its bitmap allocation for SIMD
  results; the coupled test covers the documented ABI; all three input sets
  run by default; and greedy tail returns the first candidate with probability
  1. The review also identified that production dispatch and the complete
  upstream fixture port are not present. Those remain explicit under
  REV-028, and documentation no longer claims L8 completion.
- **2026-10-04, Socrates the 2nd, sampler follow-up review:** Found stale
  fixture and production-path wording, a 2 KiB instead of 2,560 B split-part
  memory figure, a duplicated tail RNG oracle, and quantized random logits.
  Documentation now labels production wiring as planned, states the synthetic
  fixture scope, and gives the correct memory use. The test oracle uses its
  own Philox implementation with known answers; random inputs are unquantized
  and retried until the top-k boundary is separated. A second pass found greedy
  matrices had missed that retry; the final pass verified random greedy cases
  at all tested vocabulary sizes, including penalized and stale-history
  variants, and found no actionable issues. The reviewer made no edits and ran
  no tests.
- **2026-10-04, Socrates the 2nd, fixture 15 review:** Verified the
  `penalty_rows` port against upstream's tail/window order, front padding,
  empty-tail case, and 420 comparisons across the upstream parameter matrix.
  No actionable findings; the reviewer made no edits and ran no tests.
- **2026-10-04, Archimedes the 2nd, fixture 16/17 hostile review:** Found that
  `std::normal_distribution` made generated inputs implementation-dependent,
  fixture 16 omitted upstream's 262,145-entry fallback, and fixture 17's
  purported stream variants used one Metal queue. Replaced the distribution
  with a fixed-point integer generator over standardized `mt19937` outputs and
  pinned its bins with known answers; documented these as semantic ports,
  checked the 262,145 shape refusal against Metal's limit, and renamed the
  duplicated cases to draw-counter variants while documenting the shared
  queue. The review also checked row batching and source-row counter
  preservation. A follow-up review found two documentation ambiguities,
  recorded below; both were corrected and no actionable issues remain.
- **2026-10-04, Plato the 2nd, fixture 16/17 documentation hostile review:**
  Found L8 wording that could imply fixture 15 and semantic cases 16/17 had
  the same fidelity, and a CTest result without an explicit post-change
  marker. L8 now labels fixture 15 as a host fixture and 16/17 as semantic
  ports; the full 46-test run is identified as post-change. No code findings;
  the reviewer made no edits and ran no tests.
- **2026-10-04, Lagrange the 2nd, sampler encoder hostile review:** Found
  missing overlap checks, undocumented same-Context and exclusive
  single-thread CommandBuffer requirements, and a host-component table that
  claimed the absent generic Encoder existed. Encoders now reject overlap in
  BufferRef offsets, CPU address ranges and GPU address ranges, including
  distinct no-copy Metal handles over the same backing address. The API
  documents its ownership/concurrency preconditions, and the table marks the
  generic Encoder as planned. Follow-up confirmed these fixes and found no
  actionable issues. The review was read-only and ran no tests.
- **2026-10-04, Nietzsche the 2nd, shared-expert chain hostile review:** The
  read-only review found no actionable memory-safety, tensor-layout,
  partial-batch sink, oracle-independence or missing-stage issues. It
  confirmed that GPU outputs feed the following GPU stages through combine.
  The review noted that routed parts and route weights at combine are
  synthetic, so the test covers the shared-expert chain through combine but
  does not integrate the router and routed experts; D-IMPL-107 records that
  scope. The random, edge and fixture runs, 43-test CTest suite (one expected
  real-input skip) and 124.25-second ASan MoE run passed.
- **2026-10-04, Lovelace the 2nd, shared-expert SwiGLU hostile review:** A
  read-only review of the shared helper, separate entry point, manifest,
  T=1..8 twin, and documentation found no actionable findings and made no
  edits. The full 43-test CTest run (including the expected real-input skip)
  and the all-input-set ASan MoE run pass.
- **2026-10-04, Nash the 2nd, MoE combine hostile review:** The first pass
  found that the final-add fixture assumed one exact sigmoid bit pattern even
  though the math contract permits several ULPs of variation, and that an
  explicit `real` selection returned success without running the kernel. The
  final-add check now reads the device sigmoid through a same-dispatch
  unit-shared coordinate and computes both separate and fused outcomes from
  that observed value; the contraction difference was checked across 17
  sigmoid values around 2/3. A dedicated host CTest invokes `--input-set real`
  and records return 77 as skipped. The follow-up confirmed both fixes and
  found no actionable issue.
- **2026-10-04, Bacon the 2nd, first grouped IQ4_NL MoE down hostile review:**
  Found that the initial kernel combined a block's two 16-value dots, the
  fixed answers were not checked directly against GPU output, and grouped
  multi-entry parity and a weight-bit negative control were missing. The
  implementation was changed to two ordered FMA calls, exact GPU goldens,
  forced singleton parity and a bit-flip control.
- **2026-10-04, Pauli the 2nd, grouped IQ4_NL MoE down follow-up reviews:**
  The next pass found that a poisoned negative-control output could make a
  no-write regression pass, the original fixed probe did not activate both
  partials with different dots, and the first scalar oracle repeated the
  kernel's lane mapping. The negative control now starts from the valid
  singleton output and checks the analytically changed value; the fixture
  uses distinct partial dots that distinguish split from merged FMA; and the
  oracle decodes by logical position and sums independently of lane mapping.
  The final review rechecked the golden and bit-flip arithmetic, singleton
  plan construction and no-write behavior and found no remaining actionable
  findings. The focused MoE test, full 42-test CTest suite, and ASan
  `metal_mv_test`/`metal_moe_test` runs passed.
- **2026-10-04, Hubble the 2nd, grouped Q2_0 MoE down hostile review:** The
  Q2_0 byte/half mapping, Q8_1 part offsets, 20-call lane tail, half-scale
  product, FMA/butterfly order, role-2 address stride, grouped activation
  indexing, `ent_dst` bijection/restoration, 80-entry expert permutation,
  unique-scale one-hot answer and guard/CMake/manifest coverage had no
  actionable findings. The focused validation run, full 42-test CTest suite,
  ASan build and all-input-set MoE/mat-vec tests passed.
- **2026-10-04, Averroes, grouped IQ4_XS MoE gate/up hostile review:** Checked
  the split 6-bit scale unpacking, low/high nibble-to-Q8_1 lane mapping,
  integer-dot range, half conversion, FMA and butterfly order, row stride,
  four-entry token mapping, 80 distinct expert address slots, shared fixed
  packed fixture, guards, and CMake/manifest registration. No actionable
  findings. The targeted Metal shader/API-validated tests passed, the full
  42-test CTest suite passed, and the ASan build plus both all-input-set
  `metal_mv_test` and `metal_moe_test` runs passed.
- **2026-10-04, Dalton, grouped IQ3_S MoE gate/up hostile review:** The
  9-bit `qs`/`qh` indices, sign nibbles, scale nibbles, 32-lane traversal,
  four-entry batching, 10-block row stride, Q8_1 token stride, expert-slot
  mapping, 80-group one-hot fixture, guards, and manifest/CMake wiring matched
  the standalone IQ3_S implementation. No actionable finding. The reviewer
  confirmed this adds only a Metal implementation; CUDA remains an external
  reference under D-IMPL-094. Validation-enabled `metal_mv_test` and
  `metal_moe_test` passed, all 42 CTests passed, and both all-input-set ASan
  tests passed.
- **2026-10-04, Aquinas, grouped IQ3_XXS MoE gate/up hostile review:** The
  sign restoration, grid bytes, scale nibble and signed division match the
  standalone Metal row-dot and the repository's CUDA-derived reference
  arithmetic. Q8_1 layout, lane/warp and four-entry indexing, expert slot
  mapping, 80-group endpoint, shared one-hot fixture, guards, and build
  registration had no actionable code findings. The review found stale
  “before grouped integration” wording in the IQ3_XXS and IQ1_M standalone
  twin descriptions; updated both to describe their continuing decoder-check
  role. The pinned `vecdotq.cuh` is not present locally, so the original
  upstream CUDA implementation was not independently inspected. The Metal
  build and validation-enabled IQ3_XXS tests passed, the full CTest suite
  passed 42/42, and both all-input-set ASan tests passed.
- **2026-10-04, Ampere, grouped IQ2_S MoE gate/up hostile review:** Rechecked
  the 10-bit grid index assembled from `qs` and `qh`, explicit sign bytes,
  scale halves and signed divisions, plus the TF oracle, group/token mapping,
  80 unique address slots, fixture sharing, output guards and build/test
  registration. No actionable finding. API/GPU-validated `metal_moe_test` and
  `metal_mv_test` passed across random, edge and fixture inputs, all 42 CTests
  passed, and both all-input-set ASan runs passed.
- **2026-10-04, Raman, IQ2_XS TF row-oracle hostile review:** Compared the
  current scalar decoder and arithmetic with the standalone Metal row kernel,
  shared codebooks and pinned ggml/CUDA implementation. Confirmed packed word
  addressing, grid/sign/component extraction, both partial sums, signed
  division truncation, per-lane FMA and butterfly order; no actionable finding.
  This was a static review of the row oracle, before the grouped integration
  test run.
- **2026-10-04, Banach, grouped IQ2_XS MoE gate/up hostile review:** The
  kernel, oracle, packed fixture, address tests, CMake and registration had no
  actionable implementation findings. The review did find stale status text in
  the kernel inventory, which still called IQ2_XXS and IQ2_XS planned, and
  described the IQ2_XS row twin as pre-integration work. Updated the inventory
  and row-twin descriptions to match the implemented grouped kernels and their
  continuing decoder-isolation role; follow-up review confirmed the mismatch
  is resolved. API/GPU-validated `metal_moe_test` and `metal_mv_test` passed
  across random, edge and fixture inputs, all 42 CTests passed, and both
  all-input-set ASan runs passed.
- **2026-10-04, Laplace, grouped IQ2_XXS MoE gate/up hostile review:** The
  first 80-group test mapped 80 expert IDs onto ten identical weight slots, so
  it could not catch a wrong expert-to-address mapping. It now allocates 80
  distinct slots, assigns separate gate/up half scales to each expert and
  drives the fixed position-41 one-hot fixture. The test checks each oracle
  result against the independent `fma(scale, 1190, 0)` answer and verifies all
  80 gate and up answers are distinct. Follow-up review found no remaining
  issue. API/GPU-validated fixtures, all 42 CTests and the all-input-set ASan
  run pass.
- **2026-10-04, Pasteur, grouped IQ1_M MoE gate/up hostile review:** The
  initial review found that the scalar oracle dropped IQ1_M's `twin::tf`
  magnitude before the inner FMA, the fixture reached only ten unique groups,
  and the grouped test did not use packed tool-generated IQ1_M blocks.
  Follow-ups also found out-of-policy random scales and two fixture-generator
  errors: incorrect half-scale assembly and masking bits used by local scales.
  The oracle now preserves TF rounding, tests 80 unique groups, and embeds
  `tools/iq_fixture.py` raw blocks; random scales stay in [2^-12, 2^-4], and
  compile-time fixture checks pin half scale 0x3c00 and cover all eight local
  scale values. The final re-review found no actionable findings. The
  API/shader-validated Metal test, all-input-set ASan run, and all 42 CTests
  pass.
- **2026-10-04, Anscombe, routed-expert SwiGLU hostile review:** Found the
  L7 review table still listed SwiGLU as incomplete after the kernel, scalar
  oracle, and T=1..8 twins had landed; corrected REV-027 to leave only the
  genuinely unfinished L7 components open. No other actionable finding in
  the kernel, oracle, twin, CMake, or manifest. Metal API/shader validation,
  all 42 host/Metal CTests, and the all-input-set ASan run pass.
- **2026-10-04, Carson and Mendel, IQ2_XXS row-dot hostile reviews:** Carson
  found that the first packed-u16 base used `qs + 4 * iqs`; since `qs` is a
  u16 array and `iqs = 2 * part`, the correct base is `qs + 2 * iqs`. The
  correction keeps all four loaded words in the selected 32-value part, and
  the n_in=256 fixture exercises the former out-of-bounds lane. Mendel then
  found that the all-code/sign/scale sweep generated its packed words and
  expected values at runtime while the docs called them literal. The sweep is
  now labeled generated, and a separate fixture pins 32 u16 words and eight
  one-hot float answers. Mendel's follow-up confirmed that finding resolved.
  Shader/API-validated fixture and all-input-set ASan runs pass; all 41 CTests
  pass, including the 13 Metal tests.
- **2026-10-04, Heisenberg, IQ2_XS row-dot hostile review:** Compared the
  decoder and arithmetic to pinned CUDA reference
  `3cf03257f219afbe7334045ff7c6a06ac68c627d`. Confirmed the u16 part offset,
  9-bit grid and 7-bit sign extraction, both scale halves, signed `/2` and
  `/4` truncation, dp4a order, Q8_1 mapping, FMA/reduction order, inactive
  column sink, and manifest/docs registration; found no actionable source
  issue and made no edits. The subagent performed static review only. Local
  shader/API-validated fixture, random and edge runs pass; ASan runs for all
  three input sets pass; all 41 CTests pass, including all 13 Metal tests.
- **2026-10-04, Curie and Cicero, IQ2_S row-dot hostile reviews:** Curie
  found that the first fixed raw fixture only pinned components 0 and 7, while
  the generated sweep's expected values came from the scalar oracle. The
  fixture now pins all eight components of the final codeword with literal
  answers and a sign byte containing both set and clear bits. Curie also found
  the missing source inventory, implementation decision and REV-026 update;
  D-IMPL-088 and those records are now present. Cicero's follow-up confirmed
  both findings resolved and found no remaining P1/P2 issue against the pinned
  CUDA reference. Both reviews were static and made no edits or test runs.
  The final shader/API-validated fixture, random and edge runs, all three
  corresponding ASan runs, and all 41 CTests pass.
- **2026-10-04, Hegel, IQ3_XXS row-dot hostile review:** Hegel confirmed the
  packed bounds, grid/sign/parity mapping, scale, signed division, activation
  indexing, lane reduction and NC sink against the pinned source path. Hegel
  found that the initial 9/13-block FMA goldens used `d8 = 1`, so they could
  not detect reassociation of `d * d8`; the follow-up fixtures use `3.7f`
  activations (`d8` half bits `0x2775`) and literal outputs `0x45fbc643` and
  `0x46440c05`, each one ULP from the reassociated result. The follow-up
  confirmed that finding resolved and found no remaining actionable issue.
  An earlier focused run also caught and corrected the padded-column and
  9-block literals before review. Both reviews were static and ran no tests.
  Shader/API-validated fixture, random and edge runs, all three ASan runs,
  and all 41 CTests pass. Upstream CUDA SASS confirmation remains open under
  L-9 / REV-026.
- **2026-10-04, Sagan, IQ1_M row-dot hostile review:** Sagan found that the
  generated sweep's local-scale self-check selected the packed word by
  codeword instead of by the fixed part 0, and that the initial literal
  one-hot answers did not follow the CUDA low/high-nibble activation order.
  Both fixtures were corrected. Sagan also noted that the generated sweep and
  both decoders share `iq1s_grid_gpu`; the existing `codebooks` CTest compares
  all 2,048 entries byte-for-byte with the pinned llama.cpp table, and that
  test passes. The follow-up found no remaining actionable code or test issue.
  Focused fixture, random and edge runs and their AddressSanitizer runs pass.
  The review was static and made no edits or test runs. CUDA SASS confirmation
  of the FMA form remains open under L-9 / REV-002.
- **2026-10-04, Dirac and Zeno, Q3_K mat-vec read-only reviews:** Dirac
  confirmed the high-mask sign, packed signed scales, half-aligned loads,
  Q8_1 indexing, thread stride, row threshold, FMA order and reductions match
  the pinned CUDA path. Zeno found that the first literal raw pattern repeated
  code/sign pairs eight positions apart within a scale group, and that a
  zero-valued scale made one group's code results unobservable. The literal
  byte arrays and expected-code formula now differ for each `p`/`p+8` pair;
  the affected raw scale is nonzero, and the analytic fixture independently
  checks zero-scale output. Follow-up verified all 256 codes, all 16 scales
  and distinct answers for every pair, with no remaining code findings. The
  focused fixture, random and edge input sets, all 13 Metal CTests and all 41
  CTests pass with API/shader validation; the all-input-set ASan mat-vec run
  and `test_map.py` pass.
- **2026-10-04, Hume, Helmholtz and Halley, Q4_K read-only reviews:** Hume
  confirmed the packed scale/min decode, Q4/Q8 lane offsets, eight-block
  stride, row threshold, reduction and FMA mapping against the pinned native
  source. Helmholtz caught that the first FMA literals came from the Q4_K
  oracle and that the negative control compared results from different
  weights. The literals were independently re-derived with a scalar lane/warp
  calculator (`0x44bf9107` for 9 blocks; `0x4506f316` for 13), and the
  property now checks changed weights against NC = 1 runs using those same
  weights, then confirms a one-bit output difference is detected. Halley
  caught the missing Q4_K inventory and REV-026 coverage; both are now
  recorded. Production inference dispatch is a planned follow-on integration
  and remains open under REV-026. Final focused fixture, random and edge runs,
  all 13 Metal CTests and all 41 CTests pass with API/shader validation; the
  all-input-set ASan mat-vec run and `test_map.py` pass. CUDA SASS confirmation
  of the final fused expression remains open because this host has no CUDA
  reference GPU/toolchain.
- **2026-10-04, Pascal and Maxwell, Q5_K read-only reviews:** The literal
  one-hot fixture first exposed a missing Q8 lane offset on the QH read in both
  the kernel and oracle; both now use `qh[4 * q8_word + local]`. The fixture
  checks each decoded code directly from its literal packed QH/QS bytes against
  the independent position formula at all 256 positions, all 32 codes, and
  both high-bit values per group. Pascal also found that the first coverage
  mask assertions were derived only from the expected-code formula; those
  assertions now decode the literal arrays themselves. Follow-up reviews found
  no remaining mapping, scale/min, FMA/reduction, bounds, registration or
  parity issue. Independent lane/warp calculations pin the 9/13-block results
  (`0x455ecdce`, `0x4597a411`); long-K answers cover 63/64/65 blocks. The
  focused fixture, random/edge sets, all 13 Metal CTests and all 41 CTests
  pass with API/shader validation; the all-input-set ASan mat-vec run passes.
  Production inference dispatch and CUDA SASS confirmation remain open under
  REV-026 and L-9/REV-002.
- **2026-10-04, Einstein and Mill, Q6_K mat-vec read-only reviews:** Initial
  kernel/oracle review found no CUDA mapping, signed-scale, FMA, reduction,
  alignment or bounds issue. Mill found that the first raw-code sweep and the
  analytic fixture both used the test packing setter, leaving a common-mode
  packed-layout gap, and that the long-row boundary had no independent known
  answer. Follow-up hardening uses literal `ql`/`qh` arrays and independent
  answers at all 256 positions, and fixed analytical outputs cover 63/64/65
  blocks. The first partial raw fixture had three incorrect expected codes;
  the focused run exposed the mismatch, and the complete raw array now checks
  each code against the separate logical-position formula. Final review
  confirmed all 256 decoded codes, all 64 values per 64-position segment and
  the long-K answers, with no remaining P1/P2/P3 findings. The focused
  fixture with API/shader validation, all 13 Metal CTests, all 41 CTests,
  the all-input-set ASan mat-vec run and `test_map` pass.

- **2026-10-04, MoE router hostile reviews:** Franklin's read-only kernel,
  oracle and test review found no actionable P1/P2/P3 issue. Lorentz found
  missing independent route-weight goldens; fixed equal and nonuniform
  goldens were added. The follow-up noted those U(16) goldens do not reliably
  identify the selected arithmetic mode, so an exact-bit mode probe was added.
  Euclid found an earlier exp-based mode witness could vary across Apple GPUs
  and that a standalone arithmetic kernel did not cover the router call site.
  The probe now uses fixed inputs in the same macro-generated kernel body as
  `r16`/`r8`. Euclid's follow-up found no actionable issue. Focused route
  parity, exact mode bits, API/shader validation, the all-input-set ASan run
  and all 42 host/metal CTests pass; L7 plan, expert, shared-expert, combine
  and production-dispatch work remains open.
- **2026-10-04, MoE plan hostile review:** The first parity run found that the
  kernel wrote the shared `gstart_s` offsets but omitted the corresponding
  `MoePlan.grp_start` entries. Added those writes; exact full-struct comparisons
  now pass for random, unique-entry and repeated-entry layouts. Arendt's
  read-only ABI, synchronization, bounds, residency and oracle review found no
  actionable P1/P2/P3 issues. Shader/API validation, all-input-set ASan, and
  all 42 host/metal CTests pass.
- **2026-10-04, Nash, grouped Q2_0 MoE hostile review:** The first test-input
  pass reused 16-value source patterns across 32-value Q8_1 blocks, so edge and
  fixture blocks repeated and did not exercise low/subnormal scales or distinct
  token-row activation offsets. Replaced those sets with deterministic
  row- and block-specific source floats covering zero, half-subnormal and
  underflowed scales, minimum-normal scale, exact rounding ties, ordinary
  scales and large outliers; all are encoded by the Q8_1 scalar oracle. Nash's
  follow-up confirmed the finding is resolved and found no remaining actionable
  issue. Metal API/shader validation, the all-input-set ASan run and all 42
  host/Metal CTests pass.
- **2026-10-04, Schrodinger, Q8_1 read-only review:** Found that the
  subnormal division path was selected from `amax`'s exponent, which could
  misroute a block whose computed `d` was the minimum normal value. The
  quantizer now branches on `d`'s exponent and the edge fixture covers
  `amax = 127 × min-normal`. Follow-up found that a subnormal `xi` also needs
  a bitwise-safe path when `d` is normal; significand division now handles the
  minimum-normal denominator, with larger normal denominators returning zero.
  Also found CUDA parity claims that included out-of-range float-to-int8
  conversions; the docs now limit parity to defined conversions and state
  the local zero/subnormal policy, including the MoE statement. Focused tests,
  Metal validation, and the 41-test host/metal suite pass after the first
  fixes; the latest mixed-scale path has passed focused tests and Metal
  validation. The final review found no P1/P2 issues and one documentation
  threshold ambiguity; the exponent-field wording and fixtures for minimum
  normal, just above it, and twice minimum normal close that P3.
- **2026-10-04, Schrodinger and James, BF16 mat-vec read-only reviews:**
  Neither found a P1/P2 issue. Both found that testing only NC−1 live rows
  omitted the all-live case and some inactive sink offsets; `metal_mv_test`
  now covers every live-row count from 1 through NC. A hard-coded analytic
  input/weight fixture independently checks the oracle's column layout and
  inactive-column padding. Schrodinger's final sweep found no actionable
  P1/P2 issue. The focused test, GPU/API validation edge run, and all 41
  host/Metal tests pass. CUDA-produced model-real data and production dispatch
  remain open under REV-025 / REV-026.
- **2026-10-04, James, BF16 `mv_multi_eq_single` review:** Initially found a
  P3 because only the first K pair was nonzero. The fixture now has nonzero
  deterministic values at every K position for n_in = 320, 2560 and 10240,
  with column-varying inputs; each NC=2/4/8 GPU result is byte-compared with
  separate NC=1 dispatches. The one-bit negative control changes 128 to 129
  at a nonzero input. Follow-up closed the P3 with no P1/P2 findings and
  caught the stale “sparse” wording above, which is corrected. Focused CTest,
  GPU/API Validation fixture run, and all 41 host/Metal tests pass.
- **2026-10-04, Schrodinger and James, Q8_0 mat-vec read-only reviews:**
  Neither found a P1/P2 issue. Schrodinger found that the kernel source
  inventory did not name the new file; the inventory now lists the Q8_1
  quantizer and Q8_0 mat-vec separately. James found that tests missed the
  32-block row-group threshold; n_in = 992 and 1024 now exercise the 31-block
  small-K and 32-block one-row paths for every NC and live-row count. Both
  P3s were closed. Focused CTest, GPU/API Validation edge and fixture runs,
  the full 41-test host/Metal suite, and `test_map` pass.
- **2026-10-04, James, Q8_0 `mv_multi_eq_single` review:** No actionable
  findings. The review confirmed each single-column dispatch receives the
  same raw input slice as its multi-column counterpart, each exact comparison
  covers one column-major output, and the dense patterns vary across all K.
  The code-bit negative control changes output by 127 for NC column 0.
  Focused CTest, GPU/API Validation fixture run, and all 41 host/Metal tests
  pass; other format families remain open under REV-026.
- **2026-10-04, Schrodinger and James, Q4_0 mat-vec read-only reviews:**
  Neither found a P1/P2 issue. James verified that the dense, K-varying
  `mv_multi_eq_single` cases compare each NC=2/4/8 Metal column against a
  separate NC=1 dispatch across all five widths, and that the code-bit
  negative control changes output. Schrodinger found that the initial
  analytic fixture used equal low/high nibbles and constant activations,
  making it unable to detect a shared nibble-order error in the kernel and
  oracle. The fixture now uses different low/high codes, varying Q8_1 codes
  across K, and literal expected outputs; follow-up confirmed that a
  low/high swap changes the expected answer and that the D-IMPL-076 coverage
  statement matches the implementation. Focused CTest, API/shader validation
  for all 13 Metal tests, all 41 host/Metal tests, and `test_map` pass.
- **2026-10-04, Jason and Russell, Q5_0 mat-vec read-only reviews:** The
  initial review found a common-mode qh mapping gap and an analytic fixture
  without mixed high bits; both were closed with literal raw-byte coverage
  for all 32 positions, a mixed-bit analytic fixture, and an outer-FMA
  golden. Russell's follow-up found duplicate decoded values at two raw
  positions. The raw block now uses qh bytes `a5 3c 5a c3` and packed qs
  bytes `00 11 22 33 44 55 66 77 88 99 aa bb cc dd ee ff`, which uniquely
  cover codes 0 through 31. Both final reviews independently verified the
  literals against the packed low/high layout and found no remaining
  actionable findings. The fixture run, focused test, all 41 host/Metal
  tests, all 13 Metal tests with API/shader validation, and `test_map` pass.
- **2026-10-04, Newton, Tesla and Gauss, Q2_0 mat-vec read-only reviews:**
  Newton found no kernel or oracle issue. Tesla found that REV-026 omitted
  Q2_0; the coverage table now lists its widths and properties. Gauss found
  the shared Q8_1 edge generator wrote past the 64-value Q2_0 fixture and
  cleared synthesized values after its special blocks. AddressSanitizer
  confirmed the heap overrun; the generator now writes only complete blocks
  within the row and preserves later synthesized blocks. Gauss also requested
  an independent multiple-block golden. The 4160-value fixture now assigns
  distinct scales to the final weight and activation blocks and checks the
  literal 524256 result; final follow-up confirmed the address-sensitive
  calculation. ASan edge and fixture runs, all 13 Metal tests with API/shader
  validation, all 41 host/Metal tests, and `test_map` pass. Production
  dispatch, exhaustive `mv_decode` coverage and model-backed `real` inputs
  remain open under REV-025 / REV-026.
- **2026-10-04, Franklin and Lorentz, IQ4_NL mat-vec read-only reviews:**
  Franklin found no kernel/oracle mismatch. Lorentz found that long-row
  outputs relied on an oracle that mirrors the GPU schedule and that the
  block's field offsets were not pinned. The 65- and 129-block FMA-sensitive
  fixtures now check block 64 and block 128 against the literal
  `-40.222751617431640625`; compile-time assertions pin `d` at byte 0 and
  `qs` at byte 2. Lorentz's final review found these findings closed. The
  ASan mat-vec run, all 13 Metal tests with API/shader validation, and all 41
  host/Metal tests pass. Production dispatch, exhaustive `mv_decode`
  coverage and model-backed `real` inputs remain open under REV-025 / REV-026.
- **2026-10-04, Hubble and Lovelace, IQ4_XS mat-vec read-only reviews:**
  Hubble found that the initial four-row threshold differed from CUDA. The
  kernel, oracle and test dispatch now use the native 16-block threshold, and
  widths 3840/4096 exercise both sides. Lovelace found that the scale golden
  packed bytes through the same setter under test. It now specifies literal
  `scales_l` and `scales_h` bytes, with independently listed decoded scale
  codes covering each low bit and all four high-bit combinations. The first
  raw high-bit literal had a missing bit in subblock 2; the independent
  fixture caught it and the literal was corrected before the final run.
  D-IMPL-080 and this REV-026 update record the decision and coverage.
  Follow-up review confirmed the threshold and raw-scale findings closed,
  with no further code findings. API/shader validation passes for all 13
  Metal tests; all 41 CTests, the all-input-set ASan mat-vec run and
  `test_map` pass. Production dispatch, exhaustive `mv_decode` coverage and
  model-backed `real` inputs remain open under REV-025 / REV-026.
- **2026-10-04, Hilbert and Leibniz, IQ3_S mat-vec read-only reviews:**
  Hilbert caught that the initial kernel advanced `signs` by one byte per
  32-value sub-block instead of four; it must use
  `signs[4 * (iqs / 2) + l0 / 2]`, as CUDA's `get_int_b2` does. The kernel was
  fixed, and the one-hot raw fixture checks sign bytes across every part.
  Leibniz's final review found no remaining indexing, scale, FMA, reduction,
  bounds, padding or parity issues. The fixture covers all 512 codebook
  indices, qh bit positions, both signs per value position, all scale nibbles,
  independent 9/13-block lane/warp goldens and 63/64/65-block answers. All
  three input sets, all 13 Metal CTests with API/shader validation, all 41
  CTests, `test_map` and the all-input-set ASan run pass. Production inference
  binding and CUDA SASS confirmation remain open under REV-026 / L-9.
- **2026-10-03, Kierkegaard, H5 read-only review:** Found an MTP-window
  undercount, geometry-output/input alias risk, overbroad expert-count range,
  unchecked draft-vocabulary values, an unrecognized conversion method being
  accepted, and a v4 fixture that did not split roles across shards. These
  were fixed with planner scaling, output identity checks, catalogue-only
  expert counts, vocabulary validation, a packer-method allow-list and a
  three-role v4 split fixture. The claimed missing `head_count` check was
  already covered by `strata::check_architecture` (24 heads and 2 KV heads).
  SHA-256 content recomputation remains intentionally outside the payload-free
  PLAN path and is recorded in D-IMPL-043 / REV-010.
- **2026-10-03, Kierkegaard follow-up:** Found that the default `pack/full`
  path was missing from geometry-output collision checks, and that the final
  vocabulary EOF probe did not distinguish I/O errors. Both were corrected;
  the CLI regression now exercises an omitted `--pack` against the real
  default path.
- **2026-10-03, Locke, H6 read-only review:** Found that a `STOP` read just
  after a queued `GEN` could be cleared before generation, that active-request
  EOF was not covered, and that the fake vision request bypassed SVE1
  validation. The reader now associates queued STOP with the earliest queued
  request and synchronizes active state; child tests cover queued/running STOP
  and EOF. GENI loads and validates SVE1 records and checks M-RoPE positions.
  Later review rounds found and fixed further ordering and boundary cases
  recorded below. The production `main.cpp` adapter remains intentionally
  open under D-IMPL-044.
- **2026-10-03, Locke, H6 follow-up review:** Found a STOP/DONE attribution
  race, nondeterministic EOF before dequeue, unbounded SVE1 header allocation,
  partial trailing headers, token-count and progress validation gaps, and
  insufficient cancellation-result validation. The reader now linearizes
  completion and STOP under one mutex; SVE1 allocation is bounded by prompt
  image-pad runs, context and file length; callback/outcome invariants are
  checked before terminal output. Tests now cover immediate EOF, STOP after
  DONE, overproducing backends, malformed progress, multi-image 2-D M-RoPE
  and partial headers.
- **2026-10-03, Locke, second H6 follow-up:** Found unbounded input line and
  queue growth, a 2.5 GiB image payload case, PP totals detached from the
  prompt length, incomplete progress accepted before `REUSED`, and
  `finish=stop` accepted without an EOS token. Added 4 MiB line and 1,024-line /
  8 MiB queue limits, a 256 MiB SVE1 payload cap, exact PP bounds/completion,
  and EOS-ID validation. EOF now cancels active requests even if `QUIT` is
  queued; transcript tests keep stdin open until `QUIT`. Regression coverage
  includes preclosed EOF, EOF with queued QUIT, oversized lines/queues, large
  image grids, malformed PP and non-EOS stop outcomes.
- **2026-10-03, Locke, third H6 follow-up:** Found that bytes after an
  overlong line could grow without limit, read-batch overflow could be
  deferred, EOF could bypass a queued overflow error and CR stripping,
  completion could race with STOP/EOF held in reader-local state, and the
  backend could emit tokens after EOS. The reader now bounds partial-line
  storage byte by byte, parses ready input while holding the completion
  mutex, drains ready stdin before the finish snapshot, and finalizes the last
  line through the normal CR-stripping and queue-limit path. EOS now makes
  `should_stop()` true and rejects any subsequent token callback. Regressions
  cover a continuing 16 MiB overlong line, queue overflow followed by EOF,
  CR-terminated EOF and EOS callback ordering.
- **2026-10-03, Locke, H6 targeted re-review:** Found that request completion
  could stop draining at 8 MiB and snapshot before a later STOP/EOF in the
  same available stream. Completion now drains up to 64 MiB under the reader
  mutex; reaching the ceiling cancels the active request, queues a dedicated
  bounded input error and exits with status 2.
- **2026-10-03, Locke, final H6 hostile re-review:** Found that a 16 KiB read
  could overshoot the completion-drain ceiling by one chunk. Each read now
  uses only the remaining budget, and a one-byte probe distinguishes exact
  limit/EOF from actual excess. The final review found no actionable issue in
  the drain boundary or adjacent STOP/EOF handling.
- **2026-10-03, Turing, H7 read-only review:** Found the missing upstream MTP
  summary line and the zero-draft case that could suppress it or divide by
  zero. The formatter now keeps the line whenever decoding ran, uses a
  zero-safe average when no draft ran, and covers that edge case. The separate
  concern that `logits dumped` was absent was not adopted: upstream prints
  that summary after the stats block, and the formatter preserves it there.
- **2026-10-03, Turing, H7 follow-up:** Confirmed the zero-draft fix and the
  retained MTP-line classification; no further actionable findings.
- **2026-10-03, Meitner, U2 hostile review and follow-ups:** Found registry and
  requirement-map omissions, an incorrect 38 GiB test cap, a missing PLE
  boundary assertion, and the need to reconcile PLE exclusion with mandatory
  page rounding. These were fixed by mapping REQ-MEM-08, using the specified
  38.88 GiB cap, asserting no PLE slice/address mapping while bounding edge
  padding to one page, and updating the related 03/05/06 loader rules and
  `metal_never_wrap_test` contract. Follow-ups found a missing fixture-presence
  assertion, no unaligned expert split case, and missing shared CMake warning
  options; all were added. The final follow-up found that a forced
  max-buffer split could omit a merged unread gap; bounded gap-only windows now
  preserve the pre-split range union. The final hostile review found no
  remaining actionable issues.

- **2026-10-06, Banach the 2nd, BufferRef and MTP storage hostile review and
  follow-ups:** Found that slicing a buffer whose Metal GPU address is
  unavailable (`0`) synthesized a nonzero address from the slice offset. A
  follow-up found sampler and mat-vec validators did the same when checking
  native metadata. Shared offset validation now preserves zero; sampler and
  mat-vec use it, overlap checks ignore the sentinel, and BufferArena skips GPU
  alignment checks when no address exists. Full-mode and tiered raw-address
  layout builders reject a missing base instead of publishing an offset as a
  pointer. Regressions cover zero/ordinary/overflow address arithmetic and
  address-base refusal. The review also found Embed/PLE test descriptions
  overstated addressless overlap coverage: tests reject forged metadata as
  unowned, while actual overlap checks use valid bounded slices. The docs now
  distinguish these cases. The final hostile follow-up found no additional
  actionable issue.

- **2026-10-06, Banach the 2nd, MTP bind layout hostile review and
  follow-up:** The first pass found that an arbitrary positive reserved tail
  could conceal a missing or undersized region, that the sized device
  vocabulary table was not initialized, and that the separate `MtpParams`
  array was absent from runtime planning. The layout now enforces the exact
  shape-dependent tail, accounting only for the known ID-table, logits and
  attention-scratch alignment padding; the runtime copies the inspected IDs
  into the bounded device view before publication. `MtpParams` remains a
  separate pre-dispatch blocker tracked under REV-034. The follow-up confirmed
  the tail arithmetic, copy guards and documentation. An additional
  follow-up verified identity initialization for `mtp_ids` and that the owner
  exposes only named subviews; no production caller used the removed raw-arena
  accessor. The reviews found no remaining actionable issues and ran no tests.

- **2026-10-06, Banach the 2nd, MTP parameter planning hostile review and
  follow-ups:** The first pass found that a no-prefill speculative plan lacked
  the reusable MTP group slot used after short-read decode windows, and that
  `mtp_params_mib` shifted the stable PLAN key order. The planner now includes
  a nine-slot baseline, charges only extra chunk groups to prefill admission,
  and appends the new key after prior keys and any `reason`. Follow-ups found
  double-counting in the documented full-mode and anonymous-RAM formulas and
  one stale name for the nine-slot baseline; the formulas now count the base,
  group delta and prefill scratch once and use `mtp_params_base_bytes`
  consistently. The final review found no remaining actionable findings and
  ran no tests.

## Verification recorded

- **2026-10-06, reference Mac:** the main `KvStore` composition build
  completed. `metal_kv_layout_test`, `metal_window_storage_test`,
  `metal_mtp_storage_fixture` and `metal_mtp_storage_test` passed 4/4,
  including a live speculative-plan `ModelStorage::open` refusal at a
  one-byte-smaller `maxBufferLength`, Full/Tiered transaction rollback and
  bounded shared-RoPE ownership. The full
  `ctest --test-dir build-dev -L 'host|metal' --output-on-failure -j8`
  passed 61/62; the remaining model-real test was skipped because its assets
  are unavailable. After adding the live-limit test seam, its targeted
  Context/MTP/fixture/Full-Tiered selection passed 4/4 and the full host/Metal
  suite again passed 61/62 with the same expected skip. `git diff --check`
  passed.
- **2026-10-06, reference Mac:** after MTP parameter-slot planning and storage,
  the full build completed and `ctest --test-dir build-dev -L 'host|metal'
  --output-on-failure -j 8` reported 0 failures across 62 tests; 1 model-real
  input test was skipped because assets are unavailable. The focused planner,
  fixture and Metal MTP storage tests passed 3/3. Hostile review findings on
  short-read slot sizing, append-only PLAN keys and documentation arithmetic
  were fixed and re-reviewed.
- **2026-10-06, reference Mac:** `cmake --build build-dev -j8` completed.
  `ctest --test-dir build-dev -L 'host|metal' --output-on-failure -j8`
  reported 0 failures across 62 tests; 1 real-input test was skipped because
  its model assets are unavailable.
- **2026-10-06, reference Mac:** after adding identity initialization for
  `mtp_ids` and removing the public raw bind-arena accessor,
  the targeted build of `metal_mtp_storage_test`, `metal_weight_store_test`
  and `metal_mtp_bind_layout_test` completed; the focused MTP
  geometry/storage/planner CTest selection passed 4/4.
- **2026-10-03, reference Mac:** after H2 memory-probe and geometry-backed
  plan-only CLI changes, `cmake --build build-dev -j4` completed and
  `ctest --test-dir build-dev -L host --output-on-failure` passed 21/21.
  The targeted geometry CLI, planner, options, test-map and memory-probe tests
  passed 5/5. The Metal-labeled device tests were not part of this run.
- **2026-10-03, reference Mac:** `tools/metal/run_python_tests.py` — 37/37 modules
  exited successfully. Existing skips for absent model packs and optional
  `jsonschema` remain conditional.
- **2026-10-03, reference Mac cross-configured as Linux x86-64:** the pinned CMake
  and Ninja configure produced matching graphs: 57 compile commands and 1,787
  `build.ninja` lines. The pinned llama.cpp source and build tools were placed
  outside both source roots so the comparison tests the two trees equally.
- **2026-10-03, reference Mac:** `tools/metal/test_compare_build_graphs.py` —
  20/20 tests passed, including embedded path prefixes, argument boundaries,
  standalone compiler prefix-map options and trailing slashes.
- **2026-10-03, reference Mac:** `tools/metal/test_check_drift.py` — 14/14
  planted-change cases passed; the baseline check at `UPSTREAM_SYNC` reports
  `golden: still valid`.
- **2026-10-03, reference Mac:** `tools/metal/test_test_map.py` — 28/28 tests
  passed. The real `test_map.py` checker passes while reporting the currently
  planned tests and requirements without checks.
- **2026-10-03, reference Mac:** `tools/metal/test_check_release_policy.py` —
  33/33 tests passed, including paginated history, exact revision increments,
  repository fetch/push identity, annotated tag and retained-branch-head
  checks, malformed tags, invalid API responses, and release-PR head
  verification.
- **2026-10-03, reference Mac:** `tools/metal/test_workflows.py` — 11/11 tests
  passed for YAML structure, pinned Actions, job-level read-only permissions,
  runner triggers, fail-closed acceptance gates, dispatch ordering and
  validation, independent weekly builds, hardware/release queue recovery and
  Dependabot cadence.
- **2026-10-03, reference Mac:** `tools/test_strata_mcp.py` — 42/42 tests
  passed, including Metal plan matching and process-group cleanup cases.
- **2026-10-03, reference Mac:** `mac-dev` built successfully; `mac-host` passed
  17/17 tests and `mac-metal` passed 5/5 both normally and with
  `MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1`. The normal context test passed
  cold, warm and damaged-archive recovery; the validation run passed direct
  pipeline compilation and the selftest. The fault test reported the
  out-of-bounds `MTLFunctionLog` and recovered on the following command buffer.
- **2026-10-03, reference Mac:** the current `build-dev` completed, and
  `ctest --test-dir build-dev -L host --output-on-failure` passed 20/20 after
  the geometry and planner changes. The geometry review found that
  zero-padded shard IDs could alias canonical IDs; both shard key forms now
  reject non-canonical decimal spellings, with independent name and size
  regressions. The planner review found no remaining concrete deviation from
  05 §3. Real-file and Prefill allocation parity remain open under REV-007.
- **2026-10-03, reference Mac:** after H5 WeightStore, role-specific tensor
  validation, Swift and v4 split fixtures, MTP-window sizing, draft-vocabulary
  validation, atomic geometry export and input-alias protection,
  `cmake --build build-dev -j4` completed and
  `ctest --test-dir build-dev -L host --output-on-failure` passed 23/23.
  The targeted plan, WeightStore, CLI and test-map tests passed 5/5. The
  adversarial review's concrete findings were fixed and rechecked; live
  catalogue model timing and geometry-versus-real PLAN parity remain open
  under REV-007.
- **2026-10-03, reference Mac:** after H6 transcript, SVE1 resource, input
  bound, callback and EOF-ordering changes, `cmake --build build-dev -j4`
  completed with `STRATA_WERROR=ON`; `serve_transcript_test` passed 20
  consecutive runs, `ctest --test-dir build-dev -L host --output-on-failure`
  passed 24/24, `tools/metal/test_map.py` passed, and `git diff --check` was
  clean. Coverage includes fixed transcripts, partial SVE1 headers, the
  256 MiB image bound, line/queue limits, PP/EOS invariants, STOP/DONE order,
  active and preclosed EOF, and signals. The upstream Python server suite and
  production `strata --serve` path remain pending the real Window/request
  backend under REV-011.
- **2026-10-03, reference Mac:** after the final H6 input synchronization and
  drain-bound changes, `cmake --build build-dev -j4` completed with
  `STRATA_WERROR=ON`; the host suite passed 24/24 and
  `serve_transcript_test` passed 10 consecutive runs. `test_map.py` and
  `git diff --check` passed. The Metal `strata-vision` target built, its
  no-argument invocation printed the specified usage and exited 2, and
  `sysctl -n hw.perflevel0.physicalcpu` returned 6. A model-backed vision
  launch is unavailable because this checkout has no model or mmproj files
  (REV-013).
- **2026-10-03, reference Mac:** the H7 formatter built with
  `STRATA_WERROR=ON`; `oneshot_output_test` and the full host suite passed
  (25/25), `test_map.py` and `git diff --check` passed. The hostile review's
  missing MTP line was added and retested. This validates formatting only;
  production one-shot inference and the `--stats` runtime remain open under
  REV-015, and real Metal timing boundaries remain open under REV-016.
- **2026-10-03, reference Mac:** U2's host planner built with
  `STRATA_WERROR=ON`; the host suite passed 26/26, and `metal_window_test`
  passed 10 consecutive runs on the Q2_0 and Swift R1 header fixtures. Coverage
  includes aligned and unaligned expert splits, merged-gap preservation, the
  combined 252,149,760 B Swift router-conversion total plus page padding, and
  PLE mapping exclusion. `metal_weight_store_test` verified the transient range
  handoff and 512-slab expansion. `test_map.py` and `git diff --check` passed.
- **2026-10-03, Goodall, U3 read-only review:** Flagged a possible Coder row
  stride mismatch because the memory document describes maximum `[48][512]`
  capacity while the kernel ABI specifies compact `l * n_expert + e`
  indexing. Kept the kernel ABI, made table capacity and active indexing
  explicit in the memory document, and added a two-layer 256-expert test.
  Also found that production startup does not call `FullModeStorage`, converted
  Swift routers and MTP runtime files are not included in it, and the live
  warm-up fixture exercised only one page. The first two remain open U3 work;
  the fixture now spans multiple pages, a pure test checks profile order across
  eight-expert wave boundaries, and `FullModeStorage::open` verifies the exact
  plan before allocation. Cold-page order remains unverified.
- **2026-10-03, Erdos, U3 read-only review:** Found that the GPU checksum
  allocation was not committed to residency before dispatch. Added a residency
  commit before submitting the warm-up command. Also found that the live test
  stopped before the second eight-expert wave. The live fixture now uses nine
  experts, and a deterministic nine-item scheduler test verifies that all
  first-wave work is complete at the next-wave boundary. Production startup,
  converted Swift routers, MTP runtime mappings and reference-model M-5 remain
  open under REV-017 and REV-020/021.
- **2026-10-03, Rawls, U4 read-only review:** Found that the loader did not
  require a complete tensor inventory, validate converted exclusions as
  routers, reject overlapping GGUF payloads, or keep architecture and memory
  documentation aligned with the incomplete tiered implementation. Added
  explicit PLE/conversion exclusions, exact inventory and dense-pack byte
  checks, F32 router-role/shape validation, payload overlap checks, and
  corrected the stale documentation. The follow-up review found no confirmed
  bugs; added loader-path coverage for a valid converted router and a plan
  whose window would cross the PLE payload. Host 26/26, Metal 8/8,
  `test_map.py` and `git diff --check` passed. Tiered mode remains open under
  REV-022.
- **2026-10-03, Bernoulli, U5 metadata read-only review:** Found that the new
  dense-pack handoff test did not compare `file_offset` with the index row.
  Added an independent index-row comparison and an in-file range check; the
  follow-up review confirmed the finding is resolved. `metal_weight_store_test`
  passed.
- **2026-10-03, Bernoulli, U5 adversarial follow-up:** Confirmed that same-size
  GGUF and dense-pack path replacements, source/destination length mismatches,
  and a declared-versus-actual dense-pack size mismatch are rejected before
  Metal allocation. The review also found that the region-name assertion
  treated `filesystem::equivalent` errors as success; it now fails closed.
  A later focused pass found that storage-only locks began after inspection,
  leaving a cooperating-writer race. WeightStore now locks the shard and
  `dense.bin` before opening mappings or parsing pack metadata, carries those
  leases in `WindowSources`, and TieredStorage verifies them before taking its
  own descriptors. New negatives cover a writer already holding an exclusive
  lock and verify that inspection leases block a writer through startup.
  Advisory-lock-ignoring writes remain open under REV-010. The reviewer
  initially proposed 95 catalog tests; comparison against `test_map.py` showed
  `metal_tiered_copy_test` was already counted as a planned catalog row. The
  correct configured/static total remains 94 (36 host, 30 Metal, 28 model).
- **2026-10-03, Bernoulli, U5 generation-lock review:** Found a mixed-generation
  race across sequential shard/pack locks, setup recovery that could accept an
  old completion marker after a crash, and a child-packer lock reacquisition
  deadlock. Added source-set and pack sibling locks with fixed ordering,
  inherited descriptors for setup children, a durable pending-generation
  marker, and forced repacking on recovery or source changes. `WeightStore`
  rejects pending generations and `TieredStorage` retains the shared set locks.
  Follow-up found orphaned nested tokenizers and an incomplete transaction
  around the shared MTP runtime. Nested POSIX writers now inherit the locks;
  Windows runs every setup writer in-process. MTP generation now has its own
  lock, pending marker, output validation and reader-side exclusion. Native
  Windows execution and forced process interruption remain open under REV-023.
- **2026-10-03, Feynman, U6 slot-pool review:** Found that a duplicate request
  could observe a slot before its first read completed, and that the delayed
  GPU test checked before the GPU had reached its event wait. Added
  Loading/Ready publication and a GPU-arrival event. Follow-up found a timeout
  cleanup path that could strand the committed command buffer; the test now
  signals the release value and drains it before failing. Final re-review is
  pending.
- **2026-10-04, implementation review log:** Replaced the ORCA and Unsloth
  manual draft-vocabulary copy with the locked atomic publisher, after tracing
  the original `cp` as a path around MTP reader exclusion. The explicit source
  path and fail-closed marker behavior are recorded in D-IMPL-066.
- **2026-10-04, Bernoulli, U5 MTP adversarial re-review:** Found that a moved
  symbolic ref could keep the same URL, a pin fallback could retain stale
  request metadata, pinned hashes could be applied to fallback bytes, a
  standalone conversion could leave a stale source stamp, and runtime files
  were not fsynced before marker removal. Inventory and per-tensor partial-file
  metadata now include the immutable Hub commit and requested ref; symbolic
  refs without that commit fail closed, only matching resolved pinned bytes use
  pinned hashes, and standalone output is fsynced before invalidating its stamp
  and clearing the marker. Focused Python regressions cover the findings.
  Native Windows and process-kill recovery remain open under REV-023.
- **2026-10-04, Feynman, U6 slot-pool final review:** Rechecked layer-level
  retirement, coordinate validation, cached-page prefetch, and callback
  synchronization; no actionable P1/P2 findings remain. Production layer
  hookup and model parity remain open under REV-024.
- **2026-10-04, Hooke, U5 MTP adversarial review:** Found Windows sync handles
  were opened read-only, a standalone pack could enter a pending runtime
  generation without both setup locks, tensor shape checks accepted
  same-element-count transposes, and runtime geometry relied on assertions
  removed by `python -O`. The publisher now opens files read-write before
  `fsync`, validates all 31 expected BF16 source names and exact shapes plus
  per-tensor hashes, requires inherited source and runtime locks to consume a
  pending setup generation, and uses explicit runtime validation errors.
  Focused coverage exercises each case and the Windows sync dispatch; the
  final review found no actionable P1/P2 issues. Native Windows execution and
  production setup interruption remain open under REV-023.
- **2026-10-04, Hooke, U5 forced-interruption test review:** Confirmed the
  child publishes its partial artifact only while holding the generation
  locks, and that after process termination the test verifies marker
  persistence, lock reacquisition, and the recovery decision. No actionable
  correctness or portability findings.
- **2026-10-04, Schrodinger, U9 adversarial review:** Found that LOOCV
  fingerprinted source files and then reopened them for training and
  evaluation, allowing post-validation replacement; also found that opening a
  FIFO could block before the regular-file check. The implementation now
  snapshots each opened regular file once, verifies its identity and metadata
  during copying, fingerprints and uses only the private snapshots, and opens
  nonblocking where supported. Regressions mutate an original trace after
  validation and pass a FIFO to a timed subprocess. Final re-review found no
  actionable P1/P2 issues.
- **2026-10-04, Aristotle, L1 adversarial review:** Found that the first
  `metal_math_test` draft used U(16) or U(32) for every helper, although the
  L1 test map requires twice each helper's A.10 bound. Tightened the full-grid
  checks to U(2) for `exp`, U(8) for `sigmoid_k` and `silu_k`, U(10) for
  `softplus_k`, and U(6) for `log1p_k`. The rerun must pass all 2^24 inputs at
  these limits before L1 is accepted.
- **2026-10-04, Euler, L2 hostile review:** Found four P2s: unwritten values
  could pass mixed byte ranges or NaN comparisons, the all-`0xFF` sentinel
  narrowed the B/U/R contract, and dispatch did not preserve exact
  threadgroup/grid geometry. Replaced sentinel reservation with per-element
  oracle-derived poison, added no-write checks for all-`0xFF` bytes and NaN
  references, made the grid mode explicit, and recorded exact manifest axes.
  The 40-test host/Metal suite and validation-layer twin selftest pass. Euler's
  follow-up static review found no remaining P1/P2 issues. Kernel-specific
  fixtures, quantized-weight synthesis and model-backed inputs remain open
  under REV-025.
- **2026-10-05, Schrodinger the 2nd, Prefill metadata hostile review:** Found
  a P3 test gap because only `kPrefillMaxShaderRows + 1` was rejected; the
  exact shader-row limit lacked an acceptance case. Added a test that builds
  exactly 419,430 rows and checks the final live `StepRow`, while retaining
  the +1 rejection. The focused build and CTest pass. Follow-up confirmed the
  finding is closed; the review was read-only and did not rerun tests.
- **2026-10-05, Rawls the 2nd, QSA norm/RoPE hostile review:** The initial
  read-only review found three P2 items: the test inventory overstated the
  implemented kernels, no model-backed real-input run was available, and
  Metal allocation-end/GPU-address rejection paths lacked tests. The inventory
  now lists only norm/RoPE and the remaining L6 kernels as open. Added tests
  for an input view extending past its allocation, forged GPU-address metadata,
  and an incomplete RoPE row. Real-input parity remains open under REV-002/025:
  this target has no model-backed fixture and explicitly returns skip code 77
  for `--input-set real`. The focused QSA twin passes normally and with Metal
  API/shader validation enabled. Follow-up confirmed the coverage fixes and
  found no other actionable P1/P2 issue; the reviewer ran no tests.
- **2026-10-05, Lorentz the 2nd, QSA FWHT256 hostile review:** The initial
  read-only review found a P2 oracle-independence gap and a P3 gap in FWHT
  CPU-metadata and submitted-command-buffer rejection coverage. Rewrote the
  oracle as a contiguous 256-element butterfly, added analytic basis goldens
  crossing dimensions 31/32, 63/64, 127/128 and 255, a dense scalar-spec
  output digest, and direct CPU-metadata/committed-command-buffer rejection
  cases. Follow-up confirmed the fixes and found no other actionable issue.
  The requested CUDA-captured golden cannot be produced without the CUDA
  reference assets; external parity remains open under REV-025. The focused
  normal and API/shader-validation tests and full CTest pass; the reviewer ran
  no tests.
- **2026-10-05, Beauvoir the 2nd, QSA fp16 KV append hostile review and
  follow-ups:** The first read-only pass found a P2 race if two rows named the
  same cache cell, plus P3 gaps in per-head/token edge inputs and partial
  alias coverage. The encoder now rejects duplicate positions; tests vary
  edge values by row and reject partial same-handle and cross-handle no-copy
  overlaps. Follow-up found that offset-adjusted alias fixtures must preserve
  a zero native GPU address; those fixtures now do, and the final review
  confirms the prior findings are resolved with no further actionable issue.
  The review was read-only and ran no tests. The full host/Metal CTest passed
  51 tests with one expected model-input skip; final focused QSA runs pass
  normally and with Metal API/shader validation enabled.
- **2026-10-05, Heisenberg the 2nd, QSA int8 KV append hostile review and
  follow-ups:** The first pass found that validation and positive GPU tests
  were not called from `main`; these were wired up with T=1..8 coverage and
  four disjoint output views over one backing allocation. The follow-up found
  that output buffers initially reused the oracle's expected values, allowing
  a no-store kernel to pass. Tests now seed every written code/scale with a
  guaranteed-different poison and explicitly check that no-op baseline.
  Finite K/V input requirements and the implemented-vs-planned kernel
  inventory were also clarified. Final review found no remaining actionable
  issue. The focused test, Metal API/shader validation, `test_map.py`, diff
  check, and full host/Metal CTest pass; one model-input test is skipped.
  The review was read-only.
- **2026-10-05, Godel the 2nd, QSA q4_0 KV append hostile review and
  follow-ups:** The first pass found that the public API comment did not
  require K/V FWHT and q4 append to be encoded earlier in the same command
  buffer, and that q4 rejection tests lacked distinct no-copy handles over
  shared backing. The API contract now requires same-command-buffer FWHT
  ordering, and tests reject cross-handle partial overlaps for output/output
  and output/input views. Follow-up found both issues resolved and no new
  actionable finding. Focused tests pass normally and with Metal
  API/shader validation; the full host/Metal CTest passes with one expected
  model-input skip. The review was read-only.

- **2026-10-04, reference Mac:** after U5 artifact-generation recovery,
  U6 tiered ExpertStore reads and lease retirement, and U9 profile/hit-curve
  analysis, `cmake --build build-dev -j4` completed and
  `ctest --test-dir build-dev -L 'host|metal' --output-on-failure` passed
  37/37. `tools/metal/run_python_tests.py` passed 42/42 modules; the focused
  U5/U9 suite passed 45 tests with one Windows-only test skipped.
  `tools/metal/test_map.py` passed with `STRATA_BUILD_DIR=build-dev`;
  `py_compile`, `git diff --check`, and the staged diff check passed.
  These runs cover synthetic storage and host paths. Production setup
  interruption, native Windows execution, real-model inference parity and
  the production ExpertStore/kernel hookup remain open under REV-023/024 and
  the model/hardware review items.

- **2026-10-05, Herschel the 2nd, ModelStorage hostile review:** The first
  pass found that the owner did not bind every plan byte count to the supplied
  geometry, could miss an unlisted dense pack, and did not reject MTP KV-only
  plans. The next pass found that a caller could understate both geometry and
  plan against actual GGUF payload sizes, full mode did not require the
  WeightStore inspection locks, and failed opens left provisional buffers in
  the Context registry. All six findings now have fail-closed checks and
  regressions; buffer-pool transactions also roll back across full-mode
  warm-up. A final hostile follow-up was requested after these fixes but has
  not returned yet. The reviewer ran no tests; the full host/Metal CTest passes
  58/58 with the model-input test skipped, as recorded under D-IMPL-159/161.
- **2026-10-05, Herschel the 2nd, ModelStorage follow-up:** Found a P2 because
  a caller could forge the public `shared_ptr<void>` lock lifetime while
  copying correct path metadata, and a P3 because nested rollback after
  FullModeStorage committed was not directly covered. Replaced the lifetime
  with a descriptor-owning `WindowSourceLockLease`, validated the lease
  against the source identity in WeightStore, full-mode and tiered paths, and
  added regressions for forged metadata and inner-open/outer-rollback.
  The focused build and storage/WeightStore CTests pass. A fresh read-only
  hostile re-review is pending; the reviewer ran no tests.
- **2026-10-05, Fermat the 2nd, lock-lease hostile follow-up:** Found that a
  public `acquire_shared` factory still let callers acquire a new lock after
  inspection and attach it to stale `WindowSources`. Moved lease issuance
  behind the private `WeightStore` implementation and added
  a compile-time check that public callers cannot acquire leases. The focused
  build and three storage/WeightStore CTests pass. A fresh read-only review is
  pending; Fermat confirmed the tiered validation order and nested rollback
  regression, and ran no tests.
- **2026-10-05, Locke the 2nd, lock-lease hostile re-review:** Found that the
  synthetic test issuer was a friend in production headers, allowing an
  in-tree caller to define the same class and reissue leases. Guarded the
  friend declarations with `STRATA_METAL_TEST_LOCK_ACCESS` and moved the
  storage fixture executable to a separate engine build carrying that macro.
  The shipped engine target is compiled without it. The full host/Metal CTest
  passed 58/58 with the model-input test skipped; the focused lock/storage
  tests passed 3/3 after making the macro private to both test targets.
  Hypatia the 2nd found no issues in the target isolation or follow-up CMake
  change. Both reviewers ran no tests.
- **2026-10-06, hostile Encoder and platform review:** The first pass found
  persistent Metal slots could reuse stale bindings, and requested a true
  post-dispatch exception case and fresh error diagnostics. Added reflected
  exact-slot validation, stale-slot clearing, and those regressions. The next
  pass found repeated IORegistry/sysctl probes in dispatch, missing cumulative
  4 KiB boundary coverage, and no archive-hit reflection test. Cached the
  immutable `DeviceFacts` snapshot at Context creation, added exact-4-KiB and
  4,097-byte cumulative cases, and checked active/omitted slots after an
  archive hit. The final pass found no additional code issues; it noted stale
  20-marker metadata, which was updated to 22 and verified with
  `check_drift.py`, and the old appendix result, now labeled historical.
  Separately, the Darwin CUDA guard was checked to fail before CUDA toolchain
  detection. The reviewer made no edits and ran no tests; the reference Mac
  build and host/Metal test results are recorded under D-IMPL-178/179.
- **2026-10-06, Jason the 2nd, D-IMPL-184 hostile review:** The read-only pass
  found no runtime accounting or rollback defects. It found that the planner
  test did not explicitly assert the exact subtotal fits `fixed_bytes`, and
  that the decision omitted the `qwen4exp` family label used by direct
  `WeightStore` inspection. Added explicit allowance assertions for Qwen and
  Coder plans and documented the default label. Follow-up requested pinning
  the allowance itself to 256 MiB; the planner test now checks that value for
  both package families. Final review confirmed both fixes and found no new
  actionable issues. The reviewer ran no tests; validation is recorded under
  D-IMPL-184.
- **2026-10-06, Bernoulli the 2nd, WindowPleRows hostile review and
  follow-ups:** The initial pass found that an arbitrary vocabulary size could
  admit token IDs outside the supported model. The builder now accepts the
  checked `Geometry` and requires the supported qwen4exp runtime dimensions
  and vocabulary; tests reject mismatched vocabulary and family labels. The
  follow-up found reliance on a transitive `<limits>` include, which is now
  included directly. The read-only reviews ran no tests; validation is
  recorded under D-IMPL-185.

## Open review items

L3 coverage added since REV-026 was opened: IQ2_XXS now has a standalone Metal
row-dot twin for the routed-expert path. Its tests cover n_in 256/512/768/1024/
1280/1792/2048/2304/2560/6144/10240/16128/16384/16640, NC 1/2/4/8 and every
live-row count; all 256 codebook indices, all eight signed components and 16
scale nibbles in a generated raw-word sweep; a separate fixed-u16 fixture pins
packed bytes to eight known one-hot answers. The tests also cover signed `/8`
truncation; independent 9/13-block FMA and 63/64/65-block answers;
padded-column sinks and multi-column parity. Dense inference does not dispatch
this expert-only twin. L3 remains open for the other planned quantized formats,
production dispatch, model-backed real inputs and remaining exhaustive
decoder/property coverage.

IQ2_XS also has a standalone expert row-dot twin. Tests cover n_in
256/512/768/1024/1280/1792/2048/2304/2560/6144/10240/16128/16384/16640,
NC 1/2/4/8 and every live-row count. A generated sweep covers all 512 grid
indices, both signs of all eight components and all values of both scale
nibbles. A separate fixed block pins 32 packed u16 words, eight scale bytes
and eight one-hot outputs. Signed `/2` and `/4` truncation, padded-column
sinks, independent 9/13-block FMA answers, 63/64/65-block outputs, NC parity
and a weight-bit negative control are covered.

IQ2_S also has a standalone expert row-dot twin. It covers the same n_in widths,
NC 1/2/4/8 and every live-row count. A generated raw-byte sweep checks all 1024
grid indices, every qh two-bit field value, both signs at all eight components,
and every value of both scale nibbles. A fixed packed block checks eight literal
one-hot answers across all components, including the last codeword in the
82-byte block. Signed `/2` and `/4` truncation, padded-column sinks,
independent 9/13-block FMA answers, 63/64/65-block outputs, NC parity and a
weight-bit negative control are covered. Dense inference does not dispatch
this expert-only twin.

IQ3_XXS now has a standalone expert row-dot twin. It covers the same n_in
widths, NC 1/2/4/8 and every live-row count. A generated raw-byte sweep checks
all 256 grid indices, all 128 sign indices, both signs at every one of the
eight components and every scale nibble. A separate fixed 98-byte block pins
all 32 one-hot answers in its final part. Signed `/2` truncation, padded-column
sinks, independent 9/13-block FMA answers, 63/64/65-block outputs, NC parity
and a weight-bit negative control are covered. Dense inference does not
dispatch this expert-only twin.

IQ1_M now has a standalone expert row-dot twin. It covers the same n_in widths,
NC 1/2/4/8 and every live-row count. Four generated sweeps exercise all 2048
grid indices, both delta selectors, all eight one-hot components and every
value of both local scales. A separate fixed raw-byte block pins the final
32-value part, all four packed grid indices, and 32 one-hot answers. Independent
9/13-block FMA answers, 63/64/65-block outputs, padded-column sinks, NC parity
and a code-index negative control are covered. Dense inference does not dispatch
this expert-only twin.

The MTP draft-vocabulary head now has a validated engine encoder and indexed
Q5_K, Q6_K and IQ4_XS kernels. Synthetic engine tests cover its unsorted row
map and map/command-buffer contracts. Wiring stage 21 into MTP inference and
measuring L-15 remain open.

| ID | Item | Why it needs review | Blocking point |
| --- | --- | --- | --- |
| REV-001 | Confirm GitHub owner/repository and release test repository | OQ-01 leaves both open; no remote is configured | Before external URLs and release workflows are used |
| REV-002 | Confirm model assets/result storage and CUDA reference availability | The local Mac matches the required compute specs, but no model disk is mounted and no CUDA PC is available | Before model/golden tests and acceptance gates |
| REV-003 | Review self-hosted runner isolation and workflow permissions | B21 requires a second-person review | Before the first workflow runs on `strata-ref` |
| REV-004 | Verify independent local history | The rewritten `codex` history must have no Strata commit ancestry while retaining a clear source-provenance record and every project tree | Before publishing or opening a pull request |
| REV-005 | Eliminate or accept the POSIX PID/PGID signal reuse race | Identity verification and PID or process-group signaling are separate operations; post-leader cleanup no longer signals an unverified group, but a leader can still exit and its PID/PGID can be reused between verification and signaling | Before claiming process-stop identity is race-free |
| REV-006 | Decide whether `--expert-cache-per-layer` should admit zero experts when the first pass finds fewer than 48 | The literal documented quota is `floor(|R| / 48)`; clamping to one would change residency policy and memory use | Before treating per-layer admission as final |
| REV-007 | Validate planner PLAN lines against every catalogue geometry and the live model files | H5 now has a complete synthetic 1,224-tensor WeightStore path, including the Swift 342/882 split, and CLI geometry export. The reference model packs are not present here, so per-model live PLAN timing and geometry-versus-real parity remain unverified | Before accepting U1/H5 or claiming section 8 golden coverage |
| REV-008 | Independently verify Darwin available-memory probe behavior | The host test checks that the sample is present and within physical RAM but does not independently recalculate the `host_statistics64` page total; zero remains a valid low-memory reading | Before treating H2's memory-probe verification as complete |
| REV-010 | Decide whether source payload integrity needs trusted hashes or writer coordination beyond advisory locks | PLAN and geometry export validate conversion manifest declarations and hash syntax without reading tensor payloads; generation locks cover in-repo setup and pack writers, but advisory locks cannot stop an external writer that ignores them | Before claiming source-payload conversion integrity against concurrent or malicious writers |
| REV-011 | Connect the host serve protocol to the real Metal model backend | Window, Prefill, sampling, model-backed startup INFO/DONE values and H11 watchdog are not implemented; `main.cpp` must not emit a false READY from the fake | Before accepting H6 or running upstream `serve/test_*` against `strata` |
| REV-012 | Confirm the D-IMPL-046 context and vision payload ceilings | The 262,144-token context and 256 MiB aggregate SVE1 payload limits are conservative host bounds; supported YaRN configurations and vision encoder output need validation before either is raised | Before advertising Metal serve support beyond either bound |
| REV-013 | Verify H13 thread count in a model-backed `strata-vision` launch | The checkout has no text-model or mmproj files, so the program's startup thread-count line cannot be checked against the sysctl result in this environment | Before marking H13's one-off runtime check complete |
| REV-014 | Confirm the 64 MiB completion-drain ceiling against expected server traffic | The hostile review verified exact-boundary handling and STOP/EOF ordering; expected server request sizes are not measured in this checkout | Before treating the H6 input-completion policy as final |
| REV-015 | Connect the one-shot formatter to a real Metal generation loop | `Window`, `Prefill`, sampling and model loading are not implemented; the formatter is intentionally not wired into `main.cpp` | Before accepting H7 or REQ-FUN-12 |
| REV-016 | Verify `gpu window` timing boundaries on a real command buffer | The formatter avoids adding GPU time twice to commit-to-wake wait, but no Metal window exists to measure those intervals here | Before using one-shot Metal timing lines for performance decisions |
| REV-017 | Complete production model startup with validated WeightStore mappings, converted-router storage, MTP/PLE runtime buffers and request dispatch | `ModelStorage` owns the selected Full/Tiered weight storage, exact converted-router rows in a full-mode anonymous arena, planner-sized main QSA KV/indexer/shared-RoPE allocation, and exact planned Window scratch under one outer transaction; the planner reports exact session-state and Window-scratch subtotals inside its conservative fixed allowance. `--storage-check` exercises full-mode warm-up or tiered copies and speculative MTP composition. Inspection-lock claims use descriptor-owning leases checked against file identities. This diagnostic is storage-only and never sends READY; it leaves K/V and RoPE values uninitialized, while the normal driver has no model startup route and session state, PLE runtime, MTP dispatch and request dispatch remain unimplemented | Before accepting U3 or sending READY for a full-mode model |
| REV-018 | Confirm page-padding overlap at the excluded Swift PLE boundary | The R1 header places the preceding tensor end off a 16 KiB boundary; outward rounding includes fewer than 16 KiB from the PLE edge, though the PLE has no `WindowSlice` | Before claiming the PLE exclusion against live no-copy Metal buffers |
| REV-019 | Confirm the full-mode resident-table integer encoding against the production plan kernel | The host layout currently uses `1` for each all-resident entry; the plan kernel and its encoding contract are not implemented yet | Before connecting the table to a production plan kernel |
| REV-020 | Verify profile-prioritized page warm-up and M-5 on the reference model | The synthetic test executes both CPU and GPU warm-up, but the checkout has no model pack or real STRP profile for cold/warm timing and page-order observation | Before accepting U3 or changing the warm-up scheduling policy |
| REV-021 | Verify CPU profile-order waves under cold file-page conditions | Synthetic coverage now checks scheduling order across wave boundaries and runs CPU/GPU warm-up over a multi-page mapping; actual OS page-in order still needs observation on the reference Q2_0 model | Before accepting M-5 |
| REV-022 | Reject every GGUF no-copy mapping in tiered mode | `metal_tiered_copy_test` byte-compares copied data and checks all six tiered buffer address ranges against the live GGUF mapping; the GGUF mapping itself is identified by `proc_regionfilename`, and tiered storage retains source-set, pack and file locks. The initial region-name check on Metal buffer contents was removed under D-IMPL-260 because Metal can report an unrelated file for an anonymous no-copy buffer. U5 follow-up review found and prompted fixes for fail-open source-path comparison, same-inode writes, mixed generations and interrupted setup recovery; noncooperating writers and normal model-serving startup integration remain open (REV-010, REV-017) | Re-review the hardened generation-lock, recovery-marker and live Metal allocation assertions before accepting U5 |
| REV-023 | Exercise setup generation recovery under process interruption and concurrent setup | Unit and CTest coverage checks lock exclusion, inherited descriptors through the nested tokenizer, in-process Windows dispatch for all setup and MTP helpers, interrupted MTP recovery, exact tensor names/dtypes/shapes and hashes, optimized-mode runtime geometry rejection, partial-download resume and marker refusal. A subprocess-kill regression now verifies that a partial artifact and pending marker survive, locks are released, and setup selects recovery. The production setup writer has not been killed during its real GGUF/MTP publication stages or run on native Windows; power-loss durability for the entire setup transaction is not promised | Before accepting the setup transactions across supported setup platforms |
| REV-024 | Connect tiered ExpertStore to production layer handshakes and verify full-versus-tiered parity | Synthetic tests cover reads, slots, counters, prefetch and callback lifetime, but production plan-kernel ABI, resolver signaling, decode lookahead and C4 model parity are not wired or available here | Before accepting U6 or tiered inference |
| REV-025 | Complete kernel-specific twin inputs and model-backed comparisons | The shared harness selftest covers generic random, edge and scalar fixture inputs plus three negative controls. Per-area upstream fixtures and CUDA-dump/model `real` inputs are not wired; the QSA twin explicitly skips `--input-set real`, and its FWHT dense golden and fp16, int8, q4_0 and k8v4 KV append oracles are specification-derived rather than CUDA-captured. K8V4's synthetic chain checks V-only same-command-buffer FWHT and exact int8 K/q4 V output. The model assets and CUDA reference are unavailable under REV-002 | Before accepting the full L2 input-set contract or any kernel twin whose fixtures/weights require those inputs |
| REV-026 | Complete L3 mat-vec coverage | Q8_1 quantization, Q8_0, Q2_0, IQ4_NL, IQ4_XS, IQ3_S, Q3_K, Q4_K, Q5_K, Q6_K, Q4_0, Q5_0 and BF16 mat-vecs have Metal kernels and byte-exact oracles. Q8_0 covers n_in 640/992/1024/2560/6144; Q2_0 covers 64/640/2560/4032/4096/4160/6144/10240; IQ4_NL covers 32/640/2016/2048/2080/2560/6144; Q4_0 and Q5_0 cover 640/2016/2048/2560/6144; IQ3_S covers 256/512/768/1024/1280/1792/2048/2304/2560/6144/10240/16128/16384/16640, including the 3/4/5-, 8/9- and 63/64/65-block lane-stride boundaries; Q3_K, Q4_K and Q5_K cover 256/512/768/1792/2048/2304/2560/6144/10240/16128/16384/16640, including the 7/8/9 row-group and 63/64/65 block boundaries; IQ4_XS covers 256/2560/3840/4096/6144/10240/16128/16384/16640, including the 15/16 row-group and 63/64/65 block boundaries; Q6_K covers 256/512/768/1024/1280/2560/6144/10240/16128/16384/16640, including the 3/4 row-group and 63/64/65 block boundaries. These quantized formats cover NC 1/2/4/8 and every live-row count; their multi-versus-single properties and code-bit/high-bit negative controls pass. IQ4_NL and IQ4_XS check all codebook values and logical positions with raw one-hot inputs and have multi-block FMA-sensitive goldens; IQ4_XS pins raw scale bytes and an analytic padded-column fixture. IQ3_S checks all 512 codebook indices with literal raw packing and one-hot inputs, all qh bit positions, both signs at every value position, all 16 scale nibbles, independent scalar lane/warp 9/13-block FMA answers, 63/64/65-block goldens, padded columns and NC equality. Q3_K checks every signed code across all 256 positions from literal raw bytes and has fixed FMA and 63/64/65-block answers plus an analytic padded-column fixture. Q4_K checks all codes at all 256 positions from literal packed bytes, representative 6-bit scales/minima, independent 9/13-block scalar FMA answers, 63/64/65-block goldens, padded columns and NC equality. Q5_K checks all 32 codes at all 256 positions from literal qh/qs bytes, both high-bit values per group, six-bit scales/minima, independent 9/13-block scalar FMA answers, 63/64/65-block goldens, padded columns and NC equality. Q6_K independently pins every position from literal ql/qh arrays, covers signed scales and has fixed 5/9 and 63/64/65-block answers plus an analytic padded-column fixture. BF16 covers n_in 320/2560/10240, BS 160/256 and NC 1/2/4/8, with its NC property and one-bit negative control. Exhaustive decoder tests, remaining quantized formats, other property families, production dispatch and model-backed real inputs remain open. | Before accepting L3 or connecting a production Metal layer |
| REV-027 | Complete L7 MoE kernels and production integration | Standalone router `r16`/`r8`, plan, routed/shared SwiGLU, grouped Q2_0/IQ1_M/IQ2_XXS/IQ2_XS/IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS gate/up, Q2_0/IQ4_NL down, and shared-gated combine have scalar oracles and synthetic twins. A GPU twin now chains shared-expert Q8_1 input, Q8_0 gate/up, SwiGLU, Q8_1 hidden quantization, Q8_0 down, BF16 gate and combine. Production dispatch and miss-handshake wiring, external CUDA/model-backed parity, and end-to-end full/tiered inference tests remain incomplete | Before accepting L7 or connecting a production MoE layer |
| REV-028 | Complete L8 sampler fixtures and production integration | Host fixture 15's `penalty_rows` layout is ported across 420 boundary cases; Metal-derived semantic ports of upstream cases 16 and 17 preserve tie/sentinel and 17-row parameter/counter coverage with portable integer-generated background logits. They are not byte-identical upstream random vectors. Fixture 16's 262,145-entry fallback exceeds Metal's vocabulary limit and is covered only by shape refusal; fixture 17's draw-counter variants share a Metal queue and do not cover CUDA stream semantics. Literal upstream fixtures 1-14 and 18, external CUDA/model-backed parity, Window integration, head dispatch, production vocabulary-limit refusal and MTP coupled-round publication remain incomplete. Metal command-buffer encoders now expose the sampler primitives and are parity-tested, but do not constitute production inference wiring. Maximum-size greedy scratch, nonzero `kpart`, nonempty-history penalty bypass, test input options and greedy-tail semantics were fixed after hostile review | Before accepting L8 or connecting sampling to production inference |
| REV-029 | Complete L4 embedding/PLE/GR kernels and production integration | Q3_K, IQ3_S, IQ4_XS and BF16 embedding gathers are bytewise checked against ggml CPU; image copies are bit-exact for T=1..8 and GR stream broadcast is checked against raw-bit references. PLE gate, broadcast, conv-residual and history twins cover the documented reduction/order contracts, T boundaries and buffer guards. GR weighted RMS norm, norm, down, up, cvec, post, down-SiLU and pre-gated kernels have scalar twins: T=1..8 coverage where applicable; exact no-apply/inject/no-op paths; U(64) or U(16) arithmetic checks; cvec project/add, active layers 3 and 7 with distinct scales, disabled/uncovered paths and an FMA-sensitive folded-write negative control; GR-up exact normalization and stream-order controls; alias, read-only and guard checks. The weighted RMS norm, cvec, GR norm, GR down, GR up, GR post, GR down-SiLU and GR pre-gated encoders plus the PLE value-broadcast, history, gate and conv-residual encoders, the image embedding and text embedding-row encoders, and the GR stream-broadcast encoder validate their documented grids, buffer sizes/alignment/ranges/overlap and Context/command-buffer ownership; GR down and GR up also check their threadgroup-memory requirements. `encode_embed_rows` checks non-image token ids against the supplied `n_vocab` and reserves bit 31 for image rows; Window/Prefill must pass the model vocabulary and keep token values stable until completion. Production Window/Prefill/MTP dispatch remains open | Before accepting L4 or connecting embedding and GR to production inference |
| REV-030 | Confirm Prefill chunk row and position limits against every prompt kernel | `build_prefill_chunk_metadata` caps rows from `HC_DIM` and signed `StepRow` values. Later prompt kernels are not yet wired through this builder, so their full flattened-index bounds and the eventual planner integration have not been exercised | Before `Prefill::run` accepts chunks or the planner changes its chunk limits |
| REV-031 | Audit Metal queue-fence dependency registration for untracked resources | `Context::open` seeds one queue `MTLFence`; engine and twin-test compute/blit encoders wait/update it, and `CommandBuffer::depends_on` rejects submitting a consumer until an earlier-created producer has completed successfully. Per-command-buffer encoding guards prevent submit races without holding the process-wide submission mutex across encoder callbacks. The wrapper cannot infer read/write dependencies, so every production producer/consumer edge must register explicitly; production Prefill/Window wiring, failure recovery and asynchronous overlap costs remain unverified | Before production Prefill, Window, or any dependent untracked-buffer sequence is submitted |
| REV-032 | Confirm sparse QSA attention capacity and production ID validity | Standalone chunk/merge kernels and encoders pass synthetic oracle tests. The 262,144-cell bound follows the host context ceiling but lacks model-backed configuration validation; GPU-produced live IDs cannot be range-checked by the CPU encoder, and production producer/consumer ordering is not wired. The kernel masks negative, out-of-capacity and future-cell IDs before cache reads | Before production Window/Prefill integration or accepting L6 |
| REV-033 | Confirm exact main and drafter QSA KV allocation sizing | Main `KvStore` and drafter `MtpKvStore` layouts use four-cell capacity rounding and planner byte agreement; `ModelStorage` validates and allocates the exact main layout for all accepted plans under its outer transaction, while the separate MTP store holds K/V and scales, maps k8v4 to plain int8, and leaves indexer/pool state out. Speculative MTP storage now validates and retains the exact main RoPE subview from the same Context allocation; the session must initialize its values before use. Append, production use and model-backed or external-reference comparisons remain open under REV-002/025 | Before connecting `KvStore` or `MtpKvStore` to production Window/MTP or accepting exact KV memory planning |
| REV-034 | Complete MTP bind-region layout and startup composition | `MtpRuntimeStorage` validates geometry-derived weight, bind, 256-byte parameter-slot and K/V extents, copies inspected draft IDs into the device-side table, zeroes parameter slots and rolls all MTP allocations back together. Speculative `ModelStorage` and `--storage-check` compose it with main `KvStore` and Full/Tiered model storage under one outer transaction; the MTP runtime retains the validated borrowed main RoPE view and clears its accessor when invalid. The exact shape-dependent roughly 128 KiB bind tail remains reserved pending ABI reconciliation; session/encoder lifetime, RoPE initialization and dispatch are not established | Before enabling MTP inference |
| REV-035 | Review removal of non-Mac inference source trees and PLE mapping changes | The inactive CPU, HIP, prefill and discrete-GPU trees are absent from the active CMake graph; CUDA sources retained for parity are reference-only. Darwin `MADV_RANDOM` advice and page-size-based optional locking need confirmation on the reference Mac | Before claiming the macOS-only source cleanup is complete |
| REV-036 | Confirm CUDA parity code stays outside the Mac product | CUDA kernels and standalone parity harnesses are reference inputs only; the CMake source graph and Mac CI do not compile them. The external reference workflow still needs the pinned source and a separate CUDA machine for numerical captures | Before claiming Metal numerical acceptance or shipping a release |
| REV-037 | Verify the clean independent Git history | The `codex` branch has a parentless root and English commit messages; license, source notices and provenance remain; retired PC product files, removed fixtures and local build output are absent from every reachable commit | Before publication |
| REV-038 | Confirm removal of unreferenced inherited assets | No retained source or documentation points to `.github/FUNDING.yml`, `docs/paper/`, `docs/media/`, `tests/data/native_experts/` or `bench/results/2026-09-30-community-rtx-5090/`; benchmark records cited by Mac documents and the reference loader remain available | Before publication |
| REV-039 | Verify non-loopback server authentication | CLI rejects a missing or empty API key before engine startup; the server factory rejects unsafe binds before opening a socket; authenticated endpoints retain constant-time key comparison | Before enabling a non-loopback listener |
| REV-040 | Keep host-test registrations intact after relocating test sources | `conversation_cache_test` and `coupled_draft_test` remain registered in CTest and the Metal test map after moving under `tests/core/` | Before accepting the test-tree cleanup |
| REV-041 | Replace platform-bound logits comparison launchers | The new comparer reads captured dump files directly, validates positions and token IDs, keeps memory bounded in chunks, and has synthetic coverage for the documented metrics | Before accepting the comparison-tool replacement |
| REV-042 | Keep engine exit diagnostics evidence-based | Runtime messages, tests, and docs do not infer quarantine, code-signing, or memory-pressure causes from SIGKILL or log growth | Before accepting the diagnostic wording |
| REV-043 | Preserve logits-comparison input integrity and metric meaning | Empty ID fields and out-of-range IDs fail cleanly, JSON output cannot alias an input, and one-row standard error is unavailable | Before accepting the comparer hardening |
| REV-044 | Remove benchmark archives unrelated to the Mac port | No retained source or Mac documentation references the nine retired PC benchmark folders; six referenced result folders remain | Before accepting the benchmark cleanup |
| REV-045 | Verify the model contract without downloading full checkpoints | Seven pinned header snapshots are checked in Python and as compact GGUFs through the production C++ reader; synthetic loader cases mutate protected metadata, type and block geometry | Before accepting R2/R3 or loading the packs produced by Mac setup |
| REV-046 | Keep requirement mappings executable and honest | The new model-type and planner tests exist and pass; the documented Mac setup golden suite is explicitly still future work until its engine-recorded fixtures and test harness are added | Before completing the setup golden-set phase |


### D-IMPL-230: Keep the unified-memory threshold advisory until measured

- **Decision:** `min_free_unified_memory_mib` may print a warning before eager or lazy Metal model loads, but never blocks or delays them.
- **Reason:** The system-wide IORegistry counter and available-RAM figure have not been validated as a safe fit test. The engine's own Metal planner is the authority for whether the model can load.
- **Review required:** Measure the estimate against engine allocations and competing Metal processes on supported Macs before changing it into a gate.

### D-IMPL-231: Fail closed while the native Mac installer is incomplete

- **Decision:** The shell launcher, `setup.py` CLI/API, and MCP installer subprocess refuse to perform installation until a Mac-specific planner and installer are ready. The shell launcher first verifies macOS 27+ on Apple silicon.
- **Reason:** The remaining installer orchestration still contains PC GPU discovery and memory-fit assumptions. Allowing explicit model arguments to enter that code would give users an unsupported, misleading install path.
- **Review required:** Replace the refusal only after Mac planning, model selection, download validation, Metal engine installation, and an end-to-end install/start test pass on the reference Mac.

### D-IMPL-232: Do not attribute every SIGKILL to memory pressure (superseded by D-IMPL-246)

- **Status:** Superseded; retained to record the earlier diagnostic policy.
- **Decision:** Engine exit diagnostics inspect only log bytes written by the current process. A SIGKILL with no new output is a startup-policy clue; with output it names memory pressure as one possibility. Other signals include their symbolic name.
- **Reason:** Exit status -9 identifies SIGKILL, not its sender or cause. Old log lines also cannot establish what happened in a later engine process.
- **Review required:** Compare the messages against macOS Console logs for code-signing blocks, jetsam events, user termination, and ordinary engine crashes.

### D-IMPL-233: Scope the first repository build and installer paths to macOS

- **Decision:** Windows and Linux product setup, build, packaging, monitoring, and CI paths are removed or refused. Retain platform details only where they explain the upstream source baseline or a measured reference.
- **Reason:** The supported product is Apple Silicon with Metal; keeping active PC paths creates unsupported branches and misleading instructions.
- **Review required:** Check remaining historical research references separately from executable or user-facing product support before publishing the rewritten history.


### D-IMPL-234: Keep CUDA as a reference, not a Mac backend

- **Decision:** Remove Windows and Linux product, setup, packaging, and CI paths. Build the product for Apple Silicon with Metal only. Keep CUDA source references only where the Metal port uses them to verify arithmetic contracts; do not build, package, or advertise CUDA on macOS.
- **Reason:** The numerical port compares Metal kernels with the original CUDA behavior, but Macs cannot run the CUDA backend. Keeping those references preserves a testable oracle without retaining a second product path.
- **Review required:** Audit each remaining CUDA reference before release and remove it if no current parity test or documented arithmetic contract uses it.


### D-IMPL-235: Name server memory estimates for unified memory

- **Decision:** Name the server threshold, CLI option, helper, and telemetry function for unified memory. Describe GPU status as a single Apple GPU in the web monitor; do not retain multi-GPU display branches.
- **Reason:** macOS exposes one default Metal device with a recommended working-set limit over shared system memory. VRAM and per-card UI names implied a PC GPU model that the Mac server does not support. The estimate remains advisory and does not decide model fit.
- **Review required:** Verify the displayed Apple GPU name and unified-memory estimate on the reference Mac before treating the monitor as release-ready.


### D-IMPL-236: Remove PC-only calibration tooling

- **Decision:** Remove the legacy CUDA tuning script, its calibration test, and its requirement from the macOS product. Keep request-scoped sampling overrides that are part of the Mac server API.
- **Reason:** The tuner selects PC CUDA/HIP engine settings, while the Mac product has one Metal device and no compatible calibration backend. A successful no-op command would imply support that does not exist.
- **Review required:** Revisit only if a measured Metal calibration utility is designed as a separate Mac feature.

### D-IMPL-237: Keep Mac build dependencies independent of the disabled installer

- **Decision:** Fetch the pinned llama.cpp revision directly through CMake for the engine tests and vision encoder. Keep MTP and release metadata helpers in focused modules outside `setup.py`; setup and MCP install entrypoints remain fail-closed until the Mac planner and installer are implemented.
- **Reason:** Build and validation must remain usable while installation is deliberately unavailable. Importing installer code into development tools coupled independent Mac workflows to an unfinished PC-oriented module.
- **Review required:** Replace fail-closed setup behavior only after Mac planning, model selection, artifact verification, and end-to-end install/start validation pass.

### D-IMPL-238: Remove OS-specific memory probing from the Mac build

- **Decision:** Remove the old Windows/Linux conversation-memory provider and its platform-specific test. Keep the macOS Mach/sysctl memory probe as the sole runtime memory interface in the Metal target. Make atomic MTP metadata publication use the POSIX directory-fsync path unconditionally.
- **Reason:** The removed provider reads `/proc/meminfo` or calls `GlobalMemoryStatusEx`; on macOS it always returned unknown and had no consumer in the Metal engine. Retaining it added OS support code that could never serve this product.
- **Review required:** Confirm every memory fit decision continues to use the Metal planner and that no Mac server path relies on the removed legacy helper.


### D-IMPL-239: Remove unbuilt PC inference trees from the macOS repository

- **Decision:** Delete the inactive Windows/Linux CPU, HIP compatibility, CUDA host runtime, prefill and discrete-GPU planner trees. Keep only sources needed by the explicit Metal build, shared Mac host logic and active tests. Remove OS-specific PLE prefetch controls; use Darwin page advice and the system page size for optional page locking.
- **Reason:** Those files are not inputs to the Metal targets and preserve a second, untested runtime architecture in a macOS-only codebase. CUDA parity sources are handled separately under D-IMPL-240.
- **Review required:** See REV-035. Confirm no explicit Mac CMake target, test, tool or license notice depends on the removed files; verify the PLE mapping and locking boundaries on the reference Mac under memory pressure.

### D-IMPL-240: Keep CUDA parity code outside the Mac product

- **Decision:** Retain the upstream CUDA kernels and standalone parity harnesses only as offline numerical-reference and provenance inputs. They are not included in the Mac CMake target graph, macOS CI jobs, setup flow or release assets. The supported product backend remains Metal.
- **Reason:** The Metal implementation still needs a reproducible way to compare arithmetic with the upstream reference. Removing the source harnesses would prevent future reference captures; retaining them does not add a CUDA runtime or CUDA build to macOS.
- **Review required:** See REV-036. Keep the Mac source list explicit, keep CI and release checks free of CUDA compilation, and revisit these files if checked-in reference captures replace the external oracle workflow.


### D-IMPL-241: Rebuild history from the maintained macOS project

- **Decision:** Replace the inherited Git history with a new parentless root and a short series of English commits containing only maintained project files. Point `main` at the project-identity root and make `codex` a reviewable descendant. Keep the source baseline in `UPSTREAM_SYNC` and `docs/PROJECT_ORIGIN.md`; preserve license and source notices. Retain CUDA code and measurement records only where the Mac parity or port documents cite them as offline reference inputs.
- **Reason:** The user requested an independently maintained macOS-only repository and a fresh English commit history. A new history also keeps retired Windows/Linux product files, obsolete fixtures and generated build output out of all reachable commits while preserving a normal `codex`-to-`main` review base.
- **Review required:** See REV-037. Verify the root has no parent, no original Strata commit is an ancestor, `main` points to the root, `codex` descends from it, all commit subjects are English, the final source tree contains no retired product paths or local build output, and provenance and license files remain.

### D-IMPL-242: Remove unreferenced inherited assets

- **Decision:** Remove the upstream funding profile, unused paper and media files, the six static native-expert fixtures whose test consumer was deleted, and the uncited Linux/CUDA RTX 5090 benchmark package. Keep benchmark records cited by the Mac port documentation and `ref/load.py` for offline artifact reference.
- **Reason:** These assets are not used by the maintained product or current tests; retaining them would carry unused Strata identity, non-Mac runtime instructions and data into the smaller repository. The cited benchmark records and reference loader still support documented analysis.
- **Review required:** See REV-038. Search retained sources and docs for links to the removed assets and confirm the Mac test suite passes.

### D-IMPL-243: Require authentication for non-loopback listeners

- **Decision:** Permit unauthenticated requests only when the server binds to a literal loopback address or `localhost`. Require a non-empty API key before any other bind; enforce this at CLI startup and in the server factory, and keep authentication required if the key is later cleared.
- **Reason:** AGENTS.md requires the API to stay on loopback unless the operator supplies `--api-key`. A warning after binding does not satisfy that boundary.
- **Review required:** See REV-039. Confirm missing or empty keys fail before engine startup and before socket bind; confirm loopback startup remains unchanged and all API endpoints reject missing keys when authentication is enabled.

### D-IMPL-244: Keep shared host tests under the test tree

- **Decision:** Store the platform-neutral conversation-cache and coupled-draft tests in `tests/core/`; keep their CTest target names and Mac preset coverage unchanged.
- **Reason:** They are test programs for headers used by the Metal engine, not production code or remnants of a Windows/Linux runtime. The `tests/` path makes that boundary clear.
- **Review required:** See REV-040. Confirm test-map coverage and both `mac-host` and full Metal CTest runs after the move.

### D-IMPL-245: Remove Windows-bound benchmark launchers

- **Decision:** Remove the historical `esp_kl.py` and `esp_serve.py` launchers because they embed Windows checkout paths. Keep their measured outputs as historical reference data and use `tools/metal/tf_compare.py` for direct logits-dump comparisons on macOS.
- **Reason:** The launchers cannot run in the macOS-only repository and imply support for a separate Windows benchmark workflow. Keeping the measurement tables preserves the reported evidence without retaining an unusable runner.
- **Review required:** See REV-041. Confirm current docs point to the Mac comparer, the historical results are labeled accurately, the runner files are absent, and the comparer tests cover dump validation and reported metrics.


### D-IMPL-246: Keep SIGKILL diagnostics cause-neutral

- **Decision:** Report SIGKILL and direct the operator to Console and system memory pressure checks, but do not infer a cause from whether the engine log grew.
- **Reason:** SIGKILL does not identify its sender. A post-start SIGKILL can come from system pressure, an operator, or another process policy; an unchanged log is not evidence of quarantine or signing failure.
- **Review required:** See REV-042. Confirm the runtime, test contract, and setup-port documentation all match.


### D-IMPL-247: Preserve logits alignment and comparison evidence

- **Decision:** Reject empty comma-separated token fields and out-of-vocabulary IDs, prevent a JSON report path from aliasing an input file, and report standard error as unavailable for one-row comparisons.
- **Reason:** A dropped token shifts every later target position; an output alias can destroy the measurement being checked; and a one-sample standard error is undefined rather than zero.
- **Review required:** See REV-043. Check malformed inputs, hard-link aliases, large integer IDs, and one-row text/JSON output.


### D-IMPL-248: Remove uncited PC benchmark archives

- **Decision:** Delete `bench/results/2026-09-24-final/`, `2026-09-24-vision/`, `2026-09-28-coder/`, `2026-09-28-rope-scaling/`, `2026-09-28-speed-0114/`, `2026-09-29-layer-split/`, `2026-09-29-layer-split-limits/`, `2026-09-29-rtx3090-epyc-milan/` and `2026-09-30-tc-gemv-sm75/`. Keep only benchmark archives linked by the Mac port documentation or a retained source tool.
- **Reason:** The removed results and runners document Windows, Linux, NVIDIA and CUDA workloads that are not part of the maintained Mac product; no retained Mac document or source tool uses these archives.
- **Review required:** See REV-044. Confirm retained references resolve, the nine folders are absent from every commit in the rewritten history, and the Mac test suite still passes.

### D-IMPL-249: Validate model metadata and tensor contracts without full checkpoints

- **Decision:** Check all seven retained model-header snapshots for required metadata, per-role ggml types, block geometry and FF = 640 in Python. Serialize each snapshot as a header-only two-shard GGUF pair and check its metadata, production C++ type allowlist and block geometry. Exercise WeightStore rejection with sparse synthetic GGUF files, including mutated values for every protected metadata key, a Q5_0 expert-down tensor and a partial quantization block.
- **Reason:** The header snapshots preserve the real catalogue metadata and tensor tables, while full checkpoints are too large for routine host tests and the project is not yet ready for model-backed validation. Compact headers and sparse fixtures exercise production C++ paths without GPU or checkpoint downloads.
- **Review required:** See REV-045. Confirm all seven snapshots remain covered in both readers, every changed protected key is named, unsupported expert formats and partial blocks are refused with the tensor name, and revisit against setup-produced packs when those become available.



| REV-047 | Keep PLE metadata, kernel constants and table bounds consistent | All seven model headers match the emitted n-gram constants; C++ and compiled Metal parameter blocks match the artifact values; head offsets are exact prefix sums; each PLE table appears once with shape `[160, 320001536]`; value and type mutations report their keys; the final hostile re-review has no findings | Before accepting R4 |
| REV-048 | Keep the product macOS-only | Root CMake rejects non-Darwin targets; setup, server and MCP host gates reject unsupported machines; CI schedules only Mac runners; project-owned runtime and test sources expose no Windows/Linux backend; hostile review found no remaining product paths or harmful deletions | Reopen only if a supported product path or project-owned non-Mac runtime is added |
| REV-049 | Verify pack float payloads against catalogue GGUF sources | Hostile reviews found missing conversion entries, lossy output with `compat_bf16=false`, quantized required-form rows marked native, empty or source-mismatched expert tables, incomplete packs and missing split indices; regressions now cover these paths, source-derived expert-table equality, both Q2_0/Q8_0 native PLE exceptions, and incorrect expert counts and shard offsets; final review found no further code issues, and the real model run remains | Before accepting R5 |
| REV-050 | Keep native expert metadata behavior visible in the Mac CLI | v3/v4 reader tests cover accepted and malformed tables; CLI fixture contains `experts.bin`, reports that it is ignored, and emits the same PLAN line; host test-map mapping is valid | Before accepting R6 |
| REV-051 | Validate the production pack index parser | The dedicated CTest accepts the current 19-field pack fixture and rejects canonical `code_bits`, `compat_bf16: true`, and an 18-field row; Python verifier regressions reject source-mismatched expert count and shard offsets; final hostile review confirmed the code-bits mutation reaches the intended parser check | Before accepting R7; validate against the seven pinned upstream packs when T3 makes them available |
| REV-052 | Verify every expert address across supported shard layouts | `expert_address_test` checks every layer, role and expert in five sparse layouts, compares WeightStore offsets to parsed GGUF offsets, validates all window slices and checks both Swift split boundaries plus a synthetic v4 per-role split; focused tests and the full 84-test CTest suite pass, with one expected unavailable-real-input skip. Hostile review found no P1/P2 issues and caught a stale four-versus-five fixture count in this log; it is corrected | Recheck the seven pinned catalogue packs when T3 makes them available before claiming their real offsets are verified |
| REV-053 | Verify PLE oracle values through both production reader layouts | The host fixture covers a standalone Direct table and a split-shard mapped table at a nonzero offset. The model CTest scans every PLE-bearing GGUF, checks frozen head/tail samples and full-row L1 sums for three probe rows, and compares packed probe bytes between canonical ISTA Q2_0 shard 2 and Swift IQ2_XS shard 1. It validates split metadata, exact standalone extent and embedded tensor offset. The focused host test and test map pass; the full host/Metal suite passes 83 tests with one expected real-input skip. The hostile re-review found no remaining P1/P2 issues. Real model files are not present in this checkout, so the model-backed scan remains unverified | Run `ple_oracle_model_test` with the pinned model directory when T3 provides it; generated fixtures do not establish model provenance |
| REV-054 | Validate the production MTP reader against all catalogue runtimes | `mtp_reader_test` identifies all seven GGUFs by file type, expert count, Swift marker and split metadata; runs `WeightStore::inspect` on each pack/runtime; checks the three binary paths and sizes, the parsed 29-row dense index and tensor extents, and exact ordered vocabulary contents against one of the three shipped files. `mtp_reader_vocabulary_test` checks all three shipped little-endian ID lists and rejects reordered or substituted content without model assets; `metal_weight_store_test` rejects a one-byte-truncated `experts.bin`. The focused warning-as-error build, both host regressions, test-map check, expected missing-input failure and final 85-test ASan CTest suite pass; the model test itself remains unrun because pinned model files are unavailable. Final hostile re-review found no remaining P1/P2/P3 issues after correcting the writable-directory description to name the lock-containing parents; pinned model files remain unavailable for end-to-end verification. | Run `mtp_reader_test` on the pinned seven-model directory when T3 provides it; verify all seven real runtimes and their lock-file locations |
| REV-055 | Keep the tiered-copy lifetime assertion valid under AddressSanitizer | ASan found that `metal_window_storage_test` retained a reference to a `TieredTensorBinding` owned by `TieredStorage::layout()` and read it after `storage.reset()`. The test now copies the immutable binding before destruction. The focused ASan `metal_tiered_copy_test` and full host/Metal suite pass; hostile re-review confirms the post-destruction check uses only copied metadata and found no P1/P2 issues | No further review remains for this test fix; rerun the focused ASan test if its lifetime assertion changes |
| REV-056 | Verify tiered buffer ranges without querying Metal region filenames | A small Metal probe showed `proc_regionfilename` returned an unrelated ICU file for an anonymous no-copy buffer while its CPU pointer remained the original anonymous mmap. `metal_tiered_copy_test` checks each buffer range against the live source GGUF mapping, retaining path identity checks only for the source map itself. The focused ASan test and full host/Metal suite pass; hostile re-review found no P1/P2 issues and confirmed the assertion does not detect a separate alias mapping at another virtual address | Keep the documented claim scoped to address overlap with the active source mapping; rerun the ASan test if the allocator changes |
| REV-057 | Keep the GR read composition ordered and transactional | The first hostile review found missing late-failure coverage for the caller-owned `Encoder` overload. Added a case that propagates the helper's `false`, verifies the segment is discarded, and checks that outputs remain untouched. Final hostile re-review found no remaining actionable issues. The final focused serial and concurrent GR CTests pass; the full ASan non-model suite passed 84 tests with one expected missing-model skip before that test-only addition, and `test_map.py` passes in non-final mode with existing planned-work diagnostics | No further review remains for this composition; Window/Prefill wiring and model-backed decode remain open |
| REV-058 | Preserve the L9 layer-0-write-to-PLE order in both dispatch modes | `encode_window_ple_entry` applies layer 0's `strata_gr_post` in place to the residual consumed by PLE, then encodes the PLE block in the same segment. Tests exercise T=1 and T=2, compare the GR probe with its scalar oracle, use the actual GPU probe output for PLE's query-norm oracle, verify a late PLE failure discards the preceding GR dispatch through both overloads, and reject an open concurrent dispatch group. The latest focused serial/concurrent CTests pass 2/2; the full ASan non-model suite reports 84 passed and one expected unavailable-model skip (run before the final test-only regression); `test_map.py` passes with existing non-final planned-work diagnostics. Hostile review found only a stale test-pass claim, fixed by rerunning the focused tests | Review the residual handoff, both overloads' rollback behavior and dispatch ordering; complete L9 Window assembly and model-backed decode separately |
| REV-059 | Preserve the documented speculative GDN stage order and rollback in serial/concurrent encoders | Initial hostile review found no stage-order, shape, oracle, rollback, concurrency or build defect. Its T=3 padding observation predated the completed T=3 case; follow-up review verified padded activation construction, oracle output/sink comparisons and the maximum qkv sink guard. The follow-up verified the mixed/history alias regression reaches the coordinator before dispatch and the host inventory now distinguishes the fixture coordinator from pending production call sites. Focused serial/concurrent GDN CTests pass 2/2; `test_map.py` passes with existing non-final planned-work diagnostics. No code-level blockers remain | Review the stage order, Q8_1 weight-format restriction, pairwise view contract, serial/concurrent barriers and rollback evidence |
| REV-060 | Commit one layer's accepted GDN prefix in history/state order | Hostile review found no blocking coordinator, rollback, oracle or guard issue. The partial code review verified 0..8 handling, history-before-state ordering, overlap checks, no-op behavior and both late-failure rollback paths. The follow-up verified T=1/2/3 accepted-prefix slicing from GPU outputs, state/history expectations, unchanged scratch/input guards and docs consistency; its fixture-set precision note was fixed. ASan build and serial/concurrent GDN CTests pass 2/2; `test_map.py` passes with existing non-final planned-work diagnostics | Future Window scheduling must map `n_keep` consistently for all layers and KV/indexer/PLE/GDN state, fence before mutation, and retain a full rollback boundary |
| REV-061 | Expose read-only full-mode weight views from ModelStorage | Hostile review passed with no actionable blockers; the ASan storage target builds, owner CTest passes, and `test_map.py` passes | Before L9 Window binding, add a distinct input-only encoder binding for `ReadOnlyBufferRef`; keep the Context lifetime and rollback invalidation checks |
| REV-062 | Bind const model views as read-only Encoder inputs | Hostile review passed with no actionable blockers after adding distinct-handle overlap and adjacent-view regressions; ASan build/CTest and `test_map.py` pass | During L9 integration, use this binding only for MSL `const` arguments; the fixture exercises CPU-backing alias detection, while GPU-address-only alias detection is reviewed in code but not isolated by a fixture |
| REV-063 | Keep the T=1 Window input stem ordered, read-only and transactional | Hostile review found no P1/P2 issue and identified three coverage/documentation gaps: caller-owned failure propagation, proving that a late failure discards the command buffer, and direct coverage of every read-only gather format. The header now requires callers to propagate failure; tests assert discard for standalone and caller-owned segments; Q3_K, IQ3_S, IQ4_XS and BF16 read-only gathers compare with ggml CPU. Focused serial and concurrent CTests pass 2/2; the Metal test map passes with existing planned-work diagnostics | Finish all L9 layer scheduling, model-backed parity, dump hooks and state commit before describing this input stem as a complete Window decode |
| REV-064 | Preserve T=1 attention-to-GDN ordering and speculative state | Initial hostile review identified two test gaps: the zero GR block output could not distinguish apply=false, and several parameter/cross-stage rejection checks lacked fixtures. The test now seeds nonzero block output, compares a direct apply=true negative control, and verifies residual preservation; it independently rejects invalid nc, nonzero n_keep, an out-of-vocabulary token, an inconsistent StepRow and a real cross-stage alias. Follow-up hostile review found no remaining actionable issue. The ASan build succeeds, all four focused serial/concurrent embedding and GDN CTests pass, and `test_map.py` passes with its existing non-final planned-work diagnostics | Re-review the mixed handoff, alias rules, concurrent barriers, n_keep=0 contract and late-failure rollback if layer-0 assembly expands beyond this T=1 speculative segment |
| REV-065 | Preserve the layer-0 attention/GDN/FFN chain and Window scratch reuse | Design review confirmed distinct `hc_attn_*`/`hc_ffn_*` model tensors and exact planned scratch/injection reuse. Hostile implementation review found no actionable issue and confirmed the alias matrix, placeholder separation, `apply=false` coverage, stage order, both rollback paths and concurrent tests. The ASan build succeeds; all four focused serial/concurrent embedding and GDN CTests pass; `test_map.py` passes with its existing non-final planned-work diagnostics | Review layer-indexed weight selection again when a production Window binder connects this pre-resolved-view coordinator |
| REV-066 | Validate the typed MoE route encoder | Hostile review found missing caller-owned failure-propagation coverage and read-only-logits/output alias fixtures; the header contract and tests now cover caller rollback, exact/partial logits-to-ID and logits-to-weight overlap, output overlap, and open-group refusal in concurrent mode. Follow-up found no further issue in this increment. The ASan build and serial/concurrent MoE CTests pass; `test_map.py` passes with existing non-final planned-work diagnostics | Add the BF16 `ReadOnlyBufferRef` weight path to mat-vec before connecting the router to the T=1 Window chain; keep the router→route→Q8_1 sequence outside concurrent groups |
| REV-067 | Accept full-mode BF16 weights as read-only mat-vec inputs | Hostile review requested production router-width coverage and a distinct-handle shared-backing alias fixture; both were added. Follow-up found no actionable issue and confirmed direct type compatibility with `FullModeTensorView.buffer`. Focused validation covers bitwise oracle parity, input immutability, shared-segment rollback, all read-only input/output and output/output aliases, cross-handle partial overlap, short and misaligned views, and read-only/read-only overlap in serial and concurrent dispatch modes | Before T=1 router integration, confirm the live full-mode tensor view and `WindowParams` metadata can be sliced with required alignment and lifetime; keep router → route → Q8_1 ordered outside concurrent groups |
| REV-068 | Compose the T=1 router, route and Q8_1 prefix | Initial hostile review found that U(16) route-weight tolerance would not catch a division-mode regression; the fixture now requires bitwise equality with a direct reciprocal-mode GPU dispatch and proves a direct division-mode negative control differs on the same router logits. Follow-up found the router-tensor/parameter overlap check lacked a fixture; a large shared-backing alias case now reaches that check and verifies rejection before dispatch. Final hostile review found no actionable issue. The ASan target builds, serial/concurrent `metal_gdn_test` CTests pass, and `test_map.py` passes with its existing non-final planned-work diagnostics | Connect the pre-resolved-view prefix to the production Window binder only after resolving full/tiered router weight access and the tiered `MissResolver` handshake; keep `MoePlan` and expert execution as later ordered stages |
| REV-072 | Resolve tiered plans through the shared-event host bridge | The first hostile review found that the fixture CPU-prewrote its plan; the test now dispatches the real GPU plan kernel before the ready signal and compares exact plan bytes. Follow-up found the host-ready value must immediately follow plan-ready so a malformed value cannot release a later wait; the implementation checks adjacency with overflow protection and tests a rejected gap before submission. The read-only skip path is tested after poisoning plan arrays, and malformed headers leave the wait unsignaled for explicit caller cleanup. Final follow-up found only a missing validation result and review-log row; both were added. The ASan `metal_window_storage_test` target builds; the `metal_tiered_copy_test` CTest passes; serial and concurrent `metal_moe_test` pass 2/2; `test_map.py` passes with existing non-final diagnostics. | Keep production Window scheduling, timeout recovery, lookahead, Prefill `PfPlan` handling and parity retirement separate; these remain open under U6/REV-024 |
| REV-073 | Release blocked GPU waits after a stalled event | Initial hostile review found that an invalid timeout poisoned the event before returning, a fixed sleep did not prove release waited for late commit, and `signaled_value()` exposed the terminal sentinel as completion-looking state. Those fixes remain; follow-up found that advancing to the terminal value satisfies downstream command-buffer waits but does not cancel commands after them. The event and resolver headers now document this, and the engine-host plan forbids release before removing or guarding untrusted consumers. The current production failure path is still unwired. | Before using wait release for a failed tiered plan, prove no queued consumer can read an unresolved address; prefer a per-layer command-buffer boundary until a GPU-visible failure guard exists |
| REV-074 | Retire completed expert layers at later plan events | Hostile review found no remaining P1/P2 issue. After the plan-ready wait, lease retirement, miss resolution and the layer-ready signal run under the existing operation lock, while GPU waiting remains outside it so prefetch can continue. The test produces layer-0 and layer-2 plans in one command buffer and proves the later plan event releases prior leases before completion; the ASan target builds and the CTest passes 10/10 repeated runs. `test_map.py` and `git diff --check` pass with existing planned-work and missing-check diagnostics. | Preserve serial per-window ordering; overlapping windows require per-window lease pinning through completion |
| REV-075 | Keep failed tiered-plan resolution from releasing downstream expert work | Hostile review found that auto-signaling the host release event on resolver failure could let already-queued expert kernels read unresolved or stale address-table entries. Removed that release; the caller retains failure ownership. `release_gpu_waits` and `resolve_layer_after_plan` API comments plus the engine-host plan now state that wait release does not cancel downstream kernels. A real grouped IQ2_XS consumer is gated while its slot leases stay pinned; test-only failed-command cleanup clears the plan group after its GPU arrival, waits each subsequent GPU signal before host release, and verifies the guarded consumer leaves outputs untouched. Follow-ups found and drove fixes for cleanup overtaking the consumer-arrival/final layer-done signals and releasing before the guard was installed. Test-map ownership was split between default full-mode and explicit `--tiered-copy` CTests. Focused CTests pass 4/4; the tiered CTest passes 20/20 repetitions; `test_map.py`, Python suite (35/35 modules), and `git diff --check` pass. Final hostile follow-up found no actionable P1/P2 issues. | Start production tiered scheduling with expert consumers submitted only after successful resolution, or prove the GPU-visible guard before any address load |
| REV-076 | Validate typed grouped MoE gate/up dispatch | Initial hostile review found that rows/experts were not bound to the GPU plan dimensions, making a mismatched plan capable of out-of-range reads/writes, and that the header overstated ExpertStore validation on the skip path. The host now passes row count in a dedicated 12-byte gate/up parameter ABI; every format bounds-checks plan row/group headers, group ranges, expert IDs, token IDs and null role addresses before reading or writing. Runtime regressions cover mismatched rows/experts, an out-of-range token and an invalid group range. Documentation now requires the same plan dimensions and says tiered consumers are queued only after successful ExpertStore resolution; it does not claim ExpertStore validates every plan array. All eight grouped gate/up oracle paths use the engine encoder. Other fixtures cover invalid shapes/layers/formats, short/misaligned views, exact/partial aliases, caller-owned rollback and the open-group boundary. Final hostile follow-up found no remaining P1/P2 issue. Serial/concurrent CTests, `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Ensure the production scheduler submits this consumer only after plan generation/resolution, and retain the selected expert addresses through command completion |
| REV-071 | Persist MoE usage counters with model weights | Hostile review found no planner/storage implementation issue. It identified two documentation gaps: a redundant new macOS-only decision conflicted with the established project-origin record, and usage decay read as implemented. Removed the duplicate decision, clarified that decay awaits event-ordered adaptive-swap integration, and marked the already-completed macOS audit in D-IMPL-227/228/252 and REV-048. ASan storage/planner/MTP targets build; focused CTests pass 3/3; `test_map.py` and `git diff --check` pass | Add host reads/decay only after GPU plan completion is proven by the window event protocol |
| REV-069 | Add a validated MoE plan encoder | Initial hostile review found two missing alias fixtures (IDs/plans and resident/usage); both now verify rejection, command-buffer discard and output immutability through the shared rejection helper, and follow-up confirmed both are included in the test result. No other actionable issue was found. The ASan target builds; serial and fixture-only concurrent CTests pass 2/2, and `test_map.py` passes with existing non-final planned-work diagnostics | Keep plan encoding separate from expert dispatch until production layer scheduling proves the plan/event ordering and `ExpertLeaseSet` lifetime |

| REV-077 | Compose T=1 Window routing through GPU MoE plan production | Initial hostile review found that the tiered fixture did not guarantee a routed miss and that its short-plan rejection happened before prefix encoding. The fixture now marks one routed expert resident and checks that the expected plan contains both resident and missing groups. It also encodes the complete caller-owned route/Q8_1/plan prefix before injecting a containing-segment failure, then checks command-buffer discard and output immutability; the standalone short-plan case continues to verify rejection before dispatch. Review also found the decision's REV-077 reference missing; this row closes it. Follow-up review found no remaining actionable P1/P2 issue. The `metal_gdn_test` target builds and serial/concurrent CTests pass 2/2; `test_map.py` and `git diff --check` pass with existing diagnostics. | Production must signal/wait for plan readiness, resolve tiered misses successfully before submitting consumers, and retain expert leases until GPU completion |
| REV-078 | Validate routed and shared-expert SwiGLU dispatch | Hostile review found no actionable issue in shape arithmetic, 640-column dispatch coverage, read-only bindings, overlap checks, transactional behavior or converted routed/shared tests. It found the decision's REV-078 and test-map references had no review row; this record closes that gap. The `metal_moe_test` target builds and serial/concurrent CTests pass 2/2; all routed/shared fixtures cover 1..8 rows and three input sets, with validation and rollback cases. `test_map.py` and `git diff --check` pass with existing diagnostics. | Keep the 640-column tail covered by all three threadgroups and preserve the routed rows-times-10 versus shared rows entry count |
| REV-079 | Validate grouped routed-expert down dispatch | Hostile review found no P1/P2 issue. It found no malformed-plan GPU fixtures for the Q2_0 and IQ4_NL down kernels; tests now dispatch out-of-range destinations through both formats plus invalid IQ4_NL expert/group bounds and verify the whole output allocation, including trailing guards, remains unchanged. The `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. `test_map.py` and `git diff --check` pass with existing diagnostics. | Keep tiered down dispatch behind successful plan resolution, preserve Q8_1 input ordering, and test both format-specific GPU plan guards |
| REV-080 | Validate MoE routed/shared combine dispatch | Hostile review found no actionable P1/P2 issue in the shape arithmetic, typed bindings, alias checks, transactions, converted oracle paths or documentation. It found the D-IMPL-284 and test-map references had no review row; this record closes that gap. The `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. The fixture covers 1..8 rows, all four output/input aliases, invalid views, rollback and the separate shared multiply/add golden. `test_map.py` and `git diff --check` pass with existing diagnostics. | Preserve routed FMA ordering and the separate shared multiply/add behavior while keeping the output disjoint from all inputs |
| REV-081 | Compose the routed MoE expert FFN stages | Initial hostile review found that the composed GPU oracle only covered T=1; the fixture now runs T=1 and maximum T=8 with row-specific route IDs and sizes every oracle/buffer from `rows * 10`. Follow-up review found a missing writable scratch alias regression and missing public preconditions for trusted plan partitions/addresses; both are now covered in tests and docs. Final hostile review found no actionable P1/P2/P3 issue. The target builds; final serial and concurrent CTests pass 2/2, including exact GPU plan bytes and gate/up, SwiGLU, Q8_1 and down scalar-oracle comparisons at T=1 and T=8. Invalid extents, input/scratch and scratch/scratch aliases, caller-owned rollback, open-group refusal and guards are covered. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Resolve tiered plans successfully before submitting this consumer; finish shared-expert scheduling, combine and the rest of Window decode before claiming end-to-end inference |
| REV-082 | Bind Q8_0 mat-vec inputs read-only | Initial hostile review found two public-contract omissions and test gaps in the long-K oracle, NC=2 sink-tail guard, and no-padding minimum sink. The API now documents input lifetime/stability and failure propagation; the oracle checks failure and exact size; NC=1/2/8 fixtures exercise the no-padding and padded sink paths with per-case guards. Final hostile review found no actionable P1/P2/P3 issue. The target builds; serial and concurrent CTests pass 2/2; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Compose the full shared-expert and combine path without weakening the full-mode read-only binding contract |
| REV-083 | Compose the shared-expert FFN and routed combine | Initial hostile review found no P1/P2 issues. It identified a missing direct `<cstring>` include and missing decision/review records; both are fixed. The caller-owned and standalone encoders compose Q8_0 gate/up, SwiGLU, Q8_1, Q8_0 down, BF16 gate and combine in order. Tests cover rows 1..8 over random, edge and fixture inputs; scalar-oracle outputs, padded sink values and exact write ranges, read-only input guards, active-row and cross-stage alias rejection, rollback and concurrent-group refusal. Final follow-up review found no actionable issue. The target builds; serial and concurrent CTests pass 2/2; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Add the shared output's validated handoff to FFN residual write; keep tiered consumers after successful resolution and do not claim production Window integration |
| REV-084 | Apply shared MoE output to the GR residual | Initial hostile review found that unused tails of oversized residual/injection views could overlap and be rejected by GR post only after the shared FFN stages were encoded; GR post now receives bounded active slices, with a regression for adjacent active ranges and overlapping unused capacity. It also found missing public guarantees for caller-owned failure propagation and asynchronous input stability; the API and kernel documentation now require propagating `false`, keeping FFN inputs, injection and initial residual unchanged until completion, and avoiding access to device-written scratch/output before completion. Final hostile re-review found no actionable issue. The target builds; focused serial and concurrent CTests pass 2/2; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Before Window scheduling, verify production residual/injection extents and routed-output lifetime, and preserve the successful tiered `ExpertStore` resolution boundary |
| REV-085 | Compose routed/shared MoE FFN and GR residual write | Initial hostile review found that cross-stage alias checks used full view capacities and that composition tests covered only T=1 and T=8. The compositor now checks active byte ranges, and the regression rejects live-range overlap while accepting overlap confined to unused capacity. The GPU oracle runs T=1..8 and verifies padded-column sink values. An initial test run also exposed incorrect guard allowances that counted only one routed row instead of all ten expert entries; exact routed extents and explicit guard diagnostics now cover the failure. A follow-up found the missing REV-085 and test-map references; both were added. Final hostile follow-up found no actionable issue. The ASan target builds; serial and concurrent CTests pass 2/2; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Before Window integration, preserve successful tiered `ExpertStore` resolution, validate production Window ranges and lifetimes through GPU completion, and do not claim end-to-end inference |
| REV-086 | Compose the full-mode T=1 Window MoE layer | Hostile review found that a separate consumer could neither guarantee tiered `ExpertStore` readiness/leases nor prevent duplicated producer views. Follow-up reviews found address-table provenance gaps, consumer-input aliases into producer writes, handoff allowances not tied to a consumer role, omitted router/resident producer inputs, and producer-table overwrite through outputs/sinks. Fixes require `FullModeStorage`, compare all three owner tables, bind each permitted input role to its exact handoff, and reject other consumer input/output/sink aliases against every producer view. Negative tests cover tiered table injection, normalized-output aliasing, mixed/router/resident role substitutions, plus router-weight overwrite through both an output and a sink. Final hostile re-review found no remaining P1-P3 issue. `metal_gdn_test` serial/concurrent and `metal_window_storage_test` pass 3/3; the routed/shared `metal_moe_test` serial/concurrent pair passed 2/2. Build, `test_map.py` and `git diff --check` pass with the repository's existing planned-work diagnostics. | Keep tiered execution behind successful plan readiness, `ExpertStore` resolution and retained expert leases; do not claim a production Window scheduler or end-to-end inference |

| REV-087 | Preserve the layer-0 pending MoE write | Hostile review found missing tiered lease/lifetime and caller-owned failure-propagation conditions on the new no-write public API, stale full-mode residual-writeback wording in the kernel spec, and no positive public full-mode coordinator route. The API contract and docs were corrected; GPU cases exercise the no-write composition for T=1..8 and prove residual/injection remain bitwise unchanged, while Window handoff preflight rejects combined-output aliases into either region. A separate GR-post dispatch consumes the pending combine result and matches the residual oracle. The first consumer-boundary follow-up found ineffective combined-output immutability and residual write-range checks; the test now snapshots immediately before GR post, checks the full combined output and injection remain unchanged, and validates the residual's allowed write extent and guards. Final follow-up found no remaining actionable issue. D-IMPL-294 and D-IMPL-295 verify the two FullMode consumer branches in separate synthetic segments; D-IMPL-296 centralizes their `ple_enabled` selection and rejects mismatched payload tags. `Window::run` wiring and model parity remain open. The targets build; focused serial/concurrent GDN/MoE and storage CTests pass 5/5, and the updated MoE serial/concurrent pair passes 2/2; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Wire the selector into the production Window token graph and add model parity before claiming L9 |
| REV-088 | Bind the pending MoE output to the PLE entry | The new typed coordinator obtains the prior block output and injection from `MoeRoutedSharedFFNBuffers` and `GRReadBuffers`, and rejects a PLE residual that differs from the producer residual. Its fixture covers ordered GR post plus PLE, caller-owned rollback after a late history error, and a mismatched-residual rejection before dispatch. Hostile review found no P1 issue; it confirmed the integrated full-mode producer path was untested and identified the standalone ordering contract recorded under REV-089. D-IMPL-295 now covers a FullMode producer-to-PLE handoff with mapped-row staging and synthetic projection weights; D-IMPL-296 centralizes pending-consumer selection. Production Window wiring remains open. | Preserve producer-to-consumer view identity and wire the branch selector into the production Window |
| REV-089 | Specify producer ordering for standalone pending-MoE PLE entry | The standalone overload consumes output and injection views produced by an earlier command buffer, but its declaration did not state the required ordering. An initial clarification incorrectly treated same-queue commit order as sufficient; hostile follow-up found that this does not propagate producer GPU errors. The API contract now requires the caller to register `consumer.depends_on(producer, error)` and verify successful producer completion before submitting the consumer. A final read-only follow-up confirmed that this matches the existing command-buffer contract. | Keep standalone consumption after a registered dependency and successful producer completion; verify this contract in the eventual integrated Window test |
| REV-090 | Bind the pending MoE write to no-PLE attention | The new typed entry maps the layer-0 shared output and FFN GR injection into the next attention GR read, checks exact residual reuse, and forces `apply=true`. Serial and concurrent GR suites pass and compare every writable GR output against the apply=true oracle; decoy `block_out`/`inject_prev` views prove the producer binding overrides caller values, while producer views remain unchanged. Hostile review found a missing post-entry guard check; the test now checks every writable view's declared extent after both consumer forms, and follow-up found no P1/P2. Mismatched residuals are rejected without writes. D-IMPL-294 and D-IMPL-295 integrate each typed branch with FullMode; D-IMPL-296 centralizes selection. A review wording ambiguity was clarified in the kernel guide. | Wire the selector into the production Window token graph and complete model-backed parity |
| REV-091 | Integrate the FullMode producer with the no-PLE attention consumer | Initial hostile review found that zero FFN injection weights and an equal-valued attention decoy did not prove the typed handoff, and that an allocation-count assertion was vacuous; both were corrected with nonzero injection weights, a distinct decoy and explicit producer-injection checks, while removing the assertion. Follow-up found the T=1 GR low-rank comparison included unwritten rows; its snapshot now covers only the active row. Final hostile review found no actionable issue. The production-shaped sparse fixture executes the actual 256-expert FullMode producer and typed attention consumer in one `Encoder::encode` segment. It checks stable top-10 selection of the populated Q2_0 expert slabs, a finite nonzero combined output, exact GR output parity against a direct `apply=true` read using the original residual and producer injection, one-time residual application, and pending-output immutability. Focused CTests cover this target plus the existing GR, storage and ModelStorage owner paths. | Wire the typed selector into production Window scheduling and add model parity |
| REV-092 | Integrate the FullMode producer with the PLE consumer | The synthetic 256-expert FullMode producer runs before `encode_window_moe_ple_entry` in one caller-owned Encoder segment. The test builds row IDs from token 1 and committed history `[5, 0]`, then stages the sixteen selected IQ4_NL rows from FullMode's mapped sparse fixture table through `stage_window_ple_input` into `WindowScratchLayout::ple_input`; Q2_0 key and BF16 value projection weights remain synthetic. The pending MoE output uses `WindowScratchLayout::embedding`, based on the test judgment that the input-stem embedding is dead by the PLE stage; production scratch liveness must be confirmed. PLE history is deferred. The test snapshots active PLE outputs, residual, pending MoE output and committed history; a direct GR-post plus PLE-entry run from the original residual provides the comparison. Independent hostile review found no actionable findings. The target builds and the eight focused handoff, GR, PLE and storage CTests pass; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Confirm scratch liveness in production scheduling, wire the typed selector into the token graph, and verify model-backed parity |
| REV-093 | Select one FullMode pending-MoE consumer | Initial hostile review found a P2: a T=1 FullMode producer could be paired with a multi-row consumer, which would read rows the producer had not written. The selector now rejects `rows != 1` before inspecting the branch payload, and both PLE and attention integration cases verify this rejection alongside the mismatched-tag rejection. The two branch CTests and the eight focused handoff, GR, PLE and storage CTests pass. Follow-up hostile review found no actionable findings. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Have production `Window::run` derive `ple_enabled` from the token graph, pass the matching payload, propagate failures to discard the producer segment, and retain model-backed parity |
| REV-094 | Compose the FullMode T=1 token-to-consumer step | Initial hostile review found two P2 issues: embedding scratch was not checked against later live producer/consumer views, and the synthetic native FullMode table used BF16 although the loader accepts Q3_K/IQ3_S/IQ4_XS. The coordinator now rejects overlap with every later view except an exactly identical `shared.output`; regressions reject attention-weight overlap and partial `shared.output` overlap while both exact-alias branch integrations pass. The fixture now maps a Q3_K `[2560, 1]` row, matching the production loader. Follow-up reviews found P3 gaps in input-stem coverage: the standalone reference had preinitialized the coordinator's aliased embedding scratch and active residual. The integration now preserves expected values separately, resets active embedding and residual ranges to distinct sentinels, and compares consumer outputs and final residual to the independent reference. Final hostile review found no actionable issues. The target builds; eight focused CTests, `test_map.py`, and `git diff --check` pass with existing diagnostics. | Keep `Window::run`, the tiered plan-resolution handshake, later layers, state commit, head and sampler outside this T=1 coordinator until their scheduling and model-parity contracts are implemented |
| REV-095 | Add a transactional FullMode T=1 step wrapper | Initial hostile review found a P2 evidence gap: unchanged CPU-visible buffers did not prove that earlier dispatches were discarded, since the test never submitted the failed command buffer. The late-failure test now also requires `unsafe_native_handle()` to be null, proving the wrapper made that buffer unsubmitable. Final hostile follow-up found no actionable issue. The target builds; eight focused CTests, `test_map.py`, and `git diff --check` pass with existing diagnostics. | Keep the wrapper limited to fresh command buffers; do not imply that command-buffer queue order proves successful upstream producer completion for standalone consumer-only calls |

| REV-096 | Bind FullMode PLE weights through read-only views | Initial hostile review found four actionable gaps: read-only RMS gamma could not overlap input as documented; the FullMode fixture supplied PLE tensors itself and so did not prove automatic resolution; the FullMode resolver accepted Q8_0 although the model-role allowlist rejected it; and the Q2_0 read-only test stopped at exactly 64 blocks per row. Fixes now bind disjoint RMS input/output as read-only input plus read-only gamma, clear caller PLE views and declare the wrong key format in the integration fixture, permit native Q8_0 PLE keys in both model-header allowlists with a WeightStore regression, and cover 65 Q2_0 blocks per row. The final hostile pass found stale call-site status in the kernel and engine guides, an imprecise excluded PLE-table description, and a missing API lifetime/key-resolution contract; these documentation gaps were corrected and final follow-up found no actionable issue. Focused serial/concurrent GR, MV, PLE and FullMode Metal CTests pass; the host WeightStore CTest, nine model-header Python tests, test-map validation and diff check pass. The post-fix mac-metal suite passes 45/45; mac-host passes 43 tests with the one expected unavailable-real-input test skipped. The full Python runner passes 35/35 modules in a temporary environment with declared requirements; two optional `jsonschema` cases are skipped because that extra is not installed. | Keep the PLE branch tied to its `FullModeStorage` owner through command completion. Add real-artifact parity and production Window scratch/queue scheduling before claiming L9 or end-to-end inference |

| REV-097 | Verify automatic FullMode PLE key-format selection | The hostile review found no P1/P2 issue. It found one P3 documentation gap: D-IMPL-300 and the new test-map rows referenced a missing REV-097 ledger entry; this row records the review and closes that traceability gap. It verified the Q2_0/Q8_0/BF16 payload byte counts, inspected GGUF type and shape, caller-format decoys, automatic FullMode resolution against format-specific direct references, CTest registration, map consistency and synthetic-data scope. The three focused PLE handoff tests pass 3/3, the complete mac-metal suite passes 47/47, `test_map.py` passes with the repository's existing planned-work and missing-check diagnostics, and `git diff --check` passes. Final follow-up after the ledger repair found no further actionable issue. | Preserve the synthetic-fixture limitation; require shipped-model parity and production Window scheduling before claiming L9 |

| REV-098 | Track the committed PLE token suffix | Initial hostile review found a P2 prompt-boundary risk: seeding with the full prompt would include its final token even though that token is the first T=1 decode window. The API contract, kernel guide and decision record now require seeding only from IDs already applied to model state; tests build a pending final prompt token from the applied prefix and advance history only after its accepted commit. A follow-up found the missing REV-098 ledger row, which this entry supplies. A later hostile pass found the one-token prompt fixture ambiguously treated its pending token as applied; it now starts from an empty applied prefix and adds that token only after the first decode commit. Final hostile follow-up found no actionable issue. The target builds; the focused row-history CTest passes; all 44 mac-host tests pass and the existing unavailable-real-input test is skipped; `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. | Attach the suffix to the eventual session owner, seed only from model-applied prompt tokens, and advance it after successful commit; resolve Prefill/GENI image-row token semantics before wiring those paths. Production Window scheduling and real-model parity remain required |

| REV-099 | Bind PLE token history to session reset | Initial hostile review found a P2 escape hatch: mutable access to the owned helper let callers clear token history without resetting GPU state. It also found that pending/foreign-context refusal tests did not prove GPU bytes were preserved, and that the REV-099 ledger row was missing. Replaced mutable helper access with `SessionStateStorage` methods for initialization, read-only suffix access, row building and accepted-prefix commit; only owner reset clears its host suffix. Added full-allocation byte-pattern assertions around both refused reset cases, one-time-initialization checks after refused resets, and a prepopulated-output preservation check for uninitialized row-build refusal. The full build and mac-host suite (44 passed; one existing unavailable-real-input test skipped) passed before the final test-only assertions; mac-metal passed 47/47 before those assertions, then the rebuilt latest `metal_session_state_test` passed. The latest `test_map.py` and `git diff --check` pass with existing non-final/planned diagnostics. Final hostile review found no actionable code or test issue; its only finding was this evidence-provenance wording, corrected here. | Keep token-history operations serialized with the sequence lifecycle, call accepted-prefix advancement only after successful GPU commit, and wire prompt/Prefill/GENI handling before production `Window::run` |

| REV-100 | Prepare T=1 Window parameters and PLE inputs | Initial hostile review found a P3 test gap: the preparation fixture used a default sampler and asserted only its counter, so it could miss dropped or misassigned sampler fields. The test now supplies distinct temperature, top-p/min-p, penalty, top-k, min-keep, greedy and history values and checks every prepared `SampRow` field. The FullMode handoff tests also cover successful no-PLE preparation, session-derived PLE staging for Q2_0/Q8_0/BF16, exact parameter token/position/seed data, direct gathered-row equality, history immutability, and failure before ring/scratch mutation for an uninitialized suffix, invalid token or overflowing position. The target builds; the five focused ring and FullMode handoff CTests pass, `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. Follow-up hostile review found no remaining actionable issue. | Keep parameter-slot and PLE-scratch reuse serialized with GPU work; connect to the production token graph only when sampler, prompt-boundary and commit ordering are established |

| REV-101 | Compose the ordered Window GR head mixer | Initial hostile review found a P3 API-contract gap: a caller-owned `Encoder&` can already contain earlier dispatches when a later stage rejects its input, so callers needed an explicit obligation to propagate `false` from `encode_window_head_mixer` out of the enclosing `Encoder::encode` transaction. The header now states this; tests exercise the caller-owned path and a late short-up-weight rejection, and verify that the standalone wrapper discards its command buffer and clears the returned native handle. Review also checked stage order and shapes, alias validation, Context ownership, concurrent-group refusal, and the independent oracle; follow-up found no remaining actionable issue. The target builds, focused serial/concurrent GR CTests pass, the full `mac-metal` suite passes 47/47, `test_map.py` and `git diff --check` pass with the repository's existing planned-work diagnostics. | Keep the caller-owned Encoder transaction fail-closed; production Window weight/scratch resolution, vocabulary projection, sampling, commit and model-backed parity remain required |

### D-IMPL-251: Preserve the artifact's 90 trailing PLE rows

- **Decision:** Compare the 16 PLE head offsets against the prefix sums of the 16 head vocabulary sizes and require their total to fit within `PLE_TABLE_ROWS`. Do not require the total to equal the tensor's row count. For the pinned headers, the head ranges cover 320,001,446 rows in a tensor with 320,001,536 rows; the remaining 90 rows are left unused and their purpose is not inferred.
- **Reason:** The values in all seven checked-in headers and `ple_artifact_consts()` agree. The existing parity harness treats the table as a bound and reports the 90-row remainder. The earlier K4 audit line incorrectly described the final offset plus vocabulary size as equal to the table length; the test exposed that documentation error.
- **Adversarial review and resolution:** The first review found that only the C++ branch of `params.h` was exercised and Python could treat JSON booleans or floats as integers. Added Metal-branch compile-time assertions, made `metal_weight_store_test` depend on the metallib, and added strict type checks with boolean-layer and floating-point-tokenizer-EOS negatives. The final hostile re-review found no remaining issues.
- **Review required:** See REV-047. Confirm every model fixture matches the n-gram and C++ parameter sources, Metal-side static assertions compile against the same expected values, each table is located in either shard, prefixes are exact, the 90-row remainder remains within the table, PLE EOS remains distinct from tokenizer EOS, and PLE/tokenizer EOS value/type mutations name their metadata key. Keep `ple_reader_selftest` and `ple_oracle_layout_test` mapped as host tests and `ple_oracle_model_test` mapped as a model test; complete the real-model run when T3 provides inputs, as recorded under REV-053.

### D-IMPL-252: Keep the maintained product macOS-only

- **Decision:** Support only Apple Silicon Macs running macOS 27 or newer. Keep the root CMake, setup, server and MCP gates that reject other hosts; require the Mac memory sample unconditionally in the project-owned memory test. Do not advertise or maintain Windows/Linux product paths. Keep vendored llama.cpp portability conditionals intact because they belong to the pinned third-party dependency and do not create supported product targets.
- **Reason:** The user selected a simple macOS-only repository. The active build, CI and release design already targets the native Metal product; stripping platform branches from an upstream dependency would create a divergent vendored copy without changing the supported target. Negative tests that simulate Linux remain because they verify that unsupported hosts fail closed.
- **Adversarial review and resolution:** The hostile review confirmed that CMake, setup, server and MCP still reject unsupported hosts, CI uses Mac runners, and project-owned sources do not expose Windows/Linux runtime paths. It found no harmful deletions or remaining product paths.
- **Review complete:** REV-048 confirmed the CMake and host gates reject unsupported targets, CI runs on Mac runners, project-owned sources expose no Windows/Linux runtime, and third-party portability code is not part of the supported product. Keep those checks as regression expectations.

### D-IMPL-253: Verify standalone pack float payloads

- **Decision:** After a standalone native pack is generated, compare a deterministic sample of up to 32 unconverted float tensors with their source GGUF bytes. Rank each tensor by SHA-256 of its source shard name and tensor name, so the sample is repeatable and varies across model shards. Recompute every recorded conversion payload in bounded chunks using the packer's conversion code and `compat_bf16` setting. Require every non-expert source tensor with a shared pack-form rule to have an index row in that form, including quantized sources that need conversion; only Q2_0 and Q8_0 `blk.1.ple_key.weight` rows may remain native. Require `native_experts.txt` to exactly match the v3/v4 layout the packer derives from every GGUF shard, and reject a pending generation marker, malformed split metadata, and converted rows whose source shard/type/hash or payload differs. The model test compares all unconverted floats and conversion outputs across exactly the seven GGUF catalogue identities. `--base` is outside this Mac contract and is not checked.
- **Reason:** A source hash alone does not prove that converted dense bytes follow the declared operation, and a missing manifest row must not bypass the required engine form. The packer is the active point that has both the generated pack and its GGUF inputs, so validation failure prevents the command from succeeding. The model test identifies the catalogue labels from GGUF `general.name` and `general.file_type`; that metadata is not a cryptographic proof of checkpoint provenance. The T3 quick-hash preflight and a run against all pinned real packs are still required to establish catalogue inputs.
- **Adversarial review and resolution:** Hostile reviews found that required conversions could be omitted, lossy BF16 bytes could pass with `compat_bf16=false`, a quantized pack-form tensor could be relabeled native, empty or semantically incorrect expert tables could count as complete, a pending pack could count toward the seven, and a split shard could omit `split.no`. Added regressions for each false pass, share the Q2_0/Q8_0 native PLE-key exception with the packer, compare the expert table byte-for-byte with the layout derived from source shards, and corrected stale build/test status text. Final post-fix review found no remaining code issues; the real seven-pack run remains open.
- **Review required:** See REV-049. Run `test_models.py check --quick` when T3 provides that tool, then run the full comparison against the seven pinned real packs when `STRATA_TEST_MODELS` is available. Synthetic packs validate the verifier logic but cannot establish checkpoint provenance or replace the model-backed run.

### D-IMPL-250: Map only executable coverage while the setup golden suite is pending

- **Decision:** Map REQ-MOD-06 to the implemented `model_types_test` and `metal_plan_test` only. Keep the documented `tools/test_setup_golden_mac.py` check out of `tests/metal/requirements.json` until its golden runner and recorded PLAN fixtures exist.
- **Reason:** The requirements map is validated against real source paths; listing the planned but absent test would make the test-map suite fail and imply coverage that is not yet implemented. The documentation still records the intended Mac setup golden suite as follow-on work.
- **Review required:** See REV-046. Add the setup test and its recorded engine plans, then restore the requirement mapping before accepting the setup golden-set phase.


### D-IMPL-254: Report ignored legacy expert blobs in the Mac CLI

- **Decision:** The Mac `WeightStore` continues to read routed experts from GGUF shards and ignores a sibling pack `experts.bin`. Return a warning from inspection and print it to stderr only after a successful inspection. Keep the warning path covered by the CLI plan test.
- **Reason:** `experts.bin` is an optional artifact from the upstream PC pack flow and is not an input to the Mac engine. An explicit warning makes the selected Mac data source visible without pretending the legacy blob is consumed.
- **Adversarial review and resolution:** Review confirmed the reader tests cover v3/v4 acceptance and malformed metadata, and found the CLI warning itself was untested. Added a real `strata --plan-only` assertion that the warning appears and the PLAN output stays unchanged; moved warning emission after successful inspection.
- **Review required:** See REV-050. Recheck warning visibility through the executable CLI and verify the new reader CTest remains in the host test map.

### D-IMPL-255: Test the Mac pack index contract directly

- **Decision:** Add a dedicated host CTest using the production `WeightStore` parser and a compact generated pack. Require the valid 19-field index and `compat_bf16: false` manifest to load, and refuse a native-format row whose `code_bits` is not 8, a `compat_bf16: true` manifest, and an 18-field index row.
- **Reason:** The parser already enforces these constraints in the full WeightStore integration test, but an explicitly mapped R7 test keeps the Mac-only format contract and its rejection cases visible without large downloaded model packs.
- **Adversarial review and resolution:** The R5 hostile review found that source-derived expert-table equality needed direct regressions for a structurally consistent but incorrect expert count and for a source shard offset. Added both mutations alongside the existing role-type mutation; each still passes structural checks and is rejected by the exact GGUF-derived layout comparison. A later review found the initial `code_bits` negative mutated a native row and failed too early; changed it to mutate only the `code_bits` field of a valid F32 row. The final review confirmed that it reaches the intended parser check and found no further issues.
- **Review required:** See REV-051. Run this host test and check the test map, then parse the seven pinned real catalogue packs when they are available; synthetic packs do not validate their provenance or revision.


### D-IMPL-256: Verify every expert slice against its GGUF shard

- **Decision:** Add a dedicated host CTest that compares WeightStore's expert tensor source offsets with each parsed GGUF shard's `data_start + tensor.offset`, then checks every planned expert slice against `base + expert_index × slab_bytes`. Exercise all 48 layers, all three expert roles, and all experts in five sparse layouts: a 512-expert single shard, Swift IQ2_XS-like and IQ3_XXS-like two-shard boundaries, a synthetic v4 per-role shard split, and a 256-expert Coder layout. Also check the shard selected for every role and that each slice's window-relative range stays within its planned window.
- **Reason:** The Mac loader reads experts directly from GGUF, and the window planner turns each role's contiguous expert slabs into file-window slices. Checking only slab counts can miss a shifted base, wrong per-expert stride, or a role assigned to the wrong shard. Sparse fixtures cover those formulas without allocating full model payloads. The seven pinned catalogue packs are not available in this checkout, so the generated fixtures cannot establish their real offsets or provenance.
- **Adversarial review and resolution:** The hostile review found no address-arithmetic, false-pass, fixture-coverage or shard-selection issues. It caught a stale reference to four layouts in the review checklist; that count is now five. The focused CTest and test-map test pass, and the full suite has 83 passing tests and one expected real-input skip.
- **Review required:** See REV-052. Recheck the seven pinned catalogue packs when T3 supplies them before claiming their real addresses are verified.

### D-IMPL-257: Verify PLE oracle rows through the production GGUF readers

- **Decision:** Register a host CTest for a small standalone PLE file and a split GGUF whose PLE tensor has a nonzero offset. Register a model CTest that scans every `.gguf` under `STRATA_TEST_MODELS`, checks each discovered PLE tensor against the frozen head/tail samples and full-row L1 sums for the three `ple_parity` probes, and requires the canonical `qwen-Q2_0` standalone shard 2 plus the canonical `swift-IQ2_XS` embedded shard 1. Validate `split.no`, `split.count` and `split.tensors.count`; require the standalone table's documented 192-byte data start and exact file size; and require the embedded table's parsed tensor offset and exact byte size. Compare each reader result with decoding its raw tensor bytes, and compare the packed probe bytes between the required ISTA and Swift sources. The frozen oracle stores summaries for the first two rows rather than all 160 decoded values, so the test documents that precision and does not claim a full-vector independent oracle.
- **Reason:** The standalone ISTA table and Swift's table-in-shard layout have different offsets and file-extent rules. Deriving the Swift address from the tensor directory prevents a file-start assumption from appearing to work on a large quantized row. Since the pinned models are not present in this checkout, the real-model test must fail closed when its required files are absent; the compact host fixture validates reader wiring without claiming the real oracle has passed.
- **Adversarial review and resolution:** The first hostile review found that required model paths could pass without proving the expected shard positions, a filesystem query error could be missed, and the checked-in oracle stores row summaries rather than all 160 values. Required tags and split metadata are now strict, filesystem errors fail immediately, the required ISTA/Swift packed probe bytes are compared, and the documented assertion scope matches the frozen data. Re-review found no remaining P1/P2 issues. The focused host test, test map and full host/Metal suite pass; the pinned real-model scan remains open.
- **Review required:** See REV-053. Run the model CTest on the pinned catalogue inputs when T3 supplies them; the generated fixture cannot verify model provenance or the actual table values.


### D-IMPL-258: Validate MTP runtime files through the production reader

- **Decision:** Identify each of the seven catalogue shard-1 GGUFs by its documented `general.file_type`, expert count, presence or absence of Swift recurrent-layer metadata, and `split.count = 2`, `split.no = 0`, `split.tensors.count = 1224` values. Pass that identity, the matching pack directory and `mtp/rt/` runtime through `WeightStore::inspect`. Check the three binary paths and sizes; parse `dense.txt` under the production reader's 1 MiB limit and verify the exact 29 tensor names, formats, dimensions, byte counts and aligned non-overlapping extents; verify shipped draft-vocabulary sizes and unique in-range IDs. Add a host regression that truncates synthetic `experts.bin` by one byte and requires the production inspector to reject it.
- **Reason:** MTP parsing and binary-size checks already live in the production `WeightStore`; using that reader against each correctly identified catalogue model avoids a second parser and avoids falsely passing the same model copied under several directory names. The small host regression checks truncation without downloading catalogue models or reading hundreds of megabytes. The production inspector creates persistent `.strata-artifact-set.lock` files in the parent directories of the resolved GGUF parent, selected pack directory and MTP runtime directory. Those lock-containing parent directories must be writable, while the inspected model, pack and runtime directories can remain read-only.
- **Adversarial review and resolution:** The first review found that generic architecture and shard-number matching could accept the wrong catalogue model and that the docs overstated exact `dense.txt` size checking. Tag-specific metadata and split checks are now required, and the docs distinguish the three binary file sizes from the bounded parsed text index. Final review prompted exact ordered vocabulary comparison against one of the three shipped files. Focused warning-as-error build, vocabulary and truncation host regressions, test-map check, missing-input fail-closed check and full 85-test ASan CTest pass; one real-input test is expectedly skipped. Final hostile re-review found no remaining P1/P2/P3 issues after the lock-path documentation correction. Pinned catalogue files are absent, so the model-backed check remains unverified.
- **Review required:** See REV-054. Code review is complete. Run the model CTest on the pinned catalogue directory when T3 supplies it, and verify the inspector's lock files are created in the documented parent directories.


### D-IMPL-259: Copy tiered tensor metadata before destroying its owner

- **Decision:** Keep a value copy of the dense tensor's `TieredTensorBinding` in `metal_window_storage_test` before resetting `TieredStorage`, then use that copy to verify the anonymous arena remains valid through the retained `Context` buffer reference.
- **Reason:** The layout owns its binding vector. The test intentionally destroys storage before checking that the arena buffer is retained, so a reference into the destroyed layout becomes dangling even though the arena itself remains alive.
- **Adversarial review and resolution:** AddressSanitizer found the original dangling reference. The value copy removes the use-after-free while preserving the intended retained-arena check. Focused ASan verification and the full host/Metal suite pass; hostile re-review confirms only immutable copied metadata is used after destruction and found no P1/P2 issues.
- **Review required:** See REV-055. No further review remains for this fix; repeat the focused ASan test if the lifetime assertion changes.


### D-IMPL-260: Check tiered buffers against the live GGUF mapping by address range

- **Decision:** Keep `proc_regionfilename` and `fs::equivalent` for identifying the actual source GGUF mapping. Check each tiered `BufferRef` by comparing its CPU-visible address range with the live GGUF mapping range; do not ask `proc_regionfilename` for a Metal buffer's region name.
- **Reason:** A minimal Metal probe showed that a valid anonymous `mmap` passed to `newBufferWithBytesNoCopy` kept the same pointer while `proc_regionfilename` then reported `/usr/share/icu/icudt78l.dat`. Metal can therefore expose an unrelated region name for a no-copy buffer, making the old path-equivalence assertion nondeterministic and capable of false positives. The direct range comparison reliably catches a buffer that wraps the source mapping address itself. The production tiered allocator uses `MAP_ANON` for its dense arena and `newBufferWithLength` for its other arenas and tables.
- **Adversarial review and resolution:** A focused Metal probe reproduced an unrelated region filename for an anonymous no-copy buffer. The revised test uses overflow-checked CPU address ranges for all six buffers and retains fail-closed source-map path identity checks. Focused ASan verification and the full host/Metal suite pass; hostile re-review found no P1/P2 issues. This detects a buffer overlapping the active mapping, not a distinct alias at another virtual address.
- **Review required:** See REV-056. No further review remains for the current allocator; rerun the ASan tiered-copy test if the allocation strategy changes.

### D-IMPL-261: Compose the ordered GR read stages behind one encoder

- **Decision:** Add `encode_gr_read` as the ordered norm → down/injection → up/mix composition for T=1..8. Require all twelve buffer views to be pairwise disjoint, reject calls from inside an open concurrent dispatch group, and provide both a caller-owned `Encoder` overload and a `CommandBuffer` overload that owns one transactional encoder segment.
- **Reason:** L9 needs to invoke the same three validated kernels in a fixed order. A small coordinator centralizes the wiring and leaves concurrent-dispatch barriers to `Encoder`; the open-group check prevents dependent dispatches from bypassing those barriers. Disjoint views keep residual folding, stage scratch, and previous/next injection values distinct. The CommandBuffer wrapper guarantees rollback if a later stage rejects its inputs; callers composing into a larger Encoder segment must propagate failure from their enclosing `Encoder::encode` body.
- **Adversarial review and resolution:** The first hostile review found that late-stage rollback was covered through the `CommandBuffer` wrapper but not directly through the caller-owned `Encoder` overload. Added a direct test that propagates the helper's `false`, verifies the outer segment is discarded and checks every output remains untouched. Final hostile re-review found no remaining actionable issues. The focused ASan serial and concurrent GR CTests pass after the repair; the full ASan non-model suite passed 84 tests with one expected missing-model skip before this test-only addition, and the non-final test-map check passes with its existing planned-work diagnostics.
- **Review required:** See REV-057. Code review is complete for this composition. Window/Prefill production wiring and model-backed decode verification remain open.

### D-IMPL-262: Compose the prior FFN write with the Window PLE entry

- **Decision:** Add `encode_window_ple_entry` for the L9 layer-1 PLE path. It takes layer 0's block output and injection plus the existing `PLEBlockBuffers`, uses `ple.post.residual` as the in-place residual/output of `strata_gr_post`, then invokes `encode_ple_block` in the same ordered Encoder segment. Support T=1..8 and preserve the caller's `PLEHistoryUpdate` policy. Reject an open concurrent dispatch group and provide both caller-owned Encoder and standalone CommandBuffer overloads.
- **Reason:** The Window order in [04](04-metal-kernels.md), section 5.1, requires the pending layer-0 FFN write to update R before PLE computes its query norm and residual. Deriving both GR residual and output from the PLE residual view removes a second buffer reference that could disagree and makes the handoff explicit. Encoder barriers order the writes and reads in concurrent mode; the standalone wrapper discards the entire segment if PLE validation fails after GR has been encoded.
- **Adversarial review and resolution:** The hostile review found that the documentation's focused-test pass statement predated the added caller-owned late-failure regression. Rebuilt and reran both focused CTests against the final regression; both pass. The review found no code defect. The PLE fixture compares the GR probe against its scalar oracle and uses the GPU-produced probe output for PLE's query oracle, preventing an oracle mismatch from preceding-stage rounding. It checks T=1 standalone and T=2 caller-owned success, late-history rollback through both overloads, and open-group rejection in concurrent mode.
- **Validation:** The ASan `metal_ple_test` target builds cleanly; serial and concurrent `metal_ple_test` CTests pass 2/2 with the caller-owned rollback regression. The full ASan non-model suite reports 84 passed and one expected unavailable-model-input skip; that full run preceded the final test-only regression. `test_map.py` and `git diff --check` pass, with test-map reporting its existing non-final planned-work diagnostics.
- **Review required:** See REV-058. This composes only the L9 prior-write and PLE block; full Window command assembly, state commit, dump hooks, Prefill reuse and model-backed decode remain open.

### D-IMPL-263: Compose the speculative Window GDN block

- **Decision:** Add `encode_window_gdn_block` for the seven documented GDN window rows, including both Q8_1 quantizers and the qkv, gate and output mat-vectors. Derive compiled columns as the next supported width (1, 2, 4 or 8), require non-BF16 weight formats for Q8_1 activations, keep state/history read-only in window mode, and require all supplied Context-owned views to be pairwise disjoint. Provide caller-owned Encoder and transactional CommandBuffer overloads; reject calls from inside an open concurrent group.
- **Reason:** The existing validated components implement every stage, but no one API enforced their documented dependency order. A single coordinator lets Window append the chain to an ordered segment, reuses the mat-vec sink and active-row metadata serially, and rejects aliases across separate kernels that each local encoder cannot see. Speculative evaluation must leave recurrent state and convolution history untouched until accepted-prefix commit.
- **Validation:** The ASan `metal_gdn_test` target builds; focused serial and concurrent GDN CTests pass 2/2. The composed T=1 CommandBuffer, T=2 caller-owned Encoder and T=3 padded-row chains match the scalar oracles, the padded sink matches the output matvec oracle, mixed/history cross-stage aliasing is rejected, both late-failure paths discard prior stages, and state/history/input guards pass. `test_map.py` and `git diff --check` pass with the repository's existing non-final planned-work diagnostics. Both hostile reviews found no remaining code-level blockers; the follow-up verified T=3 padding/sink coverage, alias rejection and the corrected host inventory.
- **Review required:** See REV-059. Production Window assembly, commit scheduling, Prefill reuse and model-backed decode remain open.


### D-IMPL-264: Commit an accepted GDN prefix transactionally

- **Decision:** Add `encode_window_gdn_commit` for one layer and `n_keep` in 0..8. For positive prefixes, validate Context ownership and pairwise disjointness, then encode convolution-history advancement from the accepted qkv rows followed by recurrent-state replay from the accepted h/gate/beta/z rows. Require a separate output scratch view for the recurrent kernel's unused output binding, reject open concurrent groups, and provide caller-owned Encoder and transactional CommandBuffer forms. Treat zero rows as a buffer-free no-op.
- **Reason:** The speculative block coordinator deliberately leaves history and recurrent state untouched. Commit must apply only the accepted prefix, in the documented history-then-state order, and a late second-stage validation failure must roll back the first dispatch too. One coordinator gives the future multi-layer Window scheduler a safe per-layer unit without claiming that scheduler is implemented.
- **Validation:** The ASan `metal_gdn_test` target builds; serial and concurrent GDN CTests pass 2/2. T=1, T=2 and T=3 fixture cases commit prefixes 1, 1 and 2 respectively, compare exact history and scalar-oracle state, verify replay inputs and scratch remain unchanged, exercise the n_keep=0 no-op and n_keep=9 rejection, reject history/state aliasing and open concurrent groups, and verify both overloads discard a prior history dispatch after late gamma failure. `test_map.py` and `git diff --check` pass with existing non-final planned-work diagnostics. Hostile review found no blocking issue in the coordinator, prefix oracles, guards, rollback, or documentation.
- **Review required:** See REV-060. The Window scheduler must still fence after speculative use, commit all state in layer order, handle the full accepted prefix across QSA/indexer/KV/PLE/GDN state, and verify production/model-backed decode parity.

### D-IMPL-265: Expose read-only full-mode weight views through ModelStorage

- **Decision:** Add `ModelStorage::full_mode_storage()` returning a borrowed `const FullModeStorage*`. It is non-null only for an opened and warmed full-mode owner and exposes named dense tensor views, file windows, the converted-weight arena, and expert address/resident tables. Every public full-mode buffer view is a `ReadOnlyBufferRef`: it preserves the Context generation token and exposes CPU contents only as `const void*`. The accessor returns null in tiered mode, and its resolver pointer cannot outlive `ModelStorage`.
- **Reason:** The full-mode owner already retains the validated mappings and resolved GPU address tables needed by a Window encoder, but `ModelStorage` kept the owner private. A borrowed const accessor exposes those resources without transferring their lifetime or storage-mode selection to the caller. Shallow `const FullModeStorage*` alone was insufficient because the former `BufferRef::contents` was `void*`; `ReadOnlyBufferRef` closes that host-side mutation path while preserving stale-view checks. Tiered storage has a different binding path and must not surface this resolver.
- **Validation:** The ASan `metal_window_storage_test` target builds and `metal_model_storage_owner_test` passes. Its owner fixture resolves a Context-owned `token_embd.weight` view and both Context-owned expert tables through `ModelStorage`, statically checks that contents are const and no implicit conversion to mutable `BufferRef` exists, verifies that an enclosing transaction invalidates a copied read-only view, and confirms that tiered mode returns null. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile review found no actionable blockers.
- **Review required:** See REV-061. Future Window binding must carry an explicit read-only binding type for these model views; do not retain the borrowed resolver pointer beyond the ModelStorage owner.

### D-IMPL-266: Add an explicit read-only Encoder input binding

- **Decision:** Add `Buffers::read_only(BufferRef)` to create a generation-checked `ReadOnlyBufferRef` only for a live allocation owned by that Context. Add an `EncoderReadOnlyBufferBinding` overload for MSL `const` buffer inputs. It validates ownership, extent, alignment and argument-slot uniqueness like writable bindings, and rejects overlap with any writable view by handle range, CPU backing range or GPU address range. The existing dispatch signature remains available for mutable and legacy call sites.
- **Reason:** Full-mode model storage now exposes CPU data as const, but the generic Encoder accepted only mutable `BufferRef` values. An explicit binding preserves that distinction through dispatch and prevents accidental model-weight writes or read/write aliases in one kernel invocation. Existing mutable bindings remain unchanged to avoid altering kernels with in-place outputs.
- **Validation:** The ASan `metal_encoder_test` target builds and its CTest passes. A `device const uint*` self-test input is copied through the new binding and compared exactly; two distinct no-copy MTLBuffer handles over one CPU backing are rejected when their views overlap, while adjacent disjoint views are accepted and copy exact values. Foreign Context promotion and misaligned read-only input are rejected. Compile-time assertions check that the contents pointer is const and `ReadOnlyBufferRef` cannot convert to `BufferRef`. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile review found the missing distinct-handle regression; after it was added, follow-up review found no other actionable blockers. The fixture exercises CPU-backing overlap; GPU-address-only alias overlap is code-reviewed but is not isolated by a fixture.
- **Review required:** See REV-062. Use the read-only binding only when the corresponding MSL argument is declared `const`; verify the overlap and input/output access contract for each future Window dispatch.

### D-IMPL-267: Implement the T=1 Window input stem

- **Decision:** Add `encode_window_embedding` for the first T=1 Window segment only. It reads the token from the `WindowParams` token field through a bounded `ReadOnlyBufferRef`, gathers one embedding row from the read-only model table, then broadcasts it into four GR streams through another read-only input view. Keep both dispatches in one ordered Encoder segment and expose standalone transactional and caller-owned forms. The parameter block must stay unchanged and all views must stay alive until GPU completion; callers wait before reusing a WindowParams ring slot. This does not implement the remaining Window layers or a complete decode.
- **Reason:** The documented decode order begins with embedding gather and GR stream initialization. Wiring that real first stage validates the read-only model-weight path and ties the ABI-compatible token ring to existing Metal kernels without claiming the still-missing scheduler, layer stack, head, sampler or state commit.
- **Validation:** `metal_embed_test` and `metal_embed_test_concurrent` pass 2/2 after the hostile review fixes. Read-only Q3_K, IQ3_S, IQ4_XS and BF16 gathers match ggml CPU; the Window BF16 stage checks exact embedding and four-stream bytes, parameter/table immutability, guards, T=2 and out-of-vocabulary rejection, and late-failure discard for both standalone and caller-owned Encoder segments. The Metal test map passes in non-final mode with existing planned-work diagnostics.
- **Review required:** See REV-063. Do not mark L9 complete until all 48 layers, PLE/GDN/QSA/MoE, head/sampler, state commit, dump hooks and model-backed parity are connected and validated.


### D-IMPL-268: Connect the T=1 layer-0 attention read to speculative GDN

- **Decision:** Add a T=1 Window coordinator that validates the published `WindowParams` and its `StepRow`, requires the ordinary speculative `n_keep == 0` path, runs `encode_gr_read` with `apply=false` (`pending(0) == 0`), then runs the speculative GDN block in one ordered Encoder segment. The GR `mixed` output and GDN `mixed` input must be the same live view; all other cross-stage views and the parameter block must be disjoint. Do not commit GDN state/history or claim full Window decode.
- **Reason:** The documented layer order is GR attention read followed by GDN. A narrowly scoped coordinator exercises the real handoff and automatic serial/concurrent barriers while preserving speculative state until a later accepted-prefix commit coordinator decides to mutate it.
- **Validation:** The ASan `metal_gdn_test` build succeeds; `metal_gdn_test` and `metal_gdn_test_concurrent` pass, as do `metal_embed_test` and `metal_embed_test_concurrent` (4/4). The composed fixture matches the component chain byte-for-byte, proves `apply=false` against a nonzero `apply=true` control, preserves residual and speculative state/history, rejects invalid T/nc/n_keep/token/StepRow, split handoff and cross-stage alias inputs, and verifies late-failure discard through both overloads. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile review found two test-coverage gaps; both were fixed and the final follow-up found no actionable issue.
- **Review required:** See REV-064. L9 remains incomplete until the remaining layers, PLE/QSA/MoE path, head/sampler, state commit, dump hooks and model-backed parity are connected and validated.


### D-IMPL-269: Compose layer-0 attention, GDN and FFN GR reads

- **Decision:** Add T=1 caller-owned and standalone transactional coordinators for attention GR read (`apply=false`), speculative GDN block, then FFN GR read (`apply=true`) in one ordered Encoder segment. Require `n_keep == 0`; the three `mixed` roles share one exact view; GDN and FFN share the exact `block_out`; both GR reads share `residual`, `normalized`, `rs` and `lo`; their `inject_prev`/`inject_out` views swap exactly. Use a separate attention `block_out` placeholder because `pending(0) == 0` makes it unread; the Window's embedding scratch can serve after broadcast. Bind layer-0 `hc_attn_*` and `hc_ffn_*` as distinct weight views; reject all other cross-stage overlap. Stop before router/MoE and the FFN write.
- **Reason:** This follows [04](04-metal-kernels.md), section 5.1, while matching the existing `WindowScratchLayout` without allocating duplicate GR scratch. Distinct attention and FFN hyper-connection tensors are separate in the model artifact even where shapes match. The ordered segment makes the GDN mixed input complete before FFN overwrites that scratch and lets the FFN read apply the GDN block output and attention injection to the residual.
- **Validation:** The ASan `metal_gdn_test` target builds; `metal_gdn_test` and `metal_gdn_test_concurrent` pass, as do `metal_embed_test` and `metal_embed_test_concurrent` (4/4). The fixture compares the full three-stage coordinator with the component chain, checks the apply=true residual fold and swapped injection output, verifies speculative GDN state/history, rejects invalid nc/n_keep, split injection, exact and partial attention/FFN weight overlap, and checks standalone and caller-owned late FFN rollback. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile design and implementation reviews found no actionable issue.
- **Review required:** See REV-065. The API accepts pre-resolved views, so production callers must bind the correct layer-0 `hc_attn_*` and `hc_ffn_*` tensors; this coordinator does not construct model bindings or complete the Window layer.


### D-IMPL-280: Add validated grouped MoE gate/up dispatch

- **Decision:** Add a typed encoder for grouped gate/up projections across Q2_0, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S and IQ4_XS. Support 1..8 token rows, 1..512 active experts and the 48-layer plan/address tables. Bind address, Q8_1 activation and plan views read-only; bind separate gate/up outputs. Validate Context ownership, extents, alignment, shape limits, format, pairwise overlap and ordered dispatch boundaries. Pass row count in a dedicated 12-byte gate/up parameter ABI. Each kernel guards plan row/group counts, group entry ranges, selected expert IDs, token indices and null role addresses before reading or writing. Provide caller-owned Encoder and standalone transactional forms.
- **Reason:** The grouped GPU kernels and their scalar-oracle twins existed, but the oracle tests called raw dispatch and did not exercise a reusable engine binding contract. The typed API centralizes their ABI, active expert stride and transaction rules for later Window scheduling. GPU plan partition integrity and selected expert address extents remain trusted-engine preconditions; the API does not read back GPU plan arrays. Require a plan produced for the same rows, expert stride and layer by `encode_moe_plan`; tiered consumers must be queued only after successful `ExpertStore` resolution.
- **Validation:** The `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. Every supported grouped format now reaches its GPU kernel through the typed encoder and retains its oracle comparisons. Rejection fixtures cover invalid dimensions/layers/formats, mismatched GPU plan row/expert counts, invalid token/group ranges, short and misaligned views, exact/partial input-output and output-output aliases, caller-owned rollback and concurrent open-group refusal. `test_map.py` and `git diff --check` pass with existing non-final diagnostics. Hostile findings and fixes are recorded in REV-076.
- **Review required:** See REV-076. Keep production expert dispatch behind validated plan resolution and the tiered event/`ExpertLeaseSet` protocol; full Window integration and model-backed inference remain open.

### D-IMPL-282: Validate routed and shared-expert SwiGLU dispatch

- **Decision:** Add caller-owned `Encoder` and standalone transactional forms for the existing routed MoE and shared-expert SwiGLU kernels. Accept 1..8 input rows; derive the output-entry count as `rows * 10` for routed experts or `rows` for the shared expert. Bind gate/up F32 inputs as read-only and the F32 output as writable, with Context ownership, extent, alignment and output/input alias checks. Require an ordered dispatch boundary and cover all 640 columns with three 256-thread groups.
- **Reason:** The kernel and CPU oracle existed, but their tests bypassed the engine API. A typed binding covers the stage after grouped gate/up and before Q8_1 quantization, while preventing output aliasing and making the row-to-entry mapping explicit.
- **Validation:** The `metal_moe_test` target builds; its serial and concurrent CTests pass 2/2. Routed and shared SwiGLU cases for 1..8 rows and random, edge and fixture inputs now execute through the engine encoder and compare with the scalar oracle. Fixtures check read-only input guards, short and misaligned views, invalid rows, exact and partial output aliases, caller-owned rollback, successful caller-owned encoding, and open-group refusal in concurrent mode. `test_map.py` and `git diff --check` pass with existing diagnostics. Hostile findings and resolutions are recorded in REV-078.
- **Review required:** See REV-078. During later MoE composition, preserve the 640-column tail, use routed `rows * 10` entry dimensions, and keep Q8_1 quantization and grouped down projection ordered after SwiGLU; full Window scheduling remains unfinished.

### D-IMPL-283: Validate grouped routed-expert down dispatch

- **Decision:** Add caller-owned `Encoder` and standalone transactional forms for the grouped Q2_0 and IQ4_NL down kernels. Accept 1..8 rows and 1..512 active experts; bind the expert address table, Q8_1 activations and plan read-only, with a separate parts output in original route order. Validate Context ownership, extents, alignment, input overlap, supported format and ordered dispatch boundaries. Pass row count in a 12-byte parameter ABI and guard plan row/group bounds, expert IDs, output destinations and null down-weight addresses in the GPU kernels.
- **Reason:** The grouped down twins had scalar-oracle coverage but were dispatched with raw bindings. The engine API now expresses the plan/shape contract after SwiGLU Q8_1 quantization and blocks stale or overlapping views before GPU submission. Kernel destination checks prevent malformed plan indices from writing outside `parts`; group partition integrity and selected weight extents still rely on trusted engine outputs.
- **Validation:** The `metal_moe_test` target builds; full serial and concurrent MoE CTests pass 2/2. Q2_0 and IQ4_NL grouped outputs use the typed encoder across random, edge and fixture inputs, retaining the 80-group, single-entry parity and fixed IQ4_NL split-FMA probes. Rejection fixtures cover invalid dimensions/layers/formats, short and misaligned views, output/input and read-only input overlap, caller-owned rollback and concurrent open-group refusal. `test_map.py` and `git diff --check` pass with existing non-final diagnostics. Hostile findings and fixes are recorded in REV-079.
- **Review required:** See REV-079. Keep the dispatch after routed SwiGLU Q8_1 quantization and successful tiered-plan resolution; production Window scheduling and end-to-end inference remain unfinished.

### D-IMPL-284: Validate MoE routed/shared combine dispatch

- **Decision:** Add caller-owned `Encoder` and standalone transactional forms for `strata_moe_combine`, accepting 1..8 rows. Bind routed parts `[rows, 10, 2560]`, route weights `[rows, 10]`, shared output `[rows, 2560]` and shared gate logits `[rows]` as read-only inputs; require a separate writable output `[rows, 2560]`. Validate Context ownership, byte extents, F32 alignment, output/input disjointness and ordered dispatch boundaries.
- **Reason:** The combine kernel and exact arithmetic oracle existed, but both standalone and composed shared-expert tests bypassed the engine API. This makes the FFN's final routed/shared merge shape and binding contract reusable while retaining the model-specific oracle and separate multiply/add rounding checks.
- **Validation:** The `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. The 1..8 row combine and composed shared-expert chains run through the typed encoder for random, edge and fixture inputs. Validation cases reject bad row counts, short/misaligned views and each output/input alias; caller-owned late failure and concurrent open-group refusal are covered. `test_map.py` and `git diff --check` pass with the existing non-final diagnostics. Hostile findings and fixes are recorded in REV-080.
- **Review required:** See REV-080. Preserve the routed FMA order and separate shared multiply/add from the analytic fixture, and keep the combined result before the FFN residual write; production Window scheduling remains unfinished.

### D-IMPL-285: Compose the routed MoE expert FFN stages

- **Decision:** Add caller-owned `Encoder` and standalone transactional forms for the ordered grouped gate/up → routed SwiGLU → Q8_1 hidden quantization → grouped down sequence, for 1..8 rows and 1..512 active experts. Validate all input and scratch extents, Context ownership, alignment and aliasing before the first dispatch. Permit a full-mode plan dispatch immediately before the composition in the same ordered segment. Require tiered callers to submit the composition only after successful `ExpertStore` resolution.
- **Reason:** The validated primitive encoders now existed but Window assembly still had to connect four routed expert stages and validate their shared scratch contract itself. A reusable coordinator closes that boundary while preserving the critical tiered-plan handshake: queuing an unguarded consumer before resolution could read unresolved expert addresses. This intentionally stops before the shared expert, combine, residual write and remaining Window layers.
- **Validation:** The `metal_moe_test` target builds; final serial and concurrent CTests pass 2/2. GPU-produced plan bytes and every routed FFN intermediate match scalar plan, gate/up, SwiGLU, Q8_1 and down oracles at T=1 and T=8. Fixtures cover invalid extents, read-only input and scratch-output aliases, caller-owned rollback, concurrent-group refusal and buffer guards. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. Hostile findings and fixes are recorded in REV-081.
- **Review required:** See REV-081. Keep the tiered resolver handshake outside the consumer submission and complete shared-expert scheduling, combine, residual write, later layers, state commit and dump hooks before marking L9 done or describing end-to-end inference.

### D-IMPL-286: Bind Q8_0 mat-vec inputs read-only

- **Decision:** Add `ReadOnlyQ8_0MatVecBuffers` and caller-owned `Encoder` plus standalone transactional overloads for the Q8_0-by-Q8_1 mat-vec. Bind Q8_1 activations, Q8_0 weights and active-row metadata as `ReadOnlyBufferRef`; retain separate writable output and sink views. Support widths 32..16640 divisible by 32 and compiled column counts 1, 2, 4 or 8. Allow read-only inputs to overlap one another, but reject output or sink overlap with every input and with each other.
- **Reason:** Full-mode storage intentionally exposes model tensors as read-only, while the shared-expert Q8_0 projections previously had only the mutable mat-vec binding. A distinct read-only path lets those projections use model views without weakening `ModelStorage`'s host-side mutation boundary or copying weights into scratch. This adds a reusable projection primitive; it does not compose shared-expert execution, combine, or the production Window path.
- **Validation:** The `metal_mv_test` target builds; serial and concurrent CTests pass 2/2. Small-K T=1..3/NC=4 and long-K T=1/NC=1, 2 and 8 fixtures compare output and sink bytes to the scalar Q8_0 oracle, including the one-float no-padding sink. Each padded width checks its exact written range before any later output reset. Tests also cover standalone and caller-owned success, read-only input overlap, short/misaligned views, active-row mismatch, output/input and output/sink aliases, caller-owned late-failure rollback, input immutability and guards. `test_map.py` and `git diff --check` pass with existing non-final diagnostics. Hostile findings and resolutions are recorded in REV-082.
- **Review required:** See REV-082. Use this path only for MSL `const` Q8_0 kernel inputs. Next compose the shared-expert gate/up, SwiGLU, Q8_1, down, gate and combine sequence, then validate its handoff to the Window residual write; keep L9 and end-to-end inference open.

### D-IMPL-281: Extend the T=1 router prefix through GPU plan production

- **Decision:** Add standalone transactional and caller-owned `Encoder` forms for the T=1 layer-0 attention/GDN/FFN-read, BF16 router, top-10 route, Q8_1 quantization and GPU MoE plan sequence. Reuse the route-ID output as the plan's read-only input; require separate layer-0 resident, usage and plan views. Validate their extents, alignment, Context ownership and disjointness from each other and every prefix buffer before encoding any stage. Support full and tiered plan modes for 256/512 experts. Stop after the plan dispatch in both modes; this API does not signal plan-ready events, call `ExpertStore`, or encode expert consumers.
- **Reason:** The previous ordered prefix produced exactly the IDs consumed by `encode_moe_plan`, but callers had to rebuild and validate the stage boundary themselves. One coordinator makes that dependency explicit and gives late validation a single rollback boundary. Tiered resolution remains outside the command buffer: an unguarded consumer queued after a failed resolver could read unresolved addresses, and releasing an event wait does not cancel later GPU work (REV-075). The caller must establish plan completion/readiness and only then submit expert work after successful resolution. This is a T=1 building block, not production Window scheduling or end-to-end inference.
- **Validation:** The ASan `metal_gdn_test` target builds; serial and concurrent CTests pass 2/2. The fixture checks tiered 512-expert and full 256-expert outputs against the route and plan oracles, including exact plan bytes, usage updates/immutability and input guards. The tiered case explicitly contains resident and missing groups. Standalone encoding rejects an undersized plan before prefix dispatch; a caller-owned case injects failure after the complete route/Q8_1/plan prefix has encoded and verifies command-buffer discard and unchanged outputs. `test_map.py` and `git diff --check` pass with existing non-final diagnostics. Hostile findings and resolutions are recorded in REV-077.
- **Review required:** See REV-077. Production callers must prove the plan-ready signal/wait and command-buffer boundary, resolve the tiered plan successfully before expert submission, retain expert leases through GPU completion, and complete the remaining Window layers, head/sampler and state commit before claiming decode support.

### D-IMPL-278: Retire completed expert layers at later plan events

- **Decision:** After `resolve_layer_after_plan` successfully waits for layer l's plan-ready value, retire the `ExpertLeaseSet` leases of every earlier layer before reading and resolving the current plan. Keep the command-buffer completion callback as the fallback for leases retained by the current or later layers.
- **Reason:** A plan-ready signal is encoded after the prior layers’ expert reads in the same serial Window command buffer. It therefore provides the GPU ordering proof needed to release their slot pins without waiting for the whole Window or guessing from elapsed time. This enables parity-half reuse when the next cold plan is resolved. The operation lock is acquired only after the event wait and held through retirement, miss resolution and host signaling, so cache/prefetch work cannot race the slot transition.
- **Validation:** The ASan `metal_window_storage_test` target builds and the `metal_tiered_copy_test` CTest passes. Its synthetic command buffer produces plans for layers 0 and 2, dispatches the bounded IQ2_XS gate consumer for layer 0, holds the completed layer’s slots while filling that parity half, and proves the layer-2 plan event automatically releases prior leases so another same-parity slot is available before command-buffer completion. `test_map.py` and `git diff --check` pass with existing planned-work and missing-check diagnostics. Hostile findings and fixes are recorded in REV-074.
- **Review required:** See REV-074. This proof depends on one serial Window command buffer in flight. If Window scheduling permits overlapping command buffers, pin slots per window and retain them until each command buffer completes.

### D-IMPL-279: Keep failed tiered-plan wait release under request ownership

- **Decision:** `ExpertStore` leaves the GPU's layer-ready event unsignaled when plan validation or miss resolution fails. The request owner remains responsible for marking the request failed and deciding how to terminate its queued work.
- **Reason:** The shared-event release value only unblocks an already-queued command buffer; it does not cancel later expert kernels or make unresolved address-table entries safe to read. Automatically signaling from `ExpertStore` can therefore turn a resolver error into an invalid GPU memory access.
- **Validation:** The tiered bridge rejects malformed plans without signaling layer-ready. The success path gates the grouped `strata_moe_gu_iq2_xs` gate kernel while leases are pinned, verifies slot reuse is refused, then releases the consumer and checks that its routed outputs were written before layer completion. A separate failed-request fixture confirms cleanup waits for plan arrival before clearing the GPU-produced group count, then waits each interleaved GPU signal before host release and drains the command buffer with untouched consumer outputs. Failure-path execution of production Window command buffers remains unimplemented.
- **Review required:** See REV-075. The initial production scheduler must place expert consumers in a later command buffer submitted only after successful resolution, or add a GPU-visible failure guard checked before address loads. Do not release waits in a command buffer with unguarded consumers after them; `Events::release_gpu_waits` satisfies waits and does not cancel commands.

### D-IMPL-277: Release blocked Metal event waits after a failure

- **Decision:** Add `Events::release_gpu_waits(timeout_ms, error)`. It validates the timeout before changing the shared event, advances it to `UINT64_MAX`, waits for committed or enqueued command buffers containing that event's waits to complete, and permanently rejects future waits and signals. It returns no future reserved values, makes `signaled_value()` return no value, and makes `wait()` fail rather than treating the sentinel as completion.
- **Reason:** A failed or timed-out tiered plan must not leave the single Metal queue blocked behind an event wait. The terminal value satisfies already-encoded waits; it does not cancel commands after them, so callers must first ensure any downstream work is absent or guarded. Draining submitted waiters bounds cleanup, and poisoning the event prevents the failure sentinel from being reused as a valid request value. The request owner still reports failure because released events do not prove successful GPU work.
- **Validation:** The ASan `metal_events_test` and `metal_window_storage_test` targets build; focused CTests pass 2/2. The event fixture proves invalid timeout input leaves the event usable, committed and delayed-commit enqueued command buffers exit their blocked waits after release, an event wait encoded but not enqueued before release cannot be submitted later, released events expose no completion value, and the plan bridge rejects a released event. `test_map.py` and `git diff --check` pass with the existing planned-work and missing-check diagnostics. Hostile findings and fixes are recorded in REV-073.
- **Review required:** See REV-073. A caller must treat release as terminal, discard unsubmitted command buffers, and fail the request even if released GPU work drains successfully. Before release, prove queued downstream GPU work cannot consume unresolved addresses; event release is not command-buffer cancellation.

### D-IMPL-276: Resolve tiered plans through a shared-event host bridge

- **Decision:** Add `ExpertStore::resolve_layer_after_plan` to wait for the GPU plan-ready value, acquire and read the selected tiered plan, resolve misses through the existing cache and file workers, and host-signal the immediately following layer-ready value on the same monotonic `Events` object only after success. Validate the bounded plan header before copying arrays; when `skip` is set, validate the four header words only and treat all entries as resident hits. Expose a borrowed `TieredStorage*` from `ModelStorage` so a tiered `ExpertStore` can be created without transferring storage ownership.
- **Reason:** The plan buffer and event protocol already have Context-owned lifetimes, and the existing resolver owns all address publication and lease pinning. A small bridge connects those components while preserving the documented signal → resolve → wait sequence and avoiding array reads in the skip path. The plan kernel uses a dynamic table stride, so the lower-level plan encoder accepts 1..512 experts to exercise the same GPU producer with the compact nine-expert storage fixture; route selection remains limited to product widths 256/512. Failed resolution must not wake the expert kernels; the caller owns wait release and request failure.
- **Validation:** The ASan `metal_window_storage_test` and `metal_moe_test` targets build. The `metal_tiered_copy_test` CTest passes, including a pre-submission non-adjacent event-value rejection, GPU-produced plan byte equality, cold-expert resolution, skip-header-only handling after array poisoning, and malformed-header failure followed by caller cleanup. `metal_moe_test` and `metal_moe_test_concurrent` pass 2/2 after widening the plan encoder to bounded synthetic active strides. `test_map.py` and `git diff --check` pass with the existing planned-work and missing-check diagnostics. Hostile review findings and fixes are recorded in REV-072.
- **Review required:** See REV-072 and REV-075. Keep the change scoped to this host bridge; production Window layer scheduling, lookahead and Prefill `PfPlan` handling remain open under U6/REV-024. Event-wait release is implemented separately in D-IMPL-277, and completed prior-layer lease retirement is implemented in D-IMPL-278. Before production wiring, ensure a failed resolver cannot unblock queued consumers with unresolved addresses.

### D-IMPL-275: Persist MoE usage counters with model weights

- **Decision:** Allocate a zeroed Shared `usage[48][n_expert]` table inside the selected full-mode or tiered-mode storage and expose one model-level accessor. Keep the table alive across decode windows; the tiered plan kernel updates it, while full mode binds it without reading or writing it.
- **Reason:** Adaptive swaps consume counts accumulated by the plan kernel across windows. A window scratch view would lose policy state when scratch is reused and blur the lifetime of persistent counters. The selected storage owner already bounds the resident/address/plan table lifetime and rolls allocations back transactionally.
- **Validation:** The ASan `metal_window_storage_test` target builds; focused serial and fixture-only concurrent CTests pass, including zero initialization, active `n_expert` row stride, Context ownership, and no overlap with mapped GGUF data. Hostile review findings and fixes are recorded in REV-071.
- **Review required:** See REV-071. Before adaptive swaps, add event-ordered CPU reads/decay and prove plan writes are complete before the adapt thread accesses usage.

### D-IMPL-273: Add a validated MoE plan encoder

- **Decision:** Add standalone transactional and caller-owned `Encoder` forms for `strata_moe_plan`, supporting T=1..8, active expert strides from 1 through 512 and the 48-layer `TieredMoePlan` array. Production route kernels still support only 256- and 512-expert models. Bind route IDs and residency as read-only inputs; bind usage and plans as writable outputs. Validate the active layer-strided table extents, the plan prefix through the selected layer, Context ownership, alignment and writable disjointness. Keep both residency and usage views bound in full mode to satisfy the shader ABI; the full-mode kernel still marks every group resident and leaves usage unchanged. Caller-owned segments must propagate failure so an earlier unsubmitted dispatch is discarded. Stop at planning; do not imply expert execution or tiered miss resolution.
- **Reason:** The kernel and CPU oracle already define the exact first-appearance grouping, resident flags, miss count, and tiered usage increments over the shared 1,620-byte `TieredMoePlan` ABI, but the production engine had no typed host dispatch. The validated encoder makes the existing all-T/full/tiered oracle suite exercise the engine's real resource binding and transaction rules while keeping `MissResolver` ordering as a separate production integration step.
- **Validation:** The ASan `metal_moe_test` target builds; serial and fixture-only concurrent CTests pass 2/2. The existing oracle suite covers T=1..8, both expert widths, full and tiered tables, exact plan bytes, exact usage writes, and guards through the engine encoder. Validation fixtures reject invalid shapes/layers, undersized and misaligned views, all five writable/input or output/output overlap pairs, and prove standalone and caller-owned rollback leave outputs untouched; concurrent mode also rejects an open group. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile review found two missing alias fixtures; they were added and the follow-up review found no further actionable issue. See REV-069.
- **Review required:** See REV-069. Before production use, connect the plan output to the tiered layer schedule only with the documented signal/wait ordering, plan visibility, miss reads, address writes and `ExpertLeaseSet` pin/retire lifecycle; preserve full-mode usage immutability.

### D-IMPL-272: Compose the T=1 router, route and Q8_1 prefix

- **Decision:** Add a T=1 standalone and caller-owned coordinator that runs the layer-0 attention/GDN/FFN-read chain, a read-only BF16 router mat-vec, top-10 MoE routing with `divide_weights=false`, and Q8_1 quantization of FFN `mixed` in that order. Derive the router's live-row metadata from a bounded read-only slice of `WindowParams.nc`; reuse `gdn.xq` only after GDN has consumed it. Require 256 or 512 experts, disjoint router scratch and parameter views, and an ordered dispatch boundary. Stop before `MoePlan` and expert execution.
- **Reason:** The Window decode's documented layer order is router projection, route selection and Q8_1 input preparation after the layer-0 FFN read. A single validated segment captures their data dependencies and uses the planned GDN Q8_1 scratch without allocating another row. The pre-resolved API can accept full-mode BF16 model weights while keeping tiered expert readiness and production binding out of scope.
- **Validation:** The ASan `metal_gdn_test` target builds; focused serial and concurrent CTests pass 2/2. Both 512-expert and 256-expert fixtures compare router logits bitwise with the scalar BF16 oracle, route IDs exactly and route weights at U(16), require bitwise equality with direct reciprocal-mode GPU routing, and prove the division-mode negative control differs on the same logits. Q8_1 bytes match exactly. The fixture verifies read-only router weights, output/input guards, GDN `xq` reuse, short-sink and router-tensor/parameter alias rejection before dispatch, caller-owned alias failure discard, and unchanged outputs on rejection. `test_map.py` passes with its existing non-final planned-work diagnostics. Hostile review's two test-coverage findings were fixed; the final review found no actionable issue.
- **Review required:** See REV-068. The coordinator accepts pre-resolved views only; production Window binding must resolve the full/tiered router tensor, keep full-mode views within `ModelStorage` lifetime, and add the tiered `MissResolver` handshake before scheduling plan and expert stages.


### D-IMPL-271: Keep BF16 mat-vec model inputs read-only

- **Decision:** Add a separate BF16 mat-vec API whose F32 activations, BF16 weights and active-row count are `ReadOnlyBufferRef` inputs. Preserve the existing mutable `MatVecBuffers` API for current callers. Allow read-only inputs to overlap each other, but reject output or sink overlap with every input and with each other. Route the standalone overload through `Encoder::encode` so validation failure discards the unsubmitted command buffer; require a caller-owned Encoder body to propagate false.
- **Reason:** `FullModeTensorView` exposes converted and mapped model weights as `ReadOnlyBufferRef`, while the existing mat-vec signature accepts only mutable `BufferRef`. The separate path preserves the ownership boundary and lets later T=1 router composition pass model views directly to the `device const` BF16 kernel.
- **Validation:** The ASan `metal_mv_test` target builds; focused serial and concurrent CTests pass 2/2. The fixtures compare standalone and caller-owned outputs bitwise with the BF16 oracle, including a synthetic T=1 512-expert router shape at n_in=2560 (the 256-thread kernel); permit read-only input overlap; reject every input/output and output/sink overlap, including partial CPU-backing overlap across distinct no-copy handles; reject short/misaligned views and mismatched active-row metadata; preserve input/guard bytes; and discard the first dispatch after a late caller-owned failure. `test_map.py` passes with the repository's existing planned-work and missing-check diagnostics.
- **Review required:** See REV-067. Before router integration, confirm the full-mode router view uses the supported BF16 width and alignment, and derive the T=1 live-row view from the Window parameter storage without extending its lifetime beyond the owning Window.

### D-IMPL-270: Add the validated MoE route encoder

- **Decision:** Add standalone transactional and caller-owned `Encoder` APIs for the existing top-10 route kernels. Accept only 256- or 512-expert logits, bind logits read-only, require separate ID and weight outputs, and expose the multiply-by-reciprocal versus division mode explicitly. Require caller-owned bodies to propagate `false` so prior dispatches are discarded. Keep this increment at the route primitive; do not imply router mat-vec, Q8_1, expert execution or Window integration.
- **Reason:** The shader and CPU oracle already define the tie order and weight arithmetic, but there was no validated engine host API. A typed wrapper lets later Window and Prefill composition reuse the kernel contract while preserving their different normalization modes and buffer transaction rules.
- **Validation:** The ASan `metal_moe_test` target builds; its full serial CTest and fixture-only concurrent CTest pass (2/2). Route IDs match the CPU oracle exactly; route weights satisfy U(16) and normalization checks; guards and input immutability pass for 256/512 experts. Validation cases cover invalid shapes/extents, output and exact/partial logits aliases, caller-owned late failure rollback, and open-group rejection in concurrent mode. `test_map.py` passes with the repository's existing non-final planned-work and missing-check diagnostics.
- **Review required:** See REV-066. Before composing the T=1 router, extend BF16 mat-vec to accept read-only model weights, then compose router mat-vec → route → Q8_1 in one ordered segment and test the `mixed`/`xq` scratch handoffs.

### D-IMPL-287: Compose the shared-expert FFN and routed combine

- **Decision:** Add `MoeSharedFFNBuffers` and standalone transactional and caller-owned `encode_moe_shared_ffn` forms. In one ordered segment, encode the read-only Q8_0 shared gate and up projections, shared SwiGLU, Q8_1 hidden quantization, read-only Q8_0 down projection, read-only BF16 shared gate, and routed/shared combine. Derive the smallest supported compiled column count for rows 1..8. Require all writable scratch, outputs and padded-column sinks to be Context-owned, correctly sized/aligned, and disjoint from all read-only inputs and each other; allow read-only inputs to overlap. Require caller-owned bodies to propagate false. Routed parts, route weights and mixed Q8_1 rows must already be available; tiered callers submit the consumer only after successful expert-plan resolution.
- **Reason:** The primitive kernels and their oracles existed, but there was no validated engine path that carried shared-expert outputs through Q8_1, down projection, gate scaling and final combine. A separate shared-FFN coordinator makes these dependencies reusable while respecting the asynchronous tiered resolver. It stops before the GR FFN residual write because the cross-handoff into layer state needs its own shape, alias and transaction contract; this step does not add Window scheduling or complete inference.
- **Validation:** The ASan `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. The composed path covers rows 1..8 with random, edge and fixture inputs, compares gate/up/hidden/Q8_1/down/gate/combine results against the scalar references, verifies padded sink values and exact sink write ranges, and checks read-only inputs and guards. Rejection tests cover invalid rows, mismatched active-row metadata, cross-stage input/output overlap, caller-owned late-failure rollback, and concurrent open-group refusal. `test_map.py` and `git diff --check` pass with existing non-final planned-work diagnostics. Hostile review identified only a direct standard-library include omission and missing decision/review records; both were corrected, and final follow-up review found no actionable issue.
- **Review required:** See REV-083. Before using this coordinator in Window, prove the combine output handoff to `strata_gr_post`, include the FFN residual/injection views in cross-stage alias validation, and preserve the full-mode versus tiered `ExpertStore` readiness boundary. Synthetic kernel/oracle tests do not establish end-to-end inference.

### D-IMPL-288: Apply shared MoE output to the GR residual

- **Decision:** Add `MoeSharedFFNWriteBuffers` and standalone transactional and caller-owned `encode_moe_shared_ffn_write` forms. Encode the shared FFN and routed combine first, then pass the combined output as `strata_gr_post`'s `block_out` and update the four-stream residual in place using the provided FFN injection. Validate residual and injection extents, ownership, alignment, their mutual disjointness, and their disjointness from every shared FFN input, scratch, output and sink before the first dispatch; pass only the active byte ranges to GR post. Keep the read-only FFN inputs, injection and initial residual contents unchanged until command completion, and do not access device-written scratch/output before completion. Caller-owned callers must propagate false from their Encoder body to discard earlier stages on failure.
- **Reason:** The layer's GR write consumes the combined FFN block output and the FFN injection after the shared/routed combine. A distinct composition makes that GPU handoff explicit and prevents output corruption from cross-stage aliases while preserving `strata_gr_post`'s established in-place residual behavior. This remains a typed, caller-supplied-view coordinator: route weights and routed parts must already be ready, and the Window scheduler and tiered resolver remain outside its scope.
- **Validation:** The ASan `metal_moe_test` target builds; serial and concurrent CTests pass 2/2. Rows 1, 3 and 8 exercise standalone and caller-owned successful writes; resulting residual values are compared with the scalar GR-post oracle using the actual GPU combine output. Rejection coverage includes residual/read-only-input aliasing, injection/shared-scratch aliasing, residual/injection aliasing, caller-owned late-failure rollback, and open concurrent-group refusal. An oversized residual view whose unused capacity overlaps the adjacent injection view is accepted when their active ranges are disjoint. Output guards and read-only inputs are checked.
- **Review required:** See REV-084. Before Window scheduling, confirm that routed outputs are complete and remain alive through command completion, that the tiered consumer is submitted only after successful `ExpertStore` resolution, and that the residual/injection ranges match production Window storage. These synthetic kernel/oracle tests do not establish end-to-end inference.

### D-IMPL-289: Compose routed/shared MoE FFN and GR residual write

- **Decision:** Add `MoeRoutedSharedFFNWriteBuffers` and standalone transactional and caller-owned `encode_moe_routed_shared_ffn_write` forms. Preflight the routed and shared FFN chains, validate that the shared Q8_1 input and routed parts use the routed producer's corresponding buffer starts, then check every cross-stage alias using the active byte extents before the first dispatch. Encode routed gate/up, SwiGLU, Q8_1 and down projection, followed by shared gate/up, SwiGLU, Q8_1, down projection, shared gate, combine and in-place GR residual write. Preserve the successful tiered `ExpertStore` resolution boundary and require caller-owned bodies to propagate false.
- **Reason:** The preceding APIs individually validated routed FFN, shared FFN/combine and residual write, but the production-shaped handoffs and aliases crossed their boundaries. A single preflighted ordered composition verifies those relationships and carries the combined output directly into GR post. Comparing active ranges allows arena capacity tails to overlap when the bytes used by the stages remain disjoint. The compositor does not create routes or plans, resolve tiered misses, or add a production Window call site.
- **Validation:** The ASan `metal_moe_test` target builds; focused serial and concurrent CTests pass 2/2. T=1..8 exercise routed/shared/combine/GR composition, including compiled-width padding sinks; scalar oracles cover shared projections, SwiGLU, down projection, gate logits, combine, routed parts and GR post. Tests reject cross-stage output and active residual aliases, accept overlapping unused view capacity, verify caller-owned rollback across routed and shared outputs, and reject an open concurrent group. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics.
- **Review required:** See REV-085. Before Window integration, confirm the full-mode plan producer and tiered `ExpertStore` resolver feed this composition with trusted route/plan/address views, verify the real Window active extents and output lifetimes through GPU completion, and retain the no-end-to-end-inference status until the production scheduler and model-backed checks are complete.

### D-IMPL-290: Compose the full-mode T=1 Window layer-0 MoE path

- **Decision:** The initial implementation added `WindowT1MoeLayer0FullBuffers` and standalone transactional and caller-owned `encode_window_t1_moe_layer0_full_mode` forms. It embedded the actual `WindowT1MoePlanPrefixBuffers` producer and routed/shared consumer, required `FullModeStorage`, bound expert addresses/resident/usage tables to that owner, and validated producer/consumer handoffs and aliases. Its initial consumer applied the GR residual write immediately; D-IMPL-291 corrects that part to preserve the layer-0 pending-write policy.
- **Reason:** An earlier consumer-only design duplicated producer views supplied by its caller and could not guarantee `ExpertStore` plan readiness or lease ownership. Hostile review found both risks. Composing the actual prefix and consumer in one full-mode transaction closes the duplicate-view gap and ensures a failed consumer preflight discards the already encoded prefix. Tiered scheduling needs a different integration point: first establish plan completion, resolve through `ExpertStore`, retain expert leases through GPU completion, then call the lower-level consumer. That production scheduler remains open.
- **Validation:** `metal_gdn_test` checks that consumer handoffs match the actual prefix and rejects consumer address aliasing with producer normalized output, misuse of mixed/router/resident views as shared-gate weights, and router-tensor aliasing through a shared output or sink. `metal_window_storage_test` passes a real `TieredStorage` address table to the full-mode API and checks rejection before dispatch. GDN serial/concurrent plus storage CTests pass 3/3; generic routed/shared MoE serial/concurrent CTests pass 2/2. `test_map.py` and `git diff --check` pass; see REV-086 for hostile findings and fixes.
- **Review required:** Keep tiered callers behind successful plan readiness, `ExpertStore` resolution and retained expert leases. The full-mode owner must remain alive through GPU completion. Production Window scheduling and end-to-end inference remain unimplemented.

### D-IMPL-291: Preserve the layer-0 pending MoE write

- **Decision:** Add `MoeRoutedSharedFFNBuffers` and transactional/caller-owned `encode_moe_routed_shared_ffn` forms that stop after routed/shared combine, leaving its result in `shared.output`. Change `WindowT1MoeLayer0FullBuffers` to use this no-write composition and remove the consumer's residual/injection writeback views. The layer-0 output remains pending for the PLE entry when enabled or the next layer's attention GR read otherwise. Keep `encode_moe_routed_shared_ffn_write` for flows whose GR write is due at that point in the sequence.
- **Reason:** The Window order specifies `pending(1) = !ple_on`: layer 0's MoE result is folded into PLE input or layer 1 attention exactly once. Applying `strata_gr_post` at the end of layer 0 would modify the residual before either consumer and apply the same output twice. The layer-0 coordinator must preserve the pending output and let the later stage own that write.
- **Validation:** The `metal_moe_test`, `metal_gdn_test`, and `metal_window_storage_test` targets build. Focused serial/concurrent GDN/MoE and storage CTests pass 5/5; after adding the pending-write consumer, the updated MoE serial/concurrent pair passes 2/2. The MoE fixture exercises the no-write composition for T=1..8 against the combine oracle, verifies that the original residual and injection remain bitwise unchanged, then applies the pending output through a separate GR-post dispatch and checks the residual against its oracle and write-range guards. The layer-0 consumer preflight rejects a pending output that aliases either its producer residual or injection. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics.
- **Review required:** See REV-087. D-IMPL-294 and D-IMPL-295 exercise the PLE and no-PLE typed consumers separately with the production-shaped FullMode layer-0 producer. Production scheduler branch selection, real PLE row staging through that scheduler and model-backed parity remain open.

### D-IMPL-292: Bind the pending MoE result to the PLE entry

- **Decision:** Add `WindowMoePLEEntryBuffers` and `encode_window_moe_ple_entry`. Take the pending block output from `previous_ffn.shared.output`, its injection from `previous_gr_read.inject_out`, and require the PLE stage to use the exact previous GR residual view. Then encode the existing GR post and PLE block in one ordered segment.
- **Reason:** The generic PLE-entry API accepts unrelated block-output and injection views, so a caller can accidentally lose or double-apply the layer-0 pending MoE result. The typed handoff binds those views to their producer structures and makes the residual identity explicit without claiming that a Window scheduler exists.
- **Validation:** `metal_ple_test` uses the typed handoff for its Window PLE-entry fixture, rejects a mismatched residual before dispatch, and retains its caller-owned late-history rollback check. Serial and concurrent `metal_ple_test` both pass. The fixture's producer views are synthetic; it does not run the full-mode layer-0 MoE producer. Hostile review found no P1 issue and identified the standalone producer-ordering precondition; the first wording was rejected on follow-up because commit order does not propagate producer GPU errors. The API and guide now require caller-registered command-buffer dependency and successful producer completion before standalone submission (REV-089).
- **Review required:** See REV-088, REV-090, REV-092 and REV-093. D-IMPL-295 integrates the FullMode producer with the typed PLE consumer using staged sparse fixture rows and synthetic projection weights; production Window wiring, scratch-liveness confirmation and model-backed parity remain required before claiming L9. The no-PLE producer integration is covered by D-IMPL-294.

### D-IMPL-293: Bind the pending MoE result to the no-PLE attention entry

- **Decision:** Add `WindowMoeAttentionEntryBuffers` and `encode_window_moe_attention_entry`. Take the pending output from `previous_ffn.shared.output`, injection from `previous_gr_read.inject_out`, and require `attention.residual` to reuse the previous GR residual. Bind the pending views to `block_out` and `inject_prev`, then encode the next GR read with `apply=true`.
- **Reason:** The no-PLE branch consumes the same pending write through layer-1 attention instead of PLE. Binding the producer views in a typed coordinator prevents callers from dropping either input or accidentally running the GR read without applying the pending write.
- **Validation:** The GR test uses prefilled synthetic producer views and deliberately different caller-provided `block_out`/`inject_prev` decoys to run both caller-owned and standalone forms, compares every writable GR output with the `apply=true` oracle, verifies the standalone dependency path with a completed fixture producer, checks the declared write extents after both successful runs, and checks mismatched-residual rejection leaves producer and output views unchanged. It does not run the full-mode layer-0 producer or select between the PLE and attention branches.
- **Review required:** See REV-090 through REV-093. The same-segment FullMode-to-no-PLE and FullMode-to-PLE paths are covered by D-IMPL-294 and D-IMPL-295, with their branch selection centralized in D-IMPL-296. Production `Window::run` must call the selector with the token graph's `ple_enabled` decision; scratch liveness and model-backed parity remain required before claiming L9.

### D-IMPL-294: Integrate FullMode layer 0 with the no-PLE attention handoff

- **Decision:** Add `metal_fullmode_handoff_test` using a sparse GGUF fixture with one 256-expert Q2_0 layer, a converted BF16 `[256, 2560]` router, token embedding and PLE tensors. Use full qwen4exp runtime geometry for Window scratch and session state. In one caller-owned `Encoder::encode` segment, run `encode_window_t1_moe_layer0_full_mode` followed by `encode_window_moe_attention_entry`. Populate expert IDs 0–9 with finite Q2_0 weights under equal router logits, make the shared-expert output nonzero, check the combined pending output remains finite/nonzero and unchanged by its consumer, compare all GR writable outputs to a direct `apply=true` read with the original residual and producer injection, and compare the resulting residual to verify the pending write was applied once.
- **Reason:** The no-PLE entry had only synthetic producer-buffer coverage, while the public FullMode coordinator required a 256/512-expert router and live FullMode-owned tables. A sparse file keeps the fixture practical while still exercising GGUF inspection, FullMode storage/address publication, real routing/plan/expert/shared dispatch and the typed producer-consumer bindings on Metal. The test does not require model files and does not establish a production scheduler or end-to-end decode.
- **Validation:** The `metal_window_storage_test` target builds. `metal_fullmode_handoff_test`, `metal_gr_test`, `metal_gr_test_concurrent`, `metal_window_storage_test` and `metal_model_storage_owner_test` pass 5/5. The integration checks a distinct nonzero producer injection against an intentionally different consumer decoy and compares only the active T=1 GR output extents. Hostile findings and fixes are recorded in REV-091.
- **Review required:** See REV-091 through REV-093. `Window::run` must use D-IMPL-296's selector with the token graph's actual PLE decision; model-backed parity remains required before claiming L9.

### D-IMPL-295: Integrate FullMode layer 0 with the PLE handoff

- **Decision:** Add `metal_fullmode_ple_handoff_test` as a second mode of the production-shaped sparse FullMode fixture. Build T=1 row IDs from token 1 and committed history `[5, 0]`, then use `stage_window_ple_input` with FullMode's mapped PLE reader to gather the sixteen selected IQ4_NL rows from the sparse fixture GGUF into `WindowScratchLayout::ple_input`. In one caller-owned `Encoder::encode` segment, run `encode_window_t1_moe_layer0_full_mode` followed by `encode_window_moe_ple_entry`, using synthetic Q2_0 key and BF16 value projection weights. Compare active PLE outputs and residual with a direct GR-post plus PLE-entry run initialized from the original residual. Defer PLE history and verify that the pending FullMode output and committed history remain unchanged. Place the pending output in `WindowScratchLayout::embedding`, separate from the staged PLE input.
- **Reason:** D-IMPL-294 covers only the no-PLE consumer. This step exercises the FullMode producer, mapped-table gathering and typed PLE consumer in the same ordered segment. The embedding region is reused because its input-stem value is dead by the PLE stage; the named PLE input region must hold gathered rows. This is a test-level liveness judgment, and the production scheduler must confirm it. Model projection weights and production scheduler integration remain separate work.
- **Validation:** The `metal_window_storage_test` target builds. `metal_fullmode_ple_handoff_test`, `metal_fullmode_handoff_test`, `metal_window_storage_test`, `metal_model_storage_owner_test`, `metal_gr_test`, `metal_gr_test_concurrent`, `metal_ple_test` and `metal_ple_test_concurrent` pass 8/8. The sparse PLE rows and projection weights are synthetic fixture data, so this does not establish parity against a model artifact. Hostile findings and fixes are recorded in REV-092.
- **Review required:** Confirm the pending-output scratch lifetime in production Window scheduling, call D-IMPL-296 with the branch decision derived from the token graph, use model projection weights, and verify model-backed parity before claiming L9.

### D-IMPL-296: Select one T=1 pending-MoE consumer

- **Decision:** Add `WindowMoePendingConsumer`, a variant that holds exactly one typed PLE or next-attention payload, and `encode_window_moe_pending_consumer`. Require the `ple_enabled` tag to match the active variant and require `rows == 1` before dispatch. Delegate to only the selected typed coordinator in the caller-owned Encoder segment. Route both same-segment FullMode handoff tests through this selector and reject both mismatched branch tags and multi-row requests.
- **Reason:** The Window transition rule documents `pending(1) = !ple_on`: when PLE is enabled, the PLE entry applies the pending layer-0 result; otherwise the next attention read applies it. The separate typed coordinators correctly implement either consumer but leave branch selection to callers. A tagged single-payload API centralizes the rule and avoids accidentally encoding both paths or passing a branch payload inconsistent with the selection.
- **Validation:** The `metal_window_storage_test` target builds. `metal_fullmode_handoff_test`, `metal_fullmode_ple_handoff_test`, `metal_window_storage_test`, `metal_model_storage_owner_test`, `metal_gr_test`, `metal_gr_test_concurrent`, `metal_ple_test` and `metal_ple_test_concurrent` pass 8/8. Each integration checks its selected path, a mismatched tag, and a row count above the T=1 producer extent. The selector remains a coordinator primitive and is not yet called by `Window::run`; hostile findings and resolution are recorded in REV-093.
- **Review required:** See REV-093. Wire the selector into the production Window token graph, retain the selected scratch views through GPU completion, propagate failure to discard the entire producer segment, and complete model-backed parity before claiming L9.

### D-IMPL-297: Compose the FullMode T=1 token-to-consumer step

- **Decision:** Add `WindowT1FullModeStepBuffers` and `encode_window_t1_full_mode_step`. Resolve the native `token_embd.weight` through caller-owned `FullModeStorage`, require shape `[2560, vocabulary_size]`, and map only GGML Q3_K, IQ3_S and IQ4_XS to existing embedding gather formats. The optional BF16 embedding artifact is a separate loader path and is not resolved by this API. Require the broadcast output to be the exact layer-0 attention residual view. In one ordered caller-owned Encoder segment, run token gather, GR stream broadcast, the FullMode layer-0 MoE producer, and the single typed consumer selected by `ple_enabled`. Require callers to propagate false so a later validation error discards earlier encoders in the segment. Check embedding scratch against every live producer and consumer view; permit only exact view reuse as `layer0.ffn.shared.output`, whose write follows the gather's last read.
- **Reason:** The input stem, FullMode layer-0 producer and pending-consumer selector were tested as separate adjacent pieces, leaving callers to sequence them and seed the layer-0 residual correctly. This coordinator binds a loader-compatible read-only native FullMode tensor and the residual handoff while preserving the PLE/no-PLE branch invariant. Reusing scratch at `shared.output` follows the encoded ordering, and explicit overlap checks prevent that storage from corrupting model weights, session state or another live stage. The coordinator still stops before tiered readiness, later layers, QSA, state commit, head, sampler and production scheduling.
- **Validation:** The sparse FullMode fixture maps a Q3_K `[2560, 1]` token embedding row. Both PLE and no-PLE CTests run gather and GR broadcast before the actual layer-0 producer and selected consumer in one segment, starting the active embedding and residual ranges from distinct byte sentinels and comparing against an independently seeded input-stem reference; the unused residual capacity tail retains its original bytes. Regressions reject a mismatched branch, wrong stream view, overlap with a later attention weight and partial overlap with `shared.output`; they also verify a late consumer error discards the earlier segment. The exact `shared.output` reuse is exercised by both successful branch tests. The target builds; the two branch CTests and eight focused handoff, GR, PLE and storage CTests pass; `test_map.py` and `git diff --check` pass with existing planned-work and missing-check diagnostics. Initial hostile findings and repairs are recorded in REV-094; final review found no actionable issues.
- **Review required:** Confirm production scratch liveness, derive `ple_enabled` from the real token graph, retain `FullModeStorage` and selected views through GPU completion, and complete model-backed parity before claiming L9.

### D-IMPL-298: Add a transactional wrapper for the FullMode T=1 step

- **Decision:** Add a `CommandBuffer&` overload of `encode_window_t1_full_mode_step`. Delegate through `Encoder::encode` so the wrapper validates command-buffer state/ownership and discards the whole command buffer if any gather, producer or selected-consumer validation fails.
- **Reason:** The caller-owned Encoder form is useful for larger ordered compositions, but simple one-step callers otherwise must repeat the transaction wrapper and error-propagation pattern. A matching standalone form makes failure atomic at the same boundary as the integrated operation.
- **Validation:** Both FullMode branch tests call the standalone overload for the successful transaction and for a deliberately late consumer validation failure. The failure case verifies the native handle is unavailable after rollback and that CPU-visible residual/pending bytes remain unchanged. The target builds; eight focused handoff, GR, PLE and storage CTests pass; `test_map.py` and `git diff --check` pass with existing planned-work and missing-check diagnostics. Final hostile follow-up found no actionable issue.
- **Review required:** Keep this overload scoped to the integrated step. A consumer-only standalone dispatch that reads a producer from another command buffer still requires a registered dependency and successful producer completion.

### D-IMPL-299: Bind FullMode PLE weights through read-only views

- **Decision:** Extend the T=1 FullMode step's PLE branch to resolve `blk.1.ple_key.weight`, `ple_value.weight`, the three `ple_norm_*.weight` tensors and `ple_conv1d.weight` from its `FullModeStorage` owner. Validate the exact `[2560, 10240]`, `[2560, 2560]`, `[10240]` and `[4, 10240]` shapes and supported Q2_0/Q8_0/BF16 key, BF16 value, F32 gamma and F16 convolution formats; keep the Q8_0 key type aligned across the resolver and model-role allowlists. Add a read-only Q2_0-by-Q8_1 mat-vec entry point; preserve read-only gamma/weight views through PLE post-projection, and bind normalized rows, history, convolution weights and gated rows as read-only inputs to convolution-residual, leaving only the residual writable. For weighted RMS norm, bind overlapping input/gamma as read-only when the output is disjoint, while retaining exact in-place input/output support when gamma is disjoint. Extend the sparse FullMode fixture with deterministic typed tensors and include them in its inspected source inventory.
- **Reason:** D-IMPL-295's PLE handoff used caller-created projection weights, so it did not test how the FullMode step resolves and keeps actual mapped model views immutable. The model-role allowlist also lagged the documented and kernel-supported native Q8_0 PLE key format. The read-only Q2_0 path and read-only PLE post-op bindings remove that synthetic-only integration seam without changing mutable APIs used elsewhere. The sparse fixture proves lookup, shape/type validation, mapping and dispatch contracts, but its contents remain synthetic; it cannot establish parity against a shipped model artifact. Production `Window::run`, tiered scheduling, later layers, commit, head, sampler and real-model parity remain open.
- **Validation:** The `metal_window_storage_test`, `metal_ple_test`, `metal_mv_test`, `metal_gr_test` and `metal_weight_store_test` targets build. Focused serial/concurrent GR, MV and PLE CTests, both FullMode branch CTests, `metal_window_storage_test`, and the host WeightStore regression pass. The FullMode integration clears the caller's PLE tensor views and starts with the wrong key format so success requires resolving typed views from `FullModeStorage`. The Q2_0 read-only oracle covers 65 blocks per row; PLE and mat-vec tests exercise read-only inputs, output-overlap rejection, guards and late-failure transaction behavior. The nine model-header Python tests, test-map validation, full CTest and hostile review results are recorded in REV-096.
- **Review required:** See REV-096. Keep the PLE branch tied to the `FullModeStorage` owner through command completion. Add real-artifact parity and production Window scratch/queue scheduling before claiming L9 or end-to-end inference.

### D-IMPL-300: Verify automatic FullMode PLE key-format selection

- **Decision:** Parameterize the sparse FullMode handoff GGUF fixture to store the layer-1 PLE key tensor natively as Q2_0, Q8_0 or BF16, including correctly sized and encoded payload blocks. Run the same FullMode token-to-PLE segment and direct PLE reference for each format in separate Metal CTests. Supply a deliberately different caller key-format tag to the integrated step and require the mapped model metadata to select the correct dispatch format.
- **Reason:** D-IMPL-299 resolved all three formats in code but only the Q2_0 FullMode fixture proved that the resolver overrides caller data and reaches the matching kernel path. The Q8_0 and BF16 branches are valid production model formats and need the same end-to-end handoff evidence.
- **Validation:** `metal_window_storage_test` builds. The Q2_0, Q8_0 and BF16 FullMode PLE handoff CTests pass 3/3, and the full mac-metal suite passes 47/47. `test_map.py` confirms all 54 Metal test entries and the documentation list agree; existing planned-work and missing-check diagnostics remain. `git diff --check` passes. Hostile findings and their resolution are recorded in REV-097.
- **Review required:** See REV-097. The fixtures contain synthetic model values; retain real-model parity and production Window scheduling as L9 exit requirements.

### D-IMPL-301: Track the committed PLE token suffix

- **Decision:** Add a device-free `WindowPleTokenHistory` that stores only the last two token IDs already applied to model state, in oldest-first order. It can be replaced transactionally from that committed prefix, builds rows through the existing `build_window_ple_rows` validator without changing state, and advances only the accepted prefix of a 1..8-token window. For prompt startup, exclude the final prompt token while it remains pending as the first T=1 decode input; add it only after that window's successful commit. The caller must invoke the advancing method only after the matching GPU state commit succeeds. Keep this host token-ID state separate from the floating-point PLE history in `SessionStateStorage`; D-IMPL-302 attaches its lifetime to that session owner, while `Window::run` still does not consume it.
- **Reason:** `WindowPleRows` needs up to two preceding token IDs, but the engine currently has no session token-history owner. Retaining only the suffix required by the artifact hash keeps speculative rows rollback-safe and avoids duplicating the floating-point PLE state. Mirroring the existing row builder's geometry and vocabulary validation avoids accepting predecessor IDs that row construction cannot consume.
- **Validation:** The focused host CTest passes. The suite covers empty and multi-token applied prompt prefixes, the first pending prompt decode, T=1..8 row generation, accepted-prefix and zero-acceptance behavior, subsequent-window hash parity, reset, invalid IDs/geometry/counts and state preservation on failure. Full host-suite, test-map and hostile-review results are recorded in REV-098.
- **Review required:** See REV-098 and D-IMPL-302. Wire the session-owned suffix into `Window::run`, seed it from the model-applied prompt prefix, and advance it only after successful commit; define how Prefill/GENI image rows advance the normal-token history before wiring those paths. Keep production Window scheduling and real-model parity as L9 requirements.

### D-IMPL-302: Bind PLE token history to the session reset lifecycle

- **Decision:** Embed one `WindowPleTokenHistory` in `SessionStateStorage`, the existing owner for one live sequence's GDN and PLE state. Expose owner methods to initialize once per reset from an applied prefix, read the suffix, build PLE rows and advance an accepted prefix; row building and accepted-prefix updates fail until initialization, and the owned helper is not exposed mutably. Keep token IDs in host memory and do not change the checked Metal allocation layout. Clear the suffix only after `reset()` has validated Context ownership, confirmed that tracked GPU uses are complete, and zeroed the Shared state. A refused reset preserves both host token history and GPU state.
- **Reason:** D-IMPL-301 supplied the correct rollback-safe token suffix but had no session lifecycle owner. `SessionStateStorage` already defines one sequence's state lifetime and reset boundary, so attaching the CPU-only suffix there prevents it from outliving or silently diverging across a successful sequence reset. The owner remains separate from `ModelStorage`; `Window::run` wiring is a later step.
- **Validation:** The `metal_session_state_test` target builds and its latest focused Metal CTest passes. It verifies row/update refusal before initialization, preservation of preexisting output on row-build refusal, read-only row construction, duplicate-seed rejection, empty history on open, reset clears the suffix only after successful reset, pending-use and foreign-Context reset failures preserve the one-time initialization gate, token IDs and nonzero GPU-state sentinels, and the existing GPU allocation byte count remains unchanged. The full build, mac-host suite, and 47-test mac-metal suite passed before the final test-only assertions; the rebuilt latest session-state target then passed. `test_map.py` and `git diff --check` pass with the repository's planned-work diagnostics. Hostile-review results are recorded in REV-099.
- **Review required:** See REV-099. Serialize host suffix mutations with the sequence's Window/commit lifecycle. Wire prompt seeding and successful accepted-prefix updates from the production session/Window path; decide how Prefill/GENI image rows advance normal-token history before enabling those paths. This owner integration does not implement `Window::run` or establish model-backed inference.

### D-IMPL-303: Prepare T=1 Window parameters and session-owned PLE inputs

- **Decision:** Add `prepare_window_t1_input` to build one ordinary T=1 `WindowParams` row in a caller-selected `WindowParamsRing` slot and optionally stage PLE input. With PLE enabled, build row IDs through `SessionStateStorage::build_ple_rows()` from its committed suffix and gather through the `FullModeStorage`-owned table reader into the named Shared PLE scratch region. With PLE disabled, leave the returned PLE view empty and do not require token-history initialization. Use `n_keep = 0`, matching the existing T=1 layer coordinators' ordinary-window precondition; commit remains a separate operation. Return the published parameter and optional PLE views without advancing token history. Validate runtime geometry, token ID, parameter-slot range, ring availability and parameter construction before staging. The caller serializes slot/scratch reuse with GPU work. Do not wire this helper into production `Window::run` yet.
- **Reason:** D-IMPL-302 gave PLE predecessor IDs the correct session lifetime, but the FullMode T=1 segment still required callers to separately build parameters, construct rows and stage the mapped table values. A single preparation boundary makes the session-owned token suffix flow into the already-tested PLE consumer while keeping input preparation separate from asynchronous GPU commit. The one-token pending decode remains outside the applied history until its later commit.
- **Validation:** The `metal_window_storage_test` target builds. `window_params_ring_test` and the no-PLE, Q2_0 PLE, Q8_0 PLE and BF16 PLE handoff CTests pass 5/5. Tests check prepared token/position/seed fields and every non-default sampler field, direct gathered PLE values, history immutability, no-PLE operation with uninitialized history, and rejection of an uninitialized session, an out-of-range token and an overflowing position without changing the parameter slot or PLE scratch. `test_map.py` and `git diff --check` pass with existing planned-work diagnostics. Hostile review is recorded in REV-100.
- **Review required:** See REV-100. The helper depends on callers selecting an idle ring slot and PLE scratch range and keeping FullMode/session owners alive through the later command completion. Wire preparation to the actual T=1 token graph only after the pending-token boundary, sampler settings, and commit ordering are connected. This test uses a sparse synthetic GGUF and does not establish real-model parity, a production Window scheduler, or end-to-end inference.

### D-IMPL-304: Compose the ordered Window GR head mixer

- **Decision:** Add `encode_window_head_mixer` as a caller-owned Encoder composition and a transactional command-buffer wrapper. Encode weighted RMS norm over the T·4 stream rows at width 2560 with gamma period 4, BF16 `output_hc_down` (10240→320), in-place down SiLU, BF16 `output_hc_up` (320→10240), then pre-gated stream reduction. Derive `NC` as the smallest supported width at least T and require the Shared active-row word to equal T. Require scratch outputs to be disjoint from each other, the residual and immutable model inputs; permit only exact residual/normalized in-place aliasing. Refuse an open concurrent group so the Encoder can insert inter-stage barriers. Resolve weights and scratch at the caller; do not claim that this wires the production Window scheduler.
- **Reason:** The documented T=1 path had all five final-mixer kernels but no host composition proving their ordered data handoff. A small buffer-level API advances L9 and makes the documented sequence testable without inventing the still-missing full Window scheduler, weight-owner boundary, vocabulary head or sampler.
- **Validation:** `metal_gr_test` and `metal_gr_test_concurrent` pass. The new test runs T=1..8 with synthetic BF16 one-hot projection weights, compares the complete composition against an independent scalar RMS/down-SiLU/up/pre-gated oracle at U(256), checks residual and model-weight immutability, exercises the caller-owned Encoder form, and rejects active-row mismatch, writable scratch overlap and a short up projection after earlier stages have encoded; the last case proves the command buffer is discarded. The target builds with the `mac-dev` preset. `test_map.py` and `git diff --check` pass with the repository's existing planned-work diagnostics; hostile review is recorded in REV-101.
- **Review required:** See REV-101. The helper takes raw validated views, so its caller must resolve `output_hc_*` from the live FullMode owner, select scratch ranges unused by all concurrent GPU work, and keep all owners alive through completion. The synthetic fixture does not establish real-model parity. Production `Window::run`, the final vocabulary projection, sampling and accepted-prefix commit remain open.
