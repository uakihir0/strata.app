# 02. Architecture: how the Mac port fits together

Status: implementation-ready specification, 2026-10-07.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Sources: documents 03-09 and [appendix-a-measurements.md](appendix-a-measurements.md). This document is the map of
the set. Apart from the dependency rules of section 3.3, which it owns, every statement here summarises a detailed
document and names it. Where this document and a detailed one disagree, the detailed one wins, and the disagreement
is a documentation bug to fix in both.

This document covers:
- the processes of a Mac installation and the channels between them;
- the engine's layers, components, threads and dependency rules;
- the main flows: startup, one request, one decode round, the prompt path, the tiered-mode miss path and the
  failure paths;
- the memory budget and the two residency modes;
- the project's source tree;
- the decisions that shape the design, with their reasons;
- the invariants every change must keep, and the tests that check them;
- how the project follows upstream;
- a glossary of the terms the set uses.

Other documents cover the rest:
- [00-feasibility-study.md](00-feasibility-study.md): whether to port, the scope and the gates.
- [01-requirements.md](01-requirements.md): what the port must achieve, with requirement IDs.
- [03-model-reference.md](03-model-reference.md): the model, its files and its forward pass.
- [04-metal-kernels.md](04-metal-kernels.md): the kernels, their numerics and their encoding.
- [05-memory-and-residency.md](05-memory-and-residency.md): where every byte lives.
- [06-engine-host-port.md](06-engine-host-port.md): the engine process apart from the kernels.
- [07-server-setup-port.md](07-server-setup-port.md): the Python layer: setup, server, telemetry and MCP.
- [08-build-and-distribution.md](08-build-and-distribution.md): the build, the release and CI.
- [09-testing-and-validation.md](09-testing-and-validation.md): the tests, the criteria and the gates.
- [10-implementation-plan.md](10-implementation-plan.md): the phases, the tasks and upstream compatibility reviews.
- [11-risks-and-open-questions.md](11-risks-and-open-questions.md): the risk register and the open items.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. The design in brief

Upstream Strata is built for a PC with a small discrete GPU:
- the dense weights and a cache of hot experts live in VRAM;
- every expert also lives in host RAM, and the CPU computes the experts the cache misses, because PCIe is slower than
  host RAM;
- a captured CUDA graph decodes a token, and a doorbell lets the GPU spin on a flag until the CPU's experts are done.

A Mac has one memory, which the CPU and the GPU read under one shared limit of about 284 GB/s
([appendix A.2](appendix-a-measurements.md)). Five ideas follow from that, and the rest of the design follows from
them:

1. **Every expert runs on the GPU.** On unified memory a CPU expert path adds no bandwidth, only synchronisation,
   and it would make the output depend on which experts are resident (decision D1, section 7).
2. **One decode path.** Plain decode is a verify window of one token. A window of T = 1..8 tokens is one command
   buffer, encoded afresh each time, so speculation and plain decode are the same code and give the same bits (D2,
   D3).
3. **The engine plans its own memory.** At startup the engine fits everything it needs into the GPU working-set
   budget W.
   - **Full mode:** every expert fits, and the model files are wrapped in place.
   - **Tiered mode:** a hot set of experts is resident, and the GPU waits once per layer while the host reads the
     missing experts from the SSD into a slot pool.
   - Both modes run the same kernels on the same bytes (D4, D5).
4. **No GPU thread waits on memory the CPU writes.** The only host-to-GPU handshake inside a command buffer is an
   `MTLSharedEvent` wait, and only tiered mode uses it (D6).
5. **Everything above the engine stays.** The serve protocol is byte-compatible. The Python server, the web app, the
   APIs and the MCP tools run unchanged apart from marked macOS branches. Setup asks the engine, not a Python
   formula, whether a model fits (D4, D8).

The port is therefore a new GPU backend and a simpler memory manager inside a separately maintained Strata project based on the documented v0.1.34 source baseline, not a translation of the
CUDA code. The project keeps upstream edits small and marked so material compatibility changes can be reviewed and selected fixes can be carried forward under the procedure in section 9.

---

## 2. System context

### 2.1 Processes

```
  browser, API client, MCP client
        |
        | HTTP: web app, OpenAI and Anthropic APIs, MCP hub
        v
  serve/server.py  ----- ENC <image> <out> / OK ----->  strata-vision   (optional; writes the embeddings file)
        |      ^
  stdin |      | stdout
  GEN   |      | INFO, READY, RESUME, PP, REUSED, T, DONE, ERR
  GENI  |      | (stderr: diagnostics, into the server's log)
  STOP  |      |
  QUIT  v      |
  strata --serve   (the Metal engine; one per Mac)
        |
        | mmap; pread (PLE rows; tiered-mode dense tensors and experts)
        v
  data folder: GGUF shards, native pack, MTP runtime, expert profile

  setup.py  ---- strata --plan-only ... ---->  strata   (one PLAN line, exit 0 or 3)
```

| Process | What it is | Started by | Specified in |
| --- | --- | --- | --- |
| `setup.py` | The installer and launcher: finds Python, builds the venv, installs or compiles the engine, downloads the model, builds the pack, writes the run config, starts the server | `./setup.sh`, `START-HERE.command`, `SETUP.command` | [07](07-server-setup-port.md), sections 4 and 5 |
| `serve/server.py` | The HTTP server: web app, OpenAI and Anthropic APIs, MCP hub, Monitor tab. It owns the engine process and restarts it after a failure. | `start()` in setup, or a `run-<model>` script | [07](07-server-setup-port.md), sections 6 and 7 |
| `strata --serve` | The engine | The server, as `Popen([exe, "--serve", *args])` | [06](06-engine-host-port.md) |
| `strata-vision` | The image encoder: llama.cpp's mtmd at the pinned commit, built with Metal | The server, when vision is on | [03](03-model-reference.md), section 6.11; [06](06-engine-host-port.md), section 5.10; [08](08-build-and-distribution.md), section 9 |
| `tools/strata_mcp.py` | The MCP server for AI agents: recommend, install, start, stop | An MCP client | [07](07-server-setup-port.md), section 8 |
| `strata` (one-shot) | The engine without `--serve`: one prompt, then exit. Used by benchmarks, tests and `accept.py`. | A person or a test | [06](06-engine-host-port.md), sections 3 and 13.1; [09](09-testing-and-validation.md), section 3 |

### 2.2 Channels

| Channel | Format | Compatibility rule | Specified in |
| --- | --- | --- | --- |
| Server to engine | stdin lines: `GEN`, `GENI`, `STOP`, `QUIT` | Byte-compatible with upstream. `GENI` also accepts an embeddings path with spaces. | [06](06-engine-host-port.md), section 5.3 |
| Engine to server | stdout lines: `INFO`, `READY`, `RESUME`, `PP`, `REUSED`, `T`, `DONE`, `ERR`; diagnostics on stderr | Byte-compatible. `INFO` gains keys after `engine=`; DONE keeps its fields with Mac meanings. | [06](06-engine-host-port.md), sections 5.2 and 5.5-5.8 |
| Server and `strata-vision` | `READY`, `ENC`, `OK`, `ERR`, `QUIT` lines, and the embeddings file that `GENI` names | Unchanged | [06](06-engine-host-port.md), sections 5.4 and 5.10 |
| Setup to engine | `strata --plan-only ...`: one `PLAN` line on stdout, exit 0 (fits) or 3 (does not fit) | New, Mac only | [05](05-memory-and-residency.md), section 3.4; [07](07-server-setup-port.md), section 4.4 |
| Setup to engine | `strata --version`, `strata --list-devices` | New: `--version` prints `strata <x.y.z> metal <rev>` and keeps the `engine=` literal that setup scans for; `--list-devices` replaces `strata-device` | [06](06-engine-host-port.md), sections 2.1 and 3.10 |
| Engine to files | `mmap` of the GGUF shards and the MTP runtime; `pread` of PLE rows (or a copy from a plain mapping with `--ple-io mmap`), of the `conversions.json` rows of `dense.bin` and, in tiered mode, of the dense tensors and the experts | The upstream formats. A Mac pack is a native pack without `extra.bin` or `experts.bin`. | [03](03-model-reference.md), sections 2-4; [05](05-memory-and-residency.md), sections 4-6; [06](06-engine-host-port.md), section 6 |
| Clients to server | HTTP | Unchanged | upstream |

### 2.3 One engine per Mac

The engine plans against all of W and cannot see another engine. Two servers with two models on one Mac push each
other into compression and swap, and this is not supported ([07](07-server-setup-port.md), section 6.8). An expert
user can split W by hand with `--gpu-budget-mib`. The server's own unload and reload (`idle_unload_s`,
`before_load`) are the supported way to share the Mac with another GPU app.

### 2.4 What is installed where

| Path | Contents | Specified in |
| --- | --- | --- |
| The Strata folder (a checkout of the project) | The Python layer, `.venv`, run scripts and `.command` launchers, and `engine/`: `strata`, `strata.metallib`, `strata-vision`, `BUILD.json`, `LICENSES.txt` | [07](07-server-setup-port.md), sections 4.7 and 4.10; [08](08-build-and-distribution.md), section 11.2 |
| The data folder: `Strata-data` beside the Strata folder, or `~/Strata-data` when that would be in a protected or synced folder | GGUF shards, the native pack, the MTP runtime, expert profiles | [07](07-server-setup-port.md), section 4.12 |
| `~/.config/strata` | Setup's settings | [07](07-server-setup-port.md), section 4.12 |
| `~/Library/Caches/strata/` | The pipeline archive | [08](08-build-and-distribution.md), section 10.1 |
| `~/Library/Logs/DiagnosticReports/` | The crash report after a watchdog abort | [06](06-engine-host-port.md), section 14.3 |

---

## 3. Inside the engine

### 3.1 Layers

```
  src/metal/program     the driver
                        main, options, serve, oneshot, stats, watchdog, version
                        (copies of upstream policy: Options, the parser, the serve loop, the prompt-cache policy,
                         resume selection, the spec round, the DONE line)
        |  C++ calls
        v
  src/metal/engine      the model on Metal
                        Geometry, Plan, WeightStore, ExpertStore, KvStore, Window, Mtp, Prefill, Snapshot
        |  C++ calls through plain C++ headers
        v
  src/metal/host        Objective-C++ over Metal
                        Context, Buffers, Events, Encoder, PipelineCache, Capture
        |  pipelines and dispatches
        v
  src/metal/kernels     strata.metallib (MSL 4.0): common/, safe/, fast/

  Upstream code the engine links, unchanged or with marked patches:
    src/spec/* (suffix drafter, draft policy, controller), conv_cache.hpp, conversation_cache.hpp,
    conversation_buffer.hpp, progress.hpp, coupled_draft.hpp, sampler.hpp, logits_selection.hpp, gguf_reader.hpp,
    gguf_split.hpp, ple_reader.cpp, ngram.cpp, direct_file.cpp, memory.cpp
```

The upstream list and the status of each file on macOS are in [06](06-engine-host-port.md), section 1.4. Everything
else under `src/core`, `src/prefill`, `src/kernels/cuda` and `src/program/generate.cpp` is not compiled on macOS.

### 3.2 Components

The namespace is `strata::metal`, and the headers are in `include/strata/metal/`. The full table, with the upstream
code each component replaces, is [06](06-engine-host-port.md), section 1.5.

| Component | Layer | Responsibility | Specified in |
| --- | --- | --- | --- |
| `Context` | host | Device checks; the command queue and its fence; loading `strata.metallib` and checking its hash; building every pipeline at startup; the residency set; labelled command buffers | [06](06-engine-host-port.md), sections 2.3 and 8 |
| `Buffers` | host | Shared buffers from a bump arena with fixed addresses; no-copy file windows (16 KiB-aligned, each at most `maxBufferLength`) | [05](05-memory-and-residency.md), sections 4.2 and 7 |
| `Events` | host | The `MTLSharedEvent` values of the tiered-mode handshake; completion waits | [06](06-engine-host-port.md), section 9.3 |
| `Encoder` | host | The compute-encoder wrapper: dispatches, dispatch groups, the fence on every encoder, labels | [04](04-metal-kernels.md), sections 8.1 and 8.2 |
| `PipelineCache` | host | The `MTLBinaryArchive` in `~/Library/Caches/strata/` | [08](08-build-and-distribution.md), section 10.1 |
| `Capture` | host | `--metal-capture`: a GPU trace of the first decode window | [08](08-build-and-distribution.md), section 10.2 |
| `Geometry` | engine | The facts the plan needs about a model (shard sizes, tensor tables, per-layer blob sizes, MTP sizes), built from the files or read from a geometry file (`--geometry`) | [06](06-engine-host-port.md), section 1.5; [07](07-server-setup-port.md), section 4.4.3 |
| `Plan` | engine | The memory plan: full or tiered mode, the resident set, the slot pool, the prefill chunk, the window planning, the PLAN line. Plain C++ with no Metal calls, so it is tested without a GPU. | [05](05-memory-and-residency.md), sections 3 and 4.2 |
| `WeightStore` | engine | Opens the GGUF shards and the native pack, applies the format rules, maps each tensor to (buffer, offset) | [03](03-model-reference.md), sections 3-4; [06](06-engine-host-port.md), section 6 |
| `ExpertStore` | engine | The residency mode, the resident set, the address and resident tables, the slot pool, `MissResolver`, adaptive swaps, lookahead | [05](05-memory-and-residency.md), sections 3-5; [06](06-engine-host-port.md), section 9 |
| `KvStore` | engine | QSA K/V in fp16, int8, q4_0 or k8v4, and the indexer keys | [04](04-metal-kernels.md), sections 4.5 and 4.6 |
| `Window` | engine | Encodes a window of T = 1..8 rows over 48 layers, the head and the sampler; `commit(n_keep)` | [04](04-metal-kernels.md), sections 5 and 8; [06](06-engine-host-port.md), section 10.1 |
| `Mtp` | engine | The MTP drafter, with upstream's API | [04](04-metal-kernels.md), section 5.11; [06](06-engine-host-port.md), section 10.3 |
| `Prefill` | engine | The batched prompt path: chunks of up to C rows | [04](04-metal-kernels.md), section 6 |
| `Snapshot` | engine | Checkpoints, restore and conversation parking | [06](06-engine-host-port.md), section 11 |
| Driver | program | The `strata` executable: options, serve and one-shot loops, stats, watchdog, version | [06](06-engine-host-port.md), sections 2-5, 13 and 14 |

### 3.3 Dependency rules

These rules belong to this document. A change that breaks one needs a change here first.

1. **Calls go down.** The driver calls the engine, the engine calls the host layer, and the host layer dispatches
   kernels. No layer calls a layer above it. Metal completion handlers and event listeners only set flags and signal
   condition variables ([06](06-engine-host-port.md), section 2.2).
2. **Objective-C++ stays in `src/metal/host/*.mm`.** Every header in `include/strata/metal/` is plain C++, and Metal
   objects cross it as opaque handles, as the skeleton's `context.hpp` does
   ([spikes/metal-build/](spikes/metal-build/)). The driver, the engine and the reused upstream code then compile
   without Metal headers.
3. **The driver reaches the model only through the engine's headers.** It never touches a Metal object. That lets a
   `host` test link the driver against a fake: `serve_transcript_test` links `tests/metal/fake_window.cpp` in place of
   the real `Window` ([09](09-testing-and-validation.md), section 9.1).
4. **One declaration per parameter block.** The kernels and the host include the same
   `src/metal/kernels/common/types.h`, with `static_assert`s on the size and every offset, on both sides
   ([04](04-metal-kernels.md), section 4.2). Offsets that both sides read, such as those of `MoePlan` and `PfPlan`,
   come from that file.
5. **Upstream code changes only at marked sites.** A reused upstream file is edited only in a branch marked
   `// MAC:` (C and C++) or `# MAC:` (CMake and Python). Copied code carries its `synced from` header (section 9).

### 3.4 Threads

| Thread | Role | Specified in |
| --- | --- | --- |
| Main | The serve or one-shot loop: requests, prompt cache, prefill, decode rounds; encodes and commits every command buffer; applies resident-table updates between windows | [06](06-engine-host-port.md), section 2.2 |
| stdin reader | `read(2)` on fd 0; `STOP` takes effect at once, other lines are queued, EOF sets `in_eof` | [06](06-engine-host-port.md), sections 2.2 and 5.1 |
| Watchdog | One check per second; stall report and abort | [06](06-engine-host-port.md), section 14 |
| PLE I/O workers | `pread` of PLE rows (`STRATA_IO_THREADS`) | [06](06-engine-host-port.md), section 2.2 |
| Fetch workers (tiered only) | Expert reads (`STRATA_FETCH_THREADS`, default 8), so that 8 or more reads are in flight | [06](06-engine-host-port.md), section 2.2; [05](05-memory-and-residency.md), section 5.4 |
| `MissResolver` (tiered only) | The per-layer handshake: reads the plan, fills slots, writes their addresses, signals | [06](06-engine-host-port.md), section 9.3 |
| Adapt (tiered only) | Reads incoming experts for adaptive swaps | [06](06-engine-host-port.md), section 9.4 |
| Metal completion handlers | Set flags only | [06](06-engine-host-port.md), section 2.2 |

Only the main thread encodes or commits command buffers. The `MissResolver` and its fetch workers write only the
slots of half `l % 2` after layer l's signal (and of half `(l + 1) % 2` for the prefetch), and `addr[l]` before the
resolver signals ([06](06-engine-host-port.md), section 9.3). The adapt thread writes only resident-arena places that
no window reads any more, and decays the usage counters between windows ([05](05-memory-and-residency.md),
section 5.6).

---

## 4. Flows

### 4.1 Startup

The full sequence, with each step's failure and exit code, is [06](06-engine-host-port.md), section 2.3.

| Steps | What happens |
| --- | --- |
| 1-3 | Unbuffered stdout, signal handling, argument parsing; `--version` and `--list-devices` exit here |
| 4 | `Context::open`: the device and its families, `strata.metallib` and its hash, the manifest, every pipeline (from the archive when it is valid) |
| 5-7 | Memory facts (W, RAM, available memory); the GGUF shards and the pack, with the format checks; the MTP runtime |
| 8 | The memory plan ([05](05-memory-and-residency.md), section 3): full or tiered mode, the resident set, the slot pool, the prefill chunk. Printed on stderr. |
| 9 | Buffers: file windows (full mode) or copied arenas (tiered mode); the residency set; the warm-up pass |
| 10-11 | The PLE table, control vectors, the expert profile; `session_zero` |
| 12 | Serve: the memory line, `INFO`, the watchdog, `READY <max_context> stop`. One-shot: run. |

Two light paths stop early and need no Metal pipelines:
- `--plan-only` runs steps 1-8 on headers and tensor tables only (or on a geometry file), prints one `PLAN` line on
  stdout and exits 0 or 3. Setup calls it for every menu line ([07](07-server-setup-port.md), section 4.4).
- `--write-geometry FILE` reads the same facts and writes them as a geometry file, so that setup can plan before a
  model is downloaded.

### 4.2 One request

1. The stdin reader queues a `GEN` or `GENI` line. The main thread parses it under a `BusyScope`, which the watchdog
   watches ([06](06-engine-host-port.md), sections 5.3 and 14).
2. **Resume.** The longest matching prefix wins among the live session, the checkpoints and the parked conversations.
   It is restored, and the engine prints `RESUME <n>` ([06](06-engine-host-port.md), section 11.2).
3. **The rest of the prompt.** Text parts of at most `--short-read` (64) tokens with no image row go through
   write-through windows, unless `STRATA_CKPT_REREAD` is set. Every other part goes through `Prefill` in chunks;
   without `--prefill`, the whole prompt goes through windows. Checkpoints are taken at the cut points
   ([06](06-engine-host-port.md), sections 11.1 and 11.2).
4. **The prompt's last token** is the first decode window, with T = 1. Its residual seeds the drafter
   (`draft_first`).
5. **Decode rounds** (section 4.3) run until `max_new`, an EOS, `STOP` or stdin EOF. Each accepted token is one `T`
   line.
6. **The finish:** one `DONE` line with the finish reason and the counters, then a summary on stderr
   ([06](06-engine-host-port.md), sections 5.5-5.8).

### 4.3 One decode round

```
  host, main thread                                  GPU (one queue, one fence)
  -----------------                                  --------------------------
  choose T and the drafts (policy, suffix, MTP)
  pread the T rows' PLE inputs (PLE workers)
  write WindowParams and ple_in
  commit the window ------------------------------>  window: embedding, 48 layers, head, sampler
  wait                                               (tiered: one handshake per layer, section 4.5)
  accept: the longest prefix where the drafts  <---  sampled tokens of the T rows
  equal the samples, a + 1 tokens
  [tiered, every --adapt-every rounds: apply swaps]
  print T lines for the accepted tokens
  commit(a + 1) and the MTP round ---------------->  commit: GDN replay, indexer, PLE history; first draft
  wait; read the first draft
  each further draft ----------------------------->  one command buffer per draft step
```

- The steps, with upstream's line ranges, are [06](06-engine-host-port.md), section 10.2. The window's kernels, in
  order, are [04](04-metal-kernels.md), section 5.1. The command buffers are [04](04-metal-kernels.md), section 8.3.
- Plain decode (`--spec 0`) is the same round with T = 1 and no drafts. Its windows are write-through: the commit is
  folded into the window ([04](04-metal-kernels.md), section 8.5).
- A round costs two host round trips, plus one per further draft. Each is about 11 µs against a window of about
  25 ms ([appendix A.5](appendix-a-measurements.md)).

### 4.4 The prompt path

`Prefill::run(tokens, n, pos0)` reads a prompt part in chunks of up to C rows, where C comes from `--prefill` and the
plan ([05](05-memory-and-residency.md), sections 2.4 and 3.2).

- Each chunk is one command buffer. The host decodes the next chunk's PLE rows into the other half of a double
  buffer while the GPU runs the current chunk.
- Prefill runs no output norm and no head. The prompt's last token is never prefilled.
- The prompt path is deterministic, and with the `simd` GEMMs its result does not depend on C
  (`prefill_chunk_independent`). It has no bitwise contract with decode; its tolerance is in
  [04](04-metal-kernels.md), section 10.3.
- The MTP drafter fills its own K/V from the chunk's residual rows (`Mtp::prefill`).

Details: [04](04-metal-kernels.md), section 6.

### 4.5 The tiered-mode miss path

In tiered mode every layer of a window, and every layer of a prefill chunk, makes one handshake with the
`MissResolver`:

```
  GPU, layer l                         MissResolver                         fetch workers
  ------------                         ------------                         -------------
  router, route, plan -> plan[l]
  signal base + 2l + 1  ------------>  read plan[l]
  shared expert                        look up the slot cache of half l % 2
  wait base + 2l + 2                   queue 3 preads per missing expert -> read from the GGUF
                                       wait for the reads              <-   (page cache or SSD)
                                       write addr[l], release fence
                        <------------  signal base + 2l + 2
  routed experts (addresses from       prefetch layer l + 1 into half (l + 1) % 2
  addr[l]), combine, rest of layer
```

- The slot pool has two halves of H slots, where H is the largest number of non-resident experts in any layer.
  Layer l uses half `l % 2`. When layer l's signal fires, every earlier reader of that half has completed, so the
  resolver may overwrite any of its slots. Only one command buffer that reads slots is in flight
  ([06](06-engine-host-port.md), section 9.3; [05](05-memory-and-residency.md), section 5.4).
- A layer with no misses still signals, with `skip = 1` and no reads.
- Decode prefetches the experts that the lookahead predicts. Prefill prefetches every non-resident expert of the next
  layer ([05](05-memory-and-residency.md), sections 5.5 and 5.7).
- Adaptive swaps run between windows: a victim leaves the resident table first, the adapt thread reads the incoming
  expert into its place, and the main thread publishes it before the next window
  ([06](06-engine-host-port.md), section 9.4).
- A layer signal that does not arrive within 20 s ends the request with `ERR gpu error: window timeout`.

Gate G2 decides whether this path is fast enough on a 48 GB Mac ([09](09-testing-and-validation.md), section 6.2).

### 4.6 Failure paths

| Event | What the engine does | What the server does | Specified in |
| --- | --- | --- | --- |
| Bad arguments | Exit 2, with the message (and the usage for an unknown argument) | Shows the log | [06](06-engine-host-port.md), section 2.3 |
| Any other startup failure (missing file, bad format, wrong metallib) | Exit 1, with a `strata: ` message | Explains the death from the log | [06](06-engine-host-port.md), section 2.3; [07](07-server-setup-port.md), section 6.5 |
| The model does not fit | Exit 1: `the model does not fit: ...`. With `--plan-only`: `PLAN fits=0`, exit 3. | Setup offers smaller choices | [05](05-memory-and-residency.md), section 3.2 |
| A buffer cannot be created | Exit 1, with the plan and the buffer's name | As above | [06](06-engine-host-port.md), section 2.3, step 9; [09](09-testing-and-validation.md), section 9.1 |
| A command buffer fails | `strata: GPU error in "<label>": ...` on stderr; `ERR gpu error: <description>`; exit 1 | Starts the engine again on the next request | [06](06-engine-host-port.md), section 2.4 |
| A window wait times out (20 s) | `ERR gpu error: window timeout`; exit 1 | As above | [06](06-engine-host-port.md), section 14.4 |
| No progress for `STRATA_WATCHDOG_S` | The issue #29 line, the stall report, the event release, `abort()`; macOS writes a crash report | Its death note finds "issue #29" and restarts | [06](06-engine-host-port.md), section 14 |
| Critical memory pressure | Drops the parked conversations and every checkpoint but the root at the next request boundary | Nothing | [06](06-engine-host-port.md), section 11.4 |
| stdin EOF, or the parent dies | Ends the request at the next round; exit 0 | | [06](06-engine-host-port.md), section 2.4 |
| SIGINT or SIGHUP in serve mode | Ignored; the server sends `QUIT` | Catches the signal and quits the engine cleanly | [06](06-engine-host-port.md), section 2.4; [07](07-server-setup-port.md), section 6.2 |

---

## 5. Memory

### 5.1 One budget

- **W** is the GPU working-set budget: `recommendedMaxWorkingSetSize`, or `--gpu-budget-mib` when that is smaller.
  It is 51.84 GiB, 81% of RAM, on the 64 GB reference Mac ([appendix A.1](appendix-a-measurements.md)). On other
  sizes it is an estimate until measured (M-8).
- **B** = W − reserve, with `--vram-reserve-mib` 700 by default. The sum of every Metal buffer must stay within B.
- The plan adds the always-resident set A (dense weights, the MTP runtime, KV, fixed state), the prompt-path buffers
  P, and either every expert (full mode) or a resident set plus the slot pool S (tiered mode). The algorithm is
  [05](05-memory-and-residency.md), section 3.2.
- Host RAM outside the budget holds the conversation checkpoints, the parked conversations and the page cache. The
  page cache is the RAM tier of tiered mode ([05](05-memory-and-residency.md), sections 2.5 and 5.2).

### 5.2 The two modes

| | Full mode | Tiered mode |
| --- | --- | --- |
| When | Every expert fits beside A and the smallest prompt path, and no flag caps the resident set | Otherwise |
| GGUF shards | `mmap`ped and wrapped as no-copy file windows in the residency set | Never wrapped. The dense tensors and the resident experts are copied into anonymous arenas with `pread`. |
| Cold experts | None | Read per layer into the slot pool: two halves of H slots |
| Host work per window | Write the inputs, commit, wait | One handshake per layer; swaps every `--adapt-every` rounds |
| Typical cases (Q2_0 unless named) | 64 GB and up with Q2_0, IQ2_XS and IQ3_XXS, Swift included; Coder from 48 GB | 32-48 GB Macs; IQ3_S on 64 GB; any run with `--expert-cache N` or a small `--gpu-budget-mib` |
| Example | 64 GB, 262,144 cells fp16, chunk 8192: 45.18 of 51.16 GiB | 48 GB, 32,768 cells fp16, chunk 2048: 96% of experts resident, about 1,000 on the SSD |
| DONE counters | Every lookup is a hit | `hits`, `ram_blobs`, `file_blobs`, `file_MB` |

The examples are from [05](05-memory-and-residency.md), section 8, which has a plan for every Mac class. The output
does not depend on the mode (invariant I3, section 8).

### 5.3 What is never wrapped

macOS has no GPU demand paging. A kernel that touched 1 MiB of a cold 8 GiB no-copy buffer made all 8 GiB resident
([appendix A.3](appendix-a-measurements.md)). So these are never wrapped as Metal buffers
([05](05-memory-and-residency.md), section 6):
1. the PLE table, whose rows the host reads with `pread` (or copies from a plain mapping, `--ple-io mmap`) and writes
   into the window's PLE staging buffer;
2. any GGUF shard in tiered mode;
3. any byte range that the plan does not count.

`metal_never_wrap_test` checks the PLE exclusion in full mode and the full-mode byte accounting. The tiered-mode
GGUF exclusion remains planned with U5/U6, because the tiered loader is not implemented yet.

---

## 6. Source tree

The project's files that this port adds or edits. Every path is relative to the project's root.

```
METAL_REV, UPSTREAM_SYNC, CMakePresets.json     the Mac release number, the upstream commit, the presets
CMakeLists.txt                                  upstream; edits A-G, each line marked # MAC:
cmake/metal/                                    backend.cmake, metallib.cmake, metal.sh, check_manifest.cmake,
                                                tests.cmake, check_engine.cmake
include/strata/metal/                           plain C++ headers of the Mac engine
src/metal/host/                                 context.mm, buffers.mm, events.mm, encoder.mm, pipeline_cache.mm,
                                                capture.mm
src/metal/engine/                               weights.cpp, geometry.cpp, plan.cpp, experts.cpp, expert_profile.cpp,
                                                kv.cpp, window.cpp, sampler_tail.cpp, mtp.cpp, prefill.cpp, snapshot.cpp
src/metal/program/                              main.cpp, options.cpp, serve.cpp, oneshot.cpp, stats.cpp,
                                                watchdog.cpp, version.cpp
src/metal/kernels/                              common/, safe/, fast/, manifest.txt, THIRD_PARTY
src/platform/direct_file.cpp and the other      upstream; patched in // MAC: branches
  files of [06] section 7
tests/metal/                                    the Mac tests, fixtures, prompt sets, golden manifest,
                                                test_map.json, requirements.json, reviews.json
tools/metal/                                    accept.py and accept/, tf_compare.py, ladder_compare.py,
                                                make_prompts.py, record_golden.py, check_drift.py, drift.json,
                                                make_release.py, check_asset.py, verify_release.py, test_map.py, ...
serve/macos.py                                  every macOS probe of the Python layer
setup.py, setup.sh, serve/*.py, tools/*.py      upstream; Mac branches marked # MAC:
tools/make_geometry.py, data/geometry/          plans before a download
START-HERE.command, SETUP.command               double-click launchers
tools/vision/strata_vision.cpp                  upstream; the thread-count fix
.github/workflows/                              ci.yml, mac-hardware.yml, mac-release.yml
docs/mac/                                       this documentation set, copied in phase 0
```

The complete lists are:
- [08](08-build-and-distribution.md), sections 1.2 and 1.3: the build and release files, and the upstream files the
  build edits;
- [07](07-server-setup-port.md), section 1.3: the Python layer;
- [09](09-testing-and-validation.md), section 1.3: the test files;
- [06](06-engine-host-port.md), sections 1.5 and 7: the engine files and the platform patches.

The build's targets are `strata_metal_options` (the shared compile and link options), `strata_metal_platform`,
`strata_metal_host`, `strata_metal_engine`, `strata` and `strata_metallib`. The C++ targets' source lists are explicit,
never globbed. `strata_metallib` globs the `.metal` files in sorted order, and `check_manifest.cmake` checks them
against `manifest.txt` ([08](08-build-and-distribution.md), sections 4.4, 5.2 and 5.3).

---

## 7. Decisions

### 7.1 The decisions that shape the design

| ID | Decision | Reason | Rejected alternative | Specified in |
| --- | --- | --- | --- | --- |
| D1 | **Every routed expert runs on the GPU.** No CPU expert path in the first release. | CPU and GPU share one DRAM limit of about 284 GB/s, so CPU reads take bandwidth from the GPU. A CPU path adds synchronisation, and it makes the output depend on residency, which would make invariant I3 impossible. | Upstream's CPU pool. An optional CPU helper stays possible later. | [06](06-engine-host-port.md), sections 1.2 and 12; [appendix A.2](appendix-a-measurements.md) |
| D2 | **One decode path:** the verify window, T = 1..8, with T = 1 for plain decode | Speculation equals plain decode by construction, because it is the same code | A separate token graph and verify graphs, as upstream | [06](06-engine-host-port.md), sections 1.2 and 10.5; [04](04-metal-kernels.md), section 5 |
| D3 | **Re-encode one command buffer per window.** An indirect command buffer only if H-4 measures an encode time above 1 ms. | 0.78 µs per dispatch; about 0.15 ms of encoding for a T = 8 window against about 25 ms of GPU time | Recorded command sequences (the CUDA-graph analogue) | [04](04-metal-kernels.md), section 8.3; [appendix A.5](appendix-a-measurements.md) |
| D4 | **The engine plans its memory; setup reads the PLAN line.** | W is known only on the device. One fit logic, in one place, cannot drift from what the engine does. | Python fit formulas, as upstream | [05](05-memory-and-residency.md), section 3; [07](07-server-setup-port.md), section 1.2, P3 |
| D5 | **Two residency modes with the same kernels.** Tiered mode copies what is resident and reads cold experts into a slot pool, one layer at a time. | There is no GPU demand paging, so a wrapped file becomes resident as a whole. Fixed slot pools avoid sparse-resource stalls. | GPU page faults on a mapped file; sparse heaps | [05](05-memory-and-residency.md), sections 1.1, 1.3 and 5; [00](00-feasibility-study.md), section 6 |
| D6 | **No GPU thread waits on CPU-written memory.** The only in-command-buffer handshake is `encodeWaitForEvent`. | Metal gives no forward-progress guarantee for spinning, MSL device atomics are relaxed only, and a long command buffer can be killed by the GPU watchdog | Upstream's doorbell and flag spins | [06](06-engine-host-port.md), sections 1.2 and 8.2, rule 1 |
| D7 | **One queue, one fence, untracked buffers, fixed addresses, per-step values in buffers** | Explicit ordering costs little and is testable. Without the fence a second command buffer read stale data in 20 of 20 runs. Fixed addresses suit residency sets and a later ICB. | Metal's hazard tracking; per-token allocation | [04](04-metal-kernels.md), section 8.1; [06](06-engine-host-port.md), section 8.2; [appendix A.11](appendix-a-measurements.md) |
| D8 | **The serve protocol is byte-compatible.** New `INFO` keys are additive. | `serve/server.py` and everything above it run unchanged | A Mac-specific protocol | [06](06-engine-host-port.md), section 5 |
| D9 | **A slim Metal driver that copies upstream policy**, with `synced from` markers | About half of `generate.cpp` is PC policy (multi-GPU, PCIe, VRAM sizing, Windows) | A line-by-line port of `generate.cpp` | [06](06-engine-host-port.md), section 1.3 |
| D10 | **The PLE table is never wrapped.** The host reads 16 rows per token with `pread`. | 28.8 GB, and no demand paging. Measured random 16 KiB reads at 76K-173K IOPS make this cheap. | Wrapping the table and gathering on the GPU | [05](05-memory-and-residency.md), section 6; [appendix A.4](appendix-a-measurements.md) |
| D11 | **Kernels compute upstream's arithmetic in upstream's order**, except for listed deviations. The math mode is a property of the kernel's folder (`safe/` or `fast/`). | Mac-against-Mac comparisons can then be bitwise, and Mac-against-CUDA comparisons stay within the reference's own noise | Free reimplementation of the kernels | [04](04-metal-kernels.md), section 3 |
| D12 | **`strata.metallib` ships next to the engine, paired by SHA-256.** Every pipeline is built at startup and cached in a binary archive. | Setup moves `strata` and `strata.metallib` one after the other. A mismatched pair would run kernels with the wrong layout and give wrong text, not an error, so it must fail at startup. Building every pipeline at startup also moves a missing kernel's failure out of a request. | Building pipelines on first use | [08](08-build-and-distribution.md), sections 5.6 and 10.1 |
| D13 | **macOS 27 and later, arm64, GPU families `apple7` and `metal4`.** Tuned on M5; M1-M4 supported with a fallback matmul path (`simd` tiles instead of `mpp`). | One tested OS. MSL 4.0, the tensor ops and residency sets need macOS 27, so one target removes old-OS fallbacks. | Older macOS versions | [00](00-feasibility-study.md), section 5; [04](04-metal-kernels.md), section 2.1; [08](08-build-and-distribution.md), section 2 |
| D14 | **A separate Git repository with minimal, marked upstream edits and selective compatibility reviews; no upstream history merges** | Keeps compatibility visible while preserving an independent project history | Scheduled merges of upstream history, or a long-lived branch synced rarely | [08](08-build-and-distribution.md), section 1.1; [10](10-implementation-plan.md), section 4 |
| D15 | **Python probes use the standard library only.** `MAC = sys.platform == "darwin"` is the only platform test. | No new dependency for users; no `not WIN` taken to mean Linux | pyobjc | [07](07-server-setup-port.md), section 1.2 |

### 7.2 Deliberately left out of the first release

| Item | Why not now | Where it is recorded |
| --- | --- | --- |
| The CPU expert helper | D1. Revived only if a measurement shows unused memory bandwidth while the GPU computes experts. | [06](06-engine-host-port.md), section 12 |
| Indirect command buffers | Encoding is cheap unless H-4 shows otherwise | [04](04-metal-kernels.md), section 8.3 (L17) |
| A second command queue for the drafter | Only if M-2 shows a gain | [06](06-engine-host-port.md), section 15 |
| Overlapping the next window with the commit; acceptance on the GPU | Optimisations after the gates pass | [06](06-engine-host-port.md), section 10.2; [04](04-metal-kernels.md), section 8.6 (L18) |
| A KV arena per parked conversation | Not needed for parity | [06](06-engine-host-port.md), section 11.3 |
| KV streaming, multi-GPU, PCIe tuning, calibration | No Mac meaning | [06](06-engine-host-port.md), section 1.2; [07](07-server-setup-port.md), section 1.1 |
| Unsloth UD-Q4_K_XL, the Orca files, Swift Q2_0 | Swift Q2_0 splits a layer's experts across shards; Orca needs Q5_0 expert downs; Unsloth's 77 GB of experts fit no target Mac and need `--compat-bf16` | [03](03-model-reference.md), section 2.2 |
| `.pkg`, `.dmg`, Homebrew formula, Docker | The release is one zip installed by setup | [08](08-build-and-distribution.md), sections 11.2 and 14; [07](07-server-setup-port.md), section 13 |
| Smaller slot pools for 32-36 GB Macs, and other memory optimisations | After the first measurements | [05](05-memory-and-residency.md), section 11 |

---

## 8. Invariants

Every change must keep these. Each has a test or a gate that fails when it breaks.

| ID | Invariant | Checked by | Specified in |
| --- | --- | --- | --- |
| I1 | **Decode equals verify.** Row t of a window is bit-identical to the same position decoded at T = 1, and the state after `commit(n_keep)` equals n_keep single-token windows. | `row_t_independent`, `window_eq_decode`, `state_after_commit_eq_decode`, `write_through_eq_commit` | [06](06-engine-host-port.md), section 10.5; [04](04-metal-kernels.md), section 10.4 |
| I2 | **Speculation changes nothing but the time.** Greedy output equals plain decode. Sampled output is a pure function of the seed, whatever the drafts were. | Gate C2 | [06](06-engine-host-port.md), section 10.5; [09](09-testing-and-validation.md), section 4.2 |
| I3 | **Tiered mode equals full mode,** bit for bit, in tokens and in every DONE counter except the tier counters | `tiered_eq_full`, `metal_full_vs_tiered`, gate C4 | [06](06-engine-host-port.md), section 9.1; [05](05-memory-and-residency.md), section 13; [09](09-testing-and-validation.md), section 4.4 |
| I4 | **The conversation cache changes nothing but the time** | Gate C3 | [09](09-testing-and-validation.md), section 4.3 |
| I5 | **The Mac computes the same model as CUDA,** within the references' own spread | Gate C1 | [09](09-testing-and-validation.md), section 4.1 |
| I6 | **The protocol is byte-compatible** | `serve_transcript_test`; the upstream `serve/test_*` modules on a Mac | [06](06-engine-host-port.md), section 5; [07](07-server-setup-port.md), section 10 |
| I7 | **Nothing that must not be resident is wrapped** | `metal_never_wrap_test` | [05](05-memory-and-residency.md), section 6 |
| I8 | **Every flag, variable and protocol string has a disposition** | `options_dispositions_test`, `kernel_env_dispositions_test`, `check_drift.py` | [06](06-engine-host-port.md), sections 3 and 4; [04](04-metal-kernels.md), section 9 |
| I9 | **The engine runs only with its own kernels** | `strata_version`, `metal_manifest_test`, `metal_context_test` | [08](08-build-and-distribution.md), sections 5.6 and 6.3 |
| I10 | **Only the macOS arm64 Metal build is supported.** Other targets fail during platform checks before compiler detection. | `mac-dev` and `mac-release` configure tests on the reference Mac | [08](08-build-and-distribution.md), section 3.4 |
| I11 | **Same inputs, same bits.** Every Mac kernel is deterministic: no floating-point atomics, fixed orders. | Every property in [04](04-metal-kernels.md), section 10.4; principle P8 of [09](09-testing-and-validation.md) | [04](04-metal-kernels.md), sections 3.4 and 6 |

---

## 9. Following upstream

Upstream made 532 commits in the 9 days to the baseline, about 140 a day lately ([00](00-feasibility-study.md), section 6). The project
shares three kinds of code with it:

| Kind | Example | How drift is caught |
| --- | --- | --- |
| Reused unchanged | `src/spec/*`, the GGUF reader, `conv_cache.hpp` | Upstream's own tests, which the `host` label runs on a Mac |
| Patched in place | `direct_file.cpp`, `CMakeLists.txt`, `setup.py`, `serve/server.py` | Every edit carries `// MAC:` or `# MAC:`. `check_drift.py` counts the markers per file and fails when a count changes without a matching entry in `tools/metal/drift.json`. |
| Copied | `Options`, the parser, the prompt-cache policy, resume selection, the spec round, the DONE line | Each copy starts with `// synced from upstream generate.cpp:<range> @ <commit>`. `check_drift.py` hashes the upstream range and fails when it changes. |

`check_drift.py` also fails on a new `--flag` in the parser range, a new `STRATA_*` variable anywhere in `src/` or
`include/`, and a new `INFO` key, `ERR` string or `DONE` field. On the Python side it fails when an upstream change
touches a line next to a marker, or adds a platform branch to a file the port patches
([06](06-engine-host-port.md), section 1.3; [07](07-server-setup-port.md), section 9).

`UPSTREAM_SYNC` holds the source revision used as the current comparison baseline. Compatibility reviews do not
merge upstream history. They classify each `check_drift.py` difference, port only selected changes, and run the tests
for the affected areas ([09](09-testing-and-validation.md), section 7.3). The procedure is
[10](10-implementation-plan.md), section 4.

---

## 10. Glossary

Symbols and terms as the set uses them. Section references point to where each is defined.

| Term | Meaning | Defined in |
| --- | --- | --- |
| A | The always-resident set of the plan: non-expert tensors, the MTP runtime, KV and fixed state | [05](05-memory-and-residency.md), section 3.2 |
| `accept.py` | The acceptance runner: every gate, criterion and benchmark is one of its subcommands, with exit codes | [09](09-testing-and-validation.md), section 3.2 |
| Adaptive swap | A tiered-mode exchange of a resident expert for a more used one, every `--adapt-every` rounds | [06](06-engine-host-port.md), section 9.4 |
| Address table (`addr`) | 48 × 512 × 3 GPU addresses: the gate, up and down slabs of every expert | [05](05-memory-and-residency.md), section 5.3 |
| B | The budget the plan fills: W − reserve | [05](05-memory-and-residency.md), section 3.2 |
| BENCH.mtp, BENCH.plain | The decode benchmarks with and without MTP on the `ca-4k` prompt set | [09](09-testing-and-validation.md), section 5.1 |
| Blob | One expert's bytes: its gate, up and down slabs. Its size is fixed per layer. | [03](03-model-reference.md), section 3.6 |
| C | The prefill chunk size, in rows | [05](05-memory-and-residency.md), section 2.4 |
| C1-C4 | The correctness criteria: Mac against CUDA (C1), speculation (C2), the conversation cache (C3), tiered against full (C4) | [09](09-testing-and-validation.md), section 4 |
| Coder, ISTA, Swift | The model families: ISTA's GSQ-RCO files (Q2_0, IQ2_XS, IQ3_XXS, IQ3_S), ISTA's Coder IQ1_M (256 experts), ukisai's Swift 1.5 files (IQ2_XS, IQ3_XXS) | [03](03-model-reference.md), sections 2.1 and 2.2 |
| Commit | The kernels encoded after a window that make its first `n_keep` rows permanent: GDN replay, indexer append and PLE history | [04](04-metal-kernels.md), section 8.4 |
| CUDA reference PC | The upstream build on an NVIDIA GPU that records the golden set | [09](09-testing-and-validation.md), section 2.4 |
| Draft, drafter | A guessed next token, from the MTP layer (`Mtp`) or from the suffix drafter (prompt lookup) | [06](06-engine-host-port.md), sections 10.3 and 10.4 |
| Family run | A test run on a Mac of the M1, M2, M3 or M4 generation | [09](09-testing-and-validation.md), section 7.4 |
| Fence | The one `MTLFence` per queue that every encoder waits for first and updates last | [04](04-metal-kernels.md), section 8.1 |
| File window | A no-copy Metal buffer over a 16 KiB-aligned range of a mapped file, at most `maxBufferLength` long. [05](05-memory-and-residency.md) calls it a "window" where the context is clear. | [05](05-memory-and-residency.md), section 4.2 |
| Full mode | Every routed expert is resident, read in place from file windows | [05](05-memory-and-residency.md), section 4 |
| G1-G3 | The decision gates at the ends of phases 2, 4 and 6 | [00](00-feasibility-study.md), section 7; [09](09-testing-and-validation.md), section 6 |
| GDN | Gated DeltaNet: the linear-attention block of 36 of the 48 layers (`l % 4 != 3`) | [03](03-model-reference.md), sections 2.1 and 6.3 |
| Geometry file | The facts the plan needs about a model, stored without the model, so that setup can plan before a download | [07](07-server-setup-port.md), section 4.4.3 |
| Golden set | Reference outputs recorded on the CUDA reference PC and with llama.cpp, named by SHA-256 | [09](09-testing-and-validation.md), section 2.5 |
| GR | The hyper-connections: four residual streams of 2560 values, read and written around each block | [03](03-model-reference.md), section 6.2 |
| H | The slot count of one half of the slot pool: the largest number of non-resident experts in any layer | [05](05-memory-and-residency.md), section 3.2, step 5 |
| Handshake | In tiered mode, the per-layer exchange of event values between a window and the `MissResolver` | [06](06-engine-host-port.md), section 9.3 |
| Indexer | The QSA block's selector of the cells each query attends to | [03](03-model-reference.md), section 6.4 |
| KV, `--kv` | The QSA layers' key and value cells, in fp16, int8, q4_0 or k8v4 | [06](06-engine-host-port.md), section 3.3; [04](04-metal-kernels.md), section 4.5 |
| `MissResolver` | The tiered-mode thread that fills the slots of each layer's missing experts | [06](06-engine-host-port.md), section 9.3 |
| MTP | Multi-token prediction: the extra draft layer, loaded from `mtp/rt/` | [03](03-model-reference.md), section 7 |
| `nc` | The mat-vec column count of a window: the smallest of 1, 2, 4 and 8 that is at least T | [04](04-metal-kernels.md), section 4.2 |
| Oracle | A CPU transcription of upstream's arithmetic that a kernel is tested against | [04](04-metal-kernels.md), section 10.1 |
| Pack (native pack) | What `tools/iq_pack.py` writes beside the GGUF: `index.txt`, `dense.bin`, `conversions.json`, `native_experts.txt`, `tokenizer/` | [03](03-model-reference.md), section 4.5 |
| PF-1, PF-2 | The prefill-speed criteria | [09](09-testing-and-validation.md), section 5.2 |
| PLAN line | The one-line result of `strata --plan-only` | [05](05-memory-and-residency.md), section 3.4 |
| PLE | The per-layer n-gram embedding: a 28.8 GB table on the SSD, of which 16 rows are read per token | [03](03-model-reference.md), sections 2.4 and 6.8 |
| Plan (memory) | The startup computation that fits everything into B and chooses the mode | [05](05-memory-and-residency.md), section 3 |
| Plan (`plan[l]`) | The per-layer list of distinct routed experts and their resident flags, written by `strata_moe_plan` | [05](05-memory-and-residency.md), section 5.3 |
| Profile (STRP) | The ranking of (layer, expert) by use, which orders residency and the warm-up | [06](06-engine-host-port.md), section 9.5 |
| Property | A test that compares two Mac runs bitwise | [04](04-metal-kernels.md), section 10.4 |
| QSA | The sparse-attention block of 12 of the 48 layers (`l % 4 == 3`) | [03](03-model-reference.md), sections 2.1 and 6.4 |
| Reference Mac | The M5 Pro 64 GB on macOS 27.0 where the measurements were taken, and the self-hosted runner `strata-ref` | [appendix A.1](appendix-a-measurements.md); [08](08-build-and-distribution.md), section 12.4 |
| Resident set, resident table (`res`) | The experts kept in GPU memory, and the 48 × 512 table that marks them | [05](05-memory-and-residency.md), sections 3.2 and 5.3 |
| Residency set | The `MTLResidencySet` that holds every long-lived buffer, attached to the queue | [05](05-memory-and-residency.md), section 4.3 |
| Round | One decode step of the driver: a window, its acceptance, its commit and the next drafts | [06](06-engine-host-port.md), section 10.2 |
| S | With `--spec S`, the largest window; also the slot pool's size S(R) in the plan | [06](06-engine-host-port.md), section 3.7; [05](05-memory-and-residency.md), section 3.2 |
| Segment | The dispatches between two command-buffer-level operations, encoded as one compute encoder | [04](04-metal-kernels.md), section 8.1 |
| Slot, slot pool | A tiered-mode buffer place for one non-resident expert; the pool has two halves of H slots | [05](05-memory-and-residency.md), section 5.4 |
| T | The number of rows in a window, 1..8; in the prompt path, the rows of a chunk | [04](04-metal-kernels.md), sections 4.2 and 6.1 |
| Tiered mode | A hot set of experts is resident; the rest is read per layer into the slot pool | [05](05-memory-and-residency.md), section 5 |
| Twin | A kernel test that runs a Metal kernel and its CPU oracle on the same inputs | [04](04-metal-kernels.md), section 10.1 |
| W | The GPU working-set budget | [05](05-memory-and-residency.md), section 3.1 |
| Window | One command buffer that runs T rows through the whole model and samples them. "Window" alone means this, except in [05](05-memory-and-residency.md), sections 4 and 6 (file windows). | [04](04-metal-kernels.md), section 5.1 |
| Write-through window | A window whose rows are all kept, with the commit folded in | [04](04-metal-kernels.md), section 8.5 |

---

## 11. Where the details are

| Topic | This document | Detailed document |
| --- | --- | --- |
| Processes and channels | 2.1, 2.2 | [06](06-engine-host-port.md), section 5; [07](07-server-setup-port.md), sections 4 and 6 |
| Installed files | 2.4 | [07](07-server-setup-port.md), sections 4.7 and 4.12; [08](08-build-and-distribution.md), section 11.2 |
| Engine layers, components and dependency rules | 3.1-3.3 | [06](06-engine-host-port.md), sections 1.4 and 1.5; [08](08-build-and-distribution.md), section 4.4 |
| Threads | 3.4 | [06](06-engine-host-port.md), section 2.2 |
| Startup | 4.1 | [06](06-engine-host-port.md), section 2.3 |
| A request, a round | 4.2, 4.3 | [06](06-engine-host-port.md), sections 10 and 11; [04](04-metal-kernels.md), sections 5.1 and 8 |
| The prompt path | 4.4 | [04](04-metal-kernels.md), section 6 |
| The tiered-mode miss path | 4.5 | [06](06-engine-host-port.md), section 9; [05](05-memory-and-residency.md), section 5 |
| Failures | 4.6 | [06](06-engine-host-port.md), sections 2.3, 2.4 and 14 |
| Memory | 5 | [05](05-memory-and-residency.md) |
| Source tree | 6 | [08](08-build-and-distribution.md), section 1; [07](07-server-setup-port.md), section 1.3; [09](09-testing-and-validation.md), section 1.3 |
| Decisions | 7 | The documents named in each row |
| Invariants | 8 | [09](09-testing-and-validation.md), sections 4 and 9 |
| Following upstream | 9 | [10](10-implementation-plan.md), section 4; [06](06-engine-host-port.md), section 1.3 |
| Terms | 10 | The documents named in each row |
