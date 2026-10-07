# 00. Feasibility study: Strata on macOS / Apple Silicon

Status: final for the go/no-go decision, 2026-10-07.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33` (2026-10-02).

## 1. Answer

| Question | Answer |
| --- | --- |
| Can Strata run on a Mac? | **Yes.** Every operation the model needs already runs on Metal in other engines. Our own measurements on an M5 Pro 64 GB show no hardware blocker. |
| Is it worth doing? | **Yes, for a specific scope.** The port is worth it for 36-64 GB Macs, where no existing engine runs this model well, and for giving Mac users the same Strata product (installer, server, Monitor, MCP) as PC users. It is not worth it as one more engine for 96 GB+ Macs; there it would at best match existing engines. |
| Recommendation | **Go**, as a native Metal backend in a separately maintained Strata project initialized from the documented v0.1.34 source baseline, macOS 27 only, Apple M5-class first, Q2_0 first. Re-evaluate at the gates in section 7. |

The rest of this document gives the evidence.

## 2. What Strata is, and what would have to change

Strata is a one-model inference engine for Qwen3.8-Flash-Next (125.7B total, 6.7B active per token, MoE with 512
experts per layer, plus a 28.8 GB n-gram embedding table and an MTP draft layer). Details are in
[03-model-reference.md](03-model-reference.md).

Its design targets a PC with a small discrete GPU:

| Upstream (PC) | Why | On a Mac |
| --- | --- | --- |
| Dense weights plus a hot-expert cache in VRAM | VRAM is small (8-16 GB) | One unified memory pool. The GPU can read everything that is resident. |
| All experts in pinned host RAM; the CPU computes cache misses with AVX2/AVX-512 | PCIe (about 25-50 GB/s) is slower than host RAM, so running misses on the CPU beats copying them | CPU and GPU share one DRAM limit (measured: about 284 GB/s combined). A CPU expert path adds no bandwidth. **All experts run on the GPU.** |
| N-gram table on SSD, 16 row reads per token | 28.8 GB does not fit anywhere useful | Same. Measured random 16 KiB reads at 76K-173K IOPS make this cheap. |
| CUDA (and HIP for AMD) kernels | NVIDIA / AMD GPUs | Metal Shading Language kernels, about 45 to port |
| CUDA graphs, doorbell CPU/GPU handshake, PCIe DMA tuning | Launch overhead and PCIe scheduling | Not needed. Measured 0.78 µs per dispatch and 11 µs per command buffer round trip. |
| Python server, web app, `setup.py` installer, MCP | Product layer | Reused with macOS branches |

So the port is a **new GPU backend plus a simpler memory manager**, not a line-by-line translation. The memory
manager gets simpler at 64 GB+ (everything resident) and gains one new tier for smaller Macs (experts streamed from
SSD into a fixed slot pool).

## 3. Feasibility evidence

### 3.1 The model already runs on Metal elsewhere

Verified on 2026-10-02 (sources in [11-risks-and-open-questions.md](11-risks-and-open-questions.md), section 5):

| Engine | Status on Mac | Notes |
| --- | --- | --- |
| llama.cpp | Official since PR #27742 (merged 2026-08-27); stable v0.5.0 | Metal `supports_op` covers every op: Gated DeltaNet, sparse flash attention, lightning indexer, hyper-connection ops |
| DwarfStar (antirez/ds4, MIT, C + Metal) | Metal is its primary target | Own requantised GGUF (Q2: 41.73 GiB resident). SSD expert streaming in open PR #1056 |
| MTPLX (Apache-2.0) | 4-bit MLX build with MTP | Needs 96-128 GB |
| oMLX (Apache-2.0) | v0.7.0 adds MTP, expert offload, PLE on SSD | Expert offload cannot be combined with its MTP |
| mlx-vlm (MIT) | Official since v0.6.17 | `ple_storage.py` reads n-gram rows on demand |
| LM Studio, Ollama | GGUF / MLX 4-bit | LM Studio lists "96 GB minimum"; Ollama's MLX build is 105 GB |

This proves the operations are expressible and fast enough in Metal. DwarfStar and llama.cpp (both MIT) are usable
references for kernels.

### 3.2 The hardware is adequate

Measured on an M5 Pro 64 GB, macOS 27.0 ([appendix-a-measurements.md](appendix-a-measurements.md)):

| What | Measured | What it means for the port |
| --- | --- | --- |
| GPU read bandwidth | 238-282 GB/s; random 1.31 MiB expert gathers at 228-230 GB/s | Decode can read experts in any order at near-peak speed |
| CPU + GPU combined | about 284 GB/s total | No gain from a CPU expert path |
| GPU working-set budget | 51.84 GiB (`recommendedMaxWorkingSetSize`) | Q2_0 (35.0 GiB resident) fits with about 14 GiB left for KV and scratch |
| Largest single buffer | 38.88 GiB (`maxBufferLength`) | Weights must be split across several buffers |
| SSD random reads | 12.7-12.9 GB/s at 1.31 MiB with 8+ threads; 173K IOPS at 16 KiB | SSD expert tier and n-gram rows are viable |
| GPU touch of a no-copy mmap buffer | Makes the whole buffer resident (8 GiB in 1.2 s) | The n-gram table and SSD-tier experts must never be wrapped as Metal buffers |
| Dispatch overhead | 0.78 µs per dependent dispatch; 11.2 µs per round trip | About 3% of a decode window; no CUDA-graph equivalent needed |
| Half matmul (MPP tensor ops, M5 Neural Accelerators) | 22-29 TFLOPS | Prefill is compute-bound and has enough compute for 450-900 tok/s |

### 3.3 Expected performance on the reference Mac

Projections, not measurements (derivation in [appendix-a-measurements.md](appendix-a-measurements.md), section A.7):

| Workload | M5 Pro 64 GB, Q2_0 | Upstream on RTX 5070 12 GB | Best existing Mac engine (measured) |
| --- | ---: | ---: | --- |
| Decode with MTP | 50-75 tok/s | 93 | 65-90 on M3 Ultra (DwarfStar), 73.5 on M5 Max (MTPLX) |
| Prefill | 450-900 tok/s | 2,170 at 32K | 1,331 on M3 Ultra, 1,453 on M5 Max |

Agent research projected 45-60 tok/s for an M5 Pro running an engine of DwarfStar quality. Existing Mac engines use
only 12-24% of memory bandwidth during decode; they are limited by kernel count, synchronisation and serial work, not
by bytes. That is where a careful port can do better, and why the requirements set a decode target above existing
engines (see [01-requirements.md](01-requirements.md)).

### 3.4 Engineering effort

Upstream size at the baseline commit:

| Area | Lines | Port effort |
| --- | ---: | --- |
| CUDA kernels (`src/kernels/cuda`, 44 files) | 16,895 | Rewrite in MSL. The largest single job. |
| CPU expert kernels (`src/kernels/cpu`, AVX2/AVX-512) | 3,530 | Dropped on Mac, except as an optional NEON reference for tests |
| Engine core, prefill, speculative decoding, program (`src/core`, `src/prefill`, `src/spec`, `src/program`) | 22,537 | Mostly reusable C++; the CUDA runtime calls and VRAM/RAM tiering are replaced |
| Headers (`include/strata`) | 10,001 | Mostly reusable; CUDA types and HIP shims are replaced |
| Platform, artifacts, n-gram reader | 1,657 | Small port (Windows/Linux APIs to macOS) |
| Python server, setup, tools | 23,208 | Reused; macOS branches for GPU detection, memory fit, paths, launchers |

Estimate: 5-7 months for one senior engineer who knows GPU programming to reach parity on Q2_0 and IQ2_XS, or roughly
half that for a small team or with heavy use of coding agents. The phase breakdown is in
[10-implementation-plan.md](10-implementation-plan.md).

## 4. Value analysis

### 4.1 Which Macs can run the model today

Resident sizes (shard 1 of each GGUF, plus about 3-6 GiB for KV and scratch at 8-32K context) against the GPU
budget (`recommendedMaxWorkingSetSize`, measured or reported at 74-81% of RAM):

| RAM (Macs that offer it) | GPU budget | Existing engines | What a Strata port offers |
| --- | ---: | --- | --- |
| 16-24 GB (M5 Air, M6 mini, base M5 Pro) | about 12-19 GiB | Nothing usable | Not a target. Too little memory for the dense weights plus a useful expert cache. |
| 32 GB (M5, M6) | about 24 GiB | Nothing usable; DwarfStar's PR #1047 (closed, superseded by the open #1056) reached 7.86 tok/s on a 32 GiB M1 Max | Q2_0 with about half of experts on SSD, or Coder IQ1_M with about a third. Prompt processing is SSD-bound. Stretch target. |
| 36 GB (M5 Max 32-core GPU) | about 27 GiB | Nothing usable | Q2_0 with about 35% of experts on SSD |
| **48 GB (M5 Pro, M5 Max 40-core)** | about 37 GiB | DwarfStar Q2 (41.73 GiB) does not fit; llama.cpp with ISTA Q2_0 is marginal, short context only | **Primary target.** ISTA Q2_0 (35.0 GiB) with the coldest 7-10% of experts on SSD, chosen with Strata's expert profile. Close to fully resident speed. Coder IQ1_M (27.6 GiB) fully resident. |
| **64 GB (M5 Pro, M5 Max 40-core)** | about 52 GiB | llama.cpp Q2_0 (hit by issue #29465), DwarfStar Q2 at `--ctx 8192` | **Primary target.** Q2_0, IQ2_XS and IQ3_XXS fully resident with long context, because the n-gram table is never counted and KV can be 8-bit or k8v4 |
| 96-128 GB (M5 Ultra, M5 Max) | about 72-96 GiB | MTPLX, DwarfStar, oMLX, llama.cpp, LM Studio, Ollama | Parity only. Supported, not optimised for. |
| 256-512 GB (M5 Ultra) | 190+ GiB | Same | Parity only |

The 48 GB and 64 GB configurations are the mainstream high-end MacBook Pro and Mac mini models. They are exactly where
the existing engines either do not fit or cut context.

### 4.2 What is unique about a Strata port

1. **Tiered experts for mid-memory Macs.** Strata's core idea (keep hot experts close, stream cold ones) is what 36-48
   GB Macs need. Strata already ships a measured expert-popularity profile (`data/expert-profile*.bin`). On a Mac the
   tiers become "resident in unified memory" and "read from SSD into a fixed slot pool". No released Mac engine
   combines this with MTP.
2. **Uses ISTA's GSQ-RCO files as-is.** Q2_0 is 35.0 GiB resident, 6.7 GiB smaller than DwarfStar's own Q2 (41.73
   GiB), and ISTA measured Q2_0 as the fastest of its formats. Mac and PC users download the same files.
3. **The n-gram table never touches GPU memory.** llama.cpp's open issue #29465 counts it against the GPU budget; the
   port reads rows with `pread`, so 64 GB Macs keep about 14 GiB for context.
4. **Decode efficiency headroom.** Existing Mac engines use 12-24% of bandwidth in decode. Strata's verify-window
   design (one weight pass for up to 4 tokens) and fused kernels can raise that.
5. **One product across platforms.** The same installer, the OpenAI- and Anthropic-compatible server, the web app with
   Monitor, MCP client and server, and the conversation cache. Mac users currently get these only through general
   tools (LM Studio, Ollama) that need 96 GB+ for this model.

### 4.3 What a port does not add

- On 96 GB+ Macs the port would likely match, not beat, DwarfStar and MTPLX. Users there have good options.
- It does not make the model run on 16-24 GB Macs.
- It does not reach upstream's RTX 5070 prefill speed on an M5 Pro (about 2-4x lower). M5 Max and M5 Ultra close that
  gap.

### 4.4 Alternatives considered

| Alternative | Why not chosen |
| --- | --- |
| Do nothing; point Mac users to llama.cpp or DwarfStar | Leaves the 36-48 GB tier unserved and the Strata product PC-only |
| Strata server and installer in front of llama.cpp on Mac (what the `jinzy0623/Strata-macOS` project started) | Gets the UX but not the tiered experts, MTP verify windows, conversation cache or Monitor telemetry, which live in the engine. Useful only as a stopgap. |
| Contribute SSD expert streaming to llama.cpp or DwarfStar instead | Valuable for the community, but does not give Strata users anything, and both projects use different file formats and process models |
| Port via MLX (Python or Swift) | MLX would hide the memory control that the tiered design depends on (wired limits, `pread` into slots, residency sets). The 4-bit MLX path also needs 96 GB+. |
| Translate CUDA with a source-to-source tool (HIP-style) | There is no such tool for Metal; MSL has a different memory and threading model |

## 5. Scope decision

| Decision | Choice | Reason |
| --- | --- | --- |
| OS | macOS 27 and later only | One tested OS. The kernels use MSL 4.0 and the Metal Performance Primitives tensor ops, and the host uses residency sets, so a single target removes old-OS fallbacks. macOS 27 shipped 2026-09-14. |
| Hardware | Apple Silicon only; tuned for M5-generation (Neural Accelerators); M1-M4 supported with a fallback matmul path | M5 is the current lineup and the only one with Neural Accelerators |
| Memory | 48 GB and 64 GB first; 36 GB and 32 GB next; 96 GB+ supported, not tuned | Section 4.1 |
| Formats | Q2_0 first, then IQ2_XS, Coder IQ1_M, IQ3_XXS, IQ3_S; UD-Q4_K_XL later | Q2_0 has the simplest kernels and is the fastest per ISTA; IQ formats need lookup-table kernels |
| Engine | New Metal backend in a separately maintained Strata project initialized from the documented v0.1.34 source baseline, same engine process and protocol, Python server reused | Section 4.4 |
| Language | C++20 for the engine core (as upstream), Objective-C++ for the Metal host layer, MSL 4.0 for kernels | Keeps upstream core code shareable; avoids a wrapper library that can lag the SDK |

Full requirements are in [01-requirements.md](01-requirements.md).

## 6. Main risks

The full register is in [11-risks-and-open-questions.md](11-risks-and-open-questions.md). The ones that could change
the decision:

1. **Upstream moves very fast** (532 commits in the 9 days to the baseline, about 140 per day lately). A project can fall
   behind quickly. Mitigation: a backend boundary that keeps upstream files untouched where possible, and selective upstream reviews
   with a parity test gate ([10-implementation-plan.md](10-implementation-plan.md), section 4).
2. **Competitors close the gap.** DwarfStar PR #1056 (SSD expert streaming) or oMLX could ship a good 48 GB story
   first. Mitigation: gate G2 below.
3. **Decode stays overhead-bound** like existing engines and lands at 30-45 tok/s. That is still useful on 48 GB, but
   weakens the 64 GB case.
4. **macOS memory behaviour.** GPU working-set limits, wired memory and sparse-resource stalls (Splash issue #220) are
   less documented than CUDA's. Mitigation: fixed slot pools, no sparse heaps, measurements first.

## 7. Decision gates

| Gate | When | Pass criterion | If it fails |
| --- | --- | --- | --- |
| G1 | End of phase 2 (first token on Metal) | Q2_0 greedy output matches the reference for 256 tokens on 10 prompts; plain decode at least 20 tok/s on M5 Pro 64 GB | Profile; if under 15 tok/s with no clear cause, stop and contribute to llama.cpp instead |
| G2 | End of phase 4 (SSD tier) | M5 Pro 48 GB (or a 64 GB Mac limited to a 37 GiB budget) runs Q2_0 with MTP at 20 tok/s or more at 8K context | If an existing engine already ships better 48 GB numbers, refocus on product parity (server, setup, Monitor) and long context on 64 GB |
| G3 | End of phase 6 (feature parity) | All acceptance criteria in [01-requirements.md](01-requirements.md) | Release as beta with the gaps listed |

Each gate is a command with exit codes, `accept.py g1`, `g2` or `g3`. Its steps, the exact meaning of "matches the
reference", and the 64 GB form of G2 are in [09-testing-and-validation.md](09-testing-and-validation.md), section 6.
