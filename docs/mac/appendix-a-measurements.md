# Appendix A. Measurements on the reference Mac

These are the numbers that the feasibility study and the architecture rest on. Sections A.2-A.6 were measured on
2026-10-02 with the programs in [spikes/metal-feasibility/](spikes/metal-feasibility/). Sections A.8-A.12 name their
own spike folders. Rerun them on any new target machine before trusting a projection for it.

## A.1 Machine

| Item | Value |
| --- | --- |
| Chip | Apple M5 Pro (6 performance + 12 efficiency CPU cores, 20-core GPU) |
| Memory | 64 GB unified (68,719,476,736 B), Apple spec 307 GB/s |
| OS | macOS 27.0 (26A428) |
| Toolchain | Xcode 27.0 (27A266a), Metal compiler 32023.921 |
| `hasUnifiedMemory` | 1 |
| `recommendedMaxWorkingSetSize` | 55,662,788,608 B = 51.84 GiB (81.0% of RAM) |
| `maxBufferLength` | 41,747,087,360 B = 38.88 GiB |
| `maxThreadgroupMemoryLength` | 32,768 B |
| GPU families | Apple10 yes, Apple11 no, Metal4 yes |
| `sysctl iogpu.wired_limit_mb` | 0 (OS default) |

Units: GB = 10^9 B, GiB = 2^30 B. Bandwidths are GB/s.

## A.2 Memory bandwidth (`bw`, `conc`)

| Test | Result |
| --- | --- |
| GPU streaming read, shared buffer, 1 GiB | 282.4 GB/s |
| GPU streaming read, shared buffer, 4 GiB | 237.5 GB/s |
| GPU streaming read, shared buffer, 8 GiB | 237.7 GB/s |
| GPU streaming read, warm (the `conc` baseline) | 277.1 GB/s |
| GPU gather of 480 random 1.31 MiB blocks out of 16 GiB (one token's worth of routed experts, 10 x 48) | 230.0 GB/s, 2.87 ms per pass |
| GPU gather of 1,920 random 1.31 MiB blocks out of 16 GiB (a 4-token verify window) | 228.3 GB/s, 11.57 ms per pass |
| CPU read, 6 threads | 184.5 GB/s |
| CPU read, 18 threads | 174.7 GB/s |
| GPU + CPU at the same time, 4 CPU threads | GPU 195.7 + CPU 89.0 = 284.7 GB/s |
| GPU + CPU at the same time, 8 CPU threads | GPU 195.9 + CPU 87.1 = 283.0 GB/s |
| GPU + CPU at the same time, 12 CPU threads | GPU 178.5 + CPU 105.7 = 284.2 GB/s |

What this shows:

- **The GPU gets 75-90% of the spec bandwidth**, and an expert-sized random gather costs almost nothing more than a
  streaming read. Routed experts do not need to be contiguous.
- **CPU and GPU share one DRAM limit of about 284 GB/s.** Whatever the CPU reads, the GPU loses. Upstream Strata computes
  expert cache misses on the CPU because PCIe is slower than host RAM; on unified memory that path adds no bandwidth and
  only adds synchronisation. The port computes every expert on the GPU (see
  [02-architecture.md](02-architecture.md)).

## A.3 mmap and residency (`bw`, `ssd`)

| Test | Result |
| --- | --- |
| `newBufferWithBytesNoCopy` over a 4 GiB mmap'd file | Works |
| First GPU pass over that buffer (page cache warm) | 259.3 GB/s |
| Later GPU passes over that buffer | 268.8 GB/s |
| `newBufferWithBytesNoCopy` over a cold 8 GiB mmap'd file | Created in 0.001 s; resident fraction (`mincore`) 0.000 before, 0.007 after |
| GPU kernel that touches 1 MiB of that cold 8 GiB buffer | 1.205 s wall; resident fraction afterwards **1.000** |
| `newBufferWithBytesNoCopy` over `mmap(PROT_READ, MAP_PRIVATE)` and over `MAP_SHARED`, whole file and a 64 MiB window at a 16 KiB-aligned offset (`mapflags`) | All work; GPU XOR of the words equals the CPU XOR |
| The same with a 4 KiB-aligned pointer, or a length that is not a multiple of 16 KiB (`mapflags`, also under `MTL_DEBUG_LAYER=1`) | Accepted, and read correctly. This is not documented behaviour; the engine still uses 16 KiB-aligned pointers and lengths. |

What this shows:

- An mmap'd weight file can be wrapped with no copy and read at full GPU speed.
- Both `MAP_PRIVATE` (what upstream `gguf_reader.hpp` uses) and `MAP_SHARED` mappings can be wrapped, so the GGUF
  reader needs no patch.
- **The GPU does not demand-page.** Using any part of a no-copy buffer makes the whole buffer resident before the
  command buffer runs. A no-copy buffer over the 28.8 GB n-gram table, or over experts meant to stay on SSD, would pull
  all of it into memory. The n-gram rows and SSD-tier experts must be read with `pread` into small staging buffers or
  fixed slots (see [05-memory-and-residency.md](05-memory-and-residency.md)).

## A.4 SSD (`ssd`)

Files were written with `F_NOCACHE` and read back with `F_NOCACHE`, so every read came from the SSD.

| Test | Throughput | Reads/s | p50 | p99 |
| --- | ---: | ---: | ---: | ---: |
| Uncached write, file A (8 GiB) | 8.87 GB/s | | | |
| Uncached write, file B (8 GiB) | 7.89 GB/s | | | |
| Random 1,376,256 B reads, 1 thread | 4.82 GB/s | 3,503 | 0.280 ms | 0.385 ms |
| Random 1,376,256 B reads, 4 threads | 11.60 GB/s | 8,425 | 0.430 ms | 1.686 ms |
| Random 1,376,256 B reads, 8 threads | 12.68 GB/s | 9,211 | 0.851 ms | 1.017 ms |
| Random 1,376,256 B reads, 16 threads | 12.89 GB/s | 9,364 | 1.694 ms | 1.893 ms |
| Random 16 KiB reads, 1 thread | 0.18 GB/s | 11,238 | 0.086 ms | 0.136 ms |
| Random 16 KiB reads, 8 threads | 1.24 GB/s | 75,890 | 0.098 ms | 0.169 ms |
| Random 16 KiB reads, 32 threads | 2.83 GB/s | 172,823 | 0.176 ms | 0.303 ms |

1,376,256 B (1.3125 MiB) stands for one Q2_0 routed expert: gate + up + down are 3 x 2560 x 640 = 4.92M weights, and
the Q2_0 file holds about 34 GB of routed experts for 24,576 experts, about 1.38 MB each. 16 KiB stands for an n-gram
row read.

What this shows:

- **Expert-sized random reads reach 12-13 GB/s with 8 or more threads in flight.** One thread gets only 4.8 GB/s. The
  SSD expert tier needs a queue depth of at least 8.
- **The n-gram table is not an I/O problem.** Strata reads 16 rows (at most 23 KB) per token. At 0.1-0.2 ms per read
  with 16 in flight, that is about 0.2 ms per token without the page cache, and the page cache hides most of it.
- **A full expert miss per token is expensive.** 480 experts x 1.31 MiB = 660 MB per token. At 12.7 GB/s that is 52 ms,
  or about 19 tok/s at a 0% hit rate before any compute. The SSD tier works only with a high hit rate in the resident
  expert cache (see [00-feasibility-study.md](00-feasibility-study.md)).

## A.5 Dispatch and submission overhead (`conc`)

| Test | Result |
| --- | --- |
| 100 dependent dispatches in one compute encoder | GPU 0.079 ms (0.79 µs each), CPU encode 0.012 ms |
| 1,000 dependent dispatches in one compute encoder | GPU 0.779 ms (0.78 µs each), CPU encode 0.104 ms |
| Empty command buffer, commit to completion | 11.2 µs |

What this shows:

- A decode step of about 1,000 small kernels costs about 0.8 ms of pure dispatch overhead on the GPU, and about 0.1 ms
  of CPU time to encode. That is about 3% of a 25 ms verify window, so Metal needs no equivalent of CUDA graphs to reach
  the targets. Fusing kernels still pays because each dispatch boundary is a full memory barrier.
- A CPU-GPU round trip costs about 11 µs. The engine can afford one or two per verify window (sampling, MTP
  acceptance), but not one per layer. Routing decisions must stay on the GPU.

## A.6 Matrix-multiply throughput (`mm`)

Metal Performance Primitives `mpp::tensor_ops::matmul2d`, MSL 4.0, `execution_simdgroups<4>`, K = 4096. These are the
M5 Neural Accelerator paths that prefill uses.

| Kernel | M = N = 2048 | M = N = 4096 | M = N = 8192 |
| --- | ---: | ---: | ---: |
| half x half -> float, 64 x 32 output tile (`mm`) | 15.93 TFLOPS | 11.43 TFLOPS | 11.05 TFLOPS |
| half x half -> float, 64 x 64 output tile (`mm64`) | 29.37 TFLOPS | 22.36 TFLOPS | 21.71 TFLOPS |
| int8 x int8 -> int32, 64 x 64 output tile (`mmi8`) | 28.18 TOPS | 27.45 TOPS | 22.52 TOPS |

The half results match a CPU reference (2.4573). `mmi8` reinterprets the half test data as int8, so its check value is
meaningless; only its throughput counts.

What this shows:

- About 22-29 TFLOPS of half matmul is available on a 20-core M5 Pro with a naive single-level kernel. Larger tiles
  matter a lot (2x from 64 x 32 to 64 x 64).
- int8 is no faster than half here. Prefill should dequantise weights to half tiles in threadgroup memory and use the
  half path; int8 activations are not worth the accuracy risk.

## A.7 Projections derived from these numbers

These are estimates, not measurements. The byte counts come from
[03-model-reference.md](03-model-reference.md).

### Decode (Q2_0, all weights resident)

| Item | Value |
| --- | --- |
| Dense bytes per forward pass (mixers, attention, shared experts, routers, GR, head) | 3.37 GB ([03](03-model-reference.md), section 4.2) |
| Routed expert bytes per token | 0.66 GB (doc 03 section 4.3) |
| One verify window of 4 tokens (dense once, routed experts for 4 tokens with some overlap, MTP drafting) | about 6.3 GB |
| Time per window at 230-250 GB/s | 25-27 ms |
| Accepted tokens per window (upstream reports 1.6-1.8x from MTP, up to 4 per window) | 2.0-2.8 |
| Bandwidth ceiling | about 75-110 tok/s |
| Expected after overhead (60-75% of ceiling) | about 50-75 tok/s |

For comparison: upstream Strata does 93 tok/s on an RTX 5070 12 GB; existing Mac engines measured 44-50 tok/s plain
and 65-90 tok/s with MTP on Macs with 2-3x this machine's bandwidth.

### Prefill (Q2_0)

| Item | Value |
| --- | --- |
| Compute per token (2 x 6.67B active weights, doc 03 section 2.1) | about 13.3 GFLOP |
| Sustained half matmul at 50-80% of the measured 22-29 TFLOPS | 11-23 TFLOPS |
| Matmul-bound ceiling | about 830-1,700 tok/s |
| Expected after dequant, attention, GDN scan and non-matmul ops | about 400-850 tok/s |

For comparison: upstream Strata does 2,170 tok/s at 32K on an RTX 5070; MTPLX measured 1,453 tok/s at 4K on an M5 Max
(40-core GPU, twice this GPU).

### SSD expert tier (48 GB Macs)

| Hit rate in the resident expert cache | Bytes from SSD per token | I/O time per token at 12 GB/s | I/O-bound ceiling |
| ---: | ---: | ---: | ---: |
| 0% | 660 MB | 55 ms | 18 tok/s |
| 70% | 198 MB | 16.5 ms | 60 tok/s |
| 85% | 99 MB | 8.3 ms | 120 tok/s |
| 95% | 33 MB | 2.8 ms | 360 tok/s |

With a verify window, 4 tokens share one I/O wait, but the union of their experts is up to 4 times larger. Real numbers
also depend on overlapping I/O with compute, which DwarfStar PR #1056 reports as worth "up to 40%".

## A.8 Build and distribution (`spikes/metal-build/`)

Measured on 2026-10-02 with the build skeleton of [08-build-and-distribution.md](08-build-and-distribution.md): the
root edits, `cmake/metal/`, two self-test kernels, and the `src/metal/` sources as stubs. The engine sizes and times
below are the skeleton's. The real engine's are measurements B-8 and B-9 of doc 08.

### Toolchain

| Item | Value |
| --- | --- |
| Xcode | 27.0 (27A266a), macOS SDK 27.0 |
| Compiler and linker | Apple clang 21.0.0 (clang-2100.3.34.2), ld-27037.1 |
| Metal toolchain | `Apple metal version 32023.921 (metalfe-32023.921.6)`, component `com.apple.dt.toolchain.Metal.32023.921.5`, installed with `xcodebuild -downloadComponent MetalToolchain` |
| CMake and Ninja | 4.4.3 and 1.13.2, the `requirements.txt` wheels |
| Python | 3.13.12 (Homebrew) and 3.9.6 (`/usr/bin/python3`), both with zlib 1.2.12 |

### The engine skeleton

| Item | Value |
| --- | --- |
| Fresh configure and build, `mac-dev` | 6.56 s |
| Fresh configure and build, `mac-release` | 5.61 s |
| `strata`, unstripped | 120,568 B, SHA-256 `2abaa4f03fe8d1c6...` |
| `strata`, after `strip -S -x` in place | 107,576 B, SHA-256 `b9bbb74b97a17ca7...`, `flags=0x20002(adhoc,linker-signed)`, `Identifier=strata` |
| UUID of `strata` and of its dSYM | `EDBA2B49-2E05-357A-A976-4CEBF1D1CE14` for both |
| dSYM | `Info.plist` 635 B; `DWARF/strata` 722,584 B (SHA-256 `c4fff08a...`); `Relocations/aarch64/strata.yml` 90 B with `binary-path: strata` |
| `strata.metallib`, release | 8,904 B, SHA-256 `b763d0449788d2a31baff6af3ec61ab2081a4398b6af64cbdfce04e369b23cc5` |
| `strata.metallib`, `mac-dev` (sources kept) | 29,381-29,387 B, depending on the length of the build folder's path. `metal-dsymutil -flat -remove-source` on it, in place, gives the release library byte for byte (`b763d044...`). |
| `strata.metallibsym` | 29,363-29,411 B, depending on the path. It differs between folders: it records the `metallib` command line. |
| Forbidden strings (doc 08 section 11.7, check 6) in the stripped `strata` and the release library | none |
| `metal-dsymutil -flat -remove-source` on a copy of the release library | no change (check 8) |

**Reproducibility.** Two checkouts at different paths (`z-project/`, `z-project-2/`), and the build folder names `build/`
and `build-release/`, gave identical bytes for the unstripped and the stripped `strata`, the dSYM and the
library. Before the fix of doc 08 section 4.6, the stripped binaries differed. The debug map's object paths and
dates fed the UUID, and the code signature covers the UUID. With `-oso_prefix` and `ZERO_AR_DATE=1`, `dsymutil`
(run in the build folder) gives no warning. The DWARF has `comp_dir` `.` and names `./src/...`. There are no SOL
stabs, so no SDK path reaches the stripped binary. The dSYM names SDK headers by absolute path, in lambda names.

**Incremental builds.**
- A comment-only edit of a `.metal` file, with `STRATA_METAL_EMBED_SOURCES` OFF, rebuilds the AIR and relinks the
  library. The library's hash is unchanged (`b763d044...`), so `strata` is neither recompiled nor relinked. The
  next build does nothing.
- An edit of `manifest.txt` reconfigures, regenerates `kernel_manifest.inc`, recompiles `context.mm` and relinks
  `strata`.

### Pipelines and capture

| Item | Value |
| --- | --- |
| Skeleton engine, first run (empty archive): 2 pipelines compiled, added and serialised | 48.5 ms. Earlier runs: 34.8-38.5 ms. |
| Second run, both pipelines from the archive | 0.5 ms (0.4-0.5 ms over runs) |
| `STRATA_METAL_ARCHIVE=0`, repeated runs (macOS's own shader cache) | 0.8-0.9 ms |
| Archive file | 28.9 KB, mode 0644, in a 0700 folder |
| An invalid archive | Opening it fails with `has an invalid format`. The engine ignores it, compiles (2.01 ms, from the system cache) and writes a new one. |
| A pipeline label in the descriptor | Does not stop archive hits |
| Library load (`newLibraryWithData:`) | 0.17-0.42 ms |
| First compile of an elementwise kernel in a fresh process | 133 ms, which includes starting the compiler service. Warm: 4 ms. |
| First compile of the MPP `matmul2d` kernel | 87 ms. From an archive: 0.14 ms. |
| `--metal-capture` of the self-test dispatches | An 80 KB `.gputrace` |
| `MTL_CAPTURE_ENABLED` | Without it, `supportsDestination:MTLCaptureDestinationGPUTraceDocument` is NO. Set with `setenv` by the program before the device exists, the capture starts. |

The archive-against-system-cache comparison with the full kernel set is measurement B-1 of doc 08.

### Test oracle and the PC build

| Item | Value |
| --- | --- |
| ggml CPU oracle (doc 08 section 6.1), configure and build | 2.9 s and 4.2 s. It links `libSystem` and `libc++` only, and prints `iq2_xs blck=256 size=74 vec_dot=1`. |
| `compare_build_graphs.py`, upstream against the project, Linux x86-64 configure | `same build: 57 compile commands, 1787 lines of build.ninja`, exit 0 |
| The same, Intel macOS configure | Exit 1, listing only `direct_file_async_test` (intended, doc 08 section 3.3) |
| `apply.sh` on a fresh clone at `1678de33` (historical spike, before D-IMPL-179) | `20 lines with # MAC: in CMakeLists.txt`; the tree equals the scratch project |

### `strata-vision`

| Item | Value |
| --- | --- |
| Build with the release definitions (`vision-repro.sh`), `-j 14` | 11.8-17.7 s over three runs |
| `<build>/bin/strata-vision` | 8,180,336 B, identical from two build folders (`9dc9ba48...`) |
| After `strip -S -x` in place | 7,453,408 B, identical (`fdfc71f3...`), `Identifier=strata-vision`, ad hoc |
| Linked | `Accelerate`, `Foundation`, `Metal`, `MetalKit`, `CoreFoundation`; `libSystem`, `libc++`, `libobjc`. No `LC_RPATH`. |
| ggml defaults that apply | `GGML_BLAS=ON` (vendor `Apple`), `GGML_ACCELERATE=ON` |
| Absolute paths in the stripped binary | one: `/tmp/perf-metal-%d.gputrace` |
| With no argument | usage on stderr, exit 2 |
| Defined symbols of vendored libraries (`nm -U -j`, unstripped) | `stbi_` 153, `ma_decoder` 17, `subprocess_` 18, `sha256_` 6; `httplib`, `nlohmann`, `XXH`, `SHA1` 0 |

### The release zip

The zip writer of doc 08 section 11.3 (`spikes/metal-build/release/detzip.py`), on `strata.metallib` and the stripped
`strata-vision`, gave the same zip (`fc6c17fc...`, 2,612,677 B) for each of these:
- Python 3.9.6 and 3.13.12;
- umask 022 and 077;
- the source files' mtimes and modes changed (2020-01-01, mode 0600).

Both Pythons used zlib 1.2.12, so another zlib is still measurement B-7. `strata-vision` deflates to about 35% of its
size, so the skeleton's asset is about 2.6 MB.

## A.9 Model file headers (`spikes/model-headers/`)

Measured on 2026-10-02 and 2026-10-03 with HTTP range requests against the revisions pinned in
[03-model-reference.md](03-model-reference.md), section 2.2, and the Hugging Face API. No model was downloaded. Doc 03
uses these results in sections 2-4 and 9.3; the scripts and the parsed headers are in
[spikes/model-headers/](spikes/model-headers/).

### Headers

| File | `data_start` of shard 1 | Tensors in shard 1 / shard 2 | Shard 2 |
| --- | ---: | ---: | --- |
| ISTA Q2_0 | 11,024,672 | 1223 / 1 | The shared ISTA shard 2: one tensor, `data_start` 192 |
| ISTA IQ2_XS | 11,024,672 | 1223 / 1 | The same file |
| ISTA IQ3_XXS | 11,024,256 | 1223 / 1 | The same file |
| ISTA IQ3_S | 11,024,384 | 1223 / 1 | The same file |
| Coder IQ1_M | 11,024,416 | 1223 / 1 | The same file (same size and LFS object) |
| Swift IQ2_XS | 10,967,808 | 342 / 882 | `data_start` 56,320 |
| Swift IQ3_XXS | 10,966,240 | 317 / 907 | `data_start` 57,920 |

| Item | Result |
| --- | --- |
| Tensors per file | 1224 in every file, both shards together |
| Shard sizes | The table of doc 03 section 2.4. The ISTA and Coder shard 2 is 28,800,138,432 B in every repository. |
| Tokeniser | Tokens, merges and token types identical in the seven shard 1 files (`tokcmp.py`). Chat template: one text for ISTA and Coder, another for Swift. |
| Tensor types | No type outside the Phase-1 list of doc 03 section 3.3. Distinct (gate-up, down) expert type pairs over the 48 layers: 1 in Q2_0, 3 in IQ2_XS (ISTA and Swift), 7 in IQ3_S and Coder, 10 in IQ3_XXS (ISTA and Swift) (`perlayer.py`). |
| `dense.bin` size, computed | 1,485,606,400 B for Q2_0 (`dpack.py`) |
| Swift layout | The PLE table is in shard 1, at tensor offset 350,863,360 (IQ2_XS) and 450,191,360 (IQ3_XXS). Shard 2 holds the experts of the later layers (doc 03 section 3.9). |

### Single tensors and rows

| Check | Range read | Result |
| --- | --- | --- |
| `output_hc_norm.weight`, ISTA Q2_0 | 454,621,472-454,662,431 of shard 1 (40,960 B, F32 `[10240]`) | Min 0.6328, max 13.375, mean 3.75: the file stores the final multiplier (doc 03 section 3.7) |
| Layer 0 routers of Swift IQ2_XS (`ffn_gate_inp`, `ffn_gate_inp_shexp`, F32) | 30,025,918,208-30,031,171,327 of shard 1 | All 1,313,280 words have zero low 16 bits: BF16 values stored as F32, so narrowing them to BF16 is exact |
| PLE rows 0, 12345, 1,000,000, 200,000,000 and 320,001,535 | ISTA shard 2 at `192 + 90 r`; Swift IQ2_XS shard 1 at `10,967,808 + 350,863,360 + 90 r` | Byte-identical. Rows 0, 12345 and 320,001,535 decode to upstream's `kProbeHead`, `kProbeTail` and `kProbeSum`; the last row is all zeros. Upstream's PLE oracle applies to Swift. |

### MTP source

| Item | Result |
| --- | --- |
| Tensors | 31 `mtp.*` tensors, all BF16, in 28 of the 131 safetensors shards of `Qwen/Qwen3.8-Flash-Next` at `de4b8e4d…` |
| Bytes | 5,214,301,696 |
| Inventory | `mtp-inventory.json` and `.md`, with absolute byte ranges per tensor. A rerun of `mtpinv.py` on 2026-10-03 reproduced it exactly. Every range response was `206`. |

## A.10 Metal numerics (`spikes/metal-numerics/`)

Measured on 2026-10-03 on the machine of section A.1 with `Apple metal version 32023.921`. The same kernels were
built three ways:
- `safe`: the flags of [08-build-and-distribution.md](08-build-and-distribution.md), section 5.3;
- `safe_contract`: `safe` without `-ffp-contract=off`;
- `fast`: `-fmetal-math-mode=fast`.

Results are bit-for-bit comparisons with the CPU, or comparisons with a double-precision reference rounded to float.
[04-metal-kernels.md](04-metal-kernels.md), section 3, turns them into rules. The full output is
`spikes/metal-numerics/results.txt`. A second run gave identical numeric results.

### Contraction

838,389 of 2^22 uniform inputs in [-1, 1) round differently as `fma(a, b, c)` and as `a*b + c`. On those inputs:

| Kernel | `safe` | `safe_contract` | `fast` |
| --- | --- | --- | --- |
| `a*b + c` | all unfused | all fused | all fused |
| `fma(a, b, c)` | all fused | all fused | all fused |
| `a*b + c` after `#pragma METAL fp contract(off)` | all unfused | all unfused | all unfused |

The pragma compiles and works in every mode. In the fast build it removes only `contract`. The operations keep
`reassoc nnan ninf nsz arcp afn` (textual AIR).

### Division, square root, transcendentals

| Operation (inputs) | `safe` and `safe_contract` | `fast` |
| --- | --- | --- |
| `x / y` (2^24 pairs, magnitudes 2^-20 to 2^21) | correctly rounded, 0 differ | 28.4% differ, up to 2 ulp |
| `precise::divide(x, y)` | correctly rounded | correctly rounded |
| `1.0f / x` | correctly rounded | 10.4% differ, up to 1 ulp |
| `amax / 127` (magnitudes 2^-12 to 2^9) | correctly rounded | 4.7% differ, up to 1 ulp |
| `sqrt(x)` (2^22, magnitudes 2^-40 to 2^41) | correctly rounded | 28.3% differ, up to 2 ulp |
| `rsqrt(x)` against `1/sqrt` in double | correctly rounded, 0 differ | 4.3% differ, up to 1 ulp |
| `exp(x)`, x in [-87, 88], against double | up to 1 ulp (38% not correctly rounded). `precise::exp` gives the same bits. | up to 64 ulp (near `FLT_MIN`); `precise::exp` up to 1 ulp |
| `log(1 + x)`, x in (-0.999, 100) | up to 65,535 ulp, for small x | the same |
| `log1p_k(x)` = `u == 1 ? x : log(u) * (x / (u - 1))`, u = 1 + x | up to 3 ulp | up to 32,767 ulp. Fast math folds `u - 1` to `x`, so it becomes `log(1 + x)`. |
| softplus `v > 20 ? v : log1p_k(exp(v))`, v in [-30, 30] | up to 5 ulp against double and against the CPU's `log1pf(expf(v))` | up to 2× wrong (1.19e-7 for 5.96e-8 at v = -16.6) |
| sigmoid `1 / (1 + exp(-x))`, x in [-30, 30] | up to 4 ulp | up to 29 ulp |
| Apple libm on the CPU, for scale | `expf` and `log1pf`: up to 1 ulp | |

MSL has no `log1p`. The compiler stops with `use of undeclared identifier 'log1p'`.

### Rounding and conversion

The results were the same in all three builds.

| Operation | Result |
| --- | --- |
| `rint` | Half to even. It equals `rintf` on 1,060,595 inputs, which include every tie in [-2000, 2000]. |
| `round` | Half away from zero: `0.49999997` gives 0, `2.5` gives 3. It differs from `roundf` only in the sign of zero, returning +0 for inputs in (-0.5, -0]. |
| `trunc` | Equals `truncf` |
| float to half | Round to nearest even, equal to the CPU on 4,448,275 inputs. The inputs were every midpoint between adjacent finite halves and both float neighbours, of both signs, plus 4M random values from 2^-27 to 2^17. |
| Subnormal halves | Kept: 2^-25 gives 0 (a tie, to even), the next float gives `0x0001`, and `0x387fc000` gives `0x03ff` |
| Overflow | 65,519.996 gives `0x7bff`; 65,520 and above give infinity |
| float32 denormal input | +0 |
| NaN | Every NaN becomes `0x7e00`, losing sign and payload. The CPU keeps both, so 2 of 4 NaN inputs differ. |

### Simdgroups and threadgroups

| Item | Result |
| --- | --- |
| `threadExecutionWidth`, `threads_per_simdgroup` | 32 |
| `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup` | `i % 32` and `(i % 1024) / 32` in 1024-thread groups |
| `simd_ballot` | A 64-bit vote. The low word holds the 32 lanes; the high word is 0. |
| `simd_sum` | The same value in every lane. Over 32,768 simdgroups it was bitwise equal to a pairwise tree `((v0+v1)+(v2+v3))+...` every time, which is the xor butterfly in the order o = 1, 2, 4, 8, 16. It matched CUDA's order (o = 16, 8, 4, 2, 1) in 37% of groups and a sequential sum in 26%. The same held in all three builds. |
| Explicit `simd_shuffle_xor` butterfly, o = 16 down to 1 | Bitwise equal to the same order on the CPU |
| `simd_max` | Exact |
| `maxThreadgroupMemoryLength` | 32,768 B |
| A kernel with 32,768 B of static threadgroup memory | The pipeline is created, with 1024 threads, and the results are correct |
| A kernel with 32,784 B | Compiles. Pipeline creation fails with `Threadgroup memory size (32784) exceeds the maximum threadgroup memory allowed (32768)`. |

### Codebook lookups

Best of six runs, 2^20 threads × 2048 lookups. The "dependent" rows use four chains per thread, where each index
depends on the last value read; those rows measure the lookups. The "independent" rows use xorshift indices. That
loop is ALU-bound: the same loop with no lookup takes 6.07 ms.

| Table | Lookups | Program-scope `constant` array | `constant` buffer | `device` buffer | `threadgroup` copy |
| --- | --- | ---: | ---: | ---: | ---: |
| 16 × int8 (`kvalues_iq4nl`) | dependent | 4.81 ms | 4.86 | 5.01 | 5.06 |
| 16 × int8 | independent | 5.25 | 5.42 | 5.42 | 5.11 |
| 512 × u64, 4 KiB (`iq2xs_grid`) | dependent | 3.52 | 3.58 | 3.68 | 3.88 |
| 512 × u64 | independent | 6.70 | 7.13 | 7.33 | 7.47 |
| 1024 × u64, 8 KiB (`iq2s_grid`, `iq1s_grid_gpu`) | dependent | 6.44 | 7.09 | 7.80 | 8.56 |
| 1024 × u64 | independent | 9.39 | 9.93 | 10.78 | 11.88 |

A second run agreed within 2% for the 16 B and 4 KiB tables and within 9% for the 8 KiB table, with the same
ranking. Constant memory has no penalty for divergent indices, unlike CUDA's 2.12× (`s_gemv.cu`). A
program-scope `constant` array is the fastest placement, or within 3% of it, for every size. A threadgroup copy is
the slowest at 4 and 8 KiB.

### What this changes

| Audit statement | Result |
| --- | --- |
| `#pragma METAL fp contract(off)`: "availability unverified" (B.2.4, F.2) | Available. It works in safe and fast builds. |
| Division must use `precise::divide` (B.2.4, F.2) | In the safe library, `/` is already correctly rounded. `precise::divide` is needed only in the fast library. |
| `simd_sum` order "unspecified (unverified)" (F.4) | Deterministic. It is the ascending butterfly, not CUDA's descending one. |
| Verify half subnormals (B.2.4) | Correct round to nearest even, subnormals kept. NaN is canonicalised. |
| Codebooks in threadgroup memory, not `constant` (F.6) | Reversed on Apple GPUs: program-scope `constant` is fastest |
| (new) | MSL has no `log1p`. A compensated `log1p` works only in the safe library. |

## A.11 Binding and ordering (`spikes/metal-binding/`)

Measured on 2026-10-03 on the machine of section A.1. All buffers are Shared and Untracked, as in the engine.

| Test | Result |
| --- | --- |
| Weights read through a table of `uint64` GPU addresses (`gpuAddress + offset`, 32-B-aligned offsets), the weight buffer only in a residency set attached to the queue | Correct (4,096 entries; float, half and uint loads) |
| `setBuffer:offset:` at a 32-B-aligned offset (1184) | Correct |
| 64 dependent dispatches in one serial-dispatch encoder | Bitwise equal to the CPU |
| The same in a concurrent encoder with `memoryBarrierWithScope:MTLBarrierScopeBuffers` after each dispatch | Bitwise equal to the CPU |
| A concurrent encoder with no barrier, 1,000 small dependent dispatches | Wrong in one run of three |
| Command buffer 2 reads what command buffer 1 wrote, same queue, no fence | **Wrong in 20 of 20 runs** |
| The same with an `MTLFence` (`updateFence` at the end of 1, `waitForFence` at the start of 2) | Correct in 20 of 20 runs |
| 1,000 small dependent dispatches (64 threadgroups of 256 each) | serial 0.87 µs each; concurrent with a barrier after each 1.11-1.14 µs each; concurrent without barriers 0.40-0.54 µs each (wrong result) |
| 500 steps of two independent dispatches | serial 0.85 ms; concurrent with one barrier per pair 0.76-0.78 ms |
| 200 dependent dispatches of 1M floats | serial 3.25-3.46 ms; concurrent with barriers 3.29-3.30 ms |

What this shows:

- Expert kernels can read their weights through the address table of [05](05-memory-and-residency.md), section 5.3,
  in both modes. The residency set alone makes the buffers available; no per-encoder `useResource` is needed.
- **Untracked buffers are not ordered between command buffers**, even on one queue. Every command buffer that reads
  what an earlier one wrote must wait on a fence or an event.
- A serial-dispatch encoder orders its dispatches, as `MTLCommandBuffer.h` documents, and is the cheapest correct
  policy for a dependent chain: a barrier after each dispatch of a concurrent encoder costs about 0.25 µs more per
  dispatch. Concurrency pays only where dispatches are independent, and then saves about 10% of the dispatch
  overhead. [04-metal-kernels.md](04-metal-kernels.md), section 8, turns this into the encoding rules.

## A.12 Host contraction (`spikes/host-contraction/`)

Measured on 2026-10-03 with Apple clang 21.0.0 (`clang-2100.3.34.2`), `-O2`, arm64:

| Source | Default flags | `-ffp-contract=off` |
| --- | --- | --- |
| `double f(double a, double b, double c) { return a*b + c; }` | `fmadd d0, d0, d1, d2` | `fmul` then `fadd` |
| `float t = a*b; return t + c;` (two statements) | `fmul` then `fadd` | `fmul` then `fadd` |
| `f(a, a, c)` in float with `a = 1 + 2^-12`, `c = -(1 + 2^-11)` | `0x1p-24` | `0x0p+0` |

Apple clang fuses a multiply and an add written in one expression unless told not to. Upstream's host code is built
for baseline x86-64, which has no FMA, so the Mac's host code and test oracles are compiled with `-ffp-contract=off`
([08](08-build-and-distribution.md), sections 4.3 and 6.5). The last row is the probe that `metal_manifest_test` uses.
