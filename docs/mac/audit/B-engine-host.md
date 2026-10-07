> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Strata engine, host side: inventory for a macOS / Apple Silicon (Metal, unified memory) port

Upstream tree: `/tmp/strata-upstream` (read-only, engine version 0.1.34, `CMakeLists.txt:11`). Every `file:line` below is relative to that root unless an absolute path is given. This report covers the host side: drivers, protocol, artifacts, memory, expert tiers, speculation, caching, build, platform sites and telemetry. GPU kernel math is out of scope; kernels are named only where the host depends on their contract.

## 0. Conventions, scope and the model in one paragraph

### 0.1 Tags used throughout

| Tag | Meaning |
|---|---|
| **[PORT]** | Portable C++/POSIX. Compiles and behaves correctly on macOS arm64 as is. |
| **[BLOCK]** | Fails to compile on macOS arm64 (x86 intrinsic, missing Linux API, CUDA header). |
| **[SILENT]** | Compiles on macOS but returns wrong or degraded results (zeros, disabled features, wrong page size). |
| **[UMA-NOP]** | Exists only to move bytes between host RAM and a discrete GPU. Unnecessary on unified memory; delete or turn into a no-op. |
| **[DGPU]** | A policy or tuning knob that matters only with a discrete GPU or several GPUs (VRAM sizing, PCIe share, layer split). |
| **[WIN] / [LINUX]** | OS-specific branch. |
| **[DIAG]** | Diagnostic or A/B switch only; it is not on the production path. |

### 0.2 Model geometry the host code hard-codes or reads

These values come from the GGUF, from `expert.hpp:34-51` and from the comments in `session.hpp` and `verify.hpp`.

- **Embedding and layers.**
  - n_embd 2560; 48 layers.
  - QSA (sparse attention) layers are every 4th: 3, 7, ..., 47, which makes 12 QSA layers and 36 GDN (gated delta-net) layers.
- **GDN.** ssm 128, 16 k-heads, 48 v-heads, conv 4, channels 10240.
- **QSA.**
  - 24 heads, 2 KV heads, head_dim 256.
  - Indexer: 4 q-heads, key dim 128.
- **Hyper-connections.** hc 4 streams, hc_lr 320.
- **MoE.**
  - 512 routed experts (256 for the Coder pack), n_ff 640, top-K = 10, plus one shared expert.
  - Canonical Q2_0 blob: 1,382,400 B. 48 x 512 blobs = 33,973,862,400 B.
- **Vocabulary and special tokens.**
  - n_vocab 248,320.
  - `<|im_start|>` = 248045. EOS = 248044 and 248046. `<|image_pad|>` = 248056.
- **PLE (per-layer n-gram embedding) table.** IQ4_NL, 320,001,536 rows x 90 B = 28.8 GB, read from SSD per token (`ngram.cpp`, `ple_reader.cpp`).
- **MTP draft layer.** 1 layer, with its own 512 experts held entirely in VRAM (708 MB, `mtp.hpp:1-19`).

### 0.3 Host source map (line counts)

| Area | Files (lines) | Role |
|---|---|---|
| Driver | `src/program/generate.cpp` (6487) | The `strata` binary. It handles option parsing, loading, one-shot mode and `--serve`. |
| Core runtime | `src/core/layer.cpp` (1321), `session.cpp` (970), `verify.cpp` (1367), `expert_source.cpp` (2299), `mtp.cpp` (840), `remote_experts.cpp` (328), `expert_cache.cpp` (374), `weights.cpp` (479), `graph.cpp` (146), `layout.cpp` (173), `native_head.cpp` (198), `native_dense.cpp` (195), `pinned.cu` (549), `device.cu` (243), `conversation_snapshot.cpp` (268), `conversation_state.cpp` (305), `conversation_memory.cpp` (51) | Layers, token graph, verify windows, expert tiers, MTP, VRAM caches, weight loading, snapshots. |
| Small diagnostic programs | `src/core/{device,load,concurrent,overlap}_main.cpp` | `strata-device`, `strata-load`, `strata-concurrent`, `strata-overlap`. |
| CPU kernels | `src/kernels/cpu/expert.cpp` (683), `pool.cpp` (753), `expert_layout.cpp` (240), `native_expert.cpp` (152), `q2_avx2.cpp` (130), `iq_avx2.cpp` (440), `iq_avx512.cpp` (273), `kq_avx2.cpp` (265), tests | The CPU expert pool and the x86 SIMD expert kernels. |
| Spec | `src/spec/{suffix_drafter,draft_policy,controller}.cpp` plus tests | The drafter policy and the lookup drafter. No GPU code. |
| PLE I/O | `src/ngram/ple_reader.cpp` (+ test), `src/kernels/ngram.cpp` | Direct or mmap SSD reads of the n-gram table. |
| Platform | `src/platform/memory.cpp` (138), `direct_file.cpp` (365) | Page locking and unbuffered asynchronous file I/O. |
| Artifact | `include/strata/artifact/gguf_reader.hpp`, `gguf_split.hpp`, `src/artifact/{gguf_reader,dequant}.cpp` | Header-only GGUF v3 reader, with the `strata-gguf` and `strata-dequant` CLIs. |
| Plan | `src/plan/plan_main.cpp`, `include/strata/plan/*` | The `strata-plan` VRAM plan calculator. |
| Public headers | `include/strata/core/*` (24), `include/strata/platform/{memory,direct_file}.hpp`, `include/strata/program/{conv_cache,logits_selection}.hpp`, `include/strata/spec/*.hpp` | |

### 0.4 Decode architecture in brief (what the host does per token)

1. The driver stages the token into mapped memory and launches one captured CUDA graph covering all 48 layers (the *token graph*, `session.hpp:385-408`; `session.cpp:903-960`).
2. Inside the graph, each layer's router writes its top-10 expert ids, weights and the activation into mapped host memory, then rings a *doorbell*. A device kernel then **spins on a host-written flag** (`doorbell_wait`).
3. The host thread spins on the doorbell sequence number. It runs the CPU expert pool for the experts that are not resident in VRAM ("misses"), publishes the result rows, issues a store fence, and writes the flag. The GPU then copies the miss results (`parts <- y_miss`, H2D) and continues.
4. The experts that are resident in VRAM ("hits") are computed on the GPU in parallel with the CPU pool.
5. With `--spec N` the same scheme runs T tokens at a time in a *verify window* (`verify.cpp`). The window adds a second handshake that publishes a GPU plan (`flagA`) and a PCIe-DMA landing flag (`flagB`).

On Apple Silicon, steps 2-3 are the main design problem (see section 4.10). Metal has no forward-progress guarantee for a shader spinning on CPU-written memory. In addition, on unified memory the GPU can read every expert directly, so the CPU/GPU split that motivates the whole design largely disappears.

---

## 1. Engine binaries, flags, environment variables and the serve protocol

### 1.1 Build targets that are executables

| Target | Source | Built when (`CMakeLists.txt`) | Purpose | macOS today |
|---|---|---|---|---|
| `strata` | `src/program/generate.cpp` | `:406`, inside the GPU block `:181-736` (CUDA or HIP) | The engine: one-shot generation and `--serve`. | [BLOCK] It is CUDA throughout. |
| `strata-gguf` | `src/artifact/gguf_reader.cpp` | `:151`, always | `gguf_reader <file.gguf> [--check]`: dumps the header, KV and tensor directory. | [PORT] Builds. |
| `strata-dequant` | `src/artifact/dequant.cpp` | `:154`, always | `dequant <file.gguf>`: dequantizes and checks tensors (prints PASS/FAIL). | [PORT] |
| `strata-plan` | `src/plan/plan_main.cpp` | `:176`, always | `strata-plan [--max-context N] [--vram BYTES] [--dense B] [--embd B]`: the VRAM memory plan. | [PORT], but [DGPU] in meaning. |
| `strata-device` | `src/core/device_main.cpp` | `:194`, GPU block | `[--selftest] [--list-devices]`: GPU report plus an arena exercise. Also a ctest at `:704/706`. | [BLOCK] (CUDA) |
| `strata-load` | `src/core/load_main.cpp` | `:197`, GPU block | `--file experts.bin [--layers N] [--threads N] [--chunk-mb N]`: arena load benchmark. | [BLOCK] |
| `strata-concurrent` | `src/core/concurrent_main.cpp` | `:200` | Measures whether the CPU expert path and the PCIe stream overlap. | [BLOCK], [DGPU] |
| `strata-overlap` | `src/core/overlap_main.cpp` | `:204` | Measures whether an expert stream overlaps compute. | [BLOCK], [DGPU] |
| `tune_hipblaslt` | `tools/hip/tune_hipblaslt.cpp` | `:402`, HIP only | hipBLASLt tuning. | n/a |
| `strata-vision` | `tools/vision/strata_vision.cpp` | separate project `tools/vision/CMakeLists.txt` | Vision encoder helper based on llama.cpp mtmd. | [PORT] Builds and runs on macOS arm64; verified here against the pinned llama.cpp `3cf03257f`. `--gpu` sets mtmd `use_gpu`, which selects Metal on macOS. Without `--gpu` it runs on the CPU with `hardware_concurrency()/2` threads (`:101`). That divisor assumes SMT, so on Apple Silicon it uses only half the cores. |

### 1.2 `strata` command line (complete)

Defaults live in `struct Options` (`generate.cpp:160-415`). Parsing is at `generate.cpp:999-1255`.

Error handling:
- An unknown argument prints "unknown argument: X" plus usage and exits **2** (`:1248-1252`).
- A missing value prints "X needs a value" and exits 2.
- Validation failures after parsing also exit 2.

The parse line of each flag is given in parentheses.

**Model and pack**

| Flag | Default | Notes |
|---|---|---|
| `--pack DIR` (1007) | `pack/full` | Pack directory: `index.txt`, `dense.bin`, optional `experts.bin`, `native_experts.txt`. |
| `--native SHARD1` (1080) | off | The native-pack preset. It reads quantized dense tensors and experts from the GGUF and turns on every `--native-*` arm. |
| `--native-bf16` (1068), `--native-bf16-extra` (1069), `--native-ple-key` (1070), `--native-moe-combine` (1071), `--native-gdn` (1072), `--native-flash-attn-short` (1073), `--native-qsa-indexer` (1074), `--native-qsa` (1075), `--native-rope` (1076), `--native-ple-postops` (1077), `--native-router` (1078) | off | Individual native arms. [DIAG] when used separately. |
| `--native-head-gguf F` (1081), `--embd-gguf F` (1082), `--native-dense-gguf F` (1083, repeatable) | - | Alternate sources for the head, the embedding (the `strata-embd` BF16 GGUF) and dense tensors. |
| `--ple-gguf F` (1055), `--no-ple` (1056) | - | PLE table shard. `--no-ple` is [DIAG]. |
| `--ple-io direct\|mmap\|ram` (1057) | `direct` | `ram` is refused on Windows (`:1463`) because there is no mlock there. On macOS `ram` means mlock and needs RLIMIT_MEMLOCK. |
| `--ple-row-cache N` (1058) | `1<<20` rows | |
| `--ple-inflight N` (1059) | 64 | Allowed range 1..1024 (`:1456`). |
| `--ple-delay-us`, `--ple-sync-submit`, `--no-ple-prefetch` (1060, 1061, 1102) | - | [DIAG] |
| `--cpu-oracle-q8-0`, `--gr-fp32-activations`, `--gr-native-mmvf` (1079, 1066, 1067) | off | [DIAG] numerics arms. |

**Context, KV and RoPE**

| Flag | Default | Notes |
|---|---|---|
| `--max-context N` (1027) | 4096 | Setup writes the user's choice. |
| `--kv fp16\|int8\|q4_0\|k8v4` (1062) | fp16 | `q4` is an alias for `q4_0` (`:1467`). Anything else exits 2 (validated just below). |
| `--kv-resident N` (1063) | 0 | KV cells kept in VRAM; the rest are streamed (minimum 20480). [DGPU] |
| `--rope-scaling none\|linear\|yarn` (1028), `--rope-scale` (1029), `--rope-freq-base` (1030), `--rope-freq-scale` (1031), `--yarn-orig-ctx` (1032), `--yarn-ext-factor` (1033, default -1), `--yarn-attn-factor` (1034, default 1), `--yarn-beta-fast` (1035, default 32), `--yarn-beta-slow` (1036, default 1) | none | |

**Sampling (one-shot mode; serve takes these per request)**

| Flag | Default |
|---|---|
| `--max-new N` (1026) | 16 |
| `--greedy` (1037) | true |
| `--seed S` (1038) | Setting it turns sampling on. |
| `--top-k` (1039) | 20 |
| `--top-p` (1040) | 0.95 |
| `--temperature` (1041) | 1.0 |
| `--stop-eos` (1141), `--eos-ids a,b` (1203) | `{248044, 248046}`. `--eos-ids` implies `--stop-eos`. |
| `--tokens a,b,...` (1008) / `--tokens-file F` (1014) | Only one of the two is allowed. In serve mode the dummy prompt is `{248045}` (`:1443-1448`). |

**Experts, VRAM tier and CPU pool**

| Flag | Default | Notes |
|---|---|---|
| `--expert-cache N\|auto` (1109) | 0 (`auto` = -1) | VRAM expert slots. `auto` = (free VRAM - reserve - prefill buffers - MTP head) / max_blob, capped by the profile size (`:2664-2687`). [DGPU] |
| `--expert-cache-device1..3 N` (1113-1115), `--expert-cache-remote-placement stripe\|layer` (1116) | 0 / stripe | Helper GPUs that only hold experts (RemoteExperts). [DGPU] |
| `--expert-cache-per-layer` (1210) | off | Per-layer slot quotas (`expert_cache.hpp:109`). |
| `--expert-cache-cpu-order` (1209) | off | Experimental CPU-order GPU arithmetic. [DIAG] |
| `--expert-profile F` (1212) | - | STRP ranking that fills the VRAM tier (section 2). |
| `--no-hit-poke` (1211) | off | [DIAG] |
| `--vram-reserve-mib N` (1118) | 700 | [DGPU] |
| `--pool-workers N` (1090) | 0 = auto | Auto is P-cores - 1 on hybrid CPUs, otherwise one per physical core (`pool.cpp:345-353`). |
| `--pool-affinity all\|auto\|p-cores` (1091) | all | `PoolAffinity` (`pool.hpp:68`). |
| `--no-host-worker` (1101) | off | When set, the host thread does not join the pool. |
| `--mmap-experts` (1214) | off | Experts are mmapped from the pack or GGUF instead of loaded into a pinned arena. |
| `--shared-expert-arena F` (1215) | - | One arena file shared by several engine processes via MAP_SHARED with a pack-hash header (`pinned.cu:175-197`). [LINUX] |
| `--resident-cpu-experts` (1216) | off | |
| `--resident-experts` (1217) | off | Implies mmap + resident CPU + pin + soft + 4 GiB headroom (`:1217-1226`). Overridden by `STRATA_RESIDENT_PIN=0` (1222) and `STRATA_RESIDENT_HEADROOM_GIB` (1224). |
| `--resident-budget-gib G` (1227) | - | Must be > 0. Same overrides (1233, 1235). The default headroom is 8 GiB. |
| `--pcie-frac F` (1138) | -1 = auto | Auto is 0.55 (DMA) for native packs, scaled by the measured H2D bandwidth, and 0.2 (direct) for the Q2_0 pack (`:1720-1734`). [DGPU] |
| `--pcie-mode auto\|dma\|kernel\|direct` (1146) | auto | Mapped to 0 dma, 1 direct, 2 kernel (`:4215, 4227, 6015`). [DGPU] |
| `--adapt-every N` (1139) | 4 | Rounds between adaptive VRAM-tier swaps. |
| `--adapt-swaps N` (1208) | 96 | Maximum swaps per round. The exchange buffers are capped at 96 (`:3786`). |

**Prompt path (prefill)**

| Flag | Default | Notes |
|---|---|---|
| `--prefill CHUNK\|auto\|auto:16384\|auto:32768` (1119) | - | Batched prompt path. `auto` takes the largest chunk the expert cache can lend buffers for, up to 8192 (`STRATA_PREFILL_AUTO_MAX` 1123). |
| `--no-split-rows` (1129), `--no-prefill-borrow` (1130), `--prefill-until N` (1131), `--dump-final-r F` (1132) | - | `--no-prefill-borrow` disables lending VRAM-tier slots to the prompt path. [DGPU] |

**Speculation**

| Flag | Default | Notes |
|---|---|---|
| `--spec N` (1133) | 0 | Verify window size S. Setup uses 4. |
| `--mtp DIR` (1136) | - | `mtp/rt` from `tools/mtp_rt.py`. |
| `--mtp-window N` (1137) | 32768 | |
| `--mtp-max-t N` (1169) | 0 | Caps the MTP window (S_mtp = min(mtp_max_t, S) when > 0). |
| `--spec-min-p P` (1140) | 0 | Grows T while each draft probability is >= P. Setup uses 0.5. |
| `--suffix-draft N` (1168) | 3 | Lookup-drafter minimum match. |
| `--spec-split` / `--no-spec-split` (1142, 1202) | off | Row-group split of the verify window. |
| `--coupled-draft` / `--no-coupled-draft` (1103, 1104) | env `STRATA_SPEC_COUPLED` | Coupled draft sampling. |
| `--spec-oracle F` (1134), `--spec-corrupt N` (1135) | - | [DIAG] |

**Multi-GPU**

| Flag | Notes |
|---|---|
| `--layer-split K\|K1,K2\|auto` (1143) | Serve only. Later stages run layers [K_i, K_i+1) on other GPUs (`:1263-1300`). [DGPU] |
| `--split-device a,b` (1144) | Default: the next visible devices. |
| `--split-skip-if-fits` (1145) | |

`--layer-split` together with conversation parking exits 2 (`:1260-1262`).

**Serve and conversation cache**

| Flag | Default | Notes |
|---|---|---|
| `--serve` (1147) | off | Line protocol on stdin/stdout (section 1.4). |
| `--vision` (1148) | off | Enables `GENI` requests (the image path). |
| `--prompt-cache N` (1149) | 6 | Number of checkpoints (each ~118 MB of host RAM). |
| `--prompt-cache-every N` (1164) | 16384 | A checkpoint every N freshly read tokens. |
| `--prompt-cache-root N` (1165) | 2048 | Root checkpoint at the first turn boundary when that is at least N tokens in. |
| `--turn-token ID` (1166) | 248045 | |
| `--short-read N` (1167) | 64 | Text parts of at most N tokens go through verify windows instead of the batched path. |
| `--conversation-cache-mib N` (1160) | 0 | Opt-in parking of whole conversations in host RAM. |
| `--conversation-cache-slots N` (1155) | 4 | |
| `--conversation-cache-min-free-mib N` (1161) | 2560 | |

**Control vectors** (`generate.cpp:876-945`)

| Flag | Notes |
|---|---|
| `--control-vector F` (1170), `--control-vector-scaled F:S[,F:S]` (1171; split at the last colon) | GGUF with arch `controlvector`, tensors `direction.<l>` F32[2560] for 1 <= l < 48. |
| `--control-vector-layer-range A B` (1186) | Setup uses 4 44. |
| `--cvec-mode add\|project` (1190) | Default add. `project` normalizes to a unit vector and uses the norm as the scale. |
| `--cvec-dir per-layer\|single:L` (1196) | |

**Diagnostics and A/B arms** [DIAG]

- Dumps:
  - `--dump-logits F` (1042) with `--logits-stride` (1043), `--dump-residual` (1050), `--dump-mixed` (1051), `--dump-layers` (1052), `--dump-halves` (1053).
  - `--dump-routing F` (1054). Its record is `int32 layer, int32 k, k x int32 ids, k x f32 weights` (`drive_pool`, `:610-658`). This is the input of `tools/make_profile.py`.
- Run-mode switches: `--no-capture` (1084), `--no-pool` (1085), `--sync-every-layer` (1086), `--stage-timing` (1087), `--graph-only` (1088), `--gpu-only-full` (1089), `--gpu-stages` (1213), `--stats` (1238).
- Kernel A/B arms: `--shared-late` (1239), `--keep-canonical` (1240), `--no-token-graph` (1241), `--no-fused-gr` (1242), `--no-fast-attn` (1243), `--no-publish-kernel` (1244), `--no-fused-gdn` (1245), `--no-fast-select` (1246), `--stream-token` (1064), `--check-logits` (1065).

**Process setup in `main`**

- stdout is set unbuffered (`:985`).
- `CUDA_MODULE_LOADING=EAGER` is forced into the environment (`:989-995`). [UMA-NOP]; on Metal, build pipelines at startup instead.

**One-shot output.** stdout carries `prompt  : ids`, `output  : ids`, `decode N tokens in X ms -> Y tok/s` and `prefill ...` (`:6331-6342`), plus the `%-24s` stats block (section 9).

### 1.3 Every `STRATA_*` environment variable (host and engine)

Category legend: **P** = production tuning, **D** = diagnostic or A/B, **T** = test hook.

**Driver (`src/program/generate.cpp`)**

| Variable | Line | Cat | Effect | Mac |
|---|---|---|---|---|
| STRATA_REFILL_BLOCKING | 116 | D | Refill lent slots with blocking copies. | [UMA-NOP] |
| STRATA_TRACE | 811, 4571 (also prefill.cpp:1052) | D | `strata trace: %s %lld %lld` progress lines and `mem_mark` VRAM-free prints. | keep |
| STRATA_PREFILL_AUTO_MAX | 1123 | P | Largest auto prefill chunk (8192). | keep (rename to "chunk") |
| STRATA_RESIDENT_PIN | 1222, 1233 | P | `=0` keeps resident experts unpinned. | [UMA-NOP] |
| STRATA_RESIDENT_HEADROOM_GIB | 1224, 1235 | P | RAM headroom for the resident mode. | keep (RAM budget) |
| STRATA_SPLIT_OWN | 1370 | D | Layer-split ownership. | [DGPU] |
| STRATA_KV_ROT | 1476 | D | Hadamard rotation for int8 KV. | kernel |
| STRATA_EARLY_REMOTE_CONTEXTS | 1703 | D | RemoteExperts context order. | [DGPU] |
| STRATA_NO_IQ256 | 1742 (also native_expert.cpp:87) | D | | kernel |
| STRATA_SSD_KEEPALIVE, STRATA_SSD_KEEPALIVE_WINDOW | 2038, 2039 | P | Keep the SSD awake (periodic reads). | possibly useful |
| STRATA_SPLIT_OWN_PLACE | 2306 | D | | [DGPU] |
| STRATA_SPLIT_MISS_MS | 2338 | D | | [DGPU] |
| STRATA_FETCH_THREADS | 2578 | P | File-tier fetch threads (default 8, `expert_source.hpp:527`). | keep |
| STRATA_TEST_CACHE_FAIL | 2755 | T | Injects an expert-cache allocation failure. | |
| STRATA_LOOKAHEAD, STRATA_LOOKAHEAD_K | 3029, 3040 | P | RouterLookahead file prefetch. | keep if SSD tier |
| STRATA_PREFILL_LEND_PCT | 3744 | P | Share of VRAM slots lent to the prompt path. | [DGPU] |
| STRATA_SPLIT_RING | 3870 | D | | [DGPU] |
| STRATA_SPLIT_OWN_BUFFERS | 3927 | D | | [DGPU] |
| STRATA_SPLIT_SMALL_OWN, STRATA_SPLIT_SMALL_MAX | 4094, 4096 | D | | [DGPU] |
| STRATA_SNAPSHOT_VERIFY | 4370, 4912 | T | Verify each snapshot after save and restore. | keep |
| STRATA_SNAPSHOT_FULL_CAPTURE | 4930 | D | | keep |
| STRATA_WATCHDOG_S | 4635 | P | Watchdog limit (default 60 s, 0 = off). | keep |
| STRATA_CKPT_REREAD | 4964, 5019 | D | Re-read instead of restoring a checkpoint. | keep |
| STRATA_LOGPOS, STRATA_LOGPOS_EXTRA | 5056, 5060 | D | Log positions. | |
| STRATA_REFILL_SERIAL | 5111 | D | | [UMA-NOP] |
| STRATA_DECODE_TIMING | 5305 | D | Per-request `strata decode timing:` line. | keep |
| STRATA_STATE_HASH, STRATA_STATE_HASH_GDN | 5455, 5497 (also prefill.cpp:1960) | T | State fingerprints. | keep |
| STRATA_SPLIT_TIMING | 5605 | D | | [DGPU] |
| STRATA_DUMP_FIRST_LOGITS | 6183 | D | | |

**Verify windows (`src/core/verify.cpp`)**

| Variable | Line | Cat | Effect | Mac |
|---|---|---|---|---|
| STRATA_VERIFY_DEBUG | 58 | D | | |
| STRATA_TEST_VERIFY_STALL | 131 | T | Simulates a stall to test the watchdog. | keep |
| STRATA_VERIFY_PROFILE | 331 | D | | |
| STRATA_VERIFY_DEVICE_PLAN | 358 | D | Builds the plan on the device. | |
| STRATA_DEC_BATCH | 404 (also shared_expert.cu:182) | D | | |
| STRATA_VERIFY_NODES | 907 | D | Prints the graph node count. | |
| STRATA_DBG_NAN | 1183 | D | | |
| STRATA_LOGPOS_TOPK | 1300 | D | | |
| STRATA_COMMIT_SYNC | 1318 | D | Synchronous commit. | |

**Other core files**

| Variable | Location | Cat | Effect | Mac |
|---|---|---|---|---|
| STRATA_TG_FLUSH_US | session.cpp:915 | P | WDDM flush interval for the token graph (default 2000 us). | [UMA-NOP] |
| STRATA_MTP_PREFILL_SYNC | mtp.cpp:692 | D | | |
| STRATA_KV_RING_OFF, STRATA_KV_MAIN_OFF | layer.cpp:502, 503 | D | KV streaming ring. | [DGPU] |
| STRATA_FETCH_PVM | expert_source.cpp:783 | D | PrefetchVirtualMemory. | [WIN] |
| STRATA_PARTIAL_PIN, STRATA_PARTIAL_PIN_GIB | expert_source.cpp:1173, 1179 | P | Partial cudaHostRegister of the resident set. | [UMA-NOP] |
| STRATA_POOL_TRACE | expert_source.cpp:1611 | D | `pool trace: layer L what n`. | keep |
| STRATA_REMOTE_SPIN | remote_experts.cpp:77 | D | | [DGPU] |
| STRATA_REMOTE_ZEROCOPY | remote_experts.cpp:202 | D | | [DGPU] |
| STRATA_ARENA_PIN_GIB | pinned.cu:242 | P | Cap on cudaHostRegister. `auto` uses the DXGI budget (#243). | [UMA-NOP] |
| STRATA_ARENA_LOCK | pinned.cu:356, 382 | P | VirtualLock/mlock the arena. | optional |
| STRATA_NO_LARGEPAGES | pinned.cu:88, 105 | D | | [WIN] |
| STRATA_IO_THREADS | platform/direct_file.cpp:63 | P | DirectFile worker threads. | keep |

**CPU kernels (`src/kernels/cpu`)**

| Variable | Location | Cat | Effect | Mac |
|---|---|---|---|---|
| STRATA_CPU_YMM | expert.cpp:153 | D | | x86 |
| STRATA_FORCE_AVX2 | expert_layout.cpp:26 | D | | x86 |
| STRATA_IQ_GATHER | iq_avx512.cpp:33 | D | | x86 |
| STRATA_IQ_PREFETCH | iq_avx2.cpp:88, iq_avx512.cpp:37 | D | | x86 |
| STRATA_NO_IQ512 | native_expert.cpp:86 | D | | x86 |
| STRATA_IQ_MT_MIN | native_expert.cpp:92, 130 | P | | keep for a NEON pool |
| STRATA_KQ256 | native_expert.cpp:96, 131 | D | | |
| STRATA_NO_IQ4NL | native_expert.cpp:129 | D | | |
| STRATA_POOL_SPIN_US | pool.cpp:347 | T | Spin time before a worker sleeps (default 20 ms). | keep |

**Headers**

| Variable | Location | Cat | Effect |
|---|---|---|---|
| STRATA_SPEC_COUPLED | coupled_draft.hpp:54 | P | Coupled draft sampling. |
| STRATA_EMULATE_CC | emulate.hpp:17 | D | Emulates a compute capability. [DGPU] |

**GPU-side tuning variables (kernel scope; listed for completeness)**

- prefill.cpp: GR_UNFUSED 125, GROUP_COPY 569, HIP_WMMA 97, IDX_FP16_CHECK 1481, INDEXER_PER_TOKEN 1427, KV_STAGE_OWN 375, MTP_BATCH 740, MTP_BATCH_F16 772, PLE_BATCH 1026, PREFILL_BF16X2 115, PREFILL_DUMP_R 1886, PREFILL_ISSUER 1207, PREFILL_MMQ 476, PREFILL_RING 92, PREFILL_STREAM_MIN 82, PREFILL_TIMING 960, PROMPT_ATTN_OLD 1549, QSA_DUMP 1527, QSA_DUMP_LAST 1528-1529, SELECT_OLD 1448, SEL_OVERLAP 1459, STAGER_RING 188/562, STAGER_THREADS 554.
- prefill/kernels.cu: GDN_CONV_SERIAL 733, GDN_PIPELINE 750, GDN_REC_HEADS 746.
- fused_gr.cu: GR_V3 824, HC_SPLIT 791/1042.
- s2_expert_grouped.cu: GROUPED_PAIR_MIN_HITS 508, OLD_GROUPED 495.
- router_top10.cu: HIP_ROUTER_OLD 349.
- qsa_prompt_attn.cu: PROMPT_ATTN_V1 1024, QSA_WARP 994 (also qsa_select.cu:707).
- native_rope.cu: ROPE_TABLE 97.
- qsa_select.cu: SCORES_MULTI 657, SELECT_WMMA 681, TOPK_OLD 752.
- sampler.cu: OLD_SAMPLER and SAMPLER_ONE_BLOCK 763-764.
- iq_kernels.cu: OLD_IQ_MMVQ 1295.
- HIP tuning: HIPBLASLT_TUNING gemm.cu:144, HIPBLASLT_VERBOSE gemm.cu:99/190/259, HIPBLASLT_TUNING_V1 hipblaslt_tuning.hpp:47.

**Non-STRATA variables**

- `CUDA_MODULE_LOADING` (set at generate.cpp:989-995).
- `CUDA_VISIBLE_DEVICES` (tools/vision/strata_vision.cpp:81-87, which hides CUDA from mtmd when `--gpu` is absent).
- `STRATA_GGUF_PY` and `STRATA_MTP_REVISION` (Python tools).

### 1.4 The serve protocol (engine side `generate.cpp`, server side `serve/server.py`)

**Transport**
- The server starts `Popen([exe, "--serve", *args], text=True, bufsize=1)` (`server.py:246`).
- Requests go to the engine's stdin, one line each. Responses come back on stdout, one line each.
- Human-readable diagnostics go to stderr, which the server writes to the log file.
- The stdin reader is a dedicated thread using `read(2)`, not `std::cin`, because of a glibc exit hang (`generate.cpp:4514-4560`).
  - It strips `\r`.
  - A line equal to `STOP` sets `stop_req` immediately, even during generation. Every other line is queued.
  - EOF sets `in_eof`, and the main loop exits.
- [PORT] on macOS: read(2) and std::thread are POSIX.

**Startup handshake (engine to server)**
1. Load progress and warnings on stderr.
2. stderr: `strata serve: N MiB of VRAM free ...`, or a `LOW ...` variant below 256 MiB that suggests `--vram-reserve-mib` (`:4577-4592`). [DGPU]
3. stdout `INFO` (`:4613-4628`), one line of `key=value` pairs:
   `INFO context=<max_ctx> kv=<fp16|int8|q4_0|k8v4> kv_resident=<cells> expert_slots=<all GPUs> expert_cache_mib=<all> expert_slots_primary=<CUDA0> expert_cache_primary_mib=<CUDA0> spec=<S> mtp_max=<mtp_max_t> lookup=<suffix_draft> vram_free_mib=<n> cvec=<summary> arena_mib=<resident or total expert bytes> pool_workers=<n> pcie_frac=<%.2f> spec_min_p=<%.2f> conversation_cache_mib=<n> conversation_cache_slots=<n> conversation_cache_min_free_mib=<n> engine=<version>`
4. The watchdog thread starts (`:4634-4656`).
5. stdout: `READY <max_context> stop` (`:4658`). The literal `stop` advertises that the engine accepts STOP.
6. Server side: `server.py:250-259` reads lines until READY and parses INFO into a dict.

**Request grammar (server to engine, `:4675-4741`)**
```
GEN  <max_new>[ key=value ...] <id,id,...>
GENI <max_new>[ key=value ...] <embeddings_file> <id,id,...>
STOP
QUIT
```
- `QUIT` ends the loop.
- `GENI` requires `--vision`; otherwise the reply is `ERR this engine was started without --vision` (`:4744`).
- Each request runs under a `BusyScope`, which the watchdog uses, and `stop_req` is cleared at the start.
- Keys are parsed at `:4700-4735`. Unknown keys are skipped silently.

| Key | Type / clamp | Default |
|---|---|---|
| `cvec` | control-vector selector | |
| `temperature` | float | 0 (greedy) |
| `top_p` | float | 1 |
| `top_k` | int, must be 1..64 when sampling | 20 |
| `min_p` | float | 0 |
| `penalty_last_n` | int (coupled history cap 4096) | 0 |
| `penalty_repeat`, `penalty_freq`, `penalty_present` | float | neutral |
| `seed` | uint64 | |
| `pcie_frac` | clamped 0..1 (`:4725`) | engine value. [DGPU] |
| `spec_min_p` | clamped 0..1 | engine value |

**Embeddings file for GENI** (`:4745-4804`)
- The file is a concatenation of `SVE1` records: `int32 {0x31455653, n_tokens, nx, ny, n_embd}` followed by `float32[n_tokens][n_embd]`.
- `n_tokens` must equal `nx*ny`, and `n_embd` must be 2560.
- Each run of `<|image_pad|>` (248056) tokens in the prompt consumes one record in order.
- M-RoPE positions:
  - Text token: (p, p, p).
  - Image row j: (p, p + j/nx, p + j%nx).
  - After an image, p advances by max(nx, ny).
- Each image is fingerprinted (FNV-1a over the grid and the floats) for the conversation cache.
- Errors, each sent as `ERR <message>`:
  - `bad embeddings file (expected strata-vision records of width 2560)`
  - `short embeddings file`
  - `the prompt has more images than the embeddings file`
  - `image k has n rows but fewer <|image_pad|> tokens`
  - `the embeddings file has more images than the prompt`
  - `the prompt cannot end in an image`

**Responses (engine to server, stdout), in order for one request**

| Line | Where | Meaning |
|---|---|---|
| `RESUME <resume>` | `:5009` | Tokens of the prompt reused from the live session, a checkpoint or a parked conversation. |
| `PP <done> <total> <ms %.0f> <fresh tok/s %.1f>` | `:4376` (batched path), `:5071` (short-read path) | Prompt progress. The server reads f[4] as tok/s. |
| `REUSED <resume>` | `:5283` | Printed once the prompt is fully read. |
| `T <id>` | `:5389` | One emitted token per line, in order. |
| `DONE <generated> <prompt_n> <prompt_ms %.1f> <decode_ms %.1f> <finish> <drafts_accepted> <drafts_offered> <reused> <hits> <lookups> <ram_blobs> <file_blobs> <file_MB %.1f>` | `:5569` | `finish` is `stop` (EOS), `cancel` (STOP received) or `length`. |
| `ERR <message>` | various | The request failed. The engine stays alive unless the error is fatal. |

ERR messages (`generate.cpp`):
- `ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> <file> <id,id,...>` (4686)
- `ERR bad request: %s` (4739)
- `ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%lld)` (4836)
- `ERR a token id is outside the vocabulary` (4842)
- restore failures (4909, 4916, 4989)
- lend/refill failures (`ERR refilling a lent slot failed: ...` 5240, 5244, 5278)
- `ERR an adaptive refill failed` (5412)
- state-fingerprint mismatches (5460, 5555)

**stderr after each DONE** (`:5570-5605`)
```
strata serve: prompt N tokens = R reused + F read in X ms (tok/s), G generated in Y ms (tok/s), drafts accepted a of o, K checkpoints[ (cancelled)]
strata serve: decode expert cache hit rate: %.1f%% (h hits / l lookups)
strata serve: resident RAM: %.2f GiB of experts in RAM, N exchanged with the VRAM tier, M blob reads from the file
strata serve: expert tiers: GPU %lld hits this request; since the start RAM %lld blobs, files %lld blobs %.1f MB read ...
```

**Server side (`serve/server.py`)**
- `generate()` (`:403-454`):
  - It builds the head `GEN <max_new><keys>` or `GENI <max_new><keys> <emb>` and appends the comma-joined ids.
  - It keeps a 10 s heartbeat.
  - It handles `T`, `PP` (f[4] = tok/s), `DONE` and `ERR` (raised as ValueError).
  - When the consumer stops early, it sends `STOP` and drains until `DONE`.
- `close()` (`:457-466`): sends `QUIT`, waits 20 s, then terminates, then kills.
- `death_note` looks for "issue #29" in the log to explain a watchdog restart.
- Vision helper (`:495-607`):
  - It spawns `strata-vision --mmproj ... --model ... [--gpu]` and waits for `READY <n_embd>`.
  - It writes images to `<sha256[:32]>.img` and sends `ENC <image path> <out path>`.
  - It expects `OK <n_tokens> <nx> <ny> <ms>` or `ERR <msg>`, and keeps a 64-entry LRU of encodings.
  - It concatenates the SVE1 outputs into `req-<uuid>.sve` (`:1154-1158`) and sends `GENI`.
- `strata-vision` itself (`tools/vision/strata_vision.cpp`):
  - Usage `:7-8`, `:75-76`.
  - It runs a warmup encode at 2048x2048 (`:130`) and prints `READY %d` (`:149`).
  - `ENC` parsing: the image path ends at the last space before the output path (`:37-40`).
  - Replies: `OK %d %d %d %.0f` (`:192`). `QUIT` exits (`:155`).
  - Bad CLI arguments exit 2.

**Example exchange**
```
<- (stderr) strata serve: 9876 MiB of VRAM free ...
<- INFO context=65536 kv=fp16 kv_resident=0 expert_slots=7200 ... engine=0.1.34
<- READY 65536 stop
-> GEN 512 temperature=0.7 top_p=0.95 top_k=20 seed=42 248045,872,198,...
<- RESUME 1830
<- PP 2048 4096 812 2521.3
<- PP 4095 4096 1601 2556.8
<- REUSED 1830
<- T 1234
<- T 99
...
<- DONE 57 4096 1620.1 2200.4 stop 31 48 1830 9120 13680 4560 0 0.0
-> QUIT
```

**Port note.** The protocol is pure text and the engine's process model is portable. A Metal engine that keeps this exact protocol can reuse `serve/server.py` unchanged, apart from the setup and config issues covered in a separate audit.
- Keep `INFO` keys stable. On a Mac, `vram_free_mib` and `pcie_frac` become meaningless but should still be emitted, for example as `vram_free_mib=<recommendedMaxWorkingSetSize - allocated>` and `pcie_frac=0.00`.
- Keep the `stop` capability token on READY.

---

## 2. On-disk artifacts

All formats are little-endian; Apple Silicon is little-endian, so no byte swapping is needed. The producing tools are pure Python + numpy and run on macOS. Each one is described exactly in the packer audit (section 2.4 summarizes the layouts).

### 2.1 Artifact map

| Artifact | Producer | Format | Engine consumer | Platform-specific? | Mac plan |
|---|---|---|---|---|---|
| Model GGUF shards `data/models/<tag>/*.gguf` | HF download (`setup.py:3040-3071`) | GGUF v3. Q2_0 = type 42 (fork-only), 18 B per 64 weights `{fp16 d; u8 qs[16]}`, LSB-first, value = (code - 1)·d | `gguf_reader.hpp:384-458`, `gguf_split.hpp:23-50` | No | Reuse. mmap and wrap with `newBufferWithBytesNoCopy`. |
| `packs/<tag>/manifest.json` | `tools/strata_pack.py build` | JSON | only hashed into the shared-arena fingerprint (`expert_source.cpp:1995`) | Produced only on AVX-512 hosts (`setup.py:3077`) | Not needed. |
| Canonical `experts.bin` (+ `experts.json`) | `tools/pack_layer.py` via strata_pack | 24,576 blobs x 1,382,400 B. Blob = gate/up codes @0 (819,200; rows interleaved g,u), down codes @819,200 (409,600), gu fp16 scales @1,228,800 (102,400), down scales @1,331,200 (51,200) | `expert.hpp:34-51`, `expert_source.hpp:372-402` | Generic bytes, but the only CPU readers are the AVX-512 VNNI / AVX2 kernels | Optional; the split-plane layout suits SIMD-group loads. |
| Canonical `dense.bin`, `embd.bin`, `index.txt` | strata_pack + `pack_index.py` | 64-B-aligned planes; index v3 with 19 fields | `weights.cpp:89-330` | AVX-512 path only | Not needed. |
| Native `index.txt` v3, `dense.bin`, `extra.bin` | `tools/iq_pack.py` | BF16/F16/F32 raw, 64-B aligned. Native-served rows: `code_bits 8, dst_bytes 0` | `weights.cpp:64-72, 166-329`; skip set `generate.cpp:1868-1897` | No | **Reuse.** This is what setup builds on every non-AVX-512 host. |
| `native_experts.txt` v3/v4 | iq_pack | text. Per layer `L gu_type d_type offset blob gate_off up_off down_off [shard\|g,u,d]`; offsets absolute in each role's own shard | `expert_layout.cpp:132-238` (`kExpertLayoutVersion = 4`, `expert_layout.hpp:64`) | No | Reuse. |
| Native `experts.bin` + `.src.json` | `iq_pack.py --experts-bin` (low-RAM) | raw GGUF slices | `expert_source.hpp:395-402` | No | Skip; map the GGUF in place ("gguf mode"). |
| `packs/<tag>/tokenizer/*` | `tools/strata_tokenizer.py` | JSON/text | server.py only | No | Reuse. |
| `mtp/tensors/*.bin`, `mtp-manifest.json` | `tools/mtp_fetch.py` | raw BF16 plus JSON | mtp_pack.py | Paths written on Windows contain `\` | Normalize or refetch. |
| `mtp/mtp-q2_0.gguf` | `tools/mtp_pack.py` | GGUF, arch `qwen4exp-mtp` | mtp_rt.py | No | Reuse. |
| `mtp/rt/{experts.bin, dense.bin, dense.txt}` | `tools/mtp_rt.py` | experts.bin = 512 canonical blobs (707,788,800 B). dense.bin: tensors 256-B aligned, F32 norms stored as 1+w, large projections as Q8_0, the rest BF16. dense.txt rows: `short kind rows cols offset bytes` | `mtp.cpp:155-198` | No | Reuse. Metal needs a canonical-blob Q2_0 kernel and a Q8_0 one. |
| `mtp/rt/draft_vocab.bin` | copied from `data/draft_vocab*.bin` (`setup.py:2435-2452`) | raw int32, **not sorted** (en prefix, then additions) | `mtp.cpp:304-311, 389-400`; `sampler.cu:670-674` | No | Reuse. |
| `data/expert-profile.bin` (196,632 B), `expert-profile-coder.bin` (98,328 B) | `tools/make_profile.py` from `--dump-routing` traces | `STRP`: u32 version 1, n_layers, n_expert, slots, n_ranked, then n_ranked x (u16 layer, u16 expert), then the slot table (ignored by the reader) | `expert_cache.cpp:12-68` (the version is ignored) | No | Reuse if any residency tiering remains. |
| `data/experimental-speed-projection/*.gguf` (483,520 B) | shipped | GGUF, arch `controlvector`, 47 x F32[2560] | `generate.cpp:876-930` | No | Reuse. |
| Optional `token-embd-bf16.gguf` | `tools/embd_bf16_pack.py` | GGUF, arch `strata-embd`, BF16 | `native_head.cpp:108-125` | No | Reuse. |
| Optional `ple-fp8.gguf` (51.2 GB) | `tools/ple_fp8_pack.py` | GGUF, arch `strata-ple`, I8 holding FP8 E4M3 plus `strata.ple.scale` | `ngram.cpp:171-240` | No | Reuse. Metal needs an FP8 LUT decode. |
| ISTA shard 2 PLE table | download | IQ4_NL, [160, 320001536], 90 B/row, 28,800,138,240 B | `ngram.cpp:171-240`, `ple_reader.cpp` | No | Reuse (SSD-streamed). |
| `--dump-routing` trace | engine (`generate.cpp:610-658`) | records `<ii>` (layer, k), then k x i32 ids, then k x f32 weights | make_profile.py:37-48 | No | Reuse. |
| `logs/iq_fixture/*` | `tools/iq_fixture.py` | `<3i>` header plus blocks; `.f32` reference | `src/kernels/iq_parity.cpp:28-52` | No | Regenerate; it is a ready Metal parity harness. |
| `strata-<tag>.json` run config | setup.py:3182-3220 | JSON: exe, args, cwd, tokenizer, lib_dirs, port | server.py | **Yes** (CUDA args, lib_dirs, `--pcie-frac` from calibration) | Rewrite for the Mac engine. |
| Shared arena file (`--shared-expert-arena`) | engine (`pinned.cu:175-197`) | header with pack hash, then the arena | engine | Linux | Not needed. |
| Windows minidump `strata-stall-<pid>.dmp` | engine stall report | dbghelp | developer | Windows | No Mac equivalent; see section 9. |

### 2.2 Which artifacts are platform-specific

Truly platform-specific:
- `strata-<tag>.json`, which holds CUDA lib_dirs and calibrated PCIe and pool arguments.
- The shared arena file (Linux).
- Minidumps (Windows).
- The canonical Q2_0 pack. This is not by format but by policy: setup produces it only for AVX-512 VNNI+VBMI hosts (`setup.py:3077`). On a Mac, setup must take the `iq_pack.py` native route even for Q2_0.

Everything else is byte-identical across platforms.

### 2.3 Sizes that drive the Mac memory budget

| Item | Size | Residency on a Mac |
|---|---|---|
| Routed experts, Q2_0 | 34.0 GB | Needs to be in RAM for good speed (unified memory). |
| Routed experts, IQ2_XS | 35.5 GB | same |
| Routed experts, IQ3_XXS | 42.9 GB | same |
| Routed experts, IQ3_S | 50.3 GB | same |
| Routed experts, IQ1_M Coder | 23.4 GB | same |
| Routed experts, UD-Q4_K_XL | 77 GB | same |
| PLE table, IQ4_NL | 28.8 GB | Read from SSD per token (DirectFile), or mmap/ram. |
| PLE table, FP8 | 51.2 GB | same |
| MTP rt | experts 0.71 GB, dense ~0.2 GB | RAM. |
| Conversation checkpoint | ~118 MB each (6 by default) | RAM. |
| KV cache | depends on context and `--kv` | RAM (it is GPU memory on a Mac). |

Model RAM recommendations come from the `MODELS` table at `setup.py:101-121`: Q2_0 48 GB, IQ3_XXS 60 GB, IQ3_S 62 GB, IQ1_M 32 GB. That table is for a PC with a discrete GPU. On a Mac the experts and the GPU share the same pool, so the realistic minimum is about expert bytes + dense (~4-5 GB) + KV + OS headroom.

### 2.4 Format details a Metal loader must honour

1. GGUF data_start = align_up(end of tensor table, `general.alignment` or 32) (`gguf_reader.hpp:453-458`). Tensor offsets are relative to data_start. `newBufferWithBytesNoCopy` needs a page-aligned pointer (16 KiB on Apple Silicon), so wrap the whole mapped file and address tensors by offset inside it. Do not wrap per tensor.
2. Native `index.txt` rows with `code_bits 8, dst_bytes 0` carry only a shape. Their real type comes from the GGUF (`NativeDense::served_names`, `generate.cpp:1868-1883`), so a native-pack engine must open the shards too.
3. `native_experts.txt` v4 shard columns are per role. Never apply shard 1's data_start to another shard.
4. `draft_vocab.bin` is unsorted; build an id-to-subset map (`sampler.cu:670-674`).
5. `mtp_rt` F32 tensors already include +1 (GemmaRMSNorm fold).
6. Canonical `dense.bin` plane alignment is relative to each tensor's start. Absolute 64-B alignment only holds because of this model's dimensions.

---

## 3. Memory management: every OS memory API and its macOS fate

### 3.1 Principle

On a discrete GPU, Strata maintains four tiers:
1. VRAM (`ExpertCache` slots).
2. Pinned or locked host RAM (`PinnedArena`, `cudaHostRegister`, VirtualLock/mlock).
3. Pageable RAM / page cache (mmap of pack or GGUF).
4. SSD (`DirectFile`, `FileExpertSource`).

It also maintains mapped (zero-copy) host buffers for every CPU/GPU handshake.

On Apple Silicon:
- Tiers 1 and 2 merge into one: an `MTLBuffer` with `MTLResourceStorageModeShared` is both "VRAM" and "pinned host RAM".
- Every `cudaHostAlloc(Mapped)` becomes a shared MTLBuffer whose `contents` pointer the CPU uses directly.
- Every `cudaHostRegister` becomes `newBufferWithBytesNoCopy:length:options:deallocator:` over page-aligned memory (`vm_page_size` = 16384; the length rounded up to a page multiple).
- The remaining questions are residency and the GPU working-set limit:
  - `MTLDevice.recommendedMaxWorkingSetSize` and `currentAllocatedSize`.
  - On macOS 15+, `MTLResidencySet` keeps large buffers resident without per-encoder `useResource`.
  - `sysctl iogpu.wired_limit_mb` raises the wired limit (admin only). A port should document this rather than depend on it.

### 3.2 Site-by-site table

| Site | API | Purpose | macOS |
|---|---|---|---|
| `src/core/pinned.cu:31` | `#include <linux/mman.h>` (MAP_HUGE_2MB) | huge-page flag | [BLOCK] if compiled. In a Metal port this file is replaced as a whole. |
| `pinned.cu:60-83` | OpenProcessToken / AdjustTokenPrivileges (SeLockMemoryPrivilege) | enable large pages | [WIN], [UMA-NOP] |
| `pinned.cu:86-110` | `GetLargePageMinimum`, `VirtualAlloc(MEM_LARGE_PAGES)`, fallback `VirtualAlloc` | arena allocation with 2 MB pages (`STRATA_NO_LARGEPAGES` 88/105) | [WIN]. Mac: `mmap(MAP_ANON\|MAP_PRIVATE)` or `vm_allocate`. Superpages (`VM_FLAGS_SUPERPAGE_SIZE_2MB`) are x86-only on macOS and unnecessary with 16 KiB pages and the GPU's own page tables. |
| `pinned.cu:150-197` | `open`/`ftruncate`/`mmap(MAP_SHARED)` with pack-hash header | `--shared-expert-arena`, so several processes share one arena | [LINUX] in CMake tests (`:688-692`). Would work on macOS (POSIX), but is unnecessary: map the GGUF read-only and every process shares the page cache. |
| `pinned.cu:202-210` | `mmap(MAP_ANONYMOUS\|MAP_HUGETLB\|MAP_HUGE_2MB)`, fallback plain mmap | arena with hugetlb pages | [BLOCK] (`MAP_HUGETLB` is missing from the macOS SDK). Mac: plain mmap. |
| `pinned.cu:214-221` | `VirtualFree` / `munmap` | release | [PORT] (munmap) |
| `pinned.cu:242` | `STRATA_ARENA_PIN_GIB` | cap on the registered size | [UMA-NOP] |
| `pinned.cu:319, 339` | `cudaHostRegister(Portable\|Mapped)` (sliced) | make the arena GPU-visible for PCIe reads | [UMA-NOP]. Use `newBufferWithBytesNoCopy`, one buffer per mapping (or per <= `maxBufferLength` slice). |
| `pinned.cu:356, 382` | `STRATA_ARENA_LOCK` calls `platform::lock_resident` | keep the arena from being trimmed | optional on Mac; see memory.cpp below. |
| `pinned.cu:398, 400` | `cudaHostUnregister` | | [UMA-NOP] |
| `src/platform/memory.cpp:33` | `SetProcessWorkingSetSizeEx` | raise the minimum working set before VirtualLock | [WIN] |
| `memory.cpp:43` | `VirtualLock` | lock the region | [WIN] |
| `memory.cpp:60-105` | DXGI `QueryVideoMemoryInfo(NON_LOCAL)` | Windows shared-GPU-memory budget (#243) | [WIN]. Mac analogue: `recommendedMaxWorkingSetSize - currentAllocatedSize`. |
| `memory.cpp:109` | `GlobalMemoryStatusEx` | total RAM | [WIN] |
| `memory.cpp:115, 123` | `mlock` / `munlock` | lock the region | [PORT], but on macOS mlock is bounded by `RLIMIT_MEMLOCK` and the wired-page limit (`vm.global_user_wire_limit`). Wiring 30+ GB will usually fail. Prefer not locking; on UMA a GPU-resident MTLBuffer is wired by the driver while in use. |
| `memory.cpp:133` | `sysconf(_SC_PHYS_PAGES) * _SC_PAGESIZE` | total RAM | [PORT] (present in the SDK). `sysctl hw.memsize` is the canonical source. |
| `src/core/expert_source.cpp:182-217` | cgroup v2 `memory.stat` parser, guarded by `__linux__` | container-aware free RAM | [LINUX]. Compiled out on macOS. |
| `expert_source.cpp:219-283` `available_memory_bytes` | Windows `GlobalMemoryStatusEx`; Linux `/proc/meminfo` MemAvailable plus the tightest cgroup-v2 `memory.max` (226-268); otherwise (`#else`, `:276`) `sysconf(_SC_AVPHYS_PAGES)` | free-RAM probe for the resident budget (`--resident-experts`, `--resident-budget-gib`) | macOS takes the `#else` branch, and `_SC_AVPHYS_PAGES` is **absent from the macOS SDK** (checked): [BLOCK]. Add an `__APPLE__` branch: `host_statistics64(HOST_VM_INFO64)` with (free + inactive + purgeable + speculative) x `vm_page_size`, or `sysctl kern.memorystatus_level` (percent free) x `hw.memsize`. |
| `expert_source.cpp:413, 596` | `CreateFileMappingW` / `MapViewOfFile` | map experts / GGUF | [WIN] |
| `expert_source.cpp:445, 616` | `mmap(PROT_READ, MAP_SHARED/PRIVATE)` | map experts / GGUF | [PORT] |
| `expert_source.cpp:777-823` | `PrefetchVirtualMemory` (`STRATA_FETCH_PVM` 783) | warm file pages | [WIN] |
| `expert_source.cpp:840-841` | `madvise(MADV_WILLNEED)` on addresses rounded to **4096** | warm file pages (POSIX branch) | [SILENT] On 16 KiB pages a 4 KiB-aligned start that is not 16 KiB-aligned gives EINVAL, and the return value is ignored. Use `getpagesize()` / `vm_page_size`. |
| `expert_source.cpp:1123-1202` | `cudaHostRegister` of resident slices (`STRATA_PARTIAL_PIN` 1173, `_GIB` 1179) | partial pin of the resident set | [UMA-NOP] |
| `expert_source.cpp:1264-1279` | `madvise(MADV_DONTNEED)` plus `posix_fadvise(POSIX_FADV_DONTNEED)` (1274) | drop pages after the resident copy is built | [BLOCK] (`posix_fadvise` is missing). Mac: for file pages there is no exact equivalent; `madvise(MADV_DONTNEED)` on a file mapping only drops this process's mapping, and `msync(MS_INVALIDATE)` invalidates. For anonymous memory use `MADV_FREE_REUSABLE`. |
| `expert_source.cpp:1356, 2244` | `cudaHostAlloc` | exchange buffers, staging | [UMA-NOP]. Shared MTLBuffer, or nothing. |
| `src/platform/direct_file.cpp:33/35` | `VirtualAlloc` / `posix_memalign` | sector-aligned buffers | [PORT] (posix_memalign) |
| `direct_file.cpp:63` | `STRATA_IO_THREADS` | worker threads for pread | [PORT] |
| `direct_file.cpp:166` | `CreateFileW(FILE_FLAG_NO_BUFFERING\|OVERLAPPED)` | unbuffered asynchronous reads | [WIN] |
| `direct_file.cpp:263-308` | `open(O_RDONLY\|O_DIRECT)` (308), pread workers | unbuffered reads (PLE rows) | [BLOCK] (`O_DIRECT` is missing). Mac: `open` + `fcntl(fd, F_NOCACHE, 1)` + `fcntl(fd, F_RDAHEAD, 0)`; F_NOCACHE has no alignment requirement. Keep posix_memalign buffers for speed. |
| `src/kernels/ngram.cpp:262-277` | `madvise(WILLNEED)` + `mlock` with `page = 4096` (263) | `--ple-io ram` locks the 28.8 GB table | [SILENT] (page size), and mlock of 28.8 GB will fail on macOS (the code then falls back to touching pages, `:270`). |
| `ngram.cpp:332-342, 421-432` | `PrefetchVirtualMemory` | PLE warm | [WIN] |
| `src/ngram/ple_reader.cpp:339` | `PAGE = 4096` row limit and alignment | DirectFile row reads | [PORT] functionally (it is a sector size, not the VM page size). |
| `include/strata/artifact/gguf_reader.hpp:384-393` | `CreateFileMappingA` / `MapViewOfFile` | GGUF map | [WIN] |
| `gguf_reader.hpp:404-405` | `mmap(PROT_READ, MAP_PRIVATE)` | GGUF map | [PORT] |
| `src/core/weights.cpp:78-82` | `_fseeki64` / `fseeko` | pack reads | [PORT] |
| `weights.cpp:223-231` | `cudaHostAlloc` staging for conversions | dense upload | [UMA-NOP]. Convert straight into a shared MTLBuffer. |
| `generate.cpp:109` | `std::filesystem::exists("/dev/dxg")` (`under_wddm`) | WSL2 WDDM detection | [SILENT] but harmless (false). |
| `generate.cpp:716-754` | Windows `K32GetProcessMemoryInfo`/`GlobalMemoryStatusEx`; else `/proc/self/stat` (majflt), `/proc/self/status` (VmRSS, VmSwap), `/proc/meminfo` | `mem_sample` for stall reports and memory lines | [SILENT]: reports 0. Mac: `task_info(MACH_TASK_BASIC_INFO)` (resident_size) or `TASK_VM_INFO` (phys_footprint), `getrusage(RUSAGE_SELF).ru_majflt`, `sysctl vm.swapusage`, host_statistics64. |
| `generate.cpp:2772-2780` | `GlobalMemoryStatusEx` (commit left) | message when cache allocation fails | [WIN] |
| `generate.cpp:950-977` | `cudaMallocHost(256 MiB)` x 4 copies | `probe_pcie_h2d_gbps` | [UMA-NOP] |
| `generate.cpp:2001, 3323` | `cudaHostAlloc` | dump staging | [UMA-NOP] |
| `generate.cpp:4159` | `cudaHostAlloc(Mapped\|Portable)` | layer-split hand-off | [DGPU] |
| `src/core/conversation_memory.cpp:36-48` | `GlobalMemoryStatusEx` / `/proc/meminfo`; **returns `{}` on any other OS** | admission for conversation parking | [SILENT]: on macOS parking is never admitted. Add a `__APPLE__` branch using host_statistics64. |
| `src/core/layer.cpp:620-635` | `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` | Doorbell (`h_x_f, h_ids, h_weights, h_seq, h_flag`) | Shared MTLBuffer. |
| `layer.cpp:883, 998` | `cudaHostAlloc` | session staging | Shared MTLBuffer. |
| `native_head.cpp:133, 158, 173` | `cudaHostAlloc` | head / embedding staging | Shared MTLBuffer. |
| `expert_cache.cpp:76` | `cudaHostAlloc` | slot fill staging | [UMA-NOP] |
| `mtp.cpp:58-60` | `cudaHostAlloc(Mapped)` | MTP step records | Shared MTLBuffer. |
| `prefill.cpp:194, 529, 575-576, 589` | `cudaHostAlloc` | prompt staging rings | [UMA-NOP] / shared. |
| `remote_experts.cpp:190-192, 202-205` | mapped / zero-copy | helper GPU rows | [DGPU] |
| `session.cpp:531, 843` | `cudaHostAlloc` | token staging / scratch | Shared MTLBuffer. |
| `verify.cpp:72-74` | `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer` | verify staging: tok, step, pos, commit, ple, out, x, ids, w, seq, flag, flagA, flagB | Shared MTLBuffer. |

### 3.3 CUDA runtime surface to replace on the host

The HIP shim `include/strata/hip_compat/cuda_runtime.h:17-92` (~76 `#define cudaX hipX`) is in practice the complete list of CUDA runtime calls the host uses. Map them as follows.

| CUDA | Metal / Apple equivalent |
|---|---|
| `cudaMalloc` / `cudaFree` | `newBufferWithLength:options:` (Shared for CPU-touched data, Private for GPU-only hot data) |
| `cudaHostAlloc(Mapped\|Portable)` + `cudaHostGetDevicePointer` | one Shared MTLBuffer; CPU uses `contents`, GPU binds the buffer (or uses `gpuAddress` with argument buffers) |
| `cudaHostRegister` / `Unregister` | `newBufferWithBytesNoCopy` (page-aligned, length a page multiple) / release |
| `cudaMemcpy[Async]` H2D/D2H/D2D | `memcpy` on `contents`, or a blit encoder for Private buffers; mostly deleted |
| `cudaMemset[Async]` | `memset` / `fillBuffer:range:value:` |
| `cudaStreamCreate[WithFlags]` | `MTLCommandQueue` (one or two) |
| `cudaEventRecord` / `Query` / `Synchronize` | `MTLSharedEvent` signal / `signaledValue` / `waitUntilSignaledValue:timeoutMS:` (macOS 14+) or completion handlers |
| `cudaStreamWaitEvent` | `encodeWaitForEvent:value:` |
| `cudaLaunchHostFunc` | `addCompletedHandler:` or `MTLSharedEventListener` |
| `cudaStreamBeginCapture` / `cudaGraphInstantiate` / `cudaGraphLaunch` | re-encode per token (encoding costs a few microseconds per dispatch), or `MTLIndirectCommandBuffer`; Metal 4 `MTL4CommandBuffer` with argument tables |
| `cudaStreamQuery` used as a WDDM flush | nothing; `commit` submits immediately |
| `cudaMemGetInfo` | `recommendedMaxWorkingSetSize - currentAllocatedSize` |
| `cudaDeviceSynchronize` | `waitUntilCompleted` on the last command buffer |
| `cudaSetDevice` / `OnDevice` RAII (`on_device.hpp`) | single device; delete |
| `cudaGetDeviceProperties` / `luid` | `MTLDevice` name, `hasUnifiedMemory`, `maxBufferLength`, `supportsFamily` |

---

## 4. Expert tiers, adaptive swap, PCIe share, CPU pool, multi-GPU

### 4.1 Components and where they live

| Component | Declaration | Role | Discrete-only? |
|---|---|---|---|
| `ExpertSource` (abstract) | `expert_source.hpp:101-145` | Interface for "give me expert (layer, e)". Virtuals: `blob`, `reads`, `pinned`, `begin_layer`, `device_alias`, `pcie_layer`, `transient`, `copy_blob`, `prefetch`, `warm`, `warms`. | Interface is portable; `pinned`, `device_alias`, `pcie_layer` are [DGPU]. |
| `ArenaExpertSource` | `expert_source.hpp:591` | The whole `experts.bin` loaded into a `PinnedArena` (`pinned.hpp:30`). `load_experts` (101) and `load_experts_ranges` (104) use 8 threads and 16 MB chunks, with an FNV-1a checksum per layer (`LoadStats`, `pinned.hpp:65-79`). | Loading into RAM is portable; the pinning is [UMA-NOP]. |
| `FileExpertSource` | `expert_source.hpp:386` | mmap of the pack or GGUF (`gguf_mode` 407), with RAM and file tiers. Members: resident RAM complement (`pin_cache_complement` 424-427), exchange buffers (`stage_exchange` 453, `commit_exchanges` 455), counters `file_reads` 459 and `ram_reads` 462, fetch threads (`set_fetch_threads` 469; default 8 at 527; `STRATA_FETCH_THREADS`), `kStageAge 3` (520). | The file tier is relevant on a Mac when RAM < experts; the exchange machinery is [UMA-NOP]. |
| `RouterLookahead` | `expert_source.hpp:152` (`submit` 163) | Predicts the next layer's experts and warms their file pages (`STRATA_LOOKAHEAD` / `_K`, generate.cpp:3029/3040). | Useful only with an SSD tier. |
| `GpuPlanSink` | `expert_source.hpp:188` | The verify window's plan: VRAM groups `ptr/start/dst/tok`, PCIe groups `ptr2/start2`, `counts[3]`, staging slots. | [DGPU] in its PCIe half. |
| `ExpertDispatch` | `expert_source.hpp:213` | Per-layer dispatch state: lookahead 216, hit path (`x_q8_0_hit_scale` 271), `host_res` 282 (residency index per (layer, e): VRAM slot or -1), `split_rows` 284, `pcie_num` 331 (/256), `usage` 336 (decayed routing counts), latched `failed` 347. | Mixed. |
| `ExpertCache` | `expert_cache.hpp:58` | VRAM slots. `open` 69, `open_sized` 72 (per-pair sized slots for native packs, `generate.cpp:2707-2725`), `resident` 81, `set_per_layer_admission` 109 (quota = slots/n_layers), `fill_slot` 121, `fill_slot_blocking` 129, `fill_slot_queued` 132. **No runtime eviction**; only explicit swaps. | [DGPU] |
| `read_expert_profile` | `expert_cache.cpp:12-68` | STRP reader. | Portable. |
| `RemoteExperts` | `remote_experts.hpp:16-60` | A static, profile-filled expert tier on another CUDA device. CUDA0 keeps all dense state, and results return through pinned rows. API: `preflight`, `open`, `begin(layer, x, ids, n_tok, k, kind, primary_res)`, `owns`, `finish`. | [DGPU] |
| `ExpertPool` | `pool.hpp:107`; `pool.cpp` | CPU expert workers. | Needed only if the CPU computes experts. |
| CPU expert kernels | `expert.cpp` (AVX-512 VNNI canonical Q2_0), `q2_avx2.cpp`, `iq_avx2.cpp`, `iq_avx512.cpp`, `kq_avx2.cpp`, `native_expert.cpp` (ggml-cpu `vec_dot`) | Compute one expert for 1..T tokens. | [BLOCK] except native_expert.cpp. |

### 4.2 Startup: building the tiers (generate.cpp)

1. **Choose the RAM tier.**
   - Arena (`experts.bin` loaded, pinned) by default.
   - `--mmap-experts`: FileExpertSource over the pack or GGUF.
   - `--resident-experts`: mmap + a resident RAM copy of everything *not* in VRAM (the "complement") + partial pinning + a 4 GiB headroom.
   - The arena load rate is checked; below 0.2 GiB/s it warns about throttled contexts (`:2612-2618`).
2. **Size the VRAM tier.**
   - `--expert-cache auto` (`:2664-2687`): free VRAM minus (`vram_reserve_mib` + prefill buffers when not borrowing + MTP head bind bytes), divided by max_blob, capped by the profile length.
   - Native packs with a profile get per-pair sized slots (`:2707-2725`), which hold ~30% more IQ3_XXS experts.
   - If allocation fails, the cache shrinks to 3/4 and retries up to 8 times (`:2772-2810`).
3. **Fill the VRAM tier** from the STRP ranking, most important first. `host_res[l*n_expert + e]` becomes the slot index.
4. **Optional helper GPUs** (`--expert-cache-deviceN`): RemoteExperts takes the next ranked pairs, either striped or by layer.
5. **pcie_frac auto.**
   - Native packs: 0.55 x min(1, bw/26 GB/s), using `probe_pcie_h2d_gbps` (`:950-977`). 0 if the bandwidth is below 4 GB/s; the base value if it is at least 20 GB/s.
   - Q2_0 pack: 0.2, direct mode (`:1720-1734`).
   - Converted to `drive.d.pcie_num = round(pcie_frac*256)` (`:4423`).

### 4.3 Per-layer dispatch (decode and verify)

`ExpertDispatch` runs per layer, either from the doorbell (T = 1) or per verify group (`expert_source.cpp:1600-1700` for the multi-token plan).

1. `src->begin_layer(layer, ids, n)` lets the file tier start fetching.
2. If adaptive mode is on, `usage[layer, e] += 1` for each routed entry (`:1616-1618`).
3. **The plan, published first so the GPU starts while the CPU works** (`:1623-1700`):
   - Find the distinct experts in routing order.
   - Each distinct expert is classified as VRAM (`host_res >= 0`, `kind 0`); PCIe (`kind 1`), which applies to the last `nmiss*pcie_num>>8` misses when the layer allows PCIe and the blob is pinned; or CPU (`kind -1`).
   - VRAM groups are written to `P.ptr/start/dst/tok`, and PCIe groups to `P.ptr2/start2`.
   - A PCIe group's `ptr2` is either a staging slot (mode 0, DMA by the host on the `copy_` stream, then `cudaLaunchHostFunc` raises `flagB`, `verify.cpp:1243-1253`) or the expert's device alias in mapped host memory (modes 1/2, where the GPU reads RAM over PCIe directly).
   - `publish_plan` issues `_mm_sfence` and then raises `flagA` (`verify.cpp:1255-1258`).
4. The CPU pool computes the `kind -1` experts for all tokens of the group: activation quantize, then gate/up, then down. It writes `y_miss` rows into mapped memory.
5. The host issues a fence and writes `*flag = want`. The GPU (spinning in `wait_flag_ge`) copies or combines the miss rows (`moe_combine`).

The GPU hit kernels and the CPU pool overlap. The `--stats` line "R4 overlap" (`:6421`) reports how often the hit work was already done when the pool returned.

### 4.4 Adaptive swap (`--adapt-every 4`, `--adapt-swaps 96`)

**Serve path, `generate.cpp:4459-4512`** (the `adapt` lambda; one-shot copy `:6050-6080`)

`usage` holds decayed routing counts per (layer, expert). Every `adapt_every` verify rounds, the adapt lambda runs on its own thread while the GPU commits and drafts. For each layer:
1. Candidates are non-resident experts with `usage >= 2.0`. Victims are resident experts, sorted by ascending usage.
2. Pairs are taken while `cand.usage >= vict.usage + 1.5`, with gain = difference.
3. All swaps are sorted by gain and truncated to `adapt_swaps`.
4. `resident_stage_swaps` (`:130`, called at `:4485` and `:6077`) applies only in resident RAM mode. Before a slot is overwritten, its old content is copied D2H into an exchange buffer, so the CPU can keep computing the evicted expert from RAM. `commit_exchanges` later moves it into the RAM complement.
5. The swap itself is a `cudaMemcpyAsync` H2D of the incoming blob into the victim's slot on `adapt_stream` (or the owning stage's stream with a layer split).
6. `host_res[out]` is set to `kNotResident` immediately, so the CPU computes that expert meanwhile. `(in, slot)` is queued in `pending`, followed by `cudaEventRecord(adapt_ev)`.
7. `apply_pending` admits the queued experts once the event completes: `host_res[in] = slot`, then `res_upload` copies the residency table to each device's `d_res`.
8. All usage counts decay by 0.7.

Failure prints `ERR an adaptive refill failed` (`:5412`).

**Mac relevance.** If every expert lives in unified memory, the GPU reads any expert at the same cost and swapping is meaningless: [UMA-NOP]. If RAM < experts and an SSD tier remains, the same counters could drive *page residency* instead (`madvise(WILLNEED)` plus keeping a hot set wired in a MTLResidencySet). In that case keep the policy and drop the copies.

### 4.5 Hit-rate profile (`data/expert-profile.bin`, STRP)

- **Production of the profile.** `tools/make_profile.py` builds it from `--dump-routing` traces. The ranking is the base profile first, then pairs by trace frequency, then the remaining pairs interleaved across layers.
- **Use of the profile.** It sets the startup fill order of the VRAM tier and caps `--expert-cache auto`. `--prefill auto` also uses it, because cache slots are lent to the prompt path ("borrowing"; `--no-prefill-borrow`; lend/refill messages at `:5240-5281`).
- **Measuring hits.** The `DONE` fields `hits` / `lookups`, the stderr "decode expert cache hit rate", and `--stats` "R4 expert-cache hits ... admitted, refused, cache % full" (`:6422-6432`).
- **Mac.** With all experts in RAM the profile is unused. It becomes relevant again only for an SSD-backed configuration (machines with less RAM than the expert set). There it ranks which experts to keep resident and which to stream.

### 4.6 `--pcie-frac` / `--pcie-mode` [DGPU]

Purpose: move part of the miss work from the CPU (memory-bandwidth-bound, ~40 GB/s DRAM per socket on a PC) to the GPU, which reads the blob over PCIe (~26 GB/s on x16) at the same time. The share is per layer, applied to the last missed experts in routing order.

| Mode | Value | Mechanism |
|---|---|---|
| `dma` | 0 | Host-issued `cudaMemcpyAsync` into staging slots (`kStagingBlobs 16`), then `flagB`. |
| `direct` | 1 | The GPU reads `device_alias` (mapped host memory) in the expert kernel. |
| `kernel` | 2 | A copy kernel moves the blob, then the expert kernel runs on the staged copy. |

Defaults: native packs 0.55 DMA; Q2_0 pack 0.2 direct. Calibration (`tools/calibrate.py`) sweeps (0, .2, .35, .55, .75), and the request key `pcie_frac` overrides per request.

**Mac.** All of this is [UMA-NOP]. The GPU reads every expert at full memory bandwidth (M-series: ~120-800 GB/s depending on the chip), and the CPU shares that same bandwidth. The analogous Mac knob is "CPU share of the routed experts". It is worth having only if the GPU is compute-bound while bandwidth remains, which is unlikely for 2-bit MoE matvecs. Recommendation: start GPU-only and keep the knob as an experiment.

### 4.7 CPU pool (`include/strata/kernels/cpu/pool.hpp`, `src/kernels/cpu/pool.cpp`)

**Types**
- `ExpertJob` (47) and `ExpertJobMulti` (57) are the per-expert work items for 1 token and for T tokens.
- `PoolAffinity` (68): All / Auto / PCores.
- `CpuTopology` (74), `detect_cpu_topology` (83), `physical_cores` (91).
- Split limits: `kMaxSplit 16` (151) and `kMaxSplitMulti 96` (157), which split one expert's rows across workers.

**Worker loop**
- Workers spin on `epoch_` with `_mm_pause` (`pool.cpp:410-425`) and read the clock once every 1024 pauses.
- After `kSpinBeforeSleep` = 20 ms (187; `STRATA_POOL_SPIN_US` 347) a worker sleeps on a mutex + condvar, counted in `sleepers_`.
- `kStall` = 60 s (190) is the pool's own stall detector.
- The host thread can work too (`host_works`; `--no-host-worker`).
- Phase timing (wait-park, drain, re-park) is reported in `--stats` (`:6411-6417`).

**Cache-line padding.** `alignas(64)` atomics `head_, done_, parked_, epoch_, stop_, sleepers_` (`pool.hpp:236-243`), and per-worker ff buffers (253-264).

**Topology**
- Windows: `GetLogicalProcessorInformationEx` with efficiency classes, for P-core primaries, SMT siblings and E-cores (`pool.cpp:~95-136`).
- Linux: `/sys/devices/system/cpu/cpuN/topology/{core_id,physical_package_id}` and `cpu_capacity` (`:145-155`), plus `sched_getaffinity` / `cpu_set_t` (`:165-172`).
- Pinning: `SetThreadAffinityMask` or `pthread_setaffinity_np` (`:267-311`).
- Worker count: P-cores - 1 on hybrid parts with non-All affinity, otherwise all worker cores (`:345-353`).

**macOS port of the pool (if kept)**

| Concern | Replacement |
|---|---|
| topology | `sysctlbyname("hw.perflevel0.physicalcpu")` (P-cores), `hw.perflevel1.physicalcpu` (E-cores), `hw.physicalcpu`, `hw.nperflevels`. Apple Silicon has no SMT. |
| affinity | No hard affinity. `THREAD_AFFINITY_POLICY` is ignored on Apple Silicon. Use `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` so workers prefer P-cores. Under `p-cores`, size the pool to the P-core count. |
| spin | `_mm_pause` becomes `__builtin_arm_yield()` (or `asm volatile("yield")`). For long waits, consider `wfe`/`sev`, or `os_sync_wait_on_address` / `os_sync_wake_by_address_any` (macOS 14.4+) instead of mutex + condvar. |
| padding | Use 128 B: Apple M-series has 128-B cache lines, and `std::hardware_destructive_interference_size` is 128 on arm64 Apple clang. |
| kernels | Canonical Q2_0 (`expert.cpp`, `q2_avx2.cpp`) and the `q2_rows_any` / `act_quant_any` dispatch (`expert_layout.cpp:111-120`, used by `pool.cpp:550-563, 679` and `expert_source.cpp:1724`) have no ARM path. Either write NEON (sdot/i8mm) kernels, or route everything through `native_expert.cpp` (ggml-cpu `vec_dot`). The pinned llama.cpp `3cf03257f` already has a NEON `ggml_vec_dot_q2_0_q8_0` at `ggml/src/ggml-cpu/arch/arm/quants.c:222`, verified in the FetchContent checkout. |

### 4.8 Layer split and multi-GPU [DGPU]

- `--layer-split K|K1,K2|auto` is serve-only (`generate.cpp:1263-1300`). Stage i runs layers [K_i, K_i+1) on device `split_devs[i]`.
  - `auto` places the K values from each GPU's free VRAM after the weights load.
  - `split_same` (one device, split at K) exists for testing.
- `SessionState` carries `layer_lo/hi`, `qsa_ord0` and `gdn_alloc` per stage (`session.hpp:52-61`).
- Hand-off between stages uses mapped/portable host buffers (`generate.cpp:4156-4165`) plus a ring (`STRATA_SPLIT_RING`).
- Each stage has its own ExpertCache, adapt stream and event (`gs->adapt_stream`, `st->adapt_ev`).
- Checkpoints carry `stage_parts`.
- Not supported across GPUs: KV streaming, images, control vectors, remote helper caches, prompt-path slot lending (`:1263-1266`).
- Conversation parking is refused with a split (`:1260-1262`).
- RemoteExperts (helper GPUs): a static profile-filled tier, a CUDA stream per device, zero-copy returns (`STRATA_REMOTE_ZEROCOPY`), and spin waits (`STRATA_REMOTE_SPIN`).

**Mac.** A Mac has one GPU, so all of this is deletable. The code paths are threaded through generate.cpp (`stages`, `stage_of`, `OnDevice`), which is one of the main reasons to write a fresh, slimmer Metal driver rather than port generate.cpp line by line.

### 4.9 The KV ring and `--kv-resident` [DGPU]

`--kv-resident N` keeps N cells (minimum 20480) of the QSA K/V in VRAM and streams the rest from host RAM (`STRATA_KV_RING_OFF` / `KV_MAIN_OFF`, `layer.cpp:502-503`). On a Mac the whole KV lives in one shared buffer, so this is [UMA-NOP].

### 4.10 The CPU/GPU handshake on Metal (design risk)

**What Strata does**
- **Doorbell** (`layer.hpp:402-422`, used by the token graph). The device kernel `doorbell_wait` spins on `h_flag` in mapped host memory.
- **Verify windows.** `wait_flag_ge` / `wait_flag_ge_or` on `flagA`, `flagB` and `flag` inside the captured graph (`verify.cpp:731-779`).
- **Host side.**
  - The host spins `while (*seq < want) _mm_pause()` (`verify.cpp:1077`; `session.cpp:903-960` with `STRATA_SPIN_PAUSE`).
  - It polls `cudaStreamQuery` every ~2 ms to force WDDM submission.
  - It gives up after 20 s and calls `release_gpu_waits`, which writes `UINT32_MAX` to every flag (`verify.cpp:136-144`) so the GPU cannot hang forever. The watchdog uses the same mechanism.
- **Cost.** 5.2 us per handoff. Before the single token graph, per-layer graph launches cost 0.3-0.4 ms each under WDDM (`session.hpp:385-397`).

**Why this does not carry over to Metal**
1. Metal makes no forward-progress guarantee for a GPU thread spinning on memory the CPU writes. Atomics on shared buffers have no cross-device acquire/release semantics, and a command buffer that runs too long can be killed by the GPU watchdog (an "Impacting Interactivity" / timeout error).
2. Coherence of CPU writes to a Shared buffer *during* a running command buffer is not specified. In practice it often works on Apple GPUs, but it is not a supported contract.

**Options**

| Option | Mechanism | Cost | Notes |
|---|---|---|---|
| A. GPU-only experts (recommended) | All experts in shared MTLBuffers (the mmapped GGUF wrapped with `newBufferWithBytesNoCopy`). The router output stays on the GPU, and expert compute uses indirect or grouped dispatch over the selected ids. One command buffer per token, or per verify window. | No CPU round trip at all. | Removes the pool, the doorbell, flagA/flagB, the plan and pcie_frac. Requires RAM >= experts + dense + KV + headroom. |
| B. Event-split command buffers | Per layer: CB1 ends with the router and `encodeSignalEvent(ev, 2l+1)`. The CPU waits (`MTLSharedEventListener` or spinning on `signaledValue`), runs the pool, then calls `ev.signaledValue = 2l+2`. CB2 starts with `encodeWaitForEvent(ev, 2l+2)`. Encode all CBs ahead and commit them together; the GPU blocks on the events. | Event latency of roughly 10-50 us per hop, so 48 layers cost roughly 0.5-2.5 ms/token. Measure it. | Keeps CPU expert compute (useful for an SSD tier, or to add CPU throughput). |
| C. Hybrid | GPU computes all resident experts. CPU computes only experts whose pages are not resident (SSD tier), joined per layer via option B. | | Matches Strata's tiering, minus PCIe. |

Ordering primitives to replace:
- `_mm_sfence()` + seq_cst fence before writing the flag becomes `std::atomic_thread_fence(std::memory_order_release)` (`dmb ish`), followed by the MTLSharedEvent signal. The signal itself orders the CPU's prior writes for the GPU.
- `_InterlockedCompareExchange` / `__atomic` CAS (`verify.cpp:1225-1238`) is [PORT] on clang.

---

## 5. Speculative decoding

### 5.1 Pieces

| Piece | Files | Summary |
|---|---|---|
| Verifier | `include/strata/core/verify.hpp` (header doc 1-20), `src/core/verify.cpp` | T tokens (T <= 8, `exec_[9]` at 206) go through all 48 layers in one captured graph, **bit-exact with T sequential greedy steps**. One window reads the dense weights once, plus the union of the misses (1.75x the misses for T=2, 2.4x for T=3, 3.05x for T=4). |
| MTP drafter | `include/strata/core/mtp.hpp`, `src/core/mtp.cpp` | The model's single multi-token-prediction layer, on the GPU (detailed in 5.3). |
| Suffix drafter | `include/strata/spec/suffix_drafter.hpp`, `src/spec/suffix_drafter.cpp` | Prompt-lookup drafter. A trigram hash maps to the WAYS = 4 most recent end positions. It extends each match backwards up to max_match and proposes the tokens that followed the longest match. Memory is ~20 B per history token, and appends are O(1). |
| Draft policy | `include/strata/spec/draft_policy.hpp`, `src/spec/draft_policy.cpp` | Each round, chooses between the MTP window and a lookup window by expected committed tokens per millisecond (5.4). |
| Controller | `include/strata/spec/controller.hpp`, `src/spec/controller.cpp` | A cost-model controller. **Test-only; not wired into generate.cpp.** |
| Coupled draft | `include/strata/core/coupled_draft.hpp` | Under sampling, the drafter samples with the target's chain and the same Philox uniform (5.6). |

### 5.2 Verifier API (`verify.hpp`)

- `run(T, tokens, pos0, pool, user, out, err)` (76).
  - Row t samples with Philox(seed, pos0 + t) (84).
  - Penalties use one history row per window row (88-93).
- `set_stage` (111), `set_next` (116), `handoff_floats = hc*n_embd + n_embd + hc` (118). These support the layer split.
- `commit(n_keep)` (121), `set_commit_async` (126), `wait_commit` (129).
  - Commit replays GDN from the stored per-row inputs for the kept rows.
  - It restores the indexer tail from a snapshot and re-appends the kept keys.
  - It sets the PLE history from its snapshot.
  - K/V cells past the accepted prefix are left in place and overwritten later, because KV is positional.
- `set_split` (149): `--spec-split` row groups, `groups_[9]`, with a per-(layer, group) handshake.
- `set_pcie_mode` (154).
- `diag` (62) and `release_gpu_waits(timeout_ms)` (68) serve the watchdog.
- Requirements: the native configuration and a profile-filled VRAM tier.
- Mapped staging (209-219): tok, step, pos, commit, ple, out, x, ids, w, seq.
- Flags: `flag` 220, `flagA` 221 (plan ready), `flagB` 222 (PCIe DMA landed).
- Other members: `commit_done_` event 223, `copy_` stream, `flag_sets_[256]`, `ymiss`, `kStagingBlobs 16`.

Host loop of one window (`verify.cpp:1050-1135`):
1. Zero flagA and flagB, `cudaGraphLaunch(exec_[T])`, then `cudaStreamQuery`.
2. For each (layer, group):
   - Spin until `*seq >= want` (with `_mm_pause` at 1077, a `cudaStreamQuery` every 2 ms, and a 20 s timeout that calls `release_gpu_waits`).
   - `set_plan_slot`, run the pool on that group's rows (the plan is published inside dispatch), `progress_tick`.
   - Fence, `_mm_sfence` (1107). If no plan was published, write an empty plan and raise flagA and flagB. Then `*flag = want`.
3. `cudaStreamSynchronize(cs_)` and `copy_`.
4. Commit (1318 `STRATA_COMMIT_SYNC`; graph launch at 1328).

### 5.3 MTP drafter (`mtp.hpp`)

**Model**
- Cell i pairs the main model's final multi-stream residual at position i with the token at i+1, at RoPE position i. Its output predicts the token at i+2.
- Computation:
  - e = fc_embedding(rms(embed(tok))·(1+w)).
  - h = fc_hidden per stream (rms_10240(R)·(1+w)).
  - R = h + e, then the attention hyper-connection, then QSA attention (own K/V), then the MLP hyper-connection, then MoE (512 experts, top-10, shared expert), then the inject, then the final mixer, then the head over the draft vocabulary subset.

**Simplifications** (they affect only draft quality, never the output):
- Q8_0 projections.
- Dense attention over every cell. This is identical to the sparse selection below 2051 cells, so speculative cells never touch indexer state.
- All 512 experts in VRAM (708 MB).

**API**

| Call | Line | Notes |
|---|---|---|
| `load(rt_dir, g, ss, max_t, err, ...)` | 45 | Takes ~0.9 GB. It must run before the VRAM tier is sized. |
| `bind_bytes` | 59 | Reserved in `--expert-cache auto`. |
| `bind(wt, head, window_R)` | 61 | |
| `prefill(R_rows, next_tokens, n, cell0)` | 65 | Batched over the prompt. |
| `draft(T, tokens, p, a, drafts, err, probs, min_p, n_drafts)` | 70 | Catch-up over T cells from p, then the chain from accepted row a for T-1 drafts. `probs` returns per-draft probabilities for `spec_min_p`. |
| `draft_first` | 74 | First round. |
| `kv_restore(upto)` | 56 | Used by the conversation cache and rollback. |
| `set_draft_sampling` | 81 | Coupled mode. |
| `set_draft_history` | 84 | Coupled mode. |
| `first_needed` | 96 | `prompt_len - window - 64`; the MTP needs only the last `--mtp-window` cells. |

Acceptance measured offline in `tools/mtp_probe.py`: 0.89 / 0.86 / 0.85 at steps 1-3.

### 5.4 Draft policy constants (`draft_policy.cpp`)

| Constant | Value |
|---|---|
| kMaxT | 8 |
| kBuckets | 4; match-length buckets <6, <12, <24, else |
| margin | 0.03 |
| kShape (prior cost shape by T) | {0, 1, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6} |
| kCostAlpha | 0.1 |
| kTokAlpha | 0.05 |
| kDecay | 0.97 |
| kPriorQ | {0.75, 0.88, 0.93, 0.96}, with kPriorN 4 |
| kProbes | 3 |
| Unseen MTP tokens prior | 1 + 0.7(t-1) |

How `choose` decides:
- Lookup value: E(k) = 1 + q + ... + q^k, divided by cost(k+1).
- MTP value: mtp_tokens(t) / cost(t).
- The lookup window is taken when its best E/cost exceeds the MTP's by more than the margin.
- Probe rule: try the full size when q >= 0.85 and that size has fewer than 3 samples.

`observe` updates the EMAs from the window used, the number accepted and the round time. The policy changes only speed, never the output.

### 5.5 Window selection in serve (`generate.cpp:4663-4666, 5322-5430`)

**Setup**
- S = `--spec`; S_mtp = min(mtp_max_t, S) when `mtp_max_t > 0`.
- `SuffixDrafter(max(1, suffix_draft), 64, max_context + 4096)`, so min_match is 3 by default and max_match 64.
- `DraftPolicy(S)`.

**Per round**
1. T = S_mtp. If `spec_min_p > 0`, T grows only while `dprob[T-1] >= min_p` (the MTP's own draft probabilities). The first window is T = 1.
2. Suffix: `k = sfx.propose(S-1)`. If `sbuf[0] == drafts[0]` (the lookup agrees with the MTP's first draft), then `pk = policy.choose(T, k, match)`. If `pk.lookup`, T = pk.t and the lookup tokens fill the window.
3. Stage penalty rows, then `ver.run(T, window, p, ...)`.
4. **Acceptance (exact match):** `a = 0; while (a < T-1 && window[a+1] == outv[a]) ++a;`. `outv[t]` is the verifier's pick after row t.
5. Every `adapt_every` rounds an adapt thread runs (section 4.4). `ver.commit(a+1)` keeps the first a+1 rows.
6. Emit `T <id>` for outv[0..a] while produced < max_new and no EOS (`:5389`). The finish reason is `stop` (EOS), `cancel` (STOP) or `length`.
7. `mtp.draft(T, outv, p, a, drafts, err, dprob, min_p)`, then `policy.observe(from_sfx, T, a, match, round_ms)`, then `sfx.append(accepted)`.
8. x = outv[a]; p += a + 1.

### 5.6 Exactness argument (what a Metal port must preserve)

1. **Greedy.** Each emitted token is the verifier's argmax at that row. The verify graph is bit-exact with sequential decode because every row is computed by the same per-row kernels, with no batch-dependent reduction order. A Metal port must keep that property: the T-row kernels must reduce each row in the same order as the T = 1 path. The simplest way is to always decode through the verify path with T = 1.
2. **Sampling.** Row t of a window at pos0 is sampled with Philox(seed, pos0 + t) over the chain penalties, top_k, top_p, min_p, temperature, inverse-CDF pick. A draft is kept only when it **equals** that pick.
   - This is not rejection sampling with residual resampling. The output is a pure function of the seed, whatever the drafts were.
   - Penalty history row t = the last h tokens of (consumed, x, drafts 0..t-1) (`coupled_draft.hpp` header).
3. **Coupled draft (STRATA_SPEC_COUPLED / --coupled-draft).**
   - The MTP samples its draft with the target's chain and the same uniform the target will draw for the verifying row. The counter is cell + 1 (`coupled_draft_counter`). Draft j after a window at p with a accepted runs at cell p + a + j.
   - It keeps a device penalty ring of `kCoupledHistCap 4096 + max_t`.
   - This raises acceptance under sampling without changing the text.
4. **Rollback.**
   - Main model: `commit(a+1)` replays GDN from stored inputs, restores indexer tails and the PLE history, and lets KV be overwritten.
   - MTP: its dense attention makes speculative cells harmless; `kv_restore` handles conversation restore.
   - The suffix drafter only ever sees accepted tokens.
5. **Fingerprints** for regression: `STRATA_STATE_HASH` / `_GDN` (`:5455, 5497`) and `STATE_HASH` error lines (5460, 5555).

### 5.7 Controller (test-only)

`controller.hpp` / `.cpp` define K_MAX 8 and a `CostModel`:

| Field | Value |
|---|---|
| dense_ms | 11 |
| dense_ratio | {1, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3} |
| cpu_all_miss | 15.8 |
| hit_rate | 0.55 |
| distinct_ratio | {1, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2} |
| extra_use | 0.25 |
| sync | 2.4 |
| mtp_draft | 1.2 |
| lookup_draft | 0.01 |

Other parameters: min_gain 0.05, ema 0.05, acceptance buckets >=16 / >=8 / >=5.

It is not used by the engine; DraftPolicy replaced it. Its `distinct_ratio` table is still a useful model of how the union of experts grows with T, which on a Mac sets the memory traffic of a verify window.

### 5.8 Mac notes

- spec/*.cpp is [PORT] (verified to compile).
- The verify window's value on a Mac is that one window reads the dense weights once and the expert union once. On a bandwidth-bound UMA machine that is exactly the win, so keep the T-row design.
- The flagA/flagB/plan machinery is [UMA-NOP] under option A (section 4.10).
- MTP needs Metal kernels for canonical-blob Q2_0 experts and Q8_0 dense, or a re-pack of `mtp/rt` into whatever format the Metal expert kernel uses.

---

## 6. Conversation and prompt cache

### 6.1 What is cached

- **Checkpoint** (`ConversationCheckpoint`, `conversation_cache.hpp:23-40`):
  - `ids`, `imgs` (image keys {start, hash}).
  - `gdn`: recurrent state of the 36 GDN layers.
  - `ple`: PLE history.
  - `tails`: QSA indexer tails.
  - `dead`, `block_pos`, `used` (LRU stamp), `stage_parts` (layer split).
  - Size: ~118 MB of host RAM each (`generate.cpp:383-385`).
- **KV is positional.** One arena holds one branch, and retained checkpoints form a chain of prefixes of the live KV. A checkpoint is only useful while the live KV still holds its prefix.
- **Live session.** The state after the previous request.
- **Parked conversation** (`SavedConversation`, `:63-80`), opt-in with `--conversation-cache-mib > 0`:
  - `geometry[18]`, `layer_lo/hi`, `live`, `checkpoints`, `cvec`.
  - `kv` (`ConversationKv`, 43-50): format, cells, heads, head_dim, page_size, pooled_rows, idx_dim, and `ConversationBuffer` k, v, k_scale, v_scale, pooled. Main layers come first, then the draft layer.
  - A whole independent conversation, including its KV, copied to host RAM so another chat can use the GPU arena.

### 6.2 When checkpoints are taken (`generate.cpp:4318-4410, 5212-5240`)

The prompt is read in parts, cut at these positions in order (`for to : {reread_to, root_at, turn_at, n-1}`):
1. `root_at`: the first `<|im_start|>` (turn_token 248045). Taken when the prompt is read from 0 and that boundary is at least `--prompt-cache-root` 2048 tokens in. It covers the end of the system prompt and tools, which every chat of the same client shares. The root is pinned by the retention policy.
2. `turn_at`: the last `<|im_start|>` after `resume`. This is where the chat history ends and the new assistant turn begins, which is the checkpoint the next request reuses.
3. Every `--prompt-cache-every` 16384 freshly read tokens (`:4381-4398`), counted per stage with a layer split (`part_next`).

`checkpoint_at(L)` (`:4327-4360`):
- If a checkpoint at L already exists, it only bumps `used`.
- Otherwise it calls `cudaDeviceSynchronize` and `checkpoint_save`, plus each stage's part.
- When the count exceeds `--prompt-cache` (6), it calls `conv_cache::eviction_victim` (`include/strata/program/conv_cache.hpp`): index 0 (the root) is pinned and the rest are evicted LRU by stamp. With cap < 2 the oldest is evicted.

### 6.3 Resume selection (`generate.cpp:4858-5009`)

1. `starts_with(candidate, prompt)`: candidate length L <= n - 1, ids equal, and images below L equal (image keys).
2. Candidates in order:
   - The live session.
   - Checkpoints (the longest wins).
   - Parked conversations via `conversations.best` (newest first, considering live and every checkpoint). A longer parked match is taken (`take`) and validated.
3. If a parked conversation wins:
   - `park_current` saves the outgoing live state into the cache (if admitted), then the snapshot is restored.
   - Results are `ConversationRestore restored | invalid | transfer_failed`.
4. On a control-vector mismatch, checkpoints and the live session are cleared (the state depends on the cvec).
5. Checkpoints longer than `resume`, or not on this path, are erased, because the KV no longer holds them.
6. If `resume == 0`, `session_zero` runs on every stage. Otherwise the checkpoint is restored and `c->used = ++check_clock`. `STRATA_CKPT_REREAD` re-reads instead.
7. `mtp.kv_restore(resume)`, then `RESUME <resume>`.
8. Short text parts (<= `--short-read` 64 tokens) after resume go through verify windows, S tokens at a time (`:5015-5071`), instead of the batched prompt path.

### 6.4 Conversation parking (opt-in)

- `ConversationCache` (`conversation_cache.hpp:97-217`) is a deque, least recently active first. Methods:
  - `best`, `take`.
  - `can_fit`, `make_room` (FIFO eviction under the byte budget and slot count).
  - `drop_superseded` (#342), `put`, `retain` / `limit_reuse`.
- `ConversationBuffer` (`conversation_buffer.hpp`): 16 MiB segments with geometric growth and `allocation_peak`.
- Snapshot API (`conversation_snapshot.hpp`): `kv_bytes / save / capture_bytes / validate / restore / verify`, `state_sizes`, `session_sizes`, checkpoint and snapshot variants. Implementations are in `conversation_snapshot.cpp` / `conversation_state.cpp`, using cudaMemcpy D2H/H2D.
- Admission: `conversation_memory_admit(available, alloc, floor)`, with floor = `--conversation-cache-min-free-mib` 2560 and `available` from `conversation_available_memory()` (`conversation_memory.cpp:36-48`).
  - **On macOS this function returns `{}`, so parking is never admitted** [SILENT].
- Not allowed with `--layer-split` (`generate.cpp:1260-1262`). A warning is printed if `--prompt-cache 0` or slots 0 disables it (`:1257-1259`).
- Tests:
  - `conversation_cache_test` and `conversation_memory_test` (standalone, Mac OK).
  - `conversation_snapshot_test` and `conversation_validation_test` (GPU block).
  - `conversation_transfer_test`: GNU ld `--wrap` failure injection, `UNIX AND NOT APPLE` (`CMakeLists.txt:380-386`).

### 6.5 Mac port

- The cache logic (`conversation_cache.hpp`, `conv_cache.hpp`, `conversation_buffer.hpp`) is header-only and [PORT].
- The snapshot transfer is memcpy on UMA. Better still, a parked conversation's KV can stay in its own MTLBuffer arena and be "restored" by switching buffer pointers. This needs a KV arena per slot within the RAM budget, which also removes the 118 MB checkpoint copies for the GDN state when it is double-buffered.
- Implement the Darwin branch of `conversation_available_memory()` with host_statistics64:
  ```
  vm_statistics64_data_t s; mach_msg_type_number_t c = HOST_VM_INFO64_COUNT;
  host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t)&s, &c);
  avail = (s.free_count + s.inactive_count + s.purgeable_count + s.speculative_count) * vm_page_size;
  ```

---
## 7. Build and tests on macOS (measured on this machine)

### 7.1 Build system facts (`CMakeLists.txt`)

**Project settings**
- Version 0.1.34 (`:11`). CMake >= 3.24; C++20.
- `STRATA_ENABLE_CUDA` and `STRATA_ENABLE_HIP` both default OFF and are mutually exclusive (`:35-48`). HIP comes in through `cmake/hip_backend.cmake` (`:50`).
- The backend seam is `_strata_gpu_runtime_target`, `_strata_blas_target` and `_strata_gpu_include_directories` (`:111-117`). A Metal backend would add a third value here; there is no abstraction layer underneath, because the engine sources call `cuda*` directly.

**Test switches**
- `STRATA_BUILD_TESTS` defaults to ON only when `tests/CMakeLists.txt` **and** `bench/micro` exist (`:52-57`). The published tree has neither, so the default is OFF. Setting it ON still builds every in-tree test target, but `add_subdirectory(tests)` is skipped (`:1023-1029`).
- `STRATA_BUILD_CONVERSATION_TESTS` (`:58, 62-68`) builds the two standalone conversation tests.

**Always-built targets**
- `strata_artifact` (INTERFACE) with `strata-gguf` (`:151`) and `strata-dequant` (`:154`); `gguf_reader_test` (`:160`) and `gguf_split_test` (`:164`) when tests are on.
- `strata_plan` with `strata-plan` (`:176`).
- `strata_spec` (`:750`) and its tests (`:753-761`).
- `conv_cache_test`, `coupled_draft_test` and `bf16_bits_test` (`:763-776`).
- `strata_kernels_cpu` (`:778-792`), **unconditional**. It contains `expert.cpp`, `pool.cpp`, `expert_layout.cpp` and `q2_avx2.cpp`, with per-source `-mavx512*` / `-mavx2 -mfma -mf16c` flags (`:783-790`).
- `STRATA_NATIVE_EXPERTS` (default **ON**, `:800`) adds `native_expert.cpp`, `iq_avx512.cpp`, `iq_avx2.cpp` and `kq_avx2.cpp`, again with AVX flags (`:844-852`).
  - It builds static ggml from `STRATA_GGML_DIR`, or fetches llama.cpp at `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`:828-837`, `third_party/ggml/VERSION.txt`).
  - It links `ggml-cpu ggml-base` (`:853`).
  - `STRATA_PORTABLE` forces GGML_AVX/AVX2/FMA/F16C/BMI2 ON (`:817-822`). That must never be set on a Mac.
- CPU tests (`:984-1013`): `expert_multi_test`, `expert_parity` (needs a real pack), `pool_test` (needs a pack) and `pool_stress`.
- `direct_file_async_test` is gated by `UNIX AND NOT APPLE` (`:1015`).

**GPU-only block (`:181-736`)**
- `strata_core` (`:182`): device.cu, pinned.cu, platform/memory.cpp, graph.cpp, weights.cpp, layout.cpp.
- `strata_kernels` (all `.cu` sources), `strata_engine` (`:341`; layer, session, verify, expert_source, expert_cache, mtp, remote_experts, native_*, conversation_*) and `strata_prefill` (`:395`).
- Executables: `strata` (`:406`), `strata-device` / `load` / `concurrent` / `overlap` (`:194-204`).
- Most tests, including `platform_memory_test` (`:684`) and `ple_reader_test` (`:695`), even though both are pure CPU code.
- `pinned_shared_test` (Linux, `:688-692`).
- `conversation_transfer_test` (`:380-387`), which needs GNU ld `--wrap` and is gated `UNIX AND NOT APPLE`.

### 7.2 What happens on macOS 27 / arm64 / Apple clang 21 today

All of this was run out-of-source in `/tmp/strata-research/scratch/build-host`; upstream was not modified.

**Configure**

`cmake -S /tmp/strata-upstream -B . -DCMAKE_BUILD_TYPE=Release` **succeeds**.
- FetchContent clones llama.cpp `3cf03257f`.
- ggml detects ARM, picks `-mcpu=native+dotprod+i8mm+nosve+sme`, and finds Accelerate BLAS.
- The **Metal** backend is included automatically, because GGML_METAL defaults ON on Apple. Strata links only `ggml-cpu` and `ggml-base`.
- The only warning is a CMP0169 deprecation of `FetchContent_Populate`.

**Targets**

| Target / object | Result | Cause |
|---|---|---|
| `strata-gguf`, `strata-dequant`, `strata-plan` | **OK** | header-only artifact and plan code |
| `strata_spec` | **OK** | |
| `gguf_reader_test`, `gguf_split_test` | **OK, PASS** | |
| `suffix_drafter_test`, `controller_test`, `draft_policy_test` | **OK, PASS** | |
| `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test` | **OK, PASS** | |
| `conversation_cache_test` (4175 checks), `conversation_memory_test` (23 checks) | **OK, PASS** | Note: memory_test passes because it tests the admission arithmetic, not `conversation_available_memory()`. |
| `src/kernels/cpu/native_expert.o` | **OK** | ggml-cpu NEON path |
| `src/kernels/cpu/pool.o` | **FAIL** (20 errors) | `immintrin.h` "only meant to be used on x86" (`pool.cpp:9`), then `cpu_set_t` / `CPU_SETSIZE` at `:165, 168, 272, 289, 293, 305`. `sched_getaffinity` / `pthread_*affinity_np` are absent from the SDK. |
| `expert.o` | **FAIL** | clang rejects `-mavx512f` and the other AVX-512 flags for the arm64 target (`CMakeLists.txt:787-788`). The source also includes `<cpuid.h>` (`expert.cpp:12`). |
| `expert_layout.o` | **FAIL** (11 errors) | `<cpuid.h>` is "for x86 only" (`expert_layout.cpp:12`), plus `__cpuid_count` / `_xgetbv` at `:31-109` |
| `q2_avx2.o`, `iq_avx2.o`, `kq_avx2.o` | **FAIL** | `-mavx2 -mfma -mf16c` unsupported for arm64 (`:789, 850-851`) |
| `iq_avx512.o` | **FAIL** | AVX-512 flags (`:847-848`) |
| `strata_kernels_cpu` (and so every CPU test) | **FAIL** | the objects above |
| `direct_file.cpp` (syntax check with an `immintrin` shim) | **FAIL** at `:308` | `O_DIRECT` is undeclared on macOS |
| `platform/memory.cpp`, `ngram/ple_reader.cpp`, `kernels/ngram.cpp`, `conversation_memory.cpp` | compile (syntax-checked) | `ngram.cpp` uses only POSIX mmap/madvise/mlock. `conversation_memory.cpp` compiles but returns `{}` on macOS. |
| `expert_source.cpp`, `expert_cache.cpp`, `conversation_state.cpp` and every engine TU | **FAIL** | `cuda_runtime.h` not found, through the public headers `include/strata/core/{graph,mtp,on_device,remote_experts,session,verify}.hpp` and directly in `expert_cache.cpp:4`, `layer.cpp`, `weights.cpp`, `native_head.cpp`, `native_dense.cpp`, `conversation_snapshot.cpp`, `generate.cpp` |
| `strata-vision` (`tools/vision`, `-DLLAMA_DIR=<the fetched checkout>`) | **OK**, binary runs and prints usage | |

**Minimal CMake changes to get the CPU pieces building on Apple**
1. Guard every `-mavx*` `set_source_files_properties` with `if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")`, and leave out `expert.cpp`, `q2_avx2.cpp`, `iq_avx*.cpp` and `kq_avx2.cpp` on arm64.
2. Replace them with a NEON TU, or with stubs returning "unsupported" for the `cpu_avx512_ok` / `q2_rows_any` / `act_quant_any` dispatch (`expert_layout.cpp:111-120`).
3. Guard `immintrin.h` and `cpuid.h` in `pool.cpp`, `graph.cpp`, `verify.cpp`, `session.cpp`, `expert.cpp` and `expert_layout.cpp` (section 8).
4. Add Darwin branches in `pool.cpp` for topology and affinity, in `direct_file.cpp` for `F_NOCACHE`, and in `expert_source.cpp` and `conversation_memory.cpp` for memory probes.
5. Move `platform_memory_test` and `ple_reader_test` out of the GPU block (they are pure CPU), and enable `direct_file_async_test` on Apple once `F_NOCACHE` is in.
6. A Metal backend: new option `STRATA_ENABLE_METAL`, `enable_language(OBJCXX)`, link `-framework Metal -framework Foundation`, and compile `.metal` files to a `.metallib` with `xcrun metal` / `metallib`, or embed the source as ggml does.

### 7.3 Test inventory and Mac applicability

| Test | Location | Needs | Mac |
|---|---|---|---|
| gguf_reader_test, gguf_split_test | `src/artifact`, `tests/core` | synthetic | passes |
| suffix_drafter / controller / draft_policy tests | `src/spec` | | passes |
| conv_cache / coupled_draft / bf16_bits | `src/program`, `src/core`, `src/kernels` | | passes |
| conversation_cache / conversation_memory | `src/core` | | passes |
| expert_multi_test, pool_stress | `src/kernels/cpu` | CPU kernels | blocked by the x86 kernels; reusable with NEON kernels |
| expert_parity, pool_test | `src/kernels/cpu` | a real canonical pack | needs a canonical pack (AVX-512 producer) |
| direct_file_async_test | `tests/platform` | Linux | enable after `F_NOCACHE` |
| platform_memory_test, ple_reader_test | GPU block | | trivially portable once moved |
| file_expert_source_test, expert_layout_test, expert_cache_per_layer_test, native_dense_ple_key_test | `tests/core` | (the `tests/CMakeLists.txt` that would build them is missing) | Mostly portable logic; expert_source pulls CUDA headers. |
| tests/cuda/* (handoff, intrinsics, mapped_alias, router_fast, ple_iq4, prefill_*) | CUDA | | Port as Metal parity tests; `mapped_alias.cpp` and `handoff.cpp` encode exactly the zero-copy contract to re-validate on Metal. |
| model-level ctests (`:500-680`: logits parity vs ggml, prefill parity, spec exactness, snapshot round trips) | GPU block | the 38 GB model, the pack, a ggml capture | The real acceptance suite for a port. Re-run against a Metal engine with identical CLI and stdout. |
| `src/kernels/iq_parity.cpp` + `tools/iq_fixture.py` | | fixtures | ready-made per-format parity harness for Metal kernels |

---

## 8. Every x86-only or Windows/Linux-only site, with its ARM64 / macOS replacement

### 8.1 x86 intrinsics and CPU detection (compile blockers on arm64)

| Site | Code | Guarded? | Replacement on arm64 |
|---|---|---|---|
| `src/core/session.cpp:22-29` | `immintrin.h` + `STRATA_SPIN_PAUSE()` = `_mm_pause` | **guarded** (no-op elsewhere) | Make the non-x86 branch `__builtin_arm_yield()` rather than a no-op. |
| `src/core/session.cpp:951` | `_mm_sfence()` before writing the doorbell flag | **unguarded**: it uses the guarded include, so it is undeclared on arm64 | `std::atomic_thread_fence(std::memory_order_release)` (`dmb ish`). Under Metal it is replaced by the MTLSharedEvent signal anyway. |
| `src/core/verify.cpp:4` | `<intrin.h>` | MSVC-guarded | n/a |
| `src/core/verify.cpp:50` | `<immintrin.h>` | unguarded | remove or guard |
| `verify.cpp:144` | `_mm_sfence` in `release_gpu_waits` | unguarded | release fence |
| `verify.cpp:1077` | `_mm_pause` in the window spin | unguarded | `__builtin_arm_yield()`; design-wise it is replaced by an event wait |
| `verify.cpp:1107` | `_mm_sfence` before `*flag = want` | unguarded | release fence / event signal |
| `verify.cpp:1257` | `_mm_sfence` in `publish_plan` | unguarded | release fence |
| `src/core/graph.cpp:4` | `<immintrin.h>` | unguarded | guard |
| `graph.cpp:100` | `for 64 x _mm_pause` in the timed wait | unguarded | `yield` loop or `os_sync_wait_on_address` |
| `src/kernels/cpu/pool.cpp:9` | `<immintrin.h>` | unguarded | guard |
| `pool.cpp:420` | worker park spin `_mm_pause` | unguarded | `__builtin_arm_yield()`; consider `wfe` with `sevl`, or `os_sync_wait_on_address` (macOS 14.4+, present in the SDK) |
| `pool.cpp:478, 501` | host wait-for-park / wait-for-done spins | unguarded | same |
| `src/kernels/cpu/expert.cpp:8-13` | `immintrin.h`, `intrin.h`, `cpuid.h` | partly | drop the TU on arm64 |
| `expert.cpp:355-365` | `__cpuid` / `__cpuidex` / `__cpuid_count` for the AVX-512 VNNI/VBMI probe | | arm64 probe: `sysctlbyname("hw.optional.arm.FEAT_DotProd" / "FEAT_I8MM" / "FEAT_BF16" / "FEAT_SME")` |
| `src/kernels/cpu/expert_layout.cpp:8-13, 24-109` | cpuid, `_xgetbv(0)` (42, 75), brand string (92-102), `STRATA_FORCE_AVX2` (26) | | `sysctlbyname("machdep.cpu.brand_string")` gives "Apple M4 Pro"; feature bits as above |
| `expert_layout.cpp:111-120` | `q2_rows_any` / `act_quant_any`, which dispatch to AVX-512 or AVX2 or fail | | A NEON `q2_rows` (sdot/i8mm), or ggml-cpu `ggml_vec_dot_q2_0_q8_0`. The **pinned** llama.cpp `3cf03257f` already has `GGML_TYPE_Q2_0 = 42` (`ggml/include/ggml.h:432`), the NEON `ggml_vec_dot_q2_0_q8_0` (`ggml/src/ggml-cpu/arch/arm/quants.c:222`) and **Metal Q2_0 kernels** (`ggml-metal/kernels/mul_mv.metal:32-33, 386-421` `kernel_mul_mv_q2_0_f32`, plus `mul_mm.metal`, `dequantize.h`, `ggml-metal-device.m:1865, 1894`). Verified in the FetchContent checkout. These are GGUF Q2_0 blocks (18 B / 64 weights); Strata's canonical blob is a re-layout of the same codes. |
| `src/kernels/cpu/q2_avx2.cpp:9` | AVX2 canonical Q2_0 kernel | | NEON rewrite, or drop |
| `src/kernels/cpu/iq_avx2.cpp:19, 94-95` | AVX2 IQ kernels; `_mm_prefetch` | | ggml-cpu NEON `vec_dot` (already used by `native_expert.cpp`); `__builtin_prefetch` |
| `src/kernels/cpu/iq_avx512.cpp:17, 166-167` | AVX-512 IQ kernels; `_mm_prefetch` | | same |
| `src/kernels/cpu/kq_avx2.cpp:20` | AVX2 K-quant kernels | | ggml-cpu NEON |
| `CMakeLists.txt:783-790, 844-852` | `-mavx512*`, `-mavx2 -mfma -mf16c`, MSVC `/arch:` | | Make them arch-conditional. On Apple use `-mcpu=apple-m1` (portable baseline) or `-mcpu=native`. |
| `include/strata/kernels/cpu/pool.hpp:236-264` | `alignas(64)` false-sharing pads | compiles | 128 on Apple Silicon (L2 line is 128 B); `std::hardware_destructive_interference_size` |
| `include/strata/hip_compat/intrinsics.hpp` | warp intrinsics shim | GPU | n/a (kernel side) |

### 8.2 OS-specific APIs on the host

| Site | API | OS | macOS status | macOS replacement |
|---|---|---|---|---|
| `pool.cpp:~95-136` | `GetLogicalProcessorInformationEx`, EfficiencyClass | WIN | compiled out | |
| `pool.cpp:145-160` | `/sys/devices/system/cpu/cpu%d/topology/{core_id,physical_package_id}`, `cpu_capacity` | LINUX (not `#ifdef`-guarded beyond `!_WIN32`) | runtime: the files are missing, so the result is a flat topology | `sysctlbyname("hw.nperflevels")`, `hw.perflevel0.physicalcpu` (P), `hw.perflevel1.physicalcpu` (E), `hw.physicalcpu`. No SMT. |
| `pool.cpp:165-172` | `sched_getaffinity`, `cpu_set_t`, `CPU_SETSIZE` | LINUX | **compile error** | Every logical CPU is usable: `hw.logicalcpu`. |
| `pool.cpp:267-311` | `pthread_setaffinity_np` / `pthread_getaffinity_np` / `SetThreadAffinityMask` (`pin_this_thread`, `pin_current_thread`, `restore_thread_affinity`) | LINUX / WIN | **compile error** | No-op. Use `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0)` for workers and the host thread. Optionally `thread_policy_set(THREAD_AFFINITY_POLICY)` as a tag (it is ignored on Apple Silicon). |
| `src/platform/direct_file.cpp:166` | `CreateFileW(NO_BUFFERING\|OVERLAPPED)` + IOCP | WIN | compiled out | |
| `direct_file.cpp:308` | `open(O_RDONLY\|O_DIRECT)` | LINUX | **compile error** | `open(O_RDONLY)`, then `fcntl(fd, F_NOCACHE, 1)` and `fcntl(fd, F_RDAHEAD, 0)`. Keep the `posix_memalign` buffers (`:36`) and `pread` workers (`:284`). |
| `src/core/expert_source.cpp:276` | `sysconf(_SC_AVPHYS_PAGES)` in the generic `#else` | non-Win, non-Linux | **compile error** (absent from the SDK) | host_statistics64 |
| `expert_source.cpp:1274` | `posix_fadvise(POSIX_FADV_DONTNEED)` under `!_WIN32` | POSIX, missing on macOS | **compile error** | Drop it (macOS has no per-range page-cache eviction), or `msync(MS_INVALIDATE)` on the mapping. `madvise(MADV_DONTNEED)` (`:1270`) compiles. |
| `expert_source.cpp:840-841` | `madvise(WILLNEED)` on an address aligned to a hardcoded 4096 | POSIX | **silent**: EINVAL on 16 KiB pages when the start is not 16 KiB-aligned; the return value is ignored | `const uintptr_t pg = (uintptr_t) getpagesize();` |
| `expert_source.cpp:413-423, 584-597` | `CreateFileMapping` / `MapViewOfFile` | WIN | compiled out | |
| `expert_source.cpp:445, 616` | `mmap(PROT_READ, MAP_SHARED)` | POSIX | OK | Wrap the mapping with `newBufferWithBytesNoCopy` (16 KiB-aligned, length rounded up). |
| `expert_source.cpp:776-823` | `PrefetchVirtualMemory` (`STRATA_FETCH_PVM`) | WIN | compiled out | `madvise(WILLNEED)` is the POSIX branch |
| `src/kernels/ngram.cpp:262-273` | `madvise` + `mlock` on a 4096-aligned range; on mlock failure, touch the pages | POSIX | **silent page-size bug** (4096 at `:263`); mlock of 28.8 GB fails on macOS, so it falls back to touching | `getpagesize()`. Do not lock: on unified memory, a wired 28.8 GB table competes with GPU residency. Prefer `--ple-io direct`. |
| `ngram.cpp:332-342, 421-432` | `PrefetchVirtualMemory` | WIN | compiled out | |
| `src/ngram/ple_reader.cpp:339` | `PAGE = 4096` row-read granularity | sector size | OK | Keep; it is the DirectFile sector, not the VM page. |
| `src/platform/memory.cpp:4-105` | `SetProcessWorkingSetSizeEx`, `VirtualLock`, DXGI `QueryVideoMemoryInfo` | WIN | compiled out | `gpu_shared_memory_budget` becomes `MTLDevice.recommendedMaxWorkingSetSize - currentAllocatedSize` |
| `memory.cpp:115-123` | `mlock` / `munlock` | POSIX | OK | Bounded by `ulimit -l` and the kernel wire limit; avoid it for large ranges. |
| `memory.cpp:133` | `sysconf(_SC_PHYS_PAGES)` | POSIX | OK (present) | Better: `sysctlbyname("hw.memsize")` |
| `src/core/conversation_memory.cpp:36-48` | `GlobalMemoryStatusEx` / `/proc/meminfo`; `#else return {}` | WIN / LINUX | **silent**: parking is never admitted on macOS | host_statistics64 (snippet in section 6.5) |
| `src/core/pinned.cu:31` | `<linux/mman.h>` | LINUX | n/a (`.cu`) | |
| `pinned.cu:60-110` | `SeLockMemoryPrivilege`, `VirtualAlloc(MEM_LARGE_PAGES)` | WIN | | |
| `pinned.cu:150-197` | `MAP_SHARED` arena file with pack-hash header | POSIX (Linux-tested) | would compile | unnecessary |
| `pinned.cu:202-210` | `MAP_HUGETLB \| MAP_HUGE_2MB` | LINUX | absent from the SDK | plain `mmap(MAP_ANON\|MAP_PRIVATE)` |
| `pinned.cu:319-400` | `cudaHostRegister` / `Unregister` | CUDA | | `newBufferWithBytesNoCopy` |
| `src/program/generate.cpp:104-112` | `under_wddm`: `_WIN32` or `/dev/dxg` | WIN / WSL | false on macOS | delete |
| `generate.cpp:716-754` | `mem_sample`: `K32GetProcessMemoryInfo` / `GlobalMemoryStatusEx`; `/proc/self/stat` (majflt at 728), `/proc/self/status` VmRSS / VmSwap (748-749), `/proc/meminfo` (750) | WIN / LINUX | **silent zeros** in stall reports | `task_info(mach_task_self(), TASK_VM_INFO)` (`phys_footprint`, `resident_size`), `getrusage(RUSAGE_SELF).ru_majflt`, `sysctlbyname("vm.swapusage")` (struct xsw_usage), host_statistics64 |
| `generate.cpp:787-806` | `MiniDumpWriteDump` (dbghelp) into `strata-stall-<pid>.dmp` | WIN | compiled out | See 9.3. |
| `generate.cpp:989-995` | `_putenv_s` / `setenv("CUDA_MODULE_LOADING","EAGER")` | | harmless | delete (CUDA-only) |
| `generate.cpp:1463` | `--ple-io ram` refused on Windows | WIN | | On macOS, allow it but warn about wired-memory limits. |
| `generate.cpp:2772-2780` | `GlobalMemoryStatusEx` commit-left in the cache-alloc failure message | WIN | | |
| `generate.cpp:4514-4560` | stdin reader thread with `read(0, ...)` | POSIX / WIN `_read` | OK | |
| `include/strata/artifact/gguf_reader.hpp:384-405` | `CreateFileMappingA` / `mmap(PROT_READ, MAP_PRIVATE)` | WIN / POSIX | OK | |
| `src/core/weights.cpp:78-82` | `_fseeki64` / `fseeko` | WIN / POSIX | OK | |
| `tools/vision/strata_vision.cpp:81-87` | `CUDA_VISIBLE_DEVICES=-1` when `--gpu` is absent | | harmless on macOS (Metal ignores it), since `cp.use_gpu = gpu` (`:97`) already keeps mtmd on the CPU | |
| `tools/vision/strata_vision.cpp:101` | `hardware_concurrency()/2` CPU threads | assumes SMT | runs at half the cores on Apple | Use `hw.perflevel0.physicalcpu`. |

### 8.3 CUDA host API sites (all of them [UMA-NOP] or a Metal rewrite)

Mapped or pinned host buffers that become shared MTLBuffers:
- `layer.cpp:620-635, 883, 998`
- `session.cpp:531, 843`
- `verify.cpp:72-74`
- `mtp.cpp:58-60`
- `native_head.cpp:133, 158, 173`
- `expert_cache.cpp:76`
- `weights.cpp:223-231`
- `prefill.cpp:194, 529, 575-576, 589`
- `remote_experts.cpp:190-192, 202-205`
- `expert_source.cpp:1134-1135, 1194, 1356, 2244`
- `generate.cpp:956` (PCIe probe), `:2001` (half dump), `:3323` (layer dump), `:4159` (split hand-off)

Device-side host-flag spins, which have no Metal equivalent with progress guarantees:
- `doorbell_wait` (`layer.hpp:402-422`, used by the token graph).
- `wait_flag_ge` / `wait_flag_ge_or` (`verify.cpp:731-779`).
- RemoteExperts spin (`STRATA_REMOTE_SPIN`, `remote_experts.cpp:77`).

Graph capture: `graph.hpp:41`, `session.cpp` token graph, `verify.cpp:907` (exec per T), `mtp.cpp` capture_round / capture_step.

---

## 9. Telemetry

### 9.1 `--stats` (one-shot) block, `generate.cpp:6236-6470`

Printed on stdout with a `%-24s` label column.

**Speculation lines** (with `--spec`)
- `<label> N rounds of S, drafts accepted a of o (rate), x tokens per round`
- `window sizes T1:n T2:n ... (min draft probability p)`
- `suffix drafts W windows, drafts accepted a of o`
- `accepted per round 0:n 1:n ...`
- `wait for rings %.3f pool %.3f host %.3f commit %.3f ms/round; CPU experts ...`
- rows phases: `gate/up %.3f quantize %.3f down %.3f ms/round; GB/s over the rows phases`
- `dispatch: plan / activation quantize / jobs / run ms/round`
- `adaptive tier: N experts swapped into the VRAM tier (every k rounds, ms/round)` [DGPU]
- `resident RAM: GiB in RAM, exchanged, blob reads from the file`
- `expert tiers: decode: RAM blobs, files blobs, MB read (MB/round ...)`
- `routing prefetch: N experts warmed, x of the file tier's y blob reads had been warmed (%)`
- `pcie experts: %.2f distinct experts per layer read over PCIe (share n/256 of the misses)` [DGPU]

**Token lines**
- `per token %.3f ms/token (WALL CLOCK: embed, layers, head, sample)`
- token host phases: `PLE embed LAYERS head readback sample ...`
- `CPU expert pool %.3f ms/token over L layers (positions, dispatches)`
- pool phases: `wait-park drain re-park ms/token` (`:6411-6417`)
- `expert blobs N blobs read`
- per-device `CUDAn experts` [DGPU]
- `R4 overlap: x of y layers the hit work was DONE when the pool returned` (`:6421`)
- `R4 expert-cache hits: h of l = rate (admitted, refused, cache % full)` (`:6422-6432`) [DGPU]
- `token graph: wait for rings %.3f pool %.3f ms/token (flushes over positions)` (WDDM flushes are [UMA-NOP])
- `rings seen MID-GRAPH x of y over positions`
- `ring latency %.3f ms of a %.3f ms layer`
- the head line `%.3f ms/token over positions (ms/layer, after the ring)`

**Other one-shot output**
- Dump summaries: `%-24s %s (%d x %lld, nonfinite %d, mean|.| %.4g, max|.| %.4g)`.
- Prefill: `prefill ... (time to first token)` (`:6331-6342`).

**Mac mapping**
- Keep the speculation lines verbatim; they compare directly across platforms.
- Replace the ring and PCIe lines with Metal timings. Use `MTLCommandBuffer.GPUStartTime` / `GPUEndTime`, and the `MTLCounterSampleBuffer` stage boundaries (`MTLCommonCounterSetTimestamp`) per layer.
- Add `os_signpost` intervals (`OSLog` subsystem `com.strata.engine`) per layer and per window, so Instruments' Metal System Trace aligns CPU and GPU.

### 9.2 Serve-side telemetry

- **INFO line** and the per-request stderr lines after DONE (section 1.4).
- **`STRATA_DECODE_TIMING`** (`:5305`): `strata decode timing:` per request.
- **`STRATA_SPLIT_TIMING`** (`:5605`) [DGPU].
- **`STRATA_TRACE`** (`:811`, `mem_mark`): VRAM free at startup steps; also prefill progress (`prefill.cpp:1052`).
- **`STRATA_POOL_TRACE`** (`expert_source.cpp:1611`): `pool trace: layer L what n`.
- **`STRATA_VERIFY_PROFILE`** (`verify.cpp:331`), **`STRATA_VERIFY_NODES`** (`:907`), **`STRATA_PREFILL_TIMING`** (`prefill.cpp:960`).
- **State fingerprints**: `STRATA_STATE_HASH` / `_GDN` (`:5455, 5497`).
- **Routing trace**: `--dump-routing F`, flushed after every DONE (`:5577`).

### 9.3 Watchdog and stall dumps (issues #29, #31, #267)

**Progress model** (`include/strata/core/progress.hpp:17-57`)
- `beats` advance per token, prompt chunk or verify window (`progress_beat`).
- `ticks` count layers served (`progress_tick`).
- Other fields: `busy`, `where` / `detail` / `chunk` (stage text) and `since_ms`.
- `BusyScope` brackets each request.

**Watchdog thread** (`generate.cpp:4634-4656`)
- `STRATA_WATCHDOG_S` defaults to 60; 0 turns it off. It checks once per second.
- If `busy` and `beats` has not changed for `limit` seconds, it prints `strata serve: no progress for N s during a request (<stage>) - stopping the engine so the server starts it again (issue #29)`.
- It then calls `stall_report`, then `release_gpu_waits(stderr)` (#267: raises every device flag to UINT32_MAX so no spin kernel outlives the process), then `std::abort()`.

**`stall_report`** (`:763-808`)
- Header: `strata serve: stall report (engine V): stage "..." for N s; L layers served since the last finished step (0 = stopped, more = slow)`.
- Two passes, 2 s apart. Each pass prints:
  - the pool diag (`diag_pool_fn`): workers parked, jobs done;
  - the verify diag (`diag_verify_fn`, `verify.cpp:159`): the last window's layer, group and flags;
  - the memory line `memory: R MiB resident, S MiB in swap, A MiB RAM available; F major page faults so far`. **On macOS this line is all zeros** (section 8.2).
- On Windows it writes a minidump `strata-stall-<pid>.dmp` with thread info (`:787-806`).

**Pool self-watchdog** (`pool.cpp:478-510`, `kStall` 60 s)
- `strata: the CPU expert pool stalled ... (k of n workers parked) - stopping the engine ... (issue #29)`.
- Or `... u of n jobs done ...`, followed by abort.

**Verify host-loop timeout** (`verify.cpp:1050-1135`)
- After 20 s without a ring, it calls `release_gpu_waits` and fails the window.
- `STRATA_TEST_VERIFY_STALL` (`:131`) injects a stall for tests (#267).

**Server side**
- `death_note` greps the log for "issue #29" to explain a restart.
- `ERR` lines become ValueErrors.
- The engine is restarted on the next request.

**Mac equivalents**

| Windows / Linux mechanism | macOS |
|---|---|
| minidump | Use `backtrace()` / `backtrace_symbols_fd()` for the watchdog thread itself, but other threads' stacks need Mach: `task_threads` + `thread_get_state(ARM_THREAD_STATE64)` + frame-pointer walk (arm64 macOS always keeps frame pointers). Simpler: let `abort()` raise SIGABRT, so macOS writes a full crash report with every thread to `~/Library/Logs/DiagnosticReports/strata-*.ips`. Print that path hint in the report. |
| `release_gpu_waits` | With MTLSharedEvent waits: set `event.signaledValue = UINT64_MAX`, which releases every encoded wait. Otherwise `[commandQueue ...]` has no cancel; a stuck command buffer is killed by the GPU watchdog (the error is `MTLCommandBufferErrorTimeout`). |
| memory line | `task_info(TASK_VM_INFO).phys_footprint`, `vm.swapusage`, host_statistics64, `ru_majflt` |
| GPU fault reporting | Check `MTLCommandBuffer.status` / `.error` in completion handlers. Enable `MTLCommandBufferDescriptor.errorOptions = MTLCommandBufferErrorOptionEncoderExecutionStatus` in debug builds to name the failing encoder. |

---

## 10. Port recommendations and risks (host side)

### 10.1 What to keep (portable, high value)

| Component | Notes |
|---|---|
| The serve protocol (section 1.4) and `serve/server.py` | Keep the INFO keys and the READY `stop` capability. |
| `src/spec/*` (suffix drafter, draft policy) | Already builds and passes on macOS. |
| `include/strata/core/coupled_draft.hpp`, `include/strata/program/conv_cache.hpp`, `conversation_cache.hpp`, `conversation_buffer.hpp` | Header-only, tested. |
| `include/strata/artifact/*` (GGUF reader and split view) and `strata-gguf` / `strata-dequant` / `strata-plan` | |
| `src/ngram/ple_reader.cpp` + `src/platform/direct_file.cpp` | With `F_NOCACHE`. The PLE table stays on the SSD and is read per token; it is 28.8 GB that does not need RAM. |
| `src/kernels/cpu/native_expert.cpp` | ggml-cpu NEON; it compiles. Useful only if CPU expert compute is kept. |
| The pinned ggml checkout (`3cf03257f`) | It already includes Metal mat-vec and mat-mat kernels for Q2_0 and every IQ type the packs use. The kernel-side port can borrow them or treat them as parity oracles; on Apple, ggml-metal is configured automatically by the existing FetchContent. |
| Native-pack artifacts (`index.txt` v3, `dense.bin`, `extra.bin`, `native_experts.txt` v4), GGUF shards, STRP profiles, `mtp/rt`, control vectors, `draft_vocab.bin` | |
| The exactness contract | Per-row deterministic reductions, Philox(seed, pos), accept-on-equality, commit/replay rollback. |
| The watchdog model (Progress beats/ticks plus a release-all-waits escape) | |

### 10.2 What to delete on unified memory

| Area | Items |
|---|---|
| Expert transfer | PinnedArena registration; large pages / hugetlb; partial pin and complement pinning; exchange buffers; adaptive swap *copies*; slot lending to prefill; `--expert-cache` sizing via cudaMemGetInfo; `--vram-reserve-mib`. |
| PCIe | `pcie_frac` / `pcie_mode` / `flagB` / DMA staging; `probe_pcie_h2d_gbps`; `strata-concurrent` / `strata-overlap`. |
| Windows driver workarounds | WDDM flushes (`STRATA_TG_FLUSH_US`, `cudaStreamQuery` pokes, `under_wddm`); `CUDA_MODULE_LOADING`. |
| Multi-GPU | RemoteExperts; layer split; `OnDevice`; `--kv-resident` KV streaming. |
| Staging copies | Mapped staging copies; the CPU writes directly into shared MTLBuffers. |

### 10.3 What to redesign

1. **Expert placement.** Default to GPU-only experts reading straight from the mmapped GGUF (`newBufferWithBytesNoCopy`, one buffer per shard, or per `maxBufferLength` slice; check `MTLDevice.maxBufferLength`, typically about half of RAM on recent macOS).
   - Keep `FileExpertSource`'s *policy* (ram_reads / file_reads tiers, RouterLookahead) only for machines where experts + dense + KV exceed `recommendedMaxWorkingSetSize`. There, use the STRP profile and the usage counters to decide what to keep resident (MTLResidencySet) and what to page in with `madvise(WILLNEED)`.
2. **The CPU/GPU handshake.** Replace device spin-waits with either no handshake (GPU-only experts) or `MTLSharedEvent` waits per layer (option B in 4.10).
   - Measure event round-trip latency first. With 48 layers per token, more than ~20 us per hop costs most of the benefit.
3. **The CPU pool, if kept.**
   - P-core count from `hw.perflevel0.physicalcpu`; QoS instead of affinity; `yield`/`wfe` instead of `_mm_pause`; 128-B padding.
   - NEON or ggml kernels in place of the AVX ones.
   - Note that the CPU and GPU share one memory bus. The CPU pool helps only if the GPU leaves bandwidth unused.
4. **Graph capture.** Rebuild the token graph and the per-T verify graphs as pre-encoded command sequences:
   - an indirect command buffer, or Metal 4 argument tables plus re-encoding;
   - per-token constants (`pos`, `tok`) live in a shared buffer, the way the mapped staging works today.
5. **Memory probes.** One `platform::available_memory()` with an `__APPLE__` branch (host_statistics64), used by `expert_source.cpp:219-283`, `conversation_memory.cpp:36-48` and `mem_sample`. Plus `platform::gpu_budget()` from `recommendedMaxWorkingSetSize - currentAllocatedSize`.
6. **Page size.** Replace every hardcoded 4096 that is a VM page with `getpagesize()`:
   - `expert_source.cpp:840`
   - `ngram.cpp:263`
   - any `newBufferWithBytesNoCopy` alignment (16384)
7. **generate.cpp.** Do not port its 6487 lines verbatim. About half is multi-GPU, PCIe, VRAM-sizing and Windows policy. Write a slim Metal driver that reuses option parsing (`Options` `:160-415`, parse `:999-1255`), the serve loop structure (`:4510-5610`), the prompt-cache policy (`:4318-4410, 4858-5009, 5212-5240`) and the spec loop (`:5322-5430`).

### 10.4 Risk register

| Risk | Impact | Mitigation |
|---|---|---|
| GPU spin on CPU-written memory has no forward-progress guarantee and may hit the GPU watchdog | hangs or killed command buffers | Do not port `doorbell_wait` / `wait_flag_ge`; use events or GPU-only experts. |
| Wired-memory limit for GPU-resident buffers (`recommendedMaxWorkingSetSize`, about 65-75% of RAM by default) | 34-50 GB of experts plus KV may exceed it on 48-64 GB Macs | Profile-ranked residency, the SSD tier for the cold tail, and document `iogpu.wired_limit_mb`. |
| `newBufferWithBytesNoCopy` requires page-aligned pointers and lengths | GGUF tensor offsets are 32-B aligned | Wrap whole mmapped shards and address tensors by offset. |
| `maxBufferLength` limits | One 35-77 GB buffer may exceed it | One buffer per shard (the shards are ~10-20 GB), or split. |
| mlock / `--ple-io ram` / `STRATA_ARENA_LOCK` on macOS | fails above the wire limit | Disable by default and warn. |
| Silent zeros (`mem_sample`, `conversation_available_memory`) | Parking never admitted; useless stall reports | Darwin branches (8.2). |
| Bit-exactness of the verify window vs T = 1 decode | Speculation changes the output | Keep per-row reductions batch-invariant; reuse the spec exactness ctests. |
| 16 KiB pages | `madvise` EINVAL ignored; prefetch silently off | `getpagesize()` |
| `pool.hpp` 64-B padding on 128-B lines | false sharing between pool atomics | 128-B alignment |
| `STRATA_PORTABLE` forces GGML_AVX* | broken ggml-cpu on arm64 | Gate by architecture. |
