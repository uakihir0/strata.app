> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Part 05: Decode/prefill orchestration, discrete-VRAM assumptions, and the test strategy

Scope: how Strata runs one speculative decode step, the expert cache, prefill, every assumption that comes from
discrete VRAM, PCIe, or the CUDA runtime, the HIP shims, and the test suite, all read with the Metal / Apple
Silicon port in mind.
The source is the upstream clone at `/tmp/strata-upstream`, which was read only and never modified. All `file:line`
references are relative to that root.

Notation used throughout:

- T is the verify window size, from 1 to 8 tokens.
- N = n_embd = 2560.
- K = 10 routed experts per token.
- NE = 512 experts per layer.
- HC = 4 hyper-connection streams.
- FF = 640 expert intermediate width.
- The canonical Q2_0 expert blob is `BLOB` = 1,382,400 B (`include/strata/kernels/cpu/expert.hpp:34-45`).
- "Mapped" means `cudaHostAlloc(..., cudaHostAllocMapped)` plus `cudaHostGetDevicePointer`. This is pinned host
  memory that the GPU reads and writes across PCIe (`src/core/verify.cpp:72-76`).

---

## 1. The decode step, end to end

### 1.1 Actors

| Actor | What it is | Where |
|---|---|---|
| Main host thread | Runs the CLI and serve loops. It also serves the doorbell and drains pool jobs itself. | `src/program/generate.cpp:6128-6232` (CLI), `:5339-5429` (serve) |
| Expert pool workers | Compute the CPU share of the routed experts (AVX2/VNNI/AVX-512 kernels). | `src/kernels/cpu/pool.cpp`, `include/strata/kernels/cpu/pool.hpp` |
| `cs_` | Compute stream of the verifier (non-blocking). It runs the window graph and the commit graph. | `src/core/verify.cpp:347` |
| `copy_` | Copy stream of the verifier (non-blocking). It carries the DMA of the "PCIe share" and the host function that raises flag B. | `src/core/verify.cpp:343` |
| MTP stream | The drafter's own stream and graphs (`step_exec_[9]`, `round_exec_[9]`, `prefill_exec_[9]`). | `include/strata/core/mtp.hpp:100-108`, `src/core/mtp.cpp:590-661` |
| Adapt thread + `adapt_stream` | Swaps VRAM-cache residency while the GPU commits and drafts. | `src/program/generate.cpp:6195-6197`, `:4459-4512` |

### 1.2 One round in the CLI loop (`src/program/generate.cpp:6128-6232`; serve is the same at `:5339-5429`)

1. **Choose T.**
   - T starts at `S_mtp = min(mtp_max_t, spec)` (`:6117`).
   - With `--spec-min-p` it shrinks to the longest prefix whose draft probability is ≥ min_p (`:6131-6134`).
   - The first window of a native pack is T=1 (`:6107-6108`, `:6135`). It runs the last prompt token and seeds the
     MTP's first cell.
   - With `--suffix-draft`, the trigram `SuffixDrafter` proposes a draft (`:6139-6146`). `DraftPolicy::choose`
     (`include/strata/spec/draft_policy.hpp`) picks MTP or lookup by online expected-accepts per cost.
   - When the suffix drafter is on, setup raises `spec` to `min(spec+2, 8)` and `mtp_max_t=spec`
     (`generate.cpp:1493-1496`). Lookup windows can therefore be 6 tokens long, while MTP windows stay at the
     configured 4 (the setup default is `--spec 4`).
2. **Build the window.**
   - `window[0] = x` is the last accepted token. `window[1..T-1]` are the drafts.
   - `--spec-corrupt N` replaces every Nth draft with `(d+1) % vocab` (`:6158`). This is the exactness test hook.
3. **`apply_pending(false)`** (`:6164`). This admits finished adaptive swaps: if `cudaEventQuery(adapt_ev)` is
   complete, it sets `host_res[i] = slot` and re-uploads `d_res` (`:4443-4457`).
4. **`ver.run(T, window, p, &drive_pool_multi, &drive, outv)`** (`:6165`). This is the whole verify window; see 1.3.
5. **Accept.** `a` = the count of leading matches with `window[a+1] == outv[a]` (`:6177-6178`). `outv[0..a]` are
   the emitted tokens.
6. **Adaptive tier.** Every `adapt_every` rounds (default 4, `:359`), `adapt()` starts on its own thread
   (`:6195-6197`). See section 2.5.
7. **`ver.commit(a+1)`** (`:6198`). This launches the captured commit graph, asynchronously; see 1.9.
8. **Emit** `outv[0..a]` and feed it to the suffix drafter (`:6207-6211`).
9. **`mtp.draft(T, outv, p, a, drafts, ...)`** (`:6218-6219`).
   - First it catches up over the window's T cells, using the verifier's final residuals `final_R`.
   - Then it chains T-1 drafts from row `a`.
   - It runs on the MTP stream, overlapping the async commit. This is safe because the drafter reads only this
     window's final rows and its own K/V (`verify.cpp:1330-1332` comment).
10. Join the adapt thread. Then `x = outv[a]` and `p += a+1` (`:6225-6226`). `DraftPolicy::observe` learns the
    round cost (`:6229`).
11. After the loop, `ver.wait_commit()` (`:6236`) does `cudaEventSynchronize(commit_done_)`.

### 1.3 `Verifier::run` (`src/core/verify.cpp:1016-1204`)

1. **Lazy capture.** `capture(T)` builds one CUDA graph per T (`exec_[9]`, `:887-948`). `capture_commit()` builds
   the commit graph (`:950-1014`). Capture uses `cudaStreamCaptureModeThreadLocal`, then `cudaGraphInstantiate`,
   then `cudaGraphUpload` plus a sync.
2. **Stage the inputs in mapped memory** (`:1029-1047`).
   - `h_tok_[t]`.
   - `h_step_` through `qsa_step_fill`.
   - `h_pos_`: per head, per KV head, and per indexer head, all equal to `pos0+t`.
   - PLE n-gram rows gathered into `h_ple_` (`ss.ple.table->gather_batch`).
3. **Reset the doorbell words.** `h_seq_`, `h_flag_`, `h_flagA_` and `h_flagB_` are all set to 0 with volatile
   stores, followed by `std::atomic_thread_fence(seq_cst)` (`:1048-1052`).
4. **Launch.** `cudaGraphLaunch(exec_[T], cs_)` is followed by `(void) cudaStreamQuery(cs_)` (`:1058-1060`). The
   query exists only to flush the WDDM submission queue.
5. **The host doorbell loop** (`:1068-1121`). There is one iteration per ring, so `steps = (le_-lb_) * G` rings.
   For ring `k`, `want = k+1`:
   - **Spin.** `while (*seq < want) { _mm_pause(); ... }`. Every 1024 spins it reads the clock.
     - Every 2 ms it calls `cudaStreamQuery(cs_)`. If the graph has finished or faulted without ringing, the window
       fails (`:1080-1088`).
     - After 20 s it calls `release_gpu_waits(5000)` (#267, `:1089-1093`).
   - `cur_layer_ = want-1`; `set_plan_slot(grp)` points `sink_` at this group's half of the mapped plan
     (`:1226-1243`).
   - **Pool call.** `pool(user, h_x_+tb*N, h_ids_+tb*K, n, K, h_ymiss_+tb*K*N, l)` (`:1102-1104`). This is
     `drive_pool_multi` → `expert_pool_dispatch_multi` (section 1.6). It publishes flag A and starts flag B from
     inside, then computes the CPU share synchronously.
   - Then `std::atomic_thread_fence(seq_cst); _mm_sfence();` (`:1106-1107`).
   - **Empty-plan fallback.** If the pool did not publish (flag A != want), the host writes an empty plan
     (`counts[0..2]=0`, `start[0]=start2[0]=0`), sets flag A = want and calls `raise_flag(h_flagB_, want)`
     (`:1108-1117`).
   - `*flag = want` (flag "M": the CPU rows are in `h_ymiss_`) (`:1118`). The #267 test hook withholds this on the
     last ring.
6. **Finish.**
   - `cudaStreamSynchronize(cs_)` (`:1128`). This is a blocking sync, because a polling variant cost IQ3_S about
     3% (`:1124-1127`).
   - `cudaStreamSynchronize(copy_)` (`:1131`), so that no host function from this window can raise flag B in the
     next one.
7. **Sampling.**
   - Greedy is captured inside the graph.
   - A sampled or penalized request runs `sample_tokens` again over `head_logits_` with
     `sp.counter = pos0` (Philox keyed to `(seed, pos0+t)`), followed by another `cudaStreamSynchronize`
     (`:1171-1181`).
   - The result is read as `out[t] = ((volatile int32_t*) h_out_)[t]` (`:1182`).
8. **Multi-GPU layer split.** When `le_ < n_layers`, this stage copies its residual to `hand_out_` and calls
   `next_->run(...)` (`:1166-1169`). This path should be dropped on Mac.

### 1.4 The captured window graph (`record_window`, `src/core/verify.cpp:387-859`)

**Inputs** (`:410-417`). These are mapped-to-device copy kernels inside the graph:

- `copy_i32_from_mapped(tok_, m_tok_, T)`.
- The same for `step_` and `pos_`.
- `copy_from_mapped(ple_, m_ple_, T*N)` when PLE is on.

**Embeddings** (`:421-445`).

- In a later layer-split stage, the input instead comes from `hand_in_`.
- Otherwise it is `NativeEmbed::gather_dev`, or the S2/S4/S8 `embedding_gather_dev`.
- Then `broadcast_streams` writes the result to the HC=4 residual streams `R_`.

**Groups.** `G = (split_ && T>=2) ? 2 : 1` (`:403`).

- Group A covers tokens `[0, (T+1)/2)` and group B the rest (`:406`).
- Ring value: `ring = (l-lb_)*G + grp + 1` (`:728`).

**`pre(l, grp)`** (`:457-722`) runs everything up to and including the doorbell:

1. Load the GR weights. Layer 1 with PLE first applies `gr_write` + `ple_block` + `ple_history_advance` and
   snapshots `hist_snap_` (`:473-490`).
2. `fused_gr_read_multi`: the attention half of the hyper-connection read (`:491-507`). It folds in the previous
   layer's pending FFN write.
3. **GDN layer** (`:511-544`):
   - `native_quantize_q8_1`, then `native_mmvq` for qkv.
   - `gdn_conv_l2_multi`.
   - `gdn_ab_multi`.
   - `mmvq` for the gate z.
   - `gdn_step_norm_multi`. This is the recurrence from the untouched state over tokens `[0, te)`. Outputs are
     written only for this group, and the state is NOT advanced; commit does that.
   - q8_1 + `mmvq` for `ssm_out` into `bo_`.
4. **QSA layer** (`:545-659`):
   - The indexer K projection (bf16 gemv).
   - K/V `mmvq` + norm + RoPE, with optional FWHT rotation.
   - Snapshot the indexer tail (`tail_snap_`, group 0 only).
   - `kv_append_*_step` per token into the paged K/V. KV streaming also writes the host copy `&st.host`.
   - `native_qsa_indexer_append`.
   - The Q projection.
   - Split q/gate via `cudaMemcpy2DAsync` D2D (`:610`, `:623`).
   - `qsa_block_scores` + `qsa_block_topk`.
   - `qsa_kv_resolve`: the in-graph KV-streaming page-in (`:641`).
   - `qsa_decode_attn_batch`.
   - Gate, then `mmvq` for the output into `bo_`.
5. `fused_gr_read_multi`, the FFN half (`:666`).
6. **Router** (`:669-682`): one bf16 gemv over the window's rows, then `native_router_top10_multi`. The fallback
   runs `moe_route` per token.
7. **E-6 (opt-in, `STRATA_VERIFY_DEVICE_PLAN=1`):** `resident_plan` (`:683-688`). See 1.8.
8. **The doorbell.** `doorbell_publish(xm, ids, w, n*N, n*K, m_x_, m_ids_, m_w_, m_seq_)` (`:689-690`). This rings
   the host.
9. **The shared expert, while the host works** (`:692-712`): `f32_to_bf16_bulk` + `shared_expert_multi` (gate-inp
   sigmoid, gate/up/down `mmvq`) into `shared_`.
10. **Activation quantization for the GPU expert kernels** (`:713-716`): `quantize_q8_1_rows` (native pack) or
    `quantize_q8_0_scaled` (Q2_0).

**`post(l, grp)`** (`:725-806`) is the MoE tail:

1. `wait_flag_ge(m_flagA_, ring)` spins until the plan is published. Then `copy_i32_from_mapped(pl, m_plan_+grp*plan_i32_)`
   (`:731-737`).
2. `grouped(p_ptr, p_start, p_counts)` runs the **VRAM groups** (`:748-760`):
   - `native_expert_grouped` (`src/kernels/cuda/iq_kernels.cu:1442`), or
   - `moe_grouped_s2` (`src/kernels/cuda/s2_expert_grouped.cu:1099`, kernels `gu_grouped_kernel:783`,
     `down_grouped_kernel:845`, `gu_grouped_t_kernel:918`, `down_grouped_t_kernel:984`).
3. `wait_flag_ge(m_flagB_, ring)`: the PCIe share has landed (`:763-764`).
   - In `pcie_mode == 2`, `fetch_blobs` (copy kernel from mapped host into `staging_`,
     `verify_kernels.cu:218-226`, `<<<384,256>>>`) runs first, then `rebase_ptrs` (`:228-231`) (`:765-770`).
4. `grouped(p_ptr2, p_start2, p_counts+2)` runs the **PCIe groups** (`:772`).
5. `wait_flag_ge(m_flag_, ring)`: the CPU rows are ready (`:780`).
   - Then `copy_rows_from_mapped(parts_, m_ymiss_, n*K, N, p_dst, p_counts+1)` (`:782-784`;
     `elementwise.cu:235-253`). Only CPU rows cross PCIe, and rows the GPU computed are written as +0.0.
   - With `STRATA_DEC_BATCH=0` it is `copy_from_mapped` of all rows instead.
6. `moe_hit_add(parts_, hit_out, p_dst, counts+1)` adds the GPU-computed rows into `parts_` (`:788`;
   `s2_expert_grouped.cu:1138`, `add_hits_kernel:666`).
7. `native_moe_combine_multi(parts_, w_, shared_, bo_)`: Σ_k w_k·parts_k + shared → `bo_` (`:789-797`;
   `native_moe.cu:77`).
8. On the last layer, `gr_write` for each token, then `cvec_apply` (`:799-804`).

**Loop order** (`:808-814`):

```
for grp: pre(lb_, grp)
for l in [lb_, le_): for grp: post(l, grp); if l+1<le_: pre(l+1, grp)
```

- With G=1 the order is `pre0 | post0 pre1 | post1 pre2 ...`. The GPU's next-layer attention runs only after this
  layer's MoE combine, because it needs `bo_`.
- With G=2 the order is `pre(0,A) pre(0,B) | post(0,A) pre(1,A) | post(0,B) pre(1,B) | ...`. While the host and CPU
  serve group B's experts, the GPU runs group A's MoE tail and group A's next-layer attention.
- `split_` is opt-in (`--spec-split`). The paper's Finding 5 measured it about 7% slower (85.8 vs 91.8 tok/s), so
  the default is G=1.

**Head** (`:824-856`):

- `lm_head_mix` per token: the output hyper-connection norm/down/up.
- `native_quantize_q8_1` + `native_mmvq` over T columns against the native head. The fallback is the per-token
  `lm_head`.
- `sample_tokens(head_logits_, T, n_vocab, greedy) → m_out_`. Greedy argmax is baked into the graph.

**Profiling.** `gpu_stamp` (`verify_kernels.cu:546-557`) uses `%globaltimer`, or `wall_clock64()*10` on HIP. It is
on only with `STRATA_VERIFY_PROFILE` and writes 32 stamps per layer (`kProfPer`).

### 1.5 The doorbell protocol: every shared word and buffer

All of the buffers below are allocated in `Verifier::init` with `mapped()` (`verify.cpp:250-264`). They are memset to
0 and given a device alias through `cudaHostGetDevicePointer`.

| Buffer (host `h_` / device `m_`) | Size | Writer | Reader | Ordering / visibility |
|---|---|---|---|---|
| `tok` | T·4 | host, in `run()` before launch | graph start, `copy_i32_from_mapped` | plain stores; `seq_cst` fence + `cudaGraphLaunch` order them |
| `step` | T·kStepCount·4 | host (`qsa_step_fill`) | graph start | same |
| `pos` | T·(NH+NKV+IQ)·4 | host | graph start | same |
| `ple` | T·N·4 | host (PLE table gather) | graph start (`copy_from_mapped`) | same |
| `commit` | (2+T)·4+16 | host, in `commit()` | commit graph start | `seq_cst` fence, then `cudaGraphLaunch(commit_exec_)` (`:1327-1328`) |
| `out` | T·4+16 | GPU (`sample_tokens` in the graph, or the host-side resample) | host, through a `volatile` read | after `cudaStreamSynchronize(cs_)` (`:1128`, `:1182`) |
| `x` | T·N·4 (f32) | GPU, `doorbell_publish_kernel` | pool (CPU experts, activation quantization) | GPU: `__threadfence_system(); __syncthreads();` then a fence + volatile `seq++` by thread 0 |
| `ids`, `w` | T·K·4 each | GPU, same kernel | pool (planner) | same |
| `seq` | 64 B | GPU, thread 0 of `doorbell_publish_kernel` (`elementwise.cu:276-287`): `__threadfence_system(); *(volatile uint32_t*)seq = *(volatile)seq + 1` | host spin: `while (*seq < want) _mm_pause()` | increment-in-memory, never a captured literal (`elementwise.cu:186-208` explains the bug) |
| `flagA` (plan ready) | 64 B | host, `publish_plan` (`:1255-1259`): `_mm_sfence(); *(volatile)flagA = cur_layer_+1`; or the empty-plan fallback in `run()` | GPU, `wait_flag_ge_kernel` <<<1,1>>> (`verify_kernels.cu:426-429`) | GPU spin: `while (*flag < v) strata_spin_pause(); __threadfence_system();` |
| `plan` | 2·plan_i32_·4+64 | host, `sink_` arrays written by the planner (`expert_source.cpp:1622-1702`) | GPU, `copy_i32_from_mapped` after flag A | `seq_cst` fence before `publish` (`expert_source.cpp:1702`) |
| `flagB` (PCIe share in staging) | 64 B | host, `raise_flag` (`:1225-1238`), a monotonic CAS (`_InterlockedCompareExchange` / `__atomic_compare_exchange_n`). It is called from `fetch_dma` right away when n≤0, else from a `cudaLaunchHostFunc` on `copy_` after the H2D copies (`:1243-1253`) | GPU `wait_flag_ge` | monotonic: a late host function may run after a later layer already raised it |
| `ymiss` | T·K·N·4 (409.6 KB at T=4) | CPU pool. The down-row phase writes output rows directly; GPU/PCIe/remote rows are memset to 0 | GPU `copy_rows_from_mapped` after flag M | host: `seq_cst` fence + `_mm_sfence` before `*flag = want` (`:1106-1118`) |
| `flag` (M, CPU rows ready) | 64 B | host: `*flag = want` (`:1118`) | GPU `wait_flag_ge` (`:780`) | same |
| `skip_[2]` (E-6) | 64 B, **device** memory | GPU `resident_plan_kernel` (`verify_kernels.cu:433-473`): `__threadfence(); *skip = ring` | `wait_flag_ge_or`, `copy_i32_unless`, `copy_or_zero` (`:474-490`) | device-scope only |
| `staging_` | 16 × max_blob, **device** (`verify.hpp:256`, `verify.cpp:312`) | copy engine (mode 0) or `fetch_blobs` (mode 2) | PCIe grouped kernels | flag B, or stream order |

Protocol rules worth keeping:

- **Rings are monotonic within a window, and all four words are reset to 0 at window start** (`:1048-1052`). The
  GPU waits for `>= ring`, not `==`, so a late raise is harmless.
- **The GPU spin is one thread** (`<<<1,1>>>`). It occupies an SM while it waits. On CUDA,
  `strata_spin_pause` = `__nanosleep(100)` on sm_70+ (`include/strata/kernels/dp4a.hpp:36-42`). Graph order makes
  every later kernel wait on it.
- **Volatile matters on RDNA4.** A plain store to mapped memory stayed in L2 and the host saw 0 of 100 rings
  (`elementwise.cu:203-206`, `tests/hip/handoff`). With volatile, a system-scope atomic, or a fence after the store,
  it saw 100 of 100.
- **The fence matters.** Without `__threadfence_system` the host compensated with a `cudaEventQuery` per spin, at
  10-22 µs per driver call, which cost about 10 ms per token (`elementwise.cu:192-201`).
- **#267 watchdog.** `release_gpu_waits` writes `UINT32_MAX` into flag, flagA and flagB, then polls both streams
  for up to 5 s (`verify.cpp:136-155`). Without it, a GPU stuck in a spin kernel on Windows was "lost until a power
  cycle".
- The older session paths used the same doorbell idea: `doorbell_ring_kernel` / `doorbell_wait_kernel`
  (`elementwise.cu:209-218`), `TokenGraph` in `session.cpp:849-951`, and `SessionGraphs` with `cudaEventQuery`
  polling (`session.cpp:682`). The verifier supersedes both.

### 1.6 How the CPU pool produces its rows (`expert_pool_dispatch_multi`, `src/core/expert_source.cpp:1588-1807`)

The function is called on the host thread once per ring, through `drive_pool_multi` (`generate.cpp:598-660`). It runs
these steps in order:

1. **Guards.**
   - `n_tok ≤ MAXT=8` (`cpu/expert.hpp:131`).
   - `n_tok·k ≤ kMaxWindowEntries=128` (`:1584`).
   - The CS-T lookahead prefetches the next layer (`d.lookahead->submit`, mmap mode).
   - `src->begin_layer`.
2. **Usage counting** for the adaptive tier: `d.usage[l*NE+e] += 1` for every routed entry (`:1621-1623`).
3. **Plan, published first** (`:1627-1706`). The plan is decided before any CPU work so the GPU starts early.
   - Build the distinct experts in routing order, plus `first_of[]`, and count the misses (`host_res[...] < 0`).
   - `m = (nmiss * pcie_num) >> 8`, where `pcie_num = pcie_frac·256` (`generate.cpp:4423`). It is 0 unless
     `src->pcie_layer(l)`, i.e. the layer's experts are pinned.
   - Each distinct expert gets a kind:
     - **resident** (slot ≥ 0) → kind 0 (VRAM). `ptr = cache_base + slot_off[slot]`, or `slot·blob` (`:1656-1659`).
     - **one of the last m misses**, pinned, with `fetches < staging_cap` (16, or 8 per group when G=2) and
       `< 64` → kind 1 (PCIe), with `dma_src` set to the host blob (`:1661-1670`).
     - otherwise → kind -1 (CPU).
   - VRAM groups fill `ptr/start/dst/tok` first (`:1676-1686`). PCIe groups follow, appended after the VRAM entries:
     `ptr2 = staging + q·bb` (mode 0), or `device_alias(l,e)`, the mapped host address (modes 1/2)
     (`:1689-1700`).
   - `counts = {vram_groups, entries, pcie_groups}`. Then `atomic_thread_fence(seq_cst)` → `P.publish(ctx)`, which
     raises flag A → `P.fetch(ctx, dma_src, mode!=0 ? 0 : fetches, bb)`, which queues the DMA or raises B at once
     (`:1701-1705`).
4. **Remote GPU experts**, kind 2: a second GPU computes some experts (`remote_experts.cpp`, `:1713-1721`).
5. **Activation quantization** of `x` once per token: q8_1 for native packs, the legacy `ActQ`, or `act_quant_any`
   for Q2_0 (`:1723-1729`).
6. **CPU jobs.** `out` rows of every non-CPU entry are memset to 0. CPU entries are grouped per expert into
   `ExpertJobMulti` with `nt` tokens each (`:1740-1781`).
7. **Run.** `pool->run_split_multi` (canonical) or `run_split_multi_native` (`pool.cpp:624-692`), in three phases:
   - Gate/up rows for all jobs (`3·threads` tasks).
   - Quantize the intermediate.
   - Down rows. These write **directly into `out` = mapped `h_ymiss_` rows**.
8. **Pool mechanics** (`pool.hpp`, `pool.cpp`):
   - A flat job batch with a 64-bit `head_` = `epoch<<32 | njobs<<16 | index`, claimed by CAS (issue #29). Atomics
     are `alignas(64)`.
   - The host thread works too.
   - Parked workers spin on `_mm_pause` for 20 ms, then sleep on a condvar. The stall timeout is 60 s.
   - Affinity uses `SetThreadAffinityMask` / `pthread_setaffinity_np`, and the host is pinned to core 0
     (`pool.cpp:270-309`).

### 1.7 How the VRAM, PCIe and CPU results are combined

- Every routed entry `i ∈ [0, n·K)` has exactly one producer. GPU groups (VRAM, then PCIe) write rows of `hit_out_`
  at `dst[]`. The CPU writes rows of `h_ymiss_`. Remote GPUs write rows of `out` too.
- `copy_rows_from_mapped` brings in only the rows that are not GPU-computed. GPU rows become +0.0, matching the
  pool's memset. `moe_hit_add` then adds the GPU rows. The result is `parts_[i]` = one expert output per entry, in
  routing order.
- `native_moe_combine_multi` reduces in fixed `k` order, with weights `w_` plus the shared expert. **The summation
  order does not depend on which device computed an expert.** The per-expert arithmetic does differ, however:
  - The CPU kernels and the GPU grouped kernels are separate implementations.
    `moe_hit_grouped_s2_cpu_order` (`s2_expert_grouped.cu:386-460`, `:1146-1187`) exists to reproduce the CPU's
    AVX reduction order exactly.
  - Output therefore depends on residency. DETAILS.md:539-541 says that with the default adaptive tier, a sampled
    result is not reproducible run to run, and recommends `--adapt-every 100000` for seed-reproducible output.
  - **A Metal port that runs every expert on the GPU removes this source of nondeterminism.**

### 1.8 E-6 device-side plan (opt-in, `STRATA_VERIFY_DEVICE_PLAN=1`, `verify.cpp:359-372`)

- `resident_plan_kernel` (<<<1,1>>>, `verify_kernels.cu:433-473`) rebuilds the host's plan on the GPU when every
  routed expert of the group is resident in `d_res`. It then sets `skip_[grp] = ring`.
- `wait_flag_ge_or`, `copy_i32_from_mapped_unless` and `copy_or_zero_from_mapped` then skip the host round trip
  entirely.
- It is exact, but it is off by default because it was "neutral on 1-2 GPUs".
- **This is the shape a Metal port wants when all experts are GPU-resident:** the routing → grouping → expert
  kernels path with no host in the loop.

### 1.9 Commit (`capture_commit` `verify.cpp:950-1014`, `commit` `:1320-1350`, `wait_commit` `:1352-1360`)

- `h_commit_ = [n_keep, n_keep-1, pos0+t for t<n_keep else -1]`, then a `seq_cst` fence, then
  `cudaGraphLaunch(commit_exec_, cs_)`.
- The commit graph does three things:
  1. `copy_i32_from_mapped(commit_)`.
  2. For each GDN layer, `gdn_conv_commit` (conv history advanced by n_keep) and `gdn_step_norm_multi(..., commit_)`.
     This replays the recurrence over only the kept tokens into the persistent state, using the qkv/h/gate/beta the
     window cached in `qkv_L_`/`h_L_`/`gate_L_`/`beta_L_`.
  3. For each QSA layer, it restores the indexer tail from `tail_snap_` and re-appends the kept rows
     (`native_qsa_indexer_append` with `commit_+2+t`; -1 skips). Then it does `copy_indexed(ple.hist, hist_snap_,
     commit_+1)`.
  - K/V rows for rejected positions are not undone. They are simply overwritten when those positions are processed
    again.
- **Async commit:** `set_commit_async(!multi_gpu)` (`generate.cpp:2181`), unless `STRATA_COMMIT_SYNC` is set.
  - `commit()` records `commit_done_` instead of syncing.
  - The next window runs on the same stream, so it is ordered after the commit.
  - `wait_commit()` = `cudaEventSynchronize(commit_done_)` before anything else reads the session.
- The host advances `ss_->ple_prev` for n_keep tokens (`:1341-1345`).

### 1.10 Graph rules the code learned (`include/strata/core/graph.hpp`, `src/core/graph.cpp`)

1. **The wait must poll the driver on WDDM.** A memory-only spin saw 5.9M spins with the kernel never running,
   because WDDM batches submissions. `CapturedGraph::wait_ms` polls with `cudaEventQuery` + `_mm_pause`×64
   (`graph.cpp:84-102`).
2. **Kernel arguments are baked at capture**, so every per-step value (tokens, positions, flags, commit counts) must
   be device or mapped data. The doorbell's "cloned literal" bug (`elementwise.cu:188-191`) is the canonical example.
3. **Fixed addresses.** Every buffer is carved once from one arena with a `Bump` allocator (`verify.cpp:290-330`).
4. Driver calls cost 10-22 µs each, and a zero-node capture is an error.

### 1.11 Timing budget (paper Finding 1, Q2_0 at 4K context)

- One window takes about 33 ms. The CPU waits about 14 ms for the GPU, and the GPU waits about 13 ms for the CPU.
- The CPU expert rate is about 42 GB/s: 41 GB/s concurrent with the GPU, 52 GB/s alone.
- MTP speculation gives 1.6-1.8×.
- The adaptive tier moves the hit rate from 50% to 72% (91.7 → 94.4 tok/s, Finding 10).


---

## 2. The expert cache (the VRAM tier)

### 2.1 Why it exists, in the code's own numbers

`include/strata/core/expert_cache.hpp:1-26` gives the case for the cache:

- Without it, the CPU pool reads 663.6 MB of expert bytes per token at about 40 GB/s, which is 16.2 ms of a
  ~53 ms token. Only 1.055 ms of 19.035 ms was hidden by pipelining.
- A frequency profile with 4,105 slots gives hit rate h = 0.6447 [0.6110, 0.6750] in a ten-fold leave-one-out test
  (`tools/routing_kfold.py`).
- Only 4.6% of (layer, token) pairs have all ten experts resident (`h_layer` = 0.0456). So the split into VRAM, PCIe
  and CPU happens per expert, not per layer.

Paper results:

- Finding 4: a profile gives 50% hits with about 4,500 slots on a 12 GB card, and adaptive swapping raises that to
  about 72%.
- Finding 8: sizing slots in bytes fits 3,556 IQ3_XXS experts instead of 2,673, which is +13% decode speed at 1K.
- Finding 10: asynchronous refill raises Q2_0 at 4K from 91.7 to 94.4 tok/s.
- Finding 12: at 262K context the KV cache leaves room for only about 1,600 experts.

### 2.2 Storage (`src/core/expert_cache.cpp`, 374 lines; header 165 lines)

| Piece | Where | What it does |
|---|---|---|
| `open(n_slots, n_layers, n_expert, blob_bytes)` | `expert_cache.cpp:88-162` | Compares `cudaMemGetInfo` free memory against `n_slots*blob` before allocating, and refuses with both numbers (`:109-121`). Then makes one `cudaMalloc` (`:123`) and calls `cudaMemset(0)` (`:133`), so an unfilled slot gives a deterministic wrong answer. Allocates `residency_` = 48×512 int32, all `kNotResident` (-1) (`:139`; `expert_cache.hpp:39`). Seeds the per-layer cursors (`:155-160`). |
| HIP bounce buffer | `:72-86`, `:301-310` | `ensure_blocking_staging` uses `cudaHostAlloc` to make a pinned buffer of one blob, so HIP never registers the pageable mmap. |
| `open_sized(slot_bytes[])` | `:164-194` | Byte-sized slots (plan v0.3 P6). The offsets are prefix sums of `roundup(bytes, 256)` (`:170-171`). One allocation of the summed size, made through `open()` as n "slots" of 1 byte (`:175`). The comment explains 256-byte alignment: "every blob starts where the kernels' vector loads expect it". |
| `device_slot(slot)` | `:254-264` | Returns `base_ + off_[slot]` for sized slots, otherwise `base_ + slot*blob_`. |
| `layer_slot_range` | `:218-227` | R4.2g per-layer ranges `[l*q, (l+1)*q)`. The last layer gets the remainder. |
| `admit(layer, expert)` | `:234-252` | Never evicts (`expert_cache.hpp:89`). It takes the next free slot, or the next slot in the layer's own range when `--expert-cache-per-layer` is set (`generate.cpp:1210`). The global counter capped the hit rate at 2.97% (`:239-240`). |
| `fill_slot` / `fill_slot_blocking` / `fill_slot_queued` / `sync_queued` | `:266-286` / `:288-317` / `:319-334` / `:336-343` | H2D copy on a given stream, synchronous `cudaMemcpy`, async copy on the legacy stream 0, and a sync of stream 0. |
| `verify_slot` | `:345-372` | Synchronous D2H readback plus memcmp. At startup one slot is checked (`generate.cpp:2866-2873`), costing one 1.38 MB D2H. |

Slot sizes:

- The canonical Q2_0 blob is 1,382,400 B (`include/strata/kernels/cpu/expert.hpp:45`). Its layout:
  - `O_GU_CODES` at 0 (819,200 B).
  - `O_D_CODES` at 819,200 (409,600 B).
  - `O_GU_SCALES` at 1,228,800 (102,400 B).
  - `O_D_SCALES` at 1,331,200 (51,200 B).
  - Offsets are at `expert.hpp:48-51`.
- Native packs (IQ*, K-quants) have per-layer blob sizes, `expert_layout().blob_bytes(layer)`
  (`include/strata/kernels/cpu/expert_layout.hpp:32-40`). IQ3_XXS ranges from 1.3 to 2.3 MB.
- `experts.bin` for Q2_0 is 33,973,862,400 B (48×512×1,382,400). Blobs are not page-aligned. Layer starts are
  16 KiB-aligned (HIP/expert-source agent, `expert_layout.hpp`).

### 2.3 Sizing (`src/program/generate.cpp`)

- **`--expert-cache auto`** (`:1109-1111`, -1) at `:2666-2690`:
  - `slots = (free - reserve) / max_blob`.
  - `reserve = vram_reserve_mib` (default 700, `:324`) + the prefill buffers. Those are
    `160 + chunk*680/1024` MiB, but only when the prompt path does not borrow cache slots.
  - Plus `mtp.bind_bytes` for the draft head.
  - The result is capped at the profile size.
  - A count of 0 is an error, because the verify window needs the cache (#174, `:2686-2688`).
- **The cache is allocated after** the weights, the session, the drafter, the native head and `d_logits`, so
  `cudaMemGetInfo` sees the true remainder (`:2636-2662`).
- **Sized slots** (`:2705-2724`): for a native pack with a profile, walk the profile in rank order and pack
  `roundup256(blob_bytes(layer))` until `min(uniform budget, free - reserve)` is used up. This is disabled with
  `--expert-cache-per-layer` (#369), because a sized slot is cut for one specific rank.
- **Test hooks:** `STRATA_TEST_CACHE_FAIL=N` makes the first N opens fail (`:2754-2762`). There is also WDDM
  page-file handling (`:2770`).

### 2.4 Filling: the profile

- **Format** (writer `tools/make_profile.py:56-63`, reader `expert_cache.cpp:12-68`):
  - Header: `"STRP"` followed by `<5I>` = `version=1, n_layers=48, n_expert=512, slots, n_ranked`.
  - Then `n_ranked` × `<HH>(layer, expert)` in rank order.
  - Then a 48×512 int32 table. The writer stores the slot index or -1 there. The header comment
    (`expert_cache.hpp:47-50`) calls it a "frequency table".
  - The reader ignores both the trailing table and `version` (`:66`).
  - The writer sets `slots = n_ranked`. A full profile is 24 + 98,304 + 98,304 = 196,632 B.
  - Shipped files: `data/expert-profile*.bin` (2). Regenerate with
    `tools/make_profile.py TRACE... [--base] [--n-expert 256]` from `--dump-routing` traces.
- **Fill** (`generate.cpp:2837-2880`, R4.2e):
  - Runs once: `admit` followed by `fill_slot_blocking` for each ranked pair, until the cache is full.
  - In per-layer mode, a full layer skips only its own pairs (#369).
  - Ends with `verify_slot` on pair 0.
  - Because nothing is ever evicted, the profile is the policy.
- **Device residency table `d_res`** (`:3610-3640`, plan v0.3 P4):
  - With a profile-filled cache, `host_res[48*512]` (slot or -1) is copied into a `cudaMalloc`'d `d_res`, and into
    every stage's copy when the layers are split across GPUs.
  - `thits.d_res`, `cache_base` and `blob` feed the device-side hit decision (`TokenHits`, `hit_hook.hpp`) and E-6.
  - The planner reads `host_res` directly (section 1.6).

### 2.5 The adaptive tier (plan v0.3 P6; serve `generate.cpp:4424-4512`, CLI `:6025-6090`)

- **Defaults:**
  - `adapt_every` = 4 rounds (`:359`, `--adapt-every` `:1139`).
  - `adapt_swaps` = 96 (`:381`, `--adapt-swaps` `:1208`).
  - `drive.d.usage[48*512]` floats are counted by the planner (`expert_source.cpp:1621-1623`).
- **Trigger:** `(rounds+1) % adapt_every == 0`. `adapt()` runs on a `std::thread` while the GPU commits and drafts
  (`:5372-5376`, `:6194-6196`). It touches only the residency tables, which nothing reads before the next window.
- **Selection** (`:4459-4485`). For each layer:
  - Candidates are non-resident experts with usage ≥ 2.0.
  - Victims are resident experts.
  - Candidates are sorted descending and victims ascending (partial sort).
  - Pairs are taken while `cand ≥ vict + 1.5`.
  - All swaps are then sorted by gain and truncated to `adapt_swaps`.
- **Exchange in resident-RAM mode** (`resident_stage_swaps`, `:113-163`). If `in` is held only in RAM and `out` only
  in VRAM:
  1. Copy `out` D2H into an exchange buffer (`cudaMemcpyAsync` D2H, then `cudaStreamSynchronize`).
  2. `stage_exchange` lets the CPU compute `out` from RAM meanwhile.
  3. `commit_exchanges` later moves the RAM copy into `in`'s place (`expert_source.cpp:1341-1403`).
- **Copy** (`:4486-4500`):
  - `cudaMemcpyAsync` H2D on `adapt_stream`, which is non-blocking (`:4426`, CLI `:6029`).
  - `host_res[out] = kNotResident` immediately, so the CPU computes `out` in the meantime.
  - `pending.push(in, slot)`, followed by `cudaEventRecord(adapt_ev)`.
- **Admission** (`apply_pending`, `:4443-4457`): once `cudaEventQuery(adapt_ev)` succeeds (or after
  `cudaEventSynchronize` when `wait`), set `host_res[in] = slot` and re-upload `d_res` with a synchronous
  `cudaMemcpy` (`res_upload`, `:4435-4442`).
- **Decay:** `usage *= 0.7` (`:4511`; CLI `:6096`).
- **Determinism:** residency changes which experts take the CPU path and which take the GPU path. The two paths
  round differently, so seeded sampling is reproducible only with `--adapt-every 100000` (`docs/DETAILS.md:539-541`)
  or `--adapt-swaps 0` (`docs/DETAILS.md:78-91`).

### 2.6 Prefill borrows cache slots

- **`plan_lend`** (`:3748-3770`):
  - With `--prefill auto`, tries `kAutoChunks` = {32768, 16384, 8192, 6144, 4096, 3072, 2048, 1024, 512, 256}.
  - Chunks above 8192 are used only when `prefill_auto_max` and `max_context` allow them (#282).
  - It needs `k + 128 ≤ slots` and `k ≤ kAutoLendPct% × slots`. `kAutoLendPct` is 90 when the pinned share is ≥ 0.9,
    else 85 (`:3739-3746`); `STRATA_PREFILL_LEND_PCT` overrides it.
  - `lend_slots(c)` counts tail slots until their bytes cover the prompt path's need (`:3714-3724`).
- **Request sizing** (`:5660-5690`):
  - The chunk is reduced to `roundup256(n_batched)` (`request_chunk`, `:3727-3733`).
  - The last k slots are lent: their `host_res` entries are set to -1, `d_res` is re-uploaded synchronously, and
    `borrow = device_slot(slots-k)` is passed to `Prefill::init`.
- **Refill** (`:5716-5737`, perf-review D-4): `fill_slot_queued` for each lent slot, then one `sync_queued`, then a
  `d_res` upload. `STRATA_REFILL_BLOCKING=1` waits on each copy (`:115-118`).
- **Measured** (`:3735-3738`, `:3749-3751`):
  - 12 GB card, 32K Q2_0: 4096 / 6144 / 8192 chunks give 791 / 878 / 973 tok/s, with 69% of slots lent.
  - IQ2_XS on a 5090 at 32K: 5,624 tok/s with 8192 chunks, 6,465 with one 32768 chunk.

### 2.7 What it becomes on Apple unified memory

On M-series machines there is no second memory to cache into. The cache exists only to avoid PCIe and to move
work onto the GPU. On a Mac every expert byte is already visible to the GPU at full memory bandwidth, so:

1. **Drop the slot arena, the fills, `verify_slot`, prefill borrowing and refill, and adaptive copies.**
   - mmap `experts.bin` (or the GGUF shards) once.
   - Wrap it in `MTLBuffer`s with `newBufferWithBytesNoCopy`. This needs page alignment, which is 16 KiB on Apple
     Silicon; layer starts are 16 KiB-aligned already.
   - Make it resident with an `MTLResidencySet`, or with `useResource`/`useHeap` on older OS versions.
   - A slot becomes a byte offset into the arena (`expert_layout().blob_offset(layer, expert)`). `host_res` and
     `d_res` become a static "all experts GPU-addressable" table or disappear.
   - Check that `maxBufferLength` covers the arena; 34-43 GB exceeds it on some devices. If not, use one buffer per
     layer or per shard.
2. **Reinterpret residency as a work assignment**, but only if the CPU stays a compute participant (section 4.3):
   - "Resident" = computed on the GPU; "miss" = computed on the CPU.
   - The adaptive tier becomes a load balancer. It moves no bytes, so it needs no copies, events or exchange buffers.
     The usage counters and the gain rule can be reused to choose which experts the CPU takes, for example the
     coldest ones, so as not to stall the GPU.
3. **Keep the profile format and `make_profile.py`.**
   - The profile becomes optional. It is still useful for page-cache warming (`madvise(WILLNEED)` in rank order)
     when the model is larger than RAM and is streamed from SSD.
   - On a 128 GB Mac a 34-43 GB pack fits in RAM, and the profile only orders the warm-up.
   - On 64 GB machines with a larger pack it can choose what to `mlock`.
4. **Determinism improves.** If all experts run on the GPU there is no CPU/GPU rounding difference, and the
   `--pcie-frac 0` / `--adapt-swaps 0` caveats (#410) vanish. Section 6 turns this into a stricter gate.

---

## 3. Prefill (`src/prefill/*`, `include/strata/prefill/*`)

Files:

| File | Lines |
|---|---|
| `src/prefill/prefill.cpp` | 1977 |
| `src/prefill/kernels.cu` | 842 |
| `src/prefill/gemm.cu` | 432 |
| `src/prefill/moe_mmq.cu` | 231 |
| `src/prefill/ggml_cuda_host.cu` | 134 |
| `src/prefill/hipblaslt_tuning.hpp` | 123 |

Headers are `include/strata/prefill/{prefill,kernels,gemm,moe_mmq}.hpp`.

Model constants are at `prefill.cpp:68-69` and `kernels.cu:17-18`:

- N=2560, HC=4, D=10240, LR=320, K=10, NE=512.
- GDN: C=10240 conv channels, ZV=6144, HV=48, HK=16, S=128.
- Expert shapes: GU [1280, 2560], down [2560, 640].

**Paper vs code.** The paper (`/tmp/strata-paper.txt:166-170, 337-339`, Finding 11) describes an older path: 2048-token
chunks, FP16 dequant plus cuBLAS, 539 tok/s at 4K and 496 at 262K, using about 7 of 26 GB/s of PCIe. The code is the
0.1.13+ path (`docs/DETAILS.md:167-176`):

- Chunks up to 8192 tokens.
- llama.cpp MMQ for the routed experts.
- Next-layer experts streamed during attention.
- Batched PLE.
- Copy helper threads.

At 32K on an RTX 5070, Q2_0 rose from 572 to 1,290 tok/s and IQ3_S from 383 to 1,208. `DETAILS.md:725, 814` still
mention 2,048-token chunks.

### 3.1 Chunk sizing

| Control | Effect | Where (`generate.cpp`) |
|---|---|---|
| `--prefill auto` | `prefill_auto`, `prefill_auto_max=8192` | `:1119-1128` |
| `--prefill auto:N` / `STRATA_PREFILL_AUTO_MAX` | N rounded to 8192, 16384 or 32768 (#282) | `:1123-1127` |
| `--prefill CHUNK` | Fixed chunk; halved until it fits the lend rule | `:1128`, `:3762-3766` |
| `--no-prefill-borrow` | Never lend cache slots; reserve `160 + chunk·680/1024` MiB instead | `:1130`, `:2275-2290` |
| `--prefill-until N` | Batched prefill stops at N; the rest goes token by token | `:1131`, `:5654` |
| Native pack | Requires `--prefill` | `:1847-1851` |
| `auto` without a profile, or with no-borrow | Fixed 2048 ("nothing to lend from") | `:1439-1442` |
| Request shrink, lend, refill | Section 2.6 | `:5655-5737` |
| Serve | Fallback 1024 if nothing is lendable or there is no cache (`:3948-3949`, `:3963-3964`). `own_fits` needs `bytes_needed + 512 MiB` free (hipBLAS on gfx1201 hung at "fits to the last MiB", `:4008-4047`). `kStepChunks {6144, 4096, 3072, 2048, 1536, 1024, 512}`; retry on "do not fit" at `:4049-4060`; per-request `Prefill::relayout` at `:5164` | |

- Batched prefill covers `n_batched = n_prompt - 1` tokens. The last prompt token goes through the decode path
  (`:5654`, `:5712`).
- **Cost of a chunk** (`Prefill::bytes_needed`, `prefill.cpp:887-922`, counted by the `Alloc{count_only}` bump
  allocator at `:131-153`):
  - About 680 KB per token. About 260 KB of that is a region shared between attention, GDN and MoE scratch
    (`:438-458`, `:494-510`).
  - `GEMM_SCRATCH` is 32M FP16 (64 MiB) plus a 32 MiB cuBLAS workspace (`:435-436`).
  - A DQ ring of 2 × (1280·2560 + 2560·640) FP16, about 19.7 MB.
  - MMQ group buffers of 16 experts, each followed by a 4096 B tail.
  - `ring_slots(T) × MAXBLOB`, plus the PLE and KV staging buffers and 8 MiB of slack.
- **The chunk size changes which code runs:**
  - `T ≥ stream_all_min()` = 1024 (`prefill.cpp:77-84`; `STRATA_PREFILL_STREAM_MIN`) selects stream-all. Below about
    1,000 tokens "the output changed on Q2_0 (a smaller chunk takes other kernels)".
  - `T ≤ CONV_TILE` = 64 selects the serial conv (`kernels.cu:735`).
  - QSA selection runs in batches of 256 queries; the fallback attention runs 32 queries at a time.

### 3.2 One chunk, step by step (`prefill.cpp:994-1941`)

**Preamble** (`:994-1262`):

1. `mmq::iota(ids_identity)` (`:1018`). It is not done in `carve`, because those bytes may still be live cache slots
   (`:690-691`).
2. **PLE prefetch** for the next chunk (`:1019-1050`, `:1113-1131`): a `std::async` thread runs `ngram_rows` and
   `gather_batch` over the 28.8 GB SSD table into a pinned double buffer `ple_emb_host[2]`. The upload is a
   `cudaMemcpyAsync` H2D, synchronised with `ple_copied[b]` and `cudaEventSynchronize` (`:1124`).
3. **Embedding:** token H2D (`:1082`), then `gather_dev`, image rows H2D (`:1103`), then `gr_broadcast` into the 4 HC
   streams (`:1111`). Steps H2D (`:1134-1136`).
4. **Stream-all plan** (`:1143-1172`): `seq` = every non-resident (l, e) for layers [LB, LE). Unpinned or
   GGUF-in-place blobs become Stager jobs (`:162-286`, CAS-claimed `gen<<32|n<<16|idx`).
5. **Copy issue** (`:1179-1260`): entry k goes to ring slot `k % ring`.
   - The copy stream waits on `used[sl]` (`cudaStreamWaitEvent` at `:1186`, `:1233`), copies H2D from the pinned arena
     (`:1188`, `:1192`, `:1235`, `:1239`), and records `copied[sl]` (`:1195`, `:1242`).
   - An **issuer thread** (D-5, `:1202-1251`) runs up to `ring` entries ahead.

**Layer loop** (`:1263-1857`):

- **PLE (layer 1):** batched over the chunk (`:1268-1321`), calling `native_ple_postops_batch` (`:1300`).
- **Each half** (`:1322-1340`): `gr_norm_rs` (or the unfused `gr_norm`), then BF16 GEMMs `hc_down`
  [T,10240]→[T,320], `gr_silu`, `hc_up` →[T,10240], `hc_inject` →[T,4], then `gr_mix_r` →
  `mixed` f32/bf16/f16.
- **GDN half** (36 layers, `:1342-1364`):
  - Native FP16 `attn_qkv` [T,2560]→[T,10240] and `attn_gate` →[T,6144].
  - BF16 `ssm_alpha` / `ssm_beta`.
  - `gdn_gates`, then `gdn_conv` (serial, or tiled plus hist), then `gdn_l2`.
  - `gdn_rec_cols_pipe` + `gdn_out_norm`. The state layout is [k=128][head=48][v=128] f32 (`kernels.cu:283-287`).
  - `ssm_out` [T,6144]→[T,2560].
- **QSA half** (12 layers, `:1365-1562`):
  1. Native K/V [T,512], Q [T,12288]; BF16 indexer k/q.
  2. `rms_rows`, then `rope` (NEOX, first 64 dims).
  3. KV stream staging (`kv_mode==1`).
  4. `kv_append`: FP16, INT8 with a per-64 absmax, Q4 (Hadamard), or K8V4.
  5. `split_q`, `rms_rows`, `rope`.
  6. `native_qsa_indexer_append_batch`.
  7. Per 256 queries: `qsa_block_scores_tc` (3xTF32, "not bitwise") then `qsa_block_topk`.
  8. `qsa_prompt_attn_batch` (tensor cores), else `qsa_decode_attn_batch`.
  9. Inverse FWHT, then `gate_attn`, then native `attn_output`.
- **MoE half** (`:1563-1835`):
  1. BF16 router [T,2560]→[T,512] (`:1572`), then `route` → ids/w [T,10] (`:1573`).
  2. Shared expert: native gate/up → `swiglu_pair` → down, plus the scalar gate (`:1575-1580`).
  3. **Host grouping** (`:1582-1616`): ids go to a mapped table via `copy_i32` (or D2H with `STRATA_GROUP_COPY=1`).
     Then **`cudaStreamSynchronize(m.cs)` at `:1592` — once per MoE layer, 48 compute drains per chunk.** Then a CPU
     counting sort (cnt, off, slot, src), and `slot`/`src` go back through mapped memory.
  4. **Activations:**
     - MMQ: `mmq::quantize(mixed, src_dev → Xq q8_1)`. Bounds are absolute per expert plus 17 relative per 16-group
       (`:1627-1649`).
     - FP16: `gather_rows16` (`:1651`).
  5. **Streaming:**
     - Routed-only mode (T < 1024): an 8-slot ring with lookahead 7 (`:1656-1701`, `:1764-1781`).
     - Stream-all mode: walk `seq`; unrouted entries just release (`:1782-1811`).
     - Resident experts (`host_res ≥ 0`) are read from `cache->device_slot` (`:1772-1774`, `:1806-1807`).
  6. **`compute()`** (`:1704-1763`):
     - MMQ, per 16-expert group:
       1. `gather_native` (`copy16`/`copy1`) or `gather_strata_q2`.
       2. Record `used` early (`:1717`).
       3. `cudaMemsetAsync` of the 4096 B zero tail (`:1727-1728`). MMQ reads up to one 256-value tile past the last
          row (`:465-468`).
       4. GU `mul_mat_q` → `mmq::swiglu` → `quantize` → down `mul_mat_q` (`:1729-1742`).
     - FP16 fallback: `blob_dequant_f16` / `iq_dequant_*_f16` into a DQ ring of 2, then two `cublasGemmEx` per expert
       (`:1744-1761`).
  7. `moe_combine` (`:1813`): `bo = Σ_k w·Dm[slot] + shared·σ(sg)`.
- **HC write:** fused `gr_write_norm_rs` (F-2), or unfused `gr_write` plus `cvec_apply` (`:1836-1856`).

**End:** join the issuer (`:1859`); split hand-off D2H (`:1868-1885`); `on_chunk` (MTP `draft_kv`, E-9) after a sync
(`:1899-1910`); final syncs (`:1933-1941`).

### 3.3 MMQ vs cuBLAS / hipBLASLt

| Product | Path | Condition |
|---|---|---|
| Routed experts | llama.cpp MMQ: `mul_mat_q_case<T>` with q8_1 activations, int8 MMA | Built in, `STRATA_PREFILL_MMQ≠0` (`prefill.cpp:474-475`), and `mmq::fits(gu,1280) && fits(d,2560)` (`:479-481`; `moe_mmq.cu:110-136` replicates `mul_mat_q_switch_J` against `smpbo`) |
| Routed experts, fallback | Dequant to FP16, then `cublasGemmEx` f16 | Not built, disabled, unsupported type (e.g. IQ1_M, `prefill.cpp:463`), or no tile fits (`moe_mmq.cu:124-131`) |
| Dense quantized weights | Always `Gemm::native` = `dequant_f16` (`src/kernels/cuda/dequant_bf16.cu:242`) → 64 MB scratch → `cublasGemmEx` FP16 | Row slices if N·K > 32M (`gemm.cu:414-430`) |
| BF16 weights | `Gemm::bf16` = `cublasGemmEx(OP_T, OP_N, N, T, K, W bf16, X bf16, Y f32, COMPUTE_32F)` | `gemm.cu:375-393`; f16 variant at `:395-412` |
| MTP drafter K/V (E-9) | Q8_0 MMQ, or FP16 with `STRATA_MTP_BATCH_F16` | `prefill.cpp:737-882` |
| HIP | hipBLAS Ex. hipBLASLt runs only with a tuning file `STRATA_HIPBLASLT_TUNING_V1 <arch> <version>` (`hipblaslt_tuning.hpp:26-111`; `gemm.cu:126-291`). MMQ is opt-in (`-DSTRATA_PREFILL_MMQ=ON`, `CMakeLists.txt:904-934`) | |

- **No token-count threshold** chooses MMQ over GEMM.
- **MMQ build** (`CMakeLists.txt:856-896`):
  - Types: `q2_0 iq2_xxs iq2_xs iq2_s iq3_xxs iq3_s iq4_nl iq4_xs q8_0`. `+q5_0` with `STRATA_ORCA_Q4KS_MMQ`;
    `+q4_k q5_k q5_1` with `STRATA_MMQ_KQUANTS`.
  - ggml commit `3cf03257f219afbe7334045ff7c6a06ac68c627d` (`:831`), with `GGML_CUDA OFF`. Only the MMQ sources are
    compiled.
  - Flags `-use_fast_math --extended-lambda` (`:884`).
- **Host shims** (`ggml_cuda_host.cu`): `ggml_cuda_info` (`:62-101`, honouring `STRATA_EMULATE_CC` at `:84`) and a
  `CachingPool` for the stream-k fixup (`:105-130`).

### 3.4 Every `__global__` kernel in the requested files

Totals:

- `src/prefill/gemm.cu`: 0 (cuBLAS/hipBLASLt host code only).
- `src/prefill/ggml_cuda_host.cu`: 0.
- `src/core/pinned.cu`: 0.
- `src/core/device.cu`: 1.
- `src/prefill/kernels.cu`: 32.
- `src/prefill/moe_mmq.cu`: 5.
- **Total: 38.**

Conventions:

- "Def/launch" means the line of the `__global__` definition and the line of the wrapper that launches it.
- `bs` = `block_sum` (`kernels.cu:45-57`): `__shfl_xor_sync` over 32 lanes, plus `__shared__ float[32]`, plus
  `__syncthreads`.
- `bf()` is a manual FP32→BF16 round-to-nearest-even (`:30-34`). `hf_sat` saturates to ±65504 (`:40-43`).
- Every wrapper calls `check()` → `cudaGetLastError` (`:58-61`).

| # | File | Kernel | Def/launch | Purpose | Launch shape | CUDA features | Metal note |
|---|---|---|---|---|---|---|---|
| 1 | device.cu | `poison_kernel` | 22 / 213 | Fill a new `DeviceArena` with a quiet NaN (debug) | ceil(n/256) × 256, chunked to 2^31-1 blocks | `__int_as_float` | `fill_buffer` blit or a 1-line kernel; debug only |
| 2 | kernels.cu | `gr_norm_kernel` | 65 / 691 | RMS-norm per HC row × w → f32 + bf16 (F-1 A/B arm) | T·4 × 256 | `bs`, `rsqrtf` | One threadgroup per row, `simd_sum` |
| 3 | kernels.cu | `gr_norm_rs_kernel` | 84 / 696 | Row scale + bf16 image only (default) | T·4 × 256 | `bs` | Same |
| 4 | kernels.cu | `gr_mix_r_kernel` | 102 / 701 | Norm recomputed in place; Σ_c x·σ(g)/4 → mixed f32/bf16/f16 | ceil(T·N/256) × 256 | `__expf` | Elementwise; keep the operation order |
| 5 | kernels.cu | `gr_write_norm_rs_kernel` | 128 / 707 | F-2: R += bo·2σ(inj/4), then the next norm | T·4 × 256, `__launch_bounds__(256)`, `v[10]` registers | `bs` | Exactly 256 threads (GRW_PER) |
| 6 | kernels.cu | `gr_silu_kernel` | 160 / 712 | SiLU(x/4) → bf16 | ceil(T·320/256) × 256 | `__expf` | Elementwise |
| 7 | kernels.cu | `gr_mix_kernel` | 170 / 717 | Unfused mix | ceil(T·N/256) × 256 | — | Elementwise |
| 8 | kernels.cu | `gr_write_kernel` | 191 / 721 | Unfused HC write | ceil(T·D/256) × 256 | — | Elementwise |
| 9 | kernels.cu | `gr_broadcast_kernel` | 198 / 725 | Embedding → 4 streams | ceil(T·D/256) × 256 | — | Elementwise or blit |
| 10 | kernels.cu | `gdn_gates_kernel` | 206 / 729 | gate = softplus(α+dt)·a (cutoff 20); β = σ | ceil(T·48/256) × 256 | `log1pf`, `__expf` | Elementwise |
| 11 | kernels.cu | `gdn_conv_kernel` | 217 / 735 | 4-tap causal conv + SiLU, serial over T (T ≤ 64) | 80 × 128 | — | One thread per channel |
| 12 | kernels.cu | `gdn_conv_tiled_kernel` | 235 / 737 | Same, 64-token tiles (C-3, same bits) | (80, ceil(T/64)) × 128 | — | 2-D grid |
| 13 | kernels.cu | `gdn_conv_hist_kernel` | 253 / 739 | Store the last 3 inputs into hist | 80 × 128 | — | Trivial |
| 14 | kernels.cu | `gdn_l2_kernel` | 263 / 741 | L2-normalise the 32 q/k heads | (32, T) × 128 | `shfl_xor`, `__shared__ part[4]` | `simd_sum` + threadgroup[4] |
| 15 | kernels.cu | `gdn_rec_kernel` | 277 / 748 | Delta-rule recurrence over T + out-norm (A/B arm) | 48 × (128,4), LB 512 | Shared sk/sq/red[4][128]; 32-float state in registers | 512 threads per threadgroup is fine; latency-bound serial T loop |
| 16 | kernels.cu | `gdn_rec_cols_kernel` | 337 / 754 | D-2: value columns split 4 ways per head (same bits) | 192 × (32,4), LB 128 | Shared + registers | Same |
| 17 | kernels.cu | `gdn_rec_cols_pipe_kernel` | 380 / 752 | **Default:** the above + register prefetch of the next token | 192 × (32,4) | Register prefetch | Port as-is (Metal has no cp.async) |
| 18 | kernels.cu | `gdn_out_norm_kernel` | 436 / 755 | Per-head RMS × γ × σ(z) → f16 | (T,48) × 128 | `shfl`, `wsum[4]` | `simd_sum` |
| 19 | kernels.cu | `route_kernel<REG>` | 454 / 761, 763 | Softmax over 512 (REG=16) or 256 (REG=8); top-10 by 10 rounds of warp argmax (ties → lower id, NaN → -FLT_MAX); renorm floor 6.1035e-5 | ceil(T/8) × 256 (one warp per token) | `__shfl_xor_sync` of value + index, precise `expf` | One simdgroup per token; `simd_shuffle_xor` |
| 20 | kernels.cu | `blob_dequant_kernel<HALF>` | 494 / 769, 773 | Strata Q2 blob → bf16/f16 GU (interleaved) + down; v = (code-1)·d | ceil((1280·640+2560·160)/256) × 256 | `__ushort_as_half` | Unneeded with a quantized matmul |
| 21 | kernels.cu | `swiglu_il_kernel` | 518 / 778 | SiLU(g)·u, interleaved → f16, `hf_sat` | ceil(n·640/256) × 256 | `__expf` | Keep the saturation |
| 22 | kernels.cu | `swiglu_pair_kernel` | 525 / 782 | Shared-expert SwiGLU → f16, saturated | ceil(n·640/256) × 256 | — | Elementwise |
| 23 | kernels.cu | `gather_rows16_kernel` | 532 / 799 | Gather 16-bit rows by `src` | ceil(n·320/256) × 256 | `uint4` loads | Avoidable with an indexed matmul |
| 24 | kernels.cu | `moe_combine_kernel` | 540 / 804 | bo = Σ_k w·Dm[slot] + shared·σ(sg) | ceil(T·N/256) × 256 | — | Keep the k order |
| 25 | kernels.cu | `rms_rows_kernel` | 553 / 809 | In-place row RMS-norm | rows × 256 | `bs` | One threadgroup per row |
| 26 | kernels.cu | `rope_kernel<TAB>` | 566 / 824, 828 | NEOX rope, first 64 dims, M-RoPE positions, YaRN; optional f64 table | T·heads × 32 | `powf`, `mrope.hpp` helpers | A template because the compiled default path changed results otherwise (`:562-565`) |
| 27 | kernels.cu | `split_q_kernel` | 586 / 834 | q half of each 512-wide head | ceil(T·6144/256) × 256 | — | Strided view |
| 28 | kernels.cu | `gate_attn_kernel` | 592 / 838 | attn·σ(gate) → f16 | Same | `expf` | Elementwise |
| 29 | kernels.cu | `kv_append_kernel` | 602 / 668 | Paged FP16 or INT8 (per-64 absmax f16) K/V append; also host mirror + stage pool | (T,2,8) × 64 | `shfl_xor` max, `__float2int_rn` | `simd_max`; the mirror writes vanish |
| 30 | kernels.cu | `to_f16_kernel` | 645 / 675 | f32 → f16, grid-stride | min(ceil(n/256), 4096) × 256 | — | Elementwise |
| 31 | kernels.cu | `round_f16_kernel` | 649 / 680 | f32 → f16 → f32 | Same | — | Elementwise |
| 32 | kernels.cu | `to_bf16_kernel` | 653 / 685 | f32 → bf16 (+ remainder lo) | Same | — | Elementwise |
| 33 | kernels.cu | `copy_i32_kernel` | 786 / 794 | Read/write mapped pinned grouping tables without queuing behind the copy engine | min(ceil(n/256), 256) × 256 | Zero-copy pointer | **Delete**: shared buffers |
| 34 | moe_mmq.cu | `copy16_kernel` | 25 / 203 | Concatenate native GU + down into a group slot (16 B aligned) | ceil(…/256) × 256 | `uint4` | Unneeded if experts are indexed in place |
| 35 | moe_mmq.cu | `copy1_kernel` | 32 / 207 | Same, byte-wise | Same | — | Same |
| 36 | moe_mmq.cu | `strata_q2_kernel` | 43 / 214 | Strata blob → GGUF Q2_0 blocks {f16 d; 16 B codes} | ceil((1280·40+2560·10)/256) × 256 | `uint4` | Convert once offline or at load |
| 37 | moe_mmq.cu | `swiglu_kernel` | 71 / 221 | SiLU(g)·u → f32 before q8_1 (interleaved or split) | ceil(rows·640/256) × 256 | `__expf`, fast-math TU | Fuse into the GU matmul epilogue |
| 38 | moe_mmq.cu | `iota_kernel` | 81 / 227 | `ids_identity[i] = i` | ceil(n/256) × 256 | — | Trivial |

**Device code that prefill calls but that is defined elsewhere** (also has to be ported):

- ggml `mul_mat_q` + stream-k fixup, and `quantize_mmq_q8_1`.
- `src/kernels/cuda/`:
  - `dequant_bf16.cu`, `iq_kernels.cu`, `router_top10.cu`.
  - `qsa_select.cu` (`qsa_block_scores[_tc]`, `qsa_block_topk`).
  - `qsa_prompt_attn.cu`, `qsa_decode_attn_batch`.
  - `native_qsa_indexer.cu`, `kv_q4.cu` (`kv_append_q4`, `fwht256`), `kv_stream.cu`.
  - `native_ple_postops.cu`, `ple_block`, the embedding gather, `cvec_apply`.

**Porting notes for these 38:**

- Warp width 32 is hard-coded in `warp_sum`/`warp_max` (`kernels.cu:20-29`), `route_kernel` (`lane = threadIdx.x & 31`)
  and `kv_append`. This maps 1:1 to Apple's 32-wide simdgroups.
- `block_sum` assumes at most 32 warps, which matches Metal's 1024-thread threadgroup limit.
- There are no atomics, async copies, cooperative groups or tensor cores in these 38 kernels. Those live in MMQ,
  cuBLAS, `qsa_*_tc` and `qsa_prompt_attn`.
- `kernels.cu` is **not** built with fast math. Only `native_*.cu` (`CMakeLists.txt:290-303`) and `strata_mmq`
  (`:884`) are, and the code picks `__expf` vs `expf` explicitly. Metal therefore needs `MTLMathModeSafe` plus
  explicit `fast::` / `precise::` calls.
- BF16 values are uint16 bit operations, which are portable. Native `bfloat` needs MSL 3.1.

### 3.5 What prefill assumes about discrete VRAM, and the Apple replacement

| Mechanism | Where | Why (discrete) | Apple unified memory |
|---|---|---|---|
| Borrowing cache slots + refill | `generate.cpp:5655-5737`; `prefill.cpp` `Alloc` `:131-153`, `relayout` `:710-733` | Scarce VRAM | Remove. Chunk buffers are ordinary `MTLBuffer`s sized against `recommendedMaxWorkingSetSize` / free RAM |
| Copy stream `m.copy` + `copied[512]`/`used[512]` events + `stage_dev` ring (8 / 96 / 384 × MAXBLOB) | `prefill.cpp:541`, `:547-550`, `carve` `:696-700`, `:1179-1251`, `:1673-1701` | Experts are in host RAM and must be DMA'd | Remove. The kernels read the blob at `arena + blob_offset(l, e)` |
| Stager threads + pinned ring | `:162-286`, `:551-564` | Unpinned blobs need a bounce | Remove for RAM-resident packs. For SSD-backed packs, keep only a CPU read-ahead (`madvise(MADV_WILLNEED)` with a 16 KiB page size, or touching threads) |
| Issuer thread | `:1206-1251` | The launch thread must not block on copies | Remove |
| Mapped grouping tables + `copy_i32` + **48 syncs per chunk** | `:567-586`, `:1589-1616`, `:1592` | Avoids copies queued behind the copy engine (saved 4.2 s of 70 s at 128K, `:335-338`) | GPU counting sort (histogram with atomics, scan, scatter), plus grouped matmuls driven by indirect dispatch or a bounds array. No host round trip per layer |
| PLE pinned double buffer + H2D | `:587-597`, `:1113-1131` | Upload | The host thread writes straight into one of two shared buffers; order with `MTLSharedEvent` |
| Token / step / bounds uploads | `:1082`, `:1135`, `:1644` | H2D | Write into shared buffers before encoding |
| KV streaming (`kv_mode==1`), host mirror | `layer.cpp:631-645`; `kv_stage_from_host` | The KV cache does not fit in VRAM from 64K (`DETAILS.md:18`) | Keep KV resident; drop the mirror |
| `cudaMemGetInfo` headroom (512 MiB) | `generate.cpp:4010-4047` | VRAM budget | `recommendedMaxWorkingSetSize`, `currentAllocatedSize`, `os_proc_available_memory()` |
| MMQ group buffers + `gather_*` copies, `MMQ_TAIL` | `prefill.cpp:1711-1728` | MMQ needs a fixed expert stride inside one launch | An indexed grouped kernel (ggml-metal `kernel_mul_mm_id`, MLX `gather_qmm` style) reading experts in place at a per-layer constant stride. Keep a finite tail (or bounds guards) |
| 64 MB FP16 dequant scratch for dense weights | `gemm.cu:414-430` | cuBLAS needs FP16 operands (a compute choice) | A quantized GEMM that dequantizes tiles into threadgroup memory and uses `simdgroup_matrix` with FP32 accumulation |

**Mac chunk rule.** At about 680 KB/token, an 8192-token chunk is about 5.4 GB, and the 384-blob ring goes away. A
replacement for "auto": pick the largest of {8192, 6144, 4096, 2048, 1024} whose `bytes_needed` stays under a fixed
fraction of `recommendedMaxWorkingSetSize` minus the model.

Keep 1024 as a floor if output parity with CUDA matters. The floor comes from kernel choice
(`prefill.cpp:79-82`), not from memory.

---

## 4. Every discrete-VRAM, PCIe and CUDA-runtime assumption, with an Apple replacement

### 4.0 What changes on Apple Silicon

Strata's design rests on three physical facts of a desktop PC:

1. VRAM (12-16 GB) is far smaller than the model. The Q2_0 experts alone are 33.97 GB.
2. The GPU reaches host RAM only over PCIe (about 26 GB/s on x16 Gen4/5, `generate.cpp:1717-1734`). That is slower
   than the CPU's own DRAM bandwidth (about 41-52 GB/s, paper Finding 1).
3. The CPU and GPU synchronise only through mapped pinned memory, driver calls, or events. A driver call costs
   10-22 µs (`elementwise.cu:192-201`).

On Apple Silicon all three are different:

- **One physical memory.** Any page can be made GPU-visible with no copy (`newBufferWithBytesNoCopy`, or
  `MTLStorageModeShared`). The only capacity limits are total RAM and `recommendedMaxWorkingSetSize`.
- **No bus.** There is nothing for a "PCIe share" to cross.
- **Host-GPU signalling is cheap.** Shared-memory flags and `MTLSharedEvent` signal/wait cover it (the latency has
  not been measured here).

Almost every item below therefore falls into one of four kinds:

- **(D) Delete.** The mechanism only moves bytes between two memories.
- **(R) Replace with a Metal primitive.**
- **(K) Keep as is.** It is host logic that does not depend on the memory model.
- **(M) Measure first.** It is a policy whose right setting on Apple is unknown.

Where Apple behaviour has not been verified, the entry says so.

### 4.1 Device memory allocation and budgeting

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| A1 | Everything the GPU reads must live in `cudaMalloc`'d VRAM, carved once from one arena with "no frees" | `include/strata/core/device.hpp:3-11`; `src/core/device.cu:196-216` (`DeviceArena`, one `cudaMalloc` at `:203`) | (R) One or a few `MTLBuffer`s (`StorageModeShared`, or `Private` for scratch the CPU never touches), with the same bump carving. Keep the "allocate once, fixed addresses" rule: Metal argument buffers / ICBs want fixed `gpuAddress`es just as CUDA graphs do |
| A2 | Poison-fill a new arena with NaN | `device.cu:22-25`, `:205-216` | (K/R) A `fillBuffer` blit or a 1-line kernel, debug only |
| A3 | Weight arena: dense weights uploaded through two pinned stages (`CHUNK`, `2·CHUNK`), one sync `cudaMemcpy` per tensor, then `cudaDeviceSynchronize` | `src/core/weights.cpp:218-230`, `:424`, `:470`; arena `cudaMalloc` `generate.cpp:1891-1892` | (D) Convert in place into the shared buffer, or mmap the converted pack and wrap it NoCopy. The "STAGE_OUT is twice STAGE_IN" growth rule (`weights.cpp:220-222`) still applies to any conversion buffer |
| A4 | The verify arena: one `cudaMalloc` of `count.used` bytes; profile stamps; E-6 `skip_` / `slot_off_d_` | `src/core/verify.cpp:290-330` (`:326`), `:334`, `:362-364` | (R) One `MTLBuffer`; `skip_` becomes a small shared buffer |
| A5 | Session state (GDN state, conv history, KV pages, PLE history) in VRAM, sized by `session_bytes(g, max_context, K, lb, le)` | `generate.cpp:2478`; `src/core/session.cpp:138-160` (`session_zero` memsets), `:428` (saved residual) | (R) A `Private` or `Shared` `MTLBuffer`. KV fits in RAM at long context; see A16 |
| A6 | Drafter (MTP) weights and scratch: 13 `cudaMalloc`s; `cudaMemGetInfo` on failure | `src/core/mtp.cpp:168-171`, `:191`, `:220`, `:229`, `:282`, `:330-331`, `:345`, `:385`, `:395-396`, `:699` | (R) Same arena pattern |
| A7 | Native dense / head weights in VRAM; the PLE/embedding table pinned, with a VRAM fallback under WSL or a HIP alias refusal | `src/core/native_dense.cpp:162`, `:176`; `src/core/native_head.cpp:47-49`, `:128-175` (`:133` mapped, `:158`/`:173` alias, `:140`/`:161` VRAM copy) | (D) No fallback is needed: the table is mmapped and NoCopy-wrapped, and the GPU gathers from it directly |
| A8 | The expert cache: `cudaMemGetInfo` pre-check plus one huge `cudaMalloc` | `src/core/expert_cache.cpp:88-162` (`:104`, `:109-121`, `:123`, `:126`, `:133`) | (D) No separate cache: the whole expert arena is GPU-visible (section 2.7) |
| A9 | Cache sizing from free VRAM minus `vram_reserve_mib`=700 (+ prefill reserve) | `generate.cpp:324`, `:2666-2690`, `:2692`, `:2705-2724` | (D) Replaced by a whole-model fit check: weights + KV + scratch ≤ `recommendedMaxWorkingSetSize` (or ≤ RAM - OS headroom). If the experts do not fit in RAM, the "cache" becomes a RAM-resident set backed by SSD (4.9) |
| A10 | Free-VRAM probes everywhere: startup trace, layer-split planning, serve headroom warning (<256 MiB → requests may stall under WDDM paging), prefill `own_fits` (+512 MiB) | `generate.cpp:808-816` (`mem_mark`), `:2150-2158`, `:2241`, `:2306-2318`, `:2492`, `:2798`, `:3934`, `:4010-4047`, `:4576-4600` | (R) One helper over `device.currentAllocatedSize`, `recommendedMaxWorkingSetSize` and `os_proc_available_memory()`. Its unit is the process's memory, not a card's |
| A11 | "Allocate the cache LAST so `cudaMemGetInfo` sees the true remainder" | `generate.cpp:2636-2662` | (D) |
| A12 | `STRATA_TEST_CACHE_FAIL=N` simulates a failed `cudaMalloc` and retries smaller; Windows commit-limit fallback | `generate.cpp:2754-2790` | (D) Or keep the shape for an `MTLBuffer` allocation-failure test |
| A13 | Eager module loading (`CUDA_MODULE_LOADING=EAGER`): a lazily-loaded kernel found no VRAM for its code at 64K+ | `generate.cpp:986-996` | (R) Build every `MTLComputePipelineState` at startup, which is good practice anyway (pipeline compilation is slow). This is not a memory issue on Apple |
| A14 | `d_res` (48×512 residency) uploaded to VRAM and kept in step with `host_res` by `apply_pending` | `generate.cpp:3610-3640`, `:4443-4457` | (D) Not needed if every expert is GPU-resident. If an SSD tier exists, keep one shared int32 table both sides read |
| A15 | Small device scratch: `d_mrope` (+ per-stage copies, `cudaMemcpy` after a `cudaDeviceSynchronize`), PLE scratch, `d_parts`, sampler `sbuf`, `d_logits`, hit scratch, penalty history | `generate.cpp:1944`, `:2115-2116`, `:2129`, `:2224`, `:2443`, `:2660`, `:3062-3070`, `:3228`, `:3237`, `:4132`, `:4808-4818` | (R) Shared buffers written by the host directly |
| A16 | KV cache does not fit VRAM from about 64K → **KV streaming**: authoritative KV in mapped pinned host pools (WSL pins only about 1 GiB), in-graph page-in `qsa_kv_resolve` | `src/core/layer.cpp:626-645` (`:634-635`); `verify.cpp` `:641`; `DETAILS.md:18` | (D) Keep KV resident in one buffer. At 262K, 8-bit KV is a few GB. Delete `kv_stream.cu`'s page-in and the host mirror |
| A17 | The rope tables and the page table are uploaded with synchronous `cudaMemcpy`, followed by `cudaDeviceSynchronize` | `layer.cpp:666-683` | (R) The host writes the shared buffer |
| A18 | State-zeroing with `cudaMemsetAsync` | `layer.cpp:692-713`; `session.cpp:138-160` | (R) `fillBuffer` on a blit encoder |

### 4.2 Mapped pinned host memory and zero-copy aliases

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| B1 | Doorbell and verify I/O live in `cudaHostAlloc(Mapped)` + `cudaHostGetDevicePointer`, with two pointers to the same bytes | `verify.cpp:72-76`, `:250-264`; `layer.cpp:994-999` (`doorbell_init`/`doorbell_free`) | (R) One `StorageModeShared` buffer. The host uses `.contents`, the GPU uses the buffer, and the two addresses are the same bytes |
| B2 | The QSA step/pos staging is mapped so per-step values can be read inside a captured graph (`copy_i32_from_mapped`), with a `cudaMemcpyAsync` fallback | `layer.cpp:619-624`, `:880-890` | (R) The host writes a shared buffer before it commits the command buffer. The kernels read it directly; there is no copy kernel |
| B3 | Session-loop `y_miss` (the CPU rows) is mapped and portable; the token-graph mapped parts | `session.cpp:531-532`, `:626`, `:843` | (D) if experts are GPU-only; (R) a shared buffer if a CPU helper is kept |
| B4 | **The expert arena is pinned with `cudaHostRegister(Portable\|Mapped)`**: whole (`:319`) or in slices after a failure (`:339`); Windows `sliced_pin_limit` (budget - usage - 4 GiB, or RAM/2 - 8 GiB); multi-GPU WDDM cap of 8 GiB; `STRATA_ARENA_PIN_GIB` | `src/core/pinned.cu:296-392`; `:250-273`; `generate.cpp:2546-2632` (`:2594-2596`) | (R) `posix_memalign`/`mmap` at **16 KiB** alignment and length, then `newBufferWithBytesNoCopy` (one buffer or a few, each within `maxBufferLength`, flagged as unverified), then add them to an `MTLResidencySet` (macOS 15+). There is no pin budget. `lock_resident`/`mlock` is still useful to keep the arena out of compression and swap |
| B5 | `lock_resident` fallback (`mlock`/`VirtualLock`) for the unpinned tail, `STRATA_ARENA_LOCK` | `pinned.cu:355-385`; `src/platform/memory.cpp:115-118` | (K/M) `mlock` exists on macOS, but its limits differ (`ulimit -l`). Measure whether wiring 34 GB is accepted, and whether a residency set already prevents compression (unverified) |
| B6 | Arena backing: hugetlb 2 MB pages, `linux/mman.h`, `MAP_HUGE_2MB` | `pinned.cu:25-34`, `:201-212` | (D) macOS has no hugetlb. `VM_FLAGS_SUPERPAGE_SIZE_2MB` is x86-only. Use 16 KiB pages; the GPU's TLB behaviour on Apple is a measurement item |
| B7 | `MAP_SHARED` arena file (`--shared-expert-arena`), header + pack hash | `pinned.cu:114-199`; header size `pinned.cu:41` | (K) It works on macOS. `kSharedArenaHeaderBytes` is **4096** (`pinned.cu:41`), so the experts start 4 KiB into the mapping. That is not 16 KiB-aligned: wrap the whole page-aligned mapping NoCopy and use a buffer offset of 4096 (or enlarge the header to 16 KiB for macOS files) |
| B8 | Arena load: 6 threads `fread` into a buffer and `memcpy` into the arena, with an FNV hash per layer | `pinned.cu:408-512`; `generate.cpp:2603` (`threads=6`) | (K) Or mmap `experts.bin` and wrap it NoCopy with no copy at all (needs 16 KiB-aligned layer starts, which `expert_layout.hpp` already guarantees). Whether file-backed pages may be NoCopy-wrapped and made resident is unverified |
| B9 | Resident complement: `cudaHostAlloc(Mapped)` of the complement, or a partial `cudaHostRegister`, plus `cudaHostGetDevicePointer`; a pinned exchange stage | `src/core/expert_source.cpp:1123-1206`, `:1352-1360` | (D) |
| B10 | Device aliases of every registered slice (`dev_slice_`) for the PCIe share (pcie_mode 1/2) | `expert_source.cpp:2236-2250` | (D) The GPU reads `arena + blob_offset(l,e)` directly. This *is* the Apple fast path, without the bus |
| B11 | Remote-GPU expert tier: mapped I/O staging, `cudaSetDevice`, `cudaInitDevice`, preflight free memory | `src/core/remote_experts.cpp:36-50`, `:78-85`, `:156`, `:186-206` | (D) There is one GPU |
| B12 | Layer-split hand-off buffers (mapped, portable) | `generate.cpp:4152-4164`; `verify.cpp:1166-1169` | (D) |
| B13 | Prefill: pinned ring, mapped grouping tables, PLE double buffer, stage jobs | `prefill.cpp:194`, `:529`, `:567-597` | (D) See 3.5 |
| B14 | Debug dump staging (`--dump-halves`, `--dump-layers`) pinned with `cudaHostAllocDefault` | `generate.cpp:1996-2003`, `:3318-3326` | (R) Shared buffers. Keep these dumps: gate C1 needs them (section 6) |
| B15 | MTP `mapped()` helper | `mtp.cpp:56-60` | (R) Shared buffer |
| B16 | `volatile` + `__threadfence_system` because a plain store to mapped memory stayed in RDNA4's L2 | `elementwise.cu:186-208`, `:276-287`; `tests/hip/handoff` | (R) On Apple, use `atomic_store_explicit(..., memory_order_relaxed)` on `device atomic_uint*`. Metal device atomics offer only relaxed order, so pair the store with `threadgroup_barrier(mem_flags::mem_device)`. Whether a GPU store reaches CPU coherence *during* a running command buffer is unverified. Do not rely on it; use the command-buffer/event boundary instead (C1) |

### 4.3 Explicit copies (H2D, D2H, D2D) and the PCIe tier

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| C-1 | `pcie_frac` (0.55 native / 0.2 canonical), scaled by a probed H2D bandwidth (≥20 GB/s keeps the base, <4 gives 0) | `generate.cpp:1716-1734`; layer-split copy `:2232-2238`; probe `:947-978` | (D) There is no PCIe share. Delete `probe_pcie_h2d_gbps`, `--pcie-frac` and `--calibrate`'s link test |
| C-2 | `pcie_mode` 0 = DMA (`cudaMemcpyAsync` + `cudaLaunchHostFunc` raises flag B), 1 = direct (kernels read mapped host memory), 2 = copy kernel `fetch_blobs` into `staging_` (default since 0.1.14, because the DMA mode deadlocked on a driver lock, #31) | `generate.cpp:4222-4227`, `:6010-6014`; `verify.cpp:1225-1253` (`:1252`); `verify_kernels.cu:218-231` | (D) All three collapse into "kernels read the arena". Mode 1 is the closest analogue |
| C-3 | Staging capacity of 16 blobs (8 per group at G=2), at most 64 fetches | `expert_source.cpp:1661-1670`; `verify.hpp:256`; `verify.cpp:312` | (D) |
| C-4 | Expert cache fills: `fill_slot` (stream), `fill_slot_blocking`, `fill_slot_queued` on stream 0, `sync_queued`; HIP bounce through a pinned blob | `expert_cache.cpp:72-86`, `:266-343` | (D) |
| C-5 | Startup cache fill from the profile and `verify_slot` (1.38 MB D2H + memcmp) | `generate.cpp:2837-2880`; `expert_cache.cpp:345-372` | (D) Or keep it as a debug check on the arena |
| C-6 | Adaptive tier: `resident_stage_swaps`, copies on a non-blocking `adapt_stream`, `adapt_ev` queries | `generate.cpp:113-163`, `:4424-4512`, `:6025-6096`; `cudaEventQuery` `:4446`, `:4450`, `:6040` | (D) if all experts are GPU-resident. (M) Repurpose it as RAM-residency management for packs larger than RAM (4.9) |
| C-7 | Prefill slot lending and refill (`STRATA_REFILL_BLOCKING`) | `generate.cpp:115-118`, `:3714-3770`, `:5655-5737` | (D) |
| C-8 | Prefill per-chunk H2D streams of every non-resident expert (copy stream + 512 events + issuer thread) | `prefill.cpp:1179-1260`, `:1656-1811` | (D) See 3.5 |
| C-9 | Legacy hit path: memset/memcpy/event per layer | `expert_source.cpp:1865-1910` (`cudaEventQuery` `:1899`, `:1910`) | (D) |
| C-10 | `copy_rows_from_mapped` brings only CPU rows across PCIe | `verify.cpp:782-784`; `elementwise.cu:235-253` | (D) with GPU-only experts. (R) with a CPU helper: the combine kernel reads the shared rows directly |
| C-11 | Q/gate split by `cudaMemcpy2DAsync` D2D | `verify.cpp:610`, `:623`; `layer.cpp:932` | (R) A strided view in the consumer kernel, or a blit |
| C-12 | Conversation-cache snapshots: `cudaMemcpy(cudaMemcpyDefault)` of state into host RAM after `cudaDeviceSynchronize` | `src/core/conversation_snapshot.cpp:113`, `:219`; `src/core/conversation_state.cpp:26-37`; `generate.cpp:4343`, `:4348`, `:4953-4976` | (R) A `memcpy` from `.contents` of the shared state buffer after the command buffer completes, or a blit into another `MTLBuffer` (which avoids the CPU copying). Keep the `STRATA_STATE_HASH` / `STRATA_CKPT_REREAD` checks (section 6) |
| C-13 | Bandwidth tool `stream_bandwidth` (H2D) | `pinned.cu:515-545` | (D) Replace with a GPU-read-bandwidth micro-benchmark over the arena. That number is the decode roofline on Apple |

### 4.4 Host↔GPU synchronisation inside a decode window

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| S1 | **Doorbell ring per layer:** GPU `doorbell_publish` → host spin on `seq` (`_mm_pause`) → host planner → flag A / B / M → GPU `wait_flag_ge` spin kernel <<<1,1>>> with `__nanosleep(100)` | `verify.cpp:1048-1121`; `verify_kernels.cu:426-429`; `elementwise.cu:276-287`; `include/strata/kernels/dp4a.hpp:36-42` | (R) **Preferred: delete the round trip.** Plan on the GPU (E-6 `resident_plan`, generalised to "all resident"), so a window is one command buffer with no host in the loop. **If a CPU helper is kept:** split the window into per-layer command buffers or encoders. Use `MTLSharedEvent`: the GPU `encodeSignalEvent(ev, 2l+1)` after the router, the host `waitUntilSignaledValue` or a listener, the host writes rows and does `signaledValue = 2l+2`, and the next encoder `encodeWaitForEvent`. **Never spin a GPU thread on a host-written flag in Metal.** Forward-progress guarantees inside one dispatch are unverified, and a GPU timeout or watchdog is likely |
| S2 | WDDM flushes: `cudaStreamQuery` right after launch; polling `cudaStreamQuery` every 2 ms in the spin; `STRATA_TG_FLUSH_US=2000` | `verify.cpp:1058-1060`, `:1080-1088`; `session.cpp:913`, `:935` | (D) `commit` submits right away |
| S3 | #267 watchdog: after 20 s, `release_gpu_waits` writes `UINT32_MAX` to the flags so a GPU stuck in a spin kernel does not hang | `verify.cpp:136-155`, `:1089-1093` | (D) No GPU spin. Keep a host timeout around `waitUntilCompleted` / event waits, and report `MTLCommandBufferStatusError` |
| S4 | `_mm_sfence` / `std::atomic_thread_fence(seq_cst)` before raising flags; volatile reads of `h_out_` | `verify.cpp:144`, `:1048-1052`, `:1106-1107`, `:1182`, `:1257`; `session.cpp:951` | (R) `std::atomic` with release/acquire on ARM64. `_mm_*` does not exist on arm64. Results are read after `waitUntilCompleted` or an event, which orders them |
| S5 | Monotonic CAS `raise_flag` (`_InterlockedCompareExchange` / `__atomic_compare_exchange_n`) | `verify.cpp:1225-1238` | (D/R) Monotonic `MTLSharedEvent` values give the same ">= ring" semantics for free |
| S6 | `cudaLaunchHostFunc` raises flag B from the copy stream | `verify.cpp:1243-1253` | (D) |
| S7 | Blocking `cudaStreamSynchronize(cs_)` and then `(copy_)` at the end of a window (polling cost 3% on IQ3_S) | `verify.cpp:1124-1131` | (R) `waitUntilCompleted`, or `MTLSharedEvent` + `MTLSharedEventListener`. Measure the wake latency: on a 25-30 ms window, 50 µs is noise, but a 1 ms scheduler quantum would not be |
| S8 | Async commit: `commit_done_` event, ordered by stream order; `wait_commit` = `cudaEventSynchronize` | `verify.cpp:1320-1360`; `generate.cpp:2181` | (R) Encode the commit into the next window's command buffer, or into its own command buffer on the same `MTLCommandQueue` (queue order is the stream order). `wait_commit` = wait on that command buffer, or on an event value |
| S9 | Sampling: greedy argmax baked into the graph; a sampled path re-runs `sample_tokens` + another stream sync | `verify.cpp:1171-1182`; `generate.cpp:5937` comment (`sampler.cu:245` `cudaDeviceSynchronize`) | (K) The same split. Philox keyed on `(seed, pos0+t)` ports as-is |
| S10 | Device-wide syncs on many paths: arena poison, layer state init, stage timing, cache readiness, MTP, generate loop, checkpoints, prefill (17 `cudaStreamSynchronize`), **48 per chunk at `prefill.cpp:1592`** | `device.cu:216`; `layer.cpp:683`, `:807`; `expert_source.cpp:1000`; `mtp.cpp:103`, `:235`, `:352`, `:402`, `:598`, `:679`, `:739`, `:762`, `:795`, `:809`; `generate.cpp:152`, `:2461`, `:2796`, `:3262`, `:3510`, `:3522`, `:3554`, `:3564`, `:4343`, `:4348`, `:4808`, `:4816`, `:5459`, `:5863`, `:5878`, `:5913`; `prefill.cpp:418-419`, `:716`, `:876`, `:1462`, `:1507`, `:1533`, `:1592`, `:1816`, `:1872`, `:1887`, `:1901`, `:1916`, `:1933`, `:1938`, `:1961` | (R) `waitUntilCompleted` on the last command buffer of a queue. Remove the prefill grouping sync (3.5). Check each one: most exist only to make a later host read safe |
| S11 | R4.2c `HitFn` seam: `Launch` before the CPU pool, `Combine` after (measured: 18.2 → 10.2 ms drain, and no token change until the two phases were split) | `include/strata/core/hit_hook.hpp:13-39` | (D) with GPU-only experts. Keep the lesson if a CPU helper stays: the GPU share must be *submitted* before the CPU starts, and decided on *this* layer's ids (the C1 regression in `hit_hook.hpp:30-36`) |

### 4.5 CUDA graphs, streams and device queries

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| G1 | One CUDA graph per T (1-8), captured lazily with `ThreadLocal` mode; a commit graph; `cudaGraphUpload` | `verify.cpp:887-1014`; `src/core/graph.cpp:41-118`; per-layer `SessionGraphs` `session.cpp:218-256`; `TokenGraph` `session.cpp:849-951`; MTP graphs `mtp.cpp:590-661` | (R) Either (a) re-encode a command buffer per window (simple; measure the CPU encode cost for about 48×30 dispatches), or (b) an `MTLIndirectCommandBuffer` per T, encoded once (the graph analogue; ICBs support compute dispatches and buffer bindings). Both share the CUDA-graph rule: **per-step values live in buffers, not in kernel arguments** (`graph.hpp`, `elementwise.cu:188-191`) |
| G2 | `CapturedGraph::wait_ms` polls `cudaEventQuery` + `_mm_pause`×64 because a memory spin saw the WDDM queue never submit | `graph.cpp:84-102` | (D) |
| G3 | Non-blocking streams: `cs_` compute, `copy_`, MTP stream, `adapt_stream`, prefill `m.cs`/`m.copy` | `verify.cpp:343`, `:347`; `prefill.cpp:541`; `generate.cpp:4426`, `:6029` | (R) One `MTLCommandQueue` for compute. A second queue only for genuinely concurrent work (the MTP drafter overlapping the commit). The copy queues go away |
| G4 | `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` above 48 KiB at 10 sites | HIP report (section 5.3); `generate.cpp:986-988` mentions the OOM | (R) Apple threadgroup memory is 32 KiB (`maxThreadgroupMemoryLength`; unverified for M4/M5). Re-tile; see 5.3 |
| G5 | Compute-capability gate `kMinCc=75` (`STRATA_EXPERIMENTAL_SM60` → 60); HIP arch + wave32 check | `device.cu:150-195` (`:185`), `:55-62` | (R) Check the `MTLGPUFamily` (Apple7+ for `simdgroup_matrix`; `bfloat` needs MSL 3.1 / macOS 14) and require `threadExecutionWidth == 32` |
| G6 | `STRATA_EMULATE_CC` changes the host's answers to CC / smem queries (tests only) | `include/strata/core/emulate.hpp:1-36`; `ggml_cuda_host.cu:84`; `generate.cpp:2557-2559` | (D) Possibly replaced by a "force the small-tile path" switch, for testing the 32 KiB variants on large-memory parts |
| G7 | `OnDevice` RAII device switch for multi-GPU; `cudaSetDevice` in prefill/remote | `include/strata/core/on_device.hpp:10-20`; `prefill.cpp:217`; `remote_experts.cpp:36-50`; `generate.cpp:1702` | (D) |
| G8 | Multi-GPU layer split: SM×clock speed model, per-stage `pcie_frac`, hand-offs | `generate.cpp:2306-2340`, `:2150-2240`; `docs/MULTI_GPU.md` | (D) |
| G9 | `device_code_error()`: "this binary has no code for sm_XY" | `generate.cpp:2561-2566`; `device.cu` | (R) Load the `.metallib` and fail clearly if a function is missing |
| G10 | `cudaGetLastError` after every launch (`check()`) | `kernels.cu:58-61`, and everywhere | (R) Metal reports errors per command buffer (`.error`, `.status`). Add `MTLCommandBufferDescriptor.errorOptions = .encoderExecutionStatus` in debug builds |
| G11 | GPU timestamps: `%globaltimer` / `wall_clock64` (`gpu_stamp`), `STRATA_VERIFY_PROFILE` | `verify_kernels.cu:546-557` | (R) `MTLCounterSampleBuffer` (stage-boundary timestamps) or command-buffer `GPUStartTime`/`GPUEndTime`. There is no in-kernel global timer (unverified) |

### 4.6 CPU-side, OS and ISA assumptions on the orchestration path

| # | Assumption | Where | Apple replacement |
|---|---|---|---|
| O1 | x86 `_mm_pause` / `_mm_sfence` in the host spins | `verify.cpp:1077`, `:1107`, `:1257`, `:144`; `graph.cpp:91`, `:100`; `session.cpp:22-26`, `:951`; `pool.cpp:405`, `:420`, `:478`, `:501`, `:715`; `pool.hpp:162`, `:180`, `:223` | (R) `__builtin_arm_yield()` / `std::this_thread::yield()` and C++ atomics |
| O2 | Thread affinity: `pthread_setaffinity_np` / `SetThreadAffinityMask`, host pinned to core 0 | `src/kernels/cpu/pool.cpp:270-309` | (R) macOS has no affinity API. Use QoS classes (`QOS_CLASS_USER_INTERACTIVE`) to keep workers on P-cores. `thread_policy_set(THREAD_AFFINITY_POLICY)` is only a hint and is ignored on Apple Silicon (unverified) |
| O3 | CPU expert kernels in AVX2 / AVX-512 VNNI / VBMI, compiled per source with `-mavx512*` | `CMakeLists.txt:740-790`, `:845-852`; `include/strata/kernels/cpu/expert.hpp:13-21`, `:80-97`, `:172-175`; `src/kernels/cpu/{expert,iq_avx2,iq_avx512,kq_avx2,q2_avx2,expert_layout}.cpp` | (R) If a CPU helper is kept, NEON `sdot`/`usdot` (i8mm on M2+) kernels. Otherwise exclude these files on arm64 (they also break the build: `immintrin.h`). The scalar reference must stay for the tests (section 6) |
| O4 | Available memory from `/proc/meminfo` and cgroup limits; `_SC_AVPHYS_PAGES` | `expert_source.cpp:219-283` (`:228`, `:241`, `:276`); `conversation_memory.cpp:43`; `generate.cpp:728`, `:748-750` | (R) `host_statistics64(HOST_VM_INFO64)`, `os_proc_available_memory()`, and `task_info(TASK_VM_INFO)` for RSS / phys_footprint |
| O5 | `posix_fadvise(DONTNEED)` to release mmapped expert layers | `expert_source.cpp:1270-1275` | (R) `madvise(MADV_DONTNEED / MADV_FREE)`. `posix_fadvise` does not exist on macOS. Keep `madvise(MADV_WILLNEED)` (`:841`). The address arithmetic must use a 16 KiB page size (`getpagesize()`), not 4096 |
| O6 | `O_DIRECT` reads | `src/platform/direct_file.cpp:263`, `:308` | (R) `fcntl(fd, F_NOCACHE, 1)` (+ `F_RDAHEAD`). The `direct_file_async` test is already `NOT APPLE` (`CMakeLists.txt:1015-1020`) |
| O7 | `linux/mman.h`, `MAP_HUGETLB` | `pinned.cu:31-34`, `:203` | (D) |
| O8 | `/dev/dxg` WDDM detection; Windows launch-context hints | `generate.cpp:100-111`, `:2617-2625` | (D) |
| O9 | `mlock` for the PLE table (`--ple-io ram`) | `src/kernels/ngram.cpp:267-270`; `generate.cpp:1463` | (K) Available on macOS, within its limits |

### 4.7 Policy knobs whose value comes from the PCIe machine

| Knob | Default and origin | On Apple |
|---|---|---|
| `--expert-cache auto`, `--vram-reserve-mib 700` | `generate.cpp:324`, `:1109-1111` | Delete, or turn into "RAM reserve for the OS" |
| `--expert-profile` | profile-first fill | Delete, or turn into the RAM-resident set for packs larger than RAM |
| `--adapt-every 4`, `--adapt-swaps 96`, usage≥2.0, margin 1.5, decay 0.7 | `generate.cpp:359`, `:381`, `:4459-4512` | Same as above. Note DETAILS.md:539-541: adaptivity makes sampled output non-reproducible on CUDA |
| `--pcie-frac`, `--pcie-mode` | `:1716-1734`, `:4222-4227` | Delete |
| `--expert-cache-per-layer` | `:1210` | Delete |
| `--prefill auto[:N]`, `--no-prefill-borrow` | `:1119-1131` | Chunk from RAM headroom; there is nothing to borrow |
| `--kv-resident` / KV streaming | `layer.cpp:631-645` | Always resident |
| `--spec-split` (G=2) | `verify.cpp:403` | Revisit only if a CPU helper is kept (Finding 5: 7% slower on CUDA) |
| `STRATA_VERIFY_DEVICE_PLAN` (E-6) | `verify.cpp:359-372` | Make it **the** design |
| `STRATA_DEC_BATCH`, `STRATA_GROUP_COPY`, `STRATA_REFILL_BLOCKING`, `STRATA_TG_FLUSH_US`, `STRATA_ARENA_PIN_GIB`, `STRATA_ARENA_LOCK`, `STRATA_REMOTE_ZEROCOPY`, `CUDA_MODULE_LOADING` | various, above | Delete |

### 4.8 Should the CPU/GPU expert split be kept?

**Why the split exists.**

- On the PC, the CPU computes the experts the GPU does not hold, *where the bytes already are*.
  - Shipping a 1.38 MB blob over PCIe at about 26 GB/s costs about 53 µs.
  - The CPU streams the same bytes from DDR5 at about 41-52 GB/s.
- The split is therefore a way to add the CPU's DRAM bandwidth to the GPU's VRAM bandwidth.
- It is also why the paper gets 1.6-1.8× from speculation: the experts' union over a window of T tokens is read once.

**What changes on Apple.**

- The CPU and the GPU **share one memory controller**. Moving experts to the CPU does not add bandwidth; it divides
  the same bandwidth.
- What the split buys on Apple is only **(a)** a second compute engine when the GPU is *not* saturating memory
  bandwidth, and **(b)** nothing at all on capacity.

**What the split costs, as measured on CUDA:**

- About 14 ms of CPU waiting plus 13 ms of GPU waiting in a 33 ms window (Finding 1).
- One host round trip per layer, 48 per window.
- All of the doorbell machinery (4.4).
- A source of nondeterminism, because the CPU and GPU kernels round differently (section 1.7; DETAILS.md:539-541).
- The R4.2c ordering bugs (`hit_hook.hpp:30-36`).

**Roofline at Q2_0.**

- A window of T=4 tokens reads at most 4×10×48 distinct expert blobs, about 2.65 GB. With overlap in routing it is
  usually less: the union per layer is 20-30 experts, so about 1.3-2.0 GB. Dense weights come on top.
- The time per window is bounded below by bytes ÷ bandwidth. On a 400 GB/s part that is about 4-6 ms. On a 120 GB/s
  base chip it is about 12-18 ms.
- A CPU helper on the same controller cannot lower that bound. It can only fill GPU idle time if the GPU's
  small-batch grouped matvec does not reach the roofline.

**Recommendation.**

1. **Phase 1: GPU-only experts, with no host in the decode loop.**
   - The window is one command buffer: router → GPU grouping (generalised E-6 `resident_plan`, all experts resident)
     → grouped experts reading the arena in place → combine → head → greedy argmax.
   - The host only writes inputs to shared buffers, commits, and waits on completion or an event.
   - This deletes S1-S7 and B1-B3, and with them all of section 1.5's protocol.
   - It also makes output independent of residency, which makes the determinism test (section 6) meaningful
     across runs.
2. **Phase 2 (optional, behind a flag, only if a measurement shows GPU idle bandwidth): a CPU helper.**
   - Port `expert_pool_dispatch_multi`'s planner as a "CPU share" fraction, not a residency decision.
   - The CPU computes rows into a shared buffer.
   - Handshake per layer with `MTLSharedEvent` across encoder or command-buffer boundaries, never with a GPU spin.
   - Use NEON kernels whose arithmetic order matches the GPU kernels bit for bit, as
     `moe_hit_grouped_s2_cpu_order` does on CUDA (`s2_expert_grouped.cu:386-460`), so output stays deterministic.
   - Expect a gain only on chips where the GPU is the bottleneck, which is a measurement item.
3. **Capacity is the real Apple constraint (4.9).** The "expert cache" logic is still useful, but one level down:
   RAM versus SSD, not VRAM versus RAM.

### 4.9 When the pack does not fit in RAM

Sizes:

- Q2_0 experts: 33.97 GB.
- The native IQ3 packs: about 42.9 GB (`pinned.cu:462` comment).
- The PLE table: 28.8 GB. It stays on SSD (mmap) by default.
- Dense weights, KV and scratch come on top.

On machines with 64 GB or more, the experts fit in RAM. On 32-48 GB machines they do not, and the design needs a
two-level store:

- **GPU-visible RAM set.** Keep a resident set of experts in one NoCopy arena, chosen by the existing profile
  (`expert_cache.cpp:12-68`) and refreshed by the adaptive tier's usage counters (`generate.cpp:4459-4512`).
  Swaps move blobs from SSD into a slot (`pread` with `F_NOCACHE`, or mmap + `memcpy`). No GPU copy is involved.
- **Misses.** For a miss there are three options, and any of them must be measured:
  1. A GPU kernel reading a mmapped-file NoCopy buffer, which page-faults from the GPU. Whether that is allowed at
     all is unverified, and its latency would be large.
  2. A CPU helper computing the miss from the page cache.
  3. A synchronous prefetch by the host after routing. That needs the router result on the host, so it is the
     doorbell again, now in shared memory and paced by `MTLSharedEvent`.
- **Where the existing code helps.** The `FileExpertSource` lookahead (`expert_source.cpp:850-930`) and the
  8-thread prefetch (`:751-811`) are the existing CS-T machinery for exactly this, and are worth keeping.

---

## 5. HIP shims and warp-32 assumptions

### 5.1 How the HIP port is wired, and why the same trick does not work for Metal

- **The HIP backend is CUDA source recompiled, not a second implementation.**
  - `cmake/hip_backend.cmake:75-83` force-includes `include/strata/hip_compat/cuda_runtime.h` into every CXX and HIP
    translation unit (`-include`, or `/FI` with an MSVC front end).
  - `:86-92` marks every `src/*.cu`, `bench/*.cu` and `tests/*.cu` as `LANGUAGE HIP`.
  - `:66-70` defines `STRATA_USE_HIP=1` and `STRATA_HIP_ARCHS`, which the runtime check in `device.cu:55-62` reads.
- **Architecture allowlist** (`hip_backend.cmake:12-35`):
  - Validated: gfx1100 and gfx1201. Community-validated: gfx1101 and gfx1200. Warned: gfx1102 and gfx1030.
  - Anything else is a fatal error. Wave64 is excluded by policy (`docs/AMD_HIP.md:8-9`).
  - There is no `-mwavefrontsize64` handling; wave32 is checked at runtime (`device.cu:59-60`).
- **Build rules.**
  - `CMakeLists.txt:35-37` makes CUDA and HIP mutually exclusive (`STRATA_ENABLE_CUDA` / `STRATA_ENABLE_HIP`, both
    OFF by default).
  - With neither enabled, only the host libraries and host tests build. That is the natural starting point for a
    `STRATA_ENABLE_METAL` option.
  - `CMakeLists.txt:304-308` sets `-ffp-contract=off` for HIP kernels, keeping FMA contraction under control.
    `:134-140` sets `-Werror` when `STRATA_WERROR` is on.
- **Why Metal needs a different approach.** MSL is a C++14-based language with its own address spaces (`device`,
  `constant`, `threadgroup`, `thread`), no pointers-to-generic-memory and no `<<<>>>`. A header shim cannot turn `.cu`
  into `.metal`. The kernels need a port, and only the **host** side can follow HIP's pattern of one runtime facade
  with backend implementations.
- **Recommendation: a thin `strata::gpu` host interface.** It covers:
  - Buffer allocation, stream/queue, event, `launch(kernel, grid, args)`, and graph/ICB capture.
  - A CUDA implementation that is today's code.
  - A Metal implementation in Objective-C++ (`.mm`) or `metal-cpp`.
- **About 870 lines in `src/core`, `src/program/generate.cpp` and `src/prefill` call the CUDA runtime** (`grep -c "cuda[A-Z][A-Za-z]*("`; generate.cpp 152, session.cpp 109, prefill.cpp 106, mtp.cpp 89, verify.cpp 68). The launch wrappers in `src/kernels/cuda` come on top.
  Most of them disappear under section 4's deletions rather than being wrapped.

### 5.2 The shim headers, line by line

| Header | Lines | What it does | Metal relevance |
|---|---|---|---|
| `hip_compat/cuda_runtime.h` | 115 | `#include <hip/hip_runtime.h>`, undefining HIP's `__noinline__` (`:12-16`). About 70 `#define cudaX hipX` renames (`:17-92`). Host-alloc flags (`:53-56`: Mapped/Portable/Default → `hipHostMalloc*`). `malloc_host` helper (`:94-99`). `cudaFuncSetAttribute` wrapper (`:101-104`). Two `cudaGraphInstantiate` overloads (`:105-111`). `__trap` → `__builtin_trap` (`:112`). `cudaMemcpyToSymbol` → `hipMemcpyToSymbol(HIP_SYMBOL(...))` (`:113`). Includes `intrinsics.hpp` | The rename list is a ready-made **inventory of the CUDA runtime surface Strata uses**. Each entry needs a Metal answer or a deletion (section 4) |
| `hip_compat/cuda_runtime_api.h` | 2 | Forwards to the above | — |
| `hip_compat/cuda_fp16.h` | 2 | `#include <hip/hip_fp16.h>` | MSL has a native `half`. Conversions are `half(x)` / `float(h)`, which round to nearest even like `__float2half_rn` (verify subnormal and saturation behaviour against `hf_sat`) |
| `hip_compat/cublas_v2.h` | 19 | Only what `gemm.cu` uses: handle, `cublasGemmEx`, op/compute/datatype enums → hipBLAS | Replace with an MPS (`MPSMatrixMultiplication`, f16/f32), an MPSGraph matmul (bf16 on macOS 14+), or Strata's own `simdgroup_matrix` kernels. Bitwise parity with cuBLAS is not expected, and the CUDA side is "not bitwise" either |
| `hip_compat/intrinsics.hpp` | 132 | `dp4a` via `__builtin_amdgcn_sudot4` (gfx11/12) or `sdot4` (gfx103x), with a portable fallback (`:17-37`). `byte_perm` via `v_perm_b32` (`:43-46`). SWAR `vsub4` / `vsubss4` / `vcmpne4` (`:55-79`). `require_full_wave_mask` traps on any partial mask (`:83-85`). `shfl_{xor,down,up,}_sync` with width 32 by default (`:87-109`). `ballot_sync` (`:111-114`). `__nanosleep` → `s_sleep(1)` (`:130`) | **This is the list of CUDA device intrinsics MSL also lacks.** The equivalents are below |

**Device intrinsics and their MSL equivalents.**

| CUDA intrinsic | Uses (from the HIP report) | MSL equivalent | Notes |
|---|---|---|---|
| `__dp4a(a, b, c)` (signed int8×4 dot) | Every q8/q2/iq gemv and the grouped expert kernels | No direct instruction. Use `char4` unpacking with `int` multiply-adds (the `intrinsics.hpp:29-35` fallback), or `simdgroup_matrix<char…>` | **Performance-critical.** Apple GPUs have no exposed int8 dot instruction (unverified for M4/M5). Q2/IQ decode is memory-bound, so the scalar fallback may be enough. Measure it |
| `__byte_perm` | i-quant table lookups | Shifts and masks, or `as_type<uchar4>` swizzles | |
| `__vsub4` / `__vsubss4` / `__vcmpne4` | Q3_K/Q6_K dots, i-quant sign expansion | Copy the SWAR versions from `intrinsics.hpp:55-79` verbatim (portable C) | Same bits by construction |
| `__shfl_xor_sync` / `_down_` / `_up_` / `_sync` (full mask) | 30 in sampler, 17 in qsa_prompt_attn, 12 in router_top10, 6 in s2_expert_grouped, and so on (35 files, counted with grep) | `simd_shuffle_xor`, `simd_shuffle_down`, `simd_shuffle_up`, `simd_shuffle`, or `simd_sum` / `simd_max` | Apple's SIMD width is 32 on every Apple GPU so far, but assert `threads_per_simdgroup == 32` at startup |
| Sub-width shuffle (`width=8`) | **`s2_expert_grouped.cu:405-407`**: `__shfl_down_sync(mask, acc, 4, 8)`, then `1`, then `2`, each wrapped in `__fadd_rn`. The deliberate order reproduces the CPU AVX reduction (`moe_hit_grouped_s2_cpu_order`) | `simd_shuffle_down` has no width parameter. Emulate it with `simd_shuffle(v, (lane & ~7) | ((lane & 7) + d))` and mask the lanes that leave the segment | Only needed if the CPU-order variant is ported. With GPU-only experts (4.8), drop it |
| `__ballot_sync` + `__popc` | `s2_expert_grouped.cu:632-638`, `:653-659` (compacting hit lists) | `simd_ballot(pred)` returns `simd_vote`; cast to `uint64_t` (`simd_vote::vote_t`) and use `popcount` | Apple's vote is 64-bit wide. Mask the low 32 bits |
| `__syncwarp()` | 10 in sampler, 4 in qsa_prompt_attn, 4 in native_qsa_score, 1 in native_flash_attn, 1 in qsa_select | `simdgroup_barrier(mem_flags::mem_threadgroup)` | |
| `__threadfence()` / `__threadfence_system()` | Doorbell publish, E-6 `resident_plan` | `threadgroup_barrier(mem_flags::mem_device)` / `atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst)` (MSL 3.2+, unverified) | `_system` only existed for host visibility, which is deleted (4.4) |
| `__nanosleep` / `strata_spin_pause` | GPU spin waits (`dp4a.hpp:12-40`) | — | Deleted along with the GPU spins |
| `__ldg` | 51 uses | `const device T*` (or `constant` for small tables) | Purely a hint |
| `__fadd_rn`, `__fmul_rn` | Ordered reductions (not fused) | Plain `a + b` under `-fno-fast-math` and `#pragma METAL fp contract(off)`, flagged as unverified | **The bit-exactness of every "ordered" kernel depends on this** |
| Atomics: `atomicOr` ×6 (sampler), `atomicAdd` (qsa_select, kv_stream), `atomicCAS` ×1 | | `atomic_fetch_or_explicit`, `atomic_fetch_add_explicit`, `atomic_compare_exchange_weak_explicit` (relaxed only) | A float `atomicAdd` would need MSL 3.0+ `atomic<float>`. Check whether any use is float |
| `__constant__` tables + `cudaMemcpyToSymbol` | `s2_gemv_fast.cu:44`, `:59`; `s_gemv.cu:32`; `dequant_bf16.cu:33` (IQ4_NL table) | `constant` address-space arrays in the `.metal` source (compile-time), or a `constant` buffer argument | `s_gemv.cu:20-31` notes the constant-memory version cost 2.12×, so a threadgroup copy may be better. Measure |
| `%globaltimer` (`verify_kernels.cu:545-551`) | Profiling | — | Use counter sample buffers (G11) |
| Inline PTX `mma.sync` (m16n8k8 tf32, m16n8k8/k16 f16) | `qsa_prompt_attn.cu:48-57`; `qsa_select.cu:168-188`; `native_qsa_score.cu:49` | `simdgroup_float8x8` / `simdgroup_half8x8` with `simdgroup_multiply_accumulate` (8×8 tiles; or MSL 4 tensor ops on M5-class, unverified) | **Start with the ordered FP32 fallback** that both CUDA (sm<80) and HIP already have (`docs/AMD_HIP.md:13-14`). Apple GPUs have no tf32, so "3xTF32" does not apply; use plain f32 `simdgroup_matrix`, which is exact f32 FMA |
| `cvt.rna.tf32.f32` | `qsa_select.cu:178` | — | As above |
| `ldmatrix.sync` | `native_qsa_score.cu:39-44` | `simdgroup_load` from threadgroup memory | |
| `cp.async` (+ commit/wait) | `fused_gr.cu:575-592`; `qsa_prompt_attn.cu:402-412`; `qsa_select.cu:168` (comment) | Plain loads into threadgroup memory, double-buffered by hand | Metal has no async-copy instruction. Registers and the `threadgroup` prefetch pattern are what `gdn_rec_cols_pipe` already does |
| gfx12 WMMA builtins | `qsa_prompt_attn.cu:723`; `qsa_select.cu:308` | — | Show that Strata accepts per-backend matrix-core code paths behind the same contract |

### 5.3 Threadgroup-memory limits (dynamic shared memory over 48 KiB)

There are eight `cudaFuncSetAttribute(..., MaxDynamicSharedMemorySize, ...)` sites in `src/` (grep). The earlier
"ten" also counted the header define and the `generate.cpp:988` comment.

| Site | Kernel | Bytes requested | Apple (32 KiB, `maxThreadgroupMemoryLength`; unverified that M3/M4/M5 are still 32 KiB) |
|---|---|---|---|
| `qsa_prompt_attn.cu:639` | `prompt_attn_i8_kernel` | `sizeof(Smem2)` | Re-tile (fewer queries per threadgroup) |
| `qsa_prompt_attn.cu:670` | `prompt_attn_kernel<KV_MODE>` | Kernel smem struct | Re-tile |
| `qsa_select.cu:717` | `block_scores_tc_kernel` | `(TC_QT·TC_QS + TC_NB·TC_KS)·4` | Re-tile, or use the non-tc path |
| `fused_gr.cu:726`, `:728` | `gr_down_multi_kernel<1280 / 2560>` | `kFusedGrMaxT(8) × tile × 4` = **40 KiB / 80 KiB** | Neither fits. 640-wide tiles (20 KiB), or 4 tokens per launch at 1280. The existing "slice the tokens" fallback (`fused_gr.cu:707-712`: outputs are strictly per token) is exactly the needed mechanism |
| `fused_gr.cu:732` | `gr_down_staged_kernel` | `2·8·H_TILE·4` | Check H_TILE |
| `fused_gr.cu:836` | `gr_down_v3_kernel<1>` | `8·2560·4` = 80 KiB, else a split in 2 (40 KiB) | It needs a split in 4 (20 KiB). The code already has the split logic |
| `qsa.cu:684` | `qsa_attend_kernel` | Depends on `max_ids`; `exit(1)` if over the limit | **Hard failure.** Make `max_ids` per-threadgroup tiling or a two-pass softmax |

HIP had the easy case here: gfx11/12 LDS is 64 KiB, and `fused_gr.cu:713-714` simply forces `small_tile` on HIP. On
Apple, every kernel above needs a tile size chosen for 32 KiB. The existing `STRATA_EMULATE_CC=75` mechanism
(`emulate.hpp:27-34`, answering 64 KiB) is the model for a "pretend 32 KiB" test switch, so the small paths can be
validated on CUDA before Metal exists.

### 5.4 Warp-32 assumptions

Strata hard-codes 32 lanes everywhere. Apple SIMD-groups are also 32 wide, which is the single biggest piece of luck
in this port. These assumptions carry over unchanged, but each one should be backed by a startup assertion:

- **Lane arithmetic.** `lane = threadIdx.x & 31`, `warp = threadIdx.x >> 5`, `WARP=32` constants
  (`native_mmvq.cu:45-46`), `warp_sum` / `warp_max` loops `for (o = 16; o > 0; o >>= 1)` (`prefill/kernels.cu:20-29`
  and about 35 files).
  - MSL: `thread_index_in_simdgroup`, `simdgroup_index_in_threadgroup`. **Do not derive them from
    `thread_position_in_threadgroup`.** Metal does not guarantee that a 1-D threadgroup is split into simdgroups by
    consecutive thread index (unverified, but in practice it is).
- **`block_sum` assumes at most 32 warps per block** (`prefill/kernels.cu:45-57`: `__shared__ float[32]`). Metal's
  maximum is 1024 threads per threadgroup, so at most 32 simdgroups. That matches, but `maxTotalThreadsPerThreadgroup`
  can be lower for a pipeline with heavy register use. Query it per pipeline and choose the launch shape from it.
- **Hit counts by file** (HIP report): s2_expert_grouped 70, sampler 46, router_top10 23, iq_kernels 23,
  qsa_prompt_attn 19, qsa_select 18, native_mmvq 16, gr 16, fused_gr 15.
- **Full masks only.** Every `*_sync` call passes `0xffffffff` (`require_full_wave_mask` traps otherwise,
  `intrinsics.hpp:83-85`). MSL SIMD functions have no mask, and inactive lanes return undefined values. Kernels with
  early-exiting lanes before a shuffle must be restructured so all 32 lanes reach it. The HIP port already enforced
  this, so the risk is low.
- **Bit-order assumptions.**
  - The ballot result is used as a 32-bit mask with `(1u << lane) - 1` prefix popcounts (`s2_expert_grouped.cu:634`,
    `:659`). On Apple, take the low 32 bits of `simd_vote`.
  - The top-10 router breaks ties by the lower index and turns NaN into `-FLT_MAX` through warp argmax
    (`prefill/kernels.cu:454-490`; `router_top10.cu`). It is portable as long as the shuffle-reduction order is
    kept.
- **Shuffle widths of 8** (`s2_expert_grouped.cu:405-407`) are the only sub-warp segments; see 5.2.
- **Reduction order equals the bits.** Many parity tests are bitwise (section 6). A Metal kernel that changes the
  butterfly order (`simd_sum` order is unspecified; unverified) will not match the CUDA bits. Use explicit
  `simd_shuffle_xor` butterflies where a bitwise contract exists, and `simd_sum` elsewhere.

### 5.5 Lessons from the HIP port that apply to Metal directly

1. **No cross-backend bitwise claim** (`docs/AMD_HIP.md:16`: "This does not claim bit-identical model answers
   across backends"). The contract is within-backend determinism plus the model-level KL and spec-exactness gates
   (section 6). Metal should adopt the same contract explicitly.
2. **Host-visible stores from a GPU need explicit visibility.** RDNA4 kept plain stores in L2 (`elementwise.cu:203-206`,
   `tests/hip/handoff`). On Metal, GPU writes become CPU-visible at command-buffer completion (or at an event
   signal). Design for that boundary instead of in-flight visibility (4.2 B16).
3. **Mapped aliases are not portable** (#325, `native_head.cpp:154-160`: Windows HIP gives the host address, and a
   D2D copy into it does not land). Metal shared buffers are the clean version of what HIP approximated.
4. **Per-architecture tuned GEMM** (hipBLASLt tuning files, `prefill/hipblaslt_tuning.hpp:26-111`). Expect the same
   need for per-family (M1/M2/M3/M4/M5) tile tables.
5. **The HIP-only tests** (`tests/hip/*`: intrinsics, handoff, mapped_alias, expert_cache_staging,
   native_qsa_score, prefill_native_batch, router_fast, prefill_mmq_parity, ple_iq4, prefill_hipblaslt_gemm) show the
   shape of a backend bring-up suite. Section 6.4 lists their Metal twins.

---

## 6. Test strategy: what exists, and what a Metal port needs

### 6.1 Method, and the state of the public tree

**The method is a reference chain, not self-comparison.** The rule is stated at `CMakeLists.txt:220-222`: "a kernel
checked against another copy of itself proves nothing". Every GPU kernel is compared with a scalar host reference
that was itself validated against llama.cpp/ggml (gate C1, 6.3). The kernel tests then pin one of two contracts:

- **Bitwise**, where the arithmetic order is fixed by construction. Examples:
  - `quantize_act_parity` uses `memcmp` on the round trip and on values (`src/kernels/quantize_act_parity.cpp:91,106,201`).
  - `qsa_parity` asserts that batched equals sequential, bit for bit, on state words
    (`src/kernels/qsa_parity.cpp:1386-1440`, and the "moved" checks :1448, :1526-1542).
  - `hip_router_fast` compares ids and weights of 65,536 rows that include ties, near-ties, NaN and all-equal rows
    (`tests/hip/router_fast.cpp:1-3`).
  - The `multi_exact` MMVQ layouts (`src/kernels/mmvq_multi_parity.cpp:24`).
- **Tolerance against FP64/FP32**, where the order is free (gemv, GEMM, attention). Examples are
  `hip_prompt_attn_wmma` (FP32 kernel and FP64, `CMakeLists.txt:455-460`) and `prefill_mmq_parity` against CPU FP32
  (`tests/hip/prefill_mmq_parity.cpp:3`).

**Negative controls are built in.** The same comparison with the exact mode switched off must find differences,
otherwise the test has no power (`mmvq_multi_parity.cpp:24,181,237`; `qsa_parity.cpp:1298,1373`). The project's own
warning is at `CMakeLists.txt:630-631`: "A test that passes by finding nothing to compare is the failure mode this
project keeps re-learning."

**How tests report a skip.**
- **Exit 77** means skip (`SKIP_RETURN_CODE 77`). Examples:
  - `native_dense_ple_key_test` :364
  - `expert_cache_per_layer_test` :369
  - `conversation_snapshot_test` :375
  - `hip_mapped_alias` :435
  - `hip_prompt_attn_wmma` :459
  - `hip_prefill_hipblaslt_gemm`
- **Exit 3** means skip for the IQ fixture pair (`CMakeLists.txt:527-533`). This fires when numpy or the vendored
  gguf-py is missing.
- **Some tests fail rather than skip** when hardware or a fixture is absent. `cuda/hip_device_selftest` (:704/:706)
  does this, and `ple_parity` fails because `bench/micro/ple_in.bin` and `ple_out.bin` are absent
  (`CMakeLists.txt:634-639`, `src/kernels/ple_parity.cpp:9,257`).

**The public tree is trimmed.** A port team must know this before relying on "the tests":
- **No CI.** `.github/` contains only `FUNDING.yml`.
- **Default build.** `STRATA_BUILD_TESTS` defaults OFF unless `tests/CMakeLists.txt` and `bench/micro` exist
  (`CMakeLists.txt:52-57`), and both are missing. `STRATA_ENABLE_CUDA` and `STRATA_ENABLE_HIP` both default OFF (:35-37).
  The GPU block (:181-736) still registers its parity tests whenever a GPU backend is on.
- **Missing pieces:**
  - `bench/micro` (and with it the PLE fixtures);
  - `scripts/`, `tests/python`, most of `ref/` (only `load.py` remains) and `gguf-py`;
  - `tools/gpu_session.py`, `tools/kv_precision_compare.py`;
  - the sources of the `oracle` foreach (:328), `native_mmvq_multi` (:680), `graph_registry` (:716) and
    `pinned_capture` (:728). These are guarded by `if(EXISTS ...)` or silently missing.
- **Counts.** A default build registers about 33 ctests, and about 61 with both GPU options. The private tree had 70
  (`bench/results/2026-09-27-kv-q4/README.md:19`).

### 6.2 Inventory of ctests

All line numbers refer to `/tmp/strata-upstream/CMakeLists.txt`. "Metal" says what to do with each test.

**Host, backend-independent tests: keep as-is and run them on macOS from day one.**

| Test | Line | What it checks | Metal |
|---|---|---|---|
| `conversation_cache_test`, `conversation_memory_test` | :65, :68 | prefix cache, memory budget | keep |
| `gguf_reader_test`, `gguf_split_test` | :162, :166 | GGUF parsing, split shards | keep |
| `expert_arena_load` | :212 | arena load from `pack/full/experts.bin` (needs a real pack) | keep, gate on fixture |
| `qsa_select_bench` | :284 | QSA selection host bench | keep |
| `file_expert_source_test` | :350 | expert file reader | keep (check the `F_NOCACHE` path, 4.5 O-rows) |
| `expert_layout_test` | :358 | native expert layout (`tests/data/native_experts`) | keep |
| `native_dense_ple_key_test` | :363 | dense PLE key (skip 77) | keep |
| `expert_cache_per_layer_test` | :368 | per-layer cache sizing (skip 77) | keep while the cache exists; drop with it in Phase 1 |
| `conversation_snapshot_test` | :374 | snapshot save/load equality (skip 77) | keep |
| `conversation_validation_test` | :378 | snapshot validation | keep |
| `conversation_transfer_test` | :386 | transfer; `NOT APPLE` (:380) | port, then remove the guard |
| `suffix_drafter`, `controller`, `draft_policy`, `conv_cache`, `coupled_draft`, `bf16_bits` | :755-775 | spec/drafting logic | keep |
| `expert_multi`, `expert_parity`, `pool_test` | :998, :1004, :1009 | CPU expert kernels against scalar | keep scalar; add NEON if the CPU helper is kept (4.8) |
| `pool_stress` | :1013 | AVX-512 pool stress | x86-only: make conditional |
| `direct_file_async` | :1019 | O_DIRECT async reads; `UNIX AND NOT APPLE` (:1015) | rewrite for `F_NOCACHE`/`F_RDADVISE` or drop |
| `platform_memory_test` | :686 | `lock_resident` on 256 MiB (`src/platform/memory_test.cpp:1`, `memory.cpp:115` mlock) | keep; on macOS mlock is the only knob (no `ulimit -l` note) |
| `pinned_shared_test` | :691 | Linux MAP_SHARED backing of PinnedArena (`src/core/pinned_shared_test.cpp:1`) | drop (no pinned arena on Metal) |

The AVX-512 flags are applied unconditionally to `src/kernels/cpu/expert.cpp` (`CMakeLists.txt:784-789`, including
`-mavx512vnni;-mavx512vbmi`). On arm64 they must be guarded by `CMAKE_SYSTEM_PROCESSOR`, or clang will reject the TU.

**GPU kernel parity tests (block :181-736): each needs a Metal twin.** The host reference in each `*_parity.cpp` is
the asset to keep. The CUDA/HIP launch half is what gets replaced.

| Test | Line | Contract | Metal priority |
|---|---|---|---|
| `quantize_act_parity` | :592 | bitwise (`memcmp`) | P0 |
| `elementwise_parity` | :607 | bitwise / tight | P0 |
| `dequant_s2_parity` | :486 | exact dequant | P0 |
| `s2_gemv_parity`, `s2_gemv_q8_parity` | :491, :571 | tolerance vs FP64 | P1 |
| `s_gemv_parity`, `s_gemv_q8k_parity`, `bf16_gemv_parity` | :602, :617, :612 | tolerance | P1 |
| `iq_parity_fixtures` + `iq_parity` | :527, :532 | IQ dequant/gemv against gguf-py fixtures (skip 3) | P1, but restore gguf-py first |
| `iq_multi_parity` | :552 | multi-row IQ | P1 |
| `mmvq_multi_parity` | :545 | CUDA-only negative control of `multi_exact` | P1: keep the idea (negative control on simdgroup layouts) |
| `router_top10_parity` | :597 | ids exact, weights tight | P2 |
| `sampler_parity` (x3) | :576, :579, :581 | exact | P2 |
| `rope_parity` | :587 | tolerance | P2 |
| `shared_expert_parity` | :556 | tolerance | P2 |
| `gr_parity` | :561 | fused GR | P3 (smem re-tile, 5.3) |
| `gdn_parity` | :566 | GDN recurrence/state | P3 |
| `qsa_parity` | :622 | batch == sequential bitwise | P3 (hard 32 KiB exit at `qsa.cu:684`, 5.3) |
| `ple_parity` | :637 | against ggml graph capture | P3: recreate `ple_in.bin`/`ple_out.bin` first |
| `kv_q8_parity`, `kv_stream_parity`, `kv_hybrid_parity`, `kv_q4_parity` | :645, :651, :657, :662 | KV storage vs host and FP16 | P2 |
| `cvec_parity` | :666 | control vectors | P2 |
| `s2_expert_grouped_parity` | :673 | grouped expert GEMV (ballot at `s2_expert_grouped.cu:632,653`) | P4 (largest kernel; last) |
| `prefill_mmq_kquant_test` | :901 | MMQ k-quant prefill (`tests/cuda`) | P4: Metal GEMM path |
| `native_expert_parity` variants | :958-965 | native expert format | P4 |
| `cuda/hip_device_selftest` | :704, :706 | `strata-device --selftest` | write `metal_device_selftest`; make it skip (77) with no GPU |

### 6.3 Model-level gates

None of these gates is scripted in the public tree. All of them are reachable from `strata-generate` flags.

**C1: logits parity with llama.cpp** (`src/generate.cpp:9-15`). The same token ids are fed through `--tokens`, and
logits are compared with llama.cpp.
- **Hooks:**
  - `--dump-logits` (:1042).
  - `--dump-layers` (:1052). It is written only by `session_loop`, at :3203-3208, so it is not available on the
    verify path.
  - `--dump-halves` (:1053), `--dump-routing` (:1054) and `--prefill-until` (:1131).
- **Dump format** (`bench/results/2026-09-27-esp/esp_kl.py:52-54`): int32 `[n_vocab, rows]`, then f32 rows.
- **Why it matters:** the regression history shows the gate earning its keep. `include/strata/core/hit_hook.hpp:35` (and `src/core/expert_source.cpp:1494`) records
  KL rising from 9.69e-02 to 1.03.
- **Metal:** this is the first end-to-end gate. It needs a T=1 path with the dump hooks before anything is
  optimised (6.5 step 3).

**KL against the reference distribution.**
- The paper reports KL 0.022 (paper p.6). The number does not appear in the code.
- The procedure is in `docs/UNSLOTH_Q4.md:170-205`. It uses `STRATA_LOGPOS` (`generate.cpp:5056`) and
  `STRATA_LOGPOS_TOPK` (`verify.cpp:1295-1301`, `verify.hpp:131`).
- **Metal:** run the same procedure, with a CUDA/HIP run of the same pack as the second reference. The pass threshold
  is that the KL to llama.cpp should be no worse than CUDA's.

**Speculative == greedy, bit for bit.**
- The claim is made at `verify.hpp:1-18`.
- **Hooks:** `--spec-oracle` (`generate.cpp:1134`, applied at :5973-5981) feeds perfect drafts. `--spec-corrupt`
  (:1135, :6158) injects wrong drafts, so rejection is exercised.
- **The comparison:**
  - The `--spec 0` stream runs token-by-token through `session_run_token`/`session_loop`/`session_token`
    (`generate.cpp:5810-5830`).
  - The spec loop runs from :5963-5964 and requires `d_res` (:5983-5987).
  - The two must produce identical token streams.
- **Known exception:** #152, IQ multi-threading. It is cured by `STRATA_IQ_MT_MIN=1`.
- **Metal:** this gate is the main correctness test for the one-command-buffer verify window (section 1/4.8). Any
  T-dependent reduction order in the Metal GEMV/attention kernels will break it. Either make every kernel
  T-invariant in its per-row arithmetic order (the CUDA design), or document the deviation the way #152 does.

**Conversation cache equality.**
- Host test: `conversation_snapshot_test`.
- Engine hooks:
  - `STRATA_CKPT_REREAD` (`generate.cpp:4950-4980`, :5018) re-reads a checkpoint and continues from it.
  - `STRATA_STATE_HASH` (:5455-5562, GDN-state variant :5497) hashes the KV and recurrent state.
- Python gates `tools/conversation_cache_{parity,disabled,growth,isolation,soak,http_smoke}.py` (`STATE_KEYS` at L18,
  `verify_results` at L107-128). They drive the server and compare outputs with the cache on and off.
- **Metal:** these port unchanged once the server runs. `STRATA_STATE_HASH` needs the state readable from host
  memory. On unified memory that is a plain read after the command buffer completes, which is simpler than on CUDA.

**Determinism** (`docs/DETAILS.md:78-91`).
- IQ3_XXS gives 1 distinct answer out of 4 runs with `--prompt-cache 0 --adapt-swaps 0 --pcie-frac 0`, and 2 out of 4
  otherwise. The documented causes are the CPU/GPU expert split and the adaptive tier.
- `docs/AMD_HIP.md:16` makes no bitwise claim across backends.
- **Metal:** with GPU-only experts (Phase 1, 4.8) and no adaptive tier, the expected result is 1 distinct answer out
  of N without any flags. Make this an explicit gate. If a CPU helper is added later (Phase 2), the gate shows
  whether it reintroduces nondeterminism.

**Python suites that need no GPU.**
- `tools/test_*.py`: 21 files covering setup, conversation_cache, iq_pack, mtp_fetch, strata_mcp, shards and calibrate.
- `serve/test_*.py`: 7 files (detok, lifecycle, mcp, monitor, server, structured, winjob).
- `serve/chat_golden.json` has no runner.
- `tools/canonical_xcheck.py`, `tools/iq_fixture.py` (feeds `iq_parity`) and `tools/needle_bench.py`.
- These should run on macOS unchanged, except where they spawn CUDA tools. `winjob` is Windows-only.

### 6.4 Metal twins of the HIP bring-up tests

`tests/hip/*` is the template for a backend bring-up suite. They were written to catch exactly the classes of bug
that a new backend introduces. The table maps each one to a Metal equivalent.

| HIP test | What it proves on HIP | Metal twin | Notes |
|---|---|---|---|
| `intrinsics.cpp` (:417-421) | CUDA intrinsics from `hip_compat/intrinsics.hpp` match numerically on 32 lanes (`kLanes = 32`) | `metal_intrinsics`: `simd_shuffle(_xor/_down)`, `simd_sum`, `simd_ballot`, `simd_prefix_*`, `dp4a` replacement, half/bf16 conversions, `__expf`/`fast::exp` against host | must also pin `simd_sum` reduction order (unverified, 5.2), since bitwise tests depend on it |
| `handoff.cpp` (:423-427) | async CPU-to-GPU doorbell: graph replay with `doorbell_wait` + `copy_from_mapped` over 100 rounds with changing payloads (`tests/hip/handoff.cpp:1,24-27`) | `metal_shared_event_handoff`: CPU writes into a shared `MTLBuffer`, signals `MTLSharedEvent` value n, GPU command buffer waits on n, consumes and signals n+1. Repeat 100+ rounds with changing payloads | replaces the doorbell protocol (4.2 B-rows). Also a negative test: no GPU spin on host memory (unverified forward progress, 4.2) |
| `mapped_alias.cpp` (:429-436) | whether a `hipHostGetDevicePointer` alias is usable (#325; exits 0 as a diagnosis) | `metal_nocopy_alias`: `newBufferWithBytesNoCopy` over (a) a page-aligned anonymous allocation, (b) a file-backed `mmap` of an expert pack slice. GPU writes, CPU reads after completion, and the reverse | answers the open question "can NoCopy wrap a file-backed mmap" (unverified, 4.1). Skip 77 with a clear message if (b) is refused |
| `native_qsa_score.cpp` (:437-442) | scalar HIP QSA scorer against host | `metal_qsa_score` | same host reference |
| `prefill_native_batch.cpp` (:443-448) | native QSA append + embedding gather (`tests/hip/prefill_native_batch.cpp:1-3`) | `metal_prefill_native_batch` | same host reference |
| `router_fast.cpp` (:449-453) | AMD fast router == portable kernel BITWISE on 65,536 rows incl. ties/NaN (`tests/hip/router_fast.cpp:1-3`) | `metal_router_fast` against the portable router. Copy the row generator verbatim | tie-breaking and NaN order must match exactly, or spec==greedy (6.3) breaks |
| `qsa_prompt_attn_parity` as `hip_prompt_attn_wmma` (:455-460) | RDNA4 WMMA prompt attention vs FP32 kernel and FP64; skips off gfx12 | `metal_prompt_attn_simdgroup_matrix` with `simdgroup_matrix` (or the Apple-family matrix path), skip 77 on families without it | tolerance contract, three-way (fast / FP32 / FP64) |
| `prefill_hipblaslt_gemm.cpp` (:461-468) | tuned hipBLASLt table loads and matches hipBLASEx (`:1-3`) | `metal_prefill_gemm_table`: per-family tile table loads; MPS/MPSGraph or custom GEMM matches an FP32 reference on the real prefill shapes | per-family tables (5.5 item 4) |
| `prefill_mmq_parity.cpp` (:937-942) | ggml MMQ (with q8_1 activation rounding) vs CPU FP32 (`:1-3`) | `metal_prefill_mmq_parity` against ggml-metal's `mul_mm` on the same blocks | ggml-metal is a free second reference on Apple |
| `ple_iq4.cpp` (:946-) | real PLE IQ3_XXS/IQ4_XS: ggml CPU dequant vs HIP MMVQ, on a model shard (`:1-2`) | `metal_ple_iq4`, same shard input | needs a model; gate behind a fixture-path option |
| `expert_cache_staging.cpp` (:413-416) | `ExpertCache::open_sized` growth and reuse with pageable payloads (`tests/hip/expert_cache_staging.cpp:12-15`) | none in Phase 1 (the cache and staging are deleted, 4.8). Replace with `metal_residency_set`: add/remove arena buffers to an `MTLResidencySet`, commit, and confirm GPU reads see the data | reinstate a cache test only if packs larger than RAM need a RAM tier (4.9) |

**Tests to drop with no twin, because the mechanism does not exist on Metal:**
- `pinned_shared_test` (:691), `pinned_capture` (:728, already absent) and every `cudaHostRegister`/pinned-staging path.
- `direct_file_async` in its O_DIRECT form (:1015-1019). Keep it only if the SSD tier of 4.9 is built, and then
  rewrite it around `F_NOCACHE`/`F_RDADVISE`/`preadv`.
- The `--pcie-frac` part of the determinism recipe (6.3). There is no PCIe tier.
- `graph_registry` (:716, absent). Graph capture becomes command-buffer encoding; see 6.5 for what replaces it.

### 6.5 Recommended Metal test plan, in order

1. **Make the host suite green on macOS first (week 0).**
   - Guard the AVX-512 options (`CMakeLists.txt:784-789`) and `pool_stress` (:1013) on x86.
   - Remove `NOT APPLE` from `conversation_transfer_test` (:380) once it builds.
   - Rewrite or skip `direct_file_async`.
   - Add a GitHub Actions `macos-15` (arm64) job that runs the host ctests and the Python suites (`tools/test_*.py`,
     `serve/test_*.py`). Today there is no CI at all, so this is a net gain for every backend.
2. **Metal kernel harness built from the existing host references.**
   - Split each `*_parity.cpp` into a backend-neutral reference/generator part and a launch part. Add a Metal launch
     part (Objective-C++ or metal-cpp).
   - Keep the exact pass criteria, including the negative controls (6.1).
   - Bring kernels up in priority order:
     - P0: `quantize_act`, `elementwise`, `dequant_s2`. These are bitwise and catch layout and intrinsic bugs.
     - P1: gemv/dequant families (s2, s, q8k, bf16, IQ). Restore gguf-py so that `iq_parity` stops skipping.
     - P2: router, sampler, rope, shared expert, KV, cvec.
     - P3: GR, GDN, QSA, PLE. All need re-tiling for 32 KiB of threadgroup memory (5.3), and QSA has a hard exit
       at `qsa.cu:684`.
     - P4: the grouped experts and the prefill GEMM.
   - Numerics settings:
     - Compile with `MTLMathModeSafe` (or `fastMathEnabled = NO`) for every bitwise test.
     - Make FMA contraction explicit (`fma()` where the CUDA code relies on it, and no contraction elsewhere).
     - Keep `precise::` math for exp/log in the sampler and router.
     - Record the Apple GPU family and OS version in each test's output.
3. **T=1 end-to-end verifier with the dump hooks, before any optimisation.**
   - Implement `--dump-logits`, `--dump-layers` and `--dump-routing` on the Metal path, in the format read by
     `esp_kl.py:52-54`.
   - Note that `--dump-layers` today exists only in `session_loop` (`generate.cpp:3203-3208`). The Metal path should
     emit it from the verify window too, so per-layer bisection works for T>1.
   - Script gate C1 against llama.cpp (ggml-metal is available on the same machine) and the KL procedure from
     `UNSLOTH_Q4.md:170-205`.
4. **Scripted model-level gates, run per release on a real Mac with the model.**
   - Spec == greedy: compare `--spec 0` with `--spec-oracle` and with `--spec-corrupt`. They must give identical
     streams.
   - Determinism: N=8 repeats of the DETAILS.md prompt must give 1 distinct answer without flags.
   - Conversation cache: the `tools/conversation_cache_*.py` gates, plus a `STRATA_STATE_HASH` comparison between a
     cold run and a `STRATA_CKPT_REREAD` run.
   - Keep these as scripts that exit 0, 77 or 1, so they can join ctest under a `STRATA_MODEL_DIR` option.
5. **Recreate the missing artifacts.**
   - `bench/micro/ple_in.bin` and `ple_out.bin`, captured from ggml as described at `ple_parity.cpp:9`. Until then,
     `ple_parity` fails on every backend.
   - A small real-pack fixture for `expert_arena_load` (:212), and for the Metal NoCopy test on a file-backed pack.
   - The `tests/CMakeLists.txt`/`bench/micro` pair that `STRATA_BUILD_TESTS` keys on (:52-57). Alternatively,
     change the default so a fresh clone registers its tests.
6. **Cross-backend guard on the existing backends.** Add a CUDA/HIP build option, "pretend 32 KiB smem", that caps
   every `cudaFuncSetAttribute(MaxDynamicSharedMemorySize)` request (the eight sites, 5.3) at 32 KiB. It also forces
   the small-tile branches. The CUDA parity tests then validate the re-tiled kernel shapes on hardware that already
   works, before the Metal versions exist. This separates "tile logic wrong" from "Metal translation wrong".
7. **Metal-specific tests with no CUDA analogue.**
   - `metal_shared_event_handoff` (6.4).
   - `metal_nocopy_alias`, which covers file-backed mmap.
   - `metal_residency_set`.
   - A `maxBufferLength` probe that checks the arena split into several `MTLBuffer`s (4.1) addresses every expert
     correctly. Use a synthetic arena just over the limit, filled with unique per-slot patterns.
   - A visibility test. Data written by the GPU must not be read by the CPU before the command buffer completes or
     its shared event signals. The test records what is seen in flight, as a diagnosis like `hip_mapped_alias`, so
     the design never comes to depend on in-flight visibility (5.5 item 2).
   - A command-buffer timeout/watchdog test for the longest prefill window.

**Summary of the test position.** Strata's strongest asset for the port is its scalar host references with explicit
bitwise contracts and negative controls. Its weakest point is that the public tree runs almost none of them by
default: there is no CI, `STRATA_BUILD_TESTS` is off, and fixtures and scripts are missing. Its model-level gates also
exist only as flags and documentation. A Metal port should first make the host suite and a macOS CI job green. It
should then reuse every host reference for a Metal kernel harness, and only after that turn the four model-level gates
into scripts. The Metal path is GPU-only, with one command buffer per window, so it should *pass a stricter
determinism gate than the CUDA build does today*.
