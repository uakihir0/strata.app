> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# Part 01: Hyper-connections (GR), router, shared expert, elementwise, RoPE / M-RoPE / YaRN, control vectors

Scope: the CUDA/HIP kernels and the host glue for the residual hyper-connection ("GR", `build_hc_mix`), the MoE
router (softmax + top-10 + renormalisation), the shared expert's gate and the MoE combine, the generic elementwise
kernels, rotary embeddings (NEOX pairing, YaRN/linear scaling, interleaved M-RoPE) and control vectors (cvec).
Everything here was read from the read-only clone at `/tmp/strata-upstream`. All paths below are relative to
that root, and every `file:line` refers to that tree.

Conventions:
- `ne0` is the contiguous axis. "row k of W" means `W[k*n_in + i]`.
- `bf16f(u) = as_float(u << 16)`.
- `fma(a,b,c)` is a single-rounding fused multiply-add.
- `__expf`, `rsqrtf` and fast-math division are the CUDA approximate intrinsics.
- "Contracted" means nvcc turned `a*b+c` into an FMA. nvcc does this by default on every file, with or without
  fast-math.
- Each "Metal" note says whether to use `simd_shuffle_xor`, `simd_sum` or `simdgroup_matrix`, gives the closest
  ggml-metal kernel, or says the kernel can be dropped on unified memory.

---------------------------------------------------------------------------------------------------------------

## 1. Geometry and constants

| Constant | Value | Source |
|---|---|---|
| n_embd `N` | 2560 | include/strata/core/layout.hpp:27-55; fused_gr.cu:20 |
| n_layers | 48 (36 GDN + 12 QSA; `layer % 4 == 3` is QSA) | layout.hpp:27-55 |
| hc (streams) | 4 | layout.hpp; fused_gr.cu:21 |
| hc_dim `D = hc*N` | 10240 | fused_gr.cu:22 |
| hc_lr (GR bottleneck) | 320 | fused_gr.cu:23 |
| n_expert / top-k | 512 / 10 | layout.hpp |
| n_ff (routed and shared expert) | 640 | layout.hpp |
| n_head / n_head_kv / head_dim | 24 / 2 / 256 | layout.hpp |
| indexer q heads / key dim | 4 / 128 | layout.hpp |
| n_rot | 64. NEOX pairing: pair i (0..31) rotates x[i] with x[i+32]; dims 64..head_dim-1 pass through | rope.hpp:42-45, native_rope.cu:46 |
| rope freq_base | 1e7 | src/core/qsa.hpp:125; rope_scaling.hpp:65-74 default |
| RMS eps (GR, qk-norm) | 1e-6f | src/core/layer.cpp:45 (`RMS_EPS`); FusedGrArgs.eps default (fused_gr.hpp:24) |
| Router weight clamp | 6.103515625e-05 = 2^-14 (ggml's clamp on the top-k weight sum) | router_top10.cu:196; native_router.cu; prefill/kernels.cu:490 |
| IMROPE sections | {11, 11, 10, 0} | include/strata/kernels/mrope.hpp |
| Fused GR max tokens | kFusedGrMaxT = 8 | include/strata/kernels/fused_gr.hpp:48 |
| YaRN defaults | type None, factor 1, freq_scale_in 0, orig_ctx 262144, ext 0, attn 1, beta_fast 32, beta_slow 1 | rope_scaling.hpp:65-74 |

### 1.1 Tensors

Defined in src/core/layout.cpp. Engine "forms" are checked at layer.cpp:1190-1200.

| Tensor (per layer unless noted) | Engine form | Shape [rows][ne0] | Line |
|---|---|---|---|
| `hc_attn_down.weight` | Bf16InF32 (bf16 bits, 2 B/elem) | [320][10240] | layout.cpp:53 |
| `hc_attn_up.weight` | Bf16InF32 | [10240][320] | layout.cpp:54 |
| `hc_attn_inject.weight` | Bf16InF32 | [4][10240] | layout.cpp:55 |
| `hc_ffn_{down,up,inject}.weight` | Bf16InF32 | same as attn | layout.cpp:56-58 |
| `hc_attn_norm.weight`, `hc_ffn_norm.weight` | F32 | [10240], indexed `[c*2560 + d]`; **stored as (1 + w)** by the converter (gr.hpp:81) | layout.cpp:97-98 |
| `ffn_gate_inp.weight` (router) | bf16 | [512][2560] | layout.cpp:60 |
| `ffn_gate_shexp`, `ffn_up_shexp` | quantized (another part's slice) | [640][2560] | layout.cpp:61-62 |
| `ffn_down_shexp` | quantized | [2560][640] | layout.cpp:63 |
| `ffn_gate_inp_shexp.weight` (shared gate) | Bf16InF32; the only 1-D bf16 tensor | [2560] | layout.cpp:99 |
| `attn_q_norm`, `attn_k_norm` | F32 | [256] | layout.cpp:103-106 |
| `indexer.q_norm`, `indexer.k_norm` | F32 | [128] | layout.cpp:103-106 |
| Head: `output_hc_norm`, `output_hc_down`, `output_hc_up` | same forms as the per-layer GR, **no inject** | as above | layer.cpp:1060-1094 |

Activations and their buffers, in bb.* (layer.cpp):

| Buffer | Type | Shape | Notes |
|---|---|---|---|
| `R` | f32 | [4][2560] per token | the residual streams |
| `mixed` | f32 | [2560] | |
| `inject` / `inject2` | f32 | [4] | attn / ffn injection |
| `block_out` | f32 | [2560] | |
| `gr.lo` | f32 | [320] | |
| `gr_rs` | f32 | [4] | |

The GR workspace (gr.cu:303-338) has five regions, each 16-byte aligned:

| Region | Type | Size |
|---|---|---|
| `xn` | f32 | D |
| `xq` | u16 | D |
| `lq` | u16 | 320 |
| `gated` | f32 | D |
| `lo` | f32 | 320 |

`gr_read` checks the region sizes (gr.hpp:63-75).

---------------------------------------------------------------------------------------------------------------

## 2. Numeric primitives (shared by all kernels)

### BF16 and FP16 conversion

From include/strata/kernels/bf16_bits.hpp:
```
f32_from_bf16(h)  = as_float(uint32(h) << 16)
bf16_from_f32(f):                                  // bf16_bits.hpp:37-47, RNE
    i = as_uint(f)
    if isnan(f): return (i >> 16) | 64             // quiet-NaN bit forced
    return (i + ((i >> 16) & 1) + 0x7FFF) >> 16
```

Prefill's `bf()` (src/prefill/kernels.cu:30-34) is the same RNE **without** the NaN case, and
`bf_lo(f, hi) = bf(f - f32_from_bf16(hi))` (prefill/kernels.cu:37).

Packed bf16 pairs in a `uint32 v`: the first element is `as_float(v << 16)`, the second is
`as_float(v & 0xffff0000)` (fused_gr.cu:42-44).

`f16_from_f32` is in f16_bits.hpp:37+ (RNE, used by `to_f16_kernel`).

### Sigmoid and SiLU variants

These differ bit-for-bit, so a port must pick the one for the path it targets.

| Name | Expression | Where |
|---|---|---|
| gr.cu `silu_f`, `sigmoid_f` | `x/(1+expf(-x))`, `1/(1+expf(-x))`; precise expf, IEEE division (gr.cu is not fast-math) | gr.cu:36-60 |
| fused_gr / cvec `sigmoidf_` | `1.0f/(1.0f+__expf(-x))`, IEEE division | fused_gr.cu:35; cvec.cu:43 |
| fused down SiLU | `x/(1.0f+__expf(-x))` with `x = acc/(float)HC`, IEEE division | fused_gr.cu:101-102 |
| native_gr_postops `sigmoid` | `1/(1+expf(-x))` compiled with `--use_fast_math`, so `__expf` and approximate division | native_gr_postops.cu:36 |
| prefill `sigm` | `1/(1+__expf(-x))`, IEEE division | prefill/kernels.cu:38 |
| shared expert native sigmoid | `__fdividef(1, 1+__expf(-g))` | shared_expert.cu:133-141 |
| shared expert canonical | double `1/(1+exp(-acc))`, then cast to float | shared_expert.cu:109-126 |
| elementwise `silu_kernel` | double `v/(1+exp(-v))`, then cast (matches numpy reference) | elementwise.cu:72-80 |

### Fast-math scope

CMakeLists.txt:288-303.
- `--use_fast_math` applies on CUDA only, to: native_mmvq, native_bf16, native_gr_norm, native_moe, native_gdn,
  native_gr_postops, native_router, native_gdn_preprocess, native_qsa, native_rope, native_ple_postops,
  native_flash_attn, native_qsa_indexer and native_qsa_score.
- The HIP branch does **not** use fast math for native_*. HIP only gets `-ffp-contract=off` on elementwise.cu and
  quantize_act.cu.
- Not fast-math: gr.cu, fused_gr.cu, cvec.cu, router_top10.cu, shared_expert.cu, rope.cu and prefill/kernels.cu.
  fused_gr, cvec and prefill call `__expf` explicitly; their division is IEEE.
- Under fast-math, `expf` becomes `__expf` (ex2.approx), `x/y` and `1/x` become approximate, and
  `powf`/`cosf`/`sinf` become `__powf`/`__cosf`/`__sinf`. Denormals flush to zero.
- `rsqrtf` is always the MUFU approximation, whatever the flags.

### Reduction shapes

Order matters for bitwise parity.

| Reduction | Shape | Where |
|---|---|---|
| gr.cu `warp_sumf` | `__shfl_down_sync` 16,8,4,2,1, then `__shfl_sync(v, 0)` broadcast | gr.cu:62-70 |
| gr.cu `block_sumf` | leading `__syncthreads`, warp_sumf, lane0 writes `scratch[warp]`, warp 0 re-reduces `scratch[tid < nw]` with warp_sumf, thread 0 writes `scratch[0]`, all threads read it. The scratch array is declared `double[8]`. A double `block_sum` at gr.cu:102 exists but is unused on these paths | gr.cu:85-101 |
| fused_gr / cvec / prefill / native_* `warp_sum` | xor butterfly 16,8,4,2,1; every lane holds the result | fused_gr.cu:30-34 |
| native_gr_norm block reduce | xor warp sum, `sums[32]`, then **every** warp re-reduces `lane < BS/32 ? sums[lane] : 0` with xor. This pins llama.cpp ggml-cuda's `block_reduce` | native_gr_norm.cu:62-71 |
| native_bf16 mmvf | strided pairs per thread; `acc = fma(lo, x0, acc); acc = fma(hi, x1, acc)`; xor warp sum; `partials[32]` zero-initialised; thread `t < 32` xor-reduces `partials[t]` | native_bf16.cu:44-75 |
| prefill `block_sum` | xor warp sum, `sh[w]`, warp 0 xor sum of `sh[tid < nw]`, broadcast `sh[0]` | prefill/kernels.cu:44-57 |

---------------------------------------------------------------------------------------------------------------

## 3. Hyper-connection (GR) math

### 3.1 Semantic definition

This is the contract in include/strata/kernels/gr.hpp:79-95. Per token, `R` is [4][2560] f32, with stream c at
`R + c*N`.
```
gr_read(R, w_norm, w_down, w_up, w_inject):
  for c in 0..3:
      rs[c] = 1 / sqrt( mean_d R[c,d]^2 + eps )                 // eps = 1e-6
      xn[c*N+d] = R[c,d] * rs[c] * w_norm[c*N+d]                // w_norm already holds (1 + w)
  lo[k]     = silu( (sum_i a(xn)[i] * w_down[k,i]) / hc )        k = 0..319   (the /hc is INSIDE the silu)
  gate[i]   = sigmoid( sum_k a(lo)[k] * w_up[i,k] )              i = 0..10239
  mixed[d]  = (1/hc) * sum_c xn[c*N+d] * gate[c*N+d]             d = 0..2559
  inject[c] = sum_i a(xn)[i] * w_inject[c,i]                     c = 0..3, only when w_inject != null
  where a() is bf16 rounding (canonical BF16 contract) or the identity (FP32 contract; all production paths)

gr_write(R, block_out, inject):
  R[c,d] += block_out[d] * 2 * sigmoid(inject[c] / hc)

embedding:  R[c,d] = e[d] for every c             (broadcast_streams; verify_kernels.cu:203/533, prefill gr_broadcast)
head:       mixed = gr_read(output_hc_norm, output_hc_down, output_hc_up, inject = null)
            logits = output.weight @ q8_K(mixed)  (layer.cpp:1060-1094)
```

Each layer runs, in order:
1. attn read (`hc_attn_*`)
2. the QSA or GDN block on `mixed`, giving `block_out`
3. the attn write with `inject`
4. the FFN read (`hc_ffn_*`)
5. the MoE (router, shared expert and combine), giving `block_out`
6. the FFN write

### 3.2 Decode dataflow with the fused read (production `--native`)

layer.cpp:1160-1250 and 1290-1316.

```
fused       = g_fused_gr && stage_prefix == 0 && half == 0 && fused_gr_supported(2560, 4, 320)      (1160)
pending_ffn = fused && layer > 0 && !cvec().covers(layer-1)                                         (1163)

if PLE ready && layer == 1:                                           (1164-1196)
    if pending_ffn: gr_write(R, block_out, inject2);  pending_ffn = false
    ple_block(...)                  // in place on R
    history advance

ATTN READ  fused_gr_read{ R_out = R, apply = pending_ffn, bo_prev = block_out(prev MoE), inj_prev = inject2,
                          inject_out = inject, lo = gr.lo, rs = gr_rs, mixed }                       (1219-1226)
block      qsa_layer / gdn_layer(mixed) -> block_out
           (non-fused only: gr_write(R, block_out, inject))                                         (1233)
FFN READ   fused_gr_read{ apply = true, bo_prev = block_out(attn), inj_prev = inject,
                          inject_out = inject2, ... }                                                (1242-1249)
           (non-fused: gr_read(..., inject) - note the non-fused path reuses `inject`, not `inject2`)
moe_route(mixed); moe_shared early (g_shared_early = true)
POST       moe_combine_parts -> block_out                                                           (1294)
           if !fused:                     gr_write(R, block_out, inject);   if steer: cvec_apply(write = false)
           elif layer == n_layers-1:      gr_write(R, block_out, inject2);  if steer: cvec_apply(write = false)
           elif steer:                    cvec_apply(write = true, bo = block_out, inj = inject2)   (1299-1309)
           (otherwise the FFN write stays pending; the next layer's attn read folds it in)
```

So in fused decode, a layer's FFN write is applied inside the next layer's attn read, and the attn write inside
that layer's FFN read. The standalone `gr_write` (the native `post` kernel when `gr_native_mmvf` is set) runs
only at the PLE layer 1 and the last layer. Both reads keep `R_out == R`, in place.

`fused_gr_read` rejects `apply && inj_prev == inject_out` (fused_gr.cu:884-899). The inject block of gr_down
writes `inject_out` while other blocks read `inj_prev`.

### 3.3 Canonical op-level (src/kernels/cuda/gr.cu, library default, no fast-math)

There are two contracts: the BF16-activation default (`xq = bf16(xn)` and `lq = bf16(lo)`, the llama.cpp CPU
contract) and FP32 via `gr_set_fp32_activations(true)`.

```
gr_norm_kernel<FP32>     grid hc, 256 thr                                      gr.cu:136
    ss = sum_{d = tid; d += 256} R[c,d]^2           (per-thread serial, then block_sumf)
    ms = block_sumf(ss) / (float)n_embd             (IEEE division)
    rs = rsqrtf(ms + eps)
    x  = (R[c,d] * rs) * w_norm[c*N+d]               xn = x;  if !FP32: xq = bf16(x)
gr_down_kernel<Act>      grid 320, 256 thr, one block per output row k           gr.cu:165
    acc = sum over i = tid, tid+256, ... of act(xq[i]) * bf16f(w_down[k,i])  (contracted FMA, serial per thread)
    warp_sumf -> part[warp] -> warp 0: t = sum(part[lane < 8])
    lo[k] = silu_f(t / (float)hc)                    stored as bf16 (lq) when Act = uint16, f32 (lo) when Act = float
gr_gate_kernel<Act>      grid ceil(D/8), 256 thr, one warp per row i             gr.cu:234
    acc = sum over k = lane, lane+32, ... (< 320) of act(lq[k]) * bf16f(w_up[i,k]);  warp_sumf
    gated[i] = xn[i] * sigmoid_f(acc)
gr_mean_kernel           grid ceil(N/256)                                        gr.cu:251
    m = 0;  for c in 0..3: m += gated[c*N+d];  mixed[d] = m / (float)hc
gr_inject_kernel<Act>    <<<1, 32*hc>>>, warp c = stream c                       gr.cu:266
    acc = sum over i = lane, lane+32, ... of act(xq[i]) * bf16f(w_inject[c,i]);  warp_sumf;  inject[c] = acc
gr_write_kernel          grid ceil(D/256), 256 thr, dynamic smem hc floats        gr.cu:283
    w[c] = 2 * sigmoid_f(inject[c] / (float)hc)
    R[i] = R[i] + block_out[d] * w[c]                (contracted to fma(block_out, w, R))
```

Host `gr_read` (gr.cu:340-411) dispatches on two flags:
- If `native_mmvf`, it runs the native six-kernel path (§3.4).
- Otherwise it runs norm, down, gate, mean and inject in the FP32 variant (gr.cu:375-377) or the BF16 variant
  (gr.cu:380-382). The mean is at gr.cu:386 and the inject at gr.cu:394-396.

Host `gr_write` (gr.cu:413-430) calls `native_gr_post` when `native_mmvf` is set, else `gr_write_kernel`
(gr.cu:421).

The history comments explain the design choices:
- FP64 accumulation cost 5.07 ms/token; FP32 brought it to 2.38 ms (round 282).
- "The ground truth is llama.cpp, and llama.cpp accumulates in FP32."
- The FP64 inject cost 3.70 ms/token.

### 3.4 Native op-level (`gr_native_mmvf`; pins llama.cpp CUDA's graph: rms_norm, mul_mat_vec_f, unary ops)

gr.cu:345-372. Uses FP32 activations; the `fp32_activations` flag is ignored.

1. `native_gr_rms_norm_weighted(R, w_norm, xn, n_cols = 2560, n_rows = 4, eps)`
   - native_gr_norm.cu:46, launched at native_gr_norm.cu:94-96.
   - Block size is 1024 because n_cols ≥ 1024, so each thread serially sums `R[col]^2` for col = tid, tid+1024,
     tid+2048.
   - Reduction: xor warp sum, `sums[32]`, then every warp xor-reduces.
   - `mean = partial / n_cols` (fast-math division); `scale = rsqrtf(mean + eps)`.
   - `xn = (scale * R) * gamma`. This is bitwise the same as canonical `(R*rs)*w`, because multiplication commutes;
     the difference lies in how rs is computed.
2. `bf16_gemv_fp32_mmvf(xn, w_down) -> lo[320]`
   - native_bf16.cu:44, one block per row, BS = 256 (`mmvf_block_size(10240)`).
   - Each thread covers pairs p = t, t+256, ... (20 pairs) with `acc = fma(w_lo, x0, acc); acc = fma(w_hi, x1, acc)`.
   - Reduction: xor warp sum, `partials[32]`, xor.
3. `native_gr_down_silu(lo)`: `x = fma(1/hc, lo, 0)` (exact for hc = 4), then `lo = x/(1+expf(-x))` under
   fast-math (native_gr_postops.cu:42).
4. `bf16_gemv_fp32_mmvf(lo, w_up) -> gated[10240]`, with BS = 160 (one pair per thread for n_in = 320, so 5 warps).
5. `native_gr_pre_gated<Fused>(xn, gated, mixed)` (native_gr_postops.cu:49):
   ```
   for c in 0..3:
       x = xn[c*N+d];  w = sigmoid(gated[c*N+d]);  gated[c*N+d] = x*w (rn)
       Fused   (w_inject != null, i.e. layer mixers):  sum = fma(x, w, sum)      starting at 0
       Unfused (head mixer):                           sum = c == 0 ? x*w : sum + x*w     (all _rn)
   Fused:    mixed = scale*sum              (scale = 1/hc = 0.25, exact)
   Unfused:  mixed = fma(scale, sum, 0)
   ```
6. If `w_inject` is set: `bf16_gemv_fp32_mmvf(xn, w_inject) -> inject[4]` (BS 256).

`native_gr_post` (native_gr_postops.cu:65) handles the write:
```
weight = fma(2, sigmoid(fma(1/hc, inject[c], 0)), 0)
out[i] = fma(block_out[d], weight, residual[i])        // aliasing out == residual is allowed
```

### 3.5 Fused op-level, single token (src/kernels/cuda/fused_gr.cu, decode default under `--native`)

This is two kernels per read. The pending write of the previous half is folded in (`apply`).
```
gr_down_kernel  <<<41, 256>>>  static smem: xn[10240] f32 (40 KB) + part[8][4] + s_rs[4]         fused_gr.cu:49
  gw[c] = apply ? 2*sigmoidf_(inj_prev[c] / 4.0f) : 0
  for i = 4*t; i < D; i += 1024:                      // float4; c = i / 2560 (a float4 never straddles streams)
      r = R[i..i+3]
      if apply: r.* = fma(bo_prev[d..].*, gw[c], r.*)
      sq = r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w       // left to right; nvcc contracts into an fma chain
      ss[c] += sq
      xn[i..] = r.* * w_norm[i..].*
  per c: v = warp_sum_xor(ss[c]); lane0: part[warp][c] = v
  t < 4:  s = 0; for w in 0..7: s += part[w][t];  s_rs[t] = rsqrtf(s / 2560.0f + eps);  block 0 writes a.rs[t]
  xn[i] *= s_rs[i / N]                                 // so xn = (R'*g)*rs, two roundings
  row = (bid == 40) ? warp : bid*8 + warp;  the inject block keeps warps 0..3 only, and only if w_inject != null
  acc = 0
  for j = lane; j < 1280; j += 32:  acc += dot8(w_row[j], xn + 8j)     // #pragma unroll 4; a PLAIN add of each dot8
  acc = warp_sum_xor(acc)
  lane 0:  inject block: inject_out[row] = acc
           else:         x = acc / 4.0f;  lo[row] = x / (1.0f + __expf(-x))

  dot8(w, x): acc = 0; for j in 0..3: acc = fma(lo16(w[j]), x[2j], acc); acc = fma(hi16(w[j]), x[2j+1], acc)
  (fused_gr.cu:38-47)

gr_up_kernel  <<<80, 256>>>  smem lo[320] + g[4][32]                                                fused_gr.cu:107
  d0 = bid*32
  for r = warp; r < 128; r += 8:   c = r / 32;  dd = r % 32;  i = c*N + d0 + dd
      acc = dot8(w_up_row_i[lane], lo + 8*lane)
      if lane < 8: acc += dot8(w_up_row_i[32 + lane], lo + 8*(32 + lane))       // plain add of the second dot8
      acc = warp_sum_xor(acc)
      lane 0: rv = R[i]
              if apply: rv = fma(bo_prev[d0+dd], 2*sigmoidf_(inj_prev[c] / 4.0f), rv);  R_out[i] = rv
              x = (rv * w_norm[i]) * rs[c]                   // bitwise the xn of gr_down
              g[c][dd] = x * sigmoidf_(acc)
  t < 32:  s = 0;  for c in 0..3: s += g[c][t];  mixed[d0+t] = s / 4.0f
```

### 3.6 Fused multi-token (verify window and MTP: `fused_gr_read_multi`, T = 1..8)

fused_gr.cu:141-878. All tokens share the weights. These kernels are **bitwise equal** to calling the
single-token read once per token; `fused_gr_selftest` and gr_parity.cpp:351-355 check this.

| Kernel | Line | Launch | What it does |
|---|---|---|---|
| `gr_norm_multi_kernel` | 148 | one block per token | Same math as gr_down's norm step. Writes `a.rs` and `xn` to global `m.xn + bid*D`, then applies `*= s_rs` in place |
| `gr_norm_split_kernel` (#315) | 522 | grid (T, 4) | Same element visits as the plain norm, gated by `if (i/N != c) continue`. Serial sum of the 8 warp partials, then rsqrtf. Recomputes R' and writes `xn = (r*g)*rs` |
| `gr_down_multi_kernel<TILEV>` | 200 | grid 41, dynamic smem `tile[T][TILEV]` | TILEV = 2560, or 1280 on sm_75 and HIP. Per tile: prefetch `wv[q] = w4[base/8 + lane + 32q]`, sync, stage the tile from `m.xn`, sync, then `acc[k] += dot8(wv[q], tile + k*TILEV + 8j)`. Then xor warp_sum; lane k writes token k's `lo` or `inject` with the same epilogue. Chunk order is `lane + 32q` ascending, which is why it is bitwise |
| `gr_down_staged_kernel` | 619 | double-buffered, 2 × T × 1280 floats (80 KB at T = 8) | Loads the next tile with `cp.async.cg` (sm_80+, cp_async16 at fused_gr.cu:568-576; plain float4 copy elsewhere) while computing the current one. `stage_htile` (605-612) lays tiles out as [T][2 planes][160] float4; `dot8v` (591-602) is the same eight fmas on two float4. Bitwise equal to plain. On an RTX 4080S the norm fell from 0.58 to 0.23 ms per 4-token round |
| `gr_up_multi_kernel` | 258 | 160 blocks × 16 columns | smem `lo[8][320]` + `g[8][4][16]`. Lane < T fetches token lane's R, w_norm, rs, bo and inj. Per token k: dot8 + dot8, warp_sum; lane == k keeps it. Same epilogue; `mixed = (sum_c g[k][c][col]) / HC` |

`down_chunk` (fused_gr.cu:690-753) slices tokens by shared-memory capacity, with the smem opt-in.
`launch_multi` (fused_gr.cu:762-787) runs:
1. the split norm if variant ≥ split, else norm_multi
2. the down kernel per chunk
3. `gr_up_multi<<<160>>>`

Variant selection:
- `fused_gr_check` (fused_gr.cu:1046-1080, called once per card at verify.cpp:198) runs `fused_gr_selftest`
  (fused_gr.cu:902+). The selftest uses mt19937 seed 20260930, down weights 0.02·N(0,1), up weights 0.05·N(0,1),
  T = 1..8 and apply = 0/1, and compares outputs bitwise.
- It picks staged if staged passes, else split, else plain (`fused_gr_variant` at 1036-1044).
- The `STRATA_HC_SPLIT` env var overrides this (fused_gr.cu:790-796):

  | Value | Variant |
  |---|---|
  | unset or "2" | staged (kHcStaged = 3) |
  | "0" | plain (kHcPlain = 1) |
  | "1" | split (kHcSplit = 2) |

**v3** is opt-in with `STRATA_GR_V3=1` and is **not bitwise**:
- `gr_down_v3_kernel<S>` (fused_gr.cu:324) runs on grid (41, 4·S). It computes unscaled per-stream partial dots
  `p_c` plus sums of squares.
- `gr_up_v3_kernel<S>` (fused_gr.cu:408) finishes with `lo = silu((Σ_c fma(rs[c], p_c, sum)) / HC)`.
- S = 2 fails (#375). gr_parity.cpp:336-346 compares v3 at 2e-6 × max|ref|.

### 3.7 Prefill GR

src/prefill/prefill.cpp:1325-1340 and 1835-1857, with kernels in src/prefill/kernels.cu. Prefill is **not
bitwise** with decode: the projections are BF16-activation GEMMs.

```
gr_norm_rs_kernel (84)        one block per (t, c)
                              rs = rsqrtf(block_sum(Σ r^2) / N + eps)
                              xn16 = bf((r*rs)*w)            (+ the bf_lo image for STRATA_PREFILL_BF16X2)
bf16_proj(down, xn16) -> lo   (GEMM)
gr_silu_kernel (160)          x = lo/HC;  lo16 = bf(x/(1+__expf(-x)))
bf16_proj(up, lo16) -> g
bf16_proj(inject, xn16) -> inj
gr_mix_r_kernel (102)         s = 0
                              for c: x = (R*rs[c])*w;  s = fma(x, sigm(g), s)
                              s /= HC
                              mixed (+ bf16/f16 images)
  [STRATA_GR_UNFUSED=1: gr_norm_kernel (65) + gr_mix_kernel (170) instead]
write: gr_write_norm_rs_kernel (128)   x = fma(bo, 2*sigm(inj/HC), r);  then the NEXT half's rs and xn16 from x
                                       (fused unless last layer, PLE layer 1 or cvec)
       else gr_write_kernel (191)      R = fma(bo, 2*sigm(inj/HC), R)
       then cvec_apply(write = false) if covered (prefill.cpp:1853-1856)
gr_broadcast_kernel (198)     R[t,c,d] = e[t,d]
```

`STRATA_PREFILL_BF16X2` (prefill.cpp:108-121) controls the hi+lo second BF16 operand:

| Value | Applies to |
|---|---|
| 0 (default) | off |
| 1 | also the HC projections |
| 2 | all but HC |

### 3.8 Write and norm flavours (they are not interchangeable bitwise)

| Flavour | Write expression | Sigmoid |
|---|---|---|
| canonical gr_write_kernel | `fma(bo, 2*sigmoid_f(inj/hc), R)` | precise expf, IEEE division |
| native post | `fma(bo, fma(2, sigmoid(fma(0.25, inj, 0)), 0), R)` | fast-math expf and division |
| fused fold / cvec(write) / prefill | `fma(bo, 2*(1/(1+__expf(-inj/4))), R)` | `__expf`, IEEE division |

| Flavour | Norm product | rs reduction |
|---|---|---|
| canonical | `(R*rs)*w` | stride-256 shfl_down + double-typed scratch; IEEE `/N` |
| native | `(rs*R)*w`, which equals the canonical product bitwise | stride-1024 xor, all-warps re-reduce; fast-math `/N` |
| fused | `(R*w)*rs`, a **different product** | per-stream stride-1024 float4 with an in-thread `sq` FMA chain, xor warp, serial 8-warp sum; IEEE `/N` |
| prefill | `(R*rs)*w` | stride-256 xor block_sum |

| Flavour | Mixer |
|---|---|
| native layers | `0.25 * Σ_c fma(x, w, ·)` |
| native head | `fma(0.25, ((p0+p1)+p2)+p3, 0)` |
| fused | `(((0+g0)+g1)+g2+g3)/4` with `g = x*sigmoid`, rounded |
| canonical | `(Σ_c gated)/hc` |
| prefill | `Σ fma(x, sigm, s) / 4` |

---------------------------------------------------------------------------------------------------------------

## 4. Router (softmax over 512, top-10, renormalise)

Logits come from `project_bf16` (layer.cpp:94-100):
- native: `bf16_gemv_fp32_mmvf(mixed f32, ffn_gate_inp)`, a 2560-wide row with BS 256
- otherwise: `f32_to_bf16_bulk` + `bf16_gemv_split` (TPR 32)

Router selection (layer.cpp:355-389): `native_router_enabled() && n_expert == 512 && k == 10` uses
`native_router_top10`; anything else uses `router_top10`. Then `doorbell_publish` copies x, ids and weights to the
host and rings the CPU expert pool (or memcpy + `doorbell_ring`).

### 4.1 Canonical `router_top10_kernel` (router_top10.cu:69, FP64, not fast-math)

```
one block per token, nt = min(512, ...) threads
smem: taken[512 rounded to 16] u8 | double s_ex[512] | float s_p[512];  static s_red[16], s_rid[16], double s_sum
mx = max_e l[e]    (fmaxf strided, shfl_down tree, warp 0 tree; exact)
s_ex[e] = exp((double)l[e] - (double)mx)
thread 0: sum = Σ_{e = 0..511 ascending} s_ex[e]  (serial double)
inv = (float)(1.0 / sum)
s_p[e] = (float)(s_ex[e] * (double)inv)
for i in 0..k-1:                                           // k passes of block argmax
    per thread: bv = -inf, bi = 512;  scan e ascending, skip taken, take strict pe > bv
    shfl_down merge: take other if (ov > bv || (ov == bv && oi < bi));  warp 0 merges the warp winners the same way
    tid0: ids[i] = ix;  weights[i] = v;  taken[ix] = 1
tid0: s = Σ_i (double)weights[i] (rank order);  sc = fmax(s, 2^-14);  weights[i] = (float)((double)weights[i] / sc)
```

The result is a stable descending top-k with ties broken by lower index. Weights are softmax-over-all-512
probabilities renormalised over the selected 10 (ggml's `norm_w` with clamp).

`router_top10_fast_kernel<FORCE_SERIAL>` (router_top10.cu:206) is HIP only and **bitwise equal** to the canonical
kernel:
- It sums with a tree, and uses the tree sum when `(float)(1/x)` maps to one float over the bracket
  `[tree(1-2^-40), tree(1+2^-40)]`. Otherwise thread 0 falls back to the serial sum.
- Selection runs on wave 0 from registers (16 probabilities per lane).
- `STRATA_HIP_ROUTER_OLD=1` forces the old kernel.
- Covered by tests/hip/router_fast.cpp:64-71 (memcmp).

Host code is at router_top10.cu:345-399, with k ≤ 64.

### 4.2 Native `route` (native_router.cu:45, fast-math; pins llama.cpp CUDA topk-moe)

```
__launch_bounds__(256, 1); <<<n_tok, dim3(32, 8)>>> (native_router.cu:115/123); logits += blockIdx.x * 512
threads with threadIdx.y != 0 RETURN (before a __syncthreads)        -> effectively one warp per token
values[i] = logits[lane + 32*i], i = 0..15                           // expert id = lane + 32*i
maximum = xor-butterfly fmaxf of the per-lane max
values[i] = expf(values[i] - maximum)  (__expf);  sum += values[i]  (i ascending, per lane)
reciprocal = 1 / warp_sum_xor(sum)                                     (approximate under fast-math)
values[i] *= reciprocal;  if isnan: values[i] = -FLT_MAX
for rank in 0..9:
    best = values[0];  expert = lane;  for i in 1..15: if values[i] > best: best = values[i]; expert = lane + 32i
    xor butterfly 16..1:  take other if (other > best || (other == best && other_id < expert))
    if (expert & 31) == lane: values[expert / 32] = -INF;  selected_sum += best
    ids[rank] = expert (lane 0);  if rank == lane: selected = best
selected_sum = max(warp_sum_xor(selected_sum), 6.103515625e-5f)
inverse = 1 / selected_sum                                             (approximate)
lane < 10: weights[lane] = selected * inverse
```

Hosts: `native_router_top10` at native_router.cu:110 and `native_router_top10_multi` at native_router.cu:119.

Verify (verify.cpp:672-686): with `dec_batch && n > 1 && native router` it uses `bf16_gemv_fp32_mmvf_multi` +
`native_router_top10_multi`; otherwise `moe_route` per token.

### 4.3 Prefill `route_kernel<REG>` (prefill/kernels.cu:454, not fast-math)

- 8 warps per block, one warp per token. REG is 16 for 512 experts, 8 for 256.
- Same algorithm as native, but with precise `expf` and IEEE `rcp = 1.0f / warp_sum(sum)`.
- The final weight is `selected / selected_sum`, a true division rather than a multiply by a reciprocal.
- Host `route()` is at prefill/kernels.cu:755-763; other expert counts fall back to router_top10.

---------------------------------------------------------------------------------------------------------------

## 5. Shared expert and MoE combine

### Math

The projections use quantized mmvq; that is another part's slice.
```
h      = silu(x @ W_gate_shexp^T) * (x @ W_up_shexp^T)          [640]
y      = h @ W_down_shexp^T                                      [2560]
g      = sigmoid( x . w_gate_inp_shexp )                         scalar (bf16 weight, 2560)
shared = g * y                                                   (scale_kernel / scale_rows, BEFORE the combine)
bo[d]  = Σ_{k=0..9} w_k * expert_k[d] + shared[d]
```

### Per-path forms

| Item | Canonical | Native (`native_bf16_extra`, `native_moe_combine`) |
|---|---|---|
| SwiGLU | `swiglu_kernel` (shared_expert.cu:54): `(float)(x/(1.0+exp(-x)))*up` with x promoted to double | `native_swiglu_kernel` (62): `__fdividef(g, 1.0f+__expf(-g)) * up`, in place `gate = ...` |
| Gate logit | `scalar_gate_kernel` (109) `<<<1, 256>>>`: double `acc += bf16f(x_bf16[i]) * bf16f(w[i])`, warp_sum_d (shfl_down + broadcast), `scratch[8]` double, warp 0; `out = (float)(1/(1+exp(-acc)))` | `bf16_gemv_fp32_mmvf(x_f32, w_gate_inp_shexp, n_out = 1)` (BS 256), then `native_scalar_sigmoid_kernel` (133) `<<<1, 1>>>`: `g = __fdividef(1, 1+__expf(-g))` |
| Scale | `scale_kernel` (128): `out[i] *= g[0]` | same; `scale_rows_kernel` (162) for T tokens |
| Combine | `moe_combine_kernel` (143): double `acc = Σ_e (double)w[e]*(double)parts[e*n+col]` in expert order, `+= (double)shared[col]`, cast to float | `native_moe.cu combine` (34): `sum = parts[col]*w[0]; for e = 1..k-1: sum = fma(parts[e*n+col], w[e], sum)`; `if shared: sum += shared[col]` (plain add); k in 1..15 |

### Host paths

`shared_expert` (shared_expert.cu:209-323) uses native mmvq per projection when native data is present and
supported, and the legacy s2/q8k/q8_0 gemv otherwise. The down activation is quantized to the down weight's
contract.

`shared_expert_multi` (shared_expert.cu:169-199) handles 1..8 tokens:
1. `native_quantize_q8_1(x)`
2. mmvq for gate and up
3. `native_swiglu` in place
4. q8_1 of the result
5. mmvq down
6. Gate:
   - with `native_bf16 && STRATA_DEC_BATCH` (default on) `&& n > 1`: `mmvf_multi(n_out = 1)` +
     `native_scalar_sigmoid_multi_kernel<<<1, n>>>` (138)
   - otherwise per token: mmvf + sigmoid, or scalar_gate
7. `scale_rows`

Call sites:
- decode: `moe_shared` (layer.cpp:390-424) runs early, right after the router; `moe_combine_parts` at
  layer.cpp:425-446
- verify: `shared_expert_multi` at verify.cpp:709; `native_moe_combine_multi` at verify.cpp:790

**Prefill differs.** `moe_combine_kernel` (prefill/kernels.cu:540) applies the shared gate **inside** the combine:
```
s = 0
for k in 0..9:  s = fma(w[t,k], Dm[slot,d], s)
bo = s + shared[i] * sigm(sg[t])             // nvcc contracts this to fma(shared, sigm, s); sigm uses __expf
```
Decode rounds `y*g` first and then adds, so prefill has one fewer rounding.

---------------------------------------------------------------------------------------------------------------

## 6. RoPE, YaRN and M-RoPE

### 6.1 Rotation (NEOX, partial: 64 of the head dims)

```
for pair i in 0..31:   a = x[i], b = x[i+32]
    x'[i]    = a*c_i - b*s_i
    x'[i+32] = a*s_i + b*c_i            (rope.hpp:42-45; contracted)
x'[64..head_dim) = x[64..head_dim)       // copy, or untouched in place
```

Applied in decode by `normalize_rotate` (layer.cpp:853-854), in place:
1. rms_norm per head row: `rms_norm_weighted` (§8), or `native_qsa_rms_norm_weighted` under `native_qsa`
   (layer.cpp:851-852)
2. rope on the Q rows (24×256) and K rows (2×256)

MTP calls it at mtp.cpp:466 and verify at verify.cpp:560.

Prefill rope launches (prefill.cpp:1383/1416/1419):

| Target | Heads | Dim | ld |
|---|---|---|---|
| K | 2 | 256 | 512 |
| Q | 24 | 256 | 6144 |
| indexer q | 4 | 128 | 512 |

### 6.2 Angle

rope_scaling.hpp:65-193.
```
theta_extrap_i = pos_sec(i) * base^(-2i/n_rot)                  // base = 1e7, n_rot = 64
fs     = (type == None) ? 1 : (freq_scale_in > 0 ? freq_scale_in : 1/factor)
mscale = (ext == 0) ? attn : attn * (1 + 0.1*ln(1/fs))
corr_dim(beta) = n_rot * ln(orig_ctx / (beta*2*pi)) / (2*ln(base))        (double)
corr = [max(0, floor(corr_dim(beta_fast))), min(n_rot-1, ceil(corr_dim(beta_slow)))]
rope_yarn_ramp(lo, hi, i) = 1 - clamp((i - lo) / max(0.001, hi - lo), 0, 1)            (136-140)
rope_scaled_angle (148-162, float):
    theta = fs * theta_extrap
    if ext != 0:
        mix = ramp*ext;  theta = theta*(1-mix) + theta_extrap*mix
        mscale *= 1 + 0.1f*logf(1/fs)
    c = cosf(theta)*mscale;  s = sinf(theta)*mscale
```

`rope_scaling_invalid` is at rope_scaling.hpp:169-184.

**Shipped config:** no rope keys in the artifact. Resolution is CLI > GGUF > defaults (generate.cpp ~1750-1850),
so production runs type None, fs 1, ext 0, mscale 1, and `c = cos(theta_extrap)`, `s = sin(theta_extrap)`.

### 6.3 M-RoPE (interleaved, Qwen3-VL style)

mrope.hpp.
```
mrope_pos(tab, pos, pair) = tab ? tab[pos*3 + pair % 3] : pos       // 0 = t, 1 = h, 2 = w
```
With sections {11, 11, 10}, llama.cpp's IMROPE rule (`sector%3 == k && sector < 3*sections[k]`) reduces to exactly
`pair % 3` for all 32 pairs: pairs 30 and 31 are t and h with bounds 33 and 33, and pair 29 is w with bound 30.

Positions from mtmd:
- t = p, h = p + y, w = p + x
- text after an image continues at p + max(nx, ny)

When `--vision` is on, the device table holds `max_context + 64` cells, initialised to the identity
(generate.cpp:1934-1950). That costs one indirect load per pair even for text.

`rope_tab_cs` reads `[p*32 + pair]` and returns false when p is out of range.

The mrope.hpp comment notes that under fast-math the analytic angle error is about 0.0014 rad at 32K and about
0.02 rad at 262K, and that prefill (precise) and decode (fast) disagree.

### 6.4 Implementations

| Variant | File:line | Angles |
|---|---|---|
| canonical `rope_neox_kernel` | rope.cu:97 | Host table `build_rope_table` (rope.cu:34-75) in **double**: `inv = pow(theta, -2i/n_rot)`; `ang = p*inv` (scaled: `interp = fs*extrap`; ramp blend in double using a float `rope_yarn_ramp`); `tab = (float)(cos(ang)*ms)` and `(float)(sin(ang)*ms)`. Layout [max_cells][32] cos and sin, built at layer.cpp:658-670 and registered by `rope_table_set` when n_rot == 64. One thread per row, 128 threads per block; `toff = mrope_pos(...)*32` |
| native `apply<TAB>` | native_rope.cu:46, fast-math | Grid ((head_dim/2 + 127)/128, rows), 128 threads, one thread per pair; pairs ≥ 32 copy if out-of-place. TAB=false (default): `theta_extrap = mrope_pos(...) * powf(theta_scale, pair)` with host `theta_scale = powf((float)1e7, -2.0f/64)`, then `rope_scaled_angle` (fast `__powf`, `__cosf`, `__sinf`). TAB=true only with `STRATA_ROPE_TABLE=1` and matching scaling (native_rope.cu:89-95): reads the canonical double table. Host checks: rows ≤ 65535, head_dim 128 or 256, n_rot 64, valid scaling (113-145) |
| prefill `rope_kernel<TAB>` | prefill/kernels.cu:566, not fast-math | One block per (t, head), 32 threads (one per pair). Precise powf, cosf and sinf through `rope_scaled_angle`; theta_scale from host powf (prefill/kernels.cu:820) |

---------------------------------------------------------------------------------------------------------------

## 7. Control vectors (cvec)

cvec.hpp:25-37 and cvec.cu:46-164. The loader is at generate.cpp:875-945.

Loader:
- Tensors are `direction.<l>`; `d = Σ scale * src`.
- Mode 0 (project): `s_l = ||d||` (double), `dir = d / ||d||`.
- Mode 1 (add, the default; generate.cpp:413): `s_l = 1`, `dir = d`.
- `covers(l) = loaded && first ≤ l ≤ last && steered[l]`.
- `cvec_set_enabled` (cvec.cu:137-150) flips `*on` on every device.

Kernel:
```
cvec_kernel <<<(hc, T), 256>>>, MAXK 16 registers per thread (2560/256 = 10 used)          cvec.cu:46
  r = R + t*r_ld + c*n;  s = s_l[layer];  steer = (*on != 0 && s != 0)
  if !steer && !write: return
  w = write ? 2*sigmoidf_(inj[t*inj_ld + c] / (float)hc) : 0          // sigmoidf_ = 1/(1+__expf(-x)), IEEE division
  for k: d = tid + 256k;  xv = r[d];  if write: xv = fma(bo[t*bo_ld + d], w, xv)
         x[k] = xv;  if steer && mode == 0: dot = fma(xv, dir[d], dot)
  mode 0: xor warp sum -> part[8] -> threads < 32: xor sum of (t < 8 ? part[t] : 0) -> dot = part[0] * s
  out: r[d] = steer ? (mode == 0 ? fma(-dot, dir[d], xv) : xv + dir[d]) : xv
```

Mode 0 computes `R_c <- R_c - s * <R_c, dir> * dir` per stream. With s = 1 that removes the component; with s = 2
it reflects.

`write = true` performs the fused path's pending FFN write and is bitwise equal to it. That is why a loaded vector
that is switched off reproduces the stock engine (cvec_parity.cpp:179).

---------------------------------------------------------------------------------------------------------------

## 8. Elementwise kernels (src/kernels/cuda/elementwise.cu)

- `rms_norm_weighted_kernel` (88): 4 warps per block, one warp per row, with the row guard.
  ```
  acc = Σ_{c = lane; c += 32} r^2       (serial per lane)
  shfl_down tree
  lane 0: inv = rsqrtf(acc/cols + eps);  shfl broadcast
  r = (w ? r*w : r) * inv               // product order (r*w)*inv
  ```
  The row guard fixed the QSA k-norm overrunning into vcur (elementwise.cu:97-102). Used for QSA q/k norms
  (24×256, 2×256) and indexer norms (4×128) when `native_qsa` is off.
- `embedding_gather_kernel` (19):
  ```
  code = (codes[i/per_byte] >> ((i%per_byte)*bits)) & mask
  out  = __fadd_rn(__fmul_rn((float)(code + bias), scales[i/group_elems]), offsets ? offsets[group] : 0)
  ```
  Explicitly no FMA. Callers are layer.cpp:1055 and verify's `embedding_gather_dev`.
- `silu_kernel` (72) runs in double. `gdn_gate_kernel` (40) computes `softplus(alpha + dt[h]) * ssm_a[h]`, with
  `softplus = x > 20 ? x : log1pf(expf(x))`; used only when `native_gdn` is off.

---------------------------------------------------------------------------------------------------------------

## 9. Kernel catalogue (every `__global__` in the slice)

Format: purpose · I/O · launch / smem · CUDA features · order and determinism · parity test and tolerance · Metal note.

None of these kernels use atomics. Every reduction has a fixed order, so all of them are run-to-run deterministic
for a fixed launch configuration.

### 9.1 src/kernels/cuda/gr.cu (canonical; library default when the native flags are off)

**K1 `gr_norm_kernel<FP32_ACT>`** (gr.cu:136)
- Purpose: per-stream RMS norm times gamma.
- I/O: R f32 [4][2560], w_norm f32 [10240] → xn f32 [10240], xq bf16 [10240] (BF16 contract only).
- Launch: grid hc = 4, 256 threads, smem `double scratch[8]`.
- Features: `__shfl_down_sync`, `__shfl_sync` broadcast, `__syncthreads`, `rsqrtf`.
- Order: per-thread stride 256; shfl_down tree; warp 0 re-reduce.
- Parity: gr_parity.cpp. Real dims 2560/4/320 rel ≤ 1e-4 vs the CPU reference for each contract (754); scalar
  contract within 2e-7 (233); BF16 vs FP32 contract floor 1e-4 (514).
- Metal: one threadgroup per stream, 256 threads; `simd_shuffle_down` + broadcast; `rsqrt` (precise vs fast
  unspecified, so not bitwise with CUDA). ggml-metal: `kernel_rms_norm` (+ mul).

**K2 `gr_down_kernel<Act>`** (gr.cu:165)
- Purpose: down projection + SiLU(·/hc).
- I/O: xq bf16 or xn f32 [10240], w_down bf16 [320][10240] → lq bf16 [320] or lo f32 [320].
- Launch: grid 320, 256 threads, smem `part[8]`.
- Features: shfl_down, syncthreads.
- Order: thread i-strided 256 serial FMA chain; warp tree; part[0..7] tree.
- Parity: gr_parity (as K1); contract test notes "reduction order differs ... C1 is the gate".
- Metal: a simdgroup-per-row gemv; ggml-metal `kernel_mul_mv_bf16_f32` (different order).

**K3 `gr_gate_kernel<Act>`** (gr.cu:234)
- Purpose: up projection + sigmoid + gating.
- I/O: lq or lo [320], w_up bf16 [10240][320], xn → gated f32 [10240].
- Launch: grid ceil(10240/8) = 1280, 256 threads (one warp per row).
- Features: shfl_down.
- Order: lane-strided k; warp tree.
- Parity: gr_parity.
- Metal: simdgroup per row; `simd_shuffle_down` loop.

**K4 `gr_mean_kernel`** (gr.cu:251)
- Purpose: stream mean.
- I/O: gated [10240] → mixed [2560].
- Launch: grid 10, 256 threads.
- Order: c ascending, then `/hc`.
- Parity: gr_parity.
- Metal: trivial elementwise.

**K5 `gr_inject_kernel<Act>`** (gr.cu:266)
- Purpose: inject dots.
- I/O: xq or xn [10240], w_inject [4][10240] → inject [4].
- Launch: `<<<1, 128>>>`, warp c handles stream c.
- Order: lane stride 32 serial; warp tree.
- Parity: gr_parity (exact scalar inject at 261).
- Metal: one threadgroup of 4 simdgroups.

**K6 `gr_write_kernel`** (gr.cu:283)
- Purpose: residual write.
- I/O: R, block_out [2560], inject [4] → R (in place allowed).
- Launch: grid 40, 256 threads, dynamic smem hc floats.
- Features: precise expf; contracted fma.
- Parity: gr_parity: zero inject plain add < 1e-6 (647); every stream adds the same block output < 1e-6 (691);
  graph replay memcmp (397).
- Metal: elementwise; use explicit `fma`.

### 9.2 src/kernels/cuda/fused_gr.cu (decode, verify and MTP default under `--native`)

**K7 `gr_down_kernel`** (fused_gr.cu:49)
- Purpose: pending write fold + norm + down SiLU + inject, for one token.
- I/O: FusedGrArgs (fused_gr.hpp:24). R f32 [10240], bo_prev [2560], inj_prev [4], w_norm, w_down, w_inject →
  lo f32 [320], inject_out [4], rs [4] (block 0).
- Launch: grid 41, 256 threads, **static smem 40 KB xn + 128 B part + 16 B s_rs**.
- Features: float4 loads, `__ldg` uint4, xor shuffles, `__expf`, `rsqrtf`, `__launch_bounds__(256)`.
- Order: as in §3.5. Each block recomputes the norm redundantly (41×); all blocks get bitwise-identical rs.
- Parity: gr_parity.cpp:351-355 (multi ≡ single memcmp); fused_gr_selftest (bitwise, every variant); cvec_parity
  (179) uses it as the oracle. **No direct comparison against an independent reference was found in the clone for
  the single-token fused read;** its accuracy is gated end to end.
- Metal: **40 KB exceeds the 32 KB threadgroup limit.** Have a norm kernel write xn to device memory (as
  `gr_norm_multi`/`gr_norm_split` already do; bitwise identical) and let the down kernel read xn from device
  memory (it is cache-resident). Keep the `lane + 32j` chunk order, the dot8 inner FMA order and the plain
  `acc += dot8`. `simd_shuffle_xor` for warp_sum.

**K8 `gr_up_kernel`** (fused_gr.cu:107)
- Purpose: up projection + gating + mixer + write-back of R' (columns owned per block).
- I/O: lo [320], w_up [10240][320], R, bo_prev, inj_prev, w_norm, rs → R_out (if apply), mixed [2560].
- Launch: grid 80, 256 threads, smem 1.28 KB + 512 B.
- Features: `__ldg`, xor shuffles, `__expf`.
- Order: per row, dot8 of chunk lane plus (lane < 8) dot8 of chunk 32+lane, then xor warp sum; mixer
  c-ascending, `/4`.
- Parity: as K7.
- Metal: fits; port directly.

**K9 `gr_norm_multi_kernel`** (fused_gr.cu:148)
- Purpose: norm (+ fold) for T tokens.
- I/O: GrMulti{a[8], xn, T} → m.xn [T][10240] global, a.rs.
- Launch: grid T, 256 threads.
- Order: identical to K7 step 1.
- Parity: selftest and gr_parity:351-355 (memcmp).
- Metal: direct; this is the natural Metal norm for T = 1 as well.

**K10 `gr_down_multi_kernel<TILEV>`** (fused_gr.cu:200)
- Purpose: down + inject for T tokens with one weight read.
- I/O: m.xn → lo [T][320], inject [T][4].
- Launch: grid 41, 256 threads, dynamic smem T·TILEV·4 B (80 KB at T = 8 with TILEV 2560; 40 KB with 1280);
  opt-in large smem (`down_chunk`, 690-753).
- Features: `__ldg` uint4 prefetch, register weight cache.
- Order: per token, the same chunk order as single-token, hence bitwise.
- Parity: memcmp vs single (gr_parity:351-364); captured graph vs direct (376-383); changing graph replay (397-404).
- Metal: use TILEV = 512 or 768 (any multiple of 256 floats keeps the per-lane chunk mapping) so T·TILEV·4 ≤ 32 KB,
  or read xn straight from device memory. Not a `simdgroup_matrix` candidate if you want bitwise ≡ single-token.

**K11 `gr_up_multi_kernel`** (fused_gr.cu:258)
- Purpose: up + mixer + R' write for T tokens.
- Launch: grid 160, 256 threads, smem lo[8][320] + g[8][4][16] (about 12 KB).
- Order: identical to K8 per token.
- Parity: as K10.
- Metal: fits.

**K12 `gr_down_v3_kernel<S>`** (fused_gr.cu:324)
- Purpose: one-pass down with unscaled per-stream partials. Opt-in `STRATA_GR_V3=1`.
- Launch: grid (41, 4S), 256 threads, dynamic smem; writes `part` and `ssg` scratch.
- Order: **not bitwise**.
- Parity: gr_parity 2e-6 × max|ref| (336-346); S = 2 known failing (#375).
- Metal: skip.

**K13 `gr_up_v3_kernel<S>`** (fused_gr.cu:408)
- Purpose: finishes rs, lo = silu(Σ_c fma(rs_c, p_c, ·)/4), then up as in K11.
- Launch: grid 160.
- Parity and Metal: as K12 (skip).

**K14 `gr_norm_split_kernel`** (fused_gr.cu:522)
- Purpose: norm parallelised over (token, stream).
- Launch: grid (T, 4), 256 threads, smem part[8] + s_rs.
- Order: the same element visits and per-stream accumulation as plain, so bitwise.
- Parity: selftest.
- Metal: good choice (4 threadgroups per token).

**K15 `gr_down_staged_kernel`** (fused_gr.cu:619)
- Purpose: double-buffered down for T tokens. Default variant on the RTX 4080S when the selftest passes.
- Launch: grid 41, 256 threads, dynamic smem 2·T·1280·4 B = 80 KB at T = 8.
- Features: `cp.async.cg` 16 B (sm_80+), commit/wait groups.
- Order: bitwise equal to plain.
- Parity: selftest.
- Metal: no cp.async; use plain loads, and use the K10 recommendation for the tile size. Threadgroup
  double-buffering at 2·T·256·4 = 16 KB is possible but probably unnecessary on unified memory.

### 9.3 src/kernels/cuda/native_gr_norm.cu (fast-math)

**K16 `weighted_rms_norm<BS>`** (native_gr_norm.cu:46)
- Purpose: the llama.cpp CUDA `rms_norm` + `mul` replica.
- I/O: input R [4][2560], gamma [10240] (row-offset), output xn.
- Launch: grid n_rows = 4; block 1024 if n_cols ≥ 1024, else 256; smem `sums[32]`.
- Features: xor shuffles, fast-math division, `rsqrtf`.
- Order: stride BS; every warp re-reduces.
- Parity: gr_parity mode 2 ("native pinned GR", rel ≤ 1e-4 real dims; scalar 2e-7). bench/micro
  native_gr_norm_parity is referenced in CMakeLists.txt:311-330 but absent from the clone.
- Metal: 1024-thread threadgroup is allowed (check `maxTotalThreadsPerThreadgroup` for the pipeline). ggml-metal
  `kernel_rms_norm` uses a different reduction.

### 9.4 src/kernels/cuda/native_gr_postops.cu (fast-math)

**K17 `down_silu`** (native_gr_postops.cu:42)
- I/O: lo f32 [320] in place.
- Launch: grid 2, 256 threads.
- Math: `x = fma(0.25, lo, 0)`; `lo = x/(1+expf(-x))` (fast).
- Parity: gr_parity mode 2; native_gr_postops_parity (absent from the clone).
- Metal: ggml-metal `kernel_scale` + `kernel_silu`.

**K18 `pre_gated<Fused>`** (native_gr_postops.cu:49)
- I/O: xn [10240], gate [10240] (overwritten with products), mixed [2560].
- Launch: grid 10, 256 threads.
- Math: §3.4; `__fmul_rn`, `__fmaf_rn`, `__fadd_rn` pinned.
- Parity: gr_parity mode 2 (incl. scalar final mixer memcmp, 252).
- Metal: elementwise; explicit `fma`.

**K19 `post`** (native_gr_postops.cu:65)
- I/O: residual, block_out, inject → output (alias allowed).
- Launch: grid 40, 256 threads.
- Math: §3.4.
- Parity: gr_parity mode 2.
- Metal: elementwise. ggml-metal `kernel_sigmoid`/`kernel_mul`/`kernel_add` in sequence (rounds differently).

### 9.5 src/kernels/cuda/native_bf16.cu (fast-math; used by GR native, the router logits and the shared gate)

**K20 `bf16_f32_mmvf_kernel<BS>`** (native_bf16.cu:44)
- Purpose: y[n_out] = W_bf16[n_out][n_in] · x_f32. The llama.cpp `mul_mat_vec_f` replica.
- Launch: grid n_out, BS threads chosen by `mmvf_block_size` (121-133): fewest `ceil(n_in/(2·BS))` iterations,
  starting at 32 and trying 64..256 in steps of 32. Gives 2560 → 256, 10240 → 256, 320 → 160. smem
  `partials[32]` when BS > 32.
- Order: §2.
- Parity: gr_parity mode 2; the scalar inject is exact (261).
- Metal: one threadgroup per row with `simd_shuffle_xor`. ggml-metal `kernel_mul_mv_bf16_f32` (different order).

**K21 `bf16_f32_mmvf_multi_kernel<BS, NT>`** (native_bf16.cu:77)
- Purpose: same for 1..8 tokens (NT 4 or 8); bitwise per row equal to K20.
- Launch: native_bf16.cu:144-145.
- Metal: same pattern.

### 9.6 src/kernels/cuda/router_top10.cu (FP64; not fast-math)

**K22 `router_top10_kernel`** (router_top10.cu:69)
- I/O: logits f32 [T][512] → ids i32 [T][k], weights f32 [T][k].
- Launch: grid T, ≤ 512 threads, dynamic smem 512 + 4096 + 2048 B, static s_red/s_rid/s_sum.
- Features: **FP64 exp**, shfl_down.
- Order: serial double sum ascending; k passes of argmax with tie → lower index.
- Parity: router_top10_parity.cpp (NE 512, NT 64, seed 2024, K 10; random, all-equal ties, 12-way tie and
  dominant cases). ids exact, weights rel 1e-5 (119-146).
- Metal: **no FP64 in MSL.** Port K24 (FP32) instead. If canonical bits are ever needed, compute on the CPU.

**K23 `router_top10_fast_kernel<FORCE_SERIAL>`** (router_top10.cu:206)
- HIP only; bitwise equal to K22.
- Parity: tests/hip/router_fast.cpp:64-71 (memcmp, 65,536 rows).
- Metal: n/a. The bracket trick is a model for double-free exactness only if exp is also emulated.

### 9.7 src/kernels/cuda/native_router.cu (fast-math)

**K24 `route`** (native_router.cu:45)
- I/O: logits [T][512] → ids [T][10], weights [T][10].
- Launch: grid T, block dim3(32, 8) with only y = 0 working (`__launch_bounds__(256, 1)`), no smem.
- Features: xor shuffles, `__expf`, approximate rcp, `isnan`.
- Order: §4.2.
- Parity: no test of its own found in the clone (bench/micro native_router_parity is referenced in CMake but
  absent). It is exercised by verify-vs-decode end to end.
- Metal: launch **32 threads** (one simdgroup) per token and remove the barrier. 16 registers per lane; use
  `simd_shuffle_xor` with an explicit (value, index) tie-break. `isnan` and `-INFINITY` need
  `MTLMathModeSafe`/`Relaxed` (not Fast). ggml-metal: `kernel_soft_max_f32` + `kernel_argsort_f32_i32` +
  get_rows/sum_rows/clamp/div as a graph. A fused Metal topk-moe kernel: unsure.

### 9.8 src/kernels/cuda/shared_expert.cu (THREADS 128; not fast-math)

**K25 `swiglu_kernel`** (shared_expert.cu:54)
- I/O: gate [640], up [640] → out (in place: `gate`).
- Launch: grid 5, 128 threads (277-279).
- Features: **FP64 exp**.
- Parity: shared_expert_parity tolerance 1e-2 (measured 2.177e-7) at 301.
- Metal: FP32 only, so `native_swiglu` semantics. ggml-metal `kernel_swiglu`.

**K26 `native_swiglu_kernel`** (shared_expert.cu:62)
- `__fdividef(g, 1+__expf(-g))*up`; in place.
- Launch: 179 and 277.
- Parity: shared_expert_parity native gate < 2e-6 (375); captured byte-identical (389).
- Metal: `fast::divide`/`fast::exp` analogues (not bitwise with CUDA).

**K27 `to_f16_kernel`** (shared_expert.cu:71)
- Unused. Metal: drop.

**K28 `scalar_gate_kernel`** (shared_expert.cu:109)
- I/O: x_bf16 [2560], w_bf16 [2560] → g[1].
- Launch: `<<<1, 256>>>`, smem `double scratch[8]`.
- Features: **FP64** acc and exp.
- Parity: shared_expert_parity.
- Metal: replace with mmvf + sigmoid (FP32).

**K29 `scale_kernel`** (shared_expert.cu:128)
- `out[i] *= g[0]` over 2560.
- Metal: ggml-metal `kernel_mul` (broadcast scalar).

**K30 `native_scalar_sigmoid_kernel`** (shared_expert.cu:133)
- `<<<1, 1>>>`. Metal: fold into the gate gemv epilogue or the scale kernel.

**K31 `native_scalar_sigmoid_multi_kernel`** (shared_expert.cu:138)
- `<<<1, n_tok>>>`, thread t handles token t. Metal: same as K30.

**K32 `moe_combine_kernel`** (shared_expert.cu:143)
- I/O: parts [k][2560], weights [k], shared [2560] → y.
- Launch: grid at 335.
- Features: **FP64**.
- Parity: shared_expert_parity combine ≤ 1e-6 (466).
- Metal: use K34 (FP32).

**K33 `scale_rows_kernel`** (shared_expert.cu:162)
- `out[t][i] *= g[t]`; grid (ceil(2560/128), T) (195).
- Metal: elementwise.

### 9.9 src/kernels/cuda/native_moe.cu (fast-math)

**K34 `combine`** (native_moe.cu:34)
- I/O: parts [T][k][2560], weights [T][k], shared [T][2560] (optional) → bo [T][2560].
- Launch: grid (10, T), 256 threads (72/81).
- Order: expert order, `fma(parts, w, sum)` after a plain first product, then a plain `+ shared`.
- Parity: shared_expert_parity / native_moe_combine_parity (the latter absent from the clone).
- Metal: elementwise; explicit `fma`; k ≤ 15.

### 9.10 src/kernels/cuda/elementwise.cu (THREADS 256; HIP: `-ffp-contract=off`)

| K# / Kernel | Line | Purpose / I/O | Launch | Parity | Metal note |
|---|---|---|---|---|---|
| K35 `embedding_gather_kernel` | 19 | dequantize a packed embedding row: codes u8 (bits/code), scales and offsets f32 per group → f32 [2560] | grid ceil(n/256) | bit-exact incl. FMA check (elementwise_parity.cpp:362-395) | explicit non-FMA mul then add; Metal must avoid contraction (`MTLMathModeSafe`, or separate statements with `precise`). ggml-metal `kernel_get_rows_*` (different quant) |
| K36 `gdn_gate_kernel` | 40 | softplus gate for GDN (non-native only) | grid ceil(h_v/256) | 1e-6 (116) | `log1p` precise |
| K37 `scale_kernel` | 50 | `x *= s` | | exact (199) | `kernel_scale` |
| K38 `add_kernel` | 57 | `dst += src` (CPU/GPU parts merge; expert_source.cpp:1913) | | | `kernel_add` |
| K39 `to_f16_kernel` | 62 | f32 → f16 RNE | | | `kernel_cpy_f32_f16` |
| K40 `to_bf16_kernel` | 67 | f32 → bf16 RNE with NaN quieting (`f32_to_bf16_bulk`) | | bf16 NaN (413-465) | integer RNE in MSL (not `bfloat(x)` unless verified RNE + NaN) |
| K41 `silu_kernel` | 72 | double SiLU (layer.cpp:257) | | 1e-7 (177) | FP32 `precise::exp` |
| K42 `rms_norm_weighted_kernel` | 88 | §8 | grid ceil(rows/4), 128 threads | < 1e-6 with rivals (290/304); null w 1e-6 | `simd_shuffle_down` + broadcast |
| K43 `doorbell_ring_kernel` | 209 | `__threadfence_system` + volatile ++seq (GPU → host signal) | `<<<1, 1>>>` | tests/hip/handoff | `MTLSharedEvent` signal |
| K44 `doorbell_wait_kernel` | 214 | GPU spins on a host-written volatile flag (session.cpp:873) | `<<<1, 1>>>` | | **do not port as a spin**: Metal gives no forward-progress or visibility guarantee for CPU writes during a dispatch. Use `MTLSharedEvent encodeWait` or split command buffers |
| K45 `copy_from_mapped_kernel` | 226 | mapped pinned host → device copy (≤ 64 blocks grid-stride) | | | **droppable** on unified memory (bind the shared buffer) |
| K46 `copy_rows_from_mapped_kernel` | 235 | verify window CPU rows, zeroing the GPU-hit rows (verify.cpp:782) | grid rows, 128 threads | | keep only the zeroing logic or drop |
| K47 `doorbell_publish_kernel` | 276 | copy x/ids/w to mapped memory, fence, ++seq | `<<<1, 1024>>>` | | droppable; signal an event instead |
| K48 `copy_i32_from_mapped_kernel` | 297 | ids from host (verify.cpp:410-412, mtp.cpp:607+, layer.cpp:885-886) | `<<<1, 128>>>` | | droppable |

### 9.11 src/kernels/cuda/rope.cu (not fast-math)

**K49 `rope_neox_kernel`** (rope.cu:97)
- I/O: x f32 [rows][head_dim], pos i32 [rows], cos/sin tables [cells][32] → out (in place via `rope_neox_apply`,
  114-131).
- Launch: one thread per row, 128 threads per block (126); row guard.
- Order: elementwise.
- Parity: rope_parity.cpp. Table bit-exact vs float64 (76-77); rotation 1e-6 (146); pairing (153-156);
  linear/yarn tables bit-exact (190-288).
- Metal: the **recommended Metal default**. Build the double table on the CPU; it is bitwise identical to CUDA's
  canonical. ggml-metal `kernel_rope_neox` computes angles on the GPU in float (different).

### 9.12 src/kernels/cuda/native_rope.cu (fast-math)

**K50 `apply<TAB>`** (native_rope.cu:46)
- Launch and math: §6.4.
- Parity: rope_parity native vs table 3e-3 row-relative for yarn factor 2 and none (348-433).
- Metal: analytic angles with `fast::cos` on |θ| up to 2.6e5 rad lose accuracy, as `__cosf` does. Prefer the
  table (K49 semantics). If analytic, use `precise::cos`/`precise::sin` and compute `theta_scale^pair` on the host
  as a [32] table. ggml-metal `kernel_rope_multi` handles mrope; imrope support needs verification.

### 9.13 src/kernels/cuda/cvec.cu (not fast-math)

**K51 `cvec_kernel`** (cvec.cu:46)
- I/O: R [T][4][2560], dir [L][2560], s_l [L], on [1], bo [T][2560], inj [T][4] → R.
- Launch: grid (hc, T), 256 threads, smem part[8].
- Features: `__expf`, xor shuffles.
- Order: per-thread fma chain over k, xor warp, 8-partial xor.
- Parity: cvec_parity.cpp. Project < 1e-4 (111); reflection < 1e-3 (112); s = 1 < 1e-3 (130); untouched layer
  memcmp (137); off/no-write unchanged (146); off with pending write bitwise equal to the fused fold (179); on
  with write < 1e-4 (198); add bitwise (218).
- Metal: direct port; must share the fused fold's sigmoid expression bit for bit.

### 9.14 Prefill counterparts (src/prefill/kernels.cu, not fast-math)

| Kernel | Line | Notes |
|---|---|---|
| `gr_norm_kernel` | 65 | block per (t, c); `(r*rs)*w`; xn + bf16 + bf_lo |
| `gr_norm_rs_kernel` | 84 | rs + bf16 image only |
| `gr_mix_r_kernel` | 102 | recomputes x bitwise; `fma(x, sigm, s)`; `/HC`; bf16/f16 images |
| `gr_write_norm_rs_kernel` | 128 | `__launch_bounds__(256)`; write + next half's norm, GRW_PER = 10 registers |
| `gr_silu_kernel` | 160 | |
| `gr_mix_kernel` | 170 | |
| `gr_write_kernel` | 191 | |
| `gr_broadcast_kernel` | 198 | |
| `route_kernel<REG>` | 454 | §4.3 |
| `moe_combine_kernel` | 540 | §5 |
| `rms_rows_kernel` | 553 | QSA norms: `s*r*w`, block_sum |
| `rope_kernel<TAB>` | 566 | §6.4 |

---------------------------------------------------------------------------------------------------------------

## 10. Per-file summary

| File | LOC | Purpose | Kernels | CUDA features | Parity test | Metal note |
|---|---|---|---|---|---|---|
| src/kernels/cuda/gr.cu | 432 | canonical GR read/write + host dispatcher | 6 | shfl_down + broadcast, smem, precise expf | gr_parity.cpp (770) | port the dispatcher; kernels only for the BF16 reference contract |
| src/kernels/cuda/fused_gr.cu | 1081 | fused 2-kernel GR read, multi-token, split/staged/v3, selftest | 9 | 40-80 KB smem, uint4 `__ldg`, cp.async, `__expf`, xor | gr_parity:351-404, fused_gr_selftest | **restructure for 32 KB threadgroup memory**; keep the chunk order; selftest portable |
| src/kernels/cuda/native_gr_norm.cu | 102 | pinned ggml rms_norm | 1 | 1024-thread blocks, xor | gr_parity mode 2 | direct |
| src/kernels/cuda/native_gr_postops.cu | 115 | pinned ggml scale/silu/sigmoid/mul/add | 3 | `_rn` intrinsics, fast-math | gr_parity mode 2 | elementwise |
| src/kernels/cuda/native_bf16.cu | 186 | pinned ggml mul_mat_vec_f (bf16 × f32) | 2 | xor, fast-math | gr_parity mode 2 | simdgroup gemv |
| src/kernels/cuda/router_top10.cu | 401 | canonical double router + HIP fast | 2 | FP64, shfl_down | router_top10_parity.cpp (167), tests/hip/router_fast.cpp (101) | FP64 unavailable; use native |
| src/kernels/cuda/native_router.cu | 127 | pinned ggml topk-moe | 1 | xor, `__expf`, early return before barrier | none in the clone | one simdgroup per token |
| src/kernels/cuda/shared_expert.cu | 346 | shared expert host + gate/swiglu/combine | 9 | FP64 (canonical), `__fdividef`/`__expf` | shared_expert_parity.cpp (482) | FP32 native forms only |
| src/kernels/cuda/native_moe.cu | 86 | pinned combine | 1 | fma | shared_expert_parity | elementwise |
| src/kernels/cuda/elementwise.cu | 325 | gather, small ops, qk rms norm, doorbells, mapped copies | 14 | volatile, `__threadfence_system`, spin | elementwise_parity.cpp (475) | doorbells/mapped become events or are dropped |
| src/kernels/cuda/rope.cu | 136 | double-table NEOX rope | 1 | none special | rope_parity.cpp (442) | recommended |
| src/kernels/cuda/native_rope.cu | 147 | analytic fast rope (+ optional table) | 1 | fast powf/cosf/sinf, atomic host ptr table | rope_parity 3e-3 | prefer the table |
| src/kernels/cuda/cvec.cu | 166 | control vectors + fused-write fold | 1 | `__expf`, xor | cvec_parity.cpp (223) | direct |
| src/prefill/kernels.cu (slice part) | 842 total | prefill GR/route/combine/rope/rms | 12 of ~38 | `__expf`, precise libm | prefill tests (other part) | BF16-activation GEMMs |

The headers are in include/strata/kernels/. LOC:

| Header | LOC | Header | LOC |
|---|---|---|---|
| gr.hpp | 122 | fused_gr.hpp | 65 |
| router_top10.hpp | 26 | native_router.hpp | 19 |
| shared_expert.hpp | 101 | native_moe.hpp | 21 |
| elementwise.hpp | 120 | rope.hpp | 64 |
| native_rope.hpp | 20 | cvec.hpp | 61 |
| rope_scaling.hpp | 193 | mrope.hpp | 61 |
| bf16_bits.hpp | 61 | f16_bits.hpp | 104 |
| native_gr_norm.hpp | 18 | native_gr_postops.hpp | 28 |

---------------------------------------------------------------------------------------------------------------

## 11. Canonical vs native/fused, and what actually runs

### Flags

`--native` (src/program/generate.cpp:1519-1523) sets: stream_token, gr_native_mmvf, native_bf16,
native_bf16_extra, native_ple_key, native_moe_combine, native_gdn, native_router, native_qsa, native_qsa_indexer,
native_rope and native_ple_postops.

`gr_fp32_activations` stays false; native mmvf implies FP32 by itself (gr_parity.cpp:161-165).

Enablement (generate.cpp:1915-1950):
- `layer_set_fused_gr(gr_native_mmvf && !no_fused_gr && !gpu_stages && dump_layers.empty() && dump_halves.empty() && !stage_timing)`
- `shared_expert_set_native_bf16(native_bf16_extra)`
- Library defaults are all false, except layer.cpp:42 (`g_shared_early = true`, `g_fused_gr = false`,
  `g_publish_kernel = true`).

### What runs on each path

| Path | GR read | GR write | Router | Shared / combine | RoPE |
|---|---|---|---|---|---|
| Decode `--native` | fused_gr_read (K7 + K8) | folded; `post` (K19) only at PLE layer 1 and the last layer; K51 when steered | mmvf (K20) + `route` (K24) | mmvq + mmvf gate + K30 + K29; K34 | K50 analytic (fast) |
| Head (all decode) | native six-kernel gr_read with `pre_gated<false>` (or canonical without `--native`) | n/a | n/a | n/a | n/a |
| Verify window (1..8) | fused_gr_read_multi (staged/split/plain per selftest; always fused) | per-token gr_write at the last layer; cvec | mmvf_multi + route multi (if dec_batch) | shared_expert_multi; K34 multi | K50 |
| MTP | fused_gr_read_multi (mtp.cpp:461/522), gr_write (mtp.cpp:550) | | | | K50 (mtp.cpp:466) |
| Prefill | BF16 GEMMs + prefill GR kernels | gr_write_norm_rs | route_kernel (precise) | combine with gate inside | rope_kernel (precise) |
| Library default (no flags) | canonical gr.cu, BF16 activations | gr_write_kernel | router_top10 (double) | scalar_gate (double), swiglu double, moe_combine double | rope_neox (double table) |

### Environment knobs

| Knob | Effect |
|---|---|
| `STRATA_HC_SPLIT` | 0 = plain, 1 = split, unset or 2 = staged |
| `STRATA_GR_V3=1` | the non-bitwise v3 multi read |
| `STRATA_ROPE_TABLE=1` | native rope reads the double table |
| `STRATA_DEC_BATCH` | default on; batched shared gate / router in verify |
| `STRATA_PREFILL_BF16X2` | 0, 1 or 2; see §3.7 |
| `STRATA_GR_UNFUSED=1` | prefill norm + mix unfused |
| `STRATA_HIP_ROUTER_OLD=1` | forces the old HIP router kernel |

---------------------------------------------------------------------------------------------------------------

## 12. Numerical subtleties that matter for exact parity

1. **The `/hc` is inside the SiLU** (gr.hpp:88). The mixer mean is `/hc` after summing streams; the write gate is
   `2*sigmoid(inject/hc)`. Since hc = 4, every `/4` and `*0.25` is exact (except FTZ on denormals under fast-math).
2. **w_norm is stored as (1 + w)**, so do not add 1 again.
3. **Norm product order.** Fused computes `(R*g)*rs`; canonical, native and prefill compute `(R*rs)*g`. gr_up
   recomputes `(rv*w_norm)*rs` and must reproduce gr_down's xn bit for bit. Keep that if you fuse.
4. **dot8 structure.** Each 8-element chunk restarts from 0 with eight fmas (lo half first), then `acc += chunk`
   with a plain add. Chunk index `j = lane + 32q` ascending, then the xor butterfly. Multi, split and staged rely on
   this exact mapping; tiling must not change which lane owns which chunk.
5. **The `sq` chain** in the fused norm is `r.x*r.x + r.y*r.y + r.z*r.z + r.w*r.w`, contracted by nvcc into
   `fma(w,w, fma(z,z, fma(y,y, x*x)))`. Write it explicitly on Metal.
6. **rs computation** differs per flavour: serial 8-warp sum vs trees; IEEE vs fast `/N`. `rsqrtf` (MUFU) is not
   correctly rounded, and Metal's `rsqrt` is a different approximation. Decode ≡ verify bitwise on CUDA only
   because both use the same fused kernels.
7. **Three sigmoid flavours** for the write (§3.8). cvec(write = true) must use the fused flavour (`__expf` + IEEE
   division) so that "vector off" ≡ stock.
8. **Router.** Canonical computes the exponential in double and rounds `p = (float)(s_ex*inv)`. Native is
   fast-math FP32 with `NaN → -FLT_MAX`, and `values[e] = -INF` marks taken experts. The tie rule is always "larger
   value, then lower expert id". The clamp `2^-14` applies to the **sum** of the selected weights. Prefill divides;
   native multiplies by an approximate reciprocal.
9. **Shared gate placement.** Decode rounds `y*g` and then adds it in the combine. Prefill computes
   `fma(shared, sigm, s)`, one rounding fewer.
10. **Combine first term** is a plain product (`parts*w0`), then fmas in expert order; shared is a plain add.
11. **Rope.** Table angles come from double `pow` and `cos`, rounded to float. Native angles are
    `pos * powf(theta_scale, pair)` with fast intrinsics. The documented error is 1.4e-3 rad at 32K and 2e-2 rad
    at 262K. The parity tolerance is 3e-3 row-relative. Decode and prefill already differ on CUDA.
12. **M-RoPE** `pair % 3` mapping is exact for sections {11, 11, 10}. Text-only positions are the identity.
13. **Embedding gather** forbids FMA (`__fmul_rn` then `__fadd_rn`), as does `pre_gated`'s product (`__fmul_rn`).
    `scale_zero_bias` keeps the `+0` bias, which preserves signed-zero behaviour.
14. **bf16 rounding** is RNE with NaN quieting `|64`. Prefill's `bf()` lacks the NaN branch.
15. **Determinism.** No atomics anywhere in the slice. The fused selftest re-verifies bitwise multi ≡ single per
    card at verify start (verify.cpp:198) and silently downgrades staged → split → plain.

---------------------------------------------------------------------------------------------------------------

## 13. Metal porting recommendations (this slice)

1. **Target the native/fused FP32 contract**, structured as on CUDA: fused read with folded writes, decode ≡
   verify ≡ MTP bitwise *on Metal*. Accept tolerance against CUDA outputs. CUDA's `__expf`, `rsqrtf`, approximate
   rcp/div and `__cosf` have no bit-exact MSL equivalents. Reuse gr_parity's rel ≤ 1e-4 real-dims fixture,
   router_top10_parity's ids-exact / 1e-5 fixture, and rope's 3e-3 fixture as Metal acceptance tests against CPU
   references.
2. **No FP64 in MSL.** Affected: router_top10 (K22), moe_combine (K32), scalar_gate (K28), swiglu_kernel (K25),
   silu_kernel. The rope table (K49) is built on the host anyway. Use the native/FP32 forms. For a "golden"
   reference, run the double kernels on the CPU.
3. **Threadgroup memory is 32 KB.**
   - The fused gr_down's static `xn[10240]` (40 KB) does not fit, and neither do multi/staged at T = 8 (40-80 KB).
   - Use `gr_norm_split`/`gr_norm_multi` semantics to write xn to a device buffer (bitwise identical). Have the down
     kernel read xn from device memory (40 KB per token is cache resident), or stage tiles of 256·m floats
     (m·T·1 KB ≤ 32 KB).
   - gr_up_multi (about 12 KB) fits.
4. **Reductions.**
   - Apple simdgroups are 32 wide, so CUDA warp code maps 1:1.
   - Use explicit `simd_shuffle_xor` and `simd_shuffle_down` loops with the CUDA offsets (16..1) to keep the order.
     Avoid `simd_sum`, whose order is unspecified.
   - `simd_max` is fine for fmaxf maxima.
   - For argmax with ties, shuffle (value, index) pairs.
   - `threadgroup_barrier(mem_flags::mem_threadgroup)` replaces `__syncthreads`.
5. **Native router:** dispatch exactly 32 threads per token. The CUDA kernel's 7 idle warps return before a
   barrier, which is undefined on Metal.
6. **Math mode.**
   - Compile these kernels with `MTLMathModeSafe` (or `Relaxed`), not the `Fast` default. The router relies on
     `isnan` and ±INF sentinels, the gather/pre_gated rely on no contraction, and reassociation would break the
     decode ≡ verify bitwise property.
   - Write every intended FMA as `fma()` and keep every unintended one as separate rounded ops. Make the rounding
     explicit; do not rely on a contraction mode.
   - Choose `precise::` vs `fast::` exp/cos/sin deliberately and identically across decode, verify and MTP.
7. **BF16:** load as `ushort` and expand with `as_type<float>(uint(h) << 16)`. Round with the integer RNE +
   NaN-quiet routine; do not trust `bfloat(x)` conversion to match without a test.
8. **RoPE:** use the host double table (K49 semantics, [cells][32] cos/sin). It is both the most accurate option
   and bit-identical to CUDA's canonical/TAB path. With vision, keep the [cells][3] position table.
9. **Doorbells and mapped copies** (K43-K48): on unified memory, drop `copy_*_from_mapped` and publish. Replace
   ring and wait with `MTLSharedEvent` (signal, `encodeWait`, CPU `notifyListener` or polling `signaledValue`).
   Never spin on a CPU-written flag inside a dispatch.
10. **simdgroup_matrix:** none of this slice's decode GEMVs benefit from it (M = 1..8). The prefill BF16 GEMMs
    (another slice) are the candidates.
11. **ggml-metal reference kernels**, for reading and not for bit-parity:
    - `kernel_rms_norm` (with fused mul)
    - `kernel_mul_mv_bf16_f32`
    - `kernel_soft_max_f32`, `kernel_argsort_f32_i32` (fused topk-moe on Metal: unsure)
    - `kernel_rope_neox` with YaRN args; `kernel_rope_multi` (imrope: verify)
    - `kernel_swiglu`, `kernel_silu`, `kernel_sigmoid`, `kernel_scale`, `kernel_add`, `kernel_mul`
    - `kernel_cpy_f32_f16` / `_bf16`, `kernel_get_rows`

    There is no ggml equivalent for the hyper-connection read/write or for cvec's fused write.
