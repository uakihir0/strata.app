# Metal numerics

A test program for the Metal arithmetic behaviours that [04-metal-kernels.md](../../04-metal-kernels.md) relies on and
that [audit A](../../audit/A-model-and-kernels.md), sections B.2.4 and F.2-F.6, left unverified. It checks:
- whether `a*b + c` is contracted to an FMA, with and without `-ffp-contract=off` and with
  `#pragma METAL fp contract(off)`;
- whether division, reciprocal, `sqrt` and `rsqrt` are correctly rounded;
- the error of `exp`, the `log1p` substitutes, softplus and sigmoid;
- `rint`, `round` and `trunc` at ties;
- float-to-half conversion, including subnormals, overflow and NaN;
- the addition order of `simd_sum`;
- lane ids and the ballot;
- the 32 KiB threadgroup limit;
- the cost of codebook lookups from each address space.

The results are in [appendix-a-measurements.md](../../appendix-a-measurements.md), section A.10. The output of the
recorded run is `results.txt`.

| File | What it is |
| --- | --- |
| `numerics.metal` | The kernels. `build.sh` compiles it three times: once with the `safe` flags of doc 08 section 5.3, once with those flags minus `-ffp-contract=off` (`safe_contract`), and once with the `fast` flags. The `fast::` kernels exist only in the fast build. |
| `tg_over.metal` | One kernel with 32,784 B of threadgroup memory, 16 B over the limit. It is a separate file so that a refusal cannot fail the main library. |
| `gengrid.py` | Writes `grid.inc`, the codebook tables: `kvalues_iq4nl` and 512 and 1024 u64 values of splitmix64 |
| `numerics.mm` | The host program. It runs every kernel of each library and compares it with the CPU, bit for bit, or with a double-precision reference. It is built with `-ffp-contract=off`, so the CPU's `a*b + c` is two roundings. |
| `build.sh` | Builds everything into `out/`: the three libraries, their textual AIR (`out/*.ll`), the over-limit library and the host program |
| `results.txt` | The output of `out/numerics` on the reference Mac, 2026-10-03 |

## Build and run

Requires macOS 27, Xcode 27 and the Metal toolchain (doc 08 section 2.1).

```bash
./build.sh            # prints "tg_over.metal: compiled": the compiler accepts it; the pipeline is refused at run time
out/numerics          # about 30 s
grep -c 'llvm.fmuladd' out/safe.ll out/safe_contract.ll      # 0 and 2 (k_mad's call and its declaration)
grep -c 'air.fast_' out/safe.ll out/fast.ll                  # fast-math functions: 0 and 21
```

The fast build has no `llvm.fmuladd`. It marks the multiply and the add `fast`, which includes `contract`, and the
backend fuses them. Under the pragma, `k_mad_pragma_off` in the fast build keeps `reassoc nnan ninf nsz arcp afn`
but loses `contract`.

`out/` is a build folder. Delete it afterwards; `build.sh` recreates it, and `grid.inc` with it.

## Reading the output

- **Contraction lines.** These count the inputs where `fma(a, b, c)` and `a*b + c` round differently: 838,389 of 2^22
  uniform inputs. They then say which of the two the GPU returned.
- **Error lines.** These compare against a CPU reference:
  - `x / y`, `sqrt` and `1.0f / x` against the CPU's correctly rounded operation;
  - `exp`, `log1p` and softplus against `double`, rounded to float.

  "max N ulp" is the largest distance in units in the last place, with the input that produced it. The
  `CPU expf(x) vs double` and `CPU log1pf(x) vs double` lines measure Apple's libm, for scale.
- **`simdgroups` line.** For each of 32,768 simdgroups of random values, it says which CPU summation order
  reproduces `simd_sum` bit for bit. The candidates are a sequential sum, a pairwise tree (`(v0+v1)+(v2+v3)...`), and
  xor butterflies in the orders o = 16, 8, 4, 2, 1 and o = 1, 2, 4, 8, 16. The ascending butterfly gives the same
  value in every lane as the pairwise tree, because each step adds the same two partial sums.
- **Codebook lines.** These give the best GPU time of six runs of 2^20 threads × 2048 lookups. The table sizes are:
  - `_16`: 16 signed bytes, the `kvalues_iq4nl` that upstream measured as 2.12× slower in CUDA `__constant__`
    memory (`s_gemv.cu`);
  - `_512`: 4 KiB, the size of `iq2xs_grid`;
  - `_1k`: 8 KiB, the size of `iq2s_grid` and `iq1s_grid_gpu`, the largest grids.

  The lookup variants are:
  - `cb`: independent xorshift indices. This loop is ALU-bound: `k_cb_none`, the same loop with no lookup, takes as
    long.
  - `ch`: four dependent chains per thread, where each index depends on the last value read. These lines measure the
    lookups themselves.

## Limits

- One machine (M5 Pro, `Apple metal version 32023.921`). Rounding and conversion follow from the language
  specification and should hold on any Apple GPU. The `simd_sum` order and the codebook timings are hardware
  properties; recheck them on an M1-M4 GPU (doc 04 measurement L-3).
- The codebook loops use a table that stays in the GPU's L1 cache. In a real GEMV, weight reads compete for the same
  caches. Doc 04 measurement L-4 repeats the comparison inside the IQ kernels.
