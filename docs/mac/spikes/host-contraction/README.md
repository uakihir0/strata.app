# Host floating-point contraction

Two sources that show whether Apple clang fuses `a*b + c` in host C++ ([appendix A.12](../../appendix-a-measurements.md)).

| File | What it shows |
| --- | --- |
| `contract.cpp` | The instructions for `a*b + c` in one expression and in two statements |
| `probe.cpp` | A value that differs fused and unfused: `a = 1 + 2^-12`, `c = -(1 + 2^-11)`, `a*a + c` is `2^-24` fused and 0 unfused |

```bash
clang++ -O2 -S -o - contract.cpp | grep -E 'fmadd|fmul|fadd'
clang++ -O2 -ffp-contract=off -S -o - contract.cpp | grep -E 'fmadd|fmul|fadd'
clang++ -O2 probe.cpp -o on && ./on                       # 0x1p-24
clang++ -O2 -ffp-contract=off probe.cpp -o off && ./off   # 0x0p+0
```

Results (2026-10-03, Apple clang 21.0.0, arm64): the one-expression form is a single `fmadd` by default and `fmul` +
`fadd` with `-ffp-contract=off`; the two-statement form is never fused.
