#!/bin/sh
# Builds the numerics spike (README.md) into out/: three libraries from numerics.metal, the over-limit kernel, the
# textual AIR of each library, and the host program.
set -e
cd "$(dirname "$0")"
mkdir -p out
python3 gengrid.py > grid.inc

COMMON="-std=metal4.0 -mmacosx-version-min=27.0 -Wall"
for v in safe safe_contract fast; do
  case $v in
    safe)          F="-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off" ;;  # doc 08 section 5.3
    safe_contract) F="-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise" ;;
    fast)          F="-fmetal-math-mode=fast -DSPIKE_FAST" ;;                                          # doc 08 section 5.3
  esac
  xcrun -sdk macosx metal $COMMON $F -c numerics.metal -o out/$v.air
  xcrun -sdk macosx metal $COMMON $F -S -emit-llvm numerics.metal -o out/$v.ll
  xcrun -sdk macosx metallib out/$v.air -o out/$v.metallib
done

if xcrun -sdk macosx metal $COMMON -c tg_over.metal -o out/tg_over.air 2> out/tg_over.log; then
  xcrun -sdk macosx metallib out/tg_over.air -o out/tg_over.metallib
  echo "tg_over.metal: compiled"
else
  echo "tg_over.metal: refused by the compiler"; cat out/tg_over.log
fi

clang++ -std=c++20 -O2 -ffp-contract=off -fobjc-arc numerics.mm -framework Metal -framework Foundation -o out/numerics
