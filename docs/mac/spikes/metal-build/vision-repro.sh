#!/bin/zsh
# Build strata-vision twice, in two build folders, with the release definitions of
# docs/mac/08-build-and-distribution.md section 9, and compare the binaries.
#   vision-repro.sh <upstream checkout> <llama.cpp at the pinned commit> <ninja> <scratch folder>
set -u
SRC=${1:?upstream checkout} LLAMA=${2:?llama.cpp checkout} NINJA=${3:?ninja binary} OUT=${4:?scratch folder}
mkdir -p $OUT
for d in vpa vpb; do
  B=$OUT/$d
  rm -rf $B
  F="-ffile-prefix-map=$LLAMA=third_party/llama.cpp -ffile-prefix-map=$SRC=. -ffile-prefix-map=$B=build-vision"
  cmake -G Ninja -DCMAKE_MAKE_PROGRAM=$NINJA -S $SRC/tools/vision -B $B -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_DIR=$LLAMA -DSTRATA_VISION_CUDA=OFF -DSTRATA_PORTABLE=ON -DGGML_OPENMP=OFF -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 \
    "-DCMAKE_C_FLAGS=$F" "-DCMAKE_CXX_FLAGS=$F" "-DCMAKE_OBJC_FLAGS=$F" > $OUT/$d.conf.log 2>&1 \
    || { echo "$d: configure failed, see $OUT/$d.conf.log"; exit 1; }
  /usr/bin/time -p cmake --build $B --target strata-vision -j 14 > $OUT/$d.build.log 2>&1 \
    || { echo "$d: build failed, see $OUT/$d.build.log"; exit 1; }
  tail -3 $OUT/$d.build.log
  # strip under the final name: the ad hoc signature's identifier is the file name (section 11.6)
  mkdir -p $OUT/$d-stage && cp $B/bin/strata-vision $OUT/$d-stage/strata-vision && strip -S -x $OUT/$d-stage/strata-vision
done
shasum -a 256 $OUT/vpa/bin/strata-vision $OUT/vpb/bin/strata-vision $OUT/vpa-stage/strata-vision $OUT/vpb-stage/strata-vision
otool -L $OUT/vpa-stage/strata-vision
echo "absolute paths: $(strings -a $OUT/vpa-stage/strata-vision | grep -cE '/Users/|/private/|/tmp/|/var/folders/|/opt/homebrew|/usr/local/')"
