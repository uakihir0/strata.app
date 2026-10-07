# Build and toolchain experiments

Small programs and CMake projects that verified the build design in
[08-build-and-distribution.md](../../08-build-and-distribution.md). Results are in
[appendix-a-measurements.md](../../appendix-a-measurements.md), section A.8. Run on the reference Mac (M5 Pro
64 GB, macOS 27.0 build 26A428, Xcode 27.0 27A266a, Metal toolchain `com.apple.dt.toolchain.Metal.32023.921.5`).

| Path | What it shows |
| --- | --- |
| `incremental/` | The metallib build of doc 08 section 5 with one kernel: the depfile, the stamp with byproducts, the generated hash header written with `copy_if_different`, `OBJECT_DEPENDS`, and the `metal.sh` wrapper that removes absolute paths |
| `labels/` | The deferred ctest labeller of doc 08 section 6.2: every test without a label gets `host` |
| `oracle/` | ggml's CPU backend built from the pinned llama.cpp as a test oracle with Metal, BLAS, Accelerate, OpenMP and native flags off (doc 08 section 6.1) |
| `pipelines.mm` | Library load time, pipeline compile time, and reuse through an `MTLBinaryArchive`, including an invalid archive (doc 08 section 10.1) |
| `capture.mm` | Programmatic GPU capture when the program sets `MTL_CAPTURE_ENABLED=1` itself (doc 08 section 10.2) |
| `vision-repro.sh` | Two release builds of `strata-vision` in different folders, compared before and after `strip` (doc 08 section 9) |
| `mac-port/` | The verified Mac skeleton of doc 08: `apply.sh` turns an upstream checkout at `tree/UPSTREAM_SYNC` into it with `root-edits.patch` (the root edits A-G, the `direct_file.cpp` patch and the `.gitignore` / `.gitattributes` lines, doc 08 section 3) and the new files in `tree/` |
| `mac-port/apply_root_edits.py` | The root edits A-G as a script that finds its anchors by content, for an upstream commit where the patch no longer applies (doc 08 section 3.1) |
| `release/detzip.py` | The deterministic zip of doc 08 section 11.3 |
| `pc-unchanged/compare_build_graphs.py` | The check that the project's PC build equals upstream's (doc 08 section 3.4) |

## Prerequisites

Xcode 27 selected (`xcode-select -p` points into `Xcode.app`), the Metal toolchain installed
(`xcodebuild -downloadComponent MetalToolchain`), CMake 3.24 or newer and Ninja. The pinned `cmake==4.4.3` and
`ninja==1.13.2` from the upstream `requirements.txt` were used. Put ninja on PATH or pass `-DCMAKE_MAKE_PROGRAM=`.

## incremental/

```bash
cmake -S incremental -B /tmp/incr -G Ninja && cmake --build /tmp/incr && /tmp/incr/strata && shasum -a 256 /tmp/incr/strata.metallib
cmake --build /tmp/incr                     # ninja: no work to do
touch incremental/k.metal; cmake --build /tmp/incr    # k.air and the stamp only; no compile of version.cpp, no link
sed -i '' 's/2.0f/3.0f/' incremental/common/h.h; cmake --build /tmp/incr   # k.air, stamp, version.cpp, link
strings -a /tmp/incr/strata.metallib | grep -cE '/private/|/var/run|/Applications/|/tmp/|/Users/'   # 0
```

`strata` prints the SHA-256 that `shasum` prints. A second build folder gives a byte-identical `strata.metallib`.
Without the `metal.sh` wrapper and `-fdebug-compilation-dir=.`, the stripped metallib contains the toolchain's cryptex
mount path (`/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.<random>`)
and the build folder.

## labels/

```bash
cmake -S labels -B /tmp/lbl && ctest --test-dir /tmp/lbl --print-labels && ctest --test-dir /tmp/lbl -L host -N
```

`up1` and `up2` (added with plain `add_test`, before and after the include) are labelled `host`; `m1` keeps `metal` and
`k1` keeps `model`.

## oracle/

```bash
cmake -S oracle -B /tmp/oracle -G Ninja -DLLAMA=<llama.cpp at 3cf03257f> -DCMAKE_OSX_ARCHITECTURES=arm64 \
  -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 && cmake --build /tmp/oracle --target o && /tmp/oracle/o && otool -L /tmp/oracle/o
```

Prints `iq2_xs blck=256 size=74 vec_dot=1` and links only `libSystem` and `libc++`. Configure took 2.9 s and the
build 4.2 s wall. Linking `ggml-base` and `ggml-cpu` together makes ld warn about a duplicate library; linking
`ggml-cpu` alone loses ggml's include folder, so the test keeps both and passes `LINKER:-no_warn_duplicate_libraries`.

## pipelines.mm and capture.mm

```bash
for f in pipelines capture; do clang++ -std=c++17 -O2 -fobjc-arc $f.mm -framework Metal -framework Foundation -o /tmp/$f; done
/tmp/pipelines /tmp/incr/strata.metallib /tmp/ar.bin; /tmp/pipelines /tmp/incr/strata.metallib /tmp/ar.bin
head -c 32768 /dev/urandom > /tmp/bad.bin; /tmp/pipelines /tmp/incr/strata.metallib /tmp/bad.bin
/tmp/capture /tmp/t.gputrace nosetenv; /tmp/capture /tmp/t.gputrace   # started=0, then started=1
```

## vision-repro.sh

```bash
./vision-repro.sh <upstream checkout> <llama.cpp at 3cf03257f> <ninja> /tmp/vrepro
```

The binary is `<build>/bin/strata-vision` (upstream `tools/vision/CMakeLists.txt` sets the runtime output folder). Both
builds give the same hash: 8,180,336 B unstripped, and 7,453,408 B after `strip -S -x` in place under the final name,
with `Identifier=strata-vision` and an ad hoc signature. `otool -L` lists `Accelerate` (ggml's default Apple BLAS),
`Foundation`, `Metal`, `MetalKit`, `CoreFoundation`, `libSystem`, `libc++` and `libobjc`. The last line prints
`absolute paths: 1`: the one match is ggml's `/tmp/perf-metal-%d.gputrace` capture name, which doc 08 section 11.7
allows. A build took 11.8-17.7 s at `-j 14`.

## mac-port/

```bash
git clone https://github.com/Niko1221/Strata /tmp/fresh && git -C /tmp/fresh checkout -q "$(cat mac-port/tree/UPSTREAM_SYNC)"
mac-port/apply.sh /tmp/fresh
cd /tmp/fresh && cmake --preset mac-dev && cmake --build --preset mac-dev && ctest --preset mac-host && ctest --preset mac-metal
```

`apply.sh` prints `22 lines with # MAC: in CMakeLists.txt`. The configure fetches nothing when `STRATA_GGML_DIR` is
passed (`-DSTRATA_GGML_DIR=<llama.cpp at 3cf03257f>`) and falls back to FetchContent at the pinned commit when it is not.
The build ends with no warning under `-Werror`; `mac-host` runs 16 tests, `mac-metal` runs `metal_context_test`.
`build-dev/strata --version` prints `strata 0.1.34 metal 1`, and `build-dev/strata` with no argument compiles the two
self-test pipelines, writes the archive to `~/Library/Caches/strata/`, and passes the self-test. Remove that folder
afterwards, or set `STRATA_METAL_ARCHIVE=/tmp/strata-cache` for the run.

`python3 mac-port/apply_root_edits.py /tmp/fresh/CMakeLists.txt` writes the same `CMakeLists.txt` that the patch gives.

Release build and reproducibility, in two checkouts at different paths:

```bash
cmake --preset mac-release && cmake --build --preset mac-release
cd build && dsymutil strata -o strata.dSYM && strip -S -x strata   # in place: strip signs with the file name as Identifier
shasum -a 256 strata; dwarfdump --uuid strata; codesign -v --strict strata && codesign -dv strata
```

Both checkouts give the same `strata` before and after `strip` (and the same UUID as the dSYM), with `Identifier=strata` and
`flags=0x20002(adhoc,linker-signed)`; `otool -L` lists only system frameworks and libraries; `vtool -show-build`
shows `minos 27.0`. `strip -S -x -o strata.s strata` gives `Identifier=strata.s`, and renaming the file afterwards
does not change it.

## release/detzip.py

```bash
python3 release/detzip.py /tmp/a.zip 1790000000 strata=build/strata:755 strata.metallib=build/strata.metallib:644
touch build/strata; umask 077; python3 release/detzip.py /tmp/b.zip 1790000000 strata=build/strata:755 strata.metallib=build/strata.metallib:644
cmp /tmp/a.zip /tmp/b.zip && unzip -Z /tmp/a.zip
```

The two zips are identical: modification times, umask and the Python version (3.9.6 and 3.13.12, both with
zlib 1.2.12) do not reach the bytes. `unzip -Z` shows `-rwxr-xr-x  2.0 unx ... defN` and `-rw-r--r--`.

## pc-unchanged/compare_build_graphs.py

Configure upstream and the project with the same options, then compare:

```bash
opts="-G Ninja -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_BUILD_TESTS=ON -DSTRATA_BUILD_CONVERSATION_TESTS=ON \
      -DSTRATA_GGML_DIR=<llama.cpp at 3cf03257f> -DCMAKE_EXPORT_COMPILE_COMMANDS=ON"
cmake -S <upstream> -B /tmp/up ${=opts}; cmake -S <project> -B /tmp/local ${=opts}   # zsh; use $opts in bash
python3 pc-unchanged/compare_build_graphs.py /tmp/up /tmp/local
```

On Linux x86-64 (here a cross configure with `CMAKE_SYSTEM_NAME=Linux`), it prints
`same build: 57 compile commands, 1787 lines of build.ninja` and exits 0. On an Intel Mac configure, it exits 1 and
lists only `direct_file_async_test`, the test that root edit F adds for macOS on purpose.
