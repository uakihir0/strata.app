# 08. Build and distribution: compiling, testing and publishing the Mac engine

Status: design specification, 2026-10-14. The macOS CMake build, host and Metal tests, and Mac-only CI workflows are implemented. `setup.py --check-host` reports read-only host facts; the installer, full model-planning check, end-to-end inference, model acceptance runner, release asset tools, and release workflow are not implemented.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Source audits: [audit/B-engine-host.md](audit/B-engine-host.md), section 7 (build and tests on macOS), and
[audit/C-python-setup-serve.md](audit/C-python-setup-serve.md), section 9 (packaging).
Reference code: [spikes/metal-build/](spikes/metal-build/), verified on the reference Mac (M5 Pro 64 GB, macOS 27.0,
Xcode 27.0). Its `mac-port/apply.sh` turns an upstream checkout into the verified Mac build skeleton: every CMake file and
build rule in this document, with stub engine sources that build, test and run.
Every `file:line` reference is relative to the upstream repository root.

This document covers how the Mac engine is built and shipped:
- the files the port adds, and the upstream files it edits for the build;
- the toolchain;
- the macOS-only root CMake build and its Metal test targets;
- `cmake/metal/`: the Metal backend, the kernel library, the tests and the presets;
- the `setup.py` build definitions and metadata (section 8; no setup flow calls them yet), and the optional `strata-vision` helper;
- the pipeline cache and GPU capture, which the build sets up for the engine;
- the release: versioning, the asset, reproducibility, signing, symbols and the release checks;
- CI and the developer workflow.

Other documents cover the rest:
- [06-engine-host-port.md](06-engine-host-port.md): what the engine does with what this document builds.
- [07-server-setup-port.md](07-server-setup-port.md): how setup installs or compiles the engine (section 4.7), and the
  release contract it relies on (section 4.1).
- [04-metal-kernels.md](04-metal-kernels.md): the kernels that go into `strata.metallib`.
- [09-testing-and-validation.md](09-testing-and-validation.md): the test plan and the hardware acceptance runs.
- [10-implementation-plan.md](10-implementation-plan.md): the order of work and upstream compatibility reviews.

This document supersedes audit C section 9.3 where the two differ:
- the minimum is macOS 27.0, and the GPU families are `apple7` and `metal4` ([07](07-server-setup-port.md),
  section 4.7.2);
- `strata.metallib` ships next to the engine, paired with it by hash (section 5.6);
- the first release has no `.pkg`, `.dmg` or Homebrew formula.

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Principles and files

### 1.1 Principles

| # | Principle | What it means here |
| --- | --- | --- |
| P1 | macOS is the only supported build target | The root `CMakeLists.txt` rejects non-Apple hosts and always includes `cmake/metal/backend.cmake`. |
| P2 | Metal is the only product backend | No CUDA, HIP or x86 expert backend is configured or shipped. CUDA kernels and parity harnesses are kept only for offline numerical reference and provenance checks; they are outside the Mac CMake targets, CI and release assets (D-IMPL-227, D-IMPL-240). |
| P3 | One toolchain | Xcode 27 and its Metal toolchain, reached through `xcrun`. CMake and Ninja come from the venv, at the `requirements.txt` pins. Nothing comes from Homebrew, and nothing uses OpenMP. |
| P4 | Portable engine binary | The engine target is arm64, deployment target 27.0, and uses the compiler's portable CPU baseline. The standalone vision helper enables `GGML_NATIVE` for local validation and is not currently shipped. |
| P5 | Self-contained | The engine and `strata-vision` link only system frameworks and libraries: no dylib of their own, no `LC_RPATH`, and no `DYLD_*` variable. |
| P6 | Reproducible | The same commit and toolchain give the same bytes, in any folder and on any Mac (section 11.4). |
| P7 | Paired by hash | `strata` embeds the SHA-256 of the `strata.metallib` it was built with, and refuses any other library (section 5.6). |

### 1.2 Files and targets in this design

| Path | What it is | Section |
| --- | --- | --- |
| `METAL_REV` | One line holding one positive integer: the Mac release number | 11.1 |
| `UPSTREAM_SYNC` | The 40-hex upstream commit used as the current provenance and compatibility reference | 11.1; [10](10-implementation-plan.md) |
| `CMakePresets.json` | The `mac-release`, `mac-dev` and `mac-ci` configure presets, and the `mac-host`, `mac-metal` and `mac-model` test presets. Upstream has no presets file. | 7 |
| `cmake/metal/backend.cmake` | Requirement checks, options and targets of the Mac build | 4 |
| `cmake/metal/metallib.cmake` | `strata.metallib` and the two generated headers | 5 |
| `cmake/metal/metal.sh` | `xcrun metal`, with the toolchain and SDK paths mapped away | 5.3 |
| `cmake/metal/check_manifest.cmake` | The link-time check of the kernels against the manifest; writes the hash header | 5.4 |
| `cmake/metal/tests.cmake` | The ggml test oracle, the test labels and the Mac tests | 6 |
| `cmake/metal/check_engine.cmake` | The `strata_version` test | 6.3 |
| `src/metal/kernels/common/`, `safe/`, `fast/`, `manifest.txt`, `THIRD_PARTY` | The kernel sources, their manifest, and the provenance of any adapted kernel code | 5.1 |
| `src/metal/program/version.cpp` | The engine's identity strings | 4.5 |
| `src/metal/host/pipeline_cache.mm`, `src/metal/host/capture.mm` | The pipeline cache and GPU capture | 10 |
| The rest of `src/metal/**` and `include/strata/metal/` | The Mac engine | [06](06-engine-host-port.md), section 1.5 |
| `tests/metal/` | The Mac tests | 6; [09](09-testing-and-validation.md) |
| `tools/metal/make_release.py` | Builds the release asset | 11.3 |
| `tools/metal/check_asset.py` | Checks a release asset | 11.7 |
| `tools/metal/verify_release.py` | Checks a draft release on the reference Mac | 11.8 |
| `tools/metal/check_release_policy.py` | Checks the version rules | 11.1 |
| `tools/metal/run_python_tests.py` | Runs every Python test module, each in its own process | 12.1 |
| `tools/metal/check_drift.py`, `tools/metal/drift.json` | The upstream drift check | [06](06-engine-host-port.md) H12; [10](10-implementation-plan.md) |
| `tools/metal/licenses/` | Excerpts of licence texts that upstream keeps inside headers (four files) | 11.9 |
| `tools/metal/make_release.py`, `check_asset.py`, `verify_release.py` | The release tools (sections 11.3, 11.7 and 11.8) | 11 |
| `tools/test_metal_build.py` | Proposed release/build-definition tests; the file does not exist yet (B5, phase 0) | 7.2, 12.1 |
| `.github/workflows/ci.yml`, `mac-hardware.yml`, `mac-release.yml`; `.github/dependabot.yml` | macOS CI and hardware workflows, and the release workflow (written, not run) | 12 |
| `docs/mac/` | This documentation set | 1.4 |

### 1.3 Upstream files edited for the build

| File | Edit | Section |
| --- | --- | --- |
| `CMakeLists.txt` | Edits A-G: 22 lines that carry `# MAC:` | 3.1 |
| `src/platform/direct_file.cpp` | The `F_NOCACHE` branch at `:308` | 3.2 |
| `.gitattributes` | Three lines: LF for `*.metal`, the manifest and `METAL_REV` | 3.5 |
| `.gitignore` | `.DS_Store`, `*.gputrace` | 3.5 |
| `setup.py` | Read-only `--check-host`; install CLI remains fail-closed, plus the build definitions and metadata helpers of section 8 (no setup flow calls them yet) | 8 |
| `tools/vision/strata_vision.cpp` | The thread-count fix ([06](06-engine-host-port.md), H13) | 9 |

**Markers.** Every edited line is marked, so `tools/metal/check_drift.py` can count the markers per file. It fails when
a count changes and `tools/metal/drift.json` does not record the change ([10](10-implementation-plan.md)).
- CMake: every added or changed line ends with `# MAC:`. A comment line that explains a block starts with `# MAC:`.
- `.gitignore` and `.gitattributes`: one comment line that starts with `# MAC:`, above the added lines. Neither format
  allows a comment at the end of a pattern line.
- C and C++: `// MAC:` on the `#ifdef` line of the branch.
- Python: as in [07](07-server-setup-port.md), section 1.2.

### 1.4 Where the documents live

The project plan lives in `docs/mac/`, with source research in `docs/mac/audit/` and reference experiments in
`docs/mac/spikes/`. Code comments and messages cite `docs/mac/NN-...`. The upstream user documents in `docs/` change
only as [07](07-server-setup-port.md), section 11, specifies.

---

## 2. Toolchain

### 2.1 Versions

Verified on the reference Mac, 2026-10-02:

| Tool | Version | How the build finds it |
| --- | --- | --- |
| Xcode | 27.0 (27A266a) | `xcode-select -p` (`/Applications/Xcode.app/Contents/Developer`), or `DEVELOPER_DIR` |
| Apple clang | 21.0.0 (clang-2100.3.34.2) | CMake's C, C++ and Objective-C++ compiler, from Xcode |
| Linker | ld-27037.1 | Through clang |
| macOS SDK | 27.0 | `xcrun -sdk macosx --show-sdk-path` |
| Metal compiler | `Apple metal version 32023.921 (metalfe-32023.921.6)` | `xcrun -sdk macosx metal` |
| Metal toolchain component | `com.apple.dt.toolchain.Metal.32023.921.5`, build 27A266a | `xcodebuild -showComponent MetalToolchain -json` |
| `metallib`, `metal-dsymutil`, `metal-nm` | From the Metal toolchain | `xcrun -sdk macosx -f <tool>` |
| CMake | 4.4.3 | The venv: the `requirements.txt` pin. The presets need 3.24 or newer. |
| Ninja | 1.13.2 | The venv: the `requirements.txt` pin |
| Python | 3.10 or newer | The venv ([07](07-server-setup-port.md), section 12) |
| macOS | 27.0 (build 26A428) | |

**The Metal toolchain is a separate download.** Xcode 27 does not include the Metal compiler. The component is a
cryptex mounted under `/private/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.266.1.<random>/`.
The random suffix can change when the component is updated or remounted, so nothing may store that path
(section 5.3). The Command Line Tools alone have neither Xcode's SDK layout nor the Metal compiler; the build needs
full Xcode.

### 2.2 Installing it

On a developer Mac or a self-hosted runner:

1. Install Xcode 27 from the App Store, or as `Xcode_27.0.xip` from developer.apple.com.
2. Open it once, so it installs its components and accepts the licence. Or run
   `sudo xcodebuild -license accept` and `sudo xcodebuild -runFirstLaunch`.
3. `xcodebuild -downloadComponent MetalToolchain`.
4. `sudo xcode-select -s /Applications/Xcode.app`, if `xcode-select -p` points at the Command Line Tools.
5. CMake and Ninja: the setup entry point is currently fail-closed and does not create `.venv`. For development, run
   `python3 -m venv .venv && .venv/bin/pip install -r requirements.txt`.

`xcodebuild -exportComponent` and `-importComponent` exist and would let CI cache the component. Neither is verified
(measurement B-3).

### 2.3 Not used

- Homebrew, MacPorts and their compilers. A Homebrew `cmake` or `ninja` on PATH is never picked by setup (section 8.5).
- OpenMP. Apple clang has no `libomp`. ggml's OpenMP is turned off explicitly wherever ggml is built (sections 6.1
  and 9).
- Accelerate and BLAS in the Metal engine and its test oracle. The engine does its math in Metal, and the oracle
  must not depend on the Mac that runs it (section 6.1). The optional local `tools/vision` helper disables BLAS and
  Accelerate too; its current settings are listed in section 9.
- `-mcpu=native` and `GGML_NATIVE=ON` for the distributable engine. The engine is intended to use arm64's portable
  baseline. The standalone `tools/vision` helper currently enables `GGML_NATIVE` for local image-path validation; it
  is not part of the engine or a release asset. `STRATA_PORTABLE` belongs to the superseded upstream build path.

---

## 3. Historical upstream build integration (superseded by D-IMPL-227)

The original port kept upstream's Windows and Linux build graph while adding a conditional Metal backend. The user later
selected a macOS-only repository. The current root build rejects non-macOS configurations, always includes the Metal
backend, and no longer contains the CUDA/HIP target graph. Sections 3.1–3.4 below document the superseded integration
for provenance; they are not current build instructions or requirements.

### 3.1 Edits A-G to `CMakeLists.txt`

The line numbers are those of the upstream file at `1678de33`. Upstream lines are never re-indented, so later compatibility reviews can compare and selectively port
changes around the edits. The file keeps its UTF-8 BOM and LF line endings.

**A: the option**, after `:36`:
```cmake
option(STRATA_ENABLE_METAL "Build the Metal engine for Apple silicon, macOS 27+ (docs/mac/08)" OFF)  # MAC:
```

**B: keep CUDA off macOS and enable one backend at a time**, after `:48`:
```cmake
if(STRATA_ENABLE_CUDA AND CMAKE_SYSTEM_NAME STREQUAL "Darwin")  # MAC:
  message(FATAL_ERROR "The CUDA backend is not supported on macOS; use STRATA_ENABLE_METAL=ON on Apple silicon")
endif()  # MAC:
if(STRATA_ENABLE_METAL AND (STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP))  # MAC:
  message(FATAL_ERROR "STRATA_ENABLE_METAL is a separate backend; turn STRATA_ENABLE_CUDA and STRATA_ENABLE_HIP off")
endif()  # MAC:
```

**C: the x86-only CPU kernels.** The `strata_kernels_cpu` block (`:778-793`) compiles AVX2 and AVX-512 sources, which
fail on arm64 (audit B, section 7.2). Before `:778`:
```cmake
# MAC: the CPU expert kernels are x86-64 only (AVX2 / AVX-512); an arm64 build has no CPU expert path (docs/mac/08)
set(_strata_x86 OFF)  # MAC:
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")  # MAC:
  set(_strata_x86 ON)  # MAC:
endif()  # MAC:
find_package(Threads REQUIRED)  # MAC: needed without the x86 block too
if(_strata_x86)  # MAC:
```
and after `:793`:
```cmake
endif()  # MAC: _strata_x86
```
`find_package(Threads)` moves out of the block because `direct_file_async_test` (edit F) and the Mac targets link
`Threads::Threads`. Upstream's own call inside the block stays; a second call is harmless. The
`CMAKE_OSX_ARCHITECTURES` test makes an arm64 build on an Intel Mac take the arm64 path.

**D: no ggml CPU experts**, after `:801`:
```cmake
if(NOT _strata_x86)  # MAC: the ggml-cpu experts live in strata_kernels_cpu
  set(STRATA_NATIVE_EXPERTS OFF CACHE BOOL "" FORCE)  # MAC:
endif()  # MAC:
```
With `STRATA_NATIVE_EXPERTS` off, an arm64 configure does not add ggml at `:825-837`, and fetches nothing unless the
tests need the oracle (section 6.1). `STRATA_GGML_DIR` stays declared at `:801`, so passing it never gives an
"unused variable" warning.

**E: the CPU kernel tests.** `  if(_strata_x86)  # MAC:` before `:996` (`expert_multi_test`), and
`  endif()  # MAC: _strata_x86` after `:1013` (`pool_stress`).

**F: `direct_file_async_test` on macOS.** `:1015` changes from `if(UNIX AND NOT APPLE)` to:
```cmake
  if(UNIX)  # MAC: direct_file.cpp has an F_NOCACHE path on macOS (docs/mac/06 section 7.4)
```
It lands together with the `direct_file.cpp` patch (section 3.2), in work item H2 of [06](06-engine-host-port.md).

**G: the backend**, after `:1029`, the last line:
```cmake
if(STRATA_ENABLE_METAL)  # MAC:
  include(cmake/metal/backend.cmake)  # MAC:
endif()  # MAC:
```
The include is last because `backend.cmake` links the upstream target `strata_spec` (`:750`) and labels every test
defined before it (section 6.2). As a consequence, its requirement errors (section 4.1) appear at the end of the
configure.

`spikes/metal-build/mac-port/root-edits.patch` holds these edits. `spikes/metal-build/mac-port/apply_root_edits.py` applies
the same edits by content: it checks the exact upstream text at each anchor line, refuses a file that already has a
`# MAC:` line or has CRLF endings, and inserts from the bottom up. The project applies the edits once. During later compatibility reviews, compare the edits with the selected
reference revision and port only required changes ([10](10-implementation-plan.md)). The script can reapply the
project edit set to a reviewed upstream `CMakeLists.txt`; it refuses if any anchor has changed.

### 3.2 `src/platform/direct_file.cpp`

macOS has no `O_DIRECT`. `F_NOCACHE` bypasses the unified buffer cache without alignment rules, and `F_RDAHEAD` 0
turns off read-ahead ([06](06-engine-host-port.md), section 7.4). At `:308`:
```cpp
#ifdef __APPLE__   // MAC: no O_DIRECT; F_NOCACHE has no alignment rule (docs/mac/06 section 7.4)
    impl_->fd = ::open(path.c_str(), O_RDONLY);
    if (impl_->fd >= 0) { fcntl(impl_->fd, F_NOCACHE, 1); fcntl(impl_->fd, F_RDAHEAD, 0); }
#else
    impl_->fd = ::open(path.c_str(), O_RDONLY | O_DIRECT);
#endif
```
The patch is in `root-edits.patch` because `strata_metal_platform` (section 4.4) compiles the file, so the skeleton
needs it to build.

### 3.3 Current target boundary

The current build supports native Apple silicon on macOS 27 or newer. The root `CMakeLists.txt` rejects non-Darwin
hosts before compiler detection, and `cmake/metal/backend.cmake` requires arm64 plus the macOS 27 SDK and Metal
toolchain. CMake defines no Windows or Linux product target.

### 3.4 Build-graph checks

The active CI checks the macOS arm64 Metal build and its host and GPU tests. There is no cross-platform build-graph
comparison because Windows and Linux are outside the supported product scope.

### 3.5 `.gitattributes` and `.gitignore`

`.gitattributes` gets, after its last line:
```
# MAC: Metal sources and the files CMake parses keep LF on every clone (docs/mac/08-build-and-distribution.md, section 3.5)
*.metal text eol=lf
src/metal/kernels/manifest.txt text eol=lf
METAL_REV text eol=lf
```
The manifest parser and kernel identity checks use canonical LF bytes. These attributes keep the Metal sources,
manifest and release number stable in the repository and CI. The proposed source identity helper is not implemented;
see section 8.3.

`.gitignore` gets, after its last line:
```
# MAC: Finder's folder files, and GPU traces written by strata --metal-capture (docs/mac/08-build-and-distribution.md, section 3.5)
.DS_Store
*.gputrace
```
Finder writes `.DS_Store` into every folder it shows. The proposed source identity in section 8.3 must ignore
this local metadata file. GPU traces are described in section 10.2.

Upstream's `.gitignore` already covers every Mac build folder: `build*/` covers `build/`, `build-dev/`, `build-ci/`,
`build-release/` and `build-vision-release/`. `/dist/`, `/engine/` and `/third_party/llama.cpp` are covered too.

Verified: `git check-attr` reports `text: set` and `eol: lf` for the three paths, and `git check-ignore` matches
`.DS_Store`, `src/.DS_Store` and `x.gputrace`. On a fresh clone at `UPSTREAM_SYNC`, `apply.sh` gives a tree identical to the scratch project.

---

## 4. `cmake/metal/backend.cmake`

### 4.1 Requirement checks

They run in this order. Each fails the configure with a message that names the fix:

| Check | Message |
| --- | --- |
| Root target-platform guard (`APPLE` and `CMAKE_SYSTEM_NAME STREQUAL "Darwin"`) | `Strata supports macOS targets only; use an Apple Silicon Mac with Xcode 27 or newer` |
| `APPLE` in the Metal backend | `Strata requires macOS on Apple silicon` |
| `CMAKE_OSX_ARCHITECTURES STREQUAL "arm64"` | `Strata builds arm64 only: pass -DCMAKE_OSX_ARCHITECTURES=arm64 (got '<value>')` |
| `CMAKE_OSX_DEPLOYMENT_TARGET` set and `VERSION_GREATER_EQUAL 27.0` | `Strata requires macOS 27: pass -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 (got '<value>')` |
| `find_program(STRATA_XCRUN xcrun REQUIRED)` | CMake's own message |
| `xcrun -sdk macosx --show-sdk-version` succeeds and is 27.0 or newer | `Strata requires the macOS 27 SDK from Xcode 27 (found '<v>'): install Xcode 27 and run sudo xcode-select -s /Applications/Xcode.app` |
| `xcrun -sdk macosx metal --version` succeeds and prints `Apple metal version` | `the Metal toolchain is not installed: run xcodebuild -downloadComponent MetalToolchain` |
| `METAL_REV` holds one positive integer | `METAL_REV must hold one positive integer (found '<v>')` |

On success, the configure prints
`-- Metal: Apple metal version 32023.921 (metalfe-32023.921.6), macOS SDK 27.0`.

The architecture and deployment target are not defaulted. A configure that forgets them fails, instead of building a
universal or x86 binary. The presets pass both; the future setup helper described in section 8.1 is not implemented.

### 4.2 Language and options

- `enable_language(OBJCXX)`, with `CMAKE_OBJCXX_STANDARD` 20, required, extensions off: the same C++20 as upstream's
  `CMAKE_CXX_STANDARD` 20.
- `option(STRATA_METAL_EMBED_SOURCES "keep the kernel sources in strata.metallib for the Xcode shader debugger" OFF)`
  (section 5.7).
- `METAL_REV` is read with `file(STRINGS ... LIMIT_COUNT 1)` and added to `CMAKE_CONFIGURE_DEPENDS`, so editing it
  reconfigures.
- Upstream options keep their meaning. `STRATA_WERROR` (`:59`) also turns on `-Werror` for the Objective-C++ files and
  the Metal compiler. `STRATA_BUILD_TESTS` and `STRATA_BUILD_CONVERSATION_TESTS` build the upstream CPU tests as on
  any platform, plus the Mac tests (section 6).

### 4.3 Compile and link options

The INTERFACE target `strata_metal_options` carries everything the Mac targets share:

| Option | Why |
| --- | --- |
| Links `strata_warnings` | Upstream's warning flags. They are guarded with `COMPILE_LANGUAGE:CXX`, so the next three rows add them for Objective-C++. |
| `-Wall -Wextra` for OBJCXX; `-Werror` with `STRATA_WERROR` | The same warning level as the C++ files |
| `-fobjc-arc` for OBJCXX | Metal objects are reference counted by ARC; there is no manual `release` |
| `-ffp-contract=off` for CXX and OBJCXX | Apple clang contracts `a*b + c` within an expression on arm64 by default (`-ffp-contract=on`; A.12). Upstream's host code is built for baseline x86-64, which has no FMA, so there every `a*b + c` is two roundings. With contraction off, the host math (the sampler's FP64 tail, the controller's statistics, the CPU references of the tests) rounds as on a PC. Code that wants a fused operation calls `std::fma`. |
| `-ffile-prefix-map=${CMAKE_SOURCE_DIR}=.` and `-fdebug-compilation-dir=.` | No absolute checkout path in the objects or their debug information (section 11.4) |
| `-g` | Debug information for the dSYM that the release keeps (section 11.6). The release binary is stripped. |
| `STRATA_METAL=1` | Compile definition for shared headers |
| `LINKER:-dead_strip` | Drops unreferenced code and data. `[[gnu::used]]` keeps the identity tags (section 4.5). |

`strata_spec`, the upstream target that the engine links, gets `strata_metal_options` too
(`target_link_libraries(strata_spec PRIVATE strata_metal_options)`), so its objects get the same path mapping. This
changes nothing in a PC build, because `backend.cmake` only runs on a Mac.

The upstream sources that the Mac build compiles are free of warnings under `-Werror` with Apple clang 21 (verified).

### 4.4 Targets

| Target | Kind | Sources | Links |
| --- | --- | --- | --- |
| `strata_metal_options` | INTERFACE | | Section 4.3 |
| `strata_metallib` | custom, `ALL` | `src/metal/kernels/**` | Section 5 |
| `strata_metal_platform` | STATIC | `src/platform/memory.cpp`, `memory_probe.cpp`, `direct_file.cpp`, `src/ngram/ple_reader.cpp`, `src/kernels/ngram.cpp` | `strata_artifact`, `Threads::Threads` |
| `strata_metal_host` | STATIC | `src/metal/host/context.mm`, `encoder.mm`, `buffers.mm`, `events.mm`, `pipeline_cache.mm`, `capture.mm` | `strata_metal_platform`; Metal, Foundation, IOKit and CoreFoundation. Depends on `strata_metallib`. |
| `strata_metal_engine` | STATIC | Every source in the explicit `STRATA_METAL_ENGINE_SOURCES` list in `cmake/metal/backend.cmake`, including window/session storage and operation-specific dispatch files | `strata_metal_host`, `strata_spec` |
| `strata_metal_cli_options` | STATIC | `src/metal/program/options.cpp` | `strata_metal_options` |
| `strata` | executable | `main.cpp`, `serve.cpp`, `oneshot.cpp`, `stats.cpp`, `watchdog.cpp`, `version.cpp` under `src/metal/program/` | `strata_metal_engine`, `strata_metal_cli_options` |

The source lists follow [06](06-engine-host-port.md), section 1.5. A new source file is added to its list here: the
lists are explicit, never globbed, so a stray file cannot change the engine.

Mac targets outside the Metal engine include `strata-gguf`, `strata-dequant`, and `strata_spec`. The old discrete-GPU `strata-plan` utility is removed; model-fit decisions belong to the Metal planner.

The build tree has the release layout: `build/strata` and `build/strata.metallib` sit side by side, so the engine runs
from the build tree without `--metallib`.

### 4.5 `version.cpp`: the engine's identity

```cpp
// src/metal/program/version.cpp
#include "metallib_sha256.inc"

namespace strata::metal {
[[gnu::used]] static const char kEngineTag[] = "engine=" STRATA_VERSION "\n";
[[gnu::used]] static const char kMetallibTag[] = "metallib_sha256=" STRATA_METALLIB_SHA256 "\n";
const char* engine_version() { return STRATA_VERSION; }
int metal_rev() { return STRATA_METAL_REV; }
const char* metallib_sha256() { return STRATA_METALLIB_SHA256; }
}  // namespace strata::metal
```

- `STRATA_VERSION` is upstream's compile definition, the CMake project version.
- `STRATA_METAL_REV` is set on this one file (`set_source_files_properties(... COMPILE_DEFINITIONS)`), so a
  `METAL_REV` bump recompiles one file and relinks.
- `metallib_sha256.inc` is generated by the kernel build (section 5.4). `OBJECT_DEPENDS` on it makes `version.cpp`
  recompile exactly when the library's bytes change.
- The two tags are what tools read from the binary without running it:
  - the current `strata_version` CTest reads both tags (section 6.3);
  - the planned release asset validator would check them too (section 11.7).
- `strata --version` prints `strata <engine_version()> metal <metal_rev()>` ([06](06-engine-host-port.md),
  section 3.10).

Historical spike result: its stripped release-shaped binary contained the `engine` and `metallib_sha256` tags once
each. The current development build verifies the same tag format with `strata_version`; release binaries are not yet
produced by this repository.

### 4.6 Link options for reproducibility

The linker hashes the debug map into the binary's UUID, and the code signature covers the UUID. The debug map lists
every object file by path and modification time. So the stripped binary would differ between two checkouts unless
the debug map does not. `backend.cmake` sets, on `strata` only:
```cmake
file(REAL_PATH ${CMAKE_BINARY_DIR} _strata_real_build)
target_link_options(strata PRIVATE LINKER:-oso_prefix,${_strata_real_build}/)
set_target_properties(strata PROPERTIES CXX_LINKER_LAUNCHER "${CMAKE_COMMAND};-E;env;ZERO_AR_DATE=1")
```
- `-oso_prefix` makes the object paths relative to the build folder. `REAL_PATH` matters, because the linker sees
  `/private/tmp`, not `/tmp`.
- `ZERO_AR_DATE=1` makes `ar` and the linker record modification time 0. The launcher sets it for the link step only.
- `dsymutil` must then run with the build folder as its working directory (section 11.3, step 5).

Verified: two checkouts at different paths give the same unstripped `strata` (120,568 B) and the same stripped
`strata` (107,576 B). The UUID equals the dSYM's in both (appendix A.8).

---

## 5. The kernel library: `strata.metallib`

### 5.1 Layout

| Path | Contents |
| --- | --- |
| `src/metal/kernels/common/*.h` | Shared Metal headers: types, block layouts, helpers. No kernel. |
| `src/metal/kernels/safe/*.metal` | Kernels compiled with safe math. The parity gates hold them to bitwise or ulp bounds ([04](04-metal-kernels.md)). |
| `src/metal/kernels/fast/*.metal` | Kernels compiled with fast math. Only where [04](04-metal-kernels.md) allows it. |
| `src/metal/kernels/manifest.txt` | One line per kernel (section 5.2) |
| `src/metal/kernels/THIRD_PARTY` | Source, commit and licence of any kernel code adapted from elsewhere (for example ggml's Metal kernels, MIT). It goes into `LICENSES.txt` (section 11.9). While nothing is adapted, it holds the one line `none`. |

A kernel's math mode is a property of its folder, so a flag cannot be forgotten for one file. Helpers in
`common/` must be `static` or `inline`: a helper with external linkage would be defined once per `.metal` file that
includes it, and the check of section 5.4 refuses it.

### 5.2 The manifest

```
# Every kernel in strata.metallib: <function> <safe|fast> <threads-x> <threads-y> <threads-z>, one space apart.
# cmake/metal/check_manifest.cmake fails the build when the .metal files and this list disagree
# (docs/mac/08-build-and-distribution.md, section 5.4).
strata_selftest_add safe 256 1 1
strata_selftest_exp fast 256 1 1
```

- Lines that are empty or start with `#` are skipped.
- Every other line must match `^([a-z_][a-z0-9_]*) (safe|fast) ([1-9][0-9]*) ([1-9][0-9]*) ([1-9][0-9]*)$`.
- Each threadgroup axis is at most 1024, and their product is at most 1024. The shape is exact: a 256×1×1 group
  differs from a 128×2×1 group even though both contain 256 threads. `Context` checks the product against the
  pipeline's `maxTotalThreadsPerThreadgroup` at startup ([06](06-engine-host-port.md), section 2.3).

The configure reads the manifest and writes `build/generated/kernel_manifest.inc`, one row per kernel. It uses
`file(CONFIGURE)`, which only writes when the content changes:
```cpp
// generated from src/metal/kernels/manifest.txt by cmake/metal/metallib.cmake; do not edit
{"strata_selftest_add", KernelMath::safe, 256, 256, 1, 1},
{"strata_selftest_exp", KernelMath::fast, 256, 256, 1, 1},
```
`Context` builds one pipeline per row. The manifest is the only list of kernels in the repository.

Configure errors:
- `src/metal/kernels/manifest.txt: bad line '<line>' (want '<function> <safe|fast> <threads-x> <threads-y> <threads-z>', one space apart)`
- `src/metal/kernels/manifest.txt: threadgroup axes must each be in 1..1024`
- `src/metal/kernels/manifest.txt: threadgroup for <name> has <count> threads, maximum is 1024`
- `src/metal/kernels/manifest.txt: <name> is listed twice`
- `src/metal/kernels/manifest.txt lists no kernel`

### 5.3 Compiling: one AIR file per source

Each `.metal` file under `safe/` and `fast/` becomes `build/metal/<mode>/<name>.air`. The files are found with
`file(GLOB ... CONFIGURE_DEPENDS)`, which returns them sorted, so the link order is fixed and a new file
reconfigures. The command is:
```
/bin/sh cmake/metal/metal.sh <common flags> <mode flags> -c <src> -o <air> -MMD -MF <air>.d -MT <air>
```
with `DEPFILE <air>.d`, so a header edit rebuilds exactly the AIR files that include it.

| Flags | Value |
| --- | --- |
| Common | `-std=metal4.0 -mmacosx-version-min=${CMAKE_OSX_DEPLOYMENT_TARGET} -Wall -gline-tables-only -frecord-sources -ffile-prefix-map=${CMAKE_SOURCE_DIR}=. -fdebug-compilation-dir=. -I src/metal/kernels/common`, plus `-Werror` with `STRATA_WERROR` |
| `safe` | `-fmetal-math-mode=safe -fmetal-math-fp32-functions=precise -ffp-contract=off` |
| `fast` | `-fmetal-math-mode=fast` |

`-gline-tables-only -frecord-sources` keep line tables and sources for the GPU debugger and the symbols file. The
release strips the sources from the shipped library (section 5.4).

**`metal.sh`** removes the two absolute paths that the compiler would otherwise record:
```sh
#!/bin/sh
# xcrun metal with the toolchain and SDK paths mapped away; both are resolved on every call because the toolchain's
# cryptex mount point has a random suffix that can change (docs/mac/08-build-and-distribution.md, section 5.3)
tc=$(cd "$(dirname "$(xcrun -sdk macosx -f metal)")/../.." && pwd -P) || exit 1
sdk=$(xcrun -sdk macosx --show-sdk-path) || exit 1
exec xcrun -sdk macosx metal "-ffile-prefix-map=$tc=/metal-toolchain" "-ffile-prefix-map=$sdk=/macos-sdk" "$@"
```
- `pwd -P` matters. `xcrun -f` returns `/var/run/...`, but the compiler records the real path `/private/var/run/...`.
  Mapping only the first form leaves the path in the library.
- The paths are resolved on every call, never cached at configure time, because the cryptex suffix can change
  (section 2.1).

Without the wrapper and `-fdebug-compilation-dir=.`, the stripped library contains the cryptex path and the build
folder. With them, it contains no absolute path, and two build folders give the same bytes (verified,
`spikes/metal-build/incremental/`).

### 5.4 Link, strip, check, hash

One custom command, whose only declared output is a stamp file, `build/metal/strata_metallib.stamp`. Its byproducts
are `build/strata.metallib`, `build/strata.metallibsym` and `build/generated/metallib_sha256.inc`. It depends on every
AIR file, the manifest and `check_manifest.cmake`. Its steps:

1. `xcrun -sdk macosx metallib <every AIR> -o build/strata.metallib.tmp`.
2. `xcrun -sdk macosx metal-dsymutil -flat -remove-source build/strata.metallib.tmp -o build/strata.metallibsym`. This
   moves the sources and debug information into the symbols file and rewrites the library in place without them.
   With `STRATA_METAL_EMBED_SOURCES` ON, this step instead removes any old `strata.metallibsym` (section 5.7).
3. `cmake -P cmake/metal/check_manifest.cmake` with `XCRUN`, `MANIFEST`, `AIRS` (the AIR paths joined with `|`, one
   `-D` value; no path the build makes contains `|`), `TMP`, `METALLIB` and `INC`. It runs `metal-nm` on each AIR file
   and on the linked library, and collects every error before failing:

   | Rule | Error (each is listed under `strata.metallib does not match its manifest:`) |
   | --- | --- |
   | A defined symbol `_Z...` (a C++ mangled name) | `<src>: <symbol> has external linkage (make the helper static or inline)` |
   | A kernel that the manifest does not list | `<src>: kernel <f> is not in src/metal/kernels/manifest.txt` |
   | A kernel defined in two files | `<src>: kernel <f> is also defined in <other src>` |
   | A kernel in the folder of the other math mode | `<src>: kernel <f> is listed as <mode> math but lives in <folder>/` |
   | A safe-math file that calls `air.fast_*`. `fast::exp` and friends emit these even under `-fmetal-math-mode=safe`. | `<src>: calls air.fast_<x> (a fast:: function) in a safe-math file` |
   | A manifest kernel defined nowhere | `src/metal/kernels/manifest.txt: kernel <f> is defined in no .metal file` |
   | The linked library's defined functions differ from the manifest (a guard against a toolchain change) | `strata.metallib exports '<list>', the manifest lists '<list>'` |

   Only when every rule holds, the script renames `TMP` to `METALLIB`. It then writes
   `#define STRATA_METALLIB_SHA256 "<64 hex>"` to `INC.tmp`, and copies that to `INC` with `copy_if_different`.
4. `cmake -E touch` the stamp.

`add_custom_target(strata_metallib ALL DEPENDS <stamp>)`. `strata_metal_host`, `strata` and the tests that include
the hash header depend on it.

Because the library and the header are byproducts, and the header is only rewritten when the hash changes, a kernel
edit that leaves the library's bytes unchanged rebuilds nothing downstream.

### 5.5 Incremental builds

Verified with `spikes/metal-build/incremental/` and the skeleton:

| Change | What rebuilds |
| --- | --- |
| Nothing | Nothing (`ninja: no work to do`) |
| `touch` of a `.metal` file or a header | Its AIR files and the stamp. The library's bytes are the same, so `version.cpp` is not compiled and nothing is linked. |
| An edit of a `.metal` file or a header in `common/` | The AIR files that include it, the stamp, `version.cpp`, and the link of `strata` and of the tests that include the hash header |
| A new `.metal` file | A reconfigure (the glob), then its AIR file and the stamp |
| `manifest.txt` | A reconfigure, `kernel_manifest.inc`, the stamp, and the objects that include the table |
| `METAL_REV` | A reconfigure, `version.cpp` and the link |
| A Metal toolchain update | Every AIR file. The depfiles list the toolchain's headers by real path, and the old cryptex path no longer exists. |

### 5.6 Runtime pairing

`Context::open` ([06](06-engine-host-port.md), section 2.3) must:
1. Find the library: `--metallib PATH` if given, else `strata.metallib` in the folder of the real path of the
   executable (`_NSGetExecutablePath`, then `realpath`). Never the working directory.
2. Read its bytes once. If that fails: `cannot read <path> (strata.metallib ships next to strata)`.
3. Compute SHA-256 (`CC_SHA256`) and compare it with `metallib_sha256()`. On a mismatch:
   `strata.metallib does not belong to this engine (<path> has SHA-256 <16 hex>..., the engine was built with
   <16 hex>...); reinstall the engine`. Exit 1.
4. Create the library from the same bytes (`newLibraryWithData:`), so the bytes that were checked are the bytes
   loaded.

`--metallib PATH` is for development. It skips step 3, and warns
`strata: --metallib: <path> is not checked against this engine`.

Why a hash and not a version: setup moves `strata` and `strata.metallib` into `engine/` one after the other
([07](07-server-setup-port.md), section 4.7.6). A failure between the two moves, or a developer copying one file,
would otherwise run kernels with the wrong layout, and the result would be wrong text, not an error.

### 5.7 `STRATA_METAL_EMBED_SOURCES`

ON in `mac-dev` (section 7), OFF everywhere else. ON skips the strip in step 2 of section 5.4, so the library keeps
the kernel sources and Xcode's shader debugger can show them in a capture. Such a library is about three times as
large (29,387 B against 8,904 B for the skeleton's two kernels) and is never shipped:
- `make_release.py` asserts the option is OFF in its build cache (section 11.3);
- `check_asset.py` strips a copy of the shipped library again and requires the bytes to be unchanged (section 11.7).
  Verified: a stripped library is unchanged by a second strip, and a library with sources is not.

The sources are recorded with the `-ffile-prefix-map` paths, `./src/metal/kernels/...`. Xcode resolves them when the
checkout is the working directory of the capture, or through its source-path setting.

---

## 6. Tests

### 6.1 The ggml oracle

The dequantiser and kernel parity tests compare against ggml's CPU backend from the pinned llama.cpp
([09](09-testing-and-validation.md)). `tests.cmake` builds it only when `STRATA_BUILD_TESTS` is ON:

- **Source.** `STRATA_GGML_DIR` if set. Otherwise FetchContent of `https://github.com/ggml-org/llama.cpp.git` at the
  commit in `third_party/ggml/VERSION.txt` (`3cf03257f219afbe7334045ff7c6a06ac68c627d` at the baseline), with
  `SOURCE_SUBDIR strata-populate-only`. That folder does not exist, so `FetchContent_MakeAvailable` downloads without
  adding llama.cpp's own CMake project. This avoids upstream's deprecated `FetchContent_Populate` and its CMP0169
  warning.
- **Settings**, all forced in the cache before `add_subdirectory`: `GGML_METAL`, `GGML_BLAS`, `GGML_ACCELERATE`,
  `GGML_OPENMP`, `GGML_NATIVE`, `GGML_CCACHE`, `GGML_BACKEND_DL`, `GGML_BUILD_TESTS`, `GGML_BUILD_EXAMPLES` and
  `BUILD_SHARED_LIBS` OFF, and `GGML_STATIC` ON. With only the scalar and NEON code, the oracle's results do not
  depend on the Mac that runs the test. Upstream's own ggml block (`:825`) leaves `GGML_METAL` at its Apple default,
  ON, which is one reason the Mac does not reuse it.
- `add_subdirectory(<llama.cpp>/ggml ${CMAKE_BINARY_DIR}/ggml-oracle EXCLUDE_FROM_ALL)`: only what a test links gets
  built.
- After `add_subdirectory`, `target_compile_options(ggml-base PRIVATE -ffp-contract=off)` and the same for
  `ggml-cpu`. ggml's scalar reference code (`dequantize_row_*`, `quantize_row_*_ref`) then rounds as a PC without
  FMA does; its NEON paths keep their explicit `vfmaq` intrinsics.
- `strata_ggml_oracle` (INTERFACE) links `ggml-base` and `ggml-cpu`. Both are needed, because `ggml-cpu` alone does
  not carry ggml's include folder. ld then warns that `ggml-base` appears twice, which
  `LINKER:-no_warn_duplicate_libraries` silences.

Verified (`spikes/metal-build/oracle/`): configure 2.9 s, build 4.2 s; the binary links only `libSystem` and `libc++`.

### 6.2 Labels

Every test has exactly one label:

| Label | Needs | Runs on |
| --- | --- | --- |
| `host` | No GPU | Every runner, including a hosted Mac with a paravirtualised GPU |
| `metal` | A real Apple GPU, M1 or newer | The reference Mac, and hosted Macs if measurement X-8 allows ([07](07-server-setup-port.md), section 14) |
| `model` | The model files, at `STRATA_TEST_MODELS` | The reference Mac only |

`strata_metal_test(<name> <host|metal|model> <command> [args...])` adds a Mac test with its label. Any other label is
a configure error: `strata_metal_test(<name>): label must be host, metal or model, not '<label>'`.

Upstream tests are added with plain `add_test` and have no label. `tests.cmake` registers
`cmake_language(DEFER DIRECTORY ${CMAKE_SOURCE_DIR} CALL _strata_label_rest)`, which runs at the end of the root
folder and gives `host` to every test that still has no label. It covers tests defined before and after the include,
so no upstream line changes (verified, `spikes/metal-build/labels/`). An upstream test that needs a GPU or a model
does not exist on a Mac: those are all in the CUDA block.

The labeller is registered before the `if(NOT STRATA_BUILD_TESTS) return()`, so it also runs when only
`STRATA_BUILD_CONVERSATION_TESTS` adds tests.

### 6.3 `strata_version`

A `host` test on the engine binary itself. It runs `cmake -P cmake/metal/check_engine.cmake` with `STRATA`,
`METALLIB`, `VERSION` and `METAL_REV`, and fails unless:
- `strata --version` exits 0 and prints exactly `strata <VERSION> metal <METAL_REV>\n`
  (`strata --version: exit <rc>, printed '<out>'`);
- the binary has exactly the two tags `engine=<VERSION>` and `metallib_sha256=<sha256 of METALLIB>`
  (`tags in <file>: '<found>', want engine=<v> and metallib_sha256=<h>`).

`--version` opens no Metal device ([06](06-engine-host-port.md), section 2.3, step 3), so the test needs no GPU.

### 6.4 The test list

This table lists the 28-test bootstrap subset: 20 host tests and 8 Metal tests. The current `mac-dev` configuration
also registers later work-item tests; the current `mac-host` and `mac-metal` presets select 36 and 41 tests respectively.

| Test | Label | Origin |
| --- | --- | --- |
| `conversation_cache_test` | host | Upstream (`STRATA_BUILD_CONVERSATION_TESTS`) |
| `gguf_reader_test`, `gguf_split_test` | host | Upstream |
| `suffix_drafter_test`, `controller_test`, `draft_policy_test` | host | Upstream (`strata_spec`) |
| `conv_cache_test`, `coupled_draft_test`, `bf16_bits_test` | host | Upstream |
| `direct_file_async_test` | host | Upstream, enabled on macOS by edit F |
| `platform_memory_test`, `ple_reader_selftest` | host | Mac twins of upstream tests that sit in the CUDA block (`:684`, `:695`). Same sources, linked with `strata_metal_platform`. |
| `dequant_oracle_test` | host | Mac: `strata_artifact`'s Q4_K dequantiser against ggml-cpu, bitwise, on 4,096 random blocks. The pattern of every parity test. |
| `metal_manifest_test` | host | Mac: the generated manifest table and hash header agree with the built library, without a GPU |
| `test_map` | host | Mac: the documented test, requirement and review maps agree with CTest |
| `strata_version` | host | Mac (section 6.3) |
| `options_dispositions_test` | host | Mac: the full upstream and Metal CLI inventory, dispositions, warning policy, arity and validation |
| `kernel_env_dispositions_test` | host | Mac: every kernel flag of section 9.2 through the parser, with its warning and rejected value; `STRATA_METAL_DISPATCH`, `STRATA_METAL_GEMM` and `STRATA_PREFILL_BF16X2` through their readers; the section 9.3 variables by a scan of the Mac sources for a quoted name; the `STRATA_METAL_ICB` notice and the rejection of `--check-logits` by running the strata executable |
| `metal_cli_modes_test` | host | Mac: empty invocation fails closed; device listing rejects unimplemented model modes and an explicit capture request |
| `metal_context_test` | metal | Mac: `Context::open` on the GPU. A first open compiles every pipeline; a second uses the archive; a damaged archive is ignored and then rewritten; a self-test dispatch runs; a wrong hash is refused. It uses `build-dev/metal/test-cache` as its archive folder (section 10.1). |
| `metal_fault_test` | metal | Mac: a deliberately faulting GPU command buffer is reported with its label; a following command buffer completes and writes the expected result. |
| `metal_buffers_test` | metal | Mac: shared untracked allocation, bounded BufferRef slices and widening rejection, rollback/Context-destruction invalidation, unavailable GPU-address sentinel and overflow handling, CPU/GPU-aligned arena slices, poison initialization, no-copy window validation, and residency registration. |
| `metal_kv_store_test` | metal | Mac: one Context-owned allocation exposes aligned per-layer QSA K/V and indexer views plus the shared RoPE table in all four formats. |
| `metal_mtp_kv_store_test` | metal | Mac: one Context-owned MTP K/V allocation exposes K/V and scale subviews in all four formats; an outer rollback followed by another allocation cannot revive the provisional store. |
| `metal_events_test` | metal | Mac: monotonic host and GPU `MTLSharedEvent` signals, unreserved GPU signal values, timeout overflow rejection, command-buffer ordering, and Context teardown draining. |
| `metal_list_devices_test` | metal | Mac: `strata --list-devices` reports IORegistry GPU cores, Metal limits and families, RAM and macOS version without opening pipelines. |
| `metal_self_test` | metal | Mac: `strata --self-test` builds the pipelines and runs the explicit driver smoke-test path without writing an archive. |
| `metal_kv_layout_test` | host | Mac: checks the exact 12-layer QSA K/V, indexer and shared RoPE layout plus the MTP drafter's single K/V layout for all four formats, including four-cell capacity rounding and invalid inputs. |

[09](09-testing-and-validation.md) adds the kernel parity tests and the model tests, with the same helper.

### 6.5 Adding a test

```cmake
add_executable(metal_q2_matvec_test tests/metal/q2_matvec_test.cpp)
target_link_libraries(metal_q2_matvec_test PRIVATE strata_metal_host strata_ggml_oracle strata_metal_options)
strata_metal_test(metal_q2_matvec_test metal metal_q2_matvec_test ${STRATA_METALLIB})
```
The oracles in `tests/metal/oracles/` are one static library, `strata_metal_oracles`, linked by every twin. It gets
`strata_metal_options`, so it is compiled with `-ffp-contract=off` like the rest of the host code, and each oracle
calls `std::fma` exactly where its kernel calls `fma` ([04](04-metal-kernels.md), section 3.3). `metal_manifest_test`
also checks the flag. The oracle library has a probe, `float strata_contract_probe(float a, float b, float c)
{ return a*b + c; }`, in its own source file. With `a = b = 1 + 2^-12` and `c = -(1 + 2^-11)` it returns 0 unfused and
`2^-24` fused (A.12); on `2^-24` the test fails with `oracle library contracts a*b + c; build it with
-ffp-contract=off`.
A test that reads the hash header also gets `target_include_directories(... ${STRATA_METAL_GENERATED_DIR})`,
`OBJECT_DEPENDS ${STRATA_METALLIB_SHA256_INC}` on its source, and `add_dependencies(... strata_metallib)`, as
`metal_context_test` does. Tests must use the `STRATA_METALLIB` path they are given; never `--metallib`, which skips
the hash check.

---

## 7. Presets

### 7.1 `CMakePresets.json`

Version 5 (CMake 3.24, upstream's minimum). Every configure preset has the condition `hostSystemName == Darwin`, so
the file is invisible on a PC.

| Configure preset | Folder | Cache values |
| --- | --- | --- |
| `mac-release` | `build/` | Ninja; `CMAKE_BUILD_TYPE=Release`, `STRATA_BUILD_TESTS=OFF`, `CMAKE_OSX_ARCHITECTURES=arm64`, `CMAKE_OSX_DEPLOYMENT_TARGET=27.0` |
| `mac-dev` | `build-dev/` | Inherits `mac-release`; plus `STRATA_BUILD_TESTS`, `STRATA_BUILD_CONVERSATION_TESTS`, `STRATA_WERROR`, `STRATA_METAL_EMBED_SOURCES` and `CMAKE_EXPORT_COMPILE_COMMANDS` ON |
| `mac-ci` | `build-ci/` | Inherits `mac-dev`, with `STRATA_METAL_EMBED_SOURCES=OFF`: the libraries the tests see are stripped as shipped |

A build preset of the same name exists for each.

| Test preset | Configure preset | Filter | Settings |
| --- | --- | --- | --- |
| `mac-host` | `mac-dev` | label `^host$` | `outputOnFailure`; `noTestsAction: error` |
| `mac-metal` | `mac-dev` | label `^metal$` | the same |
| `mac-model` | `mac-dev` | label `^model$` | the same |

`noTestsAction: error` makes a filter that matches nothing fail, instead of passing silently. The registered
`dense_vs_gguf_test` ([09](09-testing-and-validation.md)) also fails closed until `STRATA_TEST_MODELS` contains all
seven completed catalogue packs.

`mac-release` uses `build/` for the CMake Release configuration. `setup.py --build` is not implemented; except for
the read-only `setup.py --check-host`, its CLI exits before changing files. Build the configured release with
`cmake --build --preset mac-release`.

The presets do not set `STRATA_GGML_DIR`. A developer passes it on the command line to skip the oracle's download
(section 13).

### 7.2 Planned preset and setup parity checks (not implemented)

The release/setup integration and `tools/test_metal_build.py` do not exist yet. When implemented, the proposed test
will compare the `mac-release` cache with the future setup helper definitions, verify the pinned CMake and Ninja
versions, and check that the llama.cpp revision agrees between its metadata and the CMake fetch declaration. The
current workflow runs only the Python modules present in the checkout, then its explicit drift and release-policy
checks; it does not run this proposed test.

---

## 8. Setup build integration: definitions, identity and metadata

`setup.py`'s `main()` exits with status 2 before changing files for every option except the read-only
`--check-host`. No install flow calls the functions below yet: the install flow of 07 (S7, merged from main) does not
use them, and the build flow (S8) is open. The functions exist in `setup.py` next to main's S-track helpers, with
the release helpers `llama_pin`, `get_llama_cpp` and `cmake_build` that make_release.py uses (work item B8;
D-IMPL-940 and D-IMPL-945). The tests are in `tools/test_setup_metal.py`. Sections 8.3 to 8.5 describe them as
implemented.

### 8.1 `metal_engine_defs(llama)`

```python
def metal_engine_defs(llama) -> list:  # MAC:
    """The CMake definitions of the Mac engine; CMakePresets.json mac-release must equal them (tools/test_metal_build.py)."""
    return ["-DSTRATA_BUILD_TESTS=OFF", "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=27.0", f"-DSTRATA_GGML_DIR={llama}"]
```
`build_engine_metal` ([07](07-server-setup-port.md), section 4.7.6, step 3) and `make_release.py` (section 11.3,
step 3) both call it. `STRATA_GGML_DIR` costs nothing with tests off, and keeps a later test configure of the same
folder from downloading llama.cpp again.

### 8.2 `metal_vision_defs(llama, portable)`

```python
def metal_vision_defs(llama, portable) -> list:  # MAC:
    """strata-vision on a Mac (docs/mac/08 section 9): Metal, ggml's shaders embedded, no OpenMP."""
    return [f"-DLLAMA_DIR={llama}", "-DSTRATA_VISION_CUDA=OFF", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON",
            "-DGGML_OPENMP=OFF", "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=27.0",
            *(["-DSTRATA_PORTABLE=ON"] if portable else [])]
```
- `build_engine_metal` calls it with `portable=False`: a local build may use this Mac's CPU features.
- `make_release.py` calls it with `portable=True`.
- `-DGGML_OPENMP=OFF` must be passed in both cases. With `STRATA_PORTABLE`, `tools/vision/CMakeLists.txt` turns
  OpenMP off itself. Without it, the file sets `GGML_NATIVE` ON and leaves `GGML_OPENMP` at ggml's default, ON, and
  Apple clang has no OpenMP (section 2.3).

[07](07-server-setup-port.md), section 4.7.6, step 4, lists these definitions; the function replaces the inline list.

### 8.3 Proposed macOS source identity

The future Mac installer may hash the engine's CMake and source inputs to decide whether a local rebuild is needed.
The input set would include the root `CMakeLists.txt`, `CMakePresets.json`, `cmake/metal/`, `src/`, `include/`,
`METAL_REV`, and the pinned llama.cpp revision. A Finder-created `.DS_Store` must not affect that identity. The logic
is `source_hash` in `setup.py` (D-IMPL-943); no build calls it yet.

### 8.4 BUILD.json: informational keys

The keys of [07](07-server-setup-port.md), section 4.7.2, are what setup reads. A Mac BUILD.json also has these
keys, after `vision_src`. They are for people and tools, and setup never reads them:

| Key | Value | Source |
| --- | --- | --- |
| `xcode` | `"27.0 (27A266a)"` | `xcodebuild -version`: the version and build lines |
| `metal_toolchain` | `"com.apple.dt.toolchain.Metal.32023.921.5"` | `toolchainIdentifier` from `xcodebuild -showComponent MetalToolchain -json` |
| `commit` | 40 hex, with `-dirty` appended when `git status --porcelain` prints anything; `""` when the checkout has no `.git` (a ZIP download) | `git rev-parse HEAD` |
| `metallib_sha256` | 64 hex | SHA-256 of `engine/strata.metallib` |

`verify_release.py` uses `xcode` and `metal_toolchain` to decide whether a rebuild must match (section 11.8).
`check_asset.py` checks `metallib_sha256` (section 11.7). BUILD.json has no timestamp, and is written as
`json.dumps(meta, indent=1) + "\n"`, so the same build always gives the same bytes.

One function makes the whole file, for `build_engine_metal` ([07](07-server-setup-port.md), section 4.7.6, step 6)
and for `make_release.py` (section 11.3, step 7):

```python
def metal_build_json(eng: Path, source: str) -> dict:  # MAC:
    """BUILD.json of the Mac engine in `eng` (docs/mac/07 section 4.7.2, docs/mac/08 section 8.4).  A local build
    and a release both write it, so the two describe themselves the same way."""
    def out(*cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        return r.stdout.strip() if r.returncode == 0 else ""
    xcode = out("xcodebuild", "-version").splitlines()      # ["Xcode 27.0", "Build version 27A266a"]
    xcode = f"{xcode[0].split()[-1]} ({xcode[1].split()[-1]})" if len(xcode) == 2 else ""
    tc = json.loads(out("xcodebuild", "-showComponent", "MetalToolchain", "-json") or "{}")
    commit = out("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if commit and out("git", "-C", str(ROOT), "status", "--porcelain"):
        commit += "-dirty"
    vision = "gpu" if (eng / "strata-vision").exists() else "none"
    return {"version": source_version(), "metal_rev": metal_source_rev(), "backend": "metal", "arch": "arm64",
            "macos_min": "27.0", "gpu_families": ["apple7", "metal4"], "metal_std": "metal4.0", "source": source,
            "lib_dirs": [], "vision": vision, "src": source_hash(ENGINE_SOURCES),
            "vision_src": source_hash(VISION_SOURCES) if vision == "gpu" else None,
            "xcode": xcode, "metal_toolchain": tc.get("toolchainIdentifier", ""), "commit": commit,
            "metallib_sha256": hashlib.sha256((eng / METALLIB).read_bytes()).hexdigest()}
```
- The caller writes it as `json.dumps(meta, indent=1) + "\n"`.
- `vision_src` is null when the optional validation helper is absent; the proposed metadata records helper
  identity only when a helper is included.
- `-showComponent ... -json` prints `toolchainIdentifier`, `buildVersion`, `status` and two paths. Only the identifier
  is kept, because the paths name this Mac's cryptex mount.

### 8.5 `find_tool`

The proposed Mac `find_tool` helper searches the venv's `bin` directory (`Path(sys.executable).parent`) before
PATH. Setup is expected to run with the venv's Python without activating it, so PATH-first could pick an older
Homebrew or MacPorts `cmake` instead of the pinned `cmake==4.4.3`. The helper is `find_tool` in `setup.py`, and
`cmake_build` uses it; setup's own flow does not call it yet.

---

## 9. Optional local `strata-vision` helper

Current status: `tools/vision` is a standalone macOS-only validation helper. It is not included in the root CMake
build or a release asset. Its CMake file fetches the pinned llama.cpp, enables Metal and `GGML_NATIVE`, and disables
CUDA, HIP, Vulkan, SYCL, OpenCL, BLAS, Accelerate, OpenMP and dynamic backends. This helper's native CPU tuning is
for local image-path validation; it does not set the engine's distribution baseline.

Build it with:

```
cmake -G Ninja -S tools/vision -B /tmp/strata-vision-build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/strata-vision-build --target strata-vision
```

The helper accepts the release recipe of the next paragraphs: `-DLLAMA_DIR=<checkout>` skips the fetch, and
`-DSTRATA_PORTABLE=ON` turns off the native CPU tuning (D-IMPL-941). It is still not part of the root CMake build or
a setup-installed asset; the release files of section 11 include it.

- `GGML_METAL_EMBED_LIBRARY=ON` embeds ggml's shader source in the binary. Metal compiles it at the first start, so
  `strata-vision` needs no `.metallib` beside it and no Xcode at run time. The cost of that first compile is
  measurement B-4.
- The release adds three path maps, through `CMAKE_C_FLAGS`, `CMAKE_CXX_FLAGS` and `CMAKE_OBJC_FLAGS`:
  `-ffile-prefix-map=<llama>=third_party/llama.cpp -ffile-prefix-map=<ROOT>=. -ffile-prefix-map=<build>=build-vision`.
  The flags are a space-separated string, so `make_release.py` refuses a `ROOT` that contains whitespace
  (section 11.3).

Verified with the release definitions (`spikes/metal-build/vision-repro.sh`):
- two builds in different folders give the same binary, `<build>/bin/strata-vision`, 8,180,336 B. After
  `strip -S -x` under its own name, both are 7,453,408 B with `Identifier=strata-vision`;
- 12-18 s at `-j 14` on the reference Mac, over three runs;
- it links only `Accelerate`, `Foundation`, `Metal`, `MetalKit` and `CoreFoundation` from `/System/Library/Frameworks`,
  and `libSystem`, `libc++` and `libobjc` from `/usr/lib`, with no `LC_RPATH`;
- with no argument, it prints its usage and exits 2.

Host check on a Mac that is not the reference Mac (B9, 2026-10-13, Apple M5 Pro, macOS 27.0, Xcode 27.0 27A266a,
llama.cpp at the pin; D-IMPL-957): the two builds in separate folders give identical binaries, 8,191,600 B, and
identical stripped binaries, 7,468,320 B. The spec's figures above are from the reference Mac and are not matched
exactly. Checks 2 to 6 of section 11.7 pass for the stripped binary.

The binary contains one absolute path, `/tmp/perf-metal-%d.gputrace`. It is ggml's capture file name
(`ggml/src/ggml-metal/ggml-metal-context.m:526` at the pin), used only when `GGML_METAL_CAPTURE_COMPUTE` is set.
`check_asset.py` allows exactly this string in `strata-vision` (section 11.7).

The licences of what `strata-vision` links are in section 11.9.

---

## 10. Pipeline cache and GPU capture

### 10.1 The pipeline cache

`Context` builds every pipeline at startup ([06](06-engine-host-port.md), section 2.3), as upstream's
`CUDA_MODULE_LOADING=EAGER` loads every kernel. `pipeline_cache.mm` keeps the compiled pipelines in an
`MTLBinaryArchive`, so the next start does not compile again.

**Where.** `pipeline_archive_path(sha, gpu_name, why)`:

| `STRATA_METAL_ARCHIVE` | Folder |
| --- | --- |
| unset, empty or `1` | `~/Library/Caches/strata/` (`NSCachesDirectory`, user domain) |
| `0` | No archive. The stats line says `cache: off (STRATA_METAL_ARCHIVE=0)`. |
| anything else | That folder |

The folder is created with mode 0700. The file name is
`pipelines-<first 16 hex of the metallib SHA-256>-<GPU name>-<kern.osversion>.binarchive`, with every character of
the GPU name other than letters, digits and `-` replaced by `_`. For example:
`pipelines-b763d0449788d2a3-Apple_M5_Pro-26A428.binarchive`. A new engine, a different GPU and an OS update each get a
new file. Neither an OS update nor a GPU change could reuse the old one.

**How**, in `Context::build_pipelines`:
1. Open the archive at that path. If it is missing, start an empty one. If it is invalid, print
   `strata: ignoring the pipeline cache <path>: <error>` and start an empty one.
2. For each manifest kernel, try `newComputePipelineStateWithDescriptor` with
   `MTLPipelineOptionFailOnBinaryArchiveMiss` and the archive in `binaryArchives`.
3. If any kernel missed, print once `strata: compiling GPU kernels for this Mac (once per engine update)`. Compile the
   misses in parallel (the asynchronous API, joined with a dispatch group). Add each to the archive with
   `addComputePipelineFunctionsWithDescriptor`.
4. Serialise to `<path>.tmp.<pid>`, then `rename` onto `<path>`. Two engines starting at once each write a complete
   file, and the last rename wins.
5. `prune_pipeline_archives(path)` removes every other `pipelines-*.binarchive` in the folder, and every
   `pipelines-*.tmp.*` older than 600 s. A younger temporary file may belong to an engine that is writing it.
6. Check each pipeline: `threadExecutionWidth == 32`, and `maxTotalThreadsPerThreadgroup` at least the manifest's
   threads ([06](06-engine-host-port.md), section 2.3).

No archive error is fatal: a failure to create, add or write prints a `strata: pipeline cache: ...` line, and the
engine runs with the pipelines it has. `--plan-only`, `--version` and `--list-devices` never open the archive.
When `MTL_DEBUG_LAYER=1` or `MTL_SHADER_VALIDATION=1`, `Context` skips `MTLBinaryArchive` and compiles the
pipelines directly. Validation-instrumented pipelines did not produce warm archive hits on the reference Mac, and
opening the normal cached archive under validation made a fault-test child terminate with SIGSEGV during Context
initialization. The ordinary `metal_context_test` checks cold, warm and damaged archives without validation; its
validation run checks direct compilation and a selftest dispatch.

The stats line, on stderr after the pipelines are built:
`strata: <n> pipelines (<a> from the cache, <c> compiled) in <ms> ms; cache: <path or reason>`.

Verified with the skeleton's two kernels (appendix A.8): first run 48.5 ms with 2 compiled; second run 0.5 ms with 2
from the cache; the archive is 28.9 KB with mode 0644. With `STRATA_METAL_ARCHIVE=0`, the same kernels compiled in
0.8 ms: the OS keeps its own shader cache. Whether the archive is worth it for the full kernel set is
measurement B-1. The "compiling" line prints on any archive miss, even when the system cache makes the compile fast.

**One engine at a time per folder.** Pruning deletes the archives of other engines. A development build and the
installed engine that share `~/Library/Caches/strata/` would rebuild the archive each time the user switches between
them. Development runs and tests should therefore set `STRATA_METAL_ARCHIVE` to their own folder (section 13);
`metal_context_test` uses `build-dev/metal/test-cache`.

**Metal 4 archives.** `MTL4Archive` and `MTL4PipelineDataSetSerializer` replace `MTLBinaryArchive` for pipelines made
with the Metal 4 compiler API. If the engine moves to that API ([06](06-engine-host-port.md), section 8), the same
file rules apply: path, temporary name, rename, prune, never fatal. This is not verified.

### 10.2 GPU capture

`strata --metal-capture F`, or `STRATA_METAL_CAPTURE=F`, writes a GPU trace of the first decode window to
`F.gputrace`, for Xcode's GPU debugger. It is for development only.

- **The engine enables capture itself.** It calls `setenv("MTL_CAPTURE_ENABLED", "1", 1)` before it creates the
  device. Verified: without the variable, `supportsDestination:MTLCaptureDestinationGPUTraceDocument` is NO; with it
  set by the program before the device exists, the capture starts.
- `.gputrace` is appended to `F` when missing. If the file exists, exit 2 with
  `strata: <F> exists; Metal does not overwrite a trace`.
- The capture object is the command queue. The trace covers the command buffers of the first decode window
  ([06](06-engine-host-port.md), section 10.2), from `begin_capture` to `end_capture`.
- Errors: `GPU capture is not available (MTL_CAPTURE_ENABLED must be set before Metal starts)`, and
  `cannot start the GPU capture: <error>`. Both exit 1.
- A trace holds every buffer the captured work touches. With a real model, that means the resident weights. The
  engine warns `strata: the capture includes <n> GiB of GPU buffers` when the residency set is over 8 GiB. The
  trace's size with a real model is measurement B-10.

For timing a real run, use Instruments instead:
`xcrun xctrace record --template 'Metal System Trace' --launch -- engine/strata <args>`. It records timings, not
buffers.

Verified with the skeleton: an 80 KB trace of the self-test dispatches.

---

## 11. Release design

The release tools are in `tools/metal/`: `make_release.py` (11.3), `check_asset.py` (11.7) and `verify_release.py`
(11.8), and the workflow `mac-release.yml` (12.3). Their host parts are tested in the macOS Python suite (09 section
9.3). The asset of 11.3 was built and checked on a Mac (B10, B11; D-IMPL-946). No release has been published from this
checkout, and the publication path has not been tried in a test repository (B12, B18). verify_release's step 5
installs with setup's S7 functions, and step 6 refuses until 09 T15 exists (D-IMPL-950). Signing and notarisation are not yet tested (D-IMPL-959).

### 11.1 Versions and the release policy

A Mac release is the tag `v{version}-metal.{metal_rev}` with the asset `strata-macos-arm64.zip`. This is the release
contract of [07](07-server-setup-port.md), section 4.1:
- `{version}` is `source_version()`, the CMake project version. The project keeps it equal to the upstream version selected for its current compatibility baseline.
- `{metal_rev}` is the integer in `METAL_REV`.

The rules:

1. **`METAL_REV` only grows.** It goes up by one in every Mac release. It is never reset, not even when the upstream
   version changes. So every tag is unique, and `metal_rev` alone orders all Mac engines
   ([06](06-engine-host-port.md), section 5.2).
2. **Main is always released.** For every commit on main, the release `v{source_version()}-metal.{METAL_REV}` exists,
   is published, and is the project's latest release. Setup on main builds its download URL from these two values, so a
   user who clones main always finds the asset. Four values can only reach main through a release branch, after its
   release is published:
   - the CMake project version;
   - `METAL_REV`;
   - `MIN_ENGINE`;
   - `MIN_ENGINE_METAL`.
3. **`MIN_ENGINE_METAL` is raised only when setup or the server needs it.** Raise it when they use something that an
   older engine does not have, such as a new flag, INFO key or PLAN key. The server must work with every engine from
   `MIN_ENGINE_METAL` up ([07](07-server-setup-port.md), section 4.1).
4. **An engine change without a setup change goes to main without a release.** A prebuilt install keeps the engine of
   the latest release until the next one. A local CMake build compiles main's sources; the future `setup.py --build` command described in section 8 is not implemented.

**Making a release**, here `v0.1.34-metal.2`:
1. `git switch -c release/0.1.34-metal.2 main`. If a compatibility review adopts a newer upstream CMake version,
   implement that change as a project commit on this branch; do not merge upstream history ([10](10-implementation-plan.md)).
2. Raise `METAL_REV` by one. Raise `MIN_ENGINE_METAL` if rule 3 applies. Commit and push. `ci.yml` runs on
   `release/**` (section 12.1).
3. `git tag -a v0.1.34-metal.2 -m "Strata for macOS 0.1.34, Metal release 2"` on the branch head, then
   `git push origin v0.1.34-metal.2`. The policy check rejects a lightweight tag. When a kernel change since the
   previous release changes rounding, the message has a line `Numerics: changed (<reason>)`
   ([09](09-testing-and-validation.md), section 4.3). `mac-release.yml` builds the release, verifies it and publishes it
   (section 12.3).
4. Merge the branch into main with a pull request. Keep that release branch until a newer release branch is merged:
   the main check verifies that the published annotated tag still points at the retained branch head.

**`tools/metal/check_release_policy.py [--remote]`** imports `setup` for the values and checks:

| Check | When | Failure |
| --- | --- | --- |
| `METAL_REV` is one line with one positive integer | always | `METAL_REV must hold one positive integer` |
| `MIN_ENGINE_METAL ≤ METAL_REV` | always | `MIN_ENGINE_METAL <m> is above METAL_REV <r>` |
| `MIN_ENGINE ≤ source_version()` | always | `MIN_ENGINE <m> is above the engine version <v>` |
| `UPSTREAM_SYNC` is 40 lowercase hex and a newline | always | `UPSTREAM_SYNC must hold one 40-hex commit` |
| The branch name (`GITHUB_HEAD_REF`, else `GITHUB_REF_NAME`, else `git branch --show-current`), when it starts with `release/`, is `release/{version}-metal.{metal_rev}` | always | `branch <b> does not match the version: want release/<v>-metal.<r>` |
| The tag (`GITHUB_REF_TYPE == "tag"`) is `v{version}-metal.{metal_rev}` | always | `tag <t> does not match the version: want v<v>-metal.<r>` |
| Before publication, on a `release/` branch or a tag: `METAL_REV` is exactly one greater than every published Metal release (`gh api --paginate --slurp repos/{REPOSITORY}/releases?per_page=100`, excluding drafts) | `--remote` | `METAL_REV must be exactly <r> after <t>` |
| A published Metal release tag is not a prerelease | `--remote` | `Metal release <t> must not be a prerelease` |
| Every published tag shaped like a Metal release uses the canonical positive revision form `vX.Y.Z-metal.N` | `--remote` | `published Metal release tag <t> is malformed` |
| The tag for the expected published release exists and is annotated | `--remote` on main, release branch or tag | `tag <t> must be annotated` or `cannot verify tag <t>` |
| A tag points at the head of `release/{version}-metal.{metal_rev}` | `--remote` on main, on a tag, and on a release branch after publication | `tag <t> does not point at the head of <branch>` |
| After publication, a `release/` branch PR uses the actual PR head commit, whose tag is the latest release and has `strata-macos-arm64.zip` | `--remote` on a release branch whose tag is already published | A mismatched PR head or release fails |
| The project's latest published release (`gh release view --repo {REPOSITORY} --json tagName,assets`) has the tag `v{version}-metal.{metal_rev}` and an asset `strata-macos-arm64.zip` | `--remote` on main | `main must point at a published release: the latest release is <t>, this commit needs v<v>-metal.<r>` |
| Every fetch and push URL of `origin` and, in GitHub Actions, `GITHUB_REPOSITORY` match `RELEASE_REPOSITORY` | `--remote` | The checked repository and publication destination must match |

Before the first release exists, `--remote` passes with `no release yet` when `METAL_REV` is 1. Setup then compiles
the engine locally, because no asset exists. The release workflow targets `GITHUB_REPOSITORY` explicitly and
`check_release_policy.py --remote` verifies that it matches `setup.RELEASE_REPOSITORY` and the checked-out `origin`.
An unpublished release branch cannot be merged through a pull request. After its release is published, the
release-branch pull request is allowed only if the published tag is still the latest release, has the engine asset
and points at the pull request event's `head.sha`. The check uses that event value instead of the base repository's
same-named remote branch, which may differ for a pull request from another branch. A tag created on a stale commit fails.

### 11.2 The release files

Each release has three assets and a body:

| File | Contents |
| --- | --- |
| `strata-macos-arm64.zip` | What setup installs (the table below) |
| `strata-macos-arm64-symbols.zip` | `strata.dSYM/Contents/Info.plist`, `strata.dSYM/Contents/Resources/DWARF/strata`, `strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml` and `strata.metallibsym` (section 11.6) |
| `SHA256SUMS` | The `shasum -a 256` lines of the two zips |
| The release body | `NOTES.md`, written by `make_release.py` (section 11.3, step 9) |

Members of `strata-macos-arm64.zip`, in this order:

| Member | Mode | What it is | Skeleton size |
| --- | --- | --- | ---: |
| `strata` | 0755 | The engine: stripped and signed (sections 11.5 and 11.6) | 107,576 B |
| `strata.metallib` | 0644 | The kernels, sources stripped (section 5.4) | 8,904 B |
| `strata-vision` | 0755 | The image encoder: stripped and signed (section 9) | 7,453,408 B |
| `BUILD.json` | 0644 | [07](07-server-setup-port.md), section 4.7.2, with `source: "prebuilt"` and `vision: "gpu"`, plus the keys of section 8.4 | |
| `LICENSES.txt` | 0644 | Section 11.9 | |

There are no folders and no symlinks, and every member sits at the top level. Setup's unpack
([07](07-server-setup-port.md), section 4.7.1) relies on this. The sizes with the full kernel set are
measurement B-8.

### 11.3 `tools/metal/make_release.py`

```
python3 tools/metal/make_release.py [--sign IDENTITY] [--notarize (--keychain-profile P | --key K.p8 --key-id I --issuer U)]
                                    [--notes FILE] [--out dist] [--allow-dirty]
```

Run it with the venv's Python, from a checkout. Every step either succeeds or stops with `make_release: <problem>`
and exit 1.

1. **Preconditions.**
   - `sys.platform == "darwin"` and `platform.machine() == "arm64"`.
   - `macos.translated()` is False: not under Rosetta.
   - `macos.xcode_problem()` is None ([07](07-server-setup-port.md), section 4.7.6, step 1).
   - The tree is clean: `git status --porcelain` prints nothing, unless `--allow-dirty` is given.
   - `ROOT` contains no whitespace (section 9).
   - `setup.get_llama_cpp()` gives the pinned llama.cpp in `third_party/llama.cpp`.
2. **`SOURCE_DATE_EPOCH`.** It comes from the environment, else from `git log -1 --format=%ct`. It is used only for
   the zip dates: no binary contains a date.
3. **Engine.** Remove `build-release/`, then run
   `setup.cmake_build(ROOT, ROOT / "build-release", "strata", setup.metal_engine_defs(llama), None, "")`.
   - A fresh folder, never setup's `build/`: an old cache could hold other values.
   - Then assert from `build-release/CMakeCache.txt`: `CMAKE_BUILD_TYPE=Release`,
     `STRATA_METAL_EMBED_SOURCES=OFF`, `STRATA_BUILD_TESTS=OFF`, `CMAKE_OSX_ARCHITECTURES=arm64` and
     `CMAKE_OSX_DEPLOYMENT_TARGET=27.0`.
4. **Vision.** Remove `build-vision-release/`, then run
   `setup.cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision-release", "strata-vision", defs, None, "")`.
   - `defs` is `setup.metal_vision_defs(llama, portable=True)`, plus `-DCMAKE_C_FLAGS=F`, `-DCMAKE_CXX_FLAGS=F` and
     `-DCMAKE_OBJC_FLAGS=F` with the three path maps of section 9.
   - Then assert from its cache: `GGML_NATIVE=OFF`, `GGML_OPENMP=OFF`, `GGML_METAL=ON` and
     `GGML_METAL_EMBED_LIBRARY=ON`.
   - Then the vendor check of section 11.9 runs on `build-vision-release/bin/strata-vision`. Upstream's
     `tools/vision/CMakeLists.txt` puts the binary in `bin/`.
5. **Stage** into `<out>/stage/` and `<out>/stage-symbols/`. Both are removed first.
   - `dsymutil strata -o <out>/stage-symbols/strata.dSYM`, with `build-release/` as the working directory
     (section 4.6). The argument must be the bare name `strata`, because the dSYM records it.
   - Copy `build-release/strata`, `build-release/strata.metallib` and `build-vision-release/bin/strata-vision` into
     `stage/`.
   - Run `strip -S -x stage/strata` and `strip -S -x stage/strata-vision`, in place (section 11.6).
   - `dwarfdump --uuid stage/strata` must equal the dSYM's UUID.
   - Copy `build-release/strata.metallibsym` into `stage-symbols/`.
6. **Sign**, with `--sign`:
   `codesign --force --options runtime --timestamp --sign IDENTITY stage/strata stage/strata-vision`. Without
   `--sign`, the ad hoc signature that `strip` wrote stays (section 11.5).
7. **Metadata.**
   - `stage/BUILD.json` is `setup.metal_build_json(stage, source="prebuilt")`, the function `build_engine_metal`
     uses too (section 8.4).
   - `stage/LICENSES.txt` follows section 11.9.
8. **Asset.** `<out>/strata-macos-arm64.zip` contains the members of section 11.2, in that order, written by the
   deterministic zip writer (`spikes/metal-build/release/detzip.py`, moved into the tool):
   - every member gets the date of `SOURCE_DATE_EPOCH` and `create_system` 3 (Unix);
   - `external_attr` is `(S_IFREG | mode) << 16`;
   - every member is deflated at level 9;
   - nothing else from the file system reaches the zip.
9. **The other files.**
   - `<out>/strata-macos-arm64-symbols.zip`: the dSYM's files sorted by path, then `strata.metallibsym`, all 0644,
     written by the same writer.
   - `<out>/SHA256SUMS`.
   - `<out>/NOTES.md`, which contains:
     - the tag;
     - the engine version and `METAL_REV`;
     - `commit` and `UPSTREAM_SYNC`;
     - the `xcode` and `metal_toolchain` values;
     - how the binaries are signed (`ad hoc`, or `Developer ID (<team>), notarized`);
     - the requirement `macOS 27.0 or newer on M1 or newer`;
     - the SHA-256 and size of each member;
     - the line `Install with setup (README). A zip downloaded with a browser is quarantined, and macOS refuses its
       engine.`;
     - the text of `--notes FILE`, under "Changes".
10. **Notarise**, with `--notarize`, which needs `--sign`.
    - Run `xcrun notarytool submit <out>/strata-macos-arm64.zip --wait --output-format json <credentials>`.
    - The status must be `Accepted`. Otherwise the tool prints `xcrun notarytool log <id>` and exits 1.
    - There is no stapling: a ticket cannot be stapled to a zip or to a bare executable. Gatekeeper looks it up
      online.
11. **Check.** Run `check_asset.py <out>/strata-macos-arm64.zip --symbols <out>/strata-macos-arm64-symbols.zip --run`
    (section 11.7).

It ends with `make_release: <out>/strata-macos-arm64.zip <sha256>`. Upstream's `.gitignore` already covers its
folders: `build*/` the two build folders, and `/dist/` the output.

### 11.4 Reproducibility

The same commit, built with the same Xcode build and Metal toolchain, must give the same members of
`strata-macos-arm64.zip`. That holds in any folder and on any Mac:

| Source of difference | Measure | Section |
| --- | --- | --- |
| The checkout path in objects, debug information and `__FILE__` | `-ffile-prefix-map`, `-fdebug-compilation-dir=.` | 4.3 |
| The toolchain and SDK paths in AIR | `metal.sh` | 5.3 |
| Object paths and dates in the debug map, which feed the UUID | `-oso_prefix`, `ZERO_AR_DATE=1` | 4.6 |
| The kernel sources in the library | `metal-dsymutil -remove-source` | 5.4 |
| The order of the kernel sources | Sorted glob | 5.3 |
| The paths in `strata-vision` | The three path maps | 9 |
| The signature identifier, taken from the file name | `strip` in place, under the final name | 11.6 |
| The path recorded in the dSYM | `dsymutil strata` run in the build folder | 11.3, step 5 |
| Zip dates, modes, order, umask and Python version | The deterministic zip writer and `SOURCE_DATE_EPOCH` | 11.3, step 8 |
| BUILD.json | No timestamp, fixed formatting | 8.4 |
| A Developer ID signature, which holds a timestamp and the certificate | Compare after `codesign --remove-signature` | 11.5 |

Verified on the reference Mac (appendix A.8), in two checkouts at different paths:
- `strata` is identical unstripped (120,568 B) and stripped (107,576 B), and has the same UUID as its dSYM.
- The dSYM is identical too, but only because both builds used Xcode at the same path: its DWARF names SDK headers
  by absolute path (`/Applications/Xcode.app/.../MacOSX.sdk/usr/include/c++/v1/...`, in lambda names).
- `strata.metallibsym` differs: it records the `metallib` command line, with the build folder and the Metal
  toolchain's mount path.

So the symbols zip is not part of the reproducibility check. Check 11 of section 11.7 ties it to `strata` by UUID
instead.
- `strata.metallib` is identical.
- The stripped `strata-vision` is identical.

The zip writer gave identical zips across mtimes, umasks, and Python 3.9.6 and 3.13.12, all with zlib 1.2.12.

Measured on a Mac that is not the reference Mac (B10, 2026-10-13; D-IMPL-946 and D-IMPL-947): two checkouts at different
paths, with the same commit and content, gave byte-identical asset zips (SHA-256 `866a942b…`). Their symbols zips
differ in `strata.metallibsym` only, which records the build folder, as this section predicts for that file.

**What is compared.** `verify_release.py` compares the members of both zips by SHA-256, not the zip files: another
zlib could deflate differently (measurement B-7). Whether a hosted runner gives the same bytes as the reference Mac
is measurement B-2. Until B-2 confirms it, a difference fails the verification (section 11.8).

### 11.5 Signing and Gatekeeper

**The default is ad hoc.** The arm64 linker signs every executable ad hoc (`flags=0x20002(adhoc,linker-signed)`), and
`strip` signs again in the same way. Nothing else is needed for setup's install:
- `urllib` sets no quarantine attribute, and setup strips quarantine from `engine/` anyway
  ([07](07-server-setup-port.md), section 4.10).
- Gatekeeper only assesses quarantined files.

A zip downloaded with a browser is quarantined, and macOS refuses its ad hoc engine. That path is not supported, and
`NOTES.md` says so.

**Optional Developer ID.** With `--sign "Developer ID Application: <name> (<team>)"`:
- `codesign --force --options runtime --timestamp` signs `strata` and `strata-vision`. They get the hardened
  runtime and a secure timestamp, which notarisation requires.
- No entitlements file. The engine needs no JIT, no unsigned executable memory, no library-validation exception and
  no DYLD environment. It must never be signed with `com.apple.security.cs.allow-jit`,
  `allow-unsigned-executable-memory`, `disable-library-validation` or `allow-dyld-environment-variables`.
- Whether the hardened runtime changes the Metal compile, the pipeline archive or GPU capture is measurement B-5.
  Metal compiles in its own service process, so no change is expected.
- `strata.metallib` is data, not code. It is not signed; the engine's hash check covers it (section 5.6).
- `--notarize` submits the zip. Notarisation time and the behaviour of a quarantined, notarised download are
  measurement B-6.

The identifier is the file name in both cases, `strata` and `strata-vision`. `check_asset.py` requires it.

Whether releases are Developer ID signed is up to the project's owner, who needs an Apple Developer account (open item
in section 14). Until then, releases are ad hoc.

### 11.6 Symbols

- **`strata.dSYM`** holds the engine's DWARF. Its UUID equals the release binary's.
  - A crash report (`~/Library/Logs/DiagnosticReports/strata-*.ips`) lists the binary's UUID and load address.
    `atos -o strata.dSYM/Contents/Resources/DWARF/strata -arch arm64 -l <load address> <address>...` turns its
    frames into function, file and line.
  - Spotlight indexes dSYMs by UUID, so the symbols zip unpacked anywhere on the Mac also lets Xcode and
    `symbolicatecrash` find it.
  - The stall report tells the user where the `.ips` file is ([06](06-engine-host-port.md), section 14.3). A bug
    report attaches it, and the maintainer symbolicates it with the dSYM of that release.
- **`strata.metallibsym`** holds the kernel sources and line tables that step 2 of section 5.4 took out of the
  library. Xcode's GPU debugger and shader profiler can show source lines for the release library when the file is
  imported into a trace of it.
- That both work for a release engine is measurement B-11.
- `strata-vision` has no dSYM: its Release build has no `-g`, as upstream.
- The symbols zip, unlike the asset, names folders of the machine that built it: the SDK path in the dSYM, and the
  build folder and the toolchain's mount path in `strata.metallibsym` (section 11.4). `mac-release.yml` builds in
  the runner's work folder, so no person's home folder appears. A release made by hand with `make_release.py` is
  built in a checkout under a neutral path, such as `/Users/Shared/strata-release`.

**Strip in place, under the final name.** `strip` signs the output again ad hoc with the output's file name as the
identifier:
- `strip -S -x -o strata.s strata` gives `Identifier=strata.s`;
- renaming the file afterwards does not change the identifier;
- a copy whose identifier is not its name breaks byte equality with another build, and fails `check_asset.py`.

### 11.7 `tools/metal/check_asset.py`

```
python3 tools/metal/check_asset.py ASSET.zip [--symbols SYMBOLS.zip] [--run] [--allow-dirty]
```

It runs from a checkout at the release's commit and reads the manifest and `METAL_REV` from that checkout. Every
failure prints `check_asset: <member>: <problem>`, and the run continues so that all problems are listed. The exit
code is 1 if any check failed.

| # | Check |
| --- | --- |
| 1 | The members are exactly those of section 11.2, in that order. Each has `create_system` 3, `S_ISREG`, the mode of the table, and no `/` in its name. There are no folders, symlinks or extra entries. |
| 2 | `lipo -archs` is `arm64` for `strata` and `strata-vision`. |
| 3 | `vtool -show-build` shows `platform MACOS`, `minos 27.0`, for both. |
| 4 | `otool -L` lists only paths under `/System/Library/` and `/usr/lib/`. `otool -l` has no `LC_RPATH`, and no load path starts with `@`. |
| 5 | `codesign -v --strict` passes, and `codesign -dv` shows `Identifier=<file name>`. The signature is ad hoc (`adhoc,linker-signed`) or Developer ID with the hardened runtime (`flags=0x10000(runtime)` and an `Authority=Developer ID Application:` line). `codesign -d --entitlements -` prints no entitlement. |
| 6 | No forbidden string in `strata`, `strata.metallib` or `strata-vision`: `/Users/`, `/private/`, `/tmp/`, `/var/folders/`, `/Volumes/`, `/opt/homebrew`, `/usr/local/`, `/home/`, `/Applications/`, `/Library/Developer/`, and the checkout's own path. The search is on the raw bytes. The only allowed occurrence is `/tmp/perf-metal-%d.gputrace` in `strata-vision` (section 9). The `ALLOWED` table in the script lists it. |
| 7 | `strata` contains exactly one `engine=<v>\n`, with `<v>` equal to BUILD.json's `version` and to `source_version()`. It contains exactly one `metallib_sha256=<64 hex>\n`, equal to the SHA-256 of the `strata.metallib` member and to BUILD.json's `metallib_sha256`. |
| 8 | The defined functions of `strata.metallib` (`xcrun metal-nm`) are exactly the manifest's kernels. `xcrun metal-dsymutil -flat -remove-source` on a copy leaves the copy byte-identical, so no source is left in the library (section 5.7). |
| 9 | BUILD.json has exactly the keys of [07](07-server-setup-port.md), section 4.7.2, plus those of section 8.4. The values: `backend` `metal`, `arch` `arm64`, `macos_min` `"27.0"`, `gpu_families` `["apple7", "metal4"]`, `metal_std` `metal4.0`, `source` `prebuilt`, `lib_dirs` `[]` and `vision` `gpu`. `metal_rev` equals `METAL_REV`, `src` and `vision_src` are 16 hex, and `commit` is 40 hex (`-dirty` only with `--allow-dirty`). The file ends with a newline. |
| 10 | `LICENSES.txt` has the section headers of section 11.9, in order, and no other headers. |
| 11 | With `--symbols`: the symbols zip has the members of section 11.2. The dSYM's UUID equals `strata`'s. `strata.metallibsym` is not empty. |
| 12 | With `--run`: unpack into a temporary folder, setting each mode from `external_attr`. `strata --version` must exit 0 within 30 s and print exactly `strata <v> metal <rev>\n`. `strata-vision` with no argument must exit 2 within 30 s, with stderr starting `usage: strata-vision`. Neither run needs a GPU. |

`check_asset.py` implements the twelve checks. `test_check_asset.py` runs the archive, JSON, string and licence checks on
synthetic assets in the macOS Python suite, with the macOS tools faked. On the asset built on this Mac, every check
passes with `--symbols --run` (D-IMPL-946). The run of check 12 executes the engine and `strata-vision` with no GPU.
No Linux test or release-validation job is part of this repository.

### 11.8 `tools/metal/verify_release.py`

```
python3 tools/metal/verify_release.py TAG [--repo OWNER/NAME] [--no-rebuild]
```

It runs on the reference Mac, from a checkout of the project, in `mac-release.yml`'s verify job (section 12.3). It
works on a draft release, which only users with push access can see, so `gh` needs a token with `contents: write`.
Steps:

1. `gh release download TAG` the three assets into a temporary folder.
2. `shasum -a 256 -c SHA256SUMS`.
3. `git worktree add --detach <tmp>/wt TAG`. Copy the checkout's `third_party/llama.cpp` into it with `cp -cR`, an
   APFS clone: instant, and no extra space.
4. In the worktree: `check_asset.py <asset> --symbols <symbols> --run`.
5. **Install as a user would.** Serve the download folder with `python3 -m http.server --bind 127.0.0.1 <free port>`.
   Then, with the worktree's `setup` imported:
   - set `setup.ROOT` to a fresh temporary folder;
   - call `setup.get_prebuilt_metal("http://127.0.0.1:<port>/")`;
   - call `setup.smoke_test(eng)`, which includes `--list-devices` ([07](07-server-setup-port.md),
     sections 4.7.1 and 4.7.5).

   A draft's assets have no public URL, so the local server stands in for GitHub.
6. The release acceptance runs of [09](09-testing-and-validation.md), section 8, with the installed engine and
   `STRATA_TEST_MODELS`.
7. **Rebuild**, unless `--no-rebuild`. It needs this Mac's `xcode` and `metal_toolchain` values (section 8.4) to
   equal the release BUILD.json's.
   - Run `make_release.py --out <tmp>/rebuild` in the worktree, unsigned. Compare the SHA-256 of every member of
     both `strata-macos-arm64.zip` files. The symbols zips are not compared (section 11.4).
   - For a Developer ID release, first run `codesign --remove-signature` on copies of `strata` and `strata-vision`
     from both builds.
   - When the toolchains differ, print
     `rebuild skipped: the release was built with <xcode>, <toolchain>; this Mac has <xcode>, <toolchain>`.
8. Remove the worktree and the temporary folder. Print `verify_release: TAG ok`, or
   `verify_release: TAG ok (rebuild skipped)`.

Any failure exits 1, and the release stays a draft.

### 11.9 Licences

`LICENSES.txt` is put together from files in the checkout. Each section starts with a header line, `== <title> ==`,
followed by the text and a blank line:

| Title | Text from | Covers |
| --- | --- | --- |
| `Strata` | `LICENSE` (MIT) | Everything in the asset |
| `Strata Metal kernels: adapted code` | `src/metal/kernels/THIRD_PARTY` (section 5.1), as it is | `strata.metallib` |
| `ggml block layouts (third_party/ggml)` | `third_party/ggml/LICENSE` (MIT) | `strata` |
| `llama.cpp and ggml` | `third_party/llama.cpp/LICENSE` (MIT) | `strata-vision` |
| `YaRN RoPE in ggml-metal` | `tools/metal/licenses/yarn-rope.txt`, from ggml's Metal source, embedded in `strata-vision` (MIT, Jeffrey Quesnelle and Bowen Peng) | `strata-vision` |
| `stb_image` | `tools/metal/licenses/stb_image.txt`, from the end of `vendor/stb/stb_image.h` (MIT or public domain) | `strata-vision` |
| `miniaudio` | `tools/metal/licenses/miniaudio.txt`, from the end of `vendor/miniaudio/miniaudio.h` (public domain or MIT No Attribution) | `strata-vision` |
| `subprocess.h` | `tools/metal/licenses/subprocess.txt`, from the top of `vendor/sheredom/subprocess.h` (the Unlicense) | `strata-vision` |
| `SHA-256` | `third_party/llama.cpp/vendor/hash/sha256/LICENSE` (public domain) | `strata-vision` |
| `rotate-bits` | `third_party/llama.cpp/vendor/hash/rotate-bits/LICENSE.md` (MIT) | `strata-vision` |

**Excerpts.**
- Each file in `tools/metal/licenses/` starts with the line `source: <path in llama.cpp>`, followed by the licence
  text as it appears in that file, with comment markers removed.
- `make_release.py` checks that every non-empty line of the excerpt occurs, stripped, in the named file at the pin,
  in order.
- A llama.cpp bump that changes a notice therefore stops the release until the excerpt is updated.

**The vendor check.** `make_release.py` reads `nm -U -j` of the unstripped `strata-vision`: the names of the
defined symbols, mangled C++ names included. It looks for a substring that each vendored library of
`third_party/llama.cpp/vendor/` defines:

| Substring | Library | Result | Defined symbols at the pin |
| --- | --- | --- | ---: |
| `stbi_` | stb_image | Must have its section | 153 |
| `ma_decoder` | miniaudio. Its functions have internal linkage in C++, so they appear mangled, for example `__ZL17ma_decoder_uninitP10ma_decoder`. | Must have its section | 17 |
| `subprocess_` | subprocess.h | Must have its section | 18 |
| `sha256_` | SHA-256 (`sha256.c`, `hash_sha256_hex`), which includes rotate-bits | Must have the SHA-256 and rotate-bits sections | 6 |
| `httplib` | cpp-httplib (MIT) | Fails the release | 0 |
| `nlohmann` | nlohmann/json (MIT) | Fails the release | 0 |
| `XXH` | xxHash (BSD-2-Clause) | Fails the release | 0 |
| `SHA1` | sha1 (public domain) | Fails the release | 0 |

The failure for the last four is
`strata-vision now links <library>; add its license to tools/metal/licenses/ and docs/mac/08 section 11.9`.
A library in the first four rows with no match only prints a note: its section stays until the licence review
removes it. The counts were measured on the release build of section 9 (appendix A.8).

The licence review (work item B20) confirms the table before the first release. Its host part is done: the four
excerpts in `tools/metal/licenses/` were copied from the pinned tree, and on the vision build of this Mac the vendor
check counted 153, 17, 18 and 6 defined symbols for the four vendored libraries of the first rows, and 0 for the four
refused ones (D-IMPL-949). The review itself, by a person, is open.

---

## 12. CI

Upstream has no workflows: its `.github/` holds only `FUNDING.yml`. The project's workflows are maintained here. Compatibility reviews may inspect upstream CI changes but do not
import them automatically. Common rules:
- Every `uses:` names an action by its full commit SHA, with the version in a comment, for example
  `actions/checkout@<40 hex> # v5.0.0`. `.github/dependabot.yml` updates them weekly (`package-ecosystem:
  github-actions`).
- The workflow default is `permissions: contents: read`. A job that needs more asks for it.
- `concurrency: { group: <workflow>-${{ github.ref }}, cancel-in-progress: true }` for ordinary CI.
  `mac-release.yml` and `mac-hardware.yml` use `queue: max` with `cancel-in-progress: false` to serialize without
  replacing pending work. GitHub permits up to 100 pending runs per group; once full, additional runs are canceled.
  For hardware, check Actions, wait for queue capacity, then rerun the missed weekly slice with `workflow_dispatch`
  on `main` and its `day`. For a release, wait for capacity and rerun the canceled tag workflow by run ID. Reruns
  must happen within 30 days of the original run. The hardware dispatch guard validates that the day is 1-7 before
  self-hosted jobs can start.
- Python steps run in a venv created from `requirements.txt`, so `cmake` and `ninja` are the pinned wheels.
- `ci.yml`, `mac-hardware.yml` and `.github/dependabot.yml` are prepared in this local checkout. The configured
  `strata.app` remote points to `uakihir0/strata.app` (D-IMPL-982), but no `strata-ref` runner is configured here;
  these files have not run in GitHub Actions.

### 12.1 `ci.yml`

Triggers: pushes to `main` and `release/**`. Pull requests do not run this workflow because the configurable Mac
runner is not designated as an isolated runner for unreviewed code.

| Job | Runner | Steps |
| --- | --- | --- |
| `mac-host` | `${{ vars.MAC_HOSTED_RUNNER }}`; skipped when the repository variable is empty | Select Xcode 27 and install MetalToolchain; set up Python 3.13 and pinned requirements; run every Python module, source-drift and release-policy checks; fetch pinned llama.cpp; configure and build `mac-ci`; run host CTests; build and upload a release asset when the release builder exists; attempt Metal tests with `continue-on-error` while hosted GPU availability is not a gate. |

The workflow currently has no pull-request test job. Enable one only when a dedicated macOS runner is available that
can safely execute unreviewed PR code without access to secrets or persistent reference data.

`tools/metal/run_python_tests.py` finds `tools/test_*.py`, the unittest modules in `tools/metal/`, and
`serve/test_*.py`. It excludes `tools/metal/test_map.py`, a command-line checker whose unittest coverage is
`tools/metal/test_test_map.py`. Launch the standard-library-only driver with a Python interpreter outside the project
`.venv` (`/usr/bin/python3` in the macOS developer workflow); it starts each module with the project's
`.venv/bin/python` and unittest's standard test loader in its own process, because several modules patch `setup`
globals at import time ([07](07-server-setup-port.md), section 10.1). A missing or broken `.venv` is reported before
discovery, with commands to recreate it. A module that collects zero tests is a failure. It prints a header for each
module and exits 1 when any module fails.

### 12.2 `mac-hardware.yml`

Triggers: `push` to `main`, a nightly `schedule` (`17 3 * * *`), and `workflow_dispatch`. Never `pull_request` or
`pull_request_target` (section 12.4). Every job is restricted to the `main` ref before a runner is selected. The first
step of `hardware` validates the optional weekly day for a manual dispatch. A dispatch from another ref is skipped
by the job condition and never reaches the self-hosted Mac.

The job `hardware` runs on `runs-on: [self-hosted, macOS, ARM64, strata-ref]`, with
`concurrency: { group: strata-ref, queue: max, cancel-in-progress: false }` and `timeout-minutes: 240`:
1. Create the venv from `requirements.txt` and fetch pinned llama.cpp.
2. Configure and build with `mac-dev`.
3. Set `STRATA_METAL_ARCHIVE=$RUNNER_TEMP/strata-cache`, then run `ctest --preset mac-host` and
   `ctest --preset mac-metal`.
4. A manual dispatch with no `day` runs the Python unit layer and hardware checks only. A scheduled run, or a
   manual run with a selected weekly `day`, must have `/Users/Shared/strata-models` and at least one model-labelled
   CTest. It runs `ctest --preset mac-model`, the Metal validation-layer CTest run, builds `strata-vision`, stages
   `strata`, `strata.metallib` and `strata-vision` in `engine/`, runs the Python layer with `STRATA_TEST_MODELS`,
   and then runs `accept.py nightly`. Missing model assets, model CTests, build helpers or the acceptance runner fail
   the gate. The nightly gate stays red until its specified tests and inputs exist. A dispatch with `day` set also
   runs the requested weekly slice after the nightly gate.

A second job, `weekly`, has `needs: hardware`, the same runner labels, `queue: max`, and `timeout-minutes: 600`. On
schedule or a manual dispatch from `main` with a non-empty `day`, it checks out a clean workspace, recreates the venv,
fetches pinned llama.cpp, configures and builds `mac-dev`, reruns the host and Metal CTest presets on that build, then
runs `accept.py weekly`, or `accept.py weekly --day N` for a validated `N` from 1 to 7
([09](09-testing-and-validation.md), section 7.2). It does not depend on both jobs landing on the same self-hosted
machine. Missing model assets or `accept.py` fail the weekly gate.

### 12.3 `mac-release.yml`

`.github/workflows/mac-release.yml` exists in this checkout and has the jobs of the table below (B18). It has not run:
it has not been tried in a test repository, and it must not be enabled or dispatched before that trial and the owner's
decision (D-IMPL-951). `test_workflows.py` checks its triggers, needs chain, environment, permissions and signing rules.

Trigger: `push` of a tag matching `v*-metal.*`. `concurrency: { group: mac-release, queue: max,
cancel-in-progress: false }`.

| Job | Runner | Permissions | Steps |
| --- | --- | --- | --- |
| `build` | `${{ vars.MAC_HOSTED_RUNNER \|\| 'strata-ref' }}` | `contents: write` | Check out the tag with `fetch-depth: 0`, so the tag and release-branch head refs are available. Run `check_release_policy.py --remote` (the tag rule and the growth rule; `GH_TOKEN: ${{ github.token }}`). Then the toolchain, venv and llama.cpp steps. Set up signing if the secrets exist (below). Run `make_release.py`, with `--sign` and `--notarize` when signing is set up, and `--notes` from the annotated tag's message. Run `gh release create "$TAG" --repo "$GITHUB_REPOSITORY" --draft --verify-tag --title "$TAG" --notes-file dist/NOTES.md` with the three files. Finally, delete the keychain (`if: always()`). |
| `verify` | `[self-hosted, macOS, ARM64, strata-ref]`, `needs: build` | `contents: write` (to read the draft) | `verify_release.py "$TAG"` (section 11.8) |
| `publish` | `ubuntu-24.04`, `needs: verify`, `environment: release` | `contents: write` | `gh release edit "$TAG" --repo "$GITHUB_REPOSITORY" --draft=false --latest` |

The `release` environment should require a reviewer, so that a person approves each publication. A failure in
`build` or `verify` leaves a draft or nothing; nothing is published.

If a release run is canceled because the 100-run queue is full, wait for queue capacity and rerun that canceled
workflow run by its run ID within 30 days. The rerun uses the original tag ref and commit, so the normal policy and
release checks apply again.

**Signing in CI**, only when the secret `MAC_CERT_P12_BASE64` is set:
1. Decode the secret into `$RUNNER_TEMP/cert.p12`.
2. `security create-keychain -p "$KC_PW" $RUNNER_TEMP/sign.keychain-db`. `KC_PW` is random per run.
3. `security set-keychain-settings -lut 21600` and `security unlock-keychain` on that keychain.
4. `security import cert.p12 -k <keychain> -P "$MAC_CERT_PASSWORD" -T /usr/bin/codesign`.
5. `security set-key-partition-list -S apple-tool:,apple: -s -k "$KC_PW" <keychain>`.
6. Add the keychain to the user search list.

The identity is the first `Developer ID Application` line of `security find-identity -v -p codesigning <keychain>`.
For notarisation, also set `NOTARY_KEY_P8_BASE64`, `NOTARY_KEY_ID` and `NOTARY_ISSUER`: the App Store Connect API
key, written to `$RUNNER_TEMP/AuthKey.p8` and removed with the keychain. Without the secrets, the release is ad hoc
(section 11.5).

### 12.4 The self-hosted runner `strata-ref`

The reference Mac runs the jobs that need a real GPU or the models. A self-hosted runner executes the workflow's
code with the runner user's rights, so:
- It must never run on `pull_request` or `pull_request_target`. `mac-hardware.yml` is the only current workflow
  that uses it; its triggers are pushes to main, a schedule and manual dispatch. The hardware and weekly jobs require
  the `main` ref, and the first hardware step validates a requested weekly day. Those workflow checks are not a
  security boundary because a manual dispatch can select a ref that supplies the workflow definition.
- Restrict which trusted actors may dispatch `mac-hardware.yml`, and restrict the runner group to this repository and
  the reviewed workflow files pinned to `refs/heads/main`. Only that runner-group policy prevents an unmerged branch
  from changing the workflow before it reaches the self-hosted runner. See GitHub's
  [workflow event](https://docs.github.com/en/actions/reference/events-that-trigger-workflows) and
  [runner-group access](https://docs.github.com/en/actions/how-tos/manage-runners/self-hosted-runners/manage-access)
  documentation.
- The runner runs as a dedicated, non-admin macOS user, with nothing else in its home folder.
- `/Users/Shared/strata-models` is read-only for that user.
- The repository setting "Require approval for all outside collaborators" is on.
- In an organisation, the runner is in a runner group limited to this repository.
- The runner user's `~/Library/Caches/strata` is not shared with an interactive user's engine. Section 10.1 explains
  why.
- Its toolchain is updated on purpose, never automatically. An Xcode update is a change of `xcode` and
  `metal_toolchain`, which skips the rebuild comparison (section 11.8) until the release machine has it too.

---

## 13. Current developer workflow

The setup scripts currently stop with a not-implemented message before changing files. Create a development
environment directly, then configure and test the macOS engine:

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
export PATH="$PWD/.venv/bin:$PATH"
cmake --preset mac-dev
cmake --build --preset mac-dev --parallel
ctest --preset mac-host
ctest --preset mac-metal
/usr/bin/python3 tools/metal/run_python_tests.py
```

CMake fetches the pinned llama.cpp revision automatically. Keep `.venv/bin` first in `PATH` before configuring so
CTest uses the interpreter with the pinned Python dependencies, including NumPy for conversion-payload checks.
`mac-model` includes `dense_vs_gguf_test` and remains fail-closed until `STRATA_TEST_MODELS` contains all seven
completed catalogue packs. The other model acceptance tests
and the T3 quick-hash preflight are still planned. `mac-host` runs tests without a GPU; `mac-metal` requires an Apple
GPU.

- **Editing kernels**: the manifest and `safe/` or `fast/` folder are checked during the build.
- **Debugging kernels**: `mac-dev` keeps shader sources in the library; `strata --metal-capture /tmp/run` writes
  `/tmp/run.gputrace` for Xcode. `MTL_SHADER_VALIDATION=1` and `MTL_DEBUG_LAYER=1` enable Metal validation layers.
- **Optional image validation**: build `strata-vision` with the standalone commands in section 9. This helper is not
  connected to production model inference.

---

## 14. Measurements and open items

Each B item has a provisional default in this document, so implementation does not wait for it. The measurement
replaces the default, or confirms it. Results go to [appendix-a-measurements.md](appendix-a-measurements.md).

| ID | Question | How to measure | Provisional default | Decision it drives |
| --- | --- | --- | --- | --- |
| B-1 | With the full kernel set, is `MTLBinaryArchive` faster than macOS's own shader cache? Check a cold start after an OS update and a warm start. | Time `Context::open` with the archive and with `STRATA_METAL_ARCHIVE=0`. Do it warm; after deleting `$(getconf DARWIN_USER_CACHE_DIR)com.apple.metal`; and in a new user account. | Keep the archive | Whether `pipeline_cache.mm` stays (section 10.1) |
| B-2 | Does a hosted runner build the same bytes as the reference Mac? | `verify_release.py`'s rebuild of a release built by `mac-release.yml` on a hosted runner | Expected yes when `xcode` and `metal_toolchain` match; a difference fails the verification | Whether releases may be built on hosted runners (section 12.3) |
| B-3 | How long does `xcodebuild -downloadComponent MetalToolchain` take on a hosted runner? Do `-exportComponent` and `-importComponent` work as a cache? | Time the step in `mac-host`; export to `actions/cache` and import on a fresh runner | Download on every run | The toolchain step of `mac-host` (section 12.1) |
| B-4 | How long does `strata-vision`'s first start take, while Metal compiles ggml's embedded shaders? Is the result cached across runs and across reboots? | Time the first and the second image request after an install, on M1 and on the reference Mac | Accept the cost; setup does not warm it | Whether setup warms `strata-vision` after the install, or the release ships ggml's library compiled |
| B-5 | Does the hardened runtime change the Metal compile, the pipeline archive or GPU capture? | Sign the skeleton with `--options runtime`; run a first start, a second start and `--metal-capture` | No entitlements needed | Whether `--sign` stays without entitlements (section 11.5) |
| B-6 | Developer ID in practice: signing in CI, notarisation time and result, Gatekeeper on a quarantined download, and member equality after `--remove-signature` | One signed and notarised release in a test repository | Ad hoc releases | The signing decision (open item below) |
| B-7 | Is the zip byte-identical with another Python or zlib, for example on a hosted runner? | The zip writer on both; `cmp` | Compare members, not zips | Whether `verify_release.py` may compare whole zips (section 11.4) |
| B-8 | Asset sizes with the full kernel set | `make_release.py` | None needed | `NOTES.md`; setup's download estimate |
| B-9 | How long does a full local build of the real engine take, on M1 8 GB, M1 16 GB and the reference Mac? | After the Mac installer build command is implemented, time a clean Release build with its selected job count | The text "5-10 minutes" ([07](07-server-setup-port.md), section 4.7.6, step 7) | That text |
| B-10 | How big is a GPU trace of the first decode window with a real model? | `--metal-capture` with Q2_0 on the reference Mac | Warn above 8 GiB of residency | The warning threshold (section 10.2) |
| B-11 | Do the dSYM and the `.metallibsym` symbolicate a release engine? | `kill -ABRT` a running release engine; `atos` on the `.ips` frames; import the `.metallibsym` into a trace of the release library in Xcode | Ship both | The symbols zip (section 11.6) |

Items in other documents that change values here:

| Item | Document | Values here that it changes |
| --- | --- | --- |
| X-8, the hosted runner's GPU | [07](07-server-setup-port.md), section 14 | Whether `mac-host` runs `-L metal` without `continue-on-error` (section 12.1) |
| The release acceptance runs | [09](09-testing-and-validation.md), section 8 | Step 6 of `verify_release.py` (section 11.8) |
| The kernel set and its size | [04](04-metal-kernels.md) | The input of B-1 and B-8; the manifest (section 5.2) |
| Whether the engine moves to the Metal 4 compiler API | [06](06-engine-host-port.md), section 8 | The archive type of section 10.1 |
| H-6, the upstream drift | [06](06-engine-host-port.md), section 15 | How often the compatibility reference advances and the effort per review (section 11.1) |

Open items, decided outside measurement:

| Item | Decided when | Until then |
| --- | --- | --- |
| A release-test repository for B12 and B18 | Before testing release automation | The publication repository is configured as `uakihir0/strata.app`; this identity change publishes no artifact |
| A trusted macOS 27 arm64 CI runner | When the repository owner configures `MAC_HOSTED_RUNNER` | `mac-host` is skipped while unset. PRs do not execute on the configured runner. Hardware acceptance remains on the self-hosted `strata-ref` runner. A binary with `minos 27.0` does not start on macOS 26. |
| Developer ID signing | When the owner has an Apple Developer account, after B-5 and B-6 | Ad hoc releases (section 11.5) |
| The licence review | Before the first release (work item B20) | `LICENSES.txt` as in section 11.9 |
| A `.pkg`, `.dmg` or Homebrew formula | After the first release, if users ask | Setup is the only installer |

---

## 15. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). An item is done when its
"Done when" holds on the reference Mac. B1-B5 refine work item H1 of
[06](06-engine-host-port.md).

| # | Item | Done when |
| --- | --- | --- |
| B1 | Historical root edits A-G and the PC build-graph check (section 3) | Retired by D-IMPL-227. Current validation configures and builds only the `mac-dev` or `mac-release` preset on macOS. |
| B2 | `cmake/metal/backend.cmake` (section 4) | Each requirement check of section 4.1 fails with its message: a configure without the architecture, with target 26.0, and with `DEVELOPER_DIR` at the Command Line Tools; `mac-dev` builds with `-Werror` and no warning |
| B3 | `metallib.cmake`, `metal.sh`, `check_manifest.cmake`, the manifest (section 5) | The table of section 5.5 reproduces. Each rule of section 5.4 has a fixture that fails with its message: a host test that runs `check_manifest.cmake` on AIR files compiled from small broken `.metal` files. `strings -a` finds no absolute path in `strata.metallib`. |
| B4 | `tests.cmake`, `check_engine.cmake` (section 6) | At B4 completion, `mac-host` ran the 19 tests of section 6.4 and `mac-metal` ran its five B4-registered tests (the current six-test suite also includes H4's `metal_self_test`); a bad label fails the configure with its message |
| B5 | `CMakePresets.json` and the proposed `tools/test_metal_build.py` (section 7) | Planned: the macOS Python suite checks the preset, dependency pins and setup/build definitions together |
| B6 | The pipeline cache (section 10.1), with [06](06-engine-host-port.md) H3 | `metal_context_test` passes; the second run's stats line says every pipeline came from the cache; a test shows that pruning keeps a `.tmp.` file younger than 600 s and removes an older one and a foreign archive |
| B7 | GPU capture (section 10.2) | `--metal-capture` writes a trace that Xcode opens; an existing trace exits 2; with a real model above 8 GiB the warning prints |
| B8 | The proposed setup functions in section 8, with [07](07-server-setup-port.md) S8 | Planned: after implementation, the Mac setup test checks source identity and BUILD.json; no PC build behavior is required |
| B9 | `strata-vision` on a Mac (section 9), with [06](06-engine-host-port.md) H13 | `vision-repro.sh` gives identical stripped binaries; checks 2-6 of section 11.7 pass for it |
| B10 | `make_release.py` (section 11.3) | Two runs in checkouts at different paths give the same member hashes in both zips; its `check_asset.py --run` passes |
| B11 | The proposed `check_asset.py` (section 11.7) | Planned: fixture checks run in the macOS Python suite; a valid asset passes macOS architecture, signature and runtime checks |
| B12 | `verify_release.py` (section 11.8) | A draft release in a test repository verifies; the same release with one changed byte in `strata.metallib` fails |
| B13 | `check_release_policy.py` (section 11.1) | A unit test for each row of its table, with `gh` mocked |
| B14 | `run_python_tests.py` (section 12.1) | It runs every module in its own process; a failing or empty module gives exit 1 and names the module |
| B15 | `compare_build_graphs.py` (section 3.4) | Retired by D-IMPL-227 with the non-Mac build graph. |
| B16 | `ci.yml` (section 12.1) | Every enabled Mac CI job is green on main and release branches; no pull request code reaches trusted Mac runners. |
| B17 | `mac-hardware.yml` (section 12.2) | A push to main runs the host and metal tests on `strata-ref`; the nightly run includes the model tests |
| B18 | `mac-release.yml` (section 12.3) | A test tag in a test repository gives a draft, verifies it and publishes it after approval; a failing verify leaves only the draft |
| B19 | Action pinning and `dependabot.yml` (section 12) | Every `uses:` names a 40-hex SHA; Dependabot opens its first update |
| B20 | Licence review (section 11.9) | The table of section 11.9 is confirmed; the excerpt check and the vendor check run in `make_release.py`, and each fails on a fixture |
| B21 | The `strata-ref` runner (section 12.4) | Every rule of section 12.4 holds, checked by a second person |
| B22 | Measurements B-1 to B-11 (section 14) | Each result is in appendix A, and the default it drives is updated here |

**Status after the host work of 2026-10-13** (decisions D-IMPL-940 to D-IMPL-959). An item is done only when its
"Done when" holds on the reference Mac, so each line says what is host-verified and what stays open.

- **B8:** host part done. `setup.py` has the section 8 functions and the release helpers, with `tools/test_setup_metal.py`.
  Open: the build flow that calls `cmake_build` and `metal_build_json` (07 S8), and the Mac setup test that B5's module names.
- **B9:** host part done. `tools/vision/CMakeLists.txt` has the release recipe, and two builds on a Mac that is not the
  reference Mac are identical. Open: the reference Mac's two builds and the spec's figures.
- **B10:** host part done. `tools/metal/make_release.py` builds and checks the asset; two checkouts at different paths
  give identical asset zips. Open: the reference Mac's comparison; the symbols zip differs in `strata.metallibsym`
  (D-IMPL-947); Developer ID signing and notarisation are untested.
- **B11:** host part done. `tools/metal/check_asset.py` runs checks 1 to 12, with fixtures in the macOS suite, and a
  real asset passes. Open: a Developer ID asset, which has not been built.
- **B12:** host part done. `tools/metal/verify_release.py` runs steps 1 to 4, 7 and 8. Open: a draft verified in a test
  repository, and step 6 (09 T15).
- **B13:** done as a host test, with every row of section 11.1 covered by `test_check_release_policy.py`.
- **B18:** host part done. `.github/workflows/mac-release.yml` is written and checked locally, and has not run. Open: the
  trial in a test repository and the `release` environment's reviewer.
- **B20:** host part done. The excerpts and the vendor check run in make_release. Open: the licence review by a person.
- **B22:** not started. Its measurements need the reference Mac and the models (appendix A).
