# Strata

Strata is a source-derived project for a native Metal engine on Apple Silicon. Its only supported product
and build target is macOS 27 or later. Windows and Linux builds and runtime paths are not supported; Metal is the
only graphics backend.

## Status

This repository is under active development. It contains C++ and Objective-C++ engine and host code,
Metal kernels, macOS-only CMake build files, and host, Metal and model test targets. The registered
`dense_vs_gguf_test` needs all seven catalogue packs under `STRATA_TEST_MODELS`; other model acceptance
tests are not implemented yet. It is not a ready-to-run inference product.

Production inference is incomplete. The Prefill and Window production path is not fully connected, and
the current command-line program rejects model inference options. The macOS setup and server
integration also remain in progress. `./setup.sh` currently checks for Apple silicon and macOS 27 or newer,
then exits with a clear not-implemented status before creating an environment or changing files. It is not an
installer. CUDA kernels and parity harnesses are reference-only and are not built into a Mac backend. A successful build or a
host or kernel test does not establish that end-to-end model inference is available. See the current status and design notes in
[docs/mac](docs/mac/README.md), especially the [engine port](docs/mac/06-engine-host-port.md) and
[setup and server port](docs/mac/07-server-setup-port.md).

An isolated native-weight Prefill coordinator now composes each supported
quantized-weight dequantizer with the F16 GEMM in one ordered encoder segment.
Synthetic tests cover the component chain and rollback; production
`Prefill::run` still does not call it.

One Prefill GR read half now composes normalization, down projection, SiLU,
up projection, injection and four-stream mixing. Serial and concurrent
synthetic tests cover all BF16X2 modes; layer scheduling, MoE integration and
production chunk execution are still unimplemented.

A synthetic Prefill MoE path now covers stable grouping and row gathering,
gate/up SIMD-group tiles for all eight supported formats, both down formats,
and the shared SwiGLU/combine post-ops. These are isolated components with
independent synthetic references; the production MoE schedule and
`Prefill::run` integration are still in progress.

The full-mode Prefill MoE router, divide-mode top-10 selection, grouping and
F16 row gather are now composed as one transactional steps 1–6 segment for
BF16X2 modes 1 and 2, with synthetic serial/concurrent tests and cross-stage
alias checks. The shared-expert FFN stages 7–10 also have a
caller-owned/transactional coordinator, verified with synthetic Q2_0 and
BF16X2 references. These two coordinators can be ordered by a caller when
every writable active view in either coordinator is disjoint from every active
view in the other; routed expert tile consumers and combine, tiered scheduling
and production `Prefill::run` integration remain pending.

## Build and test entry points

The CMake presets target arm64 and macOS 27. The `mac-dev` preset enables the Metal backend and
development tests. The presets require CMake 3.24 or later and use Ninja. With the Apple toolchain
available, the documented build entry points are:

```sh
cmake --preset mac-dev
cmake --build --preset mac-dev
```

The documented test entry points are:

```sh
ctest --preset mac-host
ctest --preset mac-metal
.venv/bin/python tools/metal/run_python_tests.py
```

`mac-host` runs tests that do not need a GPU; `mac-metal` needs an Apple GPU. The `mac-model` preset runs
registered model tests and fails closed until `STRATA_TEST_MODELS` contains all seven completed catalogue packs.
See the [developer workflow](docs/mac/08-build-and-distribution.md) for toolchain and archive requirements, and the
[test plan](docs/mac/09-testing-and-validation.md) for model test inputs and remaining acceptance work. These are
development commands, not an end-user setup procedure.

## Source and documentation

- `src/metal/` and `include/strata/metal/` contain the native Metal host and engine code.
- `src/metal/kernels/` contains Metal shaders and their kernel manifest.
- `cmake/metal/` contains Metal build and test integration.
- `tests/metal/` contains Metal-specific tests and fixtures.
- `docs/mac/` documents the macOS design, implementation status, build, and validation work.

## Name, origin, and license

The project name is Strata. This checkout is maintained separately rather than as a GitHub fork, and
it currently has no configured Git remote. Its source includes material imported from
Niko1221/Strata v0.1.34, commit `1678de33d0e0711bc414ad992b640e1a37dd814`. Separate maintenance or
Git history does not make that source clean-room work or remove license and attribution obligations.
See [docs/PROJECT_ORIGIN.md](docs/PROJECT_ORIGIN.md), the root [LICENSE](LICENSE), and the
applicable third-party notices.
