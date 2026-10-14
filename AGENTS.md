# AGENTS.md

## Project

Strata is a source-derived project targeting Apple Silicon Macs running macOS 27 or later. Metal is
the only graphics backend. The native engine and host code are in `src/metal/` and
`include/strata/metal/`; Metal shaders are in `src/metal/kernels/`; CMake integration is in
`cmake/metal/`. The repository also contains Python setup and server code, but the macOS product path
is still being developed.

## Product status

Treat inference and setup as work in progress. The production Prefill and Window path is not fully
wired, and the current command-line program rejects model inference options. `setup.sh` verifies Apple silicon and
macOS 27 or newer, then exits before creating an environment or changing files. `setup.py --check-host` runs
read-only host probes; full `--check`, installation and inference options remain fail-closed.
Do not describe the repository as a working end-to-end model runner or present planned setup and server behavior as
implemented. The status and specifications are in `docs/mac/README.md`,
`docs/mac/06-engine-host-port.md`, and `docs/mac/07-server-setup-port.md`.

## Build and test entry points

Use the project presets in `CMakePresets.json`. The `mac-dev` preset targets arm64 and macOS 27,
enables Metal, and builds development tests. The presets require CMake 3.24 or later and use Ninja.
The documented developer commands are:

```sh
cmake --preset mac-dev
cmake --build --preset mac-dev
ctest --preset mac-host
ctest --preset mac-metal
python3 tools/metal/run_python_tests.py
```

`mac-host` selects tests that do not need a GPU; `mac-metal` requires an Apple GPU. The `mac-model`
preset fails closed until model-backed CTests are registered. Follow
[docs/mac/08-build-and-distribution.md](docs/mac/08-build-and-distribution.md) and
[docs/mac/09-testing-and-validation.md](docs/mac/09-testing-and-validation.md) for the required
archive configuration, toolchain, and test inputs. Do not add build or test commands to project
documentation unless they are present in the presets or documented workflow.

## Documentation and changes

Keep user-facing and engineering documentation in sync with the current implementation. Mark planned
behavior as planned or in progress, and do not turn kernel-level or synthetic test results into claims
of end-to-end inference. Report measured results with the Mac model, macOS and Xcode versions, model
inputs, and test conditions.

Preserve source notices and license files. This project includes source imported from
Niko1221/Strata v0.1.34, commit `1678de33d0e0711bc414ad992b640e1a37dd814`; see
[docs/PROJECT_ORIGIN.md](docs/PROJECT_ORIGIN.md) for the provenance record.
