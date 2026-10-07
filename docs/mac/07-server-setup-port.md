# 07. Server, setup and product layer on macOS

Status: design specification for the unfinished macOS setup and server product, 2026-10-07. The current launcher
behavior is documented in section 5.1; the planned install flow elsewhere in this document is not implemented.
Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) v0.1.34, commit `1678de33`.
Source audit: [audit/C-python-setup-serve.md](audit/C-python-setup-serve.md). It covers every Python file, launcher, test
and user document, and records probes run on the reference Mac.
Reference code: [spikes/serve-macos/macos.py](spikes/serve-macos/macos.py), verified on the reference Mac (M5 Pro
64 GB, macOS 27.0) with `/usr/bin/python3` 3.9.6 and Homebrew Python 3.13.12.
Every `file:line` reference is relative to the upstream repository root.

This document specifies the planned Mac user experience outside the engine process. The current checkout has no
working installer or end-to-end inference product; its fail-closed behavior is stated explicitly in section 5.1.
- `setup.sh` and the double-click launchers;
- `setup.py`: install, `--check`, start and update;
- `serve/server.py`, the HTTP server that owns the engine process;
- `serve/telemetry.py` and the web app's Monitor tab;
- `tools/strata_mcp.py`, the MCP server for AI assistants;
- the retained MTP and model-pack tools; the inherited PC CUDA calibration utility was removed (D-IMPL-236);
- the Python tests, the user documents, the Python dependencies and Docker.

Other documents cover the rest:
- [06-engine-host-port.md](06-engine-host-port.md): the engine binary and its serve protocol.
- [05-memory-and-residency.md](05-memory-and-residency.md): the memory plan that decides whether a model fits.
- [08-build-and-distribution.md](08-build-and-distribution.md): building, signing and publishing the engine asset.
- [09-testing-and-validation.md](09-testing-and-validation.md): the test plan.

This document supersedes audit C section 2, which proposed a Python `metal_fit` formula. On a Mac the engine makes the
fit decision (principle P3).

Normative words: **must** means required for a conforming port. **Should** means the recommended default, which a
measurement or review may overturn.

---

## 1. Scope and principles

### 1.1 Current status and target behavior

| Area | Current checkout | Planned Mac product |
| --- | --- | --- |
| First run | `./setup.sh` checks Darwin, Apple silicon and macOS 27 or newer, then exits 2 before changing files. | Install a native arm64 Metal engine, Python environment, model pack and launch script. |
| Model selection | No Mac installer or usable model planner is exposed. | Ask the engine's Metal plan whether each model and context fits. |
| Server | API and Monitor scaffolding exist; production inference is incomplete. | Keep the OpenAI and Anthropic APIs, web app and MCP available with Metal telemetry. |
| Start scripts | No `.command` launcher or model start script is shipped. | Add Finder launchers after installation is implemented. |
| Windows/Linux | Not supported by this repository. | No Windows or Linux product path. |

The detailed sections below specify the target behavior. They do not imply that the setup, model selection or
production inference flows already work.

Current source behavior:
- `setup.sh`, `setup.py`, and the MCP installer entrypoint fail closed; they do not create an environment, write settings,
  download models, or expose a setup command. The setup planner is not available.
- The HTTP server and MCP entrypoint reject unsupported hosts before loading configuration. They require native Apple
  silicon, macOS 27 or newer, and Python 3.10 or newer.
- MCP hardware output describes the Mac, its memory, and one Metal device. Install recommendations remain informational
  and cannot claim that a model fits until the Metal planner is implemented.
- The C++ engine and production inference are still incomplete. Passing the Metal unit tests does not mean the model can
  be installed or served end to end.

### 1.2 Principles

| # | Principle | Consequence |
| --- | --- | --- |
| P1 | The product supports Apple Silicon on macOS only. | Setup, the engine build and the server fail clearly on unsupported operating systems or architectures. |
| P2 | The macOS product builds and runs Metal only. | There is no CUDA/HIP build, installer, or runtime path. A limited set of CUDA sources remains as numerical reference material for Metal parity checks; it is not a supported backend. |
| P3 | The fit logic exists once, in the engine. Python never computes whether a model fits. It runs `strata --plan-only` and reads the `PLAN` line. | Setup's menus, `--check`, `start()` and the MCP's `recommend` all call `metal_plan()` (section 4.4). No Python formula duplicates [05](05-memory-and-residency.md). |
| P4 | Probes use the standard library only: `ctypes`, `plistlib`, and `subprocess` with absolute tool paths (`/usr/sbin/ioreg`, `/usr/bin/xattr`). No pyobjc. `psutil` (already pinned in `requirements.txt`) is required on a Mac. | `serve/macos.py` (section 3) |
| P5 | The serve protocol and the HTTP APIs stay byte-compatible with upstream. | The server needs no protocol changes. New INFO keys are additive ([06](06-engine-host-port.md), section 5.2). |
| P6 | Setup never uses sudo and never installs system software. | It prints the command (Homebrew, Xcode, `sysctl`), then stops or goes on. |
| P7 | Setup recommends and never forces (upstream #406). `--yes` alone accepts the recommendations and keeps every stop. An explicit choice plus `--yes` is consent to the risk it carries. | `confirm_risk(..., explicit=bool(a.<flag>), yes=a.yes)`, exactly as upstream (`setup.py:1955`) |
| P8 | New code goes into new files where possible (`serve/macos.py`, `tools/make_geometry.py`, `data/geometry/`, new tests). An inline branch in an upstream file starts with a `# MAC:` comment. | `tools/metal/check_drift.py` ([06](06-engine-host-port.md), section 1.3, and [10-implementation-plan.md](10-implementation-plan.md), section 4) counts the markers per file to scope each upstream compatibility review. |
| P9 | Unsupported operating systems and backends have no product, setup or CI path in this repository. | Platform guards and the Mac-only build and test workflows |

### 1.3 Files touched

| File | Change | Size |
| --- | --- | ---: |
| `serve/macos.py` | New: every macOS probe (section 3) | ~380 lines |
| `setup.py` | MAC branches in the constants, probes, main's steps, `start()`, updates and the data folder. New functions `metal_plan`, `plan_class`, `get_prebuilt_metal`, `build_engine_metal`, `write_command_script`, `mac_step1`, `mac_check`. | ~600 lines |
| `setup.sh` | A darwin block that finds Python (section 5.1) | ~45 lines |
| `START-HERE.command`, `SETUP.command` | New double-click launchers (section 5.2) | 3 lines each |
| `.gitattributes` | `*.command text eol=lf` | 1 line |
| `serve/server.py` | Port probe, SIGHUP, firewall text, death note, child environment, free-memory check, RAM warning, `keep_awake` | ~110 lines |
| `serve/telemetry.py` | `_Apple` reader, Mac CPU name, Mac series | ~90 lines |
| `serve/web/app.js`, `serve/web/index.html` | Mac Monitor cards and texts (section 7.3) | ~90 lines |
| `serve/mcp.py` | Homebrew directories on the MCP hub's PATH (section 6.10) | ~5 lines |
| `tools/strata_mcp.py` | Process identity, system folders, RAM, `recommend`, texts (section 8) | ~130 lines |
| PC CUDA calibration utility | Removed from the macOS-only tree (D-IMPL-236); no Mac calibration command is shipped | — |
| `tools/make_geometry.py`, `data/geometry/*.txt`, `data/geometry/*.check` | New: plans before any download (section 4.4) | ~80 lines and 14 small files |
| Tests | Golden fix, AMD zip test fix, one `setUpModule` line in 13 setup test modules, a Mac golden set with `tools/record_mac_plans.py`, `serve/test_macos.py` (section 10) | ~700 lines |
| `docs/*.md`, `README.md` | Mac sections (section 11) | |

---

## 2. Inherited PC setup assumptions and the Mac port design

This is a historical analysis of the upstream v0.1.34 PC-oriented setup path. It is not a description of the current
`setup.sh`, which fails closed until a Mac installer is implemented. The 14 points below identify why the inherited
flow could not be used on a Mac and where the port specification addresses each issue:

| # | Upstream location | Inherited behavior that breaks on a Mac | Mac port design |
| --- | --- | --- | --- |
| 1 | `setup.sh:8`, `:15-33` | `/usr/bin/python3` is 3.9.6. None of apt, dnf or pacman exists, so `setup.sh` prints the Ubuntu hint and exits 1. | Section 5.1 |
| 2 | `setup.py:2685-2719` | Step 1 finds no NVIDIA or AMD card and fails "no NVIDIA GPU found" | Section 4.3, `mac_step1` |
| 3 | `setup.py:2737`, `:271-277` | `ram_gb()` opens `/proc/meminfo` and crashes | Section 4.2 |
| 4 | `setup.py:2738`, `:290-310`, `:2760-2762` | `cpu_info()` returns ("arm", False, False), then "this CPU has no AVX2" | Section 4.2 |
| 5 | `setup.py:93`, `:1509-1590` | The prebuilt asset is `strata-linux-x64.zip` on every non-Windows host. The arch check compares CUDA `sm` numbers. | Sections 4.1 and 4.7 |
| 6 | `setup.py:3020-3021` | `pip_install(CUDA_WHEELS)`: the nvidia-* wheels have no macOS builds, so pip fails | Section 4.7 |
| 7 | `setup.py:1673`, `:1803` | The local build uses winget or apt, `find_nvcc()` and `-DSTRATA_ENABLE_CUDA=ON` | Section 4.7, `build_engine_metal` |
| 8 | `setup.py:3141-3166` | 64K+ context adds `--kv-resident 32768`; vision adds `--vram-reserve-mib 700` | Section 4.9 |
| 9 | `setup.py:3214`, `:3221-3225` | Calibration is offered (it is "NVIDIA-only for now" only by a `hip` test) | Section 4.16 |
| 10 | `setup.py:2353-2381`, `:2393-2394` | `start()` probes `nvidia-smi`, may rebuild for CUDA (`ensure_engine_for`), and says the model "locks part of it for the GPU" | Section 4.11 |
| 11 | `serve/server.py:652-667` | `child_env` sets `LD_LIBRARY_PATH`, which dyld ignores | Section 6.6 |
| 12 | `serve/server.py:2442-2446` | The port probe misses a conflicting listener (BSD `SO_REUSEADDR` rules) | Section 6.1 |
| 13 | `serve/telemetry.py` | No NVML, so the Monitor shows no GPU at all | Section 7 |
| 14 | `tools/strata_mcp.py:139-142` | `proc_identity` falls back to `os.kill(pid, 0)`. A zombie counts as running, and a recycled PID counts as ours. | Section 8.1 |

Everything else in the Python layer runs unchanged on a Mac. That covers the OpenAI and Anthropic API layer,
`serve/frontend.py`, `serve/structured.py`, the tokeniser, `chat.py`, `tools/needle_bench.py`, the web chat,
`serve/web/monitor.js`, the mock engine and the HTTP tests. On the reference Mac, 238 of the 242 non-setup tests pass.
The 4 failures are in `tools/test_strata_mcp.py` and come from failure point 14 and the system-folder bug (section 8.1).

---

## 3. `serve/macos.py`

One new module holds every macOS probe. The reference implementation is
[spikes/serve-macos/macos.py](spikes/serve-macos/macos.py), which must be copied to `serve/macos.py` as the
starting point.

### 3.1 Rules

- **Standard library only** (P4). It must import and run under Python 3.9, so `python3 serve/macos.py` works as a
  diagnostic with the system Python. Use `from __future__ import annotations`, and no `match` statements.
- **Safe on every OS.** Off macOS, every function returns its documented empty value (`None`, `0`, `()`, `{}` or `[]`).
  A failed probe returns the same empty value. **No function raises.**
- **Absolute tool paths.** It calls `/usr/sbin/ioreg`, `/usr/bin/xattr`, `/usr/bin/tmutil`, `/usr/bin/caffeinate`,
  `/usr/bin/pgrep`, `/usr/bin/xcode-select`, `/usr/bin/xcrun` and
  `/usr/libexec/ApplicationFirewall/socketfilterfw`. A Homebrew GNU tool earlier on PATH must not change the result.
- **Static facts are cached** (`functools.lru_cache`): `metal_devices()`, `macos_version()` and the library handles.
  Dynamic readings (pressure, swap, ioreg statistics, thermal state) are never cached in the module. Callers decide
  how often to sample.
- **`objc_msgSend` is cast to the exact signature of each call.** On arm64 it is not variadic, so a call through the
  default prototype returns garbage.
- Importers: `setup.py` (`sys.path.insert(0, str(ROOT))`, then `from serve import macos`), `serve/server.py` and
  `serve/telemetry.py` (`from serve import macos`, or `import macos` when they run as scripts from `serve/`), and
  `tools/strata_mcp.py` (by path, as it already imports setup).

### 3.2 API

| Function | Returns | Source | Used by |
| --- | --- | --- | --- |
| `MAC` | `sys.platform == "darwin"` | | everyone |
| `sysctl_raw(name)` | bytes, or None | `sysctlbyname`, called twice (size, then value) | the helpers below |
| `sysctl_int(name, default=None)` | int. A 4-byte value is read as `<i`, an 8-byte value as `<Q`. | | |
| `sysctl_str(name, default=None)` | str up to the first NUL | | |
| `ram_bytes()` | physical RAM in bytes, 0 off macOS | `hw.memsize` | setup, server, telemetry, MCP |
| `macos_version()` | `(27, 0)`, or `()` | `kern.osproductversion`, then `platform.mac_ver()` (which can report a compatibility version such as 16.0 to a Python built against an old SDK) | setup step 1, `metal_devices` |
| `translated()` | True under Rosetta 2 | `sysctl.proc_translated` | setup step 1, `setup.sh` (section 5.1) |
| `cpu_brand()` | `"Apple M5 Pro"` | `machdep.cpu.brand_string` | setup, telemetry, MCP |
| `perf_levels()` | `[{"name": "Super", "cores": 6}, {"name": "Performance", "cores": 12}]`. Level 0 is the fastest class. | `hw.nperflevels`, `hw.perflevelN.name`, `hw.perflevelN.physicalcpu` | setup's CPU line, telemetry, Monitor |
| `fast_cores()` | the core count of perflevel0 | as above. Without perflevels it falls back to `os.cpu_count() // 2`. | vision CPU threads (section 4.9) |
| `arm_features()` | `{"DotProd", "I8MM", "BF16", "SME"}` → bool | `hw.optional.arm.FEAT_*` | diagnostics only |
| `pressure_level()` | `"normal"`, `"warn"`, `"critical"` or None | `kern.memorystatus_vm_pressure_level` (1, 2, 4) | telemetry, server |
| `swap_used()` | bytes, or None | `vm.swapusage` (`struct xsw_usage`: total, avail, used as `<QQQ`, then pagesize, encrypted) | telemetry |
| `wired_limit()` | bytes; 0 = the default | `iogpu.wired_limit_mb` × 2^20 | `metal_devices`, setup's advice (section 4.5.6) |
| `metal_devices()` | `(dev,)` or `()` | Metal via ctypes (section 3.3), and `ioreg_gpu()` for the core count | setup, MCP |
| `device_problem(dev)` | why the Metal engine cannot run on this device, or None | `unified`, `families` ⊇ `REQUIRED_FAMILIES` (`apple7`, `metal4`) | setup step 1, MCP |
| `ioreg_gpu()` | `{"model", "cores", "util", "renderer", "tiler", "in_use", "alloc", "recovery"}` or `{}` | `/usr/sbin/ioreg -r -d 1 -w 0 -c IOAccelerator -a`, parsed with `plistlib`: `model`, `gpu-core-count`, and `PerformanceStatistics` (`Device Utilization %`, `Renderer Utilization %`, `Tiler Utilization %`, `In use system memory`, `Alloc system memory`, `recoveryCount`) | telemetry, server |
| `thermal_state()` | `"nominal"`, `"fair"`, `"serious"`, `"critical"` or None | `NSProcessInfo.processInfo.thermalState` | telemetry, Monitor |
| `low_power_mode()` | bool or None | `NSProcessInfo.processInfo.isLowPowerModeEnabled` | setup step 1, telemetry |
| `proc_identity(pid)` | `"mac:<start sec>.<start usec>"`, or None for a zombie or a gone process | libproc `proc_pidinfo(pid, PROC_PIDTBSDINFO = 3, 0, &info, 136)`. It returns 136 for a live process and 0 for a zombie or a reaped one. | MCP (section 8.1) |
| `strip_quarantine(*paths)` | None | `/usr/bin/xattr -dr com.apple.quarantine <path>`. It leaves `com.apple.provenance` alone. | setup (sections 4.7 and 4.10) |
| `has_quarantine(path)` | bool | `/usr/bin/xattr -p com.apple.quarantine <path>` | setup's smoke-test hint |
| `tm_exclude(path)` | bool | `/usr/bin/tmutil addexclusion <path>` (a sticky xattr, no administrator rights) | setup's data folder (section 4.12) |
| `firewall_on()` | bool, or None when unknown | `socketfilterfw --getglobalstate` | server (section 6.3) |
| `keep_awake(pid=None)` | a `Popen`, or None | `/usr/bin/caffeinate -i -w <pid>` | setup (section 4.13), server |
| `ssl_setup()` | the CA bundle path it set, or None | Sets `SSL_CERT_FILE` to `certifi.where()`, else `pip._vendor.certifi.where()`. It acts only when `ssl.get_default_verify_paths().cafile` is None and `SSL_CERT_FILE` is unset. | setup's start (section 4.7.4) |
| `engine_pids(name="strata")` | PIDs of processes with exactly this name | `/usr/bin/pgrep -x`. pgrep excludes its own ancestors, so a setup or server never matches itself. | `start()` (section 4.11) |
| `xcode_problem()` | why a local build cannot run, or None | `/usr/bin/xcode-select -p` must point into an `.app`, and `/usr/bin/xcrun -f metal` must succeed | `build_engine_metal` (section 4.7.6) |

### 3.3 The Metal device dict

`metal_devices()` loads `/System/Library/Frameworks/Metal.framework/Metal` and calls `MTLCreateSystemDefaultDevice`.
It returns one dict with the keys setup already uses for an NVIDIA card (`index`, `name`, `vram_gb`, `arch`,
`driver`), plus the Metal facts:

| Key | Value on the reference Mac | Source |
| --- | --- | --- |
| `index` | 0 | a Mac has one GPU |
| `name` | `Apple M5 Pro` | `device.name` |
| `working_set` | 55,662,788,608 | `recommendedMaxWorkingSetSize` (W) |
| `wired_limit` | 0 | `iogpu.wired_limit_mb` |
| `max_buffer` | 41,747,087,360 | `maxBufferLength` |
| `unified` | True | `hasUnifiedMemory` |
| `families` | `apple7`, `apple8`, `apple9`, `apple10`, `metal3`, `metal4` | `supportsFamily:` with 1007-1010 and 5001-5002 |
| `cores` | 20 | ioreg `gpu-core-count` |
| `ram` | 68,719,476,736 | `hw.memsize` |
| `vram_gb` | 55.66 | W / 10^9, **for display only** |
| `arch` | `apple` | constant |
| `driver` | `27.0` | the macOS version |

No fit decision may use `vram_gb` or `working_set` (P3). They exist only so that upstream code paths that print a
card can print the Mac. The engine computes W itself and also honours `--gpu-budget-mib`.

### 3.4 Measured on the reference Mac

`python3 serve/macos.py` prints every probe as JSON. It took 0.118 s on `/usr/bin/python3` 3.9.6, with identical
output on Homebrew 3.13.12. A cached `metal_devices()` call costs 0.06 ms, and an `ioreg_gpu()` call 25-80 ms.

| Probe | Result |
| --- | --- |
| `macos_version()` | `(27, 0)` |
| `translated()` | False |
| `perf_levels()` | Super × 6, Performance × 12 |
| `arm_features()` | all four True |
| `pressure_level()`, `thermal_state()`, `low_power_mode()` | `normal`, `nominal`, False |
| `swap_used()` | 12,837,453,824 (macOS swaps even at normal pressure, so swap alone is not an alarm) |
| `ioreg_gpu()` | model `Apple M5 Pro`, 20 cores, util 0, `in_use` 1,057,800,192, `alloc` 7,127,236,608, recovery 0 |
| `proc_identity(os.getpid())` | `mac:1790931492.543160` |
| `proc_identity` of an unreaped child, and of a reaped one | None, None |
| `firewall_on()` | True |
| `xcode_problem()` | None (Xcode 27 with the Metal toolchain) |
| `engine_pids("sleep")` with sleeps running | their PIDs. pgrep excludes its own ancestors (launchd), so `engine_pids("launchd")` returns `[]`. |
| `keep_awake()` | a running `caffeinate` that ends with the process |

---

## 4. `setup.py`

### 4.1 Constants

`MAC = sys.platform == "darwin"` is defined right after `WIN` (`setup.py:60`). Then `sys.path.insert(0, str(ROOT))`
and `from serve import macos`. This import runs on every OS: the module is standard library only and never raises
(section 3.1), and `serve/__init__.py` imports nothing.

| Constant | PC (unchanged) | Mac |
| --- | --- | --- |
| `PREBUILT_ASSET` (`setup.py:93`) | `strata-windows-x64.zip` / `strata-linux-x64.zip` | `strata-macos-arm64.zip` |
| `PREBUILT_URL` (`setup.py:91`) | `https://github.com/Niko1221/Strata/releases/latest/download/` | `https://github.com/{REPOSITORY}/releases/latest/download/` |
| `PREBUILT_TAG_URL` (`setup.py:92`) | `.../releases/download/v{version}/` | `https://github.com/{REPOSITORY}/releases/download/v{version}-metal.{metal_rev}/` |
| `CUDA_WHEELS`, `MIN_DRIVER` | as upstream | never read on a Mac |
| `MIN_ENGINE` | `(0, 1, 34)` | the same value. The project's engine version is the upstream version selected for its current compatibility baseline. |
| `MIN_ENGINE_METAL` | | int, the lowest `metal_rev` this setup works with ([06](06-engine-host-port.md), section 5.2). Starts at 1. |
| `MACOS_MIN` | | `(27, 0)` |
| `MAC_RAM_FLOOR_GB` | | `32`. Below this, step 1 asks (section 4.3). |
| `METAL_SHARE_OK`, `METAL_SHARE_SLOW` | | `0.85`, `0.40`: the resident share of experts that separates "ok", "slow" and "very slow" (section 4.4). Provisional until measurement M-9 ([05](05-memory-and-residency.md), section 10) and gate G2. |
| `PLAN_TIMEOUT_S` | | `30` |
| `VISION_METAL` | | `{"gpu": {"max_tokens": 1024, "reserve_mib": 2100}, "cpu": {"max_tokens": 300, "reserve_mib": 700}}` |
| `METALLIB` | | `"strata.metallib"` |
| `GEOMETRY_DIR` | | `ROOT / "data" / "geometry"` |

`{REPOSITORY}` is the project's eventual GitHub `owner/repository`. Configure the module constant
`RELEASE_REPOSITORY` only after a publication destination is selected; the current checkout has no remote. The
`STRATA_PREBUILT_URL` environment variable and `--prebuilt` option override the default `PREBUILT_URL`.

The public repository slug may be `strata.app`; its owner and destination are not configured ([11](11-risks-and-open-questions.md), OQ-01).
Only the repository takes that name: the product stays Strata, and so do its binaries, files and folders (`strata`,
`strata.metallib`, `strata-macos-arm64.zip`, `Strata-data`, `~/Library/Caches/strata`) and its texts.
macOS treats a folder whose name ends in `.app` as an application bundle (content type
`com.apple.application-bundle`, even when the folder is empty). A plain `git clone` would make such a folder: Finder
would show it as an application that cannot open, with `START-HERE.command` hidden inside it. The user documents
therefore clone into a folder named `Strata` (section 11). A ZIP downloaded from GitHub unpacks into
`strata.app-main`, an ordinary folder.

**The release contract.** [08-build-and-distribution.md](08-build-and-distribution.md) must publish to match it:
- `{version}` is `source_version()` (`setup.py:2209`), the CMake project version. The project keeps it equal to the
  upstream version selected for the current compatibility baseline.
- `{metal_rev}` is the integer in the one-line file `METAL_REV` at the repository root. CMake reads the same file, and
  the engine reports it in `--version` and INFO ([06](06-engine-host-port.md), sections 3.10 and 5.2). A new function
  `metal_source_rev()` reads it, and returns 0 when the file is missing.
- Every release tag is `v{version}-metal.{metal_rev}`, and has a `strata-macos-arm64.zip` asset.
- On MAC, `prebuilt_bases()` (`setup.py:1500`) formats `PREBUILT_TAG_URL` with both values, and keeps the "latest"
  fallback. On the PC the function is unchanged.
- The release policy keeps this URL valid for every commit on main: the version, `METAL_REV` and
  `MIN_ENGINE_METAL` reach main only after their release is published, and `check_release_policy.py --remote`
  checks it in CI ([08](08-build-and-distribution.md), section 11.1).

**Why vision reserves 2100 MiB.** On a PC, `--vram-reserve-mib 700` keeps room on the card for the image encoder. On a
Mac, W (`recommendedMaxWorkingSetSize`) is a device constant: the engine cannot see the 1.2-1.4 GB that the encoder,
in its own process, takes from the same memory. So the reserve covers the engine's default 700 MiB plus the encoder.
Measurement item X-1 (section 14) replaces 2100 with a measured value.

**Families on Mac.** `unsloth` (UD-Q4_K_XL) is hidden from the menus and refused by `--family unsloth`, with
`not yet available on Mac (its Q4_K / Q5_K kernels are not ported)`. The PLAN line also refuses it
(`reason=unsupported_type_q4_K`, section 4.4), so the hidden family and the engine cannot disagree for long. The
upstream `MODELS[...]["ram_gb"]` and `["arena_gb"]` are never printed on a Mac; the plan replaces them.

### 4.2 Probes

| Upstream function | PC | Mac |
| --- | --- | --- |
| `ram_gb()` (`setup.py:271`) | `/proc/meminfo` or `GlobalMemoryStatusEx`, in GiB | `macos.ram_bytes() / 2**30` |
| `page_file_gb()` (`:280`) | Windows only | None (unchanged) |
| `cpu_info()` (`:290`) | `(name, avx2, avx512)` | `(macos.cpu_brand() or "Apple silicon", False, False)`. No CPUID stub runs. |
| `gpus()` (`:375`) | `nvidia-smi` | `[]`, without running anything |
| `amd_gpus()` (`:981`) | sysfs / Windows registry | `[]` |
| `gpu_info()` (`:614`) | NVIDIA card | None. Mac code uses `macos.metal_devices()`. |
| `is_wsl()` (`:2233`) | | False (unchanged) |
| `free_gb()` (`:677`) | `shutil.disk_usage` | Unchanged. APFS reports purgeable space as used, so a full-looking disk can still have room: the "not enough free disk space" hint adds `macOS may free purgeable space (local snapshots, caches) when it is needed: About This Mac > Storage`. |
| `engine_version()` (`:2215`) | BUILD.json, else `engine=x.y.z\n` in the binary | Unchanged |
| `hardware_key()` (`:2237`) | GPU name, VRAM, CPU, RAM, model, context, images | The upstream helper is calibration-only. Calibration is withdrawn for this Mac product, so do not port the helper unless a supported Mac feature later needs the same key. |

### 4.3 Step order on Mac

| Mac step | Title | Upstream step |
| --- | --- | --- |
| 0 | already installed: just start it | 0, unchanged |
| 1 | `checking this Mac` | 1, replaced by `mac_step1(a)` |
| 2 | `Python packages` | 3 |
| 3 | `the Strata engine` | 4, Mac branch (section 4.7) |
| 4 | `your choices` | 2, with Mac menus (section 4.5) |
| 5 | `downloading <model>` | 5 |
| 6 | `preparing the model for Strata` | 6, Mac branch (section 4.8) |
| 7 | `writing the start script` | 7, Mac branch (section 4.9) |

The engine comes before the menus because the menus need its plan (P3). It comes after the Python packages because a
local build needs `cmake` and `ninja` from the venv.

Implementation: steps 3 and 4 of `main()` (`setup.py:3001-3039`) go into `if not MAC:`. On MAC, `main()` runs
`mac_step1(a)`, then step 2 (`pip_install`, unchanged), then step 3 (`mac_engine(a)`, section 4.7), then
`step(4 if MAC else 2, "your choices")`. Steps 5-7 keep their numbers. Step 0 is unchanged. An adopted earlier config
(`setup.py:2614-2632`) never carries a GPU choice into a Mac install: `ch["gpu"]` and `layer_split` are dropped.

**`mac_step1(a)`** runs these checks in order. Each `fail` prints the upstream `fail(msg, hint)` form and exits 1.

| # | Check | Result |
| --- | --- | --- |
| 1 | `platform.machine() != "arm64"` and not `macos.translated()` | fail: `Strata needs a Mac with Apple silicon (M1 or newer); this Mac has an Intel CPU` |
| 2 | `macos.translated()` | fail: `this Python runs under Rosetta (x86_64)`, with hint `install an arm64 Python (Homebrew in /opt/homebrew, or python.org's universal2 installer), delete the .venv folder and run ./setup.sh again` |
| 3 | `macos.macos_version() < MACOS_MIN` | fail: `Strata needs macOS 27 or newer; this Mac has macOS <x.y>`, with hint `System Settings > General > Software Update` |
| 4 | `macos.device_problem(dev)` is not None | fail with that text |
| 5 | | ok: `GPU: Apple M5 Pro, 20-core GPU; the GPU can use 51.8 GB of the 64 GB of unified memory`. When `dev["wired_limit"]` is not 0, it adds `(iogpu.wired_limit_mb is set to N)`. |
| 6 | `ram < MAC_RAM_FLOOR_GB` and not `a.check` | `confirm_risk("RAM: N GB - Strata needs a Mac with 32 GB or more: with less, most of the model's experts are read from the SSD for every token. Expect it to be very slow.", explicit=bool(a.model), yes=a.yes, stop=..., hint="--model IQ1_M --family coder --yes installs the smallest model anyway")` |
| 7 | `macos.low_power_mode()` | warn: `Low Power Mode is on: the GPU runs slower (System Settings > Battery)` |
| 8 | | ok: `CPU: Apple M5 Pro (6 Super + 12 Performance cores)`, from `perf_levels()`. There is no AVX check. |
| 9 | `a.backend in ("cuda", "hip")` | fail: `--backend <x> is for NVIDIA / AMD cards; a Mac uses Metal (leave --backend out)` |
| 10 | `a.gpus` or `a.layer_split` | fail: `a Mac has one GPU: --gpus and --layer-split do not apply` |
| 11 | `a.gpu` not in (None, 0) | fail: `a Mac has one GPU (number 0)` |

It returns `gpu = {**dev, "count": 1, "archs": ["apple"]}`, and `main()` sets `hip = False`, `amd = []`, `multi = []`,
`sel = [0]`, `chosen = [gpu]`, `avx2 = avx512 = False`. The caffeinate assertion starts here (section 4.13).

### 4.4 `metal_plan`, `plan_class` and the geometry files

#### 4.4.1 `metal_plan`

```python
@functools.lru_cache(maxsize=None)
def metal_plan(family, model, ctx, kv, vision="none", budget_gib=None, files=None) -> dict:
```

- **Command.**
  ```
  engine/strata --plan-only <source> --expert-profile data/<profile> --expert-cache auto --prefill auto
                --spec 4 --max-context <ctx> --kv <kv> --vram-reserve-mib <R>
                [--resident-budget-gib <budget_gib>] [--gpu-budget-mib <a.gpu_budget_mib>]
  ```
  - `<source>` is `--geometry data/geometry/<family>-<model>.txt` before the download. When `files` is given (a tuple
    of `pack`, `shard1`, `ple` or None, and `mtp_rt`), it is the real-files form: `--pack P --native S
    [--ple-gguf G] --mtp RT`.
  - `<profile>` is `FAMILIES[family].get("profile", "expert-profile.bin")`.
  - `kv` is `fp16` when `ctx <= 8192`, as in step 7.
  - `R` is `VISION_METAL[vision]["reserve_mib"]` with vision, else 700.
- **Run.** `subprocess.run(cmd, capture_output=True, text=True, timeout=PLAN_TIMEOUT_S)`. The environment is setup's
  own; the engine needs no library path.
- **Accept** return codes 0 and 3 with a stdout line that starts with `PLAN `.
- **Parse.**
  - `dict(kv.split("=", 1) for kv in line.split()[1:])`.
  - Values that are all digits become int.
  - `reason_raw` keeps the raw `reason` value: some reasons contain tensor type names with underscores
    (`unsupported_type_q4_K`), so the underscore-to-space replacement cannot be undone. `reason` is
    `reason_raw.replace("_", " ")`.
- **Any other result** fails with the last 5 stderr lines. The hint depends on the result:

  | Result | Hint |
  | --- | --- |
  | `returncode == -9` (SIGKILL before any output) | `macOS refused to run the engine (code signature or quarantine): xattr -dr com.apple.quarantine engine, then run setup again` |
  | stderr contains `Library not loaded` | `the engine's files are incomplete: delete the engine folder and run setup again` |
  | timeout | `the engine took more than 30 s to plan: run engine/strata --plan-only ... by hand and report its output` |
  | anything else | `delete the engine folder and run setup again (setup --build compiles one)` |

#### 4.4.2 `plan_class`

```python
def plan_class(p: dict) -> str:
    if not p["fits"]:
        return "unsupported" if p.get("reason_raw", "").startswith("unsupported_type_") else "none"
    if p["mode"] == "full":
        return "full"
    share = p["experts_resident"] / p["experts_total"]
    return "ok" if share >= METAL_SHARE_OK else "slow" if share >= METAL_SHARE_SLOW else "very slow"
```

| Class | Text in menus and `--check` |
| --- | --- |
| `full` | `fits: every expert in GPU memory` |
| `ok` | `fits: N% of its experts in GPU memory, the rest read from the SSD` |
| `slow` | `fits: N% of its experts in GPU memory, the rest read from the SSD: slower` |
| `very slow` | `fits: N% of its experts in GPU memory, the rest read from the SSD: very slow` |
| `none` | `does not fit: <reason>` |
| `unsupported` | `not yet available on Mac` |

N is `floor(100 × experts_resident / experts_total)`. The order `full > ok > slow > very slow > none` is what "worse"
means in section 4.5.

#### 4.4.3 Geometry files

Setup must plan before it downloads 60-85 GB. The engine plans from a small text file that holds exactly the facts
its plan reads from the real files ([05](05-memory-and-residency.md), section 3.1, "Pack geometry").

Format version 1, UTF-8, one `key=value` or record per line, `#` comments allowed:

```
strata-geometry 1
family=qwen
model=Q2_0
shard1_name=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf
shard1_bytes=<exact size>
shard2_name=Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf
shard2_bytes=<exact size>
n_layers=48
n_expert=512
nonexpert_bytes=<bytes>
wrapped_unread_bytes=<bytes>
dense_pack_bytes=<bytes>
mtp_bytes=<bytes>
mtp_bind_bytes=<bytes>
n_qsa_layers=12
n_kv_head=<n>
head_dim=<n>
n_embd=<n>
vocab=<n>
types=q2_0,f16,f32
layer 0 blob=1392640 gu=q2_0 d=q2_0
...
layer 47 blob=1392640 gu=q2_0 d=q2_0
```

- One `shard<i>_name` and `shard<i>_bytes` pair per GGUF shard. `nonexpert_bytes`, `wrapped_unread_bytes` and
  `dense_pack_bytes` are the plan inputs of [05](05-memory-and-residency.md), sections 2.1 and 3.2; they cover all
  shards, so a Swift file (experts in both shards, the PLE table in shard 1) needs no other keys.
- `types` lists every tensor type the engine loads. The engine answers `fits=0 reason=unsupported_type_<T>` for a type
  it has no kernel for, both from a geometry file and from real files.
- The values are written by the engine, never by hand. They come from `strata --write-geometry FILE <real-files
  form>` ([06](06-engine-host-port.md), section 3.10).
- **Rule:** a plan from the geometry file must equal the plan from the real files, key for key, for the same flags.
  When `make_geometry.py` writes a geometry file from the real files, it also writes `data/geometry/<tag>.check`:
  the real-files PLAN lines at the W and RAM of the golden profiles `mac-24GB`, `mac-48GB` and `mac-64GB-M5Pro`
  (`--plan-w-mib` / `--plan-ram-mib`, section 10.2), with setup's default arguments for that model.
  `serve/test_macos.py` plans every shipped geometry file at the same points and compares key for key (section 10.3).
  The parity run repeats the comparison against live real files: the Python layer of the reference Mac's nightly
  run ([09](09-testing-and-validation.md), section 7.2, step 3).
- `tools/make_geometry.py --family F --model M --pack P --native S [--ple-gguf G] --mtp RT` wraps `--write-geometry`
  and writes `data/geometry/<family>-<model>.txt`. `tools/make_geometry.py --check-remote` compares every
  `shard<i>_bytes` of each file with the Content-Length of the pinned Hugging Face URL (`HF_REVISIONS`, `setup.py:63`, a `HEAD`
  request that follows redirects). It also fails when a shipped model has no geometry file. It prints one line per
  file and exits 1 on any difference. `ci.yml` runs it ([08](08-build-and-distribution.md), section 12.1), so a
  repository revision bump without a new geometry file fails CI.
- Shipped files, from the `families` of each `MODELS` entry: `qwen-Q2_0`, `qwen-IQ2_XS`, `qwen-IQ3_XXS`,
  `qwen-IQ3_S`, `swift-IQ2_XS`, `swift-IQ3_XXS`, `coder-IQ1_M`. There is none for `unsloth`.
- After step 5, setup compares every `shard<i>_bytes` with the downloaded shard. When they differ, it warns
  `the model files differ from the ones this setup was measured with; the final plan below is the one that counts`.
  Step 7's final plan (section 4.9) then decides.
- No geometry file for a model (a family added upstream before the project has one): setup plans that model after its
  download only, and the menu line says `fit known after the download`.

#### 4.4.4 Engine-side requirements

These are specified in [06](06-engine-host-port.md), section 3.10. Setup depends on them:
- `--geometry FILE` is accepted only with `--plan-only`, and never together with `--pack`, `--native`, `--ple-gguf`
  or `--mtp`. Either error exits 2.
- `--plan-only` is light. It opens the Metal device only to read W and the families (not at all with `--plan-w-mib`
  and `--plan-ram-mib`), and does not load the metallib,
  build pipelines or allocate buffers. It must take under 200 ms with a geometry file and under 1 s with real files
  on the reference Mac.
- The MTP terms of the plan count only when `--spec` is above 0.
- The PLAN line has `w_full_mib` ([05](05-memory-and-residency.md), section 3.4): the W, in MiB, at which this plan
  would be full mode.

### 4.5 Menus and consent

#### 4.5.1 Model menu

For each model of the family, setup calls `metal_plan(family, m, 32768, "int8")` (no vision) and prints:

```
  1) Q2_0     <about>; download 66 GB, fits: every expert in GPU memory, uses ~44.3 GB of the GPU's 51.8 GB
  2) IQ2_XS   <about>; download 68 GB, fits: 93% of its experts in GPU memory, the rest read from the SSD, uses ~51.1 GB of the GPU's 51.8 GB
```

`uses ~X` is `total_mib / 1024`, and `the GPU's Y` is `w_mib / 1024`. For the `none` and `unsupported` classes the
`uses` part is dropped. The upstream line format (`setup.py:2808`) is kept up to `download N GB,`.

The model menu displays each 32K plan. If that plan is `none` or `unsupported`, setup also plans 8K; when 8K fits,
the entry is selectable and is marked `fits at 8K only`. Otherwise that model is unavailable.

The recommended entry:
1. `IQ3_XXS`, when the family has it and its 32K class is `full`. (Upstream recommends it at 60 GB of RAM or more,
   `setup.py:2809`.)
2. Otherwise `names[0]`, when its 32K class is `full` or `ok`.
3. Otherwise the first model whose 32K class is `full` or `ok`.
4. Otherwise the first model whose 32K class fits (`slow` or `very slow`).
5. Otherwise the first model in menu order whose 8K plan fits.
6. Otherwise setup fails: `no <family> model fits this Mac's GPU memory (<W> GB)`, with hint
   `the Coder IQ1_M (--family coder) is the smallest model`. With `--family coder`, the hint is
   `close other apps or choose a shorter context, then run setup again`.

After the choice:
- `slow` or `very slow` goes through `confirm_risk(<class text>, explicit=bool(a.model), yes=a.yes, stop=..., hint="--model <m> --yes installs it anyway")`.
  So `--yes` alone stops on recommendation 4, as P7 requires.
- A model whose 32K and 8K plans are both `none` or `unsupported` cannot be chosen, not even with `--model X --yes`.
  It fails with the class text.
- With `--family qwen`, a note follows the list: `(Unsloth UD-Q4_K_XL is not yet available on Mac)`.

#### 4.5.2 Context menu

For each entry of `CONTEXTS`, setup calls `metal_plan(family, model, c, "int8" if c > 8192 else "fp16")` and prints
the upstream line (`setup.py:2863-2869`) with the class text in place of the RAM note:

```
  3) 64K tokens   fits: every expert in GPU memory   (recommended for this Mac)
```

The recommended context:
- When the 32K plan is `full`: the largest `c ≤ 131072` that is still `full`.
- When the 32K plan is tiered: 32768. With 8-bit KV, a 32K context costs about 0.45 GiB, about 330 Q2_0 experts, so
  a shorter context buys little.
- When the 32K plan does not fit but 8K does: 8192.

Rules:
- An entry with `fits=0` is printed with its reason and is not accepted. `--context` with such a value fails with the
  reason. This exact-context check also applies to a model admitted by its 8K fallback.
- When the chosen context gives a worse class than the model step did, `confirm_risk(<class text>,
  explicit=bool(a.context), ...)` runs.
- The KV question is upstream's (`setup.py:2900-2909`). A KV choice other than int8 is planned again, and the new
  class is printed as an ok line.
- RoPE scaling above 262144 is unchanged (`setup.py:2881-2893`).

#### 4.5.3 Images

The text above the question changes to `This adds a 0.9 GB download and keeps ~1.4 GB of GPU memory free for the image
encoder, so fewer experts may fit.` After a yes, setup plans again with `vision="gpu"` and prints the change, for
example `with images: 88% of its experts in GPU memory (was every expert)`.

- When the class gets worse: `confirm_risk(..., explicit=bool(a.vision), ...)`.
- `--yes` alone with a worse class turns images off with a warning:
  `images off: they would move experts to the SSD (--vision gpu --yes keeps them)`.
- An explicit `--vision gpu --yes` is consent, unless the plan with images has `fits=0`.
- `--vision cpu` is accepted, and uses the 700 MiB reserve and `max_tokens` 300 of `VISION_METAL["cpu"]`.
- `hip_vision()` and `prebuilt_vision()` are never called on a Mac. Every Metal `strata-vision` runs on the GPU.

#### 4.5.4 `--low-ram`

| Value | Mac behaviour |
| --- | --- |
| `auto` (default) | Normal. The plan chooses full or tiered mode. |
| `on`, `resident`, `mmap` | warn `--low-ram <v>: a Mac has no separate low-RAM mode; the engine reads experts it cannot keep from the SSD by itself`, then act as `auto` |
| `off` | Requires full mode. When the chosen plan is tiered, fail `--low-ram off: <model> does not fit this Mac's GPU memory with every expert resident`. |

`low_ram_needed`, `low_ram_fits`, `low_ram_gpu_share`, `low_ram_resident`, `low_ram_together` and
`confirm_paging` are never called on a Mac. Their RAM model is the PC's.

#### 4.5.5 `--resident-budget-gib`

On a Mac it caps the resident experts for any model ([05](05-memory-and-residency.md), section 3.2, `E_cap`). It is
passed to `metal_plan` and to the step 7 args. The upstream warning `--resident-budget-gib is for UD-Q4_K_XL`
(`setup.py:2841`) is skipped. `budget_choice()` is not called.

#### 4.5.6 Wired-limit advice

When the final plan is tiered and `w_full_mib ≤ ram_mib − 8192`, setup prints:

```
  ! This Mac could keep every expert in GPU memory if macOS let the GPU use more of it:
      sudo sysctl iogpu.wired_limit_mb=<w_full_mib rounded up to 1024>
    It lasts until the next restart and leaves less memory for macOS and other apps.
    Setup does not change it. Run setup again afterwards to use it.
```

When `w_full_mib > ram_mib − 8192`, nothing is printed. Setup never runs sudo (P6). Measurement M-8
([05](05-memory-and-residency.md), section 10) confirms that W follows the setting. Until then the advice is behind
`STRATA_WIRED_ADVICE=1`.

### 4.6 `--check` on Mac

`--check` runs `mac_step1`, then `get_prebuilt_metal()` into `engine/`. The download uses only `urllib` and
`zipfile`, so it works before step 2. The engine stays for a later install. Then setup plans every model of every
Mac family at 32K, int8, without vision, and prints:

```
  qwen Q2_0      fits: every expert in GPU memory
  qwen IQ3_S     fits: 71% of its experts in GPU memory, the rest read from the SSD: slower
  coder IQ1_M    fits: every expert in GPU memory

This Mac can run Strata. Run it again without --check to install.
```

- It returns 0 when at least one model is `full`, `ok`, `slow` or `very slow`.
- When none fits, the last line is `No model fits this Mac's GPU memory.` and it returns 1.
- When no engine can be had (no internet, no release), it returns 1 with
  `could not get the Strata engine to check this Mac: <why>`.
- No caffeinate runs during `--check`.

### 4.7 The engine (Mac step 3)

`mac_engine(a)` returns `eng` (the `engine/` folder) and `meta` (its BUILD.json):

```python
llama = get_llama_cpp()
eng = None if a.build else get_prebuilt_metal(a.prebuilt)
if eng is None:
    eng = build_engine_metal(a, llama)
smoke_test(eng)
meta = json.loads((eng / "BUILD.json").read_text())
```

`get_llama_cpp()` (`setup.py:868`) runs first on every install, as upstream: step 6 points the pack and MTP
tools at its `gguf-py` (`STRATA_GGUF_PY`, `setup.py:3076`), and a local build needs its `ggml`.

#### 4.7.1 `get_prebuilt_metal(url_base, updating=False)`

It follows `get_prebuilt` (`setup.py:1509-1590`), with these differences:

| Step | Upstream | Mac |
| --- | --- | --- |
| Installed engine | accepted when `version ≥ MIN_ENGINE` and the archs cover the card | accepted when `backend == "metal"`, `version ≥ MIN_ENGINE`, `metal_rev ≥ MIN_ENGINE_METAL`, `macos_min ≤ macos_version()`, `gpu_families ⊆ dev["families"]`, and `engine/strata` and `engine/strata.metallib` exist. `source == "local"` returns None, so `build_engine_metal` decides. An installed engine with another `backend` (copied from a PC) is deleted with a note. |
| Probe and download | `HEAD`, then `download()` | Unchanged. A `CERTIFICATE_VERIFY_FAILED` error gets the hint of section 4.7.4. |
| Unpack | `zipfile.extractall(engine/_unpack)` | The same, then each member's mode bits from `ZipInfo.external_attr >> 16` are dropped: setup sets `0o755` on `strata` and `strata-vision` and `0o644` on the rest. A member that is a symlink (`S_ISLNK`) fails the install: the asset never has any. |
| Checks | version, then archs | version, `metal_rev`, `backend == "metal"`, `arch == "arm64"`, `macos_min`, `gpu_families`. Each failure warns with the values and returns None (`compiling instead`). |
| Move | `p.replace(dst)` per member | Unchanged. `os.replace` gives the file a new inode. An in-place copy over a signed Mach-O must never be used: the kernel caches the old code signature per vnode, and the new binary is killed on its next start. |
| Quarantine | | `macos.strip_quarantine(eng)` |
| Ok line | `ready-made engine 0.1.34 for sm_75, ... (CUDA 13.0)` | `ready-made engine 0.1.34 (Metal rev N, macOS 27+)` |

`update_installed_engine()` (`setup.py:1593`) gets a metal branch before its CUDA code. It handles `source == "local"`
through `build_engine_metal` and prebuilt engines through `get_prebuilt_metal(url_base, updating=True)`. The
"engine in use" probe (`open(..., "r+b")`, `setup.py:1641-1647`) is skipped on a Mac: a running engine keeps its old
inode, and a model running in another window must not stop the update. The CUDA wheels are never installed.

#### 4.7.2 BUILD.json (metal)

```json
{
 "version": "0.1.34",
 "metal_rev": 1,
 "backend": "metal",
 "arch": "arm64",
 "macos_min": "27.0",
 "gpu_families": ["apple7", "metal4"],
 "metal_std": "metal4.0",
 "source": "prebuilt",
 "lib_dirs": [],
 "vision": "gpu",
 "src": "<source_hash(ENGINE_SOURCES)>",
 "vision_src": "<source_hash(VISION_SOURCES)>"
}
```

- `source` is `"prebuilt"` in a release asset and `"local"` after `build_engine_metal`.
- `vision` is `"gpu"` when `strata-vision` is present, else `"none"`. `vision_src` is null with `"none"`.
- `ENGINE_SOURCES` gains `"METAL_REV"` and `"cmake/metal"`; `src` and `include` already cover the Metal sources
  ([08](08-build-and-distribution.md), section 8.3). A `git pull` that changes them rebuilds a local engine.
  `source_hash` skips Finder's `.DS_Store` files on a Mac (same section).
- Four informational keys follow `vision_src`: `xcode`, `metal_toolchain`, `commit` and `metallib_sha256`
  ([08](08-build-and-distribution.md), section 8.4). Setup never reads them.
- One function writes the file, `metal_build_json(eng, source)` ([08](08-build-and-distribution.md), section 8.4),
  for a local build and for a release alike.

#### 4.7.3 No CUDA libraries

`pip_install(CUDA_WHEELS, ...)` (`setup.py:3020-3021`) is skipped. `lib_dirs = []`, and `cuda_lib_dirs()` is never
called.

#### 4.7.4 SSL certificates

`main()` calls `macos.ssl_setup()` before its first download. When it returns a path, it prints
`ok: HTTPS certificates: <path> (this Python has none of its own)`.

Any download that still fails with `CERTIFICATE_VERIFY_FAILED` fails with the hint:
`this Python has no certificates: run "Install Certificates.command" in /Applications/Python 3.x (python.org's
installer), or use Homebrew's Python`.

#### 4.7.5 Smoke test

After the engine is in place, setup runs `engine/strata --version` with a 30 s timeout:

| Result | Action |
| --- | --- |
| rc 0, stdout `strata <v> metal <rev>` | ok, then `engine/strata --list-devices` |
| rc −9 | fail `macOS stopped the engine before it started`. When `macos.has_quarantine(engine/strata)` is True, the hint is `xattr -dr com.apple.quarantine engine`; otherwise it is `codesign -dv engine/strata` and `delete the engine folder and run setup again`. |
| stderr contains `Library not loaded` | fail `the engine links a library that is not on this Mac` (a build bug: the asset must be self-contained, [08](08-build-and-distribution.md)) |
| any other failure | fail with the stderr tail |

`--list-devices` must print one device whose `families` include `apple7` and `metal4`. Anything else fails with its
output.

#### 4.7.6 `build_engine_metal(a, llama)`

1. `macos.xcode_problem()` must be None. Otherwise fail with that text and the hint:
   `install Xcode from the App Store, open it once, then run: xcodebuild -downloadComponent MetalToolchain`.
   The Command Line Tools alone have no `metal` compiler.
2. The stamp logic is upstream's (`setup.py:1806-1822`), without archs. The engine is current when
   `meta["src"] == source_hash(ENGINE_SOURCES)` and `engine/strata` exists.
3. Engine: `cmake_build(ROOT, ROOT / "build", "strata", metal_engine_defs(llama), None, "")`. The function
   ([08](08-build-and-distribution.md), section 8.1) returns
   ```
   -DSTRATA_BUILD_TESTS=OFF
   -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0 -DSTRATA_GGML_DIR=<llama>
   ```
   The release is built with the same function, so a local engine and a released one differ only in the strip and
   the signature. Then copy `build/strata` and `build/strata.metallib` to `engine/`, each through a temporary name in `engine/` and
   `os.replace` (section 4.7.1, Move).
4. Vision, unless `a.vision in ("no", "none")`: `cmake_build(ROOT / "tools" / "vision", ROOT / "build-vision",
   "strata-vision", metal_vision_defs(llama, portable=False), None, "")`, then copy
   `build-vision/bin/strata-vision` to `engine/` as in step 3. The function
   ([08](08-build-and-distribution.md), section 8.2) returns
   ```
   -DLLAMA_DIR=<llama> -DSTRATA_VISION_CUDA=OFF -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON
   -DGGML_OPENMP=OFF -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=27.0
   ```
   `-DGGML_OPENMP=OFF` is required: without `STRATA_PORTABLE`, `tools/vision/CMakeLists.txt` leaves ggml's OpenMP at
   its default, ON, and Apple clang has no OpenMP. `GGML_METAL_EMBED_LIBRARY` embeds ggml's shader source in the binary, and Metal compiles it at the first start, so
   `strata-vision` needs no `.metallib` beside it and no Xcode at run time.
5. The arm64 linker signs both binaries ad hoc. Nothing may modify them after the link; a change needs
   `codesign -s - -f <file>`.
6. Write BUILD.json (section 4.7.2): `json.dumps(metal_build_json(eng, "local"), indent=1) + "\n"`.
7. The texts: `Compiling the Strata engine for this Mac (5-10 minutes, once) ...`, and
   `The engine's source changed: compiling it again (only what changed, a few minutes) ...`.

`install_build_tools()` (`setup.py:1673`), `find_nvcc()` and `find_vcvars()` are never called on a Mac.
`cmake_build` (`setup.py:1746`) is unchanged, including its one retry. `find_tool()` (`setup.py:664`) gets a MAC branch
that looks in the venv's `bin` (`Path(sys.executable).parent`) before PATH. Setup does not activate the venv, so
upstream's PATH-first order would pick an old Homebrew or MacPorts `cmake` over the pinned `cmake==4.4.3`. `ninja`
reaches CMake as `CMAKE_MAKE_PROGRAM`, as upstream.

When the ready-made asset has no `strata-vision` and images are wanted, setup builds the encoder only if
`xcode_problem()` is None. Otherwise it turns images off with
`images off: the ready-made engine has no image encoder and Xcode is not installed to compile one`.

### 4.8 Pack, MTP and disk (step 6)

- The pack is always made by `tools/iq_pack.py`. The `strata_pack` AVX-512 repack branch for Q2_0
  (`setup.py:3077-3087`) needs AVX-512 and is never taken on a Mac.
- `--experts-bin` is never passed. The low-RAM `experts.bin` does not exist on a Mac
  ([05](05-memory-and-residency.md), section 12).
- MTP: `mtp_fetch.py`, `mtp_pack.py` and `mtp_rt.py` are unchanged.
- Disk need (`setup.py:2993`): `to_fetch + 8`, plus 1 with images. The experts.bin term is always 0.

### 4.9 Step 7: args, the final plan and the config

**Args.** The same list as upstream (`setup.py:3124-3126`), with these changes:
- No `--kv-resident`. The KV streaming block (`setup.py:3138-3173`) is skipped on a Mac, with one ok line when
  `a.kv_streaming == "on"`: `--kv-streaming: not used on a Mac (the KV cache is in unified memory)`.
- No `--resident-experts` and no `--mmap-experts`.
- `--resident-budget-gib G` only when the user gave it.
- With images: `--vision --vram-reserve-mib <VISION_METAL[vision]["reserve_mib"]>`.
- `--kv` only when `ctx > 8192`, as upstream. The control-vector flags are unchanged.

**Final plan.** Setup calls `metal_plan(..., files=(pack, shards[0], ple, rt))` with the real files.
- `fits=0`: fail with the reason. This can only happen when the files differ from the geometry.
- Another class than the menu's: warn `with the downloaded files: <class text> (the menu said: <class text>)`.
- Then the wired-limit advice runs (section 4.5.6).

**Config.** The upstream keys (`setup.py:3182-3184`), plus:
- `backend: "metal"`.
- `lib_dirs: []`.
- `plan: {"mode", "experts_resident", "experts_total", "total_mib", "w_mib"}`, from the final plan. `start()`
  compares against it (section 4.11).

Never written on a Mac: `gpu`, `gpus_asked`, `layer_split`, `env`. Vision: `"gpu": vision == "gpu"`, and for
`vision == "cpu"`, `threads = macos.fast_cores()`.

**Calibration.** It is neither offered nor applied (section 4.16).

### 4.10 Run scripts, launchers and quarantine

`write_run_script()` (`setup.py:2478`) gets a MAC branch. It writes two files:
- `run-<model>.sh`: as upstream, but each argument is quoted with `shlex.quote`. Upstream's `"..."` quoting breaks on a
  path that contains `$`, `` ` `` or `"`, and Finder makes such names easy to create.
- `run-<model>.command`: the same script with a title line, for a Finder double-click:
  ```sh
  #!/bin/sh
  printf '\033]0;Strata <model>\007'
  cd '<ROOT>' || exit 1
  exec /usr/bin/arch -arm64 '<python>' '<ROOT>/serve/server.py' --engine strata --config '<cfg>' --port <port> --open
  ```

`run-<model>.sh` also starts Python through `/usr/bin/arch -arm64`. A universal2 Python (python.org's installer) that
is started from a Terminal running under Rosetta would otherwise run as x86_64, and `serve/macos.py` and the engine's
plan would see a translated process.

Both get mode `0o755`. Setup writes them, so they carry no quarantine. The "All set" text names both.

**Launchers in the repository** (section 5.2): `START-HERE.command` and `SETUP.command`.

**Quarantine.** A ZIP downloaded with a browser marks every file inside with `com.apple.quarantine`, and Gatekeeper
then refuses to open the unsigned `.command` files. Setup strips quarantine from exactly these paths:
`START-HERE.command`, `SETUP.command`, `setup.sh` and `engine/`. It never strips it from a downloaded GGUF, from the
user's folders, or recursively from ROOT.

The first launch of a quarantined `START-HERE.command` cannot be fixed by setup, which has not run yet. The user
documents (section 11) therefore:
- recommend `git clone`, which sets no quarantine;
- for a ZIP: System Settings > Privacy & Security > "Open Anyway" after the first refusal (macOS 15 and later
  removed the Control-click > Open bypass);
- or, in Terminal: `xattr -dr com.apple.quarantine <Strata folder>`.

### 4.11 `start()` on Mac

`start()` (`setup.py:2308-2404`) gets a MAC branch after the draft-vocab refresh. It replaces everything from
`if cfg.get("backend") == "hip"` to the banner:

1. Skip `amd_gpus`, `gpus()`, `check_gpus`, `offer_together`, `split_mmap` and `ensure_engine_for`. An explicit `--gpu`
   other than 0, `--gpus` or `--layer-split` fails as in section 4.3.
2. **Plan again** with the config's own args: `engine/strata --plan-only <cfg args>`. The args already hold the real
   files.
   - `fits=0`: fail `<model> no longer fits this Mac's GPU memory: <reason>`, with hint
     `./setup.sh --setup chooses again (a shorter context or a smaller model)`.
   - A mode or resident share other than `cfg["plan"]`: ok `GPU memory: <class text> (set up as: <class text>)`.
   - `avail_mib < total_mib`: ok `macOS has <avail> GB free right now; the model needs <total> GB: it will push other
     apps' memory to compression and swap while it loads`.
3. `macos.engine_pids()` not empty: warn `another Strata engine is running (PID <n>): both share the GPU's memory, so
   this one may not fit; close the other model's window first`.
4. The banner text changes:
   ```
     Starting <model>: it maps about N GB of model files into memory and keeps them there for the GPU.
     The first start after a restart reads them from the SSD (about 10-30 s); later starts are faster.
     The browser opens when it is ready. Closing this window stops the model.
   ```
5. **Ctrl+C.** While `subprocess.call(cmd)` runs, `start()` installs `signal.signal(SIGINT, lambda *a: None)` and
   restores the previous handler afterwards. The server, in the same process group, receives the SIGINT and shuts
   the engine down with QUIT. Without this, `subprocess.call` catches the KeyboardInterrupt, waits 0.25 s and kills the
   server, which leaves the engine to die on stdin EOF. `SIG_IGN` must not be used: the child inherits it, and the
   server would then ignore Ctrl+C. Both behaviours were verified on the reference Mac.
6. `STRATA_EXECV` (`setup.py:2399`) is unchanged.

`upgrade_config()` (`setup.py:2283`) is unchanged: its WSL branch does not apply, and it never adds CUDA args.
`update_installed_engine()` uses the metal branch of section 4.7.1.

### 4.12 The data folder

`data_folder()` (`setup.py:2111`) keeps its default, `ROOT.parent / "Strata-data"`, except on a Mac when ROOT.parent
is inside one of these folders:
- `~/Desktop`, `~/Documents` or `~/Downloads`, which macOS privacy controls (TCC) protect and iCloud Drive may sync;
- `~/Library/Mobile Documents` (iCloud Drive) or `~/Library/CloudStorage` (Dropbox, OneDrive, Google Drive).

There the default is `~/Strata-data`, with an ok line saying why. An existing data folder found through the settings
file is kept wherever it is.

When setup creates the data folder on a Mac:
1. It writes an empty `.metadata_never_index`, so Spotlight does not index 70-120 GB of GGUFs.
2. It calls `macos.tm_exclude(folder)` and prints `excluded from Time Machine backups (tmutil removeexclusion
   <folder> undoes it)`.

Errors:
- `PermissionError` with errno EPERM in a TCC-protected folder fails with the hint
  `macOS blocked access: allow Terminal in System Settings > Privacy & Security > Files and Folders (or Full Disk
  Access), or choose --data-dir ~/Strata-data`.
- An explicit `--data-dir` inside iCloud Drive or CloudStorage goes through `confirm_risk(..., explicit=True, ...)`:
  the sync client may upload 70-120 GB, or evict the files it uploaded.
- The MCP install tool applies a stricter guard to both explicit and configured/default paths: it refuses
  `~/Desktop`, `~/Documents`, and `~/Downloads` too, because Desktop & Documents can be included in iCloud Drive
  sync and these locations have macOS privacy restrictions. It does not detect arbitrary user-selected sync folders
  outside the known iCloud and File Provider roots. Use `~/Strata-data` or an external unsynced volume.

The settings stay in `~/.config/strata` (`settings_path()`, `setup.py:2001`), as on Linux.

### 4.13 Sleep

An install run (one that goes past step 1: first run, `--setup`, `--build`, an engine update) starts
`macos.keep_awake()` (`caffeinate -i -w <setup pid>`) right after step 1. It prevents idle sleep during the downloads,
the pack and the builds. Setup terminates it right before it calls `start()`, and it also ends by itself when setup
exits.

No caffeinate runs for a plain start (step 0 finds a config and goes straight to `start()`) or for `--check`. While a
model is loaded, sleep is the server's business: the `keep_awake` config key (section 6.9).

Closing the lid of a MacBook still sleeps it, whatever runs; the setup completion text and the planned Mac
user guide (work item S20) say so.

### 4.14 Every CLI flag on a Mac

| Flag | Mac |
| --- | --- |
| `--family` | `qwen`, `swift`, `coder`. `unsloth` fails (section 4.1). |
| `--model` | As upstream, within the family. `UD-Q4_K_XL` fails. |
| `--context`, `--rope-scaling`, `--rope-scale` | As upstream, planned (section 4.5.2) |
| `--kv` | `int8`, `q4_0`, `k8v4`, planned |
| `--vision` | `yes`/`gpu`, `cpu`, `no`/`none` (section 4.5.3) |
| `--experimental-speed-projection` | As upstream |
| `--port`, `--host`, `--api-key` | The default host is `127.0.0.1`. Any non-loopback listener requires a non-empty API key from `--api-key`, `STRATA_API_KEY` or config; the server refuses to bind without one. |
| `--gpu` | `0` only (section 4.3) |
| `--gpus`, `--layer-split` | Fail |
| `--data-dir`, `--models-dir`, `--gguf-dir` | As upstream, plus section 4.12 |
| `--yes`, `--setup`, `--no-start` | As upstream |
| `--build` | `build_engine_metal` (section 4.7.6) |
| `--prebuilt` | A URL folder or a local folder, as upstream. Quarantine is stripped after the unpack. |
| `--check` | Section 4.6 |
| `--calibrate` | Removed from the Mac interface. No Mac calibration command is implemented; the setup CLI remains fail-closed. |
| `--draft-vocab` | As upstream |
| `--low-ram` | Section 4.5.4 |
| `--resident-budget-gib` | Section 4.5.5 |
| `--kv-streaming` | Accepted and ignored, with a note (section 4.9) |
| `--backend` | Gains `metal`, the default on darwin. `cuda` and `hip` fail. |
| `--gpu-budget-mib N` | New, hidden (`argparse.SUPPRESS`), Mac only: passed to every `metal_plan` call and stored in the config args. For tests and for two engines ([05](05-memory-and-residency.md), section 9). |
| `--skip-build` | As upstream |

### 4.15 Texts

| Upstream | Mac |
| --- | --- |
| Title (`setup.py:2604`): `Strata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)` | `Strata - Qwen3.8-Flash-Next on a Mac (unified memory)` |
| `checking your PC`, `this PC`, `for your GPU` | `checking this Mac`, `this Mac`, `for this Mac` |
| `Windows/Linux will page part of them from disk` | `macOS will compress and swap part of them` |
| `Next time: just run ./setup.sh (or run-<m>.sh)` | `Next time: ./setup.sh in Terminal, or double-click run-<m>.command (or START-HERE.command)` |
| `Other devices: the server window prints this PC's address` | `... this Mac's address`, plus `macOS will ask whether Python may accept incoming connections: allow it` when the firewall is on (section 6.3) |
| `--low-ram`, `--kv-streaming` and `--backend` help texts | Unchanged, plus one sentence each about Mac |

Units: memory sizes on a Mac print as GB = 2^30 bytes, so they match the RAM size Apple advertises ("64 GB"). File and
download sizes stay decimal GB, as upstream.

### 4.16 Calibration

Calibration is withdrawn from the macOS product (D-IMPL-236). The PC CUDA tuner, its settings, CLI flag, and tests have been removed. There is no Metal calibration command or calibration-only hardware key. If Metal calibration becomes useful later, design it around measured Mac workloads and review it as a separate feature.

---

## 5. `setup.sh` and the launchers

### 5.1 `setup.sh`

`setup.sh` currently checks the host before stopping with exit code 2. It requires Darwin, Apple silicon, and macOS 27
or newer. It does not select Python, create `.venv`, install packages, download models, or run `setup.py`. Direct
`setup.py` execution and calls to `setup.main()` also stop before changing files. This is intentional until the Mac
installer and Metal planner are implemented.

| Host | Result |
| --- | --- |
| Windows or Linux | Exit 1 with a macOS-only message; no environment is created. |
| Intel Mac | Exit 1 with an Apple-silicon requirement. |
| macOS older than 27, or an unreadable version | Exit 1 with a version or detection message. |
| Apple silicon on macOS 27 or newer | Exit 2 with the setup-not-implemented message; no environment is created. |

`tools/test_setup_shell.py` exercises the non-Mac refusal, the supported-Mac stop when run on a qualifying machine,
and the direct Python CLI refusal. The Mac-specific case is skipped when the host does not meet its OS and Python
preconditions. No installer acceptance is claimed yet.

### 5.2 Planned double-click launchers

No Finder launcher is currently shipped. These are the proposed files for a later Mac installer change:

`START-HERE.command`:
```sh
#!/bin/sh
# Double-click in Finder: installs Strata the first time, then starts the model (the same as ./setup.sh).
cd "$(dirname "$0")" || exit 1
exec ./setup.sh "$@"
```

`SETUP.command`:
```sh
#!/bin/sh
# Double-click in Finder: choose the model, context and options again (the same as ./setup.sh --setup).
cd "$(dirname "$0")" || exit 1
exec ./setup.sh --setup "$@"
```

Requirements:
- Git mode `100755` for both, and for `setup.sh` (as upstream). A ZIP from GitHub keeps the executable bits; Finder's
  Archive Utility restores them.
- `.gitattributes` gains `*.command text eol=lf`, next to `*.sh text eol=lf`. A CRLF line ending makes `/bin/sh` fail
  with `bad interpreter`.
- No `pause`, unlike the `.bat` files. Terminal keeps the window open after the script exits (its default "When the
  shell exits: Don't close the window"), so the user can read an error.
- Terminal runs a `.command` in a new login shell, so the user's `~/.zprofile` PATH applies. Section 5.1 does not
  depend on it.
- The names sort next to `START-HERE.bat` and `SETUP.bat`; a Mac user sees which file is theirs by the extension
  Finder shows.

Quarantine and Gatekeeper are covered in section 4.10.

---

## 6. `serve/server.py`

The server runs unchanged on a Mac apart from the eleven points below. Each is a `# MAC:` branch (P8) guarded by
`sys.platform == "darwin"` or by `cfg.get("backend") == "metal"`, as stated per point. Section numbers match the order
of the code.

### 6.1 The port probe

`main()` checks the port before the minutes of loading (`server.py:2442-2446`): it binds a `Server` and closes it. The
`Server` class sets `allow_reuse_address` on every non-Windows OS (`server.py:2226`).

BSD sockets, unlike Linux, let a `SO_REUSEADDR` bind succeed beside an existing listener when the addresses differ
(one wildcard, one specific). Verified on the reference Mac:

| Already listening | New probe | Upstream probe says | `connect_ex` to the port |
| --- | --- | --- | --- |
| `0.0.0.0:P` | `127.0.0.1:P` | free (wrong) | 0: in use |
| `127.0.0.1:P` | `0.0.0.0:P` | free (wrong) | 0: in use |
| `127.0.0.1:P` | `127.0.0.1:P` | in use | 0: in use |
| nothing | either | free | 61 (`ECONNREFUSED`): free |

The probe becomes a function:

```python
def port_in_use(host, port) -> bool:
    """Is something already serving this port?  A bind that fails says yes; on a Mac a bind that succeeds is not
    enough (MAC: BSD lets a SO_REUSEADDR bind succeed beside a listener on the wildcard or another address), so a
    connect to the port decides."""
    try:
        Server((host, port), BaseHTTPRequestHandler).server_close()
    except OSError:
        return True
    if sys.platform != "darwin":
        return False
    import socket
    targets = {"127.0.0.1"} | ({host} if host not in ("0.0.0.0", "", "::", "localhost") else set())
    for t in targets:
        with socket.socket(socket.AF_INET6 if ":" in t else socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.5)
            if s.connect_ex((t, port)) == 0:
                return True
    return False
```

`main()` calls `if port_in_use(a.host, a.port): ap.error(...)` with upstream's text. Linux and Windows keep their
behaviour exactly. The connect sends no data; a listener sees a connection that closes at once.

### 6.2 SIGHUP: closing the Terminal window

Closing a Terminal window (or quitting Terminal) sends SIGHUP to the foreground process group: setup's Python (when
setup started the server), the server, the engine and the vision helper all share it. Python's default action for
SIGHUP is to die at once. The engine would then be left to notice stdin EOF, and the server's shutdown (QUIT, then
terminate, then kill) never runs.

On darwin only (P9), `main()` installs a SIGHUP handler next to the SIGTERM handler (`server.py:2586-2593`):

```python
def on_sighup(signum, frame):              # MAC: the Terminal window closed; its tty is gone
    fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(fd, 1)
    os.dup2(fd, 2)                         # later prints must not fail with EIO on the dead tty
    raise KeyboardInterrupt                # the same path as Ctrl+C and SIGTERM: QUIT to the engine
```

Required alongside it:
- The engine ignores SIGHUP in `--serve` mode, as it ignores SIGINT ([06](06-engine-host-port.md), section 2.4). It
  then receives QUIT from the server and frees its memory in order.
- The vision helper may die from the SIGHUP. Its `close()` already handles a dead process (`server.py:605-611`).
- Setup's parent process (section 4.11, step 5) dies from the SIGHUP. That is harmless: it only waits for the server.

Linux keeps upstream's behaviour. A Linux port of this handler is a separate upstream decision.

### 6.3 Firewall text

When the host is not loopback, the ready text adds a Windows Firewall paragraph on `os.name == "nt"`
(`server.py:2574-2578`). On darwin, when `macos.firewall_on()` is True, it prints instead:

```
       nothing arrives? The macOS firewall asks once whether Python may accept incoming connections: click Allow.
       If you clicked Deny: System Settings > Network > Firewall > Options, set Python to "Allow incoming
       connections", or run:
         sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add <python>
         sudo /usr/libexec/ApplicationFirewall/socketfilterfw --unblockapp <python>
```

`<python>` is `os.path.realpath(sys.executable)`: the firewall rule attaches to the real binary, not to the venv's
symlink. A Homebrew Python upgrade changes that path, so macOS asks again after `brew upgrade`. The text says nothing
when `firewall_on()` is False or None.

`lan_addresses()` (`server.py:2253`) connects a UDP socket to `10.255.255.255` to learn the default route's address. It
sends nothing. Whether macOS 27's Local Network privacy prompt fires for it, and for incoming LAN connections to a
Terminal-started Python, is measurement item X-7 (section 14).

### 6.4 The RAM warning

`warn_tight_ram(arena_mib)` (`server.py:2233`, called at `:2491`) warns when RAM minus the experts' RAM leaves less than
6 GB. On a Mac every byte the model keeps resident counts, not only the experts. With `backend == "metal"`:
- the server passes `engine.info.get("plan_mib")`, the engine's planned total of resident memory (the plan's
  `total_mib`, a new INFO key, [06](06-engine-host-port.md), section 5.2), falling back to `arena_mib` from an older
  engine;
- `total` is `macos.ram_bytes()`;
- the text is:
  `[strata] WARNING: memory is tight - the model keeps N GB of this Mac's M GB resident, leaving L GB for everything
  else. macOS will compress and swap other apps' memory and may stop the engine if it runs out. Close other apps, or
  run ./setup.sh --setup and pick a shorter context or a smaller model.`

The threshold stays 6 GB. The doc-05 plan already keeps the RAM warning for plans above `memsize − 6 GiB`
([05](05-memory-and-residency.md), section 3.2), so the server and setup agree.

### 6.5 Why the engine died

`StrataEngine.death_note()` reads at most 4 KiB of log output written since that engine process started. It reports a
watchdog message, a useful final `strata` or `ERR` line after a normal exit, or the process signal. `SIGKILL` alone
does not identify why the process ended, whether or not the log grew. The server asks the user to check Console and
system memory pressure, then include the server log when reporting a failure. It does not identify an upstream issue
tracker as the project tracker.

| Condition | Diagnostic |
| --- | --- |
| `rc == -9` | Report SIGKILL without guessing the cause; check Console and system memory pressure. |
| another negative return code | Show the signal name and number. |
| watchdog line in new output | Report that the engine stopped making progress and include that line. |
| normal exit with a final `strata` or `ERR` line | Show the last line. |

The log offset is recorded from the append-only log file immediately before `Popen`. An old run's final line therefore
cannot be mistaken for this process's startup diagnostic.

### 6.6 The engine's environment

`child_env(cfg)` (`serve/server.py`) copies the current process environment, then applies only entries from `cfg.env`
(as strings). The same environment is passed to the Metal engine and, when configured, the vision encoder. It does not
select a GPU, rewrite a dynamic-library search path, or add CUDA/HIP visibility variables.

### 6.7 Advisory unified-memory estimate

`free_unified_memory_mib()` computes `min(W − ioreg in-use, available RAM)` for the Monitor and the legacy
`min_free_unified_memory_mib` warning. It is an estimate only. Measurement X-2 (section 14) has not established whether IORegistry's
system-wide `In use system memory` counter tracks the engine's allocations or how it accounts for other Metal apps.
Therefore it must never delay or refuse a model load. If the configured threshold is exceeded, the server prints an
advisory and continues. The engine's `--plan-only` remains the only model-fit decision (P3).

### 6.8 One engine per server, one model per Mac

Two Strata servers on one Mac each load a model into the same unified memory. Each engine plans against the full W
and does not know about the other. The second load then pushes the first into compression and swap, or one of them is
ended by jetsam. This is not supported:
- `start()` warns before the start (section 4.11, step 3);
- `--gpu-budget-mib` lets an expert split W by hand (section 4.14);
- [05](05-memory-and-residency.md), section 9, row "Several engines", refers to this section.

The server's own unload and reload (`idle_unload_s`, `before_load`) work unchanged and are the supported way to share
the Mac with another GPU app.

### 6.9 `keep_awake`

A Mac on battery or with the default Energy settings sleeps when idle, and a sleeping Mac answers no requests. A new
config key `keep_awake` (default false) asks the server to prevent idle sleep while the model is loaded:
- on darwin, after READY, the server starts `macos.keep_awake(os.getpid())`. caffeinate ends with the server.
- An unload (`idle_unload_s`) terminates the caffeinate process; the next load starts a new one.
- The setting is read from the config, with `--keep-awake` as a server flag that overrides it.
- When the host is not loopback and `keep_awake` is false, the ready text adds:
  `       this Mac may sleep when idle and then answers nothing: add "keep_awake": true to the config to prevent that
  while the model is loaded (closing the lid still sleeps a MacBook)`.

Off darwin the key is accepted and ignored. `caffeinate -i` prevents idle system sleep only. It does not keep the
display on and does not stop lid-close sleep. Whether App Nap or the macOS 27 background-task policy throttles a
Terminal-started engine while its window is hidden is measurement item X-5 (section 14).

### 6.10 The MCP hub's PATH

`serve/mcp.py` starts a stdio MCP server with `shutil.which(command, path=env PATH)` (`serve/mcp.py:93-108`). A
server started from Finder (`run-<model>.command`), from launchd or from another app's environment may not have
`/opt/homebrew/bin` or `/usr/local/bin` on PATH, so `npx` and `uvx` are not found. On darwin, `start()` appends (never
prepends) `/opt/homebrew/bin` and `/usr/local/bin` to the child's PATH when they are missing and exist. Appending
keeps the user's own order, and the system tools win.

### 6.11 The vision helper's temporary files

The vision helper exchanges images through `tempfile.mkdtemp(prefix="strata-vision-")` (`server.py:507`) and the
`ENC <img> <out>` line (`server.py:590`), whose fields are separated by spaces. Two Mac changes:
- **Spaces.** `$TMPDIR` on a Mac is `/var/folders/<xx>/<id>/T/`, without spaces. A user-set `TMPDIR` can have them,
  and the helper would then read a broken path. When `" " in str(self.dir)`, the server makes the folder with
  `mkdtemp(prefix="strata-vision-", dir="/private/tmp")` instead.
- **Cleanup by macOS.** `dirhelper` deletes files in `$TMPDIR` that nobody accessed for 3 days. A server that runs
  for days can therefore lose a cached `.sve`. The cache hit in `encode()` (`server.py:585-586`) must also check
  `self.cache[key][0].exists()`. A missing file drops the entry and encodes the image again.

The second change is safe on every OS and should go upstream.

---

## 7. `serve/telemetry.py` and the Monitor tab

Upstream reads the GPU through NVML (`_Nvml`) or amdgpu sysfs (`_Amd`). On a Mac neither exists, so the Monitor shows
no GPU at all (failure point 13). The CPU, RAM and disk readings already work through `psutil`.

### 7.1 The `_Apple` reader

A third reader class, after `_Amd`, with the same interface (`ok()`, `name()`, `read()`):

```python
class _Apple:
    """The Mac's GPU (MAC): load and memory in use from the IOAccelerator statistics, the GPU's working-set size W as
    its total.  macOS has no public reading of GPU temperature, power or PCIe, so those are None."""
    MIN_INTERVAL_S = 2.0                                # one ioreg call costs 25-80 ms

    def __init__(self, index=0):
        dev = macos.metal_devices()
        self.dev = dev[0] if dev else None
        self._at, self._last = 0.0, {}

    def ok(self):
        return self.dev is not None

    def name(self):
        return self.dev["name"] if self.dev else None

    def read(self):
        now = time.monotonic()
        if now - self._at >= self.MIN_INTERVAL_S:
            self._last, self._at = macos.ioreg_gpu(), now
        g = self._last
        return {"util": g.get("util"), "mem_used": g.get("in_use"), "mem_total": self.dev["working_set"],
                "temp": None, "power": None, "power_limit": None, "pcie_gen": None, "pcie_gen_max": None,
                "pcie_width": None, "pcie_rx_mb": None, "pcie_tx_mb": None, "recovery": g.get("recovery")}
```

| Reading | Source | Notes |
| --- | --- | --- |
| `util` | ioreg `Device Utilization %` | 0-100. Activity Monitor's GPU history shows the same counter. |
| `mem_used` | ioreg `In use system memory` | A system-wide byte counter. Its relation to the engine's allocation and other Metal clients remains unvalidated (X-2), so the UI labels it as an estimate. |
| `mem_total` | `recommendedMaxWorkingSetSize` (W) | Metal's recommended maximum working-set size. The pair is useful context, not a model-fit decision. |
| `recovery` | ioreg `recoveryCount` | GPU resets since boot. A change while the engine runs is a GPU error; the Monitor shows it (section 7.3). |
| `temp`, `power`, `power_limit`, `pcie_*` | None | The only sources are `powermetrics` (needs root) and private SMC keys. Neither may be used (P4, P6). |

### 7.2 Factory, sampler and static facts

- `gpu_reader(index=0, amd=False, backend=None)`: returns `_Apple(index)` when `backend == "metal"` or
  `sys.platform == "darwin"`; otherwise upstream's choice. A mock-engine server on a Mac (tests, development) therefore
  shows the Mac's GPU too.
- `free_unified_memory_mib(index=0, backend=None)` passes `backend` through. Its result is advisory telemetry only; see section 6.7.
- `Telemetry.__init__(..., amd=False, backend=None)` passes `backend` to `gpu_reader`.
  `Service.start_telemetry()` (`server.py:976`) passes `backend=getattr(self, "backend", None)`.
- `_cpu_name()` (`telemetry.py:202`) returns `macos.cpu_brand()` on darwin. Upstream's fallback,
  `platform.processor()`, returns `"arm"` there.
- `sample()` adds four readings on darwin:

| Key | Value | Source |
| --- | --- | --- |
| `thermal` | `"nominal"`, `"fair"`, `"serious"`, `"critical"` or None | `macos.thermal_state()` |
| `thermal_level` | 0-3, or None | the same, as a number for the sparkline |
| `mem_pressure` | `"normal"`, `"warn"`, `"critical"` or None | `macos.pressure_level()` |
| `swap_used` | bytes, or None | `macos.swap_used()` |

- `_loop()` records `thermal_level` and `swap_used` in the history on darwin only. The upstream history keys are
  unchanged, so `serve/test_monitor.py` keeps passing.
- `static` gains, on darwin only:
  `"unified": True`, `"gpu_cores": dev["cores"]`, `"perflevels": macos.perf_levels()`, `"computer": "Mac"`.
- `GET /health` (`server.py`) returns `"computer": "Mac"`, so the chat page can say "this Mac" before the Monitor
  has loaded. It uses the same API-key check as the other API routes. The browser sends its saved bearer key, leaves
  the current health state alone on an unauthorized response, and retries health and MCP after the key changes.
- `/v1/status`'s `machine.gpu` (`server.py:1111-1117`) works unchanged: name, used and total memory, load. `temp_c`
  and `power_w` are null.

### 7.3 The Monitor tab and the web texts

`serve/web/app.js` and `serve/web/index.html` use Mac-specific wording and readings. There is no PC UI variant.

| Element (upstream) | On a Mac |
| --- | --- |
| Card "Metal memory estimate" (`app.js`) | Value `gb(mem_used) / gb(mem_total) GB`. Sub explicitly identifies the system-wide IORegistry counter as an estimate. It does not claim model fit. |
| Card "Thermal state" | Value from macOS: `Nominal`, `Fair`, `Serious` or `Critical`. Sparkline: `thermal_level`, max 3. |
| Card "Memory pressure" | Value from `macos.pressure_level()`: `Normal`, `Warn` or `Critical`. |
| Card "macOS swap" | Current used swap in GB; sparkline from `swap_used`. |
| Card "CPU" sub (`app.js:263`) | The perflevels: `6 Super + 12 Performance cores` |
| Card "GPU load" sub | `<gpu_name> · <gpu_cores>-core GPU` |
| Bar "Resident experts" | Resident slot count and cache size, with the share shown against Metal's working-set limit. |
| Bar "Unified memory" | System RAM reading. A high used share alone is not treated as an error because macOS uses available memory for caches. |
| About, "Resident experts" | Resident slot count and cache size. |
| About, "KV cache" | Uses `unified memory` wording; it does not say VRAM. |
| About, "GPU" | Apple GPU name and Metal working-set limit when available; otherwise `not available`. |
| `runs on this PC` and `run on this PC` | `this Mac` |

`serve/web/monitor.html` and `serve/web/monitor.js` (the request monitor) read `/v1/status` and `/api/requests`
only. They show no hardware cards and need no change.

---

## 8. `tools/strata_mcp.py`

The MCP server exposes status, model inventory, an install preview, process start/stop for an existing configuration, logs, and connection details. The current implementation is macOS-only and refuses non-Apple hosts. Installation and automatic Metal planning remain unavailable; the setup tool is informational and does not write files. This section describes the intended Mac behavior and preserves the port plan, so its future-tense design details are not claims about the current checkout.

### 8.1 Shared machinery

| Item (upstream line) | Mac change |
| --- | --- |
| `proc_identity(pid)` (`strata_mcp.py:106-142`) | On darwin, use `macos.proc_identity(pid)`. It returns `mac:<sec>.<usec>`, or None for a zombie or a gone process. The start time distinguishes a reused PID when checked. Newly spawned children are probed repeatedly; if identity remains unavailable while the unreaped session leader is alive, stop its process group so setup/server descendants do not remain. If the leader has exited, only probe the old group ID and leave it untouched because ownership cannot be verified. POSIX still provides no atomic operation that both verifies this token and signals the process, so a live-leader PID signal has a narrow exit-and-reuse race; see D-IMPL-009 / REV-005. Do not describe this as an absolute guarantee. |
| `norm(p)` (`:366`) | On darwin, canonicalize the spelling of an existing path component only when `os.path.samefile()` confirms the differently cased spelling resolves to that same filesystem object. Do not blindly lowercase: case-sensitive APFS can contain distinct siblings such as `StrataApp` and `strataapp`, and merging them could incorrectly allow an external data directory. |
| `system_folders()` (`:371-377`) | A darwin list: `/System`, `/Library`, `/Applications`, `/bin`, `/sbin`, `/usr` (but not `/usr/local`), `/private/etc`, `/private/var` (but not `/private/var/folders`), `/dev`, `/cores`, `/opt/homebrew`, `/Volumes/Recovery`. `/private/tmp` and `/private/var/folders` (the temporary folders) are allowed. Upstream's list resolves `/var` to `/private/var` and refuses every temporary folder, which fails `test_data_dir_paths`. |
| `own_ram_gb()` (`:845-861`) | On darwin, `macos.ram_bytes() / 2**30` |
| `_hardware()` (`:580-636`) | On darwin, read physical RAM and CPU name from `serve/macos.py`; `hw["gpus"]` is `[dict(dev, vendor="apple", usable=problem is None, problem=macos.device_problem(dev))]` from `macos.metal_devices()`, with `vram_gb` rounded. `hw["metal"]` adds `working_set_gib`, `families`, `cores`, `macos`. The AMD and `nvidia-smi` probes are skipped. |
| `macos` import | `strata_mcp.py` always runs from a Strata checkout (`tools/`), so `serve/macos.py` is there before any install. Load it by file path with `importlib.util.spec_from_file_location`; do not mutate `sys.path`. `macos.py` runs on the client's Python (3.10+, section 8.4) without the venv, as it needs only the standard library (P4). |

### 8.2 `recommend()` and `model_table()`

Both follow P3: they never compute a fit in Python.

- When the engine is installed (`engine/strata` exists and BUILD.json says `backend: metal`), `recommend()` imports
  setup through `setup_module()` and calls `S.metal_plan()` for each model of each family at 32K context and int8 KV,
  as `--check` does (section 4.6). It returns setup's own recommendation (section 4.5.1, steps 1-5) and its
  `setup_command`, `./setup.sh --setup --yes --family F --model M --context C`. No `--backend` is needed, because
  metal is the darwin default. The family order is `FAMILIES`' insertion order; within a family it uses the model
  ordering and full/ok/slow/very slow classes above.
- Without an installed Metal engine and `setup.metal_plan()`, no plan is possible. `recommend()` returns no family or
  model and reports that it cannot safely choose from the PC RAM table; RAM and W are notes for orientation only.
  `strata_install` with no model refuses automatic selection until setup can produce a Metal plan.
- Once the planner is available, `recommend()` runs its PLAN for each model and the contexts needed by section 4.5.
  It follows the setup menu's class ordering and returns the same `setup_command`. No fit is inferred from total RAM
  or from `recommendedMaxWorkingSetSize`.
- The 32K model-menu PLAN remains the displayed class. If it is `none` or `unsupported`, setup also checks the 8K
  PLAN; the model is selectable only when that 8K plan fits. `model_table()` reports `fits at 8K only` in this case.
  The install preview checks the 8K plan and then the exact requested context, so a larger context that does not fit
  is still refused.
- `model_table()` sets `on_this_pc` from the same plans (`plan_class` text, section 4.4.2), or
  `"run ./setup.sh --check (or install) to see"` before install. It does not display the PC RAM estimates on Mac;
  `ram_needed_gb` and `experts_gb` are null. The Unsloth family is marked unavailable and excluded from Mac
  recommendations and install choices.
- For an install preview, the same planner receives the requested KV mode and image mode (`yes` normalizes to `gpu`);
  it never plans one configuration and then displays another. With context at or below 8K, setup forces FP16 KV and
  omits `--kv`; the preview uses that same effective mode. An available planner admits a model when its 32K plan fits
  or, as a last resort, its 8K plan fits; it then checks the exact requested context, vision and KV combination. A
  model whose 32K and 8K plans are both `none` or `unsupported`, or whose requested context does not fit, is refused
  before install starts. `--low-ram off` additionally requires a full-residency PLAN. If the Metal
  planner is unavailable, an explicit model/context preview reports fit as unchecked and never substitutes a RAM
  estimate.
- The Mac refusals of section 4.3 (Intel, Rosetta, macOS before 27, no Metal 4) come through
  `macos.device_problem()` and `S.mac_step1`'s texts as `why`.

### 8.3 Install, start, stop

| Item | Mac change |
| --- | --- |
| `install_plan` (`:1188`), backend and GPU checks (`:1193-1195`) | `cuda` and `hip` are refused on darwin. `auto` and `metal` use the Metal default; `--gpu 0` is accepted, other GPU IDs and `--gpus` fail. |
| Python check (`:1196-1198`) | Before setup or `.venv` creation, require Python 3.10+, native `arm64`/`aarch64`, and `macos.translated() == False`. Refuse x86_64, Rosetta or an unavailable architecture probe with an install/configuration hint. The version error is `this Python is too old: Strata needs Python 3.10+; install it with "brew install python@3.13" (or from python.org) and configure your MCP client to run strata_mcp.py with it`. |
| Default context (`:1216-1220`) | From `recommend()`'s context on darwin, not from VRAM tiers |
| `.venv` creation (`:1660-1686`) | Before `venv`, the darwin path refuses an x86_64 or Rosetta Python and checks Python 3.10+; it never creates an incompatible environment first. The Linux hint at `:1680` gets a Mac twin: `install native arm64 Python 3.10+ (brew install python@3.13), or run ./setup.sh once in Terminal`. |
| `check_data_dir` (`:1116-1135`) | The examples become `~/Strata-data` or `/Volumes/External/Strata-data` on darwin (also in the tool schema, `:970-973`, and `:1235`). MCP refuses explicit and configured/default paths inside iCloud Drive, CloudStorage, Desktop, Documents or Downloads with the reason of section 4.12. |
| `strata_start`, `strata_stop` | `strata_stop`'s SIGTERM takes the server's KeyboardInterrupt path. If escalation is needed, it checks the leader's identity before signaling its process group; POSIX does not make that check and signal atomic (REV-005). After the code observes that leader exit, it only probes whether the old group ID remains and does not signal a group whose ownership can no longer be verified. The result reports possible leftovers. |
| `INSTRUCTIONS` (`:1539`) | `runs ... locally on this PC's GPU and RAM` becomes `on this computer's GPU and memory` (one text for all) |
| `strata_benchmark` note | `the GPU's free VRAM` becomes `free GPU memory` on darwin |

The Mac model/install tool descriptions are platform-specific: no PC RAM fit is shown, the backend choices are
`auto` and `metal`, and Unsloth is omitted because its kernels are not available on this port. An install preview
returns a `fit` from the exact Metal PLAN; its legacy `ram` field is null on Mac. The default install uses the
recommended family/model/context from the planner. An explicit context is still checked against its exact PLAN
before the user can confirm an install.

### 8.4 Client configuration on a Mac

The former `docs/MCP_SERVER.md` user guide was removed with the PC setup documentation. The Claude Desktop configuration below remains a useful example; the current installer tool only returns a preview and cannot install the runtime. The configuration file is
`~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{"mcpServers": {"strata": {"command": "/opt/homebrew/bin/python3",
                           "args": ["/Users/<you>/Strata/tools/strata_mcp.py"]}}}
```

A GUI app starts MCP servers with launchd's minimal PATH (`/usr/bin:/bin:/usr/sbin:/sbin`), so `"command":
"python3"` would find Apple's 3.9 (or open the Command Line Tools dialog) and `main()` would refuse to run
(`strata_mcp.py:1698`). The command must be an absolute path to a 3.10+ Python: Homebrew's
`/opt/homebrew/bin/python3`, python.org's `/Library/Frameworks/Python.framework/Versions/3.13/bin/python3`, or, after
an install, Strata's own `<Strata>/.venv/bin/python`.

---

## 9. Tools

| Tool | Mac |
| --- | --- |
| PC CUDA calibration utility | Removed with its PC-specific settings and tests (D-IMPL-236). No Mac command or test remains. |
| `tools/iq_pack.py`, `tools/mtp_fetch.py`, `tools/mtp_pack.py`, `tools/mtp_rt.py` | Unchanged. Their tests pass on the reference Mac (`test_iq_pack` with `STRATA_GGUF_PY`). |
| `tools/strata_pack.py`, `tools/pack_index.py` (the AVX-512 Q2_0 repack) | Never run by setup on a Mac (section 4.8). They stay usable by hand. |
| `tools/needle_bench.py`, the conversation-cache tools without a `test_` prefix | Unchanged. They need a running server and become Mac engine acceptance tests ([09](09-testing-and-validation.md)). |
| `tools/make_geometry.py` | New (section 4.4.3) |
| `tools/hit_curve.py` | New; measures the expert hit curve for M-9 ([05](05-memory-and-residency.md), section 10.1) |
| `tools/metal/check_drift.py` | New ([06](06-engine-host-port.md), section 1.3). For the Python side it also fails when an upstream change touches a line next to a `# MAC:` marker, or when a new `subprocess`/`os.name`/`sys.platform` branch appears in a file this document patches ([10](10-implementation-plan.md), section 4) |
| `tools/record_mac_plans.py` | New; records the PLAN fixture of the Mac golden set (section 10.2) |
| `chat.py` | Unchanged |

---

## 10. Tests

All Python tests run with `python -m unittest <module>` from the repository root, as upstream. None needs a GPU. The
engine tests and the hardware acceptance runs are in [09](09-testing-and-validation.md).

### 10.1 Upstream tests on a Mac

At the baseline, on the reference Mac, every non-setup module passes except `tools.test_strata_mcp`. Three setup
modules fail, for reasons that are not Mac bugs:

| Module | Result on a Mac | Cause | Fix |
| --- | --- | --- | --- |
| `tools.test_setup_golden` | 46 failed subtests, all on the `log` key | `normalize()` (`tools/test_setup_golden.py:51-59`) replaces `setup.EXE` anywhere in a string. On POSIX `EXE` is `strata`, a substring of every log path (`<T>/strata-iq3_xxs.log` becomes `<T>/<EXE>-iq3_xxs.log`). It fails on Linux too. | Replace only a whole last path component, and strip the resolved temporary path (`/var` resolves to `/private/var`): see below. Should go upstream. |
| `tools.test_setup_amd` | `test_prebuilt_hip_zip` fails (`:264-292`) | The fixture zip holds `strata.exe`, but `setup.EXE` is `strata` off Windows, so `get_prebuilt_hip` refuses it. It fails on Linux too. | Patch `setup.EXE` to `"strata.exe"` inside the test's `with` block. Should go upstream. |
| `tools.test_strata_mcp` | 3 failures and 1 error | The zombie and system-folder bugs (section 8.1) | No test change: the section 8.1 code fixes them |

The `normalize()` fix:

```python
if isinstance(v, str):
    s = v.replace(str(t.resolve()), "<T>").replace(str(t), "<T>").replace("\\", "/")
    return re.sub(r"(^|/)" + re.escape(setup.EXE) + r"$", r"\1<EXE>", s)
```

**The PC tests keep testing the PC on a Mac.** The upstream setup tests describe NVIDIA and AMD PCs. Run on a Mac
with the MAC branches on, they would test the Mac path against PC goldens. Each setup-importing upstream test module
gets one line that turns the branches off for the module. At the v0.1.34 baseline,
`setup.py` does not define `MAC` yet; `create=True` lets these Phase 0 tests run
before S3 adds that constant. Remove `create=True` with S3 so a later regression
cannot silently recreate a missing production setting.

```python
def setUpModule(): p = mock.patch.object(setup, "MAC", False, create=True); p.start(); unittest.addModuleCleanup(p.stop)  # MAC:
```

The modules: `test_setup_amd`, `test_setup_choices`, `test_setup_draft_vocab`, `test_setup_golden`,
`test_setup_lowram`, `test_setup_pins`, `test_setup_prompts`, `test_setup_risk`, `test_setup_rope`,
`test_shards`, the MTP runtime tests and `test_mtp_fetch`. None of them defines `setUpModule` at the
baseline. `setup.MAC` is the only switch, because every Mac branch tests it (P1). With it False, no `serve.macos`
function is called.

**Implementation status (2026-10-03).** The two setup-test fixes and all 13 PC-path patches are in place. The
baseline MCP failures were fixed by adding `serve/macos.py` and using its process identity on darwin, plus by allowing
the documented temporary-folder exceptions in `data_dir` validation. The full `tools/metal/run_python_tests.py` run
passes all 37 modules on the M5 Pro reference Mac; tests that need model packs or optional `jsonschema` remain skipped
by their existing conditions. This completes the upstream-test portion of S17 on this host. S15's macOS hardware
probes, path validation and Metal-planner integration are implemented; the Metal planner itself is a later setup
work item, so planner-backed recommendations still need end-to-end verification after that work lands.

### 10.2 The Mac golden set

New files: `tools/test_setup_golden_mac.py`, `tools/test_setup_golden_mac.json` and
`tools/test_setup_golden_mac_plans.json`. They run on any OS: the Mac is fully mocked.

**Profiles.** Each profile patches `setup.MAC = True`, `ram_gb`, `macos.metal_devices`, `macos.macos_version`,
`macos.translated`, `macos.low_power_mode`, `macos.engine_pids` (`[]`) and `macos.cpu_brand`:

| Profile | RAM (GiB) | W (GiB) | Families | GPU cores | What it covers |
| --- | ---: | ---: | --- | ---: | --- |
| `mac-24GB` | 24 | 16.0 | apple9, metal4 | 10 | below the floor: `confirm_risk`, and every model `none` or very slow |
| `mac-32GB` | 32 | 24.0 | apple10, metal4 | 10 | the stretch class |
| `mac-36GB` | 36 | 27.0 | apple9, metal4 | 14 | tiered |
| `mac-48GB` | 48 | 36.0 | apple10, metal4 | 20 | primary target, tiered Q2_0 |
| `mac-64GB-M5Pro` | 64 | 51.84 | apple7-10, metal3, metal4 | 20 | the reference Mac: W 55,662,788,608 B, max buffer 41,747,087,360 B |
| `mac-96GB` | 96 | 72.0 | apple9, metal4 | 38 | full residency, long context |
| `mac-128GB`, `mac-256GB`, `mac-512GB` | 128, 256, 512 | 96, 192, 384 | apple9, metal4 | 40, 60, 80 | everything full |
| `mac-intel` | | | | | `translated` False, no device: the Intel refusal |
| `mac-rosetta` | 64 | | | | `translated` True: the Rosetta refusal |
| `mac-26` | 64 | 51.84 | | 20 | `macos_version` (26, 4): the macOS refusal |
| `mac-m0` | 16 | 10.7 | metal3 only | 8 | `device_problem`: no apple7 / metal4 |

**Plans come from the engine, not from the test (P3).** `metal_plan` is patched with a lookup in
`test_setup_golden_mac_plans.json`. The key is the profile's W and RAM plus the plan arguments (family, model,
context, KV, vision, budget, files or geometry). A missing key fails the test with `no recorded plan for ...: run
tools/record_mac_plans.py`.

`tools/record_mac_plans.py` (new) records the fixture on any Mac with an engine. For every key the golden tests ask
for, it runs:
```
engine/strata --plan-only --geometry data/geometry/<tag>.txt --plan-w-mib <W> --plan-ram-mib <RAM> <plan args>
```
`--plan-w-mib` and `--plan-ram-mib` are hidden engine flags that replace the device's W and `hw.memsize`, only
together with `--plan-only` ([06](06-engine-host-port.md), section 3.10). The fixture stores the engine version and
`metal_rev` that produced it. A plan change in the engine therefore shows up as a fixture diff in review, never as a
silent change of setup's choices.

**What the goldens assert**, per profile and model (with `--yes`, and with Enter for every question):
- the menu lines and the recommended model, context and KV (section 4.5);
- the config: `backend == "metal"`, `lib_dirs == []`, a `plan` object, and no `gpu`, `gpus_asked`, `layer_split` or
  `env` key;
- the args: no `--kv-resident`, `--resident-experts`, `--mmap-experts`, `--pcie-frac` or `--layer-split`;
  `--vram-reserve-mib 2100` with images on the GPU;
- the removed PC calibration utility and its CLI flag do not reappear in the Mac product;
- the two run scripts (section 4.10), through a patched `write_run_script` that records its arguments;
- the refusals: the four refusal profiles, `--yes` alone below 32 GB (it stops, as upstream's
  `test_the_refusals_with_yes_alone_stay`), an explicit `--model` below 32 GB with `--yes` (it proceeds: consent, P7),
  `--family unsloth`, `--gpu 1`, `--gpus 0,1`, `--layer-split`, `--backend cuda`, `--backend hip`;
- `tools/strata_mcp.py` on each profile (section 8.2), with `engine/strata` and a metal BUILD.json faked: `recommend()`
  returns setup's recommendation and its `setup_command`, `model_table()`'s `on_this_pc` is the plan class of the
  fixture, and without the engine `recommend()` returns its `install first` answer.

### 10.3 `serve/test_macos.py`

New. Each test is skipped off macOS unless its name says it runs everywhere.

| Test | What it checks |
| --- | --- |
| `test_no_raise_anywhere` (everywhere) | With `macos.MAC` patched False, every public function returns its documented empty value. On a Mac with `MAC` True, every function returns the documented type. |
| `test_proc_identity_zombie` | `Popen(["/usr/bin/true"])`, wait 0.2 s without reaping: `proc_identity` is None. After `wait()`: None. For `os.getpid()`: `mac:<s>.<us>`, stable across two calls. |
| `test_system_folders` | `tempfile.mkdtemp()` and `/private/tmp/x` are allowed; `/System/x`, `/Library/x`, `/private/etc/x` and `/usr/bin` are refused; `/usr/local/x` is allowed |
| `test_port_probe` | The four rows of section 6.1 with real sockets on a free port |
| `test_metal_plan_parse` (everywhere) | A fake engine (a shell script) prints PLAN lines: rc 0 and rc 3 parse, `reason_raw` keeps underscores and `reason` restores spaces; rc −9, `Library not loaded` on stderr, and a timeout give the hints of section 4.4.1 |
| `test_ssl_setup` | With `ssl.get_default_verify_paths` patched to a cafile of None and `SSL_CERT_FILE` unset, `ssl_setup()` sets `SSL_CERT_FILE` to an existing file |
| `test_sigint_handler` | A parent Python with `signal.signal(SIGINT, lambda *a: None)` around `subprocess.call` of a child that waits for KeyboardInterrupt: `os.killpg(SIGINT)` reaches the child, the child exits 0, and the parent goes on. With `SIG_IGN` instead, the child does not see it (this pins why section 4.11 forbids `SIG_IGN`). |
| `test_sighup_shutdown` | `serve/server.py --engine mock` in its own session; `SIGHUP` to it: it exits 0 within 10 s and the port is free again |
| `test_engine_swap` | Copy `/usr/bin/true` to a temporary `strata`, sign it ad hoc (`codesign -s - -f`), run it; then put a second signed copy in place with `os.replace` and run again: rc 0 both times |
| `test_setup_sh` | Copies `setup.sh` and a stub `setup.py` that prints `sys.version_info` and `platform.machine()` into a temporary folder: no `.venv` makes one and runs arm64 3.10+; a second run reuses it; a `.venv` from `/usr/bin/python3` (skipped when absent) is made again. Skipped without a 3.10+ arm64 Python. Two more cases run without one: a copy whose three absolute candidates point into an empty temporary folder, run with a PATH that holds only a `python3` stub reporting 3.9, prints the install hint of section 5.1 and exits 1; and in the Mac block, from the `Darwin` line to its `fi`, no line outside an `echo` runs `sudo`, `brew`, `installer`, `apt`, `dnf` or `pacman`, and `/usr/bin/python3` appears only in `mac_py`'s `case` pattern. |
| `test_module_rules` (everywhere) | `ast.parse(..., feature_version=(3, 9))` accepts `serve/macos.py`; every import is `__future__` or in `sys.stdlib_module_names`; every `subprocess` call whose first argument is a literal list starts with an absolute path (section 3.1) |
| `test_command_launchers` | `START-HERE.command` and `SETUP.command` have git mode `100755` (`git ls-files -s`) and LF line endings. Copied with a stub `setup.sh` that prints its arguments into a temporary ROOT with the name of `test_run_script_quoting`, and run as `/bin/sh <file> a` from another folder: `START-HERE.command` passes `a`, `SETUP.command` passes `--setup a` (section 5.2). |
| PC calibration tests | Removed with the PC-only calibration utility (D-IMPL-236) |
| `test_run_script_quoting` | `write_run_script` into a temporary ROOT named `Strata $x "q" ` + "`b`": `sh -n` accepts both scripts, and running the `.sh` with a stub Python prints the exact arguments |
| `test_data_folder` (everywhere) | `setup.MAC` patched True and `HOME` set to a temporary folder: a ROOT under `~/Documents`, `~/Desktop`, `~/Downloads`, `~/Library/Mobile Documents` or `~/Library/CloudStorage` gives `~/Strata-data` and the ok line; a ROOT of `~/src/x` gives `~/src/Strata-data`; a folder from the settings file is kept; creating the folder writes an empty `.metadata_never_index` and calls `macos.tm_exclude` (patched) once (section 4.12) |
| `test_death_note` (everywhere) | With `macos.MAC` patched True and a temporary log, `rc` −9 gives the same cause-neutral SIGKILL text with either an unchanged or grown log; `rc` −11 gives `The engine ended on signal SIGSEGV (11).`; a watchdog line is reported with the line. The server log offset excludes earlier runs. |
| `WebHealthAuthContract` | The page sends its saved key to `/health`, keeps the previous state after a non-OK response, avoids retrying a 401 until the key changes, and reloads health and MCP after the key is saved. |
| `test_keep_awake` | A `Service` with `keep_awake` true and a stub engine that has `alive()` and `unload()` (upstream's `MockEngine` has no `unload`): after `ensure_loaded()`, `pgrep -P <test pid> caffeinate` finds one `caffeinate -i -w <test pid>`; after `unload()` it is gone; a second `ensure_loaded()` starts a new one. With `keep_awake` false, none runs (section 6.9). |
| `test_geometry_plans` | Skipped without `engine/strata`: every `data/geometry/<tag>.txt`, planned at the points of its `.check` file, gives the recorded PLAN lines key for key (section 4.4.3) |
| `test_geometry_equals_files` | Skipped unless `STRATA_TEST_MODELS` points at downloaded models: for each, the geometry plan equals the live real-files plan at three W values. Part of the parity run (section 4.4.3). |

### 10.4 Where the tests run

- **macOS 27+ arm64** (CI and reference Mac): the upstream modules (MAC False), the Mac golden set, and the "everywhere" tests of
  `serve/test_macos.py`.
- **A GitHub-hosted macOS arm64 runner**: all of the above plus `serve/test_macos.py`. The runners have 7 GB of RAM
  and a paravirtualized GPU, so they never load a model. Whether their GPU passes `--list-devices` is measurement
  item X-8 (section 14).
- **The reference Mac** (self-hosted): everything, with `STRATA_TEST_MODELS` set, and the hardware acceptance runs of
  [09](09-testing-and-validation.md).

---

## 11. Project documentation

This repository supports Apple Silicon Macs running macOS 27 or later. Root installation guides for Windows, Linux,
CUDA/HIP and Docker have been removed. The root README and `docs/mac/` describe the current Mac-only source tree and
its implementation status.

The setup flow is still in progress. Until the Mac-specific installer, native model planning and model-backed
acceptance tests pass, documentation must say so and must not present `setup.sh` as a completed installer. Add a
standalone Mac user guide only when it can describe a tested end-to-end install, start, API check and update. Keep
engineering detail in this directory; do not preserve duplicate PC-specific user manuals.

## 12. Python and dependencies

| Item | Rule |
| --- | --- |
| Minimum | Python 3.10, as upstream (`setup.sh:8`, `strata_mcp.py:1698`, `requirements.txt:3`). `serve/macos.py` alone also runs on 3.9 (section 3.1). |
| Tested | 3.13 (Homebrew 3.13.12 on the reference Mac). CI also runs 3.10. |
| Recommended source | Homebrew (`brew install python@3.13`): arm64, with a working CA bundle |
| python.org installer | Works. It is universal2, so it needs `arch -arm64` from a Rosetta shell (section 5.1). Without "Install Certificates.command", HTTPS fails until `ssl_setup()` (section 4.7.4). |
| uv-managed or Conda/Miniforge Python | Works (arm64). Not the default advice. |
| `/usr/bin/python3` | 3.9.6: refused by `setup.sh` and never run (section 5.1) |
| `requirements.txt` | Every retained pin must have a macOS arm64, universal2 or pure-Python wheel for the supported Python versions. The former Windows-only `colorama` pin was removed. Recheck wheel availability when changing a pin. |
| `psutil` | Optional for server startup; CPU, RAM and disk readings are absent if it cannot be imported. It is pinned for the supported monitor path. |
| CUDA/HIP wheels | Removed from the supported setup path. |
| pyobjc | Not used (P4) |
| Build tools | `--build` needs Xcode 27 with the Metal toolchain, and the venv's cmake and ninja (section 4.7.6). Homebrew is not needed. |

---

## 13. Docker

Docker support has been removed. The Metal engine needs a native macOS process and Apple GPU; the product has no
Dockerfile, container entrypoint or supported Linux container setup path.

---

## 14. Measurements and open items

Each X item has a provisional default in this document, so implementation does not wait for it. The measurement
replaces the default, or confirms it. Results go to [appendix-a-measurements.md](appendix-a-measurements.md).

| ID | Question | How to measure | Provisional default | Decision it drives |
| --- | --- | --- | --- | --- |
| X-1 | GPU memory taken by `strata-vision` with `--vision gpu` (1024 max tokens) | On the reference Mac, encode a 4K image and a 1024-token image. Read the helper's `phys_footprint` (`footprint -p <pid>`), and the drop in `ioreg` "In use system memory". | 2100 MiB reserve (section 4.1) | `VISION_METAL["gpu"]["reserve_mib"]` |
| X-2 | Does `ioreg` "In use system memory" track the engine's `currentAllocatedSize`, and does it count another app's GPU memory? | Load Q2_0 and compare both every 2 s (INFO plus the Monitor). Then open a 4 GiB Metal allocation in a second process and watch both. | Display the formula `free = min(W − in_use, available RAM)` as advisory telemetry only (section 6.7) | Whether the formula can be used for any automated load decision; Monitor labels |
| X-3 | `--plan-only` time | `time engine/strata --plan-only ...` 20 times, cold (after `purge`) and warm, with a geometry file and with real files | under 200 ms (geometry) and under 1 s (files) (section 4.4.4) | Whether `metal_plan`'s `lru_cache` is enough or menus need a progress line |
| X-4 | Does `caffeinate -i -w` keep a MacBook awake on battery and on AC during a 2-hour download with the display asleep? | `pmset -g assertions` during the run; `pmset -g log` afterwards for sleep events | caffeinate during install runs only (section 4.13) | Whether setup also needs `-s` on AC, or a text about the power adapter |
| X-5 | Does App Nap or macOS 27 background throttling slow a server started from a hidden Terminal window, or from a LaunchAgent? The engine holds an `NSProcessInfo` activity during requests ([06](06-engine-host-port.md), section 2.2); the Python server holds none, and the engine holds none while it loads. | Time to first token, decode tok/s and load time with the Terminal window visible, hidden and minimised, and from the LaunchAgent of section 11.2. Same question as [06](06-engine-host-port.md) H-8 for the engine. | No activity in Python; `ProcessType` `Interactive` in the plist | Whether the engine also holds its activity during load and warm-up, and whether the server needs `keep_awake` by default |
| X-6 | `setup.sh` from a Rosetta shell | On a Mac with Rosetta: `arch -x86_64 /bin/zsh -c ./setup.sh` with Homebrew arm64 Python, with python.org universal2 Python, and with an Intel-only Python | arm64 run, or the Rosetta refusal (section 5.1) | The `setup.sh` rules table |
| X-7 | macOS 27 Local Network privacy prompt | With `--host 0.0.0.0`, connect from another device; also run `lan_addresses()` (a UDP `connect`, no packet sent). Note any prompt and which binary it names. | No prompt expected for a listening socket; text in section 6.3 | The firewall/Local Network text of section 6.3 |
| X-8 | Does a GitHub-hosted macOS arm64 runner pass `strata --list-devices` (apple7 and metal4 families on its paravirtualized GPU)? | One CI job | Engine tests run only on the reference Mac (section 10.4) | Whether engine unit tests run in hosted CI ([08](08-build-and-distribution.md), [09](09-testing-and-validation.md)) |

Items in other documents that change values here:

| Item | Document | Values here that it changes |
| --- | --- | --- |
| M-3, the default reserve | [05](05-memory-and-residency.md), section 10; [06](06-engine-host-port.md), section 15 | The `700` in `metal_plan`'s `--vram-reserve-mib` (section 4.4.1) and `VISION_METAL` |
| M-8, W and `maxBufferLength` on 32, 36 and 48 GB Macs, and the effect of `iogpu.wired_limit_mb` | [05](05-memory-and-residency.md), section 10 | The W column of the golden profiles (section 10.2: 0.67 of RAM at 24 GB, 0.75 above, from published values) and the wired-limit advice (section 4.5.6) |
| M-9, the expert hit curve, and gate G2 | [05](05-memory-and-residency.md), section 10.1; [00](00-feasibility-study.md), section 7 | `METAL_SHARE_OK` (0.85) and `METAL_SHARE_SLOW` (0.40) (section 4.1), so the menu classes of section 4.4.2 |

Open items, decided outside measurement:

| Item | Decided when | Until then |
| --- | --- | --- |
| GitHub publication destination (`owner/repository`) and release-test repository (OQ-01) | Before release automation or publication | No remote is configured; leave `setup.RELEASE_REPOSITORY` unset and defer remote checks. `strata.app` is only a proposed public slug. |
| Whether `normalise()` and `test_prebuilt_hip_zip` fixes (section 10.1) go upstream | Before the first release | They live in the project with `# MAC:` markers |

---

## 15. Work items for this area

Each item becomes one task in [10-implementation-plan.md](10-implementation-plan.md). An item is done when its
"Done when" holds on the reference Mac; no Linux or Windows product support is required (P9).

| # | Item | Done when |
| --- | --- | --- |
| S1 | `serve/macos.py` from the spike (section 3) | `python3 serve/macos.py` prints every probe on the reference Mac and on Linux; `test_no_raise_anywhere` passes on both |
| S2 | `setup.sh` darwin block and the two `.command` launchers (section 5) | `test_setup_sh` passes; double-clicking `START-HERE.command` in Finder runs setup in arm64 3.10+; the verified-cases table of section 5.1 reproduces |
| S3 | setup constants, probes and the Mac step 1 (sections 4.1-4.3) | The four refusal profiles give the texts of section 4.3; `--check` on the reference Mac prints the facts of section 3.4 |
| S4 | `metal_plan`, `plan_class` (section 4.4.1-4.4.2) | `test_metal_plan_parse` passes; with the engine of [06](06-engine-host-port.md) H4/H5, `metal_plan` returns a plan for every model on the reference Mac |
| S5 | Geometry files: the engine's `--geometry`/`--write-geometry`, `tools/make_geometry.py`, the 7 files and their `.check` files (section 4.4.3) | `test_geometry_plans` passes; the CI Content-Length check passes; `test_geometry_equals_files` passes in the nightly parity run (section 4.4.3) |
| S6 | Menus, consent, `--check`, `--low-ram`, `--resident-budget-gib`, wired-limit advice (sections 4.5-4.6) | The Mac golden set's menu and refusal assertions pass for every profile |
| S7 | Engine install: `get_prebuilt_metal`, BUILD.json, no CUDA wheels, `ssl_setup`, smoke test (section 4.7.1-4.7.5) | A fresh checkout on the reference Mac installs the release asset and passes the smoke test; with the python.org Python before "Install Certificates", downloads work |
| S8 | `build_engine_metal` (section 4.7.6) | `./setup.sh --build` on the reference Mac with Xcode 27 builds an engine whose BUILD.json says `source: local` and passes the smoke test; without Xcode it stops with the section 4.7.6 text |
| S9 | Step 6 and 7 on a Mac: pack, MTP, disk, args, final plan, config (sections 4.8-4.9) | The Mac golden set's config and args assertions pass; a real install of Q2_0 writes a config that the server starts |
| S10 | Run scripts, `.command` files, quarantine (section 4.10) | `test_run_script_quoting` and `test_command_launchers` pass; a ZIP download of the checkout starts after one "Open Anyway" |
| S11 | `start()` on a Mac, data folder, sleep, flags and texts (sections 4.11-4.16) | Every flag row of section 4.14 behaves as written (a table-driven test in the golden set); `test_data_folder` passes |
| S12 | `serve/server.py` changes (section 6) | `test_port_probe`, `test_sighup_shutdown`, `test_death_note` and `test_keep_awake` pass; `kill -9` of the engine after READY gives the cause-neutral SIGKILL diagnostic in section 6.5 |
| S13 | `serve/telemetry.py` `_Apple` reader and sampler (sections 7.1-7.2) | `/metrics` and `/v1/status` on the reference Mac show the readings of section 7.1; off darwin their keys are identical to upstream |
| S14 | Monitor and About texts (section 7.3) | Screenshot review of the Monitor and About tabs on the reference Mac against the section 7.3 table |
| S15 | `tools/strata_mcp.py` (section 8) | `tools.test_strata_mcp` passes on the reference Mac with no test change; `strata_models` returns engine plans (the MCP case of the golden set, section 10.2); Claude Desktop with the section 8.4 config installs and starts a model |
| S16 | PC calibration | Withdrawn: no CUDA calibration utility is shipped on macOS (D-IMPL-236) |
| S17 | Retained Python tests and Mac launcher checks (section 10.1) | Every retained Python test module passes on the reference Mac; tests for removed Windows/HIP installers are deleted |
| S18 | The Mac golden set and `tools/record_mac_plans.py` (section 10.2) | The golden set passes on the reference Mac; re-recording with the same engine gives no fixture diff |
| S19 | `serve/test_macos.py` (section 10.3) | Every required test passes on the reference Mac |
| S20 | Mac-only project documentation (section 11) | Root and docs describe the supported target and clearly mark any unimplemented setup or inference path |
| S21 | Measurements X-1 to X-8 (section 14) | Each result is in appendix A and the default it drives is updated here |

---
