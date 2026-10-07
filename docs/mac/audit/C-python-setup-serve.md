> **Audit report: research input, kept verbatim.**
> Upstream baseline: [Niko1221/Strata](https://github.com/Niko1221/Strata) at commit `1678de33` (tag v0.1.34, 2026-10-02).
> All `file:line` references are relative to the upstream repository root. `/tmp/strata-upstream` is the read-only
> clone used during the audit, and other `/tmp/...` paths are scratch files from the audit, not part of this repo.
> The curated design documents in [`docs/mac/`](../) summarise this report and take precedence where they differ.

# C. Strata outside the native engine: macOS (Apple Silicon, macOS 27) porting inventory

Scope: everything that is not the C++/CUDA/HIP engine. That is the installer (setup.py, setup.sh, the .bat files, Docker), the
Python server (serve/*.py, serve/web/*), the MCP server (tools/strata_mcp.py), the tools
(calibrate.py, needle_bench.py, _paths.py), the tests, and the user docs (INSTALL, AI_SETUP, MCP_SERVER, TROUBLESHOOTING;
DETAILS where it repeats them).

Line references use the repository root `/tmp/strata-upstream`, which is read-only and was not modified. `setup.py:L2716`
means `/tmp/strata-upstream/setup.py`, line 2716.

Evidence was collected on the live Mac below:
- All tests ran on a scratch copy, `/tmp/strata-research/scratch/strata`, using the Homebrew Python 3.13 venv at
  `/tmp/strata-research/scratch/venv`. The logs are `/tmp/strata-research/scratch/*.log`.
- Probes:
  - `/tmp/strata-research/scratch/metal_probe.py`: stdlib ctypes Metal device query.
  - `/tmp/strata-research/scratch/libproc_probe.py`: process identity via libproc.
  - `/tmp/strata-research/scratch/um_table.py`: the unified-memory fit table in section 2.
  - `/tmp/strata-research/mtlinfo.swift`: Swift cross-check of the Metal values.

---------------------------------------------------------------------------------------------------------------------

## 0. Summary, environment, blockers

### 0.1 Live reference machine

| Fact | Value | How read |
|---|---|---|
| OS | macOS 27.0 (`sw_vers` build 26A428), Darwin 27.0.0 | `sw_vers`, `uname` |
| Model | Mac17,9, "Apple M5 Pro" | `sysctl hw.model machdep.cpu.brand_string` |
| CPU | 18 cores: perflevel0 "Super" x6 + perflevel1 "Performance" x12; no SMT, so `os.cpu_count()` = `psutil.cpu_count(logical=False)` = 18 | `sysctl hw.nperflevels hw.perflevelN.name hw.perflevelN.physicalcpu` |
| CPU features | FEAT_DotProd, FEAT_I8MM, FEAT_BF16, FEAT_SME = 1; no AVX anything | `sysctl hw.optional.arm.FEAT_*` |
| RAM | 68,719,476,736 B = 64 GiB | `sysctl hw.memsize`, `os.sysconf("SC_PHYS_PAGES")*os.sysconf("SC_PAGE_SIZE")`, `NSProcessInfo.physicalMemory` |
| GPU | 20 cores, IOClass AGXAcceleratorG17X, ioreg `"model" = "Apple M5 Pro"` | `ioreg -r -d 1 -w 0 -c IOAccelerator` |
| Metal `recommendedMaxWorkingSetSize` (W) | 55,662,788,608 B = 51.84 GiB = 81.0% of RAM | ctypes probe, Swift probe |
| Metal `maxBufferLength` | 41,747,087,360 B = 38.88 GiB. Smaller than the IQ3_XXS (42.9 GB), IQ3_S (50.3 GB) and UD-Q4 (77 GB) arenas, so the engine must split the arena over several MTLBuffers | ctypes probe |
| Metal misc | hasUnifiedMemory = true; supportsFamily apple10, apple9, metal4 = true; maxThreadgroupMemoryLength = 32768 | ctypes probe |
| GPU wiring knobs | `iogpu.wired_limit_mb` = 0 (default: W applies); `vm.global_user_wire_limit` = 56,349,970,923; `vm.global_no_user_wire_amount` = 12,369,505,813; `ulimit -l` unlimited | `sysctl` |
| Memory pressure | `kern.memorystatus_vm_pressure_level` = 1 (normal); `kern.memorystatus_level` = 62-67 (% free-ish); `vm.swapusage` 14336M total (dynamic swap) | `sysctl` |
| Thermal / power | `NSProcessInfo.thermalState` = 0 (nominal), `isLowPowerModeEnabled` = False (public API, ctypes, no root) | ctypes |
| Python | `/usr/bin/python3` = 3.9.6 (CLT stub); Homebrew `/opt/homebrew/bin/python3.13` = 3.13.12; `platform.machine()` = arm64; `platform.processor()` = "arm"; `platform.mac_ver()[0]` = "27.0" | |
| Toolchain | Xcode 27.0 (27A266a). `metal` is not in CLT; it lives in a MobileAsset cryptex: `/var/run/com.apple.security.cryptexd/mnt/com.apple.MobileAsset.MetalToolchain-v27.1.../Metal.xctoolchain/usr/bin/metal` (Apple metal 32023.921). Expected install path: Xcode + `xcodebuild -downloadComponent MetalToolchain` (to verify on a clean machine) | `xcrun -f metal` |
| Firewall | Application Firewall enabled (State = 1) | `socketfilterfw --getglobalstate` |

### 0.2 What a Mac user hits today, in execution order

| # | Where | What happens on macOS |
|---|---|---|
| 1 | setup.sh:L8, L15-33 | `ok_py` needs 3.10+ with `venv` and `ensurepip`. `/usr/bin/python3` is 3.9.6 and fails. None of apt-get, dnf or pacman exists, so it prints the Ubuntu hint and `exit 1`. Without the CLT, `/usr/bin/python3` is a stub that opens the "install Command Line Tools" dialog. It passes only if Homebrew/python.org `python3` 3.10+ is first on PATH. |
| 2 | setup.py:L2685-2719 | Step 1: `gpus()` gives [] (no nvidia-smi) and `amd_gpus()` gives [] (no /sys). Then `fail("no NVIDIA GPU found (nvidia-smi did not answer)", "install the NVIDIA driver ...")`. |
| 3 | setup.py:L2737, L271-277 | If step 2 is bypassed: `ram_gb()` opens `/proc/meminfo`. The FileNotFoundError is not caught, so it crashes. |
| 4 | setup.py:L2738, L290-310, L2760-2762 | `cpu_info()` gives ("arm", False, False), then `fail("this CPU has no AVX2; Strata needs at least AVX2")`. |
| 5 | setup.py:L93, L1509-1590 | The prebuilt asset name is `strata-linux-x64.zip` on any non-Windows host, so it would download a Linux ELF. The arch check is CUDA `sm` ints (`gpu["arch"]`, `gpu["archs"]`). |
| 6 | setup.py:L3020-3021 | Not a local build: `pip_install(CUDA_WHEELS)`. The nvidia-* wheels have no macOS builds, so pip fails. |
| 7 | setup.py:L1673, L1803 | Build path: `install_build_tools` uses winget/apt; `build_engine` uses `-DSTRATA_ENABLE_CUDA=ON` and `find_nvcc()`. |
| 8 | setup.py:L3141-3166 | ctx >= 64K adds `--kv-resident 32768` (KV streaming, meaningless on unified memory). Vision adds `--vram-reserve-mib 700`. |
| 9 | setup.py:L3214, L3221-3225 | `cal = None if hip else saved_calibration(cfg)`: "NVIDIA-only for now". On metal the calibration prompt would be offered. |
| 10 | setup.py:L2353-2381, L2393-2394 | `start()` re-probes nvidia-smi, may call `ensure_engine_for` (CUDA rebuild), and says the model "locks part of it for the GPU". |
| 11 | serve/server.py:L652-667 | `child_env` sets `LD_LIBRARY_PATH` for lib_dirs (ignored by dyld). |
| 12 | serve/server.py:L2442-2446 | The port probe misses a conflicting wildcard listener (BSD SO_REUSEADDR semantics, section 3.10). |
| 13 | serve/telemetry.py | NVML missing, so the Monitor shows no GPU at all (section 4). |
| 14 | tools/strata_mcp.py:L139-142 | `proc_identity` falls back to `os.kill(pid, 0)`: zombies count as running, and the PID-reuse protection is gone (section 5). |

Python-side items that need no change on macOS:
- The OpenAI/Anthropic API layer, frontend.py, structured.py, the tokenizer, chat.py, needle_bench.py, the web chat, monitor.js,
  the mock engine and the HTTP tests all run unchanged. On macOS, 238 of 242 non-setup tests pass; the 4 failures are in
  test_strata_mcp, caused by the two macOS bugs in section 5.
- All 15 pinned requirements install as macOS arm64 cp313 wheels.

---------------------------------------------------------------------------------------------------------------------

## 1. setup.py end to end (3,249 lines)

Proposal used throughout: add a third backend, `--backend metal`, the default and only choice when
`sys.platform == "darwin"`. Gate every NVIDIA/AMD path behind `backend in ("cuda", "hip")`. Define:

```python
MAC = sys.platform == "darwin"
```

next to `WIN` (setup.py:L60). Do not use `os.name == "posix"` to mean Linux; the code currently does in several places,
listed below.

### 1.1 Module constants

| Line | Constant | macOS status / replacement |
|---|---|---|
| L59 | `ROOT = Path(__file__).resolve().parent` | OK. On macOS, `/tmp` and `/var` resolve to `/private/...`; harmless here. |
| L60 | `WIN` | Add `MAC = sys.platform == "darwin"`. |
| L84 | `LLAMA_CPP_ZIP` (pinned llama.cpp source zip) | OK (source). The vision helper build on Mac needs `-DGGML_METAL=ON` and an embedded Metal library (section 9). |
| L86-90 | comment: prebuilt zip "with strata(.exe), strata-vision(.exe) and BUILD.json, built by tools/make_release.py" | `tools/make_release.py` is not in the repository. The release pipeline must be recreated for a macOS asset. |
| L91 | `PREBUILT_URL` (GitHub releases/latest/download/) | OK |
| L92 | `PREBUILT_TAG_URL = ".../releases/download/v{version}/"` | OK |
| L93 | `PREBUILT_ASSET = "strata-windows-x64.zip" if WIN else "strata-linux-x64.zip"` | Needs a third value: `"strata-macos-arm64.zip" if MAC`. Refuse x86_64 Macs (`platform.machine() != "arm64"`) and Rosetta (`sysctl.proc_translated == 1`). |
| L94-95 | `CUDA_WHEELS` | Never on Mac. |
| L96 | `MIN_DRIVER = 580` | N/A; replace with a macOS minimum (`platform.mac_ver()`, e.g. 15.0 or 26.0, decided by the Metal feature set the engine uses). |
| L97 | `MIN_ENGINE = (0, 1, 34)` | Use a separate `MIN_ENGINE_METAL`, since the Metal engine starts at its own version. |
| L98-99 | `PY_PACKAGES`, `REQUIREMENTS` | OK (section 8) |
| L101-121 | `MODELS` (download_gb, ram_gb, arena_gb) | OK as data. `ram_gb` thresholds are dGPU-PC numbers; Mac uses the formula in section 2. |
| L135 | `UNSLOTH_RAM_LEFT_GB = 24` | Keep as a stream-tier headroom candidate; re-measure. |
| L141 | `CONTEXTS = [8192 ... 524288]` | OK |
| L180-181 | `VISION = {"gpu": {"max_tokens": 1024, "reserve_mib": 700}, "cpu": {... 300, 700}}`; comment L176-179 says "~1.2 GB at 1024 image tokens" | On Mac "gpu" means Metal; the reserve counts against W, not VRAM. |
| L182-183 | `EXE = "strata.exe" if WIN else "strata"`, `VEXE` | OK on Mac. This POSIX value also breaks two tests (section 6). |
| L967, L973 | `AMD_ARCHS`, `AMD_CARDS` | N/A |
| L1067 | `WIN_HIP_ASSET` | N/A |
| L1769-1770 | `ENGINE_SOURCES`, `VISION_SOURCES` | OK; add the Metal shader sources if they live outside `src`. |

### 1.2 Output and process helpers

- L187 `say`, L204 `fail`, L212 `ask`, L252 `mark`: portable.
- L228 `run()` uses subprocess. On Mac a GUI-launched setup (a `.command` file in Finder) inherits a minimal PATH; prepend
  `/opt/homebrew/bin:/usr/local/bin` when MAC.

### 1.3 Hardware probes

| Line | Function | Linux/Windows logic | macOS result today | macOS replacement |
|---|---|---|---|---|
| L271-277 | `ram_gb()` | WIN: GlobalMemoryStatusEx; else `/proc/meminfo` MemTotal kB * 1024 / 2**30 | **crash**: FileNotFoundError, not caught | `os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / 2**30` (verified 64.0) or `sysctl -n hw.memsize`. The Mac value is exact (64.0), while Linux reads "a little under the label" (63.7 in the golden profiles), so tests must use exact values. |
| L280-287 | `page_file_gb()` | WIN only, else None | None (fine) | Keep None. Optionally warn when `vm.swapusage` shows heavy swap use. |
| L290-310 | `cpu_info()` returns (name, avx2, avx512) | `platform.processor()`, then `/proc/cpuinfo` flags | ("arm", False, False) | Return (`sysctl -n machdep.cpu.brand_string` = "Apple M5 Pro", True, False) on arm64, or better a new tuple with neon/dotprod/i8mm/bf16/sme from `hw.optional.arm.FEAT_*`. The engine's CPU expert path needs NEON + DotProd (+ I8MM); the AVX2 gate (L2761-2762) must become a per-ISA gate. |
| L2233 | `is_wsl()` | `/proc/version` | False | OK |

Proposed probe. Verified with python 3.9 and 3.13 on the live Mac; stdlib only, no pyobjc.

```python
def metal_devices() -> list[dict]:
    """[{index, name, vram_gb (= working set), working_set, max_buffer, unified, families, cores, ram}], [] off macOS."""
    if sys.platform != "darwin":
        return []
    import ctypes, ctypes.util
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
    metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    def send(obj, sel, restype=ctypes.c_uint64, *args, argtypes=()):
        f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes))
        return f(obj, objc.sel_registerName(sel), *args)
    dev = metal.MTLCreateSystemDefaultDevice()
    if not dev:
        return []
    name = send(send(dev, b"name", ctypes.c_void_p), b"UTF8String", ctypes.c_char_p).decode()
    ws = send(dev, b"recommendedMaxWorkingSetSize")
    wired = int(out(["sysctl", "-n", "iogpu.wired_limit_mb"]).strip() or 0) * 2**20   # 0 = default
    fam = lambda n: bool(send(dev, b"supportsFamily:", ctypes.c_bool, n, argtypes=(ctypes.c_long,)))
    return [{"index": 0, "name": name, "working_set": wired or ws, "max_buffer": send(dev, b"maxBufferLength"),
             "unified": bool(send(dev, b"hasUnifiedMemory", ctypes.c_bool)),
             "families": [f for f, n in (("apple10", 1010), ("apple9", 1009), ("apple8", 1008)) if fam(n)],
             "vram_gb": (wired or ws) / 1e9, "arch": "apple", "driver": platform.mac_ver()[0]}]
```

Notes:
- `gpu-core-count` is available from `ioreg -r -d 1 -w 0 -c IOAccelerator -a` (plist, `plistlib.loads`).
- `vram_gb` is the working set in decimal GB, so the existing `"vram_gb"` consumers keep working. But see section 2: the
  existing VRAM formulas must not be fed this value.
- Optional cross-check once the engine is installed: `strata --list-devices`, mirroring `hip_devices()` (setup.py:L1182),
  with the engine printing the device line. The ctypes probe is still needed before the engine is downloaded, to choose the
  model.

### 1.4 NVIDIA GPU functions (all N/A on Mac; must be skipped when backend == "metal")

| Line | Function | Purpose | Mac action |
|---|---|---|---|
| L375 | `gpus()` | nvidia-smi query, returns cards (index, name, vram_gb, arch, driver) | Returns [] on Mac (no nvidia-smi). Keep; `metal_devices()` sits beside it. |
| L395 | `cc(g)` | compute capability string | skip |
| L399 / L405 | `experimental_sm60`, `sm60_card` | Pascal handling | skip |
| L409 | `gpu_problem(g)` | refuses < sm75 etc. | Metal equivalent: refuse when `apple9`/`apple8` family is missing (decide the minimum: M1 = apple7, M2 = apple8, M3/M4 = apple9, M5 = apple10), or when `unified` is false. |
| L421 / L427 / L431 | `gpu_rank`, `gpu_name`, `gpu_table` | display | one device; show name, working set, cores |
| L438 / L444 / L461 | `together_ok`, `parse_gpus`, `check_gpus` | multi-GPU | N/A (one GPU). `--gpu/--gpus/--layer-split` must error or be ignored on Mac. |
| L490 / L500 / L509 | `engine_archs`, `engine_archs_hip`, `engine_runs_on` | BUILD.json archs vs card | Metal: compare `BUILD.json.gpu_families` / `macos_min` with the device and the OS. |
| L517 / L529 / L562 / L578 | `start_gpus`, `choose_gpus`, `split_mmap`, `offer_together` | multi-card start logic | skip on Mac |
| L614 | `gpu_info(index)` | nvidia-smi detail for one card | metal_devices()[0] |
| L630 / L652 | `find_nvcc`, `find_vcvars` | CUDA toolchain | `xcrun -f metal` and `xcrun -f metallib` (fails without full Xcode + the MetalToolchain component), plus `xcode-select -p` |
| L943 | `cuda_lib_dirs()` | cublas64_13.dll / libcublas.so.13* | Mac: `[]`. Metal and Accelerate are system frameworks. |
| L1493 | `driver_major(g)` | NVIDIA driver | replaced by the macOS version check |

### 1.5 AMD functions (all N/A on Mac)

The AMD functions are:
- L977 `rocm_index`, L981 `amd_gpus` (sysfs KFD), L1019 `amd_problem`, L1027 `amd_gpus_win`, L1033 `amd_parse_gpus`
- L1086 `_win_display_adapters`, L1101 `_win_display_registry`, L1133 `_pci_device_id`, L1138 `win_amd_arch`, L1146 `amd_gpus_windows`
- L1182 `hip_devices`, L1226 `hip_lib_dirs`, L1235 `hip_match`, L1244 `hip_card`, L1264 `get_prebuilt_hip`
- L1332 `rocm_version`, L1341 `rocm_root`, L1384 `hipblaslt_version`, L1398 `hipblaslt_table`
- L1417 `build_engine_hip` (L1444 sets `LD_LIBRARY_PATH`), L1467 `hip_vision`

`amd_gpus()` returns [] on Mac because `/sys` is missing, so they are inert. One function is worth reusing:
L1480 `build_vision_cpu` (`-DSTRATA_VISION_CUDA=OFF`) is the closest model for a Mac vision build. Use
`-DSTRATA_VISION_CUDA=OFF` plus `-DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON` for a Metal vision helper; the CPU variant
works on Mac today in principle (ggml CPU with NEON).

### 1.6 Disk, downloads, shards

| Line | Function | macOS notes |
|---|---|---|
| L677 | `free_gb(path)`: `shutil.disk_usage(path).free / 1e9` | APFS purgeable space (snapshots, iCloud caches) is not counted, so the free space shown is lower than Finder's. Accept it, or say "free (Finder may show more: purgeable)". |
| L683 | `drop_archive` | OK |
| L690-762 | `download()`: urllib, Range resume, `.part` / `.done` marks; `file://` and plain paths via `shutil.copyfile` | Works with Homebrew Python. python.org builds need "Install Certificates.command" or urllib fails with CERTIFICATE_VERIFY_FAILED. Fix: `ssl.create_default_context(cafile=certifi.where())`; certifi is already pinned. Files fetched via urllib or curl get `com.apple.provenance` but **no** `com.apple.quarantine` (verified). `shutil.copyfile` on macOS copies data only, so no xattrs. A 70-110 GB download on a laptop needs a sleep assertion (`caffeinate -i -w <pid>`, see 1.12). |
| L765 / L780 / L800 / L847 | `whole_shard`, `gguf_dir_shards`, `verify_sha256`, `check_shards` | portable |
| L868 | `get_llama_cpp()` | portable (zip) |
| L901-940 | `req_name`, `requirement_lines`, `_installed`, `pip_install` (stamp `.strata-pip.json`) | portable; all pins have arm64 wheels |

### 1.7 Prebuilt engine (setup.py:L1500-1600)

- L1500-1506 `prebuilt_bases`: the tag release `v{source_version()}` first, then latest. A non-default `--prebuilt` /
  `STRATA_PREBUILT_URL` is used alone. OK.
- L1509-1590 `get_prebuilt(url_base, gpu, vision, updating)`:
  1. An installed engine (`engine/BUILD.json` + `engine/strata`, backend != "hip"):
     - `source == "local"` returns None (build_engine checks it).
     - The arch check is `int(a)` over `meta["archs"]` and `int(x)` over `gpu.get("archs", [gpu["arch"]])`. A metal gpu dict with
       `arch: "apple"` raises ValueError. It needs a metal branch.
     - `ver >= MIN_ENGINE` keeps the engine.
  2. HEAD `base + PREBUILT_ASSET` (60 s); on OSError, fall back to the next base, else "compiling instead".
  3. `download`, then `zipfile.ZipFile.extractall(engine/_unpack)`, then read BUILD.json, then check the version and archs,
     then move the files into `engine/`.
  4. L1584-1587 `chmod 0o755` for EXE/VEXE when not WIN. Needed because zipfile drops Unix modes **and symlinks**.
     A macOS asset must therefore contain no symlinks (no `libfoo.dylib -> libfoo.1.dylib`) and no framework bundles. Either keep
     the chmod, or extract with `ditto -x -k`, which keeps modes, symlinks and xattrs.
  5. L1588-1589 ok line `"for sm_.. (CUDA ..)"` needs a metal text.
- L1593 `update_installed_engine` uses the same path.
- Metal branch to add:
  - `meta["backend"] == "metal"`.
  - `meta["arch"] == "arm64"`.
  - `platform.mac_ver() >= meta["macos_min"]`.
  - `set(meta["gpu_families"]) & device families`.
  - version >= `MIN_ENGINE_METAL`.
  - After extraction, run `xattr -dr com.apple.quarantine engine/` (no-op when absent). Then smoke-test `engine/strata --version`
    and fail with a codesign hint if the kernel kills it (SIGKILL, exit -9 = invalid signature).

### 1.8 Build tools and local build

| Line | Function | Linux/Windows | macOS replacement |
|---|---|---|---|
| L1673 | `install_build_tools()` | winget (VS Build Tools, CUDA) / apt (build-essential, cuda-toolkit) | Check `xcode-select -p`, then `xcrun -f metal`. The CLT alone has clang but no `metal`. Ask the user to install Xcode from the App Store and run `xcodebuild -downloadComponent MetalToolchain`. Do not auto-install. cmake and ninja come from the pinned pip wheels (both have macOS arm64 wheels). |
| L1746 | `cmake_build(src, build, target, defs, env, bat)` | Ninja, `-j cpu_count//2`, `.bat` wrapper on WIN | Portable. On Mac `os.cpu_count()//2` = 9 is fine. Add `-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=<min>`. |
| L1773 | `source_hash` | | OK |
| L1784 | `engine_defs` | CUDA defs | Metal: `-DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_METAL=ON` (name to be decided by the native port) |
| L1789 | `prebuilt_vision` | | OK |
| L1803 | `build_engine(gpu, vision, yes, llama)` | `-DSTRATA_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES`, nvcc; writes BUILD.json with `archs`, `cuda_dirs` | `build_engine_metal`: no arch list; BUILD.json `{"backend": "metal", "arch": "arm64", "macos_min": ..., "gpu_families": [...], "source": "local", "lib_dirs": [], "vision": "gpu"/"cpu"/"none", "src": ..., "vision_src": ...}`. Ad-hoc sign after a local build: the linker does it on arm64; `codesign -s - -f` again if anything modifies the binary afterwards. |
| L2455 | `ensure_engine_for` | rebuilds CUDA for a newly seen card | skip on Mac |

### 1.9 Low-RAM, KV streaming and context helpers (formulas quoted in section 2)

The helpers are:
- L1858 `DATA_ITEMS`, L1861 `LOW_RAM_HEADROOM_GB = 10`, L1862 `RESIDENT_ENGINE = (0,1,30)`
- L1865 `low_ram_needed`, L1871 `low_ram_gpu_gb`, L1880 `low_ram_gpu_share`, L1885 `low_ram_resident`, L1893 `low_ram_fits`
- L1899 `low_ram_one_gpu_why`, L1920 `low_ram_together`
- L1955 `confirm_risk`
- L1967 `confirm_paging`: its text says "Windows/Linux will page"; needs a macOS wording ("macOS will compress and swap").
- L1986 `ctx_ram_need`, L1995 `ram_ctx`
- L825 `resident_budget_gib`, L833 `budget_choice`

All of them assume RAM and VRAM are separate pools that **add**. On Mac they must be bypassed by a `metal_fit()`
(section 2.4), not reused.

### 1.10 Settings, data folder, versions

| Line | Function | macOS notes |
|---|---|---|
| L2001-2004 | `settings_path()`: `%APPDATA%\Strata\settings.json`, else `$XDG_CONFIG_HOME` or `~/.config/strata/settings.json` | Works. Mac convention is `~/Library/Application Support/Strata/settings.json`, but that path has a **space** (see 3.3: ENC/GENI paths). Keeping `~/.config/strata` is simpler and already what strata_mcp.py:L428-431 reads. Recommendation: keep XDG on Mac and document it. |
| L2007-2019 | `load_settings` / `save_settings` | OK |
| L2022 / L2032 | `has_data`, `other_installs` | OK |
| L2055 | `same_drive` (`st_dev`) | OK on APFS (one container, separate volumes have different st_dev) |
| L2062 / L2083 | `move_into`, `repoint_config` | OK |
| L2111 | `data_folder(arg)`: default `ROOT.parent / "Strata-data"` | Hazard: if the checkout is in `~/Desktop` or `~/Documents` and iCloud "Desktop & Documents" sync is on, 70-120 GB of GGUF gets uploaded or evicted ("optimize storage" removes local copies, so mmap faults hit the network). Also TCC: Terminal needs permission for Desktop/Documents/Downloads/removable volumes (a prompt on first access, or "Operation not permitted"). Recommend defaulting to `~/Strata-data` on Mac when ROOT is under Desktop/Documents/iCloud Drive, or suffixing `.nosync`. Add a `.metadata_never_index` file (Spotlight) and `tmutil addexclusion <data>` (Time Machine; prompt or do it, since it is per-user and no root is needed for addexclusion). |
| L2165-2205 | `previous_config`, `choices_from_config`, `find_in`, `installed_configs` | OK |
| L2209 | `source_version()` from `CMakeLists.txt project(strata VERSION x)` | OK |
| L2215 | `engine_version()`: BUILD.json, else scans the binary for `engine=X.Y.Z\n` | OK (Mach-O string scan works the same) |
| L2237 | `hardware_key()` (calibration cache key: GPU name + VRAM, cpu name, ram, model, ctx, images) | Mac: device name + working set + core count + RAM |
| L2249 / L2278 / L2283 | `calibrate_config`, `saved_calibration`, `upgrade_config` | calibration is CUDA-specific (section 6.6) |

### 1.11 Start, run scripts, signals

- L2308-2403 `start(cfg_path, port, gpu, ...)`:
  - L2311 reads the config with `upgrade_config`; L2312-2314 checks that the files exist.
  - L2315-2321 handles `keep` and saves.
  - L2324-2325 builds the server command line.
  - L2326-2351 is the hip branch.
  - L2353 calls `found = gpus()`: nvidia-smi again, so skip it on metal.
  - L2354-2383 does the multi-GPU and `ensure_engine_for` work (L2375, L2381): skip on metal.
  - L2385-2391 sizes the banner from the `--native` GGUF's file size; L2392-2398 prints the start banner. "it loads about N GB into RAM and locks part of it for the GPU ... YOUR PC CAN BE
    SLOW OR STOP RESPONDING FOR 1-3 MINUTES": true-ish for CUDA pinning. On Mac the load is an mmap or read of the pack
    (with SSD speed 5-7 GB/s, roughly 10-20 s for 40-50 GB). The wording should change to the unified-memory story.
  - L2399-2403: `STRATA_EXECV` causes `os.execv` (Docker); otherwise `subprocess.call(cmd)`. The child shares the
    terminal's process group, so Ctrl+C reaches setup.py, server.py and the engine all at once.
- L2478-2490 `write_run_script`:
  - It writes `run-<model>.sh`: `#!/bin/sh`, `cd "<ROOT>"`, `exec "<venv python>" serve/server.py ... --open`, chmod 755.
  - Works in Terminal. But a `.sh` double-clicked in Finder opens in a text editor or Xcode, not Terminal.
  - On Mac also write `run-<model>.command` (Finder runs `.command` in Terminal). Ship a `START-HERE.command` that runs
    `./setup.sh "$@"`.
  - Caveat: a `.command` from a browser-downloaded zip is quarantined, and on macOS 15+ Gatekeeper blocks it on
    double-click (System Settings > Privacy & Security > "Open Anyway"). The right-click Open override was removed in
    macOS 15. Document `xattr -dr com.apple.quarantine <folder>` or recommend `git clone` / `curl` instead.
- Sleep: there is no sleep-prevention code anywhere in the repository. A MacBook on battery will idle-sleep during a 100 GB
  download or a long generation. Recommend:
  - wrapping long steps with `caffeinate -i -w <setup pid>` (spawned with subprocess, ends with the process);
  - in server.py, holding an `IOPMAssertionCreateWithName(kIOPMAssertionTypePreventUserIdleSystemSleep)` (IOKit via ctypes)
    or a `caffeinate -i -w <server pid>` child while a request is running or the model is loaded.
  - Closing the lid still sleeps the machine unless it is on AC with an external display.

### 1.12 main() step by step (setup.py:L2531-3249)

| Step / lines | What it does | macOS change |
|---|---|---|
| L2531-2595 | argparse (table in 1.13) | add `metal` to `--backend`; reject or ignore the multi-GPU flags |
| L2596-2604 | `--resident-budget-gib` and `--gpu` parsing; error text "numbered as nvidia-smi numbers them" | text |
| L2605 | `say("Strata - Qwen3.8-Flash-Next on a normal PC (a GPU + system RAM + CPU)")` | "on a Mac (unified memory)" |
| L2606-2609 | data_folder | see 1.10 |
| L2611-2633 | step 0: adopt an earlier install's choices (`previous_config`) | OK |
| L2639-2648 | `--calibrate` on an existing install | metal: the calibrate.py sweep must change (6.6) |
| L2649-2662 | start an installed model; `update_installed_engine` | OK after the 1.7 fixes |
| L2665-2671 | step 1: `found = gpus()`, `amd = amd_gpus()`, `hip = ...` | insert first: `if MAC: backend = "metal"; dev = metal_devices()[0]` (fail if none or unsupported) |
| L2672-2685 | both NVIDIA and AMD: ask | N/A |
| L2686-2714 | AMD path | N/A |
| L2716-2719 | `fail("no NVIDIA GPU found (nvidia-smi did not answer)", "install the NVIDIA driver ...")` | **blocker 2**; the metal branch must come first |
| L2720-2733 | choose_gpus, gpu_info, MIN_DRIVER check | metal: macOS version check instead |
| L2735-2736 | `if gpu["vram_gb"] < 11: warn("less than 12 GB of VRAM ...")` | metal: warn on a small W instead (section 2) |
| L2737 | `ram = ram_gb()` | **blocker 3** |
| L2738 | `cpu, avx2, avx512 = cpu_info()` | arm64 tuple |
| L2739-2755 | RAM gate: `need = min ram_gb` (32); `low_ok = low_ram_fits("IQ1_M", ram, vram)`; `ram < need - 4` leads to confirm_risk | metal: `metal_fit()` tiers |
| L2756-2759 | page file warning | N/A (None) |
| L2760-2762 | `ok("CPU: ... AVX2")`; `fail("this CPU has no AVX2 ...")` | **blocker 4**: replace with an arm64 / NEON / DotProd check |
| L2763-2779 | `--check` verdicts: `fits if ram >= ram_gb`, `tight if >= ram_gb-8`, low-RAM text "the GPU holds ~N% of its experts" | metal verdicts from `metal_fit()`: "fits on the GPU", "GPU + CPU (N% on GPU)", "streams N% from SSD (experimental)", "does not fit" |
| L2781-2850 | step 2: family / model / questions; `fit` texts compare against `ram_gb` | metal: model list annotated with the metal tier |
| L2852-2856 | `rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072`; budget: `8192 if small < 14 else 32768`; `rec_ctx = min(rec_ctx, ram_ctx(...))` | metal: the largest context in `CONTEXTS`, at most 131072 by default, for which `metal_fit` stays in the best tier (2.5) |
| L2857-2990 | context, rope, kv, vision, ESP questions | `--vision gpu` means Metal; `--kv k8v4` depends on whether the Metal engine supports it |
| L2932 | `vram = gpu["vram_gb"] - (VISION[vision]["reserve_mib"] / 1024 ...)` | metal: subtract the vision reserve from W |
| L2993 | disk need: `to_fetch + 8 + 40 (Q2_0, avx512, qwen) + 1 (vision) + arena+1 (low_ram)` | the "+40 for Q2_0 with avx512" item is an x86 repack; check whether a metal pack needs one |
| L3001-3004 | step 3: `pip_install` | OK |
| L3006-3039 | step 4: `get_llama_cpp`; hip+WIN uses `get_prebuilt_hip`, else `get_prebuilt`; not local means `pip_install(CUDA_WHEELS)`; else `build_engine_hip` / `build_engine`; `lib_dirs = meta lib_dirs or cuda_dirs or cuda_lib_dirs()` | metal: `get_prebuilt` metal branch, else `build_engine_metal`; **never** CUDA_WHEELS; `lib_dirs = []` |
| L3040-3116 | steps 5-6: pack, MTP | Python, portable (tools/iq_pack.py, mtp_*.py tests pass on Mac). Pack layout or alignment for Metal is a native-side decision (16 KiB pages on Apple Silicon, see 2.6). |
| L3117-3127 | step 7 args: `--pack --native [--ple-gguf] --expert-profile --expert-cache auto --prefill auto --spec 4 --spec-min-p 0.5 --mtp --max-context` | Keep. `--expert-cache auto` sizing must use W. |
| L3128-3129 | rope flags | OK |
| L3130-3131 | `--kv` above 8K | OK if supported |
| L3132-3138 | `--resident-experts` / `--mmap-experts` | metal hybrid/stream tiers map to new engine modes (2.4) |
| L3139-3170 | KV streaming: `kv_ram_gb = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9`; `stream_fits = ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1`; exclusions k8v4, WSL, `--kv-streaming off`; adds `--kv-resident 32768` | metal: **never** add `--kv-resident`; `--kv-streaming on` should warn "not used with unified memory" |
| L3173-3174 | `--resident-budget-gib` (UD-Q4) | metal stream tier: the budget is B from 2.4 |
| L3175-3176 | `--vision --vram-reserve-mib 700` | metal: the reserve counts against W (rename later, keep the flag) |
| L3177-3181 | ESP control vector flags | OK |
| L3182-3184 | `cfg = {"exe", "args", "cwd", "tokenizer", "model_name", "log", "lib_dirs", "port"}` | add `"backend": "metal"`; `lib_dirs: []` |
| L3185-3194 | hip: `backend`, `env STRATA_HIPBLASLT_TUNING`, `STRATA_RESIDENT_PIN` | N/A |
| L3195-3202 | `gpu`, `gpus_asked`, multi | none on metal |
| L3208-3212 | vision section `{"exe", "mmproj", "model", "gpu", "max_tokens"}`, cpu threads `cpu_count()//2` | Mac: threads = P-core count (`hw.perflevel0.physicalcpu`) |
| L3214-3219 | `cal = None if hip else saved_calibration(cfg)`: "tools/calibrate.py is NVIDIA-only for now" | **bug on Mac**: change to `None if backend != "cuda"` |
| L3220-3225 | write config, run script, offer calibration (L3222-3225 asks unless --yes / --no-start / hip) | same gate |
| L3227-3238 | "All set." text (L3236): "Next time: just run START-HERE.bat or ./setup.sh" | Mac: "./setup.sh (or double-click run-<model>.command)" |

### 1.13 Every CLI flag (setup.py:L2533-2594) with its macOS status

| Line | Flag | Meaning upstream | macOS |
|---|---|---|---|
| L2533 | `--family {qwen,swift,coder,unsloth}` | model family | OK |
| L2534 | `--model {Q2_0,IQ2_XS,IQ3_XXS,IQ3_S,IQ1_M,UD-Q4_K_XL}` | size | OK; recommendation via metal_fit |
| L2535 | `--context INT` | max context | OK; default from W (2.5) |
| L2536 | `--rope-scaling {none,linear,yarn}` | past 262144 | OK |
| L2540 | `--rope-scale FLOAT` | factor | OK |
| L2543 | `--kv {int8,q4_0,k8v4}` | KV precision | OK if the Metal engine implements all three; otherwise restrict |
| L2546 | `--vision {yes,no,none,gpu,cpu}` | encoder placement; yes = gpu | gpu = Metal encoder; cpu = ggml CPU |
| L2548 | `--experimental-speed-projection on/off/GGUF` | control vector | OK (engine feature) |
| L2551 | `--port INT` | server port (8080 new install) | OK. If a dev server or container already holds `*:8080`, the port probe bug in 3.10 lets Strata start anyway. |
| L2552 | `--gpu N` | "numbered as nvidia-smi numbers them" | Mac: only `0`; anything else errors "a Mac has one GPU" |
| L2554 | `--gpus LIST/all` | layer split | Mac: error |
| L2557 | `--layer-split` | split points | Mac: error |
| L2559 | `--host` | bind address | OK; macOS firewall hint (3.10) |
| L2561 | `--api-key` | auth | OK |
| L2562 | `--data-dir` | data folder | OK; iCloud/TCC/Spotlight/Time Machine advice (1.10) |
| L2564 | `--models-dir` | GGUF folder | OK |
| L2565 | `--gguf-dir` | existing GGUFs | OK (e.g. from LM Studio's `~/.lmstudio/models`, if it has the same files) |
| L2567 | `--yes` | accept defaults | OK |
| L2568 | `--setup` | reinstall / change | OK |
| L2569 | `--no-start` | install only | OK |
| L2570 | `--build` | compile instead of prebuilt | metal: needs Xcode + Metal toolchain |
| L2571 | `--prebuilt URL/folder` (env `STRATA_PREBUILT_URL`) | engine source | OK; strip quarantine on local folders (1.7) |
| L2573 | `--check` | check only | metal verdicts |
| L2574 | `--calibrate` | tune | needs a metal calibrate (6.6) or "not available on Mac yet" |
| L2576 | `--draft-vocab {cjk,en,cyrillic}` | MTP subset | OK |
| L2580 | `--low-ram {auto,on,off,resident,mmap}` | expert residency | metal: auto chooses from the tiers. `on`/`mmap` corresponds to the stream tier; `resident` corresponds to hybrid. Meaning differs: document. |
| L2585 | `--resident-budget-gib N` | UD-Q4 RAM budget | metal stream tier budget |
| L2588 | `--kv-streaming {auto,on,off}` | KV in RAM | metal: no-op, with a note |
| L2591 | `--backend {cuda,hip}` | backend | add `metal`; default on darwin |
| L2594 | `--skip-build` (suppressed) | internal | OK |

### 1.14 setup.sh, START-HERE.bat, SETUP.bat, Dockerfile, docker-entrypoint.sh

- **setup.sh** (39 lines, "Strata for Linux"):
  - L1-4 header text names Linux, apt/dnf and "Needs only an NVIDIA driver".
  - L8 `ok_py`: `import sys, venv, ensurepip` and >= 3.10.
  - L10-12 broken-venv cleanup.
  - L15-19 tries `python3` then `python`.
  - L20-33: apt-get / dnf / pacman, then the Ubuntu hint and exit 1.
  - L37 venv creation, with the hint `sudo apt install python3-venv`.
  - L39 `exec .venv/bin/python setup.py "$@"`.
  - macOS edits:
    - Try `python3.13 python3.12 python3.11 python3.10 python3 python` and the explicit paths
      `/opt/homebrew/bin/python3`, `/usr/local/bin/python3` and
      `/Library/Frameworks/Python.framework/Versions/3.1x/bin/python3`.
    - On darwin, never try sudo or a package manager. Print: "Install Python 3.12+:
      `brew install python@3.13` or the python.org macOS installer, then run ./setup.sh again".
    - Do not invoke `/usr/bin/python3` when `xcode-select -p` fails (that triggers the CLT dialog).
    - `sh` on macOS is bash 3.2 in POSIX mode (zsh is the login shell); the script is POSIX sh, so that is fine.
- **START-HERE.bat** (52 lines): winget `Python.Python.3.12` or python.org `python-3.12.10-amd64.exe`; Windows only.
  Mac equivalent: `START-HERE.command` (`#!/bin/sh`, `cd "$(dirname "$0")"`, `exec ./setup.sh "$@"`).
- **SETUP.bat** (3 lines): `call START-HERE.bat --setup`. Mac equivalent: `SETUP.command` running `./setup.sh --setup`.
- **Dockerfile** (115 lines) and **docker-entrypoint.sh** (65 lines): `nvidia/cuda:13.0.0-devel-ubuntu24.04`, `--gpus all`,
  `CUDA_ARCHITECTURES`, `STRATA_EXECV=1`. Docker Desktop for Mac runs a Linux VM with **no Metal/GPU passthrough**, so a
  container on a Mac can only be the CPU engine at best. Document "Docker is not supported on macOS; use ./setup.sh".
  - The entrypoint comment notes that setup reads `/proc/meminfo`, which in a container gives the host's RAM. In Docker
    Desktop it gives the VM's RAM.

---------------------------------------------------------------------------------------------------------------------

## 2. RAM/VRAM fit formulas and a unified-memory replacement

### 2.1 Upstream formulas, quoted

```python
# setup.py:L1865  low_ram_needed
ram < MODELS[model]["arena_gb"] + LOW_RAM_HEADROOM_GB                     # LOW_RAM_HEADROOM_GB = 10 (L1861)

# setup.py:L1871-1877  low_ram_gpu_gb: the GB of experts the GPU holds in low-RAM mode
kv_tok = 13 * (576 if kv == "q4_0" else 1056)        # bytes/token: 12 QSA layers + the draft layer
longer = max(0, ctx - 32768) * kv_tok / 1e9
return max(0.0, min(arena, vram_gb - 5 - longer))   # 5 GB: engine + 32K KV + scratch

# setup.py:L1885-1890  low_ram_resident: the rest fits in RAM as a copy
rest = arena - low_ram_gpu_gb(...)
return ram >= rest + LOW_RAM_HEADROOM_GB

# setup.py:L1893-1896  low_ram_fits
return ram - 6 + max(0.0, vram_gb - 5) >= arena

# setup.py:L1986-1992  ctx_ram_need: None unless IQ3_XXS/IQ3_S and not low_ram, else
arena + ctx * 13 * 1056 / 1e9 + 24
# setup.py:L1995  ram_ctx
max(c for c in CONTEXTS if c <= 131072 or (ctx_ram_need(...) or 0) <= ram)

# setup.py:L825-830  resident_budget_gib (UD-Q4)
gib = round(ram) - 24 - math.ceil(kv_ram_gb)
return max(8, min(gib, int(arena / 1.073741824)))

# setup.py:L2852-2856  recommended context from the smallest chosen card's VRAM
rec_ctx = 32768 if small < 14 else 65536 if small < 20 else 131072     # budget models: 8192 if small < 14 else 32768
rec_ctx = min(rec_ctx, ram_ctx(...))

# setup.py:L3141-3146  KV streaming
kv_ram_gb = ctx * (13 * (576 if kv == "q4_0" else 1056)) / 1e9
stream_fits = ram >= MODELS[model]["ram_gb"] + kv_ram_gb + 1           # then --kv-resident 32768 when ctx >= 65536

# setup.py:L2739-2741, L2763-2779  the RAM gate and --check verdicts
need = min(d["ram_gb"] for d in MODELS.values())                         # 32
ram < need - 4  -> confirm_risk(...)                                     # unless low_ok
"fits" if ram >= d["ram_gb"] else "tight" if ram >= d["ram_gb"] - 8 else "does not fit"

# setup.py:L2735-2736
if gpu["vram_gb"] < 11: warn("less than 12 GB of VRAM ...")
```

Units are mixed:
- `ram_gb()` returns **GiB** (bytes / 2**30), while `arena_gb`, `ram_gb` (MODELS) and the KV terms are decimal **GB**.
  `resident_budget_gib` divides by 1.073741824 to convert.
- On a 64 GiB Mac, `ram_gb()` with a sysconf port would return 64.0 "GB" against decimal GB thresholds. That is 7% optimistic.
  The metal formula below works in bytes.

### 2.2 Why the formulas are wrong for Apple Silicon

1. **Double counting.**
   - The full model's experts live in RAM; the GPU holds a copy of the hot ones (MODELS `ram_gb` includes the whole arena).
   - `low_ram_fits` adds `vram_gb - 5` on top of `ram - 6`.
   - On a Mac, VRAM is a subset of RAM: "copy into VRAM" is the same physical page. A Mac with 64 GiB would be credited
     64 + 52 = 116 GB.
2. **The GPU budget is W, not RAM.**
   - The GPU can only have `recommendedMaxWorkingSetSize` bytes resident (51.84 GiB here). An allocation or residency set
     beyond it fails or thrashes.
   - `iogpu.wired_limit_mb` (sudo, not persistent) can raise it.
3. **No PCIe, so no KV streaming benefit.** The `--kv-resident` reasoning ("frees VRAM for experts") does not apply; KV in "RAM"
   and "VRAM" is the same memory. Never pass `--kv-resident` on metal.
4. **`maxBufferLength` (38.88 GiB here) < some arenas.** This is a native engine concern (several buffers), but setup must
   not assume one buffer either.
5. **Headroom.** macOS needs about 6-10 GB for the system, WindowServer, the browser and the Python server. Memory beyond
   that is compressed, then swapped, rather than OOM-killed. Jetsam kills background processes first, so a too-big model
   degrades into swap thrash (very slow) rather than a crash.

### 2.3 Definitions

- `A` = `arena_gb * 1e9` (expert bytes, from MODELS).
- `D` = `4.55e9`: everything on the GPU besides the experts and the context KV. Derived from upstream's "5 GB" in
  `low_ram_gpu_gb` (engine, dense weights, the 32K KV and scratch) minus the 32K int8 KV (0.45 GB), so the KV can be
  counted per context. **To be re-measured on the Metal engine** (dense / PLE / MTP / scratch sizes from the native
  audit).
- `K` = `ctx * 13 * (576 if kv == "q4_0" else 1056)` bytes. Sizes:
  - 32K = 0.45 GB; 128K = 1.80 GB; 256K = 3.60 GB; 512K = 7.20 GB.
  - q4_0 at 128K = 0.98 GB.
  - k8v4 (816 B/cell per the `--kv` help) needs the engine's per-layer number.
- `V` = 1.2e9 when the vision encoder is on the GPU (setup.py L176-179 comment: "~1.2 GB at 1024 image tokens"; setup
  reserves 700 MiB, INSTALL.md says ~1.4 GB). 0 otherwise. The CPU encoder also uses RAM; count about 0.8 GB in H then.
- `M` = `hw.memsize`.
- `W` = `iogpu.wired_limit_mb * 2**20` if set (non-zero), else `recommendedMaxWorkingSetSize`. If Metal cannot be queried,
  estimate `0.75 * M` (`0.67 * M` at 24 GiB or less); the live 64 GiB macOS 27 machine reports 0.81 M.
- `H` = 8 GiB OS + apps headroom (between upstream's `LOW_RAM_HEADROOM_GB = 10` and the `ram - 6` in `low_ram_fits`).
- `S` = 1 GiB safety margin under W.
- `need = A + D + K + V`.

### 2.4 Tiers

```python
def metal_fit(model, ctx, kv, vision, dev):            # dev = metal_devices()[0], M = hw.memsize
    GiB = 2**30
    A = MODELS[model]["arena_gb"] * 1e9
    D = 4.55e9
    K = ctx * 13 * (576 if kv == "q4_0" else 1056)
    V = 1.2e9 if vision == "gpu" else 0.0
    M, W = dev["ram"], dev["working_set"]
    H, S = 8 * GiB, 1 * GiB
    need = A + D + K + V
    if need <= W - S and need + H <= M:
        return "gpu", 1.0                               # every expert GPU-resident, no copies
    if need + H <= M:                                   # all in RAM, the GPU's share capped by W
        return "hybrid", max(0.0, min(A, W - S - D - K - V)) / A    # the CPU runs the rest in place (no copy)
    B = M - H - D - K - V                               # RAM left for experts; the rest is read from the SSD
    if B >= 8 * GiB:
        return "stream", min(B, A) / A                  # experimental (UD-Q4 precedent: 7-8.5 tok/s at ~56% on a PC)
    return "no", 0.0
```

Tier meanings:
- **gpu**: the engine allocates (or `newBufferWithBytesNoCopy`-wraps an mmap of) the whole arena, in several MTLBuffers
  of at most `maxBufferLength` each. Then one MTLResidencySet holds them.
- **hybrid**: the experts are all in RAM (one mmap). The hottest `share` fraction is GPU-resident; the rest runs on CPU
  NEON kernels from the same pages. This is the Mac version of `--resident-experts`, minus the copy.
- **stream**: like upstream `--mmap-experts` / UD-Q4 `--resident-budget-gib`. The OS file cache serves the cold experts
  from the SSD (Apple SSDs read 5-7 GB/s).
- The engine flags for these tiers are for the native port to name. Setup only needs to emit them.

### 2.5 Table: 32K context, int8 KV, vision off

Computed by `/tmp/strata-research/scratch/um_table.py`. The W fraction is estimated except on the live 64 GiB row. A
percentage is the share of the experts on the GPU (hybrid) or in RAM (stream). "ctx" is the largest `CONTEXTS` entry that
stays in the gpu tier.

| RAM | W | IQ1_M (coder) | Q2_0 | IQ2_XS | IQ3_XXS | IQ3_S | UD-Q4_K_XL |
|---|---|---|---|---|---|---|---|
| 16 GiB | 10.7 GiB | no | no | no | no | no | no |
| 24 GiB | 16.1 GiB | stream 52% | stream 36% | stream 34% | stream 28% | stream 24% | stream 16% |
| 32 GiB | 24.0 GiB | stream 89% | stream 61% | stream 59% | stream 48% | stream 41% | stream 27% |
| 36 GiB | 27.0 GiB | hybrid 98% | stream 74% | stream 71% | stream 58% | stream 50% | stream 33% |
| 48 GiB | 36.0 GiB | gpu (ctx 512K) | hybrid 96% | hybrid 92% | stream 88% | stream 75% | stream 49% |
| 64 GiB (live) | 51.84 GiB | gpu (512K) | gpu (512K) | gpu (512K) | gpu (384K) | hybrid 99% | stream 72% |
| 96 GiB | 72 GiB | gpu (512K) | gpu | gpu | gpu (512K) | gpu (512K) | hybrid 93% |
| 128 GiB and up | 96+ GiB | gpu | gpu | gpu | gpu | gpu | gpu (512K) |

`need` at 32K is IQ1_M 28.4, Q2_0 39.0, IQ2_XS 40.5, IQ3_XXS 47.9, IQ3_S 55.3 and UD-Q4 82.0 GB. With H added that is
37.0, 47.6, 49.1, 56.5, 63.9 and 90.6 GB.

Recommendation rules for setup on metal, replacing L2852-2856 and the IQ3_XXS-from-60-GB rule:
- **Model:** the best-quality model whose tier is "gpu" at 32K. Otherwise the best "hybrid" with share >= 0.9. Otherwise
  IQ1_M or Q2_0 in "stream" with a slow/experimental warning. Otherwise refuse.
  - On the live 64 GiB machine that gives IQ3_XXS (gpu). IQ3_S is hybrid 99%, worth offering as "best quality".
- **Context:** the largest `c` in CONTEXTS with `c <= 131072` (keep upstream's 128K default cap) and `metal_fit(..., c)`
  still in the chosen tier. Show the larger contexts as options.
- **--check verdicts:** the tier text plus the percentage.
- **The VRAM < 11 GB warning becomes** `W < 16 GiB`: "this Mac's GPU can use only N GB".

### 2.6 Other memory notes for the setup or engine contract

- Apple Silicon pages are 16 KiB (`os.sysconf("SC_PAGE_SIZE")` = 16384). An mmap offset and `newBufferWithBytesNoCopy`
  need page-aligned pointers and lengths, which matters for the pack file layout (tools/iq_pack.py / strata_pack.py
  alignment).
- Wired memory: a GPU-resident buffer is wired. `vm.global_user_wire_limit` = 56.35e9 on this machine (about 82% of RAM)
  caps it. `ulimit -l` is unlimited, so mlock is not the constraint.
- `psutil.virtual_memory()` on macOS: `available` = free + inactive (+ purgeable). `used` excludes compressed memory, so
  the RAM bar under-reports pressure. Read `kern.memorystatus_vm_pressure_level` (1 normal, 2 warn, 4 critical) for a
  truthful signal.

---------------------------------------------------------------------------------------------------------------------

## 3. serve/server.py (2,613 lines)

### 3.1 Architecture

- One process, `ThreadingHTTPServer`-style `Server` (L2223; `allow_reuse_address = os.name != "nt"` at L2226, so True on macOS), `Handler` (L1723,
  HTTP/1.0).
- One `Service` (L739) holds the tokenizer, the chat template, a FIFO lock (one generation at a time), status/history, MCP
  hub, telemetry and the vision encoder.
- The engine is a child process, `StrataEngine` (L207), speaking a line protocol over stdin/stdout. `MockEngine` (L76)
  implements the same `Engine` Protocol (L71) for tests.
- `Vision` (L494) is a second child process (`strata-vision`).
- Errors: `EngineDied` L102, `EngineStarting` L106, `ModelBusy` L111, `EngineStuck` L115, `GpuBusy` L120.
- Stdlib only, plus optional psutil, jsonschema and pillow.

None of this is platform-specific except the items below. The stack runs unchanged on macOS: test_server 88 tests,
test_lifecycle 8 and test_monitor 7 all pass with the mock engine.

### 3.2 Engine protocol (`strata --serve`)

Launch: `subprocess.Popen([exe, "--serve", *args], cwd, stdin=PIPE, stdout=PIPE, stderr=log, text=True, encoding="utf-8",
bufsize=1, env)`, then `contain(proc)` (L239-241).

| Direction | Line | Meaning |
|---|---|---|
| out | `INFO k=v ...` (L245-249) | facts collected into `engine.info` and shown by the Monitor: engine version (`engine=`), kv, kv_resident, expert_slots, expert_cache_mib, arena_mib, spec, mtp_max, lookup, cvec, ... |
| out | `READY <ctx> [stop]` (L250-254) | ready; context size; `stop` = honours STOP |
| in | `GEN <max_new>[ key=val ...] <ids csv>` | generate; keys `temperature= top_p= top_k= seed=` etc. (docstring L207-214, `generate` L395-466) |
| in | `GENI <max_new>[ keys] <embeddings path> <ids>` | with image embeddings from the vision encoder (a file path) |
| out | `T <id>` | one token |
| out | `PP <done> <total> ...` | prompt progress; field 4 = prefill mean tok/s |
| out | `DONE ...` | finished, with stats |
| out | `ERR <msg>` | error |
| in | `STOP` | cancel (if READY said stop) |
| in | `QUIT` | exit. `close()`: QUIT, then terminate, then kill, 20 s each. |

The server yields a heartbeat (None) every 10 s while waiting. Mac impact: none. But the protocol carries **file paths
separated by spaces** (GENI, and the vision ENC below). A data or temp path with a space breaks it. On macOS that is likely:
- `~/Library/Application Support/...`;
- a user folder such as `/Users/Jane Doe` (rare: account names are short names without spaces, but external volumes such
  as `/Volumes/Macintosh HD 2` are common).

`tempfile.mkdtemp` gives `/var/folders/xx/.../T/strata-vision-...`, which has no spaces. Fix: quote or percent-encode the
paths, or put them last on the line.

### 3.3 Vision helper protocol

- Args: `[exe, --mmproj, --model, (--gpu), (--threads N), (--max-tokens N)]` (L499-506).
- Temp dir `tempfile.mkdtemp(prefix="strata-vision-")` (L507).
- `_start` reads one line and requires `READY` (L516-522).
- `ENC <img> <out>`, answered by `OK <ntok>` or `ERR <msg>` (L591-598).
- Cached by hash, 64 entries (L599-604).
- `QUIT` with a 10 s wait, else kill (L606-612).
- `load()` accepts `data:` URLs, http(s) (downloaded by the server) or local paths.
- Mac: the vision helper must be built with ggml Metal (llama.cpp mtmd). The `--gpu` flag means Metal. Same space-in-path caveat.

### 3.4 Endpoints (make_handler L1722)

GET (`do_GET` L1818):

| Path | Line |
|---|---|
| `/fonts/*` | L1820 |
| `/web/*` | L1835 |
| `/metrics` | L1853 |
| `/api/requests` | L1858 |
| `/settings` | L1867 |
| `/mcp` | L1871 |
| `` (index) or `/api-monitor` | L1876 |
| `/health`, `/api/health` (before the API-key gate) | L1883 |
| `/status` | L1887 |
| `/v1/models`, `/models` | L1901 |
| `/props` | L1915 |
| `/slots` | L1918 |
| `/v1/status` | L1925 |

POST (`do_POST` L1931):

| Path | Line |
|---|---|
| `/settings` | L1935 |
| `/unload` | L1938 |
| `/load` | L1946 |
| `/v1/load`, `/v1/unload` | L1957 |
| `/v1/chat/completions`, `/v1/messages` | L1981-1985 |
| `/v1/messages/count_tokens` | L1987 |

`do_OPTIONS` is at L1784 (CORS); `_authorized` at L1808; `_watch_client` at L1732 detects client disconnects.

All endpoints are portable. `/v1/status.machine.gpu` (L1110-1113) is `None` when `static.gpu_name` is None, which is the
case on Mac today. `/v1/status.machine.ram` works through psutil.

### 3.5 Config keys read by the server (main L2399-2594 and helpers)

The keys are:
- engine launch: `exe`, `args`, `cwd`, `tokenizer`, `model_name`, `log`
- `lib_dirs` (child_env)
- `env` (child_env): extra environment variables
- `gpu`: an int, a list or "0,2"
- `layer_split`, `split_skip_if_fits`: multi-GPU
- `backend`: "hip" today; add "metal"
- `hip_ordinal`
- `vision` (exe, mmproj, model, gpu, threads, max_tokens, cuda_device)
- `host`, `api_key`, `cors_origins`, `trusted_origins`, `api_monitor`
- `lazy_load`, `fit_max_tokens`, `aliases`, `anthropic_thinking`, `reasoning_budget_tokens`
- `idle_unload_s`, `min_free_vram_mib`, `before_load`
- `mcp_servers`, plus the sampling defaults (`sampling_defaults_from_config` L2341)
- `port` (setup only)

CLI flags (L2400-2436): `--engine {mock,strata}` (default mock), `--config`, `--host`, `--script`, `--port` (default 8095),
`--gpu`, `--tokenizer`, `--open`, `--fit-max-tokens`, `--api-key` (env `STRATA_API_KEY`), `--mcp-config`, `--lazy`,
`--api-monitor`, `--idle-unload`, `--min-free-vram-mib`, `--before-load`.

| Key | Mac equivalent |
|---|---|
| `lib_dirs` / `env` | `lib_dirs` must be `[]`. child_env L664-667 sets `LD_LIBRARY_PATH` (ignored by dyld). Do not switch to `DYLD_LIBRARY_PATH`: it is stripped for hardened/protected binaries and through `/bin/sh`. The engine must find its dylibs (if any) via `@rpath`/`@executable_path`. Mac branch: skip lib_dirs entirely. |
| `gpu` | not used. child_env L657-659 sets `CUDA_DEVICE_ORDER`/`CUDA_VISIBLE_DEVICES` when gpu_list is non-empty (harmless if set, but setup must not write `gpu`). |
| `backend` | `"metal"`. Add it to the free_vram_mib and telemetry dispatch. |
| `vision.cuda_device` (vision_env L670-684) | N/A |

### 3.6 Lifecycle: lazy load, idle unload, min_free_vram_mib, before_load

- **Lazy** (L2472-2474): `--lazy` or `"lazy_load": true` starts unloaded, and the first request loads. Text-only (refused
  with vision). Portable.
- **ensure_loaded** (L846-879):
  1. Run `before_load` (`subprocess.run(cmd, shell=isinstance(cmd, str), timeout=120)`).
  2. Then the `min_free_vram_mib` wait: poll `free_vram_mib()` for up to 15 s, else `GpuBusy` (HTTP 503).
  3. Then restart the vision encoder if it is down, then `engine.restart()`.
  - Portable except `free_vram_mib`.
- **free_vram_mib** (L828-844): hip uses telemetry sysfs; else NVML `nvmlDeviceGetMemoryInfo`. Returns None on Mac,
  so nothing is ever refused and the setting is silently inert.
  - Mac equivalent, "GPU memory headroom" = W minus the GPU memory other processes have allocated. Ways to read the second
    part:
    - ioreg IOAccelerator `PerformanceStatistics."In use system memory"` (no root, about 80 ms, system-wide). Gives
      `free_mib = (W - in_use) >> 20`.
    - or `MTLDevice.currentAllocatedSize` (own process only, so useless from the server).
    - or simply RAM headroom: `psutil.virtual_memory().available` combined with `kern.memorystatus_vm_pressure_level`.
  - Recommend a new config key, `min_free_ram_mib`, on metal with the same semantics (the model needs unified memory, not
    VRAM). Accept `min_free_vram_mib` as an alias that compares against the W-based number.
- **Idle unload** (L927-938): a thread sleeps `max(1, min(30, s/4))` and calls `unload(idle_for=s)`, which QUITs the engine
  and vision. Portable. More valuable on a Mac: unloading frees unified memory for every app.
- **before_load**: portable (a shell string or argv list), e.g. `lms unload --all` or `ollama stop <model>` to free memory
  held by LM Studio or Ollama.
- **restart** after death (L319-329): portable.

### 3.7 Telemetry (serve/telemetry.py) and `start_telemetry` (L976-984)

- `Telemetry(extra, gpu_index, gpu_indices, amd)` (L266):
  - Builds GPU readers: `_Nvml` (L25; libs `nvml.dll` / `libnvidia-ml.so.1`, L34-36) or `_Amd` sysfs (L136; `amd_device_dir`
    L111).
  - Uses psutil when present (L281-285), else `_CpuRamFallback` (L217: `/proc/stat` L230, `/proc/meminfo` L257, Windows
    GlobalMemoryStatusEx/GetSystemTimes).
  - Static (L289-296): `gpu_name`, `gpu_count`, `cpu_name` (`_cpu_name` L202-214: winreg, then `/proc/cpuinfo`, then
    `platform.processor()`), `cores` (psutil physical), `threads`, `psutil`.
  - `sample()` (L312+): `gpu_*` from the reader; `cpu`, `ram_used = total - available`, `ram_total`; disk read/write MB/s
    from `psutil.disk_io_counters()`.
- Live on this Mac: static = `{'gpu_name': None, 'gpu_count': 1, 'cpu_name': 'arm', 'cores': 18, 'threads': 18,
  'psutil': True}`. `now` has `cpu`, `ram_used`, `ram_total`, `disk_read_mb` and `disk_write_mb`. There are no `gpu_*` keys.
- Without psutil on Mac, `_CpuRamFallback` returns None (no /proc), so psutil is required on Mac.
- Proposed `_Apple` reader (`gpu_reader(index, backend)` dispatch, L185):

| Field | Source on Mac (no root) | Cost |
|---|---|---|
| `name` | MTLDevice.name (ctypes) or ioreg `"model"` ("Apple M5 Pro") | once |
| `util` | ioreg IOAccelerator `PerformanceStatistics."Device Utilization %"` (also Renderer/Tiler %) | `ioreg -r -d 1 -w 0 -c IOAccelerator -a`, about 80 ms, parse with plistlib; sample at 1 Hz in the existing thread, or every 2 s |
| `mem_used` | `PerformanceStatistics."In use system memory"` (bytes, all GPU clients) | same call |
| `mem_total` | W (`recommendedMaxWorkingSetSize` or the `iogpu.wired_limit_mb` override) | once |
| `temp` | not available via public API. Use `NSProcessInfo.thermalState` (0 nominal, 1 fair, 2 serious, 3 critical) via ctypes; actual degrees need private IOHID/SMC (macmon-style) or root `powermetrics` | cheap |
| `power`, `power_limit` | `sudo powermetrics --samplers gpu_power` only (root); IOReport "Energy Model" is private | omit |
| `pcie_*` | N/A (no PCIe link) | omit |
| restarts | `PerformanceStatistics.recoveryCount` / IOReport legend "GPU Restart Count", useful for death_note (GPU hang) | same call |

- Better than ioreg for frequency: the IOReport private framework. Better for per-process GPU time: `proc_pid_rusage`
  (`ri_*` has no GPU time) or `task_info`. Neither is needed for the first port.

### 3.8 Signals, process containment (winjob)

- **serve/winjob.py** (74 lines): `contain(proc)` (L61) assigns the child to a Windows Job Object with
  KILL_ON_JOB_CLOSE, and is a no-op elsewhere. On Linux nothing ties the engine's lifetime to the server either.
  - The Mac equivalent: a watchdog in the engine. The engine reads stdin; EOF on stdin means the parent died, so exit. This
    must be verified in the native engine (the `--serve` loop should exit on EOF).
  - Optional extra: a kqueue `EVFILT_PROC NOTE_EXIT` on the parent PID, or `getppid() == 1` polling.
  - No `prctl(PR_SET_PDEATHSIG)` exists on macOS.
- **SIGTERM** (L2589-2594): raises KeyboardInterrupt, which shuts down the HTTP server, closes the engine (QUIT), the
  vision encoder and the MCP hub. A second Ctrl+C kills. Portable.
- **SIGINT**: the engine is spawned without `start_new_session`, so a terminal Ctrl+C reaches the engine too. Whether the
  engine ignores SIGINT (so the server can QUIT it cleanly) is for the native audit. Same on Linux.
- **SIGHUP**: closing the Terminal window sends SIGHUP to the job. Python's default SIGHUP action terminates the server
  **without** the cleanup path; the engine then gets SIGHUP too, or EOF on stdin. Suggest handling SIGHUP like SIGTERM on
  POSIX: `signal.signal(signal.SIGHUP, on_sigterm)`. The run script's banner says "closing this window stops the model".
- **death_note** (L281-302): the RAM text "Linux then ends the biggest program (check: sudo dmesg | grep -i -E 'killed
  process|out of memory'); Windows slows down instead". The Mac text should cover:
  - macOS compresses and swaps first; under critical pressure jetsam ends processes. Check:
    `log show --last 15m --predicate 'eventMessage CONTAINS[c] "jetsam"'`, or JetsamEvent-*.ips in
    `/Library/Logs/DiagnosticReports` (to verify on the real failure).
  - A GPU fault or timeout shows as a Metal command-buffer error in the engine log (`kIOGPUCommandBufferCallbackErrorOutOfMemory`,
    `...Timeout`). `recoveryCount` increments on a GPU restart.
  - A code-signing kill shows as exit -9 with `"Killed: 9"`; see the system log for `CODESIGNING`.
- **warn_tight_ram** (L2233-2250): texts "Linux may stop the engine...", "Windows will slow down", "run START-HERE --setup".
  Needs a Mac branch.
  - It is called with `engine.info.get("arena_mib")` (L2491). On Mac compare against M - H, using W.

### 3.9 Firewall and LAN

- L2253 `lan_addresses()`: portable (UDP connect trick, getaddrinfo).
- L2578-2582: only a Windows Firewall hint. Mac equivalent, printed when the host is not loopback:
  - The macOS Application Firewall (enabled on this machine) prompts "Do you want the application 'Python' to accept
    incoming network connections?" for the **real** binary. For a Homebrew venv that is
    `/opt/homebrew/Cellar/python@3.13/3.13.x/Frameworks/Python.framework/Versions/3.13/Resources/Python.app`.
    The prompt appears on the first listen.
  - After a Homebrew Python update the binary path changes and the prompt reappears.
  - Command-line allow (admin): `sudo /usr/libexec/ApplicationFirewall/socketfilterfw --add <python> --unblockapp <python>`,
    where `<python>` is `os.path.realpath(sys.executable)`. Unsigned binaries get re-prompted; python.org builds are signed.
  - Local Network privacy (macOS 15+) applies to apps reaching **other** LAN devices, not to accepting connections.
    Clients on other devices are unaffected.
- `--open` (L2583-2585): `webbrowser.open` works on Mac (`open` command).

### 3.10 Port probe bug (macOS/BSD socket semantics)

L2442-2446:
```python
try:
    Server((a.host, a.port), BaseHTTPRequestHandler).server_close()
except OSError:
    ap.error(f"port {a.port} is already in use ...")
```

`Server.allow_reuse_address = os.name != "nt"` (L2226) sets SO_REUSEADDR on every POSIX host. Verified on this Mac:
- With SO_REUSEADDR, binding `127.0.0.1:P` succeeds while another process listens on `0.0.0.0:P`, and vice versa.
- Only the exact same address is refused (Errno 48).
- Result: Strata may start "successfully" on 8080 while, e.g., a Docker or dev server holds `*:8080`. Requests to
  127.0.0.1:8080 then go to whichever socket is more specific, so Strata's `127.0.0.1` socket wins locally. But LAN clients
  hit the other server.
- strata_mcp's `port_free` (strata_mcp.py:L298-305) binds without SO_REUSEADDR and detects the conflict correctly.
- Fix on darwin:
  - probe with a plain socket without SO_REUSEADDR (like port_free), plus a `connect_ex(("127.0.0.1", port))` check;
  - or set `allow_reuse_address = sys.platform != "darwin"` and rely on the bind error. TIME_WAIT is less of a problem
    for a listening socket restart on macOS, but keep REUSEADDR for the real server to allow fast restarts. Change only
    the probe.

### 3.11 Other server items

- L2438: the config is read as `utf-8-sig` (Notepad BOM). Harmless on Mac. TextEdit saves RTF by default, so docs should
  say "edit with nano or VS Code, or TextEdit in plain-text mode".
- `tempfile` paths live under `/var/folders/...` and are purged by the OS after about 3 days of non-access. The vision
  cache is per-process, so this is fine.
- serve/mcp.py (MCP client hub for the web chat):
  - L93-113: stdio transport via `subprocess.Popen(..., start_new_session=True)` (L103).
  - L212 `close`; L232-248 `_kill_tree` uses `os.killpg`.
  - L284 / L374: urlopen for HTTP transports.
  - Portable. Gotcha: a server started from a `.command`, a launchd agent or an MCP client inherits a minimal PATH
    (`/usr/bin:/bin:/usr/sbin:/sbin`), so `npx`/`uvx` in `/opt/homebrew/bin` are "not found". Document absolute paths, or
    prepend Homebrew dirs in `hub_from_config` on darwin.
- frontend.py (560 lines), structured.py (147 lines): portable; jsonschema is optional.

---------------------------------------------------------------------------------------------------------------------

## 4. The web app's Monitor tab: fields and what a Mac shows

Sources:
- serve/web/app.js: METRICS table L111-120, `renderMonitor` L205+, `renderAbout` L332+, and the `gb` helper L13 ("binary
  GB, as Windows shows it"; macOS Finder and Activity Monitor show decimal GB for disks but binary for memory, so keep it).
- serve/web/index.html: the cards comment at L92; the bars at L107-112.
- serve/web/monitor.js: the /api-monitor page. It uses only `/v1/status`, `/api/requests`, `/load`, `/unload`; it has no
  hardware fields and is portable.

| UI element | Field (source) | CUDA PC shows | Mac today | Mac proposal |
|---|---|---|---|---|
| Speed card (L112) | `live.tok_s` (server) | tok/s | works | same |
| GPU load card (L113; sub = `st.gpu_name` L246-247) | `hw.gpu_util`, `static.gpu_name` (NVML) | % + card name | dash, no name | ioreg "Device Utilization %"; name "Apple M5 Pro (20-core GPU)" |
| VRAM card (L114, L248-250) | `hw.gpu_mem_used` / `gpu_mem_total`, or `eng.expert_slots` "N experts cached" | GB / GB | dash (or experts cached, from INFO) | "GPU memory": ioreg "In use system memory" / W. Label "GPU memory (of N GB usable)". |
| GPU temp card (L115, L252-253) | `hw.gpu_temp` | deg C | dash | replace with "Thermal": `NSProcessInfo.thermalState` (Nominal/Fair/Serious/Critical), or hide |
| Power card (L116, L255 "of N W limit") | `gpu_power`, `gpu_power_limit` | W | dash | hide (root only) |
| PCIe card (L117, L257-260 "GenN xWidth, to GPU MB/s") | `gpu_pcie_rx_mb`, `gpu_pcie_gen(_max)`, `gpu_pcie_width` | MB/s | dash | replace with "Memory pressure" (`kern.memorystatus_vm_pressure_level`) or "Swap used" (`vm.swapusage`), or SSD read when streaming experts |
| CPU card (L118, L263 cores/threads) | `hw.cpu`, `st.cores`, `st.threads`, `st.cpu_name` | % + name | % works; name "arm"; "18 cores / 18 threads" | name from `machdep.cpu.brand_string`; "6 Super + 12 Performance cores" from hw.perflevel* |
| Disk read card (L119, L265-271 "needs psutil") | `hw.disk_read_mb` / `disk_write_mb` | MB/s | works (psutil disk_io_counters) | same; key indicator for the stream tier |
| "Experts in VRAM" bar (index.html L107; app.js L284-286 `expert_cache_mib / gpu_mem_total`) | `eng.expert_cache_mib`, `hw.gpu_mem_total` | % of VRAM | divides by a missing total, so empty | "Experts on the GPU": GPU share from INFO (gpu tier 100%) |
| "System RAM" bar (L109; app.js L287-290, red above 92%) | `ram_used` / `ram_total` | % | works, but `used` excludes compressed memory | tint by pressure level instead of a fixed 92% |
| "GPU temperature" bar (L111; app.js L291-292) | `gpu_temp` | | empty | thermal state |
| Hit-rate column tooltip (L120 "copied over PCIe") | request stats | | text only | "experts the GPU did not hold (run on the CPU or read from SSD)" |
| Multi-GPU rows | `hw.gpus[]` | per-card | none | none |
| About, "This PC" (L145; app.js L345 GPU fact "not readable (NVML)") | static | | "not readable (NVML)" | "Apple M5 Pro, 20-core GPU, 51.8 GB usable of 64 GB unified memory" |
| About, KV text ("streamed... in VRAM, the rest in RAM" / "all in VRAM") | `eng.kv`, `eng.kv_resident` | | "all in VRAM" | "in unified memory" |
| Chat empty-state "The model runs on this PC" (index.html L44), MCP note "on this PC" (L131) | text | | | "this Mac" (or neutral "this computer") |

Other engine fields shown (portable, from INFO): `max_context`, `kv`, `spec`, `mtp_max`, `lookup`, `cvec`, `version`,
`model`, `images`.

`/v1/status.machine.gpu` (server.py L1110-1113) maps `used_mib`, `total_mib`, `util_pct`, `temp_c`, `power_w` from the same
`gpu_*` keys. With the `_Apple` reader it works unchanged, with temp and power null.

---------------------------------------------------------------------------------------------------------------------

## 5. tools/strata_mcp.py (1,725 lines): the 8 tools and their platform assumptions

The MCP stdio server for AI agents. Requires Python >= 3.10 (`main` L1698). `PROTOCOL` "2025-06-18". `STATE_DIR` is
`.strata-mcp` in the Strata folder (L47). `DEFAULT_PORT` = 8080 (L46). Tool definitions are at L913.

### 5.1 Shared machinery and its macOS bugs

| Line | Item | Issue on macOS | Fix |
|---|---|---|---|
| L106-142 | `proc_identity(pid)`: WIN uses GetProcessTimes; Linux uses `/proc/<pid>/stat` field 22 and rejects Z/X; **elsewhere** `os.kill(pid, 0)` returns `"pid"` | (1) A zombie child (exited, not yet reaped) passes `kill(pid, 0)`, so it counts as running. (2) There is no start time, so a recycled PID is "ours". That breaks MCP_SERVER.md:L170-192's safety claim ("process id *and start time*"); `strata_stop` could SIGTERM an unrelated process after a reboot or PID wrap. | libproc, verified: `proc_pidinfo(pid, PROC_PIDTBSDINFO=3, 0, &proc_bsdinfo, 136)` returns 136 for a live process (pbi_status 2 = SRUN) and 0 for a zombie or reaped process. Identity = `f"mac:{pbi_start_tvsec}.{pbi_start_tvusec}"`. Struct layout in section 10 / `/tmp/strata-research/scratch/libproc_probe.py`. |
| L149 / L169 / L188 | `proc_terminate`, `proc_kill_tree` (killpg), `end_group_leftovers` | depend on proc_identity | OK once identity is fixed |
| L217-230 | `spawn_detached` (`start_new_session=True`) | portable | |
| L233 | `run_quiet` | portable | |
| L298-305 | `port_free` (bind without SO_REUSEADDR) | works correctly on macOS (section 3.10) | |
| L366 | `norm(p)`: realpath + normcase | APFS is case-insensitive by default, but `normcase` is identity on POSIX, so `/Users/x/Strata` vs `/Users/x/strata` compare unequal | on darwin, also `.lower()` (or compare with `os.path.samefile`) |
| L371-377 | `system_folders()` POSIX list: `/bin /boot /dev /etc /lib /lib32 /lib64 /proc /run /sbin /sys /usr /var /snap` | `norm("/var")` = `/private/var` and `norm("/etc")` = `/private/etc`, so every `tempfile` dir (`/var/folders/...` resolves to `/private/var/folders/...`) is rejected as "inside a system folder". The Mac system locations `/System`, `/Library`, `/Applications`, `/private`, `/cores`, `/opt` (Homebrew), `/Volumes/Recovery` are missing. | darwin list: `/System`, `/Library`, `/bin`, `/sbin`, `/usr` (but allow `/usr/local`?), `/private/etc`, `/private/var/db`, `/private/var/root`, `/dev`, `/cores`, `/Applications`, `/opt/homebrew`. Allow `/private/var/folders` (temp) and `/private/tmp`. |
| L428-431 | `settings_path()`: APPDATA or XDG | matches setup.py; OK | |
| L448-451 | `venv_python()`: `Scripts/python.exe` or `bin/python` | OK | |
| L580-636 | `_hardware()`: `S.ram_gb()` (setup), else `own_ram_gb()` (L845-861: WIN / `/proc/meminfo` / 0.0); `S.cpu_info()`; `S.gpus()` or `own_nvidia_gpus()` (L863); amd only when not WIN (`S.amd_gpus()`, else `rocm_smi_gpus()` L888); WIN `windows_video_controllers()` (L877); `hw["nvidia_smi"]` | On Mac: `S.ram_gb()` raises, `own_ram_gb()` returns **0.0**, so ram_gb = 0.0. cpu `{"name": "arm", "avx2": False}`. gpus = []. | ram via sysconf; cpu via sysctl; add `metal_devices()`; `hw["metal"] = {...}` |
| L638-687 | `recommend(hw)`: no usable GPU gives "no NVIDIA RTX 20-series-or-newer GPU found (nvidia-smi did not list one)"; RAM tiers 60 / Q2_0-4 / IQ1_M-4 / low_fits; ctx by VRAM; L679-680 refuses without AVX2; L683-684 `setup_command "./setup.sh --setup ..."` | Every Mac gets "no NVIDIA ... GPU found" | metal: reuse setup's `metal_fit` via `setup_module()`; command `./setup.sh --setup --yes --family F --model M --context C` (no `--backend` needed if metal is the darwin default) |
| L1539 | `INSTRUCTIONS` "runs ... locally on this PC's GPU and RAM" | text | "this computer's GPU and memory" |
| L1652-1690 | `install_job` / launcher: creates `.venv` with `base_python` = `sys.executable` (L1275); L1680 hint "Linux: install python3-venv" | If the MCP client runs strata_mcp.py with `/usr/bin/python3` (3.9), main refuses (>=3.10) | doc: use Homebrew python3 in the client config |

### 5.2 The tools

| # | Tool (line) | What it does | Platform assumptions | Mac fix |
|---|---|---|---|---|
| 1 | `strata_status` (L1044) | `reap()`; probes ports for Strata servers; the tracked server's state; the install job; installed models; `hardware()` + `recommend()` | proc_identity (zombie); hardware/recommend NVIDIA-only | fixes in 5.1 |
| 2 | `strata_models` (L1099) | the model table from setup's MODELS/FAMILIES/CONTEXTS (fallback copies L49+) with per-model fit for this PC (`model_table` L689) | fit = RAM/VRAM tiers | metal tiers |
| 3 | `strata_install` (L1142) | `install_plan` (L1188), then `launch_install` (L1268): runs setup.py with `--setup --yes ...` in a detached job; `check_data_dir` (L1116; example `/mnt/big` L1122) refuses system folders; progress via `install_progress` (L822) | L1193-1195 refuses hip only on WIN (a Mac would pass `backend=hip` through to setup); L1196-1198 says `./setup.sh` "installs Python" (false on Mac: it only uses apt/dnf/pacman); L1216-1220 default context from VRAM (12 GB assumed with no usable GPU); L1235 and L973 example `D:\Strata-data`; L1256 `setup_command`; system_folders bug rejects temp dirs | add metal; Mac examples (`~/Strata-data`, `/Volumes/External/Strata-data`); warn on iCloud folders |
| 4 | `strata_start` (L1298) | refuses while installing; `find_config`; port checks (`probe`, `port_free`); spawns `<venv python> serve/server.py --engine strata --config ... --port ...` detached, log `.strata-mcp/server.log`; `wait_ready` (L1352) polls `/v1/models` | `proc_identity` right after spawn returns "pid"; zombie detection in `wait_ready` (`p.poll()` is called first, which reaps, so this one path is OK) | identity fix |
| 5 | `strata_stop` (L1375) | busy check via `/status`; `POST /unload`; `proc_terminate` (SIGTERM, which runs the server's KeyboardInterrupt path); `wait_gone` 30 s; `proc_kill_tree`; `end_group_leftovers` | a recycled PID would be signalled (no start time); a zombie looks alive, so "still runs" | identity fix |
| 6 | `strata_logs` (L1427) | tails the install / server / model logs | portable | |
| 7 | `strata_benchmark` (L1450) | `POST /v1/chat/completions`, greedy, thinking off, BENCH_PROMPT, max_tokens 200; reports tok/s | portable; note text "the GPU's free VRAM" | text |
| 8 | `strata_connect_info` (L1492) | base URLs, key requirement, client config snippets (Claude Code env, `~/.codex/config.toml`, openai python, curl, Cursor/Continue/Open WebUI/aider) | portable | |

---------------------------------------------------------------------------------------------------------------------

## 6. Tests

### 6.1 Inventory, invocation, results on macOS

Invocation: from the repository root, `python -m unittest <module>`, e.g. `python -m unittest tools.test_setup_golden` and
`python -m unittest serve.test_server`. The files say this in their docstrings, e.g. test_setup_golden.py:L6-7. There is no
pytest config. `timeout(1)` is not on macOS, so run without it, or use `gtimeout` from coreutils.

None of these tests need a GPU. All run with mocks or the `--engine mock` server.

| Module | Tests | macOS result (Homebrew 3.13 venv) | Notes |
|---|---|---|---|
| tools.test_setup_choices | 15 | OK | |
| tools.test_setup_draft_vocab | 3 | OK | |
| tools.test_setup_lowram | 5 | OK | |
| tools.test_setup_pins | 13 | OK | |
| tools.test_setup_prompts | 7 | OK | |
| tools.test_setup_risk | 27 | OK | |
| tools.test_setup_rope | 19 | OK | |
| tools.test_setup_unsloth | 21 | OK | |
| tools.test_setup_amd | 18 | **1 FAIL** | `WindowsDetection.test_prebuilt_hip_zip` (L264-292): the fixture zip contains `strata.exe`, but `setup.EXE` is `"strata"` on POSIX, so `get_prebuilt_hip` returns None. **Fails on any non-Windows host**, Linux included: a pre-existing bug. Fix: build the zip with `setup.EXE`, or patch `setup.EXE`/`WIN`. |
| tools.test_setup_golden | 4 | **FAILED (failures=46)** | `test_yes` and `test_enter_for_every_question` fail 23 subtests each, only on the `"log"` key. See 6.2. |
| tools.test_shards | 7 | OK | |
| tools.test_calibrate | 10 | OK | |
| tools.test_strata_mcp | 22 (275 s) | **3 FAIL + 1 ERROR** | ERROR `test_data_dir_paths`: "/var/folders/.../Strata-big is inside a system folder" (system_folders + /private/var). FAIL `test_plan_then_install`, `test_failure_is_reported`, `test_progress_while_running_and_cancel`: `wait_for(not install_job running, 60)` never becomes true, because the finished install child is a zombie and counts as alive (`os.kill(pid, 0)`). Both are real macOS bugs (5.1), not test bugs. |
| tools.test_mtp_fetch | 8 | OK | |
| tools.test_iq_pack | 23 | OK with `STRATA_GGUF_PY=<llama.cpp>/gguf-py` | without it: "llama.cpp's gguf-py was not found" (needs setup's llama.cpp clone in third_party) |
| tools.test_conversation_cache_{disabled,growth,http_smoke,isolation,parity,soak} | 6+4+4+4+12+4 | OK | |
| serve.test_detok | 8 (5 skipped) | OK | skips need the real tokenizer pack |
| serve.test_structured | 8 (2 skipped) | OK | skips need jsonschema |
| serve.test_winjob | 2 skipped | OK | Windows only |
| serve.test_monitor | 7 | OK | |
| serve.test_lifecycle | 8 | OK | |
| serve.test_mcp | 25 | OK | uses serve/mcp_fake_server.py |
| serve.test_server | 88 (42 s) | OK | |

### 6.2 Golden test root cause and fix

- tools/test_setup_golden.py:L51-59 `normalize()`:
  ```python
  return v.replace(str(t), "<T>").replace("\\", "/").replace(setup.EXE, "<EXE>")
  ```
- The golden JSON was recorded on Windows, where `EXE` = `strata.exe`. On POSIX, `EXE` = `strata` is a substring of
  every Strata path: the log `<T>/strata-iq3_xxs.log` becomes `<T>/<EXE>-iq3_xxs.log` and mismatches. The `exe` key
  happens to normalize the same way on both. This fails on Linux too; it is not Mac-specific.
- Fix: replace only a whole path component, and also strip the resolved temp path (macOS `/var` resolves to `/private/var`):
  ```python
  s = v.replace(str(t.resolve()), "<T>").replace(str(t), "<T>").replace("\\", "/")
  return re.sub(r"(^|/)" + re.escape(setup.EXE) + r"$", r"\1<EXE>", s)
  ```
- Mocks (L93-124) patch:
  - `ROOT`, `GPU_PICK`, `data_folder`, `load_settings`, `save_settings`;
  - `gpus`, `amd_gpus`, `ram_gb`, `cpu_info -> ("Test CPU", True, avx512)`, `page_file_gb -> 16.0`, `is_wsl`, `free_gb -> 900`;
  - `pip_install`, `get_llama_cpp`, `get_prebuilt -> eng`, `update_installed_engine`, `download`, `check_shards`,
    `verify_sha256`, `run`;
  - `mtp_corrupt`, `refresh_draft_vocab`, `write_run_script`, `saved_calibration`, `calibrate_config` (raises if called);
  - `gguf_reader` in sys.modules, `argv`, `input`.
- BUILD.json fixture: `{"version": "0.1.32", "source": "local", "cuda_dirs": ["<cuda>"]}`. Profiles (L33-39) are 5 NVIDIA PCs.
- With the metal branch keyed on `sys.platform`, these NVIDIA goldens must also patch `setup.MAC = False` (or
  `metal_devices -> []`), so that they keep testing the CUDA path when run on a Mac.

### 6.3 Proposed macOS golden set (`tools/test_setup_golden_mac.json`, platform fully mocked so it runs on any host)

- Profiles, mocked `ram_gb` exact (no "a little under") and `metal_devices()`:

| Key | RAM (GiB) | W (GiB) | max_buffer | families | cores |
|---|---|---|---|---|---|
| `mac-24GB` | 24 | 16.0 | 0.75 W | apple9 | 10 |
| `mac-36GB` | 36 | 27.0 | ... | apple9 | 14 |
| `mac-48GB` | 48 | 36.0 | | apple9 | 20 |
| `mac-64GB-M5Pro` | 64 | 51.84 (live: 55,662,788,608 B; max_buffer 41,747,087,360) | | apple10, apple9 | 20 |
| `mac-96GB` | 96 | 72.0 | | apple9 | 38 |
| `mac-128GB` | 128 | 96.0 | | apple9 | 40 |
| `mac-256GB` / `mac-512GB` | 256 / 512 | 192 / 384 | | apple9 | 60-80 |

- Other patches:
  - `MAC=True`, `gpus -> []`, `amd_gpus -> []`, `cpu_info -> ("Apple M5 Pro", arm features)`, `page_file_gb -> None`,
    `is_wsl -> False`.
  - `get_prebuilt -> eng` with `BUILD.json {"version": ..., "backend": "metal", "arch": "arm64", "source": "prebuilt",
    "lib_dirs": []}`.
- Assertions per config:
  - `backend == "metal"`, `lib_dirs == []`, no `gpu` key, no `env` CUDA/HIP keys;
  - no `--kv-resident`, no `--pcie-frac`, no `--layer-split`;
  - the tier flags as decided in 2.4;
  - `--max-context` per 2.5;
  - calibration not offered;
  - run script `.command` + `.sh`.
- Refusal cases: 16 GiB (all no); 24 GiB without `--model` (stream only means refuse with `--yes` alone, matching
  upstream's "--yes alone is no consent" rule in `test_the_refusals_with_yes_alone_stay`).
- Extra unit tests:
  - `metal_fit` table rows (2.5) as data;
  - `proc_identity` zombie case (spawn `true`, don't wait, expect None);
  - `system_folders` with a `/var/folders` temp dir;
  - the port probe with a wildcard listener.

### 6.4 Tests needing hardware (not covered by unittest)

`tools/calibrate.py` (real engine), `tools/needle_bench.py` (a running server), and the conversation-cache scripts in
tools/ without the `test_` prefix (they need the engine). On a Mac they become the acceptance tests of the Metal engine.

### 6.5 What breaks on a Mac CI runner

GitHub `macos-15` / `macos-26` arm64 runners have 7 GB RAM and a 3-core Apple VM GPU (paravirtualized Metal; to be checked
for the families it reports). So:
- unit tests: fine;
- engine smoke tests: tiny fixtures only.

### 6.6 tools/calibrate.py (228 lines)

- Docstring: a Ryzen 5 7600 + RTX 5070 on PCIe 5.
- Sweeps `--pcie-frac` (`PCIE_FRACS` 0..0.75), `--spec-min-p` and `--pool-workers` using `serve.server.StrataEngine` and
  `child_env`; `apply()` rewrites args.
- `--pcie-frac` is meaningless with unified memory.
- A Mac calibration would instead sweep:
  - the GPU/CPU expert split in the hybrid tier;
  - CPU worker counts by core class: Super/P cores (`hw.perflevel0.physicalcpu` = 6) vs Performance/E (12). Thread QoS
    matters on macOS: `pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE)` in the engine.
  - `--spec-min-p` stays as is.
- Until then, gate calibration to `backend == "cuda"` (setup.py:L3214, L3221, L2639-2648).

---------------------------------------------------------------------------------------------------------------------

## 7. Docs needing macOS variants, by section

### 7.1 docs/INSTALL.md

| Section (lines) | Current | macOS change |
|---|---|---|
| L3 intro | "on Windows and Linux" | add macOS (Apple Silicon) |
| L12-21 "What you need" table: GPU, RAM, CPU (AVX2), Disk (Q2_0 AVX-512 extra), OS, Driver | NVIDIA/AMD rows | Mac column: Apple Silicon M1 or newer (decide the minimum family); unified memory 36/48/64 GB+ by model (2.5); macOS minimum; "no driver needed"; disk; Xcode only for `--build` |
| L23-28 auto-install text | | add "On a Mac, setup checks unified memory and the GPU's usable share" |
| L30-47 Windows | | new **macOS** section: install Python (`brew install python@3.13` or python.org), `git clone` (recommended; avoids quarantine) or zip download plus `xattr -dr com.apple.quarantine Strata`, `./setup.sh` in Terminal, or double-click `START-HERE.command` (Gatekeeper "Open Anyway" note) |
| L49-58 Linux | | unchanged |
| L60-74 AMD | | add "not on Mac" |
| L76-87 multi-card | | "not applicable on Mac (one GPU)" |
| L89-122 Docker | | "Docker on macOS has no GPU access: not supported" |
| L124-130 Updating | | same commands (`git pull`, `./setup.sh`) |
| L132-144 Where things are stored (L142-143 settings paths) | `%APPDATA%`, `~/.config/strata` | Mac: `~/.config/strata/settings.json` (as implemented); data-folder advice: not in iCloud-synced Desktop/Documents; Spotlight and Time Machine exclusions |
| L146-160 questions (vision GPU/CPU ~1.4 GB VRAM) | | "on a Mac the encoder uses about 1.2-1.4 GB of unified memory" |
| L162-167 Tuning / calibrate | | "not available on Mac yet" (or the new sweep) |
| L169-187 Options (D:\ examples) | | add `/Volumes/...` examples; `--backend metal` |

### 7.2 docs/TROUBLESHOOTING.md

| Section (lines) | Mac variant |
|---|---|
| L8-13 "PC froze while loading" | "Mac slow or beach-balling: memory pressure; Activity Monitor > Memory; choose a smaller model" |
| L18-20 NVIDIA driver | macOS version too old / unsupported chip (M1 family?) |
| L22-23 port in use | add `lsof -nP -iTCP:8080 -sTCP:LISTEN`; AirPlay Receiver holds **port 5000 and 7000** (not 8080); the SO_REUSEADDR caveat |
| L25-26 Python / build tools | Homebrew/python.org Python; Xcode + `xcodebuild -downloadComponent MetalToolchain` for `--build`; CLT stub dialog |
| L28-30 Task Scheduler autostart | launchd LaunchAgent plist (`~/Library/LaunchAgents/...`, `RunAtLoad`, `KeepAlive`, absolute paths, `EnvironmentVariables PATH`) |
| L34-35 disk light | SSD reads in the stream tier; Activity Monitor > Disk |
| L37-39 engine stopped ("on Linux the system then stops ...") | jetsam / memory pressure; `log show` predicate; GPU errors in the engine log; "Killed: 9" = signature |
| L45-47 slower than the tables (EXPO/XMP) | Low Power Mode (`isLowPowerModeEnabled`), battery vs AC (`pmset -g batt`), thermal throttling (thermalState), other GPU apps |
| L49-52 pictures (`--vision gpu` NVIDIA) | Metal encoder |
| L54-65 AMD | n/a |
| new | "macOS asks whether Python may accept incoming connections": the firewall (3.9) |
| new | "'strata' cannot be opened because Apple cannot check it" / "is damaged": quarantine; `xattr -dr com.apple.quarantine engine` |
| new | "Operation not permitted" on Desktop/Documents/external drive: TCC (System Settings > Privacy & Security > Files and Folders / Full Disk Access for Terminal) |

### 7.3 docs/AI_SETUP.md (the agent-facing guide)

| Section (lines) | Change |
|---|---|
| L4-5 NVIDIA/AMD | add Apple Silicon |
| L23-41 check table (Windows / Linux columns) | add a macOS column: `sysctl -n hw.memsize`, `sysctl -n machdep.cpu.brand_string`, `system_profiler SPDisplaysDataType` (GPU cores), `python3 --version`, `df -h`, the ctypes Metal probe (W) |
| L43-54 requirements | Mac rows from 2.5 |
| L56-64 get Strata | `git clone` (no quarantine) |
| L66-78 model by RAM | Mac table keyed by unified memory (2.5) |
| L80-121 setup flags table and notes | mark `--gpu/--gpus/--layer-split/--kv-streaming/--calibrate` as N/A on Mac; `--backend metal` |
| L123-130 | |
| L132-144 start (`nohup ./run-*.sh &`) | same on Mac; also `caffeinate -i ./run-<m>.sh` and the launchd option |
| L146-163 verify | same curl; `/v1/status.machine.gpu` fields |
| L165-174 connect | same |
| L176-187 failure table | Mac rows (7.2) |
| L192-196 MCP | python3 path note |

### 7.4 docs/MCP_SERVER.md

| Section (lines) | Change |
|---|---|
| L16 "64-bit on Windows" | add "macOS: Python 3.10+ from Homebrew or python.org (not /usr/bin/python3 3.9)" |
| L24-26 python3 on Linux | macOS same; use an **absolute** interpreter path in client configs (`/opt/homebrew/bin/python3`), because GUI apps do not inherit the shell PATH |
| L28-35 Claude Code | same command with `python3` |
| L37-53 Claude Desktop (`"command": "python"`) | macOS config file: `~/Library/Application Support/Claude/claude_desktop_config.json`; `"command": "/opt/homebrew/bin/python3"` |
| L55-69 Cursor, L71-87 VS Code, L89-98 Codex, L100-106 others | absolute python path note |
| L110-119 tools table | unchanged |
| L148-151 start "without a console window" / Windows job | "Terminal window" / background |
| L161-163 Stop (Linux SIGTERM / Windows job object) | macOS same as Linux (SIGTERM, process group) |
| L170-192 Safety ("process id *and start time*", L178 `D:\Strata-data`) | true on Mac only after the libproc fix; Mac path example |
| L194-205 troubleshooting | Mac rows |

### 7.5 Also affected (outside the four files)

- README.md: L49 "What you need", L61 "Install", L137 "Something went wrong".
- docs/DETAILS.md:
  - L256-280 "Before you start", L282-360 Windows (L332 Task Scheduler), L362-383 Linux;
  - L384-403 "sharing the GPU" (`idle_unload_s`, `min_free_vram_mib`, `before_load`): on Mac it is "sharing memory";
  - L774-798 troubleshooting table (cudaHostRegister, page file, dmesg, EXPO/XMP, vision gpu).

---------------------------------------------------------------------------------------------------------------------

## 8. Python version and dependency constraints

- **Minimum Python 3.10**: setup.sh:L8, START-HERE.bat:L49, strata_mcp.py `main` L1698, requirements.txt comment L3. The
  code uses `X | None` annotations and `match`-free syntax. setup.py and server.py run under 3.9 only by accident of
  `from __future__ import annotations`; the probes were verified on 3.9.6 too.
- macOS sources:
  - `/usr/bin/python3` 3.9.6 (CLT): **too old**.
  - Homebrew `python@3.12` / `python@3.13`: recommended; arm64; certifi-free SSL via Homebrew's openssl CA bundle.
  - python.org universal2 installer: needs "Install Certificates.command" for urllib SSL; its binary is signed, which
    helps with firewall prompts.
  - uv-managed Python (python-build-standalone): works; SSL via its bundled certs.
  - Conda / Miniforge: works (arm64).
- requirements.txt (23 lines). All of the following install as **macOS arm64 cp313 wheels**, verified with
  `uv pip install --only-binary :all:`:
  - `numpy==2.5.3` (py>=3.12; 2.4.6 for 3.11; 2.2.6 for <3.11);
  - `jinja2==3.1.6`, `regex==2026.9.10`, `pyyaml==6.0.3`, `tqdm==4.70.1`, `requests==2.34.2`;
  - `cmake==4.4.3`, `ninja==1.13.2`, `pillow==12.3.0`, `psutil==7.2.2`;
  - `markupsafe==3.0.3`, `certifi==2026.7.22`, `charset-normalizer==3.5.1`, `idna==3.20`, `urllib3==2.8.0`;
  - `colorama==0.4.6` is win32 only.
- Never on Mac: `CUDA_WHEELS` (nvidia-cublas/cuda-runtime etc., setup.py:L94-95); no macOS builds exist. Also ROCm.
- psutil is effectively **required** on Mac: the telemetry fallback reads /proc (3.7).
- Optional: jsonschema (structured output validation; tests skip without it). pyobjc is not needed (ctypes is enough).
- pip policy: this research environment restricts pip by package upload date (the installs were checked with
  `uv --exclude-newer` instead). setup's `pip_install` calls pip normally, so a user with a similar pip policy gets
  old wheels or resolution failures. That is not a Mac issue; just note it.
- Build tools for `--build`:
  - Xcode 27 (the `metal`/`metallib` compilers; a MobileAsset component) and cmake/ninja from pip;
  - no Homebrew needed;
  - llama.cpp's `-DGGML_METAL=ON` (vision) needs the same Metal toolchain unless `GGML_METAL_EMBED_LIBRARY` compiles from
    source at runtime. It embeds the source and compiles at first use with the runtime compiler, which has no Xcode
    dependency at run time.

---------------------------------------------------------------------------------------------------------------------

## 9. Packaging and distribution

### 9.1 What exists

- Assets: `strata-windows-x64.zip`, `strata-linux-x64.zip` (setup.py:L93), `WIN_HIP_ASSET` (L1067) for Windows AMD.
- URLs: `PREBUILT_URL` = latest, `PREBUILT_TAG_URL` = `v{version}` (L91-92). `prebuilt_bases` tries the tag first (#214).
- Versioning:
  - `source_version()` reads `CMakeLists.txt project(strata VERSION x.y.z)`.
  - The binary reports `INFO engine=X.Y.Z`; `engine_version()` scans for `engine=X.Y.Z\n`.
  - Gate `MIN_ENGINE = (0,1,34)`. Feature gates: `RESIDENT_ENGINE (0,1,30)`, `UNSLOTH_ENGINE (0,1,32)`.
- BUILD.json keys: `version`, `source` ("local" or prebuilt), `archs` (int sm list), `cuda`, `cuda_dirs`, `lib_dirs`,
  `vision` ("gpu"/"cpu"/"none"), `ptx`, `backend` ("hip"), `hipblaslt_version`, `src`, `vision_src`.
- `tools/make_release.py` is referenced (L86-90) but missing, so the release build is not reproducible from the repository.

### 9.2 How setup picks an asset (today)

`WIN` gives the windows asset, else linux. Then:
1. HEAD the tag URL, then latest.
2. Download.
3. `zipfile.extractall`.
4. Version >= MIN_ENGINE.
5. Archs cover the card (or PTX newer).
6. Move into `engine/`.
7. chmod 755.

With hip+WIN, `get_prebuilt_hip` is used instead.

### 9.3 What a macOS asset needs

1. **Name**: `strata-macos-arm64.zip`. Select it when `sys.platform == "darwin" and platform.machine() == "arm64"`, and
   refuse when `sysctl -n sysctl.proc_translated` = 1 (Python running under Rosetta reports x86_64).
2. **Contents**: `strata`, `strata-vision`, `BUILD.json`, and the Metal library.
   - Either embed the `.metallib` in the binary (a `__TEXT,__metallib` section, or `newLibraryWithData`), or ship
     `default.metallib` next to the binary, found via `[[NSBundle mainBundle] executablePath]` or `_NSGetExecutablePath`,
     never via cwd.
   - No dylibs, preferably (static ggml). If any: `@rpath` with `LC_RPATH @executable_path`, because DYLD_* variables are
     ignored under the hardened runtime and stripped across SIP-protected shells.
   - **No symlinks** (zipfile drops them).
3. **BUILD.json (metal)**: `{"version": "x.y.z", "backend": "metal", "arch": "arm64", "macos_min": "15.0",
   "gpu_families": ["apple9", "apple10"], "metal_std": "metal4", "source": "prebuilt", "lib_dirs": [], "vision": "gpu",
   "src": ..., "vision_src": ...}`.
4. **Build flags**: `-DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=<macos_min>`. `metal -std=metal4.0` (or
   3.2) with `-mmacosx-version-min`. One metallib per minimum family, if families differ in kernels.
5. **Code signing** (arm64 requires a valid signature to execute; the linker applies an ad-hoc one automatically):
   - Files downloaded by setup.py (urllib) carry **no quarantine** (verified: only `com.apple.provenance`), so an ad-hoc
     signed binary runs. That is enough for the setup.py path.
   - For browser downloads (a user fetching the zip, or `--prebuilt` from a local folder with an unzipped Finder copy), the
     quarantine flag triggers Gatekeeper. An ad-hoc or unsigned binary is blocked ("cannot be opened because Apple cannot
     check it for malicious software"; on macOS 15+ only System Settings > "Open Anyway" works).
   - Proper fix: **Developer ID Application** signing with the **hardened runtime** (`codesign -s "Developer ID ..."
     --options runtime --timestamp`), then **notarization** (`xcrun notarytool submit strata-macos-arm64.zip --wait`).
   - A bare Mach-O **cannot be stapled** (`stapler` works on .app, .pkg and .dmg only), so Gatekeeper fetches the ticket
     online on first run. Offline first runs fail unless the binaries are wrapped in a signed/notarized `.pkg` or `.dmg`
     (stapled).
   - Entitlements: probably none (no JIT; Metal needs none). If the engine uses `MAP_JIT` or loads unsigned code, it
     needs `com.apple.security.cs.allow-jit`. Avoid `disable-library-validation`.
6. **Quarantine hygiene in setup**: after placing files from a local `--prebuilt` folder, run
   `xattr -dr com.apple.quarantine engine/` (and the repository folder on first run, if it was zip-downloaded). It is
   harmless when absent. Do not remove `com.apple.provenance`.
7. **Smoke test** after extraction: run `engine/strata --version` (or `--list-devices`).
   - Exit -9 / SIGKILL with no output means a signature or quarantine problem.
   - "Library not loaded" means an rpath problem.
   - A Metal "unsupported family" error means an old chip. Map each to a clear message.
8. **Distribution alternatives**:
   - A Homebrew tap formula: `brew install strata` builds from source with Xcode, or uses a bottle. Bottles are relocatable
     and ad-hoc re-signed by brew.
   - A signed/notarized `.pkg` for the whole folder (Python sources + engine). Notarization rejects unsigned Mach-O inside,
     so the venv cannot be shipped; setup still creates it.
   - No Mac App Store: sandboxing would forbid the data folder and server model.
9. **Versioning**: keep one version line across backends. A separate `MIN_ENGINE_METAL` (and `RESIDENT_ENGINE`-style
   feature gates) lets the Metal engine start at, e.g., 0.2.0 without blocking CUDA.
10. **Release CI**: a `macos-15`/`macos-26` arm64 runner with Xcode 27 plus the Metal toolchain component. Signing secrets
    (Developer ID p12, App Store Connect API key for notarytool) are needed. `tools/make_release.py` must be (re)written
    to produce all assets plus `BUILD.json`.

---------------------------------------------------------------------------------------------------------------------

## 10. Prioritized work list

**P0: setup runs on a Mac**
1. setup.sh: the darwin Python search and hint (1.14).
2. setup.py:
   - `MAC`, `metal_devices()`, `ram_gb()` via sysconf, `cpu_info()` arm64 (1.3);
   - the metal branch in step 1 before the L2716 NVIDIA fail;
   - the ISA gate instead of AVX2 (L2761-2762).
3. `PREBUILT_ASSET` for darwin, and the `get_prebuilt` metal branch (L93, L1509-1590). No CUDA_WHEELS (L3021).
   `lib_dirs = []`.
4. `metal_fit()` and the recommendations (2.4-2.5). Never `--kv-resident` on metal (L3139-3166). The calibration gate is
   `backend == "cuda"` (L3214, L3221).
5. start(): skip `gpus()`/`ensure_engine_for`/`offer_together` on metal; banner text (L2353-2398).
6. serve/server.py `child_env`: no `LD_LIBRARY_PATH` on darwin; add `backend == "metal"` handling (L652-667).

**P1: correctness on macOS**

7. strata_mcp `proc_identity` via libproc (L106-142); `system_folders` darwin list (L371-377); `own_ram_gb` (L845-861);
   `recommend` metal (L638-687).
8. server.py: port probe without SO_REUSEADDR on darwin (L2442-2446); SIGHUP handler; firewall hint (L2578-2582);
   death_note / warn_tight_ram Mac texts (L281-302, L2233-2250).
9. Spaces in GENI/ENC paths (protocol), coordinated with the native engine.
10. Engine exits on stdin EOF (no job objects on Mac), to verify natively.

**P2: UX**

11. Telemetry `_Apple` reader (ioreg util/mem, W, thermalState) and Monitor cards (section 4).
12. `free_vram_mib` metal: `min_free_ram_mib` (3.6).
13. `.command` run scripts, START-HERE.command (1.11).
14. caffeinate / IOPM sleep assertions during downloads and generation.
15. Data folder safety: iCloud, Spotlight, Time Machine, TCC (1.10).
16. SSL context with certifi for python.org Pythons (1.6).

**P3: tests, docs, release**

17. Fix test_setup_golden `normalize` and test_setup_amd `test_prebuilt_hip_zip` (pre-existing POSIX bugs); add
    `MAC=False` patches to the NVIDIA goldens; add the macOS golden set (6.3).
18. Docs (section 7).
19. Release: make_release.py, a signed and notarized `strata-macos-arm64.zip`, a metal `BUILD.json`, a macOS CI runner
    (section 9).
20. calibrate.py metal sweep (6.6).

---------------------------------------------------------------------------------------------------------------------

## Appendix A. Probe snippets (all verified on the live Mac, no root)

```sh
sysctl -n hw.memsize machdep.cpu.brand_string hw.model hw.nperflevels \
       hw.perflevel0.name hw.perflevel0.physicalcpu hw.perflevel1.name hw.perflevel1.physicalcpu
sysctl hw.optional.arm.FEAT_DotProd hw.optional.arm.FEAT_I8MM hw.optional.arm.FEAT_BF16 hw.optional.arm.FEAT_SME
sysctl iogpu.wired_limit_mb vm.global_user_wire_limit vm.global_no_user_wire_amount
sysctl kern.memorystatus_vm_pressure_level kern.memorystatus_level vm.swapusage sysctl.proc_translated
ioreg -r -d 1 -w 0 -c IOAccelerator -a        # plist: "model", "gpu-core-count", PerformanceStatistics
xcrun -f metal; xcode-select -p
/usr/libexec/ApplicationFirewall/socketfilterfw --getglobalstate
xattr -l <file>                               # com.apple.quarantine vs com.apple.provenance
lsof -nP -iTCP:8080 -sTCP:LISTEN
```

```python
# libproc process identity (replaces os.kill(pid, 0) in strata_mcp.proc_identity on darwin)
import ctypes, ctypes.util
class proc_bsdinfo(ctypes.Structure):
    _fields_ = [("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32), ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32), ("pbi_ppid", ctypes.c_uint32), ("ids", ctypes.c_uint32 * 6),
                ("rfu_1", ctypes.c_uint32), ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32), ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32), ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64)]   # sizeof = 136
libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
def mac_identity(pid):
    b = proc_bsdinfo()
    n = libproc.proc_pidinfo(int(pid), 3, ctypes.c_uint64(0), ctypes.byref(b), ctypes.sizeof(b))   # PROC_PIDTBSDINFO
    if n != ctypes.sizeof(b):        # 0 for a zombie or a gone process (verified)
        return None
    return f"mac:{b.pbi_start_tvsec}.{b.pbi_start_tvusec}"
```

```python
# thermal state / low power (public NSProcessInfo API via ctypes)
import ctypes, ctypes.util
objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
objc.objc_getClass.restype = ctypes.c_void_p; objc.objc_getClass.argtypes = [ctypes.c_char_p]
objc.sel_registerName.restype = ctypes.c_void_p; objc.sel_registerName.argtypes = [ctypes.c_char_p]
def send(o, s, rt=ctypes.c_void_p):
    return ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(rt, ctypes.c_void_p, ctypes.c_void_p))(o, objc.sel_registerName(s))
pi = send(objc.objc_getClass(b"NSProcessInfo"), b"processInfo")
thermal = send(pi, b"thermalState", ctypes.c_long)          # 0 nominal .. 3 critical  (live: 0)
low_power = send(pi, b"isLowPowerModeEnabled", ctypes.c_bool)
```

The Metal device probe is in section 1.3 (`metal_devices()`) and `/tmp/strata-research/scratch/metal_probe.py`.
