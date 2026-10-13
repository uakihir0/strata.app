#!/usr/bin/env python3
"""macOS setup entry point. Installation is disabled until the Apple Silicon flow is complete."""
from __future__ import annotations

import errno
import functools
import hashlib
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from serve import macos  # noqa: E402
from tools.model_catalog import CONTEXTS, FAMILIES, MODELS  # noqa: E402

# True on macOS. Every Mac branch tests this one switch (docs/mac/07-server-setup-port.md, section 4.1).
MAC = sys.platform == "darwin"

# The Metal planner (docs/mac/07-server-setup-port.md, section 4.4). The values are from the table in section 4.1.
ENGINE_PATH = ROOT / "engine" / "strata"
DATA_DIR = ROOT / "data"
GEOMETRY_DIR = DATA_DIR / "geometry"
PLAN_TIMEOUT_S = 30
METAL_SHARE_OK = 0.85
METAL_SHARE_SLOW = 0.40
VISION_METAL = {"gpu": {"max_tokens": 1024, "reserve_mib": 2100}, "cpu": {"max_tokens": 300, "reserve_mib": 700}}
# --vram-reserve-mib when no vision model runs: the engine's default (05-memory-and-residency.md, section 3.1).
DEFAULT_RESERVE_MIB = 700
# The --kv names the engine accepts, which are also the names it echoes on the PLAN line (plan.cpp, kv_format_name).
PLAN_KV = ("fp16", "int8", "k8v4", "q4_0")
# Every key that plan_line() prints, in the engine's order (05 section 3.4). `reason` is printed only with fits=0.
PLAN_FIELDS = (
    "fits", "mode", "gpu", "w_mib", "reserve_mib", "ram_mib", "avail_mib", "dense_mib", "unread_mib", "pack_mib",
    "mtp_mib", "mtp_bind_mib", "fixed_mib", "kv_mib", "mtp_kv_mib", "kv", "context", "prefill_chunk", "prefill_mib",
    "experts_total", "experts_total_mib", "experts_resident", "experts_resident_mib", "slots", "slot_mib",
    "ssd_experts", "total_mib", "free_mib", "ram_warning", "w_full_mib", "mtp_params_mib", "session_state_mib",
    "window_scratch_mib",
)
PLAN_TEXT_FIELDS = ("mode", "gpu", "kv")
PLAN_REASON_PREFIXES = ("the_model_does_not_fit:", "unsupported_type_")
# Test seam: metal_plan starts the engine only through PLAN_RUNNER, so a test can replace it with a fake.
PLAN_RUNNER = subprocess.run

_PLAN_KEY = re.compile(r"[a-z][a-z0-9_]*")
_PLAN_INT = re.compile(r"-?[0-9]+")
_HINT_QUARANTINE = ("macOS refused to run the engine (code signature or quarantine): xattr -dr "
                    "com.apple.quarantine engine, then run setup again")
_HINT_INCOMPLETE = "the engine's files are incomplete: delete the engine folder and run setup again"
_HINT_TIMEOUT = (f"the engine took more than {PLAN_TIMEOUT_S} s to plan: run engine/strata --plan-only ... "
                 "by hand and report its output")
_HINT_OTHER = "delete the engine folder and run setup again (setup --build compiles one)"

# The Mac install (docs/mac/07-server-setup-port.md, sections 4.1, 4.7, 4.10 and 4.12). The release URLs need a
# publication destination (OQ-01): until RELEASE_REPOSITORY is set, no ready-made engine is looked for by default.
PREBUILT_ASSET = "strata-macos-arm64.zip"
PREBUILT_TAG_URL = "https://github.com/{repository}/releases/download/v{version}-metal.{metal_rev}/"
PREBUILT_URL = "https://github.com/{repository}/releases/latest/download/"
RELEASE_REPOSITORY = None
MIN_ENGINE = (0, 1, 34)
MIN_ENGINE_METAL = 1
MACOS_MIN = (27, 0)
METALLIB = "strata.metallib"
ENGINE_DIR = ROOT / "engine"
ENGINE_EXECUTABLES = ("strata", "strata-vision")
SMOKE_TIMEOUT_S = 30
# Folders that macOS privacy controls protect, or that iCloud Drive and sync clients may upload (section 4.12).
MAC_SYNCED_PARENTS = ("Desktop", "Documents", "Downloads", "Library/Mobile Documents", "Library/CloudStorage")
HINT_CERTIFICATES = ('this Python has no certificates: run "Install Certificates.command" in /Applications/Python 3.x '
                     "(python.org's installer), or use Homebrew's Python")
HINT_TCC = ("macOS blocked access: allow Terminal in System Settings > Privacy & Security > Files and Folders "
            "(or Full Disk Access), or choose --data-dir ~/Strata-data")
# Test seam: the smoke test starts the engine only through SMOKE_RUNNER, as the plan does through PLAN_RUNNER.
SMOKE_RUNNER = subprocess.run


class PlanError(RuntimeError):
    """The engine gave no usable plan. Setup stops here; it never substitutes a plan."""


class SetupError(RuntimeError):
    """Setup stops here with this message. Nothing after the failed check has been changed."""


def parse_plan_line(line: str) -> dict:
    """Parse one PLAN line. Anything missing, repeated, mistyped or inconsistent raises PlanError."""
    if not line.startswith("PLAN "):
        raise PlanError(f"the engine's output is not a PLAN line: {line[:120]!r}")
    raw: dict[str, str] = {}
    for token in line.split()[1:]:
        key, equals, value = token.partition("=")
        if not equals or not _PLAN_KEY.fullmatch(key) or not value:
            raise PlanError(f"malformed PLAN field {token[:80]!r}")
        if key in raw:
            raise PlanError(f"PLAN repeats {key}")
        raw[key] = value
    missing = [key for key in PLAN_FIELDS if key not in raw]
    if missing:
        raise PlanError("PLAN lacks " + ", ".join(missing))

    plan: dict = {}
    for key, value in raw.items():
        if key == "reason":
            continue
        if key in PLAN_TEXT_FIELDS:
            plan[key] = value
        elif _PLAN_INT.fullmatch(value):
            plan[key] = int(value)
        elif key in PLAN_FIELDS:
            raise PlanError(f"PLAN {key} is not an integer: {value[:80]!r}")
        else:
            plan[key] = value  # a key the engine appends later (05 section 3.4) is kept, not checked
    if plan["fits"] not in (0, 1):
        raise PlanError(f"PLAN fits must be 0 or 1, not {plan['fits']}")
    if plan["ram_warning"] not in (0, 1):
        raise PlanError(f"PLAN ram_warning must be 0 or 1, not {plan['ram_warning']}")
    if plan["mode"] not in ("full", "tiered"):
        raise PlanError(f"PLAN mode is {plan['mode']!r}")
    if plan["kv"] not in PLAN_KV:
        raise PlanError(f"PLAN kv is {plan['kv']!r}")
    for key in PLAN_FIELDS:
        if key in PLAN_TEXT_FIELDS or key in ("fits", "ram_warning", "free_mib"):
            continue
        floor = -1 if key == "avail_mib" else 0  # avail_mib is -1 when the engine was not given it
        if plan[key] < floor:
            raise PlanError(f"PLAN {key} is {plan[key]}, below {floor}")
    if plan["experts_resident"] > plan["experts_total"]:
        raise PlanError("PLAN has more resident experts than experts")

    if plan["fits"] == 1:
        if "reason" in raw:
            raise PlanError("PLAN has a reason but fits=1")
        return plan
    if "reason" not in raw:
        raise PlanError("PLAN fits=0 has no reason")
    reason_raw = raw["reason"]
    if not reason_raw.startswith(PLAN_REASON_PREFIXES):
        raise PlanError(f"PLAN reason is not one the engine gives: {reason_raw[:120]!r}")
    plan["reason_raw"] = reason_raw  # the underscores are kept: "unsupported_type_q4_K" cannot be undone
    plan["reason"] = reason_raw.replace("_", " ")
    return plan


def _plan_value(p: dict, key: str):
    if key not in p:
        raise PlanError(f"the plan has no {key}")
    return p[key]


def plan_class(p: dict) -> str:
    """Classify a parsed plan for the menus and --check (07 section 4.4.2). A missing field raises."""
    fits = _plan_value(p, "fits")
    if type(fits) is not int or fits not in (0, 1):
        raise PlanError(f"plan fits must be 0 or 1, not {fits!r}")
    if fits == 0:
        reason_raw = _plan_value(p, "reason_raw")
        if not isinstance(reason_raw, str):
            raise PlanError("plan reason_raw is not text")
        return "unsupported" if reason_raw.startswith("unsupported_type_") else "none"
    mode = _plan_value(p, "mode")
    if mode == "full":
        return "full"
    if mode != "tiered":
        raise PlanError(f"plan mode is {mode!r}")
    total = _plan_value(p, "experts_total")
    resident = _plan_value(p, "experts_resident")
    if total <= 0:
        raise PlanError(f"plan has {total} experts, so its share cannot be classed")
    share = resident / total
    return "ok" if share >= METAL_SHARE_OK else "slow" if share >= METAL_SHARE_SLOW else "very slow"


def _real_files_args(files) -> list[str]:
    if not isinstance(files, tuple) or len(files) != 4:
        raise ValueError("files must be a tuple (pack, shard1, ple or None, mtp_rt)")
    pack, shard1, ple, mtp_rt = files
    for name, value in (("pack", pack), ("shard1", shard1), ("mtp_rt", mtp_rt)):
        if value is None or str(value) == "":
            raise ValueError(f"files: {name} is missing")
    args = ["--pack", str(pack), "--native", str(shard1)]
    if ple is not None:
        if str(ple) == "":
            raise ValueError("files: ple is empty; pass None when the model has no PLE file")
        args += ["--ple-gguf", str(ple)]
    return args + ["--mtp", str(mtp_rt)]


def _plan_failure(result, what: str) -> PlanError:
    stderr_tail = "\n".join(result.stderr.splitlines()[-5:]) or "(no stderr)"
    if result.returncode == -9 and not result.stdout and not result.stderr:
        hint = _HINT_QUARANTINE
    elif "Library not loaded" in result.stderr:
        hint = _HINT_INCOMPLETE
    else:
        hint = _HINT_OTHER
    return PlanError(f"no plan from the engine ({what}).\n{stderr_tail}\n{hint}")


@functools.lru_cache(maxsize=None)
def metal_plan(family, model, ctx, kv, vision="none", budget_gib=None, files=None, gpu_budget_mib=None) -> dict:
    """The engine's memory plan for one model: `engine/strata --plan-only` and its PLAN line (07 section 4.4.1).

    Without `files` the plan comes from the model's geometry file. `files` is (pack, shard1, ple or None, mtp_rt)
    for downloaded files. A request that is not valid raises ValueError before the engine starts. A failed engine
    run, a missing or malformed PLAN line, or a plan that does not describe the request raises PlanError. The
    result is cached and shared between callers: do not modify it.
    """
    if family not in FAMILIES:
        raise ValueError(f"unknown model family {family!r}")
    if model not in MODELS:
        raise ValueError(f"unknown model {model!r}")
    if type(ctx) is not int or ctx <= 0:
        raise ValueError(f"context must be a positive integer, not {ctx!r}")
    if kv not in PLAN_KV:
        raise ValueError(f"unknown KV format {kv!r}")
    if vision == "none":
        reserve = DEFAULT_RESERVE_MIB
    elif vision in VISION_METAL:
        reserve = VISION_METAL[vision]["reserve_mib"]
    else:
        raise ValueError(f"unknown vision mode {vision!r}")

    cmd = [str(ENGINE_PATH), "--plan-only"]
    if files is None:
        cmd += ["--geometry", str(GEOMETRY_DIR / f"{family}-{model}.txt")]
    else:
        cmd += _real_files_args(files)
    profile = FAMILIES[family].get("profile", "expert-profile.bin")
    cmd += ["--expert-profile", str(DATA_DIR / profile), "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "4", "--max-context", str(ctx), "--kv", kv, "--vram-reserve-mib", str(reserve)]
    if budget_gib is not None:
        cmd += ["--resident-budget-gib", str(budget_gib)]
    if gpu_budget_mib is not None:
        cmd += ["--gpu-budget-mib", str(gpu_budget_mib)]

    try:
        result = PLAN_RUNNER(cmd, capture_output=True, text=True, timeout=PLAN_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise PlanError(f"no plan from the engine (timed out).\n{_HINT_TIMEOUT}") from exc
    except OSError as exc:
        raise PlanError(f"no plan from the engine (cannot start {ENGINE_PATH}: {exc}).\n{_HINT_OTHER}") from exc

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode not in (0, 3):
        raise _plan_failure(result, f"exit {result.returncode}")
    if len(lines) != 1 or not lines[0].startswith("PLAN "):
        raise _plan_failure(result, f"exit {result.returncode} without exactly one PLAN line on stdout")
    plan = parse_plan_line(lines[0])
    expected_rc = 0 if plan["fits"] == 1 else 3
    if result.returncode != expected_rc:
        raise PlanError(f"the engine exited {result.returncode} but its PLAN says fits={plan['fits']}")
    if plan["context"] != ctx:
        raise PlanError(f"the engine planned context {plan['context']}, not the requested {ctx}")
    if plan["kv"] != kv:
        raise PlanError(f"the engine planned KV {plan['kv']!r}, not the requested {kv!r}")
    return plan


# --- Run scripts and quarantine (docs/mac/07-server-setup-port.md, section 4.10) --------------------------------

QUARANTINE_FILES = ("START-HERE.command", "SETUP.command", "setup.sh")
_MODEL_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def quarantine_targets(root: Path = ROOT) -> list[Path]:
    """The only paths setup strips quarantine from: the two launchers, setup.sh and the engine folder. Never a
    downloaded GGUF, a folder the user chose, or ROOT itself."""
    return [root / name for name in QUARANTINE_FILES] + [root / "engine"]


def strip_quarantine_targets(root: Path = ROOT) -> None:
    macos.strip_quarantine(*quarantine_targets(root))


def write_run_script(root: Path, model: str, python: str, config: Path, port: int) -> tuple[Path, Path]:
    """Write run-<model>.sh and run-<model>.command. Every argument is quoted with shlex.quote, and the server starts
    through /usr/bin/arch -arm64, so a Rosetta shell cannot start an x86_64 Python. The .command adds a window title
    for a Finder double-click."""
    if not _MODEL_NAME.fullmatch(model):
        raise ValueError(f"model name {model!r} is not a plain file name")
    argv = ["/usr/bin/arch", "-arm64", python, str(root / "serve" / "server.py"), "--engine", "strata",
            "--config", str(config), "--port", str(port), "--open"]
    body = f"cd {shlex.quote(str(root))} || exit 1\nexec {' '.join(shlex.quote(arg) for arg in argv)}\n"
    script = root / f"run-{model}.sh"
    command = root / f"run-{model}.command"
    script.write_text("#!/bin/sh\n" + body, encoding="utf-8")
    command.write_text("#!/bin/sh\n" + f"printf '\\033]0;Strata {model}\\007'\n" + body, encoding="utf-8")
    for path in (script, command):
        path.chmod(0o755)
    return script, command


# --- The data folder (docs/mac/07-server-setup-port.md, section 4.12) ------------------------------------------

def data_folder(root: Path = ROOT, saved: str | None = None) -> Path:
    """Where models and caches are kept. A folder from the settings file is kept wherever it is. On a Mac, a ROOT
    inside a privacy-protected or synced folder uses ~/Strata-data instead, and an ok line says why."""
    if saved:
        return Path(saved).expanduser()
    default = root.parent / "Strata-data"
    if not MAC:  # MAC: the protected-folder rule of section 4.12 applies on a Mac only
        return default
    home = Path.home()
    parent = root.parent.resolve()
    for relative in MAC_SYNCED_PARENTS:
        if parent.is_relative_to((home / relative).resolve()):
            target = home / "Strata-data"
            print(f"ok: data folder {target}: {root.parent} is in {relative}, which macOS privacy controls or a sync "
                  "client may protect or upload", flush=True)
            return target
    return default


def create_data_folder(folder: Path) -> None:
    """Create the data folder. On a Mac it gets an empty .metadata_never_index, so Spotlight does not index 70-120 GB of
    GGUF files, and a Time Machine exclusion."""
    try:
        folder.mkdir(parents=True, exist_ok=True)
    except PermissionError as exc:
        if MAC and exc.errno == errno.EPERM:  # MAC: the TCC hint of section 4.12
            raise SetupError(f"cannot create {folder}. {HINT_TCC}") from exc
        raise
    if not MAC:  # MAC: Spotlight and Time Machine apply on a Mac only (section 4.12)
        return
    (folder / ".metadata_never_index").touch(exist_ok=True)
    if macos.tm_exclude(folder):
        print(f"ok: {folder} is excluded from Time Machine backups (tmutil removeexclusion {folder} undoes it)",
              flush=True)
    else:
        print(f"note: could not exclude {folder} from Time Machine backups", flush=True)


# --- The ready-made engine (docs/mac/07-server-setup-port.md, section 4.7) ------------------------------------

def source_version() -> str:
    """The CMake project version, which is the release tag's version (section 4.1)."""
    match = re.search(r"project\(\s*strata\s+VERSION\s+([0-9]+\.[0-9]+\.[0-9]+)",
                      (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    if match is None:
        raise SetupError("CMakeLists.txt has no project VERSION")
    return match.group(1)


def metal_source_rev() -> int:
    """The integer in METAL_REV at the repository root. 0 when the file is missing (section 4.1)."""
    path = ROOT / "METAL_REV"
    if not path.exists():
        return 0
    text = path.read_text(encoding="utf-8").strip()
    if not text.isdigit():
        raise SetupError(f"METAL_REV must hold one integer, not {text!r}")
    return int(text)


def prebuilt_bases(override: str | None = None) -> list[str]:
    """The folders to try for the ready-made engine, in order. `--prebuilt` (override) and STRATA_PREBUILT_URL replace
    the release folders; a local folder is used as a file:// URL. The release's tag folder comes first, then the
    latest release, whose BUILD.json the install checks."""
    chosen = override or os.environ.get("STRATA_PREBUILT_URL", "")
    if chosen:
        if chosen.startswith(("http://", "https://", "file://")):
            return [chosen.rstrip("/") + "/"]
        return [Path(chosen).expanduser().resolve().as_uri() + "/"]
    if not RELEASE_REPOSITORY:
        return []
    tag = PREBUILT_TAG_URL.format(repository=RELEASE_REPOSITORY, version=source_version(),
                                  metal_rev=metal_source_rev())
    return [tag, PREBUILT_URL.format(repository=RELEASE_REPOSITORY)]


def _engine_problem(meta: dict) -> str | None:
    """Why a BUILD.json (section 4.7.2) does not describe an engine this Mac can run, or None."""
    try:
        version = tuple(int(x) for x in str(meta.get("version", "")).split("."))
        rev = int(meta.get("metal_rev"))
        macos_min = tuple(int(x) for x in str(meta.get("macos_min", "")).split("."))
        families = list(meta.get("gpu_families", []))
    except (TypeError, ValueError):
        return "its BUILD.json is incomplete"
    if version < MIN_ENGINE:
        return f"its version {meta.get('version')} is older than {'.'.join(map(str, MIN_ENGINE))}"
    if rev < MIN_ENGINE_METAL:
        return f"its Metal rev {rev} is older than {MIN_ENGINE_METAL}"
    if meta.get("backend") != "metal":
        return f"its backend is {meta.get('backend')!r}, not metal"
    if meta.get("arch") != "arm64":
        return f"its arch is {meta.get('arch')!r}, not arm64"
    have = macos.macos_version()
    if not have:
        return "this Mac's macOS version cannot be read"
    if have < macos_min:
        return f"it needs macOS {meta.get('macos_min')} and this Mac has {'.'.join(map(str, have))}"
    devices = macos.metal_devices()
    if not devices:
        return "no Metal device was found"
    missing = [family for family in families if family not in devices[0]["families"]]
    if missing:
        return f"the GPU lacks {', '.join(missing)}"
    return None


def _installed_engine() -> Path | None:
    """The engine already in ENGINE_DIR, when this Mac can run it. An engine of another backend, copied from a PC, is
    removed. A local build is not reused here: build_engine_metal decides about it."""
    try:
        meta = json.loads((ENGINE_DIR / "BUILD.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(meta, dict):
        return None
    if meta.get("backend") != "metal":
        print(f"note: removed the engine in {ENGINE_DIR}: it is for backend {meta.get('backend')!r}, not metal",
              flush=True)
        shutil.rmtree(ENGINE_DIR, ignore_errors=True)
        return None
    if meta.get("source") == "local":
        return None
    if not (ENGINE_DIR / "strata").exists() or not (ENGINE_DIR / METALLIB).exists():
        return None
    if _engine_problem(meta) is not None:
        return None
    print(f"ok: engine {meta.get('version')} (Metal rev {meta.get('metal_rev')}) is already installed", flush=True)
    return ENGINE_DIR


def _network_reason(exc: OSError) -> str:
    text = str(getattr(exc, "reason", None) or exc)
    return HINT_CERTIFICATES if "CERTIFICATE_VERIFY_FAILED" in text else text


def _extract_engine(zip_path: Path, dest: Path) -> None:
    """Unpack the asset. A symbolic link anywhere in it fails the install before anything is written (section 4.7.1)."""
    with zipfile.ZipFile(zip_path) as archive:
        for info in archive.infolist():
            if stat.S_ISLNK(info.external_attr >> 16):
                raise SetupError(f"the engine asset has a symbolic link ({info.filename}); a release never has one, "
                                 "so the download is not trusted")
        archive.extractall(dest)


def get_prebuilt_metal(url_base: str | None, updating: bool = False) -> Path | None:
    """Install the ready-made Metal engine from `url_base` into ENGINE_DIR, or return None when it cannot be used here
    and setup should compile one (section 4.7.1). An installed engine is reused unless `updating` is True. Every check
    runs on the unpacked files before any of them replaces an installed one."""
    if not url_base:
        return None
    if not updating:
        installed = _installed_engine()
        if installed is not None:
            return installed
    url = url_base.rstrip("/") + "/" + PREBUILT_ASSET
    with tempfile.TemporaryDirectory(prefix="strata-engine-") as work:
        zip_path = Path(work) / PREBUILT_ASSET
        try:
            with urllib.request.urlopen(urllib.request.Request(url, method="HEAD"), timeout=60):
                pass
            with urllib.request.urlopen(url, timeout=60) as response, zip_path.open("wb") as out:
                shutil.copyfileobj(response, out)
        except OSError as exc:
            reason = _network_reason(exc)
            if reason == HINT_CERTIFICATES:
                raise SetupError(reason) from exc
            print(f"no ready-made engine at {url} ({reason}): compiling instead", flush=True)
            return None

        ENGINE_DIR.mkdir(parents=True, exist_ok=True)
        unpack = ENGINE_DIR / "_unpack"
        shutil.rmtree(unpack, ignore_errors=True)
        try:
            try:
                _extract_engine(zip_path, unpack)
            except zipfile.BadZipFile:
                print("warn: the ready-made engine is not a zip file: compiling instead", flush=True)
                return None
            try:
                meta = json.loads((unpack / "BUILD.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = None
            problem = "it has no readable BUILD.json" if not isinstance(meta, dict) else _engine_problem(meta)
            if problem is None and not all((unpack / name).exists() for name in ("strata", METALLIB)):
                problem = "it lacks strata or strata.metallib"
            if problem is not None:
                print(f"warn: the ready-made engine cannot run here ({problem}): compiling instead", flush=True)
                return None
            for path in unpack.iterdir():
                if path.is_file():
                    path.chmod(0o755 if path.name in ENGINE_EXECUTABLES else 0o644)
            for path in list(unpack.iterdir()):
                os.replace(path, ENGINE_DIR / path.name)  # a new inode: never copy over a signed Mach-O
        finally:
            shutil.rmtree(unpack, ignore_errors=True)
        macos.strip_quarantine(ENGINE_DIR)
        print(f"ok: ready-made engine {meta['version']} (Metal rev {meta['metal_rev']}, macOS 27+)", flush=True)
        return ENGINE_DIR


def _smoke_run(exe: Path, flag: str):
    try:
        return SMOKE_RUNNER([str(exe), flag], capture_output=True, text=True, timeout=SMOKE_TIMEOUT_S)
    except subprocess.TimeoutExpired as exc:
        raise SetupError(f"the engine did not answer {flag} within {SMOKE_TIMEOUT_S} s") from exc
    except OSError as exc:
        raise SetupError(f"cannot start {exe}: {exc}") from exc


def smoke_test(engine: Path | None = None) -> dict:
    """Run `strata --version`, then `strata --list-devices`, and check the device (section 4.7.5). Returns the version,
    the Metal rev and the device's families; any other result raises SetupError with the cause."""
    exe = (engine or ENGINE_DIR) / "strata"
    result = _smoke_run(exe, "--version")
    if result.returncode == -9:
        hint = ("xattr -dr com.apple.quarantine engine, then run setup again" if macos.has_quarantine(exe)
                else f"codesign -dv {exe}; delete the engine folder and run setup again")
        raise SetupError("macOS stopped the engine before it started. " + hint)
    if "Library not loaded" in result.stderr:
        raise SetupError("the engine links a library that is not on this Mac (a build bug: the asset must be "
                         "self-contained)")
    found = re.fullmatch(r"strata (\S+) metal (\d+)", result.stdout.strip()) if result.returncode == 0 else None
    if found is None:
        tail = "\n".join(result.stderr.splitlines()[-5:]) or "(no stderr)"
        raise SetupError(f"the engine's --version did not give 'strata <v> metal <rev>' (exit {result.returncode}):"
                         f"\n{tail}")
    listing = _smoke_run(exe, "--list-devices")
    lines = listing.stdout.splitlines()
    devices = [line for line in lines if line.startswith("device_name=")]
    families = next((line.split("=", 1)[1].split(",") for line in lines if line.startswith("families=")), [])
    if listing.returncode != 0 or len(devices) != 1 or not {"apple7", "metal4"} <= set(families):
        output = "\n".join(lines[-8:]) or "(no output)"
        raise SetupError("--list-devices must print one device whose families include apple7 and metal4; it printed:"
                         f"\n{output}")
    print(f"ok: engine {found.group(1)} (Metal rev {found.group(2)}) runs; the GPU has {', '.join(families)}",
          flush=True)
    return {"version": found.group(1), "metal_rev": int(found.group(2)), "families": families}


def use_ssl_bundle() -> str | None:
    """Give downloads a CA bundle when this Python has none (section 4.7.4), and say so."""
    path = macos.ssl_setup()
    if path:
        print(f"ok: HTTPS certificates: {path} (this Python has none of its own)", flush=True)
    return path


# ---- Mac build definitions, source identity and BUILD.json (docs/mac/08-build-and-distribution.md, section 8) ----
# No build step calls cmake_build or metal_build_json yet (S8 in docs/mac/07 is open). source_version and
# metal_source_rev are the S-track helpers above; the install helpers of S7 are also above.
# The inputs of the source identity are the list of docs/mac/08 section 8.3, with METAL_REV and cmake/metal from 07 4.7.2.
ENGINE_SOURCES = ("CMakeLists.txt", "CMakePresets.json", "cmake/metal", "src", "include", "METAL_REV",
                  "third_party/ggml/VERSION.txt")
VISION_SOURCES = ("tools/vision",)
FINDER_FILE = ".DS_Store"
LLAMA_REPO = "https://github.com/ggml-org/llama.cpp.git"


def metal_engine_defs(llama) -> list:  # MAC:
    """The CMake definitions of the Mac engine. CMakePresets.json mac-release must hold the same values."""
    return ["-DSTRATA_BUILD_TESTS=OFF", "-DCMAKE_OSX_ARCHITECTURES=arm64",
            "-DCMAKE_OSX_DEPLOYMENT_TARGET=27.0", f"-DSTRATA_GGML_DIR={llama}"]


def metal_vision_defs(llama, portable) -> list:  # MAC:
    """strata-vision on a Mac: Metal with ggml's shaders embedded, no OpenMP, and the pinned llama.cpp in `llama`.
    `portable` adds STRATA_PORTABLE, which turns off the build's native CPU tuning (docs/mac/08 section 9)."""
    return [f"-DLLAMA_DIR={llama}", "-DGGML_METAL=ON", "-DGGML_METAL_EMBED_LIBRARY=ON", "-DGGML_OPENMP=OFF",
            "-DCMAKE_OSX_ARCHITECTURES=arm64", "-DCMAKE_OSX_DEPLOYMENT_TARGET=27.0",
            *(["-DSTRATA_PORTABLE=ON"] if portable else [])]


def source_hash(paths) -> str:  # MAC:
    """The 16-hex identity of the files under `paths` (relative to ROOT). It is SHA-256 over each file's path, size and
    bytes, in path order. Finder's .DS_Store files are not part of it (docs/mac/08 section 8.3)."""
    files: list[Path] = []
    for rel in paths:
        target = ROOT / rel
        if target.is_dir():
            files.extend(path for path in target.rglob("*") if path.is_file())
        elif target.is_file():
            files.append(target)
        else:
            raise FileNotFoundError(f"source input {rel} is missing")
    digest = hashlib.sha256()
    for path in sorted(files, key=lambda item: item.relative_to(ROOT).as_posix()):
        if path.name == FINDER_FILE:
            continue
        name = path.relative_to(ROOT).as_posix()
        data = path.read_bytes()
        digest.update(f"{name}\0{len(data)}\0".encode("utf-8"))
        digest.update(data)
    return digest.hexdigest()[:16]


def find_tool(name: str) -> str | None:  # MAC:
    """The tool `name`: the bin folder of the running Python (the venv's pinned tools) before PATH (08 section 8.5)."""
    venv_tool = Path(sys.executable).parent / name
    if venv_tool.is_file() and os.access(venv_tool, os.X_OK):
        return str(venv_tool)
    return shutil.which(name)


def _command_output(*cmd: str) -> str:
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def xcode_and_toolchain() -> tuple[str, str]:  # MAC:
    """This Mac's Xcode, as `27.0 (27A266a)`, and its Metal toolchain identifier (docs/mac/08 section 8.4). Either is
    "" when the tool does not answer. verify_release.py compares them with a release's BUILD.json."""
    xcode_lines = _command_output("xcodebuild", "-version").splitlines()
    xcode = f"{xcode_lines[0].split()[-1]} ({xcode_lines[1].split()[-1]})" if len(xcode_lines) == 2 else ""
    try:
        toolchain = json.loads(_command_output("xcodebuild", "-showComponent", "MetalToolchain", "-json") or "{}")
    except json.JSONDecodeError:
        toolchain = {}
    if not isinstance(toolchain, dict):
        toolchain = {}
    return xcode, toolchain.get("toolchainIdentifier", "")


def metal_build_json(eng: Path, source: str) -> dict:  # MAC:
    """BUILD.json of the Mac engine in `eng` (docs/mac/07 section 4.7.2, docs/mac/08 section 8.4). A local build and a
    release both write it, so the two describe themselves the same way. The caller writes
    `json.dumps(meta, indent=1) + "\\n"`. The version and Metal rev come from source_version and metal_source_rev."""
    xcode, metal_toolchain = xcode_and_toolchain()
    commit = _command_output("git", "-C", str(ROOT), "rev-parse", "HEAD")
    if commit and _command_output("git", "-C", str(ROOT), "status", "--porcelain"):
        commit += "-dirty"
    vision = "gpu" if (eng / "strata-vision").exists() else "none"
    return {
        "version": source_version(),
        "metal_rev": metal_source_rev(),
        "backend": "metal",
        "arch": "arm64",
        "macos_min": ".".join(str(part) for part in MACOS_MIN),
        "gpu_families": list(macos.REQUIRED_FAMILIES),
        "metal_std": "metal4.0",
        "source": source,
        "lib_dirs": [],
        "vision": vision,
        "src": source_hash(ENGINE_SOURCES),
        "vision_src": source_hash(VISION_SOURCES) if vision == "gpu" else None,
        "xcode": xcode,
        "metal_toolchain": metal_toolchain,
        "commit": commit,
        "metallib_sha256": hashlib.sha256((eng / METALLIB).read_bytes()).hexdigest(),
    }


def llama_pin(root: Path | None = None) -> str:  # MAC:
    """The pinned llama.cpp commit: the first word of third_party/ggml/VERSION.txt, which must be 40 hex digits."""
    text = ((ROOT if root is None else root) / "third_party" / "ggml" / "VERSION.txt").read_text(encoding="utf-8")
    words = text.split()
    if not words or not re.fullmatch(r"[0-9a-f]{40}", words[0]):
        raise RuntimeError("third_party/ggml/VERSION.txt must start with a full llama.cpp commit")
    return words[0]


def get_llama_cpp(root: Path | None = None, runner=subprocess.run) -> Path:  # MAC:
    """The pinned llama.cpp in third_party/llama.cpp (docs/mac/08 section 11.3, step 1). A missing checkout is cloned
    and set to the pin; an existing one must report the pin as its HEAD. A copy without .git cannot be verified."""
    base = ROOT if root is None else root
    pin = llama_pin(base)
    target = base / "third_party" / "llama.cpp"
    if not target.exists():
        for command in (["git", "clone", "--no-checkout", LLAMA_REPO, str(target)],
                        ["git", "-C", str(target), "checkout", "--detach", pin]):
            result = runner(command, capture_output=True, text=True, check=False)
            if result.returncode:
                raise RuntimeError(f"cannot fetch the pinned llama.cpp: {result.stderr.strip()}")
    head = runner(["git", "-C", str(target), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if head.returncode or head.stdout.strip() != pin:
        raise RuntimeError(f"third_party/llama.cpp is not verified at the pin {pin}")
    return target


def cmake_build(src: Path, build: Path, target: str, defs: list[str]) -> None:  # MAC:
    """Configure `src` into the folder `build` with Ninja, then build `target`. The tools are the pinned ones next to
    this Python (find_tool). Output goes to the terminal; a failure raises CalledProcessError."""
    cmake = find_tool("cmake")
    ninja = find_tool("ninja")
    if cmake is None or ninja is None:
        raise RuntimeError("cmake and ninja are needed: create .venv from requirements.txt and run this with its Python")
    subprocess.run([cmake, "-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}", "-S", str(src), "-B", str(build),
                    "-DCMAKE_BUILD_TYPE=Release", *defs], check=True)
    subprocess.run([cmake, "--build", str(build), "--target", target], check=True)


def main() -> int:
    print("macOS setup is not implemented yet; no files were changed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
