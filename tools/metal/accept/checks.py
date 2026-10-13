"""Pre-run checks, made before any model loads (docs/mac/09-testing-and-validation.md, section 3.2).

Each check returns a Check. A check that cannot run (a file or module is absent) is unmet, never passed. Only
`--version`, `--list-devices`, the hash checks and `test_models.py check --quick` run here; none of them load a model.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from .engine import EngineOutputError, Version, parse_devices, parse_version

ENGINE_TIMEOUT_S = 60
CHECK_TIMEOUT_S = 600


@dataclass(frozen=True)
class Check:
    name: str
    ok: bool
    detail: str
    value: object = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_ids(path: Path) -> list[str]:
    """The ids of a `<set>.ids` file: commas and white space both separate them."""
    text = path.read_text(encoding="utf-8")
    return [token for token in text.replace(",", " ").split() if token]


def ids_sha256(ids: list[str]) -> str:
    """The manifest's hash of an id list: SHA-256 of `",".join(ids)` as ASCII, with no line end."""
    return hashlib.sha256(",".join(ids).encode("ascii")).hexdigest()


def prompt_manifest(root: Path, sets: tuple[str, ...]) -> Check:
    name = "prompt manifest"
    prompts = root / "tests/metal/prompts"
    manifest_path = prompts / "MANIFEST.json"
    if not manifest_path.is_file():
        return Check(name, False, "tests/metal/prompts/MANIFEST.json is absent (T1)")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check(name, False, f"cannot read the manifest: {exc}")
    if not isinstance(manifest, dict) or manifest.get("schema") != 1:
        return Check(name, False, "the manifest is not schema 1")
    entries = {entry.get("name"): entry for entry in manifest.get("sets", []) if isinstance(entry, dict)}
    for set_name in sets:
        entry = entries.get(set_name)
        if entry is None:
            return Check(name, False, f"{set_name} has no entry in the manifest")
        ids_path = prompts / f"{set_name}.ids"
        if not ids_path.is_file():
            return Check(name, False, f"{set_name}.ids is absent")
        found = ids_sha256(read_ids(ids_path))
        if found != entry.get("ids_sha256"):
            return Check(name, False, f"{set_name}: ids_sha256 {found} does not match the manifest")
    return Check(name, True, f"{len(sets)} sets match", value=manifest.get("builder_version"))


def golden_manifest(root: Path, golden: Path, reads: tuple[str, ...]) -> Check:
    name = "golden manifest"
    manifest_path = root / "tests/metal/golden/MANIFEST.json"
    if not manifest_path.is_file():
        return Check(name, False, "tests/metal/golden/MANIFEST.json is absent (T2)")
    sync_path = root / "UPSTREAM_SYNC"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        sync = sync_path.read_text(encoding="ascii").strip()
    except (OSError, ValueError) as exc:
        return Check(name, False, f"cannot read the manifest or UPSTREAM_SYNC: {exc}")
    if manifest.get("upstream_sync") != sync:
        return Check(name, False, "the manifest's upstream_sync is not UPSTREAM_SYNC")
    if not golden.is_dir():
        return Check(name, False, f"the golden folder is absent: {golden}")
    files = {entry.get("path"): entry for entry in manifest.get("files", []) if isinstance(entry, dict)}
    for relative in reads:
        entry = files.get(relative)
        if entry is None:
            return Check(name, False, f"golden: {relative}: not in the manifest")
        path = golden / relative
        if not path.is_file():
            return Check(name, False, f"golden: {relative}: absent from {golden}")
        found = sha256_file(path)
        if found != entry.get("sha256"):
            return Check(name, False, f"golden: {relative}: sha256 {found} does not match the manifest")
    return Check(name, True, f"{len(reads)} golden files match")


def model_config_args(models: Path, tag: str) -> Check:
    """The model's config.json must exist and hold an `args` list (the engine's command line)."""
    name = f"config {tag}"
    path = models / tag / "config.json"
    if not path.is_file():
        return Check(name, False, f"{path} is absent")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return Check(name, False, f"cannot read {path}: {exc}")
    args = config.get("args")
    if not isinstance(args, list) or not all(isinstance(arg, str) for arg in args):
        return Check(name, False, f"{path}: args must be a list of strings")
    return Check(name, True, str(path), value=tuple(args))


def test_models_quick(root: Path, models: Path, tag: str) -> Check:
    name = f"test_models check --quick {tag}"
    script = root / "tools/metal/test_models.py"
    if not script.is_file():
        return Check(name, False, "tools/metal/test_models.py is absent (T3): check --quick cannot run")
    try:
        result = subprocess.run(
            [sys.executable, str(script), "check", "--quick", "--models", str(models), tag],
            capture_output=True, text=True, timeout=CHECK_TIMEOUT_S, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(name, False, f"did not finish: {exc}")
    if result.returncode != 0:
        last = (result.stderr or result.stdout).strip().splitlines()[-1:] or ["no output"]
        return Check(name, False, f"exit {result.returncode}: {last[0]}")
    lines = result.stdout.strip().splitlines()
    return Check(name, True, "ok", value=lines[-1] if lines else "")


def engine_command(engine: Path) -> list[str]:
    """A Python stand-in (tests) is run by this interpreter; any other path is executed directly."""
    if engine.suffix == ".py":
        return [sys.executable, str(engine)]
    return [str(engine)]


def engine_start(engine: Path) -> tuple[list[Check], Version | None, tuple[str, ...]]:
    """`--version` and `--list-devices`, in that order (section 3.2)."""
    if not engine.is_file():
        return [Check("engine", False, f"the engine is absent: {engine}")], None, ()
    command = engine_command(engine)
    checks: list[Check] = []
    version: Version | None = None
    devices: tuple[str, ...] = ()
    for flag in ("--version", "--list-devices"):
        try:
            result = subprocess.run(command + [flag], capture_output=True, text=True,
                                    timeout=ENGINE_TIMEOUT_S, check=False)
        except (OSError, subprocess.TimeoutExpired) as exc:
            checks.append(Check(f"engine {flag}", False, f"did not run: {exc}"))
            return checks, version, devices
        if result.returncode != 0:
            checks.append(Check(f"engine {flag}", False,
                                f"exit {result.returncode}: {(result.stderr or '').strip()[-200:]}"))
            return checks, version, devices
        try:
            if flag == "--version":
                version = parse_version(result.stdout, source=f"engine {flag}")
                checks.append(Check(f"engine {flag}", True, f"strata {version.engine_version} metal "
                                                            f"{version.metal_rev}"))
            else:
                devices = parse_devices(result.stdout, source=f"engine {flag}")
                checks.append(Check(f"engine {flag}", True, f"{len(devices)} device line(s)"))
        except EngineOutputError as exc:
            checks.append(Check(f"engine {flag}", False, str(exc)))
            return checks, version, devices
    return checks, version, devices


def no_engine_running() -> Check:
    """One engine per Mac (docs/mac/07, section 6.8): `pgrep -x strata` must find nothing."""
    name = "no other strata process"
    try:
        result = subprocess.run(["pgrep", "-x", "strata"], capture_output=True, text=True, timeout=30,
                                check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Check(name, False, f"pgrep did not run: {exc}")
    if result.returncode == 0:
        return Check(name, False, f"another strata process is running (pids {result.stdout.split()})")
    if result.returncode == 1:
        return Check(name, True, "none")
    return Check(name, False, f"pgrep exited {result.returncode}")


def resolve_vision(root: Path, engine: Path, explicit: Path | None) -> Check:
    """The image encoder for S.image: `--vision`, else next to the engine, else the build tree (section 3.2)."""
    name = "vision encoder"
    if explicit is not None:
        candidates = [explicit]
    else:
        candidates = [engine.parent / "strata-vision", root / "build-vision/bin/strata-vision"]
    for candidate in candidates:
        if candidate.is_file():
            return Check(name, True, str(candidate), value=candidate)
    return Check(name, False, "no strata-vision binary: pass --vision, or build build-vision")
