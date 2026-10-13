"""The result folder and result.json, schema 1 (docs/mac/09-testing-and-validation.md, section 3.2).

Host fields are best effort: a command that fails or is absent gives null, never a guessed value.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path

HOST_TIMEOUT_S = 20
SCHEMA = 1


def run_text(argv: list[str]) -> str | None:
    try:
        result = subprocess.run(argv, capture_output=True, text=True, timeout=HOST_TIMEOUT_S, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


NO_THERMAL_WARNING = re.compile(r"No thermal warning level has been recorded")
NO_PERFORMANCE_WARNING = re.compile(r"No performance warning level has been recorded")
WARNING_LEVEL = re.compile(r"(?:thermal|performance) warning level", re.IGNORECASE)


def power_source(batteries: str | None) -> str | None:
    """The source named by `pmset -g batt`, e.g. `AC Power`; None when the text does not name one."""
    source = re.search(r"Now drawing from '([^']+)'", batteries or "")
    return source.group(1) if source else None


def low_power_mode(power: str | None) -> bool | None:
    """From `pmset -g`: True unless the key is 0. macOS names it `lowpowermode` or `powermode`."""
    value = re.search(r"(?:lowpowermode|powermode)\s+(\d+)", power or "")
    return None if value is None else value.group(1) != "0"


def thermal_warning(therm: str | None) -> bool | None:
    """From `pmset -g therm`: True when a thermal or performance warning level is recorded, False when both
    `No ... warning level has been recorded` lines are present and nothing else names a warning, None otherwise.
    """
    if not therm:
        return None
    lines = therm.splitlines()
    for line in lines:
        if WARNING_LEVEL.search(line) and not (NO_THERMAL_WARNING.search(line) or NO_PERFORMANCE_WARNING.search(line)):
            return True
    if any(NO_THERMAL_WARNING.search(line) for line in lines) and any(
        NO_PERFORMANCE_WARNING.search(line) for line in lines
    ):
        return False
    return None


def memory_status_level(text: str | None) -> int | None:
    """From `sysctl -n kern.memorystatus_level`: the percentage of memory free for use, or None."""
    value = (text or "").strip()
    return int(value) if value.isdigit() else None


def host_info() -> dict[str, object]:
    product = run_text(["sw_vers", "-productVersion"])
    build = run_text(["sw_vers", "-buildVersion"])
    memory = run_text(["sysctl", "-n", "hw.memsize"])
    displays = run_text(["system_profiler", "SPDisplaysDataType"])
    cores = re.search(r"Total Number of Cores: (\d+)", displays or "")
    return {
        "macos": None if product is None and build is None else {"product": product, "build": build},
        "chip": run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_gib": None if not memory or not memory.isdigit() else int(memory) // (1 << 30),
        "gpu_cores": int(cores.group(1)) if cores else None,
        "power": power_source(run_text(["pmset", "-g", "batt"])),
        "low_power_mode": low_power_mode(run_text(["pmset", "-g"])),
        "thermal_warning": thermal_warning(run_text(["pmset", "-g", "therm"])),
        "memory_status_level": memory_status_level(run_text(["sysctl", "-n", "kern.memorystatus_level"])),
    }


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def metallib_sha256(engine: Path) -> str | None:
    """From BUILD.json next to the engine (docs/mac/08, section 8.4), when it is there."""
    build = engine.parent / "BUILD.json"
    if not build.is_file():
        return None
    try:
        value = json.loads(build.read_text(encoding="utf-8")).get("metallib_sha256")
    except (OSError, ValueError, AttributeError):
        return None
    return value if isinstance(value, str) else None


def project_commit(root: Path) -> str | None:
    return run_text(["git", "-C", str(root), "rev-parse", "HEAD"])


def folder_name(started: datetime, engine_version: str | None, metal_rev: int | None, commit: str | None) -> str:
    stamp = started.strftime("%Y%m%dT%H%M%SZ")
    version = engine_version or "unknown"
    rev = metal_rev if metal_rev is not None else "unknown"
    return f"{stamp}-{version}-metal.{rev}-{(commit or 'unknown')[:12]}"


def write_result(folder: Path, data: dict[str, object]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "result.json"
    path.write_text(json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def build_result(
    *,
    run: str,
    status: str,
    started: datetime,
    seconds: float,
    engine: dict[str, object],
    source: dict[str, object],
    host: dict[str, object],
    models: list[dict[str, object]],
    golden: dict[str, object],
    criteria: list[dict[str, object]],
    steps: list[dict[str, object]],
    restricted: bool,
) -> dict[str, object]:
    """Schema 1. `restricted` is an addition the schema readers ignore: a run limited by --model or --prompt."""
    return {
        "schema": SCHEMA,
        "run": run,
        "status": status,
        "started": started.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "seconds": round(seconds, 3),
        "engine": engine,
        "source": source,
        "host": host,
        "models": models,
        "golden": golden,
        "criteria": criteria,
        "steps": steps,
        "restricted": restricted,
    }
