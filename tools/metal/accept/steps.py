"""Runs one step as a fresh process, with a time limit and crash-report capture (docs/mac/09, section 3.2).

Commands are argument lists. Nothing is passed through a shell, so a value from a file or the command line can only
ever be one argument. The child runs in its own session, so a timeout stops the engine's whole process group.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

GRACE_S = 30.0
MIN_LIMIT_S = 600.0
DIAGNOSTIC_DIR = Path.home() / "Library/Logs/DiagnosticReports"


@dataclass(frozen=True)
class StepResult:
    number: int
    name: str
    model: str
    prompt: str
    seconds: float
    exit: int | None
    timed_out: bool
    limit_s: float
    stdout: Path
    stderr: Path
    command: Path
    crash_reports: tuple[str, ...] = field(default_factory=tuple)


def step_limit(expected_s: float | None) -> float:
    """Three times the expected time, and at least ten minutes (measurement T-6 supplies the expected time)."""
    if expected_s is None:
        return MIN_LIMIT_S
    return max(MIN_LIMIT_S, 3.0 * expected_s)


def crash_snapshot() -> set[str]:
    if not DIAGNOSTIC_DIR.is_dir():
        return set()
    return {entry.name for entry in DIAGNOSTIC_DIR.iterdir() if "strata" in entry.name.lower()}


def copy_new_crashes(before: set[str], destination: Path) -> tuple[str, ...]:
    after = crash_snapshot()
    new = sorted(after - before)
    if not new:
        return ()
    destination.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in new:
        source = DIAGNOSTIC_DIR / name
        if source.is_file():
            shutil.copy2(source, destination / name)
            copied.append(name)
    return tuple(copied)


def write_command(path: Path, argv: list[str], env_extra: dict[str, str]) -> None:
    """The argument list and every STRATA_* variable, one per line (section 3.2)."""
    lines = list(argv)
    lines += [f"{key}={value}" for key, value in sorted(env_extra.items()) if key.startswith("STRATA_")]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _signal_group(process: subprocess.Popen, sig: int) -> None:
    try:
        os.killpg(process.pid, sig)
    except ProcessLookupError:
        pass


def run_step(
    argv: list[str],
    *,
    number: int,
    name: str,
    model: str,
    prompt: str,
    steps_dir: Path,
    env: dict[str, str],
    env_record: dict[str, str],
    cwd: Path,
    limit_s: float,
    grace_s: float = GRACE_S,
    crash_dir: Path | None = None,
) -> StepResult:
    stem = f"{number:03d}"
    stdout_path = steps_dir / f"{stem}.out"
    stderr_path = steps_dir / f"{stem}.err"
    command_path = steps_dir / f"{stem}.cmd"
    write_command(command_path, argv, env_record)

    before = crash_snapshot()
    started = time.monotonic()
    timed_out = False
    with stdout_path.open("wb") as out, stderr_path.open("wb") as err:
        process = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=out, stderr=err, env=env, cwd=str(cwd),
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=limit_s)
        except subprocess.TimeoutExpired:
            timed_out = True
            _signal_group(process, signal.SIGTERM)
            try:
                process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                _signal_group(process, signal.SIGKILL)
                process.wait()
            code = process.returncode
    seconds = time.monotonic() - started
    crashes: tuple[str, ...] = ()
    if crash_dir is not None:
        crashes = copy_new_crashes(before, crash_dir)
    return StepResult(
        number=number, name=name, model=model, prompt=prompt, seconds=seconds,
        exit=code, timed_out=timed_out, limit_s=limit_s,
        stdout=stdout_path, stderr=stderr_path, command=command_path, crash_reports=crashes,
    )
