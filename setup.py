#!/usr/bin/env python3
"""macOS setup entry point. Installation is disabled until the Apple Silicon flow is complete."""
from __future__ import annotations

import functools
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.model_catalog import CONTEXTS, FAMILIES, MODELS  # noqa: E402

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


class PlanError(RuntimeError):
    """The engine gave no usable plan. Setup stops here; it never substitutes a plan."""


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


def main() -> int:
    print("macOS setup is not implemented yet; no files were changed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
