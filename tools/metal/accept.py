#!/usr/bin/env python3
"""Run one acceptance run of the Mac port (docs/mac/09-testing-and-validation.md, section 3.2).

usage: accept.py <run> [--engine PATH] [--vision PATH] [--models DIR] [--golden DIR] [--out DIR]
                 [--model TAG ...] [--prompt SET ...] [--short] [--keep] [--day N] [--dry-run]

Exit codes: 0 every criterion passed (or a dry run whose commands and inputs are all ready), 1 a criterion failed,
2 the run could not start or could not be judged (NOT RUN). The last line of output is the status:
`accept <run>: PASS (...)`, `accept <run>: FAIL <criterion>: <value>, want <op> <threshold>`,
`accept <run>: NOT RUN: <reason>`, or `accept <run>: DRY RUN: ...` for a dry run that is ready.
"""
from __future__ import annotations

import argparse
import os
import re
import shlex
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from accept import bench, checks, criteria, engine, result, runs
from accept import steps as stepper

ROOT = Path(__file__).resolve().parents[2]
EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_NOT_RUN = 2
PY_DISPLAY = "python3"
PROMPT_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*(?:\.[A-Za-z0-9_-]+)*")


@dataclass(frozen=True)
class Settings:
    root: Path
    run: str
    engine: Path
    vision: Path | None
    models: Path | None
    golden: Path | None
    out: Path | None
    prompts: tuple[str, ...] | None
    models_restricted: tuple[str, ...] | None
    short: bool
    keep: bool
    day: int | None
    dry_run: bool
    sync: str | None

    @property
    def restricted(self) -> bool:
        return bool(self.prompts or self.models_restricted)


@dataclass
class Facts:
    """What the pre-run checks learned, for the result folder."""

    engine_version: engine.Version | None = None
    devices: tuple[str, ...] = ()
    quick: dict[str, object] = field(default_factory=dict)
    prompts_builder: object = None


def build_parser(names: list[str]) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run one acceptance run (docs/mac/09, section 3.2).")
    parser.add_argument("run", choices=names)
    parser.add_argument("--engine", type=Path, help="the engine under test (default build-dev/strata)")
    parser.add_argument("--vision", type=Path, help="the image encoder for S.image")
    parser.add_argument("--models", type=Path, help="the test models (default $STRATA_TEST_MODELS)")
    parser.add_argument("--golden", type=Path, help="the golden set (default <models>/golden/<UPSTREAM_SYNC[:12]>)")
    parser.add_argument("--out", type=Path, help="the result folder's parent (default $STRATA_RESULTS/<run>)")
    parser.add_argument("--model", action="append", default=[], metavar="TAG", help="restrict to a model")
    parser.add_argument("--prompt", action="append", default=[], metavar="SET", help="restrict to a prompt set")
    parser.add_argument("--short", action="store_true", help="the short form, where a run has one (C1)")
    parser.add_argument("--keep", action="store_true", help="keep the dumps after a pass")
    parser.add_argument("--day", type=int, help="for weekly: the slice, 1-7 (default: today, UTC)")
    parser.add_argument("--dry-run", action="store_true", help="print the commands and check the inputs")
    return parser


def validate(parser: argparse.ArgumentParser, args: argparse.Namespace, spec: runs.RunSpec) -> None:
    if args.short and not spec.accepts_short:
        parser.error(f"--short applies to c1 only, not {spec.name}")
    if args.prompt and not spec.accepts_prompts:
        parser.error(f"--prompt does not apply to {spec.name}")
    if args.day is not None:
        if not spec.accepts_day:
            parser.error(f"--day applies to weekly only, not {spec.name}")
        if not 1 <= args.day <= 7:
            parser.error("--day must be 1-7")
    for tag in args.model:
        if tag not in spec.models:
            parser.error(f"model {tag} is not in the list of {spec.name}: {', '.join(spec.models)}")
    for prompt in args.prompt:
        if not PROMPT_NAME.fullmatch(prompt):
            parser.error(f"prompt set name {prompt!r} must be letters, digits, '.', '_' or '-'")


def read_sync(root: Path) -> str | None:
    try:
        value = (root / "UPSTREAM_SYNC").read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        return None
    return value or None


def settings_from(args: argparse.Namespace, root: Path, env: dict[str, str]) -> Settings:
    sync = read_sync(root)
    engine_path = args.engine if args.engine is not None else root / "build-dev/strata"
    models_text = args.models or (Path(env["STRATA_TEST_MODELS"]) if env.get("STRATA_TEST_MODELS") else None)
    models = Path(os.path.abspath(models_text)) if models_text else None
    golden = args.golden
    if golden is None and models is not None and sync is not None:
        golden = models / "golden" / sync[:12]
    out = args.out
    if out is None and env.get("STRATA_RESULTS"):
        out = Path(env["STRATA_RESULTS"]) / args.run
    day = args.day
    if day is None:
        day = datetime.now(timezone.utc).isoweekday()
    return Settings(
        root=root,
        run=args.run,
        engine=Path(os.path.abspath(engine_path)),
        vision=None if args.vision is None else Path(os.path.abspath(args.vision)),
        models=models,
        golden=None if golden is None else Path(os.path.abspath(golden)),
        out=None if out is None else Path(os.path.abspath(out)),
        prompts=tuple(args.prompt) or None,
        models_restricted=tuple(args.model) or None,
        short=args.short,
        keep=args.keep,
        day=day if args.run == "weekly" else None,
        dry_run=args.dry_run,
        sync=sync,
    )


def model_args(settings: Settings, tags: tuple[str, ...]) -> tuple[dict[str, tuple[str, ...]], list[checks.Check]]:
    """Each model's config.json args. Without a readable config the plan keeps a visible placeholder."""
    args: dict[str, tuple[str, ...]] = {}
    found: list[checks.Check] = []
    for tag in tags:
        placeholder = (f"<config.json args of {tag}>",)
        if settings.models is None:
            args[tag] = placeholder
            continue
        check = checks.model_config_args(settings.models, tag)
        found.append(check)
        args[tag] = check.value if check.ok else placeholder
    return args, found


def gather_checks(settings: Settings, spec: runs.RunSpec, plan: list[runs.Step], tags: tuple[str, ...],
                  facts: Facts) -> list[checks.Check]:
    """The pre-run checks in the order of section 3.2. Each check is recorded, so a dry run can show all of them."""
    found: list[checks.Check] = []
    if settings.models is None:
        found.append(checks.Check("models", False, "STRATA_TEST_MODELS is not set and --models was not given"))
    else:
        found += model_args(settings, tags)[1]

    prompt_sets = tuple(dict.fromkeys(step.prompt for step in plan if step.prompt))
    if prompt_sets:
        check = checks.prompt_manifest(settings.root, prompt_sets)
        facts.prompts_builder = check.value
        found.append(check)

    reads = tuple(dict.fromkeys(name for step in plan for name in step.reads))
    if reads:
        if settings.golden is None:
            found.append(checks.Check("golden manifest", False,
                                      "the golden folder is unknown: pass --golden, or set STRATA_TEST_MODELS "
                                      "and keep UPSTREAM_SYNC"))
        else:
            found.append(checks.golden_manifest(settings.root, settings.golden, reads))

    if settings.models is not None:
        for tag in tags:
            check = checks.test_models_quick(settings.root, settings.models, tag)
            if check.ok:
                facts.quick[tag] = check.value
            found.append(check)

    engine_checks, version, devices = checks.engine_start(settings.engine)
    facts.engine_version = version
    facts.devices = devices
    found += engine_checks
    found.append(checks.no_engine_running())
    if spec.benchmark:
        found += bench.host_checks()

    if spec.needs_vision:
        found.append(checks.resolve_vision(settings.root, settings.engine, settings.vision))
    if settings.out is None and not settings.dry_run:
        found.append(checks.Check("out", False, "STRATA_RESULTS is not set and --out was not given"))
    return found


def substitutions(number: int, folder: str, python: str, engine_text: str, config_text: str) -> tuple[tuple[str, str], ...]:
    return (
        ("{python}", python),
        ("{engine}", engine_text),
        ("{config}", config_text),
        ("{result}", folder),
        ("{step}", f"steps/{number:03d}"),
    )


def substitute(text: str, mapping: tuple[tuple[str, str], ...]) -> str:
    for key, value in mapping:
        text = text.replace(key, value)
    return text


def resolve_argv(step: runs.Step, number: int, folder: str, python: str, engine_text: str,
                 config_text: str) -> list[str]:
    mapping = substitutions(number, folder, python, engine_text, config_text)
    return [substitute(token, mapping) for token in step.argv]


def resolve_env(step: runs.Step, number: int, folder: str, python: str, engine_text: str,
                config_text: str) -> dict[str, str]:
    mapping = substitutions(number, folder, python, engine_text, config_text)
    return {key: substitute(value, mapping) for key, value in step.env}


def config_text(settings: Settings, tag: str) -> str:
    if settings.models is None:
        return f"<models>/{tag}/config.json"
    return str(settings.models / tag / "config.json")


def display_command(step: runs.Step, settings: Settings, number: int) -> str:
    folder = "<result folder>"
    engine_text = str(settings.engine)
    tokens = resolve_argv(step, number, folder, PY_DISPLAY, engine_text, config_text(settings, step.model))
    if step.kind == "engine":
        tokens = [engine_text] + tokens
    env = resolve_env(step, number, folder, PY_DISPLAY, engine_text, config_text(settings, step.model))
    env_text = " ".join(f"{key}={value}" for key, value in env.items())
    command = shlex.join(tokens)
    return f"{env_text} {command}".strip()


def blocked_reason(step: runs.Step) -> str | None:
    if step.unresolved:
        return f"{step.name}: input not resolvable yet: {step.unresolved[0]}"
    if step.kind in ("server", "http"):
        return f"{step.name}: {step.kind} steps are not executed by the runner yet (T12)"
    return None


def first_reason(spec: runs.RunSpec, plan: list[runs.Step], found: list[checks.Check]) -> str | None:
    if spec.plan is None:
        return f"{spec.name} has no command plan yet: composed by {spec.task}"
    # A criterion whose reference is not named can never pass (criteria.py, basis reference): the run is NOT RUN
    # before any step starts, so no engine time is spent on it (09 sections 4.1 and 6.1).
    unnamed = [cid for cid in spec.criteria if criteria.get(cid).basis == criteria.REFERENCE]
    if unnamed:
        return f"{', '.join(unnamed)} {criteria.REFERENCE_NOTE}"
    for check in found:
        if not check.ok:
            return f"{check.name}: {check.detail}"
    for step in plan:
        reason = blocked_reason(step)
        if reason is not None:
            return reason
    if spec.criteria and spec.measure is None:
        return f"criteria not measured yet: {', '.join(spec.criteria)} (task {spec.task})"
    if not spec.criteria:
        return f"{spec.name} has no criteria of its own: composition is {spec.task}"
    return None


def print_checks(found: list[checks.Check]) -> None:
    for check in found:
        status = "ok" if check.ok else "UNMET"
        print(f"check {check.name}: {status}: {check.detail}")


def execute(spec: runs.RunSpec, settings: Settings, plan: list[runs.Step], facts: Facts,
            started: datetime) -> int:
    """Run the steps in order, judge them, and write result.json. Returns the exit code."""
    version = facts.engine_version
    commit = result.project_commit(settings.root)
    folder = settings.out / result.folder_name(
        started, version.engine_version if version else None, version.metal_rev if version else None, commit)
    steps_dir = folder / "steps"
    steps_dir.mkdir(parents=True, exist_ok=True)
    archive = folder / "archive"
    crash_dir = folder / "crash"
    clock = time.monotonic()

    step_rows: list[dict[str, object]] = []
    outputs = runs.RunResults()
    failing: criteria.Verdict | None = None
    for number, step in enumerate(plan, start=1):
        engine_text = str(settings.engine)
        step_config = config_text(settings, step.model)
        tokens = resolve_argv(step, number, str(folder), sys.executable, engine_text, step_config)
        argv = checks.engine_command(settings.engine) + tokens if step.kind == "engine" else tokens
        env_record = resolve_env(step, number, str(folder), sys.executable, engine_text, step_config)
        env_record.setdefault("STRATA_METAL_ARCHIVE", str(archive))
        env = dict(os.environ)
        env.update(env_record)
        print(f"step {number}/{len(plan)}: {step.name}", file=sys.stderr, flush=True)
        step_result = stepper.run_step(
            argv, number=number, name=step.name, model=step.model, prompt=step.prompt, steps_dir=steps_dir,
            env=env, env_record=env_record, cwd=settings.root, limit_s=stepper.step_limit(None),
            crash_dir=crash_dir,
        )
        step_rows.append({"n": number, "name": step.name, "seconds": round(step_result.seconds, 3),
                          "exit": step_result.exit, "model": step.model, "prompt": step.prompt})
        if step_result.timed_out:
            failing = criteria.judge("STEP.timeout", 1, note="timeout")
            break
        if step_result.exit != 0:
            failing = criteria.judge("STEP.exit", step_result.exit, note=f"step {step.name} exited")
            break
        outputs.outputs[step.name] = runs.Output(
            stdout=step_result.stdout.read_text(encoding="utf-8", errors="replace"),
            stderr=step_result.stderr.read_text(encoding="utf-8", errors="replace"),
            stdout_file=f"steps/{step_result.stdout.name}",
            stderr_file=f"steps/{step_result.stderr.name}",
        )

    verdicts: list[criteria.Verdict] = []
    if failing is None:
        try:
            verdicts = spec.measure(outputs) if spec.measure is not None else []
        except engine.EngineOutputError as exc:
            return _not_run(spec, settings, facts, folder, started, clock, step_rows, str(exc))
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # A measurement that cannot be judged is NOT RUN (exit 2), never a criterion failure (exit 1).
            return _not_run(spec, settings, facts, folder, started, clock, step_rows,
                            f"measurement error: {exc}")
        missing = [cid for cid in spec.criteria if cid not in {v.id for v in verdicts}]
        if missing:
            return _not_run(spec, settings, facts, folder, started, clock, step_rows,
                            f"no verdict for {', '.join(missing)}")
    else:
        verdicts = [failing]

    blocking = [verdict for verdict in verdicts if verdict.blocks]
    # A gating criterion that section 5.4 lets us skip was not measured, so the run is NOT RUN, never PASS.
    skipped_gating = tuple(dict.fromkeys(v.id for v in verdicts if v.skipped and v.gating))
    if blocking:
        status = "fail"
    elif skipped_gating:
        status = "not-run"
    else:
        status = "pass"
    seconds = time.monotonic() - clock
    _write(spec, settings, facts, folder, started, seconds, status, step_rows, verdicts)
    if not blocking and not settings.keep:
        _delete_dumps(plan, steps_dir)

    count = len(verdicts)
    if blocking:
        print(f"accept {spec.name}: FAIL {blocking[0].message}")
        return EXIT_FAIL
    if skipped_gating:
        print(f"accept {spec.name}: NOT RUN: skipped (section 5.4): {', '.join(skipped_gating)}")
        return EXIT_NOT_RUN
    suffix = " (restricted run: does not count for a gate or a release)" if settings.restricted else ""
    skipped = tuple(dict.fromkeys(verdict.id for verdict in verdicts if verdict.skipped))
    if skipped:
        # A skipped criterion that does not gate the run is named on the line.
        suffix += f" (NOT RUN, skipped: {', '.join(skipped)})"
    print(f"accept {spec.name}: PASS ({count} criteria, {seconds / 60:.1f} min){suffix}")
    return EXIT_PASS


def _delete_dumps(plan: list[runs.Step], steps_dir: Path) -> None:
    for number, step in enumerate(plan, start=1):
        for suffix in step.dumps:
            path = steps_dir / f"{number:03d}{suffix}"
            if path.is_file():
                path.unlink()


def _write(spec: runs.RunSpec, settings: Settings, facts: Facts, folder: Path, started: datetime, seconds: float,
           status: str, step_rows: list[dict[str, object]], verdicts: list[criteria.Verdict]) -> None:
    version = facts.engine_version
    engine_info = {
        "path": str(settings.engine),
        "sha256": result.file_sha256(settings.engine),
        "version": version.engine_version if version else None,
        "metal_rev": version.metal_rev if version else None,
        "metallib_sha256": result.metallib_sha256(settings.engine),
    }
    source = {
        "commit": result.project_commit(settings.root),
        "upstream_sync": settings.sync,
        "prompts_builder_version": facts.prompts_builder,
    }
    models = [{"tag": tag, "quick": value} for tag, value in facts.quick.items()]
    golden = {"folder": None if settings.golden is None else str(settings.golden), "upstream_sync": settings.sync}
    criterion_rows = [
        {"id": verdict.id, "value": verdict.value, "op": verdict.op, "threshold": verdict.threshold,
         "pass": verdict.passed, "gating": verdict.gating, "note": verdict.note}
        for verdict in verdicts
    ]
    data = result.build_result(
        run=spec.name, status=status, started=started, seconds=seconds, engine=engine_info, source=source,
        host=result.host_info(), models=models, golden=golden, criteria=criterion_rows, steps=step_rows,
        restricted=settings.restricted,
    )
    result.write_result(folder, data)


def _not_run(spec: runs.RunSpec, settings: Settings, facts: Facts, folder: Path, started: datetime,
             clock: float, step_rows: list[dict[str, object]], reason: str) -> int:
    _write(spec, settings, facts, folder, started, time.monotonic() - clock, "not-run", step_rows, [])
    print(f"accept {spec.name}: NOT RUN: {reason}")
    return EXIT_NOT_RUN


def run_cli(argv: list[str] | None, registry: dict[str, runs.RunSpec], root: Path, env: dict[str, str]) -> int:
    parser = build_parser(list(registry))
    args = parser.parse_args(argv)
    spec = registry[args.run]
    validate(parser, args, spec)
    settings = settings_from(args, root, env)

    tags = tuple(args.model) or spec.models
    model_arguments, _ = model_args(settings, tags)
    context = runs.Context(models=tags, args=model_arguments, prompts=settings.prompts, short=settings.short,
                           day=settings.day)
    plan = spec.plan(context) if spec.plan is not None else []

    facts = Facts()
    found = gather_checks(settings, spec, plan, tags, facts)
    reason = first_reason(spec, plan, found)

    if settings.dry_run:
        for number, step in enumerate(plan, start=1):
            print(f"{step.name}: {display_command(step, settings, number)}")
        print(f"{len(plan)} commands listed for {spec.name}")
        print_checks(found)
        if reason is not None:
            print(f"accept {spec.name}: NOT RUN: {reason}")
            return EXIT_NOT_RUN
        print(f"accept {spec.name}: DRY RUN: {len(plan)} commands, inputs checked, no model loaded")
        return EXIT_PASS

    if reason is not None:
        print_checks(found)
        print(f"accept {spec.name}: NOT RUN: {reason}")
        return EXIT_NOT_RUN
    started = datetime.now(timezone.utc).replace(microsecond=0)
    return execute(spec, settings, plan, facts, started)


def main(argv: list[str] | None = None) -> int:
    return run_cli(argv, runs.RUNS, ROOT, dict(os.environ))


if __name__ == "__main__":
    raise SystemExit(main())
