"""The benchmark's host-verifiable logic (docs/mac/09-testing-and-validation.md, sections 5.1-5.4).

Pure functions: how a timed run becomes a number (the one-shot lines are read by engine.py), the spread rule, PF-1 and
PF-2 from the prefill lines and the chunk line (section 5.2), the regression bar and comparison (section 5.4), the trend
rule, the BENCH rows, and the host checks that stop a run on a laptop or a hot Mac (section 5.1).

Nothing here runs a model. A value that is missing or cannot be judged raises BenchError (or EngineOutputError), which
the runner reports as NOT RUN. A criterion that section 5.4 lets a run skip is returned as skipped, which is NOT RUN
and never a pass (criteria.skip). No function returns a pass for a value it did not measure.
"""
from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

from . import checks, criteria, engine, result
from .criteria import Verdict

SPREAD_LIMIT = 0.05  # section 5.1, step 3: a spread above 5% makes three more runs (three runs) or is noisy (six)
FIRST_RUNS = 3
SETTLED_RUNS = 6
REG_BAR = 0.95  # section 5.4: the bar while T-4's coefficient of variation is at most 2%
CV_LIMIT = 0.02  # T-4's provisional default (section 11); the bar stays 0.95 until T-4 records a value
TREND_PRINT = 0.03  # section 5.4: a change of more than 3% against the previous night is printed
MEMORY_LEVEL_MIN = 50  # section 5.1: kern.memorystatus_level at least 50
AC_POWER = "AC Power"
MTP = "mtp"
PF_PROMPTS = ("ca-32k", "ca-128k")
HEADLINE = {("mtp", "ca-4k"), ("plain", "ca-4k")}
# (metric, prompt): the points the regression check (REG.*) compares; the trend rule reads the same points.
REG_POINTS = (("decode", "ca-1k"), ("decode", "ca-4k"), ("decode", "ca-32k"), ("prefill", "ca-32k"))

Key = tuple[str, str]  # (arm, prompt)


class BenchError(ValueError):
    """A measurement that cannot be judged: a missing run, a spread the plan does not re-run, an unusable rate."""


class StepOutput(Protocol):
    stdout: str
    stderr: str
    stdout_file: str
    stderr_file: str


@dataclass(frozen=True)
class Summary:
    """The median of one quantity over three or six runs, with its spread (section 5.1, steps 2-3)."""

    median: float
    spread: float
    runs: int
    rerun: bool  # three runs, spread above 5%: the three more runs of step 3 are due
    noisy: bool  # six runs, spread still above 5%: recorded as `noisy`


@dataclass(frozen=True)
class Measured:
    """One arm on one prompt: decode and prefill summaries, the time to first token and the chunk size (section 5.2)."""

    decode: Summary
    prefill: Summary
    time_to_first_token_ms: float
    chunk: int | None


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def summarize(values: Sequence[float]) -> Summary:
    """The median of three or six runs. The spread is (max - min) / median, and a spread above 5% is over the limit."""
    count = len(values)
    if count not in (FIRST_RUNS, SETTLED_RUNS):
        raise BenchError(f"bench: {count} runs, want 3 or 6")
    if not all(_is_number(value) and value > 0 for value in values):
        raise BenchError(f"bench: every rate must be a positive number, got {list(values)!r}")
    median = statistics.median(values)
    spread = (max(values) - min(values)) / median
    over = spread > SPREAD_LIMIT
    return Summary(median=median, spread=spread, runs=count, rerun=count == FIRST_RUNS and over,
                   noisy=count == SETTLED_RUNS and over)


def parse_run(outputs: Mapping[str, StepOutput], name: str, *, chunk: bool) -> engine.OneShot:
    """One step's one-shot output. The decode and prefill lines are required; the chunk line only when `chunk`."""
    output = outputs.get(name)
    if output is None:
        raise BenchError(f"bench: no output for step {name}")
    return engine.parse_one_shot(
        output.stdout, source=output.stdout_file, stderr=output.stderr, stderr_source=output.stderr_file,
        need_prefill=True, need_chunk=chunk,
    )


def measure_arm(parsed: Sequence[engine.OneShot]) -> Measured:
    """Summarize one arm on one prompt from its measured runs. The plan has no re-run, so a rerun spread is NOT RUN."""
    prefills = [run.prefill for run in parsed]
    if any(prefill is None for prefill in prefills):
        raise BenchError("bench: a run has no prefill line")
    decode = summarize([run.decode.tok_s for run in parsed])
    prefill = summarize([item.timing.tok_s for item in prefills if item is not None])
    for label, summary in (("decode", decode), ("prefill", prefill)):
        if summary.rerun:
            raise BenchError(
                f"bench: {label} spread {summary.spread:.1%} is above 5% over three runs; the three more runs of "
                "section 5.1 step 3 are not in the plan yet"
            )
    chunks = {run.chunk for run in parsed}
    if len(chunks) != 1:
        raise BenchError(f"bench: the chunk size differs between runs: {sorted(chunks, key=str)}")
    ttft = statistics.median([item.time_to_first_token_ms for item in prefills if item is not None])
    return Measured(decode=decode, prefill=prefill, time_to_first_token_ms=ttft, chunk=chunks.pop())


def _spread_note(summary: Summary) -> str:
    return f"spread {summary.spread:.1%} over {summary.runs} runs" + ("; noisy" if summary.noisy else "")


def measure_pf(outputs: Mapping[str, StepOutput], *, prefix: str = "pf") -> list[Verdict]:
    """PF-2 and PF-1 (section 5.2) from the mtp arm's measured runs of ca-32k and ca-128k, in the same run.

    PF-2 is the ca-32k prefill median in tok/s. PF-1 is the ca-128k median over the ca-32k median, both prefill rates.
    The chunk size each prompt used is in the note, and the time to first token on ca-32k is reported with PF-2.
    """
    measured = {
        prompt: measure_arm([parse_run(outputs, f"{prefix}.{MTP}.{prompt}.run{number}", chunk=True)
                             for number in (1, 2, 3)])
        for prompt in PF_PROMPTS
    }
    long, short = measured["ca-32k"], measured["ca-128k"]
    pf2 = criteria.judge(
        "PF-2", long.prefill.median,
        note=(f"ca-32k prefill median, tok/s; time to first token {long.time_to_first_token_ms:.1f} ms; "
              f"chunk {long.chunk}; {_spread_note(long.prefill)}"),
    )
    pf1 = criteria.judge(
        "PF-1", short.prefill.median / long.prefill.median,
        note=(f"ca-128k {short.prefill.median:.2f} tok/s over ca-32k {long.prefill.median:.2f} tok/s, same run; "
              f"chunks {short.chunk} and {long.chunk}"),
    )
    return [pf2, pf1]


def point(measured: Mapping[Key, Measured], arm: str, prompt: str, metric: str, who: str) -> Summary:
    """The summary of one arm, prompt and metric. A missing entry is an error, never a default."""
    entry = measured.get((arm, prompt))
    if entry is None:
        raise BenchError(f"bench: no {arm} {prompt} result for the {who}")
    return entry.decode if metric == "decode" else entry.prefill


def regression_bar(cv: float | None) -> float:
    """0.95, or 1 - 2.5 x CV above a 2% coefficient of variation (section 5.4). None means T-4 has recorded no value,
    so its provisional default (at most 2%, section 11) stands and the bar is 0.95."""
    if cv is None:
        return REG_BAR
    if not _is_number(cv) or cv < 0:
        raise BenchError(f"bench: the coefficient of variation must be a finite number of at least 0, got {cv!r}")
    if cv <= CV_LIMIT:
        return REG_BAR
    return 1 - 2.5 * cv


def previous_engine_skip(exit_code: int, stderr: str) -> str:
    """The skip note when the previous release refuses the run's arguments (section 5.4), else an empty string.

    A refused flag is `unknown argument: <flag>`. Any other failure of the previous release is NOT RUN: the spec's skip
    also covers an exit while loading, and the loader's message is not known yet, so such an exit is not guessed at.
    """
    lines = [line.strip() for line in stderr.splitlines() if line.strip()]
    last = lines[-1] if lines else "no stderr line"
    if "unknown argument:" in stderr:
        return f"previous release refused the arguments: {last}"
    if exit_code != 0:
        raise BenchError(f"bench: previous release exited {exit_code} before its measured runs: {last}")
    return ""


def _judged_ratio(criterion: str, pairs: Sequence[tuple[str, Summary, Summary]], bar: float) -> Verdict:
    """The worst current-over-previous ratio decides. A noisy pair cannot fail the check: it is noted and not judged."""
    judged: list[tuple[str, float]] = []
    noisy: list[str] = []
    for label, now, before in pairs:
        if now.noisy or before.noisy:
            noisy.append(label)
            continue
        judged.append((label, now.median / before.median))
    if not judged:
        return criteria.skip(criterion, f"every value is noisy, not judged: {', '.join(noisy)}")
    label, ratio = min(judged, key=lambda item: item[1])
    note = f"lowest of {len(judged)}: {label}, current over previous; bar {bar:.4g}"
    if noisy:
        note += f"; noisy, not judged: {', '.join(noisy)}"
    return criteria.judge(criterion, ratio, bar, note=note)


def regression_verdicts(
    current: Mapping[Key, Measured],
    previous: Mapping[Key, Measured] | None,
    *,
    cv: float | None = None,
    skip_note: str = "",
) -> list[Verdict]:
    """REG.decode and REG.prefill (section 5.4): this engine's mtp medians over the previous release's, on this Mac.

    With no previous release the two are skipped with `no previous release`; with a refusal, with the refusal's note.
    They are never passed without a measured baseline.
    """
    bar = regression_bar(cv)
    if skip_note or previous is None:
        note = skip_note or "no previous release"
        return [criteria.skip("REG.decode", note), criteria.skip("REG.prefill", note)]
    decode = [
        (f"decode {prompt}", point(current, MTP, prompt, "decode", "current"),
         point(previous, MTP, prompt, "decode", "previous release"))
        for prompt in ("ca-1k", "ca-4k", "ca-32k")
    ]
    prefill = [
        ("prefill ca-32k", point(current, MTP, "ca-32k", "prefill", "current"),
         point(previous, MTP, "ca-32k", "prefill", "previous release")),
    ]
    return [_judged_ratio("REG.decode", decode, bar), _judged_ratio("REG.prefill", prefill, bar)]


def trend_verdict(current: Mapping[Key, Measured], previous_night: Mapping[Key, Measured] | None) -> Verdict:
    """REG.trend (section 5.4): the change against the previous night's result of this engine, on the same points.

    The value is the change of the point that moved most. It is reported, never gating, and printed in the note when its
    size is more than 3%. With no previous night the check is skipped.
    """
    if previous_night is None:
        return criteria.skip("REG.trend", "no previous night's result")
    changes: list[tuple[str, float]] = []
    noisy: list[str] = []
    for metric, prompt in REG_POINTS:
        label = f"{metric} {prompt}"
        now = point(current, MTP, prompt, metric, "current")
        before = point(previous_night, MTP, prompt, metric, "previous night")
        changes.append((label, now.median / before.median - 1))
        if now.noisy or before.noisy:
            noisy.append(label)
    label, change = max(changes, key=lambda item: abs(item[1]))
    printed = abs(change) > TREND_PRINT
    note = f"largest of {len(changes)}: {label} {change:+.1%} against the previous night; "
    note += "printed: more than 3%" if printed else "within 3%, not printed"
    if noisy:
        note += f"; noisy: {', '.join(noisy)}"
    return criteria.judge("REG.trend", change, note=note)


def bench_verdicts(current: Mapping[Key, Measured]) -> list[Verdict]:
    """BENCH.mtp, BENCH.mtp.target and BENCH.plain on ca-4k, and BENCH.other for every other arm and prompt (5.1).

    One model per call. The values are decode medians in tok/s.
    """
    mtp = point(current, "mtp", "ca-4k", "decode", "current")
    plain = point(current, "plain", "ca-4k", "decode", "current")
    verdicts = [
        criteria.judge("BENCH.mtp", mtp.median, note=f"mtp decode median on ca-4k, tok/s; {_spread_note(mtp)}"),
        criteria.judge("BENCH.mtp.target", mtp.median, note="mtp decode median on ca-4k, tok/s; the target"),
        criteria.judge("BENCH.plain", plain.median, note=f"plain decode median on ca-4k, tok/s; {_spread_note(plain)}"),
    ]
    for (arm, prompt), entry in sorted(current.items()):
        if (arm, prompt) in HEADLINE:
            continue
        verdicts.append(criteria.judge("BENCH.other", entry.decode.median,
                                       note=f"{arm} {prompt} decode median, tok/s; reported"))
    return verdicts


def power_check(batteries: str | None) -> checks.Check:
    name = "host: power source"
    source = result.power_source(batteries)
    if source is None:
        return checks.Check(name, False, "pmset -g batt does not name a power source")
    if source != AC_POWER:
        return checks.Check(name, False, f"drawing from '{source}', want {AC_POWER}")
    return checks.Check(name, True, AC_POWER, value=source)


def low_power_check(power: str | None) -> checks.Check:
    name = "host: low power mode"
    state = result.low_power_mode(power)
    if state is None:
        return checks.Check(name, False, "pmset -g does not report lowpowermode or powermode")
    if state:
        return checks.Check(name, False, "low power mode is on, want 0")
    return checks.Check(name, True, "off", value=False)


def thermal_check(therm: str | None) -> checks.Check:
    name = "host: thermal state"
    state = result.thermal_warning(therm)
    if state is None:
        return checks.Check(name, False, "pmset -g therm does not report both warning levels")
    if state:
        return checks.Check(name, False, "a thermal or performance warning level is recorded")
    return checks.Check(name, True, "no thermal or performance warning", value=False)


def memory_check(text: str | None) -> checks.Check:
    name = "host: memory status"
    level = result.memory_status_level(text)
    if level is None:
        return checks.Check(name, False, "kern.memorystatus_level is not a number")
    if level < MEMORY_LEVEL_MIN:
        return checks.Check(name, False, f"kern.memorystatus_level {level}, want at least {MEMORY_LEVEL_MIN}")
    return checks.Check(name, True, str(level), value=level)


def host_checks(run: Callable[[list[str]], str | None] | None = None) -> list[checks.Check]:
    """The section 5.1 host checks, before the first run of a benchmark. Each unmet check stops the run with exit 2.

    The pgrep check (no other strata process) is shared with every run and is made by checks.no_engine_running.
    """
    read = result.run_text if run is None else run
    return [
        power_check(read(["pmset", "-g", "batt"])),
        low_power_check(read(["pmset", "-g"])),
        thermal_check(read(["pmset", "-g", "therm"])),
        memory_check(read(["sysctl", "-n", "kern.memorystatus_level"])),
    ]
