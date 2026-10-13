"""The one parser of the engine's one-shot, version and device output (docs/mac/09-testing-and-validation.md, section 3.1).

Formats follow src/metal/program/oneshot.cpp (`%-24s` labels). A line that a caller needs and that is absent or
repeated is an error: the parser never reads a default. The error text is `accept: <file>: no <label> line`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

NUM = r"-?\d+(?:\.\d+)?"
PROMPT_RE = re.compile(r"^prompt  :((?: \d+)*)$")
OUTPUT_RE = re.compile(r"^output  :((?: \d+)*)$")
DECODE_RE = re.compile(rf"^decode\s+(\d+) tokens in ({NUM}) ms\s+->\s+({NUM}) tok/s$")
PREFILL_RE = re.compile(
    rf"^prefill\s+(\d+) tokens in ({NUM}) ms\s+->\s+({NUM}) tok/s\s+\(time to first token ({NUM}) ms\)$"
)
SPECULATION_RE = re.compile(
    rf"^speculation\s+(\d+) rounds of (\d+), drafts accepted (\d+) of (\d+) \(({NUM})\), ({NUM}) tokens per round$"
)
CHUNK_RE = re.compile(r"^strata generate: prompt chunk auto: (\d+) tokens$")
VERSION_RE = re.compile(r"^strata (\S+) metal (\d+)$")


class EngineOutputError(ValueError):
    """A line the caller needs is missing or repeated."""


@dataclass(frozen=True)
class Timing:
    tokens: int
    ms: float
    tok_s: float


@dataclass(frozen=True)
class Prefill:
    timing: Timing
    time_to_first_token_ms: float


@dataclass(frozen=True)
class Speculation:
    rounds: int
    window: int
    accepted: int
    drafts: int
    acceptance: float
    tokens_per_round: float


@dataclass(frozen=True)
class OneShot:
    prompt: tuple[int, ...]
    output: tuple[int, ...]
    decode: Timing
    prefill: Prefill | None
    speculation: Speculation | None
    chunk: int | None


@dataclass(frozen=True)
class Version:
    engine_version: str
    metal_rev: int


def _lines(text: str) -> list[str]:
    return text.splitlines()


def _find(lines: list[str], label: str, pattern: re.Pattern[str], source: str, *, required: bool) -> re.Match[str] | None:
    """Return the single line that matches `pattern` and starts with the label, or fail."""
    matches = [m for m in (pattern.match(line) for line in lines) if m is not None]
    if not matches:
        if required:
            raise EngineOutputError(f"accept: {source}: no {label} line")
        return None
    if len(matches) > 1:
        raise EngineOutputError(f"accept: {source}: {len(matches)} {label} lines, want one")
    return matches[0]


def _ids(group: str) -> tuple[int, ...]:
    return tuple(int(token) for token in group.split())


def parse_one_shot(
    stdout: str,
    *,
    source: str,
    stderr: str = "",
    stderr_source: str = "",
    need_prefill: bool = False,
    need_speculation: bool = False,
    need_chunk: bool = False,
) -> OneShot:
    """Parse one one-shot run. `source` names the stdout file in errors; `stderr_source` names the stderr file."""
    out_lines = _lines(stdout)
    prompt = _find(out_lines, "prompt", PROMPT_RE, source, required=True)
    output = _find(out_lines, "output", OUTPUT_RE, source, required=True)
    decode = _find(out_lines, "decode", DECODE_RE, source, required=True)
    prefill = _find(out_lines, "prefill", PREFILL_RE, source, required=need_prefill)
    speculation = _find(out_lines, "speculation", SPECULATION_RE, source, required=need_speculation)

    chunk = None
    if need_chunk:
        err_lines = _lines(stderr)
        found = _find(err_lines, "chunk", CHUNK_RE, stderr_source or source, required=True)
        chunk = int(found.group(1))

    output_ids = _ids(output.group(1))
    if int(decode.group(1)) != len(output_ids):
        raise EngineOutputError(
            f"accept: {source}: decode line counts {decode.group(1)} tokens, output line has {len(output_ids)}"
        )
    return OneShot(
        prompt=_ids(prompt.group(1)),
        output=output_ids,
        decode=Timing(tokens=int(decode.group(1)), ms=float(decode.group(2)), tok_s=float(decode.group(3))),
        prefill=None if prefill is None else Prefill(
            timing=Timing(tokens=int(prefill.group(1)), ms=float(prefill.group(2)), tok_s=float(prefill.group(3))),
            time_to_first_token_ms=float(prefill.group(4)),
        ),
        speculation=None if speculation is None else Speculation(
            rounds=int(speculation.group(1)),
            window=int(speculation.group(2)),
            accepted=int(speculation.group(3)),
            drafts=int(speculation.group(4)),
            acceptance=float(speculation.group(5)),
            tokens_per_round=float(speculation.group(6)),
        ),
        chunk=chunk,
    )


def parse_version(stdout: str, *, source: str) -> Version:
    """Parse `strata --version`: `strata <version> metal <rev>`."""
    match = _find(_lines(stdout), "version", VERSION_RE, source, required=True)
    return Version(engine_version=match.group(1), metal_rev=int(match.group(2)))


def parse_devices(stdout: str, *, source: str) -> tuple[str, ...]:
    """Parse `strata --list-devices`. The format is not fixed by the spec, so the parser requires one non-empty line."""
    devices = tuple(line.strip() for line in _lines(stdout) if line.strip())
    if not devices:
        raise EngineOutputError(f"accept: {source}: no devices line")
    return devices
