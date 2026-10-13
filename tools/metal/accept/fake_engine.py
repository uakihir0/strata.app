#!/usr/bin/env python3
"""A stand-in for the engine's command line, for the acceptance runner's tests (docs/mac/09, section 3.2).

It accepts the engine's arguments (`--name [value]`; any flag is accepted), prints the one-shot lines in the formats of
src/metal/program/oneshot.cpp, writes the `--dump-logits` file and `STRATA_DUMP_FIRST_LOGITS`, and answers
`--version` and `--list-devices`. Its output is a fixed function of the prompt and the sampling flags, so every arm that
should agree does agree. It is not a model.

Planted differences, one per run, from FAKE_ENGINE_PLANT=<kind>:<where>:
  output:<i>        the generated token i is changed (default 0)
  logits:<r>.<c>    the dumped logit at row r, column c is changed by 0.5 (default 0.0)
  spec:accept       half of the drafts are reported as accepted
  decode:slow       the decode time is multiplied by 4
  prefill:slow      the prefill time is multiplied by 4
  chunk:<n>         the chunk size on stderr is n (default 4096)
  missing:<label>   the stdout line with that label is not printed (prompt, output, decode, prefill, speculation)
  exit:<code>       the process exits with that code after printing
  hang              the process sleeps until it is killed
"""
from __future__ import annotations

import math
import os
import re
import struct
import sys
import time

VERSION_LINE = "strata 0.0.0-fake metal 1"
DEVICES_LINE = "0 Fake Apple GPU"
N_VOCAB = 16
DECODE_MS_PER_TOKEN = 5.0
PREFILL_MS_PER_TOKEN = 2.0
DEFAULT_CHUNK = 4096


def parse_plant(value: str | None) -> tuple[str, str]:
    if not value:
        return "", ""
    kind, _, where = value.partition(":")
    return kind, where


def parse_args(argv: list[str]) -> dict[str, str]:
    """Return flags mapped to their value ("" for a value-less flag). Every token must start with `--`."""
    flags: dict[str, str] = {}
    index = 0
    while index < len(argv):
        token = argv[index]
        if not token.startswith("--"):
            raise SystemExit(f"fake_engine: unexpected argument: {token}")
        if index + 1 < len(argv) and not argv[index + 1].startswith("--"):
            flags[token] = argv[index + 1]
            index += 2
        else:
            flags[token] = ""
            index += 1
    return flags


def read_ids(flags: dict[str, str]) -> list[int]:
    if "--tokens-file" in flags:
        try:
            with open(flags["--tokens-file"], encoding="utf-8") as handle:
                text = handle.read()
        except OSError as exc:
            raise SystemExit(f"fake_engine: cannot read tokens: {exc}") from exc
    elif "--tokens" in flags:
        text = flags["--tokens"]
    else:
        return [1, 2, 3]
    tokens = [token for token in re.split(r"[,\s]+", text.strip()) if token]
    try:
        return [int(token) for token in tokens]
    except ValueError as exc:
        raise SystemExit(f"fake_engine: bad token id: {exc}") from exc


def generate(ids: list[int], flags: dict[str, str], max_new: int) -> list[int]:
    base = sum(ids)
    if "--seed" in flags and float(flags.get("--temperature", "0") or 0) > 0:
        seed = int(flags["--seed"])
        return [(seed * 13 + i * 29 + base) % 1000 for i in range(max_new)]
    return [(base * 31 + i * 17 + 7) % 1000 for i in range(max_new)]


def logits_rows(ids: list[int], rows: int, plant_row: int | None, plant_col: int | None) -> list[list[float]]:
    base = sum(ids)
    table = []
    for r in range(rows):
        row = [((base + r * 7 + c * 3) % 11) / 4.0 for c in range(N_VOCAB)]
        if plant_row == r and plant_col is not None:
            row[plant_col] += 0.5
        table.append(row)
    return table


def write_dump(path: str, rows: list[list[float]]) -> None:
    payload = struct.pack("<ii", N_VOCAB, len(rows))
    for row in rows:
        payload += struct.pack(f"<{N_VOCAB}f", *row)
    with open(path, "wb") as handle:
        handle.write(payload)


def write_first_row(path: str, row: list[float]) -> None:
    with open(path, "wb") as handle:
        handle.write(struct.pack(f"<{N_VOCAB}f", *row))


def ids_text(values: list[int]) -> str:
    return "".join(f" {value}" for value in values)


def main(argv: list[str]) -> int:
    if argv == ["--version"]:
        print(VERSION_LINE)
        return 0
    if argv == ["--list-devices"]:
        print(DEVICES_LINE)
        return 0

    kind, where = parse_plant(os.environ.get("FAKE_ENGINE_PLANT"))
    if kind == "hang":
        time.sleep(3600)
        return 0

    flags = parse_args(argv)
    ids = read_ids(flags)
    max_new = int(flags.get("--max-new", "8"))
    output = generate(ids, flags, max_new)
    if kind == "output":
        index = int(where or 0)
        if not 0 <= index < len(output):
            raise SystemExit(f"fake_engine: output index {index} out of range")
        output[index] = (output[index] + 1) % 1000

    plant_row = plant_col = None
    if kind == "logits":
        row_text, _, col_text = where.partition(".")
        plant_row, plant_col = int(row_text or 0), int(col_text or 0)

    if "--dump-logits" in flags:
        write_dump(flags["--dump-logits"], logits_rows(ids, max_new, plant_row, plant_col))
    first_logits = os.environ.get("STRATA_DUMP_FIRST_LOGITS")
    if first_logits:
        write_first_row(first_logits, logits_rows(ids, max(1, max_new), plant_row, plant_col)[0])

    lines: list[str] = []
    spec = int(flags.get("--spec", "0") or 0)
    if spec > 0:
        rounds = math.ceil(max_new / spec)
        drafts = rounds * spec
        corrupt = int(flags.get("--spec-corrupt", "0") or 0)
        accepted = drafts - (drafts // corrupt if corrupt > 0 else 0)
        if kind == "spec" and where == "accept":
            accepted = drafts // 2
        tokens_per_round = (accepted + rounds) / rounds
        acceptance = accepted / drafts if drafts else 0.0
        lines.append(
            f"{'speculation':<24} {rounds} rounds of {spec}, drafts accepted {accepted} of {drafts} "
            f"({acceptance:.3f}), {tokens_per_round:.2f} tokens per round"
        )

    decode_factor = 4.0 if kind == "decode" and where == "slow" else 1.0
    decode_ms = max_new * DECODE_MS_PER_TOKEN * decode_factor
    decode_tok_s = 1000.0 * max_new / decode_ms if decode_ms > 0 else 0.0
    lines.append("prompt  :" + ids_text(ids))
    lines.append("output  :" + ids_text(output))
    lines.append(
        f"{'decode':<24} {max_new} tokens in {decode_ms:.1f} ms  ->  {decode_tok_s:.2f} tok/s"
    )
    if len(ids) > 1:
        prefill_tokens = len(ids) - 1
        prefill_factor = 4.0 if kind == "prefill" and where == "slow" else 1.0
        prefill_ms = prefill_tokens * PREFILL_MS_PER_TOKEN * prefill_factor
        prefill_tok_s = 1000.0 * prefill_tokens / prefill_ms if prefill_ms > 0 else 0.0
        lines.append(
            f"{'prefill':<24} {prefill_tokens} tokens in {prefill_ms:.1f} ms  ->  {prefill_tok_s:.2f} tok/s"
            f"  (time to first token {prefill_ms + 1.5:.1f} ms)"
        )

    labels = ["speculation", "prompt", "output", "decode", "prefill"]
    if kind == "missing" and where in labels:
        lines = [line for line in lines if not _has_label(line, where)]

    sys.stdout.write("\n".join(lines) + "\n")
    sys.stdout.flush()

    if "--prefill" in flags and flags["--prefill"] == "auto":
        chunk = DEFAULT_CHUNK
        if kind == "chunk":
            chunk = int(where or DEFAULT_CHUNK)
        sys.stderr.write(f"strata generate: prompt chunk auto: {chunk} tokens\n")
        sys.stderr.flush()

    if kind == "exit":
        return int(where or 1)
    return 0


def _has_label(line: str, label: str) -> bool:
    if label == "prompt":
        return line.startswith("prompt  :")
    if label == "output":
        return line.startswith("output  :")
    return line.startswith(label)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
