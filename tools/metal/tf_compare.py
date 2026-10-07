#!/usr/bin/env python3
"""Compare two Strata teacher-forced logits dumps on the same token positions."""
from __future__ import annotations

import argparse
import json
import math
import re
import struct
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class LogitsDump:
    path: Path
    vocab_size: int
    declared_rows: int
    first_position: int
    logits: np.memmap


def load(path: str | Path) -> LogitsDump:
    """Read a dump header and map every complete F32 row present in the file.

    The second header integer is the full number of rows the engine planned to write.
    Prompt-path dumps may be shorter, so their first row starts at declared_rows - rows_on_disk.
    """
    source = Path(path)
    try:
        size = source.stat().st_size
        with source.open('rb') as stream:
            header = stream.read(8)
    except OSError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    if len(header) != 8 or size < 8:
        raise ValueError(f"{source}: expected an 8-byte logits header")
    vocab_size, declared_rows = struct.unpack('<ii', header)
    if vocab_size < 2 or declared_rows < 1:
        raise ValueError(f"{source}: invalid header (vocabulary {vocab_size}, rows {declared_rows})")
    row_bytes = vocab_size * 4
    payload_bytes = size - 8
    if payload_bytes % row_bytes:
        raise ValueError(f"{source}: payload is not a whole number of {vocab_size}-logit rows")
    rows = payload_bytes // row_bytes
    if rows < 1 or rows > declared_rows:
        raise ValueError(f"{source}: header advertises {declared_rows} rows but file contains {rows}")
    try:
        logits = np.memmap(source, dtype='<f4', mode='r', offset=8, shape=(rows, vocab_size))
        for start in range(0, rows, 64):
            if not np.isfinite(logits[start:start + 64]).all():
                raise ValueError('logits contain NaN or infinity')
    except (OSError, ValueError) as exc:
        raise ValueError(f"{source}: could not read logits: {exc}") from exc
    return LogitsDump(source, vocab_size, declared_rows, declared_rows - rows, logits)


def load_ids(path: str | Path) -> list[int]:
    source = Path(path)
    try:
        text = source.read_text(encoding='utf-8')
    except OSError as exc:
        raise ValueError(f"{source}: {exc}") from exc
    stripped = text.strip()
    if not stripped:
        return []
    if re.search(r'(?:^,|,$|,\s*,)', stripped):
        raise ValueError(f"{source}: empty comma-separated token ID")
    try:
        return [int(token, 10) for token in re.split(r'[\s,]+', stripped)]
    except ValueError as exc:
        raise ValueError(f"{source}: expected whitespace- or comma-separated token IDs") from exc


def lsm(logits: np.ndarray) -> np.ndarray:
    """Stable row-wise log-softmax in float64."""
    values = np.asarray(logits, dtype=np.float64)
    maximum = values.max(axis=-1, keepdims=True)
    return values - (maximum + np.log(np.exp(values - maximum).sum(axis=-1, keepdims=True)))


def parse_rows(value: str) -> tuple[int, int]:
    start, separator, end = value.partition(':')
    if not separator:
        raise argparse.ArgumentTypeError('expected START:END (zero-based, end exclusive)')
    try:
        first, last = int(start), int(end)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected integer START:END') from exc
    if first < 0 or last <= first:
        raise argparse.ArgumentTypeError('expected 0 <= START < END')
    return first, last


def parse_margins(value: str) -> tuple[float, float]:
    pieces = value.split(',')
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError('expected M0,M1')
    try:
        first, second = float(pieces[0]), float(pieces[1])
    except ValueError as exc:
        raise argparse.ArgumentTypeError('expected M0,M1') from exc
    if not math.isfinite(first) or not math.isfinite(second) or first < 0 or second <= first:
        raise argparse.ArgumentTypeError('expected finite margins with 0 <= M0 < M1')
    return first, second


def compare(reference: LogitsDump, test: LogitsDump, ids: list[int], *,
            rows: tuple[int, int] | None = None, margins: tuple[float, float] = (0.5, 1.0)) -> dict:
    if reference.vocab_size != test.vocab_size:
        raise ValueError(f"vocabulary sizes differ ({reference.vocab_size} vs {test.vocab_size})")
    if reference.first_position != test.first_position or reference.logits.shape[0] != test.logits.shape[0]:
        raise ValueError("logits dumps do not cover the same token positions")
    row_count = reference.logits.shape[0]
    start, stop = rows if rows is not None else (0, row_count)
    if start < 0 or stop <= start or stop > row_count:
        raise ValueError(f"row range {start}:{stop} is outside the {row_count} rows in the dumps")

    file_rows = np.arange(start, stop, dtype=np.int64)
    positions = reference.first_position + file_rows
    target_positions = positions + 1
    covered = target_positions < len(ids)
    file_rows = file_rows[covered]
    positions = positions[covered]
    target_positions = target_positions[covered]
    if len(file_rows) == 0:
        raise ValueError("no selected dump row has a corresponding next-token ID")

    target_values = [ids[int(position)] for position in target_positions]
    for index, token_id in enumerate(target_values):
        if token_id < 0 or token_id >= reference.vocab_size:
            raise ValueError(f"token ID {token_id} at position {target_positions[index]} is outside "
                             f"the vocabulary range 0:{reference.vocab_size}")
    # Range validation above also guarantees these Python integers safely fit int64.
    targets = np.asarray(target_values, dtype=np.int64)
    m0, m1 = margins
    if not math.isfinite(m0) or not math.isfinite(m1) or m0 < 0 or m1 <= m0:
        raise ValueError("margins must be finite and satisfy 0 <= M0 < M1")

    # Keep temporary F64 arrays bounded for long-context dumps and large vocabularies.
    chunk_size = 16
    kl_parts = []
    ref_top_parts = []
    test_top_parts = []
    margin_parts = []
    ref_target_parts = []
    test_target_parts = []
    for offset in range(0, len(file_rows), chunk_size):
        limit = min(offset + chunk_size, len(file_rows))
        selected = file_rows[offset:limit]
        ref_logits = np.asarray(reference.logits[selected], dtype=np.float64)
        test_logits = np.asarray(test.logits[selected], dtype=np.float64)
        if not np.isfinite(ref_logits).all() or not np.isfinite(test_logits).all():
            raise ValueError("a selected logits row contains NaN or infinity")
        ref_logp, test_logp = lsm(ref_logits), lsm(test_logits)
        row_index = np.arange(limit - offset)
        ref_top = ref_logits.argmax(axis=1)  # numpy returns the lowest ID when maxima tie
        test_top = test_logits.argmax(axis=1)
        ref_top_values = ref_logits[row_index, ref_top]
        second_values = np.partition(ref_logits, -2, axis=1)[:, -2]
        chunk_targets = targets[offset:limit]
        ref_target = ref_logp[row_index, chunk_targets]
        test_target = test_logp[row_index, chunk_targets]
        kl_parts.append((np.exp(ref_logp) * (ref_logp - test_logp)).sum(axis=1))
        ref_top_parts.append(ref_top)
        test_top_parts.append(test_top)
        margin_parts.append(ref_top_values - second_values)
        ref_target_parts.append(ref_target)
        test_target_parts.append(test_target)

    kl = np.concatenate(kl_parts)
    ref_top = np.concatenate(ref_top_parts)
    test_top = np.concatenate(test_top_parts)
    ref_margin = np.concatenate(margin_parts)
    ref_target_logp = np.concatenate(ref_target_parts)
    test_target_logp = np.concatenate(test_target_parts)
    same = ref_top == test_top
    bands = []
    band_masks = (ref_margin < m0, (ref_margin >= m0) & (ref_margin < m1), ref_margin >= m1)
    labels = (f'[0,{m0:g})', f'[{m0:g},{m1:g})', f'[{m1:g},infinity)')
    for label, mask in zip(labels, band_masks):
        bands.append({
            'range': label,
            'rows': int(mask.sum()),
            'same_top1': int(same[mask].sum()),
        })

    disagreements = []
    for i in np.flatnonzero(~same):
        disagreements.append({
            'position': int(positions[i]),
            'reference_margin': float(ref_margin[i]),
            'reference_top1': int(ref_top[i]),
            'test_top1': int(test_top[i]),
            'reference_next_logp': float(ref_target_logp[i]),
            'test_next_logp': float(test_target_logp[i]),
        })
    n = len(file_rows)
    delta = ref_target_logp - test_target_logp
    standard_error = float(delta.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    with np.errstate(over='ignore', invalid='ignore'):
        reference_perplexity = float(np.exp(-ref_target_logp.mean()))
        test_perplexity = float(np.exp(-test_target_logp.mean()))
    scalar_metrics = (float(kl.mean()), float(np.median(kl)), float(np.percentile(kl, 99)),
                      float(kl.max()), reference_perplexity, test_perplexity, float(delta.mean()))
    if standard_error is not None:
        scalar_metrics += (standard_error,)
    if not all(math.isfinite(value) for value in scalar_metrics):
        raise ValueError("comparison produced a non-finite metric; logits exceed the supported numeric range")
    return {
        'reference': str(reference.path),
        'test': str(test.path),
        'vocab_size': reference.vocab_size,
        'rows_compared': n,
        'first_position': int(positions[0]),
        'last_position': int(positions[-1]),
        'same_top1': int(same.sum()),
        'same_top1_fraction': float(same.mean()),
        'kl_nats': {
            'mean': float(kl.mean()),
            'median': float(np.median(kl)),
            'p99': float(np.percentile(kl, 99)),
            'max': float(kl.max()),
        },
        'perplexity': {
            'reference': reference_perplexity,
            'test': test_perplexity,
        },
        'paired_delta_nats_per_token': {
            'mean': float(delta.mean()),
            'standard_error': standard_error,
        },
        'reference_margin_bands': bands,
        'disagreements': disagreements,
    }

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('reference', type=Path)
    parser.add_argument('test', type=Path)
    parser.add_argument('--ids', required=True, type=Path, help='token IDs, whitespace- or comma-separated')
    parser.add_argument('--rows', type=parse_rows, help='zero-based dump row range START:END (end exclusive)')
    parser.add_argument('--margins', type=parse_margins, default=(0.5, 1.0), metavar='M0,M1')
    parser.add_argument('--json', type=Path, dest='json_path', help='write the full result and disagreements as JSON')
    args = parser.parse_args(argv)
    try:
        input_paths = (args.reference, args.test, args.ids)
        if args.json_path:
            output_resolved = args.json_path.resolve()
            for source in input_paths:
                if output_resolved == source.resolve() or (args.json_path.exists() and source.exists()
                                                           and args.json_path.samefile(source)):
                    raise ValueError('--json must not overwrite an input file')
        result = compare(load(args.reference), load(args.test), load_ids(args.ids),
                         rows=args.rows, margins=args.margins)
        if args.json_path:
            args.json_path.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    except (OSError, ValueError) as exc:
        print(f'tf_compare: {exc}', file=sys.stderr)
        return 2

    kl = result['kl_nats']
    ppl = result['perplexity']
    delta = result['paired_delta_nats_per_token']
    print(f"rows: {result['rows_compared']} (positions {result['first_position']}..{result['last_position']})")
    print(f"same top-1: {result['same_top1']}/{result['rows_compared']} ({100 * result['same_top1_fraction']:.3f}%)")
    print(f"KL reference||test: mean {kl['mean']:.9g}, median {kl['median']:.9g}, "
          f"p99 {kl['p99']:.9g}, max {kl['max']:.9g} nats")
    print(f"perplexity: reference {ppl['reference']:.9g}, test {ppl['test']:.9g}")
    if delta['standard_error'] is None:
        print(f"paired test-reference NLL delta: {delta['mean']:.9g} nats/token "
              "(standard error unavailable for one row)")
    else:
        print(f"paired test-reference NLL delta: {delta['mean']:.9g} +/- "
              f"{delta['standard_error']:.9g} nats/token")
    for band in result['reference_margin_bands']:
        print(f"margin {band['range']}: {band['same_top1']}/{band['rows']} same")
    for item in result['disagreements']:
        print(f"  position {item['position']}: {item['reference_top1']} -> {item['test_top1']}, "
              f"reference margin {item['reference_margin']:.9g}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
