"""Measure expert-cache hit rates from Strata routing traces without a GPU.

    python tools/hit_curve.py TRACE ... [--profile data/expert-profile.bin]
        [--n-expert 512] [--resident 2048,4105,8192,12288,16384,20480,24576]
        [--folds K] [--per-layer]

With --folds K, each TRACE is one held-out group. Its profile is built from the
other K-1 traces, without a base profile, so the reported hit rate is
leave-one-group-out rather than training-set accuracy.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import stat
import statistics
import struct
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from make_profile import (
    N_EXPERT,
    N_LAYER,
    iter_trace_records,
    read_profile,
    read_trace,
)

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESIDENT = (2048, 4105, 8192, 12288, 16384, 20480, 24576)


def parse_resident(value: str, total: int) -> list[int]:
    try:
        residents = [int(item.strip()) for item in value.split(",")]
    except ValueError as exc:
        raise ValueError("--resident must be comma-separated integers") from exc
    if not residents or any(not 0 <= count <= total for count in residents):
        raise ValueError(f"--resident counts must be between 0 and {total}")
    if len(set(residents)) != len(residents):
        raise ValueError("--resident contains a duplicate count")
    return residents


def validate_profile_capacity(ranked: list[tuple[int, int]], residents: list[int]) -> None:
    requested = max(residents)
    if requested > len(ranked):
        raise ValueError(
            f"profile ranks {len(ranked)} pairs, but the largest requested resident count is {requested}"
        )


def _routing_digest(path: Path, n_expert: int) -> str:
    """Fingerprint the order-independent multiset of layer/ID records."""
    accumulator = 0
    record_count = 0
    modulus = 1 << 256
    for layer, experts in iter_trace_records(path, n_expert):
        expert_set = tuple(sorted(experts))
        record = struct.pack("<ii", layer, len(expert_set))
        record += struct.pack(f"<{len(expert_set)}i", *expert_set)
        record_hash = hashlib.sha256(b"strata-routing-record-v1\0" + record).digest()
        accumulator = (accumulator + int.from_bytes(record_hash, "big")) % modulus
        record_count += 1
    digest = hashlib.sha256(b"strata-routing-multiset-v1\0")
    digest.update(record_count.to_bytes(8, "little"))
    digest.update(accumulator.to_bytes(32, "big"))
    return digest.hexdigest()


def snapshot_unique_trace_groups(
    paths: list[Path], n_expert: int, snapshot_dir: Path
) -> list[Path]:
    """Copy each held-out trace once and use only those validated snapshots."""
    handles = []
    identities: set[tuple[int, int]] = set()
    snapshots = []
    try:
        for path in paths:
            descriptor = os.open(
                path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
            )
            try:
                handle = os.fdopen(descriptor, "rb")
            except BaseException:
                os.close(descriptor)
                raise
            handles.append((path, handle))
            info = os.fstat(handle.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ValueError(f"{path}: each --folds trace must be a regular file")
            identity = (info.st_dev, info.st_ino)
            if identity in identities:
                raise ValueError(f"{path}: each --folds trace must identify a distinct file")
            identities.add(identity)

        for index, (path, handle) in enumerate(handles):
            before = os.fstat(handle.fileno())
            snapshot = snapshot_dir / f"group-{index:04d}.trace"
            with snapshot.open("xb") as output:
                shutil.copyfileobj(handle, output, length=1024 * 1024)
                output.flush()
            after = os.fstat(handle.fileno())
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
                before.st_ctime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
                after.st_ctime_ns,
            )
            if (before_identity != after_identity or
                    snapshot.stat().st_size != before.st_size):
                raise ValueError(
                    f"{path}: trace changed while its held-out snapshot was being made"
                )
            snapshots.append(snapshot)
    finally:
        for _path, handle in handles:
            handle.close()

    digests: dict[str, Path] = {}
    for original, snapshot in zip(paths, snapshots):
        hexdigest = _routing_digest(snapshot, n_expert)
        if hexdigest in digests:
            raise ValueError(
                f"{original}: held-out traces must have distinct routing data "
                f"(matches {digests[hexdigest]})"
            )
        digests[hexdigest] = original
    return snapshots


def rank_from_traces(paths: list[Path], n_expert: int) -> list[tuple[int, int]]:
    frequency: dict[tuple[int, int], int] = defaultdict(int)
    for path in paths:
        for pair, count in read_trace(path, n_expert).items():
            frequency[pair] += count
    ranked = sorted(frequency, key=lambda pair: (-frequency[pair], pair))
    seen = set(ranked)
    ranked.extend(
        (layer, expert)
        for expert in range(n_expert)
        for layer in range(N_LAYER)
        if (layer, expert) not in seen
    )
    return ranked


def validate_complete_trace(path: Path, n_expert: int) -> int:
    counts = [0] * N_LAYER
    for layer, _experts in iter_trace_records(path, n_expert):
        counts[layer] += 1
    if not counts[0] or any(count != counts[0] for count in counts):
        raise ValueError(
            f"{path}: expected the same nonzero number of routing records for each of {N_LAYER} layers"
        )
    return counts[0]


def evaluate(
    paths: list[Path],
    ranked: list[tuple[int, int]],
    residents: list[int],
    n_expert: int,
) -> dict[int, dict[str, float]]:
    slot = {pair: index for index, pair in enumerate(ranked)}
    counters = {
        count: {"hits": 0, "routes": 0, "all_resident": 0, "layer_tokens": 0, "misses": 0}
        for count in residents
    }
    total_tokens = 0
    for path in paths:
        total_tokens += validate_complete_trace(path, n_expert)
        layer_records = [0] * N_LAYER
        for layer, experts in iter_trace_records(path, n_expert):
            layer_records[layer] += 1
            for count in residents:
                row = counters[count]
                resident_flags = [slot.get((layer, expert), len(ranked)) < count for expert in experts]
                row["routes"] += len(experts)
                row["hits"] += sum(resident_flags)
                row["all_resident"] += all(resident_flags)
                row["layer_tokens"] += 1
                row["misses"] += len(experts) - sum(resident_flags)
        if any(layer_records[layer] != layer_records[0] for layer in range(N_LAYER)):
            raise ValueError(f"{path}: layer record counts changed while reading the trace")

    results = {}
    for count, row in counters.items():
        resident_by_layer = Counter(layer for layer, _expert in ranked[:count])
        h = max(n_expert - resident_by_layer[layer] for layer in range(N_LAYER))
        results[count] = {
            "share": count / (N_LAYER * n_expert),
            "h_expert": row["hits"] / row["routes"],
            "h_layer": row["all_resident"] / row["layer_tokens"],
            "misses_per_token": row["misses"] / total_tokens,
            "H": float(h),
        }
    return results


def summarize(
    fold_results: list[dict[int, dict[str, float]]],
    residents: list[int],
    per_layer: bool,
) -> list[str]:
    lines = []
    metrics = ("h_expert", "h_layer", "misses_per_token")
    if per_layer:
        metrics += ("H",)
    for count in residents:
        share = fold_results[0][count]["share"]
        fields = [str(count), f"{share:.6f}"]
        for metric in metrics:
            values = [fold[count][metric] for fold in fold_results]
            formatter = (lambda value: f"{value:g}") if metric == "H" else (
                lambda value: f"{value:.6f}"
            )
            if len(values) == 1:
                fields.append(formatter(values[0]))
            else:
                fields.append(
                    f"{formatter(statistics.fmean(values))}"
                    f"[{formatter(min(values))},{formatter(max(values))}]"
                )
        lines.append(" ".join(fields))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("traces", nargs="+", help="routing trace files from --dump-routing")
    parser.add_argument(
        "--profile",
        default=str(ROOT / "data" / "expert-profile.bin"),
        help="ranking whose first R pairs are treated as resident",
    )
    parser.add_argument("--n-expert", type=int, default=N_EXPERT, help="experts per layer")
    parser.add_argument(
        "--resident",
        help="comma-separated resident pair counts (default values above the model size are omitted)",
    )
    parser.add_argument("--folds", type=int, help="leave-one-trace-group-out fold count")
    parser.add_argument("--per-layer", action="store_true", help="also report H, the largest nonresident layer")
    args = parser.parse_args(argv)

    if args.n_expert <= 0:
        parser.error("--n-expert must be positive")
    total = N_LAYER * args.n_expert
    default_residents = [count for count in DEFAULT_RESIDENT if count <= total]
    try:
        residents = parse_resident(args.resident, total) if args.resident else default_residents
    except ValueError as exc:
        parser.error(str(exc))
    if not residents:
        parser.error("no default --resident counts fit this model; provide --resident explicitly")

    paths = [Path(path) for path in args.traces]
    if args.folds is not None:
        if args.folds < 2 or args.folds != len(paths):
            parser.error("--folds K requires exactly K trace files and K >= 2")
        fold_results = []
        try:
            with tempfile.TemporaryDirectory(prefix="strata-hit-curve-") as directory:
                snapshots = snapshot_unique_trace_groups(
                    paths, args.n_expert, Path(directory)
                )
                for held_out, path in enumerate(snapshots):
                    training = snapshots[:held_out] + snapshots[held_out + 1:]
                    ranked = rank_from_traces(training, args.n_expert)
                    fold_results.append(evaluate([path], ranked, residents, args.n_expert))
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
    else:
        try:
            ranked = read_profile(args.profile, args.n_expert)
            validate_profile_capacity(ranked, residents)
        except OSError as exc:
            parser.error(f"cannot read profile {args.profile}: {exc}")
        except ValueError as exc:
            parser.error(str(exc))
        try:
            fold_results = [evaluate(paths, ranked, residents, args.n_expert)]
        except (OSError, ValueError) as exc:
            parser.error(str(exc))

    for line in summarize(fold_results, residents, args.per_layer):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
