"""Synthetic routing tests for the resident-expert hit curve.

    python -m unittest tools.test_hit_curve
"""
from __future__ import annotations

import contextlib
import io
import os
import subprocess
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))

import hit_curve
from hit_curve import main
from make_profile import N_LAYER, write_profile


def write_record(output, layer: int, experts: tuple[int, ...], weight: float = 1.0) -> None:
    output.write(struct.pack("<ii", layer, len(experts)))
    output.write(struct.pack(f"<{len(experts)}i", *experts))
    output.write(struct.pack(f"<{len(experts)}f", *([weight] * len(experts))))


def write_group(path: Path, token_routes: list[tuple[int, ...]], weight: float = 1.0) -> None:
    with path.open("wb") as output:
        for _token, experts in enumerate(token_routes):
            for layer in range(N_LAYER):
                write_record(output, layer, experts, weight)


class HitCurve(unittest.TestCase):
    def test_exact_hit_miss_and_layer_capacity_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile.bin"
            trace = root / "trace.bin"
            ranked = [(layer, expert) for expert in range(4) for layer in range(N_LAYER)]
            write_profile(profile, ranked, 4)
            with trace.open("wb") as output:
                for layer in range(N_LAYER):
                    write_record(output, layer, (0, 1))
                    write_record(output, layer, (0,))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main([
                        "--profile", str(profile),
                        "--n-expert", "4",
                        "--resident", "48",
                        "--per-layer",
                        str(trace),
                    ]),
                    0,
                )
            self.assertEqual(
                stdout.getvalue(),
                "48 0.250000 0.666667 0.500000 24.000000 3\n",
            )

    def test_leave_one_group_out_builds_each_profile_from_the_other_trace(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.trace"
            second = root / "second.trace"
            write_group(first, [(0,)])
            write_group(second, [(1,), (1,), (0,)])

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main([
                        "--n-expert", "2",
                        "--resident", "48",
                        "--folds", "2",
                        "--per-layer",
                        str(first),
                        str(second),
                    ]),
                    0,
                )
            self.assertEqual(
                stdout.getvalue(),
                "48 0.500000 0.166667[0.000000,0.333333] "
                "0.166667[0.000000,0.333333] "
                "40.000000[32.000000,48.000000] 1[1,1]\n",
            )

    def test_leave_one_group_out_rejects_repeated_files_aliases_and_copies(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trace = root / "group.trace"
            write_group(trace, [(0,)])
            alias = root / "alias.trace"
            alias.symlink_to(trace)
            copied = root / "copied.trace"
            write_group(copied, [(0,)], weight=0.25)
            for other in (trace, alias, copied):
                stderr = io.StringIO()
                with self.subTest(other=other), contextlib.redirect_stderr(stderr), \
                        self.assertRaises(SystemExit):
                    main([
                        "--n-expert", "2",
                        "--resident", "48",
                        "--folds", "2",
                        str(trace),
                        str(other),
                    ])
                self.assertIn("distinct", stderr.getvalue())

    def test_leave_one_group_out_rejects_reordered_routing_records_and_ids(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = (
                ([(0,), (1,)], [(1,), (0,)]),
                ([(0, 1)], [(1, 0)]),
            )
            for index, (routes, reordered_routes) in enumerate(cases):
                first = root / f"first-{index}.trace"
                reordered = root / f"reordered-{index}.trace"
                write_group(first, routes)
                write_group(reordered, reordered_routes)
                stderr = io.StringIO()
                with self.subTest(index=index), contextlib.redirect_stderr(stderr), \
                        self.assertRaises(SystemExit):
                    main([
                        "--n-expert", "2",
                        "--resident", "48",
                        "--folds", "2",
                        str(first),
                        str(reordered),
                    ])
                self.assertIn("distinct routing data", stderr.getvalue())

    def test_leave_one_group_out_uses_the_same_snapshots_for_validation_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "first.trace"
            second = root / "second.trace"
            write_group(first, [(0,)])
            write_group(second, [(1,), (1,), (0,)])
            arguments = [
                "--n-expert", "2",
                "--resident", "48",
                "--folds", "2",
                str(first),
                str(second),
            ]

            expected = io.StringIO()
            with contextlib.redirect_stdout(expected):
                self.assertEqual(main(arguments), 0)

            original_rank = hit_curve.rank_from_traces

            def rewrite_source_after_validation(paths, n_expert):
                write_group(first, [(1,)])
                return original_rank(paths, n_expert)

            actual = io.StringIO()
            with patch("hit_curve.rank_from_traces", side_effect=rewrite_source_after_validation), \
                    contextlib.redirect_stdout(actual):
                self.assertEqual(main(arguments), 0)

            self.assertEqual(actual.getvalue(), expected.getvalue())

    @unittest.skipUnless(hasattr(os, "mkfifo"), "named pipes are POSIX-only")
    def test_leave_one_group_out_rejects_fifo_without_blocking(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            regular = root / "group.trace"
            fifo = root / "group.fifo"
            write_group(regular, [(0,)])
            os.mkfifo(fifo)
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).with_name("hit_curve.py")),
                    "--n-expert", "2",
                    "--resident", "48",
                    "--folds", "2",
                    str(regular),
                    str(fifo),
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn("regular file", result.stderr)

    def test_partial_profile_can_serve_requested_prefix(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "partial.bin"
            trace = root / "trace.bin"
            ranked = [(layer, 0) for layer in range(N_LAYER)]
            write_profile(profile, ranked, 2)
            with trace.open("wb") as output:
                for layer in range(N_LAYER):
                    write_record(output, layer, (0, 1))

            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                self.assertEqual(
                    main([
                        "--profile", str(profile),
                        "--n-expert", "2",
                        "--resident", "48",
                        str(trace),
                    ]),
                    0,
                )
            self.assertIn("48 0.500000 0.500000 0.000000 48.000000", stdout.getvalue())
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([
                    "--profile", str(profile),
                    "--n-expert", "2",
                    "--resident", "49",
                    str(trace),
                ])

    def test_rejects_incomplete_layer_coverage_and_bad_resident_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profile = root / "profile.bin"
            trace = root / "partial.trace"
            write_profile(
                profile,
                [(layer, expert) for expert in range(2) for layer in range(N_LAYER)],
                2,
            )
            with trace.open("wb") as output:
                write_record(output, 0, (0,))

            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([
                    "--profile", str(profile),
                    "--n-expert", "2",
                    "--resident", "48",
                    str(trace),
                ])
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main([
                    "--profile", str(profile),
                    "--n-expert", "2",
                    "--resident", "97",
                    str(trace),
                ])


if __name__ == "__main__":
    unittest.main()
