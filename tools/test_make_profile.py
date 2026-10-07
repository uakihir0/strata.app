"""Validation tests for STRP expert profiles and routing traces.

    python -m unittest tools.test_make_profile
"""
from __future__ import annotations

import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from make_profile import (
    MAGIC,
    N_EXPERT,
    N_LAYER,
    VERSION,
    iter_trace_records,
    read_profile,
    read_trace,
    write_profile,
)


class ProfileReader(unittest.TestCase):
    def test_profile_round_trip_and_rejects_trailing_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.bin"
            ranked = [(layer, expert) for expert in range(4) for layer in range(N_LAYER)]
            write_profile(path, ranked, 4)
            self.assertEqual(read_profile(path, 4), ranked)
            blob = bytearray(path.read_bytes())
            table_offset = 24 + 4 * len(ranked)
            blob[table_offset:table_offset + 4] = struct.pack("<i", 7)
            path.write_bytes(blob)
            self.assertEqual(read_profile(path, 4), ranked, "the engine ignores the compatibility table")
            partial = ranked[:96]
            write_profile(path, partial, 4)
            blob = bytearray(path.read_bytes())
            struct.pack_into("<I", blob, 16, len(partial) + 4)
            path.write_bytes(blob)
            self.assertEqual(read_profile(path, 4), partial, "a profile may reserve more slots than it ranks")
            write_profile(path, ranked, 4)
            path.write_bytes(path.read_bytes() + b"x")
            with self.assertRaisesRegex(SystemExit, "size or slot count"):
                read_profile(path, 4)

    def test_version_is_informational_and_out_of_range_pair_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "profile.bin"
            ranked = [(0, 3)]
            table = [[-1] * 4 for _ in range(N_LAYER)]
            blob = (
                MAGIC
                + struct.pack("<5I", 9, N_LAYER, 4, 1, 1)
                + struct.pack("<HH", *ranked[0])
                + b"".join(struct.pack("<4i", *row) for row in table)
            )
            path.write_bytes(blob)
            self.assertEqual(read_profile(path, 4), ranked, "the runtime treats the version as informational")
            path.write_bytes(blob[:24] + struct.pack("<HH", 0, 4) + blob[28:])
            with self.assertRaisesRegex(SystemExit, "out-of-range"):
                read_profile(path, 4)


class RoutingTrace(unittest.TestCase):
    @staticmethod
    def record(layer: int, experts: tuple[int, ...], weights: tuple[float, ...] | None = None) -> bytes:
        weights = weights or (1.0,) * len(experts)
        return (
            struct.pack("<ii", layer, len(experts))
            + struct.pack(f"<{len(experts)}i", *experts)
            + struct.pack(f"<{len(experts)}f", *weights)
        )

    def test_iter_and_frequency_reader(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.bin"
            path.write_bytes(self.record(2, (3, 1)) + self.record(2, (3,)))
            self.assertEqual(
                list(iter_trace_records(path, 4)),
                [(2, (3, 1)), (2, (3,))],
            )
            self.assertEqual(dict(read_trace(path, 4)), {(2, 3): 2, (2, 1): 1})

    def test_rejects_truncated_and_out_of_range_records(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "trace.bin"
            for blob, message in (
                (b"\x00" * 7, "truncated routing record header"),
                (struct.pack("<ii", 0, -1), "invalid route count"),
                (struct.pack("<ii", 0, 1) + struct.pack("<i", 0), "truncated routing ids/weights"),
                (self.record(N_LAYER, (0,)), "layer 48 outside"),
                (self.record(0, (4,)), "expert id outside"),
                (self.record(0, (1, 1)), "duplicate expert id"),
                (self.record(0, (1,), (float("nan"),)), "finite and non-negative"),
                (self.record(0, (1,), (-0.5,)), "finite and non-negative"),
                (self.record(0, tuple(range(65))), "invalid route count 65"),
            ):
                with self.subTest(message=message):
                    path.write_bytes(blob)
                    with self.assertRaisesRegex(SystemExit, message):
                        list(iter_trace_records(path, 100 if "route count" in message else 4))


if __name__ == "__main__":
    unittest.main()
