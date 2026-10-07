"""Fail-closed tests for complete, single-generation MTP packing inputs."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from artifact_set_lock import (  # noqa: E402
    LOCK_FDS_ENV,
    PENDING_ENV,
    acquire_artifact_set_locks,
    mark_generation_pending,
)
import mtp_pack  # noqa: E402


class MtpPackGenerationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.src = Path(self.temp.name) / "mtp"
        self.src.mkdir()
        self.canonical_shapes = mtp_pack.EXPECTED_SOURCE_SHAPES
        self.fixture_shapes = {
            name: ((1, 64) if name in mtp_pack.EXPERT_TENSORS else (2,))
            for name in mtp_pack.EXPECTED_TENSORS
        }
        self.shape_patch = mock.patch.object(
            mtp_pack, "EXPECTED_SOURCE_SHAPES", self.fixture_shapes
        )
        self.shape_patch.start()
        self.identity = {
            "requested_revision": "de4b8e4d43b917e7706784d8bb445c9af86a3540",
            "repo": (
                "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/"
                "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa/"
            ),
            "resolved_revision": "a" * 40,
        }
        tensor_dir = self.src / "tensors"
        tensor_dir.mkdir()
        self.rows = []
        self.records = []
        for index, name in enumerate(sorted(mtp_pack.EXPECTED_TENSORS)):
            shape = [1, 64] if name in mtp_pack.EXPERT_TENSORS else [2]
            size = math.prod(shape) * 2
            row = {
                "name": name,
                "shard": f"model-{index + 1}.safetensors",
                "dtype": "BF16",
                "shape": shape,
                "start": 12,
                "end": 11 + size,
                "bytes": size,
            }
            tensor = tensor_dir / (name + ".bin")
            tensor.write_bytes(b"\x00\x3f" * (size // 2))
            tensor.with_name(tensor.name + ".source.json").write_text(
                json.dumps(self.identity), encoding="utf-8"
            )
            self.rows.append(row)
            self.records.append({
                **row,
                "file": "tensors/" + tensor.name,
                "sha256": hashlib.sha256(tensor.read_bytes()).hexdigest(),
            })
        self._write_metadata()

    def tearDown(self):
        self.shape_patch.stop()
        self.temp.cleanup()

    def _write_metadata(self, rows=None, records=None, identity=None):
        identity = identity or self.identity
        rows = [dict(row) for row in self.rows] if rows is None else rows
        records = [dict(record) for record in self.records] if records is None else records
        (self.src / "mtp-inventory.json").write_text(
            json.dumps({**identity, "total_bytes": sum(r["bytes"] for r in rows), "tensors": rows}),
            encoding="utf-8",
        )
        (self.src / "mtp-manifest.json").write_text(json.dumps(records), encoding="utf-8")

    def test_complete_generation_is_accepted(self):
        self.assertEqual(
            [row["name"] for row in mtp_pack._validate_source_generation(self.src)],
            sorted(mtp_pack.EXPECTED_TENSORS),
        )

    def test_source_shape_catalog_matches_the_frozen_model_inventory(self):
        fixture = json.loads(
            (ROOT / "docs/mac/spikes/model-headers/mtp-inventory.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {row["name"]: tuple(row["shape"]) for row in fixture["tensors"]}
        self.assertEqual(self.canonical_shapes, expected)

    def test_same_size_but_transposed_expert_shape_is_refused(self):
        rows = [dict(row) for row in self.rows]
        records = [dict(record) for record in self.records]
        expert_name = "mtp.layers.0.mlp.experts.gate_up_proj"
        row_index = next(i for i, row in enumerate(rows) if row["name"] == expert_name)
        rows[row_index]["shape"] = [2, 32]
        records[row_index]["shape"] = [2, 32]
        self._write_metadata(rows=rows, records=records)
        with self.assertRaisesRegex(ValueError, "does not match required shape"):
            mtp_pack._validate_source_generation(self.src)

    def test_incomplete_manifest_is_refused(self):
        self._write_metadata(records=self.records[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete or stale"):
            mtp_pack._validate_source_generation(self.src)

    def test_incomplete_inventory_is_refused(self):
        self._write_metadata(rows=self.rows[:-1], records=self.records[:-1])
        with self.assertRaisesRegex(ValueError, "required tensor set"):
            mtp_pack._validate_source_generation(self.src)

    def test_non_bf16_source_is_refused(self):
        rows = [dict(row) for row in self.rows]
        records = [dict(record) for record in self.records]
        rows[0]["dtype"] = "F16"
        records[0]["dtype"] = "F16"
        self._write_metadata(rows=rows, records=records)
        with self.assertRaisesRegex(ValueError, "requires BF16"):
            mtp_pack._validate_source_generation(self.src)

    def test_inventory_manifest_and_sidecar_must_share_identity(self):
        tensor = self.src / "tensors" / (self.rows[0]["name"] + ".bin")
        tensor.with_name(tensor.name + ".source.json").write_text(
            json.dumps({**self.identity, "resolved_revision": "b" * 40}),
            encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "different source generation"):
            mtp_pack._validate_source_generation(self.src)

    def test_same_size_corruption_is_refused(self):
        tensor = self.src / "tensors" / (self.rows[0]["name"] + ".bin")
        tensor.write_bytes(bytes(reversed(tensor.read_bytes())))
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            mtp_pack._validate_source_generation(self.src)

    def test_selected_revision_must_match_inventory(self):
        with mock.patch.dict(os.environ, {"STRATA_MTP_REVISION": "main"}):
            with self.assertRaisesRegex(ValueError, "selected revision"):
                mtp_pack._validate_source_generation(self.src)

    def test_direct_packer_refuses_interrupted_runtime_generation(self):
        runtime = self.src / "rt"
        mark_generation_pending(runtime)
        stderr = io.StringIO()
        with mock.patch.dict(os.environ, {LOCK_FDS_ENV: ""}, clear=False), \
                mock.patch.object(sys, "argv", [
                    "mtp_pack.py", "--src", str(self.src), "--out", str(self.src / "draft.gguf")
                ]), contextlib.redirect_stderr(stderr):
            os.environ.pop(PENDING_ENV, None)
            result = mtp_pack.main()
        self.assertEqual(result, 2)
        self.assertIn("generation is pending", stderr.getvalue())
        self.assertFalse((self.src / "draft.gguf").exists())

    def test_pending_pack_requires_both_setup_locks_to_be_inherited(self):
        runtime = self.src / "rt"
        mark_generation_pending(runtime)
        runtime_only = acquire_artifact_set_locks([], [(runtime, True)])
        stderr = io.StringIO()
        try:
            with mock.patch.dict(os.environ, {
                LOCK_FDS_ENV: runtime_only.inherited_environment(),
                PENDING_ENV: "1",
            }), mock.patch.object(sys, "argv", [
                "mtp_pack.py", "--src", str(self.src), "--out", str(self.src / "draft.gguf")
            ]), contextlib.redirect_stderr(stderr):
                self.assertEqual(mtp_pack.main(), 2)
        finally:
            runtime_only.close()
        self.assertIn("generation is pending", stderr.getvalue())
        self.assertFalse((self.src / "draft.gguf").exists())

    def test_setup_borrowed_locks_may_complete_pending_generation(self):
        runtime = self.src / "rt"
        mark_generation_pending(runtime)
        locks = acquire_artifact_set_locks([], [(self.src, True), (runtime, True)])
        out = self.src / "draft.gguf"
        try:
            with mock.patch.dict(os.environ, {
                LOCK_FDS_ENV: locks.inherited_environment(),
                PENDING_ENV: "1",
            }), mock.patch.object(sys, "argv", [
                "mtp_pack.py", "--src", str(self.src), "--experts", "q2_0",
                "--out", str(out),
            ]), contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(mtp_pack.main(), 0)
            self.assertTrue(out.is_file())
            self.assertTrue((self.src / "draft.gguf.report.json").is_file())
        finally:
            locks.close()


if __name__ == "__main__":
    unittest.main()
