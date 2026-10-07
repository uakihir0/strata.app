"""Tests for locked, validated and atomic MTP draft-vocabulary installation."""
from __future__ import annotations

import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

import mtp_vocab  # noqa: E402
from artifact_set_lock import (  # noqa: E402
    ArtifactSetLockError,
    LOCK_FDS_ENV,
    acquire_artifact_set_locks,
    mark_generation_pending,
)


class MtpVocabularyTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.bin"
        self.runtime = self.root / "mtp" / "rt"

    def tearDown(self):
        self.tmp.cleanup()

    def write_ids(self, *ids: int) -> None:
        self.source.write_bytes(struct.pack("<" + "i" * len(ids), *ids))

    def test_installs_valid_subset_atomically_and_clears_marker(self):
        self.write_ids(0, 42, 248_319)
        self.runtime.mkdir(parents=True)
        destination = self.runtime / "draft_vocab.bin"
        destination.write_bytes(b"old")

        mtp_vocab.install(self.source, self.runtime, inherited="")

        self.assertEqual(destination.read_bytes(), self.source.read_bytes())
        self.assertFalse((self.runtime / ".strata-generation-pending").exists())
        self.assertFalse(list(self.runtime.glob(".draft_vocab.bin.tmp.*")))

    def test_rejects_invalid_ids_without_touching_runtime(self):
        for ids in ((-1,), (248_320,), (7, 7)):
            with self.subTest(ids=ids):
                self.write_ids(*ids)
                self.runtime.mkdir(parents=True, exist_ok=True)
                destination = self.runtime / "draft_vocab.bin"
                destination.write_bytes(b"keep")
                with self.assertRaises(ValueError):
                    mtp_vocab.install(self.source, self.runtime, inherited="")
                self.assertEqual(destination.read_bytes(), b"keep")
                self.assertFalse(
                    (self.runtime / ".strata-generation-pending").exists()
                )

    def test_refuses_runtime_reader_lock(self):
        self.write_ids(3)
        reader = acquire_artifact_set_locks([], [(self.runtime, False)])
        try:
            with self.assertRaisesRegex(RuntimeError, "artifact set is in use"):
                mtp_vocab.install(self.source, self.runtime, inherited="")
        finally:
            reader.close()
        self.assertFalse(self.runtime.exists())

    def test_keeps_parent_generation_marker_when_nested(self):
        self.write_ids(3, 4)
        self.runtime.mkdir(parents=True)
        locks = acquire_artifact_set_locks([], [(self.runtime, True)])
        marker = mark_generation_pending(self.runtime)
        try:
            mtp_vocab.install(
                self.source,
                self.runtime,
                inherited=locks.inherited_environment(),
            )
            self.assertTrue(marker.is_file())
            self.assertEqual(
                (self.runtime / "draft_vocab.bin").read_bytes(),
                self.source.read_bytes(),
            )
        finally:
            locks.close()

    def test_standalone_writer_cannot_clear_interrupted_runtime_marker(self):
        self.write_ids(3, 4)
        self.runtime.mkdir(parents=True)
        destination = self.runtime / "draft_vocab.bin"
        destination.write_bytes(b"old")
        marker = mark_generation_pending(self.runtime)

        with self.assertRaisesRegex(ArtifactSetLockError, "interrupted MTP"):
            mtp_vocab.install(self.source, self.runtime, inherited="")

        self.assertEqual(destination.read_bytes(), b"old")
        self.assertTrue(marker.is_file())

    def test_cli_reports_bad_input(self):
        self.source.write_bytes(b"\x01")
        stderr = io.StringIO()
        with mock.patch.object(sys, "argv", [
            "mtp_vocab.py", "--source", str(self.source),
            "--runtime", str(self.runtime),
        ]), contextlib.redirect_stderr(stderr):
            self.assertEqual(mtp_vocab.main(), 2)
        self.assertIn("little-endian int32", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
