"""Tests for MTP draft-vocabulary selection and atomic installation.

    .venv/bin/python -m unittest tools.test_mtp_draft_vocab
"""
from __future__ import annotations

import hashlib
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
import mtp_runtime_support as runtime_support  # noqa: E402
from artifact_set_lock import (  # noqa: E402
    ArtifactSetLockError,
    acquire_artifact_set_locks,
    pending_marker,
)


class SavedChoice(unittest.TestCase):
    def test_a_config_keeps_its_choice(self):
        with tempfile.TemporaryDirectory() as d:
            for choice in runtime_support.DRAFT_VOCABS:
                p = Path(d) / "strata-x.json"
                p.write_text(json.dumps({"args": [], "draft_vocab": choice}), encoding="utf-8")
                self.assertEqual(runtime_support.saved_draft_vocab(p), choice)

    def test_no_choice_without_a_valid_config(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(runtime_support.saved_draft_vocab(Path(d) / "missing.json"))
            for text in ("{}", "[1]", "not json", json.dumps({"draft_vocab": "klingon"})):
                p = Path(d) / "strata-y.json"
                p.write_text(text, encoding="utf-8")
                self.assertIsNone(runtime_support.saved_draft_vocab(p), text)


class Refresh(unittest.TestCase):
    def test_cyrillic_replaces_a_shipped_subset_and_stays(self):
        if not (ROOT / "data" / runtime_support.DRAFT_VOCABS["cyrillic"]).exists():
            self.skipTest("data/draft_vocab_cyrillic.bin is not in this checkout")
        want = hashlib.sha256((ROOT / "data" / runtime_support.DRAFT_VOCABS["cyrillic"]).read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as d:
            rt = Path(d)
            (rt / "draft_vocab.bin").write_bytes((ROOT / "data" / runtime_support.DRAFT_VOCABS["cjk"]).read_bytes())
            runtime_support.refresh_draft_vocab(rt, "cyrillic")
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)
            runtime_support.refresh_draft_vocab(rt, "cyrillic")
            self.assertEqual(hashlib.sha256((rt / "draft_vocab.bin").read_bytes()).hexdigest(), want)

    def test_invalid_saved_subset_is_replaced_with_the_selected_shipped_subset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            selected = data / "selected.bin"
            selected.write_bytes(b"\x01\x00\x00\x00\x02\x00\x00\x00")
            runtime = root / "mtp" / "rt"
            runtime.mkdir(parents=True)
            destination = runtime / "draft_vocab.bin"
            destination.write_bytes(b"\x01")
            with mock.patch.object(runtime_support, "ROOT", root), \
                    mock.patch.object(runtime_support, "DRAFT_VOCABS", {"cjk": "selected.bin"}), \
                    mock.patch.object(runtime_support, "OLD_DRAFT_VOCABS", set()):
                runtime_support.refresh_draft_vocab(runtime, "cjk")
            self.assertEqual(destination.read_bytes(), selected.read_bytes())

    def test_valid_custom_subset_is_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            (data / "selected.bin").write_bytes(b"\x01\x00\x00\x00")
            runtime = root / "mtp" / "rt"
            runtime.mkdir(parents=True)
            destination = runtime / "draft_vocab.bin"
            custom = b"\x03\x00\x00\x00\x04\x00\x00\x00"
            destination.write_bytes(custom)
            with mock.patch.object(runtime_support, "ROOT", root), \
                    mock.patch.object(runtime_support, "DRAFT_VOCABS", {"cjk": "selected.bin"}), \
                    mock.patch.object(runtime_support, "OLD_DRAFT_VOCABS", set()):
                runtime_support.refresh_draft_vocab(runtime, "cjk")
            self.assertEqual(destination.read_bytes(), custom)

    def test_vocab_refresh_obeys_a_live_runtime_reader_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            data = root / "data"
            data.mkdir()
            selected = data / "selected.bin"
            selected.write_bytes(b"\x02\x00\x00\x00")
            runtime = root / "mtp" / "rt"
            runtime.mkdir(parents=True)
            destination = runtime / "draft_vocab.bin"
            old = b"\x01\x00\x00\x00"
            destination.write_bytes(old)
            reader = acquire_artifact_set_locks([], [(runtime, False)])
            try:
                with mock.patch.object(runtime_support, "ROOT", root), \
                        mock.patch.object(runtime_support, "DRAFT_VOCABS", {"cjk": "selected.bin"}), \
                        mock.patch.object(
                            runtime_support, "OLD_DRAFT_VOCABS",
                            {hashlib.sha256(old).hexdigest()},
                        ):
                    with self.assertRaises(ArtifactSetLockError):
                        runtime_support.refresh_draft_vocab(runtime, "cjk")
                self.assertEqual(destination.read_bytes(), old)
                self.assertFalse(pending_marker(runtime).exists())
            finally:
                reader.close()


if __name__ == "__main__":
    unittest.main()
