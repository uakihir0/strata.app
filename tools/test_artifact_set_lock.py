"""Artifact-set locking and interrupted-generation marker tests."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))

import artifact_set_lock
from artifact_set_lock import (
    ArtifactSetLockError,
    acquire_artifact_set_locks,
    clear_generation_pending,
    generation_needs_repack,
    mark_generation_pending,
    pending_marker,
)


class ArtifactSetLockTests(unittest.TestCase):
    def test_inspection_locks_reject_an_exclusive_writer(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "source"
            pack = root / "packs" / "model"
            source.mkdir(parents=True)
            pack.mkdir(parents=True)
            with acquire_artifact_set_locks([source], [(pack, False)]) as first:
                self.assertEqual(len(first.inherited_fds), 2)
                with acquire_artifact_set_locks([source], [(pack, False)]):
                    self.assertTrue(first.inherited_fds)
                with self.assertRaises(ArtifactSetLockError):
                    acquire_artifact_set_locks([source], [(pack, True)])

    def test_setup_locks_can_be_borrowed_by_its_packer_child(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "source"
            pack = root / "packs" / "model"
            source.mkdir(parents=True)
            pack.mkdir(parents=True)
            with acquire_artifact_set_locks(
                [source], [(pack, True)], exclusive_sources=True
            ) as parent:
                with acquire_artifact_set_locks(
                    [source], [(pack, True)], inherited=parent.inherited_environment()
                ) as child:
                    self.assertEqual(child.inherited_fds, parent.inherited_fds)
                with self.assertRaises(ArtifactSetLockError):
                    acquire_artifact_set_locks([source], [(pack, False)])

    def test_symlinked_source_alias_uses_the_same_lock(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models" / "source"
            source.mkdir(parents=True)
            alias = root / "models" / "source-alias"
            try:
                alias.symlink_to(source, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks are unavailable: {error}")
            with acquire_artifact_set_locks([source], []) as held:
                with self.assertRaises(ArtifactSetLockError):
                    acquire_artifact_set_locks([alias], [], exclusive_sources=True)

    def test_pending_marker_survives_until_explicit_completion(self):
        with tempfile.TemporaryDirectory() as temporary:
            pack = Path(temporary) / "pack"
            marker = mark_generation_pending(pack)
            self.assertTrue(marker.is_file())
            self.assertEqual(marker, pending_marker(pack))
            mark_generation_pending(pack)
            self.assertTrue(marker.is_file())
            clear_generation_pending(pack)
            self.assertFalse(marker.exists())

    def test_killed_writer_leaves_recoverable_generation_marker_and_releases_locks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "models"
            pack = root / "pack"
            ready = root / "writer-ready"
            source.mkdir()
            pack.mkdir()
            source_file = source / "model.gguf"
            source_file.touch()
            script = (
                "import sys, time\n"
                f"sys.path.insert(0, {str(ROOT / 'tools')!r})\n"
                "from pathlib import Path\n"
                "from artifact_set_lock import acquire_artifact_set_locks, mark_generation_pending\n"
                f"source = Path({str(source)!r})\n"
                f"pack = Path({str(pack)!r})\n"
                f"ready = Path({str(ready)!r})\n"
                "with acquire_artifact_set_locks([source], [(pack, True)], exclusive_sources=True):\n"
                "    mark_generation_pending(pack)\n"
                "    (pack / 'index.txt').write_text('partial generation', encoding='utf-8')\n"
                "    ready.touch()\n"
                "    while True:\n"
                "        time.sleep(1)\n"
            )
            child = subprocess.Popen(
                [sys.executable, "-c", script],
                cwd=ROOT,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                deadline = time.monotonic() + 10
                while not ready.exists():
                    if child.poll() is not None:
                        stdout, stderr = child.communicate()
                        self.fail(f"writer exited before publishing: {stdout}\n{stderr}")
                    if time.monotonic() >= deadline:
                        self.fail("writer did not reach the interrupted publication point")
                    time.sleep(0.01)
                child.kill()
                child.wait(timeout=5)
                child.communicate()

                self.assertTrue(pending_marker(pack).is_file())
                self.assertEqual(
                    (pack / "index.txt").read_text(encoding="utf-8"),
                    "partial generation",
                )
                with acquire_artifact_set_locks(
                    [source], [(pack, True)], exclusive_sources=True
                ):
                    with patch.object(artifact_set_lock, "done", return_value=True):
                        self.assertTrue(
                            generation_needs_repack(
                                pack, [source_file], False
                            )
                        )
            finally:
                if child.poll() is None:
                    child.kill()
                    child.wait(timeout=5)


if __name__ == "__main__":
    unittest.main()
