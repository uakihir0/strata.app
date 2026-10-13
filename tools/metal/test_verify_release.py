"""Synthetic drafts for verify_release.py (docs/mac/08 section 11.8).

A draft is a real release folder made by make_release.assemble from synthetic members. gh, git, cp, codesign and the
rebuild are fakes; shasum runs for real, so the checksum rule is the tool's own. The install and the acceptance runs
are injected, or must fail closed when the checkout lacks their parts.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.metal import make_release  # noqa: E402
from tools.metal import release_metadata  # noqa: E402
from tools.metal import verify_release  # noqa: E402
from tools.metal.test_check_asset import Release  # noqa: E402

TAG = "v0.1.34-metal.1"
REPO = "owner/strata"


def completed(command, code=0, stdout="", stderr=""):
    return subprocess.CompletedProcess(command, code, stdout, stderr)


def make_dist(folder: Path, release: Release, metallib: bytes | None = None) -> Path:
    """Write a release folder (asset, symbols, SHA256SUMS, NOTES.md) for `release`, with an optional other metallib."""
    if metallib is not None:
        release.metallib = metallib
    stage = folder / "stage"
    symbols = folder / "stage-symbols"
    for name, data, _, _ in release.entries():
        stage.mkdir(parents=True, exist_ok=True)
        (stage / name).write_bytes(data)
    dsym = symbols / "strata.dSYM"
    for name, data in (
        ("Contents/Info.plist", b"<plist/>\n"),
        ("Contents/Resources/DWARF/strata", b"dwarf\n"),
        ("Contents/Resources/Relocations/aarch64/strata.yml", b"relocs\n"),
    ):
        (dsym / name).parent.mkdir(parents=True, exist_ok=True)
        (dsym / name).write_bytes(data)
    (symbols / "strata.metallibsym").write_bytes(b"symbols\n")
    out = folder / "dist"
    out.mkdir(parents=True, exist_ok=True)
    make_release.assemble(out, stage, symbols, source_date_epoch=1700000000, tag=TAG, meta=release.build_meta(),
                          upstream="b" * 40, signing="ad hoc", changes=None)
    return out


class FakeGh:
    """The runner. It serves the draft to `gh release download`, builds the rebuild with a fake make_release, and
    records every command."""

    def __init__(self, draft: Path, rebuild: Path | None = None, *, accept: bool = True, developer_id: bool = False):
        self.draft = draft
        self.rebuild = rebuild
        self.accept = accept
        self.developer_id = developer_id
        self.calls: list[list[str]] = []

    def __call__(self, command, **kwargs):
        cmd = [str(part) for part in command]
        self.calls.append(cmd)
        if cmd[:3] == ["gh", "release", "download"]:
            folder = Path(cmd[cmd.index("--dir") + 1])
            for item in self.draft.iterdir():
                if item.is_file():
                    shutil.copy2(item, folder / item.name)
            return completed(cmd)
        if cmd[0] == "shasum":
            return subprocess.run(cmd, **kwargs)
        if cmd[:4] == ["git", "worktree", "add", "--detach"]:
            worktree = Path(cmd[4])
            (worktree / "tools" / "metal").mkdir(parents=True)
            if self.accept:
                (worktree / "tools" / "metal" / "accept.py").write_text("", encoding="utf-8")
            return completed(cmd)
        if cmd[:4] == ["git", "worktree", "remove", "--force"]:
            shutil.rmtree(cmd[4], ignore_errors=True)
            return completed(cmd)
        if cmd[0] == "cp":
            return completed(cmd)
        if cmd[0] == sys.executable and cmd[1].endswith("make_release.py") and self.rebuild is not None:
            out = Path(cmd[cmd.index("--out") + 1])
            out.mkdir(parents=True)
            for item in self.rebuild.iterdir():
                if item.is_file():
                    shutil.copy2(item, out / item.name)
            return completed(cmd)
        if cmd[0] == sys.executable and (cmd[1].endswith("check_asset.py") or cmd[1].endswith("accept.py")):
            return completed(cmd)
        if cmd[0] == "codesign":
            if "-dv" in cmd:
                text = "Authority=Developer ID Application: Example (TEAM)\n" if self.developer_id else "Signature=adhoc\n"
                return completed(cmd, 0, "", text)
            return completed(cmd)
        return completed(cmd, 127, "", f"unexpected command {cmd[0]}")

    def called(self, *prefix: str) -> bool:
        return any(call[:len(prefix)] == list(prefix) for call in self.calls)

    def ran(self, script: str) -> bool:
        return any(len(call) > 1 and call[1].endswith(script) for call in self.calls)


class VerifyRelease(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.release = Release(self.folder / "release-source")
        self.draft = make_dist(self.folder / "draft", self.release)
        self.matching = make_dist(self.folder / "rebuild-source", Release(self.folder / "rebuild-release"))
        self.root = self.folder / "checkout"
        self.root.mkdir()
        self.xcode = mock.patch.object(verify_release.setup, "xcode_and_toolchain",
                                       return_value=(self.release.build_meta()["xcode"],
                                                     self.release.build_meta()["metal_toolchain"]))
        self.xcode.start()

    def tearDown(self):
        self.xcode.stop()
        self.temporary.cleanup()

    def verify(self, runner, **kwargs):
        options = {"installer": lambda url, folder: None, "root": self.root}
        options.update(kwargs)
        return verify_release.verify(TAG, REPO, runner=runner, **options)

    def test_a_matching_draft_verifies_and_its_rebuild_is_identical(self):
        runner = FakeGh(self.draft, rebuild=self.matching)

        self.assertEqual(("ok", None), self.verify(runner))
        self.assertTrue(runner.called("gh", "release", "download", TAG))
        self.assertTrue(runner.called("git", "worktree", "add", "--detach"))
        self.assertTrue(runner.called("git", "worktree", "remove", "--force"))

    def test_one_changed_byte_in_the_metallib_fails_the_checksums(self):
        changed = bytearray(self.release.metallib)
        changed[0] ^= 0x01
        tampered = make_dist(self.folder / "tampered", Release(self.folder / "tampered-source"), bytes(changed))
        (tampered / release_metadata.SUMS_NAME).write_bytes(
            (self.draft / release_metadata.SUMS_NAME).read_bytes()
        )
        runner = FakeGh(tampered)

        with self.assertRaisesRegex(verify_release.VerifyError, "SHA256SUMS does not match"):
            self.verify(runner)
        self.assertFalse(runner.called("git", "worktree", "add"))

    def test_a_rebuild_that_differs_in_a_member_fails(self):
        other = make_dist(self.folder / "other", Release(self.folder / "other-source"), b"another library\n")
        runner = FakeGh(self.draft, rebuild=other)

        with self.assertRaisesRegex(verify_release.VerifyError, "the rebuilt strata differs"):
            self.verify(runner)

    def test_a_different_toolchain_skips_the_rebuild_with_its_note(self):
        runner = FakeGh(self.draft, rebuild=self.matching)
        with mock.patch.object(verify_release.setup, "xcode_and_toolchain",
                               return_value=("27.1 (27B1)", "other.toolchain")):
            status, note = self.verify(runner)

        self.assertEqual("ok (rebuild skipped)", status)
        self.assertEqual(
            "rebuild skipped: the release was built with 27.0 (27A266a), "
            "com.apple.dt.toolchain.Metal.32023.921.5; this Mac has 27.1 (27B1), other.toolchain",
            note,
        )
        self.assertFalse(runner.ran("make_release.py"))

    def test_no_rebuild_skips_step_seven(self):
        runner = FakeGh(self.draft, rebuild=self.matching)

        status, note = self.verify(runner, rebuild=False)

        self.assertEqual(("ok (rebuild skipped)", "rebuild skipped: --no-rebuild"), (status, note))
        self.assertFalse(runner.ran("make_release.py"))

    def test_the_install_step_refuses_until_setup_has_the_engine_install(self):
        runner = FakeGh(self.draft, rebuild=self.matching)

        with self.assertRaisesRegex(verify_release.VerifyError, "step 5 needs setup.get_prebuilt_metal"):
            self.verify(runner, installer=None)
        self.assertTrue(runner.called("git", "worktree", "remove", "--force"))

    def test_the_acceptance_step_refuses_until_accept_py_exists(self):
        runner = FakeGh(self.draft, rebuild=self.matching, accept=False)

        with self.assertRaisesRegex(verify_release.VerifyError, "step 6 needs tools/metal/accept.py"):
            self.verify(runner)

    def test_a_developer_id_release_is_compared_without_its_signatures(self):
        runner = FakeGh(self.draft, rebuild=self.matching, developer_id=True)

        self.assertEqual(("ok", None), self.verify(runner))
        self.assertTrue(runner.called("codesign", "--remove-signature"))

    def test_only_a_metal_release_tag_is_accepted(self):
        runner = FakeGh(self.draft)
        with self.assertRaisesRegex(verify_release.VerifyError, "is not a Metal release tag"):
            verify_release.verify("v0.1.34-metal.1;rm", REPO, runner=runner, root=self.root)
        self.assertEqual([], runner.calls)

    def test_a_draft_without_its_checksums_is_refused(self):
        (self.draft / release_metadata.SUMS_NAME).unlink()
        runner = FakeGh(self.draft)

        with self.assertRaisesRegex(verify_release.VerifyError, "the draft v0.1.34-metal.1 has no SHA256SUMS"):
            self.verify(runner)

    def test_the_command_line_names_the_repository_when_it_is_missing(self):
        with mock.patch.dict("os.environ", {}, clear=True), \
                mock.patch.object(verify_release.release_metadata, "RELEASE_REPOSITORY", None), \
                mock.patch.object(verify_release.sys, "stderr") as stderr:
            status = verify_release.main([TAG])

        self.assertEqual(1, status)
        self.assertTrue(stderr.write.called)


if __name__ == "__main__":
    unittest.main()
