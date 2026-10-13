"""Synthetic inputs for make_release.py (docs/mac/08 section 11.3, steps 1 to 10, and section 11.9).

The real build steps need Xcode, CMake and the pinned llama.cpp, so these tests cover what the tool decides on its own:
the preconditions, the cache values it asserts, the deterministic zip, the excerpt and vendor checks, the text files,
the assembled release and its notarisation verdict. The release that `assemble` writes is checked by check_asset.py's
own checks (test_check_asset) with the same synthetic members.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.metal import check_asset  # noqa: E402
from tools.metal import make_release  # noqa: E402
from tools.metal import release_metadata  # noqa: E402
from tools.metal.test_check_asset import FakeTools, Release  # noqa: E402


def write(path: Path, text: str | bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(text, bytes):
        path.write_bytes(text)
    else:
        path.write_text(text, encoding="utf-8")
    return path


def arguments(**overrides) -> argparse.Namespace:
    values = {"sign": None, "notarize": False, "keychain_profile": None, "key": None, "key_id": None,
              "issuer": None, "notes": None, "out": "dist", "allow_dirty": False}
    values.update(overrides)
    return argparse.Namespace(**values)


class ZipWriter(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def members(self, folder: Path, mtime: int):
        paths = [write(folder / name, data) for name, data in (("a", b"alpha"), ("b", b"beta"))]
        for path in paths:
            os.utime(path, (mtime, mtime))
        return [("strata", paths[0], 0o755), ("BUILD.json", paths[1], 0o644)]

    def test_two_checkouts_at_different_paths_give_the_same_zip(self):
        first = self.folder / "one"
        second = self.folder / "two-with-other-times"
        make_release.write_zip(self.folder / "one.zip", 1700000000, self.members(first, 1_000_000_000))
        make_release.write_zip(self.folder / "two.zip", 1700000000, self.members(second, 1_800_000_000))

        self.assertEqual(
            (self.folder / "one.zip").read_bytes(), (self.folder / "two.zip").read_bytes()
        )

    def test_members_carry_the_source_date_modes_and_unix_system(self):
        make_release.write_zip(self.folder / "r.zip", 1700000000, self.members(self.folder, 0))
        with zipfile.ZipFile(self.folder / "r.zip") as archive:
            infos = archive.infolist()

        self.assertEqual(["strata", "BUILD.json"], [info.filename for info in infos])
        for info in infos:
            with self.subTest(member=info.filename):
                self.assertEqual((2023, 11, 14, 22, 13, 20), info.date_time)
                self.assertEqual(3, info.create_system)
                self.assertTrue(stat.S_ISREG(info.external_attr >> 16))
        self.assertEqual(0o755, stat.S_IMODE(infos[0].external_attr >> 16))
        self.assertEqual(0o644, stat.S_IMODE(infos[1].external_attr >> 16))

    def test_dates_before_1980_are_raised_to_the_zip_minimum(self):
        make_release.write_zip(self.folder / "old.zip", 0, self.members(self.folder, 0))
        with zipfile.ZipFile(self.folder / "old.zip") as archive:
            self.assertEqual((1980, 1, 1, 0, 0, 0), archive.infolist()[0].date_time)


class ExcerptsAndVendor(unittest.TestCase):
    def test_excerpt_lines_must_occur_in_order_with_comment_markers_allowed(self):
        source = ["// header", "// MIT licensed. Copyright (c) 2023 Example.", "int x;", "  Permission is granted", ]
        self.assertEqual([], make_release.excerpt_missing(["MIT licensed. Copyright (c) 2023 Example.", ""], source))
        self.assertEqual(
            ["MIT licensed. Copyright (c) 2023 Example."],
            make_release.excerpt_missing(["Permission is granted", "MIT licensed. Copyright (c) 2023 Example."], source),
        )
        self.assertEqual(["gone"], make_release.excerpt_missing(["gone"], source))

    def test_an_excerpt_names_a_source_that_holds_its_text(self):
        with tempfile.TemporaryDirectory() as temp:
            llama = Path(temp, "llama")
            write(llama / "vendor" / "stb" / "stb_image.h", "/*\nMIT License\nCopyright (c) 2017 Sean Barrett\n*/\n")
            good = write(Path(temp, "stb_image.txt"),
                         "source: vendor/stb/stb_image.h\nMIT License\nCopyright (c) 2017 Sean Barrett\n")
            self.assertEqual("MIT License\nCopyright (c) 2017 Sean Barrett", make_release.read_excerpt(good, llama))

            bad = write(Path(temp, "bad.txt"), "source: vendor/stb/stb_image.h\nAll rights reserved by someone\n")
            with self.assertRaisesRegex(make_release.ReleaseError, "is not in vendor/stb/stb_image.h at the pin"):
                make_release.read_excerpt(bad, llama)

            absent = write(Path(temp, "absent.txt"), "source: vendor/none.h\nx\n")
            with self.assertRaisesRegex(make_release.ReleaseError, "missing at the pin"):
                make_release.read_excerpt(absent, llama)

            unnamed = write(Path(temp, "unnamed.txt"), "MIT License\n")
            with self.assertRaisesRegex(make_release.ReleaseError, "must start with a 'source: <path>' line"):
                make_release.read_excerpt(unnamed, llama)

    def test_the_vendor_check_needs_each_linked_library_section_and_refuses_the_rest(self):
        titles = list(release_metadata.LICENSE_TITLES)
        names = ["_stbi_load", "_ma_decoder_init", "_subprocess_create", "_sha256_hex", "_main"]
        self.assertEqual(([], []), make_release.vendor_problems(names, titles))

        problems, _ = make_release.vendor_problems(names, [t for t in titles if t != "stb_image"])
        self.assertEqual(["strata-vision links stb_image; its section is missing from LICENSES.txt"], problems)

        problems, notes = make_release.vendor_problems(["_main"], titles)
        self.assertEqual([], problems)
        self.assertEqual(4, len(notes))
        self.assertIn("does not link stb_image", notes[0])

        problems, _ = make_release.vendor_problems(["_main", "__ZN8httplib6ServerE"], titles)
        self.assertEqual(
            ["strata-vision now links cpp-httplib; add its license to tools/metal/licenses/ and "
             "docs/mac/08 section 11.9"],
            problems,
        )
        for needle, library in make_release.VENDOR_FORBIDDEN:
            with self.subTest(needle=needle):
                problems, _ = make_release.vendor_problems([f"_{needle}_symbol"], titles)
                self.assertIn(f"strata-vision now links {library};", problems[0])


class LicenceText(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name, "checkout")
        self.llama = Path(self.temporary.name, "llama")
        write(self.root / "LICENSE", "MIT License for Strata\n")
        write(self.root / "src" / "metal" / "kernels" / "THIRD_PARTY", "Adapted kernels notice.\n")
        write(self.root / "third_party" / "ggml" / "LICENSE", "ggml MIT text.\n")
        write(self.llama / "LICENSE", "llama MIT text.\n")
        write(self.llama / "vendor" / "hash" / "sha256" / "LICENSE", "public domain sha256.\n")
        write(self.llama / "vendor" / "hash" / "rotate-bits" / "LICENSE.md", "rotate-bits MIT.\n")
        write(self.llama / "ggml" / "src" / "rope.metal", "// MIT licensed. Copyright (c) 2023 Example.\n")
        write(self.llama / "vendor" / "stb" / "stb_image.h", "MIT or public domain text.\n")
        write(self.llama / "vendor" / "miniaudio.h", "MIT No Attribution text.\n")
        write(self.llama / "vendor" / "subprocess.h", "The Unlicense text.\n")
        licenses = self.root / "tools" / "metal" / "licenses"
        write(licenses / "yarn-rope.txt", "source: ggml/src/rope.metal\nMIT licensed. Copyright (c) 2023 Example.\n")
        write(licenses / "stb_image.txt", "source: vendor/stb/stb_image.h\nMIT or public domain text.\n")
        write(licenses / "miniaudio.txt", "source: vendor/miniaudio.h\nMIT No Attribution text.\n")
        write(licenses / "subprocess.txt", "source: vendor/subprocess.h\nThe Unlicense text.\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_every_section_is_written_in_order_with_its_header(self):
        text = make_release.licenses_text(self.root, self.llama)

        headers = [line[3:-3] for line in text.splitlines() if line.startswith("== ") and line.endswith(" ==")]
        self.assertEqual(list(release_metadata.LICENSE_TITLES), headers)
        self.assertIn("== YaRN RoPE in ggml-metal ==\nMIT licensed. Copyright (c) 2023 Example.\n\n", text)
        self.assertNotIn("source:", text)

    def test_a_changed_excerpt_stops_the_licence_text(self):
        write(self.root / "tools" / "metal" / "licenses" / "stb_image.txt",
              "source: vendor/stb/stb_image.h\nMIT or public domain text, revised.\n")

        with self.assertRaisesRegex(make_release.ReleaseError, "licence excerpt stb_image.txt is not in"):
            make_release.licenses_text(self.root, self.llama)


class CacheValues(unittest.TestCase):
    def test_the_cache_must_hold_the_values_of_the_release_definitions(self):
        cache = "CMAKE_BUILD_TYPE:STRING=Release\nSTRATA_BUILD_TESTS:BOOL=OFF\nSTRATA_METAL_EMBED_SOURCES:BOOL=ON\n"

        problems = make_release.cache_problems(cache, make_release.ENGINE_CACHE, "engine")

        self.assertIn("engine cache has STRATA_METAL_EMBED_SOURCES='ON', want 'OFF'", problems)
        self.assertIn("engine cache has CMAKE_OSX_ARCHITECTURES=None, want 'arm64'", problems)
        self.assertEqual([], make_release.cache_problems(
            "CMAKE_BUILD_TYPE:STRING=Release\nSTRATA_BUILD_TESTS:BOOL=OFF\n"
            "STRATA_METAL_EMBED_SOURCES:BOOL=OFF\nCMAKE_OSX_ARCHITECTURES:STRING=arm64\n"
            "CMAKE_OSX_DEPLOYMENT_TARGET:STRING=27.0\n",
            make_release.ENGINE_CACHE,
            "engine",
        ))

    def test_the_vision_cache_turns_off_native_tuning_and_openmp(self):
        cache = "GGML_NATIVE:BOOL=ON\nGGML_OPENMP:BOOL=OFF\nGGML_METAL:BOOL=ON\nGGML_METAL_EMBED_LIBRARY:BOOL=ON\n"

        self.assertEqual(
            ["vision cache has GGML_NATIVE='ON', want 'OFF'"],
            make_release.cache_problems(cache, make_release.VISION_CACHE, "vision"),
        )


class Preconditions(unittest.TestCase):
    def check(self, *, platform="darwin", machine="arm64", translated=False, xcode=None, status="",
              root=make_release.ROOT, allow_dirty=False, llama_error=None):
        result = subprocess.CompletedProcess([], 0, status, "")
        with mock.patch.object(make_release.sys, "platform", platform), \
                mock.patch.object(make_release.platform, "machine", return_value=machine), \
                mock.patch.object(make_release.macos, "translated", return_value=translated), \
                mock.patch.object(make_release.macos, "xcode_problem", return_value=xcode), \
                mock.patch.object(make_release.subprocess, "run", return_value=result), \
                mock.patch.object(make_release.setup, "get_llama_cpp",
                                  side_effect=RuntimeError(llama_error) if llama_error else None,
                                  return_value=Path("/pinned/llama")):
            return make_release.preconditions(root, allow_dirty)

    def assertRefused(self, message, **kwargs):
        with self.assertRaisesRegex(make_release.ReleaseError, message):
            self.check(**kwargs)

    def test_each_host_and_tree_rule_refuses_with_its_message(self):
        self.assertRefused("a release is built on macOS", platform="linux")
        self.assertRefused("arm64 Mac", machine="x86_64")
        self.assertRefused("Rosetta", translated=True)
        self.assertRefused("Xcode is not installed", xcode="Xcode is not installed")
        self.assertRefused("the tree is not clean", status=" M setup.py\n")
        self.assertRefused("contains whitespace", root=Path("/tmp/strata checkout"))
        self.assertRefused("cannot fetch", llama_error="cannot fetch the pinned llama.cpp")

    def test_a_clean_tree_on_an_arm64_mac_gives_the_pinned_checkout(self):
        self.assertEqual(Path("/pinned/llama"), self.check())

    def test_dirty_trees_are_allowed_only_with_allow_dirty(self):
        self.assertEqual(Path("/pinned/llama"), self.check(status=" M setup.py\n", allow_dirty=True))


class SourceDateEpoch(unittest.TestCase):
    def test_the_environment_wins_then_the_commit_time(self):
        with mock.patch.dict(os.environ, {"SOURCE_DATE_EPOCH": "1700000000"}):
            self.assertEqual(1700000000, make_release.source_date_epoch(ROOT))

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(make_release.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 0, "1699999999\n", "")):
            self.assertEqual(1699999999, make_release.source_date_epoch(ROOT))

        with mock.patch.dict(os.environ, {}, clear=True), \
                mock.patch.object(make_release.subprocess, "run",
                                  return_value=subprocess.CompletedProcess([], 128, "", "fatal")):
            with self.assertRaisesRegex(make_release.ReleaseError, "SOURCE_DATE_EPOCH is not set"):
                make_release.source_date_epoch(ROOT)


class Assembly(unittest.TestCase):
    """`assemble` writes the same files as section 11.2, and check_asset.py accepts them."""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.release = Release(self.folder)
        self.stage = self.folder / "stage"
        self.symbols = self.folder / "stage-symbols"
        self.out = self.folder / "dist"
        self.out.mkdir()
        for name, data, _, _ in self.release.entries():
            write(self.stage / name, data)
        for name, data in (
            ("strata.dSYM/Contents/Info.plist", b"<plist/>\n"),
            ("strata.dSYM/Contents/Resources/DWARF/strata", b"dwarf\n"),
            ("strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml", b"relocs\n"),
            ("strata.metallibsym", b"symbols\n"),
        ):
            write(self.symbols / name, data)

    def tearDown(self):
        self.temporary.cleanup()

    def assemble(self, changes=None, signing="ad hoc"):
        return make_release.assemble(
            self.out, self.stage, self.symbols, source_date_epoch=1700000000, tag="v0.1.34-metal.1",
            meta=self.release.build_meta(), upstream="b" * 40, signing=signing, changes=changes,
        )

    def test_the_asset_and_symbols_zips_hold_the_documented_members(self):
        asset = self.assemble()

        with zipfile.ZipFile(asset) as archive:
            self.assertEqual([name for name, _ in release_metadata.ASSET_MEMBERS],
                             [info.filename for info in archive.infolist()])
        with zipfile.ZipFile(self.out / release_metadata.SYMBOLS_NAME) as archive:
            self.assertEqual(
                ["strata.dSYM/Contents/Info.plist", "strata.dSYM/Contents/Resources/DWARF/strata",
                 "strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml", "strata.metallibsym"],
                [info.filename for info in archive.infolist()],
            )

    def test_the_sums_file_names_both_zips_with_their_hashes(self):
        self.assemble()

        lines = (self.out / release_metadata.SUMS_NAME).read_text(encoding="ascii").splitlines()
        self.assertEqual(
            [f"{hashlib.sha256((self.out / name).read_bytes()).hexdigest()}  {name}"
             for name in (release_metadata.ASSET_NAME, release_metadata.SYMBOLS_NAME)],
            lines,
        )

    def test_the_notes_state_the_build_and_the_changes(self):
        self.assemble(changes="Fixed the kernel cache.\n", signing="Developer ID (TEAM), notarized")

        notes = (self.out / release_metadata.NOTES_NAME).read_text(encoding="utf-8")
        meta = self.release.build_meta()
        self.assertTrue(notes.startswith("# v0.1.34-metal.1\n"))
        self.assertIn(f"- commit {meta['commit']}, UPSTREAM_SYNC {'b' * 40}", notes)
        self.assertIn(f"- Xcode {meta['xcode']}, Metal toolchain {meta['metal_toolchain']}", notes)
        self.assertIn("- Signing: Developer ID (TEAM), notarized", notes)
        self.assertIn("- Requirement: macOS 27.0 or newer on M1 or newer", notes)
        self.assertIn(make_release.INSTALL_LINE, notes)
        self.assertIn(f"| strata | {hashlib.sha256(self.stage.joinpath('strata').read_bytes()).hexdigest()} |", notes)
        self.assertTrue(notes.rstrip().endswith("Fixed the kernel cache."))

    def test_a_missing_stage_member_stops_the_assembly(self):
        (self.stage / "strata.metallib").unlink()

        with self.assertRaisesRegex(make_release.ReleaseError, "the stage has no strata.metallib"):
            self.assemble()

    def test_check_asset_accepts_the_assembled_release(self):
        asset = self.assemble()

        problems = check_asset.check_asset(
            asset,
            symbols=self.out / release_metadata.SYMBOLS_NAME,
            run=True,
            root=self.release.checkout.root,
            runner=FakeTools(),
        )

        self.assertEqual([], problems)


class Notarisation(unittest.TestCase):
    def test_only_an_accepted_report_passes(self):
        def report(payload):
            return subprocess.CompletedProcess([], 0, json.dumps(payload), "")

        with mock.patch.object(make_release, "run", return_value=report({"status": "Accepted", "id": "x"})):
            make_release.notarize(Path("/tmp/a.zip"), ["--keychain-profile", "p"])

        with mock.patch.object(make_release, "run", return_value=report({"status": "Invalid", "id": "abc"})):
            with self.assertRaisesRegex(make_release.ReleaseError, "run: xcrun notarytool log abc"):
                make_release.notarize(Path("/tmp/a.zip"), ["--keychain-profile", "p"])

        with mock.patch.object(make_release, "run", return_value=subprocess.CompletedProcess([], 0, "no", "")):
            with self.assertRaisesRegex(make_release.ReleaseError, "printed no JSON"):
                make_release.notarize(Path("/tmp/a.zip"), ["--keychain-profile", "p"])

    def test_credentials_come_from_a_profile_or_an_api_key(self):
        self.assertEqual(["--keychain-profile", "p"],
                         make_release.notarization_credentials(arguments(keychain_profile="p")))
        self.assertEqual(
            ["--key", "k.p8", "--key-id", "i", "--issuer", "u"],
            make_release.notarization_credentials(arguments(key="k.p8", key_id="i", issuer="u")),
        )
        with self.assertRaisesRegex(make_release.ReleaseError, "needs --keychain-profile"):
            make_release.notarization_credentials(arguments())

    def test_the_signing_text_names_the_team(self):
        self.assertEqual("ad hoc", make_release.signing_text(None, False))
        identity = "Developer ID Application: Example Ltd (TEAM123)"
        self.assertEqual("Developer ID (TEAM123)", make_release.signing_text(identity, False))
        self.assertEqual("Developer ID (TEAM123), notarized", make_release.signing_text(identity, True))


class CommandLine(unittest.TestCase):
    def test_a_refused_preconditions_prints_the_problem_and_exits_one(self):
        with mock.patch.object(make_release.sys, "platform", "linux"), \
                mock.patch.object(make_release.sys, "stderr") as stderr:
            status = make_release.main([])

        self.assertEqual(1, status)
        printed = "".join(call.args[0] for call in stderr.write.call_args_list)
        self.assertIn("make_release: a release is built on macOS", printed)

    def test_notarising_without_signing_or_credentials_stops_before_the_build(self):
        with mock.patch.object(make_release, "preconditions", side_effect=AssertionError("built")):
            with self.assertRaisesRegex(make_release.ReleaseError, "needs --sign"):
                make_release.make_release(arguments(notarize=True))
            with self.assertRaisesRegex(make_release.ReleaseError, "needs --keychain-profile"):
                make_release.make_release(arguments(notarize=True, sign="Developer ID Application: X (T)"))

    def test_the_parser_defaults_to_an_unsigned_dist_build(self):
        args = make_release.parse_args([])

        self.assertEqual(("dist", None, False, False), (args.out, args.sign, args.notarize, args.allow_dirty))


if __name__ == "__main__":
    unittest.main()
