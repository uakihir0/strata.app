"""Synthetic release assets for check_asset.py (docs/mac/08 section 11.7, checks 1 to 12).

The assets are zip files of small text stand-ins, a fake checkout holds the manifest and METAL_REV, and the macOS
tools are a fake runner. The `--run` check can also execute real shell scripts in place of the engine binaries.
"""
from __future__ import annotations

import hashlib
import json
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
from tools.metal import release_metadata  # noqa: E402

KERNELS = ("strata_attn_merge", "strata_selftest_add")
VTOOL_OK = (
    "Load command 11\n      cmd LC_BUILD_VERSION\n  cmdsize 32\n platform MACOS\n    minos 27.0\n"
    "      sdk 27.0\n   ntools 1\n"
)
UUID = "A2B81753-3871-3FB0-A7DC-FA80F93CB6C5"


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_zip(path: Path, entries) -> Path:
    """Entries are (name, bytes, mode, create_system). Dates are fixed, so the archive is deterministic."""
    with zipfile.ZipFile(path, "w") as archive:
        for name, data, mode, system in entries:
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = system
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, data)
    return path


class Checkout:
    """A fake checkout: the version, METAL_REV and the kernel manifest that check 8 reads."""

    def __init__(self, folder: Path, kernels=KERNELS):
        self.root = folder / "checkout"
        (self.root / "src" / "metal" / "kernels").mkdir(parents=True)
        (self.root / "CMakeLists.txt").write_text("project(strata VERSION 0.1.34 LANGUAGES CXX)\n", encoding="utf-8")
        (self.root / "METAL_REV").write_text("1\n", encoding="ascii")
        manifest = "# every kernel\n" + "".join(f"{name} safe 256 1 1\n" for name in kernels)
        (self.root / "src" / "metal" / "kernels" / "manifest.txt").write_text(manifest, encoding="utf-8")


class Release:
    """The members of a valid release, which a test can change before it writes the zip."""

    def __init__(self, folder: Path, kernels=KERNELS):
        self.folder = folder
        self.checkout = Checkout(folder, kernels)
        self.metallib = b"strata kernel library\n"
        self.commit = "a" * 40
        self.build_overrides: dict = {}

    def build_meta(self) -> dict:
        meta = {
            "version": "0.1.34",
            "metal_rev": 1,
            "backend": "metal",
            "arch": "arm64",
            "macos_min": "27.0",
            "gpu_families": ["apple7", "metal4"],
            "metal_std": "metal4.0",
            "source": "prebuilt",
            "lib_dirs": [],
            "vision": "gpu",
            "src": "0123456789abcdef",
            "vision_src": "fedcba9876543210",
            "xcode": "27.0 (27A266a)",
            "metal_toolchain": "com.apple.dt.toolchain.Metal.32023.921.5",
            "commit": self.commit,
            "metallib_sha256": sha(self.metallib),
        }
        meta.update(self.build_overrides)
        return meta

    def entries(self):
        engine = (
            b"ENGINE\0engine=0.1.34\n" + b"metallib_sha256=" + sha(self.metallib).encode() + b"\n" + b"\0tail"
        )
        vision = b"VISION\0/tmp/perf-metal-%d.gputrace\0"
        build = (json.dumps(self.build_meta(), indent=1) + "\n").encode("utf-8")
        licenses = "".join(f"== {title} ==\ntext\n\n" for title in release_metadata.LICENSE_TITLES).encode()
        return [
            ("strata", engine, 0o755, 3),
            ("strata.metallib", self.metallib, 0o644, 3),
            ("strata-vision", vision, 0o755, 3),
            ("BUILD.json", build, 0o644, 3),
            ("LICENSES.txt", licenses, 0o644, 3),
        ]

    def asset(self, entries=None, name="strata-macos-arm64.zip") -> Path:
        return write_zip(self.folder / name, self.entries() if entries is None else entries)

    def symbols(self, entries=None) -> Path:
        if entries is None:
            entries = [
                ("strata.dSYM/Contents/Info.plist", b"<plist/>\n", 0o644, 3),
                ("strata.dSYM/Contents/Resources/DWARF/strata", b"dwarf\n", 0o644, 3),
                ("strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml", b"relocs\n", 0o644, 3),
                ("strata.metallibsym", b"symbols\n", 0o644, 3),
            ]
        return write_zip(self.folder / "strata-macos-arm64-symbols.zip", entries)


class FakeTools:
    """Answers the macOS tools of check 2 to 8 and 11 and 12. Each attribute can be changed by a test."""

    def __init__(self, kernels=KERNELS, *, execute=False):
        self.kernels = list(kernels)
        self.execute = execute
        self.lipo = "arm64\n"
        self.vtool = VTOOL_OK
        self.otool_libraries = [
            "\t/System/Library/Frameworks/Metal.framework/Versions/A/Metal (compatibility version 1.0.0)\n",
            "\t/usr/lib/libSystem.B.dylib (compatibility version 1.0.0)\n",
        ]
        self.otool_commands = "Load command 1\n      cmd LC_SEGMENT_64\n"
        self.verify_code = 0
        self.signature = "flags=0x20002(adhoc,linker-signed)\nSignature=adhoc\n"
        self.identifier = None
        self.entitlements = ""
        self.nm_output = None
        self.dsymutil_changes = False
        self.uuid = UUID
        self.dsym_uuid = None
        self.calls: list[list[str]] = []

    def __call__(self, argv, **kwargs):
        argv = list(argv)
        self.calls.append(argv)
        name = Path(argv[0]).name
        if name in ("strata", "strata-vision"):
            if self.execute:
                return subprocess.run(argv, **kwargs)
            return self.engine_answer(name, argv)
        if name == "lipo":
            return self.done(argv, self.lipo)
        if name == "vtool":
            return self.done(argv, self.vtool)
        if name == "otool":
            if argv[1] == "-L":
                return self.done(argv, "strata:\n" + "".join(self.otool_libraries))
            return self.done(argv, self.otool_commands)
        if name == "codesign":
            return self.codesign(argv)
        if name == "xcrun":
            return self.xcrun(argv)
        return subprocess.CompletedProcess(argv, 127, "", f"unexpected tool {name}")

    @staticmethod
    def done(argv, stdout="", code=0, stderr=""):
        return subprocess.CompletedProcess(argv, code, stdout, stderr)

    def engine_answer(self, name, argv):
        if name == "strata":
            return self.done(argv, "strata 0.1.34 metal 1\n")
        return self.done(argv, "", 2, "usage: strata-vision --mmproj <mmproj.gguf>\n")

    def codesign(self, argv):
        path = Path(argv[-1])
        if "--entitlements" in argv:
            return self.done(argv, self.entitlements)
        if "-v" in argv:
            return self.done(argv, "", self.verify_code, "" if self.verify_code == 0 else "invalid signature\n")
        identifier = self.identifier if self.identifier is not None else path.name
        text = f"Executable={path}\nIdentifier={identifier}\nFormat=Mach-O thin (arm64)\n{self.signature}"
        return self.done(argv, "", 0, text)

    def xcrun(self, argv):
        tool = argv[1]
        if tool == "metal-nm":
            names = self.nm_output if self.nm_output is not None else "\n".join(self.kernels) + "\n"
            return self.done(argv, names)
        if tool == "metal-dsymutil":
            target = Path(argv[-1])
            if self.dsymutil_changes:
                target.write_bytes(target.read_bytes() + b"sources")
            return self.done(argv)
        if tool == "dwarfdump":
            path = Path(argv[-1])
            uuid = self.uuid
            if path.name == "strata" and "strata.dSYM" in path.parts:
                uuid = self.dsym_uuid or self.uuid
            return self.done(argv, f"UUID: {uuid} (arm64) {path}\n")
        return self.done(argv, "", 1, f"unknown xcrun tool {tool}")


class CheckAsset(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.folder = Path(self.temporary.name)
        self.release = Release(self.folder)

    def tearDown(self):
        self.temporary.cleanup()

    def check(self, asset=None, *, tools=None, symbols=None, run=False, allow_dirty=False):
        tools = tools or FakeTools()
        return check_asset.check_asset(
            asset or self.release.asset(),
            symbols=symbols,
            run=run,
            allow_dirty=allow_dirty,
            root=self.release.checkout.root,
            runner=tools,
        )

    def assertProblem(self, problems, text):
        self.assertTrue(any(text in line for line in problems), problems)

    def test_a_valid_asset_passes_every_check(self):
        self.assertEqual([], self.check())

    def test_a_valid_asset_with_symbols_and_run_passes(self):
        problems = self.check(symbols=self.release.symbols(), run=True)

        self.assertEqual([], problems)

    def test_members_are_exact_in_order_and_at_the_top_level(self):
        entries = self.release.entries()
        reordered = list(reversed(entries))
        problems = self.check(self.release.asset(reordered, "reordered.zip"))
        self.assertProblem(problems, "members are ['LICENSES.txt', 'BUILD.json'")

        extra = self.release.entries() + [("notes.txt", b"x", 0o644, 3)]
        self.assertProblem(self.check(self.release.asset(extra, "extra.zip")), "members are")

        folder = [(("sub/strata" if name == "strata" else name), data, mode, system)
                  for name, data, mode, system in self.release.entries()]
        self.assertProblem(self.check(self.release.asset(folder, "folder.zip")), "sub/strata: is a folder")

    def test_each_member_has_its_mode_and_unix_system(self):
        entries = [(name, data, 0o644 if name == "strata" else mode, system)
                   for name, data, mode, system in self.release.entries()]
        self.assertProblem(self.check(self.release.asset(entries, "mode.zip")), "strata: mode is 0644, want 0755")

        entries = [(name, data, mode, 0 if name == "BUILD.json" else system)
                   for name, data, mode, system in self.release.entries()]
        self.assertProblem(
            self.check(self.release.asset(entries, "system.zip")),
            "BUILD.json: create_system is 0, want 3",
        )

    def test_architecture_and_platform_are_arm64_macos_27(self):
        tools = FakeTools()
        tools.lipo = "x86_64\n"
        self.assertProblem(self.check(tools=tools), "strata: lipo -archs gives x86_64, want arm64")

        tools = FakeTools()
        tools.vtool = VTOOL_OK.replace("minos 27.0", "minos 26.0")
        self.assertProblem(self.check(tools=tools), "strata-vision: vtool shows platform ['MACOS'] and minos ['26.0']")

        tools = FakeTools()
        tools.vtool = VTOOL_OK.replace("platform MACOS", "platform IOS")
        self.assertProblem(self.check(tools=tools), "strata: vtool shows platform ['IOS']")

    def test_linked_libraries_are_system_libraries_and_there_is_no_rpath(self):
        tools = FakeTools()
        tools.otool_libraries.append("\t/opt/homebrew/lib/libggml.dylib (compatibility version 0.0.0)\n")
        self.assertProblem(self.check(tools=tools), "strata: loads /opt/homebrew/lib/libggml.dylib")

        tools = FakeTools()
        tools.otool_libraries.append("\t@rpath/libmtmd.dylib (compatibility version 0.0.0)\n")
        self.assertProblem(self.check(tools=tools), "strata-vision: loads @rpath/libmtmd.dylib: no load path may start with @")

        tools = FakeTools()
        tools.otool_commands += "      cmd LC_RPATH\n"
        self.assertProblem(self.check(tools=tools), "strata: has an LC_RPATH load command")

    def test_signature_is_ad_hoc_or_developer_id_with_the_runtime_and_no_entitlements(self):
        tools = FakeTools()
        tools.verify_code = 1
        self.assertProblem(self.check(tools=tools), "strata-vision: codesign -v --strict fails")

        tools = FakeTools()
        tools.identifier = "strata.s"
        self.assertProblem(self.check(tools=tools), "strata: Identifier is strata.s, want strata")

        tools = FakeTools()
        tools.signature = "Authority=Apple Development: someone\n"
        self.assertProblem(self.check(tools=tools), "strata: signature is neither ad hoc nor Developer ID")

        tools = FakeTools()
        tools.signature = (
            "flags=0x10000(runtime)\nAuthority=Developer ID Application: Example (TEAM)\nSignature=\n"
        )
        self.assertEqual([], self.check(tools=tools))

        tools = FakeTools()
        tools.signature = "flags=0x10000(runtime)\nAuthority=Developer ID Application: Example (TEAM)\n"
        tools.entitlements = '<?xml version="1.0"?><plist><dict><key>com.apple.security.cs.allow-jit</key></dict></plist>'
        problems = self.check(tools=tools)
        self.assertProblem(problems, "strata: has entitlements")

    def test_forbidden_paths_are_refused_except_the_one_gputrace_string(self):
        self.release.metallib = b"library /tmp/build/x\n"
        problems = self.check()
        self.assertProblem(problems, "strata.metallib: contains the path '/tmp/'")

        entries = self.release.entries()
        entries[0] = ("strata", entries[0][1] + b"/Users/someone/build\0", 0o755, 3)
        self.assertProblem(self.check(self.release.asset(entries, "users.zip")), "strata: contains the path '/Users/'")

        self.release.metallib = b"strata kernel library\n"
        entries = self.release.entries()
        entries[2] = ("strata-vision", b"VISION /tmp/perf-metal-%d.gputrace /tmp/other\0", 0o755, 3)
        problems = self.check(self.release.asset(entries, "vision.zip"))
        self.assertEqual(["check_asset: strata-vision: contains the path '/tmp/'"], problems)

    def test_the_engine_banner_and_metallib_hash_must_match(self):
        entries = self.release.entries()
        entries[0] = ("strata", entries[0][1].replace(b"engine=0.1.34", b"engine=0.1.33"), 0o755, 3)
        self.assertProblem(self.check(self.release.asset(entries, "old.zip")), "engine=0.1.33 but the source version is 0.1.34")

        entries = self.release.entries()
        entries[0] = ("strata", entries[0][1] + b"engine=0.1.34\n", 0o755, 3)
        self.assertProblem(self.check(self.release.asset(entries, "twice.zip")), "contains 2 engine=<version> lines")

        entries = self.release.entries()
        entries[0] = ("strata", entries[0][1].replace(sha(self.release.metallib).encode(), b"0" * 64), 0o755, 3)
        self.assertProblem(self.check(self.release.asset(entries, "hash.zip")), "strata.metallib hashes to")

        self.release.build_overrides = {"metallib_sha256": "1" * 64}
        self.assertProblem(self.check(), "BUILD.json: metallib_sha256 does not match")

    def test_build_json_has_the_documented_keys_and_values(self):
        self.release.build_overrides = {"lib_dirs": ["/opt"]}
        self.assertProblem(self.check(), "BUILD.json: lib_dirs is ['/opt'], want []")

        self.release.build_overrides = {}
        entries = self.release.entries()
        meta = self.release.build_meta()
        del meta["lib_dirs"]
        meta["extra"] = 1
        entries[3] = ("BUILD.json", (json.dumps(meta, indent=1) + "\n").encode(), 0o644, 3)
        problems = self.check(self.release.asset(entries, "keys.zip"))
        self.assertProblem(problems, "BUILD.json: lacks keys lib_dirs")
        self.assertProblem(problems, "BUILD.json: has keys extra")

        self.release.build_overrides = {"metal_rev": 2}
        self.assertProblem(self.check(), "BUILD.json: metal_rev is 2, the checkout's METAL_REV is 1")

        self.release.build_overrides = {"src": "xyz"}
        self.assertProblem(self.check(), "BUILD.json: src must be 16 hex digits")

        self.release.build_overrides = {}
        entries = self.release.entries()
        entries[3] = ("BUILD.json", json.dumps(self.release.build_meta(), indent=1).encode(), 0o644, 3)
        self.assertProblem(self.check(self.release.asset(entries, "newline.zip")), "BUILD.json: does not end with a newline")

    def test_a_dirty_commit_needs_allow_dirty_and_a_clean_one_must_be_forty_hex(self):
        self.release.commit = "a" * 40 + "-dirty"
        self.assertProblem(self.check(), "BUILD.json: commit is -dirty")
        self.assertEqual([], self.check(allow_dirty=True))

        self.release.commit = ""
        self.assertProblem(self.check(), "BUILD.json: commit must be 40 hex digits")

    def test_the_licence_headers_are_the_documented_ones_in_order(self):
        titles = list(release_metadata.LICENSE_TITLES)
        text = "".join(f"== {title} ==\n" for title in titles[:-1])
        entries = self.release.entries()
        entries[4] = ("LICENSES.txt", text.encode(), 0o644, 3)
        self.assertProblem(self.check(self.release.asset(entries, "missing.zip")), "LICENSES.txt: section headers are")

        text = "".join(f"== {title} ==\n" for title in reversed(titles))
        entries[4] = ("LICENSES.txt", text.encode(), 0o644, 3)
        self.assertProblem(self.check(self.release.asset(entries, "order.zip")), "in this order")

        text = "".join(f"== {title} ==\n" for title in titles) + "== Extra ==\n"
        entries[4] = ("LICENSES.txt", text.encode(), 0o644, 3)
        self.assertProblem(self.check(self.release.asset(entries, "extra.zip")), "'Extra'")

    def test_the_kernels_are_the_manifest_and_no_source_is_left(self):
        tools = FakeTools(kernels=KERNELS[:1])
        problems = self.check(tools=tools)
        self.assertProblem(problems, f"strata.metallib: defined functions differ from the manifest: missing ['{KERNELS[1]}']")

        tools = FakeTools(kernels=KERNELS + ("strata_extra",))
        self.assertProblem(self.check(tools=tools), "extra ['strata_extra']")

        tools = FakeTools()
        tools.dsymutil_changes = True
        self.assertProblem(self.check(tools=tools), "strata.metallib: still holds kernel sources")

    def test_the_symbols_zip_must_match_the_engine(self):
        entries = [
            ("strata.dSYM/Contents/Info.plist", b"p", 0o644, 3),
            ("strata.dSYM/Contents/Resources/DWARF/strata", b"d", 0o644, 3),
            ("strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml", b"r", 0o644, 3),
            ("strata.metallibsym", b"s", 0o600, 3),
        ]
        problems = self.check(symbols=self.release.symbols(entries))
        self.assertProblem(problems, "strata.metallibsym: mode is 0600, want 0644")

        tools = FakeTools()
        tools.dsym_uuid = "00000000-0000-0000-0000-000000000000"
        self.assertProblem(self.check(symbols=self.release.symbols(), tools=tools), "strata.dSYM: UUID is")

        symbols = self.release.symbols([
            ("strata.dSYM/Contents/Info.plist", b"p", 0o644, 3),
            ("strata.dSYM/Contents/Resources/DWARF/strata", b"d", 0o644, 3),
            ("strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml", b"r", 0o644, 3),
            ("strata.metallibsym", b"", 0o644, 3),
        ])
        self.assertProblem(self.check(symbols=symbols), "strata.metallibsym: is empty")

    def test_run_accepts_the_documented_banner_and_usage_from_real_executables(self):
        identity = (
            b"# engine=0.1.34\n# metallib_sha256=" + sha(self.release.metallib).encode() + b"\n"
        )
        entries = self.release.entries()
        entries[0] = ("strata", b"#!/bin/sh\n" + identity + b"echo 'strata 0.1.34 metal 1'\n", 0o755, 3)
        entries[2] = ("strata-vision", b"#!/bin/sh\necho 'usage: strata-vision --mmproj x' >&2\nexit 2\n", 0o755, 3)

        problems = self.check(self.release.asset(entries, "runs.zip"), tools=FakeTools(execute=True), run=True)

        self.assertEqual([], problems)

    def test_run_refuses_a_wrong_banner_and_a_vision_helper_that_exits_zero(self):
        entries = self.release.entries()
        entries[0] = ("strata", b"#!/bin/sh\necho 'strata 0.1.33 metal 1'\n", 0o755, 3)
        entries[2] = ("strata-vision", b"#!/bin/sh\nexit 0\n", 0o755, 3)
        problems = self.check(self.release.asset(entries, "scripts.zip"), tools=FakeTools(execute=True), run=True)

        self.assertProblem(problems, "strata: --version gives exit 0 and 'strata 0.1.33 metal 1\\n'")
        self.assertProblem(problems, "strata-vision: with no argument gives exit 0, want 2")

    def test_run_refuses_an_engine_that_does_not_exit(self):
        entries = self.release.entries()
        entries[0] = ("strata", b"#!/bin/sh\nsleep 5\n", 0o755, 3)
        with mock.patch.object(check_asset, "RUN_TIMEOUT_S", 1):
            problems = self.check(self.release.asset(entries, "hang.zip"), tools=FakeTools(execute=True), run=True)

        self.assertProblem(problems, "strata: --version did not exit within 1 s")

    def test_an_unreadable_asset_is_one_problem(self):
        bad = self.folder / "bad.zip"
        bad.write_bytes(b"not a zip")

        self.assertProblem(self.check(bad), "bad.zip: not a readable zip file")

    def test_the_command_line_exits_one_for_a_failing_asset(self):
        bad = self.folder / "bad.zip"
        bad.write_bytes(b"not a zip")
        with mock.patch.object(sys, "stderr") as stderr:
            status = check_asset.main([str(bad)])

        self.assertEqual(1, status)
        self.assertTrue(stderr.write.called)


if __name__ == "__main__":
    unittest.main()
