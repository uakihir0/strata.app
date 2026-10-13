"""The Mac build definitions, source identity, tool lookup and BUILD.json of setup.py (docs/mac/08, section 8).

Every input is synthetic: a temporary tree, a fake engine folder, and a fake `xcodebuild` and `git` through
subprocess.run.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.metal import release_metadata  # noqa: E402


def write(path: Path, data: bytes | str = b"x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(data, str):
        path.write_text(data, encoding="utf-8")
    else:
        path.write_bytes(data)


def fake_source_tree(root: Path) -> None:
    write(root / "CMakeLists.txt", "project(strata VERSION 0.1.34 LANGUAGES CXX)\n")
    write(root / "CMakePresets.json", "{}\n")
    write(root / "METAL_REV", "1\n")
    write(root / "cmake" / "metal" / "backend.cmake", "# backend\n")
    write(root / "src" / "metal" / "engine.mm", "int engine;\n")
    write(root / "include" / "strata" / "metal.h", "#pragma once\n")
    write(root / "third_party" / "ggml" / "VERSION.txt", "3cf03257f219afbe7334045ff7c6a06ac68c627d\n")
    write(root / "tools" / "vision" / "CMakeLists.txt", "project(strata_vision)\n")
    write(root / "tools" / "vision" / "strata_vision.cpp", "int main() {}\n")
    write(root / "docs" / "note.md", "text\n")


class EngineDefinitions(unittest.TestCase):
    def test_engine_defines_match_the_mac_release_preset(self):
        presets = json.loads((ROOT / "CMakePresets.json").read_text(encoding="utf-8"))
        preset = next(item for item in presets["configurePresets"] if item["name"] == "mac-release")
        cache = preset["cacheVariables"]

        checked = 0
        for definition in setup.metal_engine_defs("/llama"):
            name, value = definition[2:].split("=", 1)
            if name == "STRATA_GGML_DIR":
                self.assertEqual("/llama", value)
                continue
            with self.subTest(name=name):
                self.assertEqual(cache[name], value)
                checked += 1
        self.assertEqual(3, checked)

    def test_vision_defines_are_all_read_by_the_vision_cmake_file(self):
        source = (ROOT / "tools" / "vision" / "CMakeLists.txt").read_text(encoding="utf-8")
        for definition in setup.metal_vision_defs("/llama", portable=True):
            name = definition[2:].split("=", 1)[0]
            if name.startswith("CMAKE_"):
                continue
            with self.subTest(name=name):
                self.assertIn(name, source)

    def test_portable_build_adds_only_the_portable_switch(self):
        local = setup.metal_vision_defs("/llama", portable=False)
        portable = setup.metal_vision_defs("/llama", portable=True)

        self.assertEqual(local + ["-DSTRATA_PORTABLE=ON"], portable)
        for defines in (local, portable):
            self.assertIn("-DGGML_OPENMP=OFF", defines)
            self.assertIn("-DGGML_METAL_EMBED_LIBRARY=ON", defines)
            self.assertIn("-DLLAMA_DIR=/llama", defines)


class SourceIdentity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        fake_source_tree(self.root)
        self.patch = mock.patch.object(setup, "ROOT", self.root)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temporary.cleanup()

    def test_engine_identity_is_sixteen_hex_and_ignores_finder_files(self):
        before = setup.source_hash(setup.ENGINE_SOURCES)

        self.assertRegex(before, re.compile(r"^[0-9a-f]{16}$"))
        write(self.root / "src" / ".DS_Store", b"finder")
        write(self.root / "cmake" / "metal" / ".DS_Store", b"finder")
        self.assertEqual(before, setup.source_hash(setup.ENGINE_SOURCES))

    def test_each_engine_input_changes_the_identity(self):
        before = setup.source_hash(setup.ENGINE_SOURCES)
        edits = {
            "CMakeLists.txt": lambda: write(self.root / "CMakeLists.txt", "changed\n"),
            "CMakePresets.json": lambda: write(self.root / "CMakePresets.json", "{ }\n"),
            "METAL_REV": lambda: write(self.root / "METAL_REV", "2\n"),
            "cmake/metal": lambda: write(self.root / "cmake" / "metal" / "new.cmake", "new\n"),
            "src": lambda: write(self.root / "src" / "metal" / "engine.mm", "int changed;\n"),
            "include": lambda: write(self.root / "include" / "strata" / "metal.h", "#define X\n"),
            "third_party/ggml/VERSION.txt": lambda: write(
                self.root / "third_party" / "ggml" / "VERSION.txt", "0" * 40 + "\n"
            ),
        }
        self.assertEqual(set(setup.ENGINE_SOURCES), set(edits))
        for name, edit in edits.items():
            with self.subTest(name=name):
                edit()
                self.assertNotEqual(before, setup.source_hash(setup.ENGINE_SOURCES))
                before = setup.source_hash(setup.ENGINE_SOURCES)

    def test_a_renamed_file_with_the_same_bytes_changes_the_identity(self):
        before = setup.source_hash(["src"])
        (self.root / "src" / "metal" / "engine.mm").rename(self.root / "src" / "metal" / "renamed.mm")

        self.assertNotEqual(before, setup.source_hash(["src"]))

    def test_inputs_outside_the_list_do_not_change_the_engine_identity(self):
        before = setup.source_hash(setup.ENGINE_SOURCES)
        write(self.root / "tools" / "vision" / "strata_vision.cpp", "int changed;\n")
        write(self.root / "docs" / "note.md", "changed\n")

        self.assertEqual(before, setup.source_hash(setup.ENGINE_SOURCES))

    def test_vision_identity_covers_the_vision_helper_only(self):
        before = setup.source_hash(setup.VISION_SOURCES)
        write(self.root / "src" / "metal" / "engine.mm", "int changed;\n")
        self.assertEqual(before, setup.source_hash(setup.VISION_SOURCES))

        write(self.root / "tools" / "vision" / "strata_vision.cpp", "int changed;\n")
        self.assertNotEqual(before, setup.source_hash(setup.VISION_SOURCES))

    def test_a_missing_source_input_is_an_error(self):
        with self.assertRaises(FileNotFoundError):
            setup.source_hash(["no-such-input"])

    def test_metal_revision_is_read_as_an_integer(self):
        write(self.root / "METAL_REV", "7\n")

        self.assertEqual(7, setup.metal_source_rev())


class ToolLookup(unittest.TestCase):
    def test_the_pinned_tool_next_to_the_python_wins_over_path(self):
        with tempfile.TemporaryDirectory() as temp:
            venv_bin = Path(temp, "venv", "bin")
            path_bin = Path(temp, "path")
            for folder in (venv_bin, path_bin):
                folder.mkdir(parents=True)
                tool = folder / "cmake"
                tool.write_text("#!/bin/sh\n", encoding="utf-8")
                tool.chmod(0o755)

            with mock.patch.object(setup.sys, "executable", str(venv_bin / "python3")), \
                    mock.patch.dict(os.environ, {"PATH": str(path_bin)}):
                self.assertEqual(str(venv_bin / "cmake"), setup.find_tool("cmake"))

            other = Path(temp, "other", "python3")
            with mock.patch.object(setup.sys, "executable", str(other)), \
                    mock.patch.dict(os.environ, {"PATH": str(path_bin)}):
                self.assertEqual(str(path_bin / "cmake"), setup.find_tool("cmake"))
                self.assertIsNone(setup.find_tool("no-such-strata-tool"))

    def test_a_venv_file_that_is_not_executable_is_skipped(self):
        with tempfile.TemporaryDirectory() as temp:
            venv_bin = Path(temp, "venv", "bin")
            path_bin = Path(temp, "path")
            venv_bin.mkdir(parents=True)
            path_bin.mkdir()
            (venv_bin / "ninja").write_text("not runnable\n", encoding="utf-8")
            (venv_bin / "ninja").chmod(0o644)
            (path_bin / "ninja").write_text("#!/bin/sh\n", encoding="utf-8")
            (path_bin / "ninja").chmod(0o755)

            with mock.patch.object(setup.sys, "executable", str(venv_bin / "python3")), \
                    mock.patch.dict(os.environ, {"PATH": str(path_bin)}):
                self.assertEqual(str(path_bin / "ninja"), setup.find_tool("ninja"))


class BuildJson(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name, "checkout")
        fake_source_tree(self.root)
        self.eng = Path(self.temporary.name, "engine")
        write(self.eng / "strata.metallib", b"kernel library")
        self.toolchain = "com.apple.dt.toolchain.Metal.32023.921.5"
        self.git_status = ""
        self.git_commit = "a" * 40
        self.git_ok = True
        self.patches = [
            mock.patch.object(setup, "ROOT", self.root),
            mock.patch.object(release_metadata, "source_version", return_value="0.1.34"),
            mock.patch.object(setup.subprocess, "run", side_effect=self.fake_run),
        ]
        for patch in self.patches:
            patch.start()

    def tearDown(self):
        for patch in reversed(self.patches):
            patch.stop()
        self.temporary.cleanup()

    def fake_run(self, command, **kwargs):
        command = list(command)
        if command == ["xcodebuild", "-version"]:
            stdout, code = "Xcode 27.0\nBuild version 27A266a\n", 0
        elif command == ["xcodebuild", "-showComponent", "MetalToolchain", "-json"]:
            stdout = json.dumps({
                "toolchainIdentifier": self.toolchain,
                "buildVersion": "32023",
                "status": "installed",
                "path": "/private/var/folders/example/Metal.xctoolchain",
            })
            code = 0
        elif command[:4] == ["git", "-C", str(self.root), "rev-parse"]:
            stdout, code = (self.git_commit + "\n", 0) if self.git_ok else ("", 128)
        elif command[:4] == ["git", "-C", str(self.root), "status"]:
            stdout, code = self.git_status, 0
        else:
            stdout, code = "", 1
        return subprocess.CompletedProcess(command, code, stdout, "")

    def test_a_local_build_writes_every_documented_key(self):
        meta = setup.metal_build_json(self.eng, "local")

        self.assertEqual(release_metadata.BUILD_JSON_KEYS, set(meta))
        self.assertEqual("0.1.34", meta["version"])
        self.assertEqual(1, meta["metal_rev"])
        self.assertEqual(
            {"backend": "metal", "arch": "arm64", "macos_min": "27.0", "gpu_families": ["apple7", "metal4"],
             "metal_std": "metal4.0", "source": "local", "lib_dirs": [], "vision": "none", "vision_src": None},
            {key: meta[key] for key in ("backend", "arch", "macos_min", "gpu_families", "metal_std", "source",
                                       "lib_dirs", "vision", "vision_src")},
        )
        self.assertEqual(setup.source_hash(setup.ENGINE_SOURCES), meta["src"])
        self.assertEqual("27.0 (27A266a)", meta["xcode"])
        self.assertEqual(self.toolchain, meta["metal_toolchain"])
        self.assertEqual("a" * 40, meta["commit"])
        self.assertEqual(hashlib.sha256(b"kernel library").hexdigest(), meta["metallib_sha256"])

    def test_the_text_is_deterministic_and_ends_with_a_newline(self):
        first = json.dumps(setup.metal_build_json(self.eng, "prebuilt"), indent=1) + "\n"
        second = json.dumps(setup.metal_build_json(self.eng, "prebuilt"), indent=1) + "\n"

        self.assertEqual(first, second)
        self.assertTrue(first.endswith("}\n"))
        self.assertEqual(release_metadata.BUILD_JSON_KEYS, set(json.loads(first)))

    def test_a_dirty_checkout_marks_the_commit(self):
        self.git_status = " M setup.py\n"

        self.assertEqual("a" * 40 + "-dirty", setup.metal_build_json(self.eng, "local")["commit"])

    def test_a_zip_download_without_git_has_an_empty_commit(self):
        self.git_ok = False

        meta = setup.metal_build_json(self.eng, "prebuilt")
        self.assertEqual("", meta["commit"])

    def test_a_vision_helper_sets_gpu_and_its_identity(self):
        write(self.eng / "strata-vision", b"vision binary")

        meta = setup.metal_build_json(self.eng, "local")

        self.assertEqual("gpu", meta["vision"])
        self.assertEqual(setup.source_hash(setup.VISION_SOURCES), meta["vision_src"])
        self.assertRegex(meta["vision_src"], re.compile(r"^[0-9a-f]{16}$"))


class LlamaCheckout(unittest.TestCase):
    PIN = "3cf03257f219afbe7334045ff7c6a06ac68c627d"

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        write(self.root / "third_party" / "ggml" / "VERSION.txt", f"{self.PIN} Sun Sep 20 16:08:11 2026 +0800\n")

    def tearDown(self):
        self.temporary.cleanup()

    def test_the_pin_is_the_first_word_of_the_version_file(self):
        self.assertEqual(self.PIN, setup.llama_pin(self.root))

        write(self.root / "third_party" / "ggml" / "VERSION.txt", "short\n")
        with self.assertRaisesRegex(RuntimeError, "must start with a full llama.cpp commit"):
            setup.llama_pin(self.root)

    def test_an_existing_checkout_must_report_the_pin(self):
        (self.root / "third_party" / "llama.cpp").mkdir()
        commands = []

        def runner(command, **kwargs):
            commands.append(list(command))
            return subprocess.CompletedProcess(command, 0, self.PIN + "\n", "")

        self.assertEqual(self.root / "third_party" / "llama.cpp", setup.get_llama_cpp(self.root, runner))
        self.assertEqual(
            [["git", "-C", str(self.root / "third_party" / "llama.cpp"), "rev-parse", "HEAD"]], commands
        )

        def other(command, **kwargs):
            return subprocess.CompletedProcess(command, 0, "b" * 40 + "\n", "")

        with self.assertRaisesRegex(RuntimeError, "not verified at the pin"):
            setup.get_llama_cpp(self.root, other)

        def no_git(command, **kwargs):
            return subprocess.CompletedProcess(command, 128, "", "fatal: not a git repository")

        with self.assertRaisesRegex(RuntimeError, "not verified at the pin"):
            setup.get_llama_cpp(self.root, no_git)

    def test_a_missing_checkout_is_cloned_and_set_to_the_pin(self):
        commands = []

        def runner(command, **kwargs):
            commands.append(list(command))
            stdout = self.PIN + "\n" if "rev-parse" in command else ""
            return subprocess.CompletedProcess(command, 0, stdout, "")

        setup.get_llama_cpp(self.root, runner)

        target = str(self.root / "third_party" / "llama.cpp")
        self.assertEqual(
            [["git", "clone", "--no-checkout", setup.LLAMA_REPO, target],
             ["git", "-C", target, "checkout", "--detach", self.PIN],
             ["git", "-C", target, "rev-parse", "HEAD"]],
            commands,
        )

    def test_a_failed_clone_is_reported(self):
        def runner(command, **kwargs):
            return subprocess.CompletedProcess(command, 128, "", "network unreachable")

        with self.assertRaisesRegex(RuntimeError, "cannot fetch the pinned llama.cpp: network unreachable"):
            setup.get_llama_cpp(self.root, runner)


class CMakeBuild(unittest.TestCase):
    def test_configure_then_build_with_the_venv_tools(self):
        calls = []
        tools = {"cmake": "/venv/bin/cmake", "ninja": "/venv/bin/ninja"}

        with mock.patch.object(setup, "find_tool", side_effect=tools.get), \
                mock.patch.object(setup.subprocess, "run",
                                  side_effect=lambda command, **kwargs: calls.append(list(command))):
            setup.cmake_build(Path("/src"), Path("/build"), "strata", ["-DSTRATA_BUILD_TESTS=OFF"])

        self.assertEqual(
            [
                ["/venv/bin/cmake", "-G", "Ninja", "-DCMAKE_MAKE_PROGRAM=/venv/bin/ninja", "-S", "/src",
                 "-B", "/build", "-DCMAKE_BUILD_TYPE=Release", "-DSTRATA_BUILD_TESTS=OFF"],
                ["/venv/bin/cmake", "--build", "/build", "--target", "strata"],
            ],
            calls,
        )

    def test_both_tools_are_needed(self):
        with mock.patch.object(setup, "find_tool", return_value=None):
            with self.assertRaisesRegex(RuntimeError, "cmake and ninja are needed"):
                setup.cmake_build(Path("/src"), Path("/build"), "strata", [])


if __name__ == "__main__":
    unittest.main()
