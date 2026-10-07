import json
import os
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path

from tools.metal import check_drift


class CheckDriftTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.git_env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "Drift Test",
            "GIT_AUTHOR_EMAIL": "drift-test@example.invalid",
            "GIT_COMMITTER_NAME": "Drift Test",
            "GIT_COMMITTER_EMAIL": "drift-test@example.invalid",
        }
        self.git("init", "-q")
        self.git("config", "user.name", "Drift Test")
        self.git("config", "user.email", "drift-test@example.invalid")
        self._write_upstream_baseline()
        self.git("add", ".")
        self.git("commit", "-qm", "upstream baseline")
        self.base = self.git("rev-parse", "HEAD").stdout.strip()

        self._write_project_overlay()
        self._write_drift_template()
        (self.root / "UPSTREAM_SYNC").write_text(self.base + "\n", encoding="ascii")
        self.run_drift(["--update"])

    def tearDown(self):
        self.temporary.cleanup()

    def git(self, *args, env=None, input_text=None):
        return subprocess.run(
            ["git", *args],
            cwd=self.root,
            env=self.git_env if env is None else env,
            text=True,
            input=input_text,
            capture_output=True,
            check=True,
        )

    def write(self, path, content):
        target = self.root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")

    def _write_upstream_baseline(self):
        self.write(
            "src/program/generate.cpp",
            "\n".join(
                [
                    "int main(int argc, char** argv) {",
                    '  if (a == "--help") return 0;',
                    '  if (a == "--base") return 0;',
                    "  return 0;",
                    "}",
                    "",
                    'void info() { std::printf("INFO kv=%s\\n", "fp16"); }',
                    'void error() { std::printf("ERR original error\\n"); }',
                    'void done() { std::printf("DONE %lld %s\\n", 1, "stop"); }',
                    "",
                ]
            ),
        )
        self.write(
            "src/core/runtime.cpp",
            'const char* mode = std::getenv("STRATA_BASE");\n',
        )
        self.write("src/kernels/cuda/base.cu", "__global__ void base_kernel() {}\n")
        self.write(
            "setup.py",
            'MODELS = {"BASE": {"context": 8192}}\n'
            'HF_REVISIONS = {"org/model": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
        )
        self.write("serve/server.py", "print('server')\n")
        self.write("CMakeLists.txt", "cmake_minimum_required(VERSION 3.24)\n")
        self.write(".gitattributes", "*.metal text eol=lf\n")
        self.write(".gitignore", "*.gputrace\n")
        self.write("src/platform/direct_file.cpp", "int direct_file() { return 0; }\n")

    def _write_project_overlay(self):
        self.write(
            "docs/mac/06-engine-host-port.md",
            "| Argument parser | `generate.cpp:1-5` | copied parser |\n",
        )
        self.write("CMakeLists.txt", "cmake_minimum_required(VERSION 3.24)\n# MAC: test patch\n")
        self.write(".gitattributes", "*.metal text eol=lf\n# MAC: test patch\n")
        self.write(".gitignore", "*.gputrace\n# MAC: test patch\n")
        self.write("serve/server.py", "print('server')\n# MAC: patched platform branch\n")
        self.write(
            "src/platform/direct_file.cpp",
            "int direct_file() { return 0; }\n// MAC: platform patch\n",
        )

    def _write_drift_template(self):
        drift = {
            "schema": 1,
            "classified_at": self.base,
            "markers": {
                "CMakeLists.txt": 1,
                ".gitattributes": 1,
                ".gitignore": 1,
                "setup.py": 0,
                "serve/server.py": 1,
                "src/platform/direct_file.cpp": 1,
            },
            "copies": {},
            "flags": [],
            "env": [],
            "info_keys": [],
            "err_strings": [],
            "done_fields": [],
            "cuda_kernels": {},
            "models": [],
            "revisions": {},
            "no_arithmetic": ["src/platform/**"],
        }
        (self.root / "tools/metal").mkdir(parents=True, exist_ok=True)
        if self.copy_range:
            drift["copies"] = {
                "src/metal/program/options.cpp": {
                    "from": "src/program/generate.cpp",
                    "range": "1-1",
                    "sha256": "0" * 64,
                }
            }
        (self.root / "tools/metal/drift.json").write_text(
            json.dumps(drift, indent=2) + "\n", encoding="utf-8"
        )

    @property
    def copy_range(self):
        return getattr(self, "_copy_range", False)

    @copy_range.setter
    def copy_range(self, value):
        self._copy_range = value

    def run_drift(self, argv):
        stdout = StringIO()
        stderr = StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            result = check_drift.main(self.root, argv)
        return result, stdout.getvalue(), stderr.getvalue()

    def upstream_commit(self, changes):
        index = Path(self.temporary.name) / "upstream-index"
        env = {**self.git_env, "GIT_INDEX_FILE": str(index)}
        self.git("read-tree", self.base, env=env)
        for path, content in changes.items():
            blob = self.git("hash-object", "-w", "--stdin", env=env, input_text=content).stdout.strip()
            self.git(
                "update-index",
                "--add",
                "--cacheinfo",
                f"100644,{blob},{path}",
                env=env,
            )
        tree = self.git("write-tree", env=env).stdout.strip()
        commit = subprocess.run(
            ["git", "commit-tree", tree, "-p", self.base, "-m", "upstream change"],
            cwd=self.root,
            env=env,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        (self.root / "UPSTREAM_SYNC").write_text(commit + "\n", encoding="ascii")
        return commit

    def assert_detected_then_update_accepts(self, changes, expected):
        self.upstream_commit(changes)
        code, _, stderr = self.run_drift([])
        self.assertEqual(1, code, stderr)
        self.assertIn(expected, stderr)

        code, stdout, stderr = self.run_drift(["--update"])
        self.assertEqual(0, code, stderr)
        self.assertIn("updated classification", stdout)
        code, stdout, stderr = self.run_drift([])
        self.assertEqual(0, code, stderr)
        self.assertIn("golden:", stdout)

    def test_unchanged_snapshot_passes(self):
        code, stdout, stderr = self.run_drift([])

        self.assertEqual(0, code, stderr)
        self.assertIn("upstream changed: none", stdout)
        self.assertIn("golden: still valid", stdout)

    def test_golden_verdict_uses_production_no_arithmetic_policy(self):
        drift_path = check_drift.ROOT / "tools/metal/drift.json"
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        patterns = drift["no_arithmetic"]

        self.assertEqual(["src/platform/**"], patterns)
        self.assertEqual("still valid", check_drift.golden_verdict(["src/platform/direct_file.cpp"], patterns))
        self.assertEqual("record again", check_drift.golden_verdict(["src/program/server_loop.cpp"], patterns))
        self.assertEqual("record again", check_drift.golden_verdict(["src/options.cpp"], patterns))
        self.assertEqual("record again", check_drift.golden_verdict(["include/strata/model.hpp"], patterns))
        self.assertEqual("still valid", check_drift.golden_verdict(["serve/server.py", "docs/mac/README.md"], patterns))

    def test_new_flag_is_detected(self):
        source = (self.root / "src/program/generate.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/program/generate.cpp": source.replace("  return 0;", '  if (a == "--new-flag") return 0;\n  return 0;', 1)},
            "new flag: --new-flag",
        )

    def test_new_environment_variable_is_detected(self):
        source = (self.root / "src/core/runtime.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/core/runtime.cpp": source + 'const char* extra = getenv("STRATA_NEW");\n'},
            "new environment variable: STRATA_NEW",
        )

    def test_new_info_key_is_detected(self):
        source = (self.root / "src/program/generate.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/program/generate.cpp": source.replace("INFO kv=%s", "INFO kv=%s arena=%d")},
            "new INFO key: arena",
        )

    def test_new_error_string_is_detected(self):
        source = (self.root / "src/program/generate.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/program/generate.cpp": source.replace("ERR original error", "ERR changed error")},
            "new ERR string: ERR changed error",
        )

    def test_new_done_field_is_detected(self):
        source = (self.root / "src/program/generate.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/program/generate.cpp": source.replace("DONE %lld %s", "DONE %lld %s %d")},
            "new DONE field sequence",
        )

    def test_changed_copied_range_is_detected(self):
        self.copy_range = True
        self._write_drift_template()
        self.run_drift(["--update"])
        source = (self.root / "src/program/generate.cpp").read_text()
        self.assert_detected_then_update_accepts(
            {"src/program/generate.cpp": source.replace("int main", "int renamed_main")},
            "copied range changed: src/metal/program/options.cpp",
        )

    def test_removed_marker_is_detected_and_update_records_it(self):
        path = self.root / "src/platform/direct_file.cpp"
        path.write_text("int direct_file() { return 0; }\n", encoding="utf-8")

        code, _, stderr = self.run_drift([])
        self.assertEqual(1, code, stderr)
        self.assertIn("src/platform/direct_file.cpp: marker count expected 1, found 0", stderr)
        self.assertEqual(0, self.run_drift(["--update"])[0])
        self.assertEqual(0, self.run_drift([])[0])

    def test_change_next_to_python_marker_is_detected(self):
        self.assert_detected_then_update_accepts(
            {"serve/server.py": "print('server')\nprint('new behavior')\n"},
            "Python change near # MAC: marker in serve/server.py:2",
        )

    def test_new_python_platform_branch_is_detected(self):
        self.assert_detected_then_update_accepts(
            {
                "serve/server.py": (
                    "print('server')\n"
                    "if sys.platform == 'darwin':\n"
                    "    pass\n"
                )
            },
            "new subprocess/platform branch in serve/server.py",
        )

    def test_changed_cuda_kernel_is_detected(self):
        self.assert_detected_then_update_accepts(
            {"src/kernels/cuda/base.cu": "__global__ void base_kernel() { int x = 1; }\n"},
            "CUDA kernel changed: src/kernels/cuda/base.cu",
        )

    def test_new_cuda_kernel_is_detected(self):
        self.assert_detected_then_update_accepts(
            {"src/kernels/cuda/new.cu": "__global__ void new_kernel() {}\n"},
            "CUDA kernel added: src/kernels/cuda/new.cu",
        )

    def test_new_model_is_detected(self):
        source = (self.root / "setup.py").read_text()
        self.assert_detected_then_update_accepts(
            {"setup.py": source.replace('"BASE":', '"NEW": {}, "BASE":')},
            "new model: NEW",
        )

    def test_changed_model_revision_is_detected(self):
        source = (self.root / "setup.py").read_text()
        self.assert_detected_then_update_accepts(
            {"setup.py": source.replace("a" * 40, "b" * 40)},
            "model revision changed: org/model",
        )


if __name__ == "__main__":
    unittest.main()
