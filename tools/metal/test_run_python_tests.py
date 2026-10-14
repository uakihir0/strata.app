import io
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.metal import run_python_tests
from tools.metal.run_python_tests import discover_modules, run_module


class RunPythonTestsTests(unittest.TestCase):
    def test_standalone_map_checker_is_not_loaded_as_unittest(self):
        modules = discover_modules()

        self.assertNotIn("tools.metal.test_map", modules)
        self.assertIn("tools.metal.test_test_map", modules)

    def test_empty_module_fails_instead_of_passing(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "empty_tests.py").write_text(
                "value = 1\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (temporary, env.get("PYTHONPATH")))
            )

            result = run_module("empty_tests", env=env)

        self.assertEqual(2, result.returncode)

    def test_module_with_passing_test_succeeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            Path(temporary, "passing_tests.py").write_text(
                "import unittest\n"
                "class Passing(unittest.TestCase):\n"
                "    def test_passes(self):\n"
                "        self.assertTrue(True)\n",
                encoding="utf-8",
            )
            env = os.environ.copy()
            env["PYTHONPATH"] = os.pathsep.join(
                filter(None, (temporary, env.get("PYTHONPATH")))
            )

            result = run_module("passing_tests", env=env)

        self.assertEqual(0, result.returncode)

    def test_runner_uses_project_virtualenv_interpreter(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            python = root / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)
            completed = subprocess.CompletedProcess([str(python)], 0)

            with (
                patch.object(run_python_tests, "ROOT", root),
                patch.object(run_python_tests, "PROJECT_PYTHON", python),
                patch.object(run_python_tests.subprocess, "run", return_value=completed) as run,
            ):
                result = run_module("passing_tests")

        self.assertEqual(0, result.returncode)
        self.assertEqual(str(python), run.call_args.args[0][0])

    def test_missing_project_virtualenv_fails_without_interpreter_fallback(self):
        with tempfile.TemporaryDirectory() as temporary:
            with (
                patch.object(run_python_tests, "PROJECT_PYTHON", Path(temporary) / ".venv/bin/python"),
                patch.object(run_python_tests.sys, "stderr", new=io.StringIO()) as stderr,
                patch.object(run_python_tests, "discover_modules") as discover,
                patch.object(run_python_tests.subprocess, "run") as run,
            ):
                self.assertEqual(2, run_python_tests.main())

        self.assertIn(".venv/bin/python is missing or not executable", stderr.getvalue())
        discover.assert_not_called()
        run.assert_not_called()

    def test_cli_reports_missing_virtualenv_before_discovering_tests(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "tools" / "metal" / "run_python_tests.py"
            script.parent.mkdir(parents=True)
            script.write_text(
                Path(run_python_tests.__file__).read_text(encoding="utf-8"),
                encoding="utf-8",
            )

            result = subprocess.run(
                [sys.executable, str(script)],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )

        self.assertEqual(2, result.returncode, result.stdout + result.stderr)
        self.assertIn(".venv/bin/python is missing or not executable", result.stderr)
        self.assertIn("python3 -m venv .venv", result.stderr)

    def test_unstartable_project_virtualenv_fails_once_with_repair_hint(self):
        with tempfile.TemporaryDirectory() as temporary:
            python = Path(temporary) / ".venv" / "bin" / "python"
            python.parent.mkdir(parents=True)
            python.write_text("#!/bin/sh\n", encoding="utf-8")
            python.chmod(0o755)

            with (
                patch.object(run_python_tests, "PROJECT_PYTHON", python),
                patch.object(
                    run_python_tests,
                    "discover_modules",
                    return_value=["passing_tests", "second_tests"],
                ),
                patch.object(run_python_tests.sys, "stdout", new=io.StringIO()) as stdout,
                patch.object(run_python_tests.sys, "stderr", new=io.StringIO()) as stderr,
                patch.object(
                    run_python_tests.subprocess,
                    "run",
                    side_effect=OSError("bad interpreter"),
                ) as run,
            ):
                self.assertEqual(2, run_python_tests.main())

            self.assertEqual(1, run.call_count)
            self.assertIn(
                f"cannot start {python}: bad interpreter",
                stderr.getvalue(),
            )
            self.assertIn("python3 -m venv .venv", stderr.getvalue())
            self.assertNotIn("== second_tests ==", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
