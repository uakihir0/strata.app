import os
import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
