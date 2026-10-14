"""Exercise the macOS launcher without installing packages or touching the real project environment."""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class SetupShell(unittest.TestCase):
    def test_non_macos_host_is_refused_before_venv_creation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "bin").mkdir()
            shutil.copy2(ROOT / "setup.sh", root / "setup.sh")
            uname = root / "bin" / "uname"
            uname.write_text("#!/bin/sh\nprintf '%s\\n' Linux\n", encoding="utf-8")
            uname.chmod(0o755)
            env = os.environ.copy()
            env["PATH"] = f"{root / 'bin'}:/usr/bin:/bin"

            result = subprocess.run(["/bin/sh", "setup.sh"], cwd=root, env=env, text=True,
                                    capture_output=True, check=False, timeout=10)

            self.assertEqual(result.returncode, 1, result.stdout + result.stderr)
            self.assertIn("macOS only", result.stdout)
            self.assertFalse((root / ".venv").exists())

    def test_launcher_refuses_unsupported_macs_without_creating_a_venv(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            shutil.copy2(ROOT / "setup.sh", root / "setup.sh")
            env = os.environ.copy()

            result = subprocess.run(["/bin/sh", "setup.sh", "--check"], cwd=root, env=env, text=True,
                                    capture_output=True, check=False, timeout=60)

            self.assertFalse((root / ".venv").exists())
            output = result.stdout + result.stderr
            if sys.platform != "darwin":
                self.assertEqual(result.returncode, 1, output)
                self.assertIn("macOS only", output)
            elif platform.machine().lower() not in {"arm64", "aarch64"}:
                self.assertEqual(result.returncode, 1, output)
                self.assertIn("Apple silicon", output)
            elif int(platform.mac_ver()[0].split(".", 1)[0]) < 27:
                self.assertEqual(result.returncode, 1, output)
                self.assertIn("macOS 27 or newer", output)
            else:
                self.assertEqual(result.returncode, 2, output)
                self.assertIn("macOS setup is not implemented yet", output)

    def test_setup_py_cli_and_direct_main_fail_closed(self):
        cli = subprocess.run([sys.executable, str(ROOT / "setup.py"), "--help"], cwd=ROOT, text=True,
                             capture_output=True, check=False, timeout=10)
        direct = subprocess.run([sys.executable, "-c", "import setup; raise SystemExit(setup.main())"],
                                cwd=ROOT, text=True, capture_output=True, check=False, timeout=10)
        for result in (cli, direct):
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertIn("macOS setup is not implemented yet", result.stderr)

    def test_setup_py_host_check_is_read_only_and_full_check_remains_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative in (
                "setup.py",
                "serve/__init__.py",
                "serve/macos.py",
                "tools/model_catalog.py",
            ):
                source = ROOT / relative
                destination = root / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            before = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

            host_check = subprocess.run(
                [sys.executable, str(root / "setup.py"), "--check-host"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=30,
            )
            after_host_check = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }
            full_check = subprocess.run(
                [sys.executable, str(root / "setup.py"), "--check"],
                cwd=root,
                text=True,
                capture_output=True,
                check=False,
                timeout=10,
            )
            after_full_check = {
                path.relative_to(root): path.read_bytes()
                for path in root.rglob("*")
                if path.is_file()
            }

        self.assertIn(host_check.returncode, (0, 1), host_check.stdout + host_check.stderr)
        if host_check.returncode == 0:
            self.assertIn("Strata host check passed", host_check.stdout)
        else:
            self.assertIn("error:", host_check.stderr)
        self.assertEqual(before, after_host_check, "host check must not create or modify files")
        self.assertEqual(2, full_check.returncode, full_check.stdout + full_check.stderr)
        self.assertIn("macOS setup is not implemented yet", full_check.stderr)
        self.assertEqual(before, after_full_check, "full model check must remain fail-closed")


if __name__ == "__main__":
    unittest.main()
