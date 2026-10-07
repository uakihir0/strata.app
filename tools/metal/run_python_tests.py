#!/usr/bin/env python3
"""Run every Python test module in a fresh process.

Several upstream tests patch module globals during import. Running each module in
its own process keeps those patches from leaking into later test modules.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEST_DIRS = (ROOT / "tools", ROOT / "tools" / "metal", ROOT / "serve")
# This is a command-line checker. Its unittest coverage lives in test_test_map.py.
NON_UNITTEST_MODULES = {"tools.metal.test_map"}
RUN_MODULE = """\
import sys
import unittest

suite = unittest.defaultTestLoader.loadTestsFromName(sys.argv[1])
result = unittest.TextTestRunner(verbosity=2).run(suite)
if result.testsRun == 0:
    print("run_python_tests: module collected no tests", file=sys.stderr)
    raise SystemExit(2)
raise SystemExit(not result.wasSuccessful())
"""


def discover_modules() -> list[str]:
    modules: set[str] = set()
    for directory in TEST_DIRS:
        if not directory.is_dir():
            continue
        for path in directory.glob("test_*.py"):
            relative = path.relative_to(ROOT).with_suffix("")
            module = ".".join(relative.parts)
            if module not in NON_UNITTEST_MODULES:
                modules.add(module)
    return sorted(modules)


def run_module(module: str, *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", RUN_MODULE, module],
        cwd=ROOT,
        env=env,
        check=False,
    )


def main() -> int:
    modules = discover_modules()
    if not modules:
        print("run_python_tests: no test modules found", file=sys.stderr)
        return 1

    failures: list[tuple[str, int]] = []
    for module in modules:
        print(f"== {module} ==", flush=True)
        result = run_module(module)
        if result.returncode:
            failures.append((module, result.returncode))
            print(f"FAIL {module} (exit {result.returncode})", flush=True)
        else:
            print(f"PASS {module}", flush=True)

    print(
        f"run_python_tests: {len(modules) - len(failures)}/{len(modules)} modules passed",
        flush=True,
    )
    if failures:
        for module, status in failures:
            print(f"run_python_tests: {module} exited {status}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
