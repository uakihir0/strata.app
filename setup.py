#!/usr/bin/env python3
"""macOS setup entry point. Installation is disabled until the Apple Silicon flow is complete."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from tools.model_catalog import CONTEXTS, FAMILIES, MODELS  # noqa: E402


def main() -> int:
    print("macOS setup is not implemented yet; no files were changed.", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
