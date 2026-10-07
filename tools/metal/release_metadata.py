"""Release compatibility values for the macOS Metal engine."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIN_ENGINE = (0, 1, 34)
MIN_ENGINE_METAL = 1
RELEASE_REPOSITORY: str | None = None


def source_version() -> str:
    """Read the engine version from its CMake project declaration."""
    try:
        content = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
    except OSError:
        return "0"
    match = re.search(r"project\(strata VERSION ([\d.]+)", content)
    return match.group(1) if match else "0"
