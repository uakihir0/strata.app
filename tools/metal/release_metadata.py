"""Release compatibility values for the macOS Metal engine, and the release layout shared by its tools."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MIN_ENGINE = (0, 1, 34)
MIN_ENGINE_METAL = 1
RELEASE_REPOSITORY: str | None = "uakihir0/strata.app"

# The release files (docs/mac/08-build-and-distribution.md, section 11.2).
ASSET_NAME = "strata-macos-arm64.zip"
SYMBOLS_NAME = "strata-macos-arm64-symbols.zip"
SUMS_NAME = "SHA256SUMS"
NOTES_NAME = "NOTES.md"
# Members of the asset, in order, with their modes.
ASSET_MEMBERS = (
    ("strata", 0o755),
    ("strata.metallib", 0o644),
    ("strata-vision", 0o755),
    ("BUILD.json", 0o644),
    ("LICENSES.txt", 0o644),
)
# Members of the symbols zip, in order: the dSYM's files sorted by path, then the metallib's symbols.
SYMBOL_MEMBERS = (
    "strata.dSYM/Contents/Info.plist",
    "strata.dSYM/Contents/Resources/DWARF/strata",
    "strata.dSYM/Contents/Resources/Relocations/aarch64/strata.yml",
    "strata.metallibsym",
)
# BUILD.json keys: docs/mac/07 section 4.7.2, then the informational keys of docs/mac/08 section 8.4.
BUILD_JSON_KEYS = frozenset({
    "version", "metal_rev", "backend", "arch", "macos_min", "gpu_families", "metal_std", "source", "lib_dirs",
    "vision", "src", "vision_src", "xcode", "metal_toolchain", "commit", "metallib_sha256",
})
# LICENSES.txt sections, in order, from docs/mac/08 section 11.9. Each starts with "== <title> ==".
LICENSE_TITLES = (
    "Strata",
    "Strata Metal kernels: adapted code",
    "ggml block layouts (third_party/ggml)",
    "llama.cpp and ggml",
    "YaRN RoPE in ggml-metal",
    "stb_image",
    "miniaudio",
    "subprocess.h",
    "SHA-256",
    "rotate-bits",
)


def source_version(root: Path = ROOT) -> str:
    """Read the engine version from the CMake project declaration of the checkout at `root`."""
    try:
        content = (root / "CMakeLists.txt").read_text(encoding="utf-8")
    except OSError:
        return "0"
    match = re.search(r"project\(strata VERSION ([\d.]+)", content)
    return match.group(1) if match else "0"
