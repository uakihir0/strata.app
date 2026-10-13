#!/usr/bin/env python3
"""Build the Mac release files (docs/mac/08-build-and-distribution.md, section 11.3).

    python3 tools/metal/make_release.py [--sign IDENTITY] [--notarize (--keychain-profile P | --key K.p8 --key-id I --issuer U)]
                                        [--notes FILE] [--out dist] [--allow-dirty]

Run it with the venv's Python, from a checkout. Every step either succeeds or stops with `make_release: <problem>`
and exit 1. The zip writer is deterministic: the members, their modes and their dates depend only on the files and on
SOURCE_DATE_EPOCH, so two checkouts at different paths give the same member bytes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import stat
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from serve import macos  # noqa: E402
from tools.metal import check_asset  # noqa: E402
from tools.metal import release_metadata  # noqa: E402

# The cache values of section 11.3, steps 3 and 4, that the two builds must have.
ENGINE_CACHE = {
    "CMAKE_BUILD_TYPE": "Release",
    "STRATA_METAL_EMBED_SOURCES": "OFF",
    "STRATA_BUILD_TESTS": "OFF",
    "CMAKE_OSX_ARCHITECTURES": "arm64",
    "CMAKE_OSX_DEPLOYMENT_TARGET": "27.0",
}
VISION_CACHE = {"GGML_NATIVE": "OFF", "GGML_OPENMP": "OFF", "GGML_METAL": "ON", "GGML_METAL_EMBED_LIBRARY": "ON"}
# The licence texts, in section 11.9 order. `{llama}` is the pinned llama.cpp checkout.
LICENSE_SOURCES = (
    ("Strata", "LICENSE"),
    ("Strata Metal kernels: adapted code", "src/metal/kernels/THIRD_PARTY"),
    ("ggml block layouts (third_party/ggml)", "third_party/ggml/LICENSE"),
    ("llama.cpp and ggml", "{llama}/LICENSE"),
    ("YaRN RoPE in ggml-metal", "tools/metal/licenses/yarn-rope.txt"),
    ("stb_image", "tools/metal/licenses/stb_image.txt"),
    ("miniaudio", "tools/metal/licenses/miniaudio.txt"),
    ("subprocess.h", "tools/metal/licenses/subprocess.txt"),
    ("SHA-256", "{llama}/vendor/hash/sha256/LICENSE"),
    ("rotate-bits", "{llama}/vendor/hash/rotate-bits/LICENSE.md"),
)
# The vendor check of section 11.9. A library in the first four rows must have its section; in the last four it fails.
VENDOR_SECTIONS = (
    ("stbi_", "stb_image"),
    ("ma_decoder", "miniaudio"),
    ("subprocess_", "subprocess.h"),
    ("sha256_", "SHA-256"),
)
VENDOR_FORBIDDEN = (
    ("httplib", "cpp-httplib"),
    ("nlohmann", "nlohmann/json"),
    ("XXH", "xxHash"),
    ("SHA1", "sha1"),
)
DEFAULT_SOURCE_DATE_EPOCH_FALLBACK = 315532800  # 1980-01-01, the earliest date a zip can carry
REQUIREMENT = "macOS 27.0 or newer on M1 or newer"
INSTALL_LINE = ("Install with setup (README). A zip downloaded with a browser is quarantined, and macOS "
                "refuses its engine.")


class ReleaseError(RuntimeError):
    """A step failed. The message is printed as `make_release: <problem>`."""


# ------------------------------------------------------------------------------------------------ zip and text
def write_zip(out: Path, source_date_epoch: int, members: list[tuple[str, Path, int]]) -> None:
    """The deterministic zip of section 11.3, step 8: fixed dates, Unix system, mode in external_attr, deflate 9."""
    date = time.gmtime(max(int(source_date_epoch), DEFAULT_SOURCE_DATE_EPOCH_FALLBACK))[:6]
    with zipfile.ZipFile(out, "w") as archive:
        for name, path, mode in members:
            info = zipfile.ZipInfo(name, date_time=date)
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | mode) << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(info, path.read_bytes(), compresslevel=9)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def excerpt_missing(excerpt: list[str], source: list[str]) -> list[str]:
    """The non-empty lines of an excerpt that do not occur, in order, in the source (section 11.9)."""
    missing: list[str] = []
    position = 0
    for wanted in (line.strip() for line in excerpt if line.strip()):
        found = next((index for index in range(position, len(source)) if wanted in source[index]), None)
        if found is None:
            missing.append(wanted)
        else:
            position = found + 1
    return missing


def read_excerpt(path: Path, llama: Path) -> str:
    """An excerpt file starts with `source: <path in llama.cpp>`. Its body must occur in that file at the pin."""
    lines = path.read_text(encoding="utf-8").splitlines()
    if not lines or not lines[0].startswith("source: "):
        raise ReleaseError(f"licence excerpt {path.name} must start with a 'source: <path>' line")
    source_path = llama / lines[0][len("source: "):].strip()
    if not source_path.is_file():
        raise ReleaseError(f"licence excerpt {path.name}: {source_path.name} is missing at the pin")
    source = source_path.read_text(encoding="utf-8", errors="replace").splitlines()
    missing = excerpt_missing(lines[1:], source)
    if missing:
        raise ReleaseError(
            f"licence excerpt {path.name} is not in {lines[0][len('source: '):].strip()} at the pin: {missing[0]!r}"
        )
    return "\n".join(lines[1:]).strip("\n")


def licenses_text(root: Path, llama: Path) -> str:
    """LICENSES.txt: a `== <title> ==` header, the text and a blank line, for each section of section 11.9."""
    sections = []
    for title, source in LICENSE_SOURCES:
        if source.startswith("{llama}/"):
            path = llama / source[len("{llama}/"):]
        else:
            path = root / source
        if source.startswith("tools/metal/licenses/"):
            text = read_excerpt(path, llama)
        else:
            text = path.read_text(encoding="utf-8").strip("\n")
        sections.append(f"== {title} ==\n{text}\n\n")
    return "".join(sections)


def vendor_problems(names: list[str], titles: list[str]) -> tuple[list[str], list[str]]:
    """The vendor check of section 11.9 on the defined symbols of strata-vision. Returns (problems, notes)."""
    problems: list[str] = []
    notes: list[str] = []
    for needle, section in VENDOR_SECTIONS:
        if any(needle in name for name in names):
            if section not in titles:
                problems.append(f"strata-vision links {section}; its section is missing from LICENSES.txt")
        else:
            notes.append(f"strata-vision does not link {section}; its section stays until the licence review")
    for needle, library in VENDOR_FORBIDDEN:
        if any(needle in name for name in names):
            problems.append(
                f"strata-vision now links {library}; add its license to tools/metal/licenses/ and "
                "docs/mac/08 section 11.9"
            )
    return problems, notes


def cache_problems(cache_text: str, wanted: dict[str, str], label: str) -> list[str]:
    values = {}
    for line in cache_text.splitlines():
        match = re.fullmatch(r"([A-Za-z0-9_]+):[A-Z]+=(.*)", line)
        if match:
            values[match.group(1)] = match.group(2)
    return [
        f"{label} cache has {key}={values.get(key)!r}, want {value!r}"
        for key, value in wanted.items()
        if values.get(key) != value
    ]


def notes_text(tag: str, meta: dict, sums: list[tuple[str, Path]], signing: str, upstream: str,
               changes: str | None) -> str:
    """NOTES.md of section 11.3, step 9."""
    lines = [
        f"# {tag}",
        "",
        f"- Engine: strata {meta['version']}, METAL_REV {meta['metal_rev']}",
        f"- commit {meta['commit']}, UPSTREAM_SYNC {upstream}",
        f"- Xcode {meta['xcode']}, Metal toolchain {meta['metal_toolchain']}",
        f"- Signing: {signing}",
        f"- Requirement: {REQUIREMENT}",
        "",
        "| Member | SHA-256 | Size |",
        "| --- | --- | ---: |",
    ]
    for name, path in sums:
        lines.append(f"| {name} | {sha256_of(path)} | {path.stat().st_size} |")
    lines += ["", INSTALL_LINE, "", "## Changes", "", (changes or "(none given)").rstrip("\n"), ""]
    return "\n".join(lines)


def assemble(
    out: Path,
    stage: Path,
    stage_symbols: Path,
    *,
    source_date_epoch: int,
    tag: str,
    meta: dict,
    upstream: str,
    signing: str,
    changes: str | None,
) -> Path:
    """Write the asset, the symbols zip, SHA256SUMS and NOTES.md from the staged files (section 11.3, steps 8 and 9)."""
    asset_members = [(name, stage / name, mode) for name, mode in release_metadata.ASSET_MEMBERS]
    for _, path, _ in asset_members:
        if not path.is_file():
            raise ReleaseError(f"the stage has no {path.name}")
    asset = out / release_metadata.ASSET_NAME
    write_zip(asset, source_date_epoch, asset_members)

    dsym = stage_symbols / "strata.dSYM"
    dsym_files = sorted(
        (path.relative_to(stage_symbols).as_posix(), path) for path in dsym.rglob("*") if path.is_file()
    )
    symbol_members = [(name, path, 0o644) for name, path in dsym_files]
    symbol_members.append(("strata.metallibsym", stage_symbols / "strata.metallibsym", 0o644))
    symbols = out / release_metadata.SYMBOLS_NAME
    write_zip(symbols, source_date_epoch, symbol_members)

    sums = [(release_metadata.ASSET_NAME, asset), (release_metadata.SYMBOLS_NAME, symbols)]
    (out / release_metadata.SUMS_NAME).write_text(
        "".join(f"{sha256_of(path)}  {name}\n" for name, path in sums), encoding="ascii"
    )
    member_sums = [(name, path) for name, path, _ in asset_members]
    (out / release_metadata.NOTES_NAME).write_text(
        notes_text(tag, meta, member_sums, signing, upstream, changes), encoding="utf-8"
    )
    return asset


# ------------------------------------------------------------------------------------------------ real steps
def run(command: list[str], *, cwd: Path | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    result = subprocess.run(command, cwd=cwd, capture_output=capture, text=capture, check=False)
    if result.returncode:
        detail = (result.stderr or "").strip()[-400:] if capture else ""
        raise ReleaseError(f"{' '.join(command[:2])} failed with exit {result.returncode}"
                           + (f": {detail}" if detail else ""))
    return result


def preconditions(root: Path, allow_dirty: bool) -> Path:
    """Section 11.3, step 1. Returns the pinned llama.cpp checkout."""
    if sys.platform != "darwin":
        raise ReleaseError("a release is built on macOS")
    if platform.machine() != "arm64":
        raise ReleaseError("a release is built on an arm64 Mac")
    if macos.translated():
        raise ReleaseError("this Python runs under Rosetta (x86_64); use an arm64 Python")
    problem = macos.xcode_problem()
    if problem:
        raise ReleaseError(problem)
    if not allow_dirty:
        status = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True,
                                check=False)
        if status.returncode or status.stdout.strip():
            raise ReleaseError("the tree is not clean: commit the changes, or pass --allow-dirty")
    if re.search(r"\s", str(root)):
        raise ReleaseError(f"the checkout path contains whitespace, which the path maps cannot carry: {root}")
    try:
        return setup.get_llama_cpp(root)
    except RuntimeError as exc:
        raise ReleaseError(str(exc)) from exc


def source_date_epoch(root: Path) -> int:
    value = os.environ.get("SOURCE_DATE_EPOCH")
    if value:
        return int(value)
    result = subprocess.run(["git", "-C", str(root), "log", "-1", "--format=%ct"], capture_output=True, text=True,
                            check=False)
    if result.returncode or not result.stdout.strip():
        raise ReleaseError("SOURCE_DATE_EPOCH is not set and the commit time cannot be read")
    return int(result.stdout.strip())


def check_cache(build: Path, wanted: dict[str, str], label: str) -> None:
    problems = cache_problems((build / "CMakeCache.txt").read_text(encoding="utf-8"), wanted, label)
    if problems:
        raise ReleaseError("; ".join(problems))


def build_engine(root: Path, llama: Path, build: Path) -> None:
    shutil.rmtree(build, ignore_errors=True)
    setup.cmake_build(root, build, "strata", setup.metal_engine_defs(llama))
    check_cache(build, ENGINE_CACHE, "engine")


def build_vision(root: Path, llama: Path, build: Path) -> Path:
    shutil.rmtree(build, ignore_errors=True)
    flags = (f"-ffile-prefix-map={llama}=third_party/llama.cpp -ffile-prefix-map={root}=. "
             f"-ffile-prefix-map={build}=build-vision")
    defs = setup.metal_vision_defs(llama, portable=True) + [
        f"-DCMAKE_C_FLAGS={flags}", f"-DCMAKE_CXX_FLAGS={flags}", f"-DCMAKE_OBJC_FLAGS={flags}",
    ]
    setup.cmake_build(root / "tools" / "vision", build, "strata-vision", defs)
    check_cache(build, VISION_CACHE, "vision")
    binary = build / "bin" / "strata-vision"
    names = run(["nm", "-U", "-j", str(binary)], capture=True).stdout.split()
    problems, notes = vendor_problems(names, [title for title, _ in LICENSE_SOURCES])
    for note in notes:
        print(f"make_release: note: {note}")
    if problems:
        raise ReleaseError("; ".join(problems))
    return binary


def dwarf_uuid(path: Path) -> list[str]:
    result = run(["xcrun", "dwarfdump", "--uuid", str(path)], capture=True)
    return re.findall(r"UUID: ([0-9A-Fa-f-]{36})", result.stdout)


def stage(build_engine_dir: Path, out: Path, binary_vision: Path) -> Path:
    """Section 11.3, step 5. Returns the stage folder; the symbols folder is its sibling `stage-symbols`."""
    stage_dir = out / "stage"
    symbols_dir = out / "stage-symbols"
    for folder in (stage_dir, symbols_dir):
        shutil.rmtree(folder, ignore_errors=True)
        folder.mkdir(parents=True)
    run(["dsymutil", "strata", "-o", str(symbols_dir / "strata.dSYM")], cwd=build_engine_dir)
    shutil.copy2(build_engine_dir / "strata", stage_dir / "strata")
    shutil.copy2(build_engine_dir / "strata.metallib", stage_dir / "strata.metallib")
    shutil.copy2(binary_vision, stage_dir / "strata-vision")
    for name in ("strata", "strata-vision"):
        run(["strip", "-S", "-x", str(stage_dir / name)])
    dsym_uuid = dwarf_uuid(symbols_dir / "strata.dSYM" / "Contents" / "Resources" / "DWARF" / "strata")
    if not dsym_uuid or dwarf_uuid(stage_dir / "strata") != dsym_uuid:
        raise ReleaseError("the stripped engine's UUID differs from its dSYM")
    shutil.copy2(build_engine_dir / "strata.metallibsym", symbols_dir / "strata.metallibsym")
    return stage_dir


def sign(stage_dir: Path, identity: str) -> None:
    run(["codesign", "--force", "--options", "runtime", "--timestamp", "--sign", identity,
          str(stage_dir / "strata"), str(stage_dir / "strata-vision")])


def notarize(asset: Path, credentials: list[str]) -> None:
    result = run(["xcrun", "notarytool", "submit", str(asset), "--wait", "--output-format", "json", *credentials],
                 capture=True)
    try:
        report = json.loads(result.stdout)
    except ValueError as exc:
        raise ReleaseError(f"notarytool printed no JSON: {exc}") from exc
    if report.get("status") != "Accepted":
        identifier = report.get("id", "")
        raise ReleaseError(
            f"notarisation status is {report.get('status')!r}, not Accepted; "
            f"run: xcrun notarytool log {identifier}"
        )


def signing_text(identity: str | None, notarized: bool) -> str:
    if not identity:
        return "ad hoc"
    match = re.search(r"\(([^()]+)\)\s*$", identity)
    team = match.group(1) if match else identity
    return f"Developer ID ({team}), notarized" if notarized else f"Developer ID ({team})"


def make_release(args: argparse.Namespace) -> Path:
    root = ROOT
    out = (root / args.out).resolve()
    if args.notarize and not args.sign:
        raise ReleaseError("--notarize needs --sign")
    if args.notarize:
        notarization_credentials(args)  # refuses before any build when the credentials are missing
    llama = preconditions(root, args.allow_dirty)
    epoch = source_date_epoch(root)
    build_engine_dir = root / "build-release"
    build_vision_dir = root / "build-vision-release"
    build_engine(root, llama, build_engine_dir)
    vision_binary = build_vision(root, llama, build_vision_dir)
    out.mkdir(parents=True, exist_ok=True)
    stage_dir = stage(build_engine_dir, out, vision_binary)
    if args.sign:
        sign(stage_dir, args.sign)
    meta = setup.metal_build_json(stage_dir, "prebuilt")
    (stage_dir / "BUILD.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    (stage_dir / "LICENSES.txt").write_text(licenses_text(root, llama), encoding="utf-8")
    upstream = (root / "UPSTREAM_SYNC").read_text(encoding="ascii").strip()
    tag = f"v{release_metadata.source_version(root)}-metal.{meta['metal_rev']}"
    changes = Path(args.notes).read_text(encoding="utf-8") if args.notes else None
    asset = assemble(
        out, stage_dir, out / "stage-symbols", source_date_epoch=epoch, tag=tag, meta=meta, upstream=upstream,
        signing=signing_text(args.sign, args.notarize), changes=changes,
    )
    if args.notarize:
        notarize(asset, notarization_credentials(args))
    check = [sys.executable, str(root / "tools" / "metal" / "check_asset.py"), str(asset),
             "--symbols", str(out / release_metadata.SYMBOLS_NAME), "--run"]
    if args.allow_dirty:
        check.append("--allow-dirty")
    result = subprocess.run(check, capture_output=True, text=True, check=False)
    if result.returncode:
        raise ReleaseError(f"check_asset.py failed:\n{result.stderr.strip()}")
    return asset


def notarization_credentials(args: argparse.Namespace) -> list[str]:
    if args.keychain_profile:
        return ["--keychain-profile", args.keychain_profile]
    if args.key and args.key_id and args.issuer:
        return ["--key", args.key, "--key-id", args.key_id, "--issuer", args.issuer]
    raise ReleaseError("--notarize needs --keychain-profile, or --key, --key-id and --issuer")


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sign", metavar="IDENTITY", help="Developer ID Application identity")
    parser.add_argument("--notarize", action="store_true", help="submit the asset to notarytool (needs --sign)")
    parser.add_argument("--keychain-profile", help="notarytool keychain profile")
    parser.add_argument("--key", help="App Store Connect API key (.p8)")
    parser.add_argument("--key-id", help="API key id")
    parser.add_argument("--issuer", help="API issuer id")
    parser.add_argument("--notes", metavar="FILE", help="text for the Changes section of NOTES.md")
    parser.add_argument("--out", default="dist", help="output folder (default: dist)")
    parser.add_argument("--allow-dirty", action="store_true", help="build from a tree with uncommitted changes")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        asset = make_release(args)
    except ReleaseError as exc:
        print(f"make_release: {exc}", file=sys.stderr)
        return 1
    print(f"make_release: {asset} {sha256_of(asset)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
