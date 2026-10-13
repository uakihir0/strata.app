#!/usr/bin/env python3
"""Check a Mac release asset (docs/mac/08-build-and-distribution.md, section 11.7).

    python3 tools/metal/check_asset.py ASSET.zip [--symbols SYMBOLS.zip] [--run] [--allow-dirty]

Run it from a checkout at the release's commit: the kernel manifest and METAL_REV are read from that checkout. Every
failure prints `check_asset: <member>: <problem>` to stderr, and the run goes on, so all problems are listed. The exit
code is 1 when any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.metal import release_metadata  # noqa: E402

RUN_TIMEOUT_S = 30
HEX16 = re.compile(r"[0-9a-f]{16}")
HEX40 = re.compile(r"[0-9a-f]{40}")
HEX64 = re.compile(r"[0-9a-f]{64}")
COMMIT = re.compile(r"[0-9a-f]{40}(-dirty)?")
# Paths that must not reach an asset's binaries (docs/mac/08 section 11.7, check 6). The checkout's own path is added.
FORBIDDEN_PATHS = (
    b"/Users/", b"/private/", b"/tmp/", b"/var/folders/", b"/Volumes/", b"/opt/homebrew", b"/usr/local/",
    b"/home/", b"/Applications/", b"/Library/Developer/",
)
# The one allowed occurrence: ggml's capture file name in strata-vision (docs/mac/08 section 9).
ALLOWED_PATHS = {"strata-vision": (b"/tmp/perf-metal-%d.gputrace",)}
# The values of BUILD.json that a release has (docs/mac/08 section 11.7, check 9).
EXPECTED_BUILD = {
    "backend": "metal",
    "arch": "arm64",
    "macos_min": "27.0",
    "gpu_families": ["apple7", "metal4"],
    "metal_std": "metal4.0",
    "source": "prebuilt",
    "lib_dirs": [],
    "vision": "gpu",
}
Runner = Callable[..., subprocess.CompletedProcess]
Fail = Callable[[str, str], None]


def run_tool(runner: Runner, argv: list[str], **kwargs) -> subprocess.CompletedProcess:
    """Run a macOS tool. A tool that cannot start gives a failed result, not an exception."""
    try:
        return runner(argv, capture_output=True, text=True, check=False, **kwargs)
    except OSError as exc:
        return subprocess.CompletedProcess(argv, 127, "", str(exc))


def detail(result: subprocess.CompletedProcess) -> str:
    return (result.stderr.strip() or result.stdout.strip() or f"exit {result.returncode}")


def metal_rev(root: Path) -> int | None:
    try:
        text = (root / "METAL_REV").read_text(encoding="ascii")
    except (OSError, UnicodeDecodeError):
        return None
    return int(text[:-1]) if re.fullmatch(r"[1-9]\d*\n", text) else None


def check_members(
    infos: list[zipfile.ZipInfo],
    expected: list[tuple[str, int]],
    label: str,
    fail: Fail,
    *,
    nested: bool,
) -> None:
    """The members must be exactly `expected`, in order, each a regular Unix file with its mode.

    `nested` allows folders in names, as the symbols zip does; the asset keeps every member at the top level.
    """
    names = [info.filename for info in infos]
    wanted = [name for name, _ in expected]
    if names != wanted:
        fail(label, f"members are {names}, want {wanted} in this order")
    modes = dict(expected)
    for info in infos:
        name = info.filename
        if not nested and "/" in name:
            fail(name, "is a folder or has a path; every member sits at the top level")
        if info.create_system != 3:
            fail(name, f"create_system is {info.create_system}, want 3 (Unix)")
        mode = info.external_attr >> 16
        if not stat.S_ISREG(mode):
            fail(name, "is not a regular file")
        elif name in modes and stat.S_IMODE(mode) != modes[name]:
            fail(name, f"mode is {stat.S_IMODE(mode):04o}, want {modes[name]:04o}")


def extract(archive: zipfile.ZipFile, infos: list[zipfile.ZipInfo], wanted: set[str], folder: Path) -> None:
    """Unpack the expected members, with the mode each one carries in the archive."""
    for info in infos:
        if info.filename not in wanted or info.is_dir():
            continue
        target = archive.extract(info, folder)
        os.chmod(target, stat.S_IMODE(info.external_attr >> 16))


def check_binaries(work: Path, fail: Fail, runner: Runner) -> None:
    """Checks 2 to 5: architecture, platform, linked libraries and signature."""
    for name in ("strata", "strata-vision"):
        path = work / name
        if not path.exists():
            continue
        lipo = run_tool(runner, ["lipo", "-archs", str(path)])
        if lipo.returncode or lipo.stdout.split() != ["arm64"]:
            fail(name, f"lipo -archs gives {lipo.stdout.strip() or detail(lipo)}, want arm64")

        vtool = run_tool(runner, ["vtool", "-show-build", str(path)])
        platforms = set(re.findall(r"^\s*platform (\S+)", vtool.stdout, re.MULTILINE))
        minos = set(re.findall(r"^\s*minos (\S+)", vtool.stdout, re.MULTILINE))
        if vtool.returncode or platforms != {"MACOS"} or minos != {"27.0"}:
            fail(name, f"vtool shows platform {sorted(platforms)} and minos {sorted(minos)}, want MACOS and 27.0")

        otool = run_tool(runner, ["otool", "-L", str(path)])
        if otool.returncode:
            fail(name, f"otool -L fails: {detail(otool)}")
        for load in re.findall(r"^\t(\S+)", otool.stdout, re.MULTILINE):
            if load.startswith("@"):
                fail(name, f"loads {load}: no load path may start with @")
            elif not load.startswith(("/System/Library/", "/usr/lib/")):
                fail(name, f"loads {load}, but only /System/Library/ and /usr/lib/ are allowed")
        rpath = run_tool(runner, ["otool", "-l", str(path)])
        if "cmd LC_RPATH" in rpath.stdout:
            fail(name, "has an LC_RPATH load command")

        verify = run_tool(runner, ["codesign", "-v", "--strict", str(path)])
        if verify.returncode:
            fail(name, f"codesign -v --strict fails: {detail(verify)}")
        info = run_tool(runner, ["codesign", "-dv", str(path)])
        text = info.stdout + info.stderr
        identifier = re.search(r"^Identifier=(.*)$", text, re.MULTILINE)
        if identifier is None or identifier.group(1).strip() != name:
            found = identifier.group(1).strip() if identifier else "none"
            fail(name, f"Identifier is {found}, want {name}")
        ad_hoc = "Signature=adhoc" in text
        developer_id = ("Authority=Developer ID Application:" in text
                        and re.search(r"flags=0x[0-9a-f]+\([^)]*\bruntime\b", text) is not None)
        if not ad_hoc and not developer_id:
            fail(name, "signature is neither ad hoc nor Developer ID with the hardened runtime")
        entitlements = run_tool(runner, ["codesign", "-d", "--entitlements", "-", str(path)])
        if "<key>" in entitlements.stdout + entitlements.stderr:
            fail(name, "has entitlements; the engine needs none")


def check_strings(work: Path, root: Path, fail: Fail) -> None:
    """Check 6: no forbidden path in the raw bytes, except the one allowed string of strata-vision."""
    forbidden = list(FORBIDDEN_PATHS) + [str(root).encode("utf-8")]
    for name in ("strata", "strata.metallib", "strata-vision"):
        path = work / name
        if not path.exists():
            continue
        data = path.read_bytes()
        for allowed in ALLOWED_PATHS.get(name, ()):
            data = data.replace(allowed, b"")
        for token in forbidden:
            if token in data:
                fail(name, f"contains the path {token.decode('utf-8', 'replace')!r}")


def check_engine_identity(work: Path, root: Path, build: dict | None, fail: Fail) -> None:
    """Check 7: the engine's banner and the metallib it embeds match the version and the library."""
    binary = work / "strata"
    library = work / "strata.metallib"
    if not binary.exists() or not library.exists():
        return
    data = binary.read_bytes()
    version = release_metadata.source_version(root)
    banners = re.findall(rb"engine=([0-9][0-9.]*)\n", data)
    if len(banners) != 1:
        fail("strata", f"contains {len(banners)} engine=<version> lines, want exactly 1")
    elif banners[0].decode("ascii") != version:
        fail("strata", f"engine={banners[0].decode('ascii')} but the source version is {version}")
    actual = hashlib.sha256(library.read_bytes()).hexdigest()
    hashes = re.findall(rb"metallib_sha256=([0-9a-f]{64})\n", data)
    if len(hashes) != 1:
        fail("strata", f"contains {len(hashes)} metallib_sha256=<hex> lines, want exactly 1")
    elif hashes[0].decode("ascii") != actual:
        fail("strata", f"metallib_sha256 is {hashes[0].decode('ascii')}, but strata.metallib hashes to {actual}")
    if build is not None and build.get("metallib_sha256") != actual:
        fail("BUILD.json", "metallib_sha256 does not match the strata.metallib member")


def check_kernels(work: Path, root: Path, runner: Runner, fail: Fail) -> None:
    """Check 8: the library's kernels are the manifest's, and it holds no kernel source."""
    library = work / "strata.metallib"
    if not library.exists():
        return
    try:
        manifest = (root / "src" / "metal" / "kernels" / "manifest.txt").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        fail("strata.metallib", f"cannot read the kernel manifest of the checkout: {exc}")
        return
    expected = {
        line.split()[0] for line in manifest.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    listed = run_tool(runner, ["xcrun", "metal-nm", "-j", "-U", str(library)])
    if listed.returncode:
        fail("strata.metallib", f"metal-nm fails: {detail(listed)}")
    else:
        defined = {line.strip() for line in listed.stdout.splitlines() if line.strip()}
        if defined != expected:
            fail(
                "strata.metallib",
                "defined functions differ from the manifest: "
                f"missing {sorted(expected - defined)}, extra {sorted(defined - expected)}",
            )

    copy = work / "strip-check.metallib"
    copy.write_bytes(library.read_bytes())
    stripped = run_tool(runner, ["xcrun", "metal-dsymutil", "-flat", "-remove-source", str(copy)])
    if stripped.returncode:
        fail("strata.metallib", f"metal-dsymutil -remove-source fails: {detail(stripped)}")
    elif copy.read_bytes() != library.read_bytes():
        fail("strata.metallib", "still holds kernel sources: metal-dsymutil -remove-source changes it")
    copy.unlink(missing_ok=True)


def check_build_json(work: Path, root: Path, allow_dirty: bool, fail: Fail) -> dict | None:
    """Check 9: BUILD.json has the documented keys and values. Returns the parsed object when it is one."""
    path = work / "BUILD.json"
    if not path.exists():
        return None
    data = path.read_bytes()
    if not data.endswith(b"\n"):
        fail("BUILD.json", "does not end with a newline")
    try:
        meta = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        fail("BUILD.json", "is not JSON")
        return None
    if not isinstance(meta, dict):
        fail("BUILD.json", "is not a JSON object")
        return None

    keys = set(meta)
    missing = sorted(release_metadata.BUILD_JSON_KEYS - keys)
    extra = sorted(keys - release_metadata.BUILD_JSON_KEYS)
    if missing:
        fail("BUILD.json", f"lacks keys {', '.join(missing)}")
    if extra:
        fail("BUILD.json", f"has keys {', '.join(extra)} that the documented set does not have")
    for key, want in EXPECTED_BUILD.items():
        if meta.get(key) != want:
            fail("BUILD.json", f"{key} is {meta.get(key)!r}, want {want!r}")
    rev = metal_rev(root)
    if rev is None or meta.get("metal_rev") != rev:
        fail("BUILD.json", f"metal_rev is {meta.get('metal_rev')!r}, the checkout's METAL_REV is {rev}")
    version = release_metadata.source_version(root)
    if meta.get("version") != version:
        fail("BUILD.json", f"version is {meta.get('version')!r}, the source version is {version}")
    for key in ("src", "vision_src"):
        if not isinstance(meta.get(key), str) or not HEX16.fullmatch(meta[key]):
            fail("BUILD.json", f"{key} must be 16 hex digits")
    commit = meta.get("commit")
    if not isinstance(commit, str) or not COMMIT.fullmatch(commit):
        fail("BUILD.json", "commit must be 40 hex digits")
    elif commit.endswith("-dirty") and not allow_dirty:
        fail("BUILD.json", "commit is -dirty: a release is built from a clean tree (or pass --allow-dirty)")
    if not isinstance(meta.get("metallib_sha256"), str) or not HEX64.fullmatch(meta["metallib_sha256"]):
        fail("BUILD.json", "metallib_sha256 must be 64 hex digits")
    return meta


def check_licenses(work: Path, fail: Fail) -> None:
    """Check 10: LICENSES.txt has the section headers of docs/mac/08 section 11.9, in order, and no others."""
    path = work / "LICENSES.txt"
    if not path.exists():
        return
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        fail("LICENSES.txt", "is not UTF-8 text")
        return
    titles = [match.group(1) for match in re.finditer(r"^== (.+) ==$", text, re.MULTILINE)]
    wanted = list(release_metadata.LICENSE_TITLES)
    if titles != wanted:
        fail("LICENSES.txt", f"section headers are {titles}, want {wanted} in this order")


def dwarf_uuids(runner: Runner, path: Path) -> list[str] | None:
    result = run_tool(runner, ["xcrun", "dwarfdump", "--uuid", str(path)])
    if result.returncode:
        return None
    return re.findall(r"UUID: ([0-9A-Fa-f-]{36})", result.stdout)


def check_symbols(symbols: Path, work: Path, runner: Runner, fail: Fail) -> None:
    """Check 11: the symbols zip holds the dSYM and the metallib symbols, and its UUID is the engine's."""
    label = symbols.name
    try:
        archive = zipfile.ZipFile(symbols)
    except (OSError, zipfile.BadZipFile) as exc:
        fail(label, f"not a readable zip file: {exc}")
        return
    with archive:
        expected = [(name, 0o644) for name in release_metadata.SYMBOL_MEMBERS]
        infos = archive.infolist()
        check_members(infos, expected, label, fail, nested=True)
        folder = work / "symbols"
        folder.mkdir()
        extract(archive, infos, {name for name, _ in expected}, folder)

    dwarf = folder / "strata.dSYM" / "Contents" / "Resources" / "DWARF" / "strata"
    engine = work / "strata"
    if dwarf.exists() and engine.exists():
        engine_uuids = dwarf_uuids(runner, engine)
        symbol_uuids = dwarf_uuids(runner, dwarf)
        if engine_uuids is None or symbol_uuids is None:
            fail("strata.dSYM", "dwarfdump --uuid fails")
        elif not engine_uuids or engine_uuids != symbol_uuids:
            fail("strata.dSYM", f"UUID is {symbol_uuids}, the engine's is {engine_uuids}")
    metallibsym = folder / "strata.metallibsym"
    if metallibsym.exists() and metallibsym.stat().st_size == 0:
        fail("strata.metallibsym", "is empty")


def check_run(work: Path, root: Path, runner: Runner, fail: Fail) -> None:
    """Check 12: the engine prints its banner and strata-vision its usage, without a GPU."""
    version = release_metadata.source_version(root)
    rev = metal_rev(root)
    engine = work / "strata"
    if engine.exists():
        try:
            result = runner([str(engine), "--version"], cwd=work, capture_output=True, text=True,
                            check=False, timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            fail("strata", f"--version did not exit within {RUN_TIMEOUT_S} s")
        except OSError as exc:
            fail("strata", f"cannot run --version: {exc}")
        else:
            want = f"strata {version} metal {rev}\n"
            if result.returncode != 0 or result.stdout != want:
                fail("strata", f"--version gives exit {result.returncode} and {result.stdout!r}, want {want!r}")
    vision = work / "strata-vision"
    if vision.exists():
        try:
            result = runner([str(vision)], cwd=work, capture_output=True, text=True,
                            check=False, timeout=RUN_TIMEOUT_S)
        except subprocess.TimeoutExpired:
            fail("strata-vision", f"with no argument did not exit within {RUN_TIMEOUT_S} s")
        except OSError as exc:
            fail("strata-vision", f"cannot run: {exc}")
        else:
            if result.returncode != 2 or not result.stderr.startswith("usage: strata-vision"):
                fail("strata-vision", f"with no argument gives exit {result.returncode}, want 2 and a usage line")


def check_asset(
    asset: Path,
    *,
    symbols: Path | None = None,
    run: bool = False,
    allow_dirty: bool = False,
    root: Path = ROOT,
    runner: Runner = subprocess.run,
) -> list[str]:
    """Run every check of docs/mac/08 section 11.7 and return the problem lines (empty when all pass)."""
    problems: list[str] = []

    def fail(member: str, text: str) -> None:
        problems.append(f"check_asset: {member}: {text}")

    try:
        archive = zipfile.ZipFile(asset)
    except (OSError, zipfile.BadZipFile) as exc:
        fail(asset.name, f"not a readable zip file: {exc}")
        return problems
    with archive, tempfile.TemporaryDirectory(prefix="check_asset-") as temp:
        infos = archive.infolist()
        expected = list(release_metadata.ASSET_MEMBERS)
        check_members(infos, expected, asset.name, fail, nested=False)
        work = Path(temp, "asset")
        work.mkdir()
        extract(archive, infos, {name for name, _ in expected if "/" not in name}, work)

        check_binaries(work, fail, runner)
        check_strings(work, root, fail)
        build = check_build_json(work, root, allow_dirty, fail)
        check_engine_identity(work, root, build, fail)
        check_kernels(work, root, runner, fail)
        check_licenses(work, fail)
        if symbols is not None:
            check_symbols(symbols, work, runner, fail)
        if run:
            check_run(work, root, runner, fail)
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("asset", type=Path, help="strata-macos-arm64.zip")
    parser.add_argument("--symbols", type=Path, help="strata-macos-arm64-symbols.zip")
    parser.add_argument("--run", action="store_true", help="run the engine and strata-vision (no GPU needed)")
    parser.add_argument("--allow-dirty", action="store_true", help="accept a -dirty commit in BUILD.json")
    args = parser.parse_args(argv)
    problems = check_asset(args.asset, symbols=args.symbols, run=args.run, allow_dirty=args.allow_dirty)
    for line in problems:
        print(line, file=sys.stderr)
    if problems:
        return 1
    print("check_asset: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
