#!/usr/bin/env python3
"""Verify a draft Mac release (docs/mac/08-build-and-distribution.md, section 11.8).

    python3 tools/metal/verify_release.py TAG [--repo OWNER/NAME] [--no-rebuild]

It runs on the reference Mac, from a checkout of the project, in the verify job of mac-release.yml. A draft is visible
only to people with push access, so gh needs a token with `contents: write`. Any failure exits 1, and the release stays
a draft. The acceptance runs (step 6) need tools/metal/accept.py (docs/mac/09 T15), which this checkout does not have, so
the step fails closed until it exists. Step 5 installs with setup's get_prebuilt_metal and smoke_test (docs/mac/07 S7).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import setup  # noqa: E402
from tools.metal import release_metadata  # noqa: E402

TAG_SHAPE = re.compile(r"v[0-9]+\.[0-9]+\.[0-9]+-metal\.[1-9][0-9]*")
BINARIES = ("strata", "strata-vision")
Runner = Callable[..., subprocess.CompletedProcess]
Installer = Callable[[str, Path], None]


class VerifyError(RuntimeError):
    """A step failed. The message is printed as `verify_release: <problem>`."""


def run(runner: Runner, command: list[str], *, cwd: Path | None = None, what: str) -> subprocess.CompletedProcess:
    result = runner(command, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()[-400:]
        raise VerifyError(f"{what} failed with exit {result.returncode}" + (f": {detail}" if detail else ""))
    return result


def download(runner: Runner, tag: str, repo: str, folder: Path) -> None:
    """Step 1: the three assets of the draft into `folder`."""
    run(runner, ["gh", "release", "download", tag, "--repo", repo, "--dir", str(folder)],
        what=f"gh release download {tag}")
    for name in (release_metadata.ASSET_NAME, release_metadata.SYMBOLS_NAME, release_metadata.SUMS_NAME):
        if not (folder / name).is_file():
            raise VerifyError(f"the draft {tag} has no {name}")


def check_sums(runner: Runner, folder: Path) -> None:
    """Step 2: shasum -a 256 -c SHA256SUMS."""
    result = runner(["shasum", "-a", "256", "-c", release_metadata.SUMS_NAME], cwd=folder,
                    capture_output=True, text=True, check=False)
    if result.returncode:
        raise VerifyError(f"{release_metadata.SUMS_NAME} does not match the downloaded assets: "
                          f"{(result.stdout + result.stderr).strip()}")


def default_installer(url_base: str, folder: Path) -> None:
    """Step 5's install, as setup does it (docs/mac/07 sections 4.7.1 and 4.7.5): the draft's engine from `url_base`
    into a fresh folder, then the smoke test. setup's engine folder is pointed at the fresh one for the install and
    restored afterwards, so the checkout's own engine is never replaced."""
    with tempfile.TemporaryDirectory(prefix="verify-install-") as home:
        saved = setup.ENGINE_DIR
        setup.ENGINE_DIR = Path(home) / "engine"
        try:
            engine = setup.get_prebuilt_metal(url_base, updating=True)
            if engine is None:
                raise VerifyError(f"step 5 (install): setup did not install the draft's engine from {url_base} "
                                  "(its messages say why)")
            setup.smoke_test(engine)
        except setup.SetupError as exc:
            raise VerifyError(f"step 5 (install): {exc}") from exc
        finally:
            setup.ENGINE_DIR = saved


def install_check(folder: Path, installer: Installer | None) -> None:
    """Step 5: serve the download on 127.0.0.1 and install from it, as a user would."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = subprocess.Popen(
        [sys.executable, "-m", "http.server", "--bind", "127.0.0.1", str(port)],
        cwd=folder, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        (installer or default_installer)(f"http://127.0.0.1:{port}/", folder)
    finally:
        server.terminate()
        server.wait()


def acceptance(runner: Runner, worktree: Path) -> None:
    """Step 6: the release acceptance runs of docs/mac/09 section 8. Refuses until accept.py exists."""
    accept = worktree / "tools" / "metal" / "accept.py"
    if not accept.is_file():
        raise VerifyError("step 6 needs tools/metal/accept.py (docs/mac/09 T15), which this checkout lacks")
    run(runner, [sys.executable, str(accept), "release"], cwd=worktree, what="accept.py release")


def extract_asset(asset: Path, folder: Path) -> Path:
    folder.mkdir(parents=True)
    with zipfile.ZipFile(asset) as archive:
        archive.extractall(folder)
    return folder


def developer_id_signed(runner: Runner, path: Path) -> bool:
    result = runner(["codesign", "-dv", str(path)], capture_output=True, text=True, check=False)
    return "Authority=Developer ID Application:" in (result.stdout + result.stderr)


def member_bytes(runner: Runner, folder: Path, name: str, scratch: Path, strip: bool) -> bytes:
    """The bytes of one extracted member. With `strip`, a binary is read from a copy with its signature removed."""
    source = folder / name
    if not strip or name not in BINARIES:
        return source.read_bytes()
    scratch.mkdir(parents=True, exist_ok=True)
    copy = scratch / name
    shutil.copy2(source, copy)
    run(runner, ["codesign", "--remove-signature", str(copy)], what=f"codesign --remove-signature {name}")
    return copy.read_bytes()


def rebuild_and_compare(runner: Runner, worktree: Path, asset: Path, build: dict, folder: Path) -> str | None:
    """Step 7. Returns the note `rebuild skipped: ...` when this Mac has another toolchain, else None."""
    xcode, toolchain = setup.xcode_and_toolchain()
    if (xcode, toolchain) != (build.get("xcode"), build.get("metal_toolchain")):
        return (f"rebuild skipped: the release was built with {build.get('xcode')}, "
                f"{build.get('metal_toolchain')}; this Mac has {xcode}, {toolchain}")

    rebuild_out = folder / "rebuild"
    run(runner, [sys.executable, str(worktree / "tools" / "metal" / "make_release.py"), "--out", str(rebuild_out)],
        cwd=worktree, what="make_release.py")
    release_dir = extract_asset(asset, folder / "release")
    rebuilt_dir = extract_asset(rebuild_out / release_metadata.ASSET_NAME, folder / "rebuilt-members")
    strip = developer_id_signed(runner, release_dir / "strata")
    for name, _ in release_metadata.ASSET_MEMBERS:
        released = member_bytes(runner, release_dir, name, folder / "unsigned-release", strip)
        rebuilt = member_bytes(runner, rebuilt_dir, name, folder / "unsigned-rebuilt", strip)
        if released != rebuilt:
            raise VerifyError(f"the rebuilt {name} differs from the release's; compare the SHA-256 values in "
                              f"{release_metadata.SUMS_NAME}")
    return None


def verify(tag: str, repo: str, *, rebuild: bool = True, runner: Runner = subprocess.run,
           installer: Installer | None = None, root: Path = ROOT) -> tuple[str, str | None]:
    """Steps 1 to 8. Returns (status, note): status is "ok", or "ok (rebuild skipped)" with a note; raises VerifyError
    at the first failed step. The draft is never published here."""
    if not TAG_SHAPE.fullmatch(tag):
        raise VerifyError(f"{tag!r} is not a Metal release tag (v<version>-metal.<revision>)")
    folder = Path(tempfile.mkdtemp(prefix="verify-release-"))
    worktree = folder / "wt"
    made_worktree = False
    try:
        download(runner, tag, repo, folder)
        check_sums(runner, folder)
        run(runner, ["git", "worktree", "add", "--detach", str(worktree), tag], cwd=root,
            what=f"git worktree add {tag}")
        made_worktree = True
        run(runner, ["cp", "-cR", str(root / "third_party" / "llama.cpp"),
                     str(worktree / "third_party" / "llama.cpp")], what="cp -cR third_party/llama.cpp")
        asset = folder / release_metadata.ASSET_NAME
        symbols = folder / release_metadata.SYMBOLS_NAME
        run(runner, [sys.executable, str(worktree / "tools" / "metal" / "check_asset.py"), str(asset),
                     "--symbols", str(symbols), "--run"], cwd=worktree, what="check_asset.py")
        install_check(folder, installer)
        acceptance(runner, worktree)
        if not rebuild:
            return "ok (rebuild skipped)", "rebuild skipped: --no-rebuild"
        with zipfile.ZipFile(asset) as archive:
            build = json.loads(archive.read("BUILD.json"))
        note = rebuild_and_compare(runner, worktree, asset, build, folder)
        if note:
            return "ok (rebuild skipped)", note
        return "ok", None
    finally:
        if made_worktree:
            runner(["git", "worktree", "remove", "--force", str(worktree)], cwd=root,
                   capture_output=True, text=True, check=False)
        shutil.rmtree(folder, ignore_errors=True)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tag", help="the draft's tag, for example v0.1.34-metal.1")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY") or release_metadata.RELEASE_REPOSITORY,
                        help="owner/name of the release repository")
    parser.add_argument("--no-rebuild", action="store_true", help="skip step 7")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.repo:
        print("verify_release: give --repo owner/name (RELEASE_REPOSITORY is not set)", file=sys.stderr)
        return 1
    try:
        status, note = verify(args.tag, args.repo, rebuild=not args.no_rebuild)
    except VerifyError as exc:
        print(f"verify_release: {exc}", file=sys.stderr)
        return 1
    if note:
        print(note)
    print(f"verify_release: {args.tag} {status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
