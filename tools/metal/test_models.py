#!/usr/bin/env python3
"""Fetch and verify the test models of docs/mac/09-testing-and-validation.md, section 2.3.

    test_models.py check [--quick] [--models DIR] [--manifest FILE] [TAG ...]
    test_models.py fetch [--plan] [--models DIR] [--manifest FILE] TAG [TAG ...]

`check` compares every model file's size and SHA-256 with tests/metal/models.json. `check --quick` compares the
size and the SHA-256 of the first and last 64 MiB of each file, and reads nothing else. A null value in the manifest
is not pinned, so it fails: nothing passes against a value the repository does not record.

The models folder is STRATA_TEST_MODELS, or /Users/Shared/strata-models when that is unset; --models overrides it.

Exit codes: 0 every checked file passed; 1 a file differs from its pin, a pin is null, or fetch cannot run; 2 the
models folder or a model file is missing, or the command line is wrong.

This is a command-line tool. Its unit tests are in test_test_models.py, which the runner discovers instead.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "tests" / "metal" / "models.json"
MODELS_ENV = "STRATA_TEST_MODELS"
DEFAULT_MODELS = Path("/Users/Shared/strata-models")
SCHEMA = 1
MIB = 1 << 20
QUICK_BYTES = 64 * MIB
CHUNK_BYTES = 8 * MIB
ROLES = ("shard1", "shard2", "mmproj")
PIN_KEYS = ("sha256", "head_64mib_sha256", "tail_64mib_sha256")
HEX64 = re.compile(r"[0-9a-f]{64}")
EXIT_PASS, EXIT_FAIL, EXIT_MISSING = 0, 1, 2


class ManifestError(ValueError):
    """models.json does not have the shape this tool reads."""


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ManifestError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _validate_file(where: str, entry: object) -> None:
    if not isinstance(entry, dict):
        raise ManifestError(f"{where}: each file must be an object")
    if entry.get("role") not in ROLES:
        raise ManifestError(f"{where}: role must be one of {ROLES}")
    name = entry.get("path")
    if not isinstance(name, str) or not name or "/" in name:
        raise ManifestError(f"{where}: path must be a file name inside the model folder")
    size = entry.get("bytes")
    if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
        raise ManifestError(f"{where}: {name}: bytes must be a positive integer")
    source = entry.get("source")
    if not isinstance(source, str) or not source.startswith("https://"):
        raise ManifestError(f"{where}: {name}: source must be an https URL")
    for key in PIN_KEYS:
        if key not in entry:
            raise ManifestError(f"{where}: {name}: {key} is missing (use null when not pinned)")
        value = entry[key]
        if value is not None and not (isinstance(value, str) and HEX64.fullmatch(value)):
            raise ManifestError(f"{where}: {name}: {key} must be null or 64 lowercase hex digits")


def load_manifest(path: Path = MANIFEST) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_unique_object)
    if not isinstance(data, dict) or data.get("schema") != SCHEMA:
        raise ManifestError(f"{path}: schema must be {SCHEMA}")
    models = data.get("models")
    if not isinstance(models, dict) or not models:
        raise ManifestError(f"{path}: models must be a non-empty object")
    for tag, model in models.items():
        files = model.get("files") if isinstance(model, dict) else None
        if not isinstance(files, list) or not files:
            raise ManifestError(f"{path}: {tag}: files must be a non-empty list")
        for entry in files:
            _validate_file(f"{path}: {tag}", entry)
    return data


class Digester:
    """SHA-256 of byte ranges of model files, computed once per run for each inode.

    setup hard-links the one PLE shard that ISTA and Coder share into every folder that needs it, so the
    five folders that hold it are read once.
    """

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int, int, int], str] = {}

    def range_sha256(self, path: Path, start: int, stop: int) -> str:
        info = path.stat()
        key = (info.st_dev, info.st_ino, start, stop)
        if key not in self._cache:
            digest = hashlib.sha256()
            remaining = stop - start
            with path.open("rb") as handle:
                handle.seek(start)
                while remaining > 0:
                    block = handle.read(min(CHUNK_BYTES, remaining))
                    if not block:
                        raise OSError(f"{path}: shorter than the size reported by stat")
                    digest.update(block)
                    remaining -= len(block)
            self._cache[key] = digest.hexdigest()
        return self._cache[key]


@dataclass(frozen=True)
class Outcome:
    label: str
    status: str  # "pass", "fail" or "missing"
    detail: str


def verify_file(model_folder: Path, tag: str, entry: dict, *, quick: bool, digests: Digester) -> Outcome:
    label = f"{tag}/{entry['path']}"
    path = model_folder / entry["path"]
    if not path.is_file():
        return Outcome(label, "missing", f"absent: {path}")
    keys = ("head_64mib_sha256", "tail_64mib_sha256") if quick else ("sha256",)
    unpinned = [key for key in keys if entry[key] is None]
    if unpinned:
        return Outcome(label, "fail", f"{', '.join(unpinned)} is null in models.json: not pinned")
    size = path.stat().st_size
    if size != entry["bytes"]:
        return Outcome(label, "fail", f"size {size:,} B, want {entry['bytes']:,} B")
    if quick:
        head = digests.range_sha256(path, 0, min(QUICK_BYTES, size))
        if head != entry["head_64mib_sha256"]:
            return Outcome(label, "fail", f"first 64 MiB sha256 {head}, want {entry['head_64mib_sha256']}")
        tail = digests.range_sha256(path, max(0, size - QUICK_BYTES), size)
        if tail != entry["tail_64mib_sha256"]:
            return Outcome(label, "fail", f"last 64 MiB sha256 {tail}, want {entry['tail_64mib_sha256']}")
        return Outcome(label, "pass", "size, first and last 64 MiB")
    whole = digests.range_sha256(path, 0, size)
    if whole != entry["sha256"]:
        return Outcome(label, "fail", f"sha256 {whole}, want {entry['sha256']}")
    return Outcome(label, "pass", "size and sha256")


def run_check(manifest: dict, models_root: Path, tags: list[str], *, quick: bool) -> int:
    if not models_root.is_dir():
        print(f"test_models: models folder is missing: {models_root} (set {MODELS_ENV} or pass --models)",
              file=sys.stderr)
        return EXIT_MISSING
    digests = Digester()
    outcomes = []
    for tag in tags:
        for entry in manifest["models"][tag]["files"]:
            outcomes.append(verify_file(models_root / tag, tag, entry, quick=quick, digests=digests))
    missing = [o for o in outcomes if o.status == "missing"]
    failed = [o for o in outcomes if o.status == "fail"]
    passed = len(outcomes) - len(missing) - len(failed)
    mode = "check --quick" if quick else "check"
    summary = f"test_models {mode}: {passed}/{len(outcomes)} files passed"
    if missing:
        summary += f", {len(missing)} missing"
    if failed:
        summary += f", {len(failed)} failed"
    for outcome in outcomes:
        if outcome.status == "pass":
            print(f"PASS {outcome.label} ({outcome.detail})")
    for outcome in missing + failed:
        print(f"{outcome.status.upper()} {outcome.label}: {outcome.detail}", file=sys.stderr)
    if missing:
        print(summary, file=sys.stderr)
        return EXIT_MISSING
    if failed:
        print(summary, file=sys.stderr)
        return EXIT_FAIL
    print(summary)
    return EXIT_PASS


def run_fetch(manifest: dict, models_root: Path, tags: list[str], *, plan: bool) -> int:
    if plan:
        for tag in tags:
            for entry in manifest["models"][tag]["files"]:
                print(f"{tag}: {entry['path']} {entry['bytes']:,} B from {entry['source']}")
        return EXIT_PASS
    if not models_root.is_dir():
        print(f"test_models: models folder is missing: {models_root} (set {MODELS_ENV} or pass --models)",
              file=sys.stderr)
        return EXIT_MISSING
    # setup.py is the fail-closed stub in this checkout. Its step 6 (download, pack, MTP and config.json,
    # 07-server-setup-port.md section 4.8) is not implemented, and the spec does not name the function that
    # would provide it, so fetch refuses rather than downloading by a path no reviewer has checked.
    print("test_models fetch: setup.py has no step 6 in this checkout, so no model was fetched "
          "(use --plan to list the sources)", file=sys.stderr)
    return EXIT_FAIL


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="test_models.py", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("check", help="compare model files with the pins in models.json")
    check.add_argument("--quick", action="store_true",
                       help="size and the first and last 64 MiB of each file, instead of the whole file")
    check.add_argument("--models", type=Path, help="models folder (default: $STRATA_TEST_MODELS)")
    check.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    check.add_argument("tags", nargs="*", help="model tags (default: every model)")
    fetch = commands.add_parser("fetch", help="download the models of section 2.3 (not in this checkout)")
    fetch.add_argument("--plan", action="store_true", help="list the sources and sizes, and touch nothing")
    fetch.add_argument("--models", type=Path, help="models folder (default: $STRATA_TEST_MODELS)")
    fetch.add_argument("--manifest", type=Path, default=MANIFEST, help=argparse.SUPPRESS)
    fetch.add_argument("tags", nargs="+", help="model tags")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        manifest = load_manifest(args.manifest)
    except OSError as error:
        print(f"test_models: cannot read {args.manifest}: {error}", file=sys.stderr)
        return EXIT_MISSING
    except ValueError as error:
        print(f"test_models: {error}", file=sys.stderr)
        return EXIT_FAIL
    tags = list(args.tags) or list(manifest["models"])
    unknown = [tag for tag in tags if tag not in manifest["models"]]
    if unknown:
        parser.error(f"unknown model tag(s): {', '.join(unknown)}; models.json has {', '.join(manifest['models'])}")
    models_root = args.models or Path(os.environ.get(MODELS_ENV) or DEFAULT_MODELS)
    if args.command == "check":
        return run_check(manifest, models_root, tags, quick=args.quick)
    return run_fetch(manifest, models_root, tags, plan=args.plan)


if __name__ == "__main__":
    raise SystemExit(main())
