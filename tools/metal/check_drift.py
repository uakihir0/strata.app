#!/usr/bin/env python3
"""Compare classified upstream interfaces and project markers (docs/mac/06, section 1.3)."""

from __future__ import annotations

import argparse
import ast
import difflib
import fnmatch
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
UPSTREAM_URL = "https://github.com/Niko1221/Strata"
CPP_SUFFIXES = {".c", ".cc", ".cpp", ".cxx", ".cu", ".cuh", ".h", ".hh", ".hpp", ".hxx"}
KERNEL_SUFFIXES = {".cu", ".cuh"}
OUTPUT_CALL = re.compile(
    r"\b(?:std::)?(?:printf|fprintf)\s*\([^;]*?\"((?:\\.|[^\"\\])*)\"",
    re.DOTALL,
)
FLAG = re.compile(r'\ba\s*==\s*"(-{1,2}[^"\s]+)"')
ENV_READ = re.compile(r'\b(?:std::)?getenv\s*\(\s*"((?:STRATA_)[A-Za-z0-9_]*)"')
STRING_LITERAL = re.compile(r'"((?:\\.|[^"\\])*)"')
OUTPUT_KEY = re.compile(r"(?<!\S)([A-Za-z_][A-Za-z0-9_]*)=")
PYTHON_BRANCH = re.compile(r"\b(?:sys\.platform|os\.name|subprocess(?:\.[A-Za-z_]\w*)?)\b")


def git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        text=True,
        capture_output=True,
        check=check,
    )


def commit_exists(root: Path, commit: str) -> bool:
    result = git(root, "cat-file", "-e", f"{commit}^{{commit}}", check=False)
    return result.returncode == 0


def ensure_commit(root: Path, commit: str) -> None:
    if commit_exists(root, commit):
        return
    result = git(root, "fetch", "--no-tags", UPSTREAM_URL, commit, check=False)
    if result.returncode or not commit_exists(root, commit):
        detail = result.stderr.strip() or "the object is still unavailable after fetch"
        raise RuntimeError(f"cannot fetch upstream commit {commit}: {detail}")


def upstream_file(root: Path, commit: str, path: str) -> bytes:
    result = git(root, "show", f"{commit}:{path}", check=False)
    if result.returncode:
        raise RuntimeError(f"{commit} does not contain {path}")
    return result.stdout.encode("utf-8")


def tree_paths(root: Path, commit: str) -> list[str]:
    result = git(root, "ls-tree", "-r", "--name-only", commit)
    return result.stdout.splitlines()


def source_text(root: Path, commit: str, path: str) -> str:
    return upstream_file(root, commit, path).decode("utf-8", errors="replace")


def parser_bounds(root: Path) -> tuple[int, int]:
    path = root / "docs/mac/06-engine-host-port.md"
    text = path.read_text(encoding="utf-8")
    match = re.search(r"Argument parser\s*\|\s*`generate\.cpp:(\d+)-(\d+)`", text)
    if not match:
        raise RuntimeError("docs/mac/06-engine-host-port.md is missing the parser line range")
    first, last = map(int, match.groups())
    if first < 1 or last < first:
        raise RuntimeError(f"invalid parser line range {first}-{last}")
    return first, last


def assigned_literal(source: str, name: str) -> Any:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                value = ast.literal_eval(node.value)
                if isinstance(value, dict):
                    return value
                raise RuntimeError(f"setup.py {name} must be a literal dictionary")
    raise RuntimeError(f"setup.py does not define {name}")


def normalise_c_string(value: str) -> str:
    return (
        value.replace(r"\n", "\n")
        .replace(r"\r", "\r")
        .replace(r"\t", "\t")
        .replace(r"\"", '"')
        .replace(r"\\", "\\")
    )


def emitted_strings(source: str) -> list[str]:
    return [normalise_c_string(match.group(1)) for match in OUTPUT_CALL.finditer(source)]


def copy_hash(root: Path, commit: str, source_path: str, line_range: str) -> str:
    match = re.fullmatch(r"([1-9][0-9]*)-([1-9][0-9]*)", line_range)
    if not match:
        raise RuntimeError(f"invalid copied line range {line_range!r}")
    first, last = map(int, match.groups())
    if last < first:
        raise RuntimeError(f"invalid copied line range {line_range!r}")
    text = source_text(root, commit, source_path).replace("\r\n", "\n").replace("\r", "\n")
    lines = text.splitlines()
    if last > len(lines):
        raise RuntimeError(f"copied range {source_path}:{line_range} exceeds {len(lines)} lines")
    payload = ("\n".join(lines[first - 1 : last]) + "\n").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def snapshot(root: Path, commit: str, template: dict[str, Any]) -> dict[str, Any]:
    paths = tree_paths(root, commit)
    source_paths = [
        path for path in paths
        if (path.startswith("src/") or path.startswith("include/"))
        and Path(path).suffix in CPP_SUFFIXES
    ]
    source_contents = {path: source_text(root, commit, path) for path in source_paths}

    parser_start, parser_end = parser_bounds(root)
    parser_lines = source_text(root, commit, "src/program/generate.cpp").splitlines()
    if parser_end > len(parser_lines):
        raise RuntimeError(
            f"parser range {parser_start}-{parser_end} exceeds {len(parser_lines)} lines"
        )
    parser_text = "\n".join(parser_lines[parser_start - 1 : parser_end])
    flags = sorted(set(FLAG.findall(parser_text)))

    env = sorted(
        {
            match.group(1)
            for text in source_contents.values()
            for match in ENV_READ.finditer(text)
        }
    )

    info_keys: set[str] = set()
    err_strings: set[str] = set()
    done_fields: set[str] = set()
    for text in source_contents.values():
        for emitted in emitted_strings(text):
            first_line = emitted.splitlines()[0] if emitted else ""
            if first_line.startswith("INFO "):
                info_keys.update(OUTPUT_KEY.findall(first_line[len("INFO ") :]))
            elif first_line.startswith("ERR "):
                err_strings.add(first_line)
            elif first_line.startswith("DONE "):
                done_fields.add(first_line)

    setup_source = source_text(root, commit, "setup.py")
    models = assigned_literal(setup_source, "MODELS")
    revisions = assigned_literal(setup_source, "HF_REVISIONS")

    cuda_kernels = {}
    for path in paths:
        if path.startswith("src/kernels/") and Path(path).suffix in KERNEL_SUFFIXES:
            cuda_kernels[path] = hashlib.sha256(upstream_file(root, commit, path)).hexdigest()

    markers = {}
    for path in template.get("markers", {}):
        file_path = root / path
        try:
            text = file_path.read_text(encoding="utf-8")
        except OSError:
            text = ""
        markers[path] = sum("MAC:" in line for line in text.splitlines())

    copies = {}
    for path, definition in template.get("copies", {}).items():
        if not isinstance(definition, dict):
            raise RuntimeError(f"copy definition for {path} must be an object")
        source_path = definition.get("from")
        line_range = definition.get("range")
        if not isinstance(source_path, str) or not isinstance(line_range, str):
            raise RuntimeError(f"copy definition for {path} needs from and range")
        copies[path] = {
            "from": source_path,
            "range": line_range,
            "sha256": copy_hash(root, commit, source_path, line_range),
        }

    return {
        "markers": markers,
        "copies": copies,
        "flags": flags,
        "env": env,
        "info_keys": sorted(info_keys),
        "err_strings": sorted(err_strings),
        "done_fields": sorted(done_fields),
        "cuda_kernels": dict(sorted(cuda_kernels.items())),
        "models": sorted(models),
        "revisions": dict(sorted(revisions.items())),
    }


def changed_files(root: Path, base: str, target: str) -> list[str]:
    return git(root, "diff", "--name-only", f"{base}..{target}").stdout.splitlines()


def no_arithmetic(path: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatchcase(path, pattern) for pattern in patterns)


def golden_verdict(files: list[str], patterns: list[str]) -> str:
    numerical = [
        path for path in files
        if (path.startswith("src/") or path.startswith("include/"))
        and not no_arithmetic(path, patterns)
    ]
    return "record again" if numerical else "still valid"


def python_changes_needing_review(root: Path, base: str, target: str, template: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    marker_paths = [
        path for path in template.get("markers", {})
        if path.endswith(".py")
    ]
    for path in marker_paths:
        if path not in git(root, "ls-tree", "-r", "--name-only", target).stdout.splitlines():
            continue
        old = source_text(root, base, path).splitlines()
        new = source_text(root, target, path).splitlines()
        project_path = root / path
        try:
            project_lines = project_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            project_lines = []
        marker_lines = [
            index + 1 for index, line in enumerate(project_lines) if "# MAC:" in line
        ]
        matcher = difflib.SequenceMatcher(a=old, b=new, autojunk=False)
        for tag, old_start, old_end, new_start, new_end in matcher.get_opcodes():
            if tag == "equal":
                continue
            if new_start < new_end:
                changed_span = range(new_start + 1, new_end + 1)
            elif old_start < old_end:
                changed_span = range(old_start + 1, old_end + 1)
            else:
                changed_span = ()
            for changed_line in changed_span:
                if any(abs(changed_line - marker_line) <= 1 for marker_line in marker_lines):
                    issues.append(
                        f"Python change near # MAC: marker in {path}:{changed_line}"
                    )
                    break

        diff = git(root, "diff", "--unified=0", f"{base}..{target}", "--", path).stdout
        for line in diff.splitlines():
            if line.startswith("+") and not line.startswith("+++") and PYTHON_BRANCH.search(line):
                issues.append(f"new subprocess/platform branch in {path}: {line[1:].strip()}")
    return sorted(set(issues))


def compare_snapshots(expected: dict[str, Any], actual: dict[str, Any]) -> list[str]:
    issues: list[str] = []
    list_fields = {
        "flags": "flag",
        "env": "environment variable",
        "info_keys": "INFO key",
        "err_strings": "ERR string",
        "done_fields": "DONE field sequence",
        "models": "model",
    }
    for field, label in list_fields.items():
        old = set(expected.get(field, []))
        new = set(actual.get(field, []))
        for value in sorted(new - old):
            issues.append(f"new {label}: {value}")
        for value in sorted(old - new):
            issues.append(f"removed {label}: {value}")

    for field, label in (
        ("markers", "marker count"),
        ("copies", "copied range"),
        ("cuda_kernels", "CUDA kernel"),
        ("revisions", "model revision"),
    ):
        old = expected.get(field, {})
        new = actual.get(field, {})
        if not isinstance(old, dict) or not isinstance(new, dict):
            issues.append(f"{field} must be an object")
            continue
        for key in sorted(old.keys() | new.keys()):
            if old.get(key) == new.get(key):
                continue
            if field == "markers":
                issues.append(
                    f"{key}: marker count expected {old.get(key, 0)}, found {new.get(key, 0)}"
                )
            elif field == "copies":
                issues.append(f"{label} changed: {key}")
            elif field == "cuda_kernels":
                state = "added" if key not in old else "removed" if key not in new else "changed"
                issues.append(f"{label} {state}: {key}")
            else:
                issues.append(f"{label} changed: {key}")
    return issues


def write_drift(path: Path, commit: str, current: dict[str, Any], no_arithmetic_paths: list[str]) -> None:
    value = {
        "schema": 1,
        "classified_at": commit,
        **current,
        "no_arithmetic": no_arithmetic_paths,
    }
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def main(root: Path = ROOT, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--update", action="store_true", help="record the resolved upstream snapshot")
    parser.add_argument("--root", type=Path, default=root, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    drift_path = root / "tools/metal/drift.json"

    try:
        sync = (root / "UPSTREAM_SYNC").read_text(encoding="ascii").strip()
        if not re.fullmatch(r"[0-9a-fA-F]{40}", sync):
            raise RuntimeError("UPSTREAM_SYNC must contain one 40-hex commit")
        ensure_commit(root, sync)
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        if drift.get("schema") != 1:
            raise RuntimeError("drift.json schema must be 1")
        classified = drift.get("classified_at")
        if not isinstance(classified, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", classified):
            raise RuntimeError("drift.json classified_at must contain a 40-hex commit")
        ensure_commit(root, classified)
        no_arithmetic_paths = drift.get("no_arithmetic", [])
        if not isinstance(no_arithmetic_paths, list) or any(
            not isinstance(path, str) for path in no_arithmetic_paths
        ):
            raise RuntimeError("drift.json no_arithmetic must be a list of path globs")
        actual = snapshot(root, sync, drift)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"check_drift: {exc}", file=sys.stderr)
        return 1

    if args.update:
        write_drift(drift_path, sync, actual, no_arithmetic_paths)
        print(f"check_drift: updated classification at {sync}")
        return 0

    issues = compare_snapshots(drift, actual)
    for issue in python_changes_needing_review(root, classified, sync, drift):
        issues.append(issue)
    for issue in issues:
        print(f"check_drift: {issue}", file=sys.stderr)

    try:
        files = changed_files(root, classified, sync)
    except subprocess.CalledProcessError as exc:
        print(f"check_drift: cannot compare classified commits: {exc.stderr.strip()}", file=sys.stderr)
        return 1
    if files:
        for path in files:
            print(f"upstream changed: {path}")
    else:
        print("upstream changed: none")
    verdict = golden_verdict(files, no_arithmetic_paths)
    print(f"golden: {verdict}")
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
