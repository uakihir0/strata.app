#!/usr/bin/env python3
"""Check the documented CTest, requirement and review maps (docs/mac/09, section 9.4)."""

from __future__ import annotations

import argparse
import ast
import fnmatch
import ipaddress
import json
import os
import re
import subprocess
import sys
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit


ROOT = Path(__file__).resolve().parents[2]
DOC_TESTS = ROOT / "docs/mac/09-testing-and-validation.md"
DOC_REQUIREMENTS = ROOT / "docs/mac/01-requirements.md"
TEST_MAP = ROOT / "tests/metal/test_map.json"
REQUIREMENTS = ROOT / "tests/metal/requirements.json"
REVIEWS = ROOT / "tests/metal/reviews.json"
RUNS = [
    "c1", "c2", "c3", "c4", "pp", "needle", "server", "bench",
    "pf", "g1", "g2", "g3", "nightly", "weekly", "release", "family",
]
REQUIREMENT_ID = re.compile(r"^\| (REQ-[A-Z]{3}-[0-9]{2}) \|")
CRITERION_ID = re.compile(r"\b[A-Z][A-Z0-9_-]*(?:\.[A-Za-z0-9_-]+)+\b")
WORKFLOW_JOB_ID = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
PYTHON_PATH = re.compile(r"^[\w./*?-]+\.py$")
DNS_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
BAD_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
BLOCK_SCALAR_VALUE = re.compile(
    r"^(?:(?:&[A-Za-z0-9_-]+|![^\s]+)\s+)*[|>](?:[+-]?[1-9]?|[1-9][+-]?)?$"
)


def schema_is_one(value: Any) -> bool:
    return type(value) is int and value == 1


def path_is_within_root(root: Path, path: Path) -> bool:
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except (OSError, RuntimeError, ValueError):
        return False
    return True


def valid_https_url(value: Any) -> bool:
    if (
        not isinstance(value, str)
        or not value
        or re.search(r"\s", value)
        or any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in value)
    ):
        return False
    if BAD_PERCENT_ESCAPE.search(value):
        return False
    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if (
        parsed.scheme.lower() != "https"
        or not hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.netloc.endswith(":")
    ):
        return False
    if port is not None and not (1 <= port <= 65535):
        return False
    if ":" in hostname:
        try:
            ipaddress.IPv6Address(hostname)
        except ValueError:
            return False
        return parsed.netloc.startswith("[")
    try:
        ascii_host = hostname.encode("idna").decode("ascii")
    except UnicodeError:
        return False
    if ascii_host.endswith("..") or len(ascii_host.rstrip(".")) > 253:
        return False
    labels = ascii_host.rstrip(".").split(".")
    return bool(labels) and all(DNS_LABEL.fullmatch(label) for label in labels)


def read_json(path: Path, errors: list[str]) -> dict[str, Any]:
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError(f"duplicate JSON key {key!r}")
            value[key] = item
        return value

    try:
        display_path = path.relative_to(ROOT)
    except ValueError:
        display_path = path
    try:
        value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_object)
    except (OSError, ValueError) as exc:
        errors.append(f"test_map: cannot read {display_path}: {exc}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"test_map: {display_path} must contain a JSON object")
        return {}
    return value


def section(text: str, start: str, end: str) -> str:
    begin = text.find(start)
    finish = text.find(end, begin + len(start)) if begin >= 0 else -1
    if begin < 0 or finish < 0:
        raise ValueError(f"documentation is missing {start!r} or {end!r}")
    return text[begin:finish]


def documented_tests(text: str) -> tuple[dict[str, str], dict[str, int]]:
    block = section(text, "### 9.1 CTest tests", "### 9.2 Tests outside CTest")
    names: dict[str, str] = {}
    table_start = re.search(r"^\*\*`host` \(\d+\)\.\*\*$", block, re.MULTILINE)
    if table_start is None:
        raise ValueError("section 9.1 is missing the host test table")
    label = None
    table_heading = re.compile(r"^\*\*`(host|metal|model)` \(\d+\)\.\*\*$")
    for line in block[table_start.start() :].splitlines():
        heading = table_heading.fullmatch(line)
        if heading:
            label = heading.group(1)
            continue
        if not line.startswith("|"):
            continue
        cells = line.split("|")
        if len(cells) < 3 or label is None:
            continue
        first_cell = cells[1]
        for name in re.findall(r"`([^`]+)`", first_cell):
            if name in names:
                raise ValueError(f"section 9.1 lists {name} more than once")
            names[name] = label

    intro = block.split("**`host` (", 1)[0]
    heading = re.search(
        r"registers\s+\d+\s+tests:\s*(\d+)\s+`host`,\s*"
        r"(\d+)\s+`metal`\s+and\s+(\d+)\s+`model`",
        intro,
    )
    if heading is None:
        raise ValueError("section 9.1 is missing its declared CTest counts")
    return names, {
        "host": int(heading.group(1)),
        "metal": int(heading.group(2)),
        "model": int(heading.group(3)),
    }


def documented_python_paths(text: str) -> list[str]:
    block = section(text, "### 9.3 The Python layer", "### 9.4 `tools/metal/test_map.py`")
    paths: list[str] = []
    for candidate in re.findall(r"`([^`]+)`", block):
        if PYTHON_PATH.fullmatch(candidate):
            paths.append(candidate)
    return paths


def documented_requirement_rows(text: str, errors: list[str]) -> dict[str, tuple[bool, str]]:
    rows: dict[str, tuple[bool, str]] = {}
    for line in text.splitlines():
        match = REQUIREMENT_ID.match(line)
        if match:
            cells = line.split("|")
            req_id = match.group(1)
            if req_id in rows:
                errors.append(f"test_map: {req_id}: duplicated in docs/mac/01-requirements.md")
            requirement = cells[2].strip() if len(cells) > 2 else ""
            verified_by = cells[3].strip() if len(cells) > 3 else ""
            rows[req_id] = ("Withdrawn " in requirement, verified_by)
    return rows


def documented_requirement_checks(
    root: Path,
    req_id: str,
    verified_by: str,
    test_doc: str,
    errors: list[str],
) -> set[str] | str:
    if verified_by == "withdrawn":
        return "withdrawn"
    if req_id != "REQ-SRV-12":
        return set(re.findall(r"`((?:ctest|accept|python|ci|review):[^`]+)`", verified_by))

    try:
        block = section(test_doc, "### 9.3 The Python layer", "### 9.4 `tools/metal/test_map.py`")
    except ValueError as exc:
        errors.append(f"test_map: {exc}")
        return set()
    row = next((line for line in block.splitlines() if "Mac-supported Python modules" in line), None)
    if row is None:
        errors.append("test_map: section 9.3 is missing the Mac Python module row")
        return set()
    cells = row.split("|")
    module_cell = cells[1] if len(cells) > 1 else ""
    patterns = re.findall(r"`([^`]+\.py)`", module_cell)
    declared_counts = [
        int(count)
        for count in re.findall(r"\((\d+)(?: modules?)?\)", module_cell)
    ]
    separately_documented = {
        path
        for path in documented_python_paths(test_doc)
        if not any(character in path for character in "*?")
    }
    checks: set[str] = set()
    actual_counts: list[int] = []
    for pattern in patterns:
        matches = sorted(
            candidate
            for candidate in root.glob(pattern)
            if candidate.is_file() and path_is_within_root(root, candidate)
            and candidate.relative_to(root).as_posix() not in separately_documented
        )
        actual_counts.append(len(matches))
        checks.update(
            "python:" + candidate.relative_to(root).as_posix()
            for candidate in matches
        )
    if not patterns or len(declared_counts) != len(patterns):
        errors.append("test_map: Mac Python module row must declare each pattern's module count")
    elif declared_counts != actual_counts:
        errors.append(
            "test_map: Mac Python module counts in section 9.3 do not match the files: "
            f"documented {declared_counts}, found {actual_counts}"
        )
    return checks


def path_matches_pattern(path: str, pattern: str) -> bool:
    path_parts = Path(path).parts
    pattern_parts = Path(pattern).parts
    if len(path_parts) != len(pattern_parts):
        return False
    return all(
        fnmatch.fnmatchcase(path_part, pattern_part)
        for path_part, pattern_part in zip(path_parts, pattern_parts)
    )


def _target_names(target: ast.expr) -> set[str]:
    if isinstance(target, ast.Name):
        return {target.id}
    if isinstance(target, (ast.Tuple, ast.List)):
        return set().union(*(_target_names(item) for item in target.elts))
    return set()


def _literal_criterion_id(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Constant)
        and isinstance(node.value, str)
        and CRITERION_ID.fullmatch(node.value)
    ):
        return node.value
    return None


def _criteria_table_name(name: str) -> bool:
    normalized = name.upper()
    return normalized in {
        "CRITERIA",
        "CRITERIA_BY_ID",
        "CRITERIA_ROWS",
        "CRITERIA_TABLE",
        "CRITERION_ROWS",
        "CRITERION_TABLE",
        "CRITERION_DEFINITIONS",
    } or normalized.endswith(
        ("_CRITERIA", "_CRITERIA_TABLE", "_CRITERION_ROWS", "_CRITERION_TABLE")
    )


def _call_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _criterion_row_id(row: ast.expr) -> str | None:
    if isinstance(row, (ast.Tuple, ast.List)) and len(row.elts) >= 4:
        return _literal_criterion_id(row.elts[0])
    if isinstance(row, ast.Dict):
        fields = {
            key.value: value
            for key, value in zip(row.keys, row.values)
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        criterion_id = fields.get("id") or fields.get("criterion") or fields.get("name")
        has_operator = "op" in fields or "operator" in fields
        return (
            _literal_criterion_id(criterion_id)
            if has_operator and {"threshold", "gating"} <= fields.keys()
            else None
        )
    if isinstance(row, ast.Call) and _call_name(row.func) in {
        "Criterion",
        "CriterionDefinition",
        "CriterionRow",
        "CriterionSpec",
    }:
        keyword_values = {keyword.arg: keyword.value for keyword in row.keywords if keyword.arg}
        criterion_id = (
            _literal_criterion_id(row.args[0])
            if row.args
            else _literal_criterion_id(
                keyword_values.get("id")
                or keyword_values.get("criterion")
                or keyword_values.get("name")
            )
        )
        positional_fields = max(0, len(row.args) - 1)
        keyword_fields = set(keyword_values)
        has_operator = positional_fields >= 1 or "op" in keyword_fields or "operator" in keyword_fields
        has_threshold = positional_fields >= 2 or "threshold" in keyword_fields
        has_gating = positional_fields >= 3 or "gating" in keyword_fields
        if criterion_id and has_operator and has_threshold and has_gating:
            return criterion_id
    return None


def _criterion_value_has_fields(row: ast.expr) -> bool:
    if isinstance(row, (ast.Tuple, ast.List)):
        return len(row.elts) >= 3
    if isinstance(row, ast.Dict):
        fields = {
            key.value
            for key in row.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        }
        return (
            ("op" in fields or "operator" in fields)
            and {"threshold", "gating"} <= fields
        )
    if isinstance(row, ast.Call) and _call_name(row.func) in {
        "Criterion",
        "CriterionDefinition",
        "CriterionRow",
        "CriterionSpec",
    }:
        keyword_fields = {keyword.arg for keyword in row.keywords if keyword.arg}
        positional_fields = len(row.args)
        has_operator = positional_fields >= 1 or "op" in keyword_fields or "operator" in keyword_fields
        has_threshold = positional_fields >= 2 or "threshold" in keyword_fields
        has_gating = positional_fields >= 3 or "gating" in keyword_fields
        return has_operator and has_threshold and has_gating
    return False


def _criteria_table_ids(value: ast.expr) -> set[str]:
    ids: set[str] = set()
    if isinstance(value, ast.Dict):
        for key, row in zip(value.keys, value.values):
            key_id = _literal_criterion_id(key) if key is not None else None
            row_id = _criterion_row_id(row)
            if key_id is not None:
                if row_id is not None and row_id != key_id:
                    continue
                if row_id is None and not _criterion_value_has_fields(row):
                    continue
                criterion_id = key_id
            else:
                criterion_id = row_id
            if criterion_id is not None:
                ids.add(criterion_id)
    elif isinstance(value, (ast.List, ast.Tuple, ast.Set)):
        for row in value.elts:
            criterion_id = _criterion_row_id(row)
            if criterion_id is not None:
                ids.add(criterion_id)
    return ids


def declared_criterion_ids(source: str) -> set[str]:
    """Read IDs from top-level criterion rows, never from comments or prose strings."""
    tree = ast.parse(source)
    ids: set[str] = set()
    for statement in tree.body:
        if isinstance(statement, ast.Assign):
            targets = statement.targets
            value = statement.value
        elif isinstance(statement, ast.AnnAssign):
            targets = [statement.target]
            value = statement.value
        else:
            continue
        if value is None:
            continue
        names = _target_names(targets[0]) if len(targets) == 1 else set().union(
            *(_target_names(target) for target in targets)
        )
        if any(_criteria_table_name(name) for name in names):
            ids.update(_criteria_table_ids(value))
    return ids


def documented_review_ids(text: str) -> set[str]:
    try:
        block = section(text, "### 7.6 Reviews that need a person", "## 8.")
    except ValueError:
        return set()
    return set(re.findall(r"`review:([A-Za-z0-9_-]+)`", block))


def check_document_contract(
    test_map: dict[str, Any], errors: list[str]
) -> dict[str, dict[str, Any]]:
    if not schema_is_one(test_map.get("schema")):
        errors.append("test_map: tests/metal/test_map.json: schema must be 1")
    if test_map.get("runs") != RUNS:
        errors.append("test_map: tests/metal/test_map.json: runs do not match section 9.4")

    entries = test_map.get("tests")
    if not isinstance(entries, dict):
        errors.append("test_map: tests/metal/test_map.json: tests must be an object")
        return {}

    for name, entry in entries.items():
        if not isinstance(entry, dict):
            errors.append(f"test_map: {name}: entry must be an object")
            continue
        if entry.get("label") not in {"host", "metal", "model"}:
            errors.append(f"test_map: {name}: invalid or missing label")
        if not isinstance(entry.get("origin"), str) or not entry["origin"].strip():
            errors.append(f"test_map: {name}: missing origin")
        planned = entry.get("planned")
        if planned is not None and (not isinstance(planned, str) or not planned.strip()):
            errors.append(f"test_map: {name}: planned must name a work item")
    return entries


def check_ctest(
    root: Path,
    entries: dict[str, dict[str, Any]],
    final: bool,
    errors: list[str],
    notes: list[str],
) -> None:
    build_dir = os.environ.get("STRATA_BUILD_DIR")
    if not build_dir:
        message = "test_map: CTest check skipped (STRATA_BUILD_DIR is not set)"
        (errors if final else notes).append(message)
        return

    command = ["ctest", "--test-dir", build_dir, "--show-only=json-v1"]
    try:
        result = subprocess.run(
            command,
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        errors.append(f"test_map: cannot run ctest: {exc}")
        return
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        errors.append(f"test_map: ctest --show-only failed: {detail}")
        return
    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"test_map: ctest returned invalid JSON: {exc}")
        return

    actual: dict[str, str] = {}
    for test in payload.get("tests", []):
        name = test.get("name")
        if not isinstance(name, str):
            errors.append("test_map: ctest returned a test without a name")
            continue
        label = None
        for prop in test.get("properties", []):
            if prop.get("name") == "LABELS":
                value = prop.get("value")
                if isinstance(value, list):
                    label = ";".join(str(item) for item in value)
                elif value is not None:
                    label = str(value)
        if name in actual:
            errors.append(f"test_map: {name}: duplicate name in CTest")
        actual[name] = label or ""

    for name, label in actual.items():
        entry = entries.get(name)
        if entry is None:
            errors.append(f"test_map: {name}: in CTest but not in tests/metal/test_map.json")
            continue
        if entry.get("planned") is not None:
            errors.append(f'test_map: {name}: in CTest, remove "planned"')
        if label != entry.get("label"):
            errors.append(
                f"test_map: {name}: label {label}, the map says {entry.get('label')}"
            )

    for name, entry in entries.items():
        if name in actual:
            continue
        if entry.get("planned") is not None:
            message = f"test_map: {name}: planned ({entry['planned']})"
            (errors if final else notes).append(message)
        else:
            errors.append(f"test_map: {name}: in tests/metal/test_map.json but not in CTest")


def check_test_catalog(
    root: Path,
    text: str,
    entries: dict[str, dict[str, Any]],
    final: bool,
    errors: list[str],
    notes: list[str],
) -> None:
    try:
        documented_names, declared_counts = documented_tests(text)
    except ValueError as exc:
        errors.append(f"test_map: {exc}")
        return

    mapped_names = set(entries)
    documented_name_set = set(documented_names)
    for name in sorted(documented_name_set - mapped_names):
        errors.append(f"test_map: {name}: in docs/mac/09-testing-and-validation.md but not in test_map.json")
    for name in sorted(mapped_names - documented_name_set):
        errors.append(f"test_map: {name}: in test_map.json but not in docs/mac/09-testing-and-validation.md")
    for name in sorted(mapped_names & documented_name_set):
        entry = entries[name]
        if isinstance(entry, dict) and entry.get("label") != documented_names[name]:
            errors.append(
                f"test_map: {name}: documented label {documented_names[name]}, "
                f"the map says {entry.get('label')}"
            )

    actual_counts = {
        label: sum(entry.get("label") == label for entry in entries.values() if isinstance(entry, dict))
        for label in ("host", "metal", "model")
    }
    for label, expected in declared_counts.items():
        if actual_counts[label] != expected:
            errors.append(
                f"test_map: section 9.1 declares {expected} {label} tests, "
                f"test_map.json has {actual_counts[label]}"
            )

    try:
        requirements_doc = (root / "docs/mac/01-requirements.md").read_text(encoding="utf-8")
        req_rows = documented_requirement_rows(requirements_doc, errors)
    except OSError as exc:
        errors.append(f"test_map: cannot read docs/mac/01-requirements.md: {exc}")
        return

    python_patterns = documented_python_paths(text)
    review_ids = documented_review_ids(text)
    requirements_data = read_json(root / "tests/metal/requirements.json", errors)
    if not schema_is_one(requirements_data.get("schema")):
        errors.append("test_map: tests/metal/requirements.json: schema must be 1")
    requirements = requirements_data.get("requirements")
    if not isinstance(requirements, dict):
        errors.append("test_map: tests/metal/requirements.json: requirements must be an object")
        return

    for req_id in sorted(req_rows.keys() - requirements.keys()):
        errors.append(f"test_map: {req_id}: missing from requirements.json")
    for req_id in sorted(requirements.keys() - req_rows.keys()):
        errors.append(f"test_map: {req_id}: not present in docs/mac/01-requirements.md")

    criteria_path = root / "tools/metal/accept/criteria.py"
    criteria_ids: set[str] = set()
    if not path_is_within_root(root, criteria_path):
        errors.append("test_map: tools/metal/accept/criteria.py escapes --root")
    elif criteria_path.is_file():
        try:
            criteria_text = criteria_path.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            errors.append(f"test_map: cannot read tools/metal/accept/criteria.py: {exc}")
        else:
            try:
                criteria_ids = declared_criterion_ids(criteria_text)
            except SyntaxError as exc:
                errors.append(f"test_map: cannot parse tools/metal/accept/criteria.py: {exc}")
    for req_id, checks in requirements.items():
        if req_id not in req_rows:
            continue
        withdrawn, verified_by = req_rows[req_id]
        documented_checks = documented_requirement_checks(
            root, req_id, verified_by, text, errors
        )
        if withdrawn and checks != "withdrawn":
            errors.append(f"test_map: {req_id}: withdrawn requirement must map to withdrawn")
            continue
        if not withdrawn and checks == "withdrawn":
            errors.append(f"test_map: {req_id}: requirement is not withdrawn in docs")
            continue
        if withdrawn:
            continue
        if checks == "withdrawn":
            continue
        if not isinstance(checks, list) or any(not isinstance(check, str) for check in checks):
            errors.append(f"test_map: {req_id}: checks must be a list or withdrawn")
            continue
        if not isinstance(documented_checks, set):
            errors.append(f"test_map: {req_id}: documentation checks could not be read")
            continue
        unexpected_checks = sorted(set(checks) - documented_checks)
        if unexpected_checks:
            errors.append(
                f"test_map: {req_id}: checks not listed in the requirement's Verified by cell: "
                + ", ".join(unexpected_checks)
            )
        missing_checks = sorted(documented_checks - set(checks))
        if not checks:
            message = f"test_map: {req_id}: no check yet"
            (errors if final else notes).append(message)
            continue
        if missing_checks:
            message = f"test_map: {req_id}: missing checks: " + ", ".join(missing_checks)
            (errors if final else notes).append(message)
        for check in checks:
            if check.startswith("ctest:"):
                name = check.removeprefix("ctest:")
                if name not in entries:
                    errors.append(f"test_map: {req_id}: unknown check {check}")
            elif check.startswith("accept:"):
                criterion = check.removeprefix("accept:")
                if criterion not in criteria_ids:
                    errors.append(f"test_map: {req_id}: unknown check {check}")
            elif check.startswith("python:"):
                path = check.removeprefix("python:")
                candidate = root / path
                if not path_is_within_root(root, candidate):
                    errors.append(f"test_map: {req_id}: check path escapes root: {check}")
                elif not any(path_matches_pattern(path, pattern) for pattern in python_patterns):
                    errors.append(f"test_map: {req_id}: unknown check {check}")
                elif not candidate.is_file():
                    errors.append(f"test_map: {req_id}: missing module for check {check}")
            elif check.startswith("ci:"):
                workflow_job = check.removeprefix("ci:")
                workflow, sep, job = workflow_job.partition("/")
                if (
                    not sep
                    or not workflow.endswith((".yml", ".yaml"))
                    or Path(workflow).name != workflow
                    or "\\" in workflow
                    or not job
                ):
                    errors.append(f"test_map: {req_id}: malformed check {check}")
                    continue
                workflow_root = root / ".github/workflows"
                workflow_path = workflow_root / workflow
                if not path_is_within_root(root, workflow_root) or not path_is_within_root(
                    root, workflow_path
                ):
                    errors.append(f"test_map: {req_id}: workflow path escapes root: {check}")
                    continue
                jobs = workflow_jobs(workflow_path)
                if jobs is None or job not in jobs:
                    errors.append(f"test_map: {req_id}: unknown check {check}")
            elif check.startswith("review:"):
                review = check.removeprefix("review:")
                if review not in review_ids:
                    errors.append(f"test_map: {req_id}: unknown check {check}")
            else:
                errors.append(f"test_map: {req_id}: malformed check {check}")


def _strip_yaml_comment(line: str) -> str:
    quote = None
    escaped = False
    index = 0
    while index < len(line):
        char = line[index]
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif quote == "'":
            if char == "'" and index + 1 < len(line) and line[index + 1] == "'":
                index += 1
            elif char == "'":
                quote = None
        elif char in {"'", '"'} and (
            index == 0 or line[index - 1].isspace() or line[index - 1] in "[{,:"
        ):
            quote = char
        elif char == "#" and (index == 0 or line[index - 1].isspace()):
            return line[:index].rstrip()
        index += 1
    return line.rstrip()


def _yaml_key(raw_key: str) -> str | None:
    raw_key = raw_key.strip()
    if len(raw_key) >= 2 and raw_key[0] == raw_key[-1] == "'":
        return raw_key[1:-1].replace("''", "'")
    if len(raw_key) >= 2 and raw_key[0] == raw_key[-1] == '"':
        try:
            value = json.loads(raw_key)
        except json.JSONDecodeError:
            return None
        return value if isinstance(value, str) else None
    return raw_key or None


def _yaml_mapping_entry(content: str) -> tuple[str, str] | None:
    quote = None
    escaped = False
    depth = 0
    for index, char in enumerate(content):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
            continue
        if quote == "'":
            if char == "'" and index + 1 < len(content) and content[index + 1] == "'":
                continue
            if char == "'":
                quote = None
            continue
        if char in {"'", '"'} and (
            index == 0 or content[index - 1].isspace() or content[index - 1] in "[{,:"
        ):
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
        elif (
            char == ":"
            and depth == 0
            and (index + 1 == len(content) or content[index + 1].isspace() or content[index + 1] in "[{")
        ):
            key = _yaml_key(content[:index])
            if key is not None:
                return key, content[index + 1 :].strip()
            return None
    return None


def _flow_mapping_items(value: str) -> list[tuple[str, str]] | None:
    value = value.strip()
    if not value.startswith("{") or not value.endswith("}"):
        return None
    body = value[1:-1].strip()
    if not body:
        return []

    items: list[str] = []
    quote = None
    escaped = False
    depth = 0
    start = 0
    for index, char in enumerate(body):
        if quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
            continue
        if quote == "'":
            if char == "'" and index + 1 < len(body) and body[index + 1] == "'":
                continue
            if char == "'":
                quote = None
            continue
        if char in {"'", '"'} and (
            index == 0 or body[index - 1].isspace() or body[index - 1] in "[{,:"
        ):
            quote = char
        elif char in "[{":
            depth += 1
        elif char in "]}":
            depth -= 1
            if depth < 0:
                return None
        elif char == "," and depth == 0:
            items.append(body[start:index].strip())
            start = index + 1
    if quote is not None or depth != 0:
        return None
    items.append(body[start:].strip())
    if items[-1] == "":
        items.pop()
    if not items or any(not item for item in items):
        return None

    result: list[tuple[str, str]] = []
    for item in items:
        entry = _yaml_mapping_entry(item)
        if entry is None:
            return None
        result.append(entry)
    return result


def _has_block_scalar_header(content: str) -> bool:
    entry = _yaml_mapping_entry(content)
    if entry is not None:
        value = entry[1]
    elif content.startswith("- "):
        value = content[2:].strip()
    else:
        return False
    return BLOCK_SCALAR_VALUE.fullmatch(value) is not None


def _workflow_content_lines(lines: list[str]) -> list[tuple[int, str]] | None:
    content_lines: list[tuple[int, str]] = []
    scalar_parent_indent: int | None = None
    for line in lines:
        indentation = len(line) - len(line.lstrip(" "))
        prefix = line[: len(line) - len(line.lstrip(" \t"))]
        if "\t" in prefix:
            return None
        content = _strip_yaml_comment(line[indentation:]).strip()
        if not content:
            continue
        if scalar_parent_indent is not None:
            if indentation > scalar_parent_indent:
                continue
            scalar_parent_indent = None
        content_lines.append((indentation, content))
        if _has_block_scalar_header(content):
            scalar_parent_indent = indentation
    return content_lines


def _job_value_is_block_mapping(
    content_lines: list[tuple[int, str]], index: int, job_indent: int
) -> bool:
    for indentation, content in content_lines[index + 1 :]:
        if indentation <= job_indent:
            return False
        if content == "-" or content.startswith("- "):
            return False
        return _yaml_mapping_entry(content) is not None
    return False


def _valid_job_id(job_id: str) -> bool:
    return WORKFLOW_JOB_ID.fullmatch(job_id) is not None


def workflow_jobs(path: Path) -> set[str] | None:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    content_lines = _workflow_content_lines(lines)
    if content_lines is None:
        return None

    jobs_index = None
    jobs_value = ""
    for index, (indentation, content) in enumerate(content_lines):
        if indentation != 0:
            continue
        entry = _yaml_mapping_entry(content)
        if entry is None or entry[0] != "jobs":
            continue
        if jobs_index is not None:
            return None
        jobs_index = index
        jobs_value = entry[1]
    if jobs_index is None:
        return None

    if jobs_value:
        flow_jobs = _flow_mapping_items(jobs_value)
        if flow_jobs is None:
            return None
        jobs: set[str] = set()
        for job_id, value in flow_jobs:
            job_map = _flow_mapping_items(value)
            if not _valid_job_id(job_id) or job_map is None or job_id in jobs:
                return None
            jobs.add(job_id)
        return jobs

    following = []
    for item in content_lines[jobs_index + 1 :]:
        if item[0] == 0:
            break
        following.append(item)
    if not following:
        return set()
    job_indent = following[0][0]
    if job_indent <= 0:
        return None

    jobs: set[str] = set()
    index = jobs_index + 1
    while index < len(content_lines):
        indentation, content = content_lines[index]
        if indentation == 0:
            break
        if indentation < job_indent:
            return None
        if indentation == job_indent:
            entry = _yaml_mapping_entry(content)
            if entry is None:
                return None
            job_id, value = entry
            if not _valid_job_id(job_id) or job_id in jobs:
                return None
            if value:
                if _flow_mapping_items(value) is None:
                    return None
            elif not _job_value_is_block_mapping(content_lines, index, job_indent):
                return None
            jobs.add(job_id)
        index += 1
    return jobs


def check_reviews(root: Path, review_ids: set[str], errors: list[str]) -> None:
    data = read_json(root / "tests/metal/reviews.json", errors)
    if not schema_is_one(data.get("schema")):
        errors.append("test_map: tests/metal/reviews.json: schema must be 1")
    reviews = data.get("reviews")
    if not isinstance(reviews, list):
        errors.append("test_map: tests/metal/reviews.json: reviews must be a list")
        return
    seen: set[tuple[str, str]] = set()
    for index, entry in enumerate(reviews):
        prefix = f"test_map: reviews[{index}]"
        if not isinstance(entry, dict):
            errors.append(f"{prefix}: entry must be an object")
            continue
        review = entry.get("review")
        commit = entry.get("commit")
        url = entry.get("url")
        by = entry.get("by")
        review_date = entry.get("date")
        if review not in review_ids:
            errors.append(f"{prefix}: unknown review {review}")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
            errors.append(f"{prefix}: commit must be 40 hex characters")
        if not valid_https_url(url):
            errors.append(f"{prefix}: url must be an https URL")
        if not isinstance(by, str) or not by.strip():
            errors.append(f"{prefix}: by must name the reviewer")
        if not isinstance(review_date, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", review_date):
            errors.append(f"{prefix}: date must use YYYY-MM-DD")
        else:
            try:
                parsed_date = date.fromisoformat(review_date)
            except ValueError:
                errors.append(f"{prefix}: invalid calendar date")
            else:
                if parsed_date.isoformat() != review_date:
                    errors.append(f"{prefix}: invalid calendar date")
        key = (str(review), str(commit))
        if key in seen:
            errors.append(f"{prefix}: duplicate review and commit")
        seen.add(key)


def main(root: Path = ROOT, argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final", action="store_true", help="fail on planned tests and unchecked requirements")
    parser.add_argument("--root", type=Path, default=root, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    root = args.root.resolve()
    errors: list[str] = []
    notes: list[str] = []

    test_map = read_json(root / "tests/metal/test_map.json", errors)
    entries = check_document_contract(test_map, errors)
    try:
        test_doc = (root / "docs/mac/09-testing-and-validation.md").read_text(encoding="utf-8")
    except OSError as exc:
        errors.append(f"test_map: cannot read docs/mac/09-testing-and-validation.md: {exc}")
        test_doc = ""

    check_ctest(root, entries, args.final, errors, notes)
    check_test_catalog(root, test_doc, entries, args.final, errors, notes)
    check_reviews(root, documented_review_ids(test_doc), errors)

    for line in sorted(set(notes)):
        print(line)
    for line in sorted(set(errors)):
        print(line, file=sys.stderr)
    if errors:
        return 1
    print("test_map: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
