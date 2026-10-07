#!/usr/bin/env python3
"""Check the Mac release version rules (docs/mac/08-build-and-distribution.md, section 11.1)."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tools.metal import release_metadata


VERSION_RE = re.compile(r"^[0-9]+\.[0-9]+\.[0-9]+$")
METAL_TAG = re.compile(r"^v[0-9]+\.[0-9]+\.[0-9]+-metal\.([1-9][0-9]*)$")
METAL_TAG_MARKER = re.compile(r"-metal", re.IGNORECASE)


def read_one_line(path: Path, pattern: re.Pattern[str], error: str) -> tuple[str | None, str | None]:
    try:
        value = path.read_bytes().decode("ascii")
    except (OSError, UnicodeDecodeError):
        return None, error
    if not pattern.fullmatch(value):
        return None, error
    return value[:-1], None


def version_tuple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, str) or not VERSION_RE.fullmatch(value):
        return None
    return tuple(int(part) for part in value.split("."))


def setup_version_tuple(value: Any) -> tuple[int, int, int] | None:
    if not isinstance(value, (tuple, list)) or len(value) != 3:
        return None
    if any(type(part) is not int or part < 0 for part in value):
        return None
    return tuple(value)


def gh_json(
    root: Path,
    args: list[str],
    env: dict[str, str],
    runner: Callable[..., Any],
    errors: list[str],
) -> Any | None:
    try:
        result = runner(
            ["gh", *args],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        errors.append(f"check_release_policy: cannot run gh: {exc}")
        return None
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        errors.append(f"check_release_policy: gh {' '.join(args)} failed: {detail}")
        return None
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        errors.append(f"check_release_policy: gh returned invalid JSON: {exc}")
        return None


def current_branch(root: Path, env: dict[str, str], runner: Callable[..., Any]) -> str:
    configured = env.get("GITHUB_HEAD_REF") or env.get("GITHUB_REF_NAME")
    if configured:
        return configured
    try:
        result = runner(
            ["git", "branch", "--show-current"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return result.stdout.strip() if result.returncode == 0 else ""


def repository_slug(value: Any) -> str | None:
    if not isinstance(value, str) or not re.fullmatch(r"[^/\s]+/[^/\s]+", value):
        return None
    return value.casefold()


def repository_slug_from_remote(value: str) -> str | None:
    if value.startswith("git@") and ":" in value:
        host, path = value[4:].split(":", 1)
    else:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https", "ssh", "git"}:
            return None
        host, path = parsed.hostname or "", parsed.path
    if host.casefold() != "github.com":
        return None
    parts = path.strip("/").split("/")
    if len(parts) != 2:
        return None
    if parts[1].endswith(".git"):
        parts[1] = parts[1][:-4]
    return repository_slug("/".join(parts))


def verify_release_repository(
    root: Path,
    release_repository: str,
    env: dict[str, str],
    runner: Callable[..., Any],
    errors: list[str],
) -> bool:
    expected = repository_slug(release_repository)
    if expected is None:
        errors.append("check_release_policy: RELEASE_REPOSITORY must be set to owner/repository for --remote")
        return False

    matches = True
    workflow_repository = env.get("GITHUB_REPOSITORY")
    if workflow_repository is not None and repository_slug(workflow_repository) != expected:
        errors.append(
            "check_release_policy: GITHUB_REPOSITORY "
            f"{workflow_repository!r} does not match publication repository {release_repository}"
        )
        matches = False

    remote_urls = (
        (
            ["git", "remote", "get-url", "--all", "origin"],
            "checkout origin",
        ),
        (
            ["git", "remote", "get-url", "--push", "--all", "origin"],
            "push destination",
        ),
    )
    for command, label in remote_urls:
        try:
            result = runner(
                command,
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            errors.append(f"check_release_policy: cannot identify {label}: {exc}")
            return False
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            errors.append(f"check_release_policy: cannot identify {label}: {detail}")
            return False

        urls = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not urls:
            errors.append(f"check_release_policy: {label} has no URL")
            matches = False
            continue
        for url in urls:
            checkout_repository = repository_slug_from_remote(url)
            if checkout_repository is None:
                errors.append(
                    f"check_release_policy: {label} must be a GitHub owner/repository URL"
                )
                matches = False
            elif checkout_repository != expected:
                errors.append(
                    f"check_release_policy: {label} {checkout_repository} does not match "
                    f"publication repository {release_repository}"
                )
                matches = False
    return matches


def verify_annotated_tag(
    root: Path,
    tag_name: str,
    env: dict[str, str],
    runner: Callable[..., Any],
    errors: list[str],
) -> None:
    try:
        result = runner(
            ["git", "cat-file", "-t", f"refs/tags/{tag_name}"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        errors.append(f"check_release_policy: cannot verify tag {tag_name} is annotated: {exc}")
        return
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        errors.append(f"check_release_policy: cannot verify tag {tag_name} is annotated: {detail}")
        return
    object_type = result.stdout.strip()
    if object_type != "tag":
        errors.append(
            f"check_release_policy: tag {tag_name} must be annotated; "
            f"git object type is {object_type or 'unknown'}"
        )


def verify_release_tag_targets_branch(
    root: Path,
    tag_name: str,
    branch_name: str,
    env: dict[str, str],
    runner: Callable[..., Any],
    errors: list[str],
    *,
    pull_request_head_sha: str | None = None,
) -> None:
    if pull_request_head_sha is not None:
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", pull_request_head_sha):
            errors.append("check_release_policy: pull request head SHA is missing or invalid")
            return
        comparison_commit = pull_request_head_sha
    else:
        try:
            result = runner(
                [
                    "git",
                    "rev-parse",
                    "--verify",
                    f"refs/remotes/origin/{branch_name}^{{commit}}",
                ],
                cwd=root,
                env=env,
                text=True,
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            errors.append(f"check_release_policy: cannot resolve release branch commit: {exc}")
            return
        if result.returncode:
            detail = result.stderr.strip() or result.stdout.strip()
            errors.append(f"check_release_policy: cannot resolve release branch commit: {detail}")
            return
        comparison_commit = result.stdout.strip()
        if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", comparison_commit):
            errors.append("check_release_policy: release branch did not resolve to a commit")
            return

    try:
        result = runner(
            ["git", "rev-parse", "--verify", f"refs/tags/{tag_name}^{{commit}}"],
            cwd=root,
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        errors.append(f"check_release_policy: cannot resolve tag commit: {exc}")
        return
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip()
        errors.append(f"check_release_policy: cannot resolve tag commit: {detail}")
        return
    tag_commit = result.stdout.strip()
    if not re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", tag_commit):
        errors.append("check_release_policy: tag did not resolve to a commit")
        return

    if tag_commit != comparison_commit:
        if pull_request_head_sha is not None:
            errors.append(
                f"check_release_policy: tag {tag_name} does not point at the pull request head"
            )
        else:
            errors.append(
                f"check_release_policy: tag {tag_name} does not point at the head of "
                f"{branch_name}"
            )


def verify_latest_release(
    root: Path,
    release_repository: str,
    expected_tag: str,
    context: str,
    env: dict[str, str],
    runner: Callable[..., Any],
    errors: list[str],
) -> None:
    latest = gh_json(
        root,
        ["release", "view", "--repo", release_repository, "--json", "tagName,assets"],
        env,
        runner,
        errors,
    )
    if not isinstance(latest, dict):
        errors.append("check_release_policy: gh release view must return a JSON object")
        return
    latest_tag = latest.get("tagName")
    assets = latest.get("assets")
    if latest_tag != expected_tag:
        errors.append(
            f"check_release_policy: {context}: the latest release is "
            f"{latest_tag}, this commit needs {expected_tag}"
        )
    elif not isinstance(assets, list) or not any(
        isinstance(asset, dict) and asset.get("name") == "strata-macos-arm64.zip"
        for asset in assets
    ):
        errors.append("check_release_policy: latest release is missing strata-macos-arm64.zip")


def check_release_policy(
    root: Path = ROOT,
    *,
    remote: bool = False,
    env: dict[str, str] | None = None,
    runner: Callable[..., Any] = subprocess.run,
    release_config: Any = release_metadata,
) -> tuple[list[str], list[str]]:
    environment = dict(os.environ if env is None else env)
    errors: list[str] = []
    notes: list[str] = []

    metal_rev_text, error = read_one_line(
        root / "METAL_REV",
        re.compile(r"[1-9]\d*\n"),
        "METAL_REV must hold one positive integer",
    )
    if error:
        errors.append(f"check_release_policy: {error}")
    metal_rev = int(metal_rev_text) if metal_rev_text is not None else None

    _, error = read_one_line(
        root / "UPSTREAM_SYNC",
        re.compile(r"[0-9a-f]{40}\n"),
        "UPSTREAM_SYNC must hold one 40-hex commit",
    )
    if error:
        errors.append(f"check_release_policy: {error}")

    version = release_config.source_version()
    source_version = version_tuple(version)
    if source_version is None:
        errors.append(f"check_release_policy: invalid engine source version {version!r}")

    minimum_engine = setup_version_tuple(getattr(release_config, "MIN_ENGINE", None))
    if minimum_engine is None:
        errors.append("check_release_policy: MIN_ENGINE must be a three-part integer version")
    elif source_version is not None and minimum_engine > source_version:
        minimum_text = ".".join(str(part) for part in minimum_engine)
        errors.append(
            f"check_release_policy: MIN_ENGINE {minimum_text} is above the engine version {version}"
        )

    minimum_metal = getattr(release_config, "MIN_ENGINE_METAL", None)
    if type(minimum_metal) is not int or minimum_metal < 1:
        errors.append("check_release_policy: MIN_ENGINE_METAL must be a positive integer")
    elif metal_rev is not None and minimum_metal > metal_rev:
        errors.append(
            f"check_release_policy: MIN_ENGINE_METAL {minimum_metal} is above METAL_REV {metal_rev}"
        )

    branch = current_branch(root, environment, runner)
    is_release_pull_request = (
        environment.get("GITHUB_EVENT_NAME") == "pull_request"
        and branch.startswith("release/")
    )
    pull_request_head_sha = environment.get("GITHUB_EVENT_PULL_REQUEST_HEAD_SHA", "")
    expected_tag = f"v{version}-metal.{metal_rev}" if metal_rev is not None else None
    if branch.startswith("release/") and expected_tag:
        expected_branch = f"release/{version}-metal.{metal_rev}"
        if branch != expected_branch:
            errors.append(
                f"check_release_policy: branch {branch} does not match the version: "
                f"want {expected_branch}"
            )

    is_tag = environment.get("GITHUB_REF_TYPE") == "tag"
    ref_name = environment.get("GITHUB_REF_NAME", "")
    if is_tag and expected_tag and ref_name != expected_tag:
        errors.append(
            f"check_release_policy: tag {ref_name} does not match the version: want {expected_tag}"
        )
    if is_tag:
        if ref_name:
            verify_annotated_tag(root, ref_name, environment, runner, errors)
        else:
            errors.append("check_release_policy: GITHUB_REF_NAME is required to verify an annotated tag")

    if not remote or metal_rev is None or expected_tag is None or source_version is None:
        return errors, notes
    if is_release_pull_request and not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", pull_request_head_sha
    ):
        errors.append("check_release_policy: pull request head SHA is missing or invalid")
        return errors, notes

    release_repository = getattr(release_config, "RELEASE_REPOSITORY", None)
    if repository_slug(release_repository) is None:
        errors.append("check_release_policy: RELEASE_REPOSITORY must be set to owner/repository for --remote")
        return errors, notes
    if not verify_release_repository(root, release_repository, environment, runner, errors):
        return errors, notes

    listed = gh_json(
        root,
        ["api", "--paginate", "--slurp", f"repos/{release_repository}/releases?per_page=100"],
        environment,
        runner,
        errors,
    )
    if not isinstance(listed, list):
        errors.append("check_release_policy: gh api --paginate --slurp must return an array of release pages")
        return errors, notes
    if any(not isinstance(page, list) for page in listed):
        errors.append("check_release_policy: gh api --paginate --slurp must return an array of release pages")
        return errors, notes

    releases: list[tuple[int, str]] = []
    for release in (release for page in listed for release in page):
        if not isinstance(release, dict):
            errors.append("check_release_policy: a release page contains a non-object entry")
            continue
        draft = release.get("draft")
        prerelease = release.get("prerelease")
        if type(draft) is not bool or type(prerelease) is not bool:
            errors.append("check_release_policy: a release has no valid draft or prerelease status")
            continue
        if draft:
            continue
        tag_name = release.get("tag_name")
        if not isinstance(tag_name, str) or not tag_name:
            errors.append("check_release_policy: a published release has no tagName")
            continue
        match = METAL_TAG.fullmatch(tag_name)
        if match:
            if prerelease:
                errors.append(
                    f"check_release_policy: Metal release {tag_name} must not be a prerelease"
                )
                continue
            releases.append((int(match.group(1)), tag_name))
        elif METAL_TAG_MARKER.search(tag_name):
            errors.append(
                f"check_release_policy: published Metal release tag {tag_name} is malformed"
            )

    release_branch_or_tag = branch.startswith("release/") or is_tag
    published_tags = {tag for _, tag in releases}
    release_already_published = expected_tag in published_tags

    if is_release_pull_request and not release_already_published:
        errors.append(
            "check_release_policy: release-branch pull request cannot merge before its "
            "Metal release tag is published"
        )
        return errors, notes

    if release_already_published:
        previous_revisions = [
            revision for revision, tag in releases if tag != expected_tag
        ]
        highest_previous = max(previous_revisions, default=0)
        expected_next = highest_previous + 1
        if metal_rev != expected_next:
            previous_tag = max(
                (item for item in releases if item[1] != expected_tag),
                default=(0, "no earlier Metal release"),
            )[1]
            errors.append(
                f"check_release_policy: published METAL_REV must be exactly "
                f"{expected_next} after {previous_tag}, found {metal_rev}"
            )

    if release_branch_or_tag and release_already_published:
        verify_latest_release(
            root,
            release_repository,
            expected_tag,
            "release branch must point at its published release",
            environment,
            runner,
            errors,
        )
        if branch.startswith("release/") or is_tag:
            if not is_tag:
                verify_annotated_tag(root, expected_tag, environment, runner, errors)
            verify_release_tag_targets_branch(
                root,
                ref_name if is_tag else expected_tag,
                f"release/{version}-metal.{metal_rev}",
                environment,
                runner,
                errors,
                pull_request_head_sha=pull_request_head_sha if is_release_pull_request else None,
            )
        return errors, notes

    if release_branch_or_tag:
        highest_rev = max((revision for revision, _ in releases), default=0)
        expected_next = highest_rev + 1
        if metal_rev != expected_next:
            if highest_rev:
                _, highest_tag = max(releases)
                errors.append(
                    f"check_release_policy: METAL_REV must be exactly {expected_next} after "
                    f"{highest_tag}, found {metal_rev}"
                )
            else:
                errors.append(
                    f"check_release_policy: no published Metal release yet; "
                    f"METAL_REV must be 1, found {metal_rev}"
                )
        if is_tag:
            verify_release_tag_targets_branch(
                root,
                ref_name,
                f"release/{version}-metal.{metal_rev}",
                environment,
                runner,
                errors,
            )
        return errors, notes

    if releases:
        verify_latest_release(
            root,
            release_repository,
            expected_tag,
            "main must point at a published release",
            environment,
            runner,
            errors,
        )
        verify_annotated_tag(root, expected_tag, environment, runner, errors)
        verify_release_tag_targets_branch(
            root,
            expected_tag,
            f"release/{version}-metal.{metal_rev}",
            environment,
            runner,
            errors,
        )
    elif metal_rev == 1:
        notes.append("check_release_policy: no release yet")
    else:
        errors.append(f"check_release_policy: no published release exists for METAL_REV {metal_rev}")
    return errors, notes


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--remote", action="store_true", help="check published GitHub releases with gh")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    errors, notes = check_release_policy(args.root.resolve(), remote=args.remote)
    for line in notes:
        print(line)
    for line in errors:
        print(line, file=sys.stderr)
    if errors:
        return 1
    print("check_release_policy: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
