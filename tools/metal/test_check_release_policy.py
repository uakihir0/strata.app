import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools.metal import release_metadata
from tools.metal import check_release_policy


class ReleasePolicy(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "METAL_REV").write_text("1\n", encoding="ascii")
        (self.root / "UPSTREAM_SYNC").write_text("a" * 40 + "\n", encoding="ascii")
        self.env = {
            "GITHUB_REF_NAME": "main",
            "GITHUB_REF_TYPE": "branch",
            "GITHUB_REPOSITORY": "owner/repo",
        }

    def tearDown(self):
        self.temporary.cleanup()

    def run_check(
        self,
        *,
        remote=False,
        responses=(),
        env=None,
        source_version="0.1.34",
        release_repository="owner/repo",
        origin="https://github.com/owner/repo.git",
        push_origin=None,
        tag_object_type="tag",
        tag_commit="c" * 40,
        branch_commit="c" * 40,
    ):
        response_iter = iter(responses)
        calls = []

        def runner(command, **kwargs):
            calls.append(command)
            if command[:3] == ["git", "branch", "--show-current"]:
                return subprocess.CompletedProcess(command, 0, "main\n", "")
            if command == ["git", "remote", "get-url", "--all", "origin"]:
                return subprocess.CompletedProcess(command, 0, origin + "\n", "")
            if command == ["git", "remote", "get-url", "--push", "--all", "origin"]:
                return subprocess.CompletedProcess(command, 0, (push_origin or origin) + "\n", "")
            if command[:3] == ["git", "cat-file", "-t"]:
                if tag_object_type is None:
                    return subprocess.CompletedProcess(
                        command, 1, "", "fatal: Not a valid object name"
                    )
                return subprocess.CompletedProcess(command, 0, tag_object_type + "\n", "")
            if command[:3] == ["git", "rev-parse", "--verify"]:
                ref = command[-1]
                commit = tag_commit if ref.startswith("refs/tags/") else branch_commit
                return subprocess.CompletedProcess(command, 0, commit + "\n", "")
            return next(response_iter)

        with (
            mock.patch.object(release_metadata, "source_version", return_value=source_version),
            mock.patch.object(release_metadata, "MIN_ENGINE", (0, 1, 34)),
            mock.patch.object(release_metadata, "MIN_ENGINE_METAL", 1),
            mock.patch.object(release_metadata, "RELEASE_REPOSITORY", release_repository),
        ):
            errors, notes = check_release_policy.check_release_policy(
                self.root,
                remote=remote,
                env=self.env | (env or {}),
                runner=runner,
            )
        return errors, notes, calls

    @staticmethod
    def result(command, payload, *, status=0, stderr=""):
        stdout = payload if isinstance(payload, str) else json.dumps(payload)
        return subprocess.CompletedProcess(command, status, stdout, stderr)

    def test_baseline_versions_pass(self):
        errors, notes, _ = self.run_check()

        self.assertEqual([], errors)
        self.assertEqual([], notes)

    def test_metal_revision_must_be_one_positive_integer_line(self):
        for content in ("", "0\n", "01\n", "1\n2\n", "1"):
            with self.subTest(content=content):
                (self.root / "METAL_REV").write_text(content, encoding="ascii")
                errors, _, _ = self.run_check()

                self.assertIn(
                    "check_release_policy: METAL_REV must hold one positive integer",
                    errors,
                )
        (self.root / "METAL_REV").write_text("1\n", encoding="ascii")

    def test_minimum_metal_revision_cannot_exceed_release_revision(self):
        (self.root / "METAL_REV").write_text("1\n", encoding="ascii")
        with mock.patch.object(release_metadata, "MIN_ENGINE_METAL", 2):
            errors, _ = check_release_policy.check_release_policy(self.root, env=self.env)

        self.assertIn(
            "check_release_policy: MIN_ENGINE_METAL 2 is above METAL_REV 1",
            errors,
        )

    def test_minimum_engine_cannot_exceed_source_version(self):
        errors, _, _ = self.run_check(source_version="0.1.33")

        self.assertIn(
            "check_release_policy: MIN_ENGINE 0.1.34 is above the engine version 0.1.33",
            errors,
        )

    def test_upstream_sync_must_be_lowercase_hex_with_newline(self):
        for content in ("A" * 40 + "\n", "a" * 39 + "\n", "a" * 40, "a" * 40 + "\n\n"):
            with self.subTest(content=content):
                (self.root / "UPSTREAM_SYNC").write_text(content, encoding="ascii")
                errors, _, _ = self.run_check()

                self.assertIn(
                    "check_release_policy: UPSTREAM_SYNC must hold one 40-hex commit",
                    errors,
                )
        (self.root / "UPSTREAM_SYNC").write_text("a" * 40 + "\n", encoding="ascii")

    def test_release_branch_must_match_source_and_revision(self):
        errors, _, _ = self.run_check(
            env={"GITHUB_HEAD_REF": "release/0.1.34-metal.2"}
        )

        self.assertIn(
            "check_release_policy: branch release/0.1.34-metal.2 does not match the version: "
            "want release/0.1.34-metal.1",
            errors,
        )

    def test_release_tag_must_match_source_and_revision(self):
        errors, _, _ = self.run_check(
            env={
                "GITHUB_REF_NAME": "v0.1.34-metal.2",
                "GITHUB_REF_TYPE": "tag",
            }
        )

        self.assertIn(
            "check_release_policy: tag v0.1.34-metal.2 does not match the version: "
            "want v0.1.34-metal.1",
            errors,
        )

    def test_release_tag_must_be_annotated(self):
        errors, _, calls = self.run_check(
            env={
                "GITHUB_REF_NAME": "v0.1.34-metal.1",
                "GITHUB_REF_TYPE": "tag",
            },
            tag_object_type="commit",
        )

        self.assertIn(
            "check_release_policy: tag v0.1.34-metal.1 must be annotated; "
            "git object type is commit",
            errors,
        )
        self.assertIn(
            ["git", "cat-file", "-t", "refs/tags/v0.1.34-metal.1"],
            calls,
        )

    def test_remote_check_rejects_checkout_or_publication_repository_mismatch(self):
        cases = (
            {
                "origin": "git@github.com:other/repo.git",
                "env": {"GITHUB_REPOSITORY": "owner/repo"},
                "expected": "checkout origin other/repo does not match publication repository owner/repo",
            },
            {
                "origin": "https://github.com/owner/repo.git",
                "env": {"GITHUB_REPOSITORY": "other/repo"},
                "expected": "GITHUB_REPOSITORY 'other/repo' does not match publication repository owner/repo",
            },
        )
        for case in cases:
            with self.subTest(expected=case["expected"]):
                errors, _, calls = self.run_check(
                    remote=True,
                    origin=case["origin"],
                    env=case["env"],
                )

                self.assertTrue(any(case["expected"] in error for error in errors), errors)
                self.assertFalse(any(command[0] == "gh" for command in calls))

    def test_remote_check_rejects_a_different_push_url(self):
        errors, _, calls = self.run_check(
            remote=True,
            push_origin="https://github.com/other/repo.git",
        )

        self.assertTrue(
            any(
                "push destination other/repo does not match publication repository owner/repo"
                in error
                for error in errors
            ),
            errors,
        )
        self.assertFalse(any(command[0] == "gh" for command in calls))

    def test_release_revision_must_be_exactly_one_above_all_published_metal_releases(self):
        (self.root / "METAL_REV").write_text("3\n", encoding="ascii")
        errors, _, calls = self.run_check(
            remote=True,
            env={
                "GITHUB_HEAD_REF": "release/0.1.34-metal.3",
                "GITHUB_REF_NAME": "123/merge",
            },
            responses=[
                self.result(
                    ["gh"],
                    [[
                        {"tag_name": "v0.1.34-metal.1", "draft": False, "prerelease": False},
                        {"tag_name": "v0.1.33-metal.3", "draft": False, "prerelease": False},
                    ]],
                )
            ],
        )

        self.assertIn(
            "check_release_policy: METAL_REV must be exactly 4 after "
            "v0.1.33-metal.3, found 3",
            errors,
        )
        gh_calls = [command for command in calls if command[0] == "gh"]
        self.assertEqual(1, len(gh_calls))
        self.assertEqual(
            ["gh", "api", "--paginate", "--slurp", "repos/owner/repo/releases?per_page=100"],
            gh_calls[0],
        )

    def test_remote_release_history_includes_old_releases_beyond_default_window(self):
        (self.root / "METAL_REV").write_text("101\n", encoding="ascii")
        recent_releases = [
            {
                "tag_name": f"v0.1.34-metal.{revision}",
                "draft": False,
                "prerelease": False,
            }
            for revision in range(100, 0, -1)
        ]
        older_release = [
            {"tag_name": "v0.1.20-metal.150", "draft": False, "prerelease": False}
        ]
        errors, _, calls = self.run_check(
            remote=True,
            env={
                "GITHUB_HEAD_REF": "release/0.1.34-metal.101",
                "GITHUB_REF_NAME": "123/merge",
            },
            responses=[self.result(["gh"], [recent_releases, older_release])],
        )

        self.assertIn(
            "check_release_policy: METAL_REV must be exactly 151 after "
            "v0.1.20-metal.150, found 101",
            errors,
        )
        gh_calls = [command for command in calls if command[0] == "gh"]
        self.assertEqual(
            ["gh", "api", "--paginate", "--slurp", "repos/owner/repo/releases?per_page=100"],
            gh_calls[0],
        )

    def test_release_pr_cannot_merge_before_its_release_is_published(self):
        (self.root / "METAL_REV").write_text("2\n", encoding="ascii")
        errors, _, calls = self.run_check(
            remote=True,
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_HEAD_REF": "release/0.1.34-metal.2",
                "GITHUB_REF_NAME": "123/merge",
                "GITHUB_EVENT_PULL_REQUEST_HEAD_SHA": "d" * 40,
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                )
            ],
        )

        self.assertIn(
            "check_release_policy: release-branch pull request cannot merge before its "
            "Metal release tag is published",
            errors,
        )
        self.assertEqual(1, sum(command[0] == "gh" for command in calls))

    def test_remote_check_rejects_json_null_release_history(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[self.result(["gh"], "null")],
        )

        self.assertIn(
            "check_release_policy: gh api --paginate --slurp must return an array of release pages",
            errors,
        )

    def test_main_must_point_to_the_latest_release_with_the_engine_asset(self):
        errors, notes, calls = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertEqual([], errors)
        self.assertEqual([], notes)
        gh_calls = [command for command in calls if command[0] == "gh"]
        self.assertEqual(2, len(gh_calls))
        self.assertEqual("api", gh_calls[0][1])
        self.assertEqual("release", gh_calls[1][1])
        self.assertEqual("view", gh_calls[1][2])
        self.assertIn(
            ["git", "cat-file", "-t", "refs/tags/v0.1.34-metal.1"],
            calls,
        )

    def test_main_requires_the_published_release_tag_to_be_annotated(self):
        responses = [
            self.result(
                ["gh"],
                [[{
                    "tag_name": "v0.1.34-metal.1",
                    "draft": False,
                    "prerelease": False,
                }]],
            ),
            self.result(
                ["gh"],
                {
                    "tagName": "v0.1.34-metal.1",
                    "assets": [{"name": "strata-macos-arm64.zip"}],
                },
            ),
        ]

        for tag_object_type in ("commit", None):
            with self.subTest(tag_object_type=tag_object_type):
                errors, _, _ = self.run_check(
                    remote=True,
                    tag_object_type=tag_object_type,
                    responses=responses,
                )
                self.assertTrue(
                    any(
                        "tag v0.1.34-metal.1 must be annotated" in error
                        or "cannot verify tag v0.1.34-metal.1 is annotated" in error
                        for error in errors
                    ),
                    errors,
                )

    def test_main_requires_the_published_tag_to_match_release_branch_head(self):
        errors, _, _ = self.run_check(
            remote=True,
            branch_commit="d" * 40,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: tag v0.1.34-metal.1 does not point at the head of "
            "release/0.1.34-metal.1",
            errors,
        )

    def test_main_fails_when_latest_release_tag_differs(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {"tagName": "v0.1.34-metal.2", "assets": []},
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: main must point at a published release: the latest release is "
            "v0.1.34-metal.2, this commit needs v0.1.34-metal.1",
            errors,
        )

    def test_main_fails_when_engine_asset_is_missing(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {"tagName": "v0.1.34-metal.1", "assets": [{"name": "notes.txt"}]},
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: latest release is missing strata-macos-arm64.zip",
            errors,
        )

    def test_main_rejects_json_null_latest_release(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(["gh"], "null"),
            ],
        )

        self.assertIn(
            "check_release_policy: gh release view must return a JSON object",
            errors,
        )

    def test_published_release_pr_must_reference_latest_release_and_asset(self):
        errors, notes, calls = self.run_check(
            remote=True,
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_HEAD_REF": "release/0.1.34-metal.1",
                "GITHUB_REF_NAME": "123/merge",
                "GITHUB_EVENT_PULL_REQUEST_HEAD_SHA": "c" * 40,
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertEqual([], errors)
        self.assertEqual([], notes)
        self.assertIn(
            ["git", "rev-parse", "--verify", "refs/tags/v0.1.34-metal.1^{commit}"],
            calls,
        )
        self.assertIn(
            [
                "git",
                "rev-parse",
                "--verify",
                "refs/tags/v0.1.34-metal.1^{commit}",
            ],
            calls,
        )

    def test_published_release_pr_compares_tag_to_actual_head_sha(self):
        errors, _, calls = self.run_check(
            remote=True,
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_HEAD_REF": "release/0.1.34-metal.1",
                "GITHUB_REF_NAME": "123/merge",
                "GITHUB_EVENT_PULL_REQUEST_HEAD_SHA": "d" * 40,
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: tag v0.1.34-metal.1 does not point at the pull request head",
            errors,
        )
        self.assertNotIn(
            [
                "git",
                "rev-parse",
                "--verify",
                "refs/remotes/origin/release/0.1.34-metal.1^{commit}",
            ],
            calls,
        )

    def test_published_release_pr_requires_a_valid_head_sha(self):
        errors, _, _ = self.run_check(
            remote=True,
            env={
                "GITHUB_EVENT_NAME": "pull_request",
                "GITHUB_HEAD_REF": "release/0.1.34-metal.1",
                "GITHUB_REF_NAME": "123/merge",
                "GITHUB_EVENT_PULL_REQUEST_HEAD_SHA": "",
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertIn("check_release_policy: pull request head SHA is missing or invalid", errors)

    def test_published_release_revision_must_follow_previous_revision_by_one(self):
        errors, _, _ = self.run_check(
            remote=True,
            env={
                "GITHUB_HEAD_REF": "release/0.1.34-metal.1",
                "GITHUB_REF_NAME": "123/merge",
            },
            responses=[
                self.result(
                    ["gh"],
                    [[
                        {
                            "tag_name": "v0.1.34-metal.1",
                            "draft": False,
                            "prerelease": False,
                        },
                        {
                            "tag_name": "v0.1.33-metal.3",
                            "draft": False,
                            "prerelease": False,
                        },
                    ]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: published METAL_REV must be exactly 4 after "
            "v0.1.33-metal.3, found 1",
            errors,
        )

    def test_main_cannot_point_to_a_lower_revision_than_an_older_release(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[
                        {
                            "tag_name": "v0.1.34-metal.1",
                            "draft": False,
                            "prerelease": False,
                        },
                        {
                            "tag_name": "v0.1.20-metal.5",
                            "draft": False,
                            "prerelease": False,
                        },
                    ]],
                ),
                self.result(
                    ["gh"],
                    {
                        "tagName": "v0.1.34-metal.1",
                        "assets": [{"name": "strata-macos-arm64.zip"}],
                    },
                ),
            ],
        )

        self.assertIn(
            "check_release_policy: published METAL_REV must be exactly 6 after "
            "v0.1.20-metal.5, found 1",
            errors,
        )

    def test_release_tag_must_point_to_release_branch_head(self):
        errors, _, _ = self.run_check(
            remote=True,
            env={
                "GITHUB_REF_NAME": "v0.1.34-metal.1",
                "GITHUB_REF_TYPE": "tag",
            },
            tag_commit="a" * 40,
            branch_commit="b" * 40,
            responses=[self.result(["gh"], [[]])],
        )

        self.assertIn(
            "check_release_policy: tag v0.1.34-metal.1 does not point at the head of "
            "release/0.1.34-metal.1",
            errors,
        )

    def test_release_revision_must_increase_by_exactly_one(self):
        (self.root / "METAL_REV").write_text("3\n", encoding="ascii")
        errors, _, _ = self.run_check(
            remote=True,
            env={
                "GITHUB_HEAD_REF": "release/0.1.34-metal.3",
                "GITHUB_REF_NAME": "123/merge",
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": False,
                    }]],
                )
            ],
        )

        self.assertIn(
            "check_release_policy: METAL_REV must be exactly 2 after "
            "v0.1.34-metal.1, found 3",
            errors,
        )

    def test_metal_prereleases_are_rejected(self):
        errors, _, _ = self.run_check(
            remote=True,
            env={
                "GITHUB_HEAD_REF": "release/0.1.34-metal.2",
                "GITHUB_REF_NAME": "123/merge",
            },
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "v0.1.34-metal.1",
                        "draft": False,
                        "prerelease": True,
                    }]],
                )
            ],
        )

        self.assertIn(
            "check_release_policy: Metal release v0.1.34-metal.1 must not be a prerelease",
            errors,
        )

    def test_malformed_metal_release_tags_are_rejected(self):
        malformed_tags = (
            "v0.1.34-metal",
            "0.1.34-metal.2",
            "V0.1.34-METAL.2",
            "v0.1.34-metal.0",
            "v0.1.34-metal.01",
            "v0.1.34-metal.1-rc1",
            "v0.1.34-metal.2٢",
        )

        for tag_name in malformed_tags:
            with self.subTest(tag=tag_name):
                errors, _, _ = self.run_check(
                    remote=True,
                    responses=[
                        self.result(
                            ["gh"],
                            [[{
                                "tag_name": tag_name,
                                "draft": False,
                                "prerelease": False,
                            }]]
                        )
                    ],
                )

                self.assertIn(
                    f"check_release_policy: published Metal release tag {tag_name} is malformed",
                    errors,
                )

    def test_remote_check_rejects_empty_published_release_tag(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[
                self.result(
                    ["gh"],
                    [[{
                        "tag_name": "",
                        "draft": False,
                        "prerelease": False,
                    }]],
                )
            ],
        )

        self.assertIn(
            "check_release_policy: a published release has no tagName",
            errors,
        )

    def test_first_release_is_allowed_only_at_revision_one(self):
        errors, notes, _ = self.run_check(remote=True, responses=[self.result(["gh"], [[]])])

        self.assertEqual([], errors)
        self.assertEqual(["check_release_policy: no release yet"], notes)

        (self.root / "METAL_REV").write_text("2\n", encoding="ascii")
        errors, _, _ = self.run_check(remote=True, responses=[self.result(["gh"], [[]])])

        self.assertIn(
            "check_release_policy: no published release exists for METAL_REV 2",
            errors,
        )

    def test_remote_check_requires_a_configured_release_repository(self):
        errors, _, calls = self.run_check(remote=True, release_repository=None)

        self.assertIn(
            "check_release_policy: RELEASE_REPOSITORY must be set to owner/repository for --remote",
            errors,
        )
        self.assertEqual([], calls)

    def test_remote_gh_failure_is_reported(self):
        errors, _, _ = self.run_check(
            remote=True,
            responses=[self.result(["gh"], "", status=1, stderr="not authenticated")],
        )

        self.assertIn(
            "check_release_policy: gh api --paginate --slurp repos/owner/repo/releases?per_page=100 "
            "failed: not authenticated",
            errors,
        )


if __name__ == "__main__":
    unittest.main()
