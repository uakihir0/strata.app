from __future__ import annotations

import re
import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
WORKFLOWS = ROOT / ".github" / "workflows"
PINNED_ACTION = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+@[0-9a-f]{40}$")


def load_yaml(path: Path):
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


class WorkflowConfiguration(unittest.TestCase):
    def test_every_action_is_pinned_to_a_full_commit(self):
        workflow_paths = sorted(WORKFLOWS.glob("*.yml"))
        self.assertTrue(workflow_paths, "no workflow YAML files found")

        for path in workflow_paths:
            with self.subTest(path=path.name):
                workflow = load_yaml(path)
                for job_name, job in workflow.get("jobs", {}).items():
                    for index, step in enumerate(job.get("steps", [])):
                        action = step.get("uses")
                        if action is not None:
                            self.assertRegex(
                                action,
                                PINNED_ACTION,
                                f"{path.name}:{job_name} step {index} is not pinned",
                            )

    def test_workflows_request_only_read_access(self):
        # docs/mac/08 section 12.3: the release jobs that read a draft or publish it ask for contents: write.
        write_jobs = {("mac-release.yml", "build"), ("mac-release.yml", "verify"), ("mac-release.yml", "publish")}
        for path in sorted(WORKFLOWS.glob("*.yml")):
            with self.subTest(path=path.name):
                workflow = load_yaml(path)
                self.assertEqual({"contents": "read"}, workflow.get("permissions"))
                for job_name, job in workflow.get("jobs", {}).items():
                    with self.subTest(path=path.name, job=job_name):
                        expected = {"contents": "write"} if (path.name, job_name) in write_jobs else {"contents": "read"}
                        self.assertEqual(
                            expected,
                            job.get("permissions", workflow["permissions"]),
                        )

    def test_reference_mac_workflow_does_not_accept_pull_requests(self):
        workflow = load_yaml(WORKFLOWS / "mac-hardware.yml")
        triggers = workflow["on"]
        self.assertNotIn("pull_request", triggers)
        self.assertNotIn("pull_request_target", triggers)

        jobs = workflow["jobs"]
        self.assertNotIn("dispatch-ref", jobs)
        self.assertEqual("github.ref == 'refs/heads/main'", jobs["hardware"]["if"])
        manual_guard = jobs["hardware"]["steps"][0]["run"]
        self.assertIn('if [ "$GITHUB_EVENT_NAME" != "workflow_dispatch" ]', manual_guard)
        self.assertIn("day must be 1-7", manual_guard)

        for job_name in ("hardware", "weekly"):
            job = jobs[job_name]
            with self.subTest(job=job_name):
                labels = job["runs-on"]
                self.assertIn("self-hosted", labels)
                self.assertIn("strata-ref", labels)
                self.assertIn("github.ref == 'refs/heads/main'", job["if"])
                self.assertEqual("max", job["concurrency"]["queue"])
                self.assertEqual("false", job["concurrency"]["cancel-in-progress"])

    def test_ci_checks_local_release_policy_and_defers_remote_check_until_configured(self):
        workflow = load_yaml(WORKFLOWS / "ci.yml")
        self.assertEqual(["mac-host"], list(workflow["jobs"]))
        steps = workflow["jobs"]["mac-host"]["steps"]
        policy_step = next(step for step in steps if step.get("name") == "Check release policy")
        command = policy_step["run"]

        self.assertIn("tools/metal/check_release_policy.py", command)
        self.assertIn("release_metadata.RELEASE_REPOSITORY", command)
        self.assertIn("--remote", command)
        self.assertIn("::notice::", command)
        self.assertNotIn("GITHUB_HEAD_REF", command)
        self.assertNotIn("GITHUB_EVENT_PULL_REQUEST_HEAD_SHA", policy_step.get("env", {}))
        self.assertIn("::error::", command)
        self.assertIn("exit 1", command)

    def test_python_test_driver_stays_outside_the_project_virtualenv(self):
        expected_calls = {"ci.yml": ("mac-host", 1), "mac-hardware.yml": ("hardware", 2)}
        for filename, (job_name, expected_count) in expected_calls.items():
            workflow = load_yaml(WORKFLOWS / filename)
            steps = workflow["jobs"][job_name]["steps"]
            install = next(
                step for step in steps
                if step.get("name") == "Install pinned Python and build tools"
            )
            capture = 'STRATA_DRIVER_PYTHON=$(python -c \'import sys; print(sys.executable)\')'
            path_update = 'echo "$GITHUB_WORKSPACE/.venv/bin" >> "$GITHUB_PATH"'
            self.assertIn(capture, install["run"])
            self.assertLess(install["run"].index(capture), install["run"].index(path_update))

            calls = [
                step for step in steps
                if "tools/metal/run_python_tests.py" in step.get("run", "")
            ]
            self.assertEqual(expected_count, len(calls), filename)
            for step in calls:
                runner_commands = [
                    line.strip()
                    for line in step["run"].splitlines()
                    if "tools/metal/run_python_tests.py" in line
                ]
                self.assertEqual(
                    ['"$STRATA_DRIVER_PYTHON" tools/metal/run_python_tests.py'],
                    runner_commands,
                    filename,
                )

        developer_workflow = (
            ROOT / "docs" / "mac" / "08-build-and-distribution.md"
        ).read_text(encoding="utf-8")
        self.assertIn("/usr/bin/python3 tools/metal/run_python_tests.py", developer_workflow)

    def test_ci_build_uses_cmake_fetchcontent_without_setup_helpers(self):
        workflow = load_yaml(WORKFLOWS / "ci.yml")
        steps = workflow["jobs"]["mac-host"]["steps"]
        build = next(step for step in steps if step.get("name") == "Configure and build the Mac CI engine")
        self.assertIn("cmake --preset mac-ci", build["run"])
        self.assertNotIn("STRATA_GGML_DIR", build["run"])
        self.assertFalse(any("import setup" in step.get("run", "") for step in steps))

    def test_ci_only_runs_on_pushes_to_protected_branches(self):
        workflow = load_yaml(WORKFLOWS / "ci.yml")
        self.assertEqual({"push"}, set(workflow["on"]))
        self.assertNotIn("pull_request", workflow["on"])
        self.assertNotIn("pull_request_target", workflow["on"])
        self.assertEqual(["main", "release/**"], workflow["on"]["push"]["branches"])

        job = workflow["jobs"]["mac-host"]
        self.assertEqual(
            "github.event_name == 'push' && vars.MAC_HOSTED_RUNNER != ''",
            job["if"],
        )
        self.assertEqual("${{ vars.MAC_HOSTED_RUNNER }}", job["runs-on"])

    def test_ci_metal_measurement_uses_the_configured_engine_path(self):
        workflow = load_yaml(WORKFLOWS / "ci.yml")
        steps = workflow["jobs"]["mac-host"]["steps"]
        measurement = next(
            step for step in steps
            if step.get("name") == "Run the Metal self-test when the engine binary is present"
        )

        self.assertEqual("true", measurement.get("continue-on-error"))
        self.assertIn("hashFiles('build-ci/strata')", measurement["if"])
        self.assertIn("build-ci/strata --list-devices", measurement["run"])
        self.assertNotIn("build-ci/engine/strata", measurement["run"])

    def test_nightly_acceptance_fails_when_required_inputs_are_missing(self):
        workflow = load_yaml(WORKFLOWS / "mac-hardware.yml")
        steps = workflow["jobs"]["hardware"]["steps"]
        acceptance = next(step for step in steps if step.get("name") == "Run model-backed and nightly acceptance gate")
        self.assertIn("inputs.day != ''", acceptance["if"])
        self.assertIn("::error::", acceptance["run"])
        self.assertIn("exit 1", acceptance["run"])
        self.assertIn("ctest --preset mac-model", acceptance["run"])
        self.assertIn(
            "MTL_SHADER_VALIDATION=1 MTL_DEBUG_LAYER=1 ctest --preset mac-metal",
            acceptance["run"],
        )
        self.assertIn("cmake -S tools/vision -B build-vision", acceptance["run"])
        self.assertIn("cmake --build build-vision --target strata-vision", acceptance["run"])
        self.assertNotIn("import setup", acceptance["run"])
        self.assertIn("for f in strata strata.metallib", acceptance["run"])
        self.assertIn('engine/$f', acceptance["run"])
        self.assertIn("engine/strata-vision", acceptance["run"])
        self.assertIn("tools/metal/run_python_tests.py", acceptance["run"])
        self.assertEqual(
            "/Users/Shared/strata-models",
            acceptance["env"]["STRATA_TEST_MODELS"],
        )
        self.assertIn("tools/metal/accept.py nightly", acceptance["run"])

    def test_weekly_acceptance_fails_closed_for_missing_inputs_and_invalid_day(self):
        workflow = load_yaml(WORKFLOWS / "mac-hardware.yml")
        weekly = workflow["jobs"]["weekly"]
        acceptance = next(
            step for step in weekly["steps"]
            if step.get("name") == "Run the requested weekly acceptance slice"
        )

        self.assertIn('if [ ! -d "$STRATA_TEST_MODELS" ]', acceptance["run"])
        self.assertIn("weekly acceptance cannot run", acceptance["run"])
        self.assertIn('if [ ! -f tools/metal/accept.py ]', acceptance["run"])
        self.assertIn('case "$ACCEPTANCE_DAY"', acceptance["run"])
        self.assertIn("day must be 1-7", acceptance["run"])
        self.assertIn("exit 2", acceptance["run"])
        self.assertIn("accept.py weekly --day", acceptance["run"])

    def test_vision_encoder_cmake_is_mac_metal_only(self):
        source = (ROOT / "tools" / "vision" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn("strata-vision is supported on macOS only", source)
        self.assertIn("set(GGML_METAL ON", source)
        self.assertIn("GGML_CUDA", source)
        self.assertIn("GGML_HIP", source)
        self.assertNotIn("STRATA_VISION_CUDA", source)
        # docs/mac/08 section 9: the release recipe's switches, which the release definitions pass.
        self.assertIn("option(STRATA_PORTABLE", source)
        self.assertIn("if(LLAMA_DIR)", source)
        self.assertIn("set(GGML_METAL_EMBED_LIBRARY ON", source)
        self.assertIn("if(STRATA_PORTABLE)\n  set(GGML_NATIVE OFF CACHE BOOL", source)

    def test_release_workflow_publishes_only_a_verified_draft(self):
        workflow = load_yaml(WORKFLOWS / "mac-release.yml")
        self.assertEqual({"push"}, set(workflow["on"]))
        self.assertEqual(["v*-metal.*"], workflow["on"]["push"]["tags"])
        self.assertNotIn("pull_request", workflow["on"])
        self.assertNotIn("pull_request_target", workflow["on"])
        self.assertEqual(
            {"group": "mac-release", "queue": "max", "cancel-in-progress": "false"},
            workflow["concurrency"],
        )

        jobs = workflow["jobs"]
        self.assertEqual(["build", "verify", "publish"], list(jobs))
        self.assertEqual("build", jobs["verify"]["needs"])
        self.assertEqual("verify", jobs["publish"]["needs"])
        self.assertEqual("release", jobs["publish"]["environment"])
        self.assertEqual("ubuntu-24.04", jobs["publish"]["runs-on"])
        self.assertEqual(["self-hosted", "macOS", "ARM64", "strata-ref"], jobs["verify"]["runs-on"])
        self.assertEqual("${{ vars.MAC_HOSTED_RUNNER || 'strata-ref' }}", jobs["build"]["runs-on"])

        steps = {step.get("name"): step for step in jobs["build"]["steps"]}
        create = steps["Create the draft release"]["run"]
        self.assertIn("gh release create", create)
        self.assertIn("--draft", create)
        self.assertIn("--verify-tag", create)
        self.assertIn("dist/SHA256SUMS", create)
        self.assertNotIn("--draft=false", create)
        self.assertIn("tools/metal/check_release_policy.py --remote", steps["Check release policy"]["run"])
        self.assertIn("make_release.py", next(
            step["run"] for step in jobs["build"]["steps"] if "make_release.py" in step.get("run", "")
        ))

        publish = jobs["publish"]["steps"][0]["run"]
        self.assertIn("gh release edit", publish)
        self.assertIn("--draft=false --latest", publish)
        self.assertIn("verify_release.py", next(
            step["run"] for step in jobs["verify"]["steps"] if "verify_release.py" in step.get("run", "")
        ))

    def test_release_signing_is_optional_and_its_keychain_is_always_removed(self):
        workflow = load_yaml(WORKFLOWS / "mac-release.yml")
        steps = workflow["jobs"]["build"]["steps"]
        names = [step.get("name") for step in steps]

        detect = steps[names.index("Detect the signing secrets")]
        self.assertEqual("signing", detect["id"])
        import_step = steps[names.index("Import the Developer ID certificate into a temporary keychain")]
        self.assertEqual("steps.signing.outputs.certificate == 'true'", import_step["if"])
        self.assertIn("security set-key-partition-list", import_step["run"])
        self.assertIn("-T /usr/bin/codesign", import_step["run"])
        self.assertIn("security set-keychain-settings -lut 21600", import_step["run"])
        removal = steps[names.index("Remove the signing keychain and key")]
        self.assertEqual("always()", removal["if"])
        self.assertIn("security delete-keychain", removal["run"])

    def test_ci_builds_and_checks_the_release_files_once_the_builder_exists(self):
        workflow = load_yaml(WORKFLOWS / "ci.yml")
        steps = workflow["jobs"]["mac-host"]["steps"]
        build = next(step for step in steps if step.get("name") == "Build the release files and check them")

        self.assertNotIn("if", build)
        self.assertEqual(".venv/bin/python tools/metal/make_release.py --out dist", build["run"])

    def test_weekly_job_builds_in_its_own_checkout(self):
        workflow = load_yaml(WORKFLOWS / "mac-hardware.yml")
        job = workflow["jobs"]["weekly"]
        steps = job["steps"]
        self.assertTrue(any(step.get("uses", "").startswith("actions/setup-python@") for step in steps))
        build = next(step for step in steps if step.get("name") == "Configure and build the Mac development engine")
        self.assertIn("cmake --preset mac-dev", build["run"])
        self.assertNotIn("STRATA_GGML_DIR", build["run"])
        self.assertTrue(
            any("cmake --preset mac-dev" in step.get("run", "") for step in steps)
        )
        self.assertTrue(
            any("ctest --preset mac-host" in step.get("run", "") for step in steps)
        )
        self.assertTrue(
            any("ctest --preset mac-metal" in step.get("run", "") for step in steps)
        )

    def test_dependabot_checks_github_actions_weekly(self):
        config = load_yaml(ROOT / ".github" / "dependabot.yml")
        updates = config["updates"]
        action_updates = [
            update for update in updates if update.get("package-ecosystem") == "github-actions"
        ]

        self.assertEqual(1, len(action_updates))
        self.assertEqual("/", action_updates[0]["directory"])
        self.assertEqual("weekly", action_updates[0]["schedule"]["interval"])

    def test_hardware_docs_explain_queue_overflow_recovery(self):
        document = (ROOT / "docs" / "mac" / "08-build-and-distribution.md").read_text(
            encoding="utf-8"
        )
        normalized_document = " ".join(document.split())
        self.assertIn("up to 100 pending runs", normalized_document)
        self.assertIn("wait for queue capacity", normalized_document)
        self.assertIn("rerun the missed weekly slice", normalized_document)
        self.assertIn("group: mac-release, queue: max", normalized_document)
        self.assertIn("rerun the canceled tag workflow by run ID", normalized_document)
        self.assertIn("within 30 days", normalized_document)


if __name__ == "__main__":
    unittest.main()
