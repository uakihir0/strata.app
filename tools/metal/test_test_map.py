import json
import os
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

from tools.metal import test_map


class TestMapTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name) / "repo"
        for relative in (
            "docs/mac/01-requirements.md",
            "docs/mac/09-testing-and-validation.md",
            "tests/metal/test_map.json",
            "tests/metal/requirements.json",
            "tests/metal/reviews.json",
        ):
            source = test_map.ROOT / relative
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        for source_dir in (test_map.ROOT / "tools", test_map.ROOT / "serve"):
            for source in source_dir.glob("test_*.py"):
                target = self.root / source.relative_to(test_map.ROOT)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source, target)
        for relative in (
            "tools/metal/test_check_drift.py",
            "tools/metal/test_dense_vs_gguf.py",
            ".github/workflows/ci.yml",
        ):
            source = test_map.ROOT / relative
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)

        registry = json.loads((self.root / "tests/metal/test_map.json").read_text())
        active = [
            {
                "name": name,
                "properties": [{"name": "LABELS", "value": [entry["label"]]}],
            }
            for name, entry in registry["tests"].items()
            if "planned" not in entry
        ]
        self.ctest_json = json.dumps({"tests": active})
        self.environ = mock.patch.dict(os.environ, {"STRATA_BUILD_DIR": str(self.root / "build")})
        self.environ.start()

    def tearDown(self):
        self.environ.stop()
        self.temporary.cleanup()

    def run_map(self, argv, ctest_json=None):
        result = mock.Mock(
            returncode=0,
            stdout=self.ctest_json if ctest_json is None else ctest_json,
            stderr="",
        )
        output = StringIO()
        errors = StringIO()
        with mock.patch("tools.metal.test_map.subprocess.run", return_value=result):
            with redirect_stdout(output), redirect_stderr(errors):
                code = test_map.main(self.root, argv)
        return code, output.getvalue(), errors.getvalue()

    def set_requirement_check(self, req_id, check):
        doc_path = self.root / "docs/mac/01-requirements.md"
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        found = False
        for index, line in enumerate(lines):
            if line.startswith(f"| {req_id} |"):
                cells = line.split("|")
                cells[3] = f" `{check}` "
                lines[index] = "|".join(cells)
                found = True
                break
        self.assertTrue(found, f"missing requirement row {req_id}")
        doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        requirements["requirements"][req_id] = [check]
        requirements_path.write_text(json.dumps(requirements))

    def test_cmake_only_test_fails_with_the_documented_message(self):
        payload = json.loads(self.ctest_json)
        payload["tests"].append(
            {
                "name": "only_cmake",
                "properties": [{"name": "LABELS", "value": ["host"]}],
            }
        )
        code, _, errors = self.run_map([], json.dumps(payload))

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: only_cmake: in CTest but not in tests/metal/test_map.json",
            errors,
        )

    def test_final_fails_for_unchecked_requirements_without_planned_tests(self):
        registry_path = self.root / "tests/metal/test_map.json"
        registry = json.loads(registry_path.read_text())
        for entry in registry["tests"].values():
            entry.pop("planned", None)
        registry_path.write_text(json.dumps(registry))

        code, _, errors = self.run_map(["--final"])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-PLT-01: no check yet", errors)
        self.assertNotIn(": planned (", errors)

    def test_final_requires_a_configured_ctest_build(self):
        with mock.patch.dict(os.environ, {"STRATA_BUILD_DIR": ""}):
            code, _, errors = self.run_map(["--final"])

        self.assertEqual(1, code)
        self.assertIn("test_map: CTest check skipped (STRATA_BUILD_DIR is not set)", errors)

    def test_review_url_requires_a_host(self):
        (self.root / "tests/metal/reviews.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "reviews": [
                        {
                            "review": "family",
                            "commit": "a" * 40,
                            "url": "https://?/evidence",
                            "by": "reviewer",
                            "date": "2026-10-03",
                        }
                    ],
                }
            )
        )
        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: reviews[0]: url must be an https URL", errors)

    def test_review_url_rejects_leading_c0_control_character(self):
        path = self.root / "tests/metal/reviews.json"
        path.write_text(
            json.dumps(
                {
                    "schema": 1,
                    "reviews": [
                        {
                            "review": "family",
                            "commit": "a" * 40,
                            "url": "\x00https://example.com/evidence",
                            "by": "reviewer",
                            "date": "2026-10-03",
                        }
                    ],
                }
            )
        )

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: reviews[0]: url must be an https URL", errors)

    def test_review_url_rejects_malformed_percent_escape_in_host(self):
        (self.root / "tests/metal/reviews.json").write_text(
            json.dumps(
                {
                    "schema": 1,
                    "reviews": [
                        {
                            "review": "family",
                            "commit": "a" * 40,
                            "url": "https://exa%mple.com/evidence",
                            "by": "reviewer",
                            "date": "2026-10-03",
                        }
                    ],
                }
            )
        )
        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: reviews[0]: url must be an https URL", errors)

    def test_boolean_schema_is_not_version_one(self):
        cases = (
            (
                "tests/metal/test_map.json",
                "test_map: tests/metal/test_map.json: schema must be 1",
            ),
            (
                "tests/metal/requirements.json",
                "test_map: tests/metal/requirements.json: schema must be 1",
            ),
            (
                "tests/metal/reviews.json",
                "test_map: tests/metal/reviews.json: schema must be 1",
            ),
        )
        for relative, expected in cases:
            with self.subTest(relative=relative):
                path = self.root / relative
                data = json.loads(path.read_text())
                data["schema"] = True
                path.write_text(json.dumps(data))

                code, _, errors = self.run_map([])

                self.assertEqual(1, code)
                self.assertIn(expected, errors)
                source = test_map.ROOT / relative
                shutil.copyfile(source, path)

    def test_python_check_cannot_escape_the_selected_root(self):
        outside = self.root.parent / "escape.py"
        outside.write_text("# deliberately outside the fixture root\n")
        doc_path = self.root / "docs/mac/09-testing-and-validation.md"
        doc_text = doc_path.read_text(encoding="utf-8")
        marker = "### 9.4 `tools/metal/test_map.py`"
        doc_path.write_text(
            doc_text.replace(marker, "`../escape.py`\n\n" + marker),
            encoding="utf-8",
        )
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        req_id = next(iter(requirements["requirements"]))
        requirements["requirements"][req_id] = ["python:../escape.py"]
        requirements_path.write_text(json.dumps(requirements))

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            f"test_map: {req_id}: check path escapes root: python:../escape.py",
            errors,
        )

    def test_ci_check_cannot_escape_the_workflows_root(self):
        outside = self.root.parent / "escape.yml"
        outside.write_text("jobs:\n  escaped:\n", encoding="utf-8")
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        req_id = next(iter(requirements["requirements"]))
        requirements["requirements"][req_id] = ["ci:../../../escape.yml/escaped"]
        requirements_path.write_text(json.dumps(requirements))

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            f"test_map: {req_id}: malformed check ci:../../../escape.yml/escaped",
            errors,
        )

    def test_ci_check_cannot_follow_a_workflows_symlink_outside_root(self):
        external_workflows = self.root.parent / "external-workflows"
        external_workflows.mkdir()
        (external_workflows / "ci.yml").write_text("jobs:\n  escaped:\n", encoding="utf-8")
        github_dir = self.root / ".github"
        shutil.rmtree(github_dir)
        github_dir.mkdir()
        (github_dir / "workflows").symlink_to(external_workflows, target_is_directory=True)
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        req_id = next(iter(requirements["requirements"]))
        requirements["requirements"][req_id] = ["ci:ci.yml/escaped"]
        requirements_path.write_text(json.dumps(requirements))

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            f"test_map: {req_id}: workflow path escapes root: ci:ci.yml/escaped",
            errors,
        )

    def test_requirement_checks_must_match_the_verified_by_cell(self):
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        requirements["requirements"]["REQ-MNT-01"] = ["python:tools/test_removed_pc_tool.py"]
        requirements_path.write_text(json.dumps(requirements))

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: REQ-MNT-01: checks not listed in the requirement's Verified by cell: "
            "python:tools/test_removed_pc_tool.py",
            errors,
        )

    def test_final_rejects_a_requirement_with_missing_documented_checks(self):
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        requirements["requirements"]["REQ-MNT-01"] = ["python:tools/metal/test_check_drift.py"]
        requirements_path.write_text(json.dumps(requirements))
        target = self.root / "tools/metal/test_check_drift.py"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(test_map.ROOT / "tools/metal/test_check_drift.py", target)

        code, _, errors = self.run_map(["--final"])

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: REQ-MNT-01: missing checks: ci:ci.yml/mac-host",
            errors,
        )

    def test_python_glob_does_not_match_a_nested_module(self):
        nested = self.root / "tools/test_helper/nested.py"
        nested.parent.mkdir(parents=True)
        nested.write_text("# nested file\n", encoding="utf-8")
        requirements_path = self.root / "tests/metal/requirements.json"
        requirements = json.loads(requirements_path.read_text())
        requirements["requirements"]["REQ-SRV-12"] = ["python:tools/test_helper/nested.py"]
        requirements_path.write_text(json.dumps(requirements))

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: REQ-SRV-12: unknown check python:tools/test_helper/nested.py",
            errors,
        )

    def test_mac_module_count_excludes_separately_documented_tests(self):
        test_doc = (self.root / "docs/mac/09-testing-and-validation.md").read_text(
            encoding="utf-8"
        )
        requirements_doc = (self.root / "docs/mac/01-requirements.md").read_text(
            encoding="utf-8"
        )
        errors = []
        req_rows = test_map.documented_requirement_rows(requirements_doc, errors)
        checks = test_map.documented_requirement_checks(
            self.root,
            "REQ-SRV-12",
            req_rows["REQ-SRV-12"][1],
            test_doc,
            errors,
        )

        self.assertEqual([], errors)
        self.assertEqual(20, len(checks))
        self.assertNotIn("python:serve/test_macos.py", checks)

    def test_ci_parser_ignores_jobs_text_inside_a_block_scalar(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(
            "name: ci\nrun-name: |\n  jobs:\n  fake:\non:\n  push:\n"
            "jobs:\n  real:\n    runs-on: ubuntu-latest\n    steps:\n"
            "      - run: echo ok\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/fake")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-BLD-08: unknown check ci:ci.yml/fake", errors)

    def test_ci_reference_accepts_yaml_extension(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yaml").write_text(
            "name: ci\non:\n  push:\njobs:\n    build:\n"
            "        runs-on: ubuntu-latest\n        steps:\n"
            "            - run: echo ok\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yaml/build")

        code, _, errors = self.run_map([])

        self.assertEqual(0, code, errors)

    def test_ci_parser_accepts_valid_four_space_job_indentation(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(
            "name: ci\non:\n  push:\njobs:\n"
            "    build:\n        runs-on: ubuntu-latest\n"
            "        steps:\n            - run: echo ok\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-MNT-01", "python:tools/metal/test_check_drift.py")
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/build")

        code, _, errors = self.run_map([])

        self.assertEqual(0, code, errors)

    def test_ci_parser_rejects_job_without_mapping_value(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(
            "name: ci\non:\n  push:\njobs:\n  build:\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/build")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-BLD-08: unknown check ci:ci.yml/build", errors)

    def test_ci_parser_rejects_sequence_as_job_value(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(
            "name: ci\non:\n  push:\njobs:\n"
            "  build:\n    - runs-on: ubuntu-latest\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/build")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-BLD-08: unknown check ci:ci.yml/build", errors)

    def test_ci_parser_rejects_invalid_job_id(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_text(
            "name: ci\non:\n  push:\njobs:\n"
            "  -fake:\n    runs-on: ubuntu-latest\n"
            "    steps:\n      - run: echo ok\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/-fake")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-BLD-08: unknown check ci:ci.yml/-fake", errors)

    def test_documented_ctest_group_label_must_match_map_label(self):
        doc_path = self.root / "docs/mac/09-testing-and-validation.md"
        lines = doc_path.read_text(encoding="utf-8").splitlines()
        in_test_tables = False
        host_test_line = None
        metal_test_line = None
        for index, line in enumerate(lines):
            if line == "### 9.1 CTest tests":
                in_test_tables = True
            elif line == "### 9.2 Tests outside CTest":
                in_test_tables = False
            if not in_test_tables:
                continue
            if "`platform_memory_test`" in line:
                host_test_line = index
            elif "`metal_context_test`" in line:
                metal_test_line = index
        self.assertIsNotNone(host_test_line)
        self.assertIsNotNone(metal_test_line)
        lines[host_test_line] = lines[host_test_line].replace(
            "`platform_memory_test`", "`metal_context_test`"
        )
        lines[metal_test_line] = lines[metal_test_line].replace(
            "`metal_context_test`", "`platform_memory_test`"
        )
        doc_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: platform_memory_test: documented label metal, the map says host",
            errors,
        )
        self.assertIn(
            "test_map: metal_context_test: documented label host, the map says metal",
            errors,
        )

    def test_criterion_comment_and_string_do_not_define_an_id(self):
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_text(
            "# FAKE.metric\nDESCRIPTION = 'ALSO.fake'\n"
            "CRITERIA = [('REAL.metric', 'eq', 1.0, True)]\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-FUN-11", "accept:FAKE.metric")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-FUN-11: unknown check accept:FAKE.metric", errors)

    def test_criterion_id_is_read_from_a_table_row(self):
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_text(
            "CRITERIA = [('REAL.metric', 'eq', 1.0, True)]\n"
            "# FAKE.metric\nDESCRIPTION = 'ALSO.fake'\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-FUN-11", "accept:REAL.metric")

        code, _, errors = self.run_map([])

        self.assertEqual(0, code, errors)

    def test_criterion_constructor_outside_the_table_is_not_a_declaration(self):
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_text(
            "example = Criterion('FAKE.metric', 'eq', 1.0, True)\n"
            "CRITERIA = [('REAL.metric', 'eq', 1.0, True)]\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-FUN-11", "accept:FAKE.metric")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-FUN-11: unknown check accept:FAKE.metric", errors)

    def test_criterion_mapping_key_must_match_the_row_id(self):
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_text(
            "CRITERIA = {'REAL.metric': Criterion('FAKE.metric', 'eq', 1.0, True)}\n",
            encoding="utf-8",
        )
        self.set_requirement_check("REQ-FUN-11", "accept:REAL.metric")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-FUN-11: unknown check accept:REAL.metric", errors)

    def test_criterion_source_with_invalid_utf8_is_reported(self):
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.write_bytes(b"\xff")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn(
            "test_map: cannot read tools/metal/accept/criteria.py:",
            errors,
        )

    def test_workflow_with_invalid_utf8_does_not_crash(self):
        workflows = self.root / ".github/workflows"
        workflows.mkdir(parents=True, exist_ok=True)
        (workflows / "ci.yml").write_bytes(b"jobs:\n  build:\n\xff")
        self.set_requirement_check("REQ-BLD-08", "ci:ci.yml/build")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: REQ-BLD-08: unknown check ci:ci.yml/build", errors)

    def test_criteria_file_symlink_cannot_escape_selected_root(self):
        outside = self.root.parent / "criteria-outside.py"
        outside.write_text(
            "CRITERIA = [('FAKE.metric', 'eq', 1.0, True)]\n",
            encoding="utf-8",
        )
        criteria_path = self.root / "tools/metal/accept/criteria.py"
        criteria_path.parent.mkdir(parents=True, exist_ok=True)
        criteria_path.symlink_to(outside)
        self.set_requirement_check("REQ-FUN-11", "accept:FAKE.metric")

        code, _, errors = self.run_map([])

        self.assertEqual(1, code)
        self.assertIn("test_map: tools/metal/accept/criteria.py escapes --root", errors)


if __name__ == "__main__":
    unittest.main()
