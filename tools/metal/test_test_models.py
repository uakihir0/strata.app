import contextlib
import hashlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from tools import model_catalog
from tools.metal import run_python_tests
from tools.metal import test_model_headers as header_fixtures
from tools.metal import test_models as tm

MIB = 1 << 20
QUICK = 64 * MIB
# Three quick regions and an unaligned tail: the middle is [QUICK, SIZE - QUICK), the last 64 MiB is not aligned.
SIZE = 3 * QUICK + 4097
NAME = "model.gguf"
TAG = "test-model"
SECTION_2_3_TAGS = {"Q2_0", "IQ2_XS", "IQ3_XXS", "IQ3_S", "coder-IQ1_M", "swift-IQ2_XS", "swift-IQ3_XXS"}
# docs/mac/03-model-reference.md section 2.4: (shard 1, shard 2) in bytes
SECTION_2_4_SIZES = {
    "Q2_0": (37_623_740_192, 28_800_138_432),
    "IQ2_XS": (39_225_954_592, 28_800_138_432),
    "IQ3_XXS": (47_039_860_096, 28_800_138_432),
    "IQ3_S": (54_817_524_224, 28_800_138_432),
    "coder-IQ1_M": (29_608_446_496, 28_800_138_432),
    "swift-IQ2_XS": (39_788_473_344, 28_363_693_824),
    "swift-IQ3_XXS": (39_785_790_560, 36_180_282_560),
}
# header fixture model names (test_model_headers.MODELS) to setup tags (tools/model_catalog.py)
HEADER_FIXTURE_TAGS = {
    "Q2_0": "Q2_0",
    "IQ2_XS": "IQ2_XS",
    "IQ3_XXS": "IQ3_XXS",
    "IQ3_S": "IQ3_S",
    "Coder IQ1_M": "coder-IQ1_M",
    "Swift IQ2_XS": "swift-IQ2_XS",
    "Swift IQ3_XXS": "swift-IQ3_XXS",
}
DENSE_BYTES = {  # docs/mac/03-model-reference.md section 4.5
    "Q2_0": 1_485_606_400,
    "coder-IQ1_M": 1_475_120_640,
    "IQ2_XS": 1_538_035_200,
    "IQ3_XXS": 1_538_035_200,
    "IQ3_S": 1_538_035_200,
    "swift-IQ2_XS": 1_538_035_200,
    "swift-IQ3_XXS": 1_538_035_200,
}


def sha256_span(path: Path, start: int, stop: int) -> str:
    """Independent oracle: SHA-256 of bytes [start, stop), read in 4 MiB blocks."""
    digest = hashlib.sha256()
    remaining = stop - start
    with path.open("rb") as handle:
        handle.seek(start)
        while remaining:
            block = handle.read(min(4 * MIB, remaining))
            if not block:
                raise AssertionError(f"{path} is shorter than {stop}")
            digest.update(block)
            remaining -= len(block)
    return digest.hexdigest()


def file_entry(name, size, *, sha=None, head=None, tail=None, role="shard1"):
    return {
        "role": role,
        "path": name,
        "source": f"https://example.invalid/{name}",
        "bytes": size,
        "sha256": sha,
        "head_64mib_sha256": head,
        "tail_64mib_sha256": tail,
    }


def write_manifest(path: Path, models: dict[str, list[dict]]) -> Path:
    manifest = {
        "schema": 1,
        "models": {
            tag: {
                "label": "test",
                "family": "qwen",
                "quant": "Q2_0",
                "repo": "example/test",
                "revision": "0" * 40,
                "cc_pressure_mib": None,
                "files": files,
                "derived": [],
            }
            for tag, files in models.items()
        },
    }
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def capture(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = tm.main(argv)
    return code, out.getvalue(), err.getvalue()


class SparseModelChecks(unittest.TestCase):
    """Quick and full checks on one 192 MiB sparse file. Changed bytes are restored after each case."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory(prefix="strata-test-models-")
        cls.root = Path(cls._tmp.name)
        cls.models_root = cls.root / "models"
        cls.path = cls.models_root / TAG / NAME
        cls.path.parent.mkdir(parents=True)
        with cls.path.open("wb") as handle:
            handle.truncate(SIZE)  # sparse: the unwritten middle reads as zeros and takes no disk
            handle.seek(0)
            handle.write(b"\x01")
            handle.seek(SIZE - 1)
            handle.write(b"\x02")
        pins = {
            "sha": sha256_span(cls.path, 0, SIZE),
            "head": sha256_span(cls.path, 0, QUICK),
            "tail": sha256_span(cls.path, SIZE - QUICK, SIZE),
        }
        cls.manifest = write_manifest(
            cls.root / "models.json",
            {TAG: [file_entry(NAME, SIZE, sha=pins["sha"], head=pins["head"], tail=pins["tail"])]},
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def run_cli(self, *args, manifest=None, models_root=None):
        return capture([*args, "--manifest", str(manifest or self.manifest),
                        "--models", str(models_root or self.models_root)])

    @contextlib.contextmanager
    def changed_byte(self, offset):
        with self.path.open("r+b") as handle:
            handle.seek(offset)
            original = handle.read(1)
            handle.seek(offset)
            handle.write(bytes([original[0] ^ 0xFF]))
        try:
            yield
        finally:
            with self.path.open("r+b") as handle:
                handle.seek(offset)
                handle.write(original)

    def test_clean_file_passes_quick_and_full(self):
        code, out, err = self.run_cli("check", "--quick")
        self.assertEqual(0, code, err)
        self.assertTrue(out.strip().endswith("1/1 files passed"), out)
        code, out, err = self.run_cli("check")
        self.assertEqual(0, code, err)

    def test_quick_check_fails_for_a_changed_byte_in_the_first_64_mib(self):
        for offset in (0, QUICK - 1):
            with self.subTest(offset=offset), self.changed_byte(offset):
                code, _, err = self.run_cli("check", "--quick")
                self.assertEqual(1, code, err)
                self.assertIn("first 64 MiB", err)

    def test_quick_check_fails_for_a_changed_byte_in_the_last_64_mib(self):
        for offset in (SIZE - QUICK, SIZE - 1):
            with self.subTest(offset=offset), self.changed_byte(offset):
                code, _, err = self.run_cli("check", "--quick")
                self.assertEqual(1, code, err)
                self.assertIn("last 64 MiB", err)

    def test_quick_check_passes_for_a_changed_byte_in_the_middle(self):
        # QUICK is the first byte after the head; SIZE - QUICK - 1 is the last byte before the tail.
        for offset in (QUICK, SIZE // 2, SIZE - QUICK - 1):
            with self.subTest(offset=offset), self.changed_byte(offset):
                code, _, err = self.run_cli("check", "--quick")
                self.assertEqual(0, code, err)

    def test_full_check_fails_for_a_changed_byte_in_the_middle(self):
        for offset in (QUICK, SIZE // 2, SIZE - QUICK - 1):
            with self.subTest(offset=offset), self.changed_byte(offset):
                code, _, err = self.run_cli("check")
                self.assertEqual(1, code, err)
                self.assertIn("sha256", err)

    def test_full_check_fails_for_a_changed_byte_at_either_end(self):
        for offset in (0, SIZE - 1):
            with self.subTest(offset=offset), self.changed_byte(offset):
                code, _, err = self.run_cli("check")
                self.assertEqual(1, code, err)

    def test_a_longer_file_fails_both_checks_on_size(self):
        with self.path.open("ab") as handle:
            handle.write(b"\0")
        try:
            for mode in (("check", "--quick"), ("check",)):
                with self.subTest(mode=mode):
                    code, _, err = self.run_cli(*mode)
                    self.assertEqual(1, code, err)
                    self.assertIn(f"want {SIZE:,} B", err)
        finally:
            os.truncate(self.path, SIZE)

    def test_missing_model_file_exits_2(self):
        moved = self.path.with_name("moved.gguf")
        os.rename(self.path, moved)
        try:
            for mode in (("check", "--quick"), ("check",)):
                with self.subTest(mode=mode):
                    code, _, err = self.run_cli(*mode)
                    self.assertEqual(2, code, err)
                    self.assertIn("absent", err)
        finally:
            os.rename(moved, self.path)

    def test_missing_models_folder_exits_2(self):
        code, _, err = self.run_cli("check", "--quick", models_root=self.root / "absent")
        self.assertEqual(2, code)
        self.assertIn("models folder is missing", err)

    def test_models_folder_comes_from_the_environment_by_default(self):
        with mock.patch.dict(os.environ, {"STRATA_TEST_MODELS": str(self.root / "absent")}):
            code, _, err = capture(["check", "--quick", "--manifest", str(self.manifest)])
        self.assertEqual(2, code, err)

    def test_null_pins_fail_instead_of_passing(self):
        unpinned_head = write_manifest(self.root / "no-head.json", {TAG: [
            file_entry(NAME, SIZE, sha="0" * 64, head=None, tail="0" * 64)]})
        code, _, err = self.run_cli("check", "--quick", manifest=unpinned_head)
        self.assertEqual(1, code, err)
        self.assertIn("head_64mib_sha256 is null", err)
        unpinned_sha = write_manifest(self.root / "no-sha.json", {TAG: [
            file_entry(NAME, SIZE, sha=None, head="0" * 64, tail="0" * 64)]})
        code, _, err = self.run_cli("check", manifest=unpinned_sha)
        self.assertEqual(1, code, err)
        self.assertIn("sha256 is null", err)

    def test_identical_bytes_behind_two_hard_links_are_both_checked(self):
        second = self.models_root / "second-model"
        second.mkdir()
        os.link(self.path, second / NAME)
        try:
            pins = file_entry(NAME, SIZE, sha=self.manifest_pins()["sha"],
                              head=self.manifest_pins()["head"], tail=self.manifest_pins()["tail"])
            manifest = write_manifest(self.root / "linked.json", {TAG: [pins], "second-model": [pins]})
            code, out, err = self.run_cli("check", "--quick", manifest=manifest)
            self.assertEqual(0, code, err)
            self.assertTrue(out.strip().endswith("2/2 files passed"), out)
        finally:
            (second / NAME).unlink()
            second.rmdir()

    def manifest_pins(self):
        entry = json.loads(self.manifest.read_text(encoding="utf-8"))["models"][TAG]["files"][0]
        return {"sha": entry["sha256"], "head": entry["head_64mib_sha256"], "tail": entry["tail_64mib_sha256"]}

    def test_unknown_tag_is_a_usage_error(self):
        with self.assertRaises(SystemExit) as caught:
            self.run_cli("check", "no-such-model")
        self.assertEqual(2, caught.exception.code)

    def test_manifest_rejects_a_pin_that_is_not_lowercase_hex(self):
        bad = write_manifest(self.root / "bad-pin.json", {TAG: [file_entry(NAME, SIZE, sha="A" * 64)]})
        with self.assertRaises(tm.ManifestError):
            tm.load_manifest(bad)

    def test_manifest_rejects_a_duplicate_key(self):
        bad = self.root / "dup.json"
        bad.write_text('{"schema": 1, "schema": 1, "models": {}}', encoding="utf-8")
        with self.assertRaises(tm.ManifestError):
            tm.load_manifest(bad)


class RealManifestTests(unittest.TestCase):
    def setUp(self):
        self.manifest = tm.load_manifest()

    def files(self, tag):
        return {entry["role"]: entry for entry in self.manifest["models"][tag]["files"]}

    def test_every_model_of_section_2_3_is_listed(self):
        self.assertEqual(SECTION_2_3_TAGS, set(self.manifest["models"]))

    def test_shard_sizes_match_section_2_4(self):
        for tag, (shard1, shard2) in SECTION_2_4_SIZES.items():
            with self.subTest(tag=tag):
                self.assertEqual(shard1, self.files(tag)["shard1"]["bytes"])
                self.assertEqual(shard2, self.files(tag)["shard2"]["bytes"])

    def test_shard_sizes_match_the_header_fixtures(self):
        for name, (first, second) in header_fixtures.MODELS.items():
            with self.subTest(model=name):
                files = self.files(HEADER_FIXTURE_TAGS[name])
                self.assertEqual(header_fixtures.SHARD_LAYOUT[first][0], files["shard1"]["bytes"])
                self.assertEqual(header_fixtures.SHARD_LAYOUT[second][0], files["shard2"]["bytes"])

    def test_mmproj_entries_exist_for_q2_0_and_coder_only(self):
        with_mmproj = {tag for tag in self.manifest["models"] if "mmproj" in self.files(tag)}
        self.assertEqual({"Q2_0", "coder-IQ1_M"}, with_mmproj)
        for tag in with_mmproj:
            with self.subTest(tag=tag):
                self.assertEqual("mmproj-Qwen3.8-Flash-Next-BF16.gguf", self.files(tag)["mmproj"]["path"])
                self.assertEqual(907_543_008, self.files(tag)["mmproj"]["bytes"])

    def test_every_model_has_cc_pressure_mib_as_null_or_a_positive_integer(self):
        for tag, model in self.manifest["models"].items():
            with self.subTest(tag=tag):
                self.assertIn("cc_pressure_mib", model)
                value = model["cc_pressure_mib"]
                self.assertTrue(value is None or (isinstance(value, int) and not isinstance(value, bool) and value > 0))

    def test_sources_are_at_the_pinned_revision_and_name_the_file(self):
        for tag, model in self.manifest["models"].items():
            self.assertEqual(model_catalog.HF_REVISIONS[model["repo"]], model["revision"])
            for entry in model["files"]:
                with self.subTest(tag=tag, file=entry["path"]):
                    self.assertTrue(entry["source"].startswith(f"https://huggingface.co/{model['repo']}/resolve/"))
                    self.assertIn(f"/resolve/{model['revision']}/", entry["source"])
                    self.assertTrue(entry["source"].endswith("/" + entry["path"]))

    def test_derived_outputs_have_the_sizes_the_spec_gives(self):
        for tag, model in self.manifest["models"].items():
            derived = {item["path"]: item["bytes"] for item in model["derived"]}
            with self.subTest(tag=tag):
                self.assertEqual(823_887_872, derived["mtp/rt/"])
                self.assertEqual(DENSE_BYTES[tag], derived["pack/dense.bin"])


class RegistrationTests(unittest.TestCase):
    def test_runner_discovers_these_unit_tests_and_not_the_cli(self):
        modules = run_python_tests.discover_modules()
        self.assertIn("tools.metal.test_test_models", modules)
        self.assertNotIn("tools.metal.test_models", modules)


class FetchTests(unittest.TestCase):
    def test_plan_lists_sources_and_sizes_without_a_models_folder(self):
        code, out, err = capture(["fetch", "--plan", "--models", "/nonexistent/strata", "Q2_0"])
        self.assertEqual(0, code, err)
        self.assertIn("Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf 37,623,740,192 B", out)
        self.assertIn("https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF/resolve/"
                      "ed59f92082b1e93c0e96d60a8b11aab089b52f09/Q2_0/"
                      "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf", out)

    def test_fetch_refuses_without_setup_and_writes_nothing(self):
        with tempfile.TemporaryDirectory(prefix="strata-fetch-") as temporary:
            models_root = Path(temporary) / "models"
            models_root.mkdir()
            code, _, err = capture(["fetch", "--models", str(models_root), "Q2_0"])
            self.assertEqual(1, code, err)
            self.assertIn("no model was fetched", err)
            self.assertEqual([], list(models_root.iterdir()))

    def test_fetch_without_a_models_folder_exits_2(self):
        with tempfile.TemporaryDirectory(prefix="strata-fetch-") as temporary:
            code, _, err = capture(["fetch", "--models", str(Path(temporary) / "absent"), "Q2_0"])
        self.assertEqual(2, code, err)


if __name__ == "__main__":
    unittest.main()
