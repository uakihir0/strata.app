"""Tests for the acceptance runner (docs/mac/09-testing-and-validation.md, section 3.2).

Each criterion is judged on equal inputs (it passes) and on its planted difference (it fails, with the message of
section 3.2). The engine parser is tested on captured output. The runner is tested end to end with
tools/metal/accept/fake_engine.py, through a test-only run registry, because no production run has a measurement
yet. Production runs are tested to stop with NOT RUN before any step starts.
"""
from __future__ import annotations

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import time
import struct
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from accept import checks, criteria, engine, result, runs  # noqa: E402
from accept import steps as stepper  # noqa: E402

FAKE_ENGINE = HERE / "accept" / "fake_engine.py"
ACCEPT_CLI = HERE / "accept.py"
UPSTREAM = "1678de33d0e0711bc414ad992b640e1a37dd814"

SPEC_IDS = (
    "C1.1.top1 C1.1.kl C1.1.safe C1.1.band C1.1.ppl C1.1.p99 C1.2 C1.3 C1.4.ids C1.4.weights C1.5 "
    "C2.b C2.c C2.d C2.e C2.f C2.g C2.h C2.i C2.j C2.d.accept "
    "C3.reuse C3.pressure C3.oversized C3.exchange C3.admission C3.growth C3.isolation C3.soak C3.http "
    "C3.disabled C3.reread C3.repeat C4.tiered C4.swaps PP.top1 PP.kl PP.ppl PP.chunk "
    "NEEDLE.32k NEEDLE.128k NEEDLE.262k S.models S.status S.openai S.stream S.tool S.anthropic S.image "
    "S.cancel S.unload BENCH.mtp BENCH.mtp.target BENCH.plain BENCH.other PF-1 PF-2 REG.decode REG.prefill "
    "REG.trend PRESSURE.smoke G1.tf G1.free G2.speed PLAN.models"
).split()

# id: (threshold used for the case, equal value that passes, planted difference that fails)
CASES: dict[str, tuple[object, object, object]] = {
    "C1.1.top1": (95.0, 95.0, 94.9),
    "C1.1.kl": (0.5, 0.5, 0.51),
    "C1.1.safe": (100.0, 100.0, 99.5),
    "C1.1.band": (40.0, 40.0, 36.9),
    "C1.1.ppl": (0.005, 0.005, 0.006),
    "C1.1.p99": (0.02, 0.02, 0.03),
    "C1.2": (96.0, 96.0, 95.0),
    "C1.3": (0, 0, 2),
    "C1.4.ids": (0, 0, 1),
    "C1.4.weights": (1e-5, 1e-5, 2e-5),
    "C1.5": (100.0, 100.0, 99.6),
    "C2.b": (0, 0, 1), "C2.c": (0, 0, 1), "C2.d": (0, 0, 1), "C2.e": (0, 0, 1), "C2.f": (0, 0, 1),
    "C2.g": (0, 0, 1), "C2.h": (0, 0, 1), "C2.i": (0, 0, 1), "C2.j": (0, 0, 1),
    "C2.d.accept": (95.0, 95.0, 94.0),
    "C3.reuse": (0, 0, 1), "C3.pressure": (0, 0, 1), "C3.oversized": (0, 0, 1), "C3.exchange": (0, 0, 1),
    "C3.admission": (0, 0, 1), "C3.growth": (0, 0, 1), "C3.isolation": (0, 0, 1), "C3.soak": (0, 0, 1),
    "C3.http": (0, 0, 1), "C3.disabled": (0, 0, 1), "C3.reread": (0, 0, 1), "C3.repeat": (0, 0, 1),
    "C4.tiered": (0, 0, 1), "C4.swaps": (0, 0, 1),
    "PP.top1": (96.0, 96.0, 95.9),
    "PP.kl": (0.5, 0.5, 0.51),
    "PP.ppl": (0.03, 0.03, 0.031),
    "PP.chunk": (None, 4096, 2048),
    "NEEDLE.32k": (0, 0, 1), "NEEDLE.128k": (0, 0, 1), "NEEDLE.262k": (0, 0, 1),
    "S.models": (True, True, False), "S.status": (True, True, False), "S.openai": (True, True, False),
    "S.stream": (True, True, False), "S.tool": (True, True, False), "S.anthropic": (True, True, False),
    "S.image": (True, True, False), "S.cancel": (True, True, False), "S.unload": (True, True, False),
    "BENCH.mtp": (50.0, 50.0, 49.9),
    "BENCH.mtp.target": (60.0, 60.0, 59.9),
    "BENCH.plain": (20.0, 20.0, 19.9),
    "BENCH.other": (None, 25.0, 1.0),
    "PF-1": (0.70, 0.70, 0.69),
    "PF-2": (450.0, 450.0, 449.0),
    "REG.decode": (0.95, 0.95, 0.94),
    "REG.prefill": (0.95, 0.95, 0.94),
    "REG.trend": (None, 0.0, 5.0),
    "PRESSURE.smoke": (True, True, False),
    "G1.tf": (0, 0, 1), "G1.free": (0, 0, 1),
    "G2.speed": (20.0, 20.0, 19.9),
    "PLAN.models": (0, 0, 1),
    "STEP.exit": (0, 0, 3),
    "STEP.timeout": (0, 0, 1),
}


def load_cli():
    spec = importlib.util.spec_from_file_location("accept_cli", ACCEPT_CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACCEPT = load_cli()

# Captured output: the one-shot lines the engine's formatter prints for one run (src/metal/program/oneshot.cpp),
# as the C++ test tests/metal/oneshot_output_test.cpp expects them. It is not a CUDA capture: the CUDA reference
# outputs of T-1 are not in the repository yet.
CAPTURED_ONESHOT = (
    "speculation              2 rounds of 4, drafts accepted 3 of 6 (0.500), 2.50 tokens per round\n"
    "window sizes             T1:1 T2:1 T3:0 T4:0  (min draft probability 0.25)\n"
    "prompt  : 11 12\n"
    "output  : 21 22\n"
    "decode                   2 tokens in 5.0 ms  ->  400.00 tok/s\n"
    "prefill                  1 tokens in 4.0 ms  ->  250.00 tok/s  (time to first token 6.5 ms)\n"
    "mixed dumped             mixed.bin (n_embd 2560, rms 0.1, mean|.| 0.2)\n"
    "gpu bandwidth            100.0 GB/s over the window\n"
)


def run_fake(args: list[str], plant: str | None = None, timeout: float = 60) -> subprocess.CompletedProcess:
    env = {key: value for key, value in os.environ.items() if not key.startswith("STRATA_")}
    env.pop("FAKE_ENGINE_PLANT", None)
    if plant is not None:
        env["FAKE_ENGINE_PLANT"] = plant
    return subprocess.run([sys.executable, str(FAKE_ENGINE), *args], capture_output=True, text=True,
                          env=env, timeout=timeout, check=False)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class CriteriaTable(unittest.TestCase):
    def test_table_has_every_id_of_section_3_2(self):
        self.assertEqual(set(SPEC_IDS) - set(criteria.CRITERIA), set())

    def test_no_row_outside_the_spec_except_runner_rows(self):
        extra = set(criteria.CRITERIA) - set(SPEC_IDS)
        self.assertEqual(extra, {"STEP.exit", "STEP.timeout"})

    def test_every_row_has_a_planted_case(self):
        self.assertEqual(set(criteria.CRITERIA) - set(CASES), set())

    def test_every_run_criterion_is_in_the_table(self):
        for spec in runs.RUN_LIST:
            for cid in spec.criteria:
                self.assertIn(cid, criteria.CRITERIA, f"{spec.name} names unknown {cid}")

    def test_every_spec_id_belongs_to_some_run(self):
        covered = {cid for spec in runs.RUN_LIST for cid in spec.criteria}
        self.assertEqual(set(SPEC_IDS) - covered, set())

    def test_report_rows_are_not_gating(self):
        for row in criteria.CRITERIA.values():
            if row.op == criteria.REPORT:
                self.assertFalse(row.gating, row.id)

    def test_test_map_reads_the_same_ids(self):
        from tools.metal import test_map

        declared = test_map.declared_criterion_ids(criteria_source())
        matching = {cid for cid in criteria.CRITERIA if test_map.CRITERION_ID.fullmatch(cid)}
        self.assertEqual(declared, matching)

    def test_derived_threshold_is_refused_without_a_value(self):
        with self.assertRaises(ValueError):
            criteria.judge("C1.1.top1", 99.0)

    def test_missing_value_never_passes(self):
        verdict = criteria.judge("C2.b", None, 0)
        self.assertIs(verdict.passed, False)
        self.assertTrue(verdict.blocks)

    def test_unknown_id_is_an_error(self):
        with self.assertRaises(ValueError):
            criteria.judge("C9.9", 0, 0)


def criteria_source() -> str:
    return (HERE / "accept" / "criteria.py").read_text(encoding="utf-8")


def _make_criterion_test(cid: str, threshold, ok, bad):
    def test(self):
        row = criteria.get(cid)
        if row.threshold is not None:
            self.assertEqual(row.threshold, threshold, f"{cid}: case threshold differs from the row")
        if row.op == criteria.REPORT:
            verdict = criteria.judge(cid, ok)
            self.assertIsNone(verdict.passed)
            self.assertFalse(verdict.blocks)
            verdict = criteria.judge(cid, bad)
            self.assertIsNone(verdict.passed)
            self.assertFalse(verdict.blocks, "a reported value never fails a run")
            return
        verdict = criteria.judge(cid, ok, threshold)
        self.assertIs(verdict.passed, True, f"{cid} must pass on equal inputs")
        self.assertFalse(verdict.blocks)
        verdict = criteria.judge(cid, bad, threshold)
        self.assertIs(verdict.passed, False, f"{cid} must fail on its planted difference")
        expected = f"{cid}: {criteria.format_value(bad)}, want {row.op} {criteria.format_value(threshold)}"
        self.assertEqual(verdict.message, expected)
        self.assertEqual(verdict.blocks, row.gating, f"{cid}: blocking must follow the gating flag")

    return test


class CriterionPerRow(unittest.TestCase):
    """One test method per criterion ID (pass on equal inputs, fail with the message on the planted difference)."""


for _cid, (_thr, _ok, _bad) in CASES.items():
    _name = "test_" + "".join(ch if ch.isalnum() else "_" for ch in _cid)
    setattr(CriterionPerRow, _name, _make_criterion_test(_cid, _thr, _ok, _bad))


class EngineParser(unittest.TestCase):
    def test_captured_output_parses(self):
        parsed = engine.parse_one_shot(CAPTURED_ONESHOT, source="captured", need_prefill=True,
                                       need_speculation=True)
        self.assertEqual(parsed.prompt, (11, 12))
        self.assertEqual(parsed.output, (21, 22))
        self.assertEqual(parsed.decode.tokens, 2)
        self.assertEqual(parsed.decode.tok_s, 400.0)
        self.assertEqual(parsed.prefill.timing.tok_s, 250.0)
        self.assertEqual(parsed.prefill.time_to_first_token_ms, 6.5)
        self.assertEqual(parsed.speculation.accepted, 3)
        self.assertEqual(parsed.speculation.drafts, 6)
        self.assertEqual(parsed.speculation.acceptance, 0.5)

    def test_missing_decode_line_fails_with_the_spec_message(self):
        text = "\n".join(line for line in CAPTURED_ONESHOT.splitlines() if not line.startswith("decode"))
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/003.out")
        self.assertEqual(str(raised.exception), "accept: steps/003.out: no decode line")

    def test_missing_prefill_fails_only_when_needed(self):
        text = "\n".join(line for line in CAPTURED_ONESHOT.splitlines() if not line.startswith("prefill"))
        parsed = engine.parse_one_shot(text, source="x")
        self.assertIsNone(parsed.prefill)
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/004.out", need_prefill=True)
        self.assertEqual(str(raised.exception), "accept: steps/004.out: no prefill line")

    def test_repeated_line_is_an_error(self):
        text = CAPTURED_ONESHOT + "output  : 1 2\n"
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/005.out")
        self.assertEqual(str(raised.exception), "accept: steps/005.out: 2 output lines, want one")

    def test_chunk_comes_from_stderr_and_is_required_when_asked(self):
        stdout = "prompt  : 1 2 3\noutput  :\ndecode                   0 tokens in 0.0 ms  ->  0.00 tok/s\n"
        parsed = engine.parse_one_shot(stdout, source="steps/006.out",
                                       stderr="strata generate: prompt chunk auto: 4096 tokens\n",
                                       stderr_source="steps/006.err", need_chunk=True)
        self.assertEqual(parsed.chunk, 4096)
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(stdout, source="steps/006.out", stderr="", stderr_source="steps/006.err",
                                  need_chunk=True)
        self.assertEqual(str(raised.exception), "accept: steps/006.err: no chunk line")

    def test_empty_output_is_a_valid_empty_list(self):
        parsed = engine.parse_one_shot("prompt  :\noutput  :\ndecode  0 tokens in 0.0 ms  ->  0.00 tok/s\n",
                                       source="x")
        self.assertEqual(parsed.output, ())

    def test_version_and_devices(self):
        version = engine.parse_version("strata 1.2.3 metal 7\n", source="v")
        self.assertEqual((version.engine_version, version.metal_rev), ("1.2.3", 7))
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_version("hello\n", source="steps/000.out")
        self.assertEqual(str(raised.exception), "accept: steps/000.out: no version line")
        with self.assertRaises(engine.EngineOutputError):
            engine.parse_devices("\n  \n", source="devices")
        self.assertEqual(engine.parse_devices("0 Fake Apple GPU\n", source="d"), ("0 Fake Apple GPU",))

    def test_fake_engine_output_parses(self):
        completed = run_fake(["--tokens", "11,12", "--max-new", "4", "--greedy", "--spec", "2", "--prefill", "auto"])
        self.assertEqual(completed.returncode, 0, completed.stderr)
        parsed = engine.parse_one_shot(completed.stdout, source="fake", stderr=completed.stderr,
                                       stderr_source="fake.err", need_prefill=True, need_speculation=True,
                                       need_chunk=True)
        self.assertEqual(len(parsed.output), 4)
        self.assertEqual(parsed.chunk, 4096)
        self.assertEqual(parsed.speculation.window, 2)


class FakeEngine(unittest.TestCase):
    def test_version_and_devices_answer_like_the_engine(self):
        self.assertEqual(run_fake(["--version"]).stdout.strip(), "strata 0.0.0-fake metal 1")
        self.assertEqual(run_fake(["--list-devices"]).returncode, 0)

    def test_equal_arguments_give_equal_output(self):
        first = engine.parse_one_shot(run_fake(["--tokens", "1,2,3", "--max-new", "6", "--greedy"]).stdout, source="a")
        second = engine.parse_one_shot(run_fake(["--tokens", "1,2,3", "--max-new", "6", "--greedy", "--spec", "3"]).stdout,
                                       source="b")
        self.assertEqual(first.output, second.output)

    def test_output_plant_changes_one_token(self):
        base = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "4"]).stdout, source="a")
        planted = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "4"], "output:2").stdout,
                                        source="b")
        self.assertNotEqual(base.output[2], planted.output[2])
        self.assertEqual(base.output[:2], planted.output[:2])
        self.assertEqual(base.output[3:], planted.output[3:])

    def test_logits_plant_changes_one_dumped_value(self):
        with tempfile.TemporaryDirectory() as temp:
            plain = Path(temp, "plain.logits")
            planted = Path(temp, "planted.logits")
            run_fake(["--tokens", "5", "--max-new", "2", "--dump-logits", str(plain)])
            run_fake(["--tokens", "5", "--max-new", "2", "--dump-logits", str(planted)], "logits:1.3")
            a = plain.read_bytes()
            b = planted.read_bytes()
            self.assertEqual(a[:8], b[:8])
            self.assertEqual(len(a), 8 + 4 * 16 * 2)
            changed = [i for i in range(0, len(a) - 8, 4) if a[8 + i:12 + i] != b[8 + i:12 + i]]
            self.assertEqual(len(changed), 1)
            self.assertAlmostEqual(struct.unpack("<f", b[8 + changed[0]:12 + changed[0]])[0]
                                   - struct.unpack("<f", a[8 + changed[0]:12 + changed[0]])[0], 0.5)

    def test_spec_accept_and_timing_plants(self):
        base = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "8", "--spec", "4"]).stdout,
                                     source="a", need_speculation=True, need_prefill=True)
        low = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "8", "--spec", "4"], "spec:accept")
                                    .stdout, source="b", need_speculation=True)
        self.assertLess(low.speculation.accepted, base.speculation.accepted)
        slow = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "8"], "decode:slow").stdout,
                                     source="c")
        fast = engine.parse_one_shot(run_fake(["--tokens", "1,2", "--max-new", "8"]).stdout, source="d")
        self.assertAlmostEqual(slow.decode.tok_s * 4, fast.decode.tok_s, places=1)
        slow_prefill = engine.parse_one_shot(run_fake(["--tokens", "1,2,3", "--max-new", "1"], "prefill:slow").stdout,
                                             source="e", need_prefill=True)
        fast_prefill = engine.parse_one_shot(run_fake(["--tokens", "1,2,3", "--max-new", "1"]).stdout,
                                             source="f", need_prefill=True)
        self.assertAlmostEqual(slow_prefill.prefill.timing.tok_s * 4, fast_prefill.prefill.timing.tok_s, places=1)

    def test_chunk_plant_and_missing_plant(self):
        completed = run_fake(["--tokens", "1,2", "--max-new", "1", "--prefill", "auto"], "chunk:2048")
        self.assertIn("prompt chunk auto: 2048 tokens", completed.stderr)
        missing = run_fake(["--tokens", "1,2", "--max-new", "1"], "missing:decode")
        self.assertEqual(missing.returncode, 0)
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(missing.stdout, source="steps/007.out")
        self.assertEqual(str(raised.exception), "accept: steps/007.out: no decode line")

    def test_exit_plant(self):
        self.assertEqual(run_fake(["--tokens", "1", "--max-new", "1"], "exit:3").returncode, 3)


class StepRunner(unittest.TestCase):
    def test_timeout_stops_the_process_group(self):
        with tempfile.TemporaryDirectory() as temp:
            steps_dir = Path(temp)
            started = time.monotonic()
            step = stepper.run_step(
                [sys.executable, str(FAKE_ENGINE), "--tokens", "1"], number=1, name="hang", model="",
                prompt="", steps_dir=steps_dir, env={**os.environ, "FAKE_ENGINE_PLANT": "hang"},
                env_record={}, cwd=ROOT, limit_s=0.5, grace_s=0.5,
            )
            self.assertTrue(step.timed_out)
            self.assertNotEqual(step.exit, 0)
            self.assertLess(time.monotonic() - started, 30)

    def test_command_file_lists_arguments_and_strata_variables_only(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, "001.cmd")
            stepper.write_command(path, ["strata", "--greedy"], {"STRATA_METAL_WAIT": "spin", "FAKE": "x"})
            self.assertEqual(path.read_text(encoding="utf-8").splitlines(),
                             ["strata", "--greedy", "STRATA_METAL_WAIT=spin"])

    def test_new_crash_reports_are_copied(self):
        with tempfile.TemporaryDirectory() as temp:
            diagnostics = Path(temp, "DiagnosticReports")
            diagnostics.mkdir()
            (diagnostics / "old-strata.ips").write_text("old", encoding="utf-8")
            destination = Path(temp, "crash")
            with mock.patch.object(stepper, "DIAGNOSTIC_DIR", diagnostics):
                before = stepper.crash_snapshot()
                (diagnostics / "strata-2026.ips").write_text("new", encoding="utf-8")
                (diagnostics / "other-app.ips").write_text("unrelated", encoding="utf-8")
                copied = stepper.copy_new_crashes(before, destination)
            self.assertEqual(copied, ("strata-2026.ips",))
            self.assertTrue(Path(destination, "strata-2026.ips").is_file())
            self.assertFalse(Path(destination, "other-app.ips").exists())

    def test_step_limit_is_at_least_ten_minutes_and_three_times_expected(self):
        self.assertEqual(stepper.step_limit(None), 600.0)
        self.assertEqual(stepper.step_limit(100.0), 600.0)
        self.assertEqual(stepper.step_limit(400.0), 1200.0)


class PreRunChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write_prompts(self, ids: str, manifest_hash: str | None = None):
        prompts = self.root / "tests/metal/prompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (prompts / "g1-01.ids").write_text(ids, encoding="ascii")
        digest = manifest_hash or checks.ids_sha256(checks.read_ids(prompts / "g1-01.ids"))
        (prompts / "MANIFEST.json").write_text(json.dumps({
            "schema": 1, "builder_version": "test-builder",
            "sets": [{"name": "g1-01", "ids_sha256": digest}],
        }), encoding="utf-8")

    def test_prompt_manifest_absent_is_unmet_not_passed(self):
        check = checks.prompt_manifest(self.root, ("g1-01",))
        self.assertFalse(check.ok)
        self.assertEqual(check.detail, "tests/metal/prompts/MANIFEST.json is absent (T1)")

    def test_prompt_manifest_matches_the_canonical_hash(self):
        self.write_prompts("11,12\n13\n")
        self.assertTrue(checks.prompt_manifest(self.root, ("g1-01",)).ok)

    def test_prompt_manifest_fails_on_a_changed_id(self):
        self.write_prompts("11,12,13", manifest_hash=sha256_text("11,12,99"))
        check = checks.prompt_manifest(self.root, ("g1-01",))
        self.assertFalse(check.ok)
        self.assertIn("g1-01: ids_sha256", check.detail)

    def test_golden_manifest_absent_is_unmet(self):
        check = checks.golden_manifest(self.root, self.root / "golden", ("tf-code.logits",))
        self.assertEqual(check.detail, "tests/metal/golden/MANIFEST.json is absent (T2)")

    def test_golden_file_with_wrong_hash_names_the_file(self):
        (self.root / "UPSTREAM_SYNC").write_text(UPSTREAM + "\n", encoding="ascii")
        golden = self.root / "golden"
        golden.mkdir()
        (golden / "tf-code.logits").write_bytes(b"planted")
        manifest = self.root / "tests/metal/golden/MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"upstream_sync": UPSTREAM, "files": [
            {"path": "tf-code.logits", "sha256": sha256_text("original")}]}), encoding="utf-8")
        check = checks.golden_manifest(self.root, golden, ("tf-code.logits",))
        self.assertFalse(check.ok)
        expected = f"golden: tf-code.logits: sha256 {hashlib.sha256(b'planted').hexdigest()} does not match the manifest"
        self.assertEqual(check.detail, expected)

    def test_golden_manifest_with_another_sync_is_refused(self):
        (self.root / "UPSTREAM_SYNC").write_text(UPSTREAM + "\n", encoding="ascii")
        manifest = self.root / "tests/metal/golden/MANIFEST.json"
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({"upstream_sync": "0" * 40, "files": []}), encoding="utf-8")
        check = checks.golden_manifest(self.root, self.root, ())
        self.assertEqual(check.detail, "the manifest's upstream_sync is not UPSTREAM_SYNC")

    def test_test_models_script_absent_is_unmet(self):
        check = checks.test_models_quick(self.root, self.root, "Q2_0")
        self.assertFalse(check.ok)
        self.assertIn("tools/metal/test_models.py is absent (T3)", check.detail)

    def test_engine_start_on_the_fake_engine(self):
        found, version, devices = checks.engine_start(FAKE_ENGINE)
        self.assertTrue(all(check.ok for check in found), [check.detail for check in found])
        self.assertEqual(version.engine_version, "0.0.0-fake")
        self.assertEqual(devices, ("0 Fake Apple GPU",))

    def test_engine_absent_is_unmet(self):
        found, version, _ = checks.engine_start(self.root / "build-dev/strata")
        self.assertFalse(found[0].ok)
        self.assertIsNone(version)

    def test_pgrep_decides_the_strata_process_check(self):
        for code, ok in ((0, False), (1, True)):
            completed = subprocess.CompletedProcess(args=[], returncode=code, stdout="123\n", stderr="")
            with mock.patch.object(checks.subprocess, "run", return_value=completed):
                self.assertIs(checks.no_engine_running().ok, ok)

    def test_model_config_needs_an_args_list(self):
        models = self.root / "models"
        (models / "Q2_0").mkdir(parents=True)
        (models / "Q2_0/config.json").write_text(json.dumps({"args": ["--max-context", "8192"]}), encoding="utf-8")
        check = checks.model_config_args(models, "Q2_0")
        self.assertTrue(check.ok)
        self.assertEqual(check.value, ("--max-context", "8192"))
        (models / "Q2_0/config.json").write_text(json.dumps({"args": "--max-context"}), encoding="utf-8")
        self.assertFalse(checks.model_config_args(models, "Q2_0").ok)


class PlanRules(unittest.TestCase):
    def test_with_flags_replaces_named_flags_and_keeps_others(self):
        base = ["--spec", "4", "--greedy", "--max-context", "8192"]
        self.assertEqual(runs.with_flags(base, ["--spec", "0"]), ["--greedy", "--max-context", "8192", "--spec", "0"])

    def test_plans_with_placeholder_args_name_their_unresolved_inputs(self):
        context = runs.Context(models=("Q2_0",), args={"Q2_0": ("<config.json args of Q2_0>",)})
        for name in ("c1", "c2", "c3", "c4", "pp", "needle", "server", "bench", "pf", "g1", "g2", "g3", "release"):
            spec = runs.RUNS[name]
            steps = spec.plan(context)
            self.assertTrue(steps, name)
            self.assertTrue(any(step.unresolved for step in steps) or spec.name == "g3", name)

    def test_c2_arm_d_names_its_oracle_as_an_unresolved_input(self):
        context = runs.Context(models=("Q2_0",), args={"Q2_0": ("--max-context", "8192")}, prompts=("g1-01",))
        arm_d = [step for step in runs.plan_c2(context) if step.name == "c2.d.g1-01"][0]
        self.assertEqual(arm_d.unresolved, ("<output of arm a for this prompt>",))

    def test_c1_short_form_lists_only_tf_code(self):
        context = runs.Context(models=("Q2_0",), args={"Q2_0": ("--max-context", "8192")}, short=True)
        names = [step.name for step in runs.plan_c1(context)]
        self.assertIn("c1.tf.tf-code", names)
        self.assertFalse(any(name.startswith("c1.tf.tf-doc") for name in names))
        self.assertFalse(any(name.startswith("c1.layers") for name in names))

    def test_registry_matches_section_3_2_run_names(self):
        self.assertEqual(
            set(runs.RUNS),
            {"c1", "c2", "c3", "c4", "pp", "needle", "server", "bench", "pf", "g1", "g2", "g3",
             "nightly", "weekly", "release", "family"},
        )


def load_registry(plan, measure, criteria_ids):
    return {"c2": runs.RunSpec(name="c2", criteria=criteria_ids, models=("Q2_0",), plan=plan, task="test",
                               measure=measure)}


SAMPLED = ["--seed", "42", "--temperature", "0.7"]
ARMS = (
    ("a", None, []),
    ("b", "C2.b", ["--greedy"]),
    ("c", "C2.c", ["--greedy", "--spec", "8"]),
    ("f", "C2.f", ["--greedy", "--spec-corrupt", "2"]),
    ("gref", None, [*SAMPLED, "--spec", "0"]),
    ("g", "C2.g", SAMPLED),
)
# The reference arm of each criterion: C2.g is judged against the sampled --spec 0 arm.
REFERENCE = {"C2.g": "gref"}
ARM_CRITERIA = ("C2.b", "C2.c", "C2.f", "C2.g")


def make_plan(plants: dict[str, str], ids_path: Path):
    def plan(context):
        del context
        steps = []
        for arm, criterion, extra in ARMS:
            env = (("FAKE_ENGINE_PLANT", plants[arm]),) if arm in plants else ()
            steps.append(runs.Step(
                name=f"c2.{arm}.g1-01", model="Q2_0", prompt="g1-01", env=env,
                argv=tuple(["--tokens-file", str(ids_path), "--max-new", "4", *extra]),
                criteria=(criterion,) if criterion else (),
            ))
        return steps

    return plan


def make_measure():
    def measure(outputs):
        def parsed(arm):
            output = outputs.outputs[f"c2.{arm}.g1-01"]
            return engine.parse_one_shot(output.stdout, source=output.stdout_file)

        verdicts = []
        for arm, criterion, _ in ARMS:
            if criterion is None:
                continue
            reference = parsed(REFERENCE.get(criterion, "a"))
            verdicts.append(criteria.judge(criterion, int(parsed(arm).output != reference.output)))
        return verdicts

    return measure


class RunnerEndToEnd(unittest.TestCase):
    """The runner with the fake engine and a test-only registry (no production run has a measurement yet)."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name, "root")
        self.out = Path(self.temp.name, "out")
        self.models = Path(self.temp.name, "models")
        self.root.mkdir()
        prompts = self.root / "tests/metal/prompts"
        prompts.mkdir(parents=True)
        (prompts / "g1-01.ids").write_text("11,12,13\n", encoding="ascii")
        ids = checks.ids_sha256(checks.read_ids(prompts / "g1-01.ids"))
        (prompts / "MANIFEST.json").write_text(json.dumps({"schema": 1, "builder_version": "t",
                                                           "sets": [{"name": "g1-01", "ids_sha256": ids}]}),
                                               encoding="utf-8")
        (self.root / "UPSTREAM_SYNC").write_text(UPSTREAM + "\n", encoding="ascii")
        tools = self.root / "tools/metal"
        tools.mkdir(parents=True)
        (tools / "test_models.py").write_text(
            "import sys\nprint('quick sha256 ok ' + sys.argv[-1])\n", encoding="utf-8")
        (self.models / "Q2_0").mkdir(parents=True)
        (self.models / "Q2_0/config.json").write_text(json.dumps({"args": ["--max-context", "8192"]}),
                                                      encoding="utf-8")
        self.ids = prompts / "g1-01.ids"
        self.patch = mock.patch("accept.result.host_info", return_value={})
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.temp.cleanup()

    def run_registry(self, plants, argv_extra=(), plan_override=None):
        registry = load_registry(plan_override or make_plan(plants, self.ids), make_measure(), ARM_CRITERIA)
        argv = ["c2", "--models", str(self.models), "--engine", str(FAKE_ENGINE), "--out", str(self.out),
                *argv_extra]
        out = io.StringIO()
        err = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ACCEPT.run_cli(argv, registry, self.root, {})
        lines = out.getvalue().strip().splitlines()
        return code, (lines[-1] if lines else ""), lines

    def result_json(self):
        folders = sorted(path for path in self.out.iterdir() if path.is_dir())
        self.assertEqual(len(folders), 1)
        return folders[0], json.loads((folders[0] / "result.json").read_text(encoding="utf-8"))

    def test_equal_outputs_pass_and_write_schema_1(self):
        code, last, _ = self.run_registry({})
        self.assertEqual(code, 0)
        self.assertEqual(last, "accept c2: PASS (4 criteria, 0.0 min)")
        folder, data = self.result_json()
        self.assertEqual(data["schema"], 1)
        self.assertEqual(data["status"], "pass")
        self.assertEqual(data["engine"]["version"], "0.0.0-fake")
        self.assertEqual(data["engine"]["metal_rev"], 1)
        self.assertEqual(len(data["steps"]), 6)
        self.assertEqual(len(data["criteria"]), 4)
        self.assertTrue(all(row["pass"] is True for row in data["criteria"]))
        command = (folder / "steps/002.cmd").read_text(encoding="utf-8")
        self.assertIn("--greedy", command)
        self.assertIn("STRATA_METAL_ARCHIVE=", command)

    def test_planted_output_difference_fails_with_the_message(self):
        code, last, _ = self.run_registry({"f": "output:0"})
        self.assertEqual(code, 1)
        self.assertEqual(last, "accept c2: FAIL C2.f: 1, want == 0")
        _folder, data = self.result_json()
        self.assertEqual(data["status"], "fail")
        failed = [row for row in data["criteria"] if row["pass"] is False]
        self.assertEqual([row["id"] for row in failed], ["C2.f"])

    def test_planted_missing_line_is_not_run_with_the_parser_message(self):
        code, last, _ = self.run_registry({"c": "missing:decode"})
        self.assertEqual(code, 2)
        self.assertEqual(last, "accept c2: NOT RUN: accept: steps/003.out: no decode line")
        _folder, data = self.result_json()
        self.assertEqual(data["status"], "not-run")

    def test_planted_exit_stops_the_run_and_fails_step_exit(self):
        code, last, _ = self.run_registry({"b": "exit:3"})
        self.assertEqual(code, 1)
        self.assertEqual(last, "accept c2: FAIL STEP.exit: 3, want == 0")
        _folder, data = self.result_json()
        self.assertEqual(len(data["steps"]), 2)

    def test_dry_run_lists_every_command_and_loads_nothing(self):
        code, last, lines = self.run_registry({}, argv_extra=["--dry-run"])
        self.assertEqual(code, 0)
        self.assertEqual(last, "accept c2: DRY RUN: 6 commands, inputs checked, no model loaded")
        self.assertTrue(any("--seed 42" in line for line in lines))
        self.assertFalse(self.out.exists())

    def test_dry_run_without_models_is_not_run_with_the_reason(self):
        registry = load_registry(make_plan({}, self.ids), make_measure(), ARM_CRITERIA)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ACCEPT.run_cli(["c2", "--dry-run", "--engine", str(FAKE_ENGINE)], registry, self.root, {})
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue().strip().splitlines()[-1],
                         "accept c2: NOT RUN: models: STRATA_TEST_MODELS is not set and --models was not given")

    def test_restricted_run_is_marked_in_the_pass_line_and_result(self):
        code, last, _ = self.run_registry({}, argv_extra=["--model", "Q2_0"])
        self.assertEqual(code, 0)
        self.assertIn("restricted run", last)
        _folder, data = self.result_json()
        self.assertTrue(data["restricted"])

    def test_unmeasured_production_run_never_reports_pass(self):
        out, err = io.StringIO(), io.StringIO()
        argv = ["c1", "--models", str(self.models), "--engine", str(FAKE_ENGINE), "--out", str(self.out)]
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ACCEPT.run_cli(argv, runs.RUNS, self.root, {})
        self.assertEqual(code, 2)
        self.assertIn("NOT RUN", out.getvalue().strip().splitlines()[-1])
        self.assertFalse(self.out.exists(), "a run that cannot start writes no result folder")

    def test_production_c2_is_not_ready_because_arm_d_needs_arm_a(self):
        argv = ["c2", "--dry-run", "--models", str(self.models), "--engine", str(FAKE_ENGINE), "--prompt", "g1-01"]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = ACCEPT.run_cli(argv, runs.RUNS, self.root, {})
        self.assertEqual(code, 2)
        self.assertEqual(out.getvalue().strip().splitlines()[-1],
                         "accept c2: NOT RUN: c2.d.g1-01: input not resolvable yet: "
                         "<output of arm a for this prompt>")

    def test_cli_rejects_options_that_do_not_apply(self):
        parser = ACCEPT.build_parser(list(runs.RUNS))
        for argv in (["c2", "--short"], ["c2", "--model", "Coder"], ["weekly", "--day", "9"],
                     ["c2", "--prompt", "g1-01", "--day", "2"]):
            with self.subTest(argv=argv), contextlib.redirect_stderr(io.StringIO()):
                args = parser.parse_args(argv)
                with self.assertRaises(SystemExit) as raised:
                    ACCEPT.validate(parser, args, runs.RUNS[argv[0]])
                self.assertEqual(raised.exception.code, 2)

    def test_meta_runs_say_who_composes_them(self):
        out = io.StringIO()
        registry = {"nightly": runs.RUNS["nightly"]}
        with contextlib.redirect_stdout(out):
            code = ACCEPT.run_cli(["nightly", "--dry-run"], registry, self.root, {})
        self.assertEqual(code, 2)
        self.assertIn("composed by T14", out.getvalue().strip().splitlines()[-1])


class ReviewFindings(unittest.TestCase):
    """Cases from the adversarial review of the runner: each one would have hidden a failure or a bypass."""

    def test_decode_count_must_match_the_output_line(self):
        text = CAPTURED_ONESHOT.replace("decode                   2 tokens", "decode                   3 tokens")
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/008.out")
        self.assertEqual(str(raised.exception),
                         "accept: steps/008.out: decode line counts 3 tokens, output line has 2")

    def test_prompt_names_cannot_name_a_path(self):
        parser = ACCEPT.build_parser(list(runs.RUNS))
        for name in ("../secret", "g1-01/../../x", "/abs", "a b"):
            with self.subTest(name=name), contextlib.redirect_stderr(io.StringIO()):
                args = parser.parse_args(["c2", "--prompt", name])
                with self.assertRaises(SystemExit) as raised:
                    ACCEPT.validate(parser, args, runs.RUNS["c2"])
                self.assertEqual(raised.exception.code, 2)

    def test_a_measurement_that_cannot_be_judged_is_not_run_not_failed(self):
        def broken(outputs):
            raise KeyError("c2.b.g1-01")

        registry = load_registry(make_plan({}, Path("/nonexistent.ids")), broken, ARM_CRITERIA)
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        ids = Path(temp.name, "g1-01.ids")
        ids.write_text("1,2\n", encoding="ascii")
        registry = load_registry(make_plan({}, ids), broken, ARM_CRITERIA)
        root = Path(temp.name, "root")
        models = Path(temp.name, "models")
        (root / "tests/metal/prompts").mkdir(parents=True)
        (root / "tests/metal/prompts/g1-01.ids").write_text("1,2\n", encoding="ascii")
        (root / "tests/metal/prompts/MANIFEST.json").write_text(json.dumps({"schema": 1, "sets": [
            {"name": "g1-01", "ids_sha256": checks.ids_sha256(["1", "2"])}]}), encoding="utf-8")
        (root / "tools/metal").mkdir(parents=True)
        (root / "tools/metal/test_models.py").write_text("print('ok')\n", encoding="utf-8")
        (root / "UPSTREAM_SYNC").write_text(UPSTREAM + "\n", encoding="ascii")
        (models / "Q2_0").mkdir(parents=True)
        (models / "Q2_0/config.json").write_text(json.dumps({"args": []}), encoding="utf-8")
        out = Path(temp.name, "out")
        argv = ["c2", "--models", str(models), "--engine", str(FAKE_ENGINE), "--out", str(out)]
        buffer = io.StringIO()
        with mock.patch("accept.result.host_info", return_value={}), contextlib.redirect_stdout(buffer), \
                contextlib.redirect_stderr(io.StringIO()):
            code = ACCEPT.run_cli(argv, registry, root, {})
        self.assertEqual(code, 2)
        self.assertIn("accept c2: NOT RUN: measurement error:", buffer.getvalue().strip().splitlines()[-1])

    def test_host_fields_parse_best_effort(self):
        answers = {
            ("sw_vers", "-productVersion"): "27.0",
            ("sw_vers", "-buildVersion"): "27A266a",
            ("sysctl", "-n", "hw.memsize"): str(64 * (1 << 30)),
            ("sysctl", "-n", "machdep.cpu.brand_string"): "Apple M5 Pro",
            ("pmset", "-g", "batt"): "Now drawing from 'AC Power'",
            ("pmset", "-g"): " lowpowermode         0\n",
        }

        def fake_run(argv):
            if argv[0] == "system_profiler":
                return "Chipset Model: Apple M5 Pro\n      Total Number of Cores: 20\n"
            return answers.get(tuple(argv))

        with mock.patch.object(result, "run_text", side_effect=fake_run):
            host = result.host_info()
        self.assertEqual(host["macos"], {"product": "27.0", "build": "27A266a"})
        self.assertEqual(host["memory_gib"], 64)
        self.assertEqual(host["gpu_cores"], 20)
        self.assertEqual(host["power"], "AC Power")
        self.assertIs(host["low_power_mode"], False)

    def test_host_fields_are_null_when_the_command_fails(self):
        with mock.patch.object(result, "run_text", return_value=None):
            host = result.host_info()
        self.assertIsNone(host["macos"])
        self.assertIsNone(host["gpu_cores"])
        self.assertIsNone(host["power"])
        self.assertIsNone(host["low_power_mode"])
        self.assertIsNone(host["memory_gib"])


class DiscoveryAndDocs(unittest.TestCase):
    def test_runner_discovers_this_module(self):
        from tools.metal import run_python_tests

        self.assertIn("tools.metal.test_accept", run_python_tests.discover_modules())

    def test_accept_package_is_not_a_test_module(self):
        from tools.metal import run_python_tests

        modules = run_python_tests.discover_modules()
        self.assertFalse(any("accept.criteria" in module or "fake_engine" in module for module in modules))


if __name__ == "__main__":
    unittest.main(verbosity=2)
