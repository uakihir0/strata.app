"""Tests for the benchmark's host-verifiable logic (docs/mac/09-testing-and-validation.md, sections 5.1-5.4).

The one-shot text is synthetic, in the formats of section 3.1: the `decode` and `prefill` lines padded to 24 columns,
and the `strata generate: prompt chunk auto: N tokens` stderr line. No model runs here. Each criterion passes on equal
inputs and fails on its planted difference with the message of section 3.2. A missing value, a missing baseline or a
spread that the plan does not re-run is NOT RUN, never a pass. The pmset text is captured from a Mac.
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))

from accept import bench, checks, criteria, engine, result, runs  # noqa: E402

ACCEPT_CLI = HERE / "accept.py"
FAKE_ENGINE = HERE / "accept" / "fake_engine.py"
UPSTREAM = "1678de33d0e0711bc414ad992b640e1a37dd814"
SECTION_9_3 = "### 9.3 The Python layer"
MODELS_NOT_SET = "STRATA_TEST_MODELS is not set and --models was not given"


def load_cli():
    spec = importlib.util.spec_from_file_location("accept_cli_bench", ACCEPT_CLI)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ACCEPT = load_cli()


@dataclass(frozen=True)
class FakeOutput:
    """A step's output as the runner keeps it (runs.Output): the text and the names the result folder gives it."""

    stdout: str
    stderr: str
    stdout_file: str
    stderr_file: str


def one_shot(*, decode_tok_s: float = 50.0, prefill_tok_s: float = 450.0, ttft_ms: float = 6500.0,
             tokens: int = 256, prefill_tokens: int = 32000) -> str:
    """The one-shot stdout of section 3.1: the prompt and output lines, then the decode and prefill lines."""
    ids = "".join(f" {number}" for number in range(1, tokens + 1))
    decode_ms = tokens / decode_tok_s * 1000
    prefill_ms = prefill_tokens / prefill_tok_s * 1000
    return "\n".join([
        "prompt  : 11 12",
        f"output  :{ids}",
        f"{'decode':<24}{tokens} tokens in {decode_ms:.2f} ms  ->  {decode_tok_s:.2f} tok/s",
        f"{'prefill':<24}{prefill_tokens} tokens in {prefill_ms:.2f} ms  ->  {prefill_tok_s:.2f} tok/s  "
        f"(time to first token {ttft_ms:.2f} ms)",
        "",
    ])


def chunk_line(chunk: int) -> str:
    return f"strata generate: prompt chunk auto: {chunk} tokens\n"


def output(name: str, stdout: str, stderr: str = "") -> FakeOutput:
    return FakeOutput(stdout=stdout, stderr=stderr, stdout_file=f"steps/{name}.out",
                      stderr_file=f"steps/{name}.err")


def pf_outputs(prefill32=(450.0, 450.0, 450.0), prefill128=(315.0, 315.0, 315.0), chunk32: int = 4096,
               chunk128: int = 2048, ttft_ms: float = 6500.0) -> dict[str, FakeOutput]:
    """The measured runs of `pf` (run1 to run3) for ca-32k and ca-128k, as the runner names them."""
    outputs: dict[str, FakeOutput] = {}
    for number, (rate32, rate128) in enumerate(zip(prefill32, prefill128), start=1):
        name32 = f"pf.mtp.ca-32k.run{number}"
        name128 = f"pf.mtp.ca-128k.run{number}"
        outputs[name32] = output(name32, one_shot(prefill_tok_s=rate32, ttft_ms=ttft_ms), chunk_line(chunk32))
        outputs[name128] = output(name128, one_shot(prefill_tok_s=rate128, prefill_tokens=128000, ttft_ms=ttft_ms),
                                  chunk_line(chunk128))
    return outputs


def summary(median: float, *, noisy: bool = False) -> bench.Summary:
    return bench.Summary(median=median, spread=0.0, runs=6 if noisy else 3, rerun=False, noisy=noisy)


def measured(decode: float, prefill: float | None = None, *, noisy: bool = False) -> bench.Measured:
    return bench.Measured(decode=summary(decode, noisy=noisy),
                          prefill=summary(prefill if prefill is not None else 500.0, noisy=noisy),
                          time_to_first_token_ms=6500.0, chunk=4096)


def arms(overrides: dict | None = None) -> dict[tuple[str, str], bench.Measured]:
    """The mtp results for the three decode prompts and ca-32k prefill, and the plain arm on ca-4k."""
    result_map = {
        ("mtp", "ca-1k"): measured(100.0),
        ("mtp", "ca-4k"): measured(100.0),
        ("mtp", "ca-32k"): measured(100.0, prefill=500.0),
        ("plain", "ca-4k"): measured(30.0),
    }
    result_map.update(overrides or {})
    return result_map


def by_id(verdicts: list[criteria.Verdict]) -> dict[str, criteria.Verdict]:
    return {verdict.id: verdict for verdict in verdicts}


class SummarizeTests(unittest.TestCase):
    def test_three_runs_give_the_median(self):
        found = bench.summarize([48.0, 50.0, 52.0])
        self.assertEqual(found.median, 50.0)
        self.assertAlmostEqual(found.spread, 0.08)
        self.assertTrue(found.rerun, "a spread of 8% over three runs asks for three more")
        self.assertFalse(found.noisy)

    def test_three_runs_inside_five_percent_need_no_rerun(self):
        found = bench.summarize([100.0, 101.0, 102.0])
        self.assertFalse(found.rerun)
        self.assertFalse(found.noisy)

    def test_a_spread_of_exactly_five_percent_is_not_above_the_limit(self):
        found = bench.summarize([100.0, 100.0, 105.0])
        self.assertEqual(found.spread, 0.05)
        self.assertFalse(found.rerun)

    def test_six_runs_over_five_percent_are_noisy_and_not_rerun(self):
        found = bench.summarize([100.0, 100.0, 100.0, 106.0, 106.0, 106.0])
        self.assertEqual(found.median, 103.0)
        self.assertTrue(found.noisy)
        self.assertFalse(found.rerun)

    def test_six_runs_take_the_mean_of_the_middle_two(self):
        self.assertEqual(bench.summarize([10.0, 20.0, 30.0, 40.0, 50.0, 60.0]).median, 35.0)

    def test_other_run_counts_are_refused(self):
        for count in (1, 2, 4, 5, 7):
            with self.subTest(count=count), self.assertRaises(bench.BenchError):
                bench.summarize([50.0] * count)

    def test_rates_must_be_positive_and_finite(self):
        for values in ([0.0, 1.0, 2.0], [-1.0, 1.0, 2.0], [math.nan, 1.0, 2.0], [math.inf, 1.0, 2.0],
                       [True, 1.0, 2.0], ["50", 1.0, 2.0]):
            with self.subTest(values=values), self.assertRaises(bench.BenchError):
                bench.summarize(values)


class RunLineTests(unittest.TestCase):
    def test_decode_and_prefill_lines_become_tokens_and_rates(self):
        outputs = {"pf.mtp.ca-32k.run1": output("pf.mtp.ca-32k.run1", one_shot(), chunk_line(4096))}
        parsed = bench.parse_run(outputs, "pf.mtp.ca-32k.run1", chunk=True)
        self.assertEqual(parsed.decode.tokens, 256)
        self.assertEqual(parsed.decode.tok_s, 50.0)
        self.assertEqual(parsed.prefill.timing.tokens, 32000)
        self.assertEqual(parsed.prefill.timing.tok_s, 450.0)
        self.assertEqual(parsed.prefill.time_to_first_token_ms, 6500.0)
        self.assertEqual(parsed.chunk, 4096)

    def test_decode_count_must_match_the_output_line(self):
        text = one_shot().replace("256 tokens in", "255 tokens in", 1)
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/001.out", need_prefill=True)
        self.assertIn("decode line counts 255 tokens", str(raised.exception))

    def test_a_malformed_decode_line_is_refused(self):
        text = one_shot().replace("tok/s", "tok per s", 1)
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/001.out", need_prefill=True)
        self.assertEqual(str(raised.exception), "accept: steps/001.out: no decode line")

    def test_the_prefill_line_is_required_when_measured(self):
        text = "\n".join(line for line in one_shot().splitlines() if not line.startswith("prefill"))
        with self.assertRaises(engine.EngineOutputError) as raised:
            engine.parse_one_shot(text, source="steps/001.out", need_prefill=True)
        self.assertEqual(str(raised.exception), "accept: steps/001.out: no prefill line")

    def test_a_repeated_prefill_line_is_refused(self):
        text = one_shot() + one_shot().splitlines()[3] + "\n"
        with self.assertRaises(engine.EngineOutputError):
            engine.parse_one_shot(text, source="steps/001.out", need_prefill=True)

    def test_the_chunk_line_is_required_and_must_be_a_count(self):
        outputs = {"pf.mtp.ca-32k.run1": output("pf.mtp.ca-32k.run1", one_shot(),
                                                "strata generate: prompt chunk auto: four tokens\n")}
        with self.assertRaises(engine.EngineOutputError) as raised:
            bench.parse_run(outputs, "pf.mtp.ca-32k.run1", chunk=True)
        self.assertEqual(str(raised.exception), "accept: steps/pf.mtp.ca-32k.run1.err: no chunk line")

    def test_a_run_without_its_step_output_is_not_run(self):
        with self.assertRaises(bench.BenchError) as raised:
            bench.parse_run({}, "pf.mtp.ca-32k.run2", chunk=True)
        self.assertEqual(str(raised.exception), "bench: no output for step pf.mtp.ca-32k.run2")


class PrefillTests(unittest.TestCase):
    def test_equal_runs_pass_both_at_their_limits(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(prefill32=(450.0,) * 3, prefill128=(315.0,) * 3)))
        self.assertIs(verdicts["PF-2"].passed, True)
        self.assertIs(verdicts["PF-1"].passed, True, "315 over 450 is exactly 0.7")
        self.assertFalse(verdicts["PF-1"].blocks)

    def test_planted_pf2_difference_fails_with_the_message(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(prefill32=(449.0,) * 3)))
        self.assertIs(verdicts["PF-2"].passed, False)
        self.assertEqual(verdicts["PF-2"].message, "PF-2: 449, want >= 450")
        self.assertTrue(verdicts["PF-2"].blocks)

    def test_planted_pf1_difference_fails_with_the_message(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(prefill32=(450.0,) * 3, prefill128=(314.0,) * 3)))
        self.assertIs(verdicts["PF-1"].passed, False)
        self.assertEqual(verdicts["PF-1"].message, "PF-1: 0.697778, want >= 0.7")
        self.assertTrue(verdicts["PF-1"].blocks)

    def test_the_judged_rate_is_the_median_of_three_runs(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(prefill32=(440.0, 460.0, 450.0), prefill128=(315.0,) * 3)))
        self.assertEqual(verdicts["PF-2"].value, 450.0)
        self.assertIs(verdicts["PF-2"].passed, True)

    def test_a_first_run_below_the_limit_does_not_decide_pf2(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(prefill32=(440.0, 455.0, 452.0), prefill128=(350.0,) * 3)))
        self.assertEqual(verdicts["PF-2"].value, 452.0)
        self.assertIs(verdicts["PF-2"].passed, True)

    def test_three_runs_with_a_spread_over_five_percent_are_not_run(self):
        with self.assertRaises(bench.BenchError) as raised:
            bench.measure_pf(pf_outputs(prefill32=(400.0, 450.0, 500.0)))
        self.assertIn("spread", str(raised.exception))
        self.assertIn("section 5.1 step 3", str(raised.exception))

    def test_the_chunk_must_be_the_same_in_every_run(self):
        outputs = pf_outputs()
        outputs["pf.mtp.ca-32k.run2"] = output("pf.mtp.ca-32k.run2", one_shot(prefill_tok_s=450.0),
                                               chunk_line(2048))
        with self.assertRaises(bench.BenchError) as raised:
            bench.measure_pf(outputs)
        self.assertIn("chunk size differs", str(raised.exception))

    def test_a_pf_run_without_its_prefill_line_is_refused(self):
        text = "\n".join(line for line in one_shot().splitlines() if not line.startswith("prefill"))
        outputs = pf_outputs()
        for number in (1, 2, 3):
            name = f"pf.mtp.ca-32k.run{number}"
            outputs[name] = output(name, text, chunk_line(4096))
        with self.assertRaises(engine.EngineOutputError) as raised:
            bench.measure_pf(outputs)
        self.assertEqual(str(raised.exception), "accept: steps/pf.mtp.ca-32k.run1.out: no prefill line")

    def test_a_missing_ca_128k_run_is_not_run(self):
        outputs = pf_outputs()
        del outputs["pf.mtp.ca-128k.run2"]
        with self.assertRaises(bench.BenchError) as raised:
            bench.measure_pf(outputs)
        self.assertEqual(str(raised.exception), "bench: no output for step pf.mtp.ca-128k.run2")

    def test_the_note_records_the_chunks_and_the_time_to_first_token(self):
        verdicts = by_id(bench.measure_pf(pf_outputs(ttft_ms=6500.0)))
        self.assertIn("time to first token 6500.0 ms", verdicts["PF-2"].note)
        self.assertIn("chunk 4096", verdicts["PF-2"].note)
        self.assertIn("chunks 2048 and 4096", verdicts["PF-1"].note)


class RegressionBarTests(unittest.TestCase):
    def test_the_bar_is_0_95_while_the_coefficient_is_at_most_two_percent(self):
        for cv in (None, 0.0, 0.01, 0.02):
            with self.subTest(cv=cv):
                self.assertEqual(bench.regression_bar(cv), 0.95)

    def test_the_bar_is_one_minus_two_and_a_half_times_the_coefficient_above_two_percent(self):
        self.assertAlmostEqual(bench.regression_bar(0.03), 0.925)
        self.assertAlmostEqual(bench.regression_bar(0.04), 0.9)

    def test_a_coefficient_that_is_not_a_finite_non_negative_number_is_refused(self):
        for cv in (-0.01, math.nan, math.inf, "2%", True):
            with self.subTest(cv=cv), self.assertRaises(bench.BenchError):
                bench.regression_bar(cv)


class RegressionTests(unittest.TestCase):
    def test_equal_inputs_pass_both_regression_checks(self):
        verdicts = by_id(bench.regression_verdicts(arms(), arms()))
        self.assertIs(verdicts["REG.decode"].passed, True)
        self.assertIs(verdicts["REG.prefill"].passed, True)
        self.assertEqual(verdicts["REG.decode"].value, 1.0)

    def test_a_ratio_exactly_at_the_bar_passes(self):
        current = arms({("mtp", "ca-4k"): measured(95.0)})
        verdicts = by_id(bench.regression_verdicts(current, arms()))
        self.assertIs(verdicts["REG.decode"].passed, True)
        self.assertEqual(verdicts["REG.decode"].value, 0.95)

    def test_planted_decode_drop_fails_with_the_message(self):
        current = arms({("mtp", "ca-4k"): measured(94.0)})
        verdicts = by_id(bench.regression_verdicts(current, arms()))
        self.assertIs(verdicts["REG.decode"].passed, False)
        self.assertEqual(verdicts["REG.decode"].message, "REG.decode: 0.94, want >= 0.95")
        self.assertTrue(verdicts["REG.decode"].blocks)

    def test_planted_prefill_drop_fails_with_the_message(self):
        current = arms({("mtp", "ca-32k"): measured(100.0, prefill=470.0)})
        verdicts = by_id(bench.regression_verdicts(current, arms()))
        self.assertIs(verdicts["REG.prefill"].passed, False)
        self.assertEqual(verdicts["REG.prefill"].message, "REG.prefill: 0.94, want >= 0.95")

    def test_a_coefficient_above_two_percent_lowers_the_bar_it_is_judged_against(self):
        current = arms({("mtp", "ca-4k"): measured(93.0)})
        self.assertIs(by_id(bench.regression_verdicts(current, arms(), cv=0.03))["REG.decode"].passed, True)
        self.assertIs(by_id(bench.regression_verdicts(current, arms()))["REG.decode"].passed, False)

    def test_a_missing_previous_release_is_skipped_never_passed(self):
        verdicts = by_id(bench.regression_verdicts(arms(), None))
        for cid in ("REG.decode", "REG.prefill"):
            with self.subTest(cid=cid):
                self.assertIs(verdicts[cid].passed, None)
                self.assertFalse(verdicts[cid].blocks)
                self.assertTrue(verdicts[cid].skipped)
                self.assertEqual(verdicts[cid].message, f"{cid}: NOT RUN (no previous release)")

    def test_a_previous_engine_that_refuses_a_flag_skips_with_its_last_line(self):
        stderr = "strata: loading\nunknown argument: --gpu-budget-mib\n"
        note = bench.previous_engine_skip(2, stderr)
        self.assertEqual(note, "previous release refused the arguments: unknown argument: --gpu-budget-mib")
        verdicts = by_id(bench.regression_verdicts(arms(), None, skip_note=note))
        self.assertEqual(verdicts["REG.decode"].note, note)
        self.assertFalse(verdicts["REG.decode"].blocks)

    def test_a_previous_engine_that_runs_gives_no_skip(self):
        self.assertEqual(bench.previous_engine_skip(0, "ok\n"), "")

    def test_a_previous_engine_failing_for_another_reason_is_not_run(self):
        with self.assertRaises(bench.BenchError) as raised:
            bench.previous_engine_skip(1, "segmentation fault\n")
        self.assertIn("exited 1", str(raised.exception))

    def test_a_noisy_point_is_noted_and_cannot_fail_the_check(self):
        current = arms({("mtp", "ca-4k"): measured(50.0, noisy=True)})
        verdicts = by_id(bench.regression_verdicts(current, arms()))
        self.assertIs(verdicts["REG.decode"].passed, True, "the other two points are at their bar")
        self.assertIn("noisy, not judged: decode ca-4k", verdicts["REG.decode"].note)

    def test_when_every_point_is_noisy_the_check_is_skipped(self):
        current = {key: measured(value.decode.median, noisy=True) for key, value in arms().items()}
        verdicts = by_id(bench.regression_verdicts(current, arms()))
        self.assertTrue(verdicts["REG.decode"].skipped)
        self.assertFalse(verdicts["REG.decode"].blocks)

    def test_a_missing_point_is_not_run(self):
        current = arms()
        del current[("mtp", "ca-1k")]
        with self.assertRaises(bench.BenchError) as raised:
            bench.regression_verdicts(current, arms())
        self.assertEqual(str(raised.exception), "bench: no mtp ca-1k result for the current")


class TrendTests(unittest.TestCase):
    def test_without_a_previous_night_the_trend_is_skipped(self):
        verdict = bench.trend_verdict(arms(), None)
        self.assertTrue(verdict.skipped)
        self.assertIsNone(verdict.passed)
        self.assertFalse(verdict.blocks)

    def test_equal_nights_report_no_change_and_print_nothing(self):
        verdict = bench.trend_verdict(arms(), arms())
        self.assertEqual(verdict.value, 0.0)
        self.assertIn("within 3%, not printed", verdict.note)
        self.assertIsNone(verdict.passed)
        self.assertFalse(verdict.blocks)

    def test_a_change_above_three_percent_is_printed(self):
        verdict = bench.trend_verdict(arms({("mtp", "ca-4k"): measured(104.0)}), arms())
        self.assertAlmostEqual(verdict.value, 0.04)
        self.assertIn("printed: more than 3%", verdict.note)
        self.assertFalse(verdict.blocks, "a trend is reported, never gating")

    def test_the_change_of_the_point_that_moved_most_is_the_value(self):
        verdict = bench.trend_verdict(arms({("mtp", "ca-32k"): measured(100.0, prefill=470.0)}), arms())
        self.assertAlmostEqual(verdict.value, -0.06)
        self.assertIn("prefill ca-32k", verdict.note)


class BenchVerdictTests(unittest.TestCase):
    def headline(self, *, mtp: float = 60.0, plain: float = 20.0):
        return {("mtp", "ca-4k"): measured(mtp), ("plain", "ca-4k"): measured(plain)}

    def test_equal_values_pass_the_headline_rows(self):
        verdicts = by_id(bench.bench_verdicts(self.headline()))
        self.assertIs(verdicts["BENCH.mtp"].passed, True)
        self.assertIs(verdicts["BENCH.mtp.target"].passed, True)
        self.assertIs(verdicts["BENCH.plain"].passed, True)

    def test_planted_mtp_difference_fails_with_the_message(self):
        verdicts = by_id(bench.bench_verdicts(self.headline(mtp=49.9)))
        self.assertIs(verdicts["BENCH.mtp"].passed, False)
        self.assertEqual(verdicts["BENCH.mtp"].message, "BENCH.mtp: 49.9, want >= 50")
        self.assertTrue(verdicts["BENCH.mtp"].blocks)

    def test_planted_plain_difference_fails_with_the_message(self):
        verdicts = by_id(bench.bench_verdicts(self.headline(plain=19.9)))
        self.assertEqual(verdicts["BENCH.plain"].message, "BENCH.plain: 19.9, want >= 20")
        self.assertTrue(verdicts["BENCH.plain"].blocks)

    def test_missing_the_target_is_reported_and_does_not_block(self):
        verdicts = by_id(bench.bench_verdicts(self.headline(mtp=55.0)))
        self.assertIs(verdicts["BENCH.mtp"].passed, True)
        self.assertIs(verdicts["BENCH.mtp.target"].passed, False)
        self.assertFalse(verdicts["BENCH.mtp.target"].blocks)

    def test_other_arms_and_prompts_are_reported_and_never_block(self):
        current = self.headline()
        current[("mtp", "ca-32k")] = measured(45.0)
        verdicts = [verdict for verdict in bench.bench_verdicts(current) if verdict.id == "BENCH.other"]
        self.assertEqual(len(verdicts), 1)
        self.assertIsNone(verdicts[0].passed)
        self.assertFalse(verdicts[0].blocks)

    def test_a_missing_headline_value_is_not_run(self):
        with self.assertRaises(bench.BenchError):
            bench.bench_verdicts({("mtp", "ca-4k"): measured(60.0)})


class HostCheckTests(unittest.TestCase):
    CAPTURED_BATT = (
        "Now drawing from 'AC Power'\n"
        " -InternalBattery-0 (id=23199843)\t90%; AC attached; not charging present: true\n"
    )
    CAPTURED_THERM = (
        "Note: No thermal warning level has been recorded\n"
        "Note: No performance warning level has been recorded\n"
        "Note: No CPU power status has been recorded\n"
    )

    def test_power_must_be_ac(self):
        self.assertTrue(bench.power_check(self.CAPTURED_BATT).ok)
        battery = bench.power_check("Now drawing from 'Battery Power'\n")
        self.assertFalse(battery.ok)
        self.assertEqual(battery.detail, "drawing from 'Battery Power', want AC Power")
        self.assertFalse(bench.power_check(None).ok)

    def test_low_power_mode_must_be_zero(self):
        self.assertTrue(bench.low_power_check("System-wide power settings:\n powermode 0\n").ok)
        self.assertTrue(bench.low_power_check(" lowpowermode         0\n").ok)
        self.assertFalse(bench.low_power_check(" powermode 1\n").ok)
        self.assertFalse(bench.low_power_check("System-wide power settings:\n").ok, "absent is unreadable")

    def test_thermal_state_passes_only_with_both_no_warning_lines(self):
        self.assertTrue(bench.thermal_check(self.CAPTURED_THERM).ok)
        warned = self.CAPTURED_THERM.replace(
            "Note: No thermal warning level has been recorded", "Note: Thermal warning level has been recorded: 1")
        self.assertFalse(bench.thermal_check(warned).ok)
        self.assertEqual(bench.thermal_check(warned).detail, "a thermal or performance warning level is recorded")
        self.assertFalse(bench.thermal_check("Note: No thermal warning level has been recorded\n").ok,
                         "a missing performance line is unreadable, not a pass")
        self.assertFalse(bench.thermal_check(None).ok)

    def test_memory_level_must_be_at_least_fifty(self):
        self.assertTrue(bench.memory_check("75\n").ok)
        self.assertTrue(bench.memory_check("50\n").ok)
        self.assertFalse(bench.memory_check("49\n").ok)
        self.assertEqual(bench.memory_check("49\n").detail, "kern.memorystatus_level 49, want at least 50")
        self.assertEqual(bench.memory_check("n/a").detail, "kern.memorystatus_level is not a number")
        self.assertFalse(bench.memory_check(None).ok)

    def test_host_checks_read_the_four_commands(self):
        answers = {
            ("pmset", "-g", "batt"): self.CAPTURED_BATT,
            ("pmset", "-g"): " powermode 0\n",
            ("pmset", "-g", "therm"): self.CAPTURED_THERM,
            ("sysctl", "-n", "kern.memorystatus_level"): "75\n",
        }
        asked: list[tuple[str, ...]] = []

        def fake(argv):
            asked.append(tuple(argv))
            return answers.get(tuple(argv))

        found = bench.host_checks(fake)
        self.assertEqual(asked, list(answers))
        self.assertEqual([check.name for check in found], [
            "host: power source", "host: low power mode", "host: thermal state", "host: memory status",
        ])
        self.assertTrue(all(check.ok for check in found))

    def test_one_failing_command_stops_only_its_check(self):
        def fake(argv):
            if argv[-1] == "kern.memorystatus_level":
                return None
            if "batt" in argv:
                return self.CAPTURED_BATT
            if "therm" in argv:
                return self.CAPTURED_THERM
            return " powermode 0\n"

        found = bench.host_checks(fake)
        self.assertEqual([check.ok for check in found], [True, True, True, False])

    def test_host_info_records_the_thermal_state_and_memory_level(self):
        answers = {
            ("pmset", "-g", "therm"): self.CAPTURED_THERM,
            ("sysctl", "-n", "kern.memorystatus_level"): "75\n",
        }
        with mock.patch.object(result, "run_text", side_effect=lambda argv: answers.get(tuple(argv))):
            host = result.host_info()
        self.assertIs(host["thermal_warning"], False)
        self.assertEqual(host["memory_status_level"], 75)

    def test_host_info_is_null_when_the_commands_fail(self):
        with mock.patch.object(result, "run_text", return_value=None):
            host = result.host_info()
        self.assertIsNone(host["thermal_warning"])
        self.assertIsNone(host["memory_status_level"])


class SkipVerdictTests(unittest.TestCase):
    def test_a_skip_is_not_a_pass_and_does_not_block(self):
        verdict = criteria.skip("REG.decode", "no previous release")
        self.assertIsNone(verdict.passed)
        self.assertTrue(verdict.skipped)
        self.assertFalse(verdict.blocks)
        self.assertEqual(verdict.message, "REG.decode: NOT RUN (no previous release)")

    def test_a_gating_criterion_that_fails_still_blocks(self):
        self.assertTrue(criteria.judge("PF-2", 449.0, 450.0).blocks)
        self.assertFalse(criteria.judge("PF-2", 450.0, 450.0).skipped)

    def test_a_missing_value_still_never_passes(self):
        verdict = criteria.judge("PF-2", None, 450.0)
        self.assertIs(verdict.passed, False)
        self.assertTrue(verdict.blocks)

    def test_an_unknown_skip_is_an_error(self):
        with self.assertRaises(ValueError):
            criteria.skip("REG.nope", "no previous release")


def write_root(root: Path, *, sets=("ca-32k", "ca-128k", "g1-01")) -> None:
    prompts = root / "tests/metal/prompts"
    prompts.mkdir(parents=True)
    entries = []
    for name in sets:
        (prompts / f"{name}.ids").write_text("11,12,13\n", encoding="ascii")
        entries.append({"name": name, "ids_sha256": checks.ids_sha256(checks.read_ids(prompts / f"{name}.ids"))})
    (prompts / "MANIFEST.json").write_text(json.dumps({"schema": 1, "builder_version": "t", "sets": entries}),
                                           encoding="utf-8")
    (root / "UPSTREAM_SYNC").write_text(UPSTREAM + "\n", encoding="ascii")
    (root / "tools/metal").mkdir(parents=True)
    (root / "tools/metal/test_models.py").write_text("print('quick sha256 ok')\n", encoding="utf-8")


class RunnerRules(unittest.TestCase):
    """The runner's rules for the benchmark runs: a missing model or a failed host check is NOT RUN, before any step."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name, "root")
        self.models = Path(self.temp.name, "models")
        self.out = Path(self.temp.name, "out")
        write_root(self.root)
        (self.models / "Q2_0").mkdir(parents=True)
        (self.models / "Q2_0/config.json").write_text(json.dumps({"args": ["--max-context", "8192"]}),
                                                      encoding="utf-8")
        host = mock.patch("accept.result.host_info", return_value={})
        host.start()
        self.addCleanup(host.stop)

    def run_cli(self, argv, registry=None):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = ACCEPT.run_cli(argv, registry or runs.RUNS, self.root, {})
        return code, out.getvalue().strip().splitlines()

    def test_bench_without_models_is_not_run_with_the_reason(self):
        with mock.patch.object(bench, "host_checks", return_value=[]):
            code, lines = self.run_cli(["bench", "--dry-run", "--engine", str(FAKE_ENGINE)])
        self.assertEqual(code, 2)
        self.assertEqual(lines[-1], f"accept bench: NOT RUN: models: {MODELS_NOT_SET}")

    def test_pf_without_models_is_not_run_with_the_reason(self):
        with mock.patch.object(bench, "host_checks", return_value=[]):
            code, lines = self.run_cli(["pf", "--dry-run", "--engine", str(FAKE_ENGINE)])
        self.assertEqual(code, 2)
        self.assertEqual(lines[-1], f"accept pf: NOT RUN: models: {MODELS_NOT_SET}")

    def test_battery_stops_pf_with_exit_two_before_any_step(self):
        battery = checks.Check("host: power source", False, "drawing from 'Battery Power', want AC Power")
        with mock.patch.object(bench, "host_checks", return_value=[battery]):
            code, lines = self.run_cli(["pf", "--models", str(self.models), "--engine", str(FAKE_ENGINE),
                                        "--out", str(self.out)])
        self.assertEqual(code, 2)
        self.assertEqual(lines[-1],
                         "accept pf: NOT RUN: host: power source: drawing from 'Battery Power', want AC Power")
        self.assertFalse(self.out.exists(), "a run that cannot start writes no result folder")

    def test_pf_is_measured_and_bench_is_not_yet(self):
        context = runs.Context(models=("Q2_0",), args={"Q2_0": ("--max-context", "8192")})
        ok = [checks.Check("host: power source", True, "AC Power")]
        pf_plan = runs.RUNS["pf"].plan(context)
        self.assertIsNone(ACCEPT.first_reason(runs.RUNS["pf"], pf_plan, ok))
        bench_plan = runs.RUNS["bench"].plan(context)
        reason = ACCEPT.first_reason(runs.RUNS["bench"], bench_plan, ok)
        self.assertEqual(reason, "criteria not measured yet: BENCH.mtp, BENCH.mtp.target, BENCH.plain, BENCH.other, "
                                 "REG.decode, REG.prefill, REG.trend (task T11)")

    def test_a_skipped_gating_criterion_makes_the_run_not_run_never_pass(self):
        plan = [runs.Step(name="probe", kind="tool", argv=("{python}", "-c", "pass"))]
        registry = {"bench": runs.RunSpec(
            name="bench", criteria=("BENCH.mtp", "REG.decode"), plan=lambda context: plan,
            measure=lambda outputs: [criteria.judge("BENCH.mtp", 60.0),
                                     criteria.skip("REG.decode", "no previous release")],
            task="test")}
        code, lines = self.run_cli(["bench", "--models", str(self.models), "--engine", str(FAKE_ENGINE),
                                    "--out", str(self.out)], registry)
        self.assertEqual(code, 2)
        self.assertEqual(lines[-1], "accept bench: NOT RUN: skipped (section 5.4): REG.decode", lines[-1])
        folder = next(path for path in self.out.iterdir() if path.is_dir())
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        self.assertEqual(data["status"], "not-run")
        self.assertEqual({row["id"] for row in data["criteria"]}, {"BENCH.mtp", "REG.decode"})
        folder = next(path for path in self.out.iterdir() if path.is_dir())
        data = json.loads((folder / "result.json").read_text(encoding="utf-8"))
        rows = {row["id"]: row for row in data["criteria"]}
        self.assertIsNone(rows["REG.decode"]["pass"])
        self.assertEqual(rows["REG.decode"]["note"], "no previous release")
        self.assertIs(rows["BENCH.mtp"]["pass"], True)


class DiscoveryAndDocs(unittest.TestCase):
    def test_runner_discovers_this_module(self):
        from tools.metal import run_python_tests

        self.assertIn("tools.metal.test_bench", run_python_tests.discover_modules())

    def test_section_9_3_lists_this_module(self):
        text = (ROOT / "docs/mac/09-testing-and-validation.md").read_text(encoding="utf-8")
        section = text[text.index(SECTION_9_3): text.index("### 9.4 ")]
        self.assertIn("`tools/metal/test_bench.py`", section)


if __name__ == "__main__":
    unittest.main(verbosity=2)
