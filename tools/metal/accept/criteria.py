"""Criterion IDs, operators, thresholds and gating flags (docs/mac/09-testing-and-validation.md, sections 3.2 and 4-8).

One row per ID. The runner judges a measured value with `judge()`, and a verdict is blocking only when its row is
gating and it neither passed nor was skipped (`skip()`, for the skips section 5.4 allows). Rows whose threshold is
derived from a reference (basis `reference`, and T-1 to T-9) have `threshold=None`: the caller must supply the derived
number, and `judge()` refuses to pass a value without one.
A missing value never passes. Rows with the operator "report" are printed and never block a run.

`basis` says where a row's value can come from (09 section 4.1, D-IMPL-960):
- `kernel`: passable on the Mac from fixtures and oracles, with no model;
- `model`: needs the test models on the reference Mac, and no reference other than the Mac's own;
- `reference`: compares whole-model outputs with a model-level reference that the specification does not name. `judge()`
  never passes such a row: it returns NOT RUN, whatever the value, and a gating one makes its run NOT RUN (exit 2);
- `retired`: no longer a criterion. The row stays so that the specification's table and this file list the same IDs.
  `judge()` gives it no verdict;
- `runner`: the runner's own failure rows (STEP.*).
"""
from __future__ import annotations

import operator
from dataclasses import dataclass

REPORT = "report"
OPS = {
    ">=": operator.ge,
    "<=": operator.le,
    "==": operator.eq,
    ">": operator.gt,
    "<": operator.lt,
}

KERNEL = "kernel"
MODEL = "model"
REFERENCE = "reference"
RETIRED = "retired"
RUNNER = "runner"
BASES = (KERNEL, MODEL, REFERENCE, RETIRED, RUNNER)
RETIRED_OP = "retired"
REFERENCE_NOTE = "needs a model-level reference, which the specification does not name (D-IMPL-671, D-IMPL-960)"


@dataclass(frozen=True)
class Criterion:
    id: str
    op: str
    threshold: object
    gating: bool
    section: str
    note: str = ""
    basis: str = MODEL

    def __post_init__(self) -> None:
        if self.basis not in BASES:
            raise ValueError(f"accept: {self.id}: unknown basis {self.basis!r}")


@dataclass(frozen=True)
class Verdict:
    id: str
    value: object
    op: str
    threshold: object
    gating: bool
    passed: bool | None
    note: str = ""
    # A criterion the spec lets a run skip (section 5.4): NOT RUN, never a pass, and it does not block.
    skipped: bool = False

    @property
    def message(self) -> str:
        if self.skipped:
            return f"{self.id}: NOT RUN ({self.note})"
        return f"{self.id}: {format_value(self.value)}, want {self.op} {format_value(self.threshold)}"

    @property
    def blocks(self) -> bool:
        """True when this verdict fails a run: a gating criterion that did not pass and was not skipped."""
        return self.gating and self.passed is not True and not self.skipped


CRITERIA: dict[str, Criterion] = {
    # 4.1 C1 (D-IMPL-960). C1.K is the per-kernel FP64 oracle check and the only C1 row that can pass without a model.
    # The model-level rows compare whole-model outputs with a reference that the specification does not name: NOT RUN.
    "C1.K": Criterion(id="C1.K", op="==", threshold=0, gating=True, section="4.1", basis=KERNEL,
                      note="failing metal-label tests of ctest --preset mac-metal: the per-kernel oracle twins and the synthetic full-layer compositions"),
    "C1.1.top1": Criterion(id="C1.1.top1", op=">=", threshold=None, gating=True, section="4.1", basis=REFERENCE,
                           note="same top-1 (percent) vs the model-level reference >= its floor minus 1.0 point"),
    "C1.1.kl": Criterion(id="C1.1.kl", op="<=", threshold=None, gating=True, section="4.1", basis=REFERENCE,
                         note="median KL vs the model-level reference <= 1.5 x its floor"),
    "C1.1.safe": Criterion(id="C1.1.safe", op=">=", threshold=100.0, gating=True, section="4.1", basis=REFERENCE,
                           note="same top-1 is 100% (percent) where the reference's margin is in [m1, inf)"),
    "C1.1.band": Criterion(id="C1.1.band", op=">=", threshold=None, gating=True, section="4.1", basis=REFERENCE,
                           note="same top-1 in [0.5, m1) >= the reference's floor in that band minus 3 points"),
    "C1.1.ppl": Criterion(id="C1.1.ppl", op="<=", threshold=None, gating=True, section="4.1", basis=REFERENCE,
                          note="|perplexity difference| vs the reference <= max(3 standard errors, 0.005 nats per token)"),
    "C1.1.p99": Criterion(id="C1.1.p99", op="<=", threshold=None, gating=False, section="4.1", basis=REFERENCE,
                          note="p99 KL vs the reference <= 2 x its floor; reported"),
    "C1.2": Criterion(id="C1.2", op=RETIRED_OP, threshold=None, gating=False, section="4.1", basis=RETIRED,
                      note="retired: its threshold is the CUDA-to-llama.cpp spread (T-3), which needs the CUDA reference (D-IMPL-629)"),
    "C1.3": Criterion(id="C1.3", op=RETIRED_OP, threshold=None, gating=False, section="4.1", basis=RETIRED,
                      note="retired: its per-slot bound is the CUDA-to-llama.cpp spread; a real-text residual has no kernel-level meaning"),
    "C1.4.ids": Criterion(id="C1.4.ids", op="==", threshold=0, gating=True, section="4.1", basis=MODEL,
                          note="router ids on the model's own rows vs the FP64 host router: mismatches that are not a boundary swap within 2^-16"),
    "C1.4.weights": Criterion(id="C1.4.weights", op="<=", threshold=1e-5, gating=True, section="4.1", basis=MODEL,
                              note="max |w_mac - w_fp64| / w_fp64 over matched ids, on the model's own rows"),
    "C1.5": Criterion(id="C1.5", op=">=", threshold=100.0, gating=True, section="4.1", basis=REFERENCE,
                      note="first token after the prompt path: same top-1 (percent) vs the model-level reference where its margin >= m1"),
    # 4.2 C2: Mac against Mac, bitwise. The value is the number of prompts whose output differs from arm (a).
    "C2.b": Criterion(id="C2.b", op="==", threshold=0, gating=True, section="4.2",
                      note="shipped configuration (MTP and suffix drafter) against plain decode"),
    "C2.c": Criterion(id="C2.c", op="==", threshold=0, gating=True, section="4.2",
                      note="widest MTP windows, drafted at every step"),
    "C2.d": Criterion(id="C2.d", op="==", threshold=0, gating=True, section="4.2",
                      note="accepted drafts with the oracle file"),
    "C2.e": Criterion(id="C2.e", op="==", threshold=0, gating=True, section="4.2",
                      note="every third draft wrong: rollback after a partly accepted window"),
    "C2.f": Criterion(id="C2.f", op="==", threshold=0, gating=True, section="4.2",
                      note="MTP drafter's own cells rolled back"),
    "C2.g": Criterion(id="C2.g", op="==", threshold=0, gating=True, section="4.2",
                      note="sampled arms against the sampled --spec 0 arm"),
    "C2.h": Criterion(id="C2.h", op="==", threshold=0, gating=True, section="4.2",
                      note="pipelines compiled at start (STRATA_METAL_ARCHIVE=0)"),
    "C2.i": Criterion(id="C2.i", op="==", threshold=0, gating=True, section="4.2",
                      note="concurrent encoders (STRATA_METAL_DISPATCH=concurrent)"),
    "C2.j": Criterion(id="C2.j", op="==", threshold=0, gating=True, section="4.2",
                      note="the other host wait mode (STRATA_METAL_WAIT)"),
    "C2.d.accept": Criterion(id="C2.d.accept", op=">=", threshold=95.0, gating=True, section="4.2",
                             note="arm (d): percent of drafts accepted, so the arm tests what it should"),
    # 4.3 C3: the conversation cache changes nothing but the time. Each value is the tool's exit code.
    "C3.reuse": Criterion(id="C3.reuse", op="==", threshold=0, gating=True, section="4.3",
                          note="conversation_cache_parity.py --scenario reuse, --spec 1 and 4"),
    "C3.pressure": Criterion(id="C3.pressure", op="==", threshold=0, gating=True, section="4.3",
                             note="conversation_cache_parity.py --scenario pressure"),
    "C3.oversized": Criterion(id="C3.oversized", op="==", threshold=0, gating=True, section="4.3",
                              note="conversation_cache_parity.py --scenario oversized"),
    "C3.exchange": Criterion(id="C3.exchange", op="==", threshold=0, gating=True, section="4.3",
                             note="conversation_cache_parity.py --scenario exchange"),
    "C3.admission": Criterion(id="C3.admission", op="==", threshold=0, gating=True, section="4.3",
                              note="conversation_cache_parity.py --scenario admission"),
    "C3.growth": Criterion(id="C3.growth", op="==", threshold=0, gating=True, section="4.3",
                           note="conversation_cache_growth.py, --spec 1 and 4"),
    "C3.isolation": Criterion(id="C3.isolation", op="==", threshold=0, gating=True, section="4.3",
                              note="conversation_cache_isolation.py, scenarios image, add and project"),
    "C3.soak": Criterion(id="C3.soak", op="==", threshold=0, gating=True, section="4.3",
                         note="conversation_cache_soak.py with its defaults"),
    "C3.http": Criterion(id="C3.http", op="==", threshold=0, gating=True, section="4.3",
                         note="conversation_cache_http_smoke.py against the server run"),
    "C3.disabled": Criterion(id="C3.disabled", op="==", threshold=0, gating=True, section="4.3",
                             note="conversation_cache_disabled.py against the previous release"),
    "C3.reread": Criterion(id="C3.reread", op="==", threshold=0, gating=True, section="4.3",
                           note="STRATA_CKPT_REREAD check: a count of failed conditions"),
    "C3.repeat": Criterion(id="C3.repeat", op="==", threshold=0, gating=True, section="4.3",
                           note="groups of identical requests that do not give one distinct answer"),
    # 4.4 C4: tiered mode gives the bits of full mode. Each value is the number of prompts that differ.
    "C4.tiered": Criterion(id="C4.tiered", op="==", threshold=0, gating=True, section="4.4",
                           note="--adapt-swaps 0 arm against full mode"),
    "C4.swaps": Criterion(id="C4.swaps", op="==", threshold=0, gating=True, section="4.4",
                          note="adaptive-swaps arm against full mode"),
    # 4.5 The prompt path: its yardstick Y_pp was CUDA's (T-1, out of scope, D-IMPL-629): NOT RUN until a reference is named.
    "PP.top1": Criterion(id="PP.top1", op=">=", threshold=None, gating=True, section="4.5", basis=REFERENCE,
                         note="same top-1 (percent) >= Y_pp - 3 points, Y_pp from the model-level reference"),
    "PP.kl": Criterion(id="PP.kl", op="<=", threshold=None, gating=True, section="4.5", basis=REFERENCE,
                       note="mean KL <= 1.5 x Y_pp's mean KL, Y_pp from the model-level reference"),
    "PP.ppl": Criterion(id="PP.ppl", op="<=", threshold=0.03, gating=True, section="4.5", basis=REFERENCE,
                        note="relative perplexity difference on the 255 rows (3%) vs the model-level reference"),
    "PP.chunk": Criterion(id="PP.chunk", op=REPORT, threshold=None, gating=False, section="4.5",
                          note="chunk size chosen by --prefill auto; reported"),
    # 4.6 Needles: the value is the number of needles of that length not found.
    "NEEDLE.32k": Criterion(id="NEEDLE.32k", op="==", threshold=0, gating=True, section="4.6",
                            note="needles missed at 32K"),
    "NEEDLE.128k": Criterion(id="NEEDLE.128k", op="==", threshold=0, gating=True, section="4.6",
                             note="needles missed at 128K"),
    "NEEDLE.262k": Criterion(id="NEEDLE.262k", op="==", threshold=0, gating=True, section="4.6",
                             note="needles missed at 262K"),
    # 4.7 The server end to end. Each value is a boolean.
    "S.models": Criterion(id="S.models", op="==", threshold=True, gating=True, section="4.7",
                          note="GET /v1/models lists model_name"),
    "S.status": Criterion(id="S.status", op="==", threshold=True, gating=True, section="4.7",
                          note="/v1/status and /metrics answer 200 with JSON"),
    "S.openai": Criterion(id="S.openai", op="==", threshold=True, gating=True, section="4.7",
                          note="chat completions contain the expected answers and token counts"),
    "S.stream": Criterion(id="S.stream", op="==", threshold=True, gating=True, section="4.7",
                          note="streamed deltas equal the non-streamed answer and end with [DONE]"),
    "S.tool": Criterion(id="S.tool", op="==", threshold=True, gating=True, section="4.7",
                        note="tool call to get_weather with city Osaka"),
    "S.anthropic": Criterion(id="S.anthropic", op="==", threshold=True, gating=True, section="4.7",
                             note="/v1/messages answer and count_tokens"),
    "S.image": Criterion(id="S.image", op="==", threshold=True, gating=True, section="4.7",
                         note="red and OSAKA read from the images, encoder on the GPU"),
    "S.cancel": Criterion(id="S.cancel", op="==", threshold=True, gating=True, section="4.7",
                          note="request after a cancelled stream answers within 30 s"),
    "S.unload": Criterion(id="S.unload", op="==", threshold=True, gating=True, section="4.7",
                          note="unload and load, then a correct answer"),
    # 5.1 and 5.2 Performance. Values are tokens per second, or a ratio for PF-1.
    "BENCH.mtp": Criterion(id="BENCH.mtp", op=">=", threshold=50.0, gating=True, section="5.1",
                           note="mtp decode median on ca-4k, tok/s; gating at G3 and in the release run"),
    "BENCH.mtp.target": Criterion(id="BENCH.mtp.target", op=">=", threshold=60.0, gating=False, section="5.1",
                                  note="target, reported"),
    "BENCH.plain": Criterion(id="BENCH.plain", op=">=", threshold=20.0, gating=True, section="5.1",
                             note="plain decode on ca-4k, tok/s"),
    "BENCH.other": Criterion(id="BENCH.other", op=REPORT, threshold=None, gating=False, section="5.1",
                             note="every other model, prompt and arm; reported"),
    "PF-1": Criterion(id="PF-1", op=">=", threshold=0.70, gating=True, section="5.2",
                      note="ca-128k prefill rate / ca-32k prefill rate, same run"),
    "PF-2": Criterion(id="PF-2", op=">=", threshold=450.0, gating=True, section="5.2",
                      note="ca-32k prefill rate, tok/s"),
    # 5.4 Regression. The bar is 0.95, or 1 - 2.5 x CV (derived per run).
    "REG.decode": Criterion(id="REG.decode", op=">=", threshold=None, gating=True, section="5.4",
                            note="mtp decode median >= bar x previous release, for ca-1k, ca-4k and ca-32k"),
    "REG.prefill": Criterion(id="REG.prefill", op=">=", threshold=None, gating=True, section="5.4",
                             note="ca-32k prefill median >= bar x previous release"),
    "REG.trend": Criterion(id="REG.trend", op=REPORT, threshold=None, gating=False, section="5.4",
                           note="change against the previous night's result; printed when above 3%"),
    # 5.5 Robustness. Value is true when the smoke step ended with no GPU error and no engine exit.
    "PRESSURE.smoke": Criterion(id="PRESSURE.smoke", op="==", threshold=True, gating=True, section="5.5",
                                note="metal_memory_pressure_smoke: decoding continued"),
    # 6 Gates.
    # G1.tf and G1.free were defined against the golden set's outputs, which are out of scope (D-IMPL-629). Under
    # D-IMPL-671 they need a model-level reference, which the specification does not name, so they are NOT RUN (09 6.1).
    "G1.tf": Criterion(id="G1.tf", op="==", threshold=0, gating=True, section="6.1", basis=REFERENCE,
                       note="teacher-forced positions where the reference's margin >= m1 and the top-1 differs; the 95% floor over all positions is part of this check"),
    "G1.free": Criterion(id="G1.free", op="==", threshold=0, gating=True, section="6.1", basis=REFERENCE,
                         note="free-running differences at positions where the reference's margin >= m1"),
    "G2.speed": Criterion(id="G2.speed", op=">=", threshold=20.0, gating=True, section="6.2",
                          note="Q2_0 mtp decode median on ca-4k in tiered mode, tok/s"),
    # 8 Release.
    "PLAN.models": Criterion(id="PLAN.models", op="==", threshold=0, gating=True, section="8",
                             note="catalogue files without a PLAN line"),
    # Runner-level criteria, reported as FAIL when a step itself fails (docs/mac/09, section 3.2).
    "STEP.exit": Criterion(id="STEP.exit", op="==", threshold=0, gating=True, section="3.2", basis=RUNNER,
                           note="a step exited nonzero after the engine started"),
    "STEP.timeout": Criterion(id="STEP.timeout", op="==", threshold=0, gating=True, section="3.2", basis=RUNNER,
                              note="a step ran past its limit and was stopped (timeout)"),
}


def get(criterion_id: str) -> Criterion:
    try:
        return CRITERIA[criterion_id]
    except KeyError:
        raise ValueError(f"accept: unknown criterion {criterion_id}") from None


def format_value(value: object) -> str:
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".6g")
    return str(value)


def skip(criterion_id: str, note: str) -> Verdict:
    """A criterion that is NOT RUN, with the reason: a skip the spec lets a run make (section 5.4), or a reference
    that is not named (section 4.1). passed is None. A gating skip makes its run NOT RUN, never PASS (accept.py)."""
    row = get(criterion_id)
    return Verdict(id=criterion_id, value=None, op=row.op, threshold=None, gating=row.gating, passed=None,
                   note=note, skipped=True)


def judge(criterion_id: str, value: object, threshold: object = None, note: str = "") -> Verdict:
    """Judge one measured value. Never passes a missing value or an underived threshold, and never passes a
    `reference` row: those are NOT RUN whatever the value (09 section 4.1). A `retired` row has no verdict."""
    row = get(criterion_id)
    if row.basis == RETIRED:
        raise ValueError(f"accept: {criterion_id} is retired (09 section 4.1) and has no verdict")
    if row.basis == REFERENCE:
        return skip(criterion_id, note or REFERENCE_NOTE)
    if row.op == REPORT:
        return Verdict(id=criterion_id, value=value, op=REPORT, threshold=None, gating=row.gating,
                       passed=None, note=note or row.note)
    limit = row.threshold if threshold is None else threshold
    if limit is None:
        raise ValueError(f"accept: {criterion_id}: the threshold is derived and must be given")
    if value is None:
        passed = False
        note = note or "no value was measured"
    else:
        try:
            passed = bool(OPS[row.op](value, limit))
        except TypeError:
            passed = False
            note = note or f"value {value!r} cannot be compared"
    return Verdict(id=criterion_id, value=value, op=row.op, threshold=limit, gating=row.gating,
                   passed=passed, note=note)
