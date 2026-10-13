"""The acceptance runs: their command plans, criteria and measurement hooks (docs/mac/09-testing-and-validation.md, 3.2 and 4-8).

A plan is a list of steps. A step's argv is the engine's argument list (kind "engine"), a full tool command (kind
"tool"), a server command ("server"), an HTTP request description ("http") or a CTest command ("ctest"). Placeholders:
`{python}`, `{engine}`, `{config}` (the step's model), `{result}` (the result folder) and `{step}` (the step's own
stem under steps/). Any other `<...>` text marks an input that a live run must resolve from the models, the golden
set or the host; the runner refuses to start such a step, so a placeholder cannot reach the engine.

`measure` turns the step outputs into verdicts. Only `pf` has one (bench.py, T11). Each other run gets one from the work
item that owns it (`task`); until then a live run stops with NOT RUN before any step starts, and it never reports PASS.
`bench` waits for the previous release's baseline (section 5.4), which is not in the plan yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from typing import Callable

from . import bench
from .criteria import Verdict

PROMPTS = "tests/metal/prompts"
PLACEHOLDER = re.compile(r"<[^<>]+>")

G1_SETS = tuple(f"g1-{number:02d}" for number in range(1, 11))
C2_PROMPTS = G1_SETS + ("ca-1k", "ca-4k", "ca-32k")
C2_NIGHTLY_PROMPTS = G1_SETS + ("ca-4k",)
C2_NIGHTLY_ARMS = ("a", "b", "c", "e", "g")
C4_PROMPTS = G1_SETS + ("ca-4k",)
BENCH_PROMPTS = ("ca-1k", "ca-4k", "ca-32k")
TIERED_BUDGET_MIB = "37888"


@dataclass(frozen=True)
class Step:
    name: str
    argv: tuple[str, ...]
    kind: str = "engine"
    model: str = ""
    prompt: str = ""
    env: tuple[tuple[str, str], ...] = ()
    criteria: tuple[str, ...] = ()
    reads: tuple[str, ...] = ()
    dumps: tuple[str, ...] = ()
    measured: bool = True

    @property
    def unresolved(self) -> tuple[str, ...]:
        return tuple(token for token in self.argv if PLACEHOLDER.search(token))


@dataclass(frozen=True)
class Context:
    models: tuple[str, ...]
    args: dict[str, tuple[str, ...]]
    prompts: tuple[str, ...] | None = None
    arms: tuple[str, ...] | None = None
    short: bool = False
    day: int | None = None


@dataclass(frozen=True)
class Output:
    """One step's stdout and stderr, with the names the result folder gives them (for error messages)."""

    stdout: str
    stderr: str
    stdout_file: str
    stderr_file: str


@dataclass
class RunResults:
    """Step outputs keyed by step name. Filled by the runner after each step that exits 0."""

    outputs: dict[str, Output] = field(default_factory=dict)


Measure = Callable[[RunResults], list[Verdict]]


@dataclass(frozen=True)
class RunSpec:
    name: str
    criteria: tuple[str, ...]
    models: tuple[str, ...] = ("Q2_0",)
    plan: Callable[[Context], list[Step]] | None = None
    needs_vision: bool = False
    accepts_short: bool = False
    accepts_prompts: bool = False
    accepts_day: bool = False
    task: str = ""
    measure: Measure | None = None
    # A timed benchmark: the host checks of section 5.1 run before its first step.
    benchmark: bool = False


def strip_flags(args: tuple[str, ...] | list[str], names: set[str]) -> list[str]:
    """Remove each named flag and its value (a token that does not start with `--`)."""
    out: list[str] = []
    index = 0
    items = list(args)
    while index < len(items):
        if items[index] in names:
            index += 1
            if index < len(items) and not items[index].startswith("--"):
                index += 1
            continue
        out.append(items[index])
        index += 1
    return out


def with_flags(args: tuple[str, ...] | list[str], extra: list[str]) -> list[str]:
    """Return `args` with every flag that `extra` names removed, then `extra` appended: the named changes win."""
    names = {token for token in extra if token.startswith("--")}
    return strip_flags(args, names) + list(extra)


def prompt_args(prompt: str) -> list[str]:
    return ["--tokens-file", f"{PROMPTS}/{prompt}.ids"]


def _timed(
    ctx: Context,
    arms: dict[str, list[str]],
    prompts: tuple[str, ...],
    criteria_for: Callable[[str, str], tuple[str, ...]],
    prefix: str,
) -> list[Step]:
    """A warm-up run and three measured runs of each arm and prompt, each in a fresh process (section 5.1)."""
    steps: list[Step] = []
    for tag in ctx.models:
        base = ctx.args[tag]
        for prompt in prompts:
            for arm, extra in arms.items():
                argv = tuple(with_flags(base, prompt_args(prompt) + extra))
                criteria = criteria_for(arm, prompt)
                steps.append(Step(name=f"{prefix}.{arm}.{prompt}.warmup", argv=argv, model=tag,
                                  prompt=prompt, measured=False))
                for number in (1, 2, 3):
                    steps.append(Step(name=f"{prefix}.{arm}.{prompt}.run{number}", argv=argv, model=tag,
                                      prompt=prompt, criteria=criteria))
    return steps


BENCH_ARMS = {
    "mtp": ["--greedy", "--max-new", "256", "--prompt-cache", "0"],
    "plain": ["--greedy", "--max-new", "256", "--prompt-cache", "0", "--spec", "0", "--suffix-draft", "0"],
}


def _bench_criteria(arm: str, prompt: str) -> tuple[str, ...]:
    if prompt == "ca-4k" and arm == "mtp":
        return ("BENCH.mtp", "BENCH.mtp.target")
    if prompt == "ca-4k" and arm == "plain":
        return ("BENCH.plain",)
    return ("BENCH.other",)


def plan_c1(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    sets = ("tf-code",) if ctx.short else ("tf-code", "tf-doc", "tf-chat")
    for tag in ctx.models:
        base = with_flags(ctx.args[tag], ["--prompt-cache", "0", "--spec", "0", "--adapt-swaps", "0"])
        base = strip_flags(base, {"--prefill"})
        for name in sets:
            steps.append(Step(
                name=f"c1.tf.{name}", model=tag, prompt=name,
                argv=tuple(base + ["--tokens-file", f"{PROMPTS}/{name}.ids", "--max-new", "1",
                                   "--dump-logits", "{step}.logits"]),
                dumps=(".logits",),
                reads=(f"{name}.logits", f"{name}.nocache.logits"),
                criteria=("C1.1.top1", "C1.1.kl", "C1.1.safe", "C1.1.band", "C1.1.ppl", "C1.1.p99"),
            ))
        if not ctx.short:
            for name in sets:
                steps.append(Step(
                    name=f"c1.layers.{name}", model=tag, prompt=name,
                    argv=tuple(base + ["--tokens", f"<first 65 ids of {name}>", "--max-new", "1",
                                       "--dump-layers", "{step}.layers"]),
                    dumps=(".layers",), reads=(f"{name}.layers",), criteria=("C1.3",),
                ))
        steps.append(Step(
            name="c1.router", model=tag, kind="ctest",
            argv=("ctest", "--preset", "mac-model", "-R", "metal_router_replay"),
            criteria=("C1.4.ids", "C1.4.weights"),
        ))
        if not ctx.short:
            for name in G1_SETS:
                steps.append(Step(
                    name=f"c1.first.{name}", model=tag, prompt=name,
                    argv=tuple(with_flags(base, ["--prefill", "auto"]) + ["--tokens-file",
                               f"{PROMPTS}/{name}.ids", "--max-new", "1"]),
                    env=(("STRATA_DUMP_FIRST_LOGITS", "{step}.first"),),
                    dumps=(".first",), reads=(f"{name}.first",), criteria=("C1.5",),
                ))
    return steps


C2_ARMS: tuple[tuple[str, str, list[str], tuple[tuple[str, str], ...]], ...] = (
    ("a", "a", ["--greedy", "--spec", "0"], ()),
    ("b", "b", ["--greedy"], ()),
    ("c", "c", ["--greedy", "--spec", "8", "--suffix-draft", "0", "--spec-min-p", "0"], ()),
    ("d", "d", ["--greedy", "--spec", "4", "--spec-oracle", "<output of arm a for this prompt>"], ()),
    ("e", "e", ["--greedy", "--spec", "4", "--spec-oracle", "<output of arm a for this prompt>",
                "--spec-corrupt", "3"], ()),
    ("f", "f", ["--greedy", "--spec-corrupt", "2"], ()),
    ("g", "g.plain", ["--seed", "42", "--temperature", "0.7", "--top-p", "0.95", "--spec", "0"], ()),
    ("g", "g.spec", ["--seed", "42", "--temperature", "0.7", "--top-p", "0.95"], ()),
    ("g", "g.coupled", ["--seed", "42", "--temperature", "0.7", "--top-p", "0.95", "--coupled-draft"], ()),
    ("h", "h", ["--greedy"], (("STRATA_METAL_ARCHIVE", "0"),)),
    ("i", "i", ["--greedy"], (("STRATA_METAL_DISPATCH", "concurrent"),)),
    ("j", "j", ["--greedy"], (("STRATA_METAL_WAIT", "spin"),)),
)


def plan_c2(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    prompts = ctx.prompts or C2_PROMPTS
    for tag in ctx.models:
        base = with_flags(ctx.args[tag], ["--prompt-cache", "0"])
        for prompt in prompts:
            for group, name, extra, env in C2_ARMS:
                if ctx.arms and group not in ctx.arms:
                    continue
                # Arm (a) and the sampled --spec 0 arm (g.plain) are the references for their groups.
                criteria: tuple[str, ...] = () if name in ("a", "g.plain") else (f"C2.{group}",)
                if group == "d":
                    criteria = ("C2.d", "C2.d.accept")
                steps.append(Step(
                    name=f"c2.{name}.{prompt}", model=tag, prompt=prompt, env=env,
                    argv=tuple(with_flags(base, prompt_args(prompt) + ["--max-new", "256"] + extra)),
                    criteria=criteria,
                ))
    return steps


C3_TOOLS = (
    ("reuse-1", "conversation_cache_parity.py", ["--scenario", "reuse", "--spec", "1"], ("C3.reuse",)),
    ("reuse-4", "conversation_cache_parity.py", ["--scenario", "reuse", "--spec", "4"], ("C3.reuse",)),
    ("pressure", "conversation_cache_parity.py", ["--scenario", "pressure", "--cache-mib",
                                                  "<cc_pressure_mib>"], ("C3.pressure",)),
    ("oversized", "conversation_cache_parity.py", ["--scenario", "oversized", "--cache-mib", "1"],
     ("C3.oversized",)),
    ("exchange", "conversation_cache_parity.py", ["--scenario", "exchange", "--cache-mib",
                                                  "<cc_pressure_mib>"], ("C3.exchange",)),
    ("admission", "conversation_cache_parity.py", ["--scenario", "admission", "--min-free-mib",
                                                   "<hw.memsize in MiB + 4096>"], ("C3.admission",)),
    ("growth-1", "conversation_cache_growth.py", ["--spec", "1"], ("C3.growth",)),
    ("growth-4", "conversation_cache_growth.py", ["--spec", "4"], ("C3.growth",)),
    ("isolation-image", "conversation_cache_isolation.py", ["--scenario", "image"], ("C3.isolation",)),
    ("isolation-add", "conversation_cache_isolation.py", ["--scenario", "add"], ("C3.isolation",)),
    ("isolation-project", "conversation_cache_isolation.py", ["--scenario", "project"], ("C3.isolation",)),
    ("soak", "conversation_cache_soak.py", [], ("C3.soak",)),
)


def _tool(script: str, name: str, extra: list[str], config: str, criteria: tuple[str, ...], tag: str,
          env: tuple[tuple[str, str], ...] = ()) -> Step:
    return Step(
        name=f"c3.{name}", kind="tool", model=tag, criteria=criteria, env=env,
        argv=("{python}", f"tools/{script}", "--run", "--config", config, "--engine", "{engine}",
              "--output", f"{{result}}/c3-{name}", *extra),
    )


def plan_c3(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    for tag in ctx.models:
        for name, script, extra, criteria in C3_TOOLS:
            config = "<config with --max-context 131072>" if name == "soak" else "{config}"
            steps.append(_tool(script, name, extra, config, criteria, tag))
        steps.append(Step(
            name="c3.disabled", kind="tool", model=tag, criteria=("C3.disabled",),
            argv=("{python}", "tools/conversation_cache_disabled.py", "--run", "--config", "{config}",
                  "--upstream", "<previous release engine>", "--candidate", "{engine}",
                  "--output", "{result}/c3-disabled"),
        ))
        steps.append(_tool("conversation_cache_parity.py", "reread-1", ["--scenario", "reuse", "--spec", "1"],
                           "{config}", (), tag))
        steps.append(_tool("conversation_cache_parity.py", "reread-2", ["--scenario", "reuse", "--spec", "1"],
                           "{config}", ("C3.reread",), tag, env=(("STRATA_CKPT_REREAD", "1"),)))
        for mode, flags in (("full", []), ("tiered", ["--gpu-budget-mib", TIERED_BUDGET_MIB])):
            steps.append(Step(
                name=f"c3.repeat.{mode}.server", kind="server", model=tag,
                argv=("{python}", "serve/server.py", "--engine", "{engine}", "--config", "{config}",
                      "--port", "<free port>", "--prompt-cache", "0", *flags),
            ))
            for number in range(8):
                steps.append(Step(
                    name=f"c3.repeat.{mode}.greedy.{number}", kind="http", model=tag, prompt="g1-01",
                    argv=("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-01",
                          "temperature 0", "max_tokens 64"),
                    criteria=("C3.repeat",),
                ))
            for number in range(8):
                steps.append(Step(
                    name=f"c3.repeat.{mode}.seeded.{number}", kind="http", model=tag, prompt="g1-02",
                    argv=("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-02",
                          "temperature 0.7", "seed 42", "max_tokens 64"),
                    criteria=("C3.repeat",),
                ))
    return steps


def plan_c4(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    modes = (("greedy", ["--greedy"]), ("seeded", ["--seed", "42", "--temperature", "0.7"]))
    for tag in ctx.models:
        base = ctx.args[tag]
        for mode, mode_flags in modes:
            for prompt in ctx.prompts or C4_PROMPTS:
                common = prompt_args(prompt) + ["--max-new", "256"] + mode_flags
                arms = (
                    # Full mode: the reference has no budget and no adaptive swaps.
                    ("full", strip_flags(base, {"--gpu-budget-mib", "--adapt-swaps"}), ()),
                    ("tiered", with_flags(base, ["--gpu-budget-mib", TIERED_BUDGET_MIB, "--adapt-swaps", "0"]),
                     ("C4.tiered",)),
                    ("swaps", with_flags(base, ["--gpu-budget-mib", TIERED_BUDGET_MIB]), ("C4.swaps",)),
                )
                for arm, arm_base, criteria in arms:
                    steps.append(Step(
                        name=f"c4.{arm}.{mode}.{prompt}", model=tag, prompt=prompt,
                        argv=tuple(with_flags(arm_base, common)), criteria=criteria,
                    ))
    return steps


def plan_pp(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    for tag in ctx.models:
        base = with_flags(ctx.args[tag], ["--prompt-cache", "0"])
        steps.append(Step(
            name="pp.ca-32k", model=tag, prompt="ca-32k",
            argv=tuple(with_flags(base, ["--prefill", "auto", "--prefill-until", "32000"] + prompt_args("ca-32k")
                                  + ["--max-new", "1", "--max-context", "32768", "--dump-logits", "{step}.logits"])),
            dumps=(".logits",), reads=("pp-32k.logits",),
            criteria=("PP.top1", "PP.kl", "PP.ppl", "PP.chunk"),
        ))
    return steps


def plan_bench(ctx: Context) -> list[Step]:
    arms = {arm: flags for arm, flags in BENCH_ARMS.items() if not ctx.arms or arm in ctx.arms}
    return _timed(ctx, arms, ctx.prompts or BENCH_PROMPTS, _bench_criteria, "bench")


def plan_pf(ctx: Context) -> list[Step]:
    def criteria(arm: str, prompt: str) -> tuple[str, ...]:
        if prompt == "ca-32k":
            return ("PF-2", "PF-1")
        return ("PF-1",)

    return _timed(replace(ctx, arms=("mtp",)), {"mtp": BENCH_ARMS["mtp"]}, ("ca-32k", "ca-128k"), criteria, "pf")


NEEDLE_LENGTHS = (("32k", (10, 50, 90), 32768), ("128k", (10, 90), 131072), ("262k", (50,), 262144))


def plan_needle(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    for tag in ctx.models:
        for length, depths, _context in NEEDLE_LENGTHS:
            steps.append(Step(
                name=f"needle.{length}.server", kind="server", model=tag,
                argv=("{python}", "serve/server.py", "--engine", "{engine}", "--config",
                      f"<config with --max-context {_context}>", "--port", "<free port>"),
            ))
            steps.append(Step(
                name=f"needle.{length}", kind="tool", model=tag, criteria=(f"NEEDLE.{length}",),
                argv=("{python}", "tools/needle_bench.py", "--url", "http://127.0.0.1:<port>",
                      "--lengths", length, "--depths", ",".join(str(d) for d in depths),
                      "--out", f"{{result}}/needles-{length}.json"),
            ))
    return steps


SERVER_CHECKS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("S.models", ("GET", "http://127.0.0.1:<port>/v1/models")),
    ("S.status", ("GET", "http://127.0.0.1:<port>/v1/status")),
    ("S.status", ("GET", "http://127.0.0.1:<port>/metrics")),
    ("S.openai", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-01..g1-10", "temperature 0",
                  "max_tokens 256")),
    ("S.stream", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-01", "stream true")),
    ("S.tool", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-06", "tools")),
    ("S.anthropic", ("POST", "http://127.0.0.1:<port>/v1/messages", "g1-02")),
    ("S.anthropic", ("POST", "http://127.0.0.1:<port>/v1/messages/count_tokens", "g1-02")),
    ("S.image", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "tests/metal/images/red.png",
                 "temperature 0")),
    ("S.image", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "tests/metal/images/osaka.png",
                 "temperature 0")),
    ("S.cancel", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-07", "stream, close after 8 deltas")),
    ("S.cancel", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-02")),
    ("S.unload", ("POST", "http://127.0.0.1:<port>/v1/unload")),
    ("S.unload", ("POST", "http://127.0.0.1:<port>/v1/load")),
    ("S.unload", ("POST", "http://127.0.0.1:<port>/v1/chat/completions", "g1-02")),
)


def plan_server(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    for tag in ctx.models:
        steps.append(Step(
            name="server.start", kind="server", model=tag,
            argv=("{python}", "serve/server.py", "--engine", "{engine}", "--config", "{config}",
                  "--port", "<free port>"),
        ))
        for index, (criterion, argv) in enumerate(SERVER_CHECKS):
            steps.append(Step(name=f"server.{index:02d}", kind="http", model=tag, criteria=(criterion,),
                              argv=argv))
        steps.append(Step(
            name="server.c3-http", kind="tool", model=tag, criteria=("C3.http",),
            argv=("{python}", "tools/conversation_cache_http_smoke.py", "--run", "--url",
                  "http://127.0.0.1:<port>", "--output", "{result}/c3-http.json"),
        ))
    return steps


def plan_g1(ctx: Context) -> list[Step]:
    steps = plan_c1(replace(ctx, short=False))
    steps += plan_c2(replace(ctx, arms=C2_NIGHTLY_ARMS, prompts=C2_NIGHTLY_PROMPTS))
    steps += _timed(replace(ctx, arms=("plain",), prompts=("ca-4k",)),
                    {"plain": BENCH_ARMS["plain"]}, ("ca-4k",), _bench_criteria, "bench")
    for tag in ctx.models:
        base = ctx.args[tag]
        for name in G1_SETS:
            steps.append(Step(
                name=f"g1.tf.{name}", model=tag, prompt=name, criteria=("G1.tf",), dumps=(".logits",),
                reads=(f"{name}.tf.logits",),
                argv=tuple(with_flags(base, ["--tokens-file", f"<{name}.ids + {name}.out>", "--max-new", "1",
                                             "--dump-logits", "{step}.logits"])),
            ))
            steps.append(Step(
                name=f"g1.free.{name}", model=tag, prompt=name, criteria=("G1.free",), reads=(f"{name}.out",),
                argv=tuple(with_flags(base, prompt_args(name) + ["--greedy", "--spec", "0", "--max-new", "256"])),
            ))
    return steps


def plan_g2(ctx: Context) -> list[Step]:
    steps = _timed(replace(ctx, arms=("mtp",), prompts=("ca-4k",)), {"mtp": [
        "--greedy", "--max-new", "256", "--prompt-cache", "0", "--gpu-budget-mib", TIERED_BUDGET_MIB,
        "--max-context", "8192"]}, ("ca-4k",), lambda arm, prompt: ("G2.speed",), "g2")
    return steps + plan_c4(ctx)


def plan_g3(ctx: Context) -> list[Step]:
    del ctx
    return [Step(name="g3.test_map", kind="tool", argv=("{python}", "tools/metal/test_map.py", "--final"))]


def plan_release(ctx: Context) -> list[Step]:
    steps: list[Step] = []
    for tag in ctx.models:
        steps.append(Step(
            name=f"release.plan.{tag}", model=tag, criteria=("PLAN.models",),
            argv=tuple(with_flags(ctx.args[tag], ["--plan-only"])),
        ))
    return steps


def measure_pf(results: RunResults) -> list[Verdict]:
    return bench.measure_pf(results.outputs)


RUN_LIST: tuple[RunSpec, ...] = (
    RunSpec("c1", ("C1.1.top1", "C1.1.kl", "C1.1.safe", "C1.1.band", "C1.1.ppl", "C1.1.p99", "C1.2", "C1.3",
                   "C1.4.ids", "C1.4.weights", "C1.5"), models=("Q2_0", "Coder"), plan=plan_c1,
            accepts_short=True, task="T6"),
    RunSpec("c2", ("C2.b", "C2.c", "C2.d", "C2.e", "C2.f", "C2.g", "C2.h", "C2.i", "C2.j", "C2.d.accept"),
            plan=plan_c2, accepts_prompts=True, task="T8"),
    RunSpec("c3", ("C3.reuse", "C3.pressure", "C3.oversized", "C3.exchange", "C3.admission", "C3.growth",
                   "C3.isolation", "C3.soak", "C3.disabled", "C3.reread", "C3.repeat"),
            plan=plan_c3, task="T9"),
    RunSpec("c4", ("C4.tiered", "C4.swaps"), models=("Q2_0", "Coder"), plan=plan_c4, accepts_prompts=True,
            task="T10"),
    RunSpec("pp", ("PP.top1", "PP.kl", "PP.ppl", "PP.chunk"), plan=plan_pp, task="T11"),
    RunSpec("needle", ("NEEDLE.32k", "NEEDLE.128k", "NEEDLE.262k"), plan=plan_needle, task="T12"),
    RunSpec("server", ("S.models", "S.status", "S.openai", "S.stream", "S.tool", "S.anthropic", "S.image",
                       "S.cancel", "S.unload", "C3.http"), plan=plan_server, needs_vision=True, task="T12"),
    RunSpec("bench", ("BENCH.mtp", "BENCH.mtp.target", "BENCH.plain", "BENCH.other", "REG.decode", "REG.prefill",
                      "REG.trend"), plan=plan_bench, accepts_prompts=True, task="T11", benchmark=True),
    RunSpec("pf", ("PF-1", "PF-2"), plan=plan_pf, task="T11", measure=measure_pf, benchmark=True),
    RunSpec("g1", ("G1.tf", "G1.free"), plan=plan_g1, task="T7", benchmark=True),
    RunSpec("g2", ("G2.speed",), plan=plan_g2, task="T7", benchmark=True),
    RunSpec("g3", (), plan=plan_g3, task="T7"),
    RunSpec("nightly", (), task="T14"),
    RunSpec("weekly", ("PRESSURE.smoke",), accepts_day=True, task="T14"),
    RunSpec("release", ("PLAN.models",), models=("Q2_0",), plan=plan_release, task="T15"),
    RunSpec("family", (), task="T16"),
)
RUNS: dict[str, RunSpec] = {spec.name: spec for spec in RUN_LIST}
