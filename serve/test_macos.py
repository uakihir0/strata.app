import importlib.util
import os
import select
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import macos
import setup

# PLAN lines built from the field order in docs/mac/05-memory-and-residency.md, section 3.4, and plan_line() in
# src/metal/engine/plan.cpp. They are constructed for these tests, not captured from an engine run.
PLAN_FIELDS_FULL = {
    "fits": 1, "mode": "full", "gpu": "Apple_M5_Pro", "w_mib": 53084, "reserve_mib": 700, "ram_mib": 65536,
    "avail_mib": 41210, "dense_mib": 3471, "unread_mib": 1, "pack_mib": 0, "mtp_mib": 786, "mtp_bind_mib": 17,
    "fixed_mib": 256, "kv_mib": 1648, "mtp_kv_mib": 130, "kv": "fp16", "context": 65536, "prefill_chunk": 8192,
    "prefill_mib": 2194, "experts_total": 24576, "experts_total_mib": 32400, "experts_resident": 24576,
    "experts_resident_mib": 32400, "slots": 0, "slot_mib": 0, "ssd_experts": 0, "total_mib": 40904,
    "free_mib": 11480, "ram_warning": 0, "w_full_mib": 40905, "mtp_params_mib": 1, "session_state_mib": 113,
    "window_scratch_mib": 47,
}
NO_FIT_REASON = "the_model_does_not_fit:_needs_30000_MiB_of_GPU_memory_with_no_expert_resident,_25000_MiB_available"


def plan_line(*, drop=(), **changes) -> str:
    """A PLAN line with the full field set, changed or without the keys named in `changes` and `drop`."""
    fields = dict(PLAN_FIELDS_FULL)
    for key in drop:
        del fields[key]
    if "reason" in changes:
        reason = changes.pop("reason")
        fields = {**fields, **changes}
        ordered = {}
        for key, value in fields.items():
            if key == "mtp_params_mib":
                ordered["reason"] = reason  # the engine prints reason between w_full_mib and mtp_params_mib
            ordered[key] = value
        fields = ordered
    else:
        fields.update(changes)
    return "PLAN " + " ".join(f"{key}={value}" for key, value in fields.items())


class FakeRunner:
    """Replaces setup.PLAN_RUNNER (subprocess.run). Records each command and returns the canned result or raises."""

    def __init__(self, returncode=0, stdout="", stderr="", raises=None):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.raises = raises
        self.calls = []
        self.kwargs = None

    def __call__(self, cmd, *, capture_output, text, timeout):
        self.calls.append(list(cmd))
        self.kwargs = {"capture_output": capture_output, "text": text, "timeout": timeout}
        if self.raises is not None:
            raise self.raises
        return subprocess.CompletedProcess(list(cmd), self.returncode, self.stdout, self.stderr)


class MetalPlanParse(unittest.TestCase):
    def setUp(self):
        setup.metal_plan.cache_clear()
        self.addCleanup(setup.metal_plan.cache_clear)

    def plan(self, runner, *args, **kwargs):
        setup.metal_plan.cache_clear()
        with mock.patch.object(setup, "PLAN_RUNNER", runner):
            return setup.metal_plan(*(args or ("qwen", "Q2_0", 65536, "fp16")), **kwargs)

    def test_metal_plan_parse(self):
        full = setup.parse_plan_line(plan_line())
        self.assertEqual(1, full["fits"])
        self.assertEqual("full", full["mode"])
        self.assertEqual("Apple_M5_Pro", full["gpu"])
        self.assertEqual(53084, full["w_mib"])
        self.assertEqual(11480, full["free_mib"])
        self.assertNotIn("reason", full)
        self.assertEqual("full", setup.plan_class(full))

        # avail_mib is -1 when the engine was not given the available memory, and free_mib can be negative.
        unmeasured = setup.parse_plan_line(plan_line(avail_mib=-1, free_mib=-5))
        self.assertEqual(-1, unmeasured["avail_mib"])
        self.assertEqual(-5, unmeasured["free_mib"])

        # A key appended by a later engine is kept (05 section 3.4).
        self.assertEqual(7, setup.parse_plan_line(plan_line(future_key=7))["future_key"])

        no_fit = setup.parse_plan_line(plan_line(fits=0, reason=NO_FIT_REASON))
        self.assertEqual(0, no_fit["fits"])
        self.assertEqual(NO_FIT_REASON, no_fit["reason_raw"])  # the underscores are kept
        self.assertEqual("the model does not fit: needs 30000 MiB of GPU memory with no expert resident, "
                         "25000 MiB available", no_fit["reason"])
        self.assertEqual("none", setup.plan_class(no_fit))

        unsupported = setup.parse_plan_line(plan_line(fits=0, reason="unsupported_type_q4_K"))
        self.assertEqual("unsupported_type_q4_K", unsupported["reason_raw"])
        self.assertEqual("unsupported type q4 K", unsupported["reason"])
        self.assertEqual("unsupported", setup.plan_class(unsupported))

        malformed = {
            "not a PLAN line": "plan " + plan_line()[5:],
            "empty line": "",
            "bare token": plan_line() + " bogus",
            "empty value": plan_line(w_mib=""),
            "empty text value": plan_line(gpu=""),
            "repeated key": plan_line() + " w_mib=1",
            "non-integer count": plan_line(w_mib="53k"),
            "negative count": plan_line(w_mib=-5),
            "fits not 0 or 1": plan_line(fits=2),
            "unknown mode": plan_line(mode="hybrid"),
            "ram_warning not a flag": plan_line(ram_warning=2),
            "unknown KV format": plan_line(kv="bf16"),
            "avail_mib below -1": plan_line(avail_mib=-2),
            "more resident than total": plan_line(experts_total=10, experts_resident=11),
            "fits=1 with a reason": plan_line(reason=NO_FIT_REASON),
            "fits=0 without a reason": plan_line(fits=0),
            "reason the engine does not give": plan_line(fits=0, reason="out_of_memory"),
        }
        for label, line in malformed.items():
            with self.subTest(malformed=label):
                with self.assertRaises(setup.PlanError):
                    setup.parse_plan_line(line)

        # Every field the engine prints is required: a line cut short or missing any key must raise.
        for key in setup.PLAN_FIELDS:
            with self.subTest(missing=key):
                with self.assertRaises(setup.PlanError):
                    setup.parse_plan_line(plan_line(drop=(key,)))

    def test_plan_class(self):
        def tiered(total, resident):
            return setup.parse_plan_line(plan_line(mode="tiered", experts_total=total, experts_resident=resident))

        for total, resident, expected in ((100, 100, "ok"), (100, 90, "ok"), (100, 85, "ok"), (100, 84, "slow"),
                                          (100, 40, "slow"), (100, 39, "very slow"), (100, 0, "very slow")):
            with self.subTest(total=total, resident=resident):
                self.assertEqual(expected, setup.plan_class(tiered(total, resident)))
        for incomplete in ({}, {"fits": 0}, {"fits": 1, "mode": "tiered", "experts_resident": 1},
                           {"fits": 1, "mode": "tiered", "experts_total": 0, "experts_resident": 0},
                           {"fits": 2, "mode": "full"}):
            with self.subTest(incomplete=incomplete):
                with self.assertRaises(setup.PlanError):
                    setup.plan_class(incomplete)

    def test_metal_plan_runs_the_engine_and_reads_its_plan(self):
        runner = FakeRunner(stdout=plan_line() + "\n")
        plan = self.plan(runner)
        self.assertEqual("full", plan["mode"])
        self.assertEqual(1, len(runner.calls))
        cmd = runner.calls[0]
        self.assertEqual(str(setup.ENGINE_PATH), cmd[0])
        self.assertEqual(1, cmd.count("--plan-only"))
        self.assertEqual(str(setup.GEOMETRY_DIR / "qwen-Q2_0.txt"), cmd[cmd.index("--geometry") + 1])
        self.assertEqual(str(setup.DATA_DIR / "expert-profile.bin"), cmd[cmd.index("--expert-profile") + 1])
        for flag, value in (("--expert-cache", "auto"), ("--prefill", "auto"), ("--spec", "4"),
                            ("--max-context", "65536"), ("--kv", "fp16"), ("--vram-reserve-mib", "700")):
            self.assertEqual(value, cmd[cmd.index(flag) + 1], flag)
        self.assertEqual({"capture_output": True, "text": True, "timeout": setup.PLAN_TIMEOUT_S}, runner.kwargs)
        self.assertNotIn("--resident-budget-gib", cmd)

        runner = FakeRunner(stdout=plan_line(context=8192) + "\n")
        self.plan(runner, "coder", "IQ1_M", 8192, "fp16", vision="gpu", budget_gib=8.0, gpu_budget_mib=30000,
                  files=("/m/pack", "/m/shard1.gguf", "/m/ple.gguf", "/m/mtp"))
        cmd = runner.calls[0]
        self.assertEqual(str(setup.DATA_DIR / "expert-profile-coder.bin"), cmd[cmd.index("--expert-profile") + 1])
        self.assertNotIn("--geometry", cmd)
        self.assertEqual("/m/pack", cmd[cmd.index("--pack") + 1])
        self.assertEqual("/m/shard1.gguf", cmd[cmd.index("--native") + 1])
        self.assertEqual("/m/ple.gguf", cmd[cmd.index("--ple-gguf") + 1])
        self.assertEqual("/m/mtp", cmd[cmd.index("--mtp") + 1])
        self.assertEqual("2100", cmd[cmd.index("--vram-reserve-mib") + 1])
        self.assertEqual("8.0", cmd[cmd.index("--resident-budget-gib") + 1])
        self.assertEqual("30000", cmd[cmd.index("--gpu-budget-mib") + 1])

    def test_metal_plan_fit_result_and_no_fit_result(self):
        plan = self.plan(FakeRunner(returncode=3, stdout=plan_line(fits=0, reason=NO_FIT_REASON) + "\n"))
        self.assertEqual("none", setup.plan_class(plan))
        plan = self.plan(FakeRunner(returncode=0, stdout=plan_line() + "\n"))
        self.assertEqual("full", setup.plan_class(plan))

    def test_metal_plan_refuses_a_plan_it_cannot_trust(self):
        cases = {
            "exit 0 with fits=0": FakeRunner(returncode=0, stdout=plan_line(fits=0, reason=NO_FIT_REASON) + "\n"),
            "exit 3 with fits=1": FakeRunner(returncode=3, stdout=plan_line() + "\n"),
            "exit 1 with a full plan on stdout": FakeRunner(returncode=1, stdout=plan_line() + "\n"),
            "exit 2 with a full plan on stdout": FakeRunner(returncode=2, stdout=plan_line() + "\n"),
            "SIGKILL with a full plan on stdout": FakeRunner(returncode=-9, stdout=plan_line() + "\n"),
            "exit 0 with no stdout": FakeRunner(returncode=0, stdout=""),
            "exit 0 with two PLAN lines": FakeRunner(returncode=0, stdout=plan_line() + "\n" + plan_line() + "\n"),
            "exit 0 with text beside the PLAN line": FakeRunner(returncode=0, stdout="note\n" + plan_line() + "\n"),
            "exit 0 with a malformed PLAN line": FakeRunner(returncode=0, stdout=plan_line(w_mib="x") + "\n"),
            "a different context than requested": FakeRunner(returncode=0, stdout=plan_line(context=8192) + "\n"),
            "a different KV than requested": FakeRunner(returncode=0, stdout=plan_line(kv="int8") + "\n"),
        }
        for label, runner in cases.items():
            with self.subTest(case=label):
                with self.assertRaises(setup.PlanError):
                    self.plan(runner)

    def test_metal_plan_failure_hints(self):
        messages = {}
        messages["sigkill"] = self._failure_message(FakeRunner(returncode=-9))
        messages["library"] = self._failure_message(
            FakeRunner(returncode=1, stderr="dyld: Library not loaded: @rpath/libstrata.dylib\n"))
        messages["timeout"] = self._failure_message(
            FakeRunner(raises=subprocess.TimeoutExpired(["engine"], setup.PLAN_TIMEOUT_S)))
        messages["missing binary"] = self._failure_message(FakeRunner(raises=FileNotFoundError("engine")))
        messages["other"] = self._failure_message(
            FakeRunner(returncode=1, stderr="\n".join(f"line {i}" for i in range(1, 8))))
        self.assertIn("xattr -dr com.apple.quarantine", messages["sigkill"])
        self.assertIn("files are incomplete", messages["library"])
        self.assertIn("more than 30 s", messages["timeout"])
        self.assertIn("cannot start", messages["missing binary"])
        self.assertIn("delete the engine folder", messages["other"])
        # Only the last five stderr lines are reported.
        self.assertIn("line 7", messages["other"])
        self.assertIn("line 3", messages["other"])
        self.assertNotIn("line 2", messages["other"])

    def _failure_message(self, runner) -> str:
        with self.assertRaises(setup.PlanError) as context:
            self.plan(runner)
        return str(context.exception)

    def test_metal_plan_rejects_bad_requests_before_the_engine_runs(self):
        bad = {
            "unknown family": (("nope", "Q2_0", 65536, "fp16"), {}),
            "unknown model": (("qwen", "Q9", 65536, "fp16"), {}),
            "zero context": (("qwen", "Q2_0", 0, "fp16"), {}),
            "boolean context": (("qwen", "Q2_0", True, "fp16"), {}),
            "unknown KV": (("qwen", "Q2_0", 65536, "q4"), {}),
            "unknown vision": (("qwen", "Q2_0", 65536, "fp16"), {"vision": "tpu"}),
            "incomplete files": (("qwen", "Q2_0", 65536, "fp16"), {"files": ("/p", "/s", None)}),
            "empty MTP path": (("qwen", "Q2_0", 65536, "fp16"), {"files": ("/p", "/s", None, "")}),
        }
        for label, (args, kwargs) in bad.items():
            with self.subTest(request=label):
                runner = FakeRunner(stdout=plan_line() + "\n")
                setup.metal_plan.cache_clear()
                with mock.patch.object(setup, "PLAN_RUNNER", runner):
                    with self.assertRaises(ValueError):
                        setup.metal_plan(*args, **kwargs)
                self.assertEqual([], runner.calls)


class MacSetting(unittest.TestCase):
    def test_setup_mac_switch(self):
        """Runs everywhere: setup.MAC must exist and select the Mac branches by platform (section 4.1)."""
        self.assertTrue(hasattr(setup, "MAC"), "setup.py must define MAC (docs/mac/07, section 4.1)")
        self.assertIs(setup.MAC, sys.platform == "darwin")
        for platform, expected in (("darwin", True), ("linux", False), ("win32", False)):
            with self.subTest(platform=platform), mock.patch.object(sys, "path", list(sys.path)):
                spec = importlib.util.spec_from_file_location(f"setup_as_{platform}", setup.__file__)
                module = importlib.util.module_from_spec(spec)
                with mock.patch.object(sys, "platform", platform):
                    spec.loader.exec_module(module)
                self.assertIs(module.MAC, expected)


class MacOSProbes(unittest.TestCase):
    def test_no_raise_anywhere(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(macos, "MAC", False):
            path = Path(temporary)
            calls = (
                lambda: macos.sysctl_raw("hw.memsize"),
                lambda: macos.sysctl_int("hw.memsize"),
                lambda: macos.sysctl_str("machdep.cpu.brand_string"),
                macos.ram_bytes,
                macos.macos_version,
                macos.translated,
                macos.cpu_brand,
                macos.perf_levels,
                macos.fast_cores,
                macos.arm_features,
                macos.pressure_level,
                macos.swap_used,
                macos.wired_limit,
                macos.metal_devices,
                lambda: macos.device_problem(None),
                macos.ioreg_gpu,
                macos.thermal_state,
                macos.low_power_mode,
                lambda: macos.proc_identity(os.getpid()),
                lambda: macos.strip_quarantine(path),
                lambda: macos.has_quarantine(path),
                lambda: macos.tm_exclude(path),
                macos.firewall_on,
                macos.keep_awake,
                macos.ssl_setup,
                macos.engine_pids,
                macos.xcode_problem,
            )
            for call in calls:
                with self.subTest(call=call):
                    call()

    @unittest.skipUnless(macos.MAC, "macOS only")
    def test_proc_identity_zombie(self):
        child = subprocess.Popen(["/bin/sleep", "30"])
        events = select.kqueue()
        try:
            exit_notice = select.kevent(
                child.pid,
                filter=select.KQ_FILTER_PROC,
                flags=select.KQ_EV_ADD | select.KQ_EV_ENABLE | select.KQ_EV_ONESHOT,
                fflags=select.KQ_NOTE_EXIT,
            )
            events.control([exit_notice], 1, 0)
            os.kill(child.pid, signal.SIGTERM)
            observed = events.control(None, 1, 5)
            self.assertEqual(1, len(observed), "kqueue did not report child exit")
            self.assertEqual(child.pid, observed[0].ident)
            self.assertTrue(observed[0].fflags & select.KQ_NOTE_EXIT)
            os.kill(child.pid, 0)  # It is still an unreaped child, so this must be a zombie.
            self.assertIsNone(macos.proc_identity(child.pid))
        finally:
            if child.poll() is None:
                child.kill()
            child.wait()
            events.close()
        identity = macos.proc_identity(os.getpid())
        self.assertRegex(identity or "", r"^mac:\d+\.\d+$")
        self.assertEqual(identity, macos.proc_identity(os.getpid()))


ROOT = Path(__file__).resolve().parents[1]


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class ServerSockets(unittest.TestCase):
    @unittest.skipUnless(macos.MAC, "macOS only")
    def test_port_probe(self):
        """docs/mac/07, section 6.1: the four rows of the table, with real loopback sockets on a free port.  The
        listeners set SO_REUSEADDR as the server's own socket does."""
        from serve import server
        port = free_port()
        cases = (
            ("0.0.0.0 listening, probe 127.0.0.1", "0.0.0.0", "127.0.0.1", True),
            ("127.0.0.1 listening, probe 0.0.0.0", "127.0.0.1", "0.0.0.0", True),
            ("127.0.0.1 listening, probe 127.0.0.1", "127.0.0.1", "127.0.0.1", True),
            ("nothing listening, probe 127.0.0.1", None, "127.0.0.1", False),
            ("nothing listening, probe 0.0.0.0", None, "0.0.0.0", False),
        )
        for label, listen_on, probe, expected in cases:
            with self.subTest(label):
                listener = None
                try:
                    if listen_on:
                        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                        listener.bind((listen_on, port))
                        listener.listen(1)
                    self.assertIs(server.port_in_use(probe, port), expected)
                finally:
                    if listener is not None:
                        listener.close()


class ServerLifecycle(unittest.TestCase):
    def setUp(self):
        # test_no_raise_anywhere caches macos_version() as () while MAC is patched False: read the host's value again
        macos.macos_version.cache_clear()

    @unittest.skipUnless(macos.MAC, "macOS only")
    def test_sighup_shutdown(self):
        """docs/mac/07, section 6.2: closing the Terminal window sends SIGHUP to the session.  The mock-engine server
        exits 0 within 10 s and its port is free again."""
        from serve import server
        reason = server.macos_host_error()
        if reason:
            self.skipTest(reason)
        port = free_port()
        with tempfile.TemporaryDirectory() as temp:
            log = Path(temp) / "server.log"
            with open(log, "w", encoding="utf-8") as out:
                proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "mock",
                                         "--port", str(port)], cwd=str(ROOT), stdin=subprocess.DEVNULL, stdout=out,
                                        stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + 60
                while "ready:" not in log.read_text(encoding="utf-8"):
                    if proc.poll() is not None:
                        self.fail(f"the server exited before it was ready ({proc.returncode}): "
                                  + log.read_text(encoding="utf-8"))
                    if time.monotonic() > deadline:
                        self.fail("the server was not ready within 60 s")
                    time.sleep(0.2)
                time.sleep(0.5)     # the SIGHUP handler is installed just after the ready text
                os.killpg(proc.pid, signal.SIGHUP)
                try:
                    returned = proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.fail("the server did not exit within 10 s of SIGHUP")
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
        self.assertEqual(returned, 0)
        self.assertFalse(server.port_in_use("127.0.0.1", port))

    @unittest.skipUnless(macos.MAC, "macOS only")
    def test_keep_awake(self):
        """docs/mac/07, section 6.9: caffeinate runs while the model is loaded and ends at an unload; it starts again
        at the next load, and with keep_awake off none runs.  The stub engine has no process of its own."""
        from serve import server
        from serve.frontend import ChatTemplate

        class StubEngine:
            def __init__(self):
                self.up = False

            def alive(self):
                return self.up

            def restart(self):
                self.up = True

            def unload(self):
                self.up = False

        def caffeinate_children():
            r = subprocess.run(["/usr/bin/pgrep", "-P", str(os.getpid()), "caffeinate"], capture_output=True,
                               text=True, timeout=10)
            return [int(x) for x in r.stdout.split()]

        svc = server.Service(StubEngine(), server.ByteTokenizer(), ChatTemplate(ROOT / "serve" / "chat_template.jinja"))
        self.addCleanup(lambda: (setattr(svc, "keep_awake", False), svc.keep_awake_sync()))
        svc.keep_awake = True
        svc.ensure_loaded()
        first = caffeinate_children()
        self.assertEqual(len(first), 1)
        argv = subprocess.run(["/bin/ps", "-o", "command=", "-p", str(first[0])], capture_output=True, text=True,
                              timeout=10).stdout.split()
        self.assertTrue(argv[0].endswith("caffeinate"))
        self.assertEqual(argv[-3:], ["-i", "-w", str(os.getpid())])
        self.assertEqual(svc.unload(), "unloaded")
        self.assertEqual(caffeinate_children(), [])
        svc.ensure_loaded()
        second = caffeinate_children()
        self.assertEqual(len(second), 1)
        self.assertNotEqual(second, first)
        svc.keep_awake = False
        svc.unload()
        self.assertEqual(caffeinate_children(), [])
        svc.ensure_loaded()
        self.assertEqual(caffeinate_children(), [])


class SourceRules(unittest.TestCase):
    def test_module_rules(self):
        """Runs everywhere: serve/macos.py parses as Python 3.9, imports only the standard library and starts every
        subprocess with an absolute path (docs/mac/07, sections 3.1 and 10.3)."""
        import ast
        if not hasattr(sys, "stdlib_module_names"):
            self.skipTest("the standard-library list needs Python 3.10+")
        tree = ast.parse((ROOT / "serve" / "macos.py").read_text(encoding="utf-8"), feature_version=(3, 9))
        allowed = set(sys.stdlib_module_names) | {"__future__"}
        launchers = {"run", "Popen", "call", "check_call", "check_output", "_tool"}
        checked = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                self.assertEqual(node.level, 0, "relative import in serve/macos.py")
                names = [node.module or ""]
            else:
                names = []
            for name in names:
                self.assertIn(name.split(".")[0], allowed, f"{name} is not a standard-library module")
            if isinstance(node, ast.Call) and node.args and isinstance(node.args[0], ast.List):
                func = node.func
                name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", "")
                if name in launchers:
                    checked += 1
                    first = node.args[0].elts[0] if node.args[0].elts else None
                    self.assertIsInstance(first, ast.Constant, f"line {node.lineno}: no literal program")
                    self.assertTrue(str(first.value).startswith("/"), f"line {node.lineno}: {first.value!r}")
        self.assertGreater(checked, 0, "no subprocess call was checked")


if __name__ == "__main__":
    unittest.main()
