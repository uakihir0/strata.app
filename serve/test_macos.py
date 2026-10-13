import contextlib
import errno
import http.server
import importlib.util
import io
import json
import os
import select
import shutil
import signal
import socket
import ssl
import stat
import string
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import zipfile
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from unittest import mock

from serve import macos, server, telemetry
import setup

ROOT = Path(__file__).resolve().parents[1]

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


def _stub_setup_sh(path: Path) -> None:
    """A setup.sh stand-in that prints its arguments, one per line."""
    path.write_text('#!/bin/sh\nfor a in "$@"; do printf \'%s\\n\' "$a"; done\n', encoding="utf-8")
    path.chmod(0o755)


def _enter(test: unittest.TestCase, patcher):
    """Start a patcher for one test and stop it afterwards."""
    value = patcher.start()
    test.addCleanup(patcher.stop)
    return value


class CommandLaunchers(unittest.TestCase):
    """Section 5.2: the Finder launchers are executable LF scripts that forward their arguments to setup.sh."""

    def test_command_launchers(self):
        self.assertIn("*.command text eol=lf", (ROOT / ".gitattributes").read_text(encoding="utf-8").splitlines())
        git = shutil.which("git")
        for name in ("START-HERE.command", "SETUP.command"):
            path = ROOT / name
            with self.subTest(file=name):
                self.assertEqual(0o755, stat.S_IMODE(path.stat().st_mode) & 0o755)
                self.assertNotIn(b"\r", path.read_bytes())
                if git:
                    # A tracked file must be 100755 in the index; an untracked one is checked when it is added.
                    listed = subprocess.run([git, "-C", str(ROOT), "ls-files", "-s", "--", name],
                                            capture_output=True, text=True, check=False).stdout
                    if listed:
                        self.assertTrue(listed.startswith("100755 "), listed)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp) / "Strata copy"
            root.mkdir()
            for name in ("START-HERE.command", "SETUP.command"):
                shutil.copy2(ROOT / name, root / name)
            _stub_setup_sh(root / "setup.sh")
            for name, args, expected in (("START-HERE.command", ["a"], ["a"]),
                                         ("SETUP.command", ["a"], ["--setup", "a"])):
                with self.subTest(launcher=name):
                    result = subprocess.run(["/bin/sh", str(root / name), *args], cwd=temp, capture_output=True,
                                            text=True, check=False, timeout=30)
                    self.assertEqual(0, result.returncode, result.stderr)
                    self.assertEqual(expected, result.stdout.splitlines())


class RunScripts(unittest.TestCase):
    """Section 4.10: run-<model>.sh and run-<model>.command quote every argument and start arm64 Python."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / 'Strata $x "q" `b`'
        self.root.mkdir()
        self.python = self.root / "stub python"
        _stub_setup_sh(self.python)
        self.config = self.root / 'config $HOME "x".json'

    def test_run_script_quoting(self):
        script, command = setup.write_run_script(self.root, "Q2_0", str(self.python), self.config, 8095)
        self.assertEqual("run-Q2_0.sh", script.name)
        self.assertEqual("run-Q2_0.command", command.name)
        for path in (script, command):
            with self.subTest(file=path.name):
                self.assertEqual(0o755, stat.S_IMODE(path.stat().st_mode))
                self.assertEqual(0, subprocess.run(["/bin/sh", "-n", str(path)], capture_output=True).returncode)
        self.assertIn(f"exec /usr/bin/arch -arm64 '{self.python}' ", script.read_text(encoding="utf-8"))

    @unittest.skipUnless(macos.MAC, "macOS only: the scripts start /usr/bin/arch")
    def test_run_scripts_pass_the_exact_arguments(self):
        setup.write_run_script(self.root, "Q2_0", str(self.python), self.config, 8095)
        expected = [str(self.root / "serve" / "server.py"), "--engine", "strata", "--config", str(self.config),
                    "--port", "8095", "--open"]
        result = subprocess.run(["/bin/sh", str(self.root / "run-Q2_0.sh")], capture_output=True, text=True,
                                check=False, timeout=30)
        self.assertEqual(0, result.returncode, result.stderr)
        self.assertEqual(expected, result.stdout.splitlines())
        result = subprocess.run(["/bin/sh", str(self.root / "run-Q2_0.command")], capture_output=True, text=True,
                                check=False, timeout=30)
        title = "\x1b]0;Strata Q2_0\x07"
        self.assertTrue(result.stdout.startswith(title), repr(result.stdout[:40]))
        self.assertEqual(expected, result.stdout[len(title):].splitlines())

    def test_run_script_refuses_a_model_name_that_is_not_a_file_name(self):
        for name in ("../x", "Q2_0 x", "", "a/b"):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    setup.write_run_script(self.root, name, str(self.python), self.config, 8095)
        self.assertEqual([], [p.name for p in self.root.glob("run-*")])

    @unittest.skipUnless(macos.MAC, "macOS only: uses the extended attribute")
    def test_quarantine_is_stripped_from_setup_files_only(self):
        root = self.root
        (root / "engine").mkdir()
        (root / "data").mkdir()
        files = {"START-HERE.command": True, "SETUP.command": True, "setup.sh": True,
                 str(Path("engine") / "strata"): True, str(Path("data") / "model.gguf"): False,
                 "README.md": False}
        for name in files:
            path = root / name
            path.write_bytes(b"x")
            subprocess.run(["/usr/bin/xattr", "-w", "com.apple.quarantine", "0081;00000000;Test;", str(path)],
                           check=True)
        self.assertEqual({root / "START-HERE.command", root / "SETUP.command", root / "setup.sh", root / "engine"},
                         set(setup.quarantine_targets(root)))
        setup.strip_quarantine_targets(root)
        for name, stripped in files.items():
            with self.subTest(file=name):
                self.assertEqual(not stripped, macos.has_quarantine(root / name))


class DataFolder(unittest.TestCase):
    """Section 4.12: the data folder stays out of protected and synced folders, and a new one is kept out of Spotlight
    and Time Machine."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.home = Path(temp.name) / "home"
        self.home.mkdir()
        _enter(self, mock.patch.dict(os.environ, {"HOME": str(self.home)}))
        _enter(self, mock.patch.object(setup, "MAC", True))

    def test_data_folder(self):
        for relative in ("Documents", "Desktop", "Downloads", "Library/Mobile Documents", "Library/CloudStorage"):
            with self.subTest(parent=relative):
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    folder = setup.data_folder(self.home / relative / "Strata")
                self.assertEqual(self.home / "Strata-data", folder)
                self.assertIn("ok: data folder", out.getvalue())
                self.assertIn(relative, out.getvalue())
        self.assertEqual(self.home / "src" / "Strata-data", setup.data_folder(self.home / "src" / "x"))
        kept = self.home / "Documents" / "keep-here"
        self.assertEqual(kept, setup.data_folder(self.home / "Documents" / "Strata", saved=str(kept)))
        with mock.patch.object(setup, "MAC", False):
            self.assertEqual(self.home / "Documents" / "Strata-data",
                             setup.data_folder(self.home / "Documents" / "Strata"))

    def test_create_data_folder_marks_and_excludes_a_mac_folder(self):
        folder = self.home / "Strata-data"
        out = io.StringIO()
        with mock.patch.object(macos, "tm_exclude", return_value=True) as exclude, contextlib.redirect_stdout(out):
            setup.create_data_folder(folder)
        marker = folder / ".metadata_never_index"
        self.assertTrue(marker.is_file())
        self.assertEqual(0, marker.stat().st_size)
        exclude.assert_called_once_with(folder)
        self.assertIn(f"tmutil removeexclusion {folder} undoes it", out.getvalue())

    def test_create_data_folder_gives_the_privacy_hint_on_eperm(self):
        folder = self.home / "Documents" / "Strata-data"
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError(errno.EPERM, "Operation not permitted")):
            with self.assertRaises(setup.SetupError) as context:
                setup.create_data_folder(folder)
        self.assertIn("allow Terminal in System Settings > Privacy & Security > Files and Folders",
                      str(context.exception))
        with mock.patch.object(Path, "mkdir", side_effect=PermissionError(errno.EACCES, "denied")):
            with self.assertRaises(PermissionError):
                setup.create_data_folder(folder)


class _AssetServer:
    """A local HTTP server for one folder, recording every request (no network is used)."""

    def __init__(self, folder: Path):
        self.requests: list[tuple[str, str]] = []
        requests = self.requests

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, directory=str(folder), **kwargs)

            def log_message(self, *args):
                pass

            def do_GET(self):
                requests.append(("GET", self.path))
                super().do_GET()

            def do_HEAD(self):
                requests.append(("HEAD", self.path))
                super().do_HEAD()

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}/"

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class EngineInstall(unittest.TestCase):
    """Section 4.7.1: the ready-made Metal engine, installed from a local asset that is built here."""

    GOOD = {"version": "0.1.34", "metal_rev": 1, "backend": "metal", "arch": "arm64", "macos_min": "27.0",
            "gpu_families": ["apple7", "metal4"], "metal_std": "metal4.0", "source": "prebuilt", "vision": "gpu"}

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.engine = self.dir / "engine"
        self.assets = self.dir / "assets"
        self.assets.mkdir()
        _enter(self, mock.patch.object(setup, "ENGINE_DIR", self.engine))
        _enter(self, mock.patch.object(macos, "macos_version", return_value=(27, 0, 1)))
        _enter(self, mock.patch.object(macos, "metal_devices",
                                       return_value=({"name": "Apple M5 Pro", "families": ["apple7", "apple9", "metal4"]},)))
        self.quarantine = _enter(self, mock.patch.object(macos, "strip_quarantine"))
        _enter(self, mock.patch.dict(os.environ, {"no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}))

    def asset(self, members: dict, modes=None, symlinks=()) -> None:
        """Write the release asset: `members` maps a name to its bytes; `symlinks` are (name, target) pairs."""
        with zipfile.ZipFile(self.assets / setup.PREBUILT_ASSET, "w") as archive:
            for name, data in members.items():
                info = zipfile.ZipInfo(name)
                info.external_attr = ((modes or {}).get(name, 0o644) & 0xFFFF) << 16
                archive.writestr(info, data)
            for name, target in symlinks:
                info = zipfile.ZipInfo(name)
                info.external_attr = ((stat.S_IFLNK | 0o777) & 0xFFFF) << 16
                archive.writestr(info, target)

    def members(self, **build_changes) -> dict:
        build = {**self.GOOD, **build_changes}
        return {"strata": b"#!/bin/sh\n", "strata.metallib": b"library", "strata-vision": b"vision",
                "README.txt": b"readme", "BUILD.json": json.dumps(build).encode()}

    def asset_server(self) -> "_AssetServer":
        served = _AssetServer(self.assets)
        self.addCleanup(served.close)
        return served

    def engine_names(self) -> list[str]:
        return sorted(p.name for p in self.engine.iterdir()) if self.engine.exists() else []

    def test_ready_made_engine_installs_from_the_asset(self):
        self.asset(self.members(), modes={"strata": 0o700, "strata.metallib": 0o600, "README.txt": 0o600})
        server_ = self.asset_server()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            folder = setup.get_prebuilt_metal(server_.base)
        self.assertEqual(self.engine, folder)
        self.assertEqual(0o755, stat.S_IMODE((self.engine / "strata").stat().st_mode))
        self.assertEqual(0o755, stat.S_IMODE((self.engine / "strata-vision").stat().st_mode))
        self.assertEqual(0o644, stat.S_IMODE((self.engine / "strata.metallib").stat().st_mode))
        self.assertEqual(0o644, stat.S_IMODE((self.engine / "README.txt").stat().st_mode))
        self.assertEqual(self.GOOD, json.loads((self.engine / "BUILD.json").read_text(encoding="utf-8")))
        self.assertNotIn("_unpack", self.engine_names())
        self.quarantine.assert_called_with(self.engine)
        self.assertIn("ok: ready-made engine 0.1.34 (Metal rev 1, macOS 27+)", out.getvalue())
        self.assertEqual([("HEAD", "/" + setup.PREBUILT_ASSET), ("GET", "/" + setup.PREBUILT_ASSET)], server_.requests)

    def test_installed_engine_is_reused_without_a_download(self):
        self.asset(self.members())
        server_ = self.asset_server()
        with contextlib.redirect_stdout(io.StringIO()):
            setup.get_prebuilt_metal(server_.base)
        requests = len(server_.requests)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(self.engine, setup.get_prebuilt_metal(server_.base))
        self.assertEqual(requests, len(server_.requests))
        self.assertIn("already installed", out.getvalue())
        with contextlib.redirect_stdout(io.StringIO()):
            setup.get_prebuilt_metal(server_.base, updating=True)
        self.assertGreater(len(server_.requests), requests)

    def test_engine_of_another_backend_is_removed_and_replaced(self):
        self.engine.mkdir()
        (self.engine / "BUILD.json").write_text(json.dumps({"backend": "cuda", "version": "0.1.34"}), encoding="utf-8")
        (self.engine / "strata").write_text("from a PC", encoding="utf-8")
        (self.engine / "stale.dylib").write_text("from a PC", encoding="utf-8")
        self.asset(self.members())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            folder = setup.get_prebuilt_metal(self.asset_server().base)
        self.assertEqual(self.engine, folder)
        self.assertIn("removed the engine", out.getvalue())
        self.assertFalse((self.engine / "stale.dylib").exists())
        self.assertEqual("#!/bin/sh\n", (self.engine / "strata").read_text(encoding="utf-8"))

    def test_an_engine_this_mac_cannot_run_is_not_installed(self):
        cases = {"older version": {"version": "0.1.33"}, "older Metal rev": {"metal_rev": 0},
                 "x86_64 build": {"arch": "x86_64"}, "newer macOS": {"macos_min": "28.0"},
                 "GPU family missing": {"gpu_families": ["apple10", "metal4"]},
                 "another backend": {"backend": "cuda"}}
        server_ = self.asset_server()
        for label, changes in cases.items():
            with self.subTest(case=label):
                self.asset(self.members(**changes))
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    self.assertIsNone(setup.get_prebuilt_metal(server_.base))
                self.assertIn("compiling instead", out.getvalue())
                self.assertEqual([], self.engine_names())

    def test_missing_asset_means_compile_instead(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertIsNone(setup.get_prebuilt_metal(self.asset_server().base))
        self.assertIn("no ready-made engine", out.getvalue())
        self.assertIn("compiling instead", out.getvalue())

    def test_a_symbolic_link_in_the_asset_fails_before_anything_is_written(self):
        self.asset(self.members(), symlinks=[("strata.metallib", "/etc/passwd")])
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(setup.SetupError) as context:
                setup.get_prebuilt_metal(self.asset_server().base)
        self.assertIn("symbolic link", str(context.exception))
        self.assertEqual([], self.engine_names())

    def test_a_certificate_failure_gets_the_python_org_hint(self):
        error = urllib.error.URLError(ssl.SSLCertVerificationError(
            1, "[SSL: CERTIFICATE_VERIFY_FAILED] certificate verify failed"))
        _enter(self, mock.patch.object(urllib.request, "urlopen", side_effect=error))
        with self.assertRaises(setup.SetupError) as context:
            setup.get_prebuilt_metal("https://example.invalid/")
        self.assertIn('run "Install Certificates.command" in /Applications/Python 3.x', str(context.exception))

    def test_prebuilt_bases_order_and_overrides(self):
        env = _enter(self, mock.patch.dict(os.environ))
        env.pop("STRATA_PREBUILT_URL", None)
        self.assertEqual([], setup.prebuilt_bases())
        _enter(self, mock.patch.object(setup, "RELEASE_REPOSITORY", "example/strata"))
        self.assertEqual(["https://github.com/example/strata/releases/download/v0.1.34-metal.1/",
                          "https://github.com/example/strata/releases/latest/download/"], setup.prebuilt_bases())
        os.environ["STRATA_PREBUILT_URL"] = "http://127.0.0.1:9"
        self.assertEqual(["http://127.0.0.1:9/"], setup.prebuilt_bases())
        local = self.dir / "release folder"
        local.mkdir()
        self.assertEqual([local.resolve().as_uri() + "/"], setup.prebuilt_bases(str(local)))


class EngineSmokeTest(unittest.TestCase):
    """Section 4.7.5: `strata --version` and `strata --list-devices` run for real, through fake engines (shell
    scripts), and the smoke test reads their output."""

    GOOD = ('case "$1" in\n'
            '--version) echo "strata 0.1.34 metal 1" ;;\n'
            '--list-devices) printf "device_name=Apple M5 Pro\\nfamilies=apple7,apple9,metal4\\n" ;;\n'
            'esac\n')

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.engine = Path(temp.name) / "engine"
        self.engine.mkdir()
        self.quarantined = _enter(self, mock.patch.object(macos, "has_quarantine", return_value=False))

    def fake(self, body: str) -> None:
        exe = self.engine / "strata"
        exe.write_text("#!/bin/sh\n" + body, encoding="utf-8")
        exe.chmod(0o755)

    def smoke(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return setup.smoke_test(self.engine)

    def test_smoke_test_passes_on_a_good_engine(self):
        self.fake(self.GOOD)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            result = setup.smoke_test(self.engine)
        self.assertEqual({"version": "0.1.34", "metal_rev": 1, "families": ["apple7", "apple9", "metal4"]}, result)
        self.assertIn("ok: engine 0.1.34 (Metal rev 1) runs", out.getvalue())

    def test_a_sigkilled_engine_gets_the_quarantine_fix_or_the_signature_check(self):
        self.fake("kill -9 $$\n")
        self.quarantined.return_value = True
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("macOS stopped the engine before it started", str(context.exception))
        self.assertIn("xattr -dr com.apple.quarantine engine", str(context.exception))
        self.quarantined.return_value = False
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("codesign -dv", str(context.exception))
        self.assertIn("delete the engine folder and run setup again", str(context.exception))

    def test_a_library_that_is_not_on_this_mac_is_a_build_bug(self):
        self.fake('echo "dyld: Library not loaded: @rpath/libggml.dylib" >&2\nexit 1\n')
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("not on this Mac", str(context.exception))

    def test_a_version_line_without_the_metal_rev_fails(self):
        self.fake('echo "strata 0.1.34"\n')
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("--version", str(context.exception))

    def test_a_device_without_metal4_or_two_devices_fail(self):
        self.fake(self.GOOD.replace("apple7,apple9,metal4", "apple7,apple9"))
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("apple7 and metal4", str(context.exception))
        self.fake(self.GOOD.replace("device_name=Apple M5 Pro\\n", "device_name=A\\ndevice_name=B\\n"))
        with self.assertRaises(setup.SetupError):
            self.smoke()

    def test_a_slow_engine_times_out(self):
        self.fake("exec sleep 5\n")
        _enter(self, mock.patch.object(setup, "SMOKE_TIMEOUT_S", 1))
        with self.assertRaises(setup.SetupError) as context:
            self.smoke()
        self.assertIn("did not answer", str(context.exception))


class SslSetup(unittest.TestCase):
    def test_ssl_setup(self):
        _enter(self, mock.patch.object(macos, "MAC", True))
        _enter(self, mock.patch.dict(os.environ))
        os.environ.pop("SSL_CERT_FILE", None)
        with mock.patch.object(ssl, "get_default_verify_paths", return_value=mock.Mock(cafile=None)):
            path = macos.ssl_setup()
            self.assertIsNotNone(path)
            self.assertTrue(os.path.isfile(path))
            self.assertEqual(path, os.environ["SSL_CERT_FILE"])


@unittest.skipUnless(sys.platform == "darwin", "macOS only: the BSD bind rule is what the probe corrects")
class PortProbe(unittest.TestCase):
    """Section 6.1: a port with a listener is in use, even where a SO_REUSEADDR bind on macOS succeeds."""

    def free_port(self) -> int:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            return probe.getsockname()[1]

    def listen(self, host: str, port: int) -> socket.socket:
        listener = socket.socket()
        self.addCleanup(listener.close)
        listener.bind((host, port))
        listener.listen()
        return listener

    def test_port_probe(self):
        port = self.free_port()
        self.assertFalse(server.port_in_use("127.0.0.1", port))
        self.assertFalse(server.port_in_use("0.0.0.0", port))

        wildcard = self.listen("0.0.0.0", port)
        # The probe's own bind succeeds beside the wildcard listener on macOS, so only the connect can say "in use".
        server.Server(("127.0.0.1", port), BaseHTTPRequestHandler).server_close()
        self.assertTrue(server.port_in_use("127.0.0.1", port))
        wildcard.close()

        self.listen("127.0.0.1", port)
        self.assertTrue(server.port_in_use("0.0.0.0", port))
        self.assertTrue(server.port_in_use("127.0.0.1", port))

    def test_port_probe_leaves_a_free_port_free(self):
        port = self.free_port()
        self.assertFalse(server.port_in_use("127.0.0.1", port))


@unittest.skipUnless(sys.platform == "darwin", "macOS only")
class SighupShutdown(unittest.TestCase):
    """Section 6.2: a SIGHUP (the Terminal window closed) shuts the mock server down cleanly, as Ctrl+C does."""

    def test_sighup_shutdown(self):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "mock",
                                 "--host", "127.0.0.1", "--port", str(port)], cwd=ROOT, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        deadline = time.monotonic() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    self.fail("the mock server did not start")
                time.sleep(0.2)
        os.kill(proc.pid, signal.SIGHUP)
        self.assertEqual(0, proc.wait(timeout=10))
        self.assertFalse(server.port_in_use("127.0.0.1", port))


FAKE_ENGINE = """\
import sys
print("INFO engine=0.1.34", flush=True)
print("READY 8192", flush=True)
for _ in sys.stdin:
    pass
"""


class EngineDeath(unittest.TestCase):
    """Section 6.5: the diagnostic for a dead engine comes from its own log output since it started. A fake engine
    (a Python script behind a shell wrapper, as `--serve` would be) is started and killed for real."""

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        script = self.dir / "fake_engine.py"
        script.write_text(FAKE_ENGINE, encoding="utf-8")
        self.wrapper = self.dir / "strata"
        self.wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} "{script}" "$@"\n', encoding="utf-8")
        self.wrapper.chmod(0o755)
        self.log = self.dir / "strata.log"

    def start(self):
        engine = server.StrataEngine(str(self.wrapper), ["--max-context", "8192"], log=str(self.log))
        self.addCleanup(lambda: engine.proc.poll() is None and engine.proc.kill())
        return engine

    def test_death_note_for_sigkill_is_the_same_with_unchanged_or_grown_log(self):
        notes = []
        for grown in (False, True):
            engine = self.start()
            if grown:
                with self.log.open("a", encoding="utf-8") as log:
                    log.write("decoding: 12 tokens so far\n")
            os.kill(engine.proc.pid, signal.SIGKILL)
            engine.proc.wait(timeout=10)
            notes.append(engine.death_note())
        self.assertEqual(notes[0], notes[1])
        self.assertIn("received SIGKILL", notes[0])
        self.assertIn("check Console and system memory pressure", notes[0])

    def test_death_note_names_other_signals(self):
        engine = self.start()
        os.kill(engine.proc.pid, signal.SIGSEGV)
        engine.proc.wait(timeout=10)
        self.assertIn("The engine ended on signal SIGSEGV (11).", engine.death_note())

    def test_death_note_reports_a_watchdog_line(self):
        engine = self.start()
        with self.log.open("a", encoding="utf-8") as log:
            log.write("watchdog: no token for 120 s\n")
        os.kill(engine.proc.pid, signal.SIGKILL)
        engine.proc.wait(timeout=10)
        note = engine.death_note()
        self.assertIn("stopped making progress", note)
        self.assertIn("watchdog: no token for 120 s", note)

    def test_death_note_ignores_earlier_runs_in_the_log(self):
        self.log.write_text("ERR the earlier run failed\n", encoding="utf-8")
        engine = self.start()
        engine.proc.stdin.close()
        engine.proc.wait(timeout=10)
        self.assertNotIn("earlier run", engine.death_note())
        self.assertIn("ended unexpectedly", engine.death_note())
        engine2 = self.start()
        with self.log.open("a", encoding="utf-8") as log:
            log.write("strata: finished the request\n")
        engine2.proc.stdin.close()
        engine2.proc.wait(timeout=10)
        self.assertIn("Its last log line: strata: finished the request", engine2.death_note())


def _caffeinate_under(pid: int) -> list[int]:
    """The caffeinate processes whose parent is `pid` (the server keeps the Mac awake through its own child)."""
    found = subprocess.run(["/usr/bin/pgrep", "-P", str(pid), "-x", "caffeinate"], capture_output=True, text=True,
                           check=False).stdout
    return [int(p) for p in found.split()]


def _wait_for_caffeinate(pid: int) -> list[int]:
    """The caffeinate starts just after the port opens, so a positive case waits for it."""
    deadline = time.monotonic() + 15
    while not _caffeinate_under(pid) and time.monotonic() < deadline:
        time.sleep(0.2)
    return _caffeinate_under(pid)


@unittest.skipUnless(sys.platform == "darwin", "macOS only: caffeinate")
class KeepAwakeFromTheServer(unittest.TestCase):
    """Section 6.9 end to end: the config's keep_awake key and the --keep-awake flag start one caffeinate under the
    server process (the mock engine, so no model is needed)."""

    def start(self, extra_config: dict, flags=()) -> subprocess.Popen:
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        config = Path(temp.name) / "config.json"
        config.write_text(json.dumps(extra_config), encoding="utf-8")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "mock",
                                 "--config", str(config), "--host", "127.0.0.1", "--port", str(port), *flags],
                                cwd=ROOT, start_new_session=True, stdout=subprocess.DEVNULL,
                                stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.addCleanup(lambda: proc.poll() is None and os.kill(proc.pid, signal.SIGTERM))
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                return proc
            except OSError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    self.fail("the server did not start")
                time.sleep(0.2)

    def test_keep_awake_from_the_config_and_the_flag(self):
        proc = self.start({"keep_awake": True})
        self.assertEqual(1, len(_wait_for_caffeinate(proc.pid)))
        proc.terminate()
        self.assertEqual(0, proc.wait(timeout=30))
        proc = self.start({})
        self.assertEqual([], _caffeinate_under(proc.pid))
        proc.terminate()
        proc.wait(timeout=30)
        proc = self.start({}, flags=("--keep-awake",))
        self.assertEqual(1, len(_wait_for_caffeinate(proc.pid)))
        proc.terminate()
        proc.wait(timeout=30)


@unittest.skipUnless(sys.platform == "darwin", "macOS only: caffeinate")
class KeepAwakeLazyStart(unittest.TestCase):
    """Section 6.9: a lazy server keeps the Mac awake only once its model is loaded, not before the first load."""

    def test_lazy_start_waits_for_the_first_load(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        folder = Path(temp.name)
        tokenizer = folder / "tokenizer"
        tokenizer.mkdir()
        # Every character of the special tokens is in the vocabulary (the server encodes them at start), plus one merge.
        chars = sorted(set("<|im_start|><|im_end|><|endoftext|>" + string.ascii_letters + string.digits
                           + " \n.,:;!?'\"()-_/"))
        vocab = {ch: index for index, ch in enumerate(chars)}
        vocab["ab"] = len(vocab)
        (tokenizer / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (tokenizer / "merges.txt").write_text("a b", encoding="utf-8")   # one merge, and no empty line after it
        (tokenizer / "token_type.json").write_text("[]", encoding="utf-8")
        script = folder / "fake_engine.py"
        script.write_text(FAKE_ENGINE, encoding="utf-8")
        wrapper = folder / "strata"
        wrapper.write_text(f'#!/bin/sh\nexec {sys.executable} "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
        config = folder / "config.json"
        config.write_text(json.dumps({"exe": str(wrapper), "args": ["--max-context", "8192"],
                                      "log": str(folder / "strata.log"), "cwd": str(folder), "model_name": "fake",
                                      "keep_awake": True, "lazy_load": True}), encoding="utf-8")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        proc = subprocess.Popen([sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "strata",
                                 "--config", str(config), "--tokenizer", str(tokenizer), "--host", "127.0.0.1",
                                 "--port", str(port)], cwd=ROOT, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.addCleanup(lambda: proc.poll() is None and proc.kill())
        self.addCleanup(lambda: proc.poll() is None and os.kill(proc.pid, signal.SIGTERM))
        deadline = time.monotonic() + 60
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                if time.monotonic() > deadline or proc.poll() is not None:
                    self.fail("the lazy server did not start")
                time.sleep(0.2)
        time.sleep(1)
        self.assertEqual([], _caffeinate_under(proc.pid))
        request = urllib.request.Request(f"http://127.0.0.1:{port}/load", data=b"", method="POST")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.assertEqual(200, response.status)
        self.assertEqual(1, len(_wait_for_caffeinate(proc.pid)))


@unittest.skipUnless(sys.platform == "darwin", "macOS only: caffeinate")
class KeepAwake(unittest.TestCase):
    """Section 6.9: with keep_awake, one caffeinate runs while the model is loaded; an unload ends it."""

    class StubEngine:
        def __init__(self):
            self.up = False

        def alive(self):
            return self.up

        def restart(self):
            self.up = True

        def unload(self):
            self.up = False

    def caffeinate_pids(self) -> list[int]:
        found = subprocess.run(["/usr/bin/pgrep", "-P", str(os.getpid()), "-x", "caffeinate"], capture_output=True,
                               text=True, check=False).stdout
        return [int(pid) for pid in found.split()]

    def test_keep_awake(self):
        svc = server.Service(self.StubEngine(), server.ByteTokenizer(), None)
        self.addCleanup(svc.unload)
        svc.keep_awake = False
        svc.ensure_loaded()
        self.assertEqual([], self.caffeinate_pids())
        svc.unload()

        svc.keep_awake = True
        svc.ensure_loaded()
        pids = self.caffeinate_pids()
        self.assertEqual(1, len(pids))
        command = subprocess.run(["/bin/ps", "-o", "args=", "-p", str(pids[0])], capture_output=True,
                                 text=True, check=False).stdout
        self.assertIn(f"-i -w {os.getpid()}", command)
        svc.unload()
        self.assertEqual([], self.caffeinate_pids())
        svc.ensure_loaded()
        again = self.caffeinate_pids()
        self.assertEqual(1, len(again))
        self.assertNotEqual(pids, again)
        svc.unload()


class TelemetryMac(unittest.TestCase):
    """Section 7: the Mac's GPU reader and the sampler's extra readings, with the macOS probes faked."""

    DEVICE = {"name": "Apple M5 Pro", "working_set": 51 << 30, "cores": 20, "families": ["apple7", "metal4"]}

    def make(self, mac: bool, devices=(), **probes):
        _enter(self, mock.patch.object(telemetry, "MAC", mac))
        _enter(self, mock.patch.object(telemetry.threading, "Thread"))
        _enter(self, mock.patch.object(macos, "metal_devices", return_value=tuple(devices)))
        for name, value in probes.items():
            _enter(self, mock.patch.object(macos, name, return_value=value))
        return telemetry.Telemetry(backend="metal")

    def test_apple_reader_maps_the_ioregistry_counters(self):
        _enter(self, mock.patch.object(telemetry, "MAC", True))
        _enter(self, mock.patch.object(macos, "metal_devices", return_value=(self.DEVICE,)))
        _enter(self, mock.patch.object(macos, "ioreg_gpu",
                                       return_value={"util": 42, "in_use": 300, "recovery": 0}))
        reader = telemetry._Apple(backend="metal")
        self.assertEqual("Apple M5 Pro", reader.name())
        reading = reader.read()
        self.assertEqual(42, reading["util"])
        self.assertEqual(300, reading["mem_used"])
        self.assertEqual(51 << 30, reading["mem_total"])
        self.assertEqual(0, reading["recovery"])
        for key in ("temp", "power", "power_limit", "pcie_gen", "pcie_width"):
            self.assertIsNone(reading[key], key)

    def test_apple_reader_asks_ioregistry_at_most_every_two_seconds(self):
        _enter(self, mock.patch.object(telemetry, "MAC", True))
        _enter(self, mock.patch.object(macos, "metal_devices", return_value=(self.DEVICE,)))
        ioreg = _enter(self, mock.patch.object(macos, "ioreg_gpu", return_value={"util": 1}))
        reader = telemetry._Apple(backend="metal")
        reader.read()
        reader.read()
        self.assertEqual(1, ioreg.call_count)

    def test_sample_adds_the_four_mac_readings_only_on_a_mac(self):
        mac = self.make(True, thermal_state="fair", pressure_level="warn", swap_used=4096)
        sample = mac.sample()
        self.assertEqual("fair", sample["thermal"])
        self.assertEqual(1, sample["thermal_level"])
        self.assertEqual("warn", sample["mem_pressure"])
        self.assertEqual(4096, sample["swap_used"])
        other = self.make(False, thermal_state="fair", pressure_level="warn", swap_used=4096)
        sample = other.sample()
        for key in ("thermal", "thermal_level", "mem_pressure", "swap_used"):
            self.assertNotIn(key, sample)

    def test_static_gains_the_mac_facts_on_a_mac(self):
        perf = [{"name": "Super", "cores": 6}, {"name": "Performance", "cores": 12}]
        mac = self.make(True, devices=(self.DEVICE,), perf_levels=perf, cpu_brand="Apple M5 Pro")
        static = mac.static
        self.assertIs(True, static["unified"])
        self.assertEqual(20, static["gpu_cores"])
        self.assertEqual("Mac", static["computer"])
        self.assertEqual(perf, static["perflevels"])
        self.assertEqual("Apple M5 Pro", static["cpu_name"])
        self.assertNotIn("computer", self.make(False).static)

    def test_cpu_name_is_the_brand_and_never_arm_on_a_mac(self):
        _enter(self, mock.patch.object(telemetry, "MAC", True))
        _enter(self, mock.patch.object(macos, "cpu_brand", return_value=""))
        _enter(self, mock.patch.object(telemetry.platform, "processor", return_value="arm"))
        self.assertIsNone(telemetry._cpu_name())
        _enter(self, mock.patch.object(telemetry, "MAC", False))
        self.assertEqual("arm", telemetry._cpu_name())


if __name__ == "__main__":
    unittest.main()
