"""tools/test_strata_mcp.py - tests for tools/strata_mcp.py (Strata's MCP server), standard library only.

    python -m unittest tools/test_strata_mcp.py -v

Setup and the model server are fakes (a temporary Strata folder with a model-catalog shim and serve/server.py), so
nothing is downloaded and no GPU is used. The tests cover the JSON-RPC handshake, argument checks, an informational install
preview that cannot launch the unfinished Mac installer, start/stop with real background processes, and fake-server status.
"""
from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import strata_mcp as M  # noqa: E402

FAKE_SETUP = "from tools.model_catalog import CONTEXTS, FAMILIES, MODELS\n"

FAKE_SERVER = r'''
import json, os, signal, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
a = sys.argv
port = int(a[a.index("--port") + 1])
cfg = json.loads(Path(a[a.index("--config") + 1]).read_text())
root = Path(__file__).resolve().parent.parent
print("loading the model (fake) ...", flush=True)
if cfg.get("crash"):
    print("engine failed: out of memory (fake)", flush=True)
    sys.exit(3)
time.sleep(cfg.get("delay", 0))
state = {"loaded": True}
class H(BaseHTTPRequestHandler):
    def log_message(self, *x):
        pass
    def js(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)
    def do_GET(self):
        p = self.path.split("?")[0]
        if p == "/health":
            self.js(200, {"status": "ok", "service": "strata", "model": cfg["model_name"], "max_context": 32768,
                          "images": False, "api_key": False, "loaded": state["loaded"]})
        elif p == "/status":
            self.js(200, {"busy": bool(cfg.get("busy")), "queued": 0})
        elif p == "/v1/status":
            self.js(200, {"service": "strata", "engine": "0.1.32", "uptime_s": 5, "activity": {"requests": 0},
                          "machine": {"gpu": {"name": "Fake GPU", "used_mib": 11000, "total_mib": 12227},
                                      "ram": {"used_gib": 40.0, "total_gib": 63.1}}})
        else:
            self.js(404, {"error": {"message": "not found"}})
    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        p = self.path.split("?")[0]
        if p == "/unload":
            state["loaded"] = False
            with open(root / "unloaded.txt", "a") as f:
                f.write(str(port) + "\n")
            self.js(200, {"status": "unloaded"})
        elif p == "/v1/chat/completions":
            with open(root / "bench-request.json", "w") as f:
                json.dump(body, f)
            self.js(200, {"model": cfg["model_name"], "choices": [{"message": {"content": "x"}, "finish_reason": "length"}],
                          "usage": {"prompt_tokens": 30, "completion_tokens": body.get("max_tokens", 1)},
                          "timings": {"predicted_per_second": 61.5, "prompt_per_second": 400.0}})
        else:
            self.js(404, {})
httpd = ThreadingHTTPServer(("127.0.0.1", port), H)
def term(*_):
    print("[strata] stopped (fake)", flush=True)
    os._exit(0)
signal.signal(signal.SIGTERM, term)
print(f"ready: http://127.0.0.1:{port}/v1", flush=True)
httpd.serve_forever()
'''


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def wait_for(cond, seconds=30.0):
    end = time.time() + seconds
    while time.time() < end:
        if cond():
            return True
        time.sleep(0.1)
    return cond()


FAKE_HW = {"os": "macOS 27", "ram_gb": 64.0,
           "cpu": {"name": "Apple silicon", "cores": 12},
           "metal": {"name": "Apple GPU", "working_set_gib": 51.8, "usable": True}}


class FakeRoot(unittest.TestCase):
    """A temporary Strata folder: fake setup.py and serve/server.py, its own settings file (data folder)."""

    def setUp(self):
        self.original_proc_identity = M.proc_identity
        self.platform_patch = mock.patch.object(M.sys, "platform", "darwin")
        self.platform_patch.start()
        self.identity_patch = mock.patch.object(
            M,
            "proc_identity",
            side_effect=lambda pid: (
                M.MACOS.proc_identity(pid) if M.MACOS is not None else self.original_proc_identity(pid)
            ),
        )
        self.identity_patch.start()
        temporary_parent = "/tmp" if os.path.isdir("/tmp") else None
        self.tmp = tempfile.TemporaryDirectory(dir=temporary_parent)
        base = Path(self.tmp.name)
        self.root = base / "Strata"
        (self.root / "serve").mkdir(parents=True)
        (self.root / "setup.py").write_text(FAKE_SETUP)
        (self.root / "serve" / "server.py").write_text(FAKE_SERVER)
        self.strata = M.Strata(self.root)
        self.strata.settings_path = lambda: base / "settings.json"
        self.strata.python_override = sys.executable
        self.strata.health_timeout = 1.0
        self.strata._hardware = lambda: dict(FAKE_HW)
        self.strata.disk_free = lambda p: 1000.0
        self.tools = M.Tools(self.strata)
        self.extra = []

    def tearDown(self):
        for p in self.extra:
            if p.poll() is None:
                p.kill()
                p.wait(10)
        srv = self.strata.state("server")
        if srv and M.proc_alive(srv.get("pid"), srv.get("ident")):
            M.proc_kill_tree(srv["pid"], srv["ident"])
            M.wait_gone(srv["pid"], srv["ident"], 10)
        job = self.strata.state("install")
        if job and M.proc_alive(job.get("pid"), job.get("ident")):
            M.proc_kill_tree(job["pid"], job["ident"])
            M.wait_gone(job["pid"], job["ident"], 10)
        for p in self.strata.children:
            try:
                p.wait(5)
            except subprocess.TimeoutExpired:
                pass
        self.identity_patch.stop()
        self.platform_patch.stop()
        for _ in range(20):                             # a child may still be closing its log
            try:
                self.tmp.cleanup()
                break
            except OSError:
                time.sleep(0.25)

    def call(self, name, args=None):
        """A tools/call through the JSON-RPC layer: (result dict, isError)."""
        srv = M.McpServer(self.strata, io.BytesIO())
        r = srv.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                        "params": {"name": name, "arguments": args or {}}})
        res = r["result"]
        return json.loads(res["content"][0]["text"]), res["isError"]

    def write_config(self, tag="q2_0", port=8080, **extra):
        cfg = {"exe": sys.executable, "args": ["--max-context", "32768"], "port": port,
               "model_name": f"qwen3.8-flash-next-{tag}", "log": str(self.root / f"strata-{tag}.log"), **extra}
        (self.root / f"strata-{tag}.json").write_text(json.dumps(cfg))
        (self.root / f"strata-{tag}.log").write_text("engine line 1\nengine line 2\n")
        return cfg


# ------------------------------------------------------------------------------------------------ the protocol
class Protocol(FakeRoot):
    def run_lines(self, msgs):
        inp = io.BytesIO(("\n".join(m if isinstance(m, str) else json.dumps(m) for m in msgs) + "\n").encode())
        out = io.BytesIO()
        M.McpServer(self.strata, out).serve(inp)
        return {m.get("id"): m for m in (json.loads(x) for x in out.getvalue().decode().splitlines())}

    def test_handshake_list_call(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize",
             "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "t", "version": "1"}}},
            {"jsonrpc": "2.0", "method": "notifications/initialized"},
            {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            {"jsonrpc": "2.0", "id": 3, "method": "tools/call", "params": {"name": "strata_models", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 4, "method": "ping"},
        ])
        init = r[1]["result"]
        self.assertEqual(init["protocolVersion"], "2025-06-18")
        self.assertIn("tools", init["capabilities"])
        self.assertEqual(init["serverInfo"]["name"], "strata")
        names = [t["name"] for t in r[2]["result"]["tools"]]
        self.assertEqual(names, ["strata_status", "strata_models", "strata_install", "strata_start", "strata_stop",
                                 "strata_logs", "strata_benchmark", "strata_connect_info"])
        for t in r[2]["result"]["tools"]:
            self.assertEqual(t["inputSchema"]["type"], "object")
            self.assertFalse(t["inputSchema"]["additionalProperties"])
            self.assertGreater(len(t["description"]), 60)
        call = r[3]["result"]
        self.assertFalse(call["isError"])
        self.assertIn("families", call["structuredContent"])
        recommendation = json.loads(call["content"][0]["text"])["recommendation"]
        self.assertIsNone(recommendation["model"])
        self.assertIn("Metal engine", recommendation["why"])
        self.assertEqual(r[4]["result"], {})
        self.assertNotIn(None, r)                       # the notification got no answer

    def test_older_protocol_and_errors(self):
        r = self.run_lines([
            {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2024-11-05"}},
            "{not json",
            {"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "rm_rf", "arguments": {}}},
            {"jsonrpc": "2.0", "id": 3, "method": "no/such"},
            {"jsonrpc": "2.0", "id": 4, "method": "tools/call", "params": {"name": "strata_logs", "arguments": []}},
        ])
        self.assertEqual(r[1]["result"]["protocolVersion"], "2024-11-05")
        self.assertEqual(r[None]["error"]["code"], -32700)
        self.assertEqual(r[2]["error"]["code"], -32602)
        self.assertEqual(r[3]["error"]["code"], -32601)
        self.assertTrue(r[4]["result"]["isError"])
        self.assertNotIn("structuredContent", r[4]["result"])   # 2024-11-05 has no structured results

    def test_real_stdio_process(self):
        """The script itself over a pipe: stdout carries only JSON-RPC lines."""
        msgs = [{"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-06-18"}},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
                {"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                 "params": {"name": "strata_logs", "arguments": {"source": "setup"}}}]
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", str(self.root)],
                           input="\n".join(json.dumps(m) for m in msgs) + "\n", capture_output=True, text=True,
                           timeout=60, env=dict(os.environ))
        lines = [json.loads(x) for x in p.stdout.splitlines()]
        self.assertEqual(sorted(m["id"] for m in lines), [1, 2, 3])
        self.assertEqual(len(next(m for m in lines if m["id"] == 2)["result"]["tools"]), 8)
        self.assertIn("ready", p.stderr)

    def test_not_a_strata_folder(self):
        p = subprocess.run([sys.executable, str(HERE / "strata_mcp.py"), "--root", self.tmp.name], input="",
                           capture_output=True, text=True, timeout=60)
        self.assertEqual(p.returncode, 2)


class MacOnlyEntrypoint(unittest.TestCase):
    def test_mcp_refuses_linux(self):
        with (mock.patch.object(M.sys, "platform", "linux"),
              mock.patch("sys.stderr", new_callable=io.StringIO)):
            self.assertEqual(M.main([]), 2)

    def test_host_gate_rejects_intel_and_old_macos(self):
        class FakeMac:
            MAC = True

            @staticmethod
            def translated():
                return False

            @staticmethod
            def sysctl_int(_name, default=0):
                return 1

            @staticmethod
            def macos_version():
                return (26, 0)

        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", FakeMac),
              mock.patch.object(M.platform, "machine", return_value="arm64")):
            self.assertIn("macOS 27", M.macos_host_error())
            with mock.patch.object(M.platform, "machine", return_value="x86_64"):
                self.assertIn("native Apple-silicon", M.macos_host_error())


# ------------------------------------------------------------------------------------------------ validation
class Validation(FakeRoot):
    def assertRejected(self, tool, args, text):
        res, err = self.call(tool, args)
        self.assertTrue(err, f"{tool} {args} was accepted: {res}")
        self.assertIn(text, res["error"])

    def test_install_arguments(self):
        with mock.patch.object(M.sys, "platform", "darwin"):
            self.assertRejected("strata_install", {"model": "Q9_X"}, "model must be one of")
            self.assertRejected("strata_install", {"family": "llama"}, "family must be one of")
            self.assertRejected("strata_install", {"context": 12345}, "context must be one of")
            self.assertRejected("strata_install", {"gpus": "0;rm -rf /"}, "unknown argument")
            self.assertRejected("strata_install", {"gpus": "0 && calc"}, "unknown argument")
            self.assertRejected("strata_install", {"gpu": -1}, "unknown argument")
            self.assertRejected("strata_install", {"port": 80}, "between")
            self.assertRejected("strata_install", {"backend": "rocm"}, "backend must be one of")
            self.assertRejected("strata_install", {"command": "setup.py --build"}, "unknown argument")
            self.assertRejected("strata_install", {"family": "swift", "model": "IQ3_S"}, "has no IQ3_S")
            self.assertRejected("strata_install", {"family": "unsloth", "vision": "yes"},
                                "family must be one of")
            self.assertRejected("strata_install", {"vision": "yes\nrm"}, "control characters")

    def test_data_dir_paths(self):
        self.assertRejected("strata_install", {"data_dir": "models"}, "absolute")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "x" / ".." / "Strata-x")}, "'..'")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "Documents")}, "named Strata")
        self.assertRejected("strata_install", {"data_dir": str(self.root.parent / "nope" / "Strata-data")},
                            "does not exist")
        self.assertRejected("strata_install", {"data_dir": "\\\\server\\share\\Strata-data"}, "network")
        sysdir = "/etc/Strata-data"
        self.assertRejected("strata_install", {"data_dir": sysdir}, "system folder")
        ok = [self.root.parent / "Strata-data", self.root / "models-here", self.root.parent / "Strata-big"]
        for d in ok:
            self.assertEqual(M.norm(self.tools.check_data_dir(str(d))), M.norm(d))

    def test_other_tools_arguments(self):
        self.assertRejected("strata_start", {"model": "../../etc/passwd"}, "not valid")
        self.assertRejected("strata_start", {"model": "q2_0"}, "no model is installed")
        self.write_config("q2_0")
        self.assertRejected("strata_start", {"model": "iq3_s"}, "no installed model 'iq3_s'")
        self.assertRejected("strata_start", {"wait_seconds": 100000}, "between")
        self.assertRejected("strata_logs", {"lines": 0}, "between")
        self.assertRejected("strata_logs", {"source": "/etc/passwd"}, "source must be one of")
        self.assertRejected("strata_benchmark", {"max_tokens": 100000}, "between")
        self.assertRejected("strata_stop", {"force": "yes"}, "true or false")

    def test_engine_log_outside_is_not_read(self):
        self.write_config("q2_0", log=str(Path(self.tmp.name).parent / "elsewhere.log"))
        res, err = self.call("strata_logs", {"source": "engine", "model": "q2_0"})
        self.assertTrue(err)
        self.assertIn("outside the Strata folder", res["error"])
        self.write_config("q2_0")
        res, err = self.call("strata_logs", {"source": "engine"})
        self.assertFalse(err)
        self.assertEqual(res["lines"], ["engine line 1", "engine line 2"])


# ------------------------------------------------------------------------------------------------ install
class Install(FakeRoot):
    def test_install_plan_is_informational_and_confirm_fails_closed(self):
        plan, err = self.call("strata_install", {})
        self.assertFalse(err)
        self.assertIn("informational preview", plan["summary"].lower())
        self.assertEqual((plan["plan"]["family"], plan["plan"]["model"], plan["plan"]["context"]),
                         ("qwen", "Q2_0", 32768))
        self.assertFalse((self.root / "setup-args.json").exists())

        result, err = self.call("strata_install", {"model": "Q2_0", "context": 65536, "vision": "no",
                                                     "confirm": True})
        self.assertTrue(err)
        self.assertIn("macOS installer is not implemented", result["error"])
        self.assertFalse((self.root / "setup-args.json").exists())

    def test_confirmed_install_refuses_even_when_disk_is_short(self):
        self.strata.disk_free = lambda _path: 10.0
        plan, err = self.call("strata_install", {"model": "Q2_0"})
        self.assertFalse(err)
        self.assertIsNotNone(plan["plan"]["disk_short"])
        result, err = self.call("strata_install", {"model": "Q2_0", "confirm": True})
        self.assertTrue(err)
        self.assertIn("macOS installer is not implemented", result["error"])


# ------------------------------------------------------------------------------------------------ start / stop / status
class StartStop(FakeRoot):
    def test_start_status_bench_stop(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=1.5)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertFalse(err, res)
        self.assertIn("Strata is running", res["summary"])
        pid = res["server"]["pid"]

        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn(f"running on port {port}", st["summary"])
        srv = st["running"][0]
        self.assertEqual(srv["engine_version"], "0.1.32")
        self.assertEqual(srv["gpu"]["total_mib"], 12227)
        self.assertIn("this MCP server", srv["started_by"])

        res, err = self.call("strata_start", {})            # already running: nothing new
        self.assertIn("already running", res["summary"])

        b, err = self.call("strata_benchmark", {"max_tokens": 64})
        self.assertFalse(err, b)
        self.assertEqual(b["output_tok_s"], 61.5)
        sent = json.loads((self.root / "bench-request.json").read_text())
        self.assertEqual((sent["temperature"], sent["max_tokens"], sent["reasoning_effort"]), (0, 64, "none"))

        c, _ = self.call("strata_connect_info", {})
        self.assertEqual(c["openai_base_url"], f"http://127.0.0.1:{port}/v1")

        logs, _ = self.call("strata_logs", {"source": "server"})
        self.assertTrue(any(line.startswith("ready:") for line in logs["lines"]))

        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("Strata server stopped", res["summary"])
        self.assertEqual(res["engine_unloaded"], "unloaded")
        self.assertEqual((self.root / "unloaded.txt").read_text().split(), [str(port)])
        ident = M.proc_identity(pid)
        self.assertIsNone(ident)
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertEqual(st["running"], [])
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")

    def test_start_reports_a_crash(self):
        self.write_config("q2_0", port=free_port(), crash=True)
        res, err = self.call("strata_start", {"wait_seconds": 30})
        self.assertTrue(err)
        self.assertIn("out of memory (fake)", res["error"])
        self.assertEqual(self.strata.state("server"), {})

    def test_loading_then_ready(self):
        port = free_port()
        self.write_config("q2_0", port=port, delay=3)
        res, err = self.call("strata_start", {"wait_seconds": 0})
        self.assertIn("loading", res["summary"])
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("loading", st["summary"])
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))

    def test_stop_while_busy_needs_force(self):
        port = free_port()
        self.write_config("q2_0", port=port, busy=True)
        self.call("strata_start", {"wait_seconds": 30})
        res, err = self.call("strata_stop", {})
        self.assertTrue(err)
        self.assertIn("force=true", res["error"])
        res, err = self.call("strata_stop", {"force": True})
        self.assertFalse(err, res)

    def test_stop_does_not_signal_a_group_after_its_leader_exits(self):
        port = free_port()
        self.write_config("q2_0", port=port)
        self.call("strata_start", {"wait_seconds": 30})
        with mock.patch.object(M, "end_group_leftovers", return_value=True):
            result, error = self.call("strata_stop", {})

        self.assertFalse(error, result)
        self.assertIn("ownership cannot be verified", result["summary"])
        self.assertFalse(result["ended_cleanly"])
        self.assertTrue(result["process_group_may_remain"])

    def test_external_server_is_only_unloaded(self):
        port = free_port()
        self.write_config("q2_0", port=port)
        p = subprocess.Popen([sys.executable, str(self.root / "serve" / "server.py"), "--config",
                              str(self.root / "strata-q2_0.json"), "--port", str(port)],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.extra.append(p)
        self.assertTrue(wait_for(lambda: self.strata.probe(port) is not None, 30))
        st, _ = self.call("strata_status", {"hardware": False})
        self.assertIn("outside this MCP server", st["running"][0]["started_by"])
        res, err = self.call("strata_start", {})
        self.assertIn("started outside this MCP server", res["summary"])
        res, err = self.call("strata_stop", {})
        self.assertFalse(err, res)
        self.assertIn("only its model was unloaded", res["summary"])
        time.sleep(0.5)
        self.assertIsNone(p.poll())                      # the server itself was not touched

    def test_never_kills_an_unrelated_process(self):
        other = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        self.extra.append(other)
        # a stale record: this process id, but another start time (a reused id)
        self.strata.save_state("server", {"pid": other.pid, "ident": "mac:1",
                                          "port": free_port(), "model": "q2_0"})
        res, _ = self.call("strata_stop", {})
        self.assertEqual(res["summary"], "Strata is not running")
        self.assertFalse(M.proc_terminate(other.pid, "mac:1"))
        time.sleep(0.3)
        self.assertIsNone(other.poll())

    def test_status_sees_other_programs_on_the_port(self):
        class H(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(404)
                self.end_headers()

            def log_message(self, *a):
                pass
        httpd = HTTPServer(("127.0.0.1", 0), H)
        port = httpd.server_address[1]
        threading.Thread(target=httpd.serve_forever, daemon=True).start()
        try:
            self.write_config("q2_0", port=port)
            st, _ = self.call("strata_status", {"hardware": False})
            self.assertEqual(st["running"], [])
            self.assertEqual(st["ports_used_by_other_programs"][0]["port"], port)
            res, err = self.call("strata_start", {})
            self.assertTrue(err)
            self.assertIn("used by another program", res["error"])
        finally:
            httpd.shutdown()
            httpd.server_close()


# ------------------------------------------------------------------------------------------------ helpers
class Helpers(unittest.TestCase):
    def test_builtin_table_matches_setup(self):
        s = M.Strata(HERE.parent)
        S = s.setup_module()
        self.assertIsNotNone(S, "the repository's setup.py did not import")
        self.assertEqual(list(S.MODELS), list(M.FALLBACK_MODELS))
        for m, d in S.MODELS.items():
            self.assertEqual(d["download_gb"], M.FALLBACK_MODELS[m]["download_gb"], m)
            self.assertEqual(d.get("families", ("qwen", "swift")),
                             M.FALLBACK_MODELS[m].get("families", ("qwen", "swift")), m)
        self.assertEqual(list(S.FAMILIES), list(M.FALLBACK_FAMILIES))
        for f, d in S.FAMILIES.items():
            self.assertEqual(d["tag"], M.FALLBACK_FAMILIES[f]["tag"])
        self.assertEqual(list(S.CONTEXTS), M.FALLBACK_CONTEXTS)

    def test_macos_process_identity_uses_the_start_time_probe(self):
        helper = mock.Mock()
        helper.proc_identity.return_value = "mac:1791000000.123456"
        with mock.patch.object(M.sys, "platform", "darwin"), mock.patch.object(M, "MACOS", helper):
            self.assertEqual(M.proc_identity(1234), "mac:1791000000.123456")
        helper.proc_identity.assert_called_once_with(1234)

    def test_child_identity_retries_transient_probe_failure(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        try:
            with mock.patch.object(M, "proc_identity", side_effect=[None, "mac:123"]):
                self.assertEqual(M.child_identity(child, timeout=0.2), "mac:123")
        finally:
            M.stop_untracked_child(child)

    def test_untracked_child_is_stopped_and_reaped(self):
        child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(10)"])
        M.stop_untracked_child(child)
        self.assertIsNotNone(child.poll())

    def test_untracked_session_cleanup_stops_descendants(self):
        with tempfile.TemporaryDirectory() as temporary:
            pid_path = Path(temporary) / "grandchild.pid"
            child_code = (
                "import subprocess,sys,time; "
                "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)']); "
                f"open({str(pid_path)!r},'w').write(str(p.pid)); "
                "time.sleep(30)"
            )
            leader = subprocess.Popen(
                [sys.executable, "-c", child_code],
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            try:
                deadline = time.monotonic() + 5
                while not pid_path.exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue(pid_path.exists(), "the session leader did not start its child")
                grandchild = int(pid_path.read_text())
                self.assertIsNotNone(M.proc_identity(grandchild))

                cleaned = M.stop_untracked_child(leader, process_group=True)

                self.assertTrue(cleaned)
                self.assertIsNotNone(leader.poll())
                deadline = time.monotonic() + 5
                while M.proc_identity(grandchild) is not None and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertIsNone(M.proc_identity(grandchild))
            finally:
                if leader.poll() is None:
                    M.stop_untracked_child(leader, process_group=True)

    def test_leftover_group_probe_never_sends_a_signal(self):
        with mock.patch.object(M.os, "killpg") as killpg:
            self.assertTrue(M.end_group_leftovers(43210))

        killpg.assert_called_once_with(43210, 0)

    def test_macos_hardware_reports_apple_metal_facts(self):
        device = {
            "index": 0,
            "name": "Apple M5 Pro",
            "working_set": 55_662_788_608,
            "families": ["apple10", "metal4"],
            "cores": 20,
            "unified": True,
        }

        class MacOS:
            @staticmethod
            def ram_bytes():
                return 64 * 2**30

            @staticmethod
            def cpu_brand():
                return "Apple M5 Pro"

            @staticmethod
            def metal_devices():
                return (device,)

            @staticmethod
            def device_problem(gpu):
                return None

            @staticmethod
            def macos_version():
                return (27, 0)

        s = M.Strata(HERE.parent)
        with mock.patch.object(M.sys, "platform", "darwin"), mock.patch.object(M, "MACOS", MacOS):
            hardware = s._hardware()

        self.assertEqual(hardware["ram_gb"], 64.0)
        self.assertEqual(hardware["metal"]["name"], "Apple M5 Pro")
        self.assertTrue(hardware["metal"]["usable"])
        self.assertEqual(hardware["metal"]["working_set_gib"], 51.8)
        self.assertEqual(hardware["metal"]["families"], ["apple10", "metal4"])
        self.assertEqual(hardware["metal"]["cores"], 20)
        self.assertEqual(hardware["metal"]["macos"], "27.0")
    def test_macos_recommendation_does_not_infer_fit_from_memory_size(self):
        s = M.Strata(HERE.parent)
        with mock.patch.object(M.sys, "platform", "darwin"):
            recommendation = s.recommend({
                "ram_gb": 64.0,
                "metal": {"working_set_gib": 51.8},
            })
            families = s.model_table({
                "ram_gb": 64.0,
            })

        self.assertIsNone(recommendation["model"])
        self.assertIn("Metal engine", recommendation["why"])
        for family in families:
            for model in family["sizes"]:
                expected = ("not supported in the macOS port" if family["family"] == "unsloth"
                            else "the Metal engine plan is unavailable; the macOS installer is not implemented")
                self.assertEqual(model["on_this_mac"], expected)
        self.assertFalse(next(f for f in families if f["family"] == "unsloth")["available_on_this_mac"])

    def test_macos_recommendation_uses_8k_only_models_as_a_last_resort(self):
        class Setup:
            MODELS = M.FALLBACK_MODELS
            FAMILIES = M.FALLBACK_FAMILIES
            CONTEXTS = M.FALLBACK_CONTEXTS

            @staticmethod
            def metal_plan(family, model, context, kv, vision="none"):
                return {"fits": 1, "mode": "full" if context == 8192 else "none",
                        "context": context, "experts_resident": 100, "experts_total": 100}

            @staticmethod
            def plan_class(plan):
                return "full" if plan["mode"] == "full" else "none"

        s = M.Strata(HERE.parent)
        s.setup_module = lambda: Setup
        s.engine_info = lambda: {"backend": "metal"}
        hardware = {"ram_gb": 64, "metal": {}}
        with mock.patch.object(M.sys, "platform", "darwin"):
            recommendation = s.recommend(hardware)
            families = s.model_table(hardware)

        self.assertIsNotNone(recommendation["model"])
        self.assertEqual(recommendation["context"], 8192)
        self.assertIn("8K context", recommendation["why"])
        first_qwen = next(f for f in families if f["family"] == "qwen")["sizes"][0]
        self.assertIn("fits at 8K only", first_qwen["on_this_mac"])

    def test_macos_install_allows_8k_only_fit_and_checks_larger_context(self):
        class Setup:
            MODELS = M.FALLBACK_MODELS
            FAMILIES = M.FALLBACK_FAMILIES
            CONTEXTS = M.FALLBACK_CONTEXTS

            @staticmethod
            def metal_plan(family, model, context, kv, vision="none"):
                fits = context == 8192
                return {"fits": int(fits), "mode": "full" if fits else "none",
                        "experts_resident": 100 if fits else 0, "experts_total": 100}

            @staticmethod
            def plan_class(plan):
                return "full" if plan["fits"] else "none"

        s = M.Strata(HERE.parent)
        s.setup_module = lambda: Setup
        s.engine_info = lambda: {"backend": "metal"}
        s.hardware = lambda: {"ram_gb": 64.0, "metal": {}}
        s.disk_free = lambda _path: 1000.0
        with tempfile.TemporaryDirectory() as temporary:
            s.data_dir = lambda: Path(temporary) / "Strata-data"
            tools = M.Tools(s)
            native_probe = mock.Mock(MAC=True)
            native_probe.translated.return_value = False
            with (mock.patch.object(M.sys, "platform", "darwin"),
                  mock.patch.object(M, "MACOS", native_probe),
                  mock.patch.object(M.platform, "machine", return_value="arm64")):
                plan, args = tools.install_plan(
                    "qwen", "Q2_0", 8192, "no", None, None, None, "auto", None, None, None
                )
                self.assertEqual(plan["context"], 8192)
                self.assertNotIn("--kv", args)
                with self.assertRaisesRegex(M.ToolError, "cannot run at 32768 tokens"):
                    tools.install_plan("qwen", "Q2_0", 32768, "no", None, None, None,
                                       "auto", None, None, None)

    def test_macos_recommendation_uses_the_metal_engine_plan(self):
        plan_calls = []

        class Setup:
            MODELS = M.FALLBACK_MODELS
            FAMILIES = M.FALLBACK_FAMILIES
            CONTEXTS = M.FALLBACK_CONTEXTS

            @staticmethod
            def metal_plan(family, model, context, kv, vision="none"):
                plan_calls.append((family, model, context, kv, vision))
                full_limit = 65536 if family == "qwen" and model == "IQ3_XXS" else 32768
                if context <= full_limit:
                    return {"fits": 1, "mode": "full", "experts_resident": 100, "experts_total": 100}
                return {"fits": 1, "mode": "tiered", "experts_resident": 80, "experts_total": 100}

            @staticmethod
            def plan_class(plan):
                if not plan["fits"]:
                    return "none"
                if plan["mode"] == "full":
                    return "full"
                share = plan["experts_resident"] / plan["experts_total"]
                return "ok" if share >= 0.9 else "slow"

        s = M.Strata(HERE.parent)
        s.setup_module = lambda: Setup
        s.engine_info = lambda: {"backend": "metal"}
        hardware = {
            "ram_gb": 64.0,
            "metal": {"name": "Apple GPU", "working_set_gib": 51.8, "usable": True},
        }
        with mock.patch.object(M.sys, "platform", "darwin"):
            recommendation = s.recommend(hardware)

        self.assertEqual(recommendation["family"], "qwen")
        self.assertEqual(recommendation["model"], "IQ3_XXS")
        self.assertEqual(recommendation["context"], 65536)
        self.assertEqual(recommendation["backend"], "metal")
        self.assertNotIn("setup_command", recommendation)
        self.assertIn(("qwen", "IQ3_XXS", 32768, "int8", "none"), plan_calls)

    def test_macos_install_preview_uses_vision_kv_and_context_plan(self):
        plan_calls = []

        class Setup:
            MODELS = M.FALLBACK_MODELS
            FAMILIES = M.FALLBACK_FAMILIES
            CONTEXTS = M.FALLBACK_CONTEXTS

            @staticmethod
            def metal_plan(family, model, context, kv, vision="none"):
                plan_calls.append((family, model, context, kv, vision))
                mode = "full" if context <= 65536 else "tiered"
                return {"fits": 1, "mode": mode, "experts_resident": 100, "experts_total": 100}

            @staticmethod
            def plan_class(plan):
                return "full" if plan["mode"] == "full" else "none"

        s = M.Strata(HERE.parent)
        s.setup_module = lambda: Setup
        s.engine_info = lambda: {"backend": "metal"}
        s.hardware = lambda: {
            "ram_gb": 64.0,
            "metal": {"name": "Apple GPU", "working_set_gib": 51.8, "usable": True},
        }
        s.disk_free = lambda _path: 1000.0
        s.data_dir = lambda: Path(tempfile.gettempdir()) / "Strata-data"
        tools = M.Tools(s)
        native_probe = mock.Mock(MAC=True)
        native_probe.translated.return_value = False
        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", native_probe),
              mock.patch.object(M.platform, "machine", return_value="arm64")):
            plan, args = tools.install_plan(
                "qwen", "IQ3_XXS", None, "yes", "q4_0", None, None, "metal", "off", None, None
            )

        self.assertIn(("qwen", "IQ3_XXS", 32768, "q4_0", "gpu"), plan_calls)
        self.assertIn(("qwen", "IQ3_XXS", 65536, "q4_0", "gpu"), plan_calls)
        self.assertEqual(plan["context"], 65536)
        self.assertEqual(plan["images"], "yes")
        self.assertEqual(plan["fit"], "fits: every expert fits in Metal memory")
        self.assertEqual(plan["installer_status"], "not_implemented")
        self.assertEqual(plan["kv"], "q4_0")
        self.assertEqual(args, [])

    def test_macos_preview_rejects_gpu_selectors_and_unknown_residency_mode(self):
        s = M.Strata(HERE.parent)
        s.hardware = lambda: {"ram_gb": 64.0, "metal": {}}
        s.data_dir = lambda: Path(tempfile.gettempdir()) / "Strata-data"
        tools = M.Tools(s)
        native_probe = mock.Mock(MAC=True)
        native_probe.translated.return_value = False
        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", native_probe),
              mock.patch.object(M.platform, "machine", return_value="arm64")):
            plan, args = tools.install_plan(
                "qwen", "Q2_0", 32768, "no", None, None, None, "auto", "auto", None, None
            )
            with self.assertRaisesRegex(M.ToolError, "GPU selectors are not supported"):
                tools.install_plan("qwen", "Q2_0", 32768, "no", None, 0, None, None, None, None, None)
            with self.assertRaisesRegex(M.ToolError, "GPU selectors are not supported"):
                tools.install_plan("qwen", "Q2_0", 32768, "no", None, None, "0,1", None, None, None, None)
            with self.assertRaisesRegex(M.ToolError, "uses the Metal backend"):
                tools.install_plan("qwen", "Q2_0", 32768, "no", None, None, None, "cuda", None, None, None)
            with self.assertRaisesRegex(M.ToolError, "low_ram must be one of"):
                tools.install_plan("qwen", "Q2_0", 32768, "no", None, None, None, "auto", "mmap", None, None)

        self.assertEqual(args, [])
        self.assertEqual(plan["installer_status"], "not_implemented")
        self.assertEqual(plan["fit"], "unavailable: Metal planner is not implemented")

    def test_macos_explicit_context_uses_exact_metal_plan_and_effective_kv(self):
        plan_calls = []

        class Setup:
            MODELS = M.FALLBACK_MODELS
            FAMILIES = M.FALLBACK_FAMILIES
            CONTEXTS = M.FALLBACK_CONTEXTS

            @staticmethod
            def metal_plan(family, model, context, kv, vision="none"):
                plan_calls.append((family, model, context, kv, vision))
                fits = context != 65536
                return {"fits": int(fits), "mode": "full" if fits else "none",
                        "experts_resident": 100 if fits else 0, "experts_total": 100,
                        "reason": "not enough GPU memory" if not fits else ""}

            @staticmethod
            def plan_class(plan):
                return "full" if plan["fits"] else "none"

        s = M.Strata(HERE.parent)
        s.setup_module = lambda: Setup
        s.engine_info = lambda: {"backend": "metal"}
        s.hardware = lambda: {"ram_gb": 64.0, "metal": {}}
        s.disk_free = lambda _path: 1000.0
        s.data_dir = lambda: Path(tempfile.gettempdir()) / "Strata-data"
        tools = M.Tools(s)
        native_probe = mock.Mock(MAC=True)
        native_probe.translated.return_value = False
        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", native_probe),
              mock.patch.object(M.platform, "machine", return_value="arm64")):
            with self.assertRaisesRegex(M.ToolError, "cannot run at 65536 tokens"):
                tools.install_plan("qwen", "IQ3_XXS", 65536, "no", "q4_0", None, None,
                                   "auto", None, None, None)
            plan, args = tools.install_plan("qwen", "IQ3_XXS", 8192, "no", "q4_0", None, None,
                                            "auto", None, None, None)

        self.assertIn(("qwen", "IQ3_XXS", 8192, "fp16", "none"), plan_calls)
        self.assertEqual(plan["kv"], "fp16")
        self.assertNotIn("--kv", args)

    def test_macos_install_refuses_unsloth_before_hardware_or_planning(self):
        s = M.Strata(HERE.parent)
        s.hardware = lambda: self.fail("hardware must not be queried for an unsupported family")
        tools = M.Tools(s)
        with mock.patch.object(M.sys, "platform", "darwin"):
            with self.assertRaisesRegex(M.ToolError, "not supported in the macOS port"):
                tools.install_plan("unsloth", "UD-Q4_K_XL", 32768, "no", None, None, None,
                                   "auto", None, None, None)

    def test_macos_install_refuses_rosetta_and_x86_python_before_hardware(self):
        s = M.Strata(HERE.parent)
        s.hardware = lambda: self.fail("hardware must not be queried for an unsupported Python")
        tools = M.Tools(s)
        rosetta = mock.Mock(MAC=True)
        rosetta.translated.return_value = True
        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", rosetta),
              mock.patch.object(M.platform, "machine", return_value="arm64")):
            with self.assertRaisesRegex(M.ToolError, "running under Rosetta"):
                tools.install_plan("qwen", "Q2_0", 8192, "no", None, None, None,
                                   "auto", None, None, None)

        native_probe = mock.Mock(MAC=True)
        native_probe.translated.return_value = False
        with (mock.patch.object(M.sys, "platform", "darwin"),
              mock.patch.object(M, "MACOS", native_probe),
              mock.patch.object(M.platform, "machine", return_value="x86_64")):
            with self.assertRaisesRegex(M.ToolError, "native Apple-silicon arm64"):
                tools.install_plan("qwen", "Q2_0", 8192, "no", None, None, None,
                                   "auto", None, None, None)

    def test_macos_data_dir_refuses_icloud_and_cloudstorage_roots(self):
        with tempfile.TemporaryDirectory() as temporary:
            home = Path(temporary) / "home"
            root = Path(temporary) / "Strata"
            paths = (
                home / "Library" / "Mobile Documents" / "com~apple~CloudDocs" / "Strata-data",
                home / "Library" / "CloudStorage" / "Drive" / "Strata-data",
                home / "Desktop" / "Strata-data",
                home / "Documents" / "Strata-data",
                home / "Downloads" / "Strata-data",
            )
            for path in paths:
                path.parent.mkdir(parents=True, exist_ok=True)
            tools = M.Tools(M.Strata(root))
            native_probe = mock.Mock(MAC=True)
            native_probe.translated.return_value = False
            with (mock.patch.object(M.sys, "platform", "darwin"),
                  mock.patch.object(M.Path, "home", return_value=home)):
                for path in paths:
                    with self.subTest(path=path), self.assertRaisesRegex(
                        M.ToolError, "must be outside iCloud Drive"
                    ):
                        tools.check_data_dir(str(path))

                alias = home / "local-files"
                alias.symlink_to(paths[1].parent)
                with self.assertRaisesRegex(M.ToolError, "must be outside iCloud Drive"):
                    tools.check_data_dir(str(alias / "Strata-data"))

                configured = M.Strata(root)
                configured.data_dir = lambda: paths[3]
                configured.hardware = lambda: {"ram_gb": 64.0, "metal": {}}
                configured_tools = M.Tools(configured)
                with (mock.patch.object(M, "MACOS", native_probe),
                      mock.patch.object(M.platform, "machine", return_value="arm64")):
                    with self.assertRaisesRegex(M.ToolError, "must be outside iCloud Drive"):
                        configured_tools.install_plan(
                            "qwen", "Q2_0", 8192, "no", None, None, None, "auto", None, None, None
                        )

    def test_macos_tool_schema_exposes_only_supported_platform_options(self):
        s = M.Strata(HERE.parent)
        models, families, contexts, _source = s.tables()
        with mock.patch.object(M.sys, "platform", "darwin"):
            tools = M.tool_defs(models, families, contexts)

        by_name = {tool["name"]: tool for tool in tools}
        install = by_name["strata_install"]["inputSchema"]["properties"]
        start = by_name["strata_start"]["inputSchema"]["properties"]
        self.assertNotIn("unsloth", install["family"]["enum"])
        self.assertNotIn("UD-Q4_K_XL", install["model"]["enum"])
        self.assertNotIn("gpus", install)
        self.assertEqual(install["backend"]["enum"], ["auto", "metal"])
        self.assertNotIn("gpu", install)
        self.assertNotIn("gpu", start)
        self.assertIn("Metal plan", by_name["strata_models"]["description"])

    def test_macos_start_has_no_gpu_argument(self):
        s = M.Strata(HERE.parent)
        tools = M.Tools(s)
        with mock.patch.object(M.sys, "platform", "darwin"):
            with self.assertRaises(TypeError):
                tools.strata_start(gpu=0)


class SystemFolders(unittest.TestCase):
    def test_the_macos_list_refuses_system_folders_and_allows_the_temporary_ones(self):
        """docs/mac/07, section 8.1: the darwin list of system folders; /private/tmp, /private/var/folders and
        /usr/local stay allowed, and so does a temporary folder."""
        refused = ("/System/x", "/Library/x", "/Applications/x", "/bin/x", "/sbin/x", "/usr/bin", "/private/etc/x",
                   "/private/var/x", "/dev/x", "/cores/x", "/opt/homebrew/x", "/Volumes/Recovery/x")
        allowed = ("/private/tmp/x", "/private/var/folders/x", "/usr/local/x", "/Users/x/Strata-data")
        for path in refused:
            with self.subTest(path=path):
                self.assertTrue(M.is_system_folder(Path(path)))
        for path in allowed:
            with self.subTest(path=path):
                self.assertFalse(M.is_system_folder(Path(path)))
        with tempfile.TemporaryDirectory() as temporary:
            self.assertFalse(M.is_system_folder(Path(temporary) / "Strata-data"))


if __name__ == "__main__":
    unittest.main()
