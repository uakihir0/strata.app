#!/usr/bin/env python3
"""Mac-only MCP tools for inspecting and controlling an existing Strata installation.

The setup flow is not implemented yet. `strata_install` returns an informational preview and refuses to install;
`strata_start` only works when a Strata runtime and model configuration already exist.

Run with: `python3 tools/strata_mcp.py`. The MCP protocol uses JSON-RPC 2.0 over stdin/stdout and Python's standard
library. The supported host is native Apple silicon with macOS 27 or newer.

Safety: no tool accepts a shell command. Paths used for an existing data directory are constrained to the Strata
data folder or an approved external `Strata-*` folder. This server checks the process identity before stopping a
server it started. The identity check and POSIX signal are separate operations, so an identifier reuse race remains;
see REV-005 in docs/mac/IMPLEMENTATION-DECISIONS.md. A server started elsewhere is only asked over its API to unload.

Not to be confused with serve/mcp.py: that is Strata's MCP *client* (the model calling your tools from the chat page).
This file is an MCP *server* for the AI assistant you already use.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

PROTOCOL = "2025-06-18"
PROTOCOLS = ("2025-06-18", "2025-03-26", "2024-11-05")   # the revisions this server answers in
VERSION = "0.1.0"
DEFAULT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_PORT = 8080
STATE_DIR = ".strata-mcp"                             # in the Strata folder (git-ignored)
_macos_path = DEFAULT_ROOT / "serve" / "macos.py"
_macos_spec = importlib.util.spec_from_file_location("strata_macos", _macos_path)
MACOS = importlib.util.module_from_spec(_macos_spec)
_macos_spec.loader.exec_module(MACOS)

# If setup.py cannot be imported, the read-only tools use the same shared model catalog.
if str(DEFAULT_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_ROOT))
from tools.model_catalog import CONTEXTS as FALLBACK_CONTEXTS  # noqa: E402
from tools.model_catalog import FAMILIES as FALLBACK_FAMILIES  # noqa: E402
from tools.model_catalog import MODELS as FALLBACK_MODELS  # noqa: E402

BENCH_PROMPT = ("Write a short story (about 300 words) about a lighthouse keeper who finds a message in a bottle. "
                "Plain prose, no title.")

def log(msg: str) -> None:
    """Diagnostics go to stderr: stdout carries only the protocol."""
    try:
        sys.stderr.write(f"[strata-mcp] {msg}\n")
        sys.stderr.flush()
    except (OSError, ValueError):
        pass


# ------------------------------------------------------------------------------------------------ processes
def proc_identity(pid) -> str | None:
    """Return the macOS start-time token for a running process, or None."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if pid <= 0 or sys.platform != "darwin" or MACOS is None:
        return None
    return MACOS.proc_identity(pid)


def proc_alive(pid, ident) -> bool:
    return bool(ident) and proc_identity(pid) == ident


def child_identity(child, timeout: float = 1.0) -> str | None:
    """Wait for a newly spawned child to become visible to the macOS identity probe."""
    deadline = time.monotonic() + timeout
    while child.poll() is None:
        ident = proc_identity(child.pid)
        if ident:
            return ident
        if time.monotonic() >= deadline:
            break
        time.sleep(0.05)
    return None


def end_group_leftovers(pgid) -> bool:
    """Probe whether a process group remains, without signaling it."""
    if not pgid:
        return False
    try:
        os.killpg(int(pgid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


def stop_untracked_child(child, *, process_group: bool = False) -> bool:
    """Stop and reap a child whose identity could not be recorded."""
    if child.poll() is not None:
        return not end_group_leftovers(child.pid) if process_group else True
    group_signal_sent = False
    if process_group:
        try:
            os.killpg(child.pid, signal.SIGKILL)
            group_signal_sent = True
        except OSError:
            pass
    if not group_signal_sent:
        try:
            child.kill()
        except OSError:
            if child.poll() is None:
                return False
    try:
        child.wait(timeout=5)
    except subprocess.TimeoutExpired:
        child.kill()
        child.wait()
    return child.poll() is not None and (not process_group or
                                         (group_signal_sent and not end_group_leftovers(child.pid)))


def proc_terminate(pid, ident) -> bool:
    """Send SIGTERM only while the recorded macOS process identity still matches."""
    if not proc_alive(pid, ident):
        return False
    try:
        os.kill(int(pid), signal.SIGTERM)
        return True
    except OSError:
        return False


def proc_kill_tree(pid, ident) -> bool:
    """End a process we started and its children after checking the recorded identity."""
    if not proc_alive(pid, ident):
        return False
    try:
        os.killpg(int(pid), signal.SIGKILL)
        return True
    except OSError:
        try:
            os.kill(int(pid), signal.SIGKILL)
            return True
        except OSError:
            return False


def wait_gone(pid, ident, seconds: float) -> bool:
    end = time.time() + seconds
    while time.time() < end:
        if not proc_alive(pid, ident):
            return True
        time.sleep(0.25)
    return not proc_alive(pid, ident)


def spawn_detached(cmd: list, cwd: Path, log_path: Path, env: dict | None = None) -> subprocess.Popen:
    """Start a background process in a new session, with stdout and stderr sent to its log."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "ab") as logf:
        return subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL, stdout=logf,
                               stderr=subprocess.STDOUT, env=env, close_fds=True, start_new_session=True)


def run_quiet(cmd, timeout=30) -> str:
    """Return a command's stdout, or an empty string when it fails."""
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              stdin=subprocess.DEVNULL).stdout or ""
    except (OSError, subprocess.SubprocessError, ValueError):
        return ""


def read_tail(path: Path, lines: int, max_bytes: int = 512 * 1024) -> list[str]:
    """The last lines of a log; a progress line rewritten with \\r counts as its last version."""
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    out = []
    for raw in data.split("\n"):
        part = [p for p in raw.split("\r") if p.strip()]
        if part:
            out.append(part[-1].rstrip())
    return out[-lines:]


def dir_size_gb(path: Path, limit_files: int = 20000) -> float:
    total, n = 0, 0
    for base, _dirs, files in os.walk(path):
        for f in files:
            try:
                total += os.stat(os.path.join(base, f)).st_size
            except OSError:
                pass
            n += 1
            if n >= limit_files:
                return round(total / 1e9, 1)
    return round(total / 1e9, 1)


# ------------------------------------------------------------------------------------------------ HTTP
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))   # localhost: never through a proxy


def http_json(url: str, method="GET", body=None, api_key=None, timeout=3.0):
    """(status, JSON or None); status 0 when nothing answers."""
    data = json.dumps(body).encode() if body is not None else (b"" if method == "POST" else None)
    req = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    if api_key:
        req.add_header("Authorization", f"Bearer {api_key}")
    try:
        with _OPENER.open(req, timeout=timeout) as r:
            raw = r.read()
            status = r.status
    except urllib.error.HTTPError as e:
        raw, status = e.read(), e.code
    except (OSError, ValueError):
        return 0, None
    try:
        return status, json.loads(raw or b"null")
    except ValueError:
        return status, None


def port_free(port: int, host="127.0.0.1") -> bool:
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, port))
        return True
    except OSError:
        return False
    finally:
        s.close()


# ------------------------------------------------------------------------------------------------ validation
class ToolError(Exception):
    """A clear message for the AI (and its user): the tool result is an error, the server goes on."""


def check_schema(schema: dict, args) -> dict:
    """The tools' JSON schemas, enforced: object, known properties only, types, enums, ranges, patterns."""
    if args is None:
        args = {}
    if not isinstance(args, dict):
        raise ToolError("arguments must be a JSON object")
    props = schema.get("properties", {})
    for k in args:
        if k not in props:
            raise ToolError(f"unknown argument {k!r}; this tool takes: {', '.join(props) or 'no arguments'}")
    for k in schema.get("required", []):
        if k not in args:
            raise ToolError(f"missing argument {k!r}")
    for k, v in args.items():
        p = props[k]
        t = p.get("type")
        if t == "integer" and (isinstance(v, bool) or not isinstance(v, int)):
            if isinstance(v, float) and v.is_integer():
                v = args[k] = int(v)
            elif isinstance(v, str) and re.fullmatch(r"\d{1,7}", v.strip()):
                v = args[k] = int(v.strip())              # some clients send numbers as strings
            else:
                raise ToolError(f"{k} must be a whole number")
        if t == "boolean" and not isinstance(v, bool):
            if v in ("true", "false"):
                v = args[k] = v == "true"
            else:
                raise ToolError(f"{k} must be true or false")
        if t == "string":
            if not isinstance(v, str):
                raise ToolError(f"{k} must be a string")
            if len(v) > p.get("maxLength", 300) or "\x00" in v or "\n" in v or "\r" in v:
                raise ToolError(f"{k} is too long or contains control characters")
            if "pattern" in p and not re.fullmatch(p["pattern"], v):
                raise ToolError(f"{k}={v!r} is not valid ({p.get('description', p['pattern'])})")
        if "enum" in p and v not in p["enum"]:
            raise ToolError(f"{k} must be one of: {', '.join(map(str, p['enum']))} (got {v!r})")
        if t == "integer":
            if "minimum" in p and v < p["minimum"] or "maximum" in p and v > p["maximum"]:
                raise ToolError(f"{k} must be between {p.get('minimum')} and {p.get('maximum')} (got {v})")
    return args


def is_inside(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def canonical_case_path(path: str) -> str:
    """Use the on-disk spelling for existing macOS components without merging case-sensitive siblings."""
    value = Path(path)
    current = Path(value.anchor)
    for part in value.parts[1:]:
        try:
            entries = os.listdir(current)
        except OSError:
            current /= part
            continue
        if part not in entries:
            for entry in entries:
                if entry.casefold() != part.casefold():
                    continue
                try:
                    if os.path.samefile(current / part, current / entry):
                        part = entry
                        break
                except OSError:
                    pass
        current /= part
    return str(current)


def norm(p) -> Path:
    """Normalize an absolute macOS path and preserve its on-disk component spelling."""
    value = os.path.realpath(os.path.abspath(str(p)))
    return Path(canonical_case_path(value))


def system_folders() -> list[Path]:
    return [norm(p) for p in ("/System", "/Library", "/Applications", "/bin", "/sbin", "/usr", "/private/etc",
                              "/private/var", "/dev", "/cores", "/opt/homebrew", "/Volumes/Recovery")]


def is_system_folder(path: Path) -> bool:
    allowed = (norm("/private/tmp"), norm("/private/var/folders"), norm("/usr/local"))
    if any(path == root or is_inside(path, root) for root in allowed):
        return False
    return any(path == root or is_inside(path, root) for root in system_folders())


def is_cloud_storage_folder(path: Path) -> bool:
    """Reject cloud roots and common macOS folders that can be iCloud-synced."""
    home = norm(Path.home())
    roots = (
        norm(home / "Desktop"),
        norm(home / "Documents"),
        norm(home / "Downloads"),
        norm(home / "Library" / "Mobile Documents"),
        norm(home / "Library" / "CloudStorage"),
    )
    return any(path == root or is_inside(path, root) for root in roots)


# ------------------------------------------------------------------------------------------------ Strata
class Strata:
    """One Strata folder: installed model files and the processes this server started."""

    def __init__(self, root: Path | str = DEFAULT_ROOT):
        self.root = Path(root).resolve()
        self.state_dir = self.root / STATE_DIR
        self.lock = threading.Lock()                    # install / start / stop one at a time
        self.children: list[subprocess.Popen] = []      # reaped now and then (no zombies on POSIX)
        self._setup = None
        self._setup_tried = False
        self._hw_cache = None
        self._metal_plan_cache = {}
        self.python_override: str | None = None         # tests: the interpreter that runs setup / the server
        self.health_timeout = 2.0

    # ---- setup.py's own tables and logic
    def setup_module(self):
        """setup.py imported as a module (its tables and its pure checks), or None.  Its command runner is replaced
        by one; its install path is never run."""
        if not self._setup_tried:
            self._setup_tried = True
            try:
                spec = importlib.util.spec_from_file_location("strata_setup_for_mcp", self.root / "setup.py")
                mod = importlib.util.module_from_spec(spec)
                real = sys.stdout
                sys.stdout = sys.stderr                 # nothing it prints may reach the protocol stream
                try:
                    spec.loader.exec_module(mod)
                finally:
                    sys.stdout = real
                if all(hasattr(mod, a) for a in ("MODELS", "FAMILIES", "CONTEXTS")):
                    mod.out = lambda cmd: run_quiet(cmd, 60)
                    self._setup = mod
            except (Exception, SystemExit) as e:        # noqa: BLE001 - any failure: the built-in tables
                log(f"setup.py could not be imported ({e}); using the built-in model table")
        return self._setup

    def tables(self):
        s = self.setup_module()
        if s is not None:
            return s.MODELS, s.FAMILIES, list(s.CONTEXTS), "setup.py"
        return FALLBACK_MODELS, FALLBACK_FAMILIES, FALLBACK_CONTEXTS, "built-in table (setup.py not importable)"

    @staticmethod
    def sizes_of(models, family) -> list:
        return [m for m in models if family in models[m].get("families", ("qwen", "swift"))]

    # ---- paths
    def settings_path(self) -> Path:
        return Path.home() / "Library" / "Application Support" / "Strata" / "settings.json"

    def data_dir(self) -> Path:
        try:
            s = json.loads(self.settings_path().read_text(encoding="utf-8"))
        except (OSError, ValueError):
            s = {}
        d = s.get("data_dir") if isinstance(s, dict) else None
        return Path(d) if d else self.root.parent / "Strata-data"

    def allowed_roots(self) -> list[Path]:
        return [norm(self.root), norm(self.data_dir())]

    def allowed(self, p) -> bool:
        p = norm(p)
        return any(is_inside(p, r) for r in self.allowed_roots())

    def venv_python(self) -> Path:
        return self.root / ".venv" / "bin" / "python"

    def run_python(self) -> str:
        return self.python_override or str(self.venv_python())

    def state(self, name: str) -> dict:
        try:
            d = json.loads((self.state_dir / f"{name}.json").read_text(encoding="utf-8"))
            return d if isinstance(d, dict) else {}
        except (OSError, ValueError):
            return {}

    def save_state(self, name: str, d: dict | None) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        f = self.state_dir / f"{name}.json"
        if d is None:
            f.unlink(missing_ok=True)
        else:
            tmp = f.with_suffix(".tmp")
            tmp.write_text(json.dumps(d, indent=1), encoding="utf-8")
            os.replace(tmp, f)

    def reap(self):
        self.children = [p for p in self.children if p.poll() is None]

    # ---- what is installed
    def configs(self) -> list[Path]:
        return sorted(self.root.glob("strata-*.json"), key=lambda p: p.stat().st_mtime, reverse=True)

    @staticmethod
    def read_config(path: Path) -> dict:
        try:
            cfg = json.loads(path.read_text(encoding="utf-8-sig"))
            return cfg if isinstance(cfg, dict) else {}
        except (OSError, ValueError):
            return {}

    def describe_config(self, path: Path) -> dict:
        cfg = self.read_config(path)
        a = cfg.get("args") if isinstance(cfg.get("args"), list) else []

        def val(k):
            return a[a.index(k) + 1] if k in a and a.index(k) + 1 < len(a) else None
        tag = path.stem[len("strata-"):]
        missing = [p for p in [cfg.get("exe"), *[x for x in a if isinstance(x, str) and x.endswith(".gguf")]]
                   if not p or not Path(p).exists()]
        script = self.root / f"run-{tag}.sh"
        vis = cfg.get("vision")
        return {"model": tag, "config": path.name, "model_name": cfg.get("model_name"),
                "context": int(val("--max-context")) if str(val("--max-context") or "").isdigit() else None,
                "kv": val("--kv"), "images": ("gpu" if vis.get("gpu") else "cpu") if isinstance(vis, dict) else "off",
                "port": cfg.get("port", DEFAULT_PORT), "host": cfg.get("host", "127.0.0.1"),
                "gpu": cfg.get("gpu"), "api_key_set": bool(cfg.get("api_key")),
                "low_ram": "--resident-experts" in a or "--mmap-experts" in a,
                "start_script": script.name if script.exists() else None,
                "ready": not missing and bool(cfg), "missing_files": missing[:3],
                "last_used": time.strftime("%Y-%m-%d %H:%M", time.localtime(path.stat().st_mtime))}

    def find_config(self, model: str | None) -> Path:
        have = self.configs()
        if not have:
            raise ToolError("no model is installed in " + str(self.root) + " yet; the macOS installer is not implemented")
        if not model:
            return have[0]                              # the most recently used, as setup starts it
        m = model.strip().lower()
        exact = [c for c in have if c.stem[len("strata-"):] == m]
        if exact:
            return exact[0]
        by_name = [c for c in have if str(self.read_config(c).get("model_name", "")).lower() == m]
        if by_name:
            return by_name[0]
        ends = [c for c in have if c.stem.endswith("-" + m)]
        if len(ends) == 1:
            return ends[0]
        raise ToolError(f"no installed model {model!r}; installed: " +
                        ", ".join(c.stem[len("strata-"):] for c in have))

    def engine_info(self) -> dict:
        for d in ("engine",):
            try:
                meta = json.loads((self.root / d / "BUILD.json").read_text())
                return {"folder": d, "version": meta.get("version"), "source": meta.get("source"),
                        "backend": meta.get("backend")}
            except (OSError, ValueError):
                continue
        return {"folder": None, "version": None}

    def installed(self) -> dict:
        data = self.data_dir()
        items = {}
        for sub in ("models", "packs"):
            for base in dict.fromkeys([data, self.root]):
                d = base / sub
                if d.is_dir():
                    for c in sorted(d.iterdir()):
                        if c.is_dir():
                            ggufs = list(c.glob("*.gguf"))
                            entry = {"folder": str(c), "size_gb": dir_size_gb(c)}
                            if sub == "models":
                                entry["complete"] = bool(ggufs) and all(
                                    g.with_name(g.name + ".done").exists() for g in ggufs)
                                entry["downloading"] = bool(list(c.glob("*.part")))
                            items.setdefault(sub, []).append(entry)
        mtp = data / "mtp" / "rt" / "experts.bin"
        return {"strata_folder": str(self.root), "data_folder": str(data), "data_folder_exists": data.is_dir(),
                "python_env": self.venv_python().exists(), "engine": self.engine_info(),
                "models": [self.describe_config(c) for c in self.configs()],
                "downloads": items.get("models", []), "prepared": items.get("packs", []),
                "draft_layer": mtp.exists()}

    # ---- the Mac
    def hardware(self, fresh=False) -> dict:
        if self._hw_cache and not fresh and time.time() - self._hw_cache[0] < 600:
            hw = dict(self._hw_cache[1])
        else:
            hw = self._hardware()
            self._hw_cache = (time.time(), hw)
            hw = dict(hw)
        hw["disk_free_gb"] = self.disk_free(self.data_dir())
        return hw

    @staticmethod
    def disk_free(p: Path) -> float | None:
        p = Path(p)
        while not p.exists() and p.parent != p:
            p = p.parent
        try:
            return round(shutil.disk_usage(p).free / 1e9, 1)
        except OSError:
            return None

    def _hardware(self) -> dict:
        try:
            ram = MACOS.ram_bytes() / 2**30
        except Exception:                                  # noqa: BLE001
            ram = None
        try:
            cpu_name = MACOS.cpu_brand() or "Apple silicon"
        except Exception:                                  # noqa: BLE001
            cpu_name = "Apple silicon"
        try:
            devices = MACOS.metal_devices()
        except Exception:                                  # noqa: BLE001
            devices = ()
        device = devices[0] if devices else {}
        try:
            version = ".".join(str(part) for part in MACOS.macos_version()) or None
        except Exception:                                  # noqa: BLE001
            version = None
        try:
            problem = MACOS.device_problem(device) if device else "no Metal device was found"
        except Exception:                                   # noqa: BLE001
            problem = "could not inspect Metal device support"
        return {
            "os": f"macOS {version or 'unknown'}", "os_version": version,
            "machine": platform.machine(), "python": platform.python_version(),
            "ram_gb": round(ram, 1) if ram else None,
            "cpu": {"name": cpu_name, "cores": os.cpu_count()},
            "metal": {
                "name": device.get("name"),
                "working_set_gib": round(device["working_set"] / 2**30, 1)
                if isinstance(device.get("working_set"), (int, float)) else None,
                "families": list(device.get("families", [])),
                "cores": device.get("cores"), "macos": version,
                "usable": problem is None, "problem": problem,
            },
        }

    def _mac_planner(self):
        """Return setup's Metal planner only when its matching engine is installed."""
        if self.engine_info().get("backend") != "metal":
            return None
        setup = self.setup_module()
        if not setup or not callable(getattr(setup, "metal_plan", None)) \
                or not callable(getattr(setup, "plan_class", None)):
            return None
        return setup

    @staticmethod
    def _mac_plan_at(setup, family: str, model: str, context: int, vision: str, kv: str | None) -> dict:
        plan_kv = "fp16" if context <= 8192 else (kv or "int8")
        plan = setup.metal_plan(family, model, context, plan_kv, vision)
        if not isinstance(plan, dict):
            raise ValueError("metal_plan did not return a plan")
        return {"plan": plan, "class": setup.plan_class(plan), "context": context, "kv": plan_kv}

    def _mac_model_plan(
        self,
        setup,
        family: str,
        model: str,
        *,
        vision: str = "none",
        kv: str | None = None,
    ) -> dict | None:
        """Plan one model using the Metal engine; never infer a fit from RAM or working-set size."""
        key = (family, model, vision, kv)
        if key in self._metal_plan_cache:
            return self._metal_plan_cache[key]

        def run(context: int):
            result = self._mac_plan_at(setup, family, model, context, vision, kv)
            return result["plan"], result["class"]

        try:
            plan32, class32 = run(32768)
            context = None
            context_plan = plan32
            context_class = class32
            class8 = None
            if class32 == "full":
                for candidate in sorted(
                    (c for c in setup.CONTEXTS if 32768 < c <= 131072), reverse=True
                ):
                    candidate_plan, candidate_class = run(candidate)
                    if candidate_class == "full":
                        context = candidate
                        context_plan, context_class = candidate_plan, candidate_class
                        break
                context = context or 32768
            elif class32 in {"ok", "slow", "very slow"}:
                context = 32768
            else:
                plan8, class8 = run(8192)
                if class8 in {"full", "ok", "slow", "very slow"}:
                    context = 8192
                    context_plan, context_class = plan8, class8
            result = {"plan": plan32, "class": class32, "context": context,
                      "context_plan": context_plan, "context_class": context_class, "class8": class8,
                      "selectable": class32 in {"full", "ok", "slow", "very slow"}
                      or class8 in {"full", "ok", "slow", "very slow"}}
        except Exception as exc:                       # noqa: BLE001
            log(f"Metal plan unavailable for {family}/{model}: {exc}")
            result = None
        else:
            self._metal_plan_cache[key] = result
        return result

    @staticmethod
    def _mac_plan_class_text(plan: dict, plan_class: str) -> str:
        if plan_class == "full":
            return "every expert fits in Metal memory"
        if plan_class in {"ok", "slow", "very slow"}:
            total = plan.get("experts_total")
            resident = plan.get("experts_resident")
            if isinstance(total, int) and total > 0 and isinstance(resident, int):
                percent = 100 * resident // total
                suffix = ": slower" if plan_class in {"slow", "very slow"} else ""
                return f"{percent}% of its experts fit in Metal memory; the rest are read from the SSD{suffix}"
            return f"fits ({plan_class})"
        if plan_class == "unsupported":
            return "not yet available on Mac"
        return "does not fit: " + str(plan.get("reason", "the Metal planner rejected this model"))

    @staticmethod
    def _mac_plan_text(result: dict | None) -> str:
        if not result:
            return "the Metal engine plan is unavailable; the macOS installer is not implemented"
        if result.get("context") == 8192 and result["class"] not in {"full", "ok", "slow", "very slow"}:
            return "fits at 8K only: " + Strata._mac_plan_class_text(
                result["context_plan"], result["context_class"]
            )
        text = Strata._mac_plan_class_text(result["plan"], result["class"])
        return "fits: " + text if result["class"] in {"full", "ok", "slow", "very slow"} else text

    def _mac_recommendation(
        self,
        hw: dict,
        family_only: str | None = None,
        *,
        vision: str = "none",
        kv: str | None = None,
    ) -> dict:
        if family_only == "unsloth":
            return {"family": None, "model": None,
                    "why": "Unsloth UD-Q4_K_XL is not yet available on Mac"}
        setup = self._mac_planner()
        if setup is None:
            engine = self.engine_info()
            if engine.get("backend") != "metal":
                why = "install a Metal engine first; its plan determines what fits this Mac"
            else:
                why = ("this setup.py has no Metal model planner yet; finish the Metal setup port before asking "
                       "Strata to choose a model")
            metal = hw.get("metal") or {}
            notes = []
            if hw.get("ram_gb") is not None:
                notes.append(f"{hw['ram_gb']:.1f} GiB physical RAM")
            if metal.get("working_set_gib") is not None:
                notes.append(f"{metal['working_set_gib']:.1f} GiB Metal working set (orientation only)")
            return {"family": None, "model": None, "why": why, "notes": notes}

        models, families, _contexts, _source = self.tables()
        choices = []
        families_to_check = [family_only] if family_only else [
            family for family in families if family != "unsloth"
        ]
        for family in families_to_check:
            if family not in families or family == "unsloth":
                continue
            for model in self.sizes_of(models, family):
                result = self._mac_model_plan(setup, family, model, vision=vision, kv=kv)
                if result is not None and result["selectable"]:
                    choices.append((family, model, result))

        def find(predicate):
            return next((item for item in choices if predicate(*item)), None)

        selected = find(lambda family, model, result:
                        family == "qwen" and model == "IQ3_XXS" and result["class"] == "full")
        if selected is None:
            selected = find(lambda _family, _model, result:
                             result["class"] in {"full", "ok"} and result["context"] != 8192)
        if selected is None:
            selected = find(lambda _family, _model, result:
                            result["class"] in {"slow", "very slow"} and result["context"] != 8192)
        if selected is None:
            selected = find(lambda _family, _model, result: result["context"] == 8192)
        if selected is None:
            name = families[family_only]["title"] if family_only in families else "any"
            return {"family": None, "model": None,
                    "why": f"no {name} model fits this Mac according to the Metal engine plan"}

        family, model, result = selected
        metal = hw.get("metal") or {}
        context = result["context"] or 32768
        return {
            "family": family,
            "model": model,
            "title": f"{families[family]['title']} {model}",
            "context": context,
            "backend": "metal",
            "metal": {key: metal.get(key) for key in ("name", "working_set_gib", "cores")},
            "download_gb": models[model]["download_gb"],
            "why": f"the Metal engine's PLAN says this model {result['context_class']} at "
                   f"{context // 1024}K context",
            "notes": [],
        }

    def recommend(self, hw: dict) -> dict:
        """Return a recommendation only when the installed Metal engine provides a model plan."""
        return self._mac_recommendation(hw)

    def model_table(self, hw: dict | None) -> list:
        models, families, _contexts, _source = self.tables()
        setup = self._mac_planner()
        installed = {c.stem[len("strata-"):] for c in self.configs()}
        out = []
        for family, family_data in families.items():
            sizes = []
            unavailable = family == "unsloth"
            for model in self.sizes_of(models, family):
                result = self._mac_model_plan(setup, family, model) if setup and not unavailable else None
                tag = (family_data.get("tag", "") + model).lower()
                sizes.append({
                    "model": model,
                    "about": models[model]["about"],
                    "download_gb": models[model]["download_gb"],
                    "on_this_mac": "not supported in the macOS port" if unavailable else self._mac_plan_text(result),
                    "installed": tag in installed,
                    "id": tag,
                })
            out.append({
                "family": family,
                "title": family_data["title"],
                "about": family_data.get("about"),
                "experimental": bool(family_data.get("experimental")),
                "available_on_this_mac": not unavailable,
                "images": family_data.get("vision", True) is not False,
                "sizes": sizes,
            })
        return out

    # ---- the running server
    def ports(self, port: int | None) -> list[int]:
        if port:
            return [port]
        ps = [self.state("server").get("port")] + [self.describe_config(c)["port"] for c in self.configs()]
        return list(dict.fromkeys(int(p) for p in [*ps, DEFAULT_PORT] if isinstance(p, int)))

    def api_key_for(self, port: int) -> str | None:
        """The key a config of this folder set for that port (used for our own requests; never shown)."""
        for c in self.configs():
            cfg = self.read_config(c)
            if cfg.get("api_key") and int(cfg.get("port", DEFAULT_PORT)) == port:
                return cfg["api_key"]
        for c in self.configs():
            if self.read_config(c).get("api_key"):
                return self.read_config(c)["api_key"]
        return None

    def probe(self, port: int, deep=True) -> dict | None:
        """What answers on 127.0.0.1:port: a Strata server's facts, {"other": True} for something else, None for
        nothing."""
        base = f"http://127.0.0.1:{port}"
        st, health = http_json(base + "/health", timeout=self.health_timeout)
        if st == 0:
            return None
        if st != 200 or not isinstance(health, dict) or health.get("service") != "strata":
            return {"port": port, "other": True, "note": "something else answers on this port (not a Strata server)"}
        info = {"port": port, "url": base, "model": health.get("model"), "context": health.get("max_context"),
                "images": health.get("images"), "loaded": health.get("loaded"),
                "api_key_required": bool(health.get("api_key"))}
        if not deep:
            return info
        key = self.api_key_for(port)
        st, v1 = http_json(base + "/v1/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(v1, dict):
            info["engine_version"] = v1.get("engine")
            info["uptime_s"] = v1.get("uptime_s")
            info["requests_served"] = (v1.get("activity") or {}).get("requests")
            m = v1.get("machine") or {}
            if m.get("gpu"):
                info["gpu"] = m["gpu"]
            if m.get("ram"):
                info["ram"] = m["ram"]
            if v1.get("last_timings"):
                lt = v1["last_timings"]
                info["last_request"] = {k: lt.get(k) for k in ("prompt_n", "prompt_per_second", "predicted_n",
                                                               "predicted_per_second") if k in lt}
        elif st == 401:
            info["note"] = "the server wants an API key this folder's configs do not have"
        st, s = http_json(base + "/status", api_key=key, timeout=self.health_timeout)
        if st == 200 and isinstance(s, dict):
            info["busy"] = bool(s.get("busy"))
            info["activity"] = {k: s.get(k) for k in ("phase", "generated", "tokens_per_s", "queued", "elapsed_s")
                                if s.get(k) is not None}
        if "gpu" not in info:                           # older servers: the Monitor's numbers
            st, met = http_json(base + "/metrics", api_key=key, timeout=self.health_timeout)
            if st == 200 and isinstance(met, dict):
                hwn = met.get("hardware") or {}
                if hwn.get("gpu_mem_total"):
                    info["gpu"] = {"used_mib": round(hwn.get("gpu_mem_used", 0) / 2**20),
                                   "total_mib": round(hwn["gpu_mem_total"] / 2**20)}
                if hwn.get("ram_total"):
                    info["ram"] = {"used_gib": round(hwn.get("ram_used", 0) / 2**30, 1),
                                   "total_gib": round(hwn["ram_total"] / 2**30, 1)}
                info["engine_version"] = (met.get("engine") or {}).get("version")
        return info

    def tracked_server(self) -> dict | None:
        """The server this MCP server started, while it still runs (else its record is cleared)."""
        s = self.state("server")
        if not s:
            return None
        if proc_alive(s.get("pid"), s.get("ident")):
            return s
        return {**s, "ended": True}

# ------------------------------------------------------------------------------------------------ the tools
MODEL_ID_PATTERN = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,48}"


def tool_defs(models: dict, families: dict, contexts: list) -> list:
    """The macOS-only MCP tool contract."""
    family_choices = [family for family in families if family != "unsloth"]
    model_choices = [model for model in models if model.lower() != "ud-q4_k_xl"]
    port = {"type": "integer", "minimum": 1024, "maximum": 65535,
            "description": "the server's port (default: the installed model's port, usually 8080)"}
    installed_model = {"type": "string", "pattern": MODEL_ID_PATTERN, "maxLength": 49,
                       "description": "an installed model id as listed by strata_status"}
    read_only = {"readOnlyHint": True, "destructiveHint": False, "openWorldHint": False}
    return [
        {"name": "strata_status", "title": "Strata status",
         "description": "Reports whether Strata is running, installed model files, data folder, Apple hardware, and a "
                        "Metal-based recommendation when the engine planner is available. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port,
             "hardware": {"type": "boolean", "description": "include hardware and recommendation (default true)"}}},
         "annotations": read_only},
        {"name": "strata_models", "title": "Strata models",
         "description": "Lists model variants, download sizes, installed files, context lengths, and a Metal plan "
                        "when the engine planner is available. Unsloth is not supported by the macOS port. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {}},
         "annotations": read_only},
        {"name": "strata_install", "title": "Install Strata",
         "description": "The macOS installer is not implemented. This tool returns an informational preview only; "
                        "confirm=true is refused and changes no files.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "family": {"type": "string", "enum": family_choices,
                        "description": "qwen = Qwen3.8-Flash-Next, swift = Swift 1.5, coder = the Coder"},
             "model": {"type": "string", "enum": model_choices,
                       "description": "the quantization size listed by strata_models"},
             "context": {"type": "integer", "enum": contexts,
                         "description": "context length in tokens; checked against the Metal engine plan when available"},
             "vision": {"type": "string", "enum": ["no", "yes", "gpu", "cpu"],
                        "description": "whether the model should accept images; default no"},
             "kv": {"type": "string", "enum": ["int8", "q4_0", "k8v4"],
                    "description": "KV cache precision above 8K context (default int8)"},
             "backend": {"type": "string", "enum": ["auto", "metal"],
                         "description": "Metal is the only supported backend"},
             "low_ram": {"type": "string", "enum": ["auto", "on", "off"],
                         "description": "Metal expert residency mode; default auto"},
             "port": {**port, "description": "the local server port (default 8080)"},
             "data_dir": {"type": "string", "maxLength": 260,
                          "description": "model data folder, such as ~/Strata-data or /Volumes/External/Strata-data"},
             "confirm": {"type": "boolean", "description": "request installation; refused until the macOS installer is implemented"}}},
         "annotations": read_only},
        {"name": "strata_start", "title": "Start Strata",
         "description": "Starts an existing model configuration. The installer is not implemented, so a compatible "
                        "runtime and model files must already be present. Loading can take 1-3 minutes; check progress "
                        "with strata_status. The server keeps running until strata_stop.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "model": installed_model, "port": port,
             "wait_seconds": {"type": "integer", "minimum": 0, "maximum": 600,
                              "description": "time to wait for readiness (default 50 seconds)"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_stop", "title": "Stop Strata",
         "description": "Unloads the model through the local server API and stops a server started by this tool. "
                        "A server started elsewhere is only asked to unload. Refuses while a request is running unless "
                        "force=true.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port, "force": {"type": "boolean", "description": "stop even while a request is running"}}},
         "annotations": {"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True,
                         "openWorldHint": False}},
        {"name": "strata_logs", "title": "Strata logs",
         "description": "Shows recent log output from the Strata server or engine on this Mac. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "source": {"type": "string", "enum": ["server", "engine"]}, "model": installed_model,
             "lines": {"type": "integer", "minimum": 1, "maximum": 500, "description": "default 80"}}},
         "annotations": read_only},
        {"name": "strata_benchmark", "title": "Strata speed test",
         "description": "Runs a short fixed request through a running Strata server and reports generation and "
                        "prompt speed. Refuses while the model is busy.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {
             "port": port, "max_tokens": {"type": "integer", "minimum": 16, "maximum": 512,
                                           "description": "default 200"}}},
         "annotations": read_only},
        {"name": "strata_connect_info", "title": "Connect to Strata",
         "description": "Provides local OpenAI and Anthropic API addresses and example client settings. Read-only.",
         "inputSchema": {"type": "object", "additionalProperties": False, "properties": {"port": port}},
         "annotations": read_only},
    ]


class Tools:
    def __init__(self, strata: Strata):
        self.s = strata

    def defs(self) -> list:
        return tool_defs(*self.s.tables()[:3])

    def call(self, name: str, args) -> dict:
        defs = {d["name"]: d for d in self.defs()}
        if name not in defs:
            raise KeyError(name)
        args = check_schema(defs[name]["inputSchema"], args)
        return getattr(self, name)(**args)

    # ---- strata_status
    def strata_status(self, port=None, hardware=True) -> dict:
        s = self.s
        s.reap()
        servers = []
        for p in s.ports(port):
            info = s.probe(p)
            if info is not None:
                servers.append(info)
        tracked = s.tracked_server()
        out = {}
        running = [x for x in servers if not x.get("other")]
        if tracked and not tracked.get("ended"):
            mine = next((x for x in running if x["port"] == tracked["port"]), None)
            if mine:
                mine["started_by"] = f"this MCP server (pid {tracked['pid']})"
            else:
                out["loading"] = {"model": tracked.get("model"), "port": tracked["port"], "pid": tracked["pid"],
                                  "since": tracked.get("started"),
                                  "last_log_line": (read_tail(Path(tracked.get("log", "")), 1) or [None])[-1]}
        elif tracked and tracked.get("ended"):
            out["last_start"] = {"model": tracked.get("model"), "ended": True,
                                 "last_log_lines": read_tail(Path(tracked.get("log", "")), 6)}
        for x in running:
            x.setdefault("started_by", "outside this MCP server (its own window, a start script or a service)")
        out["running"] = running
        others = [x for x in servers if x.get("other")]
        if others:
            out["ports_used_by_other_programs"] = others
        if running:
            r = running[0]
            summary = (f"Strata is running on port {r['port']}: {r.get('model')}"
                       + (", busy" if r.get("busy") else ", idle")
                       + ("" if r.get("loaded", True) else " (model unloaded; the next request loads it)"))
        elif "loading" in out:
            summary = f"Strata is loading {out['loading']['model']} (port {out['loading']['port']})"
        else:
            summary = "Strata is not running"
        inst = s.installed()
        if not inst["models"]:
            summary += "; no model is installed yet"
        else:
            summary += "; installed: " + ", ".join(m["model"] for m in inst["models"])
        out["installed"] = inst
        if hardware:
            hw = s.hardware()
            out["hardware"] = hw
            out["recommendation"] = s.recommend(hw)
        return {"summary": summary, **out}

    # ---- strata_models
    def strata_models(self) -> dict:
        s = self.s
        hw = s.hardware()
        _models, _families, contexts, source = s.tables()
        recommendation = s.recommend(hw)
        if recommendation.get("model"):
            text = (f"; recommended for this Mac: {recommendation['title']}, "
                    f"{recommendation['context'] // 1024}K context")
        else:
            text = f"; Metal recommendation unavailable: {recommendation['why']}"
        return {
            "summary": "Model families and sizes" + text,
            "families": s.model_table(hw),
            "contexts": contexts,
            "context_note": "Model fit and context recommendations require the Metal engine plan.",
            "recommendation": recommendation,
            "source": source,
            "this_mac": {
                "ram_gb": hw.get("ram_gb"),
                "metal": {key: (hw.get("metal") or {}).get(key) for key in
                          ("name", "working_set_gib", "families", "cores", "usable")},
            },
        }

    # ---- strata_install
    def check_data_dir(self, raw: str) -> Path:
        s = self.s
        example = "~/Strata-data or /Volumes/External/Strata-data"
        if raw.startswith("\\\\") or raw.startswith("//"):
            raise ToolError("data_dir: network paths are not allowed; use a local folder")
        p = Path(os.path.expandvars(raw)).expanduser()
        if not p.is_absolute():
            raise ToolError(f"data_dir must be an absolute path (e.g. {example})")
        if ".." in p.parts:
            raise ToolError("data_dir must not contain '..'")
        n = norm(p)
        if is_cloud_storage_folder(n):
            raise ToolError(f"data_dir {raw!r} must be outside iCloud Drive, ~/Library/CloudStorage, "
                            "~/Desktop, ~/Documents, and ~/Downloads")
        current = norm(s.data_dir())
        if n == current or is_inside(n, norm(s.root)) or n == norm(s.root.parent / "Strata-data"):
            pass
        else:
            if not p.name.lower().startswith("strata"):
                raise ToolError(f"data_dir {raw!r}: a new data folder must be named Strata... (e.g. "
                                f"{Path(p.anchor or '/') / 'Strata-data'}), so that no other folder is filled with "
                                "model files")
            if not p.parent.is_dir():
                raise ToolError(f"data_dir {raw!r}: its parent folder {p.parent} does not exist")
            if is_system_folder(n):
                raise ToolError(f"data_dir {raw!r} is inside a system folder")
            if p.exists() and not p.is_dir():
                raise ToolError(f"data_dir {raw!r} is a file")
        return p

    def strata_install(self, family=None, model=None, context=None, vision="no", kv="int8", gpu=None,
                       gpus=None, backend="auto", low_ram="auto", port=None, data_dir=None, confirm=False) -> dict:
        if confirm:
            raise ToolError("the macOS installer is not implemented; no files were changed and no install was started")
        plan, _ = self.install_plan(family, model, context, vision, kv, gpu, gpus, backend, low_ram, port, data_dir)
        return {"summary": f"Informational preview for {plan['title']}. The macOS installer is not implemented; "
                           "no files were changed.", "plan": plan}

    def install_plan(self, family=None, model=None, context=None, vision="no", kv="int8", gpu=None, gpus=None,
                     backend="auto", low_ram="auto", port=None, data_dir=None):
        s = self.s
        models, families, contexts, _source = s.tables()
        if family is not None and family not in families:
            raise ToolError(f"unknown model family {family!r}; choose from {', '.join(families)}")
        if model is not None and model not in models:
            raise ToolError(f"unknown model size {model!r}; choose from {', '.join(models)}")
        if family == "unsloth" or model == "UD-Q4_K_XL":
            raise ToolError("Unsloth UD-Q4_K_XL is not supported in the macOS port")
        if backend not in (None, "auto", "metal"):
            raise ToolError("a Mac uses the Metal backend")
        if gpu is not None or gpus is not None:
            raise ToolError("GPU selectors are not supported in the macOS setup")
        if low_ram not in (None, "auto", "on", "off"):
            raise ToolError("low_ram must be one of: auto, on, off")
        if context is not None and context not in contexts:
            raise ToolError(f"context must be one of: {', '.join(map(str, contexts))}")
        if port is not None and not 1024 <= port <= 65535:
            raise ToolError("port must be between 1024 and 65535")
        if MACOS is None:
            raise ToolError("cannot verify this Mac's hardware; use a native Apple-silicon Python")
        if getattr(MACOS, "MAC", False):
            machine = platform.machine().lower()
            if machine not in {"arm64", "aarch64"}:
                raise ToolError("this Python is not running as native Apple-silicon arm64")
            try:
                if MACOS.translated():
                    raise ToolError("this Python is running under Rosetta; use native Apple-silicon Python")
            except ToolError:
                raise
            except Exception as exc:                   # noqa: BLE001
                raise ToolError("cannot verify whether this Python is running under Rosetta") from exc

        if model is None and family is None:
            family, model = "qwen", "Q2_0"
        elif family is None:
            available = [name for name in families if model in s.sizes_of(models, name)]
            if not available:
                raise ToolError(f"no model family offers {model}")
            family = "qwen" if "qwen" in available else available[0]
        elif model is None:
            if family not in families:
                raise ToolError(f"unknown model family {family!r}")
            family_sizes = s.sizes_of(models, family)
            if not family_sizes:
                raise ToolError(f"{families[family]['title']} has no supported model sizes")
            model = family_sizes[0]
        sizes = s.sizes_of(models, family)
        if model not in sizes:
            raise ToolError(f"{families[family]['title']} has no {model}; its sizes: {', '.join(sizes)}")
        if vision in ("yes", "gpu", "cpu") and families[family].get("vision") is False:
            raise ToolError(f"images are not available with {families[family]['title']} yet: use vision=no")

        plan_vision = "gpu" if vision in ("yes", "gpu") else "cpu" if vision == "cpu" else "none"
        hw = s.hardware()
        recommendation = s._mac_recommendation(hw, family, vision=plan_vision, kv=kv)
        metal_setup = s._mac_planner()
        model_fit = s._mac_model_plan(metal_setup, family, model, vision=plan_vision, kv=kv) if metal_setup else None
        if context is None:
            if recommendation.get("family") == family and recommendation.get("model") == model:
                context = recommendation.get("context")
            if context is None and model_fit:
                context = model_fit.get("context")
            context = context or 32768
        exact_fit = None
        if metal_setup:
            try:
                exact_fit = s._mac_plan_at(metal_setup, family, model, context, plan_vision, kv)
            except Exception as exc:                       # noqa: BLE001
                raise ToolError(f"the Metal engine could not plan {families[family]['title']} {model} at "
                                f"{context} tokens: {exc}") from exc
            if exact_fit["class"] not in {"full", "ok", "slow", "very slow"}:
                detail = exact_fit["plan"].get("reason") or exact_fit["class"]
                raise ToolError(f"{families[family]['title']} {model} cannot run at {context} tokens on this "
                                f"Mac: {detail}")
        if low_ram == "off" and (exact_fit is None or exact_fit["class"] != "full"):
            raise ToolError("low_ram=off requires a Metal plan confirming that every expert fits in GPU memory")

        target = self.check_data_dir(data_dir) if data_dir else s.data_dir()
        if is_cloud_storage_folder(norm(target)):
            raise ToolError(f"data_dir {str(target)!r} must be outside iCloud Drive, ~/Library/CloudStorage, "
                            "~/Desktop, ~/Documents, and ~/Downloads")
        tag = families[family].get("tag", "") + model
        model_dir = target / "models" / tag
        partly = model_dir.is_dir() and any(model_dir.glob("*.gguf*"))
        needed = (8 if partly else models[model]["download_gb"] + 8) + (1 if vision in ("yes", "gpu", "cpu") else 0)
        free = s.disk_free(target)
        disk_short = (f"about {needed:.0f} GB is needed; {free:.0f} GB is free" if free is not None and free < needed
                      else None)
        fit = s._mac_plan_text({"plan": exact_fit["plan"], "class": exact_fit["class"],
                                "context": context}) if exact_fit else "unavailable: Metal planner is not implemented"
        title = f"{families[family]['title']} {model}"
        plan = {
            "title": title, "family": family, "model": model, "context": context,
            "images": vision or "no", "download_gb": models[model]["download_gb"],
            "partly_downloaded": partly, "fit": fit,
            "kv": exact_fit["kv"] if exact_fit else ("fp16" if context <= 8192 else kv),
            "data_folder": str(target), "disk_free_gb": free, "disk_needed_gb": round(needed),
            "disk_short": disk_short, "already_installed": (s.root / f"strata-{tag.lower()}.json").exists(),
            "installer_status": "not_implemented",
            "recommended_for_this_mac": recommendation if recommendation.get("model") else None,
        }
        if families[family].get("license"):
            plan["license"] = families[family]["license"]
        if families[family].get("experimental"):
            plan["experimental"] = True
        return plan, []

    # ---- strata_start
    def strata_start(self, model=None, port=None, wait_seconds=50) -> dict:
        s = self.s
        with s.lock:
            cfg_path = s.find_config(model)
            desc = s.describe_config(cfg_path)
            if not desc["ready"]:
                raise ToolError(f"{cfg_path.name} refers to missing files ({', '.join(desc['missing_files'])}); "
                                "the macOS installer cannot repair this configuration yet")
            port = port or desc["port"] or DEFAULT_PORT
            tracked = s.tracked_server()
            if tracked and not tracked.get("ended"):
                info = s.probe(tracked["port"])
                return {"summary": f"Strata is already {'running' if info else 'loading'} ({tracked.get('model')}, "
                                   f"port {tracked['port']}, started by this MCP server)"
                                   + ("" if tracked.get("model") == desc["model"] else
                                      ": stop it first (strata_stop) to start another model"),
                        "server": info}
            info = s.probe(port, deep=False)
            if info and not info.get("other"):
                return {"summary": f"a Strata server already runs on port {port} ({info.get('model')}), started "
                                   "outside this MCP server; stop it (close its window) or use another port",
                        "server": info}
            if info or not port_free(port):
                raise ToolError(f"port {port} is used by another program; choose another port (port=...)")
            py = s.run_python()
            if not Path(py).exists():
                raise ToolError("Strata's Python environment (.venv) is missing; the macOS installer is not implemented")
            cmd = [py, str(s.root / "serve" / "server.py"), "--engine", "strata", "--config", str(cfg_path),
                   "--port", str(port)]
            log_path = s.state_dir / "server.log"
            s.state_dir.mkdir(parents=True, exist_ok=True)
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"\n===== strata-mcp: start {desc['model']} on port {port} at "
                        f"{time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTHONIOENCODING="utf-8")
            try:
                os.utime(cfg_path)                      # the most recently used model, as setup's start marks it
            except OSError:
                pass
            p = spawn_detached(cmd, s.root, log_path, env)
            s.children.append(p)
            ident = child_identity(p)
            if ident is None:
                if p.poll() is None:
                    cleaned = stop_untracked_child(p, process_group=True)
                else:
                    cleaned = not end_group_leftovers(p.pid)
                s.reap()
                detail = ("the process group may still contain descendants; inspect the server log and stop them "
                          "before retrying" if not cleaned else "the server process group was stopped")
                raise ToolError(f"could not verify the server process identity; {detail}. Its last output:\n"
                                + "\n".join(read_tail(log_path, 15)))
            s.save_state("server", {"pid": p.pid, "ident": ident, "port": port, "model": desc["model"],
                                    "config": cfg_path.name, "log": str(log_path),
                                    "started": time.strftime("%Y-%m-%d %H:%M:%S")})
        return self.wait_ready(p, ident, port, desc, wait_seconds, log_path)

    def wait_ready(self, p, ident, port, desc, wait_seconds, log_path) -> dict:
        s = self.s
        end = time.time() + wait_seconds
        while True:
            if p.poll() is not None or not proc_alive(p.pid, ident):
                tail = read_tail(log_path, 15)
                group_remains = end_group_leftovers(p.pid)
                s.save_state("server", None)
                detail = ("\nA process group with the server's old ID still exists; it was left untouched because "
                          "ownership cannot be verified after the server exits."
                          if group_remains else "")
                raise ToolError(f"Strata stopped while starting (exit code {p.poll()}). Its last output:\n"
                                + "\n".join(tail) + detail)
            info = s.probe(port)
            if info and not info.get("other"):
                return {"summary": f"Strata is running: {info.get('model')} on http://127.0.0.1:{port}/v1 "
                                   "(OpenAI and Anthropic APIs; chat page at http://127.0.0.1:" + str(port) + "/)",
                        "server": {**info, "pid": p.pid}}
            if time.time() >= end:
                loading_note = "the unified memory and GPU may be busy meanwhile"
                return {"summary": f"Strata is loading {desc['model']} (this takes 1-3 minutes; {loading_note}). "
                                   "Check with strata_status in a minute.",
                        "loading": {"pid": p.pid, "port": port,
                                    "last_log_line": (read_tail(log_path, 1) or [None])[-1]}}
            time.sleep(1.0)

    # ---- strata_stop
    def strata_stop(self, port=None, force=False) -> dict:
        s = self.s
        with s.lock:
            tracked = s.tracked_server()
            if tracked and tracked.get("ended"):
                s.save_state("server", None)
                tracked = None
            if tracked:
                port = tracked["port"]
                key = s.api_key_for(port)
                st, body = http_json(f"http://127.0.0.1:{port}/status", api_key=key, timeout=3)
                if st == 200 and isinstance(body, dict) and body.get("busy") and not force:
                    raise ToolError("a request is running on Strata right now; wait for it, or call strata_stop "
                                    "with force=true to stop anyway")
                st, body = http_json(f"http://127.0.0.1:{port}/unload", "POST", api_key=key, timeout=120)
                unloaded = (body or {}).get("status") if isinstance(body, dict) else None
                if st == 409 and not force:
                    raise ToolError("Strata is busy (a request is running or queued); call again with force=true "
                                    "to stop anyway")
                proc_terminate(tracked["pid"], tracked["ident"])   # POSIX: SIGTERM = the server's own Ctrl+C path
                clean = wait_gone(tracked["pid"], tracked["ident"], 30)
                if not clean:
                    proc_kill_tree(tracked["pid"], tracked["ident"])
                    wait_gone(tracked["pid"], tracked["ident"], 15)
                s.reap()
                gone = not proc_alive(tracked["pid"], tracked["ident"])
                group_remains = end_group_leftovers(tracked["pid"]) if gone else False
                if gone:
                    s.save_state("server", None)
                summary = ("Strata server stopped" if gone else
                           "Strata could not confirm the server stopped; see strata_logs")
                if gone and group_remains:
                    summary += (". A process group with its old ID still exists; it was left untouched because "
                                "ownership cannot be verified after the server exits")
                return {"summary": summary
                                   + (" (the engine unloaded itself first)" if unloaded == "unloaded" else ""),
                        "model": tracked.get("model"), "port": port, "engine_unloaded": unloaded,
                        "ended_cleanly": bool(clean and gone and not group_remains),
                        "process_group_may_remain": group_remains}
            for p in s.ports(port):
                info = s.probe(p, deep=False)
                if info and not info.get("other"):
                    key = s.api_key_for(p)
                    st, body = http_json(f"http://127.0.0.1:{p}/unload", "POST", api_key=key, timeout=120)
                    r = body.get("status") if isinstance(body, dict) else None
                    if st == 409:
                        raise ToolError("Strata is busy (a request is running); try again when it is done")
                    if st == 401:
                        raise ToolError("Strata (started outside this MCP server) wants an API key this folder's "
                                        "configs do not have; close its window to stop it")
                    memory_status = "its unified GPU memory is free"
                    return {"summary": f"Strata on port {p} was started outside this MCP server (its own window or a "
                                       f"service), so only its model was unloaded ({r}): {memory_status}, "
                                       "but the server still listens and loads the model again on the next request. "
                                       "Close its window to end it completely.",
                            "port": p, "engine_unloaded": r}
            return {"summary": "Strata is not running"}

    # ---- strata_logs
    def strata_logs(self, source=None, model=None, lines=80) -> dict:
        s = self.s
        if source is None:
            source = "server"
        if source == "server":
            path = s.state_dir / "server.log"
        else:
            cfg = s.find_config(model or (s.state("server").get("model")))
            raw = s.read_config(cfg).get("log")
            if not raw:
                raise ToolError(f"{cfg.name} names no engine log")
            path = Path(raw)
            if not s.allowed(path):
                raise ToolError(f"{cfg.name}'s log {raw} is outside the Strata folder and its data folder: not shown")
        if not path.exists():
            return {"summary": f"no {source} log yet ({path})", "log": str(path), "lines": []}
        tail = read_tail(path, lines)
        return {"summary": f"last {len(tail)} lines of the {source} log", "log": str(path), "lines": tail}

    # ---- strata_benchmark
    def strata_benchmark(self, port=None, max_tokens=200) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if target is None:
            raise ToolError("Strata is not running: start it first (strata_start)")
        p = target["port"]
        key = s.api_key_for(p)
        st, stat = http_json(f"http://127.0.0.1:{p}/status", api_key=key, timeout=3)
        if st == 200 and isinstance(stat, dict) and (stat.get("busy") or stat.get("queued")):
            raise ToolError("Strata is busy with a request; run the speed test when it is idle")
        body = {"model": target.get("model") or "strata", "max_tokens": max_tokens, "temperature": 0,
                "reasoning_effort": "none", "stream": False,
                "messages": [{"role": "user", "content": BENCH_PROMPT}]}
        t0 = time.time()
        st, r = http_json(f"http://127.0.0.1:{p}/v1/chat/completions", "POST", body, api_key=key, timeout=600)
        wall = time.time() - t0
        if st != 200 or not isinstance(r, dict):
            msg = (r or {}).get("error", {}).get("message") if isinstance(r, dict) else None
            raise ToolError(f"the request failed (HTTP {st}): {msg or 'no answer'}")
        usage, tm = r.get("usage") or {}, r.get("timings") or {}
        n = usage.get("completion_tokens") or tm.get("predicted_n") or 0
        out = {"model": r.get("model"), "port": p, "prompt_tokens": usage.get("prompt_tokens"),
               "output_tokens": n, "wall_s": round(wall, 2),
               "output_tok_s": round(tm["predicted_per_second"], 1) if tm.get("predicted_per_second") else None,
               "prompt_tok_s": round(tm["prompt_per_second"], 1) if tm.get("prompt_per_second") else None,
               "wall_tok_s": round(n / wall, 1) if wall > 0 and n else None,
               "finish_reason": ((r.get("choices") or [{}])[0]).get("finish_reason"),
               "settings": f"greedy (temperature 0), thinking off, max_tokens {max_tokens}, a fixed story prompt"}
        speed = out["output_tok_s"] or out["wall_tok_s"]
        out["summary"] = (f"{n} tokens at {speed} tokens/s" + (f" (prompt {out['prompt_tok_s']} tokens/s)"
                                                              if out["prompt_tok_s"] else "")
                          + f", {wall:.1f} s in total")
        out["note"] = ("output speed depends on context length, Metal memory pressure, and other GPU work; "
                       "the first request after a start is slower")
        return out

    # ---- strata_connect_info
    def strata_connect_info(self, port=None) -> dict:
        s = self.s
        target = None
        for p in s.ports(port):
            info = s.probe(p, deep=False)
            if info and not info.get("other"):
                target = info
                break
        if port is None:
            cfgs = s.configs()
            port = target["port"] if target else (s.describe_config(cfgs[0])["port"] if cfgs else DEFAULT_PORT)
        model = (target or {}).get("model") or "strata"
        key_needed = (target or {}).get("api_key_required")
        if key_needed is None:
            key_needed = any(s.describe_config(c)["api_key_set"] for c in s.configs())
        key = "<your api_key from strata-<model>.json>" if key_needed else "none"
        base = f"http://127.0.0.1:{port}"
        return {
            "summary": (f"Strata {'runs' if target else 'will run'} at {base}: OpenAI base URL {base}/v1, Anthropic "
                        f"base URL {base}" + (" (an API key is required)" if key_needed else " (any API key works)")),
            "running": bool(target), "openai_base_url": f"{base}/v1", "anthropic_base_url": base,
            "anthropic_messages_endpoint": f"{base}/v1/messages", "chat_page": base + "/", "model": model,
            "api_key": ("required: the \"api_key\" value in the Strata folder's strata-<model>.json"
                        if key_needed else "not required (send any value)"),
            "examples": {
                "claude_code": {"env": {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": key,
                                        "ANTHROPIC_MODEL": "claude-sonnet-4-5"},
                                "note": "Claude Code needs a Claude model name it knows; Strata ignores the name"},
                "codex_cli": {"file": "~/.codex/config.toml",
                              "toml": f'model = "{model}"\nmodel_provider = "strata"\n\n[model_providers.strata]\n'
                                      f'name = "Strata"\nbase_url = "{base}/v1"\nwire_api = "chat"'},
                "openai_python": (f'from openai import OpenAI\nclient = OpenAI(base_url="{base}/v1", api_key="{key}")\n'
                                  f'r = client.chat.completions.create(model="{model}", messages=[{{"role": "user", '
                                  f'"content": "Hello!"}}])\nprint(r.choices[0].message.content)'),
                "curl": (f"curl {base}/v1/chat/completions -H \"Content-Type: application/json\" "
                         + (f"-H \"Authorization: Bearer {key}\" " if key_needed else "")
                         + f"-d '{{\"model\": \"{model}\", \"messages\": [{{\"role\": \"user\", \"content\": "
                           "\"Hello!\"}]}'"),
                "openai_compatible_apps": {"base_url": f"{base}/v1", "api_key": key, "model": model,
                                           "apps": "Cursor (Settings > Models > OpenAI base URL override), "
                                                   "Continue, Open WebUI, LM Studio clients, aider "
                                                   f"(--openai-api-base {base}/v1)"}},
            "other_devices": "the server listens on 127.0.0.1 on this Mac; network access is not configured"}


# ------------------------------------------------------------------------------------------------ MCP over stdio
INSTRUCTIONS = (
    "Strata is a local language model server for Apple silicon Macs. This checkout's macOS installer and automatic "
    "model recommendation are still under development. Call strata_status and strata_models to inspect an existing "
    "installation. strata_install can show a non-binding preview, but it cannot install. strata_start only works "
    "when a compatible runtime and model configuration are already present. Use strata_stop, strata_logs, and "
    "strata_connect_info for an existing server."
)


class McpServer:
    def __init__(self, strata: Strata, out=None):
        self.strata = strata
        self.tools = Tools(strata)
        self.out = out
        self.write_lock = threading.Lock()
        self.protocol = PROTOCOL

    def send(self, msg: dict) -> None:
        data = (json.dumps(msg, ensure_ascii=False) + "\n").encode("utf-8")
        with self.write_lock:
            self.out.write(data)
            self.out.flush()

    @staticmethod
    def result(rid, result):
        return {"jsonrpc": "2.0", "id": rid, "result": result}

    @staticmethod
    def error(rid, code, message):
        return {"jsonrpc": "2.0", "id": rid, "error": {"code": code, "message": message}}

    def handle(self, msg) -> dict | None:
        """One JSON-RPC message -> its response (None for a notification or a response)."""
        if not isinstance(msg, dict) or msg.get("jsonrpc") != "2.0":
            return self.error(msg.get("id") if isinstance(msg, dict) else None, -32600, "invalid request")
        if "method" not in msg:
            return None                                 # a response to something we never ask: ignored
        method, rid, params = msg["method"], msg.get("id"), msg.get("params") or {}
        if "id" not in msg:
            return None                                 # notifications (initialized, cancelled ...): nothing to say
        if not isinstance(params, dict):
            return self.error(rid, -32602, "params must be an object")
        if method == "initialize":
            asked = str(params.get("protocolVersion") or PROTOCOL)
            self.protocol = asked if asked in PROTOCOLS else PROTOCOL
            return self.result(rid, {"protocolVersion": self.protocol,
                                     "capabilities": {"tools": {"listChanged": False}},
                                     "serverInfo": {"name": "strata", "title": "Strata", "version": VERSION},
                                     "instructions": INSTRUCTIONS})
        if method == "ping":
            return self.result(rid, {})
        if method == "tools/list":
            return self.result(rid, {"tools": self.tools.defs()})
        if method == "tools/call":
            name = params.get("name")
            try:
                res = self.tools.call(name, params.get("arguments"))
                return self.result(rid, self.content(res, error=False))
            except KeyError:
                return self.error(rid, -32602, f"unknown tool: {name}")
            except ToolError as e:
                return self.result(rid, self.content({"error": str(e)}, error=True))
            except Exception as e:                      # noqa: BLE001 - a bug must not end the server
                log("tool failed:\n" + traceback.format_exc())
                return self.result(rid, self.content({"error": f"internal error in {name}: {e}"}, error=True))
        if method in ("resources/list", "prompts/list"):
            return self.result(rid, {method.split("/")[0]: []})
        return self.error(rid, -32601, f"method not found: {method}")

    def content(self, res: dict, error: bool) -> dict:
        text = json.dumps(res, indent=1, ensure_ascii=False)
        out = {"content": [{"type": "text", "text": text}], "isError": error}
        if self.protocol >= "2025-06-18":
            out["structuredContent"] = res
        return out

    def serve(self, inp) -> None:
        """Read JSON-RPC lines until stdin closes; tool calls run in their own threads (a start waits for the model
        while pings and status calls are still answered)."""
        threads = []
        for raw in inp:
            line = raw.decode("utf-8", "replace").strip() if isinstance(raw, bytes) else raw.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except ValueError:
                self.send(self.error(None, -32700, "parse error"))
                continue
            batch = isinstance(msg, list)
            if batch or (isinstance(msg, dict) and msg.get("method") == "tools/call"):
                t = threading.Thread(target=self._run, args=(msg,), daemon=True)
                t.start()
                threads.append(t)
                threads = [x for x in threads if x.is_alive()]
            else:
                self._run(msg)
        for t in threads:
            t.join(timeout=5)

    def _run(self, msg) -> None:
        if isinstance(msg, list):
            out = [r for r in (self.handle(m) for m in msg) if r is not None]
            if out:
                data = (json.dumps(out, ensure_ascii=False) + "\n").encode("utf-8")
                with self.write_lock:
                    self.out.write(data)
                    self.out.flush()
            return
        r = self.handle(msg)
        if r is not None:
            self.send(r)


def macos_host_error() -> str | None:
    if sys.platform != "darwin":
        return "Strata supports macOS only"
    if sys.version_info < (3, 10):
        return "Python 3.10 or newer is needed"
    if MACOS is None or not getattr(MACOS, "MAC", False):
        return "macOS hardware probes are unavailable"
    if platform.machine().lower() not in {"arm64", "aarch64"}:
        return "Strata needs a native Apple-silicon Python; Intel and Rosetta are not supported"
    try:
        if MACOS.translated():
            return "Strata needs a native Apple-silicon Python; Rosetta is not supported"
        if MACOS.sysctl_int("hw.optional.arm64", 0) != 1:
            return "Strata needs an Apple-silicon Mac (M1 or newer)"
        version = MACOS.macos_version()
    except Exception:                                  # noqa: BLE001
        return "Could not verify the Apple-silicon Mac and macOS version"
    if not version or version[0] < 27:
        rendered = ".".join(str(part) for part in version) or "unknown"
        return f"Strata needs macOS 27 or newer; this Mac has macOS {rendered}"
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Strata's MCP server (stdio): install, start, check and stop Strata "
                                             "from an AI assistant. This checkout supports macOS only; see docs/mac/07-server-setup-port.md.")
    ap.add_argument("--root", help="the Strata folder (default: the one this file is in)")
    a = ap.parse_args(argv)
    error = macos_host_error()
    if error:
        log(error)
        return 2
    root = Path(a.root).expanduser().resolve() if a.root else DEFAULT_ROOT
    if not (root / "setup.py").is_file() or not (root / "serve" / "server.py").is_file():
        log(f"{root} is not a Strata folder (no setup.py / serve/server.py)")
        return 2
    out = sys.stdout.buffer
    sys.stdout = sys.stderr                             # stray prints must never corrupt the protocol stream
    server = McpServer(Strata(root), out)
    log(f"ready (Strata folder {root})")
    try:
        server.serve(sys.stdin.buffer)
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
