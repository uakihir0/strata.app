"""macOS probes for setup.py, serve/server.py, serve/telemetry.py and tools/strata_mcp.py (standard library only).

Reference implementation for docs/mac/07-server-setup-port.md, section 3.  Copy it to serve/macos.py.  Every function is
safe to call on any OS: off macOS, or when a probe fails, it returns the documented empty value and never raises.
Run it directly to print every probe:  python3 serve/macos.py
"""
from __future__ import annotations

import ctypes
import ctypes.util
import functools
import os
import plistlib
import struct
import subprocess
import sys

MAC = sys.platform == "darwin"

FAMILIES = {"apple7": 1007, "apple8": 1008, "apple9": 1009, "apple10": 1010, "metal3": 5001, "metal4": 5002}
REQUIRED_FAMILIES = ("apple7", "metal4")          # the Metal engine's floor (docs/mac/06, section 2.3, Context::open)
PRESSURE = {1: "normal", 2: "warn", 4: "critical"}
THERMAL = {0: "nominal", 1: "fair", 2: "serious", 3: "critical"}


# ------------------------------------------------------------------------------------------------ sysctl
@functools.lru_cache(maxsize=None)
def _libc():
    lib = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    lib.sysctlbyname.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t), ctypes.c_void_p,
                                 ctypes.c_size_t]
    lib.sysctlbyname.restype = ctypes.c_int
    return lib


def sysctl_raw(name: str) -> bytes | None:
    """The raw bytes of a sysctl, or None when it does not exist (or off macOS)."""
    if not MAC:
        return None
    try:
        lib, size = _libc(), ctypes.c_size_t(0)
        if lib.sysctlbyname(name.encode(), None, ctypes.byref(size), None, 0) != 0:
            return None
        buf = ctypes.create_string_buffer(max(size.value, 1))
        if lib.sysctlbyname(name.encode(), buf, ctypes.byref(size), None, 0) != 0:
            return None
        return buf.raw[:size.value]
    except (OSError, AttributeError):
        return None


def sysctl_int(name: str, default=None):
    raw = sysctl_raw(name)
    if raw is None:
        return default
    if len(raw) == 4:
        return struct.unpack("<i", raw)[0]
    if len(raw) == 8:
        return struct.unpack("<Q", raw)[0]
    return default


def sysctl_str(name: str, default=None):
    raw = sysctl_raw(name)
    return raw.split(b"\0", 1)[0].decode("utf-8", "replace") if raw is not None else default


# ------------------------------------------------------------------------------------------------ machine facts
def ram_bytes() -> int:
    """Physical RAM in bytes (hw.memsize): exact, unlike /proc/meminfo's MemTotal.  0 off macOS."""
    return sysctl_int("hw.memsize", 0) or 0


@functools.lru_cache(maxsize=None)
def macos_version() -> tuple:
    """(27, 0) for macOS 27.0, () off macOS.  kern.osproductversion first: platform.mac_ver() can report a
    compatibility version (10.16 / 16.0) to a Python built against an old SDK."""
    v = sysctl_str("kern.osproductversion") or (__import__("platform").mac_ver()[0] if MAC else "")
    try:
        return tuple(int(x) for x in v.split("."))
    except ValueError:
        return ()


def translated() -> bool:
    """True when this process runs under Rosetta 2 (an x86_64 Python on Apple silicon)."""
    return sysctl_int("sysctl.proc_translated", 0) == 1


def cpu_brand() -> str:
    return sysctl_str("machdep.cpu.brand_string") or ""


def perf_levels() -> list:
    """[{"name": "Super", "cores": 6}, {"name": "Performance", "cores": 12}] on an M5 Pro; perflevel0 is the
    fastest class.  [] off macOS."""
    n = sysctl_int("hw.nperflevels", 0) or 0
    return [{"name": sysctl_str(f"hw.perflevel{i}.name") or f"level {i}",
             "cores": sysctl_int(f"hw.perflevel{i}.physicalcpu", 0) or 0} for i in range(n)]


def fast_cores() -> int:
    """Physical cores of the fastest class (hw.perflevel0.physicalcpu): the vision helper's CPU threads."""
    lv = perf_levels()
    return lv[0]["cores"] if lv else max(1, (os.cpu_count() or 2) // 2)


def arm_features() -> dict:
    return {f: sysctl_int(f"hw.optional.arm.FEAT_{f}", 0) == 1 for f in ("DotProd", "I8MM", "BF16", "SME")}


def pressure_level() -> str | None:
    """"normal", "warn" or "critical" (kern.memorystatus_vm_pressure_level), None when unreadable."""
    return PRESSURE.get(sysctl_int("kern.memorystatus_vm_pressure_level", 0))


def swap_used() -> int | None:
    """Bytes of swap in use (vm.swapusage, struct xsw_usage), None when unreadable."""
    raw = sysctl_raw("vm.swapusage")
    if raw is None or len(raw) < 28:
        return None
    _total, _avail, used = struct.unpack("<QQQ", raw[:24])
    return used


def wired_limit() -> int:
    """iogpu.wired_limit_mb in bytes; 0 means the default (Metal's recommendedMaxWorkingSetSize applies)."""
    return (sysctl_int("iogpu.wired_limit_mb", 0) or 0) << 20


# ------------------------------------------------------------------------------------------------ Objective-C
@functools.lru_cache(maxsize=None)
def _objc():
    objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
    objc.sel_registerName.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    return objc


def _send(obj, sel: bytes, restype=ctypes.c_uint64, *args, argtypes=()):
    """objc_msgSend cast to the exact signature (required on arm64, where it is not variadic)."""
    objc = _objc()
    f = ctypes.cast(objc.objc_msgSend, ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes))
    return f(obj, objc.sel_registerName(sel), *args)


@functools.lru_cache(maxsize=None)
def metal_devices() -> tuple:
    """(device,) with the one Metal GPU, () off macOS or without Metal.  The dict has the keys setup's NVIDIA cards
    have (index, name, vram_gb, arch, driver) plus the Metal facts.  vram_gb is the working set W in decimal GB, for
    code that prints it; no fit decision may use it (docs/mac/07, principle P3)."""
    if not MAC:
        return ()
    try:
        metal = ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Metal.framework/Metal")
        metal.MTLCreateSystemDefaultDevice.restype = ctypes.c_void_p
        dev = metal.MTLCreateSystemDefaultDevice()
        if not dev:
            return ()
        name = _send(_send(dev, b"name", ctypes.c_void_p), b"UTF8String", ctypes.c_char_p).decode()
        fam = [f for f, n in FAMILIES.items()
               if _send(dev, b"supportsFamily:", ctypes.c_bool, n, argtypes=(ctypes.c_long,))]
        ws = int(_send(dev, b"recommendedMaxWorkingSetSize"))
        gpu = ioreg_gpu()
        return ({"index": 0, "name": name, "working_set": ws, "wired_limit": wired_limit(),
                 "max_buffer": int(_send(dev, b"maxBufferLength")),
                 "unified": bool(_send(dev, b"hasUnifiedMemory", ctypes.c_bool)), "families": fam,
                 "cores": gpu.get("cores"), "ram": ram_bytes(), "vram_gb": ws / 1e9, "arch": "apple",
                 "driver": ".".join(str(x) for x in macos_version())},)
    except (OSError, AttributeError, ValueError):
        return ()


def device_problem(dev: dict | None) -> str | None:
    """Why the Metal engine cannot run on this device, or None."""
    if not dev:
        return "no Metal GPU was found"
    if not dev["unified"]:
        return f"{dev['name']} has no unified memory"
    missing = [f for f in REQUIRED_FAMILIES if f not in dev["families"]]
    if missing:
        return f"{dev['name']} does not support " + " and ".join(missing) + " (Strata needs an M1 or newer)"
    return None


def _process_info():
    ctypes.cdll.LoadLibrary("/System/Library/Frameworks/Foundation.framework/Foundation")
    return _send(_objc().objc_getClass(b"NSProcessInfo"), b"processInfo", ctypes.c_void_p)


def thermal_state() -> str | None:
    """"nominal", "fair", "serious" or "critical" (NSProcessInfo.thermalState), None when unreadable."""
    if not MAC:
        return None
    try:
        return THERMAL.get(int(_send(_process_info(), b"thermalState", ctypes.c_long)))
    except (OSError, AttributeError):
        return None


def low_power_mode() -> bool | None:
    if not MAC:
        return None
    try:
        return bool(_send(_process_info(), b"isLowPowerModeEnabled", ctypes.c_bool))
    except (OSError, AttributeError):
        return None


# ------------------------------------------------------------------------------------------------ IORegistry
def ioreg_gpu() -> dict:
    """The GPU's IOAccelerator entry: {"model", "cores", "util", "renderer", "tiler", "in_use", "alloc",
    "recovery"}; {} when unreadable.  About 80 ms: callers sample it at most every 2 s."""
    if not MAC:
        return {}
    try:
        raw = subprocess.run(["/usr/sbin/ioreg", "-r", "-d", "1", "-w", "0", "-c", "IOAccelerator", "-a"],
                             capture_output=True, timeout=5).stdout
        entries = plistlib.loads(raw) if raw else []
    except (OSError, subprocess.SubprocessError, plistlib.InvalidFileException, ValueError):
        return {}
    for e in entries if isinstance(entries, list) else [entries]:
        ps = e.get("PerformanceStatistics") or {}
        if not ps and "gpu-core-count" not in e:
            continue
        model = e.get("model")
        return {"model": model.decode(errors="replace").rstrip("\0") if isinstance(model, bytes) else model,
                "cores": e.get("gpu-core-count"), "util": ps.get("Device Utilization %"),
                "renderer": ps.get("Renderer Utilization %"), "tiler": ps.get("Tiler Utilization %"),
                "in_use": ps.get("In use system memory"), "alloc": ps.get("Alloc system memory"),
                "recovery": ps.get("recoveryCount")}
    return {}


# ------------------------------------------------------------------------------------------------ processes
class _BsdInfo(ctypes.Structure):
    _fields_ = [("pbi_flags", ctypes.c_uint32), ("pbi_status", ctypes.c_uint32), ("pbi_xstatus", ctypes.c_uint32),
                ("pbi_pid", ctypes.c_uint32), ("pbi_ppid", ctypes.c_uint32), ("ids", ctypes.c_uint32 * 6),
                ("rfu_1", ctypes.c_uint32), ("pbi_comm", ctypes.c_char * 16), ("pbi_name", ctypes.c_char * 32),
                ("pbi_nfiles", ctypes.c_uint32), ("pbi_pgid", ctypes.c_uint32), ("pbi_pjobc", ctypes.c_uint32),
                ("e_tdev", ctypes.c_uint32), ("e_tpgid", ctypes.c_uint32), ("pbi_nice", ctypes.c_int32),
                ("pbi_start_tvsec", ctypes.c_uint64), ("pbi_start_tvusec", ctypes.c_uint64)]   # 136 bytes


def proc_identity(pid: int) -> str | None:
    """"mac:<start sec>.<start usec>" for a live process, None for a zombie or a gone one (proc_pidinfo
    PROC_PIDTBSDINFO returns 0 for both).  The start time protects against a recycled PID."""
    if not MAC:
        return None
    try:
        libproc = ctypes.CDLL(ctypes.util.find_library("proc"))
        b = _BsdInfo()
        n = libproc.proc_pidinfo(int(pid), 3, ctypes.c_uint64(0), ctypes.byref(b), ctypes.sizeof(b))
    except (OSError, AttributeError, ValueError, OverflowError):
        return None
    if n != ctypes.sizeof(b):
        return None
    return f"mac:{b.pbi_start_tvsec}.{b.pbi_start_tvusec}"


# ------------------------------------------------------------------------------------------------ small tools
def _tool(argv, timeout=30) -> subprocess.CompletedProcess | None:
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None


def strip_quarantine(*paths) -> None:
    """Remove com.apple.quarantine (recursively) from our own files; a no-op when it is absent.  Leaves
    com.apple.provenance alone."""
    for p in paths:
        if MAC and os.path.exists(p):
            _tool(["/usr/bin/xattr", "-dr", "com.apple.quarantine", str(p)])


def has_quarantine(path) -> bool:
    r = _tool(["/usr/bin/xattr", "-p", "com.apple.quarantine", str(path)]) if MAC else None
    return bool(r and r.returncode == 0)


def tm_exclude(path) -> bool:
    """Exclude a folder from Time Machine (a sticky xattr; no administrator rights needed)."""
    r = _tool(["/usr/bin/tmutil", "addexclusion", str(path)]) if MAC else None
    return bool(r and r.returncode == 0)


def firewall_on() -> bool | None:
    r = _tool(["/usr/libexec/ApplicationFirewall/socketfilterfw", "--getglobalstate"], 5) if MAC else None
    return None if not r or r.returncode != 0 else "enabled" in r.stdout.lower()


def keep_awake(pid: int | None = None) -> subprocess.Popen | None:
    """Prevent idle sleep until process `pid` (default: this one) exits.  The lid still sleeps the Mac."""
    if not MAC:
        return None
    try:
        return subprocess.Popen(["/usr/bin/caffeinate", "-i", "-w", str(pid or os.getpid())],
                                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except OSError:
        return None


def ssl_setup() -> str | None:
    """A Python from the python.org installer has no CA bundle until its "Install Certificates.command" runs, so every
    HTTPS download fails with CERTIFICATE_VERIFY_FAILED.  When the default CA file is missing and SSL_CERT_FILE is
    unset, point SSL_CERT_FILE at certifi's bundle (the venv's certifi, else pip's vendored copy).  The variable also
    reaches the tools setup runs (pip, mtp_fetch.py).  Returns the bundle used, or None when nothing was changed."""
    import ssl
    if not MAC or os.environ.get("SSL_CERT_FILE") or ssl.get_default_verify_paths().cafile:
        return None
    for mod in ("certifi", "pip._vendor.certifi"):
        try:
            where = __import__(mod, fromlist=["where"]).where()
        except (ImportError, AttributeError, OSError):
            continue
        if os.path.isfile(where):
            os.environ["SSL_CERT_FILE"] = where
            return where
    return None


def engine_pids(name: str = "strata") -> list:
    """PIDs of running processes with this exact name (another Strata engine holding GPU memory), [] when none."""
    r = _tool(["/usr/bin/pgrep", "-x", name], 5) if MAC else None
    return [int(x) for x in r.stdout.split()] if r and r.returncode == 0 else []


def xcode_problem() -> str | None:
    """Why a local engine build cannot run, or None: full Xcode selected and the Metal compiler installed."""
    if not MAC:
        return "not a Mac"
    r = _tool(["/usr/bin/xcode-select", "-p"])
    if not r or r.returncode != 0 or ".app/" not in r.stdout:
        return "Xcode is not installed or not selected (xcode-select -p does not point into Xcode.app)"
    r = _tool(["/usr/bin/xcrun", "-f", "metal"], 60)
    if not r or r.returncode != 0:
        return "Xcode's Metal toolchain is missing"
    return None


if __name__ == "__main__":
    import json
    import time
    t = time.perf_counter()
    dev = metal_devices()
    facts = {"macos": macos_version(), "translated": translated(), "ram": ram_bytes(), "cpu": cpu_brand(),
             "perf_levels": perf_levels(), "fast_cores": fast_cores(), "arm": arm_features(),
             "pressure": pressure_level(), "swap_used": swap_used(), "thermal": thermal_state(),
             "low_power": low_power_mode(), "device": dev[0] if dev else None,
             "device_problem": device_problem(dev[0] if dev else None), "ioreg": ioreg_gpu(),
             "self_identity": proc_identity(os.getpid()), "firewall_on": firewall_on(),
             "xcode_problem": xcode_problem()}
    facts["seconds"] = round(time.perf_counter() - t, 3)
    print(json.dumps(facts, indent=1, default=str))
