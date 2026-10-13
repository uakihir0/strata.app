"""Hardware readings for the macOS Monitor tab.

A background thread samples once a second and keeps the last 60 readings for sparklines. Metal and unified-memory
readings come from IORegistry and the system default Metal device; CPU, RAM and disk readings use psutil. A failed
probe leaves its values empty and never stops the server.
"""
from __future__ import annotations

import collections
import os
import platform
import sys
import threading
import time

from serve import macos

HISTORY = 60
MAC = sys.platform == "darwin"


class _Apple:
    """Apple GPU load and unified memory; macOS exposes no public GPU temperature, power or PCIe readings."""

    MIN_INTERVAL_S = 2.0

    def __init__(self, index=0, backend=None):
        devices = macos.metal_devices() if backend in (None, "metal") and MAC else []
        self.dev = devices[0] if devices else None
        self._at = 0.0
        self._last = {}

    def ok(self):
        return self.dev is not None

    def name(self):
        return self.dev["name"] if self.dev else None

    def read(self):
        now = time.monotonic()
        if now - self._at >= self.MIN_INTERVAL_S:
            self._last, self._at = macos.ioreg_gpu(), now
        stats = self._last
        return {"util": stats.get("util"), "mem_used": stats.get("in_use"),
                "mem_total": self.dev["working_set"] if self.dev else None, "temp": None, "power": None,
                "power_limit": None, "pcie_gen": None, "pcie_gen_max": None, "pcie_width": None,
                "pcie_rx_mb": None, "pcie_tx_mb": None, "recovery": stats.get("recovery")}


def gpu_reader(index=0, backend=None):
    """Return a Metal reader only when the requested backend is Metal."""
    return _Apple(index, backend)


def free_unified_memory_mib(index=0, backend=None):
    """Advisory estimate of GPU-usable unified memory; it is not a safe model-load gate."""
    if backend not in (None, "metal") or not MAC:
        return None
    devices = macos.metal_devices()
    if not devices:
        return None
    in_use = macos.ioreg_gpu().get("in_use")
    if in_use is None:
        return None
    try:
        import psutil
        available = psutil.virtual_memory().available
    except (ImportError, OSError, RuntimeError):
        return None
    return max(0, int(min(devices[0]["working_set"] - in_use, available) >> 20))


def _cpu_name():
    """On a Mac the brand string only: platform.processor() would report "arm" there (07 section 7.2)."""
    if MAC:
        return macos.cpu_brand() or None
    return platform.processor() or None


class Telemetry:
    def __init__(self, extra=None, backend=None):
        """`extra()` supplies server readings such as token rates; the only supported GPU is Apple Metal."""
        self.extra = extra
        self.lock = threading.Lock()
        self.now: dict = {}
        self.hist = collections.defaultdict(lambda: collections.deque(maxlen=HISTORY))
        self.gpu = gpu_reader(backend=backend)
        try:
            import psutil
            self.ps = psutil
        except ImportError:
            self.ps = None
        dev = self.gpu.dev if isinstance(self.gpu, _Apple) else None
        self.static = {
            "gpu_name": self.gpu.name() if self.gpu.ok() else None,
            "gpu_count": 1 if self.gpu.ok() else 0,
            "cpu_name": _cpu_name(),
            "cores": (self.ps.cpu_count(logical=False) if self.ps else None) or None,
            "threads": os.cpu_count(),
            "psutil": self.ps is not None,
        }
        if MAC:
            self.static.update({"unified": True, "gpu_cores": dev.get("cores") if dev else None,
                                "perflevels": macos.perf_levels(), "computer": "Mac"})
        self._disk_prev = None
        threading.Thread(target=self._loop, daemon=True).start()

    def _disk(self):
        if not self.ps:
            return None, None
        try:
            c = self.ps.disk_io_counters()
        except (OSError, RuntimeError):
            return None, None
        t = time.time()
        prev, self._disk_prev = self._disk_prev, (t, c.read_bytes, c.write_bytes)
        if prev is None or t <= prev[0]:
            return None, None
        dt = t - prev[0]
        return (c.read_bytes - prev[1]) / dt / 2**20, (c.write_bytes - prev[2]) / dt / 2**20

    def sample(self):
        sample = {}
        if self.gpu.ok():
            sample.update({f"gpu_{key}": value for key, value in self.gpu.read().items()})
        if self.ps:
            try:
                sample["cpu"] = self.ps.cpu_percent(interval=None)
                vm = self.ps.virtual_memory()
                sample["ram_used"], sample["ram_total"] = vm.total - vm.available, vm.total
            except (OSError, RuntimeError):
                pass
        sample["disk_read_mb"], sample["disk_write_mb"] = self._disk()
        if MAC:
            thermal = macos.thermal_state()
            sample.update({"thermal": thermal, "thermal_level": {"nominal": 0, "fair": 1, "serious": 2,
                           "critical": 3}.get(thermal), "mem_pressure": macos.pressure_level(),
                           "swap_used": macos.swap_used()})
        if self.extra:
            try:
                sample.update(self.extra())
            except Exception:  # noqa: BLE001 - telemetry must never take the server down
                pass
        return sample

    def _loop(self):
        while True:
            sample = self.sample()
            with self.lock:
                self.now = sample
                keys = ["gpu_util", "gpu_mem_used", "gpu_temp", "gpu_power", "gpu_pcie_rx_mb", "cpu", "ram_used",
                        "disk_read_mb", "tok_s", "prefill_tok_s_mean"]
                if MAC:
                    keys.extend(("thermal_level", "swap_used"))
                for key in keys:
                    value = sample.get(key)
                    self.hist[key].append(round(value, 2) if isinstance(value, float) else value)
            time.sleep(1.0)

    def snapshot(self):
        with self.lock:
            return {"now": dict(self.now), "history": {key: list(values) for key, values in self.hist.items()},
                    "static": dict(self.static)}
