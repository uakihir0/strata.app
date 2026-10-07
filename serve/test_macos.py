import os
import select
import signal
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

from serve import macos


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


if __name__ == "__main__":
    unittest.main()
