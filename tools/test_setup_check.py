import io
import unittest
from unittest import mock

import setup
from serve import macos


GIB = 1024 ** 3


class ReadOnlyMacSetupCheck(unittest.TestCase):
    def setUp(self):
        self.device = {
            "name": "Apple M5 Pro",
            "unified": True,
            "families": ["apple7", "metal4"],
            "cores": 20,
            "working_set": 55 * GIB,
            "wired_limit": 0,
        }
        self.patch(setup, "MAC", new=True)
        self.machine = self.patch(setup.platform, "machine", return_value="arm64")
        self.translated = self.patch(macos, "translated", return_value=False)
        self.version = self.patch(macos, "macos_version", return_value=(27, 0))
        self.devices = self.patch(macos, "metal_devices", return_value=(self.device,))
        self.device_problem = self.patch(macos, "device_problem", return_value=None)
        self.ram = self.patch(macos, "ram_bytes", return_value=64 * GIB)
        self.low_power = self.patch(macos, "low_power_mode", return_value=False)
        self.cpu = self.patch(macos, "cpu_brand", return_value="Apple M5 Pro")
        self.levels = self.patch(
            macos,
            "perf_levels",
            return_value=[
                {"name": "Super", "cores": 6},
                {"name": "Performance", "cores": 12},
            ],
        )

    def patch(self, target, attribute, **kwargs):
        patcher = mock.patch.object(target, attribute, **kwargs)
        replacement = patcher.start()
        self.addCleanup(patcher.stop)
        return replacement

    def run_check(self, *arguments):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(setup.sys, "argv", ["setup.py", "--check-host", *arguments]),
            mock.patch.object(setup.sys, "stdout", stdout),
            mock.patch.object(setup.sys, "stderr", stderr),
        ):
            code = setup.main()
        return code, stdout.getvalue(), stderr.getvalue()

    def test_supported_host_reports_only_read_only_hardware_facts(self):
        with mock.patch.object(setup.subprocess, "run") as run:
            code, stdout, stderr = self.run_check()

        self.assertEqual(0, code, stderr)
        self.assertIn("GPU: Apple M5 Pro, 20-core GPU", stdout)
        self.assertIn("RAM: 64 GiB", stdout)
        self.assertIn("CPU: Apple M5 Pro (6 Super + 12 Performance cores)", stdout)
        self.assertIn("This check is read-only", stdout)
        run.assert_not_called()

    def test_low_memory_is_reported_without_an_interactive_consent_prompt(self):
        self.ram.stop()
        self.ram = self.patch(macos, "ram_bytes", return_value=16 * GIB)
        with mock.patch("builtins.input", side_effect=AssertionError("check must not prompt")):
            code, stdout, stderr = self.run_check()

        self.assertEqual(0, code, stderr)
        self.assertIn("below 32 GiB", stdout)
        self.assertIn("performance may be very slow", stdout)

    def test_low_power_mode_is_reported_and_unknown_mode_is_explicit(self):
        self.low_power.stop()
        self.low_power = self.patch(macos, "low_power_mode", return_value=True)
        code, stdout, stderr = self.run_check()
        self.assertEqual(0, code, stderr)
        self.assertIn("Low Power Mode is on", stdout)

        self.low_power.stop()
        self.low_power = self.patch(macos, "low_power_mode", return_value=None)
        code, stdout, stderr = self.run_check()
        self.assertEqual(0, code, stderr)
        self.assertIn("Low Power Mode status could not be read", stdout)

    def test_non_macos_host_stops_before_mac_probes(self):
        self.patch(setup, "MAC", new=False)
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("supports macOS only", stderr)
        self.assertEqual("", stdout)
        self.translated.assert_not_called()
        self.version.assert_not_called()
        self.devices.assert_not_called()

    def test_intel_mac_is_refused_before_version_or_gpu_probes(self):
        self.patch(setup.platform, "machine", return_value="x86_64")
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("Apple silicon", stderr)
        self.assertIn("Intel CPU", stderr)
        self.version.assert_not_called()
        self.devices.assert_not_called()

    def test_rosetta_is_refused_before_version_or_gpu_probes(self):
        self.patch(setup.platform, "machine", return_value="x86_64")
        self.patch(macos, "translated", return_value=True)
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("under Rosetta", stderr)
        self.assertIn("install an arm64 Python", stderr)
        self.version.assert_not_called()
        self.devices.assert_not_called()

    def test_architecture_is_read_before_rosetta_and_version_probes(self):
        order = []
        self.machine.side_effect = lambda: (order.append("machine"), "arm64")[1]
        self.translated.side_effect = lambda: (order.append("translated"), False)[1]
        code, stdout, stderr = self.run_check()

        self.assertEqual(0, code, stderr)
        self.assertEqual(["machine", "translated"], order)

    def test_unknown_rosetta_and_macos_version_probes_fail_closed(self):
        self.patch(macos, "translated", return_value=None)
        code, stdout, stderr = self.run_check()
        self.assertEqual(1, code)
        self.assertIn("could not determine whether Python is running under Rosetta", stderr)

        self.patch(macos, "translated", return_value=False)
        self.patch(macos, "macos_version", return_value=("27", 0))
        code, stdout, stderr = self.run_check()
        self.assertEqual(1, code)
        self.assertIn("macOS version cannot be read", stderr)

    def test_old_macos_is_refused_before_gpu_probe(self):
        self.patch(macos, "macos_version", return_value=(26, 9))
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("macOS 27 or newer", stderr)
        self.devices.assert_not_called()

    def test_missing_or_multiple_gpu_devices_are_refused(self):
        for devices, message in (
            ((), "no Metal GPU was found"),
            ((self.device, self.device), "must expose one default Metal GPU"),
        ):
            with self.subTest(message=message):
                devices_patch = mock.patch.object(macos, "metal_devices", return_value=devices)
                devices_patch.start()
                code, stdout, stderr = self.run_check()
                devices_patch.stop()
                self.assertEqual(1, code)
                self.assertIn(message, stderr)

    def test_malformed_device_is_refused_without_calling_capability_probe(self):
        device = mock.patch.object(macos, "metal_devices", return_value=({"name": "incomplete"},))
        problem = mock.patch.object(macos, "device_problem")
        device.start()
        device_problem = problem.start()
        self.addCleanup(device.stop)
        self.addCleanup(problem.stop)
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("incomplete or malformed capabilities", stderr)
        device_problem.assert_not_called()

    def test_metal_capability_refusal_is_reported(self):
        self.patch(macos, "device_problem", return_value="GPU lacks metal4")
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("GPU lacks metal4", stderr)

    def test_cuda_hip_and_multi_gpu_selectors_are_refused_after_host_report(self):
        for arguments, message in (
            (("--backend", "cuda"), "a Mac uses Metal"),
            (("--backend=hip",), "a Mac uses Metal"),
            (("--gpus", "0"), "--gpus and --layer-split do not apply"),
            (("--layer-split", "10"), "--gpus and --layer-split do not apply"),
            (("--gpu", "1"), "one GPU (number 0)"),
            (("--model", "example.gguf"), "installation options remain disabled"),
        ):
            with self.subTest(arguments=arguments):
                code, stdout, stderr = self.run_check(*arguments)
                self.assertEqual(1, code)
                self.assertIn("CPU:", stdout)
                self.assertIn(message, stderr)

    def test_gpu_zero_and_metal_backend_are_accepted_as_mac_compatible(self):
        code, stdout, stderr = self.run_check("--gpu", "0", "--backend", "metal")
        self.assertEqual(0, code, stderr)
        self.assertIn("host check passed", stdout)

    def test_unreadable_memory_is_refused(self):
        self.patch(macos, "ram_bytes", return_value=0)
        code, stdout, stderr = self.run_check()

        self.assertEqual(1, code)
        self.assertIn("physical memory cannot be read", stderr)


if __name__ == "__main__":
    unittest.main()
