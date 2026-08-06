import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import FrameworkReadinessReport
except ImportError:
    FrameworkReadinessReport = None


class TestFrameworkIntegrationLab(unittest.TestCase):
    def _report(self, checks):
        self.assertIsNotNone(FrameworkReadinessReport, "尚未实现 Framework 集成验收报告")
        return FrameworkReadinessReport(checks)

    def test_all_required_checks_make_report_ready(self):
        report = self._report({"boot": True, "zygote": True, "overlay": True, "setup_wizard": True, "permissions": True})
        self.assertTrue(report.ready)

    def test_missing_overlay_blocks_readiness(self):
        report = self._report({"boot": True, "zygote": True, "overlay": False, "setup_wizard": True, "permissions": True})
        self.assertFalse(report.ready)

    def test_missing_checks_are_listed(self):
        report = self._report({"boot": True, "overlay": False})
        self.assertIn("overlay", report.missing())
        self.assertIn("zygote", report.missing())

    def test_overlay_failure_has_command_hint(self):
        report = self._report({"boot": True, "zygote": True, "overlay": False, "setup_wizard": True, "permissions": True})
        self.assertIn("cmd overlay list", report.next_command())

    def test_boot_failure_has_event_log_hint(self):
        report = self._report({"boot": False, "zygote": True, "overlay": True, "setup_wizard": True, "permissions": True})
        self.assertIn("boot_progress", report.next_command())

    def test_ready_report_returns_final_validation_command(self):
        report = self._report({"boot": True, "zygote": True, "overlay": True, "setup_wizard": True, "permissions": True})
        self.assertEqual(report.next_command(), "adb reboot && adb logcat -b events -d")

