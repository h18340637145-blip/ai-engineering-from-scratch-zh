import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import IncidentTimeline, build_monkey_command, protolog_command
except ImportError:
    IncidentTimeline = build_monkey_command = protolog_command = None


class TestSystemDiagnosticsAndStability(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(IncidentTimeline, "尚未实现系统诊断时间线")

    def test_anr_timeline_identifies_binder_wait(self):
        self._require()
        result = IncidentTimeline.analyze(["am_anr", "BinderProxy.transact"])
        self.assertEqual(result["kind"], "ANR")
        self.assertIn("Binder", result["next_step"])

    def test_fatal_exception_is_classified_as_crash(self):
        self._require()
        self.assertEqual(IncidentTimeline.analyze(["FATAL EXCEPTION"])["kind"], "Crash")

    def test_lmkd_kill_is_classified_as_memory_reclaim(self):
        self._require()
        self.assertEqual(IncidentTimeline.analyze(["lmkd: Killing"])["kind"], "LMKD")

    def test_bugreport_starts_with_timestamp_search(self):
        self._require()
        self.assertEqual(IncidentTimeline.first_action("bugreport"), "locate timestamp")

    def test_monkey_command_scopes_package_and_throttle(self):
        self._require()
        command = build_monkey_command("com.example.app", 100, 36000)
        self.assertIn("-p com.example.app", command)
        self.assertIn("--throttle 100", command)

    def test_protolog_command_can_disable_group_after_diagnosis(self):
        self._require()
        self.assertEqual(protolog_command("WM_DEBUG_ANIM", False), "adb shell wm logging disable-text WM_DEBUG_ANIM")

