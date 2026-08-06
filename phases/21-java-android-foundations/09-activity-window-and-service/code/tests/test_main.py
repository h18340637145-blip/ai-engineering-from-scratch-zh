import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import LaunchModeResolver, LifecyclePlanner, service_mode
except ImportError:
    LaunchModeResolver = LifecyclePlanner = service_mode = None


class TestActivityWindowAndService(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(LifecyclePlanner, "尚未实现组件生命周期模型")

    def test_foreground_path_has_create_start_resume(self):
        self._require()
        self.assertEqual(LifecyclePlanner.foreground_path(), ["onCreate", "onStart", "onResume"])

    def test_return_path_has_restart_before_resume(self):
        self._require()
        self.assertEqual(LifecyclePlanner.return_path(), ["onRestart", "onStart", "onResume"])

    def test_stopped_process_may_die_without_destroy(self):
        self._require()
        self.assertTrue(LifecyclePlanner.needs_saved_state("stopped"))

    def test_single_top_reuses_top_instance(self):
        self._require()
        self.assertEqual(LaunchModeResolver.resolve("singleTop", True, False), "reuse-onNewIntent")

    def test_single_task_reuses_existing_task_instance(self):
        self._require()
        self.assertEqual(LaunchModeResolver.resolve("singleTask", False, True), "reuse-clear-above")

    def test_long_work_uses_foreground_service(self):
        self._require()
        self.assertEqual(service_mode("long-running"), "foreground service")
        self.assertEqual(service_mode("client-bound"), "bound service")

