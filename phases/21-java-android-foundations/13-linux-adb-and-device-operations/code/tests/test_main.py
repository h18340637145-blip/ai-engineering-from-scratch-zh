import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import AdbCommandPlanner, is_safe_shell_command
except ImportError:
    AdbCommandPlanner = is_safe_shell_command = None


class TestLinuxAdbAndDeviceOperations(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(AdbCommandPlanner, "尚未实现 ADB 命令规划器")

    def test_selected_serial_is_added_to_adb_command(self):
        self._require()
        self.assertEqual(
            AdbCommandPlanner.for_device("emulator-5554", "shell pm list packages"),
            "adb -s emulator-5554 shell pm list packages",
        )

    def test_no_serial_uses_default_device(self):
        self._require()
        self.assertEqual(AdbCommandPlanner.for_device(None, "devices"), "adb devices")

    def test_logcat_can_filter_by_package_pid(self):
        self._require()
        self.assertIn("pidof -s com.example.app", AdbCommandPlanner.logcat_for("com.example.app"))

    def test_dumpsys_current_focus_command_is_available(self):
        self._require()
        self.assertIn("mCurrentFocus", AdbCommandPlanner.current_focus())

    def test_unbounded_recursive_delete_is_unsafe(self):
        self._require()
        self.assertFalse(is_safe_shell_command("rm -rf *"))

    def test_targeted_non_destructive_search_is_safe(self):
        self._require()
        self.assertTrue(is_safe_shell_command("find /system -name bootanimation.zip"))

