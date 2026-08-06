import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import SystemState, GestureType, resolve_input_consumer

class TestResolveInputConsumer(unittest.TestCase):
    def _gesture_state(self, is_launcher, nav="gesture"):
        pkg = "com.android.launcher3" if is_launcher else "com.android.settings"
        return SystemState(pkg, is_launcher, nav)

    def test_launcher_swipe_up(self):
        s = self._gesture_state(True)
        result = resolve_input_consumer(s, GestureType.SWIPE_UP)
        self.assertIn("Launcher", result)

    def test_other_app_swipe_up(self):
        s = self._gesture_state(False)
        result = resolve_input_consumer(s, GestureType.SWIPE_UP)
        self.assertIn("Other", result)

    def test_swipe_left_returns_task_switch(self):
        s = self._gesture_state(False)
        result = resolve_input_consumer(s, GestureType.SWIPE_LEFT)
        self.assertIn("Task", result)

    def test_three_button_nav_bypasses_tis(self):
        s = self._gesture_state(False, "three_button")
        result = resolve_input_consumer(s, GestureType.SWIPE_UP)
        self.assertIn("三键", result)

    def test_swipe_right_returns_task_switch(self):
        s = self._gesture_state(True)
        result = resolve_input_consumer(s, GestureType.SWIPE_RIGHT)
        self.assertIn("Task", result)
