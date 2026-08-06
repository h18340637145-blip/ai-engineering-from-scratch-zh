import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import IncidentClassifier, leak_remedy
except ImportError:
    IncidentClassifier = leak_remedy = None


class TestMemoryOomAndAnr(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(IncidentClassifier, "尚未实现内存与响应事件分类器")

    def test_fatal_exception_is_crash(self):
        self._require()
        self.assertEqual(IncidentClassifier.classify("FATAL EXCEPTION: main"), "Crash")

    def test_input_timeout_is_anr(self):
        self._require()
        self.assertEqual(IncidentClassifier.classify("Input dispatching timed out"), "ANR")

    def test_allocation_failure_is_oom(self):
        self._require()
        self.assertEqual(IncidentClassifier.classify("OutOfMemoryError"), "OOM")

    def test_lmkd_kill_is_not_java_crash(self):
        self._require()
        self.assertEqual(IncidentClassifier.classify("lmkd: Killing process"), "LMKD")

    def test_static_activity_reference_uses_application_context(self):
        self._require()
        self.assertEqual(leak_remedy("static-activity"), "Application Context")

    def test_delayed_activity_callback_must_be_removed(self):
        self._require()
        self.assertEqual(leak_remedy("delayed-handler"), "remove callbacks")

