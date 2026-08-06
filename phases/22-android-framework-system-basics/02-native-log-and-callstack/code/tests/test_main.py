import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import NativeModuleConfig
except ImportError:
    NativeModuleConfig = None


class TestNativeLogAndCallstack(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(NativeModuleConfig, "尚未实现 Native 日志模块配置检查")

    def test_debug_logging_is_enabled_when_log_ndebug_zero(self):
        self._require()
        self.assertTrue(NativeModuleConfig(log_ndebug=0).debug_logging_enabled())

    def test_debug_logging_is_disabled_when_log_ndebug_one(self):
        self._require()
        self.assertFalse(NativeModuleConfig(log_ndebug=1).debug_logging_enabled())

    def test_callstack_requires_libutils(self):
        self._require()
        result = NativeModuleConfig(log_ndebug=0, libs={"liblog"}, uses_callstack=True).validate()
        self.assertFalse(result.ok)
        self.assertIn("libutils", result.messages)

    def test_log_api_requires_liblog(self):
        self._require()
        result = NativeModuleConfig(log_ndebug=0, libs={"libutils"}, uses_log=True).validate()
        self.assertFalse(result.ok)
        self.assertIn("liblog", result.messages)

    def test_complete_native_config_is_valid(self):
        self._require()
        result = NativeModuleConfig(log_ndebug=0, libs={"libutils", "liblog"}, uses_callstack=True, uses_log=True).validate()
        self.assertTrue(result.ok)

    def test_unused_libcutils_is_not_required(self):
        self._require()
        result = NativeModuleConfig(log_ndebug=0, libs={"libutils", "liblog"}, uses_callstack=True, uses_log=True).validate()
        self.assertNotIn("libcutils", result.messages)

