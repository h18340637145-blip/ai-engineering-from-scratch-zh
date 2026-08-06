import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import PropertyPolicy, SettingsObserver
except ImportError:
    PropertyPolicy = SettingsObserver = None


class TestSystemPropertiesAndSettings(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(PropertyPolicy, "尚未实现系统属性与 Settings 模型")

    def test_ro_property_is_never_writable_after_boot(self):
        self._require()
        self.assertFalse(PropertyPolicy.can_write("ro.example.flag", is_system=True))

    def test_system_component_can_write_persist_property(self):
        self._require()
        self.assertTrue(PropertyPolicy.can_write("persist.example.flag", is_system=True))

    def test_third_party_cannot_write_persist_property(self):
        self._require()
        self.assertFalse(PropertyPolicy.can_write("persist.example.flag", is_system=False))

    def test_observer_receives_registered_uri_change(self):
        self._require()
        observer = SettingsObserver()
        observer.register("settings://system/example_key")
        self.assertTrue(observer.write("settings://system/example_key", "1"))
        self.assertEqual(observer.events, [("settings://system/example_key", "1")])

    def test_unregistered_uri_does_not_notify_observer(self):
        self._require()
        observer = SettingsObserver()
        self.assertFalse(observer.write("settings://system/example_key", "1"))

    def test_unregister_stops_future_notifications(self):
        self._require()
        observer = SettingsObserver()
        uri = "settings://system/example_key"
        observer.register(uri)
        observer.unregister(uri)
        self.assertFalse(observer.write(uri, "2"))

