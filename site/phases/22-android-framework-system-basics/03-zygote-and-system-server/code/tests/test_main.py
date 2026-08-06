import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ZygotePlanner, ZygoteRc
except ImportError:
    ZygotePlanner = ZygoteRc = None


class TestZygoteAndSystemServer(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(ZygoteRc, "尚未实现 Zygote rc 解析器")

    def test_rc_parses_zygote_command(self):
        self._require()
        rc = ZygoteRc.parse("service zygote /system/bin/app_process64 -Xzygote /system/bin --zygote --start-system-server --socket-name=zygote")
        self.assertEqual(rc.command, "/system/bin/app_process64")

    def test_rc_detects_system_server_flag(self):
        self._require()
        rc = ZygoteRc.parse("service zygote /system/bin/app_process64 --zygote --start-system-server --socket-name=zygote")
        self.assertTrue(rc.starts_system_server)

    def test_rc_extracts_socket_name(self):
        self._require()
        rc = ZygoteRc.parse("service zygote /system/bin/app_process64 --zygote --socket-name=zygote")
        self.assertEqual(rc.socket_name, "zygote")

    def test_spawn_order_starts_at_init(self):
        self._require()
        self.assertEqual(ZygotePlanner.spawn("zygote64")[0], "init")

    def test_spawn_order_contains_system_server_before_apps(self):
        self._require()
        plan = ZygotePlanner.spawn("zygote64")
        self.assertLess(plan.index("system_server"), plan.index("application process"))

    def test_ro_zygote_variant_maps_to_rc_file(self):
        self._require()
        self.assertEqual(ZygotePlanner.rc_for("zygote64_32"), "init.zygote64_32.rc")

