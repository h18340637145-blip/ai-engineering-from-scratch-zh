import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import ThreadOperation, choose_ipc, main_thread_advice
except ImportError:
    ThreadOperation = choose_ipc = main_thread_advice = None


class TestProcessThreadAndIpc(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(choose_ipc, "尚未实现 IPC 选型模型")

    def test_small_rpc_uses_binder_aidl(self):
        self._require()
        self.assertEqual(choose_ipc("rpc", 64), "Binder/AIDL")

    def test_structured_shared_data_uses_content_provider(self):
        self._require()
        self.assertEqual(choose_ipc("data", 64), "ContentProvider")

    def test_one_to_many_event_uses_broadcast(self):
        self._require()
        self.assertEqual(choose_ipc("event", 1), "Broadcast")

    def test_custom_cross_device_protocol_uses_socket(self):
        self._require()
        self.assertEqual(choose_ipc("network", 32), "Socket")

    def test_start_creates_thread_but_run_is_normal_call(self):
        self._require()
        self.assertEqual(ThreadOperation.describe("start"), "creates-thread")
        self.assertEqual(ThreadOperation.describe("run"), "normal-call")

    def test_wait_releases_lock_but_sleep_does_not(self):
        self._require()
        self.assertEqual(ThreadOperation.describe("wait"), "releases-lock")
        self.assertEqual(ThreadOperation.describe("sleep"), "retains-lock")
        self.assertEqual(main_thread_advice("network"), "move-to-background")

