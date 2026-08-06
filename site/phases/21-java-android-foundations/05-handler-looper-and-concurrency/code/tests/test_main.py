import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import MessageQueueSimulator
except ImportError:
    MessageQueueSimulator = None


class TestHandlerLooperAndConcurrency(unittest.TestCase):
    def _queue(self):
        self.assertIsNotNone(MessageQueueSimulator, "尚未实现消息队列模拟器")
        return MessageQueueSimulator()

    def test_ready_message_is_delivered_at_due_time(self):
        queue = self._queue()
        queue.post("render", now_ms=0, delay_ms=10)
        self.assertEqual(queue.next_due(10).name, "render")

    def test_future_message_is_not_delivered_early(self):
        queue = self._queue()
        queue.post("render", now_ms=0, delay_ms=10)
        self.assertIsNone(queue.next_due(9))

    def test_queue_delivers_earliest_message_first(self):
        queue = self._queue()
        queue.post("late", now_ms=0, delay_ms=30)
        queue.post("early", now_ms=0, delay_ms=10)
        self.assertEqual(queue.next_due(30).name, "early")

    def test_removed_callbacks_are_not_delivered(self):
        queue = self._queue()
        queue.post("activity-callback", now_ms=0, delay_ms=10)
        queue.remove("activity-callback")
        self.assertIsNone(queue.next_due(10))

    def test_main_thread_has_prepared_looper(self):
        queue = self._queue()
        self.assertTrue(queue.has_looper("main"))

    def test_background_thread_requires_prepare(self):
        queue = self._queue()
        self.assertFalse(queue.has_looper("worker"))
        queue.prepare("worker")
        self.assertTrue(queue.has_looper("worker"))

