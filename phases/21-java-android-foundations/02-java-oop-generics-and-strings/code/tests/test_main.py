import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import can_write, choose_abstraction, choose_text_container, generic_access
except ImportError:
    can_write = choose_abstraction = choose_text_container = generic_access = None


class TestJavaOopGenericsAndStrings(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(choose_text_container, "尚未实现字符串容器决策")

    def test_small_immutable_text_uses_string(self):
        self._require()
        self.assertEqual(choose_text_container(1, False), "String")

    def test_single_threaded_many_concats_use_string_builder(self):
        self._require()
        self.assertEqual(choose_text_container(8, False), "StringBuilder")

    def test_shared_mutable_buffer_uses_string_buffer(self):
        self._require()
        self.assertEqual(choose_text_container(8, True), "StringBuffer")

    def test_extends_is_a_read_boundary(self):
        self._require()
        self.assertEqual(generic_access("extends"), "read")
        self.assertFalse(can_write("extends"))

    def test_super_is_a_write_boundary(self):
        self._require()
        self.assertEqual(generic_access("super"), "write")
        self.assertTrue(can_write("super"))

    def test_shared_state_prefers_abstract_class(self):
        self._require()
        self.assertEqual(choose_abstraction(True, False), "abstract class")
        self.assertEqual(choose_abstraction(False, True), "interface")

