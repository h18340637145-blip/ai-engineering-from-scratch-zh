import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import (
        inspect_type,
        is_collectable,
        parse_integer,
        reference_collection_timing,
        should_deserialize,
    )
except ImportError:
    inspect_type = is_collectable = parse_integer = reference_collection_timing = should_deserialize = None


class TestReflectionSerializationAndGc(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(inspect_type, "尚未实现反射与 GC 模型")

    def test_reflection_can_inspect_declared_method(self):
        self._require()
        self.assertEqual(inspect_type("User", "getName", False), "method:getName")

    def test_private_members_require_explicit_permission(self):
        self._require()
        self.assertEqual(inspect_type("User", "_token", False), "denied")
        self.assertEqual(inspect_type("User", "_token", True), "field:_token")

    def test_untrusted_bytes_must_not_be_deserialized(self):
        self._require()
        self.assertFalse(should_deserialize(False, "java-native"))
        self.assertTrue(should_deserialize(True, "json"))

    def test_object_without_gc_roots_is_collectable(self):
        self._require()
        self.assertTrue(is_collectable({"thread_stack": False, "static": False, "jni": False}))

    def test_static_root_prevents_collection(self):
        self._require()
        self.assertFalse(is_collectable({"thread_stack": False, "static": True, "jni": False}))

    def test_reference_strength_has_expected_collection_timing(self):
        self._require()
        self.assertEqual(reference_collection_timing("weak"), "next-gc")
        self.assertEqual(reference_collection_timing("soft"), "memory-pressure")
        self.assertEqual(parse_integer("ff", 16), 255)

