import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

try:
    from main import CollectionAdvisor, hash_contract_holds
except ImportError:
    CollectionAdvisor = None
    hash_contract_holds = None


class TestJavaCollectionsAndEquality(unittest.TestCase):
    def _require(self):
        self.assertIsNotNone(CollectionAdvisor, "尚未实现集合选型模型")
        self.assertIsNotNone(hash_contract_holds, "尚未实现哈希契约检查")

    def test_read_heavy_ordered_data_uses_array_list(self):
        self._require()
        self.assertEqual(CollectionAdvisor.recommend("ordered", "read-heavy"), "ArrayList")

    def test_iterative_mutation_uses_linked_list(self):
        self._require()
        self.assertEqual(CollectionAdvisor.recommend("ordered", "iterate-mutate"), "LinkedList")

    def test_insertion_ordered_unique_values_use_linked_hash_set(self):
        self._require()
        self.assertEqual(CollectionAdvisor.recommend("unique", "insertion-order"), "LinkedHashSet")

    def test_sorted_keyed_data_uses_tree_map(self):
        self._require()
        self.assertEqual(CollectionAdvisor.recommend("keyed", "sorted"), "TreeMap")

    def test_equal_objects_must_share_hash_code(self):
        self._require()
        self.assertTrue(hash_contract_holds(True, 42, 42))
        self.assertFalse(hash_contract_holds(True, 42, 7))

    def test_unequal_objects_may_share_hash_code(self):
        self._require()
        self.assertTrue(hash_contract_holds(False, 42, 42))

