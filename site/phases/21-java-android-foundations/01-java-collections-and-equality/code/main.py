# Java 集合与相等性契约模拟
# 课程文档：phases/21-java-android-foundations/01-java-collections-and-equality/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的链表、Collection 与 hashCode 章节
# 实现仅使用 Python 标准库，表达 Java 集合的选择边界。

from __future__ import annotations


class CollectionAdvisor:
    """根据顺序、唯一性和操作方式选择 Java 集合实现。"""

    @staticmethod
    def recommend(shape: str, workload: str) -> str:
        choices = {
            ("ordered", "read-heavy"): "ArrayList",
            ("ordered", "iterate-mutate"): "LinkedList",
            ("unique", "insertion-order"): "LinkedHashSet",
            ("unique", "sorted"): "TreeSet",
            ("keyed", "insertion-order"): "LinkedHashMap",
            ("keyed", "sorted"): "TreeMap",
            ("keyed", "default"): "HashMap",
        }
        if (shape, workload) not in choices:
            raise ValueError(f"未知的集合需求：{shape}/{workload}")
        return choices[(shape, workload)]


def hash_contract_holds(is_equal: bool, first_hash: int, second_hash: int) -> bool:
    """检查 equals 为真时 hashCode 必须相同的单向契约。"""
    return not is_equal or first_hash == second_hash


def main() -> None:
    print("=== Java 集合与相等性决策 ===")
    for shape, workload in (("ordered", "read-heavy"), ("unique", "sorted"), ("keyed", "default")):
        print(f"{shape}/{workload} -> {CollectionAdvisor.recommend(shape, workload)}")
    print("相等对象哈希一致：", hash_contract_holds(True, 42, 42))
    print("哈希碰撞不等于对象相等：", hash_contract_holds(False, 42, 42))


if __name__ == "__main__":
    main()
