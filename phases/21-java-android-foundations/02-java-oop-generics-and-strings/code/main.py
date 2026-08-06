# Java 对象模型、泛型与字符串容器决策
# 课程文档：phases/21-java-android-foundations/02-java-oop-generics-and-strings/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 OOP、String、内部类与泛型章节
# 实现以规则模型演示 PECS 和可变字符串的使用边界。

from __future__ import annotations


def choose_text_container(concat_count: int, shared_threads: bool) -> str:
    """少量拼接用 String；大量拼接按是否共享选择 Builder/Buffer。"""
    if concat_count <= 1:
        return "String"
    return "StringBuffer" if shared_threads else "StringBuilder"


def generic_access(bound: str) -> str:
    if bound == "extends":
        return "read"
    if bound == "super":
        return "write"
    raise ValueError("bound 只能是 extends 或 super")


def can_write(bound: str) -> bool:
    return generic_access(bound) == "write"


def choose_abstraction(has_shared_state: bool, multiple_implementations: bool) -> str:
    if has_shared_state:
        return "abstract class"
    return "interface" if multiple_implementations else "abstract class"


def main() -> None:
    print("=== Java 对象模型决策 ===")
    print("单线程多次拼接：", choose_text_container(8, False))
    print("共享缓冲区：", choose_text_container(8, True))
    print("? extends Number：", generic_access("extends"))
    print("? super Integer：", generic_access("super"))
    print("多实现能力契约：", choose_abstraction(False, True))


if __name__ == "__main__":
    main()
