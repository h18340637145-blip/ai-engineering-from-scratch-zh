# Java 反射、序列化与 GC 可达性模拟
# 课程文档：phases/21-java-android-foundations/03-reflection-serialization-and-gc/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的反射、序列化、Integer 与 GC 章节
# 实现将运行时风险变成可测试的规则，而不执行真实反射或反序列化。

from __future__ import annotations


def inspect_type(type_name: str, member_name: str, allow_private: bool) -> str:
    """模拟反射访问前的封装检查。"""
    if member_name.startswith("_") and not allow_private:
        return "denied"
    kind = "field" if member_name.startswith("_") else "method"
    return f"{kind}:{member_name}"


def should_deserialize(trusted_source: bool, data_format: str) -> bool:
    """不可信输入一律拒绝；可信 JSON 是显式、可审查的数据格式。"""
    return trusted_source and data_format in {"json", "protobuf", "java-native"}


def is_collectable(roots: dict[str, bool]) -> bool:
    """对象只有不被线程栈、静态字段或 JNI 根持有时才可回收。"""
    required_roots = ("thread_stack", "static", "jni")
    return not any(roots.get(root, False) for root in required_roots)


def reference_collection_timing(strength: str) -> str:
    timings = {
        "strong": "reachable",
        "soft": "memory-pressure",
        "weak": "next-gc",
        "phantom": "notification-only",
    }
    if strength not in timings:
        raise ValueError("未知引用强度")
    return timings[strength]


def parse_integer(value: str, radix: int = 10) -> int:
    return int(value, radix)


def main() -> None:
    print("=== Java 运行时安全边界 ===")
    print("私有字段（无授权）：", inspect_type("User", "_token", False))
    print("可信 JSON：", should_deserialize(True, "json"))
    print("无 GC Root：", is_collectable({"thread_stack": False, "static": False, "jni": False}))
    print("弱引用回收时机：", reference_collection_timing("weak"))


if __name__ == "__main__":
    main()
