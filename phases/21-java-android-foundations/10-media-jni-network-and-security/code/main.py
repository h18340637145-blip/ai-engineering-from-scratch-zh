# Android 多媒体、JNI、网络与安全决策模拟
# 课程文档：phases/21-java-android-foundations/10-media-jni-network-and-security/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的多媒体、JNI、网络与安全章节
# 用小型规则模型表达协议可靠性、加密用途与 JNI 引入条件。

from __future__ import annotations


class TransportAdvisor:
    @staticmethod
    def choose(goal: str, reliable: bool) -> str:
        if reliable:
            return "TCP"
        if goal == "low-latency":
            return "UDP"
        return "UDP"


class CryptoAdvisor:
    @staticmethod
    def recommend(goal: str) -> str:
        choices = {
            "bulk-data": "AES",
            "signing": "RSA/ECC",
            "integrity": "SHA-256",
            "password-storage": "KDF",
        }
        if goal not in choices:
            raise ValueError("未知安全目标")
        return choices[goal]


class MediaPipeline:
    @staticmethod
    def steps() -> list[str]:
        return ["capture", "encode", "timestamp", "mux or save"]


def jni_is_justified(reuse_native_library: bool, performance_sensitive: bool) -> bool:
    return reuse_native_library or performance_sensitive


def main() -> None:
    print("=== 多媒体与网络边界 ===")
    print("可靠流：", TransportAdvisor.choose("stream", True))
    print("批量数据加密：", CryptoAdvisor.recommend("bulk-data"))
    print("媒体管线：", " -> ".join(MediaPipeline.steps()))
    print("JNI 是否合理：", jni_is_justified(False, True))


if __name__ == "__main__":
    main()
