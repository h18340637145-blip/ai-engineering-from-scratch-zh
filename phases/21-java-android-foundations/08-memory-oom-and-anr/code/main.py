# Android 内存泄漏、OOM 与 ANR 分类模拟
# 课程文档：phases/21-java-android-foundations/08-memory-oom-and-anr/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的内存泄漏、OOM 与 ANR 章节
# 通过精确日志线索区分不同事件，避免把 LMKD 回收误判为应用崩溃。

from __future__ import annotations


class IncidentClassifier:
    @staticmethod
    def classify(log_line: str) -> str:
        line = log_line.lower()
        if "fatal exception" in line or "androidruntime" in line:
            return "Crash"
        if "input dispatching timed out" in line or "anr" in line:
            return "ANR"
        if "outofmemoryerror" in line:
            return "OOM"
        if "lmkd" in line and "killing" in line:
            return "LMKD"
        return "Unknown"


def leak_remedy(source: str) -> str:
    fixes = {
        "static-activity": "Application Context",
        "delayed-handler": "remove callbacks",
        "unregistered-observer": "unregister observer",
        "running-animation": "cancel animation",
    }
    if source not in fixes:
        raise ValueError("未知泄漏来源")
    return fixes[source]


def main() -> None:
    print("=== 内存与无响应诊断 ===")
    for line in ("FATAL EXCEPTION: main", "Input dispatching timed out", "OutOfMemoryError", "lmkd: Killing process"):
        print(f"{line} -> {IncidentClassifier.classify(line)}")
    print("静态 Activity 修复：", leak_remedy("static-activity"))


if __name__ == "__main__":
    main()
