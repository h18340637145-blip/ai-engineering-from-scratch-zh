# Android 进程、线程和 IPC 选型模拟
# 课程文档：phases/21-java-android-foundations/04-process-thread-and-ipc/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的进程、线程、IPC 与 wait/sleep 章节
# 模型不连接设备，仅将组件通信边界转换成可运行决策。

from __future__ import annotations


def choose_ipc(goal: str, payload_kb: int) -> str:
    """按通信目标选择 Android 常见 IPC 方式。"""
    if goal == "rpc" and payload_kb <= 1024:
        return "Binder/AIDL"
    if goal == "data":
        return "ContentProvider"
    if goal == "event":
        return "Broadcast"
    if goal == "serial-message":
        return "Messenger"
    if goal == "network" or payload_kb > 1024:
        return "Socket"
    raise ValueError(f"未知 IPC 目标：{goal}")


class ThreadOperation:
    """说明易混淆的线程方法语义。"""

    @staticmethod
    def describe(operation: str) -> str:
        meanings = {
            "start": "creates-thread",
            "run": "normal-call",
            "join": "waits-for-thread",
            "wait": "releases-lock",
            "sleep": "retains-lock",
        }
        if operation not in meanings:
            raise ValueError("未知线程操作")
        return meanings[operation]


def main_thread_advice(task: str) -> str:
    return "move-to-background" if task in {"network", "disk", "database", "decode"} else "safe-on-main"


def main() -> None:
    print("=== Android IPC 与线程边界 ===")
    for goal in ("rpc", "data", "event", "network"):
        print(f"{goal} -> {choose_ipc(goal, 64)}")
    print("wait：", ThreadOperation.describe("wait"))
    print("主线程网络请求：", main_thread_advice("network"))


if __name__ == "__main__":
    main()
