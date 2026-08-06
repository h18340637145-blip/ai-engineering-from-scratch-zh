# Android 系统诊断与稳定性测试模拟
# 课程文档：phases/21-java-android-foundations/14-system-diagnostics-and-stability/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 Bugreport、Event Log、ANR、LMKD、Monkey 与 ProtoLog 章节
# 模型以精确关键字形成首个诊断动作，不读取真实设备日志。

from __future__ import annotations


class IncidentTimeline:
    @staticmethod
    def analyze(lines: list[str]) -> dict[str, str]:
        text = "\n".join(lines).lower()
        if "am_anr" in text or "input dispatching timed out" in text:
            next_step = "检查 Binder 调用链" if "binderproxy.transact" in text else "读取 main 线程 trace"
            return {"kind": "ANR", "next_step": next_step}
        if "fatal exception" in text or "androidruntime" in text:
            return {"kind": "Crash", "next_step": "定位首个业务栈帧"}
        if "lmkd" in text and "killing" in text:
            return {"kind": "LMKD", "next_step": "检查 OOM adj 和内存压力"}
        return {"kind": "Unknown", "next_step": "按时间戳收集证据"}

    @staticmethod
    def first_action(source: str) -> str:
        if source == "bugreport":
            return "locate timestamp"
        if source == "anr-trace":
            return "find main thread"
        return "identify source"


def build_monkey_command(package: str, throttle_ms: int, events: int) -> str:
    return (
        f"adb shell monkey -p {package} --throttle {throttle_ms} "
        f"--monitor-native-crashes --bugreport -v -v {events}"
    )


def protolog_command(group: str, enable: bool) -> str:
    action = "enable-text" if enable else "disable-text"
    return f"adb shell wm logging {action} {group}"


def main() -> None:
    print("=== 系统诊断时间线 ===")
    print(IncidentTimeline.analyze(["am_anr", "BinderProxy.transact"]))
    print(build_monkey_command("com.example.app", 100, 1000))
    print(protolog_command("WM_DEBUG_ANIM", False))


if __name__ == "__main__":
    main()
