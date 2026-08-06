# Linux 与 ADB 设备操作规划器
# 课程文档：phases/21-java-android-foundations/13-linux-adb-and-device-operations/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 Linux、ADB、dumpsys 与 Logcat 章节
# 仅构造命令字符串，不会连接设备或执行任何破坏性操作。

from __future__ import annotations


class AdbCommandPlanner:
    @staticmethod
    def for_device(serial: str | None, command: str) -> str:
        prefix = f"adb -s {serial}" if serial else "adb"
        return f"{prefix} {command}"

    @staticmethod
    def logcat_for(package: str) -> str:
        return f"adb logcat --pid=$(adb shell pidof -s {package})"

    @staticmethod
    def current_focus() -> str:
        return "adb shell dumpsys window | grep -E 'mCurrentFocus|mFocusedApp'"


def is_safe_shell_command(command: str) -> bool:
    normalized = " ".join(command.split())
    if "rm -rf *" in normalized or normalized in {"rm -rf /", "rm -rf ~"}:
        return False
    return True


def main() -> None:
    print("=== ADB 安全命令规划 ===")
    print(AdbCommandPlanner.for_device("emulator-5554", "shell pm list packages"))
    print(AdbCommandPlanner.logcat_for("com.example.app"))
    print("允许 find：", is_safe_shell_command("find /system -name bootanimation.zip"))
    print("允许 rm -rf *：", is_safe_shell_command("rm -rf *"))


if __name__ == "__main__":
    main()
