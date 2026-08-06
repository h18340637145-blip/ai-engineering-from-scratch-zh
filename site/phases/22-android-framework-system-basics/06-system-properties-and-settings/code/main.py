# System Property 与 Settings ContentObserver 模拟
# 课程文档：phases/22-android-framework-system-basics/06-system-properties-and-settings/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的系统属性与跨进程设置章节
# 模型不调用隐藏 API、adb setprop 或 Settings Provider，只表达权限与观察语义。

from __future__ import annotations


class PropertyPolicy:
    @staticmethod
    def can_write(name: str, is_system: bool) -> bool:
        return is_system and name.startswith("persist.") and not name.startswith("ro.")


class SettingsObserver:
    def __init__(self) -> None:
        self._registered: set[str] = set()
        self.events: list[tuple[str, str]] = []

    def register(self, uri: str) -> None:
        self._registered.add(uri)

    def unregister(self, uri: str) -> None:
        self._registered.discard(uri)

    def write(self, uri: str, value: str) -> bool:
        if uri not in self._registered:
            return False
        self.events.append((uri, value))
        return True


def main() -> None:
    print("=== 系统属性与 Settings 观察 ===")
    print("系统可写 persist：", PropertyPolicy.can_write("persist.example.flag", True))
    print("ro 属性可写：", PropertyPolicy.can_write("ro.example.flag", True))
    observer = SettingsObserver()
    observer.register("settings://system/example_key")
    observer.write("settings://system/example_key", "1")
    print("观察事件：", observer.events)


if __name__ == "__main__":
    main()
