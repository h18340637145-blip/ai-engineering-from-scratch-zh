# SRO/RRO 资源 Overlay 优先级模拟
# 课程文档：phases/22-android-framework-system-basics/05-resource-overlays/docs/en.md
# 参考资料：docs/AndroidFramework/Android Framework 基础.md 的资源 Overlay 章节
# 该模型只选择 Overlay，不执行 adb cmd overlay 或修改设备状态。

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Overlay:
    package: str
    target: str
    category: str
    priority: int
    enabled: bool
    kind: str


class OverlayRegistry:
    def __init__(self) -> None:
        self._items: list[Overlay] = []

    def add(self, overlay: Overlay) -> None:
        self._items.append(overlay)

    def resolve(self, category: str, user: int) -> Overlay | None:
        del user
        candidates = [overlay for overlay in self._items if overlay.category == category and overlay.enabled]
        return max(candidates, key=lambda overlay: overlay.priority) if candidates else None

    def enable_exclusive(self, category: str, package: str) -> None:
        for overlay in self._items:
            if overlay.category == category and overlay.kind == "RRO":
                overlay.enabled = overlay.package == package

    def can_toggle(self, package: str) -> bool:
        found = next((overlay for overlay in self._items if overlay.package == package), None)
        return found is not None and found.kind == "RRO"


def main() -> None:
    print("=== Overlay 优先级 ===")
    registry = OverlayRegistry()
    registry.add(Overlay("com.example.buttons", "android", "navigation", 1, True, "RRO"))
    registry.add(Overlay("com.example.gesture", "android", "navigation", 10, True, "RRO"))
    print("当前导航 Overlay：", registry.resolve("navigation", user=0).package)


if __name__ == "__main__":
    main()
