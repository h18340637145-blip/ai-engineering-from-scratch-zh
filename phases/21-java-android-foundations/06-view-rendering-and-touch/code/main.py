# Android View 渲染与触摸分发模拟
# 课程文档：phases/21-java-android-foundations/06-view-rendering-and-touch/docs/en.md
# 参考资料：docs/AndroidFramework/Java android .md 的 View、SurfaceView、绘制和触摸章节
# 用事件记录展示 requestLayout、invalidate 与父容器拦截后的 CANCEL。

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RenderRequest:
    geometry_changed: bool
    pixels_changed: bool

    def plan(self) -> list[str]:
        actions: list[str] = []
        if self.geometry_changed:
            actions.append("requestLayout")
        if self.pixels_changed:
            actions.append("invalidate")
        return actions


class TouchDispatcher:
    def dispatch(self, action: str, child_consumes: bool, parent_intercepts: bool) -> list[str]:
        if parent_intercepts:
            return ["child:CANCEL", f"parent:{action}"]
        if child_consumes:
            return [f"child:{action}"]
        return [f"unhandled:{action}"]


def scroll_owner(dx: int, dy: int, child_can_scroll: bool) -> str:
    if abs(dx) > abs(dy):
        return "parent-horizontal"
    return "child-vertical" if child_can_scroll else "parent-vertical"


def main() -> None:
    print("=== View 绘制与触摸事件 ===")
    print("尺寸与像素变化：", RenderRequest(True, True).plan())
    print("父级拦截：", TouchDispatcher().dispatch("MOVE", True, True))
    print("横向滑动归属：", scroll_owner(20, 5, True))


if __name__ == "__main__":
    main()
