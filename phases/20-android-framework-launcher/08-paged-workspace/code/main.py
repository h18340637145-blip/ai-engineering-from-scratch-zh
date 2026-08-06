# PagedView 分页吸附逻辑模拟
# 课程文档：phases/20-android-framework-launcher/08-paged-workspace/docs/en.md
# 参考：AOSP packages/apps/Launcher3/src/com/android/launcher3/PagedView.java
# 坐标约定：负速度/负位移 = 向左滑 = 前进到下一页（+1）；正 = 向右 = 返回上一页（-1）

"""
模拟 PagedView 的分页吸附决策：根据手指速度和位移判断目标页。
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SwipeState:
    current_page: int
    total_pages: int
    displacement_px: int   # 正值=向右滑，负值=向左滑
    velocity_px_s: int     # 正值=向右，负值=向左
    page_width_px: int = 1080
    min_fling_velocity: int = 500
    min_snap_threshold: float = 0.4  # 位移超过页面宽度 40% 则翻页


def snap_to_page(state: SwipeState) -> int:
    """
    模拟 PagedView.snapToPageWithVelocity() 的决策逻辑：
    1. 速度足够大（fling），按速度方向翻页
    2. 位移超过阈值，按位移方向翻页
    3. 否则回到当前页

    坐标约定：负速度/负位移 = 向左滑 = 前进到下一页（+1）
              正速度/正位移 = 向右滑 = 返回上一页（-1）
    """
    total = state.total_pages
    cur = state.current_page
    disp = state.displacement_px
    vel = state.velocity_px_s
    threshold_px = state.page_width_px * state.min_snap_threshold

    # 速度 fling 判断
    if abs(vel) >= state.min_fling_velocity:
        delta = 1 if vel < 0 else -1  # 向左滑 vel<0 → 翻到下一页 (+1)
        return max(0, min(total - 1, cur + delta))

    # 位移阈值判断
    if abs(disp) >= threshold_px:
        delta = 1 if disp < 0 else -1  # 向左位移 disp<0 → 翻到下一页 (+1)
        return max(0, min(total - 1, cur + delta))

    return cur  # 回弹到当前页


def main() -> None:
    total = 4
    print(f"=== PagedView 分页吸附决策（共 {total} 页）===\n")
    cases = [
        SwipeState(current_page=1, total_pages=total, displacement_px=-500, velocity_px_s=0),
        SwipeState(current_page=1, total_pages=total, displacement_px=0, velocity_px_s=-800),
        SwipeState(current_page=1, total_pages=total, displacement_px=-100, velocity_px_s=-100),
        SwipeState(current_page=0, total_pages=total, displacement_px=500, velocity_px_s=800),
    ]
    labels = ["位移翻页（→下一页）", "速度 fling（→下一页）", "回弹（留在当前页）", "边界保护（不越界）"]
    for state, label in zip(cases, labels):
        target = snap_to_page(state)
        print(f"  [{label}]")
        print(f"    当前页={state.current_page}，位移={state.displacement_px}px，速度={state.velocity_px_s}px/s")
        print(f"    目标页: {target}\n")


if __name__ == "__main__":
    main()
