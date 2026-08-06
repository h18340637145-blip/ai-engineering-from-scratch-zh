# 技能：PagedView 分页吸附决策分析

## 用途
给定滑动手势的速度和位移，判断 Launcher3 工作区应吸附到哪一页。

## 核心接口
```python
from main import SwipeState, snap_to_page

state = SwipeState(
    current_page=1,
    total_pages=4,
    displacement_px=-500,  # 负=向左滑
    velocity_px_s=0,
    page_width_px=1080,
)
target = snap_to_page(state)
print(f"目标页: {target}")
```

## 坐标约定
- `velocity_px_s < 0` / `displacement_px < 0` → 向左滑 → 翻到下一页（+1）
- `velocity_px_s > 0` / `displacement_px > 0` → 向右滑 → 返回上一页（-1）

## 决策优先级
1. 速度 ≥ `min_fling_velocity`（默认 500 px/s）→ 按速度方向
2. 位移 ≥ 页宽 × `min_snap_threshold`（默认 40%）→ 按位移方向
3. 否则 → 回弹到当前页

## 来源
AOSP `packages/apps/Launcher3/src/com/android/launcher3/PagedView.java`
