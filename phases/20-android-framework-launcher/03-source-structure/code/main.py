# Launcher3 源码结构 - 模拟 LoaderCursor 的 ItemInfo 坐标校验
# 课程文档：phases/20-android-framework-launcher/03-source-structure/docs/en.md
# 参考：AOSP packages/apps/Launcher3/src/com/android/launcher3/model/LoaderCursor.java

"""
模拟 LoaderCursor 校验 ItemInfo 坐标合法性的核心逻辑。

LoaderCursor 在读取数据库时，会检查每个 Item 的 (cellX, cellY, spanX, spanY)
是否超出当前设备的网格边界。非法 Item 会被丢弃而不渲染。
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional


@dataclass
class GridConfig:
    """设备网格配置，对应 InvariantDeviceProfile 中的列数和行数。"""
    columns: int
    rows: int


@dataclass
class ItemInfo:
    """
    桌面元素数据，对应 AOSP 中的 ItemInfo 核心字段。

    container:
        -100 = CONTAINER_DESKTOP (桌面)
        -101 = CONTAINER_HOTSEAT (底部常驻区域)
    """
    label: str
    cell_x: int
    cell_y: int
    span_x: int = 1
    span_y: int = 1
    container: int = -100  # CONTAINER_DESKTOP

    def validate(self, grid: GridConfig) -> Tuple[bool, Optional[str]]:
        """
        校验 Item 在给定网格内是否合法。

        返回 (is_valid, error_message)，合法时 error_message 为 None。
        """
        if self.cell_x < 0:
            return False, "cellX 不能为负数"
        if self.cell_y < 0:
            return False, "cellY 不能为负数"
        if self.span_x < 1 or self.span_y < 1:
            return False, "span 不能小于 1"
        if self.cell_x + self.span_x > grid.columns:
            return False, f"X 方向越界 (cellX={self.cell_x} + spanX={self.span_x} > {grid.columns})"
        if self.cell_y + self.span_y > grid.rows:
            return False, f"Y 方向越界 (cellY={self.cell_y} + spanY={self.span_y} > {grid.rows})"
        return True, None


def load_and_validate(items: List[ItemInfo], grid: GridConfig) -> Tuple[List[ItemInfo], List[Tuple[ItemInfo, str]]]:
    """
    模拟 LoaderCursor.loadWorkspace() 的校验阶段。

    返回 (valid_items, invalid_items_with_reason)。
    """
    valid: List[ItemInfo] = []
    invalid: List[Tuple[ItemInfo, str]] = []
    for item in items:
        ok, reason = item.validate(grid)
        if ok:
            valid.append(item)
        else:
            invalid.append((item, reason))
    return valid, invalid


def main() -> None:
    grid = GridConfig(columns=5, rows=5)

    items = [
        ItemInfo("应用图标 A", cell_x=0, cell_y=0),
        ItemInfo("双格小部件", cell_x=1, cell_y=2, span_x=2, span_y=2),
        ItemInfo("越界应用 B", cell_x=4, cell_y=0, span_x=2),       # X 越界
        ItemInfo("越界应用 C", cell_x=0, cell_y=5),                  # Y 越界
        ItemInfo("负坐标应用", cell_x=-1, cell_y=0),                  # 负数坐标
        ItemInfo("正常应用 D", cell_x=3, cell_y=4),
    ]

    print(f"=== 校验 {grid.columns} 列 × {grid.rows} 行 网格 ===\n")
    valid, invalid = load_and_validate(items, grid)

    for item in valid:
        print(
            f"  ✅ {item.label} "
            f"[cellX={item.cell_x}, cellY={item.cell_y}, "
            f"span={item.span_x}×{item.span_y}] 合法"
        )
    for item, reason in invalid:
        print(
            f"  ❌ {item.label} "
            f"[cellX={item.cell_x}, cellY={item.cell_y}, "
            f"span={item.span_x}×{item.span_y}] {reason}"
        )

    print(f"\n合法 Item: {len(valid)} / {len(items)}")


if __name__ == "__main__":
    main()
