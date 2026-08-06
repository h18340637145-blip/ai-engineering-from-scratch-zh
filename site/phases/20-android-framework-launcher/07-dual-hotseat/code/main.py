# 双排 Hotseat 坐标映射与验证
# 课程文档：phases/20-android-framework-launcher/07-dual-hotseat/docs/en.md
# 参考：AOSP Hotseat.java / DeviceProfile.java / LauncherModel

"""
模拟双排 Hotseat 中 rank → (row, col) 坐标映射逻辑。

单排 Hotseat：rank = cellX，cellY = 0
双排 Hotseat：rank 按列数折叠到两行
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List


@dataclass
class HotseatConfig:
    rows: int       # 行数（单排=1，双排=2）
    cols: int       # 每排图标数（如 5）

    @property
    def total(self) -> int:
        return self.rows * self.cols


def rank_to_cell(rank: int, cfg: HotseatConfig) -> Tuple[int, int]:
    """
    将 rank 映射到 (cell_y, cell_x) 双排坐标。

    row = rank // cols  （第几行）
    col = rank % cols   （该行第几列）
    """
    if rank < 0 or rank >= cfg.total:
        raise ValueError(f"rank {rank} 超出范围 [0, {cfg.total})")
    row = rank // cfg.cols
    col = rank % cfg.cols
    return row, col


def cell_to_rank(row: int, col: int, cfg: HotseatConfig) -> int:
    """逆映射：(row, col) → rank"""
    return row * cfg.cols + col


def print_mapping(cfg: HotseatConfig) -> None:
    print(f"=== Hotseat {cfg.rows}行×{cfg.cols}列 rank 映射 ===\n")
    for rank in range(cfg.total):
        row, col = rank_to_cell(rank, cfg)
        print(f"  rank={rank:2d}  →  row={row}, col={col}")


def main() -> None:
    # 单排（标准 Hotseat）
    single = HotseatConfig(rows=1, cols=5)
    print_mapping(single)

    print()
    # 双排
    dual = HotseatConfig(rows=2, cols=5)
    print_mapping(dual)

    # 验证逆映射
    print("\n=== 逆映射验证（双排）===\n")
    for rank in range(dual.total):
        row, col = rank_to_cell(rank, dual)
        back = cell_to_rank(row, col, dual)
        status = "✅" if back == rank else "❌"
        print(f"  {status} rank={rank} → ({row},{col}) → rank={back}")


if __name__ == "__main__":
    main()
