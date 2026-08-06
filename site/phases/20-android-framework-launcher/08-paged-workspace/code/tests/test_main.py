import unittest, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from main import SwipeState, snap_to_page

# 约定：
#   向左滑（前进到下一页）：displacement_px < 0，velocity_px_s < 0
#   向右滑（返回上一页）：displacement_px > 0，velocity_px_s > 0

class TestSnapToPage(unittest.TestCase):
    def _s(self, cur, disp, vel):
        return SwipeState(current_page=cur, total_pages=4, displacement_px=disp, velocity_px_s=vel)

    def test_fling_left_goes_next(self):
        # vel=-800 表示向左 fling → 翻到下一页
        self.assertEqual(snap_to_page(self._s(1, 0, -800)), 2)

    def test_fling_right_goes_prev(self):
        # vel=800 表示向右 fling → 返回上一页
        self.assertEqual(snap_to_page(self._s(2, 0, 800)), 1)

    def test_displacement_left_goes_next(self):
        # disp=-500 表示向左位移 → 翻到下一页
        self.assertEqual(snap_to_page(self._s(1, -500, 0)), 2)

    def test_displacement_right_goes_prev(self):
        # disp=500 表示向右位移 → 返回上一页
        self.assertEqual(snap_to_page(self._s(2, 500, 0)), 1)

    def test_bounce_back_small_disp_small_vel(self):
        self.assertEqual(snap_to_page(self._s(1, -100, -100)), 1)

    def test_boundary_min_no_underflow(self):
        # 在第 0 页向右 fling，不能到 -1 页
        self.assertEqual(snap_to_page(self._s(0, 500, 800)), 0)

    def test_boundary_max_no_overflow(self):
        # 在最后一页（3）向左 fling，不能到 4 页
        self.assertEqual(snap_to_page(self._s(3, -500, -800)), 3)

    def test_exact_threshold_snaps(self):
        # 位移 = -432 = 1080*0.4，应触发翻页
        s = SwipeState(current_page=1, total_pages=4, displacement_px=-432, velocity_px_s=0, page_width_px=1080)
        self.assertEqual(snap_to_page(s), 2)
