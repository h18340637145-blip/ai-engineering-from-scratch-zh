# View 渲染与事件分发检查卡

## 使用时机

出现掉帧、重复布局、点击丢失或嵌套滑动冲突时使用。

## 检查步骤

1. 几何变化使用 `requestLayout()`；仅像素变化使用 `invalidate()`。
2. 跟踪 `dispatchTouchEvent()`、`onInterceptTouchEvent()` 与 `onTouchEvent()` 的返回值。
3. 父级拦截后，确认子级是否收到 `ACTION_CANCEL`。
4. 依据方向和 `canScrollVertically()`/`canScrollHorizontally()` 分配滚动。
5. 优先使用 Nested Scrolling，避免无限叠加手写拦截逻辑。
