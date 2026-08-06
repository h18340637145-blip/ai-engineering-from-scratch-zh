# 桌面左右滑动：PagedView 触摸事件与 OverScroller 动画

> 以运行实现为中心的 Android Framework 课程

**Type:** Build
**Languages:** Python
**Prerequisites:** 07-dual-hotseat
**Time:** ~50 分钟

## 学习目标

见 quiz.json 的 pre/check/post 阶段问题。

## 概念

PagedView 是 Workspace 的基类，管理桌面的左右分页和吸附动画。

### 完整触摸事件链路

```text
ACTION_DOWN  → 记录起点，停止旧动画
ACTION_MOVE  → determineScrollingStart 判断是否开始拖动 → scrollBy 跟手
ACTION_UP    → VelocityTracker 计算速度 → snapToPageWithVelocity → OverScroller.startScroll
重绘循环      → computeScroll → computeScrollOffset → scrollTo → 到达目标页更新 mCurrentPage
```

### 核心成员变量

```java
protected OverScroller mScroller;  // 动画计算器
protected int mCurrentPage;        // 当前稳定页
protected int mNextPage;           // 动画目标页
protected boolean mIsBeingDragged; // 是否在拖动
protected VelocityTracker mVelocityTracker;
```

### 日志埋点位置

```java
// PagedView.java
private static final String TAG = "PagedViewStudy";

@Override
protected void onPageBeginTransition() {
    Log.d(TAG, "page transition begin, target=" + mNextPage);
}
```

### 验证命令

```bash
# 过滤分页滑动日志
adb logcat -d | grep -iE "PagedViewStudy|snapToPage"
```

## 构建它

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```
