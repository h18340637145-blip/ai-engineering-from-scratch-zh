# Handler 队列审计卡

## 使用时机

页面旋转、返回或销毁后仍执行旧回调、内存持续增长或主线程任务乱序时使用。

## 检查步骤

1. 标记每个 `post()`/`sendMessage()` 的投递者、目标 Looper 和延迟。
2. 确认后台线程在使用 Handler 前已准备 Looper。
3. 页面销毁时调用 `removeCallbacks()` / `removeMessages()`。
4. 让延迟任务按生命周期取消，而不是保留 Activity 引用。
5. 先看消息到期时间，不凭投递顺序推断执行顺序。
