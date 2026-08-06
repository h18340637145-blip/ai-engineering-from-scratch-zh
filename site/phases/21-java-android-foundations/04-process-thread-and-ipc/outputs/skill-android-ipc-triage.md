# Android IPC 选型卡

## 使用时机

需要在两个进程、组件或设备之间传递数据或调用能力时使用。

## 选择规则

1. 少量 RPC：Binder/AIDL。
2. 结构化共享数据：ContentProvider。
3. 一对多事件：Broadcast。
4. 简单串行消息：Messenger。
5. 自定义跨设备或跨语言协议：Socket。

## 保护边界

- 不把大位图、大集合塞进 Binder 事务。
- 不让主线程等待网络、磁盘或长时间 IPC。
- `wait()` 在同步条件循环中使用；`sleep()` 不会释放锁。
