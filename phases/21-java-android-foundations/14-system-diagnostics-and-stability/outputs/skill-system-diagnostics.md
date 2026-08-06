# Android 系统事故证据链模板

## 使用时机

定位 Crash、ANR、LMKD、窗口动画或稳定性随机测试失败时使用。

## 采集顺序

1. 记录时间、包名、PID、设备 Build fingerprint。
2. 在 Event Log 中定位 `am_anr`、`am_crash`、`am_kill`、`am_proc_died`。
3. 在 Bugreport 中查内存、服务、属性和前后台状态。
4. 对 ANR 阅读 main 线程，并继续追踪持锁线程或 Binder 服务端。
5. Monkey 限定包名和节流；ProtoLog 完成诊断后关闭对应组。
