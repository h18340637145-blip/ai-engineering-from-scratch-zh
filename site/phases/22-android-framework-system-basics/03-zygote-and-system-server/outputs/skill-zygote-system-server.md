# Zygote 启动链检查卡

## 使用时机

系统 Java 服务或应用进程未按预期启动时使用。

## 检查顺序

1. 读取 `ro.zygote` 并确认导入的 `init.zygote*.rc`。
2. 检查 app_process、`--start-system-server` 与 socket-name。
3. 设备端确认 zygote/zygote64 进程和 `/dev/socket/zygote*`。
4. 在 system buffer 关联 Zygote 与 SystemServer 日志。
5. 区分 Zygote 问题、SystemServer 问题和应用组件启动问题。
