# 系统属性与 Settings 检查卡

## 使用时机

跨进程配置没有生效、设置观察不到变化或权限被拒绝时使用。

## 检查顺序

1. `ro.*` 只读；确认是否应使用 `persist.*` 或 Settings。
2. 普通应用优先公开 API、Settings 或 Binder，而非隐藏 SystemProperties。
3. 核对 Settings 的 System、Secure、Global 表和当前 user。
4. 注册正确 URI 的 ContentObserver，并在结束时注销。
5. 对 Secure/Global 写入同时核对签名、allowlist 和 SELinux。
