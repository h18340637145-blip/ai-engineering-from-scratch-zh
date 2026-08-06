# Binder、类加载、APK 构建与安装

> Android 运行时的“能调用”与“能安装”之间，隔着 Binder 边界、DEX、签名和包管理状态。

**Type:** Build
**Languages:** Python
**Prerequisites:** 11-architecture-performance-and-storage
**Time:** ~90 分钟

## 学习目标

- 说明 Binder 代理、驱动和服务端线程池的基本角色
- 识别大对象跨 Binder 传输的风险
- 区分 `PathClassLoader` 与 `DexClassLoader`
- 排列现代 APK/AAB 的构建关键步骤
- 描述 Package Manager 的安装职责

## 概念

Binder 是 Android 的核心本地 IPC。客户端代理写入事务，服务端读取并返回结果。它降低复制成本但仍有事务大小边界，因此大位图和大集合不适合作为常规 Binder 参数。

```mermaid
flowchart LR
    A[Manifest / 资源 / 源码] --> B[AAPT2]
    B --> C[编译 Java/Kotlin]
    C --> D[D8 生成 DEX]
    D --> E[R8 压缩与优化]
    E --> F[打包 APK/AAB]
    F --> G[签名与 zipalign]
    G --> H[Package Manager 解析与安装]
```

`PathClassLoader` 加载安装包和系统路径代码；受控插件场景可用 `DexClassLoader`，但动态代码必须来自可信来源，并承担兼容性与调试成本。

## 构建它

`BinderTransaction` 以 1 MiB 的教学边界拒绝过大事务；`BuildPipeline` 输出从资源处理到签名的现代流水线。

```bash
cd phases/21-java-android-foundations/12-binder-classloading-build-and-install/code
python3 main.py
python3 -m unittest discover tests -v
```

## 诊断练习

当跨进程调用抛出 `TransactionTooLargeException`，不要只重试；缩小事务、传 URI/文件描述符，或改变数据流设计。

## 发布它

Binder 与构建检查卡见 `outputs/skill-binder-build-flow.md`。
