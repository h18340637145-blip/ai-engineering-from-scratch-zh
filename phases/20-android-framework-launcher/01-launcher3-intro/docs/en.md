# Launcher3 是什么：系统桌面在 Android 中的角色

> 从零认识 Android 系统桌面：为什么 Launcher3 不是 Framework，却能成为桌面。

**Type:** Learn
**Languages:** Python
**Prerequisites:** None
**Time:** ~30 分钟

## 学习目标

- 解释 Launcher3 在 Android 架构分层中的位置
- 区分 Launcher3 与 Launcher3QuickStep 的能力差异
- 列出 Launcher3 依赖的五个核心 Framework 服务
- 说明 `CATEGORY_HOME` 为何能让一个普通 Activity 成为系统主屏幕
- 识别 QuickStep 上滑手势的大致调用链路

## 概念

Launcher3 是 Android 系统的默认桌面应用，它运行在 Android Framework **之上**，而不是 Framework 本身。

### Android 系统分层

```text
应用层      : Launcher3、设置、相机 ...
Framework   : ActivityTaskManagerService、PackageManagerService ...
Native      : Bionic、OpenGL、SQLite ...
内核         : Linux Kernel
```

Launcher3 处于应用层，通过 Binder IPC 调用 Framework 服务。

### 为什么它能成为桌面

关键在 `AndroidManifest.xml` 的 Intent Filter：

```xml
<intent-filter>
    <action android:name="android.intent.action.MAIN" />
    <category android:name="android.intent.category.HOME" />
    <category android:name="android.intent.category.DEFAULT" />
</intent-filter>
```

当用户按下 Home 键，系统会广播 `CATEGORY_HOME` 意图。拥有该声明的 Activity 即成为候选默认桌面。

### Launcher3 vs Launcher3QuickStep

| 特性 | Launcher3 | Launcher3QuickStep |
|---|---|---|
| Workspace 分页 | ✅ | ✅ |
| All Apps | ✅ | ✅ |
| Folder / Widget | ✅ | ✅ |
| 最近任务 Overview | ❌ | ✅ |
| 全面屏手势 | ❌ | ✅ |
| 与 SystemUI 交互 | ❌ | ✅ |

QuickStep 对应源码：`packages/apps/Launcher3/quickstep/`

### Launcher3 依赖的核心 Framework 服务

| Framework 服务 | 用途 |
|---|---|
| ActivityTaskManagerService | 启动应用、任务管理 |
| PackageManagerService | 查询已安装应用和图标信息 |
| WindowManagerService | 窗口、动画和屏幕布局 |
| AppWidgetService | 桌面小部件管理 |
| InputManagerService | 手势和输入事件支持 |

### 点击桌面图标后发生了什么

```text
点击图标
    ↓
Launcher3 获取 ComponentName / Intent
    ↓
startActivity()
    ↓
ActivityTaskManagerService 处理
    ↓
目标应用 Activity 启动
```

## 构建它

本课实现一个 Python 脚本，解析 AndroidManifest 片段并判断一个组件是否声明了 `CATEGORY_HOME`——模拟系统选择默认桌面的核心逻辑。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

输出示例：

```text
Launcher3QuickStep: 可作为系统桌面 (CATEGORY_HOME ✅)
设置应用: 无法成为系统桌面 (缺少 CATEGORY_HOME)
```

## 发布它

输出技能见 `outputs/skill-launcher3-basics.md`。
