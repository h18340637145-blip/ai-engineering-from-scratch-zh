# Launcher3 源码目录与核心架构

> 读懂目录结构是阅读源码的地图；理解核心对象是修改任何功能的前提。

**Type:** Learn
**Languages:** Python
**Prerequisites:** 02-logcat-and-tooling
**Time:** ~35 分钟

## 学习目标

- 定位 Launcher3 源码在 AOSP 中的位置及主要子目录
- 说明 Launcher / Workspace / Hotseat / DragLayer / LauncherModel 各自的职责
- 描述 `ItemInfo` 的核心字段及其在数据库中的映射
- 画出 Launcher3 从启动到绑定数据的单向流水线
- 解释 `DeviceProfile` 与 `InvariantDeviceProfile` 的区别

## 概念

### 源码根目录

```text
packages/apps/Launcher3/
├── Android.bp
├── AndroidManifest.xml
├── res/
│   ├── layout/          # 视图布局 XML
│   ├── values/          # 字符串、尺寸、颜色
│   └── xml/             # 默认桌面布局配置
├── src/com/android/launcher3/
│   ├── Launcher.java         # 主 Activity
│   ├── LauncherModel.java    # 数据模型入口
│   ├── LauncherProvider.java # launcher.db 的 ContentProvider
│   ├── Workspace.java        # 桌面分页容器
│   ├── Hotseat.java          # 底部常驻区域
│   ├── DragLayer.java        # 最外层拖拽容器
│   ├── PagedView.java        # 分页滑动基类
│   ├── DeviceProfile.java    # 实际布局参数
│   ├── InvariantDeviceProfile.java  # 网格、图标数等稳定配置
│   ├── model/
│   │   ├── LoaderTask.java   # 后台数据加载线程
│   │   └── LoaderCursor.java # 数据库逐行读取校验
│   └── widget/
└── quickstep/src/com/android/quickstep/
    ├── TouchInteractionService.java   # 手势导航全局入口
    ├── RecentsView.java               # 最近任务界面
    └── TaskAnimationManager.java
```

### 核心对象职责

| 类 | 职责 |
|---|---|
| `Launcher` | 主 Activity，生命周期、视图初始化、状态切换 |
| `Workspace` | 桌面分页容器，承载图标、文件夹和小部件 |
| `Hotseat` | 底部常驻区域，本质上是一个 CellLayout |
| `DragLayer` | 最外层容器，承载拖拽、弹窗和触摸分发 |
| `LauncherModel` | 数据模型入口，负责加载、更新和绑定 |
| `LoaderTask` | 后台线程加载桌面和应用列表数据 |
| `LauncherProvider` | 创建和升级 `launcher.db` |
| `ItemInfo` | 桌面元素（图标/文件夹/小部件）的通用数据结构 |
| `DeviceProfile` | 当前设备下实际的尺寸和布局参数 |
| `InvariantDeviceProfile` | 网格列数、图标数等相对稳定的配置 |

### ItemInfo 核心字段

```java
public int id;        // 数据库记录 ID
public int itemType;  // ITEM_TYPE_APPLICATION / FOLDER / APPWIDGET
public int container; // CONTAINER_DESKTOP / CONTAINER_HOTSEAT
public int screenId;  // Workspace 页面 ID 或 Hotseat rank
public int cellX;     // 网格 X 坐标
public int cellY;     // 网格 Y 坐标
public int spanX;     // 横向占用网格数
public int spanY;     // 纵向占用网格数
```

### 数据加载流水线（单向）

```mermaid
graph LR
    A["launcher.db"] --> B["LoaderCursor\n逐行校验"]
    B --> C["LoaderTask\n组装 ItemInfo 列表"]
    C --> D["LauncherModel\n触发绑定回调"]
    D --> E["Launcher / Workspace\n渲染到屏幕"]
```

## 构建它

实现一个 Python 脚本，模拟 `LoaderCursor` 校验 ItemInfo 坐标合法性的逻辑：给定一个网格尺寸（列 × 行），检查每个 Item 的 `(cellX, cellY, spanX, spanY)` 是否越界，并过滤非法项。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

输出示例：

```text
=== 校验 5 列 × 5 行 网格 ===
✅ App [cellX=0, cellY=0, span=1×1] 合法
✅ Widget [cellX=1, cellY=2, span=2×2] 合法
❌ App [cellX=4, cellY=0, span=2×1] 越界（X 方向）
❌ App [cellX=0, cellY=5, span=1×1] 越界（Y 方向）

合法 Item: 2 / 4
```

## 发布它

输出技能见 `outputs/skill-iteminfo-validator.md`。
