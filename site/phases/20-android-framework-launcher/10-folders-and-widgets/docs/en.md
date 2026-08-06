# 文件夹与小组件：数据库更新与默认小部件添加

> 以运行实现为中心的 Android Framework 课程

**Type:** Build
**Languages:** Python
**Prerequisites:** 09-recents-and-quickstep
**Time:** ~45 分钟

## 学习目标

见 quiz.json 的 pre/check/post 阶段问题。

## 概念

文件夹和小组件都存储在 favorites 表，通过 container 和 itemType 区分。

### 文件夹数据模型

```text
文件夹 Item:   itemType=2 (ITEM_TYPE_FOLDER), container=CONTAINER_DESKTOP
子图标:        container = 文件夹 Item 的 id
```

### 文件夹创建日志位置

```text
FolderNameEditText → FolderInfo.setTitle() → LauncherModel.updateItemInDatabase()
```

### 验证文件夹

```bash
# 查询所有文件夹
adb shell sqlite3 /data/data/com.android.launcher3/databases/launcher.db \
  "SELECT id, title FROM favorites WHERE itemType=2;"

# 查询文件夹内的图标
adb shell sqlite3 .../launcher.db \
  "SELECT title, cellX, cellY FROM favorites WHERE container=<folder_id>;"
```

### 默认小组件添加方式

1. 在默认布局 XML 中声明 `<appwidget>` 节点
2. 或在 LoaderTask 首次加载回调中用 AppWidgetManager API 添加

### FeatureFlags 学习方法

```bash
# 查看所有 Launcher 相关的 device_config
adb shell device_config list launcher

# 打开某个 flag
adb shell device_config put launcher ENABLE_TWO_PANEL_HOME true
```

## 构建它

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```
