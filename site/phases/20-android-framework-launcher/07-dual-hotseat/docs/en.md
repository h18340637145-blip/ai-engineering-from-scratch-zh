# 双排 Hotseat：从数据到视图的协同改造

**Type:** Build
**Languages:** Python
**Prerequisites:** 06-partner-overlay
**Time:** ~60 分钟

## 学习目标

- 解释 Hotseat 在 Launcher3 中的数据模型（rank → cellX 映射）
- 列出实现双排 Hotseat 需要改动的五个层次
- 说明 DeviceProfile 中 Hotseat 相关参数的作用
- 分析双排 Hotseat 中图标拖拽目标计算的变化
- 编写 rank 到双排坐标的映射函数

## 概念

Hotseat 不是单纯的视觉改造，而是数据、配置、绑定、布局和拖拽的协同修改。

### Hotseat 数据模型

在单排 Hotseat 中，图标的 screenId 字段存储的是 rank（位置序号，从 0 开始），
而不是 Workspace 的页面 ID：

```java
// favorites 表中 Hotseat 图标的字段
container = CONTAINER_HOTSEAT  // -101
screen = rank                  // 0, 1, 2, 3, 4
cellX = rank                   // 与 screen 相同（单排情况）
cellY = 0
```

### 双排 Hotseat 改造层次

| 层次 | 改动内容 |
|---|---|
| InvariantDeviceProfile | numHotseatIcons 翻倍，定义行数 |
| DeviceProfile | Hotseat 高度、图标间距计算 |
| Hotseat.java | 视图结构改为两行 CellLayout |
| LauncherModel | rank 映射到 (row, col) 坐标 |
| DragController | 拖拽落点计算适配双排 |

### rank → 双排坐标映射

假设每排 5 个图标，rank 0-4 在第一行，rank 5-9 在第二行：

```python
def rank_to_cell(rank: int, cols_per_row: int):
    row = rank // cols_per_row
    col = rank % cols_per_row
    return row, col
```

### 验证查询

```bash
# 查看 Hotseat 中所有图标的坐标
adb shell sqlite3 /data/data/com.android.launcher3/databases/launcher.db \
  "SELECT title, screen, cellX, cellY FROM favorites WHERE container=-101;"
```

## 构建它

实现一个 Python 脚本，将单排 Hotseat rank 转换为双排坐标，并验证转换结果，模拟 LauncherModel 中的坐标映射逻辑。

参见：`code/main.py`

## 使用它

```bash
python3 main.py
```

## 发布它

输出技能见 `outputs/skill-dual-hotseat.md`。
