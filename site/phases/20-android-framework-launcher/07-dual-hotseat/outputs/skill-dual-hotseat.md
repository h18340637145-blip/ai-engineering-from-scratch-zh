# Skill: 双排 Hotseat 坐标映射

## 核心映射函数

```python
def rank_to_dual_row(rank: int, cols: int) -> tuple:
    return rank // cols, rank % cols
```

## 改造检查清单

- [ ] InvariantDeviceProfile.numHotseatIcons 已更新
- [ ] DeviceProfile Hotseat 高度计算已适配
- [ ] Hotseat.java 视图结构改为两行
- [ ] LauncherModel rank 映射已更新
- [ ] DragController 拖拽落点已适配

## 数据库验证

```bash
sqlite3 /data/data/com.android.launcher3/databases/launcher.db \
  "SELECT title, screen, cellX, cellY FROM favorites WHERE container=-101 ORDER BY screen;"
```
