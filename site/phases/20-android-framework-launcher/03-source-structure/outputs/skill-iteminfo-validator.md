# Skill: ItemInfo 坐标校验

## 核心规则

对于网格 `columns × rows`，一个 ItemInfo 合法的条件：

- `cellX >= 0`
- `cellY >= 0`
- `spanX >= 1`, `spanY >= 1`
- `cellX + spanX <= columns`
- `cellY + spanY <= rows`

## 调试查询

```bash
# 查询 launcher.db 中所有桌面 Item 的坐标
adb shell sqlite3 \
  /data/data/com.android.launcher3/databases/launcher.db \
  "SELECT title, cellX, cellY, spanX, spanY FROM favorites WHERE container=-100;"
```

## 常见越界原因

- 修改网格列数后旧数据未清除
- 默认布局 XML 坐标超出设备行/列数
- Partner APK 布局与主布局叠加
