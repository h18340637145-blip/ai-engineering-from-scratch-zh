# 技能：LauncherDatabase 文件夹操作

## 用途
在 Launcher3 的内存数据库中模拟创建文件夹、添加子图标、重命名等 CRUD 操作。

## 核心接口
```python
from main import LauncherDatabase

db = LauncherDatabase()

# 创建文件夹
folder = db.create_folder("工具", cellX=2, cellY=1)

# 添加应用到文件夹
app = db.create_app("日历", "com.android.calendar", cellX=0, cellY=0)
folder.add_child(app)

# 重命名文件夹
folder.rename("效率工具")

print(f"文件夹 id={folder.id}，子图标数={len(folder.children)}")
```

## 数据模型
- `FolderItem`: container=CONTAINER_DESKTOP，item_type=ITEM_TYPE_FOLDER
- `AppItem`: 添加到文件夹后 container 更新为 folder.id
- `LauncherDatabase`: 持有全局 id 计数器，模拟 favorites 表

## 来源
AOSP `packages/apps/Launcher3/src/com/android/launcher3/model/data/FolderInfo.java`
AOSP `packages/apps/Launcher3/src/com/android/launcher3/LauncherProvider.java`
