# Skill: Partner APK 验证工作流

## 验证 Partner APK 是否被发现

```bash
adb shell cmd package query-receivers \
  -a com.android.launcher3.action.PARTNER_CUSTOMIZATION
```

## 壁纸不显示排查清单

1. APK 是否为系统应用（priv-app）？
2. BroadcastReceiver Action 名称是否正确？
3. drawable 名称是否与 string-array 中的条目一致？
4. 图片文件是否已打包进 APK？
5. adb sync 后是否重启？
