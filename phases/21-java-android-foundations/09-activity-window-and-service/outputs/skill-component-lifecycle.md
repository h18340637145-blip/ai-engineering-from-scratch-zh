# Android 组件生命周期检查卡

## 使用时机

排查旋转、返回前台、Fragment 空引用、启动模式或后台服务问题时使用。

## 检查步骤

1. 不依赖 `onDestroy()` 必然执行；停止状态保存可恢复状态。
2. Fragment 的 View 观察者绑定 `viewLifecycleOwner`。
3. 通过 Activity → PhoneWindow → DecorView 理解 View 树归属。
4. `singleTop` 只复用栈顶实例；`singleTask` 会清除其上 Activity。
5. 长期用户可见工作使用前台服务；延后短任务优先考虑 WorkManager。
