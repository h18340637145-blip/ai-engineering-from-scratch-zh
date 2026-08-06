# Native 日志与调用栈检查卡

## 使用时机

给 C/C++ 系统模块添加调试日志或调用栈时使用。

## 检查顺序

1. `LOG_NDEBUG 0` 用于不禁用 debug 日志。
2. 代码使用 CallStack 时声明 `libutils`。
3. 代码使用 Log API 时声明 `liblog`。
4. 不因习惯添加 `libcutils`；依赖必须对应实际 API。
5. 只编译目标模块，并用 `adb logcat -s <Tag>` 验证。
