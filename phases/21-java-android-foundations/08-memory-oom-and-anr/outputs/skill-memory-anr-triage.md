# 内存、OOM 与 ANR 分类卡

## 使用时机

收到“卡死”“闪退”“被系统杀掉”报告时，先用本卡分类。

## 证据顺序

1. `FATAL EXCEPTION`：读取首个业务栈帧，按 Crash 处理。
2. `Input dispatching timed out`：读取 main 线程、锁和 Binder 等待，按 ANR 处理。
3. `OutOfMemoryError`：检查大对象、缓存和泄漏链。
4. `lmkd Killing`：检查 OOM adj、系统总内存与回收频率。
5. 静态 Activity、未取消回调、未注销监听器：逐个切断生命周期外引用。
