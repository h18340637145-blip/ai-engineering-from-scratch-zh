# Android 架构与性能决策卡

## 使用时机

新功能架构设计、性能事故复盘或跨进程数据一致性问题时使用。

## 决策顺序

1. 被动 View 与 Presenter 协调适合 MVP。
2. 可观察状态驱动 UI 适合 MVVM，ViewModel 不持有 View。
3. 没有测量证据时先运行 Profiler、Perfetto 或 Macrobenchmark。
4. 只优化已证实的布局、启动、内存或包体热点。
5. 多进程数据用 Provider、Binder 或单写入者，避免 SharedPreferences。
