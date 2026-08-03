# Memory Blocks 与睡眠时间计算：异步记忆整理与归纳

> 借鉴生物学中睡眠对记忆重构与巩固（Memory Consolidation）的作用，Agent 系统利用离线/异步计算资源对原始会话积攒的碎片记忆进行清洗、去重、冲突消除与高阶概念抽取。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 Lesson 07
**Time:** ~60 分钟

## 学习目标

- 掌握内存块（Memory Blocks）的结构化定义与划分原则。
- 设计睡眠时间（Sleep-Time / Background）异步计算流水线。
- 实现记忆的矛盾检测、衰减遗忘与高阶总结蒸馏。
- 保障前台实时对话与后台记忆整理并发一致性。

## 机制解析

```mermaid
graph LR
    User[用户交互] -->|产生原始日志| EventLog[Raw Conversation Logs]
    EventLog -->|前台低延迟读取| ActiveMemory[Active Context]

    subgraph Background Task (Sleep-Time Compute)
        EventLog -->|后台异步拉取| Consolidation[记忆巩固与蒸馏]
        Consolidation -->|冲突消解 & 结构化| Blocks[Structured Memory Blocks]
    end

    Blocks -->|更新| ActiveMemory
```

## 动手实现

运行 `code/main.py` 观察后台睡眠计算任务如何将混乱的对话历史清洗重构为紧凑的 Memory Blocks：

```bash
python3 code/main.py
```
