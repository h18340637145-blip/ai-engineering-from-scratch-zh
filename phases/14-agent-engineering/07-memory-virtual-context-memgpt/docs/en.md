# MemGPT：操作系统式虚拟上下文管理与多层存储 hierarchy

> MemGPT（Packer et al., 2023）借鉴操作系统的虚拟内存管理思想，将有限的 LLM 上下文窗口视为内存（RAM），将持久化向量库与数据库视为磁盘（Disk），通过 Agent 主动内存指令实现超越窗口限制的长期记忆。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 Lesson 01, Lesson 06
**Time:** ~60 分钟

## 学习目标

- 理解 MemGPT 的分层内存架构：Working Memory（Core Memory + Context Window）、Recall Memory 与 Archival Memory。
- 实现主动内存管理工具（`core_memory_append`, `core_memory_replace`, `archival_memory_insert`, `archival_memory_search`）。
- 掌握如何通过固定的 Core Memory 区块维护长期用户画像（Human Block）与角色定义（Persona Block）。
- 构建具备自我记忆维护能力的跨会话长寿命 Agent。

## 架构示意

```mermaid
graph TD
    subgraph Context Window (RAM)
        System[System Prompt & Persona/Human Core Memory]
        Messages[近期对话 Context Window]
    end

    subgraph External Storage (Disk)
        Recall[Recall Memory: 历史对话日志库]
        Archival[Archival Memory: 向量知识库与海量文档]
    end

    Agent[LLM Agent] -->|主动调用工具| MemTools[Memory Management Functions]
    MemTools -->|编辑| System
    MemTools -->|检索/写入| Recall
    MemTools -->|向量检索/写入| Archival
```

## 动手实现

运行 `code/main.py` 观察 Agent 如何通过工具指令自主读写 Core Memory 与 Archival Memory：

```bash
python3 code/main.py
```
