# LangGraph 与状态机 Agent 架构（LangGraph & State Machines）

> 超越线性链：使用显式图结构、状态循环与人工干预构建复杂 Agent 工作流。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 9（Function Calling）
**Time:** ~90 分钟

## 学习目标

- 理解状态机（State Machine）模型在复杂 Agent 系统中的优势
- 掌握 LangGraph 的核心组件：State, Nodes, Edges 与 Conditional Edges
- 实现带循环重试、自我反思（Self-Reflection）与检查点断点恢复的 Agent
- 结合 Human-in-the-loop 实现敏感步骤的人工审批机制

## 核心问题

传统的 LangChain 链式调用缺乏对复杂分支、循环重试和状态持久化的有效控制。基于有向图（DAG/Graph）和明确状态树的架构（如 LangGraph）能够构建出可预测、易调试且健壮的高级 Agent。

## 动手构建

参见 `code/main.py`。
