# Agent 框架选型与权衡（Agent Framework Trade-offs）

> 从零手写 vs 选用框架：LangChain, LlamaIndex, AutoGen, CrewAI 与 Custom Engine 深度对比。

**Type:** Learn / Reference
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lessons 1-16
**Time:** ~60 分钟

## 学习目标

- 评估主流量子与 Agent 框架（LangChain, LlamaIndex, AutoGen, CrewAI）的优缺点
- 掌握“何时使用现有框架，何时手写轻量级控制循环”的决策树
- 分析框架抽象过重（Over-abstraction）带来的黑盒调试难题与性能损失
- 设计轻量、可维护且无vendor-lockin 的生产级 Agent 架构

## 核心问题

市场上存在大量 Agent 框架，但许多框架封装过度，导致在遇到边界情况时难以调试。理解各框架的适用边界并具备从零手写轻量 Agent 引擎的能力，是资深 AI 工程师的关键区别。

## 动手构建

参见 `code/main.py`。
