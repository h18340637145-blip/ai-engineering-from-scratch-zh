# 缓存优化与成本控制（Caching & Cost Optimization）

> 降低 API 开销与响应延迟：语义缓存、Token 预算控制与模型分级路由。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 4（Embeddings）
**Time:** ~60 分钟

## 学习目标

- 实现精确匹配缓存（Exact Match Cache）与基于 Embedding 的语义缓存（Semantic Cache）
- 构建智能模型路由器（Model Router）：小任务用小模型，复杂任务路由至强模型
- 掌握 Token 预算限制、流式传输（Streaming）与中间响应截断策略
- 计算并优化端到端系统的吞吐量与单请求成本

## 核心问题

随着大模型应用的规模化，高昂的 API 调用账单与较高的首 Token 延迟（TTFT）成为制约商业化落地的核心阻碍。通过缓存相似问题和动态模型分级，可以在保障体验的同时节省高达 80% 的成本。

## 动手构建

参见 `code/main.py`。
