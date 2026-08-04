# Prompt 缓存机制（Prompt Caching）

> 利用前缀复用节省高达 90% 的长上下文 Token 费用与推理延迟。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 5（上下文工程）、Phase 7 Lesson 12（KV Cache）
**Time:** ~60 分钟

## 学习目标

- 理解 Prompt Caching 在大模型 API 供应商侧（如 Anthropic / OpenAI）的运作原理
- 掌握 Prompt 前缀一致性（Prefix Consistency）与 Cache Key 命中规律
- 优化长文档分析、大代码库上下文与固定 System Prompt 的缓存命中率
- 量化分析启用 Prompt Caching 后的成本与延迟降低效果

## 核心问题

当每次请求都包含大量重复的背景文档或长 System Prompt 时，重新处理这些 Token 会造成极大的算力浪费。Prompt Caching 在服务器端复用已计算的 KV Cache 状态，大幅降低了重复前缀的计费与首 Token 延时。

## 动手构建

参见 `code/main.py`。
