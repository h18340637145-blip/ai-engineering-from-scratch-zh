# 检索增强生成基础（RAG - Retrieval-Augmented Generation）

> 弥合大模型预训练知识的滞后与私有数据壁垒：检索真实文档并作为上下文注入。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 4（嵌入模型与文本表征）
**Time:** ~75 分钟

## 学习目标

- 深入理解 RAG 架构的三个核心阶段：Indexing（索引）、Retrieval（检索）与 Generation（生成）
- 实现文本分块（Chunking）、Embedding 生成与向量数据库存取
- 构建包含 Top-k 检索与上下文拼装的基础 RAG 管道
- 评估标准 RAG 系统面临的挑战（如幻觉、召回不全、噪音干扰）

## 核心问题

LLM 的知识停留在预训练截止时刻，且无法访问企业内部私有数据。RAG 通过在回答问题前从向量数据库中检索相关文档块（Chunks），将其作为事实依据写入上下文，从而使模型能够精准回答最新或特定域的问题。

## 动手构建

参见 `code/main.py`。
