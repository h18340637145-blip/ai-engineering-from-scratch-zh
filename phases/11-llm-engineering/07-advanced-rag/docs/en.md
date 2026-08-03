# 高级 RAG 架构（Advanced RAG）

> 从简单的向量匹配走向多阶段管道：重排序、查询改写、混合检索与自适应 RAG。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 11 Lesson 6（RAG 基础）
**Time:** ~90 分钟

## 学习目标

- 掌握混合检索（Hybrid Search：BM25 稀疏检索 + Vector 密集检索）
- 实现交叉编码器重排序（Cross-Encoder Reranking）精细过滤候选 Chunk
- 应用查询扩展（Query Expansion）、HyDE（Hypothetical Document Embeddings）与查询改写
- 搭建带有自我评估与路由的 Self-RAG / Corrective RAG 流程

## 核心问题

基础 RAG（Embedding 向量匹配）容易受到词汇错配、短查询语义模糊以及包含无关段落的干扰。高级 RAG 引入多阶段管道与重排序，极大地提高了检索精度和相关度。

## 动手构建

参见 `code/main.py`。
