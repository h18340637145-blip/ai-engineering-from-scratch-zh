# 嵌入模型与文本表征（Embeddings）

> 将语言转换为高维连续向量空间：语义相似度计算与向量检索的基础。

**Type:** Learn / Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 10 Lesson 1（Tokenizers）、Phase 5 Lesson 22（嵌入模型）
**Time:** ~60 分钟

## 学习目标

- 理解文本嵌入（Text Embeddings）的数学原理与向量空间几何学
- 掌握使用 OpenAI / HuggingFace 生成与归一化 Embedding 的方法
- 深入比较余弦相似度、点积与欧氏距离的适用场景
- 评估不同嵌入维度与分块策略对下游检索任务精度的影响

## 核心问题

计算机无法直接计算文本之间的“意思是否接近”。嵌入模型通过深度神经网络将离散文本映射到高维密集向量空间（如 1536 维），使相似含义的句子在向量空间中距离相近。

## 动手构建

参见 `code/main.py` 与 `code/main.ts`。
