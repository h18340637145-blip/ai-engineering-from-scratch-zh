# 词嵌入：Word2Vec（Word Embeddings: Word2Vec）

> 依据上下文分布推断语义：Skip-gram, CBOW 与负采样（Negative Sampling）。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 3（深度学习核心）、Phase 5 Lesson 1
**Time:** ~75 分钟

## 学习目标

- 理解分布假说（Distributional Hypothesis）：“观其伴而知其意”
- 掌握 Skip-gram 与 CBOW (Continuous Bag-of-Words) 架构的区别
- 推导负采样（Negative Sampling）与层次 Softmax（Hierarchical Softmax）降低计算精度的推导
- 使用 PyTorch 从零训练 Word2Vec 并可视化词向量语义聚类（如 `king - man + woman = queen`）

## 动手构建

参见 `code/main.py`。
