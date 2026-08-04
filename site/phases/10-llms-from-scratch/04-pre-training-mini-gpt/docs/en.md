# 预训练微型 GPT（Pre-training Mini-GPT）

> 从随机初始化参数开始，在真实文本语料上完整训练一个微型自回归 LLM。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 7（GPT 架构）、Phase 10 Lesson 1-3
**Time:** ~120 分钟

## 学习目标

- 组装包含 Embedding、Transformer Block、LayerNorm 和 LM Head 的 Mini-GPT 模型
- 实现带有 Cosine 学习率衰减、Warmup 与 Gradient Clipping 的训练循环
- 监控交叉熵损失（Cross-Entropy Loss）与困惑度（Perplexity）的下降过程
- 实现自回归采样生成（Greedy / Top-k / Top-p Sampling）

## 动手构建

参见 `code/main.py`。
