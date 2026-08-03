# 差分注意力机制 V2（Differential Attention V2）

> 消除注意力噪音：通过两组注意力的差值提升相关上下文召回率。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 2
**Time:** ~45 分钟

## 学习目标

- 理解 Differential Attention 的数学推导：`Softmax(Q1 K1^T) - lambda * Softmax(Q2 K2^T)`
- 掌握其消除无关上下文噪音、防范幻觉的原理

## 动手构建

参见 `code/main.py`。
