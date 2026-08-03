# GPT 与因果语言模型（GPT & Causal Language Modeling）

> 仅使用解码器与因果掩码，GPT 将 NLP 统一为了自回归下一个 Token 预测任务。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 5（完整 Transformer 架构）
**Time:** ~60 分钟

## 学习目标

- 理解单向自回归 Transformer 解码器架构
- 实现上三角因果注意力掩码（Causal Attention Mask）
- 理解下一个 Token 预测（Next-Token Prediction）如何涌现出上下文学习（In-Context Learning）能力
- 编写从 logits 计算交叉熵损失并生成文本的完整循环

## 核心问题

与 BERT 观察双向上下文不同，GPT（Generative Pre-trained Transformer）专注于自回归生成。通过在海量无标注文本上仅预测下一个 Token，GPT 展现出了惊人的泛化能力。

## 概念详解

因果掩码矩阵形式：

```
1  0  0  0
1  1  0  0
1  1  1  0
1  1  1  1
```

在 Softmax 之前，所有 0 对应的位置被置为 `-inf`。

## 动手构建

参见 `code/main.py`。
