# 完整 Transformer 架构（Full Transformer Architecture）

> 将自注意力、前馈网络、层归一化与残差连接组装成一个完整的 Transformer 编码器-解码器模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lessons 1-4
**Time:** ~90 分钟

## 学习目标

- 从零构建 Transformer 编码器块（Encoder Block）与解码器块（Decoder Block）
- 正确实现 Pre-LN 与 Post-LN 架构并理解其训练稳定性差异
- 整合因果掩码与编码器-解码器交叉注意力（Cross-Attention）
- 搭建一个端到端的 Transformer 序列到序列模型并运行前向传播

## 核心问题

前面几课中我们分别构建了注意力机制和位置编码。要搭建一个完整的 Transformer，我们需要将这些模块通过残差连接（Residual Connections）、层归一化（Layer Normalization）以及逐位置前馈网络（Feed-Forward Network, FFN）组合起来。

## 概念详解

### 编码器块结构

```
Input -> Positional Encoding -> [ Pre-LN -> Multi-Head Attention -> Add ] -> [ Pre-LN -> FFN -> Add ] -> Output
```

### FFN 模块

前馈网络通常由两个线性变换组成，中间包含 ReLU 或 SwiGLU 激活函数：

```
FFN(x) = max(0, x W1 + b1) W2 + b2
```

维度通常从 `d_model` 扩展到 `4 * d_model` 再还原。

## 动手构建

参见 `code/main.py` 完整实现。

## 练习题

1. 将标准的 FFN 替换为 SwiGLU 激活函数。
2. 比较 Pre-LN 与 Post-LN 在深层 Transformer 中的梯度传播情况。
