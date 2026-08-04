# 位置编码（Positional Encoding）

> 自注意力机制具有置换不变性。如果不显式添加位置信息，"dog bites man" 和 "man bites dog" 在模型看来将完全相同。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 2（自注意力机制）
**Time:** ~60 分钟

## 学习目标

- 仅使用 NumPy 实现 Vaswani 等人提出的正弦位置编码（Sinusoidal Positional Encoding）
- 解释为什么正弦波交替编码能够允许模型通过线性变换表示相对位置
- 实现可学习的位置嵌入（Learned Positional Embeddings），并对比其与正弦编码在长序列外推上的优劣
- 深入理解旋转位置编码（RoPE, Rotary Position Embedding）及其在现代 LLM 中的广泛应用

## 核心问题

因为自注意力在序列维度上并行计算 `Q @ K^T`，它对于 Token 的先后顺序没有任何内置感知。颠倒输入 Token 序列的顺序，输出将经历相同的置换。这对于语言模型是不可接受的。

为了给模型引入位置顺序感，必须将位置向量加到输入 Token 嵌入中，或者在计算注意力时融合位置矩阵。

## 概念详解

### 1. 正弦位置编码（Sinusoidal Positional Encoding）

Vaswani 等人在 2017 年提出的固定编码公式：

```
PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))
```

这种设计的巧妙之处在于：对于任何固定的偏移量 `k`，`PE(pos + k)` 可以表示为 `PE(pos)` 的线性映射。这使模型非常容易学会关注相对距离。

### 2. 旋转位置编码（RoPE）

现代模型（LLaMA、Qwen、DeepSeek）普遍采用旋转位置编码（RoPE）。RoPE 不在输入层相加，而是在注意力计算前将 Query 和 Key 向量在复平面上旋转与位置对应的角度：

```
R_pos * q_pos
```

这保证了 `(R_m q_m)^T (R_n k_n) = q_m^T R_{n-m} k_n`，即点积只取决于相对距离 `m - n`。

## 动手构建

参见 `code/main.py` 获取 NumPy 和 PyTorch 实现。

## 练习题

1. 实现 RoPE（旋转位置编码）的 2D 旋转矩阵版本。
2. 绘制正弦位置编码热力图并观察频率变化趋势。
3. 比较可学习位置编码超出现有训练长度时的表现。

## 核心术语

| 术语 | 真实含义 |
|------|-----------------------|
| Sinusoidal PE | 基于正弦和余弦函数的固定绝对位置编码 |
| RoPE (Rotary Position Embedding) | 通过复数旋转将相对位置显式编码入 Q/K 点积的技术 |
| Permutation Invariance | 缺乏位置感知时自注意力对输入顺序的不敏感性 |
