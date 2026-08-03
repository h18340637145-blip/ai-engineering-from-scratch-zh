# BERT 与掩码语言模型（BERT & Masked Language Modeling）

> 通过掩盖 15% 的 Token 并预测它们，BERT 学会了双向深度文本表示。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 5（完整 Transformer 架构）
**Time:** ~60 分钟

## 学习目标

- 理解 BERT 的双向 Transformer 编码器架构与预训练目标
- 实现掩码语言模型（MLM）的数据遮蔽逻辑（80% [MASK], 10% 随机词, 10% 保持不变）
- 理解下一句预测（NSP）任务及其在后续研究中的演变
- 使用纯 Python/PyTorch 实现一个微型 BERT 模型并完成微调测试

## 核心问题

GPT 等自回归模型只能单向（从左到右）观察上下文。对于阅读理解、命名实体识别和分类等任务，同时观察上下文两侧的信息至关重要。BERT（Bidirectional Encoder Representations from Transformers）通过 MLM 预训练实现了真正的双向表示。

## 概念详解

### MLM 掩蔽策略

在预训练期间，随机选择 15% 的 Token：
- 80% 的情况下：替换为 `[MASK]` 标记
- 10% 的情况下：替换为随机的 Token
- 10% 的情况下：保持原有 Token 不变

这种设计防止了模型在下游任务中因从未见过 `[MASK]` 标记而产生 mismatch。

## 动手构建

参见 `code/main.py` 获取 MLM 遮蔽生成器与模型训练逻辑。
