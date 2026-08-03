# LoRA 高效微调（LoRA Fine-Tuning）

> 冻结预训练模型权重，仅训练低秩分解矩阵：以 < 1% 的参数开销定制专属模型。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 6（SFT 指令微调）、Phase 10 Lesson 11（量化技术）
**Time:** ~90 分钟

## 学习目标

- 理解低秩自适应（LoRA: Low-Rank Adaptation）的数学原理（W = W0 + B * A）
- 实现 QLoRA（4-bit 量化结合 LoRA）在单张消费级 GPU 上的高效微调
- 掌握准备领域微调数据集与设定超参数（r, alpha, dropout）的最佳实践
- 使用 PEFT 与 Unsloth 库完成自定义数据集的微调与权重合并

## 核心问题

全参数微调（Full Fine-Tuning）需要更新模型的数百亿参数，显存开销巨大。LoRA 通过将权重更新量分解为两个低秩矩阵 A 和 B，不仅大幅降低了显存需求，还能在运行时无缝切换不同的适配器（Adapters）。

## 动手构建

参见 `code/main.py`。
