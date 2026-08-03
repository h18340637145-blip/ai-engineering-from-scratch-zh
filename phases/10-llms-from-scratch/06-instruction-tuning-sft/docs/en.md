# 指令微调与 SFT（Instruction Tuning & SFT）

> 让预训练模型听懂人类指令：从补全续写迈向问答对话。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 4
**Time:** ~75 分钟

## 学习目标

- 掌握 SFT（Supervised Fine-Tuning）的数据集格式（ChatML / User-Assistant 模板）
- 实现掩码交叉熵损失（Masked Cross-Entropy Loss），仅计算 Assistant 响应部分的梯度
- 理解高质量指令数据对模型遵循能力与语气塑造的关键作用
- 完成基于 PyTorch 的微型指令微调流程

## 动手构建

参见 `code/main.py`。
