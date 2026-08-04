# 少样本与思维链提示（Few-Shot & Chain-of-Thought Prompting）

> 给模型提供思考的“计算空间”：通过示范与中间推理步骤释放复杂逻辑解决能力。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 11 Lesson 1（Prompt 工程）
**Time:** ~60 分钟

## 学习目标

- 掌握 Zero-Shot, Few-Shot, CoT (Chain-of-Thought) 以及 Zero-Shot-CoT 机制
- 设计针对数学、代码分析与多步逻辑推理的高质量 Few-Shot 示例
- 理解自一致性采样（Self-Consistency Sampling）提升推理稳定性的原理
- 实现动选示例（Dynamic Example Selection）检索系统

## 核心问题

直接向模型索要复杂问题的最终结论往往会导致错误，因为 Transformer 生成单 Token 的计算能力有限。思维链（CoT）通过让模型显式输出中间推导 Token，为其赋予了“边想边写”的计算空间。

## 动手构建

参见 `code/main.py`。
