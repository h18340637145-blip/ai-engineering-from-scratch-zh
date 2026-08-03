# 大模型 Scaling Laws 扩展定律

> 算力、数据量与参数量的定量法则：Kaplan 与 Chinchilla 的启示。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 Lesson 1, Lesson 12
**Time:** ~45 分钟

## 学习目标

- 理解 Kaplan et al. (2020) 与 Chinchilla (Hoffmann et al., 2022) 扩展定律的区别
- 掌握在给定 FLOPs 预算下，参数量（N）与训练 Token 数（D）的最优配置比例（1:20）
- 学会根据 Scaling Law 预测小模型实验在拉大算力后的 Loss 表现
- 理解 Inference-time Compute Scaling（如 OpenAI o1 / DeepSeek-R1）新范式

## 核心问题

模型效果不是随机提升的，而是严格遵循幂律（Power Law）扩展。了解 Scaling Laws 能够帮助研究员在投入数百万美元训练前，精准规划算力和数据配比。
