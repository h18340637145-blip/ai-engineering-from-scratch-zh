# 分布式训练与扩展（Scaling & Distributed Training）

> 从单卡到千卡集群：数据并行、张量并行与流水线并行深度拆解。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 4（预训练 Mini-GPT）
**Time:** ~90 分钟

## 学习目标

- 理解 DDP（数据并行）、FSDP（全深度切片数据并行）与 ZeRO-1/2/3 优化器状态切分
- 掌握 Tensor Parallelism（Megatron-LM 风格的张量并行）切分 QKV 和 MLP 矩阵的原理
- 掌握 Pipeline Parallelism（1F1B 调度的流水线并行）减少气泡（Bubble）的机制
- 在 PyTorch Distributed 中运行多 GPU 混合并行训练演示

## 动手构建

参见 `code/main.py`。
