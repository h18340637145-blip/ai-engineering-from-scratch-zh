# 梯度检查点（Gradient Checkpointing）

> 用计算换显存：深层模型训练必备的显存节省利器。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 3, Phase 10 Lesson 4
**Time:** ~45 分钟

## 学习目标

- 理解在前向传播时不保存中间激活值，而在反向传播时重新计算的机制
- 在 PyTorch 中应用 `torch.utils.checkpoint` 节省 60%+ 的激活显存

## 动手构建

参见 `code/main.py`。
