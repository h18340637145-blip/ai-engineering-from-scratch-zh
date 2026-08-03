# 基于人类反馈的强化学习（RLHF & PPO）

> 对齐人类偏好：奖赏模型训练与 PPO 强化学习算法。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 6（SFT）、Phase 9（强化学习）
**Time:** ~90 分钟

## 学习目标

- 掌握基于 Pairwise 偏对数据训练 Reward Model（奖赏模型）的 Loss 函数
- 理解 PPO（Proximal Policy Optimization）在 LLM 对齐中的四个网络：Policy、Value、Reference 与 Reward
- 实现 KL 散度惩罚（KL Penalty），防止 Policy 模型过度偏离参考模型
- 搭建微型 RLHF 对齐训练循环

## 动手构建

参见 `code/main.py`。
