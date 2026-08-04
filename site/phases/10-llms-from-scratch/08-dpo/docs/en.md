# 直接偏好优化（DPO - Direct Preference Optimization）

> 无需显式奖赏模型与 RL 采样：通过隐式奖励闭式解直接优化偏好概率。

**Type:** Learn / Build
**Languages:** Python
**Prerequisites:** Phase 10 Lesson 7（RLHF）
**Time:** ~75 分钟

## 学习目标

- 推导 DPO 的数学公式：将 Bradley-Terry 偏好模型转化为纯交叉熵目标
- 对比 DPO 与传统 RLHF 在训练稳定性、显存占用与收敛速度上的优势
- 实现 DPO 损失函数：处理 Winner / Loser 序列的策略与参考 Logits
- 在自定义偏好数据集上运行 DPO 训练并验证输出分布变化

## 动手构建

参见 `code/main.py`。
