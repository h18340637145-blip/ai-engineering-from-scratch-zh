# 直接偏好优化家族

> Rafailov 等人（2023）表明，RLHF 的最优解在偏好数据方面具有闭合形式，因此可以跳过显式的奖励模型，直接优化策略。这一见解催生了一个家族 —— IPO、KTO、SimPO、ORPO、BPO，每个都修复了 DPO 的一个失败模式。到 2026 年，直接对齐算法在前沿微调运行中比 PPO 更多。但第二课的过度优化曲线仍然适用：DAAs 并没有逃离 Goodhart，它们只是将 Goodhart 的影响移动到了其他地方。

**Type:** 学习
**Languages:** Python（标准库，六种变体偏好损失比较器）
**Prerequisites:** 第 18 阶段 · 01（InstructGPT），第 18 阶段 · 02（奖励黑客），第 10 阶段 · 08（DPO 基础）
**Time:** ~75 分钟

## 学习目标

- 从 RLHF-with-KL 最优解推导出 DPO 的闭合形式。
- 说明 IPO、KTO、SimPO、ORPO、BPO 每个修复了 DPO 的哪种失败模式。
- 区分“隐式奖励差距”和“偏好强度”，并解释为什么 IPO 的恒等映射很重要。
- 解释为什么 Rafailov 等人（NeurIPS 2024）证明 DAAs 会过度优化，尽管它们没有显式的 RM。

## 问题

RLHF 目标（第一课）：

```
max_pi E_{x,y~pi} [ r(x, y) ] - beta * KL(pi || pi_ref)
```

有一个已知的最优解：

```
pi*(y|x) = (1/Z(x)) * pi_ref(y|x) * exp(r(x, y) / beta)
```

因此，奖励隐式地由最优策略与参考策略的比率定义：

```
r(x, y) = beta * log(pi*(y|x) / pi_ref(y|x)) + beta * log Z(x)
```

将这个代入 Bradley-Terry 偏好似然函数中，分母函数 `Z(x)` 会因为只依赖于 `x` 而被抵消。剩下的只是策略参数的损失 —— 不需要奖励模型。这就是 DPO。

问题所在：推导假设最优解是可达的，偏好数据是分布内的，参考策略是真实的模式锚点。这些假设都不完全成立。每个家族成员修复了不同的被违反的假设。

## 概念

### DPO（Rafailov 等人，2023）

```
L_DPO = -log sigmoid(
  beta * log(pi(y_w | x) / pi_ref(y_w | x))
  - beta * log(pi(y_l | x) / pi_ref(y_l | x))
)
```

可能出现的问题：

- 隐式奖励差距 `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 是无界的。一个微小的偏好可以产生任意大的差距。
- 损失使选择和拒绝的对数概率朝相反方向移动。只要拒绝的下降速度更快，它就可以将选择的绝对对数概率压低。这是“退化选择响应”现象。
- 分布外偏好（罕见对罕见对）会产生任意的隐式奖励。

### IPO（Azar 等人，2024）

身份偏好优化将对数 sigmoid 替换为偏好概率上的恒等映射。损失变为一个有界目标的平方误差：

```
L_IPO = (log(pi(y_w | x) / pi_ref(y_w | x)) - log(pi(y_l | x) / pi_ref(y_l | x)) - 1/(2 beta))^2
```

边距被限制在 `1/(2 beta)`。偏好强度和隐式奖励差距成正比。没有爆炸。

### KTO（Ethayarajh 等人，2024）

Kahneman-Tversky 优化完全去除了成对结构。给定一个单个标记的输出和一个二元“可取”或“不可取”的信号，它映射到前景理论效用：

```
v(x, y) = sigma(beta * log(pi(y|x) / pi_ref(y|x)) - z_ref)
```

对收益和损失有不同的权重（损失厌恶）。好处：你可以使用未配对的数据，这要丰富得多。

### SimPO（Meng 等人，2024）

简单偏好优化将训练信号与生成对齐。完全移除参考策略并按长度归一化对数似然：

```
L_SimPO = -log sigmoid(
  (beta / |y_w|) * log pi(y_w | x)
  - (beta / |y_l|) * log pi(y_l | x)
  - gamma
)
```

使用边距 `gamma` 来稳定。长度归一化消除了利用 DPO 的长度偏差失败模式的动机（通过构造，更长的 `y_w` 会产生更大的对数概率差距）。

### ORPO（Hong 等人，2024）

赔率比偏好优化向标准 SFT 负对数似然中添加了一个偏好项：

```
L_ORPO = L_NLL(y_w) + lambda * L_OR
L_OR = -log sigmoid(log(odds(y_w) / odds(y_l)))
```

没有参考策略 —— SFT 项是正则化项。从基础模型到对齐模型进行单阶段训练。不需要单独的 SFT 检查点。

### BPO（ICLR 2026 提交，OpenReview id=b97EwMUWu7）

识别出“退化选择响应”问题：DPO 保留了 `y_w > y_l` 的排序，但 `y_w` 的绝对对数概率可能会下降。BPO 添加了一个单行修正，对选择响应的下降进行惩罚。在数学推理上，BPO 相比 DPO 在 Llama-3.1-8B-Instruct 上报告了 +10.1% 的准确率。

### 普遍结果：DAAs 仍然过度优化

Rafailov 等人“直接对齐算法中奖励模型过度优化的扩展定律”（NeurIPS 2024）在多个数据集上使用 DPO、IPO、SLiC 训练策略，跨越 KL 预算。黄金奖励与 KL 曲线具有相同的 Gao 等人峰值和崩溃形状。隐式奖励在训练期间查询分布外样本；KL 正则化无法稳定这一点。

DAAs 并没有逃离 Goodhart。它们只是将 Goodhart 的影响从“奖励模型过度优化”转移到了“参考策略比率过度优化”。普遍的修复方法 —— 更好的数据、集成、提前停止 —— 适用于两者。

### 在它们之间选择（2026）

- 如果你有大量配对偏好数据：使用保守的 beta 的 DPO，如果长度偏差明显，使用 SimPO。
- 如果你有未配对的二元反馈：使用 KTO。
- 如果你希望从基础模型进行单阶段流程：使用 ORPO。
- 如果你在 DPO 日志中看到退化选择对数概率：使用 BPO。
- 如果偏好强度差异很大且 DPO 饱和：使用 IPO。

每个实验室都会在一组数据上运行所有五种方法，并根据任务选择胜者。数学推理和安全的最优解没有理由相同。

```figure
dpo-margin
```

## 使用它

`code/main.py` 在一个真实偏好强度随配对变化的玩具偏好数据集上比较了六种损失（DPO、IPO、KTO、SimPO、ORPO、BPO）。每种损失都针对相同的 500 对样本进行优化，使用一个小型 softmax 策略。绘制最终的胜率、选择对数概率漂移和隐式奖励分布。

## 部署它

本课生成 `outputs/skill-preference-loss-selector.md`。根据数据集统计（配对 vs 未配对、变量 vs 均匀偏好强度、长度分布）和目标（单阶段或 SFT-然后-偏好），推荐一个偏好损失并报告它保护的失败模式。

## 练习

1. 运行 `code/main.py`。报告 DPO 和 BPO 的最终选择对数概率下降。BPO 应该保留更高的选择绝对概率 —— 验证这一点。

2. 修改偏好数据，使所有配对具有相同的强度。六种方法中哪种最稳健？哪种退化？解释 IPO 在此的优势。

3. 使拒绝响应的平均长度是选择响应的两倍。在不改变其他任何内容的情况下，数值上展示 DPO 的长度利用和 SimPO 的修复。

4. Rafailov 等人（NeurIPS 2024）声称 DAAs 过度优化。重现单点版本：绘制选择减去拒绝的 KL 散度，并观察 DPO 在大 beta 时的过度优化。

5. 阅读 BPO 论文摘要（OpenReview b97EwMUWu7）。写下 BPO 添加到 DPO 的一行修正。与 `code/main.py` 中的实现进行验证。

## 关键术语

| 术语 | 人们说 | 实际含义 |
|------|-----------------|------------------------|
| DPO | “没有奖励模型的 RLHF” | 从闭合形式 RLHF 最优解推导出的损失；仅策略参数 |
| 隐式奖励 | “对数比率” | `beta * log(pi(y|x) / pi_ref(y|x))` —— DPO 推导出的奖励 |
| IPO | “有界 DPO” | 用恒等映射替换对数 sigmoid；隐式奖励差距上限为 `1/(2 beta)` |
| KTO | “未配对 DPO” | 单标签上的前景理论效用，具有损失厌恶 |
| SimPO | “无参考 DPO” | 长度归一化的对数似然 + 边距；无参考策略 |
| ORPO | “单阶段 DPO” | NLL + 赔率比偏好项；从基础模型单次训练 |
| BPO | “保留选择的 DPO” | DPO 加上对降低选择响应绝对对数概率的惩罚 |
| 退化选择 | “选择下降” | 只要拒绝下降更快，DPO 就会降低选择对数概率 |
| DAA | “直接对齐算法” | 任何跳过显式 RM 的偏好损失方法 |

## 进一步阅读

- [Rafailov 等人 —— 直接偏好优化（NeurIPS 2023，arXiv:2305.18290）](https://arxiv.org/abs/2305.18290)
- [Azar 等人 —— 一种理解从人类偏好中学习的通用理论范式（AISTATS 2024，arXiv:2310.12036）](https://arxiv.org/abs/2310.12036) —— IPO
- [Ethayarajh 等人 —— KTO：基于前景理论优化的模型对齐（arXiv:2402.01306）](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen —— SimPO（NeurIPS 2024，arXiv:2405.14734）](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne —— ORPO（EMNLP 2024，arXiv:2403.07691）](https://arxiv.org/abs/2403.07691)
- [BPO —— 行为保留优化（ICLR 2026 OpenReview b97EwMUWu7）](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov 等人 —— DAAs 中 RM 过度优化的扩展定律（NeurIPS 2024，arXiv:2406.02900）](https://arxiv.org/abs/2406.02900)
