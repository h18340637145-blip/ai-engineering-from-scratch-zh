# 随机过程与马尔可夫链

> 建模随时间演化的随机系统。掌握马尔可夫无记忆性、平稳分布与扩散过程。

**Type:** 学习
**Language:** Python
**Prerequisites:** Phase 1, Lesson 06 (概率论与概率分布)
**Time:** ~45 分钟

## 学习目标

- 模拟 1D 和 2D 随机游走并验证位移的 sqrt(n) 缩放
- 构建马尔可夫链模拟器并通过特征分解计算其平稳分布
- 实现 Metropolis-Hastings MCMC 和 Langevin 动力学以从目标分布中采样
- 将前向扩散过程与布朗运动联系起来，并解释反向过程如何生成数据

## 问题

许多 AI 系统涉及随时间演化的随机性。不是静态的随机性——而是结构化的、顺序的随机性，每一步都依赖于之前的内容。

语言模型逐个生成标记。每个标记都依赖于之前的上下文。模型输出一个概率分布，从中采样，然后继续。这是一个随机过程。

扩散模型逐步向图像添加噪声，直到它变成纯静态。然后它们反转这个过程，逐步去噪，直到出现新的图像。前向过程是一个马尔可夫链。反向过程是一个学习到的反向运行的马尔可夫链。

强化学习代理在环境中采取动作。每个动作以一定的概率导致一个新的状态。代理在一个随机的世界中遵循一个随机的策略。整个过程是一个马尔可夫决策过程。

MCMC 采样——贝叶斯推断的支柱——构建一个平稳分布为你要采样的后验分布的马尔可夫链。

所有这些都基于四个基本概念：
1. 随机游走——最简单的随机过程
2. 马尔可夫链——具有转移矩阵的结构化随机性
3. Langevin 动力学——带有噪声的梯度下降
4. Metropolis-Hastings——从任何分布中采样

## 概念

### 随机游走

从位置 0 开始。在每一步，抛一枚公平的硬币。正面：向右移动 (+1)。反面：向左移动 (-1)。

经过 n 步后，你的位置是 n 个随机的 +/-1 值的总和。期望位置是 0（行走是无偏的）。但期望的与原点的距离随着 sqrt(n) 增长。

这是反直觉的。行走是公平的——在任何方向都没有漂移。但随着时间的推移，它会越来越远离起点。n 步后的标准差是 sqrt(n)。```
Step 0:  Position = 0
Step 1:  Position = +1 or -1
Step 2:  Position = +2, 0, or -2
...
Step 100: Expected distance from origin ~ 10 (sqrt(100))
Step 10000: Expected distance from origin ~ 100 (sqrt(10000))
```**在二维情况下**，行走会以相等的概率向上、向下、向左或向右移动。同样的 sqrt(n) 缩放规律适用于距离原点的距离。路径会描绘出类似分形的图案。

**为什么是 sqrt(n)？** 每一步以相等的概率为 +1 或 -1。经过 n 步之后，位置 S_n = X_1 + X_2 + ... + X_n，其中每个 X_i 为 ±1。每一步的方差为 1，且各步之间相互独立，因此 Var(S_n) = n。标准差 = sqrt(n)。根据中心极限定理，S_n / sqrt(n) 收敛到标准正态分布。

这个 sqrt(n) 缩放规律在机器学习中随处可见。SGD 噪声的缩放比例为 1/sqrt(batch_size)。嵌入维度的缩放比例为 sqrt(d)。平方根是独立随机添加的特征标志。

**与布朗运动的联系。** 取一个步长为 1/sqrt(n) 的随机行走，并在每个单位时间内进行 n 步。当 n 趋近于无穷大时，行走会收敛到布朗运动 B(t) —— 一个连续时间过程，其中 B(t) 服从均值为 0、方差为 t 的正态分布。

布朗运动是扩散的数学基础。它模拟了流体中粒子的随机抖动、股票价格的波动，以及——最关键的是——扩散模型中的噪声过程。

**赌徒破产。** 一个从位置 k 出发的随机行走者，具有在 0 和 N 处的吸收屏障。在到达 0 之前到达 N 的概率是多少？对于公平的行走：P(到达 N) = k/N。这令人惊讶地简单且优雅。它与鞅理论相关——公平的随机行走是一个鞅（未来期望值 = 当前值）。

### 马尔可夫链

马尔可夫链是一个根据固定概率在状态之间转换的系统。其关键属性：下一个状态仅取决于当前状态，而不取决于历史。```
P(X_{t+1} = j | X_t = i, X_{t-1} = ...) = P(X_{t+1} = j | X_t = i)
```这就是马尔可夫性质。它意味着你可以用一个转移矩阵 P 来描述整个动态过程：```
P[i][j] = probability of going from state i to state j
```P 的每一行加起来都等于 1（你必须去某个地方）。

**示例 -- 天气：**```
States: Sunny (0), Rainy (1), Cloudy (2)

P = [[0.7, 0.1, 0.2],    (if sunny: 70% sunny, 10% rainy, 20% cloudy)
     [0.3, 0.4, 0.3],    (if rainy: 30% sunny, 40% rainy, 30% cloudy)
     [0.4, 0.2, 0.4]]    (if cloudy: 40% sunny, 20% rainy, 40% cloudy)
```从任意状态开始。经过许多次转移后，状态的分布会收敛到平稳分布 pi，其中 pi * P = pi。这是 P 的特征值为 1 的左特征向量。

对于天气链，平稳分布是 [0.55, 0.18, 0.27] —— 从长远来看，无论起始状态如何，有 55% 的时间是晴天。```mermaid
graph LR
    S["Sunny"] -->|0.7| S
    S -->|0.1| R["Rainy"]
    S -->|0.2| C["Cloudy"]
    R -->|0.3| S
    R -->|0.4| R
    R -->|0.3| C
    C -->|0.4| S
    C -->|0.2| R
    C -->|0.4| C
```**计算平稳分布。** 有两种方法：

1. **幂法**：将任意初始分布反复乘以 P。经过足够多的迭代后，它将收敛。
2. **特征值方法**：找到 P 的特征值为 1 的左特征向量。这等价于找到 P^T 的特征值为 1 的特征向量。

这两种方法都要求链满足收敛条件。

**收敛条件。** 如果马尔可夫链满足以下条件，它将收敛到唯一的平稳分布：
- **不可约**：从任意状态都可以到达其他所有状态。
- **非周期**：链不具有固定的周期性。

在机器学习中遇到的大多数链都满足这两个条件。

**吸收状态。** 如果一旦进入某个状态就再也无法离开（P[i][i] = 1），则该状态为吸收状态。吸收马尔可夫链用于建模具有终止状态的过程——例如，结束的游戏、流失的客户、达到文本结尾标记的标记序列。

**混合时间。** 链需要多少步才能接近平稳分布？严格地说，混合时间是总变分距离从平稳性下降到某个阈值以下所需的步数。快速混合意味着需要的步数较少。P 的谱隙（1 减去第二大特征值）控制混合时间。谱隙越大，混合越快。

### 与语言模型的联系

语言模型中的标记生成近似于马尔可夫过程。给定当前上下文，模型会输出下一个标记的分布。温度控制分布的锐度：```
P(token_i) = exp(logit_i / temperature) / sum(exp(logit_j / temperature))
```- Temperature = 1.0：标准分布
- Temperature < 1.0：更尖锐（更确定性）
- Temperature > 1.0：更平坦（更随机）
- Temperature -> 0：argmax（贪婪）

Top-k 采样将概率截断为前 k 个概率最高的 token。Top-p（核）采样将概率截断为累积概率超过 p 的最小 token 集。两者都修改了马尔可夫转移概率。

### 布朗运动

随机游走的连续时间极限。位置 B(t) 具有三个特性：
1. B(0) = 0
2. B(t) - B(s) 是均值为 0、方差为 t - s 的正态分布（当 t > s 时）
3. 非重叠区间的增量是独立的

布朗运动是连续的但处处不可微的——在每个尺度上它都会抖动。路径在平面上的分形维数为 2。

在离散模拟中，你通过以下方式近似布朗运动：```
B(t + dt) = B(t) + sqrt(dt) * z,    where z ~ N(0, 1)
```sqrt(dt) 的缩放是重要的。它来自于对随机游走应用中心极限定理。

### Langevin 动力学

梯度下降法寻找函数的最小值。Langevin 动力学寻找与 exp(-U(x)/T) 成比例的概率分布，其中 U 是能量函数，T 是温度。```
x_{t+1} = x_t - dt * gradient(U(x_t)) + sqrt(2 * T * dt) * z_t
```有两个力作用于粒子：
1. **梯度力** (-dt * gradient(U))：向低能量区域推动（类似于梯度下降）
2. **随机力** (sqrt(2*T*dt) * z)：向随机方向推动（探索）

在温度 T = 0 时，这是纯粹的梯度下降。在高温时，它几乎是一个随机游走。在适当的温度下，粒子会探索能量景观，并在低能量区域花费更多时间。

**与扩散模型的联系。** 扩散模型的正向过程是：

 /no_think

<>

有两个力作用于粒子：
1. **梯度力** (-dt * gradient(U))：向低能量区域推动（类似于梯度下降）
2. **随机力** (sqrt(2*T*dt) * z)：向随机方向推动（探索）

在温度 T = 0 时，这是纯粹的梯度下降。在高温时，它几乎是一个随机游走。在适当的温度下，粒子会探索能量景观，并在低能量区域花费更多时间。

**与扩散模型的联系。** 扩散模型的正向过程是：

```python
def forward(x, t):
    # add noise
    x = x + sqrt(2*T*dt) * z
    # apply gradient
    x = x - dt * gradient(U)
    return x
``````
x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * noise
```这是一个逐渐将数据与噪声混合的马尔可夫链。经过足够多的步骤后，x_T 就变成了纯高斯噪声。

反向过程——从噪声回到数据——也是一个马尔可夫链，但它的转移概率是由神经网络学习得到的。该网络学习预测每一步添加的噪声，然后将其减去。```mermaid
graph LR
    subgraph "Forward Process (add noise)"
        X0["x_0 (data)"] -->|"+ noise"| X1["x_1"]
        X1 -->|"+ noise"| X2["x_2"]
        X2 -->|"..."| XT["x_T (pure noise)"]
    end
    subgraph "Reverse Process (denoise)"
        XT2["x_T (noise)"] -->|"neural net"| XR2["x_{T-1}"]
        XR2 -->|"neural net"| XR1["x_{T-2}"]
        XR1 -->|"..."| XR0["x_0 (generated data)"]
    end
```### MCMC：马尔可夫链蒙特卡洛方法

有时你需要从一个分布 p(x) 中抽样，这个分布你可以计算（最多到一个常数），但不能直接抽样。贝叶斯后验分布就是经典的例子——你知道似然乘以先验，但归一化常数是难以计算的。

**Metropolis-Hastings** 构造了一个马尔可夫链，其平稳分布为 p(x)：

1. 从某个位置 x 开始
2. 从提议分布 Q(x'|x) 中提出一个新的位置 x'
3. 计算接受率：a = p(x') * Q(x|x') / (p(x) * Q(x'|x))
4. 以概率 min(1, a) 接受 x'。否则留在 x。
5. 重复。

如果 Q 是对称的（例如，Q(x'|x) = Q(x|x') = N(x, sigma^2)），比率简化为 a = p(x') / p(x)。你只需要概率的比率——归一化常数被抵消了。

在温和的条件下，链可以保证收敛到 p(x)。但如果提议的步长太小（随机游走）或太大（高拒绝率），收敛可能很慢。调整提议分布是MCMC的艺术。

**为什么它有效。** 接受率确保了详细平衡：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。详细平衡意味着 p(x) 是链的平稳分布。因此，经过足够多的步骤后，样本将来自 p(x)。

**实际考虑因素：**
- **Burn-in（预热期）**：丢弃前 N 个样本。链需要时间从起点到达平稳分布。
- **Thinning（稀疏化）**：保留每隔 k 个样本以减少自相关。
- **多个链**：从不同的起点运行多个链。如果它们收敛到相同的分布，你就有收敛的证据。
- **接受率**：在 d 维空间中，对于高斯提议分布，最佳接受率约为 23%（Roberts & Rosenthal, 2001）。太高意味着链几乎不动。太低意味着它拒绝所有提议。

### 人工智能中的随机过程

| 过程 | 人工智能应用 |
|------|-------------|
| 随机游走 | 强化学习中的探索，Node2Vec 嵌入 |
| 马尔可夫链 | 文本生成，MCMC 抽样 |
| 布朗运动 | 扩散模型（正向过程） |
| 朗之万动力学 | 基于分数的生成模型，SGLD |
| 马尔可夫决策过程 | 强化学习 |
| Metropolis-Hastings | 贝叶斯推理，后验抽样 |```figure
random-walk-diffusion
```## 构建它

### 步骤 1：随机行走模拟器```python
import numpy as np

def random_walk_1d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    steps = rng.choice([-1, 1], size=n_steps)
    positions = np.concatenate([[0], np.cumsum(steps)])
    return positions


def random_walk_2d(n_steps, seed=None):
    rng = np.random.RandomState(seed)
    directions = rng.choice(4, size=n_steps)
    dx = np.zeros(n_steps)
    dy = np.zeros(n_steps)
    dx[directions == 0] = 1   # right
    dx[directions == 1] = -1  # left
    dy[directions == 2] = 1   # up
    dy[directions == 3] = -1  # down
    x = np.concatenate([[0], np.cumsum(dx)])
    y = np.concatenate([[0], np.cumsum(dy)])
    return x, y
```一维随机游走存储累积和。每一步是 +1 或 -1。经过 n 步后，位置就是这个和。方差随着 n 线性增长，因此标准差与 sqrt(n) 成正比。

### 步骤 2：马尔可夫链```python
class MarkovChain:
    def __init__(self, transition_matrix, state_names=None):
        self.P = np.array(transition_matrix, dtype=float)
        self.n_states = len(self.P)
        self.state_names = state_names or [str(i) for i in range(self.n_states)]

    def step(self, current_state, rng=None):
        if rng is None:
            rng = np.random.RandomState()
        probs = self.P[current_state]
        return rng.choice(self.n_states, p=probs)

    def simulate(self, start_state, n_steps, seed=None):
        rng = np.random.RandomState(seed)
        states = [start_state]
        current = start_state
        for _ in range(n_steps):
            current = self.step(current, rng)
            states.append(current)
        return states

    def stationary_distribution(self):
        eigenvalues, eigenvectors = np.linalg.eig(self.P.T)
        idx = np.argmin(np.abs(eigenvalues - 1.0))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / stationary.sum()
        return np.abs(stationary)
```平稳分布是矩阵 P 的特征值为 1 的左特征向量。我们通过计算 P^T 的特征向量来找到它（转置将左特征向量转换为右特征向量）。

### 步骤 3：朗之万动力学```python
def langevin_dynamics(grad_U, x0, dt, temperature, n_steps, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    trajectory = [x.copy()]
    for _ in range(n_steps):
        noise = rng.randn(*x.shape)
        x = x - dt * grad_U(x) + np.sqrt(2 * temperature * dt) * noise
        trajectory.append(x.copy())
    return np.array(trajectory)
```梯度将 x 推向低能量区域。噪声防止其陷入局部极小值。在平衡状态下，样本的分布与 exp(-U(x)/温度) 成正比。

### 步骤 4：Metropolis-Hastings```python
def metropolis_hastings(target_log_prob, proposal_std, x0, n_samples, seed=None):
    rng = np.random.RandomState(seed)
    x = np.array(x0, dtype=float)
    samples = [x.copy()]
    accepted = 0
    for _ in range(n_samples - 1):
        x_proposed = x + rng.randn(*x.shape) * proposal_std
        log_ratio = target_log_prob(x_proposed) - target_log_prob(x)
        if np.log(rng.rand()) < log_ratio:
            x = x_proposed
            accepted += 1
        samples.append(x.copy())
    acceptance_rate = accepted / (n_samples - 1)
    return np.array(samples), acceptance_rate
```该算法提出一个新的点，检查其是否具有更高的概率（或按比例接受），然后重复这一过程。对于良好的混合效果，接受率应在23-50%之间。

## 使用它

在实际应用中，你通常会使用已有的库来实现这些算法。但理解其原理对于调试和调优非常重要。```python
import numpy as np

rng = np.random.RandomState(42)
walk = np.cumsum(rng.choice([-1, 1], size=10000))
print(f"Final position: {walk[-1]}")
print(f"Expected distance: {np.sqrt(10000):.1f}")
print(f"Actual distance: {abs(walk[-1])}")
```### numpy 用于转移矩阵```python
import numpy as np

P = np.array([[0.7, 0.1, 0.2],
              [0.3, 0.4, 0.3],
              [0.4, 0.2, 0.4]])

distribution = np.array([1.0, 0.0, 0.0])
for _ in range(100):
    distribution = distribution @ P

print(f"Stationary distribution: {np.round(distribution, 4)}")
```重复将初始分布乘以 P。经过足够多次迭代后，无论从哪里开始，它都会收敛到平稳分布。这就是用于寻找主左特征向量的幂法。

### 与实际框架的联系

- **PyTorch 扩散：** Hugging Face 的 `DDPMScheduler` `diffusers` 实现了前向和反向马尔可夫链
- **NumPyro / PyMC：** 使用 MCMC（NUTS 采样器，改进了 Metropolis-Hastings）进行贝叶斯推断
- **Gymnasium（强化学习）：** 环境的 step 函数定义了一个马尔可夫决策过程

### 验证马尔可夫链的收敛性```python
import numpy as np

P = np.array([[0.9, 0.1], [0.3, 0.7]])

eigenvalues = np.linalg.eigvals(P)
spectral_gap = 1 - sorted(np.abs(eigenvalues))[-2]
print(f"Eigenvalues: {eigenvalues}")
print(f"Spectral gap: {spectral_gap:.4f}")
print(f"Approximate mixing time: {1/spectral_gap:.1f} steps")
```谱隙告诉你链忘记其初始状态的速度。谱隙为 0.2 表示大约需要 5 步才能达到混合。谱隙为 0.01 表示大约需要 100 步。在运行长时间模拟之前，始终要检查这一点 —— 混合缓慢的链会浪费计算资源。

## 发布它

本课将产出：
- `outputs/prompt-stochastic-process-advisor.md` -- 一个有助于识别哪种随机过程框架适用于给定问题的提示

## 联系

| 概念 | 出现的地方 |
|---------|------------------|
| 随机游走 | Node2Vec 图嵌入，强化学习中的探索 |
| 马尔可夫链 | 大语言模型中的 Token 生成，MCMC 抽样 |
| 布朗运动 | DDPM 中的前向扩散过程，基于 SDE 的模型 |
| 朗之万动力学 | 基于分数的生成模型，随机梯度朗之万动力学 (SGLD) |
| 稳态分布 | MCMC 收敛目标，PageRank |
| Metropolis-Hastings | 贝叶斯后验抽样，模拟退火 |
| 温度 | 大语言模型抽样，强化学习中的玻尔兹曼探索，模拟退火 |
| 混合时间 | MCMC 的收敛速度，谱隙分析 |
| 吸收态 | 序列结束 Token，在强化学习中的终止状态 |
| 详细平衡 | MCMC 抽样器的正确性保证 |

扩散模型需要特别关注。DDPM（Ho 等，2020）定义了一个前向马尔可夫链：```
q(x_t | x_{t-1}) = N(x_t; sqrt(1-beta_t) * x_{t-1}, beta_t * I)
```其中 beta_t 是一个噪声计划。经过 T 步之后，x_T 近似服从 N(0, I) 分布。反向过程由一个神经网络参数化，该网络预测噪声：```
p_theta(x_{t-1} | x_t) = N(x_{t-1}; mu_theta(x_t, t), sigma_t^2 * I)
```生成过程的每一步都是一个学习到的马尔可夫链中的步骤。理解马尔可夫链意味着理解扩散模型是如何以及为何生成数据的。

SGLD（随机梯度朗之万动力学）将小批量梯度下降与朗之万噪声相结合。你不是计算完整的梯度，而是使用一个随机估计，并添加校准的噪声。随着学习率的衰减，SGLD从优化转换为采样——你可以免费获得近似的贝叶斯后验样本。这是从神经网络中获取不确定性估计的最简单方法之一。

贯穿所有这些联系的关键洞察：随机过程不仅仅是理论工具。它们是现代人工智能系统内部的计算机制。当你调整一个大语言模型的温度时，你实际上是在调整一个马尔可夫链。当你训练一个扩散模型时，你实际上是在学习逆转一个类似于布朗运动的过程。当你运行贝叶斯推断时，你实际上是在构建一个收敛到后验的链。

## 练习

1. **模拟1000个随机游走，每个有10000步。** 绘制最终位置的分布。验证它近似于均值为0、标准差为sqrt(10000)=100的高斯分布。

2. **使用马尔可夫链构建一个文本生成器。** 在一个小语料库上进行训练：对于每个词，统计到下一个词的转移次数。构建转移矩阵。通过从链中采样生成新的句子。

3. **使用Metropolis-Hastings实现模拟退火。** 从高温（几乎接受所有内容）开始，逐渐降温（只接受改进）。用它来寻找一个具有多个局部最小值的函数的最小值。

4. **比较不同温度下的朗之万动力学。** 从一个双井势能U(x) = (x² - 1)²中采样。在低温下，样本集中在其中一个井中。在高温下，它们分布在两个井中。找到链在井之间混合的临界温度。

5. **实现前向扩散过程。** 从一个1D信号（例如正弦波）开始，通过100步的线性噪声计划逐步添加噪声。展示信号如何退化为纯噪声。然后实现一个简单的去噪器，逆转该过程（即使是一个简单的仅减去估计噪声的去噪器）。

## 关键术语

| 术语 | 人们常说 | 它实际上意味着 |
|------|----------------|------------------|
| 随机游走 | “硬币翻转运动” | 每一步位置随机变化的过程 |
| 马尔可夫性质 | “无记忆” | 未来仅依赖于当前状态，不依赖于历史 |
| 转移矩阵 | “概率表” | P[i][j] = 从状态i转移到状态j的概率 |
| 静态分布 | “长期平均” | 满足pi*P = pi的分布pi——链的平衡状态 |
| 布朗运动 | “随机抖动” | 随机游走的连续时间极限，B(t) ~ N(0, t) |
| 朗之万动力学 | “带噪声的梯度下降” | 将确定性梯度与随机扰动相结合的更新规则 |
| MCMC | “走向目标” | 构建一个静态分布是你想要的马尔可夫链 |
| Metropolis-Hastings | “提议并接受/拒绝” | 使用接受率确保收敛的MCMC算法 |
| 温度 | “随机性旋钮” | 控制探索与利用之间权衡的参数 |
| 扩散过程 | “输入噪声，输出噪声” | 前向：逐步添加噪声。反向：逐步去除噪声。生成数据。 |

## 进一步阅读

- **Ho, Jain, Abbeel (2020)** -- "Denoising Diffusion Probabilistic Models." 启动扩散模型革命的DDPM论文。清晰推导前向和反向马尔可夫链。
- **Song & Ermon (2019)** -- "Generative Modeling by Estimating Gradients of the Data Distribution." 使用朗之万动力学进行采样的基于分数的方法。
- **Roberts & Rosenthal (2004)** -- "General state space Markov chains and MCMC algorithms." MCMC工作原理的理论背景。
- **Norris (1997)** -- "Markov Chains." 标准教科书。涵盖收敛性、静态分布和首次到达时间。
- **Welling & Teh (2011)** -- "Bayesian Learning via Stochastic Gradient Langevin Dynamics." 将SGD与朗之万动力学结合，用于可扩展的贝叶斯推断。
