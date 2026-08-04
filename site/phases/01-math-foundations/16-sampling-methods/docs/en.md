# 采样方法与 Monte Carlo 算法

> 在复杂分布中抽取代表性样本。掌握拒绝采样、重要性采样、MCMC 与重参数化技巧。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 06 (概率论与概率分布)
**Time:** ~50 分钟

## 学习目标

- 仅使用均匀随机数从头实现逆 CDF、拒绝采样和重要性采样
- 为语言模型的 token 生成构建温度采样、top-k 采样和 top-p (nucleus) 采样
- 解释重参数化技巧及其为何使 VAE 中通过采样进行反向传播成为可能
- 运行 Metropolis-Hastings MCMC 从非归一化的目标分布中进行采样

## 问题

一个语言模型处理完你的提示后，会生成一个包含 50,000 个 logit 的向量。每个词汇表中的 token 都对应一个。现在它必须选择一个。如何选择？

如果它总是选择概率最高的 token，那么每个响应都是一样的。确定性。乏味。如果它完全随机选择，输出将是无意义的。答案介于这两个极端之间，而这个“中间点”由采样控制。

采样不仅限于文本生成。强化学习通过采样轨迹来估计策略梯度。VAEs 通过从学习到的分布中采样并反向传播通过随机性来学习潜在表示。扩散模型通过采样噪声并逐步去噪来生成图像。Monte Carlo 方法用于估计没有闭式解的积分。MCMC 算法探索无法枚举的高维后验分布。

每个生成式 AI 系统都是一个采样系统。采样策略决定了输出的质量、多样性和可控性。本节课从头构建每种主要的采样方法，从均匀随机数开始，直到现代大语言模型和生成模型所依赖的技术。

## 概念

### 为什么采样很重要

采样在人工智能和机器学习中扮演着四个基本角色：

**生成。** 语言模型、扩散模型和 GANs 都通过采样生成输出。采样算法直接控制创造力、连贯性和多样性。温度采样、top-k 采样和 nucleus 采样是工程师每天调整的旋钮。

**训练。** 随机梯度下降采样小批量。Dropout 采样神经元以关闭。数据增强采样随机变换。重要性采样在强化学习（PPO、TRPO）中重新加权样本以减少梯度方差。

**估计。** 机器学习中许多量没有闭式解。数据分布上的期望损失、基于能量模型的分区函数、贝叶斯推断中的证据。Monte Carlo 估计通过样本的平均来近似所有这些。

**探索。** MCMC 算法在贝叶斯推断中探索后验分布。进化策略采样参数扰动。汤普森采样在 bandits 中平衡探索和利用。

核心挑战：你只能直接从简单分布（均匀、正态）中采样。对于其他所有情况，你需要一种方法将简单样本转换为来自目标分布的样本。

### 均匀随机采样

每个采样方法都从这里开始。一个均匀随机数生成器生成 [0, 1) 范围内的值，其中每个长度相等的子区间具有相等的概率。```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```要从一个包含 n 个离散项目的集合中均匀采样，生成 U 并返回 floor(n * U)。要从连续区间 [a, b] 中采样，计算 a + (b - a) * U。

关键见解：一个单一的均匀随机数包含恰好足够产生来自任何分布的一个样本的随机性。关键是找到正确的变换。

### 反向累积分布函数方法（反向变换采样）

累积分布函数（CDF）将值映射到概率：```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```逆累积分布函数（inverse CDF）将概率映射回对应的值。如果 U ~ Uniform(0, 1)，那么 X = F_inverse(U) 服从目标分布。```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```**指数分布示例：**```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```当你可以将 F_inverse 以闭式形式写出时，这种方法非常有效。对于正态分布，没有闭式逆 CDF，因此我们使用其他方法（如 Box-Muller 方法或数值近似）。

**离散版本：** 对于离散分布，构建 CDF 为累积和，生成 U，然后找到累积和首次超过 U 的索引。这就是 Lesson 06 中 `sample_categorical` 的工作原理。

### 拒绝抽样

当你无法对 CDF 进行求逆，但可以评估目标 PDF（最多相差一个常数）时，拒绝抽样方法是有效的。```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```约束 M 越紧，接受率越高。在低维空间（1-3 维）中，拒绝采样效果很好。在高维空间中，接受率会呈指数级下降，因为大部分提议的体积都会被拒绝。这是拒绝采样所面临的维度诅咒问题。

**示例：从截断正态分布中采样。** 在截断范围内使用均匀提议分布。包络 M 是该范围内正态 PDF 的最大值。

**示例：从半圆中采样。** 在包围矩形内均匀提议。如果点落在半圆内则接受。这就是蒙特卡洛计算 pi 的方法：接受率等于面积比 pi/4。

### 重要性采样

有时候你并不需要从目标分布 p(x) 中获取样本。你只需要在 p(x) 下估计一个期望，并且你拥有来自另一个分布 q(x) 的样本。```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```这在强化学习中是至关重要的。在PPO（近端策略优化）中，你是在旧策略pi_old下收集轨迹，但想要优化新策略pi_new。重要性权重是pi_new(a|s) / pi_old(a|s)。PPO通过截断这些权重来防止新策略偏离旧策略太远。

重要性抽样估计量的方差取决于q与p的相似程度。如果q与p相差很大，少数样本会获得非常大的权重，并主导估计结果。自归一化重要性抽样通过除以权重的总和来减少这个问题：```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```### 蒙特卡洛估计

蒙特卡洛估计通过平均随机样本来近似积分。大数定律保证了收敛性。```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```错误率与维度无关。这就是为什么蒙特卡洛方法在高维空间中占主导地位，因为在高维空间中基于网格的积分是不可能的。

**估计圆周率：**```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```**估计期望值：**```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```### 马尔可夫链蒙特卡洛（MCMC）：Metropolis-Hastings 算法

MCMC 构造一个马尔可夫链，其平稳分布为目标分布 p(x)。经过足够多的步骤后，从链中采样的样本（近似地）就是从 p(x) 中采样的样本。```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```对于对称性提议（q(x'|x) = q(x|x')），该比例简化为 p(x')/p(x)。这是原始的 Metropolis 算法。

**为什么有效。** 接受规则确保了详细平衡：处于 x 并移动到 x' 的概率等于处于 x' 并移动到 x 的概率。详细平衡意味着 p(x) 是链的平稳分布。

**实际考虑因素：**
- 预热期（Burn-in）：在链达到平衡之前丢弃早期样本
- 薄化（Thinning）：保留每 k 个样本以减少自相关
- 提议尺度：太小会导致链移动缓慢（高接受率，探索速度慢）；太大则大多数提议会被拒绝（低接受率，停滞不动）
- 高维空间中，高斯提议的最优接受率约为 0.234

### Gibbs 抽样

Gibbs 抽样是用于多变量分布的 MCMC 的一种特例。它不是一次性在所有维度上提出移动，而是从条件分布中依次更新每个变量。```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```吉布斯采样要求你可以从每个条件分布 p(x_i | x_{-i}) 中进行采样。这对于许多模型来说是直接的：
- 贝叶斯网络：条件分布由图结构决定
- 高斯混合模型：条件分布是高斯分布
- 伊辛模型：每个自旋的条件分布只依赖于其邻居

接受率总是为 1（每个提议都被接受），因为从精确的条件分布中采样会自动满足细致平衡。

**限制。** 当变量高度相关时，吉布斯采样混合速度较慢，因为一次更新一个变量无法在分布中进行大的对角移动。

### 温度采样（用于大语言模型）

语言模型为词汇表中的每个标记输出对数几率 z_1, ..., z_V。Softmax 将这些转换为概率。温度在 Softmax 之前对对数几率进行重新缩放：```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```**为什么有效。** 将logits除以T < 1会放大logits之间的差异。如果z₁ = 2，z₂ = 1，除以T = 0.5后，z₁/T = 4，z₂/T = 2，使得差距更大。经过softmax后，logits最高的标记将获得更大的概率份额。

**实际应用：**
- T = 0.0: 贪婪解码，最适合事实性问答
- T = 0.3-0.7: 稍微有创意，适合代码生成
- T = 0.7-1.0: 平衡，适合一般对话
- T = 1.0-1.5: 创意写作，头脑风暴
- T > 1.5: 越来越随机，很少有用

温度值不会改变哪些标记是可能的。它改变的是分配给每个标记的概率质量。

### Top-k 采样

Top-k 采样将候选集限制为概率最高的k个标记，然后对这个受限集合进行重新归一化，并从中进行采样。```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```Top-k 防止模型选择词汇分布长尾中出现的极不可能的标记（拼写错误、无意义内容）。问题在于：k 是固定的，不随上下文变化。当模型很自信（某个标记的概率为 95%）时，k = 40 仍然允许有 39 个替代选项。当模型不确定（概率分布在 1000 个标记上）时，k = 40 会截断可能的选项。

### Top-p（核）采样

Top-p 采样会动态调整候选集合的大小。它不是保留固定数量的标记，而是保留累积概率超过 p 的最小标记集合。```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```当模型非常自信时，核采样（nucleus sampling）只保留少量的 token（可能为 2-3 个）。当模型不确定时，它会保留大量的 token（可能为 200 个）。这种自适应行为就是为什么核采样通常比 top-k 生成更高质量文本的原因。

**常见组合：**
- 温度 0.7 + top-p 0.9：良好的通用设置
- 温度 0.0（贪婪）：最适合确定性任务
- 温度 1.0 + top-k 50：Fan 等人（2018）原始论文设置

top-k 和 top-p 可以结合使用。先应用 top-k，然后在剩余的集合上应用 top-p。

### 重参数化技巧（用于 VAEs）

变分自编码器（VAEs）通过将输入编码为潜在空间中的分布，从该分布中采样，然后将样本解码回原始空间来学习。问题在于：你无法通过采样操作进行反向传播。```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```重参数化技巧将随机性与参数分离：```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```这之所以有效，是因为 N(mu, sigma^2) 的分布与 mu + sigma * N(0, 1) 的分布相同。关键的洞察是：将随机性转移到一个无参数的来源（epsilon），然后将样本表示为参数的可微分变换。

**在 VAE 训练循环中：**
1. 编码器为每个输入输出 mu 和 log(sigma^2)
2. 采样 epsilon ~ N(0, 1)
3. 计算 z = mu + sigma * epsilon
4. 解码 z 以重建输入
5. 通过步骤 4、3、2、1 反向传播（因为步骤 3 是可微分的）

没有重参数化技巧，VAEs 无法使用标准的反向传播进行训练。这个单一的洞察使 VAEs 变得实用。

### Gumbel-Softmax（可微分的分类采样）

重参数化技巧适用于连续分布（高斯分布）。对于离散分类分布，我们需要不同的方法。Gumbel-Softmax 提供了对分类采样的可微分近似。

**Gumbel-Max 技巧（不可微分）：**```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```**Gumbel-Softmax（可微分近似）：**```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```Gumbel-Softmax 产生离散样本的连续松弛。输出是一个概率向量（软独热）而不是硬独热。梯度通过 softmax 流动。在训练的前向传递过程中，你可以使用“直通”估计器：前向传递使用硬 argmax，反向传递使用软 Gumbel-Softmax 梯度。

**应用：**
- VAEs 中的离散潜在变量
- 神经网络架构搜索（选择离散操作）
- 硬注意力机制
- 离散动作的强化学习

### 分层抽样

标准的蒙特卡洛抽样可能会偶然在样本空间中留下空白。分层抽样通过将空间划分为若干层并从每一层进行抽样，强制实现均匀覆盖。```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```分层抽样总是具有比标准蒙特卡洛方法更低或相等的方差：```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```**应用：**
- 数值积分（准蒙特卡洛方法）
- 训练数据划分（确保每个折叠中的类别平衡）
- 带分层的重采样（结合两种技术）
- NeRF（神经辐射场）在相机射线上使用分层采样

### 与扩散模型的联系

扩散模型通过采样过程生成图像。前向过程在T步中逐步向图像添加高斯噪声，直到图像变为纯噪声。反向过程学习去噪，逐步恢复原始图像。```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```本课内容与方法的联系：
- 每个去噪步骤都使用重参数化技巧（采样噪声，应用确定性变换）
- 噪声计划 {alpha_t} 控制一种温度退火方式
- 训练使用蒙特卡洛估计来近似 ELBO（证据下界）
- 扩散模型中的祖先采样是一个马尔可夫链（每一步只依赖于当前状态）

整个图像生成过程是迭代采样：从噪声开始，每一步都根据学习到的去噪模型，采样一个略微更少噪声的版本。```figure
monte-carlo-pi
```## 构建它

### 步骤 1：均匀分布和逆累积分布函数采样```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```生成 10,000 个指数分布样本，并验证其均值是否为 1/lambda。

### 步骤 2：拒绝采样```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```使用拒绝采样方法从截断正态分布中进行抽样。通过直方图验证样本的形状。

### 步骤 3：重要性采样```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```使用均匀提议分布估计正态分布下 E[X^2]。与已知答案（mu^2 + sigma^2）进行比较。

### 步骤 4：蒙特卡洛方法估计 pi```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```### 步骤 5：Metropolis-Hastings MCMC```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```来自双峰分布（两个高斯分布的混合）的样本。可视化链的轨迹。

### 步骤 6：Gibbs 采样```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```### 步骤 7：温度采样```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```展示温度如何改变一组 token logits 的输出分布。

### 第 8 步：Top-k 和 top-p 采样```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```### 步骤 9：重参数化技巧```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```证明梯度可以通过重参数化的样本流动，但不能通过直接采样流动。

### 第 10 步：Gumbel-Softmax```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```展示温度降低如何使输出趋近于独热向量。

完整实现及所有可视化内容在 `code/sampling.py` 中。

## 使用方法

使用 NumPy 和 SciPy 的生产版本：```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```对于大规模的 MCMC，使用专用库：
- PyMC：使用 NUTS（自适应 HMC）进行完整的贝叶斯建模
- emcee：集成 MCMC 采样器
- NumPyro/JAX：加速 GPU 的 MCMC

你从零开始构建了这些。现在你知道库调用在做什么。

## 练习

1. 为柯西分布实现逆 CDF 采样。CDF 为 F(x) = 0.5 + arctan(x)/pi。生成 10,000 个样本，并将直方图与真实 PDF 进行对比。注意重尾（远离中心的极端值）。

2. 使用均匀分布 Uniform(0, 1) 提议，用拒绝采样生成 Beta(2, 5) 分布的样本。将接受的样本与真实 Beta PDF 进行对比。理论上的接受率是多少？

3. 使用 1,000、10,000 和 100,000 个样本，用蒙特卡洛方法估计 sin(x) 从 0 到 pi 的积分。比较每个层次的误差。验证误差是否按 O(1/sqrt(N)) 缩放。

4. 实现 Metropolis-Hastings 算法，从二维分布 p(x, y) 采样，该分布与 exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2) 成正比。绘制样本和链轨迹。尝试不同的提议标准差。

5. 构建一个完整的文本生成演示：给定一个包含 10 个词的词库及其 logits，使用 (a) 贪婪，(b) temperature=0.7，(c) top-k=3，(d) top-p=0.9 生成 20 个 token 的序列。比较 5 次运行中输出的多样性。

## 关键术语

| 术语 | 人们怎么说 | 实际含义 |
|------|----------------|-----------|
| 采样 | “抽取随机值” | 根据概率分布生成值。所有生成式 AI 的机制 |
| 均匀分布 | “所有值都同样可能” | [a, b] 中的每个值都有相等的概率密度 1/(b-a)。所有采样方法的起点 |
| 逆 CDF | “概率变换” | F_inverse(U) 将一个均匀样本转换为任何具有已知 CDF 的分布的样本。精确且高效 |
| 拒绝采样 | “提议并接受/拒绝” | 从简单提议中生成样本，接受概率与目标/提议比率成比例。精确但浪费样本 |
| 重要性采样 | “重新加权样本” | 使用来自 q(x) 的样本并按 p(x)/q(x) 加权来估计 p(x) 下的期望。强化学习中 PPO 的核心 |
| 蒙特卡洛 | “平均随机样本” | 将积分近似为样本平均值。误差为 O(1/sqrt(N))，无论维度如何 |
| MCMC | “收敛的随机游走” | 构造一个平稳分布为目标分布的马尔可夫链。Metropolis-Hastings 是基础算法 |
| Metropolis-Hastings | “接受上坡，有时下坡” | 提议移动，根据密度比接受。详细平衡确保收敛到目标分布 |
| Gibbs 采样 | “一次一个变量” | 从每个变量的条件分布中更新变量，保持其他变量不变。100% 接受率 |
| 温度 | “置信度旋钮” | 在 softmax 之前将 logits 除以 T。T<1 使分布更尖锐（更自信），T>1 使分布更平坦（更多样） |
| Top-k 采样 | “保留最好的 k 个” | 将除了最高的 k 个概率 token 外的所有概率设为零，重新归一化后进行采样。候选集大小固定 |
| Nucleus 采样（top-p） | “保留可能的 token” | 保留累积概率超过 p 的最小 token 集。候选集大小自适应 |
| 重参数化技巧 | “将随机性移出” | 写为 z = mu + sigma * epsilon，其中 epsilon ~ N(0,1)。使采样可微。变分自编码器训练的关键 |
| Gumbel-Softmax | “软分类采样” | 使用 Gumbel 噪声 + softmax 和温度的可微分类采样近似 |
| 分层采样 | “强制覆盖” | 将样本空间划分为层，从每层中采样。方差总是低于朴素蒙特卡洛 |
| Burn-in | “预热期” | 在链达到平稳分布之前丢弃的初始 MCMC 样本 |
| 详细平衡 | “可逆性条件” | p(x) * T(x->y) = p(y) * T(y->x)。马尔可夫链平稳分布为 p 的充分条件 |
| 扩散采样 | “迭代去噪” | 从噪声开始，应用学习到的去噪步骤生成数据。每一步是一个条件采样操作 |

## 进一步阅读

- [Holbrook (2023): Metropolis-Hastings 算法](https://arxiv.org/abs/2304.07010) - MCMC 基础的详细教程
- [Jang, Gu, Poole (2017): Gumbel-Softmax 的分类重参数化](https://arxiv.org/abs/1611.01144) - 原始 Gumbel-Softmax 论文
- [Holtzman 等 (2020): 神经文本退化奇特案例](https://arxiv.org/abs/1904.09751) - nucleus (top-p) 采样论文
- [Kingma & Welling (2014): 自编码变分贝叶斯](https://arxiv.org/abs/1312.6114) - 引入重参数化技巧的 VAE 论文
- [Ho, Jain, Abbeel (2020): 去噪扩散概率模型](https://arxiv.org/abs/2006.11239) - DDPM 将采样连接到图像生成
