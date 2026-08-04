# 数值稳定性与浮点精度

> 防止 NaN 崩溃与精度损失。掌握 Log-Sum-Exp 稳定技巧、灾难性抵消与 Loss Scaling。

**Type:** 学习
**Language:** Python
**Prerequisites:** Phase 1, Lesson 04 (机器学习中的微积分)
**Time:** ~40 分钟

## 学习目标

- 使用最大值减法技巧实现数值稳定的 softmax 和 log-sum-exp
- 识别浮点计算中的溢出、下溢和灾难性抵消
- 使用中心有限差分验证分析梯度与数值梯度
- 解释为什么在训练中更倾向于使用 bfloat16 而不是 float16，以及损失缩放如何防止梯度下溢

## 问题

你的模型训练了三小时，然后损失变为 NaN。你添加了一个打印语句。在第 9,000 步时，logits 是正常的。在第 9,001 步时，它们是 `inf`。到第 9,002 步时，所有梯度都变为 `nan`，训练已经失败。

或者：你的模型训练完成，但准确率比论文中声称的低 2%。你检查了所有内容。架构匹配。超参数匹配。数据匹配。问题是论文中使用的是 float32，而你使用的是 float16，但没有进行适当的缩放。累积的舍入误差悄悄地吃掉了你的准确率。

或者：你从头开始实现交叉熵损失。它在小 logits 上工作正常。当 logits 超过 100 时，它返回 `inf`。因为 `exp(100)` 大于 float32 能表示的范围，softmax 溢出了。每个 ML 框架都用两行技巧处理这个问题。你不知道这个技巧的存在。

数值稳定性不是一个理论问题。它决定了训练运行成功与静默失败之间的差别。你将调试的每个严重 ML 错误最终都会归结为浮点问题。

## 概念

### IEEE 754：计算机如何存储实数

计算机根据 IEEE 754 标准将实数存储为浮点数。一个浮点数有三个部分：符号位、指数和尾数（有效数字）。```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```尾数决定精度（有效数字的位数）。指数决定范围（数字可以有多大或有多小）。```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```float32 给你大约 7 位十进制数字的精度。这意味着它可以区分 1.0000001 和 1.0000002，但无法区分 1.00000001 和 1.00000002。在 7 位之后，所有数字都会变成四舍五入的噪声。

float16 给你大约 3 位数字。它能表示的最大数字是 65,504。对于机器学习来说，这令人不安地小，因为在机器学习中，logits、梯度和激活值经常超过这个数值。

bfloat16 是 Google 对 float16 的范围问题提出的解决方案。它和 float32 一样有 8 位指数（相同的范围，最高到 3.4e38），但只有 7 位尾数（精度比 float16 低）。在训练神经网络时，范围比精度更重要，因此 bfloat16 通常更胜一筹。

### 为什么 0.1 + 0.2 != 0.3

数字 0.1 无法在二进制浮点数中精确表示。在二进制中，它是一个无限循环小数：

```markdown
``````
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```Float32 将其截断为 23 位尾数。存储的值约为 0.100000001490116。同样，0.2 被存储为约 0.200000002980232。它们的和为 0.300000004470348，而不是 0.3。```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```这在机器学习中很重要，因为：

1. 类似 `if loss < threshold` 的损失比较可能会给出错误的答案
2. 累加许多小值（在数千步的梯度更新中）会导致与真实总和的偏差
3. 如果使用 `==` 比较浮点数，校验和和可重复性测试会失败

解决方法：永远不要用 `==` 来比较浮点数。使用 `abs(a - b) < epsilon` 或者 `math.isclose()`。

### 灾难性抵消及其规避

当你用两个几乎相等的浮点数相减时，有效数字会被抵消，剩下的只是被提升到高位的舍入噪声。```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```这来自于一次减法操作，产生了19%的相对误差。在机器学习中，这发生在以下情况时：

- 计算具有大均值的数据的方差：`E[x^2] - E[x]^2` 当E[x]很大时
- 减去几乎相等的对数概率
- 使用过小的epsilon计算有限差分梯度

解决方法：重新排列公式，避免减去大而几乎相等的数。对于方差，使用Welford算法或先对数据进行中心化处理。对于对数概率，始终在对数空间中进行计算。

### 溢出和下溢

溢出发生在结果太大而无法表示时。下溢发生在结果太小（小于可表示的最小正数）时。```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
````exp()` 函数是 ML 中溢出的主要来源：```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
````log()` 函数处理相反的方向：

 /no_think

<>

`log()` 函数处理相反的方向：```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```在机器学习中，`exp()` 出现在 softmax、sigmoid 和概率计算中。`log()` 出现在交叉熵、对数似然和 KL 散度中。组合 `log(exp(x))` 在没有正确技巧的情况下是一个雷区。

### Log-Sum-Exp 极值稳定算法

直接计算 `log(sum(exp(x_i)))` 在数值上是危险的。如果任何 `x_i` 很大，`exp(x_i)` 将溢出。如果所有 `x_i` 都非常小，每个 `exp(x_i)` 都会下溢到零，而 `log(0)` 将是 `-inf`。

技巧：在指数运算之前减去最大值。```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```为什么这有效：减去 `max(x)` 之后，最大的指数是 `exp(0) = 1`。不可能发生溢出。求和中的至少一项是 1，因此总和至少是 1，且 `log(1) = 0`。不可能发生下溢到 `-inf`。

证明：```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```设置 `c = max(x)` 并消除溢出。

这个技巧在 ML 中随处可见：
- Softmax 归一化
- 交叉熵损失计算
- 序列模型中的对数概率求和
- 高斯混合
- 变分推断

### 为什么 Softmax 需要最大值减法技巧

Softmax 将 logits 转换为概率：```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```没有这个技巧，logits [100, 101, 102] 会导致溢出：```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```使用这个技巧，减去 max(x) = 102：```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```概率是相同的。计算是安全的。这不是一种优化。这是正确性的要求。

### NaN 和 Inf：检测与预防

`nan`（非数字）和 `inf`（无穷大）在计算中会像病毒一样传播。梯度更新中出现一个 `nan` 会使权重变为 `nan`，这又会使所有后续的输出变为 `nan`。训练在一步之内就失效了。

`inf` 出现的方式：
- 大正数的 `exp()`
- 除以零：`1.0 / 0.0`
- 累加过程中的 `float32` 溢出

`nan` 出现的方式：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 负数的 `sqrt()`
- 负数的 `log()`
- 任何包含现有 `nan` 的算术运算

检测：```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```预防策略：

1. 将输入限制在 `exp()`：`exp(clamp(x, -80, 80))`
2. 向分母中添加 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 中添加 epsilon：`log(x + 1e-8)`
4. 使用稳定实现（log-sum-exp、稳定 softmax）
5. 梯度裁剪以防止权重爆炸
6. 在调试过程中每次前向传播后检查 `nan`/`inf`

### 数值梯度检查

解析梯度（来自反向传播）可能会有错误。数值梯度检查通过使用有限差分来计算梯度，以验证它们。

中心差分公式：```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```这是 O(h²) 精度，比仅 O(h) 的前向差分 `(f(x+h) - f(x)) / h` 好得多。

选择 h：太大，近似会出错。太小，灾难性抵消会破坏答案。`h = 1e-5` 到 `1e-7` 是典型的。

检查：计算解析梯度和数值梯度之间的相对差异。```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```经验法则：
- relative_error < 1e-7: 完美，梯度正确
- relative_error < 1e-5: 可接受，可能正确
- relative_error > 1e-3: 有问题
- relative_error > 1: 梯度完全错误

实现新的层或损失函数时，始终要检查梯度。PyTorch 提供了 `torch.autograd.gradcheck()` 来实现这一点。

### 混合精度训练

现代 GPU 有专门的硬件（Tensor Cores），可以将 float16 矩阵乘法运算速度提高 2-8 倍，比 float32 快。混合精度训练利用了这一点：```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。float16 会将小于约 6e-8 的数值下溢为零。你的模型会停止学习，因为所有的梯度更新都为零。

解决方法是使用损失缩放：

 /no_think

<>

纯 float16 训练的问题：梯度通常非常小（1e-8 或更小）。float16 会将小于约 6e-8 的数值下溢为零。你的模型会停止学习，因为所有的梯度更新都为零。

解决方法是使用损失缩放：```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```动态损失缩放会自动调整缩放因子。起始值设为一个较大的数值（65536）。如果梯度溢出到 `inf`，则将其减半。如果连续 N 步没有溢出，则将其加倍。

### bfloat16 与 float16：为什么 bfloat16 更适合训练```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```float16 的精度更高（尾数位 10 位 vs 7 位），但范围有限（最大值约 65,504）。bfloat16 精度较低，但其范围与 float32 相同（最大值约 3.4e38）。

对于训练神经网络：

- 激活值和 logits 在训练过程中经常超过 65,504。float16 会溢出；bfloat16 可以处理这种情况。
- 使用 float16 需要损失缩放，但通常使用 bfloat16 时不需要，因为其范围覆盖了梯度幅度的整个范围。
- bfloat16 是 float32 的简单截断：丢弃尾数的最低 16 位。在指数部分，转换是简单且无损的。

float16 更适合用于推理，因为此时值是有限的，精度更为重要。bfloat16 更适合用于训练，因为此时范围更为重要。这就是为什么 TPU 和现代 NVIDIA GPU（如 A100、H100）都原生支持 bfloat16。

### 梯度裁剪

爆炸梯度发生在梯度通过许多层时呈指数级增长（常见于 RNN、深度网络和 Transformer）。一个大的梯度可能在一步中破坏所有权重。

两种类型的裁剪：

**按值裁剪：** 独立地对每个梯度元素进行限制。```
grad = clamp(grad, -max_val, max_val)
```简单但可以改变梯度向量的方向。

**按范数裁剪：** 缩放整个梯度向量，使其范数不超过一个阈值。```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```保留梯度的方向。这是 `torch.nn.utils.clip_grad_norm_()` 所做的事情。它是标准选择。

典型值：对于变压器使用 `max_norm=1.0`，对于强化学习使用 `max_norm=0.5`，对于更简单的网络使用 `max_norm=5.0`。

梯度裁剪不是一个技巧。它是一种安全机制。没有它，一个异常的批次可能会产生足够大的梯度，从而毁掉数周的训练。

### 归一化层作为数值稳定器

批量归一化、层归一化和 RMS 归一化通常被呈现为有助于训练收敛的正则化器。它们同时也是数值稳定器。

没有归一化，激活值可能会通过各层呈指数级增长或缩小：

```python
# 示例代码
def example_function(x):
    return x * 2
``````
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```归一化在每一层重新定位并重新缩放激活值：```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
````epsilon`（通常为1e-5）可在所有激活值相同时防止除以零。学习到的参数 `gamma` 和 `beta` 使网络能够恢复其所需的任何尺度。

这使数值在整个网络中保持在安全范围内，防止前向传播中的溢出和反向传播中的梯度爆炸。

### 常见机器学习数值错误

**错误：几个周期后损失变为 NaN。**
原因：logits 增长得太大，softmax 溢出。或者学习率过高导致权重发散。
解决方法：使用稳定的 softmax（最大值减法），降低学习率，添加梯度裁剪。

**错误：损失卡在 log(num_classes)。**
原因：模型输出接近均匀的概率。通常意味着梯度消失或模型根本没有学习。
解决方法：检查数据标签是否正确，验证损失函数，检查是否有死亡的 ReLU。

**错误：验证准确率比预期低 1-3%。**
原因：混合精度没有进行适当的损失缩放。梯度下溢会静默地将小的更新置零。
解决方法：启用动态损失缩放，或切换为 bfloat16。

**错误：某些层的梯度范数为 0.0。**
原因：死亡的 ReLU 神经元（所有输入为负数），或 float16 下溢。
解决方法：使用 LeakyReLU 或 GELU，使用梯度缩放，检查权重初始化。

**错误：模型在一个 GPU 上运行正常，但在另一个 GPU 上给出不同的结果。**
原因：非确定性的浮点累加顺序。GPU 并行归约在不同硬件上以不同顺序求和，而浮点加法不满足结合律。
解决方法：接受小的差异（1e-6），或设置 `torch.use_deterministic_algorithms(True)` 并接受速度上的惩罚。

**错误：在损失计算中 `exp()` 返回 `inf`。**
原因：没有使用最大值减法技巧，直接将原始 logits 传递给 `exp()`。
解决方法：使用 `torch.nn.functional.log_softmax()`，它内部实现了 log-sum-exp。

**错误：从 float32 切换到 float16 后训练发散。**
原因：float16 无法表示小于 6e-8 的梯度幅度或大于 65,504 的激活值。
解决方法：使用带有损失缩放的混合精度（AMP），或改用 bfloat16。```figure
logsumexp-stability
```## 构建它

### 步骤 1：演示浮点数精度限制```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```### 步骤 2：实现朴素的 vs 稳定的 softmax```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```### 步骤 3：实现稳定的 log-sum-exp```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```### 步骤 4：实现稳定的交叉熵```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```### 步骤 5：梯度检查```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```## 使用它

### 混合精度模拟```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```### 梯度裁剪```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```### NaN/Inf 检测```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```参见 `code/numerical.py` 以查看包含所有边缘情况演示的完整实现。

## 发布它

本课将产生以下内容：
- `code/numerical.py`，具有稳定 softmax、log-sum-exp、交叉熵、梯度检查和混合精度模拟
- `outputs/prompt-numerical-debugger.md`，用于诊断训练中的 NaN/Inf 和数值问题

这些稳定的实现会在第 3 阶段构建训练循环时再次出现，并在第 4 阶段实现注意力机制时再次出现。

## 练习

1. **灾难性抵消。** 使用简单的公式 `E[x^2] - E[x]^2` 在 float32 中计算 [1000000.0, 1000001.0, 1000002.0] 的方差。然后使用 Welford 的在线算法计算它。将其与真实方差（0.6667）进行比较，对比误差。

2. **精度探索。** 找出最小的正 float32 值 `x`，使得在 Python 中 `1.0 + x == 1.0`。这就是机器精度。验证它与 `numpy.finfo(numpy.float32).eps` 是否一致。

3. **log-sum-exp 的边缘情况。** 用以下情况测试你的 `logsumexp_stable` 函数：(a) 所有值相等，(b) 一个值远大于其他值，(c) 所有值都非常小（-1000）。验证它在简单版本失败时给出正确的结果。

4. **神经网络层的梯度检查。** 实现一个线性层 `y = Wx + b` 及其分析的反向传播。使用 `numerical_gradient` 来验证 3x2 权重矩阵的正确性。

5. **损失缩放实验。** 模拟使用 float16 的训练：在 [1e-9, 1e-3] 范围内生成随机梯度，转换为 float16，并测量变成零的比例。然后应用损失缩放（乘以 1024），转换为 float16，再缩放回来，再次测量变成零的比例。

## 关键术语

| 术语 | 人们说 | 实际含义 |
|------|----------------|--------------|
| IEEE 754 | "浮点标准" | 定义二进制浮点格式、舍入规则和特殊值（inf，nan）的国际标准。每个现代 CPU 和 GPU 都实现了它。 |
| 机器精度 | "精度限制" | 在给定浮点格式中，使得 1.0 + e != 1.0 的最小值 e。对于 float32，约为 1.19e-7。 |
| 灾难性抵消 | "减法导致的精度损失" | 当减去几乎相等的浮点数时，有效数字相互抵消，舍入噪声主导结果。 |
| 溢出 | "数值太大" | 结果超过可表示的最大值，变成 inf。exp(89) 在 float32 中溢出。 |
| 欠溢出 | "数值太小" | 结果比最小的正可表示数值更接近于零，变成 0.0。exp(-104) 在 float32 中欠溢出。 |
| log-sum-exp 技巧 | "先减去最大值" | 通过将 exp(max(x)) 提取出来计算 log(sum(exp(x)))，以防止溢出和欠溢出。用于 softmax、交叉熵和对数概率计算。 |
| 稳定 softmax | "不会爆炸的 softmax" | 在对数指数化之前减去最大值。数值上与原始结果相同，不会发生溢出。 |
| 梯度检查 | "验证反向传播" | 将反向传播的分析梯度与有限差分法的数值梯度进行比较，以发现实现中的错误。 |
| 混合精度 | "前向使用 float16，反向使用 float32" | 在速度关键的操作中使用低精度浮点，在数值敏感的操作中使用高精度浮点。通常速度提高 2-3 倍。 |
| 损失缩放 | "防止梯度欠溢出" | 在反向传播之前将损失乘以一个大常数，使梯度保持在 float16 可表示范围内，然后在权重更新之前用相同的常数除以它。 |
| bfloat16 | "脑浮点" | Google 的 16 位格式，有 8 个指数位（与 float32 范围相同）和 7 个尾数位（精度低于 float16）。适用于训练。 |
| 梯度裁剪 | "限制梯度范数" | 缩放梯度向量，使其范数不超过阈值。防止爆炸梯度破坏权重。 |
| NaN | "不是数字" | 来自未定义操作（0/0，inf - inf，sqrt(-1)）的特殊浮点值。会传播到所有后续算术运算中。 |
| Inf | "无穷大" | 来自溢出或除以零的特殊浮点值。可以结合产生 NaN（inf - inf，inf * 0）。 |
| 数值梯度 | "暴力导数" | 通过计算 f(x+h) 和 f(x-h) 并除以 2h 来近似导数。虽然慢，但适用于验证。 |

## 进一步阅读

- [每个计算机科学家都应该知道的浮点运算知识（Goldberg 1991）](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 权威参考，内容密集但完整
- [混合精度训练（Micikevicius 等，2018）](https://arxiv.org/abs/1710.03740) -- 引入 float16 训练损失缩放的 NVIDIA 论文
- [AMP：自动混合精度（PyTorch 文档）](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中混合精度的实用指南
- [bfloat16 格式（Google Cloud TPU 文档）](https://cloud.google.com/tpu/docs/bfloat16) -- Google 选择这种格式用于 TPU 的原因
- [Kahan 求和（维基百科）](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 用于减少浮点求和中舍入误差的算法
