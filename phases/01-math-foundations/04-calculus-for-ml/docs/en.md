# 机器学习中的微积分

> 梯度是机器学习优化的罗盘。掌握偏导数、梯度向量与二阶黑塞矩阵。

**Type:** 学习
**Language:** Python
**Prerequisites:** 无
**Time:** ~45 分钟

## 学习目标

- 计算常见 ML 函数（x^2，sigmoid，交叉熵）的数值和解析导数
- 从头开始实现梯度下降，以在 1D 和 2D 中最小化损失函数
- 推导线性回归模型的梯度，并通过手动更新权重进行训练
- 解释黑塞矩阵、泰勒级数近似及其与优化方法的联系

## 问题

你有一个拥有数百万个权重的神经网络。每个权重是一个旋钮。你需要弄清楚如何转动每一个旋钮，以使模型略微减少错误。微积分给你这个方向。

没有微积分，训练神经网络意味着尝试随机更改并希望得到最好的结果。有了导数，你确切地知道每个权重如何影响误差。每次你都正确地转动每个旋钮。

## 概念

### 什么是导数？

导数衡量变化率。对于函数 y = f(x)，导数 f'(x) 告诉你：如果你将 x 稍微改变一点，y 会变化多少？

从几何上看，导数是某点切线的斜率。

**f(x) = x^2:**

| x | f(x) | f'(x) (斜率) |
|---|---|-----|
| 0 | 0    | 0 (平坦，底部) |
| 1 | 1    | 2 |
| 2 | 4    | 4 (这点的切线斜率) |
| 3 | 9    | 6 |

在 x=2 处，斜率为 4。如果你将 x 稍微向右移动一点，y 会增加大约 4 倍的量。在 x=0 处，斜率为 0。你位于碗的底部。

正式定义：

```
f'(x) = lim   f(x + h) - f(x)
        h->0  -----------------
                     h
```

在代码中，你可以跳过这个限制，直接使用一个非常小的 h。这就是数值导数。

### 偏导数：一次一个变量

实际的函数有很多输入。神经网络的损失函数依赖于成千上万的权重。偏导数将所有变量保持不变，除了一个变量，然后对这个变量求导。

```
f(x, y) = x^2 + 3xy + y^2

df/dx = 2x + 3y     (treat y as a constant)
df/dy = 3x + 2y     (treat x as a constant)
```

每个偏导数回答的问题是：如果我仅仅微调这个权重，损失会如何变化？

### 梯度：所有偏导数的向量

梯度将每一个偏导数收集到一个向量中。对于一个函数 f(x, y, z)，梯度是：

```
grad f = [ df/dx, df/dy, df/dz ]
```

梯度指向最陡上升的方向。为了最小化一个函数，应朝相反方向前进。

**f(x,y) = x^2 + y^2 的等高线图：**

该函数形成一个碗状，等高线为同心圆。最小值在 (0, 0)。

| 点 | grad f | -grad f（下降方向） |
|--|--------|----------|
| (1, 1) | [2, 2]（指向山上，远离最小值） | [-2, -2]（指向山下，朝向最小值） |
| (0, 0) | [0, 0]（平坦，位于最小值） | [0, 0] |

这是一幅梯度下降的图示。计算梯度，取反，然后迈出一步。

### 与优化的联系

训练神经网络是优化。你有一个损失函数 L(w1, w2, ..., wn)，用于衡量模型的错误程度。你希望将其最小化。

```
Gradient descent update rule:

  w_new = w_old - learning_rate * dL/dw

For every weight:
  1. Compute the partial derivative of loss with respect to that weight
  2. Subtract a small multiple of it from the weight
  3. Repeat
```

学习率控制步长。太大了会过冲，太小了则会缓慢爬行。

**损失景观（1D切片）：**

当权重 w 变化时，损失函数 L(w) 形成一个具有峰和谷的曲线。

| 特性 | 描述 |
|-----|-----|
| 全局最小值 | 曲线上最低的点 -- 最优解 |
| 局部最小值 | 比其邻近区域低但不是整体最低的谷 |
| 斜率 | 梯度下降从任何起点沿着斜坡向下移动 |

梯度下降沿着斜坡向下移动。它可能会陷入局部最小值，但在高维空间（数以百万计的权重）中，这很少是实际问题。

### 数值导数与解析导数

计算导数有两种方法。

解析：手动应用微积分规则。对于 f(x) = x^2，导数是 f'(x) = 2x。精确。快速。

数值：使用定义进行近似。对于一个很小的 h，计算 f(x+h) 和 f(x-h)，然后使用差值。

```
Numerical (central difference):

f'(x) ~= f(x + h) - f(x - h)
          -----------------------
                  2h

h = 0.0001 works well in practice
```

数值导数速度较慢，但适用于任何函数。解析导数速度快，但需要你推导公式。神经网络框架使用第三种方法：自动微分，它能够机械地计算精确的导数。你将在第三阶段看到这一点。

### 手动计算简单函数的导数

这些是你会在机器学习中反复看到的导数。

```
Function        Derivative       Used in
--------        ----------       -------
f(x) = x^2     f'(x) = 2x      Loss functions (MSE)
f(x) = wx + b  f'(w) = x        Linear layer (gradient w.r.t. weight)
                f'(b) = 1        Linear layer (gradient w.r.t. bias)
                f'(x) = w        Linear layer (gradient w.r.t. input)
f(x) = e^x     f'(x) = e^x     Softmax, attention
f(x) = ln(x)   f'(x) = 1/x     Cross-entropy loss
f(x) = 1/(1+e^-x)  f'(x) = f(x)(1-f(x))   Sigmoid activation
```

对于 f(x) = x^2:

```
f(x) = x^2    f'(x) = 2x

  x    f(x)   f'(x)   meaning
  -2    4      -4      slope tilts left (decreasing)
  -1    1      -2      slope tilts left (decreasing)
   0    0       0      flat (minimum!)
   1    1       2      slope tilts right (increasing)
   2    4       4      slope tilts right (increasing)
```

对于 f(w) = wx + b，其中 x=3，b=1：

```
f(w) = 3w + 1    f'(w) = 3

The derivative with respect to w is just x.
If x is big, a small change in w causes a big change in output.
```

### 链式法则

当函数被组合时，链式法则告诉你如何进行求导。

```
If y = f(g(x)), then dy/dx = f'(g(x)) * g'(x)

Example: y = (3x + 1)^2
  outer: f(u) = u^2       f'(u) = 2u
  inner: g(x) = 3x + 1    g'(x) = 3
  dy/dx = 2(3x + 1) * 3 = 6(3x + 1)
```

神经网络是由一系列函数组成的链条：输入 -> 线性 -> 激活 -> 线性 -> 激活 -> 损失。反向传播是从输出到输入反复应用链式法则。这就是整个算法。

### Hessian 矩阵

梯度告诉你斜率。Hessian 矩阵告诉你曲率。

Hessian 矩阵是二阶偏导数的矩阵。对于一个函数 f(x1, x2, ..., xn)，Hessian 矩阵的第 (i, j) 个元素是：

```
H[i][j] = d^2f / (dx_i * dx_j)
```

对于一个二元函数 f(x, y):

```
H = | d^2f/dx^2    d^2f/dxdy |
    | d^2f/dydx    d^2f/dy^2 |
```**Hessian 在临界点（梯度 = 0）处告诉你的信息：**

| Hessian 特性 | 含义 | 示例曲面 |
|------|---------|--|
| 正定（所有特征值 > 0） | 局部最小值 | 向上开口的碗 |
| 负定（所有特征值 < 0） | 局部最大值 | 向下开口的碗 |
| 不定（混合特征值） | 马鞍点 | 马鞍形状 |

**示例：** f(x, y) = x^2 - y^2（一个马鞍函数）

```
df/dx = 2x       df/dy = -2y
d^2f/dx^2 = 2    d^2f/dy^2 = -2    d^2f/dxdy = 0

H = | 2   0 |
    | 0  -2 |

Eigenvalues: 2 and -2 (one positive, one negative)
--> Saddle point at (0, 0)
```

与 f(x, y) = x² + y²（一个碗）进行比较：

```
H = | 2  0 |
    | 0  2 |

Eigenvalues: 2 and 2 (both positive)
--> Local minimum at (0, 0)
```**为什么 Hessian 在机器学习中很重要：**

牛顿法使用 Hessian 来进行比梯度下降更好的优化步骤。它不仅仅跟随斜率，还考虑了曲率：

```
Newton's update:    w_new = w_old - H^(-1) * gradient
Gradient descent:   w_new = w_old - lr * gradient
```

牛顿法收敛得更快，因为海森矩阵对梯度进行了“重新缩放”——梯度陡峭的方向会取更小的步长，而梯度平缓的方向会取更大的步长。

问题在于：对于一个有 N 个参数的神经网络，海森矩阵是 N x N 的。一个有 100 万个参数的模型将需要一个包含 1 万亿个元素的矩阵。这就是为什么我们要使用近似方法。

| 方法 | 使用的内容 | 成本 | 收敛性 |
|--------|------|------|---|
| 梯度下降法 | 仅使用一阶导数 | 每次迭代 O(N) | 慢（线性） |
| 牛顿法 | 使用完整的海森矩阵 | 每次迭代 O(N^3) | 快（二次） |
| L-BFGS | 从梯度历史中近似海森矩阵 | 每次迭代 O(N) | 中等（超线性） |
| Adam | 每个参数自适应的学习率（对角海森近似） | 每次迭代 O(N) | 中等 |
| 自然梯度 | 费舍信息矩阵（统计海森矩阵） | 每次迭代 O(N^2) | 快 |

在实践中，Adam 是深度学习的默认优化器。它通过跟踪每个参数梯度的运行均值和方差，以较低的成本近似二阶信息。

### 泰勒级数近似

任何光滑函数都可以用多项式在局部进行近似：

$$ f(x) \approx f(a) + f'(a)(x - a) + \frac{f''(a)}{2}(x - a)^2 + \cdots $$

```
f(x + h) = f(x) + f'(x)*h + (1/2)*f''(x)*h^2 + (1/6)*f'''(x)*h^3 + ...
```

包含的项越多，近似效果越好 —— 但这种近似只在点 x 附近有效。

**泰勒级数对机器学习的重要性：**

- **一阶泰勒展开 = 梯度下降。** 当你使用 f(x + h) ~ f(x) + f'(x)*h 时，你实际上是在进行线性近似。梯度下降通过最小化这个线性模型来选择 h = -lr * f'(x)。

- **二阶泰勒展开 = 牛顿法。** 使用 f(x + h) ~ f(x) + f'(x)*h + (1/2)*f''(x)*h^2 时，你会得到一个二次模型。最小化它得到 h = -f'(x)/f''(x) —— 这就是牛顿法的步长。

- **损失函数设计。** 均方误差（MSE）和交叉熵是光滑的，这意味着它们的泰勒展开是良好行为的。这并非偶然。光滑的损失函数使优化过程变得可预测。

```
Approximation order    What it captures    Optimization method
-------------------    -----------------   -------------------
0th order (constant)   Just the value      Random search
1st order (linear)     Slope               Gradient descent
2nd order (quadratic)  Curvature           Newton's method
Higher orders          Finer structure     Rarely used in ML
```

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

<>

关键见解：所有基于梯度的优化实际上都是在局部近似损失函数，并朝着该近似值的最小值进行更新。

### 机器学习中的积分

导数告诉你变化的速率。积分用于计算累积量——曲线下的面积。

在机器学习中，你很少手动计算积分，但这个概念无处不在：

**概率。** 对于具有密度 p(x) 的连续随机变量：
 /no_think

```
P(a < X < b) = integral from a to b of p(x) dx
```

概率密度曲线在 a 和 b 之间的面积是落在该范围内的概率。

**期望值。** 概率加权的平均结果：

```
E[f(X)] = integral of f(x) * p(x) dx
```

在数据分布上的预期损失是一个积分。训练过程最小化这个积分的经验近似。

**KL散度。** 衡量两个分布之间的差异：

```
KL(p || q) = integral of p(x) * log(p(x) / q(x)) dx
```

用于变分自编码器（VAEs）、知识蒸馏和贝叶斯推理。

**归一化常数。** 在贝叶斯推理中：

```
p(w | data) = p(data | w) * p(w) / integral of p(data | w) * p(w) dw
```

分母是对所有可能参数值的积分。它通常难以直接计算，这就是我们使用诸如MCMC和变分推断等近似方法的原因。

| 积分概念 | 在机器学习中的出现位置 |
|------|----|
| 曲线下面积 | 从密度函数中获得的概率 |
| 期望值 | 损失函数，风险最小化 |
| KL散度 | VAEs，策略优化，蒸馏 |
| 归一化 | 贝叶斯后验，softmax分母 |
| 边缘似然 | 模型比较，证据下界（ELBO） |

### 计算图中的多变量链式法则

链式法则不仅适用于线性标量函数。在神经网络中，变量会发散和合并。这里展示的是导数如何通过一个简单的前向传播过程流动：

```mermaid
graph LR
    x["x (input)"] -->|"*w"| z1["z1 = w*x"]
    z1 -->|"+b"| z2["z2 = w*x + b"]
    z2 -->|"sigmoid"| a["a = sigmoid(z2)"]
    a -->|"loss fn"| L["L = -(y*log(a) + (1-y)*log(1-a))"]
```

反向传播从右到左计算梯度：

```python
# 示例代码
def backward_pass():
    # 计算梯度的逻辑
    pass
```

```mermaid
graph RL
    dL["dL/dL = 1"] -->|"dL/da"| da["dL/da = -y/a + (1-y)/(1-a)"]
    da -->|"da/dz2 = a(1-a)"| dz2["dL/dz2 = dL/da * a(1-a)"]
    dz2 -->|"dz2/dw = x"| dw["dL/dw = dL/dz2 * x"]
    dz2 -->|"dz2/db = 1"| db["dL/db = dL/dz2 * 1"]
```

每条箭头都乘以局部导数。任何参数的梯度是从损失到该参数路径上所有局部导数的乘积。当路径分支和合并时，你需要对贡献进行求和（多元链式法则）。

这就是反向传播的全部内容：系统地通过计算图应用链式法则，从输出到输入。

### 雅可比矩阵

当一个函数将一个向量映射到另一个向量（如神经网络层）时，它的导数是一个矩阵。雅可比矩阵包含了每个输出相对于每个输入的所有偏导数。

对于 f: R^n -> R^m，雅可比矩阵 J 是一个 m x n 的矩阵：

| | x1 | x2 | ... | xn |
|---|---|---|---|---|
| f1 | df1/dx1 | df1/dx2 | ... | df1/dxn |
| f2 | df2/dx1 | df2/dx2 | ... | df2/dxn |
| ... | ... | ... | ... | ... |
| fm | dfm/dx1 | dfm/dx2 | ... | dfm/dxn |

你不会手动计算神经网络的雅可比矩阵。PyTorch 会处理它。但了解它的存在有助于你理解反向传播中的形状：如果某一层将 R^n 映射到 R^m，它的雅可比矩阵是 m x n。梯度会通过该矩阵的转置向后流动。

### 为什么这对神经网络很重要

神经网络中的每个权重都会得到一个梯度。梯度告诉你如何调整该权重以减少损失。

```mermaid
graph LR
    subgraph Forward["Forward Pass"]
        I["input"] --> W1["W1"] --> R["relu"] --> W2["W2"] --> S["softmax"] --> L["loss"]
    end
```

```mermaid
graph RL
    subgraph Backward["Backward Pass"]
        dL["dL/dloss"] --> dW2["dL/dW2"] --> d2["..."] --> dW1["dL/dW1"]
    end
```

每次权重更新：
- `W1 = W1 - lr * dL/dW1`
- `W2 = W2 - lr * dL/dW2`

前向传播计算预测值和损失。反向传播计算损失相对于每个权重的梯度。然后每个权重都向下山的方向迈出一小步。重复数百万次这样的步骤。这就是深度学习。

```figure
derivative-tangent
```

## 构建它

### 第一步：从零开始构建数值导数

```python
def numerical_derivative(f, x, h=1e-7):
    return (f(x + h) - f(x - h)) / (2 * h)

def f(x):
    return x ** 2

for x in [-2, -1, 0, 1, 2]:
    numerical = numerical_derivative(f, x)
    analytical = 2 * x
    print(f"x={x:2d}  f'(x) numerical={numerical:.6f}  analytical={analytical:.1f}")
```

数值导数与解析导数在许多小数位上一致。

### 步骤 2：偏导数和梯度

```python
def numerical_gradient(f, point, h=1e-7):
    gradient = []
    for i in range(len(point)):
        point_plus = list(point)
        point_minus = list(point)
        point_plus[i] += h
        point_minus[i] -= h
        partial = (f(point_plus) - f(point_minus)) / (2 * h)
        gradient.append(partial)
    return gradient

def f_multi(point):
    x, y = point
    return x**2 + 3*x*y + y**2

grad = numerical_gradient(f_multi, [1.0, 2.0])
print(f"Numerical gradient at (1,2): {[f'{g:.4f}' for g in grad]}")
print(f"Analytical gradient at (1,2): [2*1+3*2, 3*1+2*2] = [{2*1+3*2}, {3*1+2*2}]")
```

### 步骤 3：使用梯度下降法找到 f(x) = x^2 的最小值

```python
x = 5.0
lr = 0.1
for step in range(20):
    grad = 2 * x
    x = x - lr * grad
    print(f"step {step:2d}  x={x:8.4f}  f(x)={x**2:10.6f}")
```

从 x=5 开始，每一步都更接近 x=0（最小值）。

### 步骤 4：在二维函数上进行梯度下降

```python
def f_2d(point):
    x, y = point
    return x**2 + y**2

point = [4.0, 3.0]
lr = 0.1
for step in range(30):
    grad = numerical_gradient(f_2d, point)
    point = [p - lr * g for p, g in zip(point, grad)]
    loss = f_2d(point)
    if step % 5 == 0 or step == 29:
        print(f"step {step:2d}  point=({point[0]:7.4f}, {point[1]:7.4f})  f={loss:.6f}")
```

### 步骤 5：比较数值导数和解析导数

```python
import math

test_functions = [
    ("x^2",      lambda x: x**2,          lambda x: 2*x),
    ("x^3",      lambda x: x**3,          lambda x: 3*x**2),
    ("sin(x)",   lambda x: math.sin(x),   lambda x: math.cos(x)),
    ("e^x",      lambda x: math.exp(x),   lambda x: math.exp(x)),
    ("1/x",      lambda x: 1/x,           lambda x: -1/x**2),
]

x = 2.0
print(f"{'Function':<12} {'Numerical':>12} {'Analytical':>12} {'Error':>12}")
print("-" * 50)
for name, f, df in test_functions:
    num = numerical_derivative(f, x)
    ana = df(x)
    err = abs(num - ana)
    print(f"{name:<12} {num:12.6f} {ana:12.6f} {err:12.2e}")
```

### 步骤 6：数值计算 Hessian 矩阵

```python
def hessian_2d(f, x, y, h=1e-5):
    fxx = (f(x + h, y) - 2 * f(x, y) + f(x - h, y)) / (h ** 2)
    fyy = (f(x, y + h) - 2 * f(x, y) + f(x, y - h)) / (h ** 2)
    fxy = (f(x + h, y + h) - f(x + h, y - h) - f(x - h, y + h) + f(x - h, y - h)) / (4 * h ** 2)
    return [[fxx, fxy], [fxy, fyy]]

def saddle(x, y):
    return x ** 2 - y ** 2

def bowl(x, y):
    return x ** 2 + y ** 2

H_saddle = hessian_2d(saddle, 0.0, 0.0)
H_bowl = hessian_2d(bowl, 0.0, 0.0)
print(f"Saddle Hessian: {H_saddle}")  # [[2, 0], [0, -2]] -- mixed signs
print(f"Bowl Hessian:   {H_bowl}")    # [[2, 0], [0, 2]]  -- both positive
```

鞍函数的 Hessian 矩阵的特征值为 2 和 -2（符号混合，确认为鞍点）。碗状函数的特征值为 2 和 2（两者均为正，确认为最小值）。

### 步骤 7：泰勒近似的应用

```python
import math

def taylor_approx(f, f_prime, f_double_prime, x0, h, order=2):
    result = f(x0)
    if order >= 1:
        result += f_prime(x0) * h
    if order >= 2:
        result += 0.5 * f_double_prime(x0) * h ** 2
    return result

x0 = 0.0
for h in [0.1, 0.5, 1.0, 2.0]:
    true_val = math.sin(h)
    t1 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=1)
    t2 = taylor_approx(math.sin, math.cos, lambda x: -math.sin(x), x0, h, order=2)
    print(f"h={h:.1f}  sin(h)={true_val:.4f}  order1={t1:.4f}  order2={t2:.4f}")
```

在 x₀=0 附近，sin(x) ~ x（一阶泰勒展开）。对于小的 h，这种近似非常精确，但对于大的 h，这种近似就不成立了。这就是为什么梯度下降在使用小的学习率时效果最好——每一步都假设线性近似是准确的。

### 步骤 8：这对神经网络的意义

```python
import random

random.seed(42)

w = random.gauss(0, 1)
b = random.gauss(0, 1)
lr = 0.01

xs = [1.0, 2.0, 3.0, 4.0, 5.0]
ys = [3.0, 5.0, 7.0, 9.0, 11.0]

for epoch in range(200):
    total_loss = 0
    dw = 0
    db = 0
    for x, y in zip(xs, ys):
        pred = w * x + b
        error = pred - y
        total_loss += error ** 2
        dw += 2 * error * x
        db += 2 * error
    dw /= len(xs)
    db /= len(xs)
    total_loss /= len(xs)
    w -= lr * dw
    b -= lr * db
    if epoch % 40 == 0 or epoch == 199:
        print(f"epoch {epoch:3d}  w={w:.4f}  b={b:.4f}  loss={total_loss:.6f}")

print(f"\nLearned: y = {w:.2f}x + {b:.2f}")
print(f"Actual:  y = 2x + 1")
```

每个基于梯度的训练循环都遵循以下模式：预测，计算损失，计算梯度，更新权重。

## 使用它

使用 NumPy，相同的操作更快且更简洁：

```python
import numpy as np

x = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([3, 5, 7, 9, 11], dtype=float)

w, b = np.random.randn(), np.random.randn()
lr = 0.01

for epoch in range(200):
    pred = w * x + b
    error = pred - y
    loss = np.mean(error ** 2)
    dw = np.mean(2 * error * x)
    db = np.mean(2 * error)
    w -= lr * dw
    b -= lr * db

print(f"Learned: y = {w:.2f}x + {b:.2f}")
```

你刚刚从零开始构建了梯度下降。PyTorch 会自动计算梯度，但更新循环是相同的。

## 练习

1. 使用两次调用的 `numerical_derivative` 实现 `numerical_second_derivative(f, x)`。验证 x^3 在 x=2 处的二阶导数是 12。
2. 使用梯度下降法找到 f(x, y) = (x - 3)^2 + (y + 1)^2 的最小值。从 (0, 0) 开始。答案应该收敛到 (3, -1)。
3. 为梯度下降循环添加动量：维护一个累积过去梯度的速度向量。比较在 f(x) = x^4 - 3x^2 上有和没有动量的收敛速度。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 导数 | “斜率” | 函数在某一点的变化率。告诉你输入每单位变化时输出变化多少。 |
| 偏导数 | “单变量的导数” | 在固定其他变量的情况下，对一个变量求导。 |
| 梯度 | “最陡上升的方向” | 所有偏导数组成的向量。指向函数增长最快的方向。 |
| 梯度下降 | “下山” | 从参数中减去梯度（乘以学习率）以减少损失。神经网络训练的核心。 |
| 学习率 | “步长” | 控制每次梯度下降步长的标量。太大：发散。太小：收敛缓慢。 |
| 链式法则 | “相乘导数” | 对复合函数求导的规则：df/dx = df/dg * dg/dx。反向传播的数学基础。 |
| 雅可比矩阵 | “导数矩阵” | 当函数将向量映射到向量时，雅可比矩阵是所有输出相对于输入的偏导数组成的矩阵。 |
| 数值导数 | “有限差分” | 通过在两个邻近点评估函数并计算它们之间的斜率来近似导数。 |
| 反向传播 | “反向自动微分” | 使用链式法则从输出到输入逐层计算梯度。神经网络学习的方式。 |
| 海森矩阵 | “二阶导数矩阵” | 所有二阶偏导数组成的矩阵。描述函数的曲率。临界点的正定海森矩阵表示局部最小值。 |
| 泰勒级数 | “多项式近似” | 使用导数在某一点附近近似函数：f(x+h) ~ f(x) + f'(x)h + (1/2)f''(x)h^2 + ... 理解梯度下降和牛顿法为何有效的基础。 |
| 积分 | “曲线下的面积” | 在一个范围内的数量累积。在机器学习中，积分定义概率、期望值和KL散度。 |

## 进一步阅读

- [3Blue1Brown: 微积分的本质](https://www.3blue1brown.com/topics/calculus) - 导数、积分和链式法则的视觉直觉
- [Stanford CS231n: 反向传播](https://cs231n.github.io/optimization-2/) - 梯度如何通过神经网络层流动
