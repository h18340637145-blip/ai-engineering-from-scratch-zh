# 最优化方法

> 在陡峭的损失曲面上下行。掌握 SGD、Momentum 动量与 Adam 自适应优化算法。

**Type:** 构建  
**Language:** Python  
**Prerequisites:** Phase 1, Lesson 04 (机器学习中的微积分)  
**Time:** ~45 分钟  

## Learning Objectives

- 从零实现 vanilla gradient descent、带动量的 SGD 与 Adam  
- 在 Rosenbrock 函数上比较优化器的收敛性，并解释为什么 Adam 可以按权重自适应学习率  
- 区分凸与非凸损失曲面，并解释高维空间中鞍点的作用  
- 为训练稳定性配置学习率调度（step decay、cosine annealing、warmup）  

## The Problem

你有一个损失函数。它告诉你你的模型有多错误。你有梯度。它们告诉你哪个方向会使损失变得更糟。现在你需要一个策略来向下坡走。

最简单的方法是：沿梯度的反方向移动。按某个称为学习率的数值缩放步长。重复。这就是梯度下降，它确实有效。但“有效”有一些限制条件。学习率太大，你会完全错过山谷，来回反弹。学习率太小，你则需要数千步才能缓慢接近答案。遇到鞍点时，即使没有找到最小值，你也会停止移动。

深度学习中的每个优化器都是对同一个问题的回答：如何更快且更可靠地到达山谷底部？

## The Concept

### 优化的含义

优化是寻找使函数最小化（或最大化）的输入值。在机器学习中，函数是损失函数，输入是模型的权重。训练就是优化。```
minimize L(w) where:
  L = loss function
  w = model weights (could be millions of parameters)
```### 梯度下降（vanilla）

最简单的优化器。计算损失函数相对于每个权重的梯度。将每个权重沿着其梯度的相反方向移动。根据学习率对步长进行缩放。```
w = w - lr * gradient
```这就是整个算法。一行代码。```mermaid
graph TD
    A["* Starting point (high loss)"] --> B["Moving downhill along gradient"]
    B --> C["Approaching minimum"]
    C --> D["o Minimum (low loss)"]
```### 学习率：最重要的超参数

学习率控制步长。它决定了收敛的各个方面。```mermaid
graph LR
    subgraph TooLarge["Too Large (lr = 1.0)"]
        A1["Step 1"] -->|overshoot| A2["Step 2"]
        A2 -->|overshoot| A3["Step 3"]
        A3 -->|diverging| A4["..."]
    end
    subgraph TooSmall["Too Small (lr = 0.0001)"]
        B1["Step 1"] -->|tiny step| B2["Step 2"]
        B2 -->|tiny step| B3["Step 3"]
        B3 -->|10,000 steps later| B4["Minimum"]
    end
    subgraph JustRight["Just Right (lr = 0.01)"]
        C1["Start"] --> C2["..."] --> C3["Converged in ~100 steps"]
    end
```没有正确的学习率公式。你只能通过实验找到它。常见的起始点：Adam 使用 0.001，带动量的 SGD 使用 0.01。

### SGD 与批量（batch）与小批量（mini-batch）

普通梯度下降法在采取一步之前，会计算整个数据集的梯度。这叫做批量梯度下降法。它稳定但缓慢。

随机梯度下降法（SGD）对单个随机样本计算梯度并立即采取一步。它嘈杂但快速。

小批量梯度下降法取两者之间的折中。对小批量（32、64、128、256 个样本）计算梯度，然后采取一步。这是每个人实际上使用的。

| 变体       | 批量大小       | 梯度质量       | 每步速度       | 噪声     |
|------------|----------------|----------------|----------------|----------|
| 批量梯度下降 | 整个数据集     | 精确           | 慢             | 无       |
| SGD        | 1 个样本       | 非常嘈杂       | 快             | 高       |
| 小批量     | 32-256         | 良好的估计     | 平衡           | 中等     |

SGD 和小批量中的噪声不是错误。它有助于逃离浅层局部最小值和鞍点。

### 动量：滚动下坡的球

普通梯度下降法只关注当前梯度。如果梯度出现来回震荡（在狭窄山谷中常见），进展会很缓慢。动量通过将过去的梯度累积到速度项中来解决这个问题。```
v = beta * v + gradient
w = w - lr * v
```类比：一个球沿着下坡滚动。它在遇到每一个障碍时不会停止然后再启动。它以一致的方向加速，并减弱震荡。```mermaid
graph TD
    subgraph Without["Without Momentum (zigzag, slow)"]
        W1["Start"] -->|left| W2[" "]
        W2 -->|right| W3[" "]
        W3 -->|left| W4[" "]
        W4 -->|right| W5[" "]
        W5 -->|left| W6[" "]
        W6 --> W7["Minimum"]
    end
    subgraph With["With Momentum (smooth, fast)"]
        M1["Start"] --> M2[" "] --> M3[" "] --> M4["Minimum"]
    end
````beta`（通常为0.9）控制保留多少历史信息。较高的beta值意味着更强的动量，路径更平滑，但对方向变化的响应更慢。

### Adam：自适应学习率

不同的权重需要不同的学习率。一个很少产生大梯度的权重，在最终产生梯度时应该采取更大的步长。一个经常产生巨大梯度的权重应该采取更小的步长。

Adam（自适应动量估计）为每个权重跟踪两件事：

1. 一阶矩（m）：梯度的移动平均值（类似于动量）
2. 二阶矩（v）：梯度平方的移动平均值（梯度幅度）```
m = beta1 * m + (1 - beta1) * gradient
v = beta2 * v + (1 - beta2) * gradient^2

m_hat = m / (1 - beta1^t)    bias correction
v_hat = v / (1 - beta2^t)    bias correction

w = w - lr * m_hat / (sqrt(v_hat) + epsilon)
```由 `sqrt(v_hat)` 进行的除法是关键的见解。梯度较大的权重会被一个较大的数除（有效步长较小）。梯度较小的权重会被一个较小的数除（有效步长较大）。每个权重都会拥有自己自适应的学习率。

默认的超参数：`lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8`。这些默认值对于大多数问题都能很好地工作。

### 学习率调度

固定的学习率是一种折中方案。训练初期，你希望使用较大的步长以快速取得进展。训练后期，你希望使用较小的步长以在最小值附近进行微调。

常见的调度方式：

| 调度方式 | 公式 | 使用场景 |
|---------|------|---------|
| 步长衰减 | 每 N 个周期，lr = lr * factor | 简单，手动控制 |
| 指数衰减 | lr = lr_0 * decay^t | 平滑减少 |
| 余弦退火 | lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + cos(pi * t / T)) | Transformer，现代训练 |
| 预热 + 衰减 | 线性上升，然后衰减 | 大型模型，防止早期不稳定性 |

### 凸函数与非凸函数

凸函数只有一个最小值。梯度下降总是能找到这个最小值。像 `f(x) = x^2` 这样的二次函数是凸函数。

神经网络的损失函数是非凸的。它们有很多局部最小值、鞍点和平坦区域。```mermaid
graph LR
    subgraph Convex["Convex: One valley, one answer"]
        direction TB
        CV1["High loss"] --> CV2["Global minimum"]
    end
    subgraph NonConvex["Non-convex: Multiple valleys, saddle points"]
        direction TB
        NC1["Start"] --> NC2["Local minimum"]
        NC1 --> NC3["Saddle point"]
        NC1 --> NC4["Global minimum"]
    end
```实际上，在高维神经网络中，局部最小值很少成为问题。大多数局部最小值的损失值都接近全局最小值。真正的问题是鞍点（在某些方向上平坦，在其他方向上弯曲）。动量和小批量的噪声有助于逃离这些鞍点。

### 损失景观可视化

损失是所有权重的函数。对于一个拥有100万权重的模型，损失景观存在于100万+1维的空间中。我们通过在权重空间中选择两个随机方向，并沿着这两个方向绘制损失，从而对其进行可视化，生成一个二维曲面。```mermaid
graph TD
    HL["High loss region"] --> SP["Saddle point"]
    HL --> LM["Local minimum"]
    SP --> LM
    SP --> GM["Global minimum"]
    LM -.->|"shallow barrier"| GM
    style HL fill:#ff6666,color:#000
    style SP fill:#ffcc66,color:#000
    style LM fill:#66ccff,color:#000
    style GM fill:#66ff66,color:#000
```尖锐的极小值泛化能力差。平坦的极小值泛化能力好。这是为什么带有动量的随机梯度下降（SGD）通常在最终测试准确率上优于Adam的一个原因：它的噪声防止了陷入尖锐的极小值。```figure
gradient-descent
```## 构建它

### 步骤 1：定义一个测试函数

Rosenbrock 函数是一个经典的优化基准测试函数。它的最小值位于一个狭窄的弯曲山谷内部的 (1, 1) 点，这个点很容易找到，但很难跟踪。```
f(x, y) = (1 - x)^2 + 100 * (y - x^2)^2
```

```python
def rosenbrock(params):
    x, y = params
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2

def rosenbrock_gradient(params):
    x, y = params
    df_dx = -2 * (1 - x) + 200 * (y - x ** 2) * (-2 * x)
    df_dy = 200 * (y - x ** 2)
    return [df_dx, df_dy]
```### 步骤 2：普通梯度下降```python
class GradientDescent:
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return [p - self.lr * g for p, g in zip(params, grads)]
```### 步骤 3：带有动量的 SGD```python
class SGDMomentum:
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = None

    def step(self, params, grads):
        if self.velocity is None:
            self.velocity = [0.0] * len(params)
        self.velocity = [
            self.momentum * v + g
            for v, g in zip(self.velocity, grads)
        ]
        return [p - self.lr * v for p, v in zip(params, self.velocity)]
```### 步骤 4: Adam```python
class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = None
        self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = [0.0] * len(params)
            self.v = [0.0] * len(params)

        self.t += 1

        self.m = [
            self.beta1 * m + (1 - self.beta1) * g
            for m, g in zip(self.m, grads)
        ]
        self.v = [
            self.beta2 * v + (1 - self.beta2) * g ** 2
            for v, g in zip(self.v, grads)
        ]

        m_hat = [m / (1 - self.beta1 ** self.t) for m in self.m]
        v_hat = [v / (1 - self.beta2 ** self.t) for v in self.v]

        return [
            p - self.lr * mh / (vh ** 0.5 + self.epsilon)
            for p, mh, vh in zip(params, m_hat, v_hat)
        ]
```### 步骤 5：运行和比较```python
def optimize(optimizer, func, grad_func, start, steps=5000):
    params = list(start)
    history = [params[:]]
    for _ in range(steps):
        grads = grad_func(params)
        params = optimizer.step(params, grads)
        history.append(params[:])
    return history

start = [-1.0, 1.0]

gd_history = optimize(GradientDescent(lr=0.0005), rosenbrock, rosenbrock_gradient, start)
sgd_history = optimize(SGDMomentum(lr=0.0001, momentum=0.9), rosenbrock, rosenbrock_gradient, start)
adam_history = optimize(Adam(lr=0.01), rosenbrock, rosenbrock_gradient, start)

for name, history in [("GD", gd_history), ("SGD+M", sgd_history), ("Adam", adam_history)]:
    final = history[-1]
    loss = rosenbrock(final)
    print(f"{name:6s} -> x={final[0]:.6f}, y={final[1]:.6f}, loss={loss:.8f}")
```预期输出：Adam 收敛最快。带动量的 SGD 路径更平滑。普通 GD 在狭窄的山谷中进展缓慢。

## 使用方法

在实践中，使用 PyTorch 或 JAX 的优化器。它们可以处理参数组、权重衰减、梯度裁剪和 GPU 加速。```python
import torch

model = torch.nn.Linear(784, 10)

sgd = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
adam = torch.optim.Adam(model.parameters(), lr=0.001)
adamw = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(adam, T_max=100)
```经验法则：

- 从 Adam（lr=0.001）开始。它在大多数问题中无需调整即可正常工作。
- 当你需要最佳最终精度并且能够承担更多调整时，切换为带有动量的 SGD（lr=0.01, momentum=0.9）。
- 对于变压器模型，使用 AdamW（带有解耦权重衰减的 Adam）。
- 对于训练周期超过几个 epoch 的情况，始终使用学习率调度。
- 如果训练不稳定，降低学习率。如果训练太慢，增加它。

## 发布它

本课生成了一个提示，用于选择合适的优化器。参见 `outputs/prompt-optimizer-guide.md`。

在这里构建的优化器类在第 3 阶段会出现，那时我们将从头开始训练神经网络。

## 练习

1. **学习率扫描。** 在 Rosenbrock 函数上运行标准梯度下降，使用学习率 [0.0001, 0.0005, 0.001, 0.005, 0.01]。对每个学习率，在 5000 步后绘制或打印最终损失。找到仍能收敛的最大学习率。

2. **动量比较。** 在 Rosenbrock 函数上运行带有动量值 [0.0, 0.5, 0.9, 0.99] 的 SGD。跟踪每一步的损失。哪个动量值收敛最快？哪个会过冲？

3. **鞍点逃离。** 定义函数 `f(x, y) = x^2 - y^2`（原点处的鞍点）。从 (0.01, 0.01) 开始。比较标准 GD、带动量的 SGD 和 Adam 的行为。哪个能逃离鞍点？

4. **实现学习率衰减。** 向 GradientDescent 类添加一个指数衰减调度：`lr = lr_0 * 0.999^step`。比较在 Rosenbrock 函数上使用和不使用衰减时的收敛情况。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 梯度下降 | “向山下走” | 通过减去学习率缩放后的梯度来更新权重。最基础的优化器。 |
| 学习率 | “步长” | 一个标量，控制每次更新移动权重的距离。太大导致发散，太小浪费计算资源。 |
| 动量 | “持续滚动” | 将过去的梯度累积成一个速度向量。减少振荡，加速在一致方向上的移动。 |
| SGD | “随机采样” | 随机梯度下降。在随机子集上计算梯度，而不是全部数据集。实践中几乎总是指小批量 SGD。 |
| 小批量 | “一组数据” | 用于估计梯度的训练数据的小子集（32-256 个样本）。平衡速度和梯度准确性。 |
| Adam | “默认优化器” | 自适应矩估计。跟踪每个权重的梯度和梯度平方的运行平均值，为每个权重提供自己的学习率。 |
| 偏差校正 | “修正冷启动” | Adam 的一阶和二阶矩初始化为零。偏差校正通过除以 (1 - beta^t) 来补偿早期步骤。 |
| 学习率调度 | “随时间改变学习率” | 一个在训练期间调整学习率的函数。早期大步，后期小步。 |
| 凸函数 | “一个山谷” | 任何局部最小值都是全局最小值的函数。梯度下降总能找到它。神经网络损失不是凸函数。 |
| 鞍点 | “平坦但不是最小值” | 一个梯度为零的点，但在某些方向是极小值，在其他方向是极大值。在高维中很常见。 |
| 损失地形 | “地形” | 在权重空间中绘制的损失函数。通过沿两个随机方向切片进行可视化。 |
| 收敛 | “到达那里” | 优化器已达到一个点，进一步的步骤不会显著降低损失。 |

## 进一步阅读

- [Sebastian Ruder: 梯度下降优化算法概述](https://ruder.io/optimizing-gradient-descent/) - 所有主要优化器的全面综述
- [为什么动量真的有效（Distill）](https://distill.pub/2017/momentum/) - 动量动态的交互可视化
- [Adam: 随机优化的一种方法（Kingma & Ba, 2014）](https://arxiv.org/abs/1412.6980) - 原始 Adam 论文，可读且简短
- [可视化神经网络的损失地形（Li 等人，2018）](https://arxiv.org/abs/1712.09913) - 展示尖锐最小值与平坦最小值的论文
