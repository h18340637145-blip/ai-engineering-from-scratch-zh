# 多层网络与前向传播

> 一个神经元可以画一条线。将它们堆叠起来，你就可以画出任何东西。

**类型:** 构建
**语言:** Python
**先决条件:** 第01阶段（数学基础），第03.01课（感知器）
**时间:** ~90分钟

## 学习目标

- 使用Layer和Network类从头构建一个可以完成完整前向传播的多层网络
- 追踪网络中每一层的矩阵维度，并识别形状不匹配的情况
- 解释堆叠非线性激活函数如何使网络能够学习弯曲的决策边界
- 使用手动调整的sigmoid权重，通过2-2-1架构解决XOR问题

## 问题

单个神经元是一个画线工具。仅此而已。它只能在你的数据上画一条直线。在人工智能中的每一个实际问题——图像识别、语言理解、下围棋——都需要曲线。将神经元堆叠成层，你才能得到曲线。

1969年，明斯基和帕皮特证明了这一限制是致命的：单层网络无法学习XOR。不是“难以学习”，而是数学上根本无法做到。XOR真值表将[0,1]和[1,0]放在一侧，将[0,0]和[1,1]放在另一侧。没有任何一条直线可以将它们分开。

这导致神经网络的资金支持在十年内被终止。但事后看来，解决方法是显而易见的：停止使用单层。将神经元堆叠成层。让第一层将输入空间划分为新的特征，让第二层将这些特征组合成单条直线无法实现的决策。

这种堆叠结构就是多层网络。它是一切当前生产中的深度学习模型的基础。前向传播——数据从输入层经过隐藏层流向输出层——是你在任何其他东西能正常工作之前，需要首先构建的东西。

## 概念

### 层：输入层、隐藏层、输出层

一个多层网络有三种类型的层：

**输入层** —— 严格来说并不是一层。它保存你的原始数据。如果有两个特征，那就意味着有两个输入节点。这里不会进行任何计算。

**隐藏层** —— 这里才是工作的核心。每个神经元都接收前一层的所有输出，应用权重和偏置，然后将结果通过激活函数进行传递。“隐藏”是因为这些值在训练数据中永远无法直接看到。

**输出层** —— 最终的答案。对于二分类，一个带有sigmoid函数的神经元；对于多分类，每个类别一个神经元。```mermaid
graph LR
    subgraph Input["Input Layer"]
        x1["x1"]
        x2["x2"]
    end
    subgraph Hidden["Hidden Layer (3 neurons)"]
        h1["h1"]
        h2["h2"]
        h3["h3"]
    end
    subgraph Output["Output Layer"]
        y["y"]
    end
    x1 --> h1
    x1 --> h2
    x1 --> h3
    x2 --> h1
    x2 --> h2
    x2 --> h3
    h1 --> y
    h2 --> y
    h3 --> y
```这是一个 2-3-1 网络。两个输入，三个隐藏神经元，一个输出。每条连接都携带一个权重。每个神经元（除了输入）都携带一个偏置。

每一层都会产生一个称为隐藏状态的数字向量。对于文本，隐藏状态会增加维度——将一个词编码为 768 个数字以捕捉语义信息。对于图像，它们会减少维度——将数百万个像素压缩成可管理的表示形式。隐藏状态是学习发生的地方。

### 神经元和激活函数

每个神经元执行以下三个步骤：

1. 将每个输入乘以对应的权重
2. 将所有乘积相加并加上偏置
3. 将总和通过一个激活函数

目前，激活函数是 sigmoid：```
sigmoid(z) = 1 / (1 + e^(-z))
```Sigmoid 将任何数字压缩到范围 (0, 1) 内。较大的正输入会推向 1。较大的负输入会推向 0。零映射到 0.5。这种平滑的曲线使得学习成为可能 —— 与感知机的硬阈值不同，Sigmoid 在任何地方都有梯度。

### 前向传播：数据如何流动

前向传播将输入数据逐层通过网络，直到到达输出。在前向传播过程中不会发生学习。它纯粹是计算：相乘、相加、激活、重复。```mermaid
graph TD
    X["Input: [x1, x2]"] --> WH["Multiply by Weight Matrix W1 (2x3)"]
    WH --> BH["Add Bias Vector b1 (3,)"]
    BH --> AH["Apply sigmoid to each element"]
    AH --> H["Hidden Output: [h1, h2, h3]"]
    H --> WO["Multiply by Weight Matrix W2 (3x1)"]
    WO --> BO["Add Bias Vector b2 (1,)"]
    BO --> AO["Apply sigmoid"]
    AO --> Y["Output: y"]
```在每一层，依次发生三个操作：```
z = W * input + b       (linear transformation)
a = sigmoid(z)           (activation)
```一层的输出成为下一层的输入。这就是整个前向传播过程。

### 矩阵维度

跟踪维度是深度学习中最重要的调试技巧。以下是一个2-3-1网络：

| 步骤 | 操作 | 维度 | 结果形状 |
|------|-----------|------|----------|
| 输入 | x | -- | (2,) |
| 隐藏层线性变换 | W1 * x + b1 | W1: (3, 2), b1: (3,) | (3,) |
| 隐藏层激活函数 | sigmoid(z1) | -- | (3,) |
| 输出层线性变换 | W2 * h + b2 | W2: (1, 3), b2: (1,) | (1,) |
| 输出激活函数 | sigmoid(z2) | -- | (1,) |

规则：第k层的权重矩阵W的形状是（第k层的神经元数目，第k-1层的神经元数目）。行对应当前层，列对应前一层。如果形状不匹配，说明你有错误。

### 普适逼近定理

1989年，George Cybenko证明了一个惊人的结论：一个具有单个隐藏层和足够多神经元的神经网络可以以任意所需的精度逼近任何连续函数。

这并不意味着单个隐藏层总是最优的。这意味着从理论上讲，这种架构是可行的。在实践中，深度网络（更多层，每层更少的神经元）使用远少于浅而宽网络的总参数数就能学习到相同的函数。这就是深度学习之所以有效的原因。

直觉：隐藏层中的每个神经元学习一个“凸起”或特征。足够多的凸起放置在正确的位置，可以逼近任何平滑曲线。更多的神经元，更多的凸起，更好的逼近。```mermaid
graph LR
    subgraph FewNeurons["4 Hidden Neurons"]
        A["Rough approximation"]
    end
    subgraph MoreNeurons["16 Hidden Neurons"]
        B["Close approximation"]
    end
    subgraph ManyNeurons["64 Hidden Neurons"]
        C["Near-perfect fit"]
    end
    FewNeurons --> MoreNeurons --> ManyNeurons
```### 可组合性

神经网络是可组合的。你可以将它们堆叠、串联，或者并行运行。Whisper 模型使用一个编码器网络来处理音频，使用一个独立的解码器网络来生成文本。现代的大型语言模型（LLMs）是仅解码器的。BERT 是仅编码器的。T5 是编码器-解码器结构的。架构的选择决定了模型能做什么。```figure
mlp-forward
```## 构建它

纯 Python。不使用 numpy。所有矩阵运算都从零开始编写。

### 步骤 1：Sigmoid 激活函数```python
import math

def sigmoid(x):
    x = max(-500.0, min(500.0, x))
    return 1.0 / (1.0 + math.exp(-x))
```将值限制在 [-500, 500] 范围内可以防止溢出。`math.exp(500)` 是一个大但有限的数值。`math.exp(1000)` 是无穷大。

### 第二步：层类

在所有深度学习操作中，最重要的操作是矩阵乘法。每一层、每一个注意力头、每一次前向传播——归根结底都是矩阵乘法。一个线性层接受一个输入向量，将其乘以一个权重矩阵，并加上一个偏置向量：y = Wx + b。这个单一的方程占神经网络计算量的 90%。

一个层保存一个权重矩阵和一个偏置向量。它的前向方法接受一个输入向量，并返回激活后的输出。```python
class Layer:
    def __init__(self, n_inputs, n_neurons, weights=None, biases=None):
        if weights is not None:
            self.weights = weights
        else:
            import random
            self.weights = [
                [random.uniform(-1, 1) for _ in range(n_inputs)]
                for _ in range(n_neurons)
            ]
        if biases is not None:
            self.biases = biases
        else:
            self.biases = [0.0] * n_neurons

    def forward(self, inputs):
        self.last_input = inputs
        self.last_output = []
        for neuron_idx in range(len(self.weights)):
            z = sum(
                w * x for w, x in zip(self.weights[neuron_idx], inputs)
            )
            z += self.biases[neuron_idx]
            self.last_output.append(sigmoid(z))
        return self.last_output
```权重矩阵的形状为 (n_neurons, n_inputs)。每一行代表一个神经元在所有输入上的权重。前向方法遍历每个神经元，计算加权和加上偏置，应用sigmoid函数，并收集结果。

### 步骤3：网络类

网络是由多个层组成的列表。前向传播将它们串联起来：第k层的输出作为第k+1层的输入。```python
class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, inputs):
        current = inputs
        for layer in self.layers:
            current = layer.forward(current)
        return current
```这就是整个前向传播过程。四行逻辑。数据输入，流经每一层，从另一侧输出。

### 第4步：使用手动调整的权重进行异或（XOR）

在第01课中，我们通过组合或（OR）、与非（NAND）和与（AND）感知器解决了异或（XOR）问题。现在用我们的Layer类和Network类完成同样的事情。2-2-1的架构：两个输入，两个隐藏神经元，一个输出。```python
hidden = Layer(
    n_inputs=2,
    n_neurons=2,
    weights=[[20.0, 20.0], [-20.0, -20.0]],
    biases=[-10.0, 30.0],
)

output = Layer(
    n_inputs=2,
    n_neurons=1,
    weights=[[20.0, 20.0]],
    biases=[-30.0],
)

xor_net = Network([hidden, output])

xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

for inputs, expected in xor_data:
    result = xor_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    print(f"  {inputs} -> {result[0]:.6f} (rounded: {predicted}, expected: {expected})")
```大权重（20, -20）使 sigmoid 函数表现得像一个阶跃函数。第一个隐藏神经元近似 OR 运算。第二个近似 NAND 运算。输出神经元将它们组合成 AND 运算，也就是 XOR 运算。

### 步骤 5：圆分类

一个更难的问题：将二维点分类为以原点为中心、半径为 0.5 的圆的内部或外部。这需要一个弯曲的决策边界——单个感知机无法实现。```python
import random
import math

random.seed(42)

data = []
for _ in range(200):
    x = random.uniform(-1, 1)
    y = random.uniform(-1, 1)
    label = 1 if (x * x + y * y) < 0.25 else 0
    data.append(([x, y], label))

circle_net = Network([
    Layer(n_inputs=2, n_neurons=8),
    Layer(n_inputs=8, n_neurons=1),
])
```使用随机权重时，网络将无法很好地进行分类。但正向传播仍然可以运行。这一点很重要 —— 正向传播只是计算。学习合适的权重是反向传播，将在第 03 课中讲解。```python
correct = 0
for inputs, expected in data:
    result = circle_net.forward(inputs)
    predicted = 1 if result[0] >= 0.5 else 0
    if predicted == expected:
        correct += 1

print(f"Accuracy with random weights: {correct}/{len(data)} ({100*correct/len(data):.1f}%)")
```随机权重的准确性很差 -- 通常甚至不如猜测多数类。经过训练（第 03 课）后，使用相同架构但包含 8 个隐藏神经元的模型将绘制出一条曲线边界，将内部与外部分隔开。

## 使用方法

PyTorch 用四行代码即可完成上述所有操作：```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Sigmoid(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

x = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
output = model(x)
print(output)
````nn.Linear(2, 8)` 是你的 Layer 类：形状为 (8, 2) 的权重矩阵，形状为 (8,) 的偏置向量。`nn.Sigmoid()` 是你的 sigmoid 函数，按元素应用。`nn.Sequential` 是你的 Network 类：按顺序链接各层。

区别在于速度和规模。PyTorch 运行在 GPU 上，可以处理数百万样本的批次，并自动计算反向传播所需的梯度。但前向传播的逻辑与你刚刚从零开始构建的一样。

## 发布它

本课生成一个可重复使用的提示，用于设计网络架构：

- `outputs/prompt-network-architect.md`

当你需要决定对于某个特定问题使用多少层、每层使用多少个神经元、使用哪些激活函数时，使用这个提示。

## 练习

1. 构建一个 2-4-2-1 网络（两个隐藏层），在 XOR 数据上使用随机权重运行前向传播。打印中间隐藏层的输出，观察每一层表示如何变换。

2. 将圆分类器的隐藏层大小从 8 改为 2，然后改为 32。每次使用随机权重运行前向传播。隐藏神经元的数量会改变输出范围或分布吗？为什么？

3. 在 Network 类中实现一个 `count_parameters` 方法，返回可训练权重和偏置的总数。在 784-256-128-10 网络（经典的 MNIST 架构）上测试它。它有多少个参数？

4. 为一个 3-4-4-2 网络构建前向传播。输入 RGB 颜色值（归一化为 0-1），观察两个输出。这是用于两个类的简单颜色分类器的架构。

5. 将 sigmoid 替换为“泄漏阶梯”函数：如果 z < 0，返回 0.01 * z；否则返回 1.0。使用步骤 4 中相同的手动调整权重，在 XOR 上运行前向传播。它仍然有效吗？为什么平滑的 sigmoid 优于硬截断？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------|
| 前向传播 | “运行模型” | 将输入通过每一层传递，执行乘以权重、加上偏置、激活等操作，以产生输出 |
| 隐藏层 | “中间部分” | 输入层和输出层之间的任何层，其值在数据中无法直接观察到 |
| 多层网络 | “深度神经网络” | 神经元层依次堆叠，每一层的输出作为下一层的输入 |
| 激活函数 | “非线性” | 在线性变换后应用的函数，使决策边界出现曲线 |
| Sigmoid | “S 形曲线” | sigma(z) = 1/(1+e^(-z))，将任何实数压缩到 (0,1)，处处平滑且可微 |
| 权重矩阵 | “参数” | 形状为 (当前层神经元数, 前一层神经元数) 的矩阵，包含可学习的连接强度 |
| 偏置向量 | “偏移” | 在矩阵乘法后添加的向量，使神经元即使在所有输入为零时也能激活 |
| 通用逼近 | “神经网络可以学习任何东西” | 单个隐藏层如果有足够多的神经元，可以逼近任何连续函数——但“足够”可能意味着数十亿个 |
| 线性变换 | “矩阵乘法步骤” | z = W * x + b，激活前的计算，将输入映射到新空间 |
| 决策边界 | “分类器切换的地方” | 输入空间中网络输出跨越分类阈值的表面 |

## 进一步阅读

- Michael Nielsen，"Neural Networks and Deep Learning"，第 1-2 章（http://neuralnetworksanddeeplearning.com/）——关于前向传播和网络结构最清晰的免费解释，附有交互式可视化
- Cybenko，“Approximation by Superpositions of a Sigmoidal Function”（1989）——原始的通用逼近定理论文，出人意料地易读
- 3Blue1Brown，“But what is a neural network?”（https://www.youtube.com/watch?v=aircAruvnKk）——20 分钟的可视化讲解，构建正确的心理模型，涵盖层、权重和前向传播
- Goodfellow, Bengio, Courville，“Deep Learning”，第 6 章（https://www.deeplearningbook.org/）——多层网络的标准参考书，免费在线阅读
