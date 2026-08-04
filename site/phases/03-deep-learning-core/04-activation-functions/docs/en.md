# 激活函数

> 没有非线性，你的100层网络只是一个花哨的矩阵乘法。激活函数是让神经网络以曲线方式思考的门。

**类型:** 构建
**语言:** Python
**先决条件:** 第03.03课（反向传播）
**时间:** ~75分钟

## 学习目标

- 从零开始实现带有其导数的sigmoid、tanh、ReLU、Leaky ReLU、GELU、Swish和softmax
- 通过测量具有不同激活函数的10层以上网络中的激活幅度，诊断梯度消失问题
- 检测ReLU网络中的死神经元，并解释GELU为何避免了这种失效模式
- 为给定的架构（transformer、CNN、RNN、输出层）选择正确的激活函数

## 问题

堆叠两个线性变换：y = W2(W1x + b1) + b2。展开它：y = W2W1x + W2b1 + b2。这只是一个y = Ax + c的线性变换。无论你堆叠多少个线性层，结果都会塌缩为一个矩阵乘法。你的100层网络与单个层具有相同的表示能力。

这不是一个理论上的奇观。这意味着一个深度线性网络实际上无法学习XOR，无法对螺旋数据集进行分类，无法识别一张人脸。没有激活函数，深度只是一种幻觉。

激活函数打破了线性性。它们通过非线性函数扭曲每一层的输出，使网络具有弯曲决策边界、逼近任意函数和真正学习的能力。但选择错误的激活函数会导致你的梯度消失为零（深度网络中的sigmoid）、爆炸到无限（没有谨慎初始化的无界激活函数），或者你的神经元永久死亡（ReLU具有大的负偏差）。激活函数的选择直接决定了你的网络是否能够学习。

## 概念

### 为什么非线性是必要的

矩阵乘法是可组合的。先将一个向量乘以矩阵A，然后乘以矩阵B，等同于将该向量乘以AB。这意味着堆叠十个线性层在数学上等同于一个具有一个大矩阵的线性层。所有这些参数、所有这些深度——浪费了。你需要某个东西来打破这个链条。这就是激活函数的作用。

下面是证明。线性层计算f(x) = Wx + b。堆叠两个：```
Layer 1: h = W1 * x + b1
Layer 2: y = W2 * h + b2
```替换：```
y = W2 * (W1 * x + b1) + b2
y = (W2 * W1) * x + (W2 * b1 + b2)
y = A * x + c
```一层。在层之间插入一个非线性激活函数 g()：

```python
layer = Dense(units=64, activation='relu')
``````
h = g(W1 * x + b1)
y = W2 * h + b2
```现在替换过程被打破。W2 * g(W1 * x + b1) + b2 无法简化为一个单一的线性变换。网络可以表示非线性函数。每一层额外的激活函数都会增加表示能力。

### Sigmoid

神经网络最初的激活函数。```
sigmoid(x) = 1 / (1 + e^(-x))
```输出范围：(0, 1)。平滑、可微，将任何实数映射到类似概率的值。

导数：```
sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
```这个导数的最大值是 0.25，出现在 x = 0 处。在反向传播中，梯度会通过各层相乘。十个 sigmoid 层意味着梯度最多会被乘以 0.25 十次：```
0.25^10 = 0.000000953674
```原始信号的百万分之一以下。这就是消失梯度问题。早期层的梯度变得非常小，以至于权重几乎不更新。网络似乎在学习——后面的层损失减少——但前面的层却冻结了。使用深层的sigmoid网络根本无法进行训练。

附加问题：sigmoid的输出始终为正（0到1），这意味着权重上的梯度始终具有相同的符号。这会导致梯度下降过程中出现锯齿状波动。

### Tanh

sigmoid的中心化版本。```
tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
```输出范围：(-1, 1)。以零为中心，消除了锯齿问题。

导数：```
tanh'(x) = 1 - tanh(x)^2
```在 x = 0 处，导数最大为 1.0 -- 比 sigmoid 函数好四倍。但梯度消失问题仍然存在。对于较大的正数或负数输入，导数会趋近于零。即使有十层网络，仍然会压缩梯度，只是程度较轻。

### ReLU：突破性进展

修正线性单元。由 Nair 和 Hinton 在 2010 年推广用于深度学习（该函数本身可追溯到 Fukushima 在 1969 年的工作），它改变了所有的一切。```
relu(x) = max(0, x)
```输出范围：[0, infinity)。导数显然很简单：```
relu'(x) = 1  if x > 0
            0  if x <= 0
```正输入不会出现梯度消失。梯度正好为 1，直接传递。这就是为什么深度网络变得可以训练 -- ReLU 在各层之间保留梯度的大小。

但是存在一种失效模式：死亡神经元问题。如果一个神经元的加权输入始终为负（由于较大的负偏置或不幸的权重初始化），其输出始终为零，梯度始终为零，且不会更新。它将永久死亡。在实践中，ReLU 网络在训练过程中可能有 10-40% 的神经元死亡。

### Leaky ReLU

修复死亡神经元的最简单方法。```
leaky_relu(x) = x        if x > 0
                alpha * x if x <= 0
```其中 alpha 是一个小的常数，通常为 0.01。负数侧有一个较小的斜率而不是零，因此死神经元仍然可以获得梯度信号并恢复。

### GELU：现代默认激活函数

高斯误差线性单元。由 Hendrycks 和 Gimpel 于 2016 年引入。BERT、GPT 和大多数现代变压器模型的默认激活函数。```
gelu(x) = x * Phi(x)
```其中 Phi(x) 是标准正态分布的累积分布函数。实际中使用的近似方法：```
gelu(x) ~= 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
```GELU 在所有地方都是平滑的，允许出现小的负值（与硬截断到零的 ReLU 不同），并且具有概率解释：它根据输入在高斯分布下为正的概率来对每个输入进行加权。这种平滑的门控机制在 transformer 架构中优于 ReLU，因为它提供了更好的梯度流动，并完全避免了死神经元问题。

### Swish / SiLU

由 Ramachandran 等人在 2017 年通过自动化搜索发现的自门控激活函数。```
swish(x) = x * sigmoid(x)
```Swish 正式定义为 x * sigmoid(x)。Google 通过在激活函数空间上进行自动搜索发现了它——这是神经网络设计神经网络部分的一种方法。

与 GELU 类似，Swish 是平滑的、非单调的，并且允许小的负值。区别是微妙的：Swish 使用 sigmoid 进行门控，而 GELU 使用高斯累积分布函数。在实践中，它们的性能几乎相同。Swish 被用于 EfficientNet 和一些视觉模型中。GELU 在语言模型中占主导地位。

### Softmax：输出激活函数

不用于隐藏层。Softmax 将一个原始分数（logits）的向量转换为概率分布。```
softmax(x_i) = e^(x_i) / sum(e^(x_j) for all j)
```每个输出都在 0 到 1 之间。所有输出的总和为 1。这使得它成为多类分类的标准最终激活函数。最大的 logit 获得最高的概率，但与 argmax 不同，softmax 是可微的，并且保留了关于相对置信度的信息。

### 形状比较```mermaid
graph LR
    subgraph "Activation Functions"
        S["Sigmoid<br/>Range: (0,1)<br/>Saturates both ends"]
        T["Tanh<br/>Range: (-1,1)<br/>Zero-centered"]
        R["ReLU<br/>Range: [0,inf)<br/>Dead neurons"]
        G["GELU<br/>Range: ~(-0.17,inf)<br/>Smooth gating"]
    end
    S -->|"Vanishing gradient"| Problem["Deep networks<br/>don't train"]
    T -->|"Less severe but<br/>still vanishes"| Problem
    R -->|"Gradient = 1<br/>for x > 0"| Solution["Deep networks<br/>train fast"]
    G -->|"Smooth gradient<br/>everywhere"| Solution
```### 梯度流比较```mermaid
graph TD
    Input["Input Signal"] --> L1["Layer 1"]
    L1 --> L5["Layer 5"]
    L5 --> L10["Layer 10"]
    L10 --> Output["Output"]

    subgraph "Gradient at Layer 1"
        SigGrad["Sigmoid: ~0.000001"]
        TanhGrad["Tanh: ~0.001"]
        ReluGrad["ReLU: ~1.0"]
        GeluGrad["GELU: ~0.8"]
    end
```### 何时使用哪种激活函数```mermaid
flowchart TD
    Start["What are you building?"] --> Hidden{"Hidden layers<br/>or output?"}

    Hidden -->|"Hidden layers"| Arch{"Architecture?"}
    Hidden -->|"Output layer"| Task{"Task type?"}

    Arch -->|"Transformer / NLP"| GELU["Use GELU"]
    Arch -->|"CNN / Vision"| ReLU["Use ReLU or Swish"]
    Arch -->|"RNN / LSTM"| Tanh["Use Tanh"]
    Arch -->|"Simple MLP"| ReLU2["Use ReLU"]

    Task -->|"Binary classification"| Sigmoid["Use Sigmoid"]
    Task -->|"Multi-class classification"| Softmax["Use Softmax"]
    Task -->|"Regression"| Linear["Use Linear (no activation)"]
```

```figure
softmax-temperature
```## 构建它

### 第一步：实现所有激活函数及其导数

每个函数接受一个浮点数并返回一个浮点数。每个导数函数接受相同的输入并返回梯度。```python
import math

def sigmoid(x):
    x = max(-500, min(500, x))
    return 1.0 / (1.0 + math.exp(-x))

def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)

def tanh_act(x):
    return math.tanh(x)

def tanh_derivative(x):
    t = math.tanh(x)
    return 1 - t * t

def relu(x):
    return max(0.0, x)

def relu_derivative(x):
    return 1.0 if x > 0 else 0.0

def leaky_relu(x, alpha=0.01):
    return x if x > 0 else alpha * x

def leaky_relu_derivative(x, alpha=0.01):
    return 1.0 if x > 0 else alpha

def gelu(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))

def gelu_derivative(x):
    phi = 0.5 * (1 + math.erf(x / math.sqrt(2)))
    pdf = math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)
    return phi + x * pdf

def swish(x):
    return x * sigmoid(x)

def swish_derivative(x):
    s = sigmoid(x)
    return s + x * s * (1 - s)

def softmax(xs):
    max_x = max(xs)
    exps = [math.exp(x - max_x) for x in xs]
    total = sum(exps)
    return [e / total for e in exps]
```### 步骤 2：可视化梯度消失的位置

计算从 -5 到 5 的 100 个等距点处的梯度。打印一个文本直方图，显示每个激活函数的梯度接近零的位置。```python
def gradient_scan(name, derivative_fn, start=-5, end=5, n=100):
    step = (end - start) / n
    near_zero = 0
    healthy = 0
    for i in range(n):
        x = start + i * step
        g = derivative_fn(x)
        if abs(g) < 0.01:
            near_zero += 1
        else:
            healthy += 1
    pct_dead = near_zero / n * 100
    print(f"{name:15s}: {healthy:3d} healthy, {near_zero:3d} near-zero ({pct_dead:.0f}% dead zone)")

gradient_scan("Sigmoid", sigmoid_derivative)
gradient_scan("Tanh", tanh_derivative)
gradient_scan("ReLU", relu_derivative)
gradient_scan("Leaky ReLU", leaky_relu_derivative)
gradient_scan("GELU", gelu_derivative)
gradient_scan("Swish", swish_derivative)
```### 步骤 3：消失梯度实验

使用 sigmoid 和 ReLU 通过 N 层进行前向传播。测量激活值的幅度变化。```python
import random

def vanishing_gradient_experiment(activation_fn, name, n_layers=10, n_inputs=5):
    random.seed(42)
    values = [random.gauss(0, 1) for _ in range(n_inputs)]

    print(f"\n{name} through {n_layers} layers:")
    for layer in range(n_layers):
        weights = [random.gauss(0, 1) for _ in range(n_inputs)]
        z = sum(w * v for w, v in zip(weights, values))
        activated = activation_fn(z)
        magnitude = abs(activated)
        bar = "#" * int(magnitude * 20)
        print(f"  Layer {layer+1:2d}: magnitude = {magnitude:.6f} {bar}")
        values = [activated] * n_inputs

vanishing_gradient_experiment(sigmoid, "Sigmoid")
vanishing_gradient_experiment(relu, "ReLU")
vanishing_gradient_experiment(gelu, "GELU")
```### 步骤 4：死神经元检测器

创建一个 ReLU 网络，通过随机输入传递数据，统计有多少个神经元从未激活。```python
def dead_neuron_detector(n_inputs=5, hidden_size=20, n_samples=1000):
    random.seed(0)
    weights = [[random.gauss(0, 1) for _ in range(n_inputs)] for _ in range(hidden_size)]
    biases = [random.gauss(0, 1) for _ in range(hidden_size)]

    fire_counts = [0] * hidden_size

    for _ in range(n_samples):
        inputs = [random.gauss(0, 1) for _ in range(n_inputs)]
        for neuron_idx in range(hidden_size):
            z = sum(w * x for w, x in zip(weights[neuron_idx], inputs)) + biases[neuron_idx]
            if relu(z) > 0:
                fire_counts[neuron_idx] += 1

    dead = sum(1 for c in fire_counts if c == 0)
    rarely_fire = sum(1 for c in fire_counts if 0 < c < n_samples * 0.05)
    healthy = hidden_size - dead - rarely_fire

    print(f"\nDead Neuron Report ({hidden_size} neurons, {n_samples} samples):")
    print(f"  Dead (never fired):     {dead}")
    print(f"  Barely alive (<5%):     {rarely_fire}")
    print(f"  Healthy:                {healthy}")
    print(f"  Dead neuron rate:       {dead/hidden_size*100:.1f}%")

    for i, c in enumerate(fire_counts):
        status = "DEAD" if c == 0 else "WEAK" if c < n_samples * 0.05 else "OK"
        bar = "#" * (c * 40 // n_samples)
        print(f"  Neuron {i:2d}: {c:4d}/{n_samples} fires [{status:4s}] {bar}")

dead_neuron_detector()
```### 步骤 5：训练比较 -- Sigmoid 与 ReLU 与 GELU

在 circle 数据集（圆内点 = 类 1，圆外点 = 类 0）上训练相同的两层网络，使用三种不同的激活函数。比较收敛速度。```python
def make_circle_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        x = random.uniform(-2, 2)
        y = random.uniform(-2, 2)
        label = 1.0 if x * x + y * y < 1.5 else 0.0
        data.append(([x, y], label))
    return data


class ActivationNetwork:
    def __init__(self, activation_fn, activation_deriv, hidden_size=8, lr=0.1):
        random.seed(0)
        self.act = activation_fn
        self.act_d = activation_deriv
        self.lr = lr
        self.hidden_size = hidden_size

        self.w1 = [[random.gauss(0, 0.5) for _ in range(2)] for _ in range(hidden_size)]
        self.b1 = [0.0] * hidden_size
        self.w2 = [random.gauss(0, 0.5) for _ in range(hidden_size)]
        self.b2 = 0.0

    def forward(self, x):
        self.x = x
        self.z1 = []
        self.h = []
        for i in range(self.hidden_size):
            z = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            self.z1.append(z)
            self.h.append(self.act(z))

        self.z2 = sum(self.w2[i] * self.h[i] for i in range(self.hidden_size)) + self.b2
        self.out = sigmoid(self.z2)
        return self.out

    def backward(self, target):
        error = self.out - target
        d_out = error * self.out * (1 - self.out)

        for i in range(self.hidden_size):
            d_h = d_out * self.w2[i] * self.act_d(self.z1[i])
            self.w2[i] -= self.lr * d_out * self.h[i]
            for j in range(2):
                self.w1[i][j] -= self.lr * d_h * self.x[j]
            self.b1[i] -= self.lr * d_h
        self.b2 -= self.lr * d_out

    def train(self, data, epochs=200):
        losses = []
        for epoch in range(epochs):
            total_loss = 0
            correct = 0
            for x, y in data:
                pred = self.forward(x)
                self.backward(y)
                total_loss += (pred - y) ** 2
                if (pred >= 0.5) == (y >= 0.5):
                    correct += 1
            avg_loss = total_loss / len(data)
            accuracy = correct / len(data) * 100
            losses.append(avg_loss)
            if epoch % 50 == 0 or epoch == epochs - 1:
                print(f"    Epoch {epoch:3d}: loss={avg_loss:.4f}, accuracy={accuracy:.1f}%")
        return losses


data = make_circle_data()

configs = [
    ("Sigmoid", sigmoid, sigmoid_derivative),
    ("ReLU", relu, relu_derivative),
    ("GELU", gelu, gelu_derivative),
]

results = {}
for name, act_fn, act_d_fn in configs:
    print(f"\n=== Training with {name} ===")
    net = ActivationNetwork(act_fn, act_d_fn, hidden_size=8, lr=0.1)
    losses = net.train(data, epochs=200)
    results[name] = losses

print("\n=== Final Loss Comparison ===")
for name, losses in results.items():
    print(f"  {name:10s}: start={losses[0]:.4f} -> end={losses[-1]:.4f} (improvement: {(1 - losses[-1]/losses[0])*100:.1f}%)")
```## 使用它

PyTorch 以函数形式和模块形式同时提供了所有这些：```python
import torch
import torch.nn as nn
import torch.nn.functional as F

x = torch.randn(4, 10)

relu_out = F.relu(x)
gelu_out = F.gelu(x)
sigmoid_out = torch.sigmoid(x)
swish_out = F.silu(x)

logits = torch.randn(4, 5)
probs = F.softmax(logits, dim=1)

model = nn.Sequential(
    nn.Linear(10, 64),
    nn.GELU(),
    nn.Linear(64, 32),
    nn.GELU(),
    nn.Linear(32, 5),
)
```Transformer 中的隐藏层：GELU。CNN 中的隐藏层：ReLU。分类的输出层：softmax。回归的输出层：无（线性）。概率的输出层：sigmoid。就这些。从这些默认值开始。只有在有证据时才更改它们。

RNN 和 LSTM 使用 tanh 作为隐藏状态，使用 sigmoid 作为门控函数，但如果你今天是从零开始构建模型，可能不会使用 RNN。如果你的 ReLU 网络中神经元死亡了，换成 GELU。除非有特定原因，否则不要使用 Leaky ReLU —— GELU 解决了神经元死亡问题并提供了更好的梯度流动。

## 发布它

本课内容包括：
- `outputs/prompt-activation-selector.md` —— 一个可重复使用的提示，帮助你为任何架构选择合适的激活函数

## 练习

1. 实现 Parametric ReLU（PReLU），其中负斜率 alpha 是一个可学习的参数。在 circle 数据集上训练它，并与固定斜率的 Leaky ReLU 进行比较。

2. 使用 50 层而不是 10 层运行梯度消失实验。绘制每层中 sigmoid、tanh、ReLU 和 GELU 的梯度幅度。每个激活函数的信号在哪个层有效达到零？

3. 实现 ELU（Exponential Linear Unit）：elu(x) = x（如果 x > 0），alpha * (e^x - 1)（如果 x <= 0）。在相同网络中将其与 ReLU 的神经元死亡率进行比较。

4. 构建一个“梯度健康监测器”，在训练期间运行：在每个 epoch，计算每个层的平均梯度幅度。当任何层的梯度低于 0.001 或高于 100 时，打印警告。

5. 修改训练比较，使用第 01 课中的 XOR 数据集而不是 circles。哪种激活函数在 XOR 上收敛最快？为什么这与 circle 的结果不同？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----|
| 激活函数 | “非线性部分” | 应用于每个神经元输出的函数，打破线性，使网络能够学习非线性映射 |
| 梯度消失 | “梯度在深层网络中消失” | 当激活函数的导数小于 1 时，梯度通过各层呈指数级缩小，使得早期层无法训练 |
| 梯度爆炸 | “梯度爆炸” | 当有效乘数超过 1 时，梯度通过各层呈指数级增长，导致训练不稳定 |
| 死亡神经元 | “停止学习的神经元” | ReLU 神经元的输入永久为负，输出和梯度都为零 |
| Sigmoid | “将值压缩到 0-1” | 逻辑函数 1/(1+e^-x)，历史上重要，但在深层网络中会导致梯度消失 |
| ReLU | “将负值截断为零” | max(0, x) —— 通过保留梯度幅度使深度学习成为可能的激活函数 |
| GELU | “Transformer 的激活函数” | 高斯误差线性单元，一种平滑的激活函数，根据输入为正的概率加权输入 |
| Swish/SiLU | “自门控的 ReLU” | x * sigmoid(x)，通过自动化搜索发现，用于 EfficientNet |
| Softmax | “将得分转换为概率” | 将一个对数几率向量归一化为概率分布，所有值在 (0,1) 之间并总和为 1 |
| Leaky ReLU | “不会死亡的 ReLU” | max(alpha*x, x) ，其中 alpha 很小（0.01），通过允许小的负梯度防止死亡神经元 |
| 饱和 | “Sigmoid 的平坦部分” | 激活函数导数接近零的区域，会阻断梯度流动 |
| Logit | “Softmax 之前的原始得分” | 应用 softmax 或 sigmoid 之前的最终层的未归一化输出 |

## 进一步阅读

- Nair & Hinton, "Rectified Linear Units Improve Restricted Boltzmann Machines" (2010) —— 引入 ReLU 并使深度网络训练成为可能的论文
- Hendrycks & Gimpel, "Gaussian Error Linear Units (GELUs)" (2016) —— 引入成为 Transformer 默认激活函数的激活函数的论文
- Ramachandran 等人, "Searching for Activation Functions" (2017) —— 通过自动化搜索发现 Swish，展示了激活函数设计可以自动化的论文
- Glorot & Bengio, "Understanding the difficulty of training deep feedforward neural networks" (2010) —— 诊断梯度消失/爆炸并提出 Xavier 初始化的论文
- Goodfellow, Bengio, Courville, "Deep Learning" 第 6.3 章 (https://www.deeplearningbook.org/) —— 对隐藏单元和激活函数的严格处理
