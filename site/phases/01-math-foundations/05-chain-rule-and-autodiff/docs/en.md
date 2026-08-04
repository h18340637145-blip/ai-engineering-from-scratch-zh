# 链式法则与自动微分

> 反向传播的数学引擎。掌握多元链式法则、计算图与反向模式自动微分算法。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 04 (机器学习中的微积分)
**Time:** ~50 分钟

## 学习目标

- 构建一个最小的自动梯度引擎（Value类），用于记录操作并通过反向模式自动微分计算梯度
- 使用拓扑排序实现计算图的前向和反向传递
- 仅使用从零开始的自动梯度引擎构建并训练一个用于XOR问题的多层感知机
- 通过梯度检查与数值有限差分对比验证自动微分的正确性

## 问题

你可以计算简单函数的导数。但神经网络不是一个简单的函数。它是数百个函数的组合：矩阵乘法、添加偏置、应用激活函数、再次矩阵乘法、softmax、交叉熵损失。输出是一个函数的函数的函数。

为了训练网络，你需要损失相对于每个权重的梯度。对于数以百万计的参数，手动计算这是不可能的。用数值方法（有限差分）计算则太慢。

链式法则为你提供了数学方法。自动微分为你提供了算法。它们共同使你可以在与单次前向传递成比例的时间内，计算任意函数组合的精确梯度。

这就是PyTorch、TensorFlow和JAX的工作原理。你将从零开始构建一个微型版本。

## 概念

### 链式法则

如果 `y = f(g(x))`，那么 `y` 对于 `x` 的导数是：```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```沿着链乘以导数。每条链都贡献其局部导数。

示例：`y = sin(x^2)````
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```对于更深层次的组合，链式结构会继续扩展：```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```神经网络中的每一层都是这个链条中的一个环节。

### 计算图

计算图使链式法则变得可视化。每个操作都变成一个节点。数据正向流经图中。梯度反向流动。

**前向传播（计算值）：**```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```**反向传播（计算梯度）：**```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```反向传播在每个节点应用链式法则，将梯度从输出传播到输入。

### 前向模式与反向模式

有两种方式通过图来应用链式法则。

**前向模式**从输入开始，向前推送导数。它计算 `dx/dx = 1` 并通过每个操作进行传播。当输入较少而输出较多时表现良好。```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```**反向模式**从输出开始，向后拉取梯度。它计算 `dy/dy = 1` 并按相反顺序通过每个操作进行传播。当有大量输入和少量输出时效果很好。```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```神经网络有数以百万计的输入（权重）和一个输出（损失）。反向模式在一个反向传递中计算所有梯度。这就是为什么反向传播使用反向模式的原因。

| 模式 | 种子 | 方向 | 最适合情况 |
|------|------|-----------|-------|
| 正向 | `dx_i/dx_i = 1` | 输入到输出 | 输入少，输出多 |
| 反向 | `dy/dy = 1` | 输出到输入 | 输入多，输出少（神经网络） |

### 正向模式的双重数

正向模式可以优雅地通过双重数来实现。一个双重数的形式为 `a + b*epsilon`，其中 `epsilon^2 = 0`。```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```用导数 1 对输入变量进行初始化。导数会自动通过每一个操作传播。

### 构建一个自动求导引擎

一个自动求导引擎需要三样东西：

1. **值包装。** 将每一个数字包装到一个对象中，该对象存储其值和梯度。
2. **图记录。** 每一个操作都会记录它的输入和局部梯度函数。
3. **反向传播。** 对图进行拓扑排序，然后反向遍历它，在每个节点应用链式法则。

这正是 PyTorch 的 `autograd` 所做的事情。`torch.Tensor` 类包装值，在 `requires_grad=True` 时记录操作，并在你调用 `.backward()` 时计算梯度。

### PyTorch 自动求导的内部工作原理

当你编写 PyTorch 代码时：```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```PyTorch 内部：

1. 为 `x` 创建一个带有 `requires_grad=True` 的 `Tensor` 节点
2. 每个操作（`**`, `*`, `+`）都会创建一个新节点并记录反向函数
3. `y.backward()` 通过记录的图触发反向模式自动微分
4. 每个节点的 `grad_fn` 计算局部梯度并将其传递给父节点
5. 梯度通过加法（而非替换）累积到 `.grad` 属性中

该图是动态的（通过运行定义）。每次前向传递都会构建一个新的图。这就是为什么 PyTorch 支持模型内部的控制流（if/else, loops）。```figure
chain-rule
```## 构建它

### 步骤 1：Value 类```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```每个 `Value` 存储其数值数据、其梯度（初始为零）、一个反向传播函数，以及指向生成它的子节点的指针。

### 第二步：带梯度追踪的算术运算```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```每个操作都会创建一个闭包，该闭包知道如何计算局部梯度并乘以上游梯度（`out.grad`）。`+=`处理值在多个操作中被使用的情况。

### 第三步：反向传播```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```拓扑排序确保在梯度传播到其子节点之前，每个节点的梯度已被完全计算。初始梯度为 1.0（dy/dy = 1）。

### 步骤 4：为完整引擎添加更多操作

基本的 Value 类处理加法、乘法和 ReLU。一个真正的自动微分引擎需要更多操作。以下是构建神经网络所需的操作：```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```**为什么每一步操作都很重要：**

| 操作 | 反向传播规则 | 使用场景 |
|------|------|------|
| `__sub__` | 重用加法 + 取反 | 损失计算（pred - target） |
| `__pow__` | n * x^(n-1) | 多项式激活函数，均方误差（error^2） |
| `__truediv__` | 重用乘法 + 幂运算（-1） | 归一化，学习率缩放 |
| `exp` | exp(x) * upstream | Softmax，对数似然 |
| `log` | (1/x) * upstream | 交叉熵损失，对数概率 |
| `tanh` | (1 - tanh^2) * upstream | 经典激活函数 |

聪明的部分：`__sub__` 和 `__truediv__` 是根据已有的操作定义的。它们可以免费获得正确的梯度，因为链式法则会通过底层的加法/乘法/幂运算进行组合。

### 步骤 5：从零开始构建小型 MLP

有了完整的 Value 类，你可以构建一个神经网络。不需要 PyTorch。不需要 NumPy。只需要 Values 和链式法则。```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```一个 `Neuron` 计算 `tanh(w1*x1 + w2*x2 + ... + b)`。一个 `Layer` 是一个神经元列表。一个 `MLP` 堆叠层。每个权重都是一个 `Value`，因此调用 `loss.backward()` 会将梯度传播到每个参数。

**在 XOR 上训练：**```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```这是 micrograd。一个使用纯 Python 和自动微分实现的完整神经网络训练循环。每个商业深度学习框架都在大规模上做同样的事情。

### 第 6 步：梯度检查

你如何知道你的自动微分是否正确？将其与数值导数进行比较。这就是梯度检查。```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```在复杂表达式上测试它：```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```梯度检查在实现新操作时是必不可少的。如果你的反向传播中存在错误，数值检查会发现它。每个严肃的深度学习实现都会在开发过程中运行梯度检查。

**何时使用梯度检查：**

| 情况 | 是否进行梯度检查？ |
|-----------|-------------------|
| 向你的自动微分中添加新操作 | 是，总是进行 |
| 调试无法收敛的训练循环 | 是，首先检查梯度 |
| 生产训练 | 否，太慢（每个参数需要两次前向传递） |
| 自动微分代码的单元测试 | 是，自动化进行 |

### 第7步：与手动计算进行验证```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```手动检查：`y = relu(x1*x2 + 1)`。自 `x1*x2 + 1 = 7 > 0` 起，relu 是恒等函数。
`dy/dx1 = x2 = 3`。`dy/dx2 = x1 = 2`。引擎匹配。

## 使用它

### 与 PyTorch 对比验证```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```相同的梯度。你的引擎计算出的结果与 PyTorch 相同，因为数学原理是一样的：通过链式法则进行的反向自动微分。

### 更复杂的表达式```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```## 发布它

本课将产出：
- `outputs/skill-autodiff.md` -- 用于构建和调试自动梯度系统的技能
- `code/autodiff.py` -- 一个你可以扩展的最小自动梯度引擎

在这里构建的 Value 类是第三阶段神经网络训练循环的基础。

## 练习

1. 向 Value 类添加 `__pow__`，以便你可以计算 `x ** n`。验证在 `x=2` 处的 `d/dx(x^3)` 等于 `12.0`。

2. 将 `tanh` 添加为激活函数。验证 `tanh'(0) = 1` 和 `tanh'(2) = 0.0707`（近似值）。

3. 为一个单神经元构建计算图：`y = relu(w1*x1 + w2*x2 + b)`。计算所有五个梯度并与 PyTorch 进行验证。

4. 使用双数实现前向模式自动微分。创建一个 `Dual` 类并验证它与你的反向模式引擎给出相同的导数。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 链式法则 | "将导数相乘" | 复合函数的导数等于每个函数在正确点的局部导数的乘积 |
| 计算图 | "网络图示" | 有向无环图，其中节点是操作，边在前向传播中携带值或在反向传播中携带梯度 |
| 前向模式 | "将导数前向传播" | 自动微分的一种，从输入传播到输出导数。每个输入变量一次遍历。 |
| 反向模式 | "反向传播" | 自动微分的一种，从输出传播到输入梯度。每个输出变量一次遍历。 |
| Autograd | "自动梯度" | 一个系统，记录对值的操作，构建图，并通过链式法则计算精确梯度 |
| 双数 | "值加导数" | 形如 a + b*epsilon（epsilon^2 = 0）的数，通过算术携带导数信息 |
| 拓扑排序 | "依赖顺序" | 对图节点的排序，使得每个节点在其所有依赖节点之后。正确梯度传播所需的条件。 |
| 梯度累积 | "相加而非替换" | 当一个值输入到多个操作时，其梯度是所有传入梯度贡献的总和 |
| 动态图 | "运行时定义" | 每次前向传播时重新构建的计算图，允许模型中使用 Python 控制流（PyTorch 风格） |
| 梯度检查 | "数值验证" | 将自动微分梯度与数值有限差分梯度进行比较，以验证正确性。调试时至关重要。 |
| MLP | "多层感知器" | 一个具有一个或多个隐藏层神经元的神经网络。每个神经元计算加权和加偏置，然后应用激活函数。 |
| 神经元 | "加权和加激活" | 基本单元：输出 = 激活（w1*x1 + w2*x2 + ... + b）。权重和偏置是可学习参数。 |

## 进一步阅读

- [3Blue1Brown：反向传播微积分](https://www.youtube.com/watch?v=tIeHLnjs5U8) -- 神经网络中链式法则的视觉解释
- [PyTorch Autograd 机制](https://pytorch.org/docs/stable/notes/autograd.html) -- 实际系统的运行方式
- [Baydin 等，机器学习中的自动微分综述](https://arxiv.org/abs/1502.05767) -- 全面参考资料
