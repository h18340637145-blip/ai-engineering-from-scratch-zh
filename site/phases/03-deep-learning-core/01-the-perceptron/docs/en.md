# 感知机

> 感知机是神经网络的原子。把它拆开，你会发现权重、偏置和一个决策。

**类型:** 构建  
**语言:** Python  
**先决条件:** 第一阶段（线性代数直觉）  
**时间:** ~60 分钟

## 学习目标

- 在 Python 中从零开始实现一个感知机，包括权重更新规则和阶跃激活函数
- 解释为什么单个感知机只能解决线性可分问题，并演示 XOR 失败案例
- 通过组合 OR、NAND 和 AND 门构建一个多层感知机以解决 XOR 问题
- 使用 sigmoid 激活函数和反向传播训练一个两层网络以自动学习 XOR

## 问题

你了解向量和点积。你了解矩阵如何将输入转换为输出。但机器是如何 *学习* 使用哪种转换的？

感知机回答了这个问题。它是最简单的学习机器：接受一些输入，乘以权重，加上偏置，然后做出二元决策。然后进行调整。仅此而已。每一个曾经构建的神经网络都是这个想法的层层堆叠。

理解感知机意味着理解代码中“学习”实际上意味着什么：调整数字，直到输出与现实相匹配。

## 概念

### 一个神经元，一个决策

感知机接受 n 个输入，将每个输入乘以一个权重，将它们相加，加上一个偏置，然后将结果传递给一个激活函数。```mermaid
graph LR
    x1["x1"] -- "w1" --> sum["Σ(wi*xi) + b"]
    x2["x2"] -- "w2" --> sum
    x3["x3"] -- "w3" --> sum
    bias["bias"] --> sum
    sum --> step["step(z)"]
    step --> out["output (0 or 1)"]
```阶跃函数是残酷的：如果加权和加上偏置 >= 0，输出 1。否则，输出 0。```
step(z) = 1  if z >= 0
           0  if z < 0
```这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think

<>

这是一个线性分类器。权重和偏置定义了一条线（或在更高维空间中的超平面），将输入空间分成两个区域。

### 决策边界

对于两个输入，感知机在二维空间中绘制一条线：

 /no_think```
  x2
  ┤
  │  Class 1        /
  │    (0)          /
  │                /
  │               / w1·x1 + w2·x2 + b = 0
  │              /
  │             /     Class 2
  │            /        (1)
  ┼───────────/──────────── x1
```线一侧的所有内容输出 0，另一侧的所有内容输出 1。训练过程会移动这条线，直到它能正确地将类别分开。

### 学习规则

感知器学习规则很简单：```
For each training example (x, y_true):
    y_pred = predict(x)
    error = y_true - y_pred

    For each weight:
        w_i = w_i + learning_rate * error * x_i
    bias = bias + learning_rate * error
```如果预测正确，误差为 0，没有任何变化。如果预测为 0 但实际应为 1，权重会增加。如果预测为 1 但实际应为 0，权重会减少。学习率控制每次调整的幅度。

### XOR 问题

这就是它失效的地方。看看这些逻辑门：```
AND gate:           OR gate:            XOR gate:
x1  x2  out         x1  x2  out         x1  x2  out
0   0   0           0   0   0           0   0   0
0   1   0           0   1   1           0   1   1
1   0   0           1   0   1           1   0   1
1   1   1           1   1   1           1   1   0
```AND 和 OR 是线性可分的：你可以画一条直线将 0 与 1 分开。XOR 不是。没有任何一条直线可以将 [0,1] 和 [1,0] 与 [0,0] 和 [1,1] 分开。```
AND (separable):        XOR (not separable):

  x2                      x2
  1 ┤  0     1            1 ┤  1     0
    │     /                 │
  0 ┤  0 / 0              0 ┤  0     1
    ┼──/──────── x1         ┼──────────── x1
       line works!          no single line works!
```这是一个根本性的限制。单个感知器只能解决线性可分问题。明斯基和帕皮特于1969年证明了这一点，这几乎让神经网络研究停滞了整整一个十年。

解决方法：将感知器堆叠成多层。多层感知器可以通过将两个线性决策组合成一个非线性决策来解决异或（XOR）问题。```figure
perceptron-boundary
```## 构建它

### 第一步：Perceptron 类```python
class Perceptron:
    def __init__(self, n_inputs, learning_rate=0.1):
        self.weights = [0.0] * n_inputs
        self.bias = 0.0
        self.lr = learning_rate

    def predict(self, inputs):
        total = sum(w * x for w, x in zip(self.weights, inputs))
        total += self.bias
        return 1 if total >= 0 else 0

    def train(self, training_data, epochs=100):
        for epoch in range(epochs):
            errors = 0
            for inputs, target in training_data:
                prediction = self.predict(inputs)
                error = target - prediction
                if error != 0:
                    errors += 1
                    for i in range(len(self.weights)):
                        self.weights[i] += self.lr * error * inputs[i]
                    self.bias += self.lr * error
            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return
        print(f"Did not converge after {epochs} epochs")
```### 步骤 2：在逻辑门上进行训练```python
and_data = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
]

or_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 1),
]

not_data = [
    ([0], 1),
    ([1], 0),
]

print("=== AND Gate ===")
p_and = Perceptron(2)
p_and.train(and_data)
for inputs, _ in and_data:
    print(f"  {inputs} -> {p_and.predict(inputs)}")

print("\n=== OR Gate ===")
p_or = Perceptron(2)
p_or.train(or_data)
for inputs, _ in or_data:
    print(f"  {inputs} -> {p_or.predict(inputs)}")

print("\n=== NOT Gate ===")
p_not = Perceptron(1)
p_not.train(not_data)
for inputs, _ in not_data:
    print(f"  {inputs} -> {p_not.predict(inputs)}")
```### 步骤 3：观察 XOR 失败```python
xor_data = [
    ([0, 0], 0),
    ([0, 1], 1),
    ([1, 0], 1),
    ([1, 1], 0),
]

print("\n=== XOR Gate (single perceptron) ===")
p_xor = Perceptron(2)
p_xor.train(xor_data, epochs=1000)
for inputs, expected in xor_data:
    result = p_xor.predict(inputs)
    status = "OK" if result == expected else "WRONG"
    print(f"  {inputs} -> {result} (expected {expected}) {status}")
```它永远无法收敛。这是单个感知器无法学习异或（XOR）的硬性证明。

### 步骤 4：用两层解决 XOR

窍门：XOR = (x1 OR x2) AND NOT (x1 AND x2)。将三个感知器组合起来：```mermaid
graph LR
    x1["x1"] --> OR["OR neuron"]
    x1 --> NAND["NAND neuron"]
    x2["x2"] --> OR
    x2 --> NAND
    OR --> AND["AND neuron"]
    NAND --> AND
    AND --> out["output"]
```

```python
def xor_network(x1, x2):
    or_neuron = Perceptron(2)
    or_neuron.weights = [1.0, 1.0]
    or_neuron.bias = -0.5

    nand_neuron = Perceptron(2)
    nand_neuron.weights = [-1.0, -1.0]
    nand_neuron.bias = 1.5

    and_neuron = Perceptron(2)
    and_neuron.weights = [1.0, 1.0]
    and_neuron.bias = -1.5

    hidden1 = or_neuron.predict([x1, x2])
    hidden2 = nand_neuron.predict([x1, x2])
    output = and_neuron.predict([hidden1, hidden2])
    return output


print("\n=== XOR Gate (multi-layer network) ===")
for inputs, expected in xor_data:
    result = xor_network(inputs[0], inputs[1])
    print(f"  {inputs} -> {result} (expected {expected})")
```所有四个案例都正确。将感知机堆叠成层可以创建出单个感知机无法产生的决策边界。

### 第5步：训练一个两层网络

第4步手动设置了权重。这在XOR问题中有效，但在实际问题中，当你事先不知道正确的权重时，这种方法就不适用了。解决方法：将阶跃函数替换为Sigmoid函数，并通过反向传播自动学习权重。```python
class TwoLayerNetwork:
    def __init__(self, learning_rate=0.5):
        import random
        random.seed(0)
        self.w_hidden = [[random.uniform(-1, 1), random.uniform(-1, 1)] for _ in range(2)]
        self.b_hidden = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.w_output = [random.uniform(-1, 1), random.uniform(-1, 1)]
        self.b_output = random.uniform(-1, 1)
        self.lr = learning_rate

    def sigmoid(self, x):
        import math
        x = max(-500, min(500, x))
        return 1.0 / (1.0 + math.exp(-x))

    def forward(self, inputs):
        self.inputs = inputs
        self.hidden_outputs = []
        for i in range(2):
            z = sum(w * x for w, x in zip(self.w_hidden[i], inputs)) + self.b_hidden[i]
            self.hidden_outputs.append(self.sigmoid(z))
        z_out = sum(w * h for w, h in zip(self.w_output, self.hidden_outputs)) + self.b_output
        self.output = self.sigmoid(z_out)
        return self.output

    def train(self, training_data, epochs=10000):
        for epoch in range(epochs):
            total_error = 0
            for inputs, target in training_data:
                output = self.forward(inputs)
                error = target - output
                total_error += error ** 2

                d_output = error * output * (1 - output)

                saved_w_output = self.w_output[:]
                hidden_deltas = []
                for i in range(2):
                    h = self.hidden_outputs[i]
                    hd = d_output * saved_w_output[i] * h * (1 - h)
                    hidden_deltas.append(hd)

                for i in range(2):
                    self.w_output[i] += self.lr * d_output * self.hidden_outputs[i]
                self.b_output += self.lr * d_output

                for i in range(2):
                    for j in range(len(inputs)):
                        self.w_hidden[i][j] += self.lr * hidden_deltas[i] * inputs[j]
                    self.b_hidden[i] += self.lr * hidden_deltas[i]
```

```python
net = TwoLayerNetwork(learning_rate=2.0)
net.train(xor_data, epochs=10000)
for inputs, expected in xor_data:
    result = net.forward(inputs)
    predicted = 1 if result >= 0.5 else 0
    print(f"  {inputs} -> {result:.4f} (rounded: {predicted}, expected {expected})")
```与第4步相比有两个关键的不同之处。首先，sigmoid函数取代了阶跃函数——它是平滑的，因此梯度存在。其次，`train`方法从输出层向隐藏层反向传播误差，根据每个权重对误差的贡献成比例地调整每个权重。这就是用20行代码实现的反向传播。

这是通向第03课的桥梁。`d_output`和`hidden_deltas`背后的数学是将链式法则应用于网络图。我们将在那里正确地推导它。

## 使用它

你刚刚从零开始构建的一切都存在于一个导入中：

```python
``````python
from sklearn.linear_model import Perceptron as SkPerceptron
import numpy as np

X = np.array([[0,0],[0,1],[1,0],[1,1]])
y = np.array([0, 0, 0, 1])

clf = SkPerceptron(max_iter=100, tol=1e-3)
clf.fit(X, y)
print([clf.predict([x])[0] for x in X])
```五行代码。你的30行`Perceptron`类做的是同样的事情。sklearn版本增加了收敛检查、多种损失函数和支持稀疏输入的功能——但核心循环是一样的：加权求和、阶跃函数、根据误差更新权重。

真正的差距在规模上显现。生产网络中会发生的变化：

- 阶跃函数变为Sigmoid、ReLU或其他平滑激活函数
- 权重通过反向传播自动学习（第03课）
- 网络层数更深：3层、10层、100+层
- 同样的原理适用：每一层都从前一层的输出中创建新的特征

单个感知器只能画出直线。将它们堆叠起来，你就能画出任何形状。

## 发布它

本课内容包括：
- `outputs/skill-perceptron.md` - 一项技能，涵盖何时需要单层架构和何时需要多层架构

## 练习

1. 在一个NAND门（通用门——任何逻辑电路都可以由NAND门构建）上训练一个感知器。验证其权重和偏置形成有效的决策边界。
2. 修改Perceptron类，以便在每个epoch中跟踪决策边界（w1*x1 + w2*x2 + b = 0）。打印在AND门训练过程中线条如何移动。
3. 构建一个3输入感知器，当且仅当三个输入中至少有两个是1时输出1（一个多数投票函数）。这是否是线性可分的？为什么？

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------|
| 感知器 | “一个假的神经元” | 一个线性分类器：输入与权重的点积加上偏置，通过一个阶跃函数 |
| 权重 | “输入的重要性” | 一个乘数，用于放大每个输入对决策的贡献 |
| 偏置 | “阈值” | 一个常数，用于移动决策边界，使感知器即使在没有输入时也能激活 |
| 激活函数 | “压缩值的工具” | 在加权求和之后应用的函数——感知器使用阶跃函数，现代网络使用Sigmoid/ReLU |
| 线性可分 | “你可以画一条线将它们分开” | 一个数据集，其中单个超平面可以完美地将类别分开 |
| XOR问题 | “感知器无法解决的问题” | 证明单层网络无法学习非线性可分函数的证据 |
| 决策边界 | “分类器切换的地方” | 将输入空间分成两个类别的超平面 w*x + b = 0 |
| 多层感知器 | “一个真正的神经网络” | 一层接一层堆叠的感知器，其中每一层的输出作为下一层的输入 |

## 进一步阅读

- Frank Rosenblatt, "The Perceptron: A Probabilistic Model for Information Storage and Organization in the Brain" (1958) —— 开启这一切的原始论文
- Minsky & Papert, "Perceptrons" (1969) —— 证明XOR问题无法由单层网络解决的书籍，使感知器研究停滞了十年
- Michael Nielsen, "Neural Networks and Deep Learning", 第一章 (http://neuralnetworksanddeeplearning.com/) —— 免费在线，对感知器如何组成网络的最佳视觉解释
