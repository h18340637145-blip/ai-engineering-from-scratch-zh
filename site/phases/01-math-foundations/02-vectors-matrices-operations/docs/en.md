# 向量与矩阵运算

> 从点积到矩阵乘法。掌握机器学习中大规模张量计算的基础代数工具。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 1, Lesson 01 (线性代数直觉)
**Time:** ~45 分钟

## Learning Objectives

- 构建一个矩阵类，包含逐元素运算、矩阵乘法、转置、行列式和逆矩阵
- 区分逐元素乘法与矩阵乘法，并解释每种情况适用的场景
- 仅使用从零开始实现的矩阵类来实现一个密集的神经网络层 (`relu(W @ x + b)`)
- 解释广播规则以及在神经网络框架中偏置加法的工作原理

## The Problem

你想构建一个神经网络。你阅读代码并看到如下内容：```
output = activation(weights @ input + bias)
````@` 是矩阵乘法。`weights` 是矩阵。`input` 是向量。如果你不知道这些操作的作用，这一行就是魔法。如果你知道，它就是用三次操作完成一个层的整个前向传播过程。

你模型处理的每一张图像都是像素值的矩阵。每个词嵌入都是一个向量。每个神经网络的每一层都是一个矩阵变换。没有对矩阵操作的熟练掌握，你无法构建人工智能系统，就像不了解变量就无法编写代码一样。

本课将从零开始培养这种熟练度。

## 概念

### 向量：有序的数字列表

向量是具有方向和大小的数字列表。在人工智能中，向量表示数据点、特征或参数。```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```一个二维向量 `[3, 4]` 指向平面上的坐标 (3, 4)。它的长度（模）是 5（3-4-5 三角形）。

### 矩阵：数字的网格

矩阵是一个二维网格。行和列。一个 m x n 矩阵有 m 行和 n 列。```
A = | 1  2  3 |     -- 2x3 matrix (2 rows, 3 columns)
    | 4  5  6 |
```在神经网络中，权重矩阵将输入向量转换为输出向量。一个有784个输入和128个输出的层使用一个128x784的权重矩阵。

### 为什么形状很重要

矩阵乘法有一个严格的规则：`(m x n) @ (n x p) = (m x p)`。内部维度必须匹配。```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```如果你在 PyTorch 中遇到形状不匹配的错误，原因就在这里。

### 操作映射

| 操作 | 功能 | 神经网络中的用途 |
|------|-----|--------|
| 加法 | 按元素组合 | 给输出添加偏置 |
| 标量乘法 | 缩放每个元素 | 学习率 * 梯度 |
| 矩阵乘法 | 转换向量 | 层的前向传播 |
| 转置 | 交换行和列 | 反向传播 |
| 行列式 | 单个数字的摘要 | 检查可逆性 |
| 逆矩阵 | 撤销一个变换 | 解线性系统 |
| 单位矩阵 | 什么都不做的矩阵 | 初始化，残差连接 |

### 按元素乘法与矩阵乘法

这个区别常常让初学者感到困惑。

按元素乘法：相匹配的位置相乘。两个矩阵的形状必须相同。```
| 1  2 |   | 5  6 |   | 5  12 |
| 3  4 | * | 7  8 | = | 21 32 |
```矩阵乘法：行和列的点积。内维必须匹配。```
| 1  2 |   | 5  6 |   | 1*5+2*7  1*6+2*8 |   | 19  22 |
| 3  4 | @ | 7  8 | = | 3*5+4*7  3*6+4*8 | = | 43  50 |
```不同的操作，不同的结果，不同的规则。

### 广播

当你将一个偏置向量添加到一个输出矩阵时，它们的形状不匹配。广播会将较小的数组扩展以适应较大的数组。```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```每个现代框架都会自动完成这一点。理解它可以在形状看起来不对但代码却能运行时避免混淆。```figure
vector-projection
```## 构建它

### 第一步：向量类```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```### 步骤 2：具有核心操作的矩阵类```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```### 步骤 3：看到它运行```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```### 步骤 4：连接到神经网络```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```这是一个单一的密集层：`output = relu(W @ x + b)`。每个神经网络中的每个密集层都确切地执行这个操作。

## 使用它

NumPy 用更少的代码行数，以更高的速度完成以上所有操作。```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```Python 中的 `@` 运算符调用 `__matmul__`。NumPy 使用用 C 和 Fortran 编写的优化 BLAS 程序实现它。同样的数学运算，速度快 100 倍。

NumPy 中的广播：```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```NumPy 自动将 1D 偏置广播到所有行。这是所有神经网络框架中偏置加法的工作方式。

## 发布它

本课产生一个通过几何直觉教学矩阵运算的提示。详见 `outputs/prompt-matrix-operations.md`。

此处构建的 Matrix 类是第 3 阶段第 10 课中我们构建的微型神经网络框架的基础。

## 练习

1. **验证逆矩阵。** 将 `A @ A.inverse_2x2()` 相乘并确认得到单位矩阵。用三个不同的 2x2 矩阵尝试。当行列式为零时会发生什么？

2. **实现 3x3 逆矩阵。** 扩展 Matrix 类，使用伴随矩阵法计算 3x3 矩阵的逆矩阵。与 NumPy 的 `np.linalg.inv` 进行测试。

3. **构建一个两层网络。** 仅使用你的 Matrix 类（不使用 NumPy），创建一个两层神经网络：输入（3）→ 隐藏层（4）→ 输出（2）。初始化随机权重，运行前向传播，并验证所有形状是否正确。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|------------------|
| 向量 | “一个箭头” | 一组有序的数字。在 AI 中：高维空间中的一个点。 |
| 矩阵 | “一个数字表格” | 一个线性变换。它将一个空间中的向量映射到另一个空间中。 |
| 矩阵乘法 | “只是将数字相乘” | 第一个矩阵的每一行与第二个矩阵的每一列之间的点积。顺序非常重要。 |
| 转置 | “翻转它” | 交换行和列。将一个 m x n 矩阵变为 n x m。在反向传播中非常重要。 |
| 行列式 | “矩阵中的某个数字” | 衡量矩阵对面积（二维）或体积（三维）的缩放程度。零意味着变换压缩了一个维度。 |
| 逆矩阵 | “撤销矩阵” | 能够逆转变换的矩阵。只有当行列式不为零时才存在。 |
| 单位矩阵 | “无聊的矩阵” | 相当于乘以 1 的矩阵。用于残差连接（ResNets）。 |
| 广播 | “神奇的形状修复” | 通过沿缺失维度重复较小的数组来扩展它，以匹配较大的数组。 |
| 元素级运算 | “普通的乘法” | 相同位置的元素相乘。两个数组的形状必须相同（或可广播）。 |

## 进一步阅读

- [3Blue1Brown: 线性代数的本质](https://www.3blue1brown.com/topics/linear-algebra) - 本节所涉及所有操作的视觉直觉
- [NumPy 广播文档](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的确切规则
- [斯坦福 CS229 线性代数复习](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 机器学习相关线性代数的简洁参考
