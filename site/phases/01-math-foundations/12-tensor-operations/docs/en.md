# 张量运算与切片技巧

> 高维数组运算。掌握张量阶数、爱因斯坦求和（Einsum）、内存连续性与多头维度变换。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 02 (向量与矩阵运算)
**Time:** ~40 分钟

## Learning Objectives

- 从零开始实现一个张量类，包括形状（shape）、步长（strides）、reshape、transpose和逐元素操作
- 应用广播规则，在不复制数据的情况下对不同形状的张量进行操作
- 编写爱因斯坦求和（einsum）表达式，用于点积、矩阵乘法、外积和批量操作
- 追踪多头注意力中每一步的张量形状

## The Problem

你构建了一个transformer。前向传播看起来很干净。你运行它，得到：`RuntimeError: mat1 and mat2 shapes cannot be multiplied (32x768 and 512x768)`。你盯着形状看。你尝试转置。现在它说：`Expected 4D input (got 3D input)`。你添加一个unsqueeze。其他东西又出错了。

形状错误是深度学习代码中最常见的错误。它们在概念上并不难——每个操作都有一个形状契约——但它们会迅速扩散。一个transformer有数十个reshape、transpose和broadcast操作串联在一起。一个错误的轴会导致错误级联。更糟糕的是，一些形状错误根本不会抛出错误。它们通过在错误的维度上广播或对错误的轴求和，静默地生成垃圾数据。

矩阵处理两个事物集合之间的成对关系。真实的数据并不适合两个维度。32个RGB图像在224x224的批次是一个4D张量：`(32, 3, 224, 224)`。具有12个头的自注意力也是一个4D张量：`(batch, heads, seq_len, head_dim)`。你需要一种可以推广到任意数量维度的数据结构，以及可以在所有维度上清晰组合的操作。这种结构就是张量。掌握它的操作，形状错误将变得微不足道。

## The Concept

### 什么是张量

张量是一个多维数组，具有统一的数据类型。维度的数量称为**秩**（或**阶数**）。每个维度是一个**轴**。**形状**是一个元组，列出了每个轴上的大小。```mermaid
graph LR
    S["Scalar<br/>rank 0<br/>shape: ()"] --> V["Vector<br/>rank 1<br/>shape: (3,)"]
    V --> M["Matrix<br/>rank 2<br/>shape: (2,3)"]
    M --> T3["3D Tensor<br/>rank 3<br/>shape: (2,2,2)"]
    T3 --> T4["4D Tensor<br/>rank 4<br/>shape: (B,C,H,W)"]
```总元素数 = 所有尺寸的乘积。一个形状 `(2, 3, 4)` 包含 `2 * 3 * 4 = 24` 个元素。

### 深度学习中的张量形状

不同的数据类型按照惯例映射到特定的张量形状。```mermaid
graph TD
    subgraph Vision
        V1["(B, C, H, W)<br/>32, 3, 224, 224"]
    end
    subgraph NLP
        N1["(B, T, D)<br/>16, 128, 768"]
    end
    subgraph Attention
        A1["(B, H, T, D)<br/>16, 12, 128, 64"]
    end
    subgraph Weights
        W1["Linear: (out, in)<br/>Conv2D: (out_c, in_c, kH, kW)<br/>Embedding: (vocab, dim)"]
    end
```PyTorch 使用 NCHW（通道优先）。TensorFlow 默认使用 NHWC（通道最后）。不匹配的布局会导致静默的性能下降或错误。

### 内存布局是如何工作的

内存中的二维数组是一个一维的字节序列。**步长**告诉你沿着每个轴移动一步需要跳过多少个元素。```mermaid
graph LR
    subgraph "Row-major (C order)"
        R["a b c d e f<br/>strides: (3, 1)"]
    end
    subgraph "Column-major (F order)"
        C["a d b e c f<br/>strides: (1, 2)"]
    end
```转置不会移动数据。它只是交换了步长，使张量变为**非连续**的——行的元素在内存中不再相邻。

### 广播规则

广播允许你在不复制数据的情况下对不同形状的张量进行操作。从右侧对齐形状。当两个维度相等，或者其中一个为1时，这两个维度是兼容的。维度较少的张量在左侧用1进行填充。```
Tensor A:     (8, 1, 6, 1)
Tensor B:        (7, 1, 5)
Padded B:     (1, 7, 1, 5)
Result:       (8, 7, 6, 5)
```### Einsum：通用张量操作

爱因斯坦求和法用一个字母标记每个轴。输入中但输出中没有的轴会被求和。在输入和输出中都存在的轴会被保留。```mermaid
graph LR
    subgraph "matmul: ik,kj -> ij"
        A["A(I,K)"] --> |"sum over k"| C["C(I,J)"]
        B["B(K,J)"] --> |"sum over k"| C
    end
```关键模式：`i,i->`（点积），`i,j->ij`（外积），`ii->`（迹），`ij->ji`（转置），`bij,bjk->bik`（批量矩阵乘法），`bhtd,bhsd->bhts`（注意力得分）。```figure
tensor-broadcast
```## 构建它

代码位于 `code/tensors.py`。每一步都引用了那里的实现。

### 步骤 1：张量存储和步长

张量存储一个数字的扁平列表加上形状元数据。步长告诉索引逻辑如何将多维索引映射到扁平位置。```python
class Tensor:
    def __init__(self, data, shape=None):
        if isinstance(data, (list, tuple)):
            self._data, self._shape = self._flatten_nested(data)
        elif isinstance(data, np.ndarray):
            self._data = data.flatten().tolist()
            self._shape = tuple(data.shape)
        else:
            self._data = [data]
            self._shape = ()

        if shape is not None:
            total = reduce(lambda a, b: a * b, shape, 1)
            if total != len(self._data):
                raise ValueError(
                    f"Cannot reshape {len(self._data)} elements into shape {shape}"
                )
            self._shape = tuple(shape)

        self._strides = self._compute_strides(self._shape)

    @staticmethod
    def _compute_strides(shape):
        if len(shape) == 0:
            return ()
        strides = [1] * len(shape)
        for i in range(len(shape) - 2, -1, -1):
            strides[i] = strides[i + 1] * shape[i + 1]
        return tuple(strides)
```对于形状 `(3, 4)`，步长为 `(4, 1)` -- 跳过 4 个元素以前进一行，跳过 1 个元素以前进一列。

### 步骤 2：重塑、压缩、扩展

重塑会改变形状而不改变元素顺序。元素总数必须保持不变。使用 `-1` 表示一个维度以推断其大小。```python
t = Tensor(list(range(12)), shape=(2, 6))
r = t.reshape((3, 4))
r = t.reshape((-1, 3))
```Squeeze 移除大小为 1 的轴。Unsqueeze 插入一个。Unsqueezing 对于广播至关重要 -- 一个偏置向量 `(D,)` 添加到一个批次 `(B, T, D)` 需要 unsqueeze 以 `(1, 1, D)`。```python
t = Tensor(list(range(6)), shape=(1, 3, 1, 2))
s = t.squeeze()
v = Tensor([1, 2, 3])
u = v.unsqueeze(0)
```### 步骤 3：转置和排列

转置交换两个轴。排列重新排列所有轴。这是你如何在 NCHW 和 NHWC 之间进行转换的方法。```python
mat = Tensor(list(range(6)), shape=(2, 3))
tr = mat.transpose(0, 1)

t4d = Tensor(list(range(24)), shape=(1, 2, 3, 4))
perm = t4d.permute((0, 2, 3, 1))
```在转置或排列之后，张量在内存中是非连续的。在 PyTorch 中，`view` 在非连续张量上会失败——请使用 `reshape` 或者先调用 `.contiguous()`。

### 第 4 步：逐元素操作和归约操作

逐元素操作（加法、乘法、减法）独立地应用于每个元素，并保留形状。归约操作（总和、均值、最大值）会折叠一个或多个轴。```python
a = Tensor([[1, 2], [3, 4]])
b = Tensor([[10, 20], [30, 40]])
c = a + b
d = a * 2
s = a.sum(axis=0)
```卷积神经网络中的全局平均池化：`(B, C, H, W).mean(axis=[2, 3])` 产生 `(B, C)`。自然语言处理中的序列平均池化：`(B, T, D).mean(axis=1)` 产生 `(B, D)`。

### 步骤 5：使用 NumPy 进行广播

`tensors.py` 中的 `demo_broadcasting_numpy()` 函数展示了核心模式。```python
activations = np.random.randn(4, 3)
bias = np.array([0.1, 0.2, 0.3])
result = activations + bias

images = np.random.randn(2, 3, 4, 4)
scale = np.array([0.5, 1.0, 1.5]).reshape(1, 3, 1, 1)
result = images * scale

a = np.array([1, 2, 3]).reshape(-1, 1)
b = np.array([10, 20, 30, 40]).reshape(1, -1)
outer = a * b
```通过广播计算成对距离：将 `(M, 2)` 重塑为 `(M, 1, 2)`，将 `(N, 2)` 重塑为 `(1, N, 2)`，然后相减，平方，沿最后一个轴求和，再取平方根。结果：`(M, N)`。

### 第6步：Einsum 操作

`demo_einsum()` 和 `demo_einsum_gallery()` 函数会遍历所有常见的模式。```python
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
dot = np.einsum("i,i->", a, b)

A = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
B = np.array([[7, 8, 9], [10, 11, 12]], dtype=float)
matmul = np.einsum("ik,kj->ij", A, B)

batch_A = np.random.randn(4, 3, 5)
batch_B = np.random.randn(4, 5, 2)
batch_mm = np.einsum("bij,bjk->bik", batch_A, batch_B)
```收缩的计算成本是所有索引大小（保留的和求和的）的乘积。对于 `bij,bjk->bik`，其中 B=32，I=128，J=64，K=128：`32 * 128 * 64 * 128 = 33,554,432` 个乘加操作。

### 步骤 7：通过 einsum 实现注意力机制

`demo_attention_einsum()` 函数实现了端到端的多头注意力。```python
B, H, T, D = 2, 4, 8, 16
E = H * D

X = np.random.randn(B, T, E)
W_q = np.random.randn(E, E) * 0.02

Q = np.einsum("bte,ek->btk", X, W_q)
Q = Q.reshape(B, T, H, D).transpose(0, 2, 1, 3)

scores = np.einsum("bhtd,bhsd->bhts", Q, K) / np.sqrt(D)
weights = softmax(scores, axis=-1)
attn_output = np.einsum("bhts,bhsd->bhtd", weights, V)

concat = attn_output.transpose(0, 2, 1, 3).reshape(B, T, E)
output = np.einsum("bte,ek->btk", concat, W_o)
```每一步都是一个张量操作：投影（通过 einsum 的 matmul）、头拆分（reshape + transpose）、注意力得分（通过 einsum 的 batch matmul）、加权求和（通过 einsum 的 batch matmul）、头合并（transpose + reshape）、输出投影（通过 einsum 的 matmul）。

## 使用它

### 从零开始 vs NumPy

| 操作 | 从零开始（Tensor 类） | NumPy |
|---|---|---|
| 创建 | `Tensor([[1,2],[3,4]])` | `np.array([[1,2],[3,4]])` |
| 重塑 | `t.reshape((3,4))` | `a.reshape(3,4)` |
| 转置 | `t.transpose(0,1)` | `a.T` 或 `a.transpose(0,1)` |
| 压缩 | `t.squeeze(0)` | `np.squeeze(a, 0)` |
| 求和 | `t.sum(axis=0)` | `a.sum(axis=0)` |
| Einsum | 无 | `np.einsum("ij,jk->ik", a, b)` |

### 从零开始 vs PyTorch```python
import torch

t = torch.tensor([[1, 2, 3], [4, 5, 6]], dtype=torch.float32)
t.shape
t.stride()
t.is_contiguous()

t.reshape(3, 2)
t.unsqueeze(0)
t.transpose(0, 1)
t.transpose(0, 1).contiguous()

torch.einsum("ik,kj->ij", A, B)
```PyTorch 添加了 autograd、GPU 支持和优化的 BLAS 内核。形状语义是完全相同的。如果你理解了从零开始的版本，PyTorch 的形状错误将变得可读。

### 每个神经网络层都作为张量操作

| 操作 | 张量形式 | Einsum |
|---|---|---|
| 线性层 | `Y = X @ W.T + b` | `"bd,od->bo"` + bias |
| 注意力 QKV | `Q = X @ W_q` | `"btd,dh->bth"` |
| 注意力得分 | `Q @ K.T / sqrt(d)` | `"bhtd,bhsd->bhts"` |
| 注意力输出 | `softmax(scores) @ V` | `"bhts,bhsd->bhtd"` |
| 批归一化 | `(X - mu) / sigma * gamma` | 元素级 + 广播 |
| Softmax | `exp(x) / sum(exp(x))` | 元素级 + 求和 |

## 发布它

本课生成两个可重复使用的提示：

1. **`outputs/prompt-tensor-shapes.md`** -- 一个系统化的提示，用于调试张量形状不匹配。包含每个常见操作（矩阵乘法、广播、拼接、线性层、卷积层、批归一化、softmax）的决策表，以及修复查找表。

2. **`outputs/prompt-tensor-debugger.md`** -- 一个逐步调试提示，当你遇到形状错误时，将其粘贴到任何 AI 助手中。提供错误信息和张量形状，可以得到确切的修复方法。

## 练习

1. **简单 -- 重塑往返。** 取一个形状为 `(2, 3, 4)` 的张量。将其重塑为 `(6, 4)`，然后重塑为 `(24,)`，再重塑回 `(2, 3, 4)`。通过打印扁平数据来验证每一步的元素顺序是否保持不变。

2. **中等 -- 实现广播。** 扩展 `Tensor` 类，添加一个 `broadcast_to(shape)` 方法，该方法将大小为 1 的维度扩展以匹配目标形状。然后修改 `_elementwise_op`，使其在操作前自动广播。使用形状 `(3, 1)` 和 `(1, 4)` 生成 `(3, 4)` 进行测试。

3. **困难 -- 从零开始构建 einsum。** 实现一个基本的 `einsum(subscripts, *tensors)` 函数，至少处理以下操作：点积（`i,i->`）、矩阵乘法（`ij,jk->ik`）、外积（`i,j->ij`）和转置（`ij->ji`）。解析下标字符串，识别收缩的索引，并遍历所有索引组合。将你的结果与 `np.einsum` 进行比较。

4. **困难 -- 注意力形状追踪器。** 编写一个函数，以 `batch_size`、`seq_len`、`embed_dim` 和 `num_heads` 作为输入，并在多头注意力的每个步骤中打印确切的形状：输入、Q/K/V 投影、头分割、注意力得分、softmax 权重、加权和、头合并、输出投影。与 `demo_attention_einsum()` 的输出进行验证。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|---|---|---|
| 张量 | “一个矩阵但有更多维度” | 多维数组，具有统一类型、定义的形状、步长和操作 |
| 秩 | “维度的数量” | 轴的数量。矩阵有秩 2，而不是等于其矩阵秩 |
| 形状 | “张量的大小” | 沿每个轴的大小的元组。`(2, 3)` 表示 2 行、3 列 |
| 步长 | “内存是如何布局的” | 沿每个轴前进一个位置所跳过的元素数量 |
| 广播 | “当形状不同时它自动工作” | 一组严格的规则：从右对齐，维度必须相等或其中一个必须是 1 |
| 连续 | “张量是正常的” | 元素在内存中顺序存储，没有间隙或与逻辑布局的重新排序 |
| Einsum | “一种写矩阵乘法的花哨方式” | 一种通用的表示法，用一行代码表达任何张量收缩、外积、迹或转置 |
| View | “和 reshape 一样” | 与原张量共享相同内存缓冲区，但具有不同的形状/步长元数据。在非连续数据上会失败 |
| 收缩 | “对索引求和” | 张量之间共享索引相乘并求和的一般操作，生成秩更低的结果 |
| NCHW / NHWC | “PyTorch 与 TensorFlow 格式” | 图像张量的内存布局惯例。NCHW 将通道放在空间维度之前，NHWC 将它们放在之后 |

## 进一步阅读

- [NumPy 广播](https://numpy.org/doc/stable/user/basics.broadcasting.html) -- 典型规则与可视化示例
- [PyTorch 张量视图](https://pytorch.org/docs/stable/tensor_view.html) -- 视图何时工作以及何时复制
- [einops](https://github.com/arogozhnikov/einops) -- 使张量重塑可读且安全的库
- [图解 Transformer](https://jalammar.github.io/illustrated-transformer/) -- 可视化注意力中张量形状的流动
- [NumPy 中的爱因斯坦求和](https://numpy.org/doc/stable/reference/generated/numpy.einsum.html) -- 完整的 einsum 文档与示例
