# 线性方程组系统与矩阵分解

> 求解 A x = b。掌握高斯消元、LU 分解、Cholesky 分解与正规方程。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 02 (向量与矩阵运算)
**Time:** ~45 分钟

## 学习目标

- 使用带部分选主元的高斯消元和回代法求解 Ax = b
- 使用 LU、QR 和 Cholesky 分解对矩阵进行分解，并解释每种方法适用的场景
- 推导最小二乘的正规方程，并将其与线性回归和岭回归联系起来
- 使用条件数诊断病态系统，并通过正则化稳定它们

## 问题

每次你训练线性回归模型时，你都在求解一个线性系统。每次你计算最小二乘拟合时，你都在求解一个线性系统。每次神经网络层计算 `y = Wx + b` 时，它都在评估线性系统的一边。当你添加正则化时，你修改了这个系统。当你使用高斯过程时，你对矩阵进行分解。当你为马氏距离反转协方差矩阵时，你求解一个线性系统。

方程 Ax = b 出现在各个地方。A 是已知系数的矩阵。b 是已知输出的向量。x 是你想找到的未知变量向量。在线性回归中，A 是你的数据矩阵，b 是你的目标向量，x 是权重向量。整个模型简化为：找到 x，使得 Ax 尽可能接近 b。

本课将从头开始构建求解该方程的主要方法。你将了解为什么一些方法速度快而另一些方法稳定，为什么一些方法只适用于方阵而另一些方法可以处理超定系统，以及为什么你的矩阵的条件数决定了你的答案是否有任何意义。

## 概念

### Ax = b 几何上的含义

线性方程组具有几何解释。每个方程定义一个超平面。解是所有超平面相交的点（或点集）。```
2x + y = 5          Two lines in 2D.
x - y  = 1          They intersect at x=2, y=1.
```

```mermaid
graph LR
    A["2x + y = 5"] --- S["Solution: (2, 1)"]
    B["x - y = 1"] --- S
```可能发生三件事：```mermaid
graph TD
    subgraph "One Solution"
        A1["Lines intersect at a single point"]
    end
    subgraph "No Solution"
        A2["Lines are parallel — no intersection"]
    end
    subgraph "Infinite Solutions"
        A3["Lines are identical — every point is a solution"]
    end
```用矩阵形式表示，“唯一解”意味着矩阵 A 是可逆的。“无解”意味着该系统是不一致的。“无限解”意味着矩阵 A 有零空间。大多数机器学习问题都属于“没有精确解”的类别，因为通常你拥有的方程（数据点）比未知数（参数）更多。这就是最小二乘法派上用场的地方。

### 列图像与行图像

有两种方式来理解 Ax = b。

**行图像。** 矩阵 A 的每一行定义一个方程。每个方程对应一个超平面。解是所有这些超平面的交点。

**列图像。** 矩阵 A 的每一列都是一个向量。问题就变成了：A 的列向量的什么线性组合可以产生向量 b？```
A = | 2  1 |    b = | 5 |
    | 1 -1 |        | 1 |

Row picture: solve 2x + y = 5 and x - y = 1 simultaneously.

Column picture: find x1, x2 such that:
  x1 * [2, 1] + x2 * [1, -1] = [5, 1]
  2 * [2, 1] + 1 * [1, -1] = [4+1, 2-1] = [5, 1]   check.
```列图像更为基本。如果 b 位于 A 的列空间中，方程组有解。如果不在，你将找到列空间中距离 b 最近的点。这个最近的点就是最小二乘解。

### 高斯消元法

高斯消元法将 Ax = b 转化为一个上三角系统 Ux = c，然后通过回代求解。这是最直接的方法。

算法：```
1. For each column k (the pivot column):
   a. Find the largest entry in column k at or below row k (partial pivoting).
   b. Swap that row with row k.
   c. For each row i below k:
      - Compute multiplier m = A[i][k] / A[k][k]
      - Subtract m times row k from row i.
2. Back substitute: solve from the last equation upward.
```将以下 Markdown 文本完整翻译为简体中文。保留 Markdown 标记、段落、列表、标题层级、占位符、变量名和专有技术名词；不要省略任何标题，不要输出解释，不要输出思考过程，只输出译文。


示例:```
Original:
| 2  1  1 | 8 |       R2 = R2 - (2)R1     | 2  1   1 |  8 |
| 4  3  3 |20 |  -->  R3 = R3 - (1)R1 --> | 0  1   1 |  4 |
| 2  3  1 |12 |                            | 0  2   0 |  4 |

                       R3 = R3 - (2)R2     | 2  1   1 |  8 |
                                       --> | 0  1   1 |  4 |
                                           | 0  0  -2 | -4 |

Back substitute:
  -2 * x3 = -4    -->  x3 = 2
  x2 + 2  = 4     -->  x2 = 2
  2*x1 + 2 + 2 = 8 --> x1 = 2
```高斯消元法需要 $O(n^3)$ 次操作。对于一个 1000x1000 的系统，大约需要进行十亿次浮点运算。虽然很快，但如果你需要求解多个具有相同矩阵 $A$ 的系统，可以做得更好。

### 部分选主元：为什么它很重要

如果不进行选主元，高斯消元法可能会失败或产生错误的结果。如果主元元素为零，就会出现除以零的情况。如果主元元素很小，就会放大舍入误差。```
Bad pivot:                       With partial pivoting:
| 0.001  1 | 1.001 |            Swap rows first:
| 1      1 | 2     |            | 1      1 | 2     |
                                 | 0.001  1 | 1.001 |
m = 1/0.001 = 1000              m = 0.001/1 = 0.001
R2 = R2 - 1000*R1               R2 = R2 - 0.001*R1
| 0.001  1     | 1.001   |      | 1      1     | 2     |
| 0     -999   | -999.0  |      | 0      0.999 | 0.999 |

x2 = 1.000 (correct)            x2 = 1.000 (correct)
x1 = (1.001 - 1)/0.001          x1 = (2 - 1)/1 = 1.000 (correct)
   = 0.001/0.001 = 1.000        Stable because the multiplier is small.
```在有限精度的浮点运算中，未进行选主元的版本可能会丢失有效数字。部分选主元总是选择可用的主元中最大的一个，以最小化误差放大。

### LU 分解

LU 分解将矩阵 A 分解为一个下三角矩阵 L 和一个上三角矩阵 U：A = LU。L 矩阵存储了高斯消元法中的乘数。U 矩阵是消元后的结果。```
A = L @ U

| 2  1  1 |   | 1  0  0 |   | 2  1   1 |
| 4  3  3 | = | 2  1  0 | @ | 0  1   1 |
| 2  3  1 |   | 1  2  1 |   | 0  0  -2 |
```为什么使用分解而不是直接消元？因为一旦你得到了 L 和 U，对于任何新的 b 解方程 Ax = b 的成本只需 O(n²)：```
Ax = b
LUx = b
Let y = Ux:
  Ly = b    (forward substitution, O(n^2))
  Ux = y    (back substitution, O(n^2))
```O(n³) 的成本在因式分解时只支付一次。之后每次求解的代价都是 O(n²)。如果你需要求解 1000 个具有相同 A 但不同 b 向量的系统，LU 分解在总工作量上可以节省 1000/3 的因素。

使用部分选主元时，你可以得到 PA = LU，其中 P 是一个记录行交换的排列矩阵。

### QR 分解

QR 分解将矩阵 A 分解为一个正交矩阵 Q 和一个上三角矩阵 R：A = QR。

正交矩阵具有 QᵀQ = I 的性质。它的列是正交单位向量。乘以 Q 会保持长度和角度不变。```
A = Q @ R

Q has orthonormal columns: Q^T Q = I
R is upper triangular

To solve Ax = b:
  QRx = b
  Rx = Q^T b    (just multiply by Q^T, no inversion needed)
  Back substitute to get x.
```QR 算法在求解最小二乘问题时比 LU 分解数值上更稳定。Gram-Schmidt 过程按列构建 Q：```
Given columns a1, a2, ... of A:

q1 = a1 / ||a1||

q2 = a2 - (a2 . q1) * q1        (subtract projection onto q1)
q2 = q2 / ||q2||                (normalize)

q3 = a3 - (a3 . q1) * q1 - (a3 . q2) * q2
q3 = q3 / ||q3||

R[i][j] = qi . aj    for i <= j
```每一步都会沿着所有先前的 q 向量方向移除该分量，只留下新的正交方向。

### 乔列斯基分解

当 A 是对称矩阵（A = A^T）且正定（所有特征值为正）时，可以将其分解为 A = L L^T 的形式，其中 L 是下三角矩阵。这就是乔列斯基分解。```
A = L @ L^T

| 4  2 |   | 2  0 |   | 2  1 |
| 2  5 | = | 1  2 | @ | 0  2 |

L[i][i] = sqrt(A[i][i] - sum(L[i][k]^2 for k < i))
L[i][j] = (A[i][j] - sum(L[i][k]*L[j][k] for k < j)) / L[j][j]    for i > j
```Cholesky 的速度是 LU 的两倍，所需存储空间仅为 LU 的一半。它仅适用于对称正定矩阵，但这类矩阵在实际中经常出现：

- 协方差矩阵是对称半正定的（通过正则化后变为正定）。
- 高斯过程中核矩阵是对称正定的。
- 凸函数在极小值处的海森矩阵是对称正定的。
- A^T A 总是对称半正定的。

在高斯过程中，你使用 Cholesky 分解核矩阵 K，然后求解 K alpha = y 得到预测均值。Cholesky 分解因子还能提供边缘似然的对数行列式：log det(K) = 2 * sum(log(diag(L)))。

### 最小二乘法：当 Ax = b 没有精确解时

如果 A 是 m x n 的矩阵，且 m > n（方程数量多于未知数数量），那么系统是超定的。此时没有精确解。取而代之的是，你最小化平方误差：```
minimize ||Ax - b||^2

This is the sum of squared residuals:
  sum((A[i,:] @ x - b[i])^2 for i in range(m))
```最小化器满足正规方程：```
A^T A x = A^T b
```推导：展开 ||Ax - b||² = (Ax - b)^T (Ax - b) = x^T A^T A x - 2 x^T A^T b + b^T b。对 x 求梯度，设为零：2 A^T A x - 2 A^T b = 0.```
Original system (overdetermined, 4 equations, 2 unknowns):
| 1  1 |         | 3 |
| 1  2 | x     = | 5 |       No exact x satisfies all 4 equations.
| 1  3 |         | 6 |
| 1  4 |         | 8 |

Normal equations:
A^T A = | 4  10 |    A^T b = | 22 |
        | 10 30 |            | 63 |

Solve: x = [1.5, 1.7]

This is linear regression. x[0] is the intercept, x[1] is the slope.
```### 正则方程 = 线性回归

这种联系是精确的。在线性回归中，你的数据矩阵 X 每个样本对应一行，每个特征对应一列。你的目标向量 y 每个样本对应一个条目。权重向量 w 满足：```
X^T X w = X^T y
w = (X^T X)^(-1) X^T y
```这是线性回归的闭合形式解。每次调用 `sklearn.linear_model.LinearRegression.fit()` 都会计算这个（或通过 QR 或 SVD 等价的形式）。

向矩阵中添加一个正则化项 lambda * I，你就会得到岭回归：```
(X^T X + lambda * I) w = X^T y
w = (X^T X + lambda * I)^(-1) X^T y
```正则化使矩阵的条件更好（更容易准确求逆），并通过将权重向零收缩来防止过拟合。当 lambda > 0 时，矩阵 X^T X + lambda * I 始终是对称正定的，因此你可以使用 Cholesky 分解来求解它。

### 伪逆（Moore-Penrose）

伪逆 A+ 将矩阵求逆推广到非方阵和奇异矩阵。对于任何矩阵 A：```
x = A+ b

where A+ = V Sigma+ U^T    (computed via SVD)
```Sigma+ 是通过对每个非零奇异值取倒数，并将结果转置而形成的。如果 A = U Sigma V^T，那么 A+ = V Sigma+ U^T。```
A = U Sigma V^T        (SVD)

Sigma = | 5  0 |       Sigma+ = | 1/5  0  0 |
        | 0  2 |                | 0  1/2  0 |
        | 0  0 |

A+ = V Sigma+ U^T
```伪逆给出了最小范数的最小二乘解。如果系统有：
- 一个解：A+ b 给出这个解。
- 没有解：A+ b 给出最小二乘解。
- 无限多解：A+ b 给出其中范数 ||x|| 最小的那个。

NumPy 的 `np.linalg.lstsq` 和 `np.linalg.pinv` 都在内部使用 SVD。

### 条件数

条件数衡量了解对输入微小变化的敏感程度。对于矩阵 A，条件数是：```
kappa(A) = ||A|| * ||A^(-1)|| = sigma_max / sigma_min
```其中 sigma_max 和 sigma_min 分别是最大和最小的奇异值。```
Well-conditioned (kappa ~ 1):        Ill-conditioned (kappa ~ 10^15):
Small change in b -->                Small change in b -->
small change in x                    huge change in x

| 2  0 |   kappa = 2/1 = 2          | 1   1          |   kappa ~ 10^15
| 0  1 |   safe to solve            | 1   1+10^(-15) |   solution is garbage
```经验法则：
- kappa < 100：安全，解是准确的。
- kappa ~ 10^k：你的浮点运算会损失大约k位精度。
- kappa ~ 10^16（对于float64）：解是没有意义的。矩阵实际上可以视为奇异矩阵。

在机器学习中，当特征几乎共线时会发生病态条件。正则化（添加lambda * I）将条件数从sigma_max / sigma_min 改善为 (sigma_max + lambda) / (sigma_min + lambda)。

### 迭代方法：共轭梯度法

对于非常大的稀疏系统（数百万个未知数），像LU或Cholesky这样的直接方法成本太高。迭代方法通过在许多迭代中改进猜测来近似解。

共轭梯度法（CG）用于求解对称正定矩阵A的Ax = b问题。在精确算术中，它最多需要n次迭代找到精确解，但如果A的特征值聚集在一起，通常会收敛得更快。```
Algorithm sketch:
  x0 = initial guess (often zero)
  r0 = b - A x0           (residual)
  p0 = r0                 (search direction)

  For k = 0, 1, 2, ...:
    alpha = (rk . rk) / (pk . A pk)
    x_{k+1} = xk + alpha * pk
    r_{k+1} = rk - alpha * A pk
    beta = (r_{k+1} . r_{k+1}) / (rk . rk)
    p_{k+1} = r_{k+1} + beta * pk
    if ||r_{k+1}|| < tolerance: stop
```CG 用于以下方面：
- 大规模优化（牛顿-CG 方法）
- 求解偏微分方程离散化
- 核方法中核矩阵太大无法分解的情况
- 为其他迭代求解器进行预处理

收敛速度取决于条件数。条件更好的系统收敛更快，这也是正则化有所帮助的另一个原因。

### 全貌：何时使用哪种方法

| 方法 | 要求 | 成本 | 使用场景 |
|-----|-----|-----|---------|
| 高斯消元 | 方阵，非奇异 A | O(n^3) | 单次求解方阵系统 |
| LU 分解 | 方阵，非奇异 A | O(n^3) 分解 + O(n^2) 求解 | 多次求解相同的 A |
| QR 分解 | 任意 A（m >= n） | O(mn^2) | 最小二乘，数值稳定 |
| 乔列斯基分解 | 对称正定 A | O(n^3/3) | 协方差矩阵，高斯过程，岭回归 |
| 正则方程 | 过定（m > n） | O(mn^2 + n^3) | 线性回归（小 n） |
| SVD / 伪逆 | 任意 A | O(mn^2) | 秩亏系统，最小范数解 |
| 共轭梯度法 | 对称正定，稀疏 A | O(n * k * nnz) | 大型稀疏系统，k = 迭代次数 |

### 与机器学习的联系

本课中的每种方法都在生产机器学习中出现：

**线性回归。** 闭式解用于求解正规方程 X^T X w = X^T y。这可以通过乔列斯基分解（如果 n 很小）或 QR 分解（如果数值稳定性很重要）或 SVD（如果矩阵可能存在秩亏）来完成。

**岭回归。** 向 X^T X 添加 lambda * I。正则化系统 (X^T X + lambda * I) w = X^T y 总是可以通过乔列斯基分解求解，因为当 lambda > 0 时，X^T X + lambda * I 是对称正定的。

**高斯过程。** 预测均值需要求解 K alpha = y，其中 K 是核矩阵。K 的乔列斯基分解是标准方法。对数边缘似然使用 log det(K) = 2 sum(log(diag(L)))。

**神经网络初始化。** 正交初始化使用 QR 分解来创建列正交的权重矩阵。这可以防止深度网络中的信号崩溃。

**预处理。** 大规模优化器使用不完全乔列斯基分解或不完全 LU 分解作为共轭梯度求解器的预处理子。

**特征工程。** X^T X 的条件数告诉你特征是否共线。如果 kappa 很大，丢弃特征或添加正则化。```figure
linear-system-conditioning
```## 构建它

### 步骤 1：带部分选主元的高斯消去法```python
import numpy as np

def gaussian_elimination(A, b):
    n = len(b)
    Ab = np.hstack([A.astype(float), b.reshape(-1, 1).astype(float)])

    for k in range(n):
        max_row = k + np.argmax(np.abs(Ab[k:, k]))
        Ab[[k, max_row]] = Ab[[max_row, k]]

        if abs(Ab[k, k]) < 1e-12:
            raise ValueError(f"Matrix is singular or nearly singular at pivot {k}")

        for i in range(k + 1, n):
            m = Ab[i, k] / Ab[k, k]
            Ab[i, k:] -= m * Ab[k, k:]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (Ab[i, -1] - Ab[i, i+1:n] @ x[i+1:n]) / Ab[i, i]

    return x
```### 步骤 2：LU 分解```python
def lu_decompose(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.astype(float).copy()
    P = np.eye(n)

    for k in range(n):
        max_row = k + np.argmax(np.abs(U[k:, k]))
        if max_row != k:
            U[[k, max_row]] = U[[max_row, k]]
            P[[k, max_row]] = P[[max_row, k]]
            if k > 0:
                L[[k, max_row], :k] = L[[max_row, k], :k]

        for i in range(k + 1, n):
            L[i, k] = U[i, k] / U[k, k]
            U[i, k:] -= L[i, k] * U[k, k:]

    return P, L, U

def lu_solve(P, L, U, b):
    n = len(b)
    Pb = P @ b.astype(float)

    y = np.zeros(n)
    for i in range(n):
        y[i] = Pb[i] - L[i, :i] @ y[:i]

    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - U[i, i+1:] @ x[i+1:]) / U[i, i]

    return x
```### 步骤 3：Cholesky 分解```python
def cholesky(A):
    n = A.shape[0]
    L = np.zeros_like(A, dtype=float)

    for i in range(n):
        for j in range(i + 1):
            s = A[i, j] - L[i, :j] @ L[j, :j]
            if i == j:
                if s <= 0:
                    raise ValueError("Matrix is not positive definite")
                L[i, j] = np.sqrt(s)
            else:
                L[i, j] = s / L[j, j]

    return L
```### 步骤 4：通过正规方程进行最小二乘法```python
def least_squares_normal(A, b):
    AtA = A.T @ A
    Atb = A.T @ b
    return gaussian_elimination(AtA, Atb)

def ridge_regression(A, b, lam):
    n = A.shape[1]
    AtA = A.T @ A + lam * np.eye(n)
    Atb = A.T @ b
    L = cholesky(AtA)
    y = np.zeros(n)
    for i in range(n):
        y[i] = (Atb[i] - L[i, :i] @ y[:i]) / L[i, i]
    x = np.zeros(n)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - L.T[i, i+1:] @ x[i+1:]) / L.T[i, i]
    return x
```### 步骤 5：条件数```python
def condition_number(A):
    U, S, Vt = np.linalg.svd(A)
    return S[0] / S[-1]
```## 使用它

将各部分组合起来，在真实数据上进行线性回归和岭回归：```python
np.random.seed(42)
X_raw = np.random.randn(100, 3)
w_true = np.array([2.0, -1.0, 0.5])
y = X_raw @ w_true + np.random.randn(100) * 0.1

X = np.column_stack([np.ones(100), X_raw])

w_ols = least_squares_normal(X, y)
print(f"OLS weights (ours):    {w_ols}")

w_np = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"OLS weights (numpy):   {w_np}")
print(f"Max difference: {np.max(np.abs(w_ols - w_np)):.2e}")

w_ridge = ridge_regression(X, y, lam=1.0)
print(f"Ridge weights (ours):  {w_ridge}")

from sklearn.linear_model import Ridge
ridge_sk = Ridge(alpha=1.0, fit_intercept=False)
ridge_sk.fit(X, y)
print(f"Ridge weights (sklearn): {ridge_sk.coef_}")
```## 发布它

本课将产出：
- `code/linear_systems.py` 包含从零开始实现的高斯消元法、LU分解、Cholesky分解、最小二乘法和岭回归
- 一个演示，展示正规方程和 sklearn 的 LinearRegression 产生相同的权重

## 练习

1. 使用你的高斯消元法、你的 LU 求解器和 `np.linalg.solve` 解方程组 `[[1,2,3],[4,5,6],[7,8,10]] x = [6, 15, 27]`。验证这三个方法在浮点精度范围内给出相同的结果。

2. 生成一个 50x5 的随机矩阵 X 和目标 y = X @ w_true + 噪声。使用正规方程、QR（通过 `np.linalg.qr`）、SVD（通过 `np.linalg.svd`）和 `np.linalg.lstsq` 解出 w。比较这四个解。测量 X^T X 的条件数并解释它如何影响你对哪种方法的信任程度。

3. 创建一个几乎奇异的矩阵，使两列几乎相同（例如，第 2 列 = 第 1 列 + 1e-10 * 噪声）。计算它的条件数。在有和没有正则化（添加 0.01 * I）的情况下解 Ax = b。比较解和残差。解释为什么正则化有帮助。

4. 为一个 100x100 的随机对称正定矩阵实现共轭梯度算法。计算它收敛到 1e-8 精度所需的迭代次数。与理论最大迭代次数 n 进行比较。

5. 在大小为 10、50、200、500 的对称正定矩阵上，将你的 Cholesky 求解器与你的 LU 求解器与 `np.linalg.solve` 进行计时。绘制结果。验证 Cholesky 比 LU 快约两倍。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|----------------------|
| 线性系统 | “求解 x” | 一组线性方程 Ax = b。找到 x 意味着找到在变换 A 下产生输出 b 的输入。 |
| 高斯消元法 | “行简化” | 使用行操作系统地将对角线以下的元素归零，生成一个可通过回代求解的上三角系统。复杂度为 O(n^3)。 |
| 部分选主元 | “交换行以保证稳定性” | 在对第 k 列进行消元之前，将该列中绝对值最大的行交换到主元位置。防止除以小数。 |
| LU 分解 | “分解为三角形” | 将 A 写成 LU，其中 L 是下三角矩阵（存储乘数），U 是上三角矩阵（消元后的矩阵）。将 O(n^3) 的成本分摊到多次求解中。 |
| QR 分解 | “正交分解” | 将 A 写成 QR，其中 Q 的列是正交的，R 是上三角矩阵。比 LU 更适合最小二乘问题。 |
| Cholesky 分解 | “矩阵的平方根” | 对于对称正定矩阵 A，写成 A = LL^T。成本是 LU 的一半。用于协方差矩阵、核矩阵和岭回归。 |
| 最小二乘法 | “当无法精确求解时的最佳拟合” | 当系统是超定（方程数多于未知数）时，最小化残差的平方和 ||Ax - b||^2。 |
| 正规方程 | “微积分的捷径” | A^T A x = A^T b。将 ||Ax - b||^2 的梯度设为零。这确实是线性回归的闭合解。 |
| 伪逆 | “非方阵的逆” | A+ = V Sigma+ U^T 通过 SVD。为任何矩阵（方阵或非方阵，奇异或非奇异）提供最小范数的最小二乘解。 |
| 条件数 | “这个答案有多可靠” | kappa = sigma_max / sigma_min。衡量对输入扰动的敏感性。精度会损失约 log10(kappa) 位。 |
| 岭回归 | “正则化的最小二乘法” | 解 (X^T X + lambda I) w = X^T y。添加 lambda I 改善条件并使权重向零收缩。防止过拟合。 |
| 共轭梯度法 | “用于大型矩阵的迭代 Ax = b” | 用于对称正定系统的迭代求解器。最多在 n 步内收敛。适用于大型稀疏系统，其中分解太昂贵。 |
| 超定系统 | “数据多于参数” | 在 m-by-n 系统中 m > n。不存在精确解。最小二乘法找到最佳近似。这是每个回归问题。 |
| 回代 | “从下往上求解” | 给定上三角系统，先解最后一个方程，然后回代。复杂度为 O(n^2)。 |
| 前代 | “从上往下求解” | 给定下三角系统，先解第一个方程，然后前代。复杂度为 O(n^2)。用于 LU 求解的 L 步骤。 |

## 进一步阅读

- [MIT 18.06: 线性代数](https://ocw.mit.edu/courses/18-06-linear-algebra-spring-2010/)（Gilbert Strang）——关于线性系统和矩阵分解的权威课程
- [数值线性代数](https://people.maths.ox.ac.uk/trefethen/text.html)（Trefethen & Bau）——理解数值稳定性、条件和算法失败的标准参考资料
- [矩阵计算](https://www.cs.cornell.edu/cv/GolubVanLoan4/golubandvanloan.htm)（Golub & Van Loan）——每种矩阵算法的百科全书式参考
- [3Blue1Brown: 逆矩阵](https://www.3blue1brown.com/lessons/inverse-matrices)——几何上对求解 Ax = b 的视觉直觉
