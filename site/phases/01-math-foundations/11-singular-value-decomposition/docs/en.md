# 奇异值分解 (SVD)

> 矩阵分解的瑞士军刀。掌握 SVD 的极分解、低秩近似与数据压缩。

**Type:** 构建
**Languages:** Python
**Prerequisites:** Phase 1, Lesson 02 (向量与矩阵运算)
**Time:** ~45 分钟

## 学习目标

- 通过幂迭代实现 SVD，并解释 U、Sigma 和 V^T 的几何含义
- 应用截断 SVD 进行图像压缩，并测量压缩比与重建误差之间的关系
- 通过 SVD 计算 Moore-Penrose 伪逆，以解决超定最小二乘系统
- 将 SVD 与 PCA、推荐系统（潜在因子）以及 NLP 中的潜在语义分析联系起来

## 问题

你有一个 1000x2000 的矩阵。也许它是用户-电影评分。也许它是一份文档-词频表。也许它是图像的像素值。你需要压缩它、去噪它、在其中发现隐藏的结构，或者用它来解决最小二乘系统。特征分解只能用于方阵。即使如此，它还要求矩阵有一组完整的线性无关的特征向量。

SVD 可以用于任何矩阵。任何形状。任何秩。没有任何条件。它将矩阵分解成三个因子，揭示了矩阵对空间所做操作的几何特性。它是线性代数中最为通用且最有用的分解方法。```
A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
     (any)    (rotate)  (scale)  (rotate)
```给定任何矩阵 A，SVD 将其分解为：
- V^T 在输入空间（n 维）中旋转向量
- Sigma 沿每个轴进行缩放（拉伸或压缩）
- U 将结果旋转到输出空间（m 维）```mermaid
graph LR
    A["Input space (n-dim)\nData cloud\n(arbitrary orientation)"] -->|"V^T\n(rotate)"| B["Scaled space\nAligned with axes\nthen scaled by Sigma"]
    B -->|"U\n(rotate)"| C["Output space (m-dim)\nRotated to output\norientation"]
```你可以这样理解。你把一个矩阵交给SVD。它会告诉你：“这个矩阵将输入的球体首先通过V^T进行旋转，然后通过Sigma将其拉伸成一个椭球体，最后再通过U对这个椭球体进行旋转。” 奇异值就是这个椭球体的轴长。

### 完整的分解

对于一个形状为m x n的矩阵A：```
A = U * Sigma * V^T

where:
  U     is m x m, orthogonal (U^T U = I)
  Sigma is m x n, diagonal (singular values on the diagonal)
  V     is n x n, orthogonal (V^T V = I)

The singular values sigma_1 >= sigma_2 >= ... >= sigma_r > 0
where r = rank(A)
```矩阵 $ U $ 的列被称为左奇异向量。矩阵 $ V $ 的列被称为右奇异向量。矩阵 $ \Sigma $ 的对角线元素被称为奇异值。它们总是非负的，并且通常按降序排列。

### 左奇异向量、奇异值、右奇异向量

SVD 的每个组成部分都有其独特的几何意义。

**右奇异向量（$ V $ 的列）：** 它们构成了输入空间（$ \mathbb{R}^n $）的一个正交基。它们是输入空间中被矩阵映射到输出空间中正交方向的方向。可以把它们看作定义域的自然坐标系。

**奇异值（$ \Sigma $ 的对角线）：** 这些是缩放因子。第 $ i $ 个奇异值告诉你矩阵沿着第 $ i $ 个右奇异向量方向拉伸向量的程度。奇异值为零意味着矩阵完全压缩了该方向。

**左奇异向量（$ U $ 的列）：** 它们构成了输出空间（$ \mathbb{R}^m $）的一个正交基。第 $ i $ 个左奇异向量是输出空间中第 $ i $ 个右奇异向量（缩放后）所到达的方向。

它们之间的关系：```
A * v_i = sigma_i * u_i

The matrix A takes the i-th right singular vector v_i,
scales it by sigma_i, and maps it to the i-th left singular vector u_i.
```这给你一个逐坐标地描述任何矩阵作用的图像。

### 外积形式

SVD 可以写成秩为 1 的矩阵之和：```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

Each term sigma_i * u_i * v_i^T is a rank-1 matrix (an outer product).
The full matrix is the sum of r such matrices, where r is the rank.
```这种形式是低秩近似的基础。每一项都增加了一层结构。第一项捕捉到最重要的单一模式。第二项捕捉到下一个最重要的模式。依此类推。截断这个总和可以在任何给定的秩下给出最佳可能的近似。```
Rank-1 approx:    A_1 = sigma_1 * u_1 * v_1^T
                  (captures the dominant pattern)

Rank-2 approx:    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
                  (captures the two most important patterns)

Rank-k approx:    A_k = sum of top k terms
                  (optimal by the Eckart-Young theorem)
```### 与特征分解的关系

SVD 和特征分解紧密相关。矩阵 A 的奇异值和奇异向量直接来源于 A^T A 和 A A^T 的特征值和特征向量。```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

where D = Sigma^T * Sigma is a diagonal matrix with sigma_i^2 on the diagonal.

So:
- The right singular vectors (V) are eigenvectors of A^T A
- The singular values squared (sigma_i^2) are eigenvalues of A^T A

Similarly:
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

So:
- The left singular vectors (U) are eigenvectors of A A^T
- The eigenvalues of A A^T are also sigma_i^2
```此连接告诉你三件事：
1. 奇异值始终是实数且非负（它们是半正定矩阵的特征值的平方根）。
2. 你可以通过 $A^T A$ 的特征分解来计算 SVD，但这样会平方条件数并损失数值精度。专用的 SVD 算法避免了这个问题。
3. 当 $A$ 是方阵且对称半正定时，SVD 和特征分解是相同的事情。

### 截断 SVD：低秩近似

Eckart-Young-Mirsky 定理指出，对 $A$ 的最佳秩-k 近似（在 Frobenius 范数和谱范数下）是仅保留前 k 个奇异值及其对应的向量：```
A_k = U_k * Sigma_k * V_k^T

where:
  U_k     is m x k  (first k columns of U)
  Sigma_k is k x k  (top-left k x k block of Sigma)
  V_k     is n x k  (first k columns of V)

Approximation error = sigma_{k+1}  (in spectral norm)
                    = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)  (in Frobenius norm)
```这不仅仅是“一个好”的近似。它被证明是秩为k的最佳可能近似。没有其他秩为k的矩阵比它更接近A。

| 组件 | 相对大小 | 在秩-3近似中保留？ |
|------|---------|------------------|
| sigma_1 | 最大 | 是 |
| sigma_2 | 大 | 是 |
| sigma_3 | 中等偏大 | 是 |
| sigma_4 | 中等 | 否（误差） |
| sigma_5 | 中等偏小 | 否（误差） |
| sigma_6 | 小 | 否（误差） |
| sigma_7 | 非常小 | 否（误差） |
| sigma_8 | 极小 | 否（误差） |

保留前3个：A_3捕获了三个最大的奇异值。误差 = 剩下的值（sigma_4到sigma_8）。

如果奇异值衰减得很快，很小的k就能捕获矩阵的大部分信息。如果它们衰减得慢，矩阵就没有低秩结构。

### 使用SVD进行图像压缩

灰度图像是一张像素强度的矩阵。一张800x600的图像有480,000个值。SVD允许你用远少的数值来近似它。```
Original image: 800 x 600 = 480,000 values

SVD with rank k:
  U_k:      800 x k values
  Sigma_k:  k values
  V_k:      600 x k values
  Total:    k * (800 + 600 + 1) = k * 1401 values

  k=10:   14,010 values   (2.9% of original)
  k=50:   70,050 values  (14.6% of original)
  k=100: 140,100 values  (29.2% of original)

  The compression ratio improves as k gets smaller,
  but visual quality degrades.
```关键见解：自然图像的奇异值衰减得非常快。前几个奇异值捕捉了图像的整体结构（形状、梯度）。后面的奇异值则捕捉了细节和噪声。通常在秩为50时截断，可以产生一个看起来几乎与原图相同的图像，同时存储空间减少了85%。

### 推荐系统中的SVD

Netflix Prize使这一点变得著名。你有一个用户-电影评分矩阵，其中大部分条目是缺失的。```
             Movie1  Movie2  Movie3  Movie4  Movie5
  User1      [  5      ?       3       ?       1  ]
  User2      [  ?      4       ?       2       ?  ]
  User3      [  3      ?       5       ?       ?  ]
  User4      [  ?      ?       ?       4       3  ]

  ? = unknown rating
```想法：这个评分矩阵具有低秩。用户们的口味并非完全独立。少数潜在因素（如动作对剧情、老片对新片、脑力对感官）可以解释大部分的偏好。

对（已填充的）评分矩阵进行SVD分解，可以将其分解为：
- U：用户在潜在因素空间中的用户画像
- Sigma：每个潜在因素的重要性
- V^T：电影在潜在因素空间中的电影画像

用户对某部电影的预测评分，是用户画像与电影画像的点积（权重为奇异值）。低秩近似可以填补缺失的条目。

在实践中，你会使用像Simon Funk的增量SVD或者ALS（交替最小二乘法）这样的变体，它们可以直接处理缺失的数据。但核心思想是一样的：通过SVD进行潜在因素分解。

### NLP中的SVD：潜在语义分析

潜在语义分析（LSA），也称为潜在语义索引（LSI），将SVD应用于一个词-文档矩阵。```
             Doc1   Doc2   Doc3   Doc4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

After SVD with rank k=2:

  Each document becomes a point in 2D "concept space."
  Each term becomes a point in the same 2D space.
  Documents about similar topics cluster together.
  Terms with similar meanings cluster together.

  "cat" and "dog" end up near each other (land pets).
  "fish" and "ocean" end up near each other (water concepts).
  Doc1 and Doc3 cluster if they share similar topics.
```LSA 是最早成功从原始文本中捕捉语义相似性的方法之一。它之所以有效，是因为同义词倾向于出现在相似的文档中，因此 SVD 将它们归入相同的潜在维度。现代词嵌入（Word2Vec、GloVe）可以被视为这一思想的后代。

### 使用 SVD 进行降噪

噪声数据的信号集中在前几个奇异值中，而噪声则分布在所有奇异值中。截断可以去除噪声地板。

**干净信号的奇异值：**

| 成分   | 幅度   | 类型   |
|--------|--------|--------|
| sigma_1 | 非常大 | 信号   |
| sigma_2 | 大     | 信号   |
| sigma_3 | 中等   | 信号   |
| sigma_4 | 接近零 | 可忽略 |
| sigma_5 | 接近零 | 可忽略 |

**噪声信号的奇异值（噪声添加到所有）：**

| 成分   | 幅度   | 类型   |
|--------|--------|--------|
| sigma_1 | 非常大 | 信号   |
| sigma_2 | 大     | 信号   |
| sigma_3 | 中等   | 信号   |
| sigma_4 | 小     | 噪声   |
| sigma_5 | 小     | 噪声   |
| sigma_6 | 小     | 噪声   |
| sigma_7 | 小     | 噪声   |```mermaid
graph TD
    A["All singular values"] --> B{"Clear gap?"}
    B -->|"Above gap"| C["Signal: keep these (top k)"]
    B -->|"Below gap"| D["Noise: discard these"]
    C --> E["Reconstruct with A_k to get denoised version"]
```这在信号处理、科学测量和数据清洗中都有应用。任何时候你有一个被加性噪声破坏的矩阵，截断的SVD是一种原理性的方式来将信号与噪声分离。

### 通过SVD求伪逆

Moore-Penrose伪逆 $A^+$ 将矩阵求逆推广到了非方阵和奇异矩阵。SVD使得计算它变得非常简单。```
If A = U * Sigma * V^T, then:

A+ = V * Sigma+ * U^T

where Sigma+ is formed by:
  1. Transpose Sigma (swap rows and columns)
  2. Replace each non-zero diagonal entry sigma_i with 1/sigma_i
  3. Leave zeros as zeros

For A (m x n):      A+ is (n x m)
For Sigma (m x n):  Sigma+ is (n x m)
```伪逆用于求解最小二乘问题。如果方程 Ax = b 没有精确解（即超定系统），那么 x = A⁺b 是最小二乘解（即最小化 ||Ax - b||）。```
Overdetermined system (more equations than unknowns):

  [1  1]         [3]
  [2  1] x   =   [5]       No exact solution exists.
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  This gives the x that minimizes the sum of squared residuals.
  Same result as the normal equations (A^T A)^(-1) A^T b,
  but numerically more stable.
```### 数值稳定性优势

计算 A^T A 的特征分解会将奇异值平方（A^T A 的特征值是 sigma_i^2）。这会将条件数平方，放大数值误差。```
Example:
  A has singular values [1000, 1, 0.001]
  Condition number of A: 1000 / 0.001 = 10^6

  A^T A has eigenvalues [10^6, 1, 10^{-6}]
  Condition number of A^T A: 10^6 / 10^{-6} = 10^{12}

  Computing SVD directly: works with condition number 10^6
  Computing via A^T A:     works with condition number 10^{12}
                           (6 extra digits of precision lost)
```现代 SVD 算法（Golub-Kahan 双对角化）直接作用于 A，从不形成 A^T A。这就是为什么你应该始终优先选择 `np.linalg.svd(A)` 而不是 `np.linalg.eig(A.T @ A)`。

### 与 PCA 的联系

PCA 就是对居中数据进行的 SVD。这并不是一种类比。这实际上是完全相同的计算。```
Given data matrix X (n_samples x n_features), centered (mean subtracted):

Covariance matrix: C = (1/(n-1)) * X^T X

PCA finds eigenvectors of C. But:

  X = U * Sigma * V^T    (SVD of X)

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

So the principal components are exactly the right singular vectors V.
The explained variance for each component is sigma_i^2 / (n-1).

In sklearn, PCA is implemented using SVD, not eigendecomposition.
It is faster and more numerically stable.
```这意味着你在第 10 课中学到的所有关于降维的知识实际上都是在使用 SVD（奇异值分解）。PCA（主成分分析）是机器学习中 SVD 最常见的应用。```figure
svd-rank-reconstruction
```## 构建它

### 步骤 1：使用幂迭代从零开始构建 SVD

思路：为了找到最大的奇异值及其对应的向量，对 A^T A（或 A A^T）使用幂迭代。然后对矩阵进行降维，并重复该过程以找到下一个奇异值。```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```### 步骤 2：测试并和 NumPy 进行比较```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```### 步骤 3：图像压缩演示```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```### 步骤 4：降噪```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```### 步骤 5：伪逆```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```## 使用它

完整的可运行演示在 `code/svd.py` 中。运行它以查看 SVD 在图像压缩、推荐系统、潜在语义分析和降噪中的应用。```bash
python svd.py
````code/svd.jl` 中的 Julia 版本使用 Julia 本机的 `svd()` 函数和 `LinearAlgebra` 包来演示相同的概念。```bash
julia svd.jl
```## 发布它

本课内容包括：
- `outputs/skill-svd.md` - 一项技能，用于了解何时以及如何在实际项目中应用SVD

## 练习

1. 不使用幂迭代法，从零开始实现完整的SVD。相反，计算A^T A的特征分解以获得V和奇异值，然后计算U = A V Sigma^{-1}。将数值精度与你的幂迭代版本和NumPy进行比较。

2. 加载一张真实的灰度图像（或将其转换为灰度图像）。在秩1、5、10、25、50、100时进行压缩。对于每个秩，计算压缩比和相对误差。找到图像变得视觉上可接受的秩。

3. 构建一个小型的推荐系统。创建一个10x8的用户-电影评分矩阵，其中包含一些已知的条目。用行均值填充缺失的条目。计算SVD并重建一个秩3的近似。使用重建的矩阵预测缺失的评分。验证预测是否合理。

4. 创建一个100x50的文档-术语矩阵，其中包含3个合成主题。每个主题有5个相关术语。添加噪声。应用SVD并验证前3个奇异值远大于其他值。将文档投影到3D潜在空间，并检查来自同一主题的文档是否聚在一起。

5. 生成一个干净的低秩矩阵（秩3，大小为50x40）并在不同噪声水平（sigma = 0.1、0.5、1.0、2.0）下添加高斯噪声。对于每个噪声水平，通过从1到40遍历k并测量与干净矩阵的重建误差，找到最佳的截断秩。绘制最佳k如何随噪声水平变化。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|--------|
| SVD | "分解任何矩阵" | 将A分解为U Sigma V^T，其中U和V是正交的，Sigma是对角矩阵，且元素非负。适用于任何形状的矩阵。 |
| 奇异值 | "这个组件的重要性" | Sigma的第i个对角元素。衡量矩阵在第i个主方向上拉伸的程度。始终非负，按降序排列。 |
| 左奇异向量 | "输出方向" | U的一列。输出空间中第i个右奇异向量映射到的方向（在乘以sigma_i后）。 |
| 右奇异向量 | "输入方向" | V的一列。输入空间中矩阵映射到第i个左奇异向量的方向（在乘以sigma_i后）。 |
| 截断SVD | "低秩近似" | 仅保留前k个奇异值及其对应的向量。产生原始矩阵的可证明的最佳秩-k近似（Eckart-Young定理）。 |
| 秩 | "真实维度" | 非零奇异值的个数。告诉你矩阵实际使用的独立方向数量。 |
| 伪逆 | "广义逆" | V Sigma+ U^T。反转非零奇异值，保持零为零。用于解决非方阵或奇异矩阵的最小二乘问题。 |
| 条件数 | "对误差的敏感度" | sigma_max / sigma_min。大的条件数意味着小的输入变化会导致大的输出变化。SVD直接揭示这一点。 |
| 潜在因子 | "隐藏变量" | SVD在低秩空间中发现的一个维度。在推荐系统中，潜在因子可能对应于类型偏好。在NLP中，它可能对应于一个主题。 |
| Frobenius范数 | "总矩阵大小" | 所有元素平方和的平方根。等于所有奇异值平方和的平方根。用于衡量近似误差。 |
| Eckart-Young定理 | "SVD提供最佳压缩" | 对于任何目标秩k，截断SVD在所有可能的秩-k矩阵中最小化近似误差。 |
| 幂迭代 | "找到最大的特征向量" | 重复将随机向量与矩阵相乘并进行归一化。收敛到具有最大特征值的特征向量。许多SVD算法的基本构建块。 |

## 进一步阅读

- [Gilbert Strang: 《线性代数及其应用》第7章](https://math.mit.edu/~gs/linearalgebra/) - 对SVD及其应用的全面介绍
- [3Blue1Brown: 《但什么是SVD？》](https://www.youtube.com/watch?v=vSczTbgc8Rc) - SVD的几何直觉
- [《我们推荐奇异值分解》](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) - 美国数学学会的易懂概述
- [《Netflix Prize和矩阵分解》](https://sifter.org/~simon/journal/20061211.html) - Simon Funk关于推荐系统中SVD的原始博客文章
- [潜在语义分析](https://en.wikipedia.org/wiki/Latent_semantic_analysis) - SVD最初的NLP应用
- [《数值线性代数》Trefethen和Bau](https://people.maths.ox.ac.uk/trefethen/text.html) - 理解SVD算法及其数值特性的黄金标准
