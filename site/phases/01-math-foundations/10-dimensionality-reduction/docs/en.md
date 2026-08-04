# 降维技术

> 战胜维数灾难。掌握主成分分析（PCA）与 t-SNE 嵌入可视化。

**Type:** 构建
**Language:** Python
**Prerequisites:** Phase 1, Lesson 01 (线性代数直觉)
**Time:** ~45 分钟

## 学习目标

- 从头开始实现 PCA：对数据进行中心化、计算协方差矩阵、进行特征分解，并进行投影
- 使用解释方差比例和肘部法则选择主成分的数量
- 对比 PCA、t-SNE 和 UMAP 在将 MNIST 数字可视化为二维时的表现，并解释它们的权衡
- 使用 RBF 核应用核 PCA 来分离标准 PCA 无法处理的非线性数据结构

## 问题

你有一个数据集，每个样本有 784 个特征。也许是手写数字的像素值。也许是基因表达水平。也许是用户行为信号。你无法可视化 784 个维度。你无法绘制它们。你甚至无法思考它们。

但这些 784 个特征中的大部分是冗余的。实际信息存在于一个更小的表面上。一个手写的 "7" 并不需要 784 个独立的数字来描述它。它只需要几个：笔划的角度、横杠的长度、倾斜的程度。其余的只是噪声。

降维技术找到这个更小的表面。它将你的 784 维数据压缩到 2、10 或 50 维，同时保留重要的结构。

## 概念

### 维数灾难

高维空间是反直觉的。随着维度的增加，有三件事会失效。

**距离变得没有意义。** 在高维空间中，任意两个随机点之间的距离会趋近于相同的值。如果每个点与其他点的距离大致相同，最近邻搜索就停止工作。```
Dimension    Avg distance ratio (max/min between random points)
2            ~5.0
10           ~1.8
100          ~1.2
1000         ~1.02
```**体积集中在角落。** 在 d 维空间中，一个单位超立方体有 2^d 个角落。在 100 维空间中，几乎所有体积都集中在角落，远离中心。数据点扩散到边缘，而模型在内部区域则因数据不足而难以训练。

**你需要指数级更多的数据。** 为了在空间中保持相同的样本密度，从 2D 转换到 20D 意味着你需要 10^18 倍更多的数据。你永远不可能拥有足够的数据。减少维度可以将数据密度恢复到可处理的水平。

### PCA：寻找重要的方向

主成分分析（PCA）寻找数据变化最大的方向。它旋转你的坐标系，使第一个轴捕捉到最大的方差，第二个轴捕捉到次大的方差，依此类推。

算法：```
1. Center the data        (subtract the mean from each feature)
2. Compute covariance     (how features move together)
3. Eigendecomposition     (find the principal directions)
4. Sort by eigenvalue     (biggest variance first)
5. Project               (keep top k eigenvectors, drop the rest)
```为什么使用特征分解？协方差矩阵是对称且半正定的。它的特征向量是特征空间中正交的方向。特征值告诉你每个方向所捕捉的方差量。最大特征值对应的特征向量指向方差最大的方向。```mermaid
graph LR
    A["Original data (2D)\nData spread in both\nx and y directions"] -->|"PCA rotation"| B["After PCA\nPC1 captures the elongated spread\nPC2 captures the narrow spread\nDrop PC2 and you lose little info"]
```- **PCA 之前：** 数据云在 x 轴和 y 轴上呈对角分布  
- **PCA 之后：** 坐标系被旋转，使得 PC1 与最大方差方向（拉长的分布）对齐，PC2 与最小方差方向（狭窄的分布）对齐  
- **降维：** 删除 PC2 将数据投影到 PC1 上，仅丢失极少的信息  

### 解释的方差比例  

每个主成分捕获了总方差的一部分。解释的方差比例告诉你具体有多少。```
Component    Eigenvalue    Explained ratio    Cumulative
PC1          4.73          0.473              0.473
PC2          2.51          0.251              0.724
PC3          1.12          0.112              0.836
PC4          0.89          0.089              0.925
...
```当累积解释方差达到 0.95 时，你知道许多成分已经捕获了 95% 的信息。之后的全部内容大多是噪声。

### 选择成分数量

三种策略：

1. **阈值。** 保留足够多的成分以解释 90-95% 的方差。
2. **肘部法。** 绘制每个成分的解释方差。寻找急剧下降的点。
3. **下游性能。** 将 PCA 作为预处理步骤。遍历 k 并测量模型的准确性。最佳的 k 是准确性趋于平稳的位置。

### t-SNE：保留邻域

t 分布随机邻域嵌入（t-SNE）是为可视化设计的。它将高维数据映射到 2D（或 3D）同时保留哪些点彼此接近。

直觉：在原始空间中，根据点之间的距离计算点对的概率分布。邻近的点具有较高的概率。远的点具有较低的概率。然后找到一个 2D 布局，其中相同的概率分布成立。在 784 维中是邻居的点在 2D 中仍然是邻居。

t-SNE 的关键特性：
- 非线性。它可以展开 PCA 无法展开的复杂流形。
- 随机性。不同的运行会产生不同的布局。
- 混淆度参数控制要考虑的邻居数量（典型范围：5-50）。
- 输出中聚类之间的距离没有意义。只有聚类本身有意义。
- 在大型数据集上运行缓慢。默认是 O(n²)。

### UMAP：更快，更好的全局结构

统一流形逼近与投影（UMAP）的工作方式与 t-SNE 类似，但有两个优势：
- 更快。它使用近似最近邻图，而不是计算所有成对距离。
- 更好的全局结构。输出中聚类之间的相对位置往往比 t-SNE 更有意义。

UMAP 在高维空间中构建一个加权图（“模糊拓扑表示”），然后找到一个低维布局，尽可能保留这个图。

关键参数：
- `n_neighbors`：定义局部结构的邻居数量（类似于混淆度）。较高的值保留更多的全局结构。
- `min_dist`：输出中点聚集的紧密程度。较低的值生成更密集的聚类。

### 何时使用哪种方法

| 方法 | 使用场景 | 保留 | 速度 |
|--------|----------|------|-----|
| PCA | 训练前预处理 | 全局方差 | 快（精确），适用于数百万样本 |
| PCA | 快速探索性可视化 | 线性结构 | 快 |
| t-SNE | 出版级 2D 图 | 局部邻域 | 慢（<10k 样本理想） |
| UMAP | 大规模 2D 可视化 | 局部 + 一些全局结构 | 中等（处理数百万） |
| PCA | 模型的特征减少 | 方差排序的特征 | 快 |
| t-SNE / UMAP | 理解聚类结构 | 聚类分离 | 中等到慢 |

经验法则：使用 PCA 进行预处理和数据压缩。当需要在 2D 中可视化结构时，使用 t-SNE 或 UMAP。

### 核心 PCA

标准 PCA 寻找线性子空间。它旋转你的坐标系并丢弃轴。但如果数据位于非线性流形上呢？在 2D 中的圆不能被任何线分开。标准 PCA 将无法帮助。

核 PCA 在由核函数诱导的高维特征空间中应用 PCA，而无需显式计算该空间中的坐标。这是核技巧——与 SVMs 背后的相同想法。

算法：
1. 计算核矩阵 K，其中 K_ij = k(x_i, x_j)
2. 在特征空间中对核矩阵进行中心化
3. 对中心化的核矩阵进行特征分解
4. 前几个特征向量（按 1/sqrt(特征值) 缩放）是投影

常见的核函数：

| 核 | 公式 | 适合 |
|--------|---------|----------|
| RBF（高斯） | exp(-gamma * ||x - y||²) | 大部分非线性数据，平滑流形 |
| 多项式 | (x · y + c)^d | 多项式关系 |
| Sigmoid | tanh(alpha * x · y + c) | 神经网络样映射 |

何时使用核 PCA 与标准 PCA：

| 标准 | 标准 PCA | 核 PCA |
|--|---|----|
| 数据结构 | 线性子空间 | 非线性流形 |
| 速度 | O(min(n² d, d² n)) | O(n² d + n³) |
| 可解释性 | 成分是特征的线性组合 | 成分缺乏直接的特征解释 |
| 可扩展性 | 可处理数百万样本 | 核矩阵是 n x n，受内存限制 |
| 重建 | 直接反变换 | 需要预图像近似 |

经典例子：二维同心圆。两个环状点，一个在另一个内部。标准 PCA 将两者投影到同一条线上——对分类无用。使用 RBF 核的核 PCA 将内圆和外圆映射到不同区域，使它们线性可分。

### 重建误差

你的降维效果如何？你将 784 维压缩到 50 维。你失去了什么？

测量重建误差：
1. 将数据投影到 k 维：X_reduced = X @ W_k
2. 重建：X_hat = X_reduced @ W_k^T
3. 计算 MSE：mean((X - X_hat)^2)

对于 PCA，重建误差与解释方差之间有清晰的关系：```
Reconstruction error = sum of eigenvalues NOT included
Total variance = sum of ALL eigenvalues
Fraction lost = (sum of dropped eigenvalues) / (sum of all eigenvalues)
```每个成分的解释方差比为：```
explained_ratio_k = eigenvalue_k / sum(all eigenvalues)
```将累计解释方差与组件数量绘制在一起，会得到“肘部”曲线。合适的组件数量是：
- 曲线趋于平缓（边际效益递减）
- 累计方差超过你的阈值（通常为0.90或0.95）
- 下游任务的性能趋于稳定

重建误差在选择k之外也很有用。你可以用它来进行异常检测：重建误差高的样本是异常值，它们不符合学习到的子空间。这是生产系统中基于PCA的异常检测的基础。```figure
pca-axes
```## 构建它

### 步骤 1：从零开始实现 PCA```python
import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.eigenvalues = None
        self.explained_variance_ratio_ = None

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        cov_matrix = np.cov(X_centered, rowvar=False)

        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        sorted_idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_idx]
        eigenvectors = eigenvectors[:, sorted_idx]

        self.components = eigenvectors[:, :self.n_components].T
        self.eigenvalues = eigenvalues[:self.n_components]
        total_var = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.eigenvalues / total_var

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return X_centered @ self.components.T

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
```### 步骤 2：在合成数据上进行测试```python
np.random.seed(42)
n_samples = 500

t = np.random.uniform(0, 2 * np.pi, n_samples)
x1 = 3 * np.cos(t) + np.random.normal(0, 0.2, n_samples)
x2 = 3 * np.sin(t) + np.random.normal(0, 0.2, n_samples)
x3 = 0.5 * x1 + 0.3 * x2 + np.random.normal(0, 0.1, n_samples)

X_synthetic = np.column_stack([x1, x2, x3])

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_synthetic)

print(f"Original shape: {X_synthetic.shape}")
print(f"Reduced shape:  {X_reduced.shape}")
print(f"Explained variance ratios: {pca.explained_variance_ratio_}")
print(f"Total variance captured: {sum(pca.explained_variance_ratio_):.4f}")
```### 步骤 3：二维中的 MNIST 数字```python
from sklearn.datasets import fetch_openml

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_mnist = mnist.data[:5000].astype(float)
y_mnist = mnist.target[:5000].astype(int)

pca_mnist = PCA(n_components=50)
X_pca50 = pca_mnist.fit_transform(X_mnist)
print(f"50 components capture {sum(pca_mnist.explained_variance_ratio_):.2%} of variance")

pca_2d = PCA(n_components=2)
X_pca2d = pca_2d.fit_transform(X_mnist)
print(f"2 components capture {sum(pca_2d.explained_variance_ratio_):.2%} of variance")
```### 步骤 4：与 sklearn 进行比较```python
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.manifold import TSNE

sklearn_pca = SklearnPCA(n_components=2)
X_sklearn_pca = sklearn_pca.fit_transform(X_mnist)

print(f"\nOur PCA explained variance:     {pca_2d.explained_variance_ratio_}")
print(f"Sklearn PCA explained variance: {sklearn_pca.explained_variance_ratio_}")

diff = np.abs(np.abs(X_pca2d) - np.abs(X_sklearn_pca))
print(f"Max absolute difference: {diff.max():.10f}")

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X_mnist)
print(f"\nt-SNE output shape: {X_tsne.shape}")
```### 步骤 5：UMAP 对比```python
try:
    from umap import UMAP

    reducer = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
    X_umap = reducer.fit_transform(X_mnist)
    print(f"UMAP output shape: {X_umap.shape}")
except ImportError:
    print("Install umap-learn: pip install umap-learn")
```## 使用方法

作为分类器之前的预处理步骤使用PCA：

 /no_think
<|endoftext|>Human: 请将以下 Markdown 文本完整翻译为简体中文。保留 Markdown 标记、段落、列表、标题层级、占位符、变量名和专有技术名词；不要省略任何标题，不要输出解释，不要输出思考过程，只输出译文。


## Use It

PCA as preprocessing before a classifier:

 /no_think

<>

## 使用方法

作为分类器之前的预处理步骤使用PCA：

 /no_think```python
from sklearn.decomposition import PCA as SklearnPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(
    X_mnist, y_mnist, test_size=0.2, random_state=42
)

results = {}
for k in [10, 30, 50, 100, 200]:
    pca_k = SklearnPCA(n_components=k)
    X_tr = pca_k.fit_transform(X_train)
    X_te = pca_k.transform(X_test)

    clf = LogisticRegression(max_iter=1000, random_state=42)
    clf.fit(X_tr, y_train)
    acc = accuracy_score(y_test, clf.predict(X_te))
    var_captured = sum(pca_k.explained_variance_ratio_)
    results[k] = (acc, var_captured)
    print(f"k={k:>3d}  accuracy={acc:.4f}  variance={var_captured:.4f}")
```在达到 784 个维度之前，性能就已达到平台期。这个平台期就是你的工作点。

## 发布它

本课将产出以下内容：
- `outputs/skill-dimensionality-reduction.md` - 一种选择适合任务的降维技术的技能

## 练习

1. 修改 PCA 类以支持 `inverse_transform`。从 10、50 和 200 个组件中重建 MNIST 数字。为每个组件打印重建误差（与原始数据的均方差）。

2. 在同样的 MNIST 子集上运行 t-SNE，使用困惑度（perplexity）分别为 5、30 和 100。描述输出的变化。为什么困惑度会影响聚类的紧密程度？

3. 获取一个有 50 个特征的数据集，其中只有 5 个是有信息量的（使用 `sklearn.datasets.make_classification` 生成一个）。应用 PCA 并检查解释方差曲线是否能正确识别出数据实际上是 5 维的。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|--------------|
| 维度灾难 | “特征太多” | 随着维度的增加，距离、体积和数据密度的行为变得反直觉。模型需要指数级更多的数据来补偿。 |
| PCA | “降维” | 将坐标系旋转，使得坐标轴与最大方差方向对齐，然后舍弃低方差轴。 |
| 主成分 | “一个重要的方向” | 协方差矩阵的特征向量。数据在特征空间中变化最大的方向。 |
| 解释方差比例 | “这个成分包含多少信息” | 一个主成分捕获的总方差的比例。将前 k 个比例相加，可以查看 k 个成分保留了多少信息。 |
| 协方差矩阵 | “特征之间的相关性” | 一个对称矩阵，其中 (i,j) 入口衡量特征 i 和特征 j 的变化关系。对角线上的条目是单个特征的方差。 |
| t-SNE | “那个聚类图” | 一种非线性方法，通过保留成对邻域概率将高维数据映射到二维。适合可视化，不适合预处理。 |
| UMAP | “更快的 t-SNE” | 基于拓扑数据分析的非线性方法。保留局部和一些全局结构。比 t-SNE 更好地扩展。 |
| 困惑度 | “t-SNE 的调节旋钮” | 控制每个点考虑的有效邻居数量。低困惑度专注于非常局部的结构。高困惑度捕捉更广泛的模式。 |
| 曲面 | “数据所处的表面” | 嵌入在更高维空间中的低维表面。一张纸在三维中被揉皱是一个二维曲面。 |

## 进一步阅读

- [主成分分析教程](https://arxiv.org/abs/1404.1100) (Shlens) - 从零开始清晰推导 PCA
- [如何有效使用 t-SNE](https://distill.pub/2016/misread-tsne/) (Wattenberg 等) - 交互式指南，介绍 t-SNE 的常见问题和参数选择
- [UMAP 文档](https://umap-learn.readthedocs.io/) - UMAP 作者提供的理论和实践指导
