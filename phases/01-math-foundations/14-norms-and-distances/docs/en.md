# 范数与距离度量

> 从几何空间到特征度量。掌握 L1/L2 范数、曼哈顿/欧氏距离、余弦相似度与马氏距离。

**Type:** 学习
**Language:** Python
**Prerequisites:** Phase 1, Lesson 02 (向量与矩阵运算)
**Time:** ~35 分钟

## 学习目标

- 从零开始实现 L1、L2、余弦、马氏、Jaccard 和编辑距离函数
- 为给定的机器学习任务选择合适的距离度量，并解释为什么其他选择会失败
- 将 L1 和 L2 范数与 LASSO 和 Ridge 正则化及其几何约束区域联系起来
- 演示相同的数据集在不同度量下会产生不同的最近邻

## 问题

你有两个向量。它们可能是词嵌入，也可能是用户画像，也可能是像素数组。你需要知道：它们有多接近？

答案完全取决于你选择的距离函数。在一种度量下，两个数据点可能是最近邻，而在另一种度量下，它们可能相距很远。你的 KNN 分类器、推荐引擎、向量数据库、聚类算法、损失函数——它们都依赖于这个选择。如果选择错误，你的模型将优化错误的目标。

没有一种通用的最佳距离。L2 适用于空间数据。余弦相似度主导 NLP。Jaccard 适用于集合。编辑距离适用于字符串。马氏距离考虑了相关性。Wasserstein 移动概率质量。每种距离都对“相似”的含义有不同的假设。

本节课将从零开始构建每种主要的距离函数，告诉你何时使用每种工具，并演示相同的数据在使用不同的度量时会产生完全不同的最近邻。

## 概念

### 范数：衡量向量大小

范数衡量一个向量的“大小”。两个向量之间的任何距离函数都可以写成它们差值的范数：d(a, b) = ||a - b||。因此，理解范数就是理解距离。

### L1 范数（曼哈顿距离）

L1 范数计算所有分量的绝对值之和。

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

它被称为曼哈顿距离，因为它衡量的是你在只能沿坐标轴移动的城市网格上行走的距离。不允许对角线移动。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何时使用 L1：
- 高维稀疏数据（文本特征、独热编码）
- 当你希望对异常值具有鲁棒性（一个巨大的差异不会占主导地位）
- 特征选择问题（L1 正则化促进稀疏性）

与 L1 正则化（Lasso）的联系：将 ||w||_1 添加到损失函数中，会对绝对权重值的总和进行惩罚。这会将小权重精确地推到零，从而实现自动特征选择。L1 惩罚在权重空间中创建了钻石形状的约束区域，钻石的角位于某些权重为零的轴上。

与损失函数的联系：平均绝对误差（MAE）是预测值与目标值之间平均的 L1 距离。它对所有误差进行线性惩罚，与均方误差（MSE）相比，对异常值具有更强的鲁棒性。

### L2 范数（欧几里得距离）

L2 范数是直线距离。平方根的平方和的总和。

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

这是你在几何课上学到的距离。n维空间中的毕达哥拉斯定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何时使用 L2：
- 低至中维的连续数据
- 当特征的尺度可比时
- 物理距离（空间数据、传感器读数）
- 像素级别的图像相似性

与 L2 正则化（岭回归）的联系：将 ||w||_2^2 添加到损失函数中会惩罚较大的权重。与 L1 不同，它不会将权重推向零。它按比例将所有权重向零收缩。L2 惩罚创建了圆形的约束区域，因此在坐标轴上没有角落。权重会变得很小，但很少恰好为零。

与损失函数的联系：均方误差（MSE）是 L2 距离平方的平均值。平方操作对较大的误差惩罚比对较小的误差更重。

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp 范数：一般的家族

L1 和 L2 是 Lp 范数的特例：

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

不同的 $ p $ 值会产生不同形状的“单位球”（所有到原点距离为 1 的点的集合）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-无穷范数（切比雪夫距离）

当 p 趋近于无穷大时，Lp 范数收敛到最大绝对分量。

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

两个点之间的距离由它们差异最大的那个单一维度决定。其他所有维度都被忽略。

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

何时使用 L-无穷范数：
- 当任何单个维度的最坏情况偏差很重要时
- 棋盘游戏（国际象棋中的国王以 L-无穷范数移动：向任何方向移动一步的代价都是 1）
- 制造公差（每个维度都必须符合规格）

### 余弦相似度和余弦距离

余弦相似度衡量两个向量之间的角度，忽略它们的大小。

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

它的取值范围从 -1（相反方向）到 +1（相同方向）。垂直向量的余弦相似度为 0。

余弦距离将其转换为距离：cosine_distance = 1 - cosine_similarity。它的取值范围从 0（相同方向）到 2（相反方向）。

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

为什么余弦在NLP和嵌入中占主导地位：在文本中，文档长度不应影响相似性。一篇关于猫的文档长度是另一篇关于猫的文档的两倍，它仍然应该是“相似”的。余弦相似性忽略幅度（长度），只关心方向。两个具有相同词分布但长度不同的文档指向相同的方向，并且余弦相似性为1.0。

何时使用余弦相似性：
- 文本相似性（TF-IDF向量、词嵌入、句子嵌入）
- 任何幅度是噪声而方向是信号的领域
- 推荐系统（用户偏好向量）
- 嵌入搜索（向量数据库几乎总是使用余弦或点积）

### 点积相似性 vs 余弦相似性

两个向量的点积是：

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

余弦相似度是通过两个向量的模长对点积进行归一化后的结果。当两个向量已经进行单位归一化（模长 = 1）时，点积和余弦相似度是相同的。

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

当它们不同时：点积包含幅度信息。幅度较大的向量会得到更高的点积得分。这在某些检索系统中很重要，因为在这些系统中你希望“流行”的项目排名更高。幅度起到了一个隐式的质量或重要性信号的作用。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

实际上：
- 当你想要纯粹的方向相似性时，使用余弦相似度
- 当幅度携带有意义的信息时，使用点积
- 许多向量数据库（Pinecone、Weaviate、Qdrant）允许你在它们之间进行选择
- 如果你的嵌入是L2归一化的，选择不会影响结果

### 马氏距离

欧几里得距离将所有维度视为同等重要。但如果你的特征之间存在相关性或具有不同的尺度，L2距离会给出误导性的结果。

马氏距离考虑了数据的协方差结构。

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

其中，S 是数据的协方差矩阵。

直观上：马氏距离首先对数据进行去相关和归一化处理（白化），然后在转换后的空间中计算 L2 距离。如果 S 是单位矩阵（即特征之间不相关，且方差为 1），那么马氏距离就退化为欧几里得距离。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何时使用马氏距离：
- 离群点检测（与均值的马氏距离较大的点为离群点）
- 当特征具有不同尺度和相关性时进行分类
- 当有足够的数据来估计一个可靠的协方差矩阵时
- 制造业中的质量控制（多变量过程监控）

### Jaccard 相似度（用于集合）

Jaccard 相似度衡量两个集合之间的重叠程度。

```
J(A, B) = |A intersect B| / |A union B|
```

它的取值范围从 0（没有重叠）到 1（完全相同的集合）。Jaccard 距离 = 1 - Jaccard 相似度。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何时使用 Jaccard：
- 比较标签、类别或特征的集合
- 基于单词存在情况（而非频率）的文档相似性
- 近似重复检测（Jaccard 的 MinHash 近似）
- 比较二进制特征向量（存在/不存在数据）
- 评估分割模型（交并比 = Jaccard）

### 编辑距离（Levenshtein 距离）

编辑距离计算将一个字符串转换为另一个字符串所需的最小单字符操作次数。这些操作包括：插入、删除或替换。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

通过动态规划计算得出。填充一个矩阵，其中条目 (i, j) 表示字符串 A 的前 i 个字符与字符串 B 的前 j 个字符之间的编辑距离。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何时使用编辑距离：
- 拼写检查与纠正
- DNA序列比对（使用加权操作）
- 模糊字符串匹配
- 混乱文本数据的去重

### KL散度（不是距离，但被当作距离使用）

KL散度衡量一个概率分布与另一个概率分布的差异。在第09课中已覆盖，但它属于本讨论的一部分，因为尽管它不是距离，人们仍然将其当作“距离”使用。

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

关键属性：KL散度**不是**对称的。

```
D_KL(P || Q) != D_KL(Q || P)
```

这意味着它未能满足距离度量的基本要求。它也不满足三角不等式。它是一种散度（divergence），而不是距离。

前向KL散度（D_KL(P || Q)）是“均值导向”的：Q试图覆盖P的所有模式。  
后向KL散度（D_KL(Q || P)）是“模式导向”的：Q专注于P的单一模式。

当你看到KL散度时：  
- 变分自编码器（VAE，ELBO中的KL项推动潜在分布向先验分布靠近）  
- 知识蒸馏（学生模型试图匹配教师模型的分布）  
- 基于人类反馈的强化学习（RLHF，KL惩罚项使微调模型保持接近基础模型）  
- 策略梯度方法（限制策略更新）

### Wasserstein距离（Earth Mover's Distance）

Wasserstein距离衡量将一个概率分布转换为另一个分布所需的最小“工作量”。可以这样理解：如果一个分布是一堆泥土，另一个是一个坑，你需要移动多少泥土，移动多远？

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

对于一维分布，它简化为累积分布函数的绝对差的积分：

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

为什么 Wasserstein 重要：
- 它是一个真正的度量（对称，满足三角不等式）
- 即使分布不重叠时，它也能提供梯度（KL 散度会趋于无穷大）
- 这一特性使其成为 Wasserstein GANs（WGANs）的核心，解决了原始 GANs 训练不稳定的问题

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何时使用 Wasserstein：
- GAN 训练（WGAN，WGAN-GP）
- 比较可能不重叠的分布
- 最优运输问题
- 图像检索（比较颜色直方图）

### 为什么不同任务需要不同的距离度量

| 任务 | 最佳距离 | 原因 |
|------|---------|------|
| 文本相似性 | Cosine | 幅度是噪声，方向是意义 |
| 图像像素比较 | L2 | 空间关系重要，特征具有可比较的尺度 |
| 稀疏高维特征 | L1 | 稳健，不会放大罕见的大幅差异 |
| 集合重叠（标签、类别） | Jaccard | 数据本质上是集合值，而非向量值 |
| 字符串匹配 | 编辑距离 | 操作映射到人类编辑的直觉 |
| 异常检测 | Mahalanobis | 考虑了特征的相关性和尺度 |
| 比较分布 | KL 散度 | 衡量使用 Q 而非 P 所丢失的信息 |
| GAN 训练 | Wasserstein | 即使分布不重叠时也能提供梯度 |
| 嵌入（向量数据库） | Cosine 或点积 | 嵌入被训练以在方向上编码意义 |
| 推荐 | 点积 | 幅度可以编码流行度或置信度 |
| DNA 序列 | 加权编辑距离 | 替换成本因核苷酸对而异 |
| 制造业质量控制 | L-无穷 | 任何维度的最坏情况偏差很重要 |

### 与损失函数的联系

损失函数是应用于预测值与目标值之间的距离函数。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 与正则化的联系

正则化在损失函数中对权重添加了一个范数惩罚。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

为什么 L1 会产生稀疏性而 L2 不会：想象在二维权重空间中约束区域的形状。L1 是一个菱形，L2 是一个圆形。损失函数的等高线（椭圆）最有可能在菱形的角上接触，此时一个权重为零。它们在圆形的平滑点上接触，此时两个权重都不为零。

### 最近邻搜索

每种距离函数都隐含着一个最近邻搜索问题：给定一个查询点，找到数据集中最近的点。

精确的最近邻搜索在包含 n 个点、d 维的数据集中，每次查询的时间复杂度为 O(n * d)。对于大规模数据集来说，这太慢了。

近似最近邻（Approximate Nearest Neighbor, ANN）算法通过牺牲一小部分精度来换取巨大的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```HNSW（Hierarchical Navigable Small World）是现代向量数据库中的主导算法。它构建一个多层图，其中每个节点连接到其近似的最近邻。搜索从顶层（稀疏，长跳）开始，并逐步下降到底层（密集，短跳）。

```figure
norm-unit-balls
```

## 构建它

### 步骤 1：所有规范和距离函数

请参见 `code/distances.py` 以查看完整的实现。每个函数都是从零开始使用基本的 Python 数学功能构建的。

### 步骤 2：相同数据，不同距离，不同邻居

`distances.py` 中的演示创建了一个数据集，选择了一个查询点，并展示了最近邻如何根据距离度量而变化。在 L1 下“最近”的点可能在 L2 或余弦下不是最近的。

### 步骤 3：嵌入相似性搜索

代码包含一个模拟的嵌入相似性搜索，使用余弦相似性与 L2 距离来查找与查询最相似的“文档”，展示了排名可能不同。

## 使用它

最常见的实际用途：在向量数据库中查找相似的项目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

当你调用 `model.encode(text)` 并随后搜索一个向量数据库时，幕后会发生以下过程。嵌入模型将文本映射到向量。向量数据库使用近似最近邻（ANN）算法，在你的查询向量和所有存储的向量之间计算余弦相似度（或点积），而无需逐一检查所有向量。

## 练习

1. 计算点 (1, 2, 3) 和 (4, 0, 6) 之间的 L1、L2 和 L-infinity 距离。验证对于任何两点对，L-inf <= L2 <= L1 总是成立。证明为什么这种排序是保证的。

2. 创建两个向量，它们的余弦相似度很高（> 0.9），但 L2 距离很大（> 10）。从几何角度解释这是怎么回事。然后创建两个向量，它们的余弦相似度很低（< 0.3），但 L2 距离很小（< 0.5）。

3. 实现一个函数，该函数接受一个数据集和一个查询点，返回在 L1、L2、余弦和马氏距离下的最近邻。找到一个数据集，使得这四种方法在哪个点是最近邻上存在分歧。

4. 使用 CDF 方法手动计算 [0.5, 0.5, 0, 0] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。然后计算 [0.25, 0.25, 0.25, 0.25] 和 [0, 0, 0.5, 0.5] 之间的 Wasserstein 距离。哪一个更大，为什么？

5. 实现 MinHash 以近似计算 Jaccard 相似度。生成 100 个随机集合，计算所有点对的精确 Jaccard 相似度，并使用 50、100 和 200 个哈希函数比较 MinHash 近似值。绘制近似误差。

## 关键术语

| 术语 | 人们常说 | 实际含义 |
|------|----------------|-------------------|
| 范数 | "向量的大小" | 一个函数，将向量映射到非负标量，满足三角不等式、绝对齐次性和零向量唯一为零 |
| L1 范数 | "曼哈顿距离" | 绝对分量值的总和。在优化中产生稀疏性。对异常值具有鲁棒性 |
| L2 范数 | "欧几里得距离" | 平方分量的总和的平方根。欧几里得空间中的直线距离 |
| Lp 范数 | "广义范数" | 绝对分量的 p 次幂的总和的 p 次根。L1 和 L2 是特殊情况 |
| L-infinity 范数 | "最大范数" 或 "切比雪夫距离" | 绝对分量的最大值。Lp 在 p 趋近于无穷大时的极限 |
| 余弦相似度 | "向量之间的角度" | 用两个模长归一化后的点积。范围从 -1 到 +1。忽略向量长度 |
| 余弦距离 | "1 减去余弦相似度" | 将余弦相似度转换为距离。范围从 0 到 2 |
| 点积 | "未归一化的余弦" | 分量乘积的总和。等于余弦相似度乘以两个模长 |
| 马氏距离 | "考虑相关性的距离" | 在使用数据协方差矩阵进行白化（去相关和归一化）后的空间中的 L2 距离 |
| Jaccard 相似度 | "集合重叠" | 交集大小除以并集大小。适用于集合，而非向量 |
| 编辑距离 | "莱文斯坦距离" | 将一个字符串转换为另一个字符串所需的最小插入、删除和替换次数 |
| KL 散度 | "分布之间的距离" | 不是真正的距离（非对称）。衡量使用 Q 编码 P 所需的额外比特数 |
| Wasserstein 距离 | "地球移动者距离" | 将一个分布的质移动到另一个分布所需的最小工作量。一个真正的度量 |
| 近似最近邻 | "ANN 搜索" | 算法（HNSW、LSH、IVF）比精确搜索更快地找到近似最近的点 |
| HNSW | "向量数据库算法" | 层次可导航小世界图。用于快速近似最近邻搜索的多层图 |
| L1 正则化 | "Lasso" | 将权重的 L1 范数加到损失函数中。驱动权重趋向于零（稀疏性） |
| L2 正则化 | "岭回归" 或 "权重衰减" | 将权重的平方 L2 范数加到损失函数中。使权重趋向于零但不稀疏 |
| 弹性网络 | "L1 + L2" | 结合 L1 和 L2 正则化。比单独使用任一方法更好地处理相关特征组 |

## 进一步阅读

- [FAISS: 一个用于高效相似性搜索的库](https://github.com/facebookresearch/faiss) - Meta 的用于十亿级近似最近邻搜索的库
- [Wasserstein GAN (Arjovsky 等人，2017)](https://arxiv.org/abs/1701.07875) - 将地球移动者距离引入 GAN 的论文
- [局部敏感哈希 (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基础的近似最近邻算法
- [高效词表示估计 (Mikolov 等人，2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，其中余弦相似度成为嵌入的默认方法
- [sklearn.neighbors 文档](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距离度量和邻居算法的实用指南
