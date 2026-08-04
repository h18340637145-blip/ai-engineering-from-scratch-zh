# 特征选择

> 更多的特征并不一定更好。正确的特征才是更好的。

**类型:** 构建
**语言:** Python
**前提条件:** 第二阶段，第 01-09 课，第 08 课（特征工程）
**时间:** ~75 分钟

## 学习目标

- 从头开始实现过滤方法（方差阈值、互信息、卡方检验）和包装方法（RFE、前向选择）
- 解释为什么互信息能够捕捉相关系数无法捕捉的非线性特征-目标关系
- 比较 L1 正则化（嵌入式选择）与 RFE（包装式选择），并评估它们的计算权衡
- 构建一个结合多种方法的特征选择流水线，并在保留数据上展示其泛化性能的提升

## 问题

你有 500 个特征。你的模型训练缓慢，经常过拟合，而且没人能解释它学到了什么。你添加了更多特征，希望提升性能。结果变得更差。

这是维度灾难的体现。随着特征数量的增加，特征空间的体积急剧膨胀。数据点变得稀疏。点之间的距离趋近于相同。模型需要指数级更多的数据才能发现真正的模式。噪声特征掩盖了信号特征。过拟合成为默认状态。

特征选择是解药。去除噪声。消除冗余。保留那些对目标变量携带实际信息的特征。结果：更快的训练，更好的泛化，以及可以实际解释的模型。

目标不是使用所有可用的信息。而是使用正确的信息。

## 概念

### 特征选择的三类

每个特征选择方法都属于以下三类中的一种：```mermaid
flowchart TD
    A[Feature Selection Methods] --> B[Filter Methods]
    A --> C[Wrapper Methods]
    A --> D[Embedded Methods]

    B --> B1["Variance Threshold"]
    B --> B2["Mutual Information"]
    B --> B3["Chi-squared Test"]
    B --> B4["Correlation Filtering"]

    C --> C1["Recursive Feature Elimination"]
    C --> C2["Forward Selection"]
    C --> C3["Backward Elimination"]

    D --> D1["L1 / Lasso Regularization"]
    D --> D2["Tree-based Importance"]
    D --> D3["Elastic Net"]
```**过滤方法** 使用统计量独立地对每个特征进行评分。它们不使用模型。速度快，但会忽略特征之间的相互作用。

**包装方法** 训练一个模型来评估特征子集。它们使用模型性能作为评分标准。结果更好，但代价较高，因为需要多次重新训练模型。

**嵌入方法** 在模型训练过程中选择特征。L1 正则化会将权重驱动为零。决策树在最有用的特征上进行划分。特征选择发生在拟合过程中，而不是作为单独的步骤。

### 方差阈值

最简单的过滤方法。如果一个特征在样本之间几乎没有变化，那么它几乎不携带任何信息。

考虑一个特征，在1000个样本中有999个样本的值为0.0。它的方差接近于零。没有任何模型可以利用它来区分类别。应该将其删除。```
variance(x) = mean((x - mean(x))^2)
```设置一个阈值（例如，0.01）。删除所有方差低于该阈值的特征。这种方法无需查看目标变量，即可去除常数或接近常数的特征。

使用场景：作为其他方法之前的预处理步骤。它几乎不耗费成本，即可识别明显无用的特征。

局限性：一个特征可能具有高方差，但仍可能是纯粹的噪声。方差阈值是必要的，但并不充分。

### 互信息

互信息衡量的是，已知特征 X 的值在多大程度上能减少对目标 Y 的不确定性。```
I(X; Y) = sum_x sum_y p(x, y) * log(p(x, y) / (p(x) * p(y)))
```如果 X 和 Y 是独立的，那么 p(x, y) = p(x) * p(y)，所以对数项为零，I(X; Y) = 0。X 越能告诉你关于 Y 的信息，互信息就越高。

与相关性相比的关键优势：互信息能够捕捉非线性关系。一个特征可能与目标变量的相关系数为零，但互信息很高，因为它们之间的关系可能是二次的或周期性的。

对于连续特征，首先将其离散化为若干个区间（基于直方图的估计）。区间的数量会影响估计结果——区间太少会丢失信息，太多则会引入噪声。常用的选择方法：使用 sqrt(n) 个区间，或使用斯特格斯规则（Sturges' rule）：1 + log2(n)。```mermaid
flowchart LR
    A[Feature X] --> B[Discretize into Bins]
    B --> C["Compute Joint Distribution p(x,y)"]
    C --> D["Compute MI = sum p(x,y) * log(p(x,y) / p(x)p(y))"]
    D --> E["Rank Features by MI Score"]
    E --> F[Select Top K]
```### 递归特征消除（RFE）

RFE 是一种包装方法。它使用模型自身的特征重要性来迭代地修剪：

1. 使用所有特征训练模型
2. 按重要性对特征进行排序（线性模型使用系数，树模型使用不纯度减少量）
3. 移除最不重要的特征
4. 重复直到剩余所需数量的特征```mermaid
flowchart TD
    A["Start: All N Features"] --> B["Train Model"]
    B --> C["Rank Feature Importances"]
    C --> D["Remove Least Important"]
    D --> E{"Features == Target Count?"}
    E -->|No| B
    E -->|Yes| F["Return Selected Features"]
```RFE 考虑特征交互，因为模型会同时看到所有剩余特征。移除一个特征会改变其他特征的重要性。这使它比过滤方法更彻底。

代价：你需要训练模型 N - target 次。如果有 500 个特征，目标为 10，那就是 490 次训练运行。对于昂贵的模型来说，这会很慢。你可以通过每一步移除多个特征来加速（例如，每次移除底部的 10%）。

### L1（Lasso）正则化

L1 正则化将权重的绝对值添加到损失函数中：```
loss = prediction_error + alpha * sum(|w_i|)
```alpha 参数控制特征被剪枝的激进程度。alpha 值越高，意味着更多的权重会被精确地置零。

为什么是精确地置零？L1 惩罚在权重空间中创建了一个钻石形状的约束区域。最优解倾向于落在这个钻石的角落，此时一个或多个权重为零。L2 正则化（岭回归）创建了一个圆形的约束区域，权重会缩小但很少会达到零。

这是嵌入式的特征选择：模型在训练过程中学习忽略哪些特征。权重为零的特征实际上被移除了。

优势：单次训练运行，处理相关特征（选择其中一个，将其他置零），大多数线性模型实现中都内置了该功能。

局限性：仅适用于线性模型。无法捕捉非线性特征的重要性。

### 基于树的特征重要性

决策树及其集成方法（随机森林、梯度提升）自然地对特征进行排序。每一次分裂都会减少不纯度（分类使用基尼指数或熵，回归使用方差）。产生较大不纯度减少的特征更为重要。

对于包含 T 棵树的随机森林：```
importance(feature_j) = (1/T) * sum over all trees of
    sum over all nodes splitting on feature_j of
        (n_samples * impurity_decrease)
```这为每个特征提供了一个归一化的的重要性评分。它可以自动处理非线性关系和特征交互。

注意：基于树的重要性评分倾向于偏向具有许多唯一值（高基数）的特征。一个随机ID列会显得很重要，因为它可以完美地分割每一个样本。使用置换重要性作为合理性检查。

### 置换重要性

一种模型无关的方法：

1. 训练模型并在验证数据上记录基准性能
2. 对于每个特征：随机打乱其值，测量性能的下降
3. 下降越大，特征越重要

如果打乱某个特征不会影响性能，说明模型不依赖于它。如果性能崩溃，说明该特征至关重要。

置换重要性避免了基于树的重要性评分中的基数偏差。但是它速度较慢：每个特征需要一次完整的评估，为了稳定性需要多次重复。

### 对比表格

| 方法 | 类型 | 速度 | 非线性 | 特征交互 |
|--------|------|-------|---------|---------|
| 方差阈值 | 过滤 | 非常快 | 否 | 否 |
| 互信息 | 过滤 | 快 | 是 | 否 |
| 相关性过滤 | 过滤 | 快 | 否 | 否 |
| 递归特征消除（RFE） | 包裹 | 慢 | 依赖模型 | 是 |
| L1 / Lasso | 嵌入 | 快 | 否（线性） | 否 |
| 树重要性 | 嵌入 | 中等 | 是 | 是 |
| 置换重要性 | 模型无关 | 慢 | 是 | 是 |

### 决策流程图```mermaid
flowchart TD
    A[Start: Feature Selection] --> B{How many features?}
    B -->|"< 50"| C["Start with variance threshold + mutual information"]
    B -->|"50-500"| D["Variance threshold, then L1 or tree importance"]
    B -->|"> 500"| E["Variance threshold, then mutual info filter, then RFE on survivors"]

    C --> F{Using linear model?}
    D --> F
    E --> F

    F -->|Yes| G["L1 regularization for final selection"]
    F -->|No - trees| H["Tree importance + permutation importance"]
    F -->|No - other| I["RFE with your model"]

    G --> J[Validate: compare selected vs all features]
    H --> J
    I --> J

    J --> K{Performance improved?}
    K -->|Yes| L["Ship with selected features"]
    K -->|No| M["Try different method or keep all features"]
```## 构建它

### 步骤 1：生成具有已知特征结构的合成数据```python
import numpy as np


def make_feature_selection_data(n_samples=500, seed=42):
    rng = np.random.RandomState(seed)

    x1 = rng.randn(n_samples)
    x2 = rng.randn(n_samples)
    x3 = rng.randn(n_samples)
    x4 = x1 + 0.1 * rng.randn(n_samples)
    x5 = x2 + 0.1 * rng.randn(n_samples)

    informative = np.column_stack([x1, x2, x3, x4, x5])

    correlated = np.column_stack([
        x1 * 0.9 + 0.1 * rng.randn(n_samples),
        x2 * 0.8 + 0.2 * rng.randn(n_samples),
        x3 * 0.7 + 0.3 * rng.randn(n_samples),
        x1 * 0.5 + x2 * 0.5 + 0.1 * rng.randn(n_samples),
        x2 * 0.6 + x3 * 0.4 + 0.1 * rng.randn(n_samples),
    ])

    noise = rng.randn(n_samples, 10) * 0.5

    X = np.hstack([informative, correlated, noise])
    y = (2 * x1 - 1.5 * x2 + x3 + 0.5 * rng.randn(n_samples) > 0).astype(int)

    feature_names = (
        [f"info_{i}" for i in range(5)]
        + [f"corr_{i}" for i in range(5)]
        + [f"noise_{i}" for i in range(10)]
    )

    return X, y, feature_names
```我们知道真实情况：特征 0-4 是有信息的（其中 3 和 4 是 0 和 1 的相关副本），特征 5-9 与有信息的特征相关，特征 10-19 是纯粹的噪声。一个好的选择方法应该将 0-4 的排名最高，将 10-19 的排名最低。

### 步骤 2：方差阈值```python
def variance_threshold(X, threshold=0.01):
    variances = np.var(X, axis=0)
    mask = variances > threshold
    return mask, variances
```### 步骤 3：互信息（离散）```python
def discretize(x, n_bins=10):
    min_val, max_val = x.min(), x.max()
    if max_val == min_val:
        return np.zeros_like(x, dtype=int)
    bin_edges = np.linspace(min_val, max_val, n_bins + 1)
    binned = np.digitize(x, bin_edges[1:-1])
    return binned


def mutual_information(X, y, n_bins=10):
    n_samples, n_features = X.shape
    mi_scores = np.zeros(n_features)

    y_vals, y_counts = np.unique(y, return_counts=True)
    p_y = y_counts / n_samples

    for f in range(n_features):
        x_binned = discretize(X[:, f], n_bins)
        x_vals, x_counts = np.unique(x_binned, return_counts=True)
        p_x = dict(zip(x_vals, x_counts / n_samples))

        mi = 0.0
        for xv in x_vals:
            for yi, yv in enumerate(y_vals):
                joint_mask = (x_binned == xv) & (y == yv)
                p_xy = np.sum(joint_mask) / n_samples
                if p_xy > 0:
                    mi += p_xy * np.log(p_xy / (p_x[xv] * p_y[yi]))
        mi_scores[f] = mi

    return mi_scores
```### 步骤 4：递归特征消除```python
def simple_logistic_importance(X, y, lr=0.1, epochs=100):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y
        w -= lr * (X.T @ error) / n_samples
        b -= lr * np.mean(error)

    return w, b


def rfe(X, y, n_features_to_select=5, lr=0.1, epochs=100):
    n_total = X.shape[1]
    remaining = list(range(n_total))
    rankings = np.ones(n_total, dtype=int)
    rank = n_total

    while len(remaining) > n_features_to_select:
        X_subset = X[:, remaining]
        w, _ = simple_logistic_importance(X_subset, y, lr, epochs)
        importances = np.abs(w)

        least_idx = np.argmin(importances)
        original_idx = remaining[least_idx]
        rankings[original_idx] = rank
        rank -= 1
        remaining.pop(least_idx)

    for idx in remaining:
        rankings[idx] = 1

    selected_mask = rankings == 1
    return selected_mask, rankings
```### 步骤 5：L1 特征选择```python
def soft_threshold(w, alpha):
    return np.sign(w) * np.maximum(np.abs(w) - alpha, 0)


def l1_feature_selection(X, y, alpha=0.1, lr=0.01, epochs=500):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
        error = pred - y

        gradient_w = (X.T @ error) / n_samples
        gradient_b = np.mean(error)

        w -= lr * gradient_w
        w = soft_threshold(w, lr * alpha)
        b -= lr * gradient_b

    selected_mask = np.abs(w) > 1e-6
    return selected_mask, w
```### 第6步：基于树的重要性（简单决策树）```python
def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1.0 - np.sum(probs ** 2)


def best_split(X, y, feature_idx):
    values = np.unique(X[:, feature_idx])
    if len(values) <= 1:
        return None, -1.0

    best_threshold = None
    best_gain = -1.0
    parent_gini = gini_impurity(y)
    n = len(y)

    for i in range(len(values) - 1):
        threshold = (values[i] + values[i + 1]) / 2.0
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask

        n_left = np.sum(left_mask)
        n_right = np.sum(right_mask)

        if n_left == 0 or n_right == 0:
            continue

        gain = parent_gini - (n_left / n) * gini_impurity(y[left_mask]) - (n_right / n) * gini_impurity(y[right_mask])

        if gain > best_gain:
            best_gain = gain
            best_threshold = threshold

    return best_threshold, best_gain


def tree_importance(X, y, n_trees=50, max_depth=5, seed=42):
    rng = np.random.RandomState(seed)
    n_samples, n_features = X.shape
    importances = np.zeros(n_features)

    for _ in range(n_trees):
        sample_idx = rng.choice(n_samples, size=n_samples, replace=True)
        feature_subset = rng.choice(n_features, size=max(1, int(np.sqrt(n_features))), replace=False)

        X_boot = X[sample_idx]
        y_boot = y[sample_idx]

        tree_imp = _build_tree_importance(X_boot, y_boot, feature_subset, max_depth)
        importances += tree_imp

    total = importances.sum()
    if total > 0:
        importances /= total

    return importances


def _build_tree_importance(X, y, feature_subset, max_depth, depth=0):
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    if depth >= max_depth or len(np.unique(y)) <= 1 or len(y) < 4:
        return importances

    best_feature = None
    best_threshold = None
    best_gain = -1.0

    for f in feature_subset:
        threshold, gain = best_split(X, y, f)
        if gain > best_gain:
            best_gain = gain
            best_feature = f
            best_threshold = threshold

    if best_feature is None or best_gain <= 0:
        return importances

    importances[best_feature] += best_gain * len(y)

    left_mask = X[:, best_feature] <= best_threshold
    right_mask = ~left_mask

    importances += _build_tree_importance(X[left_mask], y[left_mask], feature_subset, max_depth, depth + 1)
    importances += _build_tree_importance(X[right_mask], y[right_mask], feature_subset, max_depth, depth + 1)

    return importances
```### 第7步：运行所有方法并进行比较

该代码文件在相同的合成数据集上运行所有五种方法，并打印一个比较表，显示每种方法选择的特征。

## 使用方法

使用scikit-learn，特征选择已集成到流程中：```python
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_classif,
    RFE,
    SelectFromModel,
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestClassifier

vt = VarianceThreshold(threshold=0.01)
X_filtered = vt.fit_transform(X)

mi_scores = mutual_info_classif(X, y)
top_k = np.argsort(mi_scores)[-10:]

rfe_selector = RFE(LogisticRegression(), n_features_to_select=10)
rfe_selector.fit(X, y)
X_rfe = rfe_selector.transform(X)

lasso_selector = SelectFromModel(Lasso(alpha=0.01))
lasso_selector.fit(X, y)
X_lasso = lasso_selector.transform(X)

rf = RandomForestClassifier(n_estimators=100)
rf.fit(X, y)
importances = rf.feature_importances_
```从零开始的实现展示了每个方法内部发生的确切过程。方差阈值只是计算 `var(X, axis=0)` 并应用掩码。互信息是在列联表中计算联合频率和边缘频率。RFE 是一个循环，依次训练、排序和修剪。L1 是带有软阈值步骤的梯度下降。树重要性是跨分割累积不纯度的减少。没有魔法，只有统计和循环。

sklearn 的版本增加了鲁棒性（例如，mutual_info_classif 使用 k-NN 密度估计而不是分箱）、速度（C 实现）和管道集成。

## 发布它

这节课生成以下内容：
- `outputs/skill-feature-selector.md` -- 一个快速参考决策树，用于选择正确的特征选择方法

## 练习

1. **向前选择**：实现 RFE 的相反方法。从零个特征开始。在每一步中，添加对模型性能提升最大的特征。当添加特征不再有帮助时停止。将所选特征与 RFE 结果进行比较。哪个更快？哪个结果更好？

2. **稳定性选择**：运行 L1 特征选择 50 次，每次在数据的随机 80% 子样本上运行，使用略微不同的 alpha 值。统计每个特征被选中的次数。在超过 80% 的运行中被选中的特征被认为是“稳定的”。将稳定的特征与单次运行的 L1 选择进行比较。哪个更可靠？

3. **多重共线性检测**：计算所有特征的相关矩阵。实现一个函数，给定一个相关阈值（例如，0.9），从每个高度相关的特征对中删除一个特征（保留与目标具有更高互信息的特征）。在合成数据集上测试该函数，并验证它是否删除了冗余的相关特征。

4. **特征选择管道**：将方差阈值、互信息过滤器和 RFE 链接成一个单一的管道。首先删除近似零方差的特征，然后保留互信息最高的 50% 的特征，然后对剩余的特征运行 RFE。将该管道与仅对所有特征运行 RFE 进行比较。该管道是否更快？是否同样准确？

5. **从零开始的置换重要性**：实现置换重要性。对每个特征，将其值洗牌 10 次，测量 F1 分数的平均下降。将排名与基于树的重要性进行比较。找到它们意见不一致的情况，并解释原因（提示：相关特征）。

## 关键术语

| 术语 | 人们怎么说 | 它实际意味着 |
|------|----------------|----------------|
| 过滤方法 | “独立评分特征” | 一种特征选择方法，使用统计度量对特征进行排序，而不训练模型，单独评估每个特征 |
| 包装方法 | “用模型选择特征” | 一种特征选择方法，通过训练模型并使用其性能作为选择标准来评估特征子集 |
| 嵌入方法 | “模型在训练过程中选择特征” | 特征选择作为模型拟合的一部分发生，例如 L1 正则化将权重驱动到零 |
| 互信息 | “一个变量告诉你的关于另一个变量的信息量” | 在已知 X 的情况下，对 Y 的不确定性减少的度量，捕捉线性和非线性依赖 |
| 递归特征消除 | “训练、排序、修剪、重复” | 一种迭代包装方法，训练模型，删除最不重要的特征，并重复直到达到目标数量 |
| L1 / Lasso 正则化 | “杀死特征的惩罚项” | 将权重的绝对值之和添加到损失函数中，使不重要的特征权重精确为零 |
| 方差阈值 | “删除常量特征” | 删除样本方差低于指定阈值的特征，过滤掉不携带信息的特征 |
| 特征重要性 | “哪些特征最重要” | 一个分数，表示每个特征对模型预测的贡献程度，从分割增益（树）或系数幅度（线性）计算得出 |
| 置换重要性 | “洗牌并测量损失” | 通过随机洗牌每个特征的值并测量模型性能的下降来评估特征重要性 |
| 维度灾难 | “特征太多，数据太少” | 添加特征会使特征空间的体积呈指数增长，导致数据稀疏和距离无意义的现象 |

## 进一步阅读

- [变量和特征选择简介（Guyon & Elisseeff, 2003）](https://jmlr.org/papers/v3/guyon03a.html) -- 特征选择方法的奠基性综述，至今仍被广泛引用
- [scikit-learn 特征选择指南](https://scikit-learn.org/stable/modules/feature_selection.html) -- 包含过滤、包装和嵌入方法的实用参考，附有代码示例
- [稳定性选择（Meinshausen & Buhlmann, 2010）](https://arxiv.org/abs/0809.2932) -- 结合子采样与特征选择，以获得稳健、可重复的结果
- [警惕默认的随机森林重要性（Strobl 等，2007）](https://bmcbioinformatics.biomedcentral.com/articles/10.1186/1471-2105-8-25) -- 展示了基于树的重要性中的基数偏差，并提出条件重要性作为替代方案
