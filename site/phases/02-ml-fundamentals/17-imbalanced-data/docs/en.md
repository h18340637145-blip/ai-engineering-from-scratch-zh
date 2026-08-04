# 处理不平衡数据

> 当你的数据中 99% 都是“正常”时，准确率是一个谎言。

**类型:** 构建
**语言:** Python
**前提条件:** 第二阶段，第 01-09 课（尤其是评估指标）
**时间:** ~90 分钟

## 学习目标

- 从零开始实现 SMOTE，并解释合成过采样与随机复制的区别
- 使用 F1、AUPRC 和马修斯相关系数（Matthews Correlation Coefficient）代替准确率来评估不平衡分类器
- 比较类别权重、阈值调整和重采样策略，并为给定的不平衡比例选择合适的方法
- 构建一个完整的不平衡数据管道，结合 SMOTE、类别权重和阈值优化

## 问题

你构建了一个欺诈检测模型。它获得了 99.9% 的准确率。你庆祝。然后你意识到它对每笔交易都预测为“非欺诈”。

这不是一个错误。当只有 0.1% 的交易是欺诈时，这是合理的行为。模型学会了总是猜测多数类可以最小化总体错误。虽然技术上是正确的，但完全没用。

这在所有实际分类中都可能发生。疾病诊断：1% 的阳性率。网络入侵：0.01% 的攻击。制造缺陷：0.5% 的缺陷。垃圾邮件过滤：20% 的垃圾邮件。客户流失预测：5% 的流失者。少数类越重要，它就越稀有。

准确率失败的原因是它将所有正确预测视为同等重要。正确标记一个合法交易和正确识别欺诈都算作一个准确点。但识别欺诈是模型存在的全部原因。我们需要指标、技术和训练策略，迫使模型关注稀有但重要的类别。

## 概念

### 为什么准确率失败

考虑一个有 1000 个样本的数据集：990 个负样本，10 个正样本。一个总是预测负的模型：

|  | 预测为正 | 预测为负 |
|--|---|---|
| 实际为正 | 0 (TP) | 10 (FN) |
| 实际为负 | 0 (FP) | 990 (TN) |

准确率 = (0 + 990) / 1000 = 99.0%

该模型没有识别任何欺诈。没有识别任何疾病。没有识别任何缺陷。但准确率显示 99%。这就是为什么准确率对不平衡问题很危险。

### 更好的指标

**精确率** = TP / (TP + FP)。所有标记为正的样本中，有多少是真正的正样本？高精确率意味着较少的误报。

**召回率** = TP / (TP + FN)。所有实际为正的样本中，我们识别出了多少？高召回率意味着较少的漏报。

**F1 分数** = 2 * 精确率 * 召回率 / (精确率 + 召回率)。调和平均数。相比算术平均数，对精确率和召回率之间的极端不平衡惩罚更多。

**F-beta 分数** = (1 + beta^2) * 精确率 * 召回率 / (beta^2 * 精确率 + 召回率)。当 beta > 1 时，召回率更重要；当 beta < 1 时，精确率更重要。F2 在欺诈检测中很常见（漏掉欺诈比误报更糟糕）。

**AUPRC**（精确率-召回率曲线下的面积）。类似于 AUC-ROC，但对不平衡数据更有信息量。一个随机分类器的 AUPRC 等于正类比例（不是像 ROC 中的 0.5）。这使得改进更容易被看到。

**马修斯相关系数** = (TP * TN - FP * FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))。范围从 -1 到 +1。只有在模型在两个类别上表现良好时才给出高分。即使类别大小差异很大，也是平衡的。

对于上面的“总是预测为负”模型：精确率 = 0/0（未定义，通常设置为 0），召回率 = 0/10 = 0，F1 = 0，MCC = 0。这些指标正确地识别了模型毫无价值。

### 不平衡数据管道```mermaid
flowchart TD
    A[Imbalanced Dataset] --> B{Imbalance Ratio?}
    B -->|Mild: 80/20| C[Class Weights]
    B -->|Moderate: 95/5| D[SMOTE + Threshold Tuning]
    B -->|Severe: 99/1| E[SMOTE + Class Weights + Threshold]
    C --> F[Train Model]
    D --> F
    E --> F
    F --> G[Evaluate with F1 / AUPRC / MCC]
    G --> H{Good Enough?}
    H -->|No| I[Try Different Strategy]
    H -->|Yes| J[Deploy with Monitoring]
    I --> B
```### SMOTE: 合成少数类过采样技术

随机过采样会复制现有的少数类样本。这虽然有效，但存在过拟合的风险，因为模型会反复看到相同的样本点。

SMOTE 会生成新的合成少数类样本，这些样本是合理的，但不是复制的。该算法：

1. 对于每个少数类样本 x，在其他少数类样本中找到其 k 个最近邻
2. 随机选择其中一个邻居
3. 在 x 与该邻居之间的线段上生成一个新样本

公式：`new_sample = x + random(0, 1) * (neighbor - x)`

这在真实少数类点之间进行插值，生成特征空间中相同区域的样本，而不仅仅是复制现有数据。```mermaid
flowchart LR
    subgraph Original["Original Minority Points"]
        P1["x1 (1.0, 2.0)"]
        P2["x2 (1.5, 2.5)"]
        P3["x3 (2.0, 1.5)"]
    end
    subgraph SMOTE["SMOTE Generation"]
        direction TB
        S1["Pick x1, neighbor x2"]
        S2["random t = 0.4"]
        S3["new = x1 + 0.4*(x2-x1)"]
        S4["new = (1.2, 2.2)"]
        S1 --> S2 --> S3 --> S4
    end
    Original --> SMOTE
    subgraph Result["Augmented Set"]
        R1["x1 (1.0, 2.0)"]
        R2["x2 (1.5, 2.5)"]
        R3["x3 (2.0, 1.5)"]
        R4["synthetic (1.2, 2.2)"]
    end
    SMOTE --> Result
```### 采样策略比较

**随机过采样**：复制少数类样本以匹配多数类数量。
- 优点：简单，不丢失信息
- 缺点：完全复制会导致过拟合，增加训练时间

**随机欠采样**：移除多数类样本以匹配少数类数量。
- 优点：训练速度快，简单
- 缺点：丢弃了可能有用的多数类数据，方差更高

**SMOTE**：通过插值生成合成的少数类样本。
- 优点：生成新的数据点，相比随机过采样减少了过拟合
- 缺点：在决策边界附近可能生成噪声样本，不考虑多数类分布

| 策略 | 数据改变 | 风险 | 何时使用 |
|--|--|--|--|
| 过采样 | 少数类复制 | 过拟合 | 小数据集，中等不平衡 |
| 欠采样 | 多数类移除 | 信息丢失 | 大数据集，希望快速训练 |
| SMOTE | 合成少数类添加 | 边界噪声 | 中等不平衡，有足够的少数类样本用于k-NN |

### 类别权重

与其改变数据，不如改变模型对错误的处理方式。对少数类的错误分类赋予更高的权重。

对于一个包含950个负样本和50个正样本的二分类问题：
- 负类权重 = n_samples / (2 * n_negative) = 1000 / (2 * 950) = 0.526
- 正类权重 = n_samples / (2 * n_positive) = 1000 / (2 * 50) = 10.0

正类的权重是负类的19倍。将一个正类样本错误分类的成本相当于将19个负类样本错误分类的成本。模型被迫关注少数类。

在逻辑回归中，这会修改损失函数：```
weighted_loss = -sum(w_i * [y_i * log(p_i) + (1-y_i) * log(1-p_i)])
```其中 $ w_i $ 依赖于样本 $ i $ 的类别。

类别权重在数学上等价于期望意义上的过采样，但不会生成新的数据点。这使得它们更快，并避免了复制样本所带来的过拟合风险。

### 阈值调节

大多数分类器输出一个概率。默认阈值是 0.5：如果 $ P(\text{positive}) \geq 0.5 $，则预测为正类。但 0.5 是任意设定的。当类别不平衡时，最佳阈值通常要低得多。

过程如下：
1. 训练一个模型
2. 在验证集上获取预测概率
3. 从 0.0 到 1.0 扫描阈值
4. 在每个阈值下计算 F1（或你选择的指标）
5. 选择使你的指标最大化的阈值```mermaid
flowchart LR
    A[Model] --> B[Predict Probabilities]
    B --> C[Sweep Thresholds 0.0 to 1.0]
    C --> D[Compute F1 at Each]
    D --> E[Pick Best Threshold]
    E --> F[Use in Production]
```一个模型可能对一笔欺诈交易输出 P(fraud) = 0.15。在阈值为 0.5 时，该交易会被归类为非欺诈。而在阈值为 0.10 时，该交易会被正确识别。概率校准的重要性不如排序——只要欺诈交易的概率高于非欺诈交易的概率，就存在一个可以将它们区分开的阈值。

### 成本敏感学习

类别权重的推广。不再使用统一的成本，而是分配特定的分类错误成本：

| | 预测为正类 | 预测为负类 |
|--|---|---|
| 实际为正类 | 0（正确） | C_FN = 100 |
| 实际为负类 | C_FP = 1 | 0（正确） |

漏掉一笔欺诈交易（FN）的成本是误报（FP）的 100 倍。模型优化的是总成本，而不是总错误数量。

当你能够估算现实世界中的成本时，这是最合理的方法。漏诊癌症的成本与因误报而进行额外活检的成本有非常大的差异。明确这些成本可以促使做出正确的权衡。

### 决策流程图```mermaid
flowchart TD
    A[Start: Imbalanced Dataset] --> B{How imbalanced?}
    B -->|"< 70/30"| C["Mild: try class weights first"]
    B -->|"70/30 to 95/5"| D["Moderate: SMOTE + class weights"]
    B -->|"> 95/5"| E["Severe: combine multiple strategies"]
    C --> F{Enough data?}
    D --> F
    E --> F
    F -->|"< 1000 samples"| G["Oversample or SMOTE, avoid undersampling"]
    F -->|"1000-10000"| H["SMOTE + threshold tuning"]
    F -->|"> 10000"| I["Undersampling OK, or class weights"]
    G --> J[Train + Evaluate with F1/AUPRC]
    H --> J
    I --> J
    J --> K{Recall high enough?}
    K -->|No| L[Lower threshold]
    K -->|Yes| M{Precision acceptable?}
    M -->|No| N[Raise threshold or add features]
    M -->|Yes| O[Ship it]
```

```figure
class-imbalance
```## 构建它

### 步骤 1：生成一个不平衡的数据集```python
import numpy as np


def make_imbalanced_data(n_majority=950, n_minority=50, seed=42):
    rng = np.random.RandomState(seed)

    X_maj = rng.randn(n_majority, 2) * 1.0 + np.array([0.0, 0.0])
    X_min = rng.randn(n_minority, 2) * 0.8 + np.array([2.5, 2.5])

    X = np.vstack([X_maj, X_min])
    y = np.concatenate([np.zeros(n_majority), np.ones(n_minority)])

    shuffle_idx = rng.permutation(len(y))
    return X[shuffle_idx], y[shuffle_idx]
```### 步骤 2：从零开始实现 SMOTE```python
def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b) ** 2))


def find_k_neighbors(X, idx, k):
    distances = []
    for i in range(len(X)):
        if i == idx:
            continue
        d = euclidean_distance(X[idx], X[i])
        distances.append((i, d))
    distances.sort(key=lambda x: x[1])
    return [d[0] for d in distances[:k]]


def smote(X_minority, k=5, n_synthetic=100, seed=42):
    rng = np.random.RandomState(seed)
    n_samples = len(X_minority)
    k = min(k, n_samples - 1)
    synthetic = []

    for _ in range(n_synthetic):
        idx = rng.randint(0, n_samples)
        neighbors = find_k_neighbors(X_minority, idx, k)
        neighbor_idx = neighbors[rng.randint(0, len(neighbors))]
        t = rng.random()
        new_point = X_minority[idx] + t * (X_minority[neighbor_idx] - X_minority[idx])
        synthetic.append(new_point)

    return np.array(synthetic)
```### 步骤 3：随机过采样和欠采样```python
def random_oversample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    max_count = counts.max()

    X_resampled = list(X)
    y_resampled = list(y)

    for cls, count in zip(classes, counts):
        if count < max_count:
            cls_indices = np.where(y == cls)[0]
            n_needed = max_count - count
            chosen = rng.choice(cls_indices, size=n_needed, replace=True)
            X_resampled.extend(X[chosen])
            y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]


def random_undersample(X, y, seed=42):
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(y, return_counts=True)
    min_count = counts.min()

    X_resampled = []
    y_resampled = []

    for cls in classes:
        cls_indices = np.where(y == cls)[0]
        chosen = rng.choice(cls_indices, size=min_count, replace=False)
        X_resampled.extend(X[chosen])
        y_resampled.extend(y[chosen])

    X_out = np.array(X_resampled)
    y_out = np.array(y_resampled)
    shuffle = rng.permutation(len(y_out))
    return X_out[shuffle], y_out[shuffle]
```### 步骤 4：带类别权重的逻辑回归```python
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def logistic_regression_weighted(X, y, weights, lr=0.01, epochs=200):
    n_samples, n_features = X.shape
    w = np.zeros(n_features)
    b = 0.0

    for _ in range(epochs):
        z = X @ w + b
        pred = sigmoid(z)
        error = pred - y
        weighted_error = error * weights

        gradient_w = (X.T @ weighted_error) / n_samples
        gradient_b = np.mean(weighted_error)

        w -= lr * gradient_w
        b -= lr * gradient_b

    return w, b


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    n_samples = len(y)
    n_classes = len(classes)
    weight_map = {}
    for cls, count in zip(classes, counts):
        weight_map[cls] = n_samples / (n_classes * count)
    return np.array([weight_map[yi] for yi in y])
```### 步骤 5：阈值调整```python
def find_optimal_threshold(y_true, y_probs, metric="f1"):
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.05, 0.96, 0.01):
        y_pred = (y_probs >= threshold).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))

        if metric == "f1":
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        elif metric == "recall":
            score = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        elif metric == "precision":
            score = tp / (tp + fp) if (tp + fp) > 0 else 0.0

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
```### 步骤 6：评估函数```python
def confusion_matrix_values(y_true, y_pred):
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    return tp, tn, fp, fn


def compute_metrics(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix_values(y_true, y_pred)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = (tp * tn - fp * fn) / denom if denom > 0 else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mcc": mcc,
    }
```### 步骤 7：比较所有方法```python
X, y = make_imbalanced_data(950, 50, seed=42)
split = int(0.8 * len(y))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Baseline: no treatment
w_base, b_base = logistic_regression_weighted(
    X_train, y_train, np.ones(len(y_train)), lr=0.1, epochs=300
)
probs_base = sigmoid(X_test @ w_base + b_base)
preds_base = (probs_base >= 0.5).astype(int)

# Oversampled
X_over, y_over = random_oversample(X_train, y_train)
w_over, b_over = logistic_regression_weighted(
    X_over, y_over, np.ones(len(y_over)), lr=0.1, epochs=300
)
preds_over = (sigmoid(X_test @ w_over + b_over) >= 0.5).astype(int)

# SMOTE
minority_mask = y_train == 1
X_minority = X_train[minority_mask]
synthetic = smote(X_minority, k=5, n_synthetic=len(y_train) - 2 * int(minority_mask.sum()))
X_smote = np.vstack([X_train, synthetic])
y_smote = np.concatenate([y_train, np.ones(len(synthetic))])
w_sm, b_sm = logistic_regression_weighted(
    X_smote, y_smote, np.ones(len(y_smote)), lr=0.1, epochs=300
)
preds_smote = (sigmoid(X_test @ w_sm + b_sm) >= 0.5).astype(int)

# Class weights
sample_weights = compute_class_weights(y_train)
w_cw, b_cw = logistic_regression_weighted(
    X_train, y_train, sample_weights, lr=0.1, epochs=300
)
probs_cw = sigmoid(X_test @ w_cw + b_cw)
preds_cw = (probs_cw >= 0.5).astype(int)

# Threshold tuning (tune on held-out validation set, not test set)
probs_val = sigmoid(X_val @ w_cw + b_cw)
best_thresh, best_f1 = find_optimal_threshold(y_val, probs_val, metric="f1")
preds_thresh = (probs_cw >= best_thresh).astype(int)
```该代码文件在一个脚本中运行所有这些内容并打印结果。

## 使用方法

使用 scikit-learn 和 imbalanced-learn，这些技术都可以用一行代码实现：```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y)

model_weighted = LogisticRegression(class_weight="balanced")
model_weighted.fit(X_train, y_train)
print(classification_report(y_test, model_weighted.predict(X_test)))

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
model_smote = LogisticRegression()
model_smote.fit(X_resampled, y_resampled)
print(classification_report(y_test, model_smote.predict(X_test)))

pipeline = Pipeline([
    ("smote", SMOTE()),
    ("model", LogisticRegression(class_weight="balanced")),
])
pipeline.fit(X_train, y_train)
print(classification_report(y_test, pipeline.predict(X_test)))
```从零开始的实现展示了每种技术的确切作用。SMOTE只是对少数类进行k-NN插值。类别权重会乘以损失函数。阈值调整只是一个对截断值的循环。没有魔法。

## 发布它

本课将产出：
- `outputs/skill-imbalanced-data.md` -- 处理不平衡分类问题的决策清单

## 练习

1. **Borderline-SMOTE**：修改SMOTE实现，仅对靠近决策边界的少数类样本（那些k个最近邻中包含多数类样本的少数类样本）生成合成样本。在类别重叠的数据集上，将结果与标准SMOTE进行比较。

2. **成本矩阵优化**：实现成本敏感学习，其中成本矩阵是一个参数。创建一个函数，该函数接受一个成本矩阵并返回最小化预期成本的最优预测。使用不同的成本比例（1:10, 1:100, 1:1000）进行测试，并绘制精确率-召回率权衡的变化情况。

3. **阈值校准**：实现Platt缩放（对模型的原始输出拟合一个逻辑回归，以生成校准后的概率）。比较校准前后精确率-召回率曲线。证明校准不会改变排名（AUC保持不变），但会使概率更具意义。

4. **平衡袋外集成**：训练多个模型，每个模型基于一个平衡的自助采样（所有少数类+多数类的随机子集）。对它们的预测进行平均。将这种方法与使用SMOTE的单个模型进行比较。测量不同运行下的性能和方差。

5. **不平衡率实验**：从一个平衡的数据集开始，逐步增加不平衡率（50/50, 70/30, 90/10, 95/5, 99/1）。对每个不平衡率，使用和不使用SMOTE进行训练。绘制两种方法的F1与不平衡率的关系图。在哪个不平衡率下，SMOTE开始产生有意义的差异？

## 关键术语

| 术语 | 人们常说 | 它实际上意味着 |
|------|----------------|-----------------|
| 类别不平衡 | “一个类别有大量样本” | 数据集中类别分布显著偏斜，导致模型偏向多数类 |
| SMOTE | “合成过采样” | 通过在现有少数类样本及其k个最近邻少数类样本之间进行插值，生成新的少数类样本 |
| 类别权重 | “对罕见类的错误代价更高” | 通过类别特定的权重乘以损失函数，使模型更严厉地惩罚少数类的错误分类 |
| 阈值调整 | “移动决策边界” | 将分类的概率阈值从默认的0.5调整为一个优化所需指标的值 |
| 精确率-召回率权衡 | “你不能两者兼得” | 降低阈值会捕捉更多正例（更高的召回率），但也带来更多假正例（更低的精确率），反之亦然 |
| AUPRC | “PR曲线下的面积” | 将精确率-召回率曲线总结为一个数字；在类别严重不平衡时，比AUC-ROC更有信息量 |
| 马修斯相关系数 | “平衡的指标” | 预测标签与实际标签之间的相关性，只有在模型在两个类别上都表现良好时才会得到高分 |
| 成本敏感学习 | “不同的错误代价不同” | 将真实世界的误分类成本纳入训练目标，使模型优化总成本，而非错误数量 |
| 随机过采样 | “复制少数类” | 重复少数类样本以平衡类别数量；简单但可能过度拟合复制的样本点 |

## 进一步阅读

- [SMOTE: Synthetic Minority Over-sampling Technique (Chawla et al., 2002)](https://arxiv.org/abs/1106.1813) -- 原始的SMOTE论文，仍是不平衡学习领域被引用最多的论文
- [Learning from Imbalanced Data (He & Garcia, 2009)](https://ieeexplore.ieee.org/document/5128907) -- 涵盖采样、成本敏感和算法方法的全面综述
- [imbalanced-learn 文档](https://imbalanced-learn.org/stable/) -- 包含SMOTE变体、欠采样策略和管道集成的Python库
- [The Precision-Recall Plot Is More Informative than the ROC Plot (Saito & Rehmsmeier, 2015)](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0118432) -- 在不平衡问题中何时以及为何应优先使用PR曲线而不是ROC曲线
