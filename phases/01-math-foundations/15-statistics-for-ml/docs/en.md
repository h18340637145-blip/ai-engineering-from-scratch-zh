# 机器学习中的统计学

> 假设检验、置信区间与偏差-方差权衡。用严格的统计推断评估模型与特征。

**Type:** 学习
**Language:** Python
**Prerequisites:** Phase 1, Lesson 06 (概率论与概率分布)
**Time:** ~45 分钟

## 学习目标

- 从零开始计算描述性统计、皮尔逊/斯皮尔曼相关系数以及协方差矩阵
- 执行假设检验（t检验、卡方检验）并正确解释p值和置信区间
- 使用自助抽样（bootstrap resampling）在不依赖分布假设的情况下构造任何指标的置信区间
- 使用效应量指标区分统计显著性与实际显著性

## 问题

你训练了两个模型。模型A在测试集上得分为0.87。模型B得分为0.89。你部署了模型B。三周后，生产指标比之前更差。发生了什么？

模型B实际上并没有优于模型A。0.02的差异只是噪声。你的测试集太小，或者方差太高，或者两者皆有。你部署的是伪装成改进的随机性。

这种情况经常发生。Kaggle排行榜的剧烈变化。无法复现的论文。基于几百个样本的A/B测试声称找到胜者。根本原因总是相同：有人跳过了统计学。

统计学为你提供了区分信号与噪声的工具。它告诉你差异是否真实、你应该多自信、以及在可以信任结果之前需要多少数据。每个机器学习管道、每个模型比较、每个实验都需要统计学。没有它，你只是在猜测。

## 概念

### 描述性统计：总结你的数据

在你对任何东西进行建模之前，你需要了解你的数据是什么样子的。描述性统计将数据集压缩成几个数字，以捕捉其形状。

**集中趋势的度量**回答“中间在哪里？”

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

平均数是平衡点。中位数是中间的标记。当它们出现偏差时，说明你的分布是偏斜的。收入分布的平均数远大于中位数（右偏，来自亿万富翁）。训练过程中的损失分布通常平均数远小于中位数（左偏，来自简单的样本）。

**离散程度的度量** 回答“数据有多分散？”

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```**百分位数**将已排序的数据分成100个相等的部分。第25百分位数（Q1）表示有25%的值低于该点。第50百分位数是中位数。第75百分位数是Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在机器学习中，你关注推理延迟的百分位数、预测置信度分布以及误差分布的理解。一个平均误差较低但P99误差极差的模型，可能在安全关键型应用中毫无用处。

**样本统计与总体统计。** 在从样本计算方差时，应除以(n-1)而不是n。这是贝塞尔修正。它补偿了样本均值并非真实总体均值的事实。如果分母使用n，会系统性低估真实方差。如果使用(n-1)，则估计是无偏的。

```
Population variance: sigma^2 = (1/N) * sum((x_i - mu)^2)
Sample variance:     s^2     = (1/(n-1)) * sum((x_i - x_bar)^2)
```

实际上：如果 n 很大（数千个样本），差异可以忽略不计。如果 n 很小（几十个样本），则会产生影响。

### 相关性：变量如何共同变化

相关性衡量两个变量之间线性关系的强度和方向。

**皮尔逊相关系数**衡量线性关联：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```Pearson 假设变量之间的关系是线性的，并且两个变量大致呈正态分布。它对异常值比较敏感。一个极端的点可以将 r 从 0.1 拉到 0.9。

**Spearman 等级相关** 测量单调关联：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```**何时使用每个：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```**黄金法则：** 相关性不等于因果性。冰淇淋销售和溺水死亡人数是相关的，因为两者在夏季都会增加。你的模型的准确性和参数数量是相关的，但增加参数并不自动提高准确性（参见：过拟合）。

### 协方差矩阵

两个变量之间的协方差衡量它们如何共同变化：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

对于 d 个特征，协方差矩阵 C 是一个 d x d 的矩阵，其中 C[i][j] = Cov(feature_i, feature_j)。对角线元素 C[i][i] 是每个特征的方差。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```**与PCA的联系。** PCA对协方差矩阵进行特征分解。特征向量是主成分（最大方差的方向）。特征值告诉你每个成分所捕捉的方差量。这正是第10课所讲的内容，但现在你明白了为什么协方差矩阵是正确的分解对象：它编码了数据中所有两两线性关系。

**与相关性的联系。** 相关性矩阵是标准化变量（每个变量除以其标准差）的协方差矩阵。相关性对协方差进行了归一化，使得所有值落在[-1, 1]范围内。

### 假设检验

假设检验是在不确定性下进行决策的一种框架。你从一个主张开始，收集数据，并确定数据是否与该主张一致。

**设定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```**p值** 是在假设 H0 为真时，观察到的数据与实际数据一样极端的概率。它 **不是** H0 为真的概率。这是统计学中最常见的误解之一。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```**置信区间** 给出参数的可能值范围：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

置信区间的宽度告诉你关于精度的信息。宽的区间意味着高不确定性。窄的区间意味着你的估计是精确的（但如果数据存在偏差，不一定准确）。

### t检验

t检验比较均值。有几种不同的类型。

**单样本t检验：** 总体均值是否与假设值不同？

```
t = (x_bar - mu_0) / (s / sqrt(n))

degrees of freedom = n - 1
```**双样本 t 检验（独立样本）：** 两个组的均值是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```**配对t检验：** 当测量值成对出现时（同一模型在相同数据划分上进行评估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在机器学习中，配对t检验是常见的：你将两个模型都运行在相同的10个交叉验证折叠上，并成对比较它们的得分。

### 卡方检验

卡方检验用于检查观察到的频率是否与预期频率相匹配。适用于分类数据。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 模型的 A/B 测试

ML 中的 A/B 测试与网页 A/B 测试并不相同。模型比较有其特定的挑战：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 统计显著性 vs 实际显著性

一个结果可能具有统计显著性，但实际意义却可能微乎其微。只要有足够多的数据，即使是一个微不足道的差异也可能变得具有统计显著性。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```**效应量**量化了差异的大小，与样本量无关：

```
Cohen's d = (mean_1 - mean_2) / pooled_std

d = 0.2:  small effect
d = 0.5:  medium effect
d = 0.8:  large effect
```

始终报告 p 值和效应量。p 值告诉你差异是否真实存在。效应量告诉你这个差异是否重要。

### 多重比较问题

当你测试多个假设时，有些会因为偶然而“显著”。如果你在 alpha = 0.05 的水平下测试 20 个事物，即使没有任何真实效应，你也会预期出现 1 个假阳性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```**Bonferroni校正：** 将alpha除以检验的数目。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在机器学习中，当你跨多个指标比较模型、测试许多超参数配置或在多个数据集上进行评估时，这就会变得重要。

### 自举方法（Bootstrap Methods）

自举方法通过有放回地重新采样数据来估计统计量的抽样分布。不需要对底层分布做任何假设。

**算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```**Bootstrap 置信区间（百分位数方法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```**为什么 Bootstrap 对于机器学习很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```**用于模型比较的引导法：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

这比配对t检验更稳健，因为它不作任何分布假设。

### 参数检验与非参数检验

**参数检验**假设特定的分布（通常是正态分布）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```**非参数检验** 不做任何分布假设：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```**何时使用非参数方法：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```**何时使用参数化：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在机器学习实验中，通常样本量 n 较小（5 或 10 个交叉验证折叠），因此非参数检验（如 Wilcoxon 符号秩检验）通常比 t 检验更为合适。

### 中心极限定理：实际影响

中心极限定理指出，当样本量 n 增加时，样本均值的分布会趋近于正态分布，无论总体分布如何。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```**为什么这对机器学习很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```**CLT 不做的事情：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### 机器学习论文中常见的统计错误

1. **在训练集上进行测试。** 这会导致过拟合。在训练过程中，模型永远不应该看到用于测试的数据。

2. **没有置信区间。** 报告一个没有不确定性的单一准确率数值，使得结果无法复现和验证。

3. **忽略多重比较。** 测试了50种配置，但没有进行修正就报告最好的一个，会增加假阳性率。

4. **混淆统计显著性和实际显著性。** 对于0.01%的准确率提升，p值为0.001并不具有实际意义。

5. **在不平衡数据上使用准确率。** 如果数据集中99%是负类，那么99%的准确率意味着模型什么也没学到。应使用精确率、召回率、F1或AUC。

6. **选择性地使用指标。** 只报告你模型表现最好的指标。诚实的评估应该报告所有相关的指标。

7. **在训练/测试数据划分之间泄露信息。** 在划分数据之前进行标准化，或使用未来数据预测过去。

8. **小测试集且没有方差估计。** 在100个样本上进行评估并声称有2%的提升，这其实是噪声，而不是信号。

9. **在数据不独立时假设独立性。** 来自同一患者的医学图像、同一文档中的多个句子。组内观测值是相关的。

10. **p值挖掘。** 尝试不同的测试、子集或排除标准，直到获得p < 0.05。该结果是搜索过程的产物。

## 实现过程

你将实现：

1. **从零开始的描述性统计**（均值、中位数、众数、标准差、百分位数、IQR）
2. **相关函数**（皮尔逊和斯皮尔曼，以及协方差矩阵）
3. **假设检验**（单样本t检验、双样本t检验、卡方检验）
4. **引导置信区间**（适用于任何统计量，无需任何假设）
5. **A/B测试模拟器**（生成数据，进行测试，检查I型和II型错误）
6. **统计显著性与实际显著性演示**（展示大样本量会使一切看起来“显著”）

全部从零开始实现，只使用`math`和`random`。不使用numpy，不使用scipy。

## 术语表

| 术语 | 定义 |
|---|---|
| 均值 | 所有值的总和除以数量。对异常值敏感。 |
| 中位数 | 排序后的数据的中间值。对异常值稳健。 |
| 标准差 | 方差的平方根。衡量原始单位下的分布范围。 |
| 百分位数 | 数据中低于该值的给定百分比。 |
| IQR | 四分位距。Q3减去Q1。中间50%数据的分布范围。 |
| 皮尔逊相关系数 | 衡量两个变量之间的线性关系。范围[-1, 1]。 |
| 斯皮尔曼相关系数 | 使用排名衡量单调关系。 |
| 协方差矩阵 | 所有特征之间成对协方差的矩阵。 |
| 零假设 | 默认的无效应或无差异假设。 |
| p值 | 给定零假设成立时，出现如此极端数据的概率。 |
| 置信区间 | 在给定置信水平下，参数的可能值范围。 |
| t检验 | 检验均值是否显著不同。使用t分布。 |
| 卡方检验 | 检验观察频率是否与预期频率不同。 |
| 效应量 | 差异的大小，与样本量无关。Cohen's d是常用的。 |
| Bonferroni校正 | 将显著性阈值除以测试数量以控制假阳性。 |
| 引导法 | 有放回的重采样，用于估计抽样分布。 |
| I型错误 | 假阳性。当零假设为真时拒绝零假设。 |
| II型错误 | 假阴性。当零假设为假时未能拒绝零假设。 |
| 统计功效 | 正确拒绝假零假设的概率。功效 = 1 减去 II型错误率。 |
| 中心极限定理 | 随着样本量增加，样本均值趋近于正态分布。 |
| 参数检验 | 假设数据有特定分布（通常是正态分布）。 |
| 非参数检验 | 不做任何分布假设。基于排名或符号进行操作。
