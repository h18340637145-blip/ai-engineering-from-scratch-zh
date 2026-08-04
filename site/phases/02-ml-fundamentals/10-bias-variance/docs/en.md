# 偏差-方差权衡

> 每个模型的误差都来自三个来源之一：偏差、方差或噪声。你只能控制前两者。

**类型:** 学习
**语言:** Python
**先决条件:** 第二阶段，第 01-09 课（机器学习基础、回归、分类、评估）
**时间:** ~75 分钟

## 学习目标

- 推导出预期预测误差的偏差-方差分解，并解释不可约噪声的作用
- 使用训练和测试误差模式判断模型是否存在高偏差或高方差
- 解释正则化技术（L1、L2、dropout、early stopping）如何通过增加偏差来减少方差
- 实现可视化展示随着模型复杂度增加而产生的偏差-方差权衡的实验

## 问题

你训练了一个模型。它在测试数据上存在一些误差。这些误差来自哪里？

如果模型过于简单（在弯曲数据集上使用线性回归），它将始终无法捕捉到真实模式。这就是偏差。如果模型过于复杂（在 15 个数据点上使用 20 次多项式），它将完美拟合训练数据，但在新数据上会给出完全不同的预测。这就是方差。

对于固定容量的模型，你无法同时最小化偏差和方差。降低偏差会导致方差增加，降低方差会导致偏差增加。理解这种权衡是机器学习中最有用的诊断技能。它告诉你是否应该让模型更复杂或更简单，是否应该获取更多数据或设计更好的特征，是否应该增加或减少正则化。

## 概念

### 偏差：系统误差

偏差衡量的是模型的平均预测值与真实值之间的差距。如果你在许多从相同分布中抽取的不同训练集上训练相同的模型，并对预测值取平均，偏差就是这个平均值与真实值之间的差距。

高偏差意味着模型过于僵硬，无法捕捉真实模式。用一条直线拟合抛物线将始终无法捕捉到曲线，无论你提供多少数据。这就是欠拟合。```
High bias (underfitting):
  Model always predicts roughly the same wrong thing.
  Training error: HIGH
  Test error: HIGH
  Gap between them: SMALL
```### 方差：对训练数据的敏感性

方差衡量的是当你在不同的数据子集上训练时，你的预测会发生多大的变化。如果训练集的小变化导致模型的大幅变化，说明方差较高。

高方差意味着模型在拟合训练数据中的噪声，而不是潜在的信号。一个20次多项式会穿过每一个训练点，但它们之间会剧烈震荡。这就是过拟合。```
High variance (overfitting):
  Model fits training data perfectly but fails on new data.
  Training error: LOW
  Test error: HIGH
  Gap between them: LARGE
```### 分解

对于任何点 x，在平方损失下，期望预测误差可以精确地分解为：```
Expected Error = Bias^2 + Variance + Irreducible Noise

where:
  Bias^2   = (E[f_hat(x)] - f(x))^2
  Variance = E[(f_hat(x) - E[f_hat(x)])^2]
  Noise    = E[(y - f(x))^2]             (sigma^2)
```- `f(x)` 是真实函数
- `f_hat(x)` 是你模型的预测
- `E[...]` 是对不同训练集的期望
- `y` 是观测到的标签（真实函数加上噪声）

噪声项是不可约减的。在有噪声的数据上，没有模型可以做得比 sigma^2 更好。你的任务是在偏差的平方和方差之间找到合适的平衡。

### 模型复杂度与误差```mermaid
graph LR
    A[Simple Model] -->|increase complexity| B[Sweet Spot]
    B -->|increase complexity| C[Complex Model]

    style A fill:#f9f,stroke:#333
    style B fill:#9f9,stroke:#333
    style C fill:#f99,stroke:#333
```经典的U型曲线：

| 复杂度 | 偏差 | 方差 | 总误差 |
|------|---|----|---|
| 太低 | 高 | 低 | 高（欠拟合） |
| 刚好 | 中等 | 中等 | 最低 |
| 太高 | 低 | 高 | 高（过拟合） |

### 正则化作为偏差-方差控制

正则化有意增加偏差以减少方差。它限制模型，使其不能追逐噪声。

- **L2（岭回归）：** 将所有权重向零收缩。保留所有特征但减少其影响。
- **L1（Lasso）：** 将一些权重精确地推至零。进行特征选择。
- **Dropout：** 在训练期间随机禁用神经元。强制产生冗余表示。
- **早停法：** 在模型完全拟合训练数据之前停止训练。

正则化强度（lambda、dropout率、epoch数量）直接控制你在偏差-方差曲线上的位置。更多的正则化意味着更多的偏差，更少的方差。

### 双降：现代视角

经典理论认为：在最佳点之后，更多的复杂度总是有害的。但自2019年以来的研究显示了一些意想不到的结果。如果你继续增加模型容量，远远超过插值阈值（模型具有足够参数以完美拟合训练数据），测试误差可能会再次降低。```mermaid
graph LR
    A[Underfit Zone] --> B[Classical Sweet Spot]
    B --> C[Interpolation Threshold]
    C --> D[Double Descent - Error Drops Again]

    style A fill:#fdd,stroke:#333
    style B fill:#dfd,stroke:#333
    style C fill:#fdd,stroke:#333
    style D fill:#dfd,stroke:#333
```这种“双重下降”现象解释了为什么高度过参数化的神经网络（参数数量远多于训练样本数量）仍然能够很好地泛化。经典的偏差-方差权衡并没有错，但它对于现代的场景是不完整的。

关于双重下降的关键观察：
- 它出现在线性模型、决策树和神经网络中
- 在插值区域（样本级双重下降）中，更多的数据实际上可能会带来伤害
- 更多的训练轮数也可能导致双重下降（轮数级双重下降）
- 正则化可以平滑峰值，但无法消除它

为什么会发生这种情况？在插值阈值处，模型刚好有足够的容量来拟合所有训练点。它被迫进入一个非常特定的解，该解穿过每一个点，而数据中的微小扰动会导致拟合结果出现大的变化。这就是方差达到峰值的地方。超过阈值后，模型有许多可能的解，这些解都能完美拟合数据。学习算法（例如带有隐式正则化的梯度下降）倾向于在这些解中选择最简单的一个。这种对简单解的隐式偏见就是高度过参数化模型能够泛化的原因。

| 场景 | 参数与样本数量 | 行为 |
|------|----------------|------|
| 欠参数化 | p << n | 经典的权衡适用 |
| 插值阈值 | p ~ n | 方差达到峰值，测试误差激增 |
| 过参数化 | p >> n | 隐式正则化开始起作用，测试误差下降 |

从实际应用的角度来看：如果你正在使用神经网络或大型树集成，不要在插值阈值处停止。要么保持远低于该阈值（使用显式正则化），要么远远超过该阈值。最糟糕的情况是正好处于阈值处。

### 诊断你的模型```mermaid
flowchart TD
    A[Compare train error vs test error] --> B{Large gap?}
    B -->|Yes| C[High variance - overfitting]
    B -->|No| D{Both errors high?}
    D -->|Yes| E[High bias - underfitting]
    D -->|No| F[Good fit]

    C --> G[More data / Regularize / Simpler model]
    E --> H[More features / Complex model / Less regularization]
    F --> I[Deploy]
```| Symptom | Diagnosis | Fix |
|---------|-----------|------|
| 高训练误差，高测试误差 | 偏差 | 增加特征，更复杂的模型，减少正则化 |
| 低训练误差，高测试误差 | 方差 | 更多数据，正则化，更简单的模型，dropout |
| 低训练误差，低测试误差 | 良好拟合 | 发布 |
| 训练误差下降，测试误差上升 | 过拟合 | 早停 |

### 实用策略

**当问题是偏差时：**
- 添加多项式或交互特征
- 使用更灵活的模型（树集成而不是线性）
- 减少正则化强度
- 更长时间训练（如果尚未收敛）

**当问题是方差时：**
- 获取更多训练数据
- 使用装袋（随机森林）
- 增加正则化（更高的lambda，更多的dropout）
- 特征选择（移除噪声特征）
- 使用交叉验证提前检测

### 集成方法和方差减少

集成方法是应对方差的最实用工具。

**装袋（Bootstrap Aggregating）** 在训练数据的不同bootstrap样本上训练多个模型，然后平均它们的预测结果。每个单独模型有高方差，但平均后的方差要小得多。随机森林是将装袋应用于决策树。

数学上的原理：如果你平均N个独立预测，每个预测的方差为sigma^2，那么平均后的方差为sigma^2 / N。模型并非真正独立（它们都看到类似的数据），所以减少的幅度小于1/N，但仍然显著。

**提升（Boosting）** 通过顺序构建模型，每个新模型都关注集成到目前为止的误差来减少偏差。梯度提升和AdaBoost是主要例子。如果添加太多模型，提升可能会过拟合，所以需要早停或正则化。

| 方法 | 主要效果 | 偏差变化 | 方差变化 |
|------|----------|----------|----------|
| 装袋 | 减少方差 | 无变化 | 减少 |
| 提升 | 减少偏差 | 减少 | 可能增加 |
| 栈式集成 | 减少两者 | 取决于元学习器 | 取决于基模型 |
| Dropout | 隐式装袋 | 略微增加 | 减少 |

**实用规则：** 如果你的基模型有高方差（深树，高次多项式），使用装袋。如果你的基模型有高偏差（浅树桩，简单的线性模型），使用提升。

### 学习曲线

学习曲线将训练和验证误差绘制为训练集大小的函数。它们是你拥有的最实用的诊断工具。与单次训练/测试比较不同，学习曲线显示你的模型轨迹，并告诉你更多的数据是否会有帮助。```mermaid
flowchart TD
    subgraph HB["High Bias Learning Curve"]
        direction LR
        HB1["Small N: both errors high"]
        HB2["Large N: both errors converge to HIGH error"]
        HB1 --> HB2
    end

    subgraph HV["High Variance Learning Curve"]
        direction LR
        HV1["Small N: train low, test high (big gap)"]
        HV2["Large N: gap shrinks but slowly"]
        HV1 --> HV2
    end

    subgraph GF["Good Fit Learning Curve"]
        direction LR
        GF1["Small N: some gap"]
        GF2["Large N: both converge to LOW error"]
        GF1 --> GF2
    end
```如何解读它们：

| 场景 | 训练误差 | 验证误差 | 差值 | 含义 | 做法 |
|------|---------|---------|-----|-----|-----|
| 高偏差 | 高 | 高 | 小 | 模型无法捕捉模式 | 增加特征、更复杂的模型、减少正则化 |
| 高方差 | 低 | 高 | 大 | 模型记住了训练数据 | 增加数据、正则化、更简单的模型 |
| 良好拟合 | 中等 | 中等 | 小 | 模型泛化能力好 | 上线使用 |
| 高方差，正在改善 | 低 | 随着更多数据减少 | 缩小 | 数据可以解决的方差问题 | 收集更多数据 |
| 高偏差，平缓 | 高 | 高且平缓 | 小且平缓 | 更多数据不会有所帮助 | 更改模型结构 |

关键见解：如果两条曲线都已经趋于平稳，且差值小但两个误差都高，更多数据是没有用的。你需要一个更好的模型。如果差值大且还在缩小，更多数据会有帮助。

### 如何生成学习曲线

有两种方法：

**方法1：改变训练集大小，保持模型不变。** 保持模型和超参数不变。在训练数据的越来越大的子集上进行训练。在每个大小上测量训练误差和验证误差。这就是标准的学习曲线。

**方法2：改变模型复杂度，保持数据不变。** 保持数据不变。调整一个复杂度参数（多项式次数、树的深度、层数）。在每个复杂度上测量训练误差和验证误差。这就是验证曲线，可以直接显示偏差-方差权衡。

两种方法相互补充。第一种方法告诉你更多数据是否有帮助。第二种方法告诉你不同模型是否有帮助。在做出下一步决策之前，两种方法都应该运行。```mermaid
flowchart TD
    A[Model underperforming] --> B[Generate learning curve]
    B --> C{Gap between train and val?}
    C -->|Large gap, val still decreasing| D[More data will help]
    C -->|Small gap, both high| E[More data will NOT help]
    C -->|Large gap, val flat| F[Regularize or simplify]
    E --> G[Generate validation curve]
    G --> H[Try more complex model]
```

```figure
bias-variance
```## 构建它

`code/bias_variance.py` 中的代码运行完整的偏差-方差分解实验。以下是逐步的方法。

### 步骤 1：从已知函数生成合成数据

我们使用添加了高斯噪声的 `f(x) = sin(1.5x) + 0.5x`。知道真实函数使我们能够计算精确的偏差和方差。```python
def true_function(x):
    return np.sin(1.5 * x) + 0.5 * x

def generate_data(n_samples=30, noise_std=0.5, x_range=(-3, 3), seed=None):
    rng = np.random.RandomState(seed)
    x = rng.uniform(x_range[0], x_range[1], n_samples)
    y = true_function(x) + rng.normal(0, noise_std, n_samples)
    return x, y
```### 步骤 2：引导抽样和多项式拟合

对于每个多项式次数，我们抽取许多引导训练集，拟合多项式，并在固定的测试网格上记录预测结果。这为我们提供了每个测试点处的预测分布。```python
def fit_polynomial(x_train, y_train, degree, lam=0.0):
    X = np.column_stack([x_train ** d for d in range(degree + 1)])
    if lam > 0:
        penalty = lam * np.eye(X.shape[1])
        penalty[0, 0] = 0
        w = np.linalg.solve(X.T @ X + penalty, X.T @ y_train)
    else:
        w = np.linalg.lstsq(X, y_train, rcond=None)[0]
    return w
```我们在 200 个不同的自助（bootstrap）样本上进行拟合。每个自助样本都来自相同的潜在分布，但包含不同的点。

### 步骤 3：计算 Bias^2，方差分解

在每个测试点上，我们有 200 组预测值，因此可以直接根据定义计算分解：```python
mean_pred = predictions.mean(axis=0)
bias_sq = np.mean((mean_pred - y_true) ** 2)
variance = np.mean(predictions.var(axis=0))
total_error = np.mean(np.mean((predictions - y_true) ** 2, axis=1))
```- `mean_pred` 是从 bootstrap 样本中估计的 E[f_hat(x)]
- `bias_sq` 是平均预测值与真实值之间差距的平方
- `variance` 是 bootstrap 样本中预测值的平均差异
- `total_error` 应该大约等于偏差的平方加上方差加上噪声

### 第 4 步：学习曲线

学习曲线在保持模型复杂度不变的情况下，改变训练集的大小。它们可以显示你的模型是数据受限还是容量受限。```python
def demo_learning_curves():
    sizes = [10, 15, 20, 30, 50, 75, 100, 150, 200, 300]
    degree = 5

    for n in sizes:
        train_errors = []
        test_errors = []
        for seed in range(50):
            x_train, y_train = generate_data(n_samples=n, seed=seed * 100)
            w = fit_polynomial(x_train, y_train, degree)
            train_pred = predict_polynomial(x_train, w)
            train_mse = np.mean((train_pred - y_train) ** 2)
            test_pred = predict_polynomial(x_test, w)
            test_mse = np.mean((test_pred - y_test) ** 2)
            train_errors.append(train_mse)
            test_errors.append(test_mse)
        # Average over runs gives the learning curve point
```对于一个高方差模型（5次多项式且数据量小），你将看到：
- 训练误差开始较低，随着数据量增加，记忆难度加大，误差逐渐上升
- 测试误差开始较高，随着模型接收到更多信号，误差逐渐下降
- 随着数据量的增加，训练误差和测试误差之间的差距逐渐缩小

对于一个高偏差模型（1次多项式），训练误差和测试误差都迅速收敛到相同的高值，更多的数据并没有帮助。

### 步骤5：正则化扫描

代码中还包含了 `demo_regularization_sweep()`，它固定了一个高次多项式（15次多项式），并扫描了岭回归正则化强度，范围从0.001到100。这展示了偏差-方差权衡的另一个角度：不是通过改变模型复杂度，而是通过改变约束强度。```python
def demo_regularization_sweep():
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    for alpha in alphas:
        results = bias_variance_decomposition([15], lam=alpha)
        r = results[15]
        print(f"alpha={alpha:.3f}  bias={r['bias_sq']:.4f}  var={r['variance']:.4f}")
```在 alpha 值较低时，15 次多项式几乎没有任何约束。方差占主导地位，因为模型在每次自助采样中追逐噪声。在 alpha 值较高时，惩罚非常强烈，模型实际上变成了一个近似常数的函数。此时偏差占主导地位。最佳的 alpha 值位于这两个极端之间。

这与通过改变多项式次数所得到的 U 型曲线相同，但这里是由一个连续的旋钮控制，而不是由一个离散的旋钮控制。在实践中，正则化是控制这种权衡的首选方法，因为它允许在不改变特征集的情况下进行精细的控制。

## 使用它

sklearn 提供了 `learning_curve` 和 `validation_curve`，用于在不编写自助采样循环的情况下自动执行这些诊断。

### 验证曲线：扫过模型复杂度```python
from sklearn.model_selection import validation_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

degrees = list(range(1, 16))
train_scores_all = []
val_scores_all = []

for d in degrees:
    pipe = make_pipeline(PolynomialFeatures(d), Ridge(alpha=0.01))
    train_scores, val_scores = validation_curve(
        pipe, X, y, param_name="polynomialfeatures__degree",
        param_range=[d], cv=5, scoring="neg_mean_squared_error"
    )
    train_scores_all.append(-train_scores.mean())
    val_scores_all.append(-val_scores.mean())
```这直接给出了偏差-方差权衡曲线。当验证得分相对于训练得分最差时，方差占主导地位。当两者都较差时，偏差占主导地位。

### 学习曲线：遍历训练集大小```python
from sklearn.model_selection import learning_curve

pipe = make_pipeline(PolynomialFeatures(5), Ridge(alpha=0.01))
train_sizes, train_scores, val_scores = learning_curve(
    pipe, X, y, train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5, scoring="neg_mean_squared_error"
)
train_mse = -train_scores.mean(axis=1)
val_mse = -val_scores.mean(axis=1)
```将 `train_mse` 和 `val_mse` 以 `train_sizes` 为横轴绘制图形。图形的形状会告诉你关于你的模型的一切信息。

### 带正则化扫描的交叉验证```python
from sklearn.model_selection import cross_val_score

alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
for alpha in alphas:
    pipe = make_pipeline(PolynomialFeatures(10), Ridge(alpha=alpha))
    scores = cross_val_score(pipe, X, y, cv=5, scoring="neg_mean_squared_error")
    print(f"alpha={alpha:>7.3f}  MSE={-scores.mean():.4f} +/- {scores.std():.4f}")
```这会针对固定模型复杂度调整正则化强度。你将看到同样的偏差-方差权衡：低 alpha 意味着高方差，高 alpha 意味着高偏差。

### 综合应用：完整的诊断工作流程

在实践中，你按顺序运行这些诊断：

1. 训练你的模型。计算训练和测试误差。
2. 如果两者都高：你存在偏差问题。跳到第4步。
3. 如果训练误差低但测试误差高：你存在方差问题。生成学习曲线以查看更多数据是否会有帮助。如果没有，进行正则化。
4. 生成验证曲线，调整你的主要复杂度参数。找到最佳平衡点。
5. 在最佳平衡点处生成学习曲线。如果差距仍然很大，你需要更多数据或正则化。
6. 使用 `cross_val_score` 尝试 Ridge/Lasso 并使用不同的 alpha 值。选择交叉验证误差最低的 alpha。

这通常需要对大多数表格数据集进行10到15分钟的计算，节省了数小时的猜测时间。

## 发布它

本课生成：`outputs/prompt-model-diagnostics.md`

## 练习

1. 使用 `noise_std=0` 进行分解（无噪声）。不可约误差项会发生什么变化？最优复杂度是否改变？

2. 将训练集大小从 30 增加到 300。这如何影响方差部分？最优多项式次数是否改变？

3. 向实验中添加 L2 正则化（岭回归）。对于固定高次多项式（次数 15），调整 lambda 从 0 到 100。将偏差平方和方差绘制为 lambda 的函数。

4. 将真实函数从多项式修改为 `sin(x)`。偏差-方差分解会如何变化？是否仍有明确的最优次数？

5. 实现一个简单的自助聚合（bagging）包装器：在自助样本上训练 10 个模型并平均预测。展示这如何减少方差而不显著增加偏差。

## 关键术语

| 术语 | 人们说的 | 实际含义 |
|------|----------------|----------|
| 偏差 | "模型太简单" | 由于错误假设产生的系统性误差。平均模型预测与真实值之间的差距。 |
| 方差 | "模型过拟合" | 对训练数据敏感产生的误差。不同训练集下预测结果的变化程度。 |
| 不可约误差 | "数据中的噪声" | 来自真实数据生成过程的随机性误差。没有模型可以消除它。 |
| 欠拟合 | "学习不足" | 模型偏差高。即使在训练数据上，也未能捕捉到真实模式。 |
| 过拟合 | "记住了数据" | 模型方差高。它拟合了训练数据中的噪声，这些噪声无法推广。 |
| 正则化 | "限制模型" | 添加惩罚项以降低模型复杂度，用增加偏差换取降低方差。 |
| 双降现象 | "更多参数可能有帮助" | 当模型容量远超过插值阈值时，测试误差再次下降。 |
| 模型复杂度 | "模型的灵活性" | 模型拟合任意模式的能力。受架构、特征或正则化的控制。 |

## 进一步阅读

- [Hastie, Tibshirani, Friedman: 统计学习基础，第7章](https://hastie.su.domains/ElemStatLearn/) —— 偏差-方差分解的权威论述
- [Belkin 等，协调现代机器学习实践和偏差-方差权衡（2019）](https://arxiv.org/abs/1812.11118) —— 双降现象论文
- [Nakkiran 等，深度双降（2019）](https://arxiv.org/abs/1912.02292) —— 按轮次和样本的双降现象
- [Scott Fortmann-Roe: 理解偏差-方差权衡](http://scott.fortmann-roe.com/docs/BiasVariance.html) —— 清晰的视觉解释
